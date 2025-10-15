"""Training script for the DistilBERT price regressor."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
from accelerate import Accelerator
from torch import nn
from transformers import AutoTokenizer, get_scheduler

from master.data import (
    TextPriceDataset,
    build_data_paths,
    create_dataloader,
    load_dataframe,
    stratified_split,
)
from master.models import BertRegressor
from master.utils import ConfigError, TrainingConfig, compute_smape, load_config, set_seed, to_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for price regression.")
    parser.add_argument("--config", type=str, default="config/paths.yml", help="Path to YAML config file.")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Accelerate mixed precision mode.")
    parser.add_argument("--output-suffix", type=str, default="", help="Optional suffix appended to checkpoint/log directories.")
    return parser.parse_args()


def prepare_directories(config: Dict[str, object], suffix: str) -> Tuple[Path, Path, Path]:
    paths_cfg = config.get("paths", {})
    project_root = Path(config.get("__project_root__"))

    def _resolve(relative: str) -> Path:
        return (project_root / Path(relative)).resolve()

    checkpoints_dir = _resolve(str(paths_cfg.get("checkpoints_dir", "checkpoints/bert")))
    logs_dir = _resolve(str(paths_cfg.get("logs_dir", "outputs/logs")))
    splits_dir = _resolve(str(paths_cfg.get("splits_dir", "outputs/splits")))

    if suffix:
        checkpoints_dir = checkpoints_dir / suffix
        logs_dir = logs_dir / suffix
        splits_dir = splits_dir / suffix

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    return checkpoints_dir, logs_dir, splits_dir


def training_loop() -> None:
    args = parse_args()
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc

    training_cfg = to_training_config(config)
    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    accelerator.print(f"Using device: {accelerator.device}")
    data_paths = build_data_paths(config)

    train_df = load_dataframe(data_paths.train_csv)
    train_df = train_df[train_df["catalog_content"].notna()].reset_index(drop=True)

    train_split, val_split = stratified_split(train_df, training_cfg.validation_size, seed)

    checkpoints_dir, logs_dir, splits_dir = prepare_directories(config, args.output_suffix)

    if accelerator.is_main_process:
        split_manifest = {
            "train_samples": train_split["sample_id"].tolist(),
            "val_samples": val_split["sample_id"].tolist(),
        }
        (splits_dir / "bert_split.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    hf_cache_dir = Path(config.get("paths", {}).get("hf_cache_dir", "~/.cache/huggingface")).expanduser()

    tokenizer = AutoTokenizer.from_pretrained(
        training_cfg.pretrained_name,
        use_fast=True,
        cache_dir=str(hf_cache_dir),
    )

    train_dataset = TextPriceDataset(train_split, include_price=True)
    val_dataset = TextPriceDataset(val_split, include_price=True)

    train_loader = create_dataloader(
        train_dataset,
        tokenizer=tokenizer,
        max_length=training_cfg.max_length,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        include_price=True,
        num_workers=training_cfg.dataloader_num_workers,
    )
    val_loader = create_dataloader(
        val_dataset,
        tokenizer=tokenizer,
        max_length=training_cfg.max_length,
        batch_size=training_cfg.batch_size,
        shuffle=False,
        include_price=True,
        num_workers=training_cfg.dataloader_num_workers,
    )

    steps_per_epoch = math.ceil(len(train_loader) / max(1, training_cfg.gradient_accumulation_steps))
    total_training_steps = steps_per_epoch * training_cfg.num_epochs
    warmup_steps = int(total_training_steps * training_cfg.warmup_ratio)

    model = BertRegressor(
        pretrained_name=training_cfg.pretrained_name,
        dropout=training_cfg.dropout,
        hf_cache_dir=str(hf_cache_dir),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
    )

    lr_scheduler = get_scheduler(
        name="cosine" if training_cfg.scheduler == "cosine" else "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    loss_fn = nn.L1Loss()
    history = []
    best_smape = float("inf")
    patience = training_cfg.early_stopping_patience
    patience_counter = 0

    accelerator.print(f"Training for {training_cfg.num_epochs} epochs")

    for epoch in range(1, training_cfg.num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels")
            batch.pop("sample_ids", None)
            predictions = model(**batch)
            loss = loss_fn(predictions, labels)
            loss = loss / training_cfg.gradient_accumulation_steps
            accelerator.backward(loss)

            if step % training_cfg.gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), training_cfg.grad_clip_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * training_cfg.gradient_accumulation_steps

        avg_epoch_loss = epoch_loss / len(train_loader)

        model.eval()
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                labels = batch.pop("labels")
                batch.pop("sample_ids", None)
                outputs = model(**batch)
                preds, refs = accelerator.gather_for_metrics((outputs, labels))
                all_predictions.append(preds)
                all_targets.append(refs)

        predictions_tensor = torch.cat(all_predictions)
        targets_tensor = torch.cat(all_targets)

        val_mae = nn.functional.l1_loss(predictions_tensor, targets_tensor).item()
        val_smape = compute_smape(predictions_tensor, targets_tensor).item()

        accelerator.print(
            f"Epoch {epoch}: TrainLoss={avg_epoch_loss:.4f} | ValMAE={val_mae:.4f} | ValSMAPE={val_smape:.2f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_epoch_loss,
                "val_mae": val_mae,
                "val_smape": val_smape,
            }
        )

        improved = val_smape < best_smape
        if improved:
            best_smape = val_smape
            patience_counter = 0
            accelerator.print(f"New best SMAPE: {best_smape:.2f}. Saving checkpoint...")
            accelerator.wait_for_everyone()
            unwrapped = accelerator.unwrap_model(model)
            if accelerator.is_main_process:
                checkpoint_dir = checkpoints_dir / "best"
                unwrapped.save_pretrained(checkpoint_dir)
                metrics_path = checkpoints_dir / "best_metrics.json"
                metrics = {
                    "epoch": epoch,
                    "val_mae": val_mae,
                    "val_smape": val_smape,
                }
                metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        else:
            patience_counter += 1
            accelerator.print(f"No improvement. Patience {patience_counter}/{patience}")

        if patience_counter >= patience:
            accelerator.print("Early stopping triggered.")
            break

    if accelerator.is_main_process:
        logs_dir.mkdir(parents=True, exist_ok=True)
        history_path = logs_dir / "bert_training_history.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    training_loop()
