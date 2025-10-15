"""Validation/evaluation script for the BERT regressor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from accelerate import Accelerator
from torch import nn
from transformers import AutoTokenizer

from master.data import (
    TextPriceDataset,
    build_data_paths,
    create_dataloader,
    load_dataframe,
    stratified_split,
)
from master.models import BertRegressor
from master.utils import ConfigError, compute_smape, load_config, set_seed, to_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned BERT regressor.")
    parser.add_argument("--config", type=str, default="config/paths.yml", help="Path to YAML config file.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/bert/best",
        help="Directory containing the saved checkpoint (backbone + regressor).",
    )
    parser.add_argument("--output", type=str, default="outputs/logs/bert_eval.json", help="Path to write evaluation metrics JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc

    training_cfg = to_training_config(config)
    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)

    accelerator = Accelerator()

    data_paths = build_data_paths(config)
    train_df = load_dataframe(data_paths.train_csv)
    train_df = train_df[train_df["catalog_content"].notna()].reset_index(drop=True)
    _, val_split = stratified_split(train_df, training_cfg.validation_size, seed)

    hf_cache_dir = Path(config.get("paths", {}).get("hf_cache_dir", "~/.cache/huggingface")).expanduser()
    tokenizer = AutoTokenizer.from_pretrained(training_cfg.pretrained_name, use_fast=True, cache_dir=str(hf_cache_dir))

    val_dataset = TextPriceDataset(val_split, include_price=True)
    val_loader = create_dataloader(
        val_dataset,
        tokenizer=tokenizer,
        max_length=training_cfg.max_length,
        batch_size=training_cfg.batch_size,
        shuffle=False,
        include_price=True,
        num_workers=training_cfg.dataloader_num_workers,
    )

    checkpoint_dir = Path(args.checkpoint)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    model = BertRegressor.from_pretrained(checkpoint_dir, hf_cache_dir=str(hf_cache_dir))
    model, val_loader = accelerator.prepare(model, val_loader)

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

    mae = nn.functional.l1_loss(predictions_tensor, targets_tensor).item()
    smape = compute_smape(predictions_tensor, targets_tensor).item()

    metrics: Dict[str, float] = {
        "validation_mae": mae,
        "validation_smape": smape,
        "num_validation_samples": float(predictions_tensor.shape[0]),
    }

    accelerator.print(json.dumps(metrics, indent=2))

    if accelerator.is_main_process:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
