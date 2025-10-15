"""Generate price predictions on the test set using the fine-tuned BERT model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from typing import List

import pandas as pd
import torch
from accelerate import Accelerator

try:
    # accelerate>=1.0 exposes gather_object via accelerate.utils
    from accelerate.utils import gather_object as _accelerate_gather_object
except ImportError:  # pragma: no cover - older accelerate versions
    _accelerate_gather_object = None
from transformers import AutoTokenizer

from master.data import TextPriceDataset, build_data_paths, create_dataloader, load_dataframe
from master.models import BertRegressor
from master.utils import ConfigError, load_config, set_seed, to_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the fine-tuned BERT regressor.")
    parser.add_argument("--config", type=str, default="config/paths.yml", help="Path to YAML config file.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/bert/best",
        help="Directory containing the saved checkpoint (backbone + regressor).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/predictions/bert.csv",
        help="CSV path to write the predictions (sample_id, price).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc

    accelerator = Accelerator()
    training_cfg = to_training_config(config)
    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)

    data_paths = build_data_paths(config)
    test_df = load_dataframe(data_paths.test_csv)

    hf_cache_dir = Path(config.get("paths", {}).get("hf_cache_dir", "~/.cache/huggingface")).expanduser()
    tokenizer = AutoTokenizer.from_pretrained(training_cfg.pretrained_name, use_fast=True, cache_dir=str(hf_cache_dir))

    test_dataset = TextPriceDataset(test_df, include_price=False)
    test_loader = create_dataloader(
        test_dataset,
        tokenizer=tokenizer,
        max_length=training_cfg.max_length,
        batch_size=training_cfg.batch_size,
        shuffle=False,
        include_price=False,
        num_workers=training_cfg.dataloader_num_workers,
    )

    checkpoint_dir = Path(args.checkpoint)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    model = BertRegressor.from_pretrained(checkpoint_dir, hf_cache_dir=str(hf_cache_dir))
    model, test_loader = accelerator.prepare(model, test_loader)

    model.eval()
    gathered_predictions = []
    gathered_ids: List[str] = []
    with torch.no_grad():
        for batch in test_loader:
            sample_ids = batch.pop("sample_ids")
            outputs = model(**batch)
            preds = accelerator.gather(outputs)
            gathered_predictions.append(preds)

            ids: List[str]
            if _accelerate_gather_object is not None:
                ids = _accelerate_gather_object(list(sample_ids))
            else:
                ids = list(sample_ids)

            if accelerator.is_main_process:
                gathered_ids.extend(ids)

    all_predictions = torch.cat(gathered_predictions)

    if accelerator.is_main_process:
        if not gathered_ids:
            gathered_ids = test_dataset.sample_ids
        prices = all_predictions.cpu().numpy()[: len(gathered_ids)]
        predictions = pd.DataFrame({
            "sample_id": gathered_ids,
            "price": prices,
        })
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
        metadata = {
            "checkpoint": str(checkpoint_dir),
            "num_samples": len(predictions),
        }
        (output_path.parent / "bert_prediction_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
