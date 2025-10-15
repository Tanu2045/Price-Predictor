"""Shared training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class TrainingConfig:
    pretrained_name: str
    max_length: int
    dropout: float
    learning_rate: float
    weight_decay: float
    num_epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    validation_size: float
    early_stopping_patience: int
    scheduler: str
    grad_clip_norm: float
    dataloader_num_workers: int


def to_training_config(config: Dict[str, object]) -> TrainingConfig:
    bert_cfg = config.get("model", {}).get("bert", {})
    return TrainingConfig(
        pretrained_name=str(bert_cfg.get("pretrained_name", "distilbert-base-uncased")),
        max_length=int(bert_cfg.get("max_length", 384)),
        dropout=float(bert_cfg.get("dropout", 0.1)),
        learning_rate=float(bert_cfg.get("learning_rate", 5e-5)),
        weight_decay=float(bert_cfg.get("weight_decay", 0.01)),
        num_epochs=int(bert_cfg.get("num_epochs", 5)),
        batch_size=int(bert_cfg.get("batch_size", 16)),
        gradient_accumulation_steps=int(bert_cfg.get("gradient_accumulation_steps", 1)),
        warmup_ratio=float(bert_cfg.get("warmup_ratio", 0.1)),
        validation_size=float(bert_cfg.get("validation_size", 0.1)),
        early_stopping_patience=int(bert_cfg.get("early_stopping_patience", 3)),
        scheduler=str(bert_cfg.get("scheduler", "cosine")),
        grad_clip_norm=float(bert_cfg.get("grad_clip_norm", 1.0)),
        dataloader_num_workers=int(bert_cfg.get("dataloader_num_workers", 0)),
    )


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_smape(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    numerator = torch.abs(predictions - targets)
    denominator = torch.clamp((torch.abs(targets) + torch.abs(predictions)) / 2.0, min=epsilon)
    return (numerator / denominator).mean() * 100
