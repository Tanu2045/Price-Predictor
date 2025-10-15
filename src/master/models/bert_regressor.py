"""BERT-based regression head for price prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class BertRegressor(nn.Module):
    """Backbone encoder paired with a lightweight regression head."""

    def __init__(
        self,
        pretrained_name: str,
        dropout: float = 0.1,
        hf_cache_dir: str | None = None,
        **model_kwargs: Any,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(pretrained_name, cache_dir=hf_cache_dir)
        self.dropout = dropout
        self.backbone = AutoModel.from_pretrained(
            pretrained_name,
            config=config,
            cache_dir=hf_cache_dir,
            **model_kwargs,
        )
        hidden_size = config.hidden_size

        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **_: Any) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        return self.regressor(pooled).squeeze(-1)

    def save_pretrained(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(path / "backbone")
        torch.save(self.regressor.state_dict(), path / "regressor.pt")
        metadata = {"dropout": self.dropout}
        (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, output_dir: str | Path, hf_cache_dir: str | None = None, **kwargs: Any) -> "BertRegressor":
        path = Path(output_dir)
        backbone_dir = path / "backbone"
        if not backbone_dir.exists():
            raise FileNotFoundError(f"Backbone weights not found at {backbone_dir}")

        metadata_path = path / "metadata.json"
        dropout = 0.1
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            dropout = metadata.get("dropout", dropout)

        config = AutoConfig.from_pretrained(backbone_dir, cache_dir=hf_cache_dir, **kwargs)

        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.dropout = dropout
        instance.backbone = AutoModel.from_pretrained(
            backbone_dir,
            config=config,
            cache_dir=hf_cache_dir,
            **kwargs,
        )
        hidden_size = config.hidden_size
        instance.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        regressor_weights = path / "regressor.pt"
        instance.regressor.load_state_dict(torch.load(regressor_weights, map_location="cpu"))
        return instance

    def export_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "backbone": self.backbone.state_dict(),
            "regressor": self.regressor.state_dict(),
        }
