"""Simple prediction averaging utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd


def average_prediction_files(csv_files: Iterable[str | Path], output_path: str | Path) -> Path:
    """Average predictions from multiple CSV files.

    Each input CSV must contain ``sample_id`` and ``price`` columns. The
    function validates alignment of sample identifiers before averaging.
    """

    csv_paths: List[Path] = [Path(p).resolve() for p in csv_files]
    if not csv_paths:
        raise ValueError("At least one prediction file is required for ensembling.")

    merged = None
    for path in csv_paths:
        if not path.exists():
            raise FileNotFoundError(f"Prediction file not found: {path}")
        df = pd.read_csv(path)
        if {"sample_id", "price"} - set(df.columns):
            raise ValueError(f"Missing required columns in {path}")
        df = df[["sample_id", "price"]].rename(columns={"price": f"price_{path.stem}"})
        merged = df if merged is None else merged.merge(df, on="sample_id", how="inner")

    price_cols = [col for col in merged.columns if col.startswith("price_")]
    merged["price"] = merged[price_cols].mean(axis=1)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged[["sample_id", "price"]].to_csv(output_path, index=False)
    return output_path
