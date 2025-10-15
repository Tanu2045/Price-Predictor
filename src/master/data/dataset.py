"""Dataset utilities for the product pricing challenge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


@dataclass
class DataPaths:
    """Container for dataset-related file system locations."""

    root_dir: Path
    train_csv: Path
    test_csv: Path
    image_dir: Path
    catalog_db: Path
    extra_features_dir: Path


def build_data_paths(config: Dict[str, object]) -> DataPaths:
    """Derive absolute dataset paths from the loaded configuration mapping."""

    project_root = Path(config.get("__project_root__"))
    data_cfg = config.get("data", {})

    def _resolve(base: Path, candidate: str) -> Path:
        path = Path(candidate)
        return (path if path.is_absolute() else (base / path)).resolve()

    root_dir = _resolve(project_root, str(data_cfg.get("root_dir", "../test1")))

    return DataPaths(
        root_dir=root_dir,
        train_csv=_resolve(root_dir, str(data_cfg.get("train_csv", "dataset/train.csv"))),
        test_csv=_resolve(root_dir, str(data_cfg.get("test_csv", "dataset/test.csv"))),
        image_dir=_resolve(root_dir, str(data_cfg.get("image_dir", "images"))),
        catalog_db=_resolve(root_dir, str(data_cfg.get("catalog_db", "catalog.db"))),
        extra_features_dir=_resolve(root_dir, str(data_cfg.get("extra_features_dir", "dataset/feature_cache"))),
    )


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    """Load a CSV file and normalise the text fields."""

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "catalog_content" not in df.columns:
        raise ValueError("Expected 'catalog_content' column in dataset.")

    df["catalog_content"] = df["catalog_content"].fillna("")
    return df


def stratified_split(
    df: pd.DataFrame,
    validation_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into train/validation subsets."""

    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be within (0, 1).")

    stratify_labels = None
    if "price" in df.columns and df["price"].notna().all():
        # Bin prices to approximate stratification while keeping regression target.
        stratify_labels = pd.qcut(df["price"], q=20, duplicates="drop", labels=False)

    train_df, val_df = train_test_split(
        df,
        test_size=validation_size,
        random_state=seed,
        stratify=stratify_labels,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


class TextPriceDataset(Dataset):
    """Torch dataset that yields text samples and optional targets/features."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        include_price: bool = True,
        tabular_features: Optional[np.ndarray] = None,
    ) -> None:
        self.sample_ids: List[str] = dataframe["sample_id"].astype(str).tolist()
        self.texts: List[str] = dataframe["catalog_content"].astype(str).tolist()
        self.include_price = include_price and "price" in dataframe.columns
        self.prices: Optional[np.ndarray] = (
            dataframe["price"].astype(np.float32).to_numpy() if self.include_price else None
        )

        if tabular_features is not None and len(tabular_features) != len(dataframe):
            raise ValueError("Tabular feature array must align with dataframe length.")
        self.tabular_features = tabular_features

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Dict[str, object]:
        item: Dict[str, object] = {
            "sample_id": self.sample_ids[index],
            "text": self.texts[index],
        }
        if self.include_price and self.prices is not None:
            item["price"] = float(self.prices[index])

        if self.tabular_features is not None:
            item["tabular_features"] = self.tabular_features[index]

        return item


def make_collate_fn(
    tokenizer,
    max_length: int,
    include_price: bool = True,
    include_tabular: bool = False,
) -> Callable[[List[Dict[str, object]]], Dict[str, object]]:
    """Create a collate function that tokenizes batches on the fly."""

    def collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
        texts = [example["text"] for example in batch]
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        tokenized["sample_ids"] = [example["sample_id"] for example in batch]

        if include_price and "price" in batch[0]:
            prices = torch.tensor([example.get("price", 0.0) for example in batch], dtype=torch.float32)
            tokenized["labels"] = prices

        if include_tabular:
            features = [example.get("tabular_features") for example in batch]
            if features[0] is not None:
                tokenized["tabular_features"] = torch.tensor(np.stack(features), dtype=torch.float32)

        return tokenized

    return collate


def create_dataloader(
    dataset: Dataset,
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    include_price: bool = True,
    include_tabular: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    collate_fn = make_collate_fn(
        tokenizer,
        max_length=max_length,
        include_price=include_price,
        include_tabular=include_tabular,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def placeholder_feature_harness(
    sample_ids: Iterable[str],
    feature_loaders: Optional[Iterable[Callable[[Iterable[str]], np.ndarray]]] = None,
) -> Optional[np.ndarray]:
    """Facilitate future regex/parsing feature extraction hooks.

    Each callable in ``feature_loaders`` should accept a sequence of sample IDs
    and return a NumPy array aligned with that sequence. Arrays are concatenated
    column-wise to create the final tabular feature matrix.
    """

    if not feature_loaders:
        return None

    features: List[np.ndarray] = []
    for loader in feature_loaders:
        candidate = loader(sample_ids)
        if not isinstance(candidate, np.ndarray):
            raise TypeError("Feature loader must return a NumPy array.")
        features.append(candidate)

    return np.concatenate(features, axis=1)
