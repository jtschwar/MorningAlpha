"""Sector-grouped dataset for Set Transformer training.

Each item is one (sector, date) set — all stocks in the same sector on the same date.
The model receives [N, D] features and outputs per-stock scores with cross-stock context.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List


class SectorSetDataset(Dataset):
    """
    Groups stocks by (sector, date). Each __getitem__ returns one set, padded to
    max_set_size with a boolean mask indicating real vs padding positions.

    Features must already be preprocessed (winsorized + rank-normalized) before
    passing to this class — use the 'split' column from the training parquet directly.
    """

    MIN_SET_SIZE = 4  # skip tiny sets — too little cross-stock context to learn from

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        max_set_size: int = 80,
    ):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.max_set_size = max_set_size
        self.n_features = len(feature_cols)

        self.sets: List[dict] = []

        for (sector, date), group in df.groupby(["sector", "date"], sort=False):
            n = len(group)
            if n < self.MIN_SET_SIZE:
                continue

            # Truncate oversized sets (rare) — sample randomly so no position bias
            if n > max_set_size:
                group = group.sample(max_set_size, random_state=42)
                n = max_set_size

            feats = group[feature_cols].values.astype(np.float32)
            targets = group[target_col].values.astype(np.float32)
            tickers = group["ticker"].values

            # Post rank-normalization, 0 = cross-sectional median — safe NaN fill
            feats = np.nan_to_num(feats, nan=0.0)
            # Target NaN → 0 (neutral rank); these rows contribute 0 gradient in MSE
            targets = np.nan_to_num(targets, nan=0.0)

            self.sets.append({
                "features": feats,
                "targets": targets,
                "tickers": tickers,
                "sector": int(sector),
                "date": str(date.date() if hasattr(date, "date") else date),
                "n": n,
            })

    def __len__(self) -> int:
        return len(self.sets)

    def __getitem__(self, idx: int) -> dict:
        item = self.sets[idx]
        N = item["n"]
        M = self.max_set_size
        D = self.n_features

        padded_features = np.zeros((M, D), dtype=np.float32)
        padded_targets = np.zeros(M, dtype=np.float32)
        padded_features[:N] = item["features"]
        padded_targets[:N] = item["targets"]

        # mask: True = real stock, False = padding
        mask = torch.zeros(M, dtype=torch.bool)
        mask[:N] = True

        return {
            "features": torch.tensor(padded_features),   # [M, D]
            "targets": torch.tensor(padded_targets),      # [M]
            "mask": mask,                                 # [M]  bool
            "n": N,
            "sector": item["sector"],
            "date": item["date"],
        }

    def summary(self) -> dict:
        ns = [s["n"] for s in self.sets]
        sectors = [s["sector"] for s in self.sets]
        return {
            "n_sets": len(self.sets),
            "n_features": self.n_features,
            "mean_set_size": float(np.mean(ns)),
            "median_set_size": float(np.median(ns)),
            "min_set_size": int(np.min(ns)),
            "max_set_size": int(np.max(ns)),
            "n_sectors": len(set(sectors)),
        }
