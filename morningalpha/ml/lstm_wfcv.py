"""LSTM walk-forward CV utilities.

Provides:
  - make_wfcv_folds()      — expanding-window date splits with embargo
  - LSTMDateRangeDataset   — LSTMSequenceDataset filtered by date range;
                             stores sequence end-dates for EMA weighting
  - make_ema_sampler()     — WeightedRandomSampler with exponential time decay
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from morningalpha.ml.lstm_model import LSTM_HORIZONS
from morningalpha.ml.train_lstm import TARGET_COLS, RANK_TARGET_COLS


# ---------------------------------------------------------------------------
# Walk-forward fold definitions
# ---------------------------------------------------------------------------

def make_wfcv_folds(
    df: pd.DataFrame,
    n_folds: int = 6,
    embargo_days: int = 10,
) -> List[Dict]:
    """Expanding-window walk-forward splits.

    Divides the date range into (n_folds + 1) equal chunks.
    Fold k trains on chunks 0..k and validates on chunk k+1,
    with an embargo gap between train end and val start.

    Returns a list of dicts with keys:
        fold, train_start, train_end, val_start, val_end
    """
    dates = sorted(df["date"].dt.normalize().unique())
    n = len(dates)
    chunk_size = n // (n_folds + 1)

    folds = []
    for k in range(n_folds):
        train_end_idx = (k + 1) * chunk_size - 1
        val_start_idx = train_end_idx + 1 + embargo_days
        val_end_idx   = min((k + 2) * chunk_size - 1, n - 1)

        if val_start_idx >= n:
            break

        folds.append({
            "fold":        k + 1,
            "train_start": dates[0],
            "train_end":   dates[train_end_idx],
            "val_start":   dates[val_start_idx],
            "val_end":     dates[val_end_idx],
        })

    return folds


# ---------------------------------------------------------------------------
# Date-range dataset
# ---------------------------------------------------------------------------

class LSTMDateRangeDataset(Dataset):
    """Builds per-ticker sequences from a date range (not a split column).

    Identical target-mode logic to LSTMSequenceDataset, but filtered by
    [start_date, end_date].  Also stores sequence end-dates so an EMA
    sampler can weight recent windows more heavily.

    Parameters
    ----------
    df          : full preprocessed DataFrame (all splits)
    feat_cols   : feature column names
    start_date  : inclusive start date for this dataset
    end_date    : inclusive end date for this dataset
    lookback    : sequence length (trading days)
    stride      : step between windows per ticker
    target_mode : "log" | "clip" | "rank"
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feat_cols: List[str],
        start_date,
        end_date,
        lookback: int = 60,
        stride: int = 3,
        target_mode: str = "log",
    ) -> None:
        self.feat_cols   = feat_cols
        self.lookback    = lookback
        self.n_features  = len(feat_cols)
        self.target_mode = target_mode

        mask = (
            (df["date"] >= pd.Timestamp(start_date)) &
            (df["date"] <= pd.Timestamp(end_date))
        )
        sub = df[mask].copy()

        self._xs:        List[np.ndarray]          = []
        self._ys:        List[np.ndarray]          = []
        self._end_dates: List[np.datetime64]       = []  # for EMA weighting

        for _ticker, grp in sub.groupby("ticker"):
            grp = grp.sort_values("date").reset_index(drop=True)
            n   = len(grp)
            if n < lookback + 1:
                continue

            X     = grp[feat_cols].values.astype(np.float32)
            dates = grp["date"].values  # numpy datetime64

            if target_mode == "rank":
                rank_cols = [c for c in RANK_TARGET_COLS if c in grp.columns]
                if not rank_cols:
                    continue
                Y = grp[rank_cols].values.astype(np.float32)
            elif target_mode == "clip":
                Y = np.clip(grp[TARGET_COLS].values.astype(np.float32), -2.0, 2.0)
            else:  # "log"
                Y_raw = grp[TARGET_COLS].values.astype(np.float32)
                Y = Y_raw.copy()
                Y[:, 1:] = np.log1p(np.clip(Y_raw[:, 1:], -0.99, 5.0))

            for start in range(0, n - lookback, stride):
                end = start + lookback
                x = X[start:end]
                y = Y[end - 1]
                if np.any(np.isnan(x)) or np.any(np.isnan(y)):
                    continue
                self._xs.append(x)
                self._ys.append(y)
                self._end_dates.append(dates[end - 1])

        self._end_dates_arr = np.array(self._end_dates, dtype="datetime64[D]")

    def __len__(self) -> int:
        return len(self._xs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self._xs[idx]),
            torch.from_numpy(self._ys[idx]),
        )


# ---------------------------------------------------------------------------
# EMA sample weighting
# ---------------------------------------------------------------------------

def make_ema_sampler(
    dataset: LSTMDateRangeDataset,
    halflife_days: Optional[float],
) -> Optional[WeightedRandomSampler]:
    """Return a WeightedRandomSampler that up-weights recent sequences.

    Weight for sequence i: exp(-ln2 * age_i / halflife_days)
    where age_i = (max_date - end_date_i).days

    Returns None if halflife_days is None (uniform sampling).
    """
    if halflife_days is None or len(dataset) == 0:
        return None

    end_dates = pd.DatetimeIndex(dataset._end_dates_arr)
    max_date  = end_dates.max()
    age_days  = (max_date - end_dates).days.values.astype(np.float32)

    weights = np.exp(-np.log(2) * age_days / halflife_days).astype(np.float64)
    weights /= weights.sum()

    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(dataset),
        replacement=True,
    )
