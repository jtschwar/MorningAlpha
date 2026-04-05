"""LSTM training methodology ablation study.

Sweeps three dimensions to find the best training recipe:
  - split_strategy : "simple" (train/val/test split column) vs "walk_forward" (expanding WFCV)
  - ema_halflife   : None | 365 | 180 | 90  (days; None = uniform sampling)
  - target_mode    : "log" | "clip" | "rank"

Architecture is fixed at h128-L2 (small, CPU-inference-friendly).
Results are written to results/lstm_ablation/results.csv.

Usage
-----
    python scripts/lstm_ablation.py
    python scripts/lstm_ablation.py --dataset data/training/dataset.parquet
    python scripts/lstm_ablation.py --n-folds 6 --epochs 50
    python scripts/lstm_ablation.py --hidden 256 --layers 2   # larger model
"""
from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))

from morningalpha.ml.lstm_model import StockPriceLSTM, LSTM_HORIZONS
from morningalpha.ml.features import FEATURE_COLUMNS, MARKET_CONTEXT_COLUMNS
from morningalpha.ml.train_lstm import (
    LSTMSequenceDataset, TARGET_COLS,
    fit_feature_scaler, apply_feature_scaler, COLS_TO_SCALE,
)
from morningalpha.ml.lstm_wfcv import (
    make_wfcv_folds, LSTMDateRangeDataset, make_ema_sampler,
)


# ---------------------------------------------------------------------------
# Ablation grid
# ---------------------------------------------------------------------------

SPLIT_STRATEGIES = ["simple", "walk_forward"]
EMA_HALFLIVES    = [None, 365, 180, 90]   # days; None = uniform
TARGET_MODES     = ["log", "clip", "rank"]

LOOKBACK   = 60
STRIDE     = 3
DROPOUT    = 0.3
PATIENCE   = 8
BATCH_SIZE = 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"  Device: CUDA ({name})")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("  Device: Apple MPS")
        return torch.device("mps")
    print("  Device: CPU")
    return torch.device("cpu")


def _rank_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    mask = ~(np.isnan(pred) | np.isnan(actual))
    if mask.sum() < 10:
        return float("nan")
    return float(spearmanr(pred[mask], actual[mask]).correlation)


def _evaluate(
    model: StockPriceLSTM,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Return IC per horizon, keyed by e.g. 'ic_5d'."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).cpu().numpy())
            targets.append(y.numpy())
    if not preds:
        return {f"ic_{h}d": float("nan") for h in LSTM_HORIZONS}

    P = np.concatenate(preds)
    T = np.concatenate(targets)
    return {
        f"ic_{h}d": _rank_ic(P[:, i], T[:, i])
        for i, h in enumerate(LSTM_HORIZONS)
    }


def _train_one_fold(
    model: StockPriceLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
) -> Tuple[StockPriceLSTM, Dict[str, float]]:
    """Train with early stopping. Returns (best_model, val_metrics)."""
    criterion = nn.HuberLoss(delta=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    best_val_loss = float("inf")
    best_state    = None
    patience_cnt  = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation loss for early stopping
        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item() * len(x)
                n += len(x)
        val_loss = val_loss / max(n, 1)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, _evaluate(model, val_loader, device)


def _make_model(n_features: int, hidden: int, layers: int, device: torch.device) -> StockPriceLSTM:
    return StockPriceLSTM(
        n_features=n_features,
        hidden_dim=hidden,
        num_layers=layers,
        dropout=DROPOUT,
        horizon_days=LSTM_HORIZONS,
    ).to(device)


# ---------------------------------------------------------------------------
# Run one config
# ---------------------------------------------------------------------------

def run_config(
    df: pd.DataFrame,
    feat_cols: List[str],
    split_strategy: str,
    ema_halflife: Optional[int],
    target_mode: str,
    n_folds: int,
    epochs: int,
    hidden: int,
    layers: int,
    workers: int,
    device: torch.device,
) -> Dict:
    """Run a single ablation config. Returns a result dict."""
    ema_label = str(ema_halflife) if ema_halflife else "none"
    label = f"{split_strategy} | ema={ema_label}d | target={target_mode}"
    print(f"\n  {label}")

    ds_kwargs = dict(lookback=LOOKBACK, stride=STRIDE, target_mode=target_mode)
    fold_ics: List[Dict[str, float]] = []
    t0 = time.perf_counter()

    if split_strategy == "simple":
        # ---- Simple train/val/test split --------------------------------
        train_ds = LSTMSequenceDataset(df, feat_cols, split="train", **ds_kwargs)
        val_ds   = LSTMSequenceDataset(df, feat_cols, split="val",   **ds_kwargs)
        test_ds  = LSTMSequenceDataset(df, feat_cols, split="test",  **ds_kwargs)

        print(f"    train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")
        if len(train_ds) == 0 or len(val_ds) == 0:
            print("    [SKIP] empty split")
            return {}

        # EMA weighting not applicable to the split-column dataset
        # (we'd need date info; skip EMA for simple strategy)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=workers, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=True)

        model = _make_model(len(feat_cols), hidden, layers, device)
        model, val_ics = _train_one_fold(model, train_loader, val_loader, epochs, device)
        test_ics = _evaluate(model, test_loader, device)

        # Report test IC as the primary metric (comparable to last WFCV fold)
        fold_ics = [test_ics]

    else:
        # ---- Walk-forward CV (expanding window) -------------------------
        folds = make_wfcv_folds(df, n_folds=n_folds, embargo_days=10)
        print(f"    {len(folds)} WFCV folds")

        for fold in folds:
            tr_ds = LSTMDateRangeDataset(
                df, feat_cols,
                start_date=fold["train_start"],
                end_date=fold["train_end"],
                **ds_kwargs,
            )
            va_ds = LSTMDateRangeDataset(
                df, feat_cols,
                start_date=fold["val_start"],
                end_date=fold["val_end"],
                **ds_kwargs,
            )

            if len(tr_ds) == 0 or len(va_ds) == 0:
                print(f"      fold {fold['fold']}: empty — skipping")
                continue

            sampler = make_ema_sampler(tr_ds, ema_halflife)
            if sampler:
                train_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=workers, pin_memory=True)
            else:
                train_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=workers, pin_memory=True)
            val_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=True)

            model = _make_model(len(feat_cols), hidden, layers, device)
            model, ics = _train_one_fold(model, train_loader, val_loader, epochs, device)
            fold_ics.append(ics)

            ic_63 = ics.get("ic_63d", float("nan"))
            ic_5  = ics.get("ic_5d",  float("nan"))
            print(
                f"      fold {fold['fold']}  "
                f"({str(fold['val_start'])[:10]} → {str(fold['val_end'])[:10]})  "
                f"IC-5d={ic_5:.4f}  IC-63d={ic_63:.4f}"
            )

    elapsed = time.perf_counter() - t0

    if not fold_ics:
        return {}

    # Aggregate across folds
    result = {
        "split_strategy": split_strategy,
        "ema_halflife":   ema_label,
        "target_mode":    target_mode,
        "n_folds":        len(fold_ics),
        "elapsed_s":      round(elapsed, 1),
    }
    for h in LSTM_HORIZONS:
        key = f"ic_{h}d"
        vals = [f[key] for f in fold_ics if not np.isnan(f.get(key, float("nan")))]
        result[f"mean_{key}"]   = round(float(np.mean(vals)),   4) if vals else float("nan")
        result[f"std_{key}"]    = round(float(np.std(vals)),    4) if vals else float("nan")
        result[f"min_{key}"]    = round(float(np.min(vals)),    4) if vals else float("nan")

    ic_63 = result.get("mean_ic_63d", float("nan"))
    ic_5  = result.get("mean_ic_5d",  float("nan"))
    print(f"    → mean IC-5d={ic_5:.4f}  IC-63d={ic_63:.4f}  ({elapsed:.0f}s)")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LSTM training methodology ablation")
    parser.add_argument("--dataset",  default="data/training/dataset.parquet")
    parser.add_argument("--output-dir", dest="output_dir", default="results/lstm_ablation")
    parser.add_argument("--n-folds",  dest="n_folds",  type=int, default=6)
    parser.add_argument("--epochs",   type=int, default=50)
    parser.add_argument("--hidden",   type=int, default=128, help="LSTM hidden dim (default 128 for CPU inference)")
    parser.add_argument("--layers",   type=int, default=2)
    parser.add_argument("--workers",  type=int, default=4)
    parser.add_argument(
        "--split",   dest="splits",   nargs="+", default=SPLIT_STRATEGIES,
        choices=SPLIT_STRATEGIES,  help="Split strategies to run (default: all)"
    )
    parser.add_argument(
        "--target",  dest="targets",  nargs="+", default=TARGET_MODES,
        choices=TARGET_MODES,        help="Target modes to run (default: all)"
    )
    parser.add_argument(
        "--ema",     dest="emas",     nargs="+", default=["none", "365", "180", "90"],
        help="EMA half-lives to run: 'none' or integer days (default: all)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse EMA values
    ema_values: List[Optional[int]] = []
    for v in args.emas:
        ema_values.append(None if v == "none" else int(v))

    device = _select_device()

    # --- Load and prepare dataset ---
    print(f"\nLoading dataset: {args.dataset}")
    df = pd.read_parquet(args.dataset)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  {len(df):,} rows  |  {df['date'].min().date()} → {df['date'].max().date()}  |  {df['ticker'].nunique():,} tickers")

    feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    df_train_full = df[df["split"] == "train"]

    zero_var = [c for c in feat_cols if df_train_full[c].std() == 0]
    if zero_var:
        print(f"  Dropping {len(zero_var)} constant feature(s): {zero_var}")
        feat_cols = [c for c in feat_cols if c not in zero_var]
    print(f"  {len(feat_cols)} features")

    scale_cols = [c for c in COLS_TO_SCALE if c in feat_cols]
    scaler, scaled_cols = fit_feature_scaler(df_train_full, scale_cols)
    df = apply_feature_scaler(df, scaler, scaled_cols)
    print(f"  Scaled {len(scaled_cols)} columns")

    if "forward_1d" not in df.columns:
        df["forward_1d"] = df.get("forward_5d", np.nan)

    # --- Build ablation grid ---
    configs = list(itertools.product(args.splits, ema_values, args.targets))
    n_total = len(configs)
    print(f"\nAblation grid: {n_total} configs  ({args.n_folds} WFCV folds, {args.epochs} max epochs)")
    print(f"Architecture: h{args.hidden}-L{args.layers}  dropout={DROPOUT}")
    print(f"Output: {output_dir}/results.csv\n")

    all_results = []
    csv_path = output_dir / "results.csv"

    # Write CSV header
    header_written = False

    for i, (split, ema, target) in enumerate(configs, 1):
        print(f"[{i}/{n_total}]", end=" ")
        result = run_config(
            df, feat_cols,
            split_strategy=split,
            ema_halflife=ema,
            target_mode=target,
            n_folds=args.n_folds,
            epochs=args.epochs,
            hidden=args.hidden,
            layers=args.layers,
            workers=args.workers,
            device=device,
        )
        if not result:
            continue

        all_results.append(result)

        # Append to CSV incrementally so we have results even if job is killed
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            if not header_written:
                writer.writeheader()
                header_written = True
            writer.writerow(result)

    # --- Summary table ---
    if not all_results:
        print("\nNo results collected.")
        return

    results_df = pd.DataFrame(all_results)

    print(f"\n{'=' * 90}")
    print("  ABLATION SUMMARY — ranked by mean IC-63d")
    print(f"{'=' * 90}")
    ranked = results_df.sort_values("mean_ic_63d", ascending=False)
    cols_to_show = ["split_strategy", "ema_halflife", "target_mode",
                    "mean_ic_5d", "mean_ic_63d", "std_ic_63d", "n_folds", "elapsed_s"]
    cols_to_show = [c for c in cols_to_show if c in ranked.columns]
    print(ranked[cols_to_show].to_string(index=False))

    print(f"\n{'=' * 90}")
    print("  TOP CONFIGS (IC-5d and IC-63d)")
    print(f"{'=' * 90}")
    for label, sort_col in [("IC-5d", "mean_ic_5d"), ("IC-63d", "mean_ic_63d")]:
        if sort_col not in ranked.columns:
            continue
        best = ranked.sort_values(sort_col, ascending=False).iloc[0]
        print(
            f"  Best {label}: {best['split_strategy']} | ema={best['ema_halflife']}d "
            f"| target={best['target_mode']}  →  {best[sort_col]:.4f}"
        )

    print(f"\nFull results saved to: {csv_path}")
    print("To train with the winning config:")
    print("  alpha ml train lstm --hidden <H> --layers <L> --target-mode <mode> --name lstm_v1")


if __name__ == "__main__":
    main()
