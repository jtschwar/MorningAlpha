"""LSTM model exploration: size vs. CPU inference speed vs. validation IC.

Runs a grid of model configurations. For each one it measures:
  - Parameter count and checkpoint size
  - CPU inference latency (single ticker, 6 MC paths) — the production scenario
  - Batch inference latency (500 tickers at once)
  - Quick training IC (5 epochs on a 50k-row sample) — rough signal quality proxy

Use this to find the smallest model that still has acceptable IC, then plug
those dimensions into  alpha ml train lstm --hidden X --layers Y.

Usage
-----
    python scripts/lstm_explore.py
    python scripts/lstm_explore.py --dataset data/training/dataset.parquet
    python scripts/lstm_explore.py --epochs 10 --sample 100000
    python scripts/lstm_explore.py --no-train   # latency-only, skips training
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

# Make sure morningalpha is importable from repo root
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))

from morningalpha.ml.lstm_model import StockPriceLSTM, LSTM_HORIZONS
from morningalpha.ml.features import FEATURE_COLUMNS, MARKET_CONTEXT_COLUMNS
from morningalpha.ml.train_lstm import (
    LSTMSequenceDataset, TARGET_COLS,
    fit_feature_scaler, apply_feature_scaler, COLS_TO_SCALE,
)

# ---------------------------------------------------------------------------
# Config grid
# ---------------------------------------------------------------------------

GRID = [
    # (hidden_dim, num_layers)  — dropout fixed at 0.3
    (64,  1),
    (128, 1),
    (128, 2),
    (256, 2),   # <-- current default
    (256, 3),
    (512, 2),
    (512, 3),
]

LOOKBACK   = 60
N_PATHS    = 6
DROPOUT    = 0.3
N_WARMUP   = 20    # inference timing warm-up reps
N_TIMING   = 100   # inference timing measurement reps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def checkpoint_size_kb(model: nn.Module) -> float:
    """Estimate .pt file size in KB by serialising to an in-memory buffer."""
    buf = io.BytesIO()
    torch.save({"model_state_dict": model.state_dict()}, buf)
    return buf.tell() / 1024


def measure_inference(model: StockPriceLSTM, n_features: int) -> dict:
    """Measure CPU inference latency for both single-ticker and batch scenarios."""
    device = torch.device("cpu")
    model.eval().to(device)

    # Single ticker (the live UI scenario): [1, lookback, n_features]
    x_single = torch.randn(1, LOOKBACK, n_features)
    # Warm-up
    for _ in range(N_WARMUP):
        _ = model.predict_paths(x_single, n_paths=N_PATHS)

    t0 = time.perf_counter()
    for _ in range(N_TIMING):
        _ = model.predict_paths(x_single, n_paths=N_PATHS)
    single_ms = (time.perf_counter() - t0) / N_TIMING * 1000

    # Batch of 500 (full scoring run)
    x_batch = torch.randn(500, LOOKBACK, n_features)
    model.train()  # MC dropout active
    # Warm-up
    for _ in range(5):
        with torch.no_grad():
            _ = model(x_batch)
    t0 = time.perf_counter()
    reps = 10
    for _ in range(reps):
        with torch.no_grad():
            _ = model(x_batch)
    batch_ms = (time.perf_counter() - t0) / reps * 1000
    model.eval()

    return {"single_ms": single_ms, "batch500_ms": batch_ms}


def quick_train(
    model: StockPriceLSTM,
    train_ds,
    val_ds,
    epochs: int,
    batch_size: int = 256,
) -> dict:
    """Run a short training loop and return val IC per horizon."""
    device = torch.device("cpu")
    model.to(device)

    loader_tr = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    loader_va = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    criterion = nn.HuberLoss(delta=0.05)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(epochs):
        model.train()
        for x, y in loader_tr:
            opt.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader_va:
            preds.append(model(x.to(device)).cpu().numpy())
            targets.append(y.numpy())

    P = np.concatenate(preds)    # [N, n_horizons]
    T = np.concatenate(targets)

    ics = {}
    for i, h in enumerate(LSTM_HORIZONS):
        mask = ~(np.isnan(P[:, i]) | np.isnan(T[:, i]))
        if mask.sum() > 10:
            ics[f"{h}d"] = round(float(spearmanr(P[mask, i], T[mask, i]).correlation), 4)
        else:
            ics[f"{h}d"] = float("nan")
    return ics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_architecture_grid(
    n_features: int,
    do_train: bool,
    train_ds,
    val_ds,
    epochs: int,
) -> List[dict]:
    """Run the GRID of (hidden_dim, num_layers) configs. Returns list of result dicts."""
    col_w = {"config": 18, "params": 10, "size_kb": 9, "single_ms": 11, "batch_ms": 10}
    horizon_headers = ["1d", "5d", "10d", "21d", "63d"]

    header = (
        f"{'Config':<{col_w['config']}} "
        f"{'Params':>{col_w['params']}} "
        f"{'Size(KB)':>{col_w['size_kb']}} "
        f"{'Single(ms)':>{col_w['single_ms']}} "
        f"{'Batch500(ms)':>{col_w['batch_ms']}}"
    )
    if do_train:
        for h in horizon_headers:
            header += f"  IC-{h:>3}"
    print(header)
    print("-" * len(header))

    results = []
    for hidden_dim, num_layers in GRID:
        label = f"h{hidden_dim}-L{num_layers}"

        model = StockPriceLSTM(
            n_features=n_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=DROPOUT,
            horizon_days=LSTM_HORIZONS,
        )

        params    = count_params(model)
        size_kb   = checkpoint_size_kb(model)
        latency   = measure_inference(model, n_features)
        single_ms = latency["single_ms"]
        batch_ms  = latency["batch500_ms"]

        ics: Optional[dict] = None
        if do_train and train_ds and val_ds:
            model = StockPriceLSTM(
                n_features=n_features,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=DROPOUT,
                horizon_days=LSTM_HORIZONS,
            )
            ics = quick_train(model, train_ds, val_ds, epochs=epochs)

        row = (
            f"{label:<{col_w['config']}} "
            f"{params:>{col_w['params']},} "
            f"{size_kb:>{col_w['size_kb']}.0f} "
            f"{single_ms:>{col_w['single_ms']}.2f} "
            f"{batch_ms:>{col_w['batch_ms']}.1f}"
        )
        if do_train and ics:
            for h in ["1d", "5d", "10d", "21d", "63d"]:
                v = ics.get(h, float("nan"))
                row += f"  {v:>6.4f}" if not np.isnan(v) else f"  {'n/a':>6}"

        print(row)
        results.append({
            "config": label, "hidden": hidden_dim, "layers": num_layers,
            "params": params, "size_kb": size_kb,
            "single_ms": single_ms, "batch500_ms": batch_ms,
            **(ics or {}),
        })

    return results


def _print_recommendation(results: List[dict]) -> None:
    LATENCY_BUDGET_MS = 50.0
    fast = [r for r in results if r["single_ms"] <= LATENCY_BUDGET_MS]
    if fast:
        print(f"Configs within {LATENCY_BUDGET_MS}ms single-ticker budget:")
        for r in fast:
            ic_63 = r.get("63d", float("nan"))
            ic_str = f"  IC-63d={ic_63:.4f}" if not np.isnan(ic_63) else ""
            print(f"  {r['config']:<18} {r['single_ms']:.2f}ms  {r['params']:,} params{ic_str}")
    else:
        print("All configs exceed the latency budget — consider reducing lookback or n_paths.")


def main():
    parser = argparse.ArgumentParser(description="LSTM model size vs. speed vs. IC exploration")
    parser.add_argument("--dataset", default="data/training/dataset.parquet")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Quick-train epochs per config (0 = latency-only)")
    parser.add_argument("--sample", type=int, default=50_000,
                        help="Max rows to use for quick training (keeps it fast)")
    parser.add_argument("--no-train", dest="no_train", action="store_true",
                        help="Skip training entirely — latency and size only")
    parser.add_argument(
        "--target-mode", dest="target_mode", default="log",
        choices=["log", "clip", "rank", "all"],
        help=(
            "Target encoding to benchmark: log (default), clip (±2.0), "
            "rank (cross-sectional rank), or all (sweeps all three in sequence)"
        ),
    )
    args = parser.parse_args()

    do_train = not args.no_train and args.epochs > 0
    modes_to_run = ["log", "clip", "rank"] if args.target_mode == "all" else [args.target_mode]

    # --- Load and prepare dataset once (shared across all target modes) ---
    feat_cols: List[str] = []
    df = None

    if do_train:
        print(f"Loading dataset: {args.dataset}")
        df = pd.read_parquet(args.dataset)
        df["date"] = pd.to_datetime(df["date"])

        feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

        # Drop zero-variance features (e.g. has_fundamentals=1 always)
        df_train_full = df[df["split"] == "train"]
        zero_var = [c for c in feat_cols if df_train_full[c].std() == 0]
        if zero_var:
            print(f"  Dropping {len(zero_var)} constant feature(s): {zero_var}")
            feat_cols = [c for c in feat_cols if c not in zero_var]

        print(f"  {len(df):,} rows  |  {len(feat_cols)} features")

        # Fit StandardScaler once on train split, apply to all splits
        scale_cols = [c for c in COLS_TO_SCALE if c in feat_cols]
        scaler, scaled_cols = fit_feature_scaler(df_train_full, scale_cols)
        df = apply_feature_scaler(df, scaler, scaled_cols)
        print(f"  Scaled {len(scaled_cols)} market-context/categorical columns")

        # Sample to keep quick-train fast
        if len(df) > args.sample:
            df = df.sort_values("date").tail(args.sample)
            print(f"  Sampled to {len(df):,} most recent rows")

        if "forward_1d" not in df.columns:
            df["forward_1d"] = df.get("forward_5d", np.nan)
    else:
        print("Loading feature columns from dataset...")
        sample_df = pd.read_parquet(args.dataset)
        feat_cols = [c for c in FEATURE_COLUMNS if c in sample_df.columns]
        df_train_full = sample_df[sample_df["split"] == "train"]
        zero_var = [c for c in feat_cols if df_train_full[c].std() == 0]
        feat_cols = [c for c in feat_cols if c not in zero_var]
        print(f"  {len(feat_cols)} features\n")

    n_features = len(feat_cols) if feat_cols else 77  # fallback

    # --- Run architecture grid for each target mode ---
    all_results: dict = {}  # mode -> List[dict]

    for mode in modes_to_run:
        print(f"\n{'=' * 70}")
        print(f"  Target mode: {mode}")
        print(f"{'=' * 70}")

        train_ds = val_ds = None
        if do_train and df is not None:
            ds_kwargs = dict(lookback=LOOKBACK, stride=5, target_mode=mode)
            train_ds = LSTMSequenceDataset(df, feat_cols, split="train", **ds_kwargs)
            val_ds   = LSTMSequenceDataset(df, feat_cols, split="val",   **ds_kwargs)
            print(f"  train={len(train_ds):,}  val={len(val_ds):,}\n")
            if len(train_ds) == 0:
                print("  No training samples for this mode — skipping.\n")
                continue

        results = _run_architecture_grid(n_features, do_train, train_ds, val_ds, args.epochs)
        all_results[mode] = results
        print()
        _print_recommendation(results)

    # --- Cross-mode IC-63d comparison (when sweeping all modes) ---
    if args.target_mode == "all" and do_train and all_results:
        print(f"\n{'=' * 70}")
        print("  Cross-mode IC-63d comparison (best config per mode)")
        print(f"{'=' * 70}")
        print(f"  {'Mode':<8}  {'Best config':<18}  {'IC-63d':>8}  {'Single(ms)':>10}")
        print(f"  {'-'*8}  {'-'*18}  {'-'*8}  {'-'*10}")
        for mode, results in all_results.items():
            ranked = sorted(
                [r for r in results if not np.isnan(r.get("63d", float("nan")))],
                key=lambda r: r.get("63d", float("-inf")),
                reverse=True,
            )
            if ranked:
                best = ranked[0]
                print(
                    f"  {mode:<8}  {best['config']:<18}  "
                    f"{best['63d']:>8.4f}  {best['single_ms']:>10.2f}ms"
                )

    print()
    print("To train with a specific config:")
    print("  alpha ml train lstm --hidden <hidden_dim> --layers <num_layers> --target-mode <mode>")


if __name__ == "__main__":
    main()
