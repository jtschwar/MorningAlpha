"""alpha ml backfill — Populate the predictions ledger from historical dataset rows.

For each active model, scores the last N trading days of snapshot dates from the
training dataset, fills in realized returns from the OHLCV cache, and appends to
predictions_ledger.parquet.  Safe to re-run: skips (ticker, scored_date, model)
combos already present in the ledger.

Usage:
    alpha ml backfill
    alpha ml backfill --lookback 63 --dataset data/training/dataset.parquet
    alpha ml backfill --models-dir models --dry-run
"""
import json
import logging
import pickle
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import rich_click as click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from morningalpha.ml.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)
console = Console()

_LEDGER_PATH   = Path("data/factors/predictions_ledger.parquet")
_RAW_CACHE_DIR = Path("data/raw_ohlcv")

# Eval horizons — must match score.py
_HORIZONS = [
    (5,  7,  "5d"),
    (13, 7,  "13d"),
    (63, 14, "63d"),
]

def _td_calendar(trading_days: int, buffer: int) -> int:
    return trading_days * 7 // 5 + buffer


def _load_models(models_dir: Path) -> list[dict]:
    """Return active (non-retired) models that have checkpoint files."""
    config_path = models_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"models/config.json not found at {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    active = []
    for m in config.get("models", []):
        if m.get("status", "candidate") == "retired":
            continue
        ckpt = models_dir / f"{m['id']}.pkl"
        if not ckpt.exists():
            logger.warning("Checkpoint not found for %s — skipping", m["id"])
            continue
        with open(ckpt, "rb") as f:
            m["_model"] = pickle.load(f)
        m["_ckpt"] = ckpt
        active.append(m)
    return active


def _score_date_slice(date_df: pd.DataFrame, model) -> np.ndarray:
    """Score a cross-section of dataset rows for one snapshot date.

    Dataset features are already rank-normalized — pass directly to model.predict().
    _predict_raw selects the right feature subset via model.feature_name_.
    """
    feat_cols = [c for c in FEATURE_COLUMNS if c in date_df.columns]
    X = date_df[feat_cols].fillna(0.0)

    try:
        trained_features = model.model.feature_name_
        X = X.reindex(columns=list(trained_features), fill_value=0.0)
    except AttributeError:
        pass

    return model.predict(X)


def _load_ohlcv(ticker: str) -> pd.DataFrame | None:
    path = _RAW_CACHE_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _get_price_at(ohlcv: pd.DataFrame, date: pd.Timestamp) -> float | None:
    """Close price on or before date."""
    idx = ohlcv.index[ohlcv.index <= date]
    if len(idx) == 0:
        return None
    return float(ohlcv.loc[idx[-1], "Close"])


def _get_forward_return(ohlcv: pd.DataFrame, date: pd.Timestamp, n_days: int) -> float | None:
    """Return at exactly n_days forward trading days from date. None if unavailable."""
    closes = ohlcv["Close"].dropna()
    idx = closes.index
    prior = idx[idx <= date]
    if len(prior) == 0:
        return None
    pos = idx.get_loc(prior[-1])
    if pos + n_days >= len(idx):
        return None
    p0 = closes.iloc[pos]
    pn = closes.iloc[pos + n_days]
    if p0 == 0 or pd.isna(p0):
        return None
    return float((pn / p0) - 1)


def _already_ledgered_key(ledger: pd.DataFrame) -> set:
    """Set of (ticker, scored_date) tuples already in the ledger."""
    if ledger.empty:
        return set()
    return set(zip(ledger["ticker"], ledger["scored_date"].dt.normalize()))


@click.command("backfill")
@click.option(
    "--dataset",
    "dataset_path",
    default="data/training/dataset.parquet",
    show_default=True,
    help="Path to the training dataset parquet.",
)
@click.option(
    "--lookback",
    default=63,
    show_default=True,
    help="Number of most-recent snapshot dates in the dataset to backfill.",
)
@click.option(
    "--models-dir",
    "models_dir",
    default="models",
    show_default=True,
    help="Directory containing model checkpoints and config.json.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print what would be written without touching the ledger.",
)
def backfill(dataset_path, lookback, models_dir, dry_run):
    """Backfill the predictions ledger from historical dataset snapshots.

    Scores the last --lookback snapshot dates using all active models, fills in
    realized returns from the OHLCV cache, and appends to predictions_ledger.parquet.
    Already-ledgered (ticker, scored_date) pairs are skipped.
    """
    dataset_path = Path(dataset_path)
    models_path  = Path(models_dir)

    if not dataset_path.exists():
        console.print(f"[red]Dataset not found: {dataset_path}[/red]")
        raise SystemExit(1)

    # ── Load dataset ──────────────────────────────────────────────────────────
    console.print(f"Loading dataset from [bold]{dataset_path}[/bold]…")
    ds = pd.read_parquet(dataset_path)
    ds["date"] = pd.to_datetime(ds["date"])

    all_dates = sorted(ds["date"].unique())
    window_dates = all_dates[-lookback:]
    console.print(
        f"Backfill window: [bold]{window_dates[0].date()}[/bold] → "
        f"[bold]{window_dates[-1].date()}[/bold] ({len(window_dates)} snapshot dates)"
    )

    ds = ds[ds["date"].isin(window_dates)].copy()

    # ── Load models ───────────────────────────────────────────────────────────
    console.print(f"Loading models from [bold]{models_path}[/bold]…")
    active_models = _load_models(models_path)
    if not active_models:
        console.print("[red]No active models found — nothing to backfill.[/red]")
        raise SystemExit(1)
    console.print(f"Active models: {[m['id'] for m in active_models]}")

    # ── Load existing ledger ──────────────────────────────────────────────────
    existing_ledger = pd.read_parquet(_LEDGER_PATH) if _LEDGER_PATH.exists() else pd.DataFrame()
    already_done = _already_ledgered_key(existing_ledger)
    console.print(f"Existing ledger rows: {len(existing_ledger):,}  |  Already-done keys: {len(already_done):,}")

    # ── Score each snapshot date ──────────────────────────────────────────────
    new_rows: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Scoring dates", total=len(window_dates))

        for snap_date in window_dates:
            snap_ts = pd.Timestamp(snap_date).normalize()
            date_df = ds[ds["date"] == snap_date].copy()

            if date_df.empty:
                progress.advance(task)
                continue

            # Skip tickers already ledgered for this date
            all_tickers = date_df["ticker"].tolist() if "ticker" in date_df.columns else []
            skip = {t for t in all_tickers if (t, snap_ts) in already_done}
            score_df = date_df[~date_df["ticker"].isin(skip)] if skip else date_df

            if score_df.empty:
                progress.advance(task)
                continue

            # Score all models for this date slice
            model_scores: dict[str, np.ndarray] = {}
            for m in active_models:
                try:
                    model_scores[m["id"]] = _score_date_slice(score_df, m["_model"])
                except Exception as exc:
                    logger.warning("Scoring failed for %s on %s: %s", m["id"], snap_date, exc)

            if not model_scores:
                progress.advance(task)
                continue

            # Build one row per ticker with all model scores merged
            for i, (_, row) in enumerate(score_df.iterrows()):
                ticker = row.get("ticker", "")
                if not ticker:
                    continue

                ohlcv = _load_ohlcv(ticker)
                price_at = _get_price_at(ohlcv, snap_ts) if ohlcv is not None else None

                ledger_row: dict = {
                    "ticker":         ticker,
                    "scored_date":    snap_ts,
                    "price_at_score": price_at,
                }

                for model_id, raw in model_scores.items():
                    ledger_row[f"raw_{model_id}"] = float(raw[i])

                for td, buf, suffix in _HORIZONS:
                    eval_after = snap_ts + timedelta(days=_td_calendar(td, buf))
                    ledger_row[f"eval_after_{suffix}"] = eval_after
                    fwd = _get_forward_return(ohlcv, snap_ts, td) if ohlcv is not None else None
                    ledger_row[f"realized_return_{suffix}"] = fwd
                    ledger_row[f"matured_{suffix}"] = fwd is not None

                new_rows.append(ledger_row)

            progress.advance(task)

    if not new_rows:
        console.print("[yellow]No new rows to append — ledger is already up to date.[/yellow]")
        return

    new_df = pd.DataFrame(new_rows)
    console.print(f"\nNew rows generated: [bold]{len(new_df):,}[/bold]")

    # Summarise maturity
    for _, _, suffix in _HORIZONS:
        col = f"matured_{suffix}"
        if col in new_df.columns:
            pct = new_df[col].mean() * 100
            console.print(f"  {suffix} matured: {pct:.0f}%")

    if dry_run:
        console.print("\n[yellow]--dry-run: ledger not written.[/yellow]")
        console.print(new_df.head(10).to_string())
        return

    # ── Write ledger ─────────────────────────────────────────────────────────
    _LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([existing_ledger, new_df], ignore_index=True) if not existing_ledger.empty else new_df
    combined.to_parquet(_LEDGER_PATH, index=False)
    console.print(f"\n[bold green]Ledger saved → {_LEDGER_PATH}[/bold green]  ({len(combined):,} total rows)")
