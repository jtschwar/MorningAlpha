"""alpha ml backfill — Populate the predictions ledger from historical dataset snapshots.

For each active LGBM model, scores every snapshot date in the training dataset
that falls AFTER the model's train_cutoff (out-of-sample period only).  Realized
returns are read directly from the dataset's forward_5d / forward_21d / forward_63d
columns — no OHLCV cache required.

All appended rows are marked is_backfill=True and excluded from calibration model
training (they suffer selection bias — v10 was chosen *because* it performed well on
this test period).  They ARE included in the IC timeseries so you can see the model's
historical accuracy curve from day one.

Usage:
    alpha ml backfill
    alpha ml backfill --dataset data/training/dataset.parquet
    alpha ml backfill --models-dir models --dry-run
"""
import json
import logging
import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import rich_click as click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from morningalpha.ml.features import FEATURE_COLUMNS, SPREAD_TO_ML

logger = logging.getLogger(__name__)
console = Console()

_LEDGER_PATH        = Path("data/factors/predictions_ledger.parquet")
_IC_TIMESERIES_PATH = Path("data/factors/live_ic_timeseries.parquet")
_ALERTS_PATH        = Path("data/factors/model_alerts.json")

# Eval horizons — must match score.py
_HORIZONS = [
    (5,  7,  "5d"),
    (21, 7,  "21d"),
    (63, 14, "63d"),
]

_SECTOR_CODES: dict[str, int] = {
    "Technology": 1, "Healthcare": 2, "Financial Services": 3,
    "Consumer Cyclical": 4, "Communication Services": 5, "Industrials": 6,
    "Consumer Defensive": 7, "Energy": 8, "Utilities": 9,
    "Real Estate": 10, "Basic Materials": 11,
}
# Inverse: encoded int → sector name (for lookup)
_INT_TO_SECTOR = {v: k for k, v in _SECTOR_CODES.items()}


def _load_active_lgbm_models(models_dir: Path) -> list[dict]:
    """Return active (non-retired) LGBM models with loaded checkpoints and train_cutoffs."""
    config_path = models_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"models/config.json not found at {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    active = []
    for m in config.get("models", []):
        if m.get("status") == "retired":
            continue
        if m.get("type", "lgbm") != "lgbm":
            continue  # backfill only supports LGBM for now
        ckpt = models_dir / f"{m['id']}.pkl"
        if not ckpt.exists():
            logger.warning("Checkpoint not found for %s — skipping", m["id"])
            continue

        # Load train_cutoff from feature_config.json, fall back to None
        fc_path = models_dir / f"{m['id']}_feature_config.json"
        train_cutoff = None
        if fc_path.exists():
            with open(fc_path) as f:
                fc = json.load(f)
            tc = fc.get("train_cutoff")
            if tc:
                train_cutoff = pd.Timestamp(tc)

        with open(ckpt, "rb") as f:
            model_obj = pickle.load(f)

        active.append({
            **m,
            "_model":       model_obj,
            "_train_cutoff": train_cutoff,
        })

    return active


def _score_date_slice(date_df: pd.DataFrame, model) -> np.ndarray:
    """Score a cross-section of dataset rows for one snapshot date.

    Dataset features are already preprocessed (rank-normalized, winsorized) —
    pass directly to model.predict() using the model's own feature name list.
    """
    try:
        trained_features = list(model.model.feature_name_)
    except AttributeError:
        trained_features = [c for c in FEATURE_COLUMNS if c in date_df.columns]

    X = date_df.reindex(columns=trained_features, fill_value=0.0).fillna(0.0)
    return model.predict(X)


def _fetch_market_context_history(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, dict]:
    """Fetch historical SPY 21-day trailing returns and VIX closes for a date range.

    Returns a dict mapping date-string (YYYY-MM-DD) →
    {'market_return_21d': float, 'vix_at_prediction': float}.
    Falls back to NaN for any date that can't be computed.
    """
    result: dict[str, dict] = {}
    try:
        import yfinance as yf

        fetch_start = (start - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
        fetch_end   = (end + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

        spy = yf.download("SPY", start=fetch_start, end=fetch_end, progress=False, auto_adjust=True)
        vix = yf.download("^VIX", start=fetch_start, end=fetch_end, progress=False, auto_adjust=True)

        def _series(df, col="Close"):
            s = df[col] if col in df.columns else df.iloc[:, 0]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            return s.dropna()

        spy_close = _series(spy)
        vix_close = _series(vix)

        # For each business day in the range, compute 21d trailing return and VIX close
        for ts in pd.bdate_range(start, end):
            date_str = ts.strftime("%Y-%m-%d")
            spy_ret = float("nan")
            vix_val = float("nan")

            prior_spy = spy_close[spy_close.index <= ts]
            if len(prior_spy) >= 22:
                spy_ret = float(prior_spy.iloc[-1] / prior_spy.iloc[-22] - 1)

            prior_vix = vix_close[vix_close.index <= ts]
            if len(prior_vix) >= 1:
                vix_val = float(prior_vix.iloc[-1])

            result[date_str] = {"market_return_21d": spy_ret, "vix_at_prediction": vix_val}

    except Exception as exc:
        logger.warning("Could not fetch market context history (%s) — will use NaN", exc)

    return result


def _already_done_keys(ledger: pd.DataFrame) -> set:
    """Set of (ticker, date_str) pairs already in the ledger as backfill rows."""
    if ledger.empty:
        return set()
    mask = ledger.get("is_backfill", pd.Series(False, index=ledger.index)).fillna(False)
    sub = ledger[mask]
    if sub.empty:
        return set()
    return set(zip(sub["ticker"], sub["scored_date"].dt.strftime("%Y-%m-%d")))


@click.command("backfill")
@click.option(
    "--dataset",
    "dataset_path",
    default="data/training/dataset.parquet",
    show_default=True,
    help="Path to the training dataset parquet.",
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
@click.option(
    "--skip-ic",
    "skip_ic",
    is_flag=True,
    default=False,
    help="Skip IC timeseries update after writing ledger.",
)
def backfill(dataset_path, models_dir, dry_run, skip_ic):
    """Backfill the predictions ledger from out-of-sample dataset snapshots.

    For each active LGBM model, scores every snapshot date AFTER the model's
    train_cutoff using the training dataset, fills realized returns from the
    dataset's forward_*d columns, and appends to predictions_ledger.parquet.

    All appended rows are marked is_backfill=True.  They populate the IC
    timeseries for monitoring but are excluded from calibration model training.
    """
    dataset_path = Path(dataset_path)
    models_path  = Path(models_dir)

    if not dataset_path.exists():
        console.print(f"[red]Dataset not found: {dataset_path}[/red]")
        raise SystemExit(1)

    # ── Load models ───────────────────────────────────────────────────────────
    console.print(f"Loading models from [bold]{models_path}[/bold]…")
    active_models = _load_active_lgbm_models(models_path)
    if not active_models:
        console.print("[red]No active LGBM models found — nothing to backfill.[/red]")
        raise SystemExit(1)

    for m in active_models:
        tc = m["_train_cutoff"]
        console.print(
            f"  [cyan]{m['id']}[/cyan]  train_cutoff={tc.date() if tc else 'unknown'}"
        )

    # ── Load dataset ──────────────────────────────────────────────────────────
    console.print(f"\nLoading dataset from [bold]{dataset_path}[/bold]…")
    ds = pd.read_parquet(dataset_path)
    ds["date"] = pd.to_datetime(ds["date"])

    # Only keep rows that are post-cutoff for at least one model
    earliest_cutoff = min(
        (m["_train_cutoff"] for m in active_models if m["_train_cutoff"] is not None),
        default=None,
    )
    if earliest_cutoff is not None:
        ds = ds[ds["date"] > earliest_cutoff].copy()

    all_dates = sorted(ds["date"].unique())
    if not all_dates:
        console.print("[yellow]No post-cutoff dates in the dataset — nothing to backfill.[/yellow]")
        return

    console.print(
        f"Post-cutoff window: [bold]{all_dates[0].date()}[/bold] → "
        f"[bold]{all_dates[-1].date()}[/bold]  ({len(all_dates)} snapshot dates)"
    )

    # ── Fetch historical market context (SPY 21d + VIX) ─────────────────────
    console.print("\nFetching historical market context (SPY + VIX)…")
    market_ctx_cache = _fetch_market_context_history(
        pd.Timestamp(all_dates[0]),
        pd.Timestamp(all_dates[-1]),
    )
    console.print(f"  Market context entries: {len(market_ctx_cache)}")

    # ── Load existing ledger ──────────────────────────────────────────────────
    existing_ledger = pd.read_parquet(_LEDGER_PATH) if _LEDGER_PATH.exists() else pd.DataFrame()
    already_done    = _already_done_keys(existing_ledger)
    console.print(f"Existing ledger: {len(existing_ledger):,} rows  |  backfill already done: {len(already_done):,} keys")

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
        task = progress.add_task("Scoring dates", total=len(all_dates))

        for snap_date in all_dates:
            snap_ts  = pd.Timestamp(snap_date).normalize()
            date_str = snap_ts.strftime("%Y-%m-%d")
            date_df  = ds[ds["date"] == snap_date].copy()

            if date_df.empty:
                progress.advance(task)
                continue

            ctx = market_ctx_cache.get(date_str, {})

            # Score each model that has this date in its post-cutoff window
            for m in active_models:
                tc = m["_train_cutoff"]
                if tc is not None and snap_ts <= tc:
                    continue  # skip training-period rows for this model

                # Skip rows already backfilled for this (ticker, date, model) combo
                model_date_df = date_df[
                    ~date_df["ticker"].apply(lambda t: (t, date_str) in already_done)
                ] if already_done else date_df

                if model_date_df.empty:
                    continue

                try:
                    raw_scores = _score_date_slice(model_date_df, m["_model"])
                except Exception as exc:
                    logger.warning("Scoring failed for %s on %s: %s", m["id"], snap_date, exc)
                    continue

                # Build ledger rows for this model × date
                for i, (_, row) in enumerate(model_date_df.iterrows()):
                    ticker = row.get("ticker", "")
                    if not ticker:
                        continue

                    # Sector code from encoded integer
                    sector_int = int(row.get("sector", 0) or 0)
                    sector_name = _INT_TO_SECTOR.get(sector_int, "")
                    sector_code = _SECTOR_CODES.get(sector_name, 0)

                    # Momentum bucket (quintile) — compute cross-sectionally per date
                    # We'll fill this after building all rows for this date slice
                    mom_val = float(row.get("momentum_12_1", 0.0) or 0.0)

                    ledger_row: dict = {
                        "ticker":             ticker,
                        "scored_date":        snap_ts,
                        "price_at_score":     float(row.get("close", np.nan)) if "close" in row.index else np.nan,
                        f"raw_{m['id']}":     float(raw_scores[i]),
                        "sector_code":        np.int8(sector_code),
                        "_momentum_12_1":     mom_val,  # temp column for bucket computation
                        "market_return_21d":  ctx.get("market_return_21d", float("nan")),
                        "vix_at_prediction":  ctx.get("vix_at_prediction", float("nan")),
                        "is_backfill":        True,
                    }

                    # Eval-after dates + realized returns from dataset forward columns
                    for td, buf, suffix in _HORIZONS:
                        cal_days = td * 7 // 5 + buf
                        ledger_row[f"eval_after_{suffix}"] = snap_ts + timedelta(days=cal_days)
                        fwd_col = f"forward_{suffix}"
                        fwd_val = row.get(fwd_col, np.nan)
                        if pd.isna(fwd_val):
                            fwd_val = np.nan
                        ledger_row[f"realized_return_{suffix}"] = fwd_val
                        ledger_row[f"matured_{suffix}"] = not pd.isna(fwd_val)

                    new_rows.append(ledger_row)

            progress.advance(task)

    if not new_rows:
        console.print("[yellow]No new rows to append — already up to date.[/yellow]")
        return

    new_df = pd.DataFrame(new_rows)

    # Compute momentum_bucket cross-sectionally per (scored_date, model)
    # (grouped within each snapshot date for proper cross-sectional quintile)
    if "_momentum_12_1" in new_df.columns:
        new_df["momentum_bucket"] = (
            new_df.groupby("scored_date")["_momentum_12_1"]
            .transform(lambda x: pd.qcut(x.rank(method="first"), q=5, labels=False).fillna(2))
            .astype(np.int8)
        )
        new_df = new_df.drop(columns=["_momentum_12_1"])
    else:
        new_df["momentum_bucket"] = np.int8(2)

    console.print(f"\nNew rows generated: [bold]{len(new_df):,}[/bold]")

    # Maturity summary
    for _, _, suffix in _HORIZONS:
        col = f"matured_{suffix}"
        if col in new_df.columns:
            pct = new_df[col].mean() * 100
            n   = int(new_df[col].sum())
            console.print(f"  {suffix} matured: {pct:.0f}%  ({n:,} rows)")

    if dry_run:
        console.print("\n[yellow]--dry-run: ledger not written.[/yellow]")
        preview_cols = ["ticker", "scored_date", "is_backfill",
                        "realized_return_5d", "realized_return_21d", "realized_return_63d",
                        "matured_5d", "matured_21d", "matured_63d"]
        preview_cols = [c for c in preview_cols if c in new_df.columns]
        console.print(new_df[preview_cols].head(10).to_string())
        return

    # ── Write ledger ─────────────────────────────────────────────────────────
    _LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined = (
        pd.concat([existing_ledger, new_df], ignore_index=True)
        if not existing_ledger.empty else new_df
    )
    combined.to_parquet(_LEDGER_PATH, index=False)
    console.print(f"\n[bold green]✓ Ledger saved → {_LEDGER_PATH}[/bold green]  ({len(combined):,} total rows)")

    # ── Update IC timeseries (for monitoring dashboard) ───────────────────────
    if not skip_ic:
        console.print("\nUpdating IC timeseries…")
        try:
            from morningalpha.ml.score import _update_ic_timeseries, _check_and_save_alerts
            from scipy.stats import spearmanr

            model_ids = [m["id"] for m in active_models]
            today     = pd.Timestamp(date.today())
            ic_ts     = _update_ic_timeseries(combined, today, model_ids)
            _check_and_save_alerts(ic_ts, model_ids)

            console.print("[dim]IC timeseries and model_alerts.json updated.[/dim]")
        except Exception as exc:
            logger.warning("IC timeseries update failed (%s) — skipping", exc)

    console.print("\n[bold green]✓ Backfill complete[/bold green]")


# ---------------------------------------------------------------------------
# calibration_daily.parquet helpers
# ---------------------------------------------------------------------------

_CALIB_DAILY_PATH = Path("data/factors/calibration_daily.parquet")
_CALIB_DAILY_COLS = ["date", "ticker"] + FEATURE_COLUMNS

# Features that the spread pipeline (stocks_3m.csv) actually emits.
# These are the columns that will populate on every daily update.
# Market context (SPY/VIX/WML) and cross-sectional features are computed by
# inference.py and NOT written to the spread CSV — they are absent here.
# Dataset seeding will include them since dataset.parquet has the full feature set.
_SPREAD_FEATURE_COLS = [
    # Core metrics
    "sharpe_ratio", "sortino_ratio", "max_drawdown", "consistency_score",
    "volume_trend", "quality_score", "rsi", "momentum_accel", "volume_surge",
    "entry_score", "volatility_20d", "volatility_ratio", "avg_drawdown",
    "volume_consistency", "distance_from_high", "pct_days_positive_21d",
    # Tier 2 technicals
    "rsi_7", "rsi_21", "macd", "macd_signal", "macd_hist",
    "bollinger_pct_b", "bollinger_bandwidth", "stoch_k", "stoch_d",
    "roc_5", "roc_10", "roc_21", "atr_14",
    "price_to_sma20", "price_to_sma50", "price_to_sma200",
    "price_vs_52wk_high", "momentum_12_1", "momentum_intermediate",
    "momentum_accel_long",
    # Fundamentals present in spread CSV
    "roe", "debt_to_equity", "revenue_growth", "profit_margin",
    "current_ratio", "short_pct_float",
    # Note: sector and exchange are excluded — both are strings in the spread CSV
    # but integer-encoded in dataset.parquet, causing parquet concat failures.
    # Sector context is captured via sector_code in the predictions ledger.
]


def update_calibration_daily(
    spread_df: pd.DataFrame,
    today: pd.Timestamp,
    output_path: Path = _CALIB_DAILY_PATH,
    max_days: int = 63,
) -> int:
    """Append today's spread features to the rolling calibration window.

    Renames spread CSV columns (PascalCase) to ML feature names (snake_case),
    appends today's rows, and drops anything older than ``max_days`` trading days.

    Called by ``alpha ml score`` after scoring completes.
    Returns the total row count after update.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Rename spread columns to ML names
        today_df = spread_df.rename(columns=SPREAD_TO_ML).copy()
        today_df["date"] = today

        # Normalise ticker column name
        if "Ticker" in today_df.columns and "ticker" not in today_df.columns:
            today_df = today_df.rename(columns={"Ticker": "ticker"})

        # Keep only recognised feature columns + identifiers
        keep = ["date", "ticker"] + [c for c in _SPREAD_FEATURE_COLS if c in today_df.columns]
        today_df = today_df[keep]

        if output_path.exists():
            existing = pd.read_parquet(output_path)
            # Drop today if already present (idempotent re-run)
            existing = existing[existing["date"] != today]
            combined = pd.concat([existing, today_df], ignore_index=True)
        else:
            combined = today_df

        # Trim to last max_days unique trading dates
        unique_dates = sorted(combined["date"].unique())
        if len(unique_dates) > max_days:
            cutoff = unique_dates[-max_days]
            combined = combined[combined["date"] >= cutoff]

        combined.to_parquet(output_path, compression="zstd", index=False)
        return len(combined)
    except Exception as exc:
        logger.warning("update_calibration_daily failed (%s) — skipping", exc)
        return 0


@click.command("seed-calibration-daily")
@click.option("--dataset", "dataset_path", default="data/training/dataset.parquet",
              show_default=True, help="Path to training dataset.parquet (local only).")
@click.option("--output", "output_path", default=str(_CALIB_DAILY_PATH),
              show_default=True, help="Output path for calibration_daily.parquet.")
@click.option("--days", default=252, show_default=True,
              help="Number of trading days to keep (use 252 to cover the full backfill range).")
def seed_calibration_daily(dataset_path, output_path, days):
    """Seed calibration_daily.parquet from the training dataset.

    Extracts the most recent N trading days of features from dataset.parquet and
    writes them to data/factors/calibration_daily.parquet.  Run once locally
    after a dataset extension, then commit the output.

    After seeding, alpha ml score maintains the file daily (append + trim to 63d).
    Use --days 252 on the initial seed so backfill rows have rich feature context.

    Example:
        alpha ml backfill seed-calibration-daily
        alpha ml backfill seed-calibration-daily --days 252
    """
    ds_path = Path(dataset_path)
    out_path = Path(output_path)

    if not ds_path.exists():
        console.print(f"[red]Dataset not found: {ds_path}[/red]")
        raise SystemExit(1)

    console.print(f"Loading dataset: [dim]{ds_path}[/dim]")
    dataset = pd.read_parquet(ds_path)

    # Ensure date column is a Timestamp
    if "date" not in dataset.columns:
        console.print("[red]Dataset has no 'date' column.[/red]")
        raise SystemExit(1)
    dataset["date"] = pd.to_datetime(dataset["date"])

    # Normalise ticker column
    if "ticker" not in dataset.columns and "Ticker" in dataset.columns:
        dataset = dataset.rename(columns={"Ticker": "ticker"})

    # Get last N unique trading dates
    unique_dates = sorted(dataset["date"].unique())
    if len(unique_dates) < days:
        console.print(f"[yellow]Only {len(unique_dates)} dates in dataset — using all.[/yellow]")
        cutoff = unique_dates[0]
    else:
        cutoff = unique_dates[-days]

    recent = dataset[dataset["date"] >= cutoff].copy()
    console.print(
        f"  Using {recent['date'].nunique()} trading days "
        f"({cutoff.date()} → {recent['date'].max().date()}), "
        f"{recent['ticker'].nunique()} tickers, {len(recent):,} rows"
    )

    # Keep only feature columns that exist in the dataset (already snake_case)
    keep = ["date", "ticker"] + [c for c in _SPREAD_FEATURE_COLS if c in recent.columns]
    missing = [c for c in _SPREAD_FEATURE_COLS if c not in recent.columns]
    if missing:
        console.print(f"[yellow]  {len(missing)} feature(s) absent from dataset (will be NaN): "
                      f"{', '.join(missing[:8])}{'…' if len(missing) > 8 else ''}[/yellow]")

    recent = recent[keep]

    # Fill missing feature columns with NaN so schema stays consistent
    for col in _SPREAD_FEATURE_COLS:
        if col not in recent.columns:
            recent[col] = float("nan")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    recent.to_parquet(out_path, compression="zstd", index=False)

    size_mb = out_path.stat().st_size / 1e6
    console.print(
        f"\n[bold green]✓ Seeded calibration_daily.parquet[/bold green]  "
        f"({len(recent):,} rows, {size_mb:.1f} MB) → {out_path}"
    )
    console.print("[dim]Commit this file to enable rich calibration context in GitHub Actions.[/dim]")

    console.print("\n[bold green]✓ Backfill complete[/bold green]")
