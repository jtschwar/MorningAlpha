"""alpha ml dataset — Build a point-in-time labeled dataset from raw OHLCV history."""
import json
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import rich_click as click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from morningalpha.spread.metrics import calculate_all_metrics
from morningalpha.ml.features import (
    FLOAT_FEATURES,
    encode_categoricals,
    winsorize,
)
from morningalpha.fundamentals import compute_fundamental_features, load_cached_fundamentals, _null_fundamental_features
from morningalpha.ml.features import FUNDAMENTAL_FEATURE_NAMES, FUNDAMENTAL_FLOAT_FEATURES

logger = logging.getLogger(__name__)
console = Console()

RAW_CACHE_DIR = Path("data/raw_ohlcv")
TRAINING_DIR = Path("data/training")
FAILED_TICKERS_FILE = RAW_CACHE_DIR / "failed_tickers.txt"

CACHE_TTL_DAYS = 7
BATCH_SIZE = 50
BATCH_PAUSE = 2.0
MIN_HISTORY_DAYS = 63  # ~3 months of data required to compute metrics
PRIMARY_HORIZON = 10   # default non-overlap step (trading days)

# Exchange encoding
EXCHANGE_INT: Dict[str, int] = {"NASDAQ": 0, "NYSE": 1, "S&P500": 0}

# Market-cap category string → int (3-tier from search.py → 5-tier scale)
MKTCAP_STR_INT: Dict[str, int] = {"Small": 1, "Mid": 2, "Large": 3, "Mega": 4}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _parse_lookback(lookback: str) -> int:
    """Parse '3y' / '6m' / '252d' → calendar days."""
    s = lookback.strip().lower()
    if s.endswith("y"):
        return int(s[:-1]) * 365
    if s.endswith("m"):
        return int(s[:-1]) * 30
    if s.endswith("d"):
        return int(s[:-1])
    raise ValueError(f"Unknown lookback format: {lookback!r}. Use e.g. '3y', '6m', '252d'.")


def _encode_exchange(ex: str) -> int:
    return EXCHANGE_INT.get(str(ex).strip().upper(), 2)


def _categorize_market_cap(mc: Optional[float]) -> int:
    if mc is None or (isinstance(mc, float) and np.isnan(mc)) or mc <= 0:
        return 0
    if mc >= 200e9:
        return 4
    if mc >= 10e9:
        return 3
    if mc >= 2e9:
        return 2
    if mc >= 300e6:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Ticker / metadata loading
# ---------------------------------------------------------------------------

def _load_tickers_from_csv(path: str) -> Tuple[List[str], Dict[str, dict]]:
    """Load tickers + optional metadata from a spread output CSV.

    Returns (tickers, meta_dict) where meta_dict maps ticker → {market_cap,
    market_cap_cat, exchange}.
    """
    df = pd.read_csv(path)

    ticker_col = next((c for c in ("Ticker", "ticker", "Symbol", "symbol") if c in df.columns), None)
    if ticker_col is None:
        raise ValueError(f"No ticker column found in {path}. Expected 'Ticker' or 'Symbol'.")

    tickers = df[ticker_col].dropna().astype(str).str.strip().tolist()

    meta: Dict[str, dict] = {}
    for _, row in df.iterrows():
        t = str(row[ticker_col]).strip()

        mc_raw = row.get("MarketCap", None)
        mc = float(mc_raw) if mc_raw is not None and not pd.isna(mc_raw) else np.nan

        mcc_raw = row.get("MarketCapCategory", None)
        if mcc_raw is not None and not pd.isna(mcc_raw):
            mcc = MKTCAP_STR_INT.get(str(mcc_raw).strip(), _categorize_market_cap(mc))
        else:
            mcc = _categorize_market_cap(mc)

        ex_raw = row.get("Exchange", None)
        ex = _encode_exchange(str(ex_raw)) if ex_raw is not None and not pd.isna(ex_raw) else 2

        meta[t] = {"market_cap": mc, "market_cap_cat": mcc, "exchange": ex}

    return tickers, meta


def _load_ticker_universe() -> Tuple[List[str], Dict[str, dict]]:
    from morningalpha.spread.search import read_nasdaq, read_sp500
    console.print("[bold]Fetching ticker universe from S&P 500 + NASDAQ...[/bold]")
    sp = read_sp500()
    nq = read_nasdaq()
    combined = pd.concat([sp, nq]).drop_duplicates(subset=["Ticker"])
    tickers = combined["Ticker"].tolist()
    meta = {
        row["Ticker"]: {"market_cap": np.nan, "market_cap_cat": 0, "exchange": _encode_exchange(row["Exchange"])}
        for _, row in combined.iterrows()
    }
    return tickers, meta


# ---------------------------------------------------------------------------
# OHLCV cache
# ---------------------------------------------------------------------------

def _batch_fetch_ohlcv(
    tickers: List[str],
    lookback_days: int,
    from_cache: bool,
    progress: Progress,
    task_id,
) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV for all tickers using two-layer cache + batched yfinance downloads."""
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")

    failed: set = set()
    if FAILED_TICKERS_FILE.exists():
        failed = set(FAILED_TICKERS_FILE.read_text().strip().splitlines())

    results: Dict[str, pd.DataFrame] = {}
    to_fetch_all: List[str] = []

    # Pass 1: load from cache where possible
    for ticker in tickers:
        cache_path = RAW_CACHE_DIR / f"{ticker}.parquet"
        if cache_path.exists():
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
            if (datetime.now() - mtime).days <= CACHE_TTL_DAYS or from_cache:
                try:
                    results[ticker] = pd.read_parquet(cache_path)
                    progress.advance(task_id)
                    continue
                except Exception:
                    pass
        if not from_cache and ticker not in failed:
            to_fetch_all.append(ticker)
        else:
            progress.advance(task_id)

    if from_cache or not to_fetch_all:
        return results

    # Pass 2: batched yfinance downloads for cache misses
    batches = [to_fetch_all[i : i + BATCH_SIZE] for i in range(0, len(to_fetch_all), BATCH_SIZE)]

    for batch in batches:
        try:
            raw = yf.download(
                batch,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                for t in batch:
                    failed.add(t)
                    progress.advance(task_id)
                continue

            for ticker in batch:
                try:
                    if len(batch) == 1:
                        df = raw.copy()
                    else:
                        # MultiIndex: level-0 = price field, level-1 = ticker
                        df = pd.DataFrame({
                            col: raw[col][ticker]
                            for col in ("Open", "High", "Low", "Close", "Volume")
                            if col in raw.columns.get_level_values(0)
                        })
                    df = df.dropna(how="all")
                    if len(df) >= MIN_HISTORY_DAYS:
                        df.to_parquet(RAW_CACHE_DIR / f"{ticker}.parquet")
                        results[ticker] = df
                    else:
                        failed.add(ticker)
                except Exception:
                    failed.add(ticker)
                progress.advance(task_id)

        except Exception as exc:
            logger.warning("Batch download failed (%s), falling back to individual fetches", exc)
            for ticker in batch:
                if ticker in failed:
                    progress.advance(task_id)
                    continue
                try:
                    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
                    df = df.dropna(how="all")
                    if len(df) >= MIN_HISTORY_DAYS:
                        df.to_parquet(RAW_CACHE_DIR / f"{ticker}.parquet")
                        results[ticker] = df
                    else:
                        failed.add(ticker)
                except Exception:
                    failed.add(ticker)
                progress.advance(task_id)

        FAILED_TICKERS_FILE.write_text("\n".join(sorted(failed)))
        time.sleep(BATCH_PAUSE)

    return results


# ---------------------------------------------------------------------------
# Feature / label computation
# ---------------------------------------------------------------------------

def _should_include(ohlcv: Optional[pd.DataFrame]) -> bool:
    if ohlcv is None or len(ohlcv) < MIN_HISTORY_DAYS:
        return False
    return not ohlcv["Close"].isna().all()


def _get_snapshot_dates(
    ohlcv: pd.DataFrame,
    no_overlap: bool,
    primary_horizon: int,
    max_horizon: int,
) -> List[pd.Timestamp]:
    """Return list of snapshot dates from which to compute features.

    Dates must have MIN_HISTORY_DAYS of history before them and at least
    max_horizon trading days of future prices after them.
    """
    idx = ohlcv.index
    # First usable snapshot index: we need MIN_HISTORY_DAYS prior rows
    first_pos = MIN_HISTORY_DAYS - 1
    # Last usable snapshot index: we need max_horizon future rows
    last_pos = len(idx) - max_horizon - 1

    if last_pos < first_pos:
        return []

    valid_idx = idx[first_pos : last_pos + 1]

    if no_overlap:
        return valid_idx[::primary_horizon].tolist()
    return valid_idx.tolist()


def _compute_features_at_date(
    ohlcv: pd.DataFrame,
    t: pd.Timestamp,
    ticker_meta: dict,
) -> Optional[Dict]:
    """Compute features for ticker at date t using only data up to t (point-in-time)."""
    subset = ohlcv.loc[:t]
    if len(subset) < MIN_HISTORY_DAYS:
        return None

    prices = subset["Close"].dropna()
    if len(prices) < 2:
        return None

    volumes = subset["Volume"].reindex(prices.index).fillna(0)
    returns = prices.pct_change().dropna()

    try:
        metrics = calculate_all_metrics(prices, volumes, returns)
    except Exception:
        return None

    return_pct = ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100

    mc = ticker_meta.get("market_cap", np.nan)
    if mc is None:
        mc = np.nan

    technical = {
        "return_pct": np.float32(return_pct),
        "sharpe_ratio": np.float32(metrics.get("sharpe_ratio", np.nan)),
        "sortino_ratio": np.float32(metrics.get("sortino_ratio", np.nan)),
        "max_drawdown": np.float32(metrics.get("max_drawdown", np.nan)),
        "consistency_score": np.float32(metrics.get("consistency_score", np.nan)),
        "volume_trend": np.float32(metrics.get("volume_trend", np.nan)),
        "quality_score": np.float32(metrics.get("quality_score", np.nan)),
        "rsi": np.float32(metrics.get("rsi", np.nan)),
        "momentum_accel": np.float32(metrics.get("momentum_acceleration", np.nan)),
        "price_vs_20d_high": np.float32(metrics.get("price_vs_20d_high", np.nan)),
        "volume_surge": np.float32(metrics.get("volume_surge", np.nan)),
        "entry_score": np.float32(metrics.get("entry_score", np.nan)),
        "market_cap": np.float64(mc),
        "market_cap_cat": int(ticker_meta.get("market_cap_cat", 0)),
        "exchange": int(ticker_meta.get("exchange", 2)),
    }

    # Merge fundamental features from cache
    ticker_str = ticker_meta.get("ticker", "")
    fund_raw = load_cached_fundamentals(ticker_str) if ticker_str else None
    if fund_raw is not None:
        close_price = float(prices.iloc[-1])
        fund_feats = compute_fundamental_features(fund_raw, close_price)
    else:
        fund_feats = _null_fundamental_features()

    return {**technical, **fund_feats}


def _compute_labels(
    ohlcv: pd.DataFrame,
    t: pd.Timestamp,
    horizons: List[int],
) -> Optional[Dict]:
    """Compute forward return labels at snapshot date t.

    Returns None when future price window is unavailable.
    """
    prices = ohlcv["Close"]
    try:
        idx_pos = prices.index.get_loc(t)
    except KeyError:
        return None

    max_h = max(horizons)
    if idx_pos + max_h >= len(prices):
        return None

    price_t = prices.iloc[idx_pos]
    if price_t == 0 or pd.isna(price_t):
        return None

    labels: Dict = {}
    for h in horizons:
        price_h = prices.iloc[idx_pos + h]
        labels[f"forward_{h}d"] = float((price_h / price_t) - 1)

    if 10 in horizons:
        future_slice = prices.iloc[idx_pos : idx_pos + 11]
        rolling_max = future_slice.expanding().max()
        drawdowns = (future_slice - rolling_max) / rolling_max
        max_dd = float(drawdowns.min())
        labels["adj_forward_10d"] = labels["forward_10d"] - 0.5 * abs(max_dd)

    return labels


# ---------------------------------------------------------------------------
# Train/val/test split
# ---------------------------------------------------------------------------

def _assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Assign temporal train / val / test split column.

    Uses date range: oldest 2/3 → train, next 1/6 → val, last 1/6 → test.
    For a 3-year lookback this yields ≈2y train, 6m val, 6m test.
    """
    df = df.copy()
    min_date = df["date"].min()
    max_date = df["date"].max()
    total_days = (max_date - min_date).days

    train_end = min_date + pd.Timedelta(days=int(total_days * 2 / 3))
    val_end = min_date + pd.Timedelta(days=int(total_days * 5 / 6))

    def _label(d: pd.Timestamp) -> str:
        if d <= train_end:
            return "train"
        if d <= val_end:
            return "val"
        return "test"

    df["split"] = df["date"].apply(_label)
    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _apply_preprocessing(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Winsorize (fit on train) then rank-normalize cross-sectionally per date.

    Returns (processed_df, scaler_params) where scaler_params stores the
    winsorize bounds computed from the training fold.
    """
    df = df.copy()
    train_mask = df["split"] == "train"
    float_feats = [f for f in FLOAT_FEATURES if f in df.columns]

    # --- Winsorize: compute bounds from training rows, apply globally ---
    scaler_params: dict = {}
    for col in float_feats:
        lo = float(df.loc[train_mask, col].quantile(0.01))
        hi = float(df.loc[train_mask, col].quantile(0.99))
        scaler_params[col] = {"lower": lo, "upper": hi}
        df[col] = df[col].clip(lo, hi)

    # --- Rank-normalize cross-sectionally (per snapshot date) ---
    for date_val, group_idx in df.groupby("date").groups.items():
        group = df.loc[group_idx, float_feats]
        for col in float_feats:
            ranks = group[col].rank(method="average", na_option="keep")
            n = int(ranks.notna().sum())
            if n > 1:
                df.loc[group_idx, col] = 2.0 * (ranks - 1.0) / (n - 1.0) - 1.0
            else:
                df.loc[group_idx, col] = 0.0

    return df, scaler_params


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------

@click.command("dataset")
@click.option("--lookback", default="3y", show_default=True, help="Lookback period (e.g. 1y, 3y, 5y).")
@click.option(
    "--output",
    default="data/training/dataset.parquet",
    show_default=True,
    help="Output parquet path.",
)
@click.option("--tickers-from", "tickers_from", default=None, help="CSV with tickers (e.g. data/latest/stocks_3m.csv).")
@click.option("--horizons", default="5,10,21", show_default=True, help="Comma-separated forward return horizons (trading days).")
@click.option("--no-overlap/--overlap", "no_overlap", default=True, show_default=True, help="Non-overlapping snapshot windows.")
@click.option("--refresh-only", "refresh_only", is_flag=True, default=False, help="Update raw OHLCV cache only; skip dataset build.")
@click.option("--from-cache", "from_cache", is_flag=True, default=False, help="Build dataset from existing cache (no network).")
def dataset(lookback, output, tickers_from, horizons, no_overlap, refresh_only, from_cache):
    """Build a point-in-time labeled dataset for ML training.

    \b
    Examples:
      alpha ml dataset --tickers-from data/latest/stocks_3m.csv --output data/training/dataset.parquet
      alpha ml dataset --refresh-only
      alpha ml dataset --from-cache --output data/training/dataset.parquet
    """
    horizons_list = sorted(int(h.strip()) for h in horizons.split(","))
    max_horizon = max(horizons_list)
    lookback_days = _parse_lookback(lookback)
    output_path = Path(output)

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load ticker list + metadata ---
    if tickers_from:
        tickers, meta_from_csv = _load_tickers_from_csv(tickers_from)
        console.print(f"Loaded [bold]{len(tickers)}[/bold] tickers from {tickers_from}")
    else:
        tickers, meta_from_csv = _load_ticker_universe()
        console.print(f"Loaded [bold]{len(tickers)}[/bold] tickers from universe")

    # --- Fetch / load OHLCV ---
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching raw OHLCV", total=len(tickers))
        ohlcv_cache = _batch_fetch_ohlcv(tickers, lookback_days, from_cache, progress, task)

    console.print(f"[green]{len(ohlcv_cache)}[/green] tickers with valid OHLCV data.")

    if refresh_only:
        console.print("[bold green]Cache refreshed.[/bold green]")
        return

    # --- Build rows ---
    rows: List[dict] = []
    valid_tickers = [t for t in tickers if t in ohlcv_cache and _should_include(ohlcv_cache[t])]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Computing features", total=len(valid_tickers))

        for ticker in valid_tickers:
            progress.advance(task)
            ohlcv = ohlcv_cache[ticker]
            ticker_meta = {**meta_from_csv.get(ticker, {"market_cap": np.nan, "market_cap_cat": 0, "exchange": 2}), "ticker": ticker}

            dates = _get_snapshot_dates(ohlcv, no_overlap, PRIMARY_HORIZON, max_horizon)
            for t in dates:
                features = _compute_features_at_date(ohlcv, t, ticker_meta)
                if features is None:
                    continue
                labels = _compute_labels(ohlcv, t, horizons_list)
                if labels is None:
                    continue
                rows.append({**features, **labels, "ticker": ticker, "date": t})

    if not rows:
        console.print("[bold red]No rows generated. Check ticker history length.[/bold red]")
        return

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    # Cross-sectional median imputation for fundamental features (per snapshot date)
    for col in FUNDAMENTAL_FLOAT_FEATURES:
        if col in df.columns:
            df[col] = df.groupby("date")[col].transform(lambda x: x.fillna(x.median()))

    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    console.print(f"Raw dataset: [bold]{len(df):,}[/bold] rows × {len(df.columns)} columns")

    # --- Temporal splits ---
    df = _assign_splits(df)

    # --- Preprocessing (fit on train only) ---
    df, scaler_params = _apply_preprocessing(df)

    # --- Encode categoricals ---
    df = encode_categoricals(df)

    # --- Save scaler params ---
    scaler_path = TRAINING_DIR / "scaler_params.json"
    with open(scaler_path, "w") as f:
        json.dump(scaler_params, f, indent=2)
    console.print(f"Scaler params saved to {scaler_path}")

    # --- Save parquet ---
    df.to_parquet(output_path, index=False)

    console.print(f"\n[bold green]Dataset saved → {output_path}[/bold green]")
    console.print(f"Shape: {df.shape}")
    console.print(f"\nSplit distribution:\n{df['split'].value_counts().to_string()}")
    if "forward_10d" in df.columns:
        pos_pct = (df["forward_10d"] > 0).mean()
        console.print(f"\nLabel distribution (forward_10d > 0): [bold]{pos_pct:.1%}[/bold]")
        if pos_pct < 0.40 or pos_pct > 0.60:
            console.print("[yellow]WARNING: label distribution outside 40–60% — check for survivorship bias.[/yellow]")
