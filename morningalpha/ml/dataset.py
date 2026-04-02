"""alpha ml dataset — Build a point-in-time labeled dataset from raw OHLCV history."""
import json
import logging
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Suppress noisy-but-harmless numpy/pandas warnings from rolling windows on short series
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0", category=RuntimeWarning)

import numpy as np
import pandas as pd
import yfinance as yf
import rich_click as click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from morningalpha.spread.metrics import calculate_all_metrics
from morningalpha.ml.features import (
    FLOAT_FEATURES,
    MARKET_CONTEXT_COLUMNS,
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
# Parallel worker — module-level so it can be pickled by ProcessPoolExecutor
# ---------------------------------------------------------------------------

# Worker process globals — set once via initializer, reused for every ticker
_W_OHLCV: Dict = {}
_W_META: Dict = {}
_W_DATES: Optional[List] = None
_W_MAX_HORIZON: int = 63
_W_NO_OVERLAP: int = 10
_W_HORIZONS: List[int] = [5, 10, 21, 63]
_W_FUNDAMENTALS: Dict = {}


def _worker_init(ohlcv_cache, meta_from_csv, universal_dates, max_horizon, no_overlap, horizons_list, fundamentals_lookup):
    """Initialize worker process globals once — avoids re-pickling large data per ticker."""
    global _W_OHLCV, _W_META, _W_DATES, _W_MAX_HORIZON, _W_NO_OVERLAP, _W_HORIZONS, _W_FUNDAMENTALS
    _W_OHLCV = ohlcv_cache
    _W_META = meta_from_csv
    _W_DATES = universal_dates
    _W_MAX_HORIZON = max_horizon
    _W_NO_OVERLAP = no_overlap
    _W_HORIZONS = horizons_list
    _W_FUNDAMENTALS = fundamentals_lookup


def _process_ticker_worker(ticker: str) -> List[dict]:
    """Compute all feature rows for a single ticker. Runs in a worker process."""
    ohlcv = _W_OHLCV[ticker]
    ticker_meta = {**_W_META.get(ticker, {"market_cap": np.nan, "market_cap_cat": 0, "exchange": 2}), "ticker": ticker}
    if _W_DATES is not None:
        dates = _snap_universal_dates_to_ohlcv(_W_DATES, ohlcv, _W_MAX_HORIZON)
    else:
        dates = _get_snapshot_dates(ohlcv, _W_NO_OVERLAP, PRIMARY_HORIZON, _W_MAX_HORIZON)
    ticker_rows = []
    for t in dates:
        features = _compute_features_at_date(ohlcv, t, ticker_meta, _W_FUNDAMENTALS)
        if features is None:
            continue
        labels = _compute_labels(ohlcv, t, _W_HORIZONS)
        if labels is None:
            continue
        ticker_rows.append({**features, **labels, "ticker": ticker, "date": t})
    return ticker_rows


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
    from morningalpha.spread.search import read_nasdaq, read_nyse
    console.print("[bold]Fetching ticker universe from NASDAQ + NYSE...[/bold]")
    nq = read_nasdaq()
    ny = read_nyse()
    combined = pd.concat([nq, ny]).drop_duplicates(subset=["Ticker"])
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
# Fundamentals CSV loader
# ---------------------------------------------------------------------------

def _load_fundamentals_from_csv(csv_path: str) -> Dict[str, dict]:
    """Load fundamentals.csv into a per-ticker lookup dict.

    Column mapping (fundamentals.csv → yfinance field names used by
    compute_fundamental_features):
      EPS          → trailingEps
      PB           → priceToBook  (we invert: book_to_market = 1/PB)
      PS           → priceToSalesTrailing12Months  (inverted: sales_to_price = 1/PS)
      ROE          → returnOnEquity
      DebtEquity   → debtToEquity  (already %; divided by 100 downstream)
      RevenueGrowth → revenueGrowth
      NetMargin    → profitMargins
      CurrentRatio → currentRatio
      ShortFloat   → shortPercentOfFloat
      Sector       → sector
    fcf_yield, asset_growth, accruals_ratio — not in CSV; left None (imputed later).
    """
    path = Path(csv_path)
    if not path.exists():
        logger.warning("Fundamentals CSV not found at %s — skipping", csv_path)
        return {}
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Failed to read fundamentals CSV %s: %s", csv_path, exc)
        return {}

    ticker_col = next((c for c in ("Ticker", "ticker", "Symbol") if c in df.columns), None)
    if ticker_col is None:
        logger.warning("No Ticker column in fundamentals CSV — skipping")
        return {}

    lookup: Dict[str, dict] = {}
    for _, row in df.iterrows():
        t = str(row.get(ticker_col, "")).strip()
        if not t:
            continue
        lookup[t] = {
            "trailingEps": row.get("EPS"),
            # bookValue not directly available; set None so compute_fundamental_features
            # skips book_to_market; we compute it via PB below.
            "_pb": row.get("PB"),
            "_ps": row.get("PS"),
            "returnOnEquity": row.get("ROE"),
            "debtToEquity": row.get("DebtEquity"),
            "revenueGrowth": row.get("RevenueGrowth"),
            "profitMargins": row.get("NetMargin"),
            "freeCashflow": None,
            "currentRatio": row.get("CurrentRatio"),
            "shortPercentOfFloat": row.get("ShortFloat"),
            "sector": row.get("Sector"),
            "total_assets_current": None,
            "total_assets_prev": None,
            "net_income": None,
            "operating_cashflow": None,
            "marketCap": row.get("MarketCap"),
        }
    logger.info("Loaded fundamentals from CSV for %d tickers", len(lookup))
    return lookup


def _compute_fundamental_from_lookup(
    fund_raw: dict,
    close_price: float,
) -> dict:
    """Wrapper around compute_fundamental_features that handles the CSV-specific
    _pb / _ps keys (price-to-book / price-to-sales ratios) when raw bookValue /
    totalRevenue are unavailable.
    """
    from morningalpha.fundamentals import compute_fundamental_features, _null_fundamental_features

    def sf(v):
        if v is None:
            return None
        try:
            f = float(v)
            return None if (isinstance(f, float) and np.isnan(f)) else f
        except (ValueError, TypeError):
            return None

    # If data came from the CSV it may have _pb / _ps ratio keys instead of raw values.
    augmented = dict(fund_raw)
    if augmented.get("bookValue") is None:
        pb = sf(augmented.pop("_pb", None))
        if pb is not None and pb > 0 and close_price > 0:
            augmented["bookValue"] = pb * close_price  # reverse: PB = price/book → book = price/PB
    else:
        augmented.pop("_pb", None)

    if augmented.get("totalRevenue") is None:
        ps = sf(augmented.pop("_ps", None))
        mc = sf(augmented.get("marketCap"))
        if ps is not None and ps > 0 and mc is not None and mc > 0:
            # PS = price / (revenue / shares) ≈ marketCap / revenue → revenue = marketCap / PS
            augmented["totalRevenue"] = mc / ps
    else:
        augmented.pop("_ps", None)

    try:
        return compute_fundamental_features(augmented, close_price)
    except Exception:
        return _null_fundamental_features()


# ---------------------------------------------------------------------------
# Market context (SPY regime features)
# ---------------------------------------------------------------------------

MARKET_TICKER = "SPY"

_NULL_MARKET_FEATURES: dict = {
    "spy_return_5d": np.nan,
    "spy_return_10d": np.nan,
    "spy_return_21d": np.nan,
    "spy_return_63d": np.nan,
    "spy_drawdown_from_peak": np.nan,
    "spy_volatility_20d": np.nan,
    "spy_rsi_14": np.nan,
    "spy_above_sma200": np.nan,
    "spy_momentum_regime": np.nan,
}


def _load_market_data(lookback_days: int, from_cache: bool) -> Optional[pd.DataFrame]:
    """Fetch or load SPY OHLCV for market context features."""
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_CACHE_DIR / f"{MARKET_TICKER}.parquet"

    if cache_path.exists():
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if (datetime.now() - mtime).days <= CACHE_TTL_DAYS or from_cache:
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                pass

    if from_cache:
        logger.warning("SPY cache missing and --from-cache set; market features will be NaN.")
        return None

    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")
        df = yf.download(MARKET_TICKER, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if not df.empty:
            df.to_parquet(cache_path)
            return df
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", MARKET_TICKER, exc)

    return None


def _compute_market_features_lookup(
    spy_ohlcv: Optional[pd.DataFrame],
    snapshot_dates: List[pd.Timestamp],
) -> Dict[pd.Timestamp, dict]:
    """Pre-compute SPY market context features for every snapshot date.

    Returns a dict mapping date → feature dict. All features are scalars
    (same value for every stock on that date).
    """
    if spy_ohlcv is None or spy_ohlcv.empty:
        return {d: _NULL_MARKET_FEATURES.copy() for d in snapshot_dates}

    prices = spy_ohlcv["Close"].squeeze().dropna()
    lookup: Dict[pd.Timestamp, dict] = {}

    for t in snapshot_dates:
        subset = prices.loc[:t].dropna()
        n = len(subset)

        if n < 22:
            lookup[t] = _NULL_MARKET_FEATURES.copy()
            continue

        p_t = float(subset.iloc[-1])

        spy_ret_5d  = ((p_t / float(subset.iloc[-6]))   - 1) if n >=  6 else np.nan
        spy_ret_10d = ((p_t / float(subset.iloc[-11]))  - 1) if n >= 11 else np.nan
        spy_ret_21d = ((p_t / float(subset.iloc[-22]))  - 1) if n >= 22 else np.nan
        spy_ret_63d = ((p_t / float(subset.iloc[-64]))  - 1) if n >= 64 else np.nan

        # Drawdown from 52-week peak — key recovery signal: deeply oversold = high snap-back potential
        peak_252 = float(subset.iloc[-252:].max()) if n >= 252 else float(subset.max())
        spy_drawdown_from_peak = (p_t / peak_252) - 1  # 0 = at peak, -0.15 = 15% below peak

        daily_rets = subset.pct_change().dropna()
        vol = float(daily_rets.iloc[-20:].std() * np.sqrt(252)) if len(daily_rets) >= 20 else np.nan

        rsi = _rsi_period(subset, 14)

        sma200_v = float(subset.iloc[-200:].mean()) if n >= 200 else float(subset.mean())
        above_sma200 = 1.0 if p_t > sma200_v else 0.0

        # Regime signal: SPY 10-day return expressed in units of expected 10-day vol.
        # Positive = trending/momentum regime; negative = choppy/mean-reversion regime.
        # Uses daily std over 21 days × sqrt(10) as expected 10-day vol.
        if not np.isnan(spy_ret_10d) and len(daily_rets) >= 21:
            daily_std = float(daily_rets.iloc[-21:].std())
            expected_10d_vol = daily_std * np.sqrt(10)
            spy_momentum_regime = float(spy_ret_10d / expected_10d_vol) if expected_10d_vol > 0 else np.nan
        else:
            spy_momentum_regime = np.nan

        lookup[t] = {
            "spy_return_5d": spy_ret_5d,
            "spy_return_10d": spy_ret_10d,
            "spy_return_21d": spy_ret_21d,
            "spy_return_63d": spy_ret_63d,
            "spy_drawdown_from_peak": spy_drawdown_from_peak,
            "spy_volatility_20d": vol,
            "spy_rsi_14": rsi,
            "spy_above_sma200": above_sma200,
            "spy_momentum_regime": spy_momentum_regime,
        }

    return lookup


# ---------------------------------------------------------------------------
# VIX + WML factor regime features (historical, per snapshot date)
# ---------------------------------------------------------------------------

_NULL_FACTOR_FEATURES: dict = {
    "vix_level": np.nan,
    "vix_percentile": np.nan,
    "vix_1m_change": np.nan,
    "vix_term_structure": np.nan,
    "wml_realized_vol_126d": np.nan,
    "wml_trailing_1m": np.nan,
    "wml_trailing_3m": np.nan,
}


def _load_factor_data(lookback_days: int, from_cache: bool) -> dict:
    """Fetch or load historical VIX/VIX3M and Ken French WML factor data.

    Returns {"vix": DataFrame, "wml": Series} or empty values on failure.
    All data is point-in-time safe (published at or before market close).
    """
    import io, zipfile, urllib.request

    FACTOR_CACHE_DIR = RAW_CACHE_DIR / "factors"
    FACTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    result = {"vix": None, "wml": None}

    # --- VIX / VIX3M (Yahoo Finance) ---
    vix_cache = FACTOR_CACHE_DIR / "vix_history.parquet"
    need_vix = not vix_cache.exists() or (
        not from_cache
        and (datetime.now() - datetime.fromtimestamp(vix_cache.stat().st_mtime)).days > CACHE_TTL_DAYS
    )
    if need_vix and not from_cache:
        try:
            start = (datetime.now() - timedelta(days=lookback_days + 60)).strftime("%Y-%m-%d")
            end = datetime.now().strftime("%Y-%m-%d")
            raw = yf.download("^VIX ^VIX3M", start=start, end=end,
                              interval="1d", progress=False, auto_adjust=False)
            if not raw.empty:
                close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
                df_vix = pd.DataFrame({
                    "VIX":  close.get("^VIX",  pd.Series(dtype=float)),
                    "VIX3M": close.get("^VIX3M", pd.Series(dtype=float)),
                }).dropna(how="all")
                df_vix.to_parquet(vix_cache)
                console.print(f"VIX history fetched: {len(df_vix)} trading days.")
        except Exception as exc:
            logger.warning("VIX fetch failed (%s) — VIX features will be NaN.", exc)
    elif need_vix and from_cache:
        logger.warning("VIX cache missing and --from-cache set — VIX features will be NaN. Run without --from-cache once to populate.")
    if vix_cache.exists():
        try:
            result["vix"] = pd.read_parquet(vix_cache)
        except Exception:
            pass

    # --- Ken French daily momentum factor (WML) ---
    wml_cache = FACTOR_CACHE_DIR / "umd_daily.parquet"
    need_wml = not wml_cache.exists() or (
        not from_cache
        and (datetime.now() - datetime.fromtimestamp(wml_cache.stat().st_mtime)).days > 7
    )
    if need_wml and from_cache:
        logger.warning("WML cache missing and --from-cache set — WML features will be NaN. Run without --from-cache once to populate.")
    if need_wml and not from_cache:
        try:
            url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                fname = next(f for f in z.namelist() if f.lower().endswith(".csv"))
                raw_text = z.read(fname).decode("utf-8", errors="replace")
            lines = [l for l in raw_text.splitlines() if l.strip()[:8].strip().isdigit()]
            df_wml = pd.read_csv(io.StringIO("\n".join(lines)), header=None, names=["Date", "Mom"])
            df_wml["Date"] = pd.to_datetime(df_wml["Date"].astype(str).str.strip(),
                                            format="%Y%m%d", errors="coerce")
            df_wml = df_wml.dropna(subset=["Date"]).set_index("Date")
            df_wml["Mom"] = pd.to_numeric(df_wml["Mom"], errors="coerce") / 100.0
            df_wml.to_parquet(wml_cache)
            console.print(f"WML factor fetched: {len(df_wml)} trading days.")
        except Exception as exc:
            logger.warning("WML fetch failed (%s) — WML features will be NaN.", exc)
    if wml_cache.exists():
        try:
            result["wml"] = pd.read_parquet(wml_cache)["Mom"].dropna()
        except Exception:
            pass

    return result


def _compute_factor_features_lookup(
    factor_data: dict,
    snapshot_dates: List[pd.Timestamp],
) -> Dict[pd.Timestamp, dict]:
    """Pre-compute VIX + WML regime features for every snapshot date.

    Mirrors _compute_market_features_lookup — returns {date: feature_dict}.
    All values are point-in-time (use only data available on or before that date).
    """
    vix_df = factor_data.get("vix")
    wml_s = factor_data.get("wml")

    lookup: Dict[pd.Timestamp, dict] = {}

    for t in snapshot_dates:
        feats = _NULL_FACTOR_FEATURES.copy()

        # --- VIX features ---
        if vix_df is not None and not vix_df.empty:
            try:
                vix_hist = vix_df.loc[:t]["VIX"].dropna()
                v3m_hist = vix_df.loc[:t]["VIX3M"].dropna()
                if len(vix_hist) >= 22:
                    vix_val = float(vix_hist.iloc[-1])
                    feats["vix_level"] = vix_val
                    feats["vix_1m_change"] = vix_val - float(vix_hist.iloc[-22])
                    # Percentile rank of today's VIX vs trailing 252 trading days
                    trailing = vix_hist.iloc[-252:] if len(vix_hist) >= 252 else vix_hist
                    feats["vix_percentile"] = float((trailing < vix_val).mean())
                    if len(v3m_hist) > 0 and vix_val > 0:
                        feats["vix_term_structure"] = float(v3m_hist.iloc[-1]) / vix_val
            except Exception:
                pass

        # --- WML features ---
        if wml_s is not None and len(wml_s) > 0:
            try:
                wml_hist = wml_s.loc[:t].dropna()
                if len(wml_hist) >= 126:
                    feats["wml_realized_vol_126d"] = float(wml_hist.iloc[-126:].std() * np.sqrt(252))
                if len(wml_hist) >= 21:
                    feats["wml_trailing_1m"] = float((1 + wml_hist.iloc[-21:]).prod() - 1)
                if len(wml_hist) >= 63:
                    feats["wml_trailing_3m"] = float((1 + wml_hist.iloc[-63:]).prod() - 1)
            except Exception:
                pass

        lookup[t] = feats

    return lookup


def _compute_spy_forward_return_lookup(
    spy_ohlcv: Optional[pd.DataFrame],
    dates: List[pd.Timestamp],
    horizons: List[int],
) -> Dict[pd.Timestamp, dict]:
    """Compute SPY forward returns at each snapshot date for market-excess targets."""
    if spy_ohlcv is None or spy_ohlcv.empty:
        return {}
    prices = spy_ohlcv["Close"].squeeze().dropna()
    price_array = prices.values
    date_index = prices.index
    result: Dict[pd.Timestamp, dict] = {}
    for d in dates:
        t = pd.Timestamp(d)
        pos = date_index.searchsorted(t, side="right") - 1
        if pos < 0:
            continue
        price_t = float(price_array[pos])
        if price_t <= 0 or np.isnan(price_t):
            continue
        entry: dict = {}
        for h in horizons:
            if pos + h < len(price_array):
                price_h = float(price_array[pos + h])
                entry[f"forward_{h}d"] = (price_h / price_t) - 1
            else:
                entry[f"forward_{h}d"] = np.nan
        result[t] = entry
    return result


# ---------------------------------------------------------------------------
# Label rank normalization
# ---------------------------------------------------------------------------

def _rank_norm_series(x: pd.Series) -> pd.Series:
    """Cross-sectionally rank-normalize a series to (−1, 1)."""
    n = len(x)
    if n <= 1:
        return pd.Series([0.0] * n, index=x.index, dtype="float64")
    ranks = x.rank(method="average")
    return (2.0 * (ranks - 1.0) / (n - 1.0) - 1.0).astype("float64")


# ---------------------------------------------------------------------------
# Extended technical indicators (Tier 2)
# ---------------------------------------------------------------------------

def _safe_float(v) -> float:
    """Return float or nan."""
    try:
        if isinstance(v, pd.Series):
            v = v.iloc[0]
        f = float(v)
        return f if not np.isnan(f) else np.nan
    except (ValueError, TypeError):
        return np.nan


def _rsi_period(prices: pd.Series, period: int) -> float:
    """RSI with a configurable period using Wilder's EMA smoothing."""
    if len(prices) < period + 1:
        return np.nan
    deltas = prices.diff().dropna()
    if len(deltas) < period:
        return np.nan
    up = deltas.clip(lower=0)
    down = (-deltas).clip(lower=0)
    com = period - 1
    rs = (
        up.ewm(com=com, min_periods=period).mean()
        / down.ewm(com=com, min_periods=period).mean()
    )
    rsi_series = 100 - (100 / (1 + rs))
    last = rsi_series.iloc[-1]
    return _safe_float(last)


def _compute_extended_technicals(subset: pd.DataFrame) -> dict:
    """Compute Tier-2 technical indicators from a point-in-time OHLCV subset.

    All calculations use only data in `subset` (caller guarantees no lookahead).
    Returns a flat dict; values are float or np.nan on insufficient history.
    """
    prices = subset["Close"].dropna()
    n = len(prices)

    result: dict = {
        "rsi_7": np.nan,
        "rsi_21": np.nan,
        "macd": np.nan,
        "macd_signal": np.nan,
        "macd_hist": np.nan,
        "bollinger_pct_b": np.nan,
        "bollinger_bandwidth": np.nan,
        "stoch_k": np.nan,
        "stoch_d": np.nan,
        "roc_5": np.nan,
        "roc_10": np.nan,
        "roc_21": np.nan,
        "atr_14": np.nan,
        "price_to_sma20": np.nan,
        "price_to_sma50": np.nan,
        "price_to_sma200": np.nan,
    }

    if n < 2:
        return result

    price_t = float(prices.iloc[-1])

    # --- RSI variants ---
    result["rsi_7"] = _rsi_period(prices, 7)
    result["rsi_21"] = _rsi_period(prices, 21)

    # --- MACD (12/26/9) ---
    if n >= 26:
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        hist_line = macd_line - signal_line
        result["macd"] = _safe_float(macd_line.iloc[-1])
        result["macd_signal"] = _safe_float(signal_line.iloc[-1])
        result["macd_hist"] = _safe_float(hist_line.iloc[-1])

    # --- Bollinger Bands (20-day) ---
    if n >= 20:
        sma20 = prices.rolling(20).mean()
        std20 = prices.rolling(20).std()
        sma20_v = float(sma20.iloc[-1])
        std20_v = float(std20.iloc[-1])
        if sma20_v > 0 and std20_v > 0:
            upper = sma20_v + 2 * std20_v
            lower = sma20_v - 2 * std20_v
            band_range = upper - lower
            if band_range > 0:
                result["bollinger_pct_b"] = (price_t - lower) / band_range
            result["bollinger_bandwidth"] = band_range / sma20_v

    # --- Stochastic K/D (14-day) ---
    has_hl = "High" in subset.columns and "Low" in subset.columns
    if has_hl and n >= 14:
        high = subset["High"].reindex(prices.index)
        low = subset["Low"].reindex(prices.index)
        lowest_low = low.rolling(14).min()
        highest_high = high.rolling(14).max()
        denom = highest_high - lowest_low
        stoch_k_series = pd.Series(
            np.where(denom > 0, (prices - lowest_low) / denom * 100, np.nan),
            index=prices.index,
        )
        stoch_d_series = stoch_k_series.rolling(3).mean()
        result["stoch_k"] = _safe_float(stoch_k_series.iloc[-1])
        result["stoch_d"] = _safe_float(stoch_d_series.iloc[-1])

    # --- Rate of Change ---
    def _roc(n_periods: int) -> float:
        if n <= n_periods:
            return np.nan
        p_past = float(prices.iloc[-1 - n_periods])
        return ((price_t / p_past) - 1) * 100 if p_past != 0 else np.nan

    result["roc_5"] = _roc(5)
    result["roc_10"] = _roc(10)
    result["roc_21"] = _roc(21)

    # --- ATR-14 (normalized by close) ---
    if has_hl and n >= 15:
        high = subset["High"].reindex(prices.index)
        low = subset["Low"].reindex(prices.index)
        prev_close = prices.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        atr = tr.rolling(14).mean()
        atr_v = float(atr.iloc[-1])
        result["atr_14"] = (atr_v / price_t) if price_t > 0 else np.nan

    # --- SMA ratios ---
    def _sma_ratio(period: int) -> float:
        if n < period:
            return np.nan
        sma_v = float(prices.rolling(period).mean().iloc[-1])
        return ((price_t - sma_v) / sma_v) if sma_v > 0 else np.nan

    result["price_to_sma20"] = _sma_ratio(20)
    result["price_to_sma50"] = _sma_ratio(50)
    result["price_to_sma200"] = _sma_ratio(200)

    # 52-week high proximity — 0 = at high, negative = below
    try:
        high_52wk = float(prices.iloc[-252:].max()) if n >= 252 else float(prices.max())
        result["price_vs_52wk_high"] = ((price_t - high_52wk) / high_52wk) if high_52wk > 0 else np.nan
    except Exception:
        result["price_vs_52wk_high"] = np.nan

    # % of last 21 trading days that were positive — momentum quality
    try:
        daily_rets = prices.pct_change().dropna()
        recent = daily_rets.iloc[-21:] if len(daily_rets) >= 21 else daily_rets
        result["pct_days_positive_21d"] = float((recent > 0).mean()) if len(recent) > 0 else np.nan
    except Exception:
        result["pct_days_positive_21d"] = np.nan

    # --- Volume trend confirmation (up-day vol / down-day vol, 21 days) ---
    # >1 means more volume on up days (healthy trend); <1 = distribution (warning signal)
    try:
        if "Volume" in subset.columns and n >= 22:
            vol_series = subset["Volume"].reindex(prices.index).dropna()
            daily_rets = prices.pct_change().dropna()
            common_idx = daily_rets.index.intersection(vol_series.index)
            if len(common_idx) >= 21:
                dr = daily_rets.loc[common_idx].iloc[-21:]
                vl = vol_series.loc[common_idx].iloc[-21:]
                up_vol = vl[dr.values > 0].sum()
                dn_vol = vl[dr.values <= 0].sum()
                result["volume_trend_confirmation"] = float(up_vol / dn_vol) if dn_vol > 0 else 2.0
        else:
            result["volume_trend_confirmation"] = np.nan
    except Exception:
        result["volume_trend_confirmation"] = np.nan

    # --- Long-horizon momentum (academic factors) ---
    # momentum_12_1: Jegadeesh-Titman — return from month -12 to -1 (skip last month)
    # momentum_intermediate: Novy-Marx — return from month -12 to -7 (most predictive window)
    # momentum_accel_long: 3-month ROC minus momentum_12_1 (acceleration vs trend)
    try:
        if n >= 252:
            price_252 = float(prices.iloc[-252])  # ~12 months ago
            price_21 = float(prices.iloc[-21])    # ~1 month ago (skip month)
            price_147 = float(prices.iloc[-147])  # ~7 months ago
            if price_252 > 0 and price_21 > 0:
                result["momentum_12_1"] = (price_21 / price_252) - 1
                result["log_momentum_12_1"] = float(np.log1p(max(result["momentum_12_1"], -0.99)))
            if price_252 > 0 and price_147 > 0:
                result["momentum_intermediate"] = (price_147 / price_252) - 1
            if n >= 63 and price_252 > 0 and price_21 > 0:
                price_63 = float(prices.iloc[-63])
                roc_63 = (price_t / price_63 - 1) if price_63 > 0 else np.nan
                mom_12_1 = result.get("momentum_12_1", np.nan)
                if not np.isnan(roc_63) and not np.isnan(mom_12_1):
                    result["momentum_accel_long"] = roc_63 - mom_12_1
    except Exception:
        pass

    return result


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


def _build_universal_date_grid(lookback_days: int, freq: str) -> List[pd.Timestamp]:
    """Return a fixed calendar of snapshot dates shared across all tickers.

    All tickers are evaluated on the same dates, so (sector, date) sets contain
    every stock in that sector active on that date — full cross-sectional cohorts.

    freq choices:
      'weekly'   — every Friday  (~52 dates/year)
      'biweekly' — every other Friday (~26 dates/year)
      'monthly'  — last business day of each month (~12 dates/year)
    """
    freq_map = {"weekly": "W-FRI", "biweekly": "2W-FRI", "monthly": "BME"}
    pandas_freq = freq_map.get(freq, "W-FRI")
    end = pd.Timestamp.now().normalize()
    start = end - pd.Timedelta(days=lookback_days)
    return pd.date_range(start=start, end=end, freq=pandas_freq).tolist()


def _snap_universal_dates_to_ohlcv(
    universal_dates: List[pd.Timestamp],
    ohlcv: pd.DataFrame,
    max_horizon: int,
) -> List[pd.Timestamp]:
    """For each universal calendar date, snap to the nearest prior trading day
    in this ticker's OHLCV.  Filters out dates with insufficient history or
    insufficient future prices for label computation.
    """
    idx = ohlcv.index
    result: List[pd.Timestamp] = []
    for d in universal_dates:
        prior = idx[idx <= d]
        if len(prior) == 0:
            continue
        t = prior[-1]
        pos = idx.get_loc(t)
        if pos < MIN_HISTORY_DAYS - 1:
            continue
        # Labels need max_horizon future bars — _compute_labels will return None
        # if unavailable, but skip early here to avoid the call overhead.
        if pos + max_horizon >= len(idx):
            continue
        result.append(t)
    return result


def _compute_features_at_date(
    ohlcv: pd.DataFrame,
    t: pd.Timestamp,
    ticker_meta: dict,
    fundamentals_lookup: Optional[Dict[str, dict]] = None,
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

    p0 = prices.iloc[0]
    return_pct = ((prices.iloc[-1] / p0) - 1) * 100 if p0 != 0 else np.nan

    # Point-in-time market cap proxy: scale current market cap by price ratio.
    # current_mc comes from the CSV (snapshot date), current_price is the
    # latest close in the full OHLCV (not subset), so we adjust for price change.
    current_mc = ticker_meta.get("market_cap", np.nan)
    if current_mc is None:
        current_mc = np.nan
    try:
        current_price = float(ohlcv["Close"].dropna().iloc[-1])
        snapshot_price = float(prices.iloc[-1])
        if not np.isnan(current_mc) and current_price > 0 and snapshot_price > 0:
            mc = current_mc * (snapshot_price / current_price)
        else:
            mc = np.nan
    except Exception:
        mc = np.nan

    mc_cat = _categorize_market_cap(mc)

    # --- Tier 1 + original 15 ---
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
        "market_cap_cat": int(mc_cat),
        "exchange": int(ticker_meta.get("exchange", 2)),
        # Tier 1: already in calculate_all_metrics, now wired up
        "volatility_20d": np.float32(metrics.get("volatility_20d", np.nan)),
        "volatility_ratio": np.float32(metrics.get("volatility_ratio", np.nan)),
        "avg_drawdown": np.float32(metrics.get("avg_drawdown", np.nan)),
        "volume_consistency": np.float32(metrics.get("volume_consistency", np.nan)),
        "distance_from_high": np.float32(metrics.get("distance_from_high", np.nan)),
    }

    # --- Tier 2: extended technical indicators ---
    extended = _compute_extended_technicals(subset)
    for k, v in extended.items():
        technical[k] = np.float32(v)

    # --- Fundamental features ---
    # Priority: JSON cache → CSV lookup → null
    ticker_str = ticker_meta.get("ticker", "")
    fund_raw = load_cached_fundamentals(ticker_str) if ticker_str else None
    if fund_raw is not None:
        close_price = float(prices.iloc[-1])
        fund_feats = compute_fundamental_features(fund_raw, close_price)
    elif fundamentals_lookup and ticker_str in fundamentals_lookup:
        close_price = float(prices.iloc[-1])
        fund_feats = _compute_fundamental_from_lookup(
            fundamentals_lookup[ticker_str], close_price
        )
    else:
        fund_feats = _null_fundamental_features()

    # Composite: value × quality — penalises cheap-but-deteriorating stocks
    ey = fund_feats.get("earnings_yield", np.nan)
    roe = fund_feats.get("roe", np.nan)
    fund_feats["earnings_yield_quality"] = float(ey * roe) if (
        ey is not None and roe is not None
        and not np.isnan(float(ey)) and not np.isnan(float(roe))
    ) else np.nan

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

        # Quality components of the forward return window
        future_slice = prices.iloc[idx_pos : idx_pos + h + 1]
        daily_rets = future_slice.pct_change().dropna().values

        # Max drawdown over forward window
        rolling_max = future_slice.expanding().max()
        drawdowns = (future_slice - rolling_max) / rolling_max
        max_dd = float(drawdowns.min())
        labels[f"forward_{h}d_max_drawdown"] = max_dd

        # Drawdown-adjusted return (generalised from adj_forward_10d)
        labels[f"adj_forward_{h}d"] = labels[f"forward_{h}d"] - 0.5 * abs(max_dd)

        # Forward Sharpe: annualised mean/std of daily returns in the window
        if len(daily_rets) >= 5:
            std = float(np.std(daily_rets, ddof=1))
            labels[f"forward_{h}d_sharpe"] = (
                float(np.mean(daily_rets) / std * np.sqrt(252)) if std > 1e-8 else 0.0
            )
        else:
            labels[f"forward_{h}d_sharpe"] = float("nan")

        # Consistency: fraction of positive daily returns
        labels[f"forward_{h}d_consistency"] = (
            float((daily_rets > 0).mean()) if len(daily_rets) > 0 else float("nan")
        )

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


def _assign_walk_forward_folds(
    df: pd.DataFrame,
    fold_step_months: int = 1,
    min_train_months: int = 18,
    test_months: int = 3,
) -> Tuple[pd.DataFrame, List]:
    """Assign expanding-window walk-forward fold labels.

    Adds integer column ``test_fold``:
      - 0  → always train-eligible (never a test row)
      - N  → this row is in the test set for fold N

    Test windows are non-overlapping blocks of ``test_months`` calendar
    months (~63 trading days), rolling forward by ``fold_step_months``
    months. The earliest possible test window begins ``min_train_months``
    after the first date in the dataset.

    Uses calendar-month offsets rather than snapshot counts so the window
    size is independent of snapshot frequency (weekly, monthly, etc.).
    """
    df = df.copy()
    df["test_fold"] = 0

    all_dates = sorted(df["date"].unique())
    if not all_dates:
        return df, []

    min_date = pd.Timestamp(all_dates[0])
    first_test_start = min_date + pd.DateOffset(months=min_train_months)

    fold_boundaries: List[Tuple] = []
    current_start = first_test_start

    while True:
        remaining = [d for d in all_dates if pd.Timestamp(d) >= current_start]
        if not remaining:
            break
        fold_test_start = pd.Timestamp(remaining[0])
        fold_test_end_cutoff = fold_test_start + pd.DateOffset(months=test_months)

        # All snapshot dates within [fold_test_start, fold_test_end_cutoff)
        test_snapshots = [d for d in remaining if pd.Timestamp(d) < fold_test_end_cutoff]
        if not test_snapshots:
            break
        fold_test_end = pd.Timestamp(test_snapshots[-1])
        fold_boundaries.append((fold_test_start, fold_test_end))
        current_start = fold_test_start + pd.DateOffset(months=fold_step_months)

    for fold_num, (start, end) in enumerate(fold_boundaries, start=1):
        mask = (df["date"] >= start) & (df["date"] <= end)
        df.loc[mask, "test_fold"] = fold_num

    return df, fold_boundaries


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _apply_preprocessing(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Winsorize (fit on train) then rank-normalize cross-sectionally per date.

    Market context features (MARKET_CONTEXT_COLUMNS) are winsorized but NOT
    rank-normalized — they are constant per date so cross-sectional ranking
    would zero them out, destroying their regime signal.

    Returns (processed_df, scaler_params) where scaler_params stores the
    winsorize bounds computed from the training fold.
    """
    df = df.copy()
    train_mask = df["split"] == "train"
    float_feats = [f for f in FLOAT_FEATURES if f in df.columns]
    market_feats = [f for f in MARKET_CONTEXT_COLUMNS if f in df.columns]

    # --- Winsorize: compute bounds from training rows, apply globally ---
    scaler_params: dict = {}
    for col in float_feats + market_feats:
        lo = float(df.loc[train_mask, col].quantile(0.01))
        hi = float(df.loc[train_mask, col].quantile(0.99))
        scaler_params[col] = {"lower": lo, "upper": hi}
        df[col] = df[col].clip(lo, hi)

    # --- Rank-normalize cross-sectionally (per snapshot date) ---
    # Market context features are intentionally excluded from this step.
    for _, group_idx in df.groupby("date").groups.items():
        group = df.loc[group_idx, float_feats]
        for col in float_feats:
            ranks = group[col].rank(method="average", na_option="keep")
            n = int(ranks.notna().sum())
            if n > 1:
                df.loc[group_idx, col] = (2.0 * (ranks - 1.0) / (n - 1.0) - 1.0).astype("float32")
            else:
                df.loc[group_idx, col] = 0.0

    return df, scaler_params


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------

@click.command("dataset")
@click.option("--lookback", default="10y", show_default=True, help="Lookback period (e.g. 1y, 3y, 10y).")
@click.option(
    "--output",
    default="data/training/dataset.parquet",
    show_default=True,
    help="Output parquet path.",
)
@click.option("--tickers-from", "tickers_from", default=None, help="CSV with tickers. Omit to use full NASDAQ+NYSE universe (reduces survivorship bias).")
@click.option("--horizons", default="5,10,21,63", show_default=True, help="Comma-separated forward return horizons (trading days).")
@click.option("--no-overlap/--overlap", "no_overlap", default=True, show_default=True, help="Non-overlapping snapshot windows (staggered mode only).")
@click.option(
    "--snapshot-freq",
    "snapshot_freq",
    default="weekly",
    show_default=True,
    type=click.Choice(["weekly", "biweekly", "monthly", "staggered"]),
    help=(
        "Snapshot date strategy. 'weekly/biweekly/monthly' use a universal calendar so all "
        "tickers share the same dates — required for set transformer cross-sectional learning. "
        "'staggered' uses per-ticker non-overlapping windows (legacy behaviour)."
    ),
)
@click.option("--n-workers", "n_workers", default=1, show_default=True,
              help="Number of parallel worker processes for feature computation. "
                   "Set to --cpus-per-task on HPC. Each worker is independent (no API calls).")
@click.option("--refresh-only", "refresh_only", is_flag=True, default=False, help="Update raw OHLCV cache only; skip dataset build.")
@click.option("--from-cache", "from_cache", is_flag=True, default=False, help="Build dataset from existing cache (no network).")
@click.option(
    "--fundamentals-csv",
    "fundamentals_csv",
    default="data/latest/fundamentals.csv",
    show_default=True,
    help="Path to fundamentals CSV for broad fundamental feature coverage.",
)
@click.option(
    "--min-market-cap",
    "min_market_cap",
    default="1b",
    show_default=True,
    help="Minimum market cap to include (e.g. 1b, 500m, 1000000000). 0 = no filter.",
)
@click.option(
    "--fold-step",
    "fold_step",
    default=1,
    show_default=True,
    help="Walk-forward fold step size in months.",
)
@click.option(
    "--min-train",
    "min_train",
    default=18,
    show_default=True,
    help="Minimum months of training data before the first walk-forward test fold.",
)
def dataset(lookback, output, tickers_from, horizons, snapshot_freq, no_overlap, n_workers, refresh_only, from_cache, fundamentals_csv, min_market_cap, fold_step, min_train):
    """Build a point-in-time labeled dataset for ML training.

    \b
    Examples:
      alpha ml dataset --output data/training/dataset.parquet
      alpha ml dataset --snapshot-freq weekly --lookback 10y --horizons 5,10,21,63
      alpha ml dataset --tickers-from data/latest/stocks_3m.csv --snapshot-freq staggered
      alpha ml dataset --refresh-only
      alpha ml dataset --from-cache --output data/training/dataset.parquet
    """
    horizons_list = sorted(int(h.strip()) for h in horizons.split(","))

    # Parse min_market_cap: accept "1b", "500m", or raw integer string
    def _parse_market_cap(val: str) -> float:
        v = val.strip().lower()
        if v.endswith("b"):
            return float(v[:-1]) * 1_000_000_000
        if v.endswith("m"):
            return float(v[:-1]) * 1_000_000
        return float(v)

    min_cap = _parse_market_cap(min_market_cap)
    max_horizon = max(horizons_list)
    lookback_days = _parse_lookback(lookback)
    output_path = Path(output)

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load fundamentals CSV lookup (broad coverage fallback for JSON cache) ---
    fundamentals_lookup = _load_fundamentals_from_csv(fundamentals_csv)
    if fundamentals_lookup:
        console.print(f"Loaded fundamentals from CSV for [bold]{len(fundamentals_lookup)}[/bold] tickers ({fundamentals_csv})")
    else:
        console.print("[yellow]No fundamentals CSV loaded — fundamental features will be null for uncached tickers.[/yellow]")

    # --- Load ticker list + metadata ---
    if tickers_from:
        tickers, meta_from_csv = _load_tickers_from_csv(tickers_from)
        console.print(f"Loaded [bold]{len(tickers)}[/bold] tickers from {tickers_from}")
    else:
        tickers, meta_from_csv = _load_ticker_universe()
        console.print(f"Loaded [bold]{len(tickers)}[/bold] tickers from universe")
        # Backfill market cap from fundamentals CSV for universe tickers
        if fundamentals_lookup:
            filled = 0
            for t, meta in meta_from_csv.items():
                if np.isnan(meta.get("market_cap", np.nan)):
                    mc_raw = fundamentals_lookup.get(t, {}).get("marketCap")
                    if mc_raw is not None:
                        try:
                            mc = float(mc_raw)
                            meta["market_cap"] = mc
                            meta["market_cap_cat"] = _categorize_market_cap(mc)
                            filled += 1
                        except (ValueError, TypeError):
                            pass
            console.print(f"[dim]Backfilled market cap for {filled} tickers from fundamentals CSV.[/dim]")

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

    # --- Fetch SPY for market context features ---
    spy_ohlcv = _load_market_data(lookback_days, from_cache)
    if spy_ohlcv is not None:
        console.print(f"SPY data loaded: [bold]{len(spy_ohlcv)}[/bold] trading days.")
    else:
        console.print("[yellow]SPY data unavailable — market context features will be NaN.[/yellow]")

    # --- Fetch VIX/WML for regime features ---
    factor_data = _load_factor_data(lookback_days, from_cache)

    if refresh_only:
        console.print("[bold green]Cache refreshed.[/bold green]")
        return

    # --- Build rows ---
    rows: List[dict] = []
    valid_tickers = [t for t in tickers if t in ohlcv_cache and _should_include(ohlcv_cache[t])]

    if min_cap > 0:
        before = len(valid_tickers)
        valid_tickers = [
            t for t in valid_tickers
            if meta_from_csv.get(t, {}).get("market_cap") is not None
            and not np.isnan(float(meta_from_csv[t]["market_cap"]))
            and float(meta_from_csv[t]["market_cap"]) >= min_cap
        ]
        console.print(
            f"[dim]Market cap filter (>= ${min_cap/1e9:.1f}B): "
            f"{before} → {len(valid_tickers)} tickers[/dim]"
        )

    # Pre-build universal date grid once (shared across all tickers)
    if snapshot_freq != "staggered":
        universal_dates = _build_universal_date_grid(lookback_days, snapshot_freq)
        console.print(
            f"Universal snapshot grid: [bold]{len(universal_dates)}[/bold] dates "
            f"({snapshot_freq}, {lookback_days // 365}y lookback)"
        )
    else:
        universal_dates = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Computing features", total=len(valid_tickers))

        if n_workers > 1:
            console.print(f"[dim]Parallel feature computation: {n_workers} workers[/dim]")
            init_args = (ohlcv_cache, meta_from_csv, universal_dates, max_horizon,
                         no_overlap, horizons_list, fundamentals_lookup)
            with ProcessPoolExecutor(max_workers=n_workers,
                                     initializer=_worker_init,
                                     initargs=init_args) as executor:
                futures = {executor.submit(_process_ticker_worker, t): t for t in valid_tickers}
                for future in as_completed(futures):
                    progress.advance(task)
                    try:
                        rows.extend(future.result())
                    except Exception as exc:
                        logger.warning("Ticker %s failed: %s", futures[future], exc)
        else:
            _worker_init(ohlcv_cache, meta_from_csv, universal_dates, max_horizon,
                         no_overlap, horizons_list, fundamentals_lookup)
            for ticker in valid_tickers:
                progress.advance(task)
                rows.extend(_process_ticker_worker(ticker))

    if not rows:
        console.print("[bold red]No rows generated. Check ticker history length.[/bold red]")
        return

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    # --- Add market context features (same value for all stocks per date) ---
    all_dates = sorted(df["date"].unique().tolist())
    market_lookup = _compute_market_features_lookup(spy_ohlcv, all_dates)
    for col in _NULL_MARKET_FEATURES:
        df[col] = df["date"].map(lambda d, _c=col: market_lookup.get(d, _NULL_MARKET_FEATURES)[_c])
    console.print(
        f"Market context features added. "
        f"SPY coverage: [bold]{sum(1 for v in market_lookup.values() if not np.isnan(v['spy_return_10d']))}[/bold]/{len(all_dates)} dates."
    )

    # --- Add VIX/WML regime features (historical, per snapshot date) ---
    factor_lookup = _compute_factor_features_lookup(factor_data, all_dates)
    for col in _NULL_FACTOR_FEATURES:
        df[col] = df["date"].map(lambda d, _c=col: factor_lookup.get(d, _NULL_FACTOR_FEATURES)[_c])
    vix_coverage = sum(1 for v in factor_lookup.values() if not np.isnan(v["vix_level"]))
    wml_coverage = sum(1 for v in factor_lookup.values() if not np.isnan(v["wml_trailing_3m"]))
    console.print(
        f"Regime features added. "
        f"VIX coverage: [bold]{vix_coverage}[/bold]/{len(all_dates)} dates, "
        f"WML coverage: [bold]{wml_coverage}[/bold]/{len(all_dates)} dates."
    )

    # --- Compute sector-relative features (cross-sectional, pre-rank-normalization) ---
    df["sector_return_rank"] = np.nan
    df["earnings_yield_vs_sector"] = np.nan
    df["book_to_market_vs_sector"] = np.nan
    if "sector" in df.columns:
        known_mask = df["sector"].notna() & (df["sector"].astype(float) >= 0)
        if known_mask.any() and "return_pct" in df.columns:
            df.loc[known_mask, "sector_return_rank"] = (
                df.loc[known_mask]
                .groupby(["date", "sector"])["return_pct"]
                .transform(lambda x: x.rank(pct=True))
            )
        if known_mask.any() and "earnings_yield" in df.columns:
            df.loc[known_mask, "earnings_yield_vs_sector"] = (
                df.loc[known_mask]
                .groupby(["date", "sector"])["earnings_yield"]
                .transform(lambda x: x - x.median())
            )
        if known_mask.any() and "book_to_market" in df.columns:
            df.loc[known_mask, "book_to_market_vs_sector"] = (
                df.loc[known_mask]
                .groupby(["date", "sector"])["book_to_market"]
                .transform(lambda x: x - x.median())
            )

    # --- Regime-conditional momentum interaction feature ---
    # Computed from raw return_pct × spy_momentum_regime before rank normalization.
    # On momentum days: high-return_pct stocks get high positive values.
    # On reversal days: high-return_pct stocks get negative values.
    # NaN where regime signal is unavailable → filled to 0 (neutral) at train time.
    if "return_pct" in df.columns and "spy_momentum_regime" in df.columns:
        df["return_pct_x_regime"] = df["return_pct"] * df["spy_momentum_regime"]
    else:
        df["return_pct_x_regime"] = np.nan

    # --- Sector momentum rank (within-sector percentile rank of momentum_12_1) ---
    df["sector_momentum_rank"] = np.nan
    if "sector" in df.columns and "momentum_12_1" in df.columns:
        known_mask = (
            df["sector"].notna()
            & (df["sector"].astype(float) >= 0)
            & df["momentum_12_1"].notna()
        )
        if known_mask.any():
            df.loc[known_mask, "sector_momentum_rank"] = (
                df.loc[known_mask]
                .groupby(["date", "sector"])["momentum_12_1"]
                .transform(lambda x: x.rank(pct=True))
            )

    # --- Value × momentum and quality × momentum interaction features ---
    # Computed on raw values; rank-normalized by _apply_preprocessing like all float features.
    df["value_x_momentum"] = np.nan
    df["quality_x_momentum"] = np.nan
    if "earnings_yield" in df.columns and "momentum_12_1" in df.columns:
        both = df["earnings_yield"].notna() & df["momentum_12_1"].notna()
        df.loc[both, "value_x_momentum"] = (
            df.loc[both, "earnings_yield"] * df.loc[both, "momentum_12_1"]
        )
    if "roe" in df.columns and "momentum_12_1" in df.columns:
        both = df["roe"].notna() & df["momentum_12_1"].notna()
        df.loc[both, "quality_x_momentum"] = (
            df.loc[both, "roe"] * df.loc[both, "momentum_12_1"]
        )

    # --- RS rating: universe-wide percentile rank of momentum_12_1 per date (0–1) ---
    # This is the IBD-style Relative Strength number. A stock with rs_rating=0.99
    # has stronger 12-month momentum than 99% of all stocks on that date.
    if "momentum_12_1" in df.columns:
        df["rs_rating"] = (
            df.groupby("date")["momentum_12_1"]
            .transform(lambda x: x.rank(pct=True))
        )
    else:
        df["rs_rating"] = np.nan

    # --- Rank-normalize labels cross-sectionally per date ---
    # forward_{h}d_rank is the primary training target; raw forward_{h}d is kept for evaluation.
    for h in horizons_list:
        col = f"forward_{h}d"
        if col in df.columns:
            df[f"{col}_rank"] = df.groupby("date")[col].transform(_rank_norm_series)

    # --- Composite quality target ---
    # Mirrors the traditional investment score weighting:
    #   Return 30% | Sharpe 35% | Consistency 20% | Drawdown protection 15%
    # Trains the model to predict which stocks will have the best QUALITY returns
    # going forward, not just the highest raw return.
    for h in horizons_list:
        needed = [f"forward_{h}d", f"forward_{h}d_sharpe", f"forward_{h}d_consistency", f"forward_{h}d_max_drawdown"]
        if not all(c in df.columns for c in needed):
            continue
        # Rank each component cross-sectionally per date (0→1, higher = better)
        r_return      = df.groupby("date")[f"forward_{h}d"].transform(lambda x: x.rank(pct=True))
        r_sharpe      = df.groupby("date")[f"forward_{h}d_sharpe"].transform(lambda x: x.rank(pct=True))
        r_consistency = df.groupby("date")[f"forward_{h}d_consistency"].transform(lambda x: x.rank(pct=True))
        # Drawdown is negative — invert so lower drawdown scores higher
        r_drawdown    = df.groupby("date")[f"forward_{h}d_max_drawdown"].transform(lambda x: (-x).rank(pct=True))

        _tmp = f"_tmp_composite_{h}d"
        df[_tmp] = (
            0.30 * r_return
            + 0.35 * r_sharpe
            + 0.20 * r_consistency
            + 0.15 * r_drawdown
        )
        # Final cross-sectional rank-normalize to (−1, 1) to match other targets
        df[f"forward_{h}d_composite_rank"] = df.groupby("date")[_tmp].transform(_rank_norm_series)
        df.drop(columns=[_tmp], inplace=True)

    # --- Market-excess and sector-relative return targets ---
    spy_fwd_lookup = _compute_spy_forward_return_lookup(spy_ohlcv, all_dates, horizons_list)
    for h in horizons_list:
        raw_col = f"forward_{h}d"
        if raw_col not in df.columns:
            continue
        # Market-excess: stock return minus SPY return over same window, then rank within date
        if spy_fwd_lookup:
            spy_fwd = df["date"].map(
                lambda d, _h=h: spy_fwd_lookup.get(d, {}).get(f"forward_{_h}d", np.nan)
            )
            df["_tmp_excess"] = df[raw_col] - spy_fwd
            df[f"forward_{h}d_market_excess_rank"] = (
                df.groupby("date")["_tmp_excess"].transform(_rank_norm_series)
            )
            df.drop(columns=["_tmp_excess"], inplace=True)
        # Sector-relative: stock return minus sector median return, then rank within date
        if "sector" in df.columns:
            known_mask = df["sector"].notna() & (df["sector"].astype(float) >= 0)
            sector_rel = df[raw_col].copy()
            if known_mask.any():
                sector_medians = (
                    df.loc[known_mask]
                    .groupby(["date", "sector"])[raw_col]
                    .transform("median")
                )
                sector_rel.loc[known_mask] = df.loc[known_mask, raw_col] - sector_medians
            df["_tmp_sector_rel"] = sector_rel
            df[f"forward_{h}d_sector_relative_rank"] = (
                df.groupby("date")["_tmp_sector_rel"].transform(_rank_norm_series)
            )
            df.drop(columns=["_tmp_sector_rel"], inplace=True)

    # Cross-sectional median imputation for fundamental features (per snapshot date)
    for col in FUNDAMENTAL_FLOAT_FEATURES:
        if col in df.columns:
            df[col] = df.groupby("date")[col].transform(
                lambda x: x.fillna(x.median())
            ).infer_objects(copy=False)

    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    console.print(f"Raw dataset: [bold]{len(df):,}[/bold] rows × {len(df.columns)} columns")

    # --- Temporal splits (backward-compat static split) ---
    df = _assign_splits(df)

    # --- Walk-forward fold labels ---
    df, fold_boundaries = _assign_walk_forward_folds(
        df, fold_step_months=fold_step, min_train_months=min_train
    )
    console.print(
        f"Walk-forward folds: [bold]{len(fold_boundaries)}[/bold] "
        f"(step={fold_step}m, min_train={min_train}m, test_window=63 trading days)"
    )
    if fold_boundaries:
        console.print(
            f"  First fold: {fold_boundaries[0][0].date()} → {fold_boundaries[0][1].date()}"
        )
        console.print(
            f"  Last fold:  {fold_boundaries[-1][0].date()} → {fold_boundaries[-1][1].date()}"
        )

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
