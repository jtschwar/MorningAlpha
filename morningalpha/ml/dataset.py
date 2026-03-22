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
    "spy_return_10d": np.nan,
    "spy_return_21d": np.nan,
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

    prices = spy_ohlcv["Close"].dropna()
    lookup: Dict[pd.Timestamp, dict] = {}

    for t in snapshot_dates:
        subset = prices.loc[:t].dropna()
        n = len(subset)

        if n < 22:
            lookup[t] = _NULL_MARKET_FEATURES.copy()
            continue

        p_t = float(subset.iloc[-1])

        spy_ret_10d = ((p_t / float(subset.iloc[-11])) - 1) if n >= 11 else np.nan
        spy_ret_21d = ((p_t / float(subset.iloc[-22])) - 1) if n >= 22 else np.nan

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
            "spy_return_10d": spy_ret_10d,
            "spy_return_21d": spy_ret_21d,
            "spy_volatility_20d": vol,
            "spy_rsi_14": rsi,
            "spy_above_sma200": above_sma200,
            "spy_momentum_regime": spy_momentum_regime,
        }

    return lookup


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

    return_pct = ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100

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
@click.option("--lookback", default="7y", show_default=True, help="Lookback period (e.g. 1y, 3y, 5y).")
@click.option(
    "--output",
    default="data/training/dataset.parquet",
    show_default=True,
    help="Output parquet path.",
)
@click.option("--tickers-from", "tickers_from", default=None, help="CSV with tickers. Omit to use full NASDAQ+S&P500 universe (reduces survivorship bias).")
@click.option("--horizons", default="5,10,21", show_default=True, help="Comma-separated forward return horizons (trading days).")
@click.option("--no-overlap/--overlap", "no_overlap", default=True, show_default=True, help="Non-overlapping snapshot windows.")
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
def dataset(lookback, output, tickers_from, horizons, no_overlap, refresh_only, from_cache, fundamentals_csv, min_market_cap):
    """Build a point-in-time labeled dataset for ML training.

    \b
    Examples:
      alpha ml dataset --tickers-from data/latest/stocks_3m.csv --output data/training/dataset.parquet
      alpha ml dataset --min-market-cap 1b --output data/training/dataset.parquet
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
                features = _compute_features_at_date(ohlcv, t, ticker_meta, fundamentals_lookup)
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

    # --- Add market context features (same value for all stocks per date) ---
    all_dates = sorted(df["date"].unique().tolist())
    market_lookup = _compute_market_features_lookup(spy_ohlcv, all_dates)
    for col in _NULL_MARKET_FEATURES:
        df[col] = df["date"].map(lambda d, _c=col: market_lookup.get(d, _NULL_MARKET_FEATURES)[_c])
    console.print(
        f"Market context features added. "
        f"SPY coverage: [bold]{sum(1 for v in market_lookup.values() if not np.isnan(v['spy_return_10d']))}[/bold]/{len(all_dates)} dates."
    )

    # --- Compute sector-relative return (alpha vs. sector beta) ---
    # Use raw return_pct before rank normalization; stocks with unknown sector get NaN → filled to 0.
    df["return_vs_sector"] = np.nan
    if "sector" in df.columns and "return_pct" in df.columns:
        known_mask = df["sector"].notna() & (df["sector"].astype(float) >= 0)
        if known_mask.any():
            df.loc[known_mask, "return_vs_sector"] = (
                df.loc[known_mask]
                .groupby(["date", "sector"])["return_pct"]
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

    # --- Rank-normalize labels cross-sectionally per date ---
    # forward_{h}d_rank is the primary training target; raw forward_{h}d is kept for evaluation.
    for h in horizons_list:
        col = f"forward_{h}d"
        if col in df.columns:
            df[f"{col}_rank"] = df.groupby("date")[col].transform(_rank_norm_series)

    # Cross-sectional median imputation for fundamental features (per snapshot date)
    for col in FUNDAMENTAL_FLOAT_FEATURES:
        if col in df.columns:
            df[col] = df.groupby("date")[col].transform(
                lambda x: x.fillna(x.median())
            ).infer_objects(copy=False)

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
