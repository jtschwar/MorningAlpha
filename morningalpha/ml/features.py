"""Feature definitions and preprocessing utilities for the ML pipeline."""
import numpy as np
import pandas as pd
from typing import List

# ---------------------------------------------------------------------------
# Technical features (OHLCV-derived, from metrics.py)
# ---------------------------------------------------------------------------

TECHNICAL_FEATURE_COLUMNS: List[str] = [
    # Original 15
    "return_pct",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "consistency_score",
    "volume_trend",
    "quality_score",
    "rsi",
    "momentum_accel",
    "volume_surge",
    "entry_score",
    "market_cap",
    "market_cap_cat",
    "exchange",
    # Tier 1: already returned by calculate_all_metrics, now wired up
    "volatility_20d",
    "volatility_ratio",
    "avg_drawdown",
    "volume_consistency",
    "distance_from_high",
    "pct_days_positive_21d",
    # Tier 2: extended technical indicators (OHLCV-derived, point-in-time)
    "rsi_7",
    "rsi_21",
    "macd",
    "macd_signal",
    "macd_hist",
    "bollinger_pct_b",
    "bollinger_bandwidth",
    "stoch_k",
    "stoch_d",
    "roc_5",
    "roc_10",
    "roc_21",
    "atr_14",
    "price_to_sma20",
    "price_to_sma50",
    "price_to_sma200",
    "price_vs_52wk_high",   # % below 52-week high; 0 = at high, negative = below
    # Tier 2 extended: long-horizon momentum (academic factors)
    "momentum_12_1",         # Jegadeesh-Titman: return from month -12 to -1 (skip last month)
    "momentum_intermediate", # Novy-Marx: return from month -12 to -7 (intermediate horizon)
    "momentum_accel_long",   # 3-month ROC minus momentum_12_1 (long-term trend acceleration)
    # Tier 3: cross-sectional alpha features
    "sector_return_rank",    # percentile rank of return_pct within sector (0–1) — cross-sectional alpha vs peers
    "return_pct_x_regime",   # return_pct × spy_momentum_regime — regime-conditional momentum.
                             # Positive in trending markets, negative in reversal markets.
                             # Cross-sectional (varies by stock) → rank-normalized.
    "sector_momentum_rank",  # within-sector percentile rank of momentum_12_1
    "value_x_momentum",      # raw earnings_yield × momentum_12_1 (value trap filter)
    "quality_x_momentum",    # raw ROE × momentum_12_1 (quality + trend)
]

# ---------------------------------------------------------------------------
# Market context features — same value for all stocks on a given date.
# Cross-sectionally constant → excluded from per-date rank normalization.
# Winsorized globally; tree models use raw scale directly.
# ---------------------------------------------------------------------------

MARKET_CONTEXT_COLUMNS: List[str] = [
    "spy_return_10d",      # SPY return over last 10 trading days
    "spy_return_21d",      # SPY return over last 21 trading days
    "spy_volatility_20d",  # SPY annualized 20-day realized volatility
    "spy_rsi_14",          # SPY RSI(14) — momentum vs. oversold regime signal
    "spy_above_sma200",    # 1.0 if SPY > 200-day SMA (bull market), 0.0 otherwise
    "spy_momentum_regime", # SPY 10d return / expected 10d vol (Sharpe-like).
                           # >0 = momentum regime, <0 = mean-reversion regime.
                           # Constant per date — excluded from cross-sectional rank norm.
]

# ---------------------------------------------------------------------------
# Fundamental features (yfinance .info + financial statements)
# ---------------------------------------------------------------------------

FUNDAMENTAL_FEATURE_NAMES: List[str] = [
    "earnings_yield",
    "book_to_market",
    "sales_to_price",
    "roe",
    "debt_to_equity",
    "revenue_growth",
    "profit_margin",
    "fcf_yield",
    "current_ratio",
    "short_pct_float",
    "asset_growth",
    "accruals_ratio",
    "earnings_yield_vs_sector",   # earnings_yield minus sector median — value relative to peers
    "book_to_market_vs_sector",   # book_to_market minus sector median — value relative to peers
    "earnings_yield_quality",  # earnings_yield × ROE — value + quality composite
    "sector",
    "has_fundamentals",
]

SECTOR_MAP = {
    "technology": 0,
    "healthcare": 1,
    "financial services": 2,
    "consumer cyclical": 3,
    "communication services": 4,
    "industrials": 5,
    "consumer defensive": 6,
    "energy": 7,
    "utilities": 8,
    "real estate": 9,
    "basic materials": 10,
}

# ---------------------------------------------------------------------------
# Combined feature list (order matters for model input)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: List[str] = (
    TECHNICAL_FEATURE_COLUMNS
    + MARKET_CONTEXT_COLUMNS
    + FUNDAMENTAL_FEATURE_NAMES
)

# Categorical features — ordinal encoded as int8, not rank-normalized
CATEGORICAL_FEATURES: List[str] = [
    "market_cap_cat",
    "exchange",
    "sector",
    "has_fundamentals",
]

# Continuous float features — winsorized AND cross-sectionally rank-normalized.
# Market context features are excluded: they are constant per date, so
# cross-sectional rank normalization would zero them out.
FLOAT_FEATURES: List[str] = [
    f for f in FEATURE_COLUMNS
    if f not in CATEGORICAL_FEATURES and f not in MARKET_CONTEXT_COLUMNS
]

# Fundamental float features (subset of FLOAT_FEATURES, for median imputation)
FUNDAMENTAL_FLOAT_FEATURES: List[str] = [
    f for f in FUNDAMENTAL_FEATURE_NAMES if f not in CATEGORICAL_FEATURES
]


# ---------------------------------------------------------------------------
# Preprocessing functions
# ---------------------------------------------------------------------------

def winsorize(df: pd.DataFrame, cols: List[str], lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Cross-sectional winsorization using global quantile bounds.

    Bounds are computed from the passed DataFrame (caller passes training-fold
    rows only when fitting).
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        lo = df[col].quantile(lower)
        hi = df[col].quantile(upper)
        df[col] = df[col].clip(lo, hi)
    return df


def rank_normalize(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Rank-normalize each column to (−1, 1) — Gu/Kelly/Xiu standard."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        ranks = df[col].rank(method="average", na_option="keep")
        n = int(ranks.notna().sum())
        if n > 1:
            df[col] = 2.0 * (ranks - 1.0) / (n - 1.0) - 1.0
        else:
            df[col] = 0.0
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure categorical features are stored as int8."""
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(-1 if col == "sector" else 0).astype("int8")
    return df
