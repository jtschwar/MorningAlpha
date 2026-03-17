"""Feature definitions and preprocessing utilities for the ML pipeline."""
import numpy as np
import pandas as pd
from typing import List

# ---------------------------------------------------------------------------
# Technical features (OHLCV-derived, from metrics.py)
# ---------------------------------------------------------------------------

TECHNICAL_FEATURE_COLUMNS: List[str] = [
    "return_pct",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "consistency_score",
    "volume_trend",
    "quality_score",
    "rsi",
    "momentum_accel",
    "price_vs_20d_high",
    "volume_surge",
    "entry_score",
    "market_cap",
    "market_cap_cat",
    "exchange",
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

FEATURE_COLUMNS: List[str] = TECHNICAL_FEATURE_COLUMNS + FUNDAMENTAL_FEATURE_NAMES

# Categorical features — ordinal encoded as int8, not rank-normalized
CATEGORICAL_FEATURES: List[str] = [
    "market_cap_cat",
    "exchange",
    "sector",
    "has_fundamentals",
]

# Continuous float features — winsorized and rank-normalized
FLOAT_FEATURES: List[str] = [f for f in FEATURE_COLUMNS if f not in CATEGORICAL_FEATURES]

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
