"""Feature definitions and preprocessing utilities for the ML pipeline."""
import numpy as np
import pandas as pd
from typing import List

# ---------------------------------------------------------------------------
# Technical features (OHLCV-derived, from metrics.py)
# ---------------------------------------------------------------------------

TECHNICAL_FEATURE_COLUMNS: List[str] = [
    # Core metrics (pruned: removed return_pct, sharpe_ratio, volume_trend, quality_score,
    # momentum_accel, entry_score, exchange — all had SHAP ≈ 0 across 10d/21d/63d horizons)
    "sortino_ratio",
    "max_drawdown",
    "consistency_score",
    "rsi",
    "volume_surge",
    "market_cap_cat",
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
    # macd_signal removed — zero SHAP, redundant with macd_hist
    "macd_hist",
    # bollinger_pct_b removed — zero SHAP
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
    "price_vs_5yr_high",    # % below 5-year high; -0.96 = 96% below peak (falling knife filter)
    # Tier 2 extended: long-horizon momentum (academic factors)
    "momentum_12_1",         # Jegadeesh-Titman: return from month -12 to -1 (skip last month)
    "momentum_intermediate", # Novy-Marx: return from month -12 to -7 (intermediate horizon)
    "momentum_accel_long",   # 3-month ROC minus momentum_12_1 (long-term trend acceleration)
    "log_momentum_12_1",     # log(1 + momentum_12_1) — compresses extreme values (2000%+) to prevent winsorization from flattening the signal
    "info_discreteness",     # Discrete vs. continuous information arrival (Da & Warachka 2011)
    "rs_rating",             # Universe-wide percentile rank of momentum_12_1 (0–1, IBD-style RS)
    "rs_rating_delta_21d",   # Change in rs_rating over 21 trading days — rising RS = gaining relative strength
    "volume_trend_confirmation", # Up-day vol / down-day vol over last 21 days — confirms trend with volume
    # New: momentum structure features (Phase 1)
    "moving_average_alignment",    # Ordinal 0–3: Price>SMA20>SMA50>SMA200 (trend health score)
    "days_consecutive_above_sma20", # How many consecutive days price has held above SMA20
    "up_down_volume_ratio_63d",    # Up-day vol / down-day vol over 63 days (institutional accumulation)
    # New: pre-breakout compression & catalyst detection (Phase 2)
    "roc_63",                      # 63-day rate of change
    "vol_compression_5d_63d",      # 5d vol / 63d vol — drops < 0.5 in a volatility squeeze
    "consolidation_tightness_10d", # (10d high - 10d low) / close — tighter = more explosive potential
    "max_single_day_return_21d",   # Largest single-day gain in 21 days — catalyst event detector
    "gap_up_magnitude_10d",        # Largest gap-up (open > prior high) in 10 days
    "trend_linearity_63d",         # R² of 63d linear price trend — smooth vs. choppy
    "days_since_52wk_high",        # Days since 52-week high, /252 — near 0 = active breakout
    # New: normalized & accelerating momentum (Phase 2)
    "norm_momentum_5d",            # roc_5 / vol_20d — risk-adjusted 5d return
    "norm_momentum_21d",           # roc_21 / vol_20d — risk-adjusted 21d return
    "norm_momentum_63d",           # roc_63 / vol_20d — risk-adjusted 63d return
    "momentum_accel_5_21",         # roc_5 / roc_21 — short vs. medium acceleration (clipped ±5)
    "momentum_accel_21_63",        # roc_21 / roc_63 — medium vs. long acceleration (clipped ±5)
    "volume_confirmed_momentum",   # roc_21 × volume_surge — momentum with institutional footprint
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
    "spy_return_5d",       # SPY return over last 5 trading days (1-week crash/bounce signal)
    "spy_return_10d",      # SPY return over last 10 trading days
    "spy_return_21d",      # SPY return over last 21 trading days
    "spy_return_63d",      # SPY return over last 63 trading days (~3 months; bull/bear context)
    "spy_drawdown_from_peak",  # SPY price / 52-week high - 1 (0 = at peak, -0.15 = 15% correction)
    "spy_volatility_20d",  # SPY annualized 20-day realized volatility
    "spy_rsi_14",          # SPY RSI(14) — momentum vs. oversold regime signal
    "spy_above_sma200",    # 1.0 if SPY > 200-day SMA (bull market), 0.0 otherwise
    "spy_momentum_regime", # SPY 10d return / expected 10d vol (Sharpe-like).
                           # >0 = momentum regime, <0 = mean-reversion regime.
                           # Constant per date — excluded from cross-sectional rank norm.
    # VIX term structure — regime risk signal
    "vix_level",           # CBOE VIX spot level
    "vix_percentile",      # Percentile rank of vix_level vs trailing 252 trading days (0–1). More informative than raw level: VIX=20 means different things in different regimes.
    "vix_1m_change",       # VIX change over last 22 trading days (rising = risk-off)
    "vix_term_structure",  # VIX3M / VIX — contango (>1) = normal, backwardation (<1) = fear
    # WML (momentum) factor — crowding and crash risk
    "wml_realized_vol_126d",  # 6-month realized vol of Ken French momentum factor
    "wml_trailing_1m",        # 1-month cumulative WML return (momentum factor trend)
    "wml_trailing_3m",        # 3-month cumulative WML return (momentum regime)
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
    "current_ratio",
    "short_pct_float",
    "earnings_yield_vs_sector",   # earnings_yield minus sector median — value relative to peers
    "book_to_market_vs_sector",   # book_to_market minus sector median — value relative to peers
    "earnings_yield_quality",  # earnings_yield × ROE — value + quality composite
    "sector",
    # has_fundamentals removed — constant column (always True after NaN filtering), SHAP = 0
]

# ---------------------------------------------------------------------------
# Column name mapping: spread CSV names (PascalCase) → ML feature names (snake_case)
# Single source of truth — imported by inference.py, score.py, and backfill.py.
# ---------------------------------------------------------------------------

SPREAD_TO_ML: dict[str, str] = {
    # Core metrics
    "Return":              "return_pct",
    "SharpeRatio":         "sharpe_ratio",
    "SortinoRatio":        "sortino_ratio",
    "MaxDrawdown":         "max_drawdown",
    "ConsistencyScore":    "consistency_score",
    "VolumeTrend":         "volume_trend",
    "QualityScore":        "quality_score",
    "RSI":                 "rsi",
    "MomentumAccel":       "momentum_accel",
    "VolumeSurge":         "volume_surge",
    "EntryScore":          "entry_score",
    "AnnualizedVol":       "volatility_20d",
    "VolatilityRatio":     "volatility_ratio",
    "AvgDrawdown":         "avg_drawdown",
    "VolumeConsistency":   "volume_consistency",
    "DistanceFromHigh":    "distance_from_high",
    "PctDaysPositive21d":  "pct_days_positive_21d",
    "PriceVs20dHigh":      "price_vs_20d_high",
    # Tier 2 technicals
    "RSI7":                "rsi_7",
    "RSI21":               "rsi_21",
    "MACD":                "macd",
    "MACDSignal":          "macd_signal",
    "MACDHist":            "macd_hist",
    "BollingerPctB":       "bollinger_pct_b",
    "BollingerBandwidth":  "bollinger_bandwidth",
    "StochK":              "stoch_k",
    "StochD":              "stoch_d",
    "ROC5":                "roc_5",
    "ROC10":               "roc_10",
    "ROC21":               "roc_21",
    "ATR14":               "atr_14",
    "PriceToSMA20Pct":     "price_to_sma20",
    "PriceToSMA50Pct":     "price_to_sma50",
    "PriceToSMA200Pct":    "price_to_sma200",
    "PriceVs52wkHighPct":  "price_vs_52wk_high",
    "PriceVs5yrHighPct":   "price_vs_5yr_high",
    # Long-horizon momentum
    "Momentum12_1":        "momentum_12_1",
    "MomentumIntermediate":"momentum_intermediate",
    "MomentumAccelLong":   "momentum_accel_long",
    "InfoDiscreteness":    "info_discreteness",
    "VolumeUpDnRatio":     "volume_trend_confirmation",
    # Cross-sectional features
    "SectorReturnRank":    "sector_return_rank",
    "ReturnXRegime":       "return_pct_x_regime",
    "SectorMomentumRank":  "sector_momentum_rank",
    "ValueXMomentum":      "value_x_momentum",
    "QualityXMomentum":    "quality_x_momentum",
    # Market context
    "SPYReturn5d":         "spy_return_5d",
    "SPYReturn10d":        "spy_return_10d",
    "SPYReturn21d":        "spy_return_21d",
    "SPYReturn63d":        "spy_return_63d",
    "SPYDrawdownFromPeak": "spy_drawdown_from_peak",
    "SPYVolatility20d":    "spy_volatility_20d",
    "SPYRSI14":            "spy_rsi_14",
    "SPYAboveSMA200":      "spy_above_sma200",
    "SPYMomentumRegime":   "spy_momentum_regime",
    "VIXLevel":            "vix_level",
    "VIXPercentile":       "vix_percentile",
    "VIX1mChange":         "vix_1m_change",
    "VIXTermStructure":    "vix_term_structure",
    "WMLRealizedVol126d":  "wml_realized_vol_126d",
    "WMLTrailing1m":       "wml_trailing_1m",
    "WMLTrailing3m":       "wml_trailing_3m",
    # Fundamentals
    "EarningsYield":       "earnings_yield",
    "BookToMarket":        "book_to_market",
    "SalesToPrice":        "sales_to_price",
    "ROE":                 "roe",
    "DebtEquity":          "debt_to_equity",
    "RevenueGrowth":       "revenue_growth",
    "NetMargin":           "profit_margin",
    "CurrentRatio":        "current_ratio",
    "ShortFloat":          "short_pct_float",
    "EarningsYieldVsSector":   "earnings_yield_vs_sector",
    "BookToMarketVsSector":    "book_to_market_vs_sector",
    "EarningsYieldQuality":    "earnings_yield_quality",
    # Categoricals
    "Sector":              "sector",
    "MarketCapCat":        "market_cap_cat",
    "Exchange":            "exchange",
    "HasFundamentals":     "has_fundamentals",
    "LogMomentum12_1":     "log_momentum_12_1",
    "RSRating":            "rs_rating",
    "RSRatingDelta21d":    "rs_rating_delta_21d",
    # Phase 1 momentum structure features
    "MovingAvgAlignment":       "moving_average_alignment",
    "DaysAboveSMA20":           "days_consecutive_above_sma20",
    "UpDnVolumeRatio63d":       "up_down_volume_ratio_63d",
    # Phase 2 pre-breakout compression & catalyst features
    "ROC63":                    "roc_63",
    "VolCompression5d63d":      "vol_compression_5d_63d",
    "ConsolidationTightness10d":"consolidation_tightness_10d",
    "MaxSingleDayReturn21d":    "max_single_day_return_21d",
    "GapUpMagnitude10d":        "gap_up_magnitude_10d",
    "TrendLinearity63d":        "trend_linearity_63d",
    "DaysSince52wkHigh":        "days_since_52wk_high",
    # Phase 2 normalized & accelerating momentum
    "NormMomentum5d":           "norm_momentum_5d",
    "NormMomentum21d":          "norm_momentum_21d",
    "NormMomentum63d":          "norm_momentum_63d",
    "MomentumAccel5_21":        "momentum_accel_5_21",
    "MomentumAccel21_63":       "momentum_accel_21_63",
    "VolumeConfirmedMomentum":  "volume_confirmed_momentum",
}

# Reverse mapping: ML feature names → spread CSV names
ML_TO_SPREAD: dict[str, str] = {v: k for k, v in SPREAD_TO_ML.items()}


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
    "sector",
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
