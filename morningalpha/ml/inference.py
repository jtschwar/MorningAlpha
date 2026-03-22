"""Live inference: score the current spread result with the trained LightGBM model.

Called from spread/access.py after `analyze_stocks` + fundamentals merge.
Adds a 'MLScore' column (0–100 percentile rank within the batch).
Returns the DataFrame unchanged if the model is not found or inference fails.
"""
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from morningalpha.ml.features import (
    FEATURE_COLUMNS,
    FLOAT_FEATURES,
    MARKET_CONTEXT_COLUMNS,
    SECTOR_MAP,
    winsorize,
    rank_normalize,
)

logger = logging.getLogger(__name__)

# Look in repo-tracked models/ first, fall back to ~/.morningalpha/models/
_REPO_MODELS = Path(__file__).parents[2] / "models"
_HOME_MODELS = Path.home() / ".morningalpha" / "models"
MODEL_DIR = _REPO_MODELS if _REPO_MODELS.exists() else _HOME_MODELS

# ---------------------------------------------------------------------------
# Column mapping: spread CSV names → ML feature names
# ---------------------------------------------------------------------------

_SPREAD_TO_ML: dict = {
    # Core metrics (PascalCase in CSV)
    "SharpeRatio":        "sharpe_ratio",
    "SortinoRatio":       "sortino_ratio",
    "MaxDrawdown":        "max_drawdown",
    "ConsistencyScore":   "consistency_score",
    "VolumeTrend":        "volume_trend",
    "QualityScore":       "quality_score",
    "RSI":                "rsi",
    "MomentumAccel":      "momentum_accel",
    "PriceVs20dHigh":     "price_vs_20d_high",
    "VolumeSurge":        "volume_surge",
    "EntryScore":         "entry_score",
    "MarketCap":          "market_cap",
    "AnnualizedVol":      "volatility_20d",
    # Tier 2 technicals
    "RSI7":               "rsi_7",
    "RSI21":              "rsi_21",
    "MACD":               "macd",
    "MACDSignal":         "macd_signal",
    "MACDHist":           "macd_hist",
    "BollingerPctB":      "bollinger_pct_b",
    "BollingerBandwidth": "bollinger_bandwidth",
    "StochK":             "stoch_k",
    "StochD":             "stoch_d",
    "ROC5":               "roc_5",
    "ROC10":              "roc_10",
    "ROC21":              "roc_21",
    "ATR14":              "atr_14",
    "PriceToSMA20Pct":    "price_to_sma20",
    "PriceToSMA50Pct":    "price_to_sma50",
    "PriceToSMA200Pct":   "price_to_sma200",
    "PriceVs52wkHighPct": "price_vs_52wk_high",
    # Fundamentals — raw yfinance column names that end up in the CSV
    "returnOnEquity":          "roe",
    "debtToEquity":            "debt_to_equity",
    "revenueGrowth":           "revenue_growth",
    "profitMargins":           "profit_margin",
    "currentRatio":            "current_ratio",
    "shortPercentOfFloat":     "short_pct_float",
    # Pre-computed ML fundamental columns already in the CSV
    "earnings_yield":    "earnings_yield",
    "book_to_market":    "book_to_market",
    "sales_to_price":    "sales_to_price",
    "fcf_yield":         "fcf_yield",
    "asset_growth":      "asset_growth",
    "accruals_ratio":    "accruals_ratio",
}

_MARKET_CAP_CAT_MAP = {
    "Mega": 5, "Large": 4, "Mid": 3, "Small": 2, "Micro": 1,
}

_EXCHANGE_MAP = {
    "NASDAQ": 0, "NYSE": 1, "S&P500": 2,
}


# ---------------------------------------------------------------------------
# SPY market context
# ---------------------------------------------------------------------------

def _compute_spy_features() -> dict:
    """Fetch recent SPY data and compute today's market context features."""
    try:
        import yfinance as yf
        spy = yf.download("SPY", period="1y", interval="1d", progress=False, auto_adjust=True)
        if spy.empty or len(spy) < 22:
            return {}
        # Flatten multi-level columns if present
        if hasattr(spy.columns, "levels"):
            spy.columns = spy.columns.get_level_values(0)

        closes = spy["Close"].values.astype(float)

        spy_return_10d = float((closes[-1] / closes[-11]) - 1) if len(closes) >= 11 else 0.0
        spy_return_21d = float((closes[-1] / closes[-22]) - 1) if len(closes) >= 22 else 0.0

        daily_rets = np.diff(closes[-22:]) / closes[-22:-1]
        spy_vol_20d = float(np.std(daily_rets, ddof=1) * np.sqrt(252))

        # RSI-14
        rets14 = np.diff(closes[-15:]) / closes[-15:-1]
        gains = np.where(rets14 > 0, rets14, 0.0)
        losses = np.where(rets14 < 0, -rets14, 0.0)
        avg_gain = float(np.mean(gains))
        avg_loss = float(np.mean(losses))
        spy_rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss) if avg_loss > 0 else 100.0

        sma200 = float(np.mean(closes[-200:])) if len(closes) >= 200 else float(closes[-1])
        spy_above_sma200 = 1.0 if closes[-1] > sma200 else 0.0

        daily_std_21d = float(np.std(daily_rets, ddof=1))
        spy_momentum_regime = (
            float(spy_return_10d / (daily_std_21d * np.sqrt(10)))
            if daily_std_21d > 1e-8 else 0.0
        )

        return {
            "spy_return_10d":     spy_return_10d,
            "spy_return_21d":     spy_return_21d,
            "spy_volatility_20d": spy_vol_20d,
            "spy_rsi_14":         spy_rsi,
            "spy_above_sma200":   spy_above_sma200,
            "spy_momentum_regime": spy_momentum_regime,
        }
    except Exception as exc:
        logger.warning("SPY fetch failed (%s) — market context features will be 0", exc)
        return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_spread_result(
    df: pd.DataFrame,
    model_id: str = "lgbm_v4",
) -> pd.DataFrame:
    """Score each stock in the spread result. Adds 'MLScore' (0–100) column.

    Returns df unchanged if the model checkpoint is not found or inference fails.
    """
    model_path = MODEL_DIR / f"{model_id}.pkl"
    if not model_path.exists():
        logger.info("ML model not found at %s — skipping ML scoring", model_path)
        return df

    try:
        return _score(df, model_path)
    except Exception as exc:
        logger.warning("ML scoring failed (%s) — MLScore will be absent", exc, exc_info=True)
        return df


# ---------------------------------------------------------------------------
# Internal scoring logic
# ---------------------------------------------------------------------------

def _build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build the ML feature DataFrame from a spread CSV result (without running inference).

    Returns a DataFrame with FEATURE_COLUMNS, ready for model.predict().
    """
    feat = pd.DataFrame(index=df.index)

    # return_pct: find the Return_* column
    return_col = next((c for c in df.columns if c.lower().startswith("return_")), None)
    feat["return_pct"] = pd.to_numeric(df[return_col], errors="coerce") if return_col else np.nan

    # Rename spread columns → ML feature names
    for src, dst in _SPREAD_TO_ML.items():
        if src in df.columns:
            feat[dst] = pd.to_numeric(df[src], errors="coerce")

    # Categorical: market_cap_cat
    if "MarketCapCategory" in df.columns:
        feat["market_cap_cat"] = (
            df["MarketCapCategory"].map(_MARKET_CAP_CAT_MAP).fillna(0).astype("int8")
        )
    else:
        feat["market_cap_cat"] = np.int8(0)

    # Categorical: exchange
    if "Exchange" in df.columns:
        feat["exchange"] = df["Exchange"].map(_EXCHANGE_MAP).fillna(0).astype("int8")
    else:
        feat["exchange"] = np.int8(0)

    # Categorical: sector (use pre-encoded column if available, otherwise encode from string)
    if "sector_encoded" in df.columns:
        feat["sector"] = pd.to_numeric(df["sector_encoded"], errors="coerce").fillna(-1).astype("int8")
    elif "Sector" in df.columns:
        feat["sector"] = (
            df["Sector"]
            .apply(lambda s: SECTOR_MAP.get(str(s).lower(), -1) if pd.notna(s) else -1)
            .astype("int8")
        )
    else:
        feat["sector"] = np.int8(-1)

    feat["has_fundamentals"] = (feat["sector"] >= 0).astype("int8")

    # Market context features (same for all stocks — today's SPY state)
    spy = _compute_spy_features()
    for col in MARKET_CONTEXT_COLUMNS:
        feat[col] = spy.get(col, 0.0)

    # Composite: earnings_yield × ROE — penalises cheap-but-deteriorating stocks
    if "earnings_yield" in feat.columns and "roe" in feat.columns:
        feat["earnings_yield_quality"] = feat["earnings_yield"] * feat["roe"]

    # Cross-sectional derived features
    spy_regime = spy.get("spy_momentum_regime", 0.0)
    ret = feat.get("return_pct", pd.Series(np.nan, index=feat.index))

    sector_median = feat.groupby(feat["sector"])["return_pct"].transform("median")
    feat["return_vs_sector"] = (ret - sector_median.fillna(ret)).fillna(0.0)
    feat["return_pct_x_regime"] = ret.fillna(0.0) * spy_regime

    # Ensure every expected feature column exists (fill missing with NaN → 0 after preprocessing)
    for col in FEATURE_COLUMNS:
        if col not in feat.columns:
            feat[col] = np.nan

    # Preprocessing: winsorize then cross-sectional rank-normalize float features
    float_cols = [c for c in FLOAT_FEATURES if c in feat.columns]
    mkt_cols = [c for c in MARKET_CONTEXT_COLUMNS if c in feat.columns]
    feat = winsorize(feat, float_cols + mkt_cols)
    feat = rank_normalize(feat, float_cols)

    # Ensure categoricals are int8
    for col in ["market_cap_cat", "exchange", "has_fundamentals"]:
        if col in feat.columns:
            feat[col] = feat[col].fillna(0).astype("int8")
    feat["sector"] = feat["sector"].fillna(-1).astype("int8")

    return feat[FEATURE_COLUMNS].fillna(0).astype(np.float32)


def _predict_raw(model, X: pd.DataFrame) -> np.ndarray:
    """Run model.predict() and return raw scores as numpy array."""
    return model.predict(X)


def _score(df: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    df = df.copy()
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    X = _build_feature_matrix(df)
    raw = _predict_raw(model, X)
    df["MLScore"] = pd.Series(raw, index=df.index).rank(pct=True).mul(100).round(1).values
    return df


def get_raw_scores(df: pd.DataFrame, model_path: Path) -> np.ndarray:
    """Return raw model predictions for df (no percentile ranking)."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    X = _build_feature_matrix(df)
    return _predict_raw(model, X)
