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
    "AnnualizedVol":      "volatility_20d",
    "DistanceFromHigh":   "distance_from_high",
    "AvgDrawdown":        "avg_drawdown",
    "VolumeConsistency":  "volume_consistency",
    "VolatilityRatio":    "volatility_ratio",
    "VolumeUpDnRatio":   "volume_trend_confirmation",
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
    "PriceVs52wkHighPct":   "price_vs_52wk_high",
    "PctDaysPositive21d":   "pct_days_positive_21d",
    # Long-horizon momentum (academic factors)
    "Momentum12_1":         "momentum_12_1",
    "MomentumIntermediate": "momentum_intermediate",
    "MomentumAccelLong":    "momentum_accel_long",
    "InfoDiscreteness":     "info_discreteness",
    # Fundamentals — actual column names from the spread CSV (merged from fundamentals.csv)
    "ROE":          "roe",
    "DebtEquity":   "debt_to_equity",
    "RevenueGrowth":"revenue_growth",
    "NetMargin":    "profit_margin",
    "CurrentRatio": "current_ratio",
    "ShortFloat":   "short_pct_float",
}

_MARKET_CAP_CAT_MAP = {
    "Mega": 5, "Large": 4, "Mid": 3, "Small": 2, "Micro": 1,
}

_EXCHANGE_MAP = {
    "NASDAQ": 0, "NYSE": 1, "S&P500": 2,
}

_FACTOR_CACHE_DIR = Path("data/factors")
_SCORE_CACHE = _FACTOR_CACHE_DIR / "mlscore_cache.parquet"


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
# VIX + WML factor features
# ---------------------------------------------------------------------------

def _compute_factor_features() -> dict:
    """Fetch/refresh VIX term structure and WML factor features for today."""
    import io, zipfile, urllib.request
    result: dict = {}

    # --- VIX term structure ---
    try:
        _FACTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        vix_cache = _FACTOR_CACHE_DIR / "vix_inference.parquet"
        refresh_vix = True
        if vix_cache.exists():
            import datetime as _dt
            age = (_dt.datetime.now() - _dt.datetime.fromtimestamp(vix_cache.stat().st_mtime))
            if age.total_seconds() < 86400:
                refresh_vix = False
        if refresh_vix:
            import yfinance as yf
            raw = yf.download("^VIX ^VIX3M", period="3mo", interval="1d", progress=False, auto_adjust=False)
            if not raw.empty:
                if hasattr(raw.columns, "levels"):
                    close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", axis=1, level=1)
                else:
                    close = raw[["Close"]]
                df = pd.DataFrame({
                    "VIX": close.get("^VIX", pd.Series(dtype=float)),
                    "VIX3M": close.get("^VIX3M", pd.Series(dtype=float)),
                }).dropna(how="all")
                df.to_parquet(vix_cache)
        if vix_cache.exists():
            df = pd.read_parquet(vix_cache)
            vix_s = df["VIX"].dropna()
            vix3m_s = df["VIX3M"].dropna()
            if len(vix_s) >= 22:
                vix_val = float(vix_s.iloc[-1])
                result["vix_level"] = vix_val
                result["vix_1m_change"] = vix_val - float(vix_s.iloc[-22])
                trailing = vix_s.iloc[-252:] if len(vix_s) >= 252 else vix_s
                result["vix_percentile"] = float((trailing < vix_val).mean())
                if len(vix3m_s) > 0 and vix_val > 0:
                    result["vix_term_structure"] = float(vix3m_s.iloc[-1]) / vix_val
    except Exception as exc:
        logger.warning("VIX feature computation failed (%s) — VIX features will be 0", exc)

    # --- WML factor features ---
    try:
        wml_cache = _FACTOR_CACHE_DIR / "umd_daily.parquet"
        refresh_wml = True
        if wml_cache.exists():
            import datetime as _dt
            age = (_dt.datetime.now() - _dt.datetime.fromtimestamp(wml_cache.stat().st_mtime))
            if age.days < 7:
                refresh_wml = False
        if refresh_wml:
            url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                fname = next(f for f in z.namelist() if f.lower().endswith(".csv"))
                raw_text = z.read(fname).decode("utf-8", errors="replace")
            lines = [l for l in raw_text.splitlines() if l.strip()[:8].strip().isdigit()]
            df_wml = pd.read_csv(io.StringIO("\n".join(lines)), header=None, names=["Date", "Mom"])
            df_wml["Date"] = pd.to_datetime(df_wml["Date"].astype(str).str.strip(), format="%Y%m%d", errors="coerce")
            df_wml = df_wml.dropna(subset=["Date"]).set_index("Date")
            df_wml["Mom"] = pd.to_numeric(df_wml["Mom"], errors="coerce") / 100.0
            df_wml.to_parquet(wml_cache)
        if wml_cache.exists():
            wml = pd.read_parquet(wml_cache)["Mom"].dropna()
            if len(wml) >= 126:
                result["wml_realized_vol_126d"] = float(wml.iloc[-126:].std() * np.sqrt(252))
            if len(wml) >= 21:
                result["wml_trailing_1m"] = float((1 + wml.iloc[-21:]).prod() - 1)
            if len(wml) >= 63:
                result["wml_trailing_3m"] = float((1 + wml.iloc[-63:]).prod() - 1)
    except Exception as exc:
        logger.warning("WML feature computation failed (%s) — WML features will be 0", exc)

    return result


# ---------------------------------------------------------------------------
# Score delta
# ---------------------------------------------------------------------------

def _get_score_delta(tickers: pd.Series, current_scores: pd.Series) -> pd.Series:
    """Compare current MLScores to the previous run's cached scores.

    Falling score on a held stock = warning signal. NaN if no prior cache or ticker is new.
    Saves current scores to cache for the next run.
    """
    delta = pd.Series(np.nan, index=current_scores.index)
    if _SCORE_CACHE.exists():
        try:
            prev = pd.read_parquet(_SCORE_CACHE).set_index("Ticker")["MLScore"]
            for idx, ticker in tickers.items():
                if ticker in prev.index:
                    delta[idx] = round(float(current_scores[idx]) - float(prev[ticker]), 1)
        except Exception as exc:
            logger.warning("Score cache read failed (%s) — MLScoreDelta will be NaN", exc)
    try:
        _SCORE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Ticker": tickers.values, "MLScore": current_scores.values}).to_parquet(
            _SCORE_CACHE, index=False
        )
    except Exception as exc:
        logger.warning("Score cache save failed (%s)", exc)
    return delta


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

    # log_momentum_12_1: compress extreme values (2000%+ stocks) to prevent winsorization
    # from flattening the signal. Computed from the already-mapped momentum_12_1.
    if "momentum_12_1" in feat.columns:
        feat["log_momentum_12_1"] = np.log1p(feat["momentum_12_1"].clip(lower=-0.99))

    # Derived fundamentals: compute from price ratios in the spread CSV
    # earnings_yield = 1/PE, book_to_market = 1/PB, sales_to_price = 1/PS
    for ratio_col, feat_name in [("PE", "earnings_yield"), ("PB", "book_to_market"), ("PS", "sales_to_price")]:
        if ratio_col in df.columns:
            ratio = pd.to_numeric(df[ratio_col], errors="coerce")
            if feat_name == "earnings_yield":
                # Only meaningful for profitable companies (positive PE).
                # Cap at 0.20 (P/E floor = 5) to prevent pre-revenue data artifacts
                # from dominating sector-relative rankings.
                ratio = ratio.where(ratio > 0)
                feat[feat_name] = (1.0 / ratio).replace([np.inf, -np.inf], np.nan).clip(upper=0.20)
            else:
                feat[feat_name] = (1.0 / ratio).replace([np.inf, -np.inf], np.nan)

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

    # Market context features (same for all stocks — today's SPY state + VIX/WML)
    spy = _compute_spy_features()
    factors = _compute_factor_features()
    market_ctx = {**spy, **factors}
    for col in MARKET_CONTEXT_COLUMNS:
        feat[col] = market_ctx.get(col, 0.0)

    # Composite: earnings_yield × ROE — penalises cheap-but-deteriorating stocks
    if "earnings_yield" in feat.columns and "roe" in feat.columns:
        feat["earnings_yield_quality"] = feat["earnings_yield"] * feat["roe"]

    # Cross-sectional derived features
    spy_regime = spy.get("spy_momentum_regime", 0.0)
    ret = feat.get("return_pct", pd.Series(np.nan, index=feat.index))

    sector_grp = feat.groupby(feat["sector"])
    feat["sector_return_rank"] = sector_grp["return_pct"].transform(lambda x: x.rank(pct=True)).fillna(0.5)
    feat["return_pct_x_regime"] = ret.fillna(0.0) * spy_regime

    if "earnings_yield" in feat.columns:
        feat["earnings_yield_vs_sector"] = (
            feat["earnings_yield"] - sector_grp["earnings_yield"].transform("median")
        ).fillna(0.0)
    if "book_to_market" in feat.columns:
        feat["book_to_market_vs_sector"] = (
            feat["book_to_market"] - sector_grp["book_to_market"].transform("median")
        ).fillna(0.0)

    # Long-horizon momentum cross-sectional features
    if "momentum_12_1" in feat.columns:
        feat["sector_momentum_rank"] = (
            sector_grp["momentum_12_1"].transform(lambda x: x.rank(pct=True))
        ).fillna(0.5)
        # RS rating: universe-wide percentile rank of momentum_12_1 (IBD-style)
        feat["rs_rating"] = feat["momentum_12_1"].rank(pct=True).fillna(0.5)
    if "earnings_yield" in feat.columns and "momentum_12_1" in feat.columns:
        feat["value_x_momentum"] = feat["earnings_yield"] * feat["momentum_12_1"]
    if "roe" in feat.columns and "momentum_12_1" in feat.columns:
        feat["quality_x_momentum"] = feat["roe"] * feat["momentum_12_1"]

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
    """Run model.predict() and return raw scores as numpy array.

    If the model was trained on a subset of FEATURE_COLUMNS (e.g. value features
    were excluded for a momentum-only model), subset X to match.
    """
    try:
        trained_features = model.model.feature_name_
        if list(trained_features) != list(X.columns):
            X = X[list(trained_features)]
    except AttributeError:
        pass
    return model.predict(X)


def _score(df: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    df = df.copy()
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    X = _build_feature_matrix(df)
    raw = _predict_raw(model, X)
    scores = pd.Series(raw, index=df.index).rank(pct=True).mul(100).round(1)
    df["MLScore"] = scores.values
    if "Ticker" in df.columns:
        df["MLScoreDelta"] = _get_score_delta(df["Ticker"], scores).values
    return df


def get_raw_scores(df: pd.DataFrame, model_path: Path) -> np.ndarray:
    """Return raw model predictions for df (no percentile ranking)."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    X = _build_feature_matrix(df)
    return _predict_raw(model, X)
