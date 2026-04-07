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
    SPREAD_TO_ML,
    winsorize,
    rank_normalize,
)

logger = logging.getLogger(__name__)

# Look in repo-tracked models/ first, fall back to ~/.morningalpha/models/
_REPO_MODELS = Path(__file__).parents[2] / "models"
_HOME_MODELS = Path.home() / ".morningalpha" / "models"
MODEL_DIR = _REPO_MODELS if _REPO_MODELS.exists() else _HOME_MODELS

# Column mapping imported from features.py — single source of truth.
_SPREAD_TO_ML = SPREAD_TO_ML

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

_SPY_CACHE = _FACTOR_CACHE_DIR / "spy_inference.parquet"
_NETWORK_TIMEOUT = 20  # seconds before giving up on any download


def _download_with_timeout(fn, *args, **kwargs):
    """Run a download function in a thread; return None if it exceeds _NETWORK_TIMEOUT."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as _Timeout
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args, **kwargs)
        try:
            return fut.result(timeout=_NETWORK_TIMEOUT)
        except _Timeout:
            logger.warning("Download timed out after %ds — skipping", _NETWORK_TIMEOUT)
            return None
        except Exception as exc:
            logger.warning("Download failed (%s) — skipping", exc)
            return None


def _compute_spy_features() -> dict:
    """Fetch recent SPY data and compute today's market context features.

    Result is cached for 4 hours so repeated calls within the same scoring run
    (one per model) hit disk instead of re-downloading.
    """
    try:
        import datetime as _dt
        _FACTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        refresh = True
        if _SPY_CACHE.exists():
            age = _dt.datetime.now() - _dt.datetime.fromtimestamp(_SPY_CACHE.stat().st_mtime)
            if age.total_seconds() < 4 * 3600:
                refresh = False

        if refresh:
            import yfinance as yf
            spy = _download_with_timeout(
                yf.download, "SPY", period="1y", interval="1d", progress=False, auto_adjust=True
            )
            if spy is not None and not spy.empty:
                if hasattr(spy.columns, "levels"):
                    spy.columns = spy.columns.get_level_values(0)
                spy[["Close"]].to_parquet(_SPY_CACHE)

        if not _SPY_CACHE.exists():
            return {}

        spy_df = pd.read_parquet(_SPY_CACHE)
        closes = spy_df["Close"].dropna().values.astype(float)
        if len(closes) < 22:
            return {}
        import yfinance as yf  # noqa: F811 — needed for fallback path above

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


# ---------------------------------------------------------------------------
# LSTM shared helpers
# ---------------------------------------------------------------------------

def _lstm_load_model_and_device(model_path: Path, dropout: float = 0.0):
    """Load LSTM checkpoint, rebuild model, select device. Returns (model, ckpt, device)."""
    import torch
    from morningalpha.ml.lstm_model import StockPriceLSTM
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    cfg = dict(ckpt["config"])
    cfg["dropout"] = dropout
    model = StockPriceLSTM.from_config(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    if dropout == 0.0:
        model.eval()
    else:
        model.train()  # keep dropout active for MC paths
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    return model, ckpt, device


def _lstm_prepare_dataset(ckpt: dict) -> pd.DataFrame:
    """Load dataset.parquet and apply the checkpoint's stored StandardScaler."""
    dataset_path = Path("data/training/dataset.parquet")
    if not dataset_path.exists():
        raise FileNotFoundError(f"LSTM dataset not found: {dataset_path}")
    ds = pd.read_parquet(dataset_path)
    scaler_info = ckpt.get("feature_scaler", {})
    if scaler_info and "cols" in scaler_info:
        full_cols  = scaler_info["cols"]
        mean_map   = dict(zip(full_cols, scaler_info["mean"]))
        scale_map  = dict(zip(full_cols, scaler_info["scale"]))
        sc_cols    = [c for c in full_cols if c in ds.columns]
        for col in sc_cols:
            ds[col] = (ds[col].fillna(0) - mean_map[col]) / max(scale_map[col], 1e-8)
    feat_cols = ckpt["feature_cols"]
    for c in [c for c in feat_cols if c not in ds.columns]:
        ds[c] = 0.0
    return ds.sort_values(["ticker", "date"])


def _lstm_build_batch(tickers, ticker_groups, feat_cols: list, lookback: int,
                      batch_size: int = 512):
    """Yield (seq_batch [B,T,F], ticker_indices) in chunks of batch_size."""
    import torch
    seqs: list = []
    indices: list = []
    for i, ticker in enumerate(tickers):
        if ticker not in ticker_groups.groups:
            continue
        grp = ticker_groups.get_group(ticker)
        if len(grp) < lookback:
            continue
        seq = np.nan_to_num(grp[feat_cols].tail(lookback).values.astype(np.float32), nan=0.0)
        seqs.append(seq)
        indices.append(i)
        if len(seqs) == batch_size:
            yield torch.tensor(np.stack(seqs)), indices
            seqs, indices = [], []
    if seqs:
        yield torch.tensor(np.stack(seqs)), indices


# ---------------------------------------------------------------------------
# Public LSTM API
# ---------------------------------------------------------------------------

def get_lstm_raw_scores(df_score: pd.DataFrame, model_path: Path) -> np.ndarray:
    """Return raw LSTM predictions (63d horizon) for each row in df_score.

    Unlike LightGBM (cross-sectional snapshot), the LSTM needs a 60-day
    historical sequence per ticker from data/training/dataset.parquet.
    Returns the 63d-horizon prediction for each ticker, in df_score row order.
    Falls back to 0.0 for tickers with insufficient history.
    """
    try:
        import torch
        from morningalpha.ml.lstm_model import LSTM_HORIZONS
    except ImportError as exc:
        raise ImportError(f"PyTorch required for LSTM inference: {exc}") from exc

    model, ckpt, device = _lstm_load_model_and_device(model_path, dropout=0.0)
    feat_cols    = ckpt["feature_cols"]
    lookback     = ckpt.get("lookback", 60)
    horizon_days = ckpt.get("horizon_days", LSTM_HORIZONS)
    is_combo     = ckpt.get("config", {}).get("combo", False)
    n_horizons   = len(horizon_days)
    try:
        idx_63d = horizon_days.index(63)
    except ValueError:
        idx_63d = n_horizons - 1
    # In combo mode outputs are [rank_0..rank_N, clip_0..clip_N].
    # For scoring we want rank half — idx_63d is already correct (rank half is first).
    _ = is_combo  # used below in batch loop

    try:
        ds = _lstm_prepare_dataset(ckpt)
    except FileNotFoundError as exc:
        logger.warning("%s — returning zeros", exc)
        return np.zeros(len(df_score), dtype=np.float32)

    ticker_groups = ds.groupby("ticker", sort=False)
    tickers       = df_score["Ticker"].values if "Ticker" in df_score.columns else df_score.index.values
    raw_scores    = np.zeros(len(tickers), dtype=np.float32)

    for batch, valid_indices in _lstm_build_batch(tickers, ticker_groups, feat_cols, lookback):
        with torch.no_grad():
            preds = model(batch.to(device)).cpu().numpy()  # [B, n_horizons]
        for j, idx in enumerate(valid_indices):
            raw_scores[idx] = float(preds[j, idx_63d])

    n_scored = int((raw_scores != 0.0).sum())
    logger.info("LSTM inference: scored %d / %d tickers (63d horizon)", n_scored, len(tickers))
    return raw_scores


def generate_forecast_paths(
    df_score: pd.DataFrame,
    model_path: Path,
    n_paths: int = 6,
) -> dict:
    """Generate MC-dropout forecast paths for all scored tickers.

    Returns a dict ready to serialise as forecast_paths.json:
    {
        "model":     "lstm_clip_v1",
        "horizons":  [1, 5, 10, 21, 63],
        "generated_at": "2026-04-05",
        "last_prices": {"AAPL": 185.5, ...},
        "paths": {
            "AAPL": [[log_ret_1d, log_ret_5d, ...], ...],   # n_paths × n_horizons
            ...
        }
    }

    log-returns are relative to the ticker's last known price; the frontend
    converts them to price levels via price * exp(log_ret).
    Paths that exceed 2× historical max or go below historical min are filtered.
    Tickers with < lookback rows in the dataset are omitted.
    """
    try:
        import torch
        from morningalpha.ml.lstm_model import LSTM_HORIZONS
    except ImportError as exc:
        raise ImportError(f"PyTorch required for LSTM forecast paths: {exc}") from exc

    import torch as _torch
    _ckpt_tmp = _torch.load(str(model_path), map_location="cpu", weights_only=False)
    _mc_dropout = float(_ckpt_tmp.get("config", {}).get("dropout", 0.3))
    model, ckpt, device = _lstm_load_model_and_device(model_path, dropout=_mc_dropout)
    feat_cols    = ckpt["feature_cols"]
    lookback     = ckpt.get("lookback", 60)
    horizon_days = ckpt.get("horizon_days", LSTM_HORIZONS)
    is_combo     = ckpt.get("config", {}).get("combo", False)
    n_horizons   = len(horizon_days)

    try:
        ds = _lstm_prepare_dataset(ckpt)
    except FileNotFoundError as exc:
        logger.warning("%s — returning empty paths", exc)
        return {}

    ticker_groups = ds.groupby("ticker", sort=False)
    tickers       = df_score["Ticker"].values if "Ticker" in df_score.columns else df_score.index.values

    # Resolve last price from spread CSV (used for path filtering + JSON output).
    # The spread CSV has no raw close column but does have SMA20 + PriceToSMA20Pct,
    # so we reconstruct: price = SMA20 * (1 + PriceToSMA20Pct / 100).
    price_map: dict[str, float] = {}
    if "Ticker" in df_score.columns:
        if "SMA20" in df_score.columns and "PriceToSMA20Pct" in df_score.columns:
            sma20  = pd.to_numeric(df_score["SMA20"], errors="coerce")
            pct    = pd.to_numeric(df_score["PriceToSMA20Pct"], errors="coerce")
            prices = sma20 * (1.0 + pct / 100.0)
            price_map = (
                df_score.assign(_price=prices)
                .dropna(subset=["_price"])
                .set_index("Ticker")["_price"]
                .to_dict()
            )
        else:
            # Fallback: any column that looks like a raw price
            price_col = next(
                (c for c in df_score.columns
                 if c.lower() in ("close", "price", "last", "lastprice")), None
            )
            if price_col:
                price_map = df_score.set_index("Ticker")[price_col].dropna().to_dict()

    paths_out:  dict[str, list] = {}
    prices_out: dict[str, float] = {}

    for batch, valid_indices in _lstm_build_batch(tickers, ticker_groups, feat_cols, lookback):
        batch_tickers = [tickers[i] for i in valid_indices]
        x = batch.to(device)

        # n_paths forward passes with MC dropout active
        all_paths = []
        for _ in range(n_paths):
            with torch.no_grad():
                preds = model(x).cpu().numpy()  # [B, n_horizons] or [B, 2*n_horizons]
            # Combo: use clip half (second n_horizons cols) for price-level fan chart
            if is_combo:
                preds = preds[:, n_horizons:]
            all_paths.append(preds)
        # all_paths: [n_paths, B, n_horizons] → [B, n_paths, n_horizons]
        stacked = np.stack(all_paths, axis=1)

        for j, ticker in enumerate(batch_tickers):
            last_price = float(price_map.get(ticker, 0.0))
            ticker_paths = stacked[j]  # [n_paths, n_horizons]

            # Filter extreme paths: reject if any horizon implies price < 0
            # or > 3× last price (catches runaway MC samples)
            if last_price > 0:
                implied = last_price * np.exp(ticker_paths)  # [n_paths, n_horizons]
                keep = (implied > 0).all(axis=1) & (implied < last_price * 3).all(axis=1)
                ticker_paths = ticker_paths[keep]

            if len(ticker_paths) == 0:
                continue

            paths_out[ticker]  = [[round(float(v), 5) for v in path] for path in ticker_paths]
            prices_out[ticker] = last_price

    from datetime import date
    return {
        "model":        model_path.stem,
        "horizons":     horizon_days,
        "generated_at": str(date.today()),
        "last_prices":  prices_out,
        "paths":        paths_out,
    }


def ckpt_dropout(ckpt: dict) -> float:
    """Return the training-time dropout rate from a checkpoint config."""
    return float(ckpt.get("config", {}).get("dropout", 0.3))


def get_st_raw_scores(df: pd.DataFrame, model_path: Path) -> np.ndarray:
    """Return raw Set Transformer scores for each row in df.

    Unlike LightGBM (flat feature matrix), the ST needs stocks grouped by
    sector to compute cross-stock attention. Scores are returned in the same
    row order as df. Falls back to 0.5 for stocks in sectors with < 2 members.
    """
    try:
        import torch
        from morningalpha.ml.set_transformer import SectorSetRanker
    except ImportError as exc:
        raise ImportError(f"PyTorch required for Set Transformer inference: {exc}") from exc

    # Build feature matrix (same preprocessing as LightGBM)
    X = _build_feature_matrix(df)

    # Load checkpoint — contains model state, config, and feature_cols
    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    ckpt_feature_cols = checkpoint.get("feature_cols", list(X.columns))

    # Align feature order to match training
    missing = set(ckpt_feature_cols) - set(X.columns)
    if missing:
        logger.warning("ST inference: %d features missing from feature matrix: %s", len(missing), missing)
    X = X.reindex(columns=ckpt_feature_cols, fill_value=0.0).fillna(0.0)

    # Select device: Apple Metal (MPS) > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("ST inference: using Apple Metal (MPS)")
    else:
        device = torch.device("cpu")

    # Build model with training-time architecture
    model = SectorSetRanker(
        dim_input=len(ckpt_feature_cols),
        d_model=config.get("d_model", 128),
        num_heads=config.get("num_heads", 4),
        num_blocks=config.get("num_blocks", 3),
        dropout=0.0,  # disable dropout at inference
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    max_set_size = config.get("max_set_size", 80)
    raw_scores = np.full(len(df), 0.5, dtype=np.float32)
    D = X.shape[1]

    sector_values = X["sector"].values if "sector" in X.columns else np.zeros(len(df), dtype=np.int8)

    # Phase 1: collect all chunks across all sectors into flat lists.
    # For sectors larger than max_set_size we use overlapping windows (50% overlap)
    # and accumulate scores per stock. All chunks are padded to max_set_size so we
    # can stack them into one batched tensor and do a single forward pass.
    chunk_feats_list: list[np.ndarray] = []
    chunk_sizes: list[int] = []
    # Each entry is either a global index array (direct write) or a tuple
    # (sector_id, local_start, local_end) for overlapping-window accumulation.
    chunk_dest: list = []

    # Per-sector accumulation buffers for oversized sectors
    sector_acc: dict = {}  # sector_id -> (sum [N], cnt [N], global_indices [N])

    step = max_set_size // 2  # 50% overlap

    for sector_id in np.unique(sector_values):
        indices = np.where(sector_values == sector_id)[0]
        if len(indices) < 2:
            continue

        feats = X.iloc[indices].values.astype(np.float32)
        n = len(feats)

        if n <= max_set_size:
            chunk_feats_list.append(feats)
            chunk_sizes.append(n)
            chunk_dest.append(indices)
        else:
            sector_acc[sector_id] = (
                np.zeros(n, dtype=np.float32),
                np.zeros(n, dtype=np.int32),
                indices,
            )
            for start in range(0, n, step):
                end = min(start + max_set_size, n)
                chunk_feats_list.append(feats[start:end])
                chunk_sizes.append(end - start)
                chunk_dest.append((sector_id, start, end))
                if end == n:
                    break

    if not chunk_feats_list:
        return raw_scores

    # Phase 2: pad all chunks to max_set_size and run ONE batched forward pass.
    C = len(chunk_feats_list)
    padded = np.zeros((C, max_set_size, D), dtype=np.float32)
    masks = np.zeros((C, max_set_size), dtype=bool)
    for i, (feats, n) in enumerate(zip(chunk_feats_list, chunk_sizes)):
        padded[i, :n] = feats
        masks[i, :n] = True

    feat_t = torch.tensor(padded).to(device)           # [C, max_set_size, D]
    mask_t = torch.tensor(masks).to(device)            # [C, max_set_size]
    with torch.no_grad():
        all_scores = model(feat_t, mask_t).cpu().numpy()  # [C, max_set_size]

    # Phase 3: scatter scores back to raw_scores.
    for i, (dest, n) in enumerate(zip(chunk_dest, chunk_sizes)):
        scores = all_scores[i, :n]
        if isinstance(dest, np.ndarray):
            raw_scores[dest] = scores
        else:
            sector_id, start, end = dest
            sum_arr, cnt_arr, _ = sector_acc[sector_id]
            sum_arr[start:end] += scores
            cnt_arr[start:end] += 1

    for sum_arr, cnt_arr, global_indices in sector_acc.values():
        raw_scores[global_indices] = sum_arr / np.maximum(cnt_arr, 1)

    return raw_scores
