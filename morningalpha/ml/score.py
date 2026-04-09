"""alpha ml score — score all period CSVs with registered ML models.

Usage:
    alpha ml score
    alpha ml score --data-dir data/latest --models-dir models
"""
import json
import logging
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import rich_click as click
from rich.console import Console
from rich.table import Table
from scipy.stats import spearmanr

# Suppress pandas FutureWarning about object-dtype downcasting in fillna/astype chains.
# These are cosmetic deprecations that don't affect correctness.
warnings.filterwarnings(
    "ignore",
    message="Downcasting object dtype arrays",
    category=FutureWarning,
)

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_DATA_DIR = "data/latest"
_WEIGHTS_PATH        = Path("data/factors/model_weights.json")
_LEDGER_PATH         = Path("data/factors/predictions_ledger.parquet")
_IC_TIMESERIES_PATH  = Path("data/factors/live_ic_timeseries.parquet")
_MODEL_HEALTH_PATH   = Path("data/factors/model_health.json")
_CORRECTION_PATH     = Path("data/factors/correction_models.joblib")
_CALIBRATION_PATH    = Path("data/factors/calibration_models.joblib")
_CALIB_DAILY_PATH    = Path("data/factors/calibration_daily.parquet")
_CALIB_DIR           = Path("data/factors")
# How many trading days before we evaluate a prediction against realized returns
_EVAL_HORIZON   = 63
DEFAULT_MODELS_DIR = "models"
# 3M is the canonical scoring file — matches the ~3-month lookback used in training.
# Scores are computed once here, then merged into all period CSVs by ticker.
SCORE_SOURCE = "stocks_3m.csv"
PERIOD_FILES = ["stocks_2w.csv", "stocks_1m.csv", "stocks_3m.csv", "stocks_6m.csv"]


# ---------------------------------------------------------------------------
# Dynamic model weighting (Hedge algorithm on rolling IC)
# ---------------------------------------------------------------------------

def _load_model_weights(active_model_ids: list) -> dict:
    """Return normalized weights for each active model.

    Reads data/factors/model_weights.json. Falls back to equal weights if the
    file is missing or a model isn't listed yet.
    """
    equal = {m: 1.0 / len(active_model_ids) for m in active_model_ids}
    if not _WEIGHTS_PATH.exists():
        return equal
    try:
        with open(_WEIGHTS_PATH) as f:
            data = json.load(f)
        stored = data.get("weights", {})
        known = [float(stored[m]) for m in active_model_ids if m in stored]
        default_w = min(known) if known else 1.0 / len(active_model_ids)
        weights = {m: float(stored.get(m, default_w)) for m in active_model_ids}
        total = sum(weights.values())
        return {m: w / total for m, w in weights.items()} if total > 0 else equal
    except Exception as exc:
        logger.warning("Could not load model weights (%s) — using equal weights", exc)
        return equal


def _save_model_weights(weights: dict, ic_entry: dict | None = None) -> None:
    """Persist updated weights and append an IC history entry."""
    try:
        _WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if _WEIGHTS_PATH.exists():
            with open(_WEIGHTS_PATH) as f:
                existing = json.load(f)

        history: list = existing.get("ic_history", [])
        if ic_entry:
            history.append(ic_entry)
            window = existing.get("window_weeks", 12)
            history = history[-window:]  # keep rolling window only

        payload = {
            "updated":      str(date.today()),
            "method":       existing.get("method", "rolling_ic_hedge"),
            "window_weeks": existing.get("window_weeks", 12),
            "eta":          existing.get("eta", 0.5),
            "weights":      weights,
            "ic_history":   history,
        }
        with open(_WEIGHTS_PATH, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as exc:
        logger.warning("Could not save model weights (%s)", exc)


def _hedge_update(current_weights: dict, ic_by_model: dict, eta: float = 0.5) -> dict:
    """Apply one step of the Hedge update rule.

    w_i(t+1) ∝ w_i(t) * exp(η * IC_i)

    IC acts as the reward signal: higher IC = model gets more weight next period.
    Negative IC models are penalized but never zeroed out entirely.
    """
    new_weights = {
        m: current_weights.get(m, 1.0) * np.exp(eta * ic_by_model.get(m, 0.0))
        for m in current_weights
    }
    total = sum(new_weights.values())
    return {m: w / total for m, w in new_weights.items()}


# ---------------------------------------------------------------------------
# Prediction ledger — append-only record for future IC evaluation
# ---------------------------------------------------------------------------

# Three eval horizons: (trading_days, calendar_buffer, column_suffix)
# Aligned with training targets (forward_5d, forward_21d, forward_63d).
_HORIZONS = [
    (5,  7,  "5d"),   # ≈ 1 week  — first results after ~2 weeks
    (21, 7,  "21d"),  # ≈ 1 month — medium-term signal, matches training target
    (63, 14, "63d"),  # ≈ 13 wks  — primary calibration target
]


def _td(trading_days: int, buffer: int) -> int:
    """Approximate calendar days for a given number of trading days + buffer."""
    return trading_days * 7 // 5 + buffer


def _fetch_market_context(today: pd.Timestamp) -> dict:
    """Fetch SPY 21-day trailing return and VIX close for today.

    Used as context features for the residual correction and calibration models.
    Returns NaN values on failure (safe fallback).
    """
    try:
        import yfinance as yf
        start = (today - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
        end   = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        spy = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
        vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)

        spy_ret = float("nan")
        vix_val = float("nan")

        # Handle multi-level columns from newer yfinance
        spy_close = spy["Close"] if "Close" in spy.columns else spy.iloc[:, 0]
        vix_close = vix["Close"] if "Close" in vix.columns else vix.iloc[:, 0]

        if isinstance(spy_close, pd.DataFrame):
            spy_close = spy_close.iloc[:, 0]
        if isinstance(vix_close, pd.DataFrame):
            vix_close = vix_close.iloc[:, 0]

        spy_close = spy_close.dropna()
        vix_close = vix_close.dropna()

        if len(spy_close) >= 22:
            spy_ret = float(spy_close.iloc[-1] / spy_close.iloc[-22] - 1)
        if len(vix_close) >= 1:
            vix_val = float(vix_close.iloc[-1])

        return {"market_return_21d": spy_ret, "vix_at_prediction": vix_val}
    except Exception as exc:
        logger.warning("Could not fetch market context (%s) — using NaN", exc)
        return {"market_return_21d": float("nan"), "vix_at_prediction": float("nan")}


_SECTOR_CODES: dict[str, int] = {
    "Technology": 1, "Healthcare": 2, "Financial Services": 3,
    "Consumer Cyclical": 4, "Communication Services": 5, "Industrials": 6,
    "Consumer Defensive": 7, "Energy": 8, "Utilities": 9,
    "Real Estate": 10, "Basic Materials": 11,
}


def _append_predictions_ledger(df_score: pd.DataFrame, raw_scores: dict) -> None:
    """Append this run's raw model scores to the predictions ledger.

    Stores eval_after dates for 5d, 21d, 63d horizons plus context features
    (sector, market return, VIX, momentum bucket) needed for the residual
    correction and calibration models.
    """
    try:
        _LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        today = pd.Timestamp(date.today())

        rows = pd.DataFrame({"ticker": df_score["Ticker"].values if "Ticker" in df_score.columns else df_score.index})
        rows["scored_date"] = today

        for td, buf, suffix in _HORIZONS:
            rows[f"eval_after_{suffix}"] = today + timedelta(days=_td(td, buf))

        price_col = next((c for c in df_score.columns if c.lower().startswith("price") or c == "Close"), None)
        rows["price_at_score"] = df_score[price_col].values if price_col else np.nan

        for model_id, raw in raw_scores.items():
            rows[f"raw_{model_id}"] = raw

        # Context features for correction/calibration models
        if "Sector" in df_score.columns:
            rows["sector_code"] = df_score["Sector"].map(_SECTOR_CODES).fillna(0).astype(np.int8).values
        else:
            rows["sector_code"] = np.int8(0)

        if "Momentum12_1" in df_score.columns:
            mom = pd.to_numeric(df_score["Momentum12_1"], errors="coerce")
            rows["momentum_bucket"] = pd.qcut(mom.rank(method="first"), q=5, labels=False).fillna(2).astype(np.int8).values
        else:
            rows["momentum_bucket"] = np.int8(2)

        ctx = _fetch_market_context(today)
        rows["market_return_21d"]  = ctx["market_return_21d"]
        rows["vix_at_prediction"]  = ctx["vix_at_prediction"]
        rows["is_backfill"]        = False

        # Return columns filled in later by _evaluate_and_update_weights
        for _, _, suffix in _HORIZONS:
            rows[f"realized_return_{suffix}"] = np.nan
            rows[f"matured_{suffix}"]         = False

        existing = pd.read_parquet(_LEDGER_PATH) if _LEDGER_PATH.exists() else pd.DataFrame()
        updated = pd.concat([existing, rows], ignore_index=True)
        updated.to_parquet(_LEDGER_PATH, index=False)
        console.print(f"[dim]Ledger updated: {len(rows)} predictions appended → {_LEDGER_PATH}[/dim]")
    except Exception as exc:
        logger.warning("Ledger append failed (%s) — skipping", exc)


def _evaluate_and_update_weights(df_current: pd.DataFrame, raw_scores: dict) -> None:
    """Backfill realized returns for matured rows across all three horizons.

    For each horizon (5d, 13d, 63d), finds rows whose eval_after date has
    passed, computes realized return from the current spread prices, and writes
    it back to the ledger.  Uses the longest available horizon's IC for the
    Hedge weight update (63d preferred → 13d → 5d).
    """
    if not _LEDGER_PATH.exists():
        return

    try:
        ledger = pd.read_parquet(_LEDGER_PATH)
        today  = pd.Timestamp(date.today())

        price_col = next((c for c in df_current.columns if c.lower().startswith("price") or c == "Close"), None)
        if price_col is None or "Ticker" not in df_current.columns:
            return
        current_prices = df_current.set_index("Ticker")[price_col].rename("current_price")

        any_new = False
        for _, _, suffix in _HORIZONS:
            eval_col    = f"eval_after_{suffix}"
            ret_col     = f"realized_return_{suffix}"
            matured_col = f"matured_{suffix}"

            # Skip horizons not yet in the ledger (old ledger schema)
            if eval_col not in ledger.columns:
                continue

            matured_series = ledger.get(matured_col, pd.Series(False, index=ledger.index))
            if matured_series is None:
                matured_series = pd.Series(False, index=ledger.index)
            matured_bool = matured_series.fillna(False).astype(bool)
            due_mask = (ledger[eval_col] <= today) & (~matured_bool)
            if not due_mask.any():
                continue

            due = ledger[due_mask].copy()
            due = due.join(current_prices, on="ticker", how="inner")
            due = due.dropna(subset=["price_at_score", "current_price"])
            due = due[due["price_at_score"] > 0]
            if due.empty:
                continue

            due[ret_col] = due["current_price"] / due["price_at_score"] - 1
            ledger.loc[due.index, ret_col]     = due[ret_col].values
            ledger.loc[due.index, matured_col] = True
            any_new = True
            console.print(f"[dim]Backfilled {len(due)} realized_return_{suffix} rows[/dim]")

        if any_new:
            ledger.to_parquet(_LEDGER_PATH, index=False)

        # Hedge update — use the longest horizon that has enough matured data
        ic_by_model: dict = {}
        used_horizon = None
        for _, _, suffix in reversed(_HORIZONS):   # 63d first, then 21d, then 5d
            ret_col     = f"realized_return_{suffix}"
            matured_col = f"matured_{suffix}"
            if ret_col not in ledger.columns:
                continue
            matured_col_vals = ledger.get(matured_col, pd.Series(False, index=ledger.index))
            if matured_col_vals is None:
                matured_col_vals = pd.Series(False, index=ledger.index)
            mature = ledger[matured_col_vals.fillna(False).astype(bool)]
            if len(mature) < 50:
                continue
            model_cols = [c for c in mature.columns if c.startswith("raw_")]
            for col in model_cols:
                model_id = col[len("raw_"):]
                valid = mature[[ret_col, col]].dropna()
                if len(valid) < 20:
                    continue
                ic = float(spearmanr(valid[col], valid[ret_col]).correlation)
                ic_by_model[model_id] = ic
            if ic_by_model:
                used_horizon = suffix
                break

        if not ic_by_model:
            console.print("[dim]No mature predictions yet — weights unchanged[/dim]")
            return

        with open(_WEIGHTS_PATH) as f:
            data = json.load(f)
        current_weights = data.get("weights", {m: 1.0 for m in ic_by_model})
        new_weights = _hedge_update(current_weights, ic_by_model, eta=data.get("eta", 0.5))
        ic_entry = {"week_end": str(today.date()), "horizon": used_horizon, **ic_by_model}
        _save_model_weights(new_weights, ic_entry=ic_entry)

        ic_str = "  ".join(f"{m}: {ic:+.4f}" for m, ic in sorted(ic_by_model.items()))
        w_str  = "  ".join(f"{m}: {w:.3f}" for m, w in sorted(new_weights.items()))
        console.print(f"[bold cyan]Weight update ({used_horizon}):[/bold cyan]  IC → {ic_str}")
        console.print(f"[bold cyan]New weights:[/bold cyan]  {w_str}")

    except Exception as exc:
        logger.warning("Weight evaluation failed (%s) — weights unchanged", exc)


def _run_calibration(model_ids: list, active_model_ids: list | None = None) -> None:
    """Run calibration pipeline: IC timeseries, alerts, residual correction, and
    cross-model calibration model.

    Also writes per-model live_ic_{model_id}.json for the dashboard.
    Only runs when invoked with --calibrate (intended for the daily workflow).
    """
    if not _LEDGER_PATH.exists():
        console.print("[dim]--calibrate: no ledger found yet — skipping[/dim]")
        return

    if active_model_ids is None:
        active_model_ids = model_ids

    try:
        import joblib
    except ImportError:
        logger.warning("joblib not available — calibration skipped")
        return

    try:
        ledger = pd.read_parquet(_LEDGER_PATH)
        today  = pd.Timestamp(date.today())
        _CALIB_DIR.mkdir(parents=True, exist_ok=True)

        # Load calibration_daily for rich feature context (join on ticker + scored_date)
        calib_ctx: pd.DataFrame | None = None
        if _CALIB_DAILY_PATH.exists():
            try:
                calib_ctx = pd.read_parquet(_CALIB_DAILY_PATH)
                calib_ctx = calib_ctx.rename(columns={"date": "scored_date"})
                console.print(
                    f"[dim]calibration_daily loaded: {len(calib_ctx):,} rows "
                    f"({calib_ctx['scored_date'].nunique()} days)[/dim]"
                )
            except Exception as exc:
                logger.warning("Could not load calibration_daily (%s) — using basic features", exc)

        # --- Part 1: IC timeseries + consolidated model_health.json ---
        ic_ts = _update_ic_timeseries(ledger, today, model_ids)
        _check_and_save_alerts(ic_ts, model_ids)

        # --- Part 2: Residual correction models → single correction_models.joblib ---
        correction_dict: dict = {}
        if _CORRECTION_PATH.exists():
            try:
                correction_dict = joblib.load(_CORRECTION_PATH)
            except Exception:
                correction_dict = {}

        for model_id in model_ids:
            raw_col = f"raw_{model_id}"
            if raw_col not in ledger.columns:
                console.print(f"[dim]--calibrate: {model_id} — no score column in ledger yet[/dim]")
                continue

            for _, _, suffix in _HORIZONS:
                cm = _fit_residual_correction(ledger, model_id, suffix, calib_ctx)
                if cm is not None:
                    key = f"{model_id}_{suffix}"
                    correction_dict[key] = cm
                    console.print(f"[dim]Correction model updated: {key}[/dim]")

        if correction_dict:
            joblib.dump(correction_dict, _CORRECTION_PATH)
            console.print(f"[cyan]Correction models saved:[/cyan] {len(correction_dict)} models → {_CORRECTION_PATH.name}")

        # --- Part 3: Cross-model calibration model → single calibration_models.joblib ---
        calibration_dict: dict = {}
        if _CALIBRATION_PATH.exists():
            try:
                calibration_dict = joblib.load(_CALIBRATION_PATH)
            except Exception:
                calibration_dict = {}

        for _, _, suffix in _HORIZONS:
            cal_model = _fit_calibration_model(ledger, suffix, active_model_ids, calib_ctx)
            if cal_model is not None:
                calibration_dict[suffix] = cal_model
                n_resolved = int(
                    ledger.get(f"matured_{suffix}", pd.Series(False, index=ledger.index))
                    .fillna(False).astype(bool).sum()
                )
                console.print(
                    f"[cyan]Multi-model calibrator ({suffix}):[/cyan] updated  "
                    f"(n={n_resolved})"
                )
            else:
                matured_col = f"matured_{suffix}"
                n_resolved = int(
                    ledger.get(matured_col, pd.Series(False, index=ledger.index))
                    .fillna(False).astype(bool).sum()
                ) if matured_col in ledger.columns else 0
                console.print(
                    f"[dim]Multi-model calibrator ({suffix}): cold-start "
                    f"({n_resolved} / 500 resolved predictions)[/dim]"
                )

        if calibration_dict:
            joblib.dump(calibration_dict, _CALIBRATION_PATH)
            console.print(f"[cyan]Calibration models saved:[/cyan] {len(calibration_dict)} horizons → {_CALIBRATION_PATH.name}")

    except Exception as exc:
        logger.warning("--calibrate step failed (%s) — continuing", exc)


# ---------------------------------------------------------------------------
# Daily IC timeseries + model alerts
# ---------------------------------------------------------------------------

def _update_ic_timeseries(
    ledger: pd.DataFrame,
    today: pd.Timestamp,
    model_ids: list[str],
) -> pd.DataFrame:
    """Compute cross-sectional Spearman IC for each (model, horizon) pair and
    append new rows to live_ic_timeseries.parquet.

    Only computes IC for cohorts whose realized returns have been filled in
    (matured == True).  Returns the updated timeseries DataFrame.
    """
    try:
        existing = pd.read_parquet(_IC_TIMESERIES_PATH) if _IC_TIMESERIES_PATH.exists() else pd.DataFrame()
    except Exception:
        existing = pd.DataFrame()

    new_rows: list[dict] = []

    for _, _, suffix in _HORIZONS:
        ret_col     = f"realized_return_{suffix}"
        matured_col = f"matured_{suffix}"
        if ret_col not in ledger.columns or matured_col not in ledger.columns:
            continue

        mature = ledger[ledger[matured_col].fillna(False).astype(bool)].copy()
        if mature.empty:
            continue

        for model_id in model_ids:
            raw_col = f"raw_{model_id}"
            if raw_col not in mature.columns:
                continue

            for scored_date, grp in mature.groupby("scored_date"):
                # Skip if we already have this row
                if not existing.empty and len(existing[
                    (existing["date"] == today) &
                    (existing["prediction_date"] == scored_date) &
                    (existing["model_id"] == model_id) &
                    (existing["horizon"] == suffix)
                ]) > 0:
                    continue

                valid = grp[[raw_col, ret_col]].dropna()
                if len(valid) < 30:
                    continue
                try:
                    ic, _ = spearmanr(valid[raw_col], valid[ret_col])
                except Exception:
                    continue
                if np.isnan(ic):
                    continue

                hit_rate = float(
                    ((valid[raw_col] > valid[raw_col].median()) ==
                     (valid[ret_col] > 0)).mean()
                )
                new_rows.append({
                    "date":            today,
                    "prediction_date": pd.Timestamp(scored_date),
                    "model_id":        model_id,
                    "horizon":         suffix,
                    "ic":              round(float(ic), 4),
                    "hit_rate":        round(hit_rate, 4),
                    "n_tickers":       len(valid),
                })

    if not new_rows:
        return existing

    updated = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    _IC_TIMESERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    updated.to_parquet(_IC_TIMESERIES_PATH, index=False)
    console.print(f"[dim]IC timeseries updated: {len(new_rows)} new rows → {_IC_TIMESERIES_PATH}[/dim]")
    return updated


def _check_and_save_alerts(ic_ts: pd.DataFrame, active_model_ids: list[str]) -> None:
    """Check rolling IC thresholds per model and write model_health.json.

    model_health.json consolidates alerts + per-model IC summaries in one place,
    replacing the old per-model live_ic_*.json files and model_alerts.json.
    """
    alerts: list[dict] = []
    model_health: dict = {}
    today_str = str(date.today())

    for model_id in active_model_ids:
        mdf = ic_ts[ic_ts["model_id"] == model_id] if not ic_ts.empty else pd.DataFrame()
        health_entry: dict = {"horizons": {}}
        status = "warming_up"

        for _, _, suffix in _HORIZONS:
            h_df = mdf[mdf["horizon"] == suffix]["ic"] if not mdf.empty else pd.Series(dtype=float)
            if len(h_df) < 3:
                continue
            ic_vals   = h_df.tolist()
            rolling   = float(h_df.tail(10).mean())
            ic_mean   = float(h_df.mean())
            ic_std    = float(h_df.std()) if len(h_df) > 1 else 0.0
            health_entry["horizons"][suffix] = {
                "ic_mean":    round(ic_mean, 4),
                "ic_std":     round(ic_std, 4),
                "rolling_10": round(rolling, 4),
                "n":          len(h_df),
                "status": (
                    "healthy"   if rolling >= 0.03 else
                    "degrading" if rolling >= 0.02 else
                    "retrain"
                ),
            }

            if rolling < 0.0:
                alerts.append({
                    "level": "CRITICAL", "model_id": model_id,
                    "message": f"{suffix} rolling IC = {rolling:.4f} (negative — model is harmful)",
                    "date": today_str,
                })
                status = "critical"
            elif suffix == "63d" and rolling < 0.02 and status != "critical":
                alerts.append({
                    "level": "WARNING", "model_id": model_id,
                    "message": f"63d rolling IC = {rolling:.4f} (below 0.02 threshold)",
                    "date": today_str,
                })
                if status == "warming_up":
                    status = "warning"
            elif status == "warming_up" and len(h_df) >= 5:
                status = "healthy"

        health_entry["status"] = status
        model_health[model_id] = health_entry

    payload = {
        "last_updated": today_str,
        "alerts":       alerts,
        "models":       model_health,
    }
    _MODEL_HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_MODEL_HEALTH_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    if alerts:
        for a in alerts:
            lvl_color = "red" if a["level"] == "CRITICAL" else "yellow"
            console.print(f"[{lvl_color}]{a['level']}: {a['message']}[/{lvl_color}]")
    else:
        console.print(f"[dim]Model alerts: all clear[/dim]")


# ---------------------------------------------------------------------------
# Residual correction (Part 2)
# ---------------------------------------------------------------------------

# Basic context features always available from the ledger
_CORRECTION_FEATURES_BASIC = [
    "raw_score", "sector_code", "market_return_21d", "vix_at_prediction", "momentum_bucket",
]
# Richer features available when calibration_daily.parquet is joined in
_CORRECTION_FEATURES_RICH = _CORRECTION_FEATURES_BASIC + [
    "rsi", "momentum_12_1", "price_to_sma200", "volatility_20d",
    "bollinger_pct_b", "vix_level",
]
_CORRECTION_FEATURES = _CORRECTION_FEATURES_BASIC  # default; overridden at fit time


def _fit_residual_correction(
    ledger: pd.DataFrame,
    model_id: str,
    horizon_suffix: str,
    calib_ctx: "pd.DataFrame | None" = None,
    lookback_pairs: int = 3000,
):
    """Fit a Ridge correction model on recent (prediction, actual) residuals.

    When ``calib_ctx`` is provided (calibration_daily.parquet), joins richer
    features (RSI, momentum, SMA ratio, etc.) for a more powerful correction.
    Falls back to basic ledger context columns if not available.

    Returns the fitted Ridge model, or None if not enough resolved data.
    """
    try:
        from sklearn.linear_model import Ridge
    except ImportError:
        return None

    raw_col     = f"raw_{model_id}"
    ret_col     = f"realized_return_{horizon_suffix}"
    matured_col = f"matured_{horizon_suffix}"

    if raw_col not in ledger.columns or ret_col not in ledger.columns:
        return None
    if matured_col not in ledger.columns:
        return None

    resolved = ledger[
        ledger[matured_col].fillna(False).astype(bool) &
        ledger[raw_col].notna() &
        ledger[ret_col].notna()
    ].sort_values("scored_date", ascending=False).head(lookback_pairs).copy()

    if len(resolved) < 200:
        return None

    # Join rich features from calibration_daily when available
    if calib_ctx is not None:
        resolved = resolved.merge(
            calib_ctx[["ticker", "scored_date"] + [
                c for c in _CORRECTION_FEATURES_RICH
                if c not in _CORRECTION_FEATURES_BASIC and c in calib_ctx.columns
            ]],
            on=["ticker", "scored_date"],
            how="left",
        )
        features = [f for f in _CORRECTION_FEATURES_RICH if f in resolved.columns or f in _CORRECTION_FEATURES_BASIC]
    else:
        features = _CORRECTION_FEATURES_BASIC

    resolved["raw_score"] = resolved[raw_col].values
    for col in features:
        if col not in resolved.columns:
            resolved[col] = 0.0

    # Downweight backfill rows (0.3) vs live rows (1.0)
    sample_weight = np.where(
        resolved.get("is_backfill", pd.Series(False, index=resolved.index)).fillna(False),
        0.3, 1.0
    )

    X = resolved[features].fillna(0).values.astype(float)
    y = (resolved[ret_col] - resolved[raw_col]).values  # residual

    try:
        mdl = Ridge(alpha=10.0)
        mdl.fit(X, y, sample_weight=sample_weight)
        return mdl
    except Exception as exc:
        logger.warning("Residual correction fit failed for %s %s: %s", model_id, horizon_suffix, exc)
        return None


def _apply_correction(
    raw: np.ndarray,
    df_score: pd.DataFrame,
    correction_model,
    ctx: dict,
) -> np.ndarray:
    """Apply residual correction to today's raw scores array.

    Builds the full rich feature set (matching what the model was trained on):
    basic ledger context + RSI, momentum, SMA200, volatility, bollinger, VIX.
    Falls back to 0 for any feature not available in df_score.
    """
    if correction_model is None:
        return raw
    try:
        feat_df = pd.DataFrame({"raw_score": raw})
        # Basic context features
        feat_df["sector_code"] = (
            df_score["Sector"].map(_SECTOR_CODES).fillna(0).values
            if "Sector" in df_score.columns else 0
        )
        feat_df["market_return_21d"] = ctx.get("market_return_21d", 0.0)
        feat_df["vix_at_prediction"] = ctx.get("vix_at_prediction", 20.0)
        if "Momentum12_1" in df_score.columns:
            mom = pd.to_numeric(df_score["Momentum12_1"], errors="coerce")
            feat_df["momentum_bucket"] = pd.qcut(mom.rank(method="first"), q=5, labels=False).fillna(2).values
        else:
            feat_df["momentum_bucket"] = 2

        # Rich features from today's spread CSV (PascalCase → values)
        def _spread_col(pascal, default):
            return pd.to_numeric(df_score[pascal], errors="coerce").fillna(default).values \
                if pascal in df_score.columns else default

        feat_df["rsi"]             = _spread_col("RSI", 50.0)
        feat_df["momentum_12_1"]   = _spread_col("Momentum12_1", 0.0)
        feat_df["price_to_sma200"] = _spread_col("PriceToSMA200Pct", 1.0)
        feat_df["volatility_20d"]  = _spread_col("AnnualizedVol", 0.02)
        feat_df["bollinger_pct_b"] = _spread_col("BollingerPctB", 0.5)

        # Use whichever features the stored model was trained on
        n_expected = correction_model.n_features_in_
        available_features = _CORRECTION_FEATURES_RICH
        features_to_use = available_features[:n_expected]

        # Ensure all required columns exist
        for col in features_to_use:
            if col not in feat_df.columns:
                feat_df[col] = 0.0

        X = feat_df[features_to_use].fillna(0).values.astype(float)
        corrections = correction_model.predict(X)
        corrections = np.clip(corrections, -0.1, 0.1)
        return raw + corrections
    except Exception as exc:
        logger.warning("Correction apply failed: %s", exc)
        return raw


# ---------------------------------------------------------------------------
# Learned calibration model (Part 3) — replaces hedge + isotonic
# ---------------------------------------------------------------------------

def _build_calib_features(row: pd.Series, model_ids: list[str]) -> dict:
    """Build the feature vector for the cross-model calibration logistic regression."""
    scores = {mid: row.get(f"raw_{mid}", np.nan) for mid in model_ids}
    available = {mid: s for mid, s in scores.items() if not np.isnan(s)}

    feat: dict = {}
    for mid in model_ids:
        s = scores[mid]
        feat[f"score_{mid}"] = s if not np.isnan(s) else 0.5
        feat[f"has_{mid}"]   = float(not np.isnan(s))

    # Pairwise agreement / spread between first two models (if available)
    mids = list(available.keys())
    if len(mids) >= 2:
        a, b = mids[0], mids[1]
        feat["agreement_top2"]  = available[a] * available[b]
        feat["spread_top2"]     = abs(available[a] - available[b])
    else:
        feat["agreement_top2"]  = list(available.values())[0] * 0.5 if available else 0.25
        feat["spread_top2"]     = 0.0

    def _fval(row, key, default):
        """Safe float extraction — treats None and NaN as missing."""
        v = row.get(key)
        if v is None:
            return default
        try:
            f = float(v)
            return default if f != f else f   # NaN check: NaN != NaN
        except (TypeError, ValueError):
            return default

    feat["market_return_21d"] = _fval(row, "market_return_21d", 0.0)
    feat["vix_at_prediction"] = _fval(row, "vix_at_prediction", 20.0)
    feat["sector_code"]       = _fval(row, "sector_code", 0.0)
    return feat


def _fit_calibration_model(
    ledger: pd.DataFrame,
    horizon_suffix: str,
    active_model_ids: list[str],
    calib_ctx: "pd.DataFrame | None" = None,
    min_samples: int = 500,
    rolling_window: int = 5000,
):
    """Fit a LogisticRegression calibration model on resolved predictions.

    When ``calib_ctx`` is provided (calibration_daily.parquet), joins richer
    stock-level features (RSI, momentum, SMA ratio, etc.) for better regime-
    conditional calibration.  Backfill rows are downweighted (0.3) so the model
    activates immediately from historical data but shifts toward live signal.

    Returns None during cold-start (< min_samples resolved rows).
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None

    ret_col     = f"realized_return_{horizon_suffix}"
    matured_col = f"matured_{horizon_suffix}"

    if ret_col not in ledger.columns or matured_col not in ledger.columns:
        return None

    resolved = ledger[
        ledger[matured_col].fillna(False).astype(bool) &
        ledger[ret_col].notna()
    ].sort_values("scored_date", ascending=False).head(rolling_window).copy()

    if len(resolved) < min_samples:
        return None

    # Require at least one model score column
    raw_cols = [f"raw_{mid}" for mid in active_model_ids if f"raw_{mid}" in resolved.columns]
    if not raw_cols:
        return None

    resolved = resolved.dropna(subset=raw_cols, how="all")
    if len(resolved) < min_samples:
        return None

    # Join rich features from calibration_daily when available
    _RICH_CALIB_COLS = ["rsi", "momentum_12_1", "price_to_sma200", "volatility_20d",
                        "bollinger_pct_b", "vix_level"]
    if calib_ctx is not None:
        ctx_cols = [c for c in _RICH_CALIB_COLS if c in calib_ctx.columns]
        if ctx_cols:
            resolved = resolved.merge(
                calib_ctx[["ticker", "scored_date"] + ctx_cols],
                on=["ticker", "scored_date"],
                how="left",
            )

    feature_rows = resolved.apply(lambda r: _build_calib_features(r, active_model_ids), axis=1)
    X = pd.DataFrame(feature_rows.tolist()).fillna(0).values.astype(float)
    y = (resolved[ret_col] > 0).astype(int).values

    # Downweight backfill rows (selection-bias risk) while still using them for cold-start
    sample_weight = np.where(
        resolved.get("is_backfill", pd.Series(False, index=resolved.index)).fillna(False),
        0.3, 1.0
    )

    try:
        mdl = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
        mdl.fit(X, y, sample_weight=sample_weight)
        return mdl
    except Exception as exc:
        logger.warning("Calibration model fit failed for horizon %s: %s", horizon_suffix, exc)
        return None


def _apply_calibration_model(
    df_score: pd.DataFrame,
    raw_scores: dict,
    cal_model,
    active_model_ids: list[str],
    ctx: dict,
) -> np.ndarray | None:
    """Apply learned calibration model to today's scored stocks.

    Returns probability array P(positive return) or None if model unavailable.
    """
    if cal_model is None:
        return None
    try:
        feat_rows = []
        for i in range(len(df_score)):
            row = {f"raw_{mid}": raw_scores[mid][i] if mid in raw_scores else np.nan
                   for mid in active_model_ids}
            row["market_return_21d"] = ctx.get("market_return_21d", 0.0)
            row["vix_at_prediction"] = ctx.get("vix_at_prediction", 20.0)
            if "Sector" in df_score.columns:
                row["sector_code"] = _SECTOR_CODES.get(df_score["Sector"].iloc[i], 0)
            else:
                row["sector_code"] = 0
            feat_rows.append(_build_calib_features(pd.Series(row), active_model_ids))

        X = pd.DataFrame(feat_rows).fillna(0).values.astype(float)
        return cal_model.predict_proba(X)[:, 1]
    except Exception as exc:
        logger.warning("Calibration apply failed: %s", exc)
        return None


def _load_config(models_dir: Path) -> dict:
    config_path = models_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    # Auto-discover: treat all .pkl files as models, no champion
    pkls = sorted(models_dir.glob("*.pkl"))
    return {
        "champion": pkls[0].stem if pkls else None,
        "models": [{"id": p.stem, "type": "lgbm", "status": "champion" if i == 0 else "challenger"}
                   for i, p in enumerate(pkls)],
    }


def _generate_all_forecast_paths(
    active_models: list,
    df_score: pd.DataFrame,
    data_path: Path,
    n_paths: int = 6,
) -> None:
    """Generate MC-dropout forecast paths for every active LSTM model.

    Writes data/latest/forecast_paths_<model_id>.json for each LSTM.
    Skips non-LSTM models (LightGBM has no sequential forecast).
    Mirrors output to the web public dir alongside ticker_index.json.
    """
    lstm_models = [m for m in active_models if m.get("type") == "lstm"]
    if not lstm_models:
        return

    from morningalpha.ml.inference import generate_forecast_paths

    _REPO_ROOT   = Path(__file__).parents[2]
    _WEB_PUBLIC  = _REPO_ROOT / "morningalpha" / "web" / "public" / "data" / "latest"

    for m in lstm_models:
        model_path = Path(m["checkpoint"]) if "checkpoint" in m else (
            Path(__file__).parents[2] / "models" / f"{m['id']}.pt"
        )
        if not model_path.exists():
            continue
        try:
            console.print(f"[dim]Generating forecast paths for {m['id']} ({len(df_score)} tickers)…[/dim]")
            result = generate_forecast_paths(df_score, model_path, n_paths=n_paths)
            n_with_paths = len(result.get("paths", {}))

            out_path = data_path / f"forecast_paths_{m['id']}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, separators=(",", ":"))

            console.print(f"[dim]✓ Forecast paths: {n_with_paths} tickers → {out_path.name}[/dim]")

            # Mirror to web public dir
            if _WEB_PUBLIC.exists():
                import shutil
                shutil.copy(out_path, _WEB_PUBLIC / out_path.name)
        except Exception as exc:
            logger.warning("Forecast path generation failed for %s: %s", m["id"], exc)


def _write_ticker_index(
    scored_df: pd.DataFrame,
    active_models: list,
    raw_scores: dict,
    output_dir: Path,
) -> None:
    """Write a lightweight JSON for frontend stock search / Forecast page autocomplete.

    Includes MLScore (consensus), per-model scores, and calibrated signals for
    each scored ticker.  Stored at: {output_dir}/ticker_index.json
    """
    # Friendly field names for per-model scores in the JSON.
    MODEL_FIELD_MAP = {
        # current champions / candidates
        "lgbm_breakout_v10":          "mlScore_breakout",
        "lgbm_composite_v10":         "mlScore_composite",
        "lgbm_breakout_medium_v1":    "mlScore_breakout_medium",
        "lgbm_composite_medium_v1":   "mlScore_composite_medium",
        "lstm_clip_v1":               "mlScore_lstm",
        "lstm_clip_v3":               "mlScore_lstm",
        # legacy (retired but kept for back-compat if old model files present)
        "lgbm_breakout_v7":           "mlScore_breakout",
        "lgbm_composite_v7":          "mlScore_composite",
        "lgbm_breakout_v5":           "mlScore_breakout",
        "lgbm_composite_v6":          "mlScore_composite",
        "st_sector_relative_v1":      "mlScore_st",
    }

    def _f(val, default=None):
        """Convert a value to a rounded float, returning default for NaN/None/inf."""
        if val is None:
            return default
        try:
            f = float(val)
            return default if (f != f or f == float("inf") or f == float("-inf")) else round(f, 1)
        except (TypeError, ValueError):
            return default

    def _investment_score(row) -> float | None:
        """Mirror of scoring.ts calculateInvestmentScore — Return/Quality/Sharpe/Consistency/Drawdown."""
        return_pct = _f(row.get("Return_3M_%"))
        quality    = _f(row.get("QualityScore"))
        sharpe     = _f(row.get("SharpeRatio"))
        consistency = _f(row.get("ConsistencyScore"))
        drawdown   = _f(row.get("MaxDrawdown"))

        score = 0.0
        factors = 0.0
        if return_pct is not None:
            score += min(30.0, (return_pct / 200.0) * 30.0); factors += 30.0
        if quality is not None:
            score += (quality / 100.0) * 25.0; factors += 25.0
        if sharpe is not None:
            score += max(0.0, min(20.0, ((sharpe + 1.0) / 4.0) * 20.0)); factors += 20.0
        if consistency is not None:
            score += (consistency / 100.0) * 15.0; factors += 15.0
        if drawdown is not None:
            score += max(0.0, min(10.0, ((drawdown + 50.0) / 50.0) * 10.0)); factors += 10.0
        if factors == 0.0:
            return None
        return round((score / factors) * 1000.0) / 10.0

    records = []
    for _, row in scored_df.iterrows():
        sector = row.get("Sector", row.get("sector", None))
        entry: dict = {
            "ticker":  row.get("Ticker", ""),
            "name":    str(row.get("Name", row.get("name", ""))),
            "sector":  None if (sector is None or (isinstance(sector, float) and sector != sector)) else sector,
            "mlScore": _f(row.get("MLScore"), default=0.0),
            "investmentScore": _investment_score(row),
        }
        for m in active_models:
            if m["id"] not in raw_scores:
                continue
            col = f"MLScore_{m['id']}"
            field = MODEL_FIELD_MAP.get(m["id"], f"mlScore_{m['id']}")
            entry[field] = _f(row.get(col))

        # Calibrated signal fields (absent until calibrators are trained)
        calib_prob = row.get("CalibratedProb")
        if calib_prob is not None and not (isinstance(calib_prob, float) and calib_prob != calib_prob):
            entry["calibratedProb"]   = round(float(calib_prob), 3)
            entry["calibratedSignal"] = row.get("CalibratedSignal", None)
        # Per-model calibrated probs (useful for multi-model display on Forecast page)
        for m in active_models:
            col = f"CalibProb_{m['id']}"
            if col in row.index and not pd.isna(row[col]):
                field = MODEL_FIELD_MAP.get(m["id"], m["id"]).replace("mlScore_", "calibProb_")
                entry[field] = round(float(row[col]), 3)

        records.append(entry)

    out_path = output_dir / "ticker_index.json"
    with open(out_path, "w") as f:
        json.dump(records, f)
    console.print(f"[dim]✓ Wrote ticker_index.json ({len(records)} tickers)[/dim]")

    # Mirror to the web public directory so the dev server and GitHub Pages build
    # pick up the latest scores without a manual copy step.
    web_public = Path(__file__).parents[2] / "morningalpha" / "web" / "public" / "data" / "latest"
    if web_public.exists():
        import shutil
        shutil.copy2(out_path, web_public / "ticker_index.json")
        console.print(f"[dim]✓ Mirrored ticker_index.json → {web_public}[/dim]")


@click.command("score")
@click.option("--data-dir", default=DEFAULT_DATA_DIR, show_default=True,
              help="Directory containing period CSVs from `alpha spread`.")
@click.option("--models-dir", default=DEFAULT_MODELS_DIR, show_default=True,
              help="Directory containing model .pkl files and config.json.")
@click.option("--score-only", "score_only", is_flag=True, default=False,
              help="Score only — skip ledger, weight updates, and calibration. "
                   "Use for local development and model testing.")
@click.option("--calibrate", "run_calibrate", is_flag=True, default=False,
              help="After scoring, fit isotonic calibrator and write live_ic.json. "
                   "Intended for the weekly workflow.")
def score(data_dir, models_dir, score_only, run_calibrate):
    """Score all stocks in data-dir CSVs with registered ML models.

    Adds MLScore (champion/consensus 0-100) and MLScore_{model_id} per-model
    columns to each period CSV. Run after `alpha spread`.

    Examples:
        alpha ml score
        alpha ml score --data-dir data/latest --models-dir models
    """
    from morningalpha.ml.inference import get_raw_scores

    data_path = Path(data_dir)
    models_path = Path(models_dir)

    if not models_path.exists():
        console.print(f"[red]Models directory not found: {models_path}[/red]")
        raise SystemExit(1)

    config = _load_config(models_path)
    champion_id = config.get("champion")
    model_entries = config.get("models", [])

    # Only score models that are active (champion/candidate) and have a checkpoint file.
    # LightGBM models use .pkl; Set Transformer models use .pt (path from config or default).
    def _checkpoint_exists(m: dict) -> bool:
        if "checkpoint" in m:
            return Path(m["checkpoint"]).exists()
        if m.get("type") in ("set_transformer", "lstm"):
            return (models_path / f"{m['id']}.pt").exists()
        return (models_path / f"{m['id']}.pkl").exists()

    active_models = [
        m for m in model_entries
        if m.get("status", "candidate") not in ("retired",)
        and _checkpoint_exists(m)
    ]

    if not active_models:
        console.print("[yellow]No model checkpoint files found — nothing to score.[/yellow]")
        return

    console.print(f"\n[bold]ML Scoring[/bold] — {len(active_models)} model(s): "
                  + ", ".join(
                      f"[cyan]{m['id']}[/cyan]" + (" [green](champion)[/green]" if m['id'] == champion_id else "")
                      for m in active_models
                  ))

    # -----------------------------------------------------------------------
    # Step 1: Score the 3M CSV — canonical source matching training distribution
    # -----------------------------------------------------------------------
    source_path = data_path / SCORE_SOURCE
    if not source_path.exists():
        console.print(f"[red]Canonical score source not found: {source_path}[/red]")
        raise SystemExit(1)

    df3m = pd.read_csv(source_path, index_col=0)

    # Update the rolling 63-day feature window used for LSTM inference + rich calibration
    if not score_only:
        try:
            from morningalpha.ml.backfill import update_calibration_daily
            n_rows = update_calibration_daily(df3m, pd.Timestamp(date.today()), _CALIB_DAILY_PATH)
            console.print(f"[dim]calibration_daily updated: {n_rows:,} rows[/dim]")
        except Exception as exc:
            logger.warning("calibration_daily update failed (%s) — continuing", exc)

    # Backfill MarketCap from fundamentals.csv if the spread CSV has it empty
    # (happens when spread CSVs predate the fundamentals-merge fix in access.py).
    fund_path = data_path / "fundamentals.csv"
    mc_col = df3m.get("MarketCap", pd.Series(dtype=float))
    if fund_path.exists() and (mc_col.isna().all() or "MarketCap" not in df3m.columns):
        fund_df = pd.read_csv(fund_path, usecols=["Ticker", "MarketCap"])
        df3m = df3m.drop(columns=["MarketCap"], errors="ignore")
        df3m = df3m.merge(fund_df, on="Ticker", how="left")
        console.print("[dim]MarketCap backfilled from fundamentals.csv[/dim]")

    # Filter to large-enough stocks before scoring — micro/small caps produce
    # noisy signals (illiquid, mean-reversion artifacts) and are untradeable at scale.
    MIN_MARKET_CAP = 100_000_000  # $100M — matches training dataset floor
    mc_numeric = df3m["MarketCap"].apply(pd.to_numeric, errors="coerce") if "MarketCap" in df3m.columns else pd.Series(dtype=float)
    if mc_numeric.notna().any():
        eligible = mc_numeric >= MIN_MARKET_CAP
        n_filtered = int((~eligible & mc_numeric.notna()).sum())
        df_score = df3m[eligible | mc_numeric.isna()].copy()
        if n_filtered:
            console.print(f"[dim]Filtered out {n_filtered} stocks below ${MIN_MARKET_CAP/1e9:.0f}B market cap[/dim]")
    else:
        df_score = df3m.copy()

    # No pre-scoring return or quality filters — the model was trained on the full universe
    # and learned to rank good from bad. Pre-filtering by return defeats the purpose.

    # SMA200 flag — model already learned price_to_sma200 as a feature and prices it in.
    # Hard-gating would remove potential recovery/mean-reversion plays and mismatches
    # the training distribution. Instead, surface it as a warning flag in the output.
    if "PriceToSMA200Pct" in df_score.columns:
        sma200_pct = pd.to_numeric(df_score["PriceToSMA200Pct"], errors="coerce")
        df_score["AboveSMA200"] = (sma200_pct > 0).fillna(True)
        n_below = int((~df_score["AboveSMA200"]).sum())
        if n_below:
            console.print(f"[dim]SMA200 flag: {n_below} stocks below 200-day MA (scored, not removed)[/dim]")

    raw_scores: dict[str, np.ndarray] = {}

    for m in active_models:
        model_type = m.get("type", "lgbm")
        # Resolve checkpoint path: explicit in config, or default by type
        if "checkpoint" in m:
            model_path = Path(m["checkpoint"])
        elif model_type in ("set_transformer", "lstm"):
            model_path = models_path / f"{m['id']}.pt"
        else:
            model_path = models_path / f"{m['id']}.pkl"

        if not model_path.exists():
            console.print(f"[yellow]Checkpoint not found for {m['id']}: {model_path} — skipping[/yellow]")
            continue

        try:
            if model_type == "set_transformer":
                from morningalpha.ml.inference import get_st_raw_scores
                raw = get_st_raw_scores(df_score, model_path)
            elif model_type == "lstm":
                from morningalpha.ml.inference import get_lstm_raw_scores
                raw = get_lstm_raw_scores(df_score, model_path)
            else:
                raw = get_raw_scores(df_score, model_path)
            raw_scores[m["id"]] = raw
            pct = pd.Series(raw, index=df_score.index).rank(pct=True).mul(100).round(1)
            df_score[f"MLScore_{m['id']}"] = pct.values
        except Exception as exc:
            logger.warning("Scoring with %s failed: %s", m["id"], exc)

    if not raw_scores:
        console.print("[yellow]All models failed to score — nothing written.[/yellow]")
        return

    if score_only:
        console.print("[dim]--score-only: skipping ledger, weight updates, and calibration[/dim]")
    else:
        # Evaluate mature predictions (≥63 trading days old) → Hedge weight update
        _evaluate_and_update_weights(df_score, raw_scores)
        # Append this run's predictions to the ledger for future IC evaluation
        _append_predictions_ledger(df_score, raw_scores)

    # Consensus: dynamic IC-weighted blend.
    # In --score-only mode uses current weights from file without updating them.
    model_weights = _load_model_weights(list(raw_scores.keys()))
    if len(raw_scores) > 1:
        weights = np.array([model_weights[mid] for mid in raw_scores])
        stacked = np.column_stack(list(raw_scores.values()))
        consensus = (stacked * weights).sum(axis=1)
        df_score["MLScore"] = pd.Series(consensus, index=df_score.index).rank(pct=True).mul(100).round(1).values
        w_str = "  ".join(f"{m}: {model_weights[m]:.3f}" for m in raw_scores)
        console.print(f"[dim]Ensemble weights: {w_str}[/dim]")
    else:
        df_score["MLScore"] = df_score[f"MLScore_{list(raw_scores.keys())[0]}"]

    # Sector diversity cap: per-sector limits on final MLScore ranking.
    # High-opportunity sectors (Technology, Healthcare) get more slots.
    # Individual MLScore_* columns are preserved unchanged for analysis.
    DEFAULT_SECTOR_CAP = 5
    SECTOR_CAPS: dict[str, int] = {
        "Technology": 15,
        "Healthcare": 10,
        "Financial Services": 10,
        "Consumer Cyclical": 7,
        "Communication Services": 7,
        "Industrials": 3,   # shipping/transport stocks often have clean technicals but slow growth
        "Energy": 3,
        "Utilities": 2,
    }
    if "Sector" in df_score.columns:
        sectors = df_score["Sector"].fillna("Unknown")
        sector_rank = df_score.groupby(sectors)["MLScore"].rank(ascending=False, method="first")
        cap_per_row = sectors.map(lambda s: SECTOR_CAPS.get(s, DEFAULT_SECTOR_CAP))
        within_cap = sector_rank <= cap_per_row
        diversity_bonus = within_cap.astype(float) * 1000
        df_score["MLScore"] = (
            (df_score["MLScore"] + diversity_bonus)
            .rank(pct=True).mul(100).round(1)
        )
        n_deprioritized = int((~within_cap).sum())
        cap_summary = ", ".join(f"{s}:{c}" for s, c in sorted(SECTOR_CAPS.items()))
        console.print(
            f"[dim]Sector diversity cap (default {DEFAULT_SECTOR_CAP}, overrides: {cap_summary}): "
            f"{n_deprioritized} stocks deprioritized[/dim]"
        )

    # Score delta — compare to previous run's cached scores (falling = warning signal)
    _score_cache_path = Path("data/factors/mlscore_cache.parquet")
    try:
        if _score_cache_path.exists():
            prev = pd.read_parquet(_score_cache_path).set_index("Ticker")["MLScore"]
            current = df_score.set_index("Ticker")["MLScore"]
            delta = current.sub(prev, fill_value=float("nan")).reindex(current.index).round(1)
            df_score["MLScoreDelta"] = delta.values
        _score_cache_path.parent.mkdir(parents=True, exist_ok=True)
        df_score[["Ticker", "MLScore"]].to_parquet(_score_cache_path, index=False)
    except Exception as exc:
        logger.warning("MLScoreDelta computation failed (%s) — delta will be absent", exc)

    # -----------------------------------------------------------------------
    # Part 4: Apply residual corrections + learned calibration
    # -----------------------------------------------------------------------
    _SIGNAL_THRESHOLDS = [
        (0.70, "STRONG BUY"),
        (0.55, "BUY"),
        (0.45, "HOLD"),
        (0.30, "SELL"),
        (float("-inf"), "STRONG SELL"),
    ]

    def _prob_to_signal(prob: float) -> str:
        for threshold, label in _SIGNAL_THRESHOLDS:
            if prob >= threshold:
                return label
        return "STRONG SELL"

    try:
        import joblib as _joblib
    except ImportError:
        _joblib = None

    # Fetch today's market context once (used by correction + calibration)
    market_ctx = _fetch_market_context(pd.Timestamp(date.today()))

    # Load correction and calibration dicts once (avoids repeated file I/O per model)
    _correction_dict: dict = {}
    if _joblib is not None and _CORRECTION_PATH.exists():
        try:
            _correction_dict = _joblib.load(_CORRECTION_PATH)
        except Exception as exc:
            logger.warning("Could not load correction_models.joblib (%s)", exc)

    _calibration_dict: dict = {}
    if _joblib is not None and _CALIBRATION_PATH.exists():
        try:
            _calibration_dict = _joblib.load(_CALIBRATION_PATH)
        except Exception as exc:
            logger.warning("Could not load calibration_models.joblib (%s)", exc)

    # Step 4a: Apply per-model residual corrections (Part 2)
    # Try 63d correction first (most relevant to primary signal), fall back to shorter horizons.
    corrected_scores: dict[str, np.ndarray] = {}
    for model_id, raw in raw_scores.items():
        corrected = raw
        for _, _, suffix in reversed(_HORIZONS):   # 63d first
            key = f"{model_id}_{suffix}"
            if key in _correction_dict:
                corrected = _apply_correction(corrected, df_score, _correction_dict[key], market_ctx)
                break  # use the longest available horizon's correction model
        corrected_scores[model_id] = corrected

    # Step 4b: Try multi-model calibration model (Part 3)
    # Uses the primary scoring horizon (63d) to generate calibrated probabilities.
    active_ids = list(raw_scores.keys())
    consensus_prob: np.ndarray | None = None

    for _, _, suffix in reversed(_HORIZONS):   # 63d first
        if suffix not in _calibration_dict:
            continue
        try:
            cal_multi = _calibration_dict[suffix]
            # Build feature matrix from corrected scores
            tmp_scores = {mid: corrected_scores[mid] for mid in active_ids if mid in corrected_scores}
            consensus_prob = _apply_calibration_model(df_score, tmp_scores, cal_multi, active_ids, market_ctx)
            if consensus_prob is not None:
                console.print(f"[dim]Multi-model calibration active ({suffix})[/dim]")
            break
        except Exception as exc:
            logger.warning("Multi-model calibration load failed: %s", exc)

    calibrated_probs: dict[str, np.ndarray] = {}

    if consensus_prob is not None:
        df_score["CalibratedProb"]   = np.round(consensus_prob, 3)
        df_score["CalibratedSignal"] = [_prob_to_signal(float(p)) for p in consensus_prob]
        dist = {s: int((df_score["CalibratedSignal"] == s).sum())
                for s in ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]}
        dist_str = "  ".join(f"{s}={n}" for s, n in dist.items())
        console.print(f"[dim]Calibrated signals: {dist_str}[/dim]")
    else:
        console.print("[dim]No calibrators found yet — CalibratedSignal will be absent[/dim]")

    # -----------------------------------------------------------------------
    # Step 2: Build ticker → score lookup from the eligible scored stocks
    # -----------------------------------------------------------------------
    delta_col  = ["MLScoreDelta"] if "MLScoreDelta" in df_score.columns else []
    calib_cols = (
        ["CalibratedProb", "CalibratedSignal"]
        + [f"CalibProb_{mid}" for mid in calibrated_probs]
        if calibrated_probs else []
    )
    sma200_col = ["AboveSMA200"] if "AboveSMA200" in df_score.columns else []
    score_cols = ["MLScore"] + delta_col + sma200_col + [f"MLScore_{m['id']}" for m in active_models if m["id"] in raw_scores] + calib_cols
    scores_by_ticker = df_score.set_index("Ticker")[score_cols]

    top_ticker = scores_by_ticker["MLScore"].idxmax()

    # -----------------------------------------------------------------------
    # Step 3: Merge scores into all period CSVs by Ticker
    # -----------------------------------------------------------------------
    for filename in PERIOD_FILES:
        csv_path = data_path / filename
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path, index_col=0)

        # Drop any stale score/signal columns before merging fresh values
        stale = [c for c in df.columns
                 if c.startswith("MLScore") or c.startswith("CalibProb_")
                 or c in ("CalibratedProb", "CalibratedSignal", "AboveSMA200")]
        if stale:
            df = df.drop(columns=stale)

        df = df.merge(scores_by_ticker, on="Ticker", how="left")
        df.to_csv(csv_path, index_label="Rank")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    table = Table(title="ML Scoring Summary", show_header=True, header_style="bold cyan")
    table.add_column("Source", style="dim")
    table.add_column("Stocks scored", justify="right")
    table.add_column("Models", justify="right")
    table.add_column("Top ML Pick", style="bold green")
    table.add_column("Merged into", style="dim")
    table.add_row(
        SCORE_SOURCE,
        f"{len(df_score)} / {len(df3m)}",
        str(len(raw_scores)),
        str(top_ticker),
        ", ".join(f for f in PERIOD_FILES if (data_path / f).exists()),
    )
    console.print(table)

    # Write generation timestamp for the dashboard
    import datetime
    generated_path = data_path / "_generated.json"
    with open(generated_path, "w") as f:
        json.dump({"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}, f)

    # Write ticker_index.json for the Forecast/Portfolio frontend pages
    _write_ticker_index(df_score, active_models, raw_scores, data_path)

    if run_calibrate and not score_only:
        _run_calibration(list(raw_scores.keys()), active_model_ids=list(raw_scores.keys()))

    console.print("\n[bold green]✓ ML scoring complete[/bold green]")
