"""alpha ml backtest — Compute evaluation metrics and write dashboard JSON files.

Reads data/training/dataset.parquet (test split) + the trained model checkpoint,
runs inference, computes the full evaluation suite, and writes JSON to
morningalpha/web/public/data/backtest/{model_id}/.

Usage:
    alpha ml backtest
    alpha ml backtest --model lgbm_v4
    alpha ml backtest --out morningalpha/web/public/data/backtest/
"""
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rich_click as click
from rich.console import Console
from rich.table import Table
from scipy.stats import spearmanr

from morningalpha.ml.features import (
    TECHNICAL_FEATURE_COLUMNS,
    MARKET_CONTEXT_COLUMNS,
    FUNDAMENTAL_FEATURE_NAMES,
)

logger = logging.getLogger(__name__)
console = Console()

_REPO_MODELS = Path(__file__).parents[2] / "models"
_HOME_MODELS = Path.home() / ".morningalpha" / "models"
MODEL_DIR = _REPO_MODELS if _REPO_MODELS.exists() else _HOME_MODELS
DATASET_PATH = Path("data/training/dataset.parquet")
DEFAULT_OUT = Path("morningalpha/web/public/data/backtest")

# 10-day non-overlapping windows → ~25.2 periods per year
PERIODS_PER_YEAR = 252 / 10

# Round-trip transaction cost (10 bps)
TX_COST = 0.001

# Feature category sets for the importance chart
_DERIVED = {"return_vs_sector", "return_pct_x_regime"}
_TECHNICAL = set(TECHNICAL_FEATURE_COLUMNS) - _DERIVED
_MARKET = set(MARKET_CONTEXT_COLUMNS)
_FUNDAMENTAL = set(FUNDAMENTAL_FEATURE_NAMES)


def _feature_category(name: str) -> str:
    if name in _DERIVED:
        return "derived"
    if name in _TECHNICAL:
        return "technical"
    if name in _MARKET:
        return "market_context"
    if name in _FUNDAMENTAL:
        return "fundamental"
    return "other"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_model_and_config(model_id: str) -> Tuple[object, dict]:
    """Load LightGBMModel wrapper (pickled) and feature config for the given model_id."""
    model_path = MODEL_DIR / f"{model_id}.pkl"
    config_path = MODEL_DIR / "feature_config.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Feature config not found: {config_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(config_path) as f:
        config = json.load(f)

    return model, config


def _load_test_data(config: dict) -> pd.DataFrame:
    """Load test split from the training dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_parquet(DATASET_PATH)
    test = df[df["split"] == "test"].copy()
    console.print(
        f"Test split: [bold]{len(test):,}[/bold] rows  "
        f"[bold]{test['date'].nunique()}[/bold] dates  "
        f"[bold]{test['ticker'].nunique():,}[/bold] tickers"
    )
    return test


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_inference(booster, df: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    """Run model inference on the test set. Returns df with pred_score column."""
    # Prefer the model's own feature list — it reflects the exact columns used at
    # training time (e.g. value features dropped for momentum-universe models).
    try:
        feat_cols = list(booster.model.feature_name_)
    except AttributeError:
        pass

    available = [c for c in feat_cols if c in df.columns]
    missing = set(feat_cols) - set(available)
    if missing:
        logger.warning("%d feature columns missing from dataset: %s", len(missing), missing)

    X = df[available].fillna(0).astype(np.float32)
    df = df.copy()
    df["pred_score"] = booster.predict(X)
    return df


# ---------------------------------------------------------------------------
# IC metrics
# ---------------------------------------------------------------------------

def _compute_snapshot_ic(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional rank IC per snapshot date."""
    rows = []
    for date, grp in df.groupby("date"):
        if len(grp) < 10:
            continue
        ic, _ = spearmanr(grp["pred_score"], grp["forward_10d_rank"])
        if not np.isnan(ic):
            rows.append({"date": pd.Timestamp(date), "ic": float(ic)})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _monthly_ic(snapshot_ic: pd.DataFrame) -> pd.DataFrame:
    """Aggregate snapshot ICs to monthly mean."""
    tmp = snapshot_ic.copy()
    tmp["month"] = tmp["date"].dt.to_period("M")
    monthly = (
        tmp.groupby("month")["ic"]
        .mean()
        .reset_index()
        .rename(columns={"ic": "ic"})
    )
    monthly["month"] = monthly["month"].astype(str)
    return monthly


def _ic_summary(monthly: pd.DataFrame, snapshot_ic: pd.DataFrame) -> dict:
    """ICIR, hit rate, t-stat from monthly IC series."""
    ics = monthly["ic"].values
    n = len(ics)
    mean_ic = float(np.mean(ics))
    std_ic = float(np.std(ics, ddof=1)) if n > 1 else float("nan")
    icir = mean_ic / std_ic if std_ic > 0 else float("nan")
    hit_rate = float((ics > 0).mean())
    tstat = (mean_ic / (std_ic / np.sqrt(n))) if (std_ic > 0 and n > 1) else float("nan")
    overall_ic = float(spearmanr(
        snapshot_ic["ic"].notna() * snapshot_ic["ic"],
        np.zeros(len(snapshot_ic))
    ).correlation) if False else float(np.mean(snapshot_ic["ic"]))
    return {
        "ic_mean": round(mean_ic, 4),
        "ic_std": round(std_ic, 4),
        "icir": round(icir, 3),
        "ic_hit_rate": round(hit_rate, 3),
        "ic_tstat": round(tstat, 3),
        "n_months": n,
    }


# ---------------------------------------------------------------------------
# Long-short portfolio
# ---------------------------------------------------------------------------

def _build_ls_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Equal-weighted long top decile / short bottom decile, rebalanced each snapshot.
    Returns a DataFrame with columns: date, period_return, cumulative_return, underwater.
    forward_10d is winsorized at 1st/99th percentile to remove data errors.
    """
    # Winsorize forward_10d to remove extreme outliers (e.g. max=22999)
    lo = df["forward_10d"].quantile(0.01)
    hi = df["forward_10d"].quantile(0.99)
    df = df.copy()
    df["fwd"] = df["forward_10d"].clip(lo, hi)

    rows = []
    for date, grp in df.groupby("date"):
        if len(grp) < 20:
            continue
        grp = grp.sort_values("pred_score")
        n = len(grp)
        n_decile = max(1, n // 10)

        short_ret = float(grp["fwd"].iloc[:n_decile].mean())
        long_ret = float(grp["fwd"].iloc[-n_decile:].mean())

        # Long − Short − round-trip cost
        period_ret = long_ret - short_ret - TX_COST
        rows.append({"date": pd.Timestamp(date), "period_return": period_ret})

    portfolio = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    portfolio["cumulative_return"] = (1 + portfolio["period_return"]).cumprod()

    # Underwater (drawdown from peak)
    running_max = portfolio["cumulative_return"].cummax()
    portfolio["underwater"] = (portfolio["cumulative_return"] - running_max) / running_max

    return portfolio


def _ls_summary(portfolio: pd.DataFrame) -> dict:
    """Compute annualized Sharpe, return, and max drawdown from the L/S portfolio."""
    rets = portfolio["period_return"].values
    mean_r = float(np.mean(rets))
    std_r = float(np.std(rets, ddof=1)) if len(rets) > 1 else float("nan")
    sharpe = (mean_r / std_r) * np.sqrt(PERIODS_PER_YEAR) if std_r > 0 else float("nan")
    ann_return = float((1 + mean_r) ** PERIODS_PER_YEAR - 1)
    max_dd = float(portfolio["underwater"].min())
    return {
        "ls_sharpe": round(sharpe, 3),
        "ls_ann_return": round(ann_return, 4),
        "ls_max_drawdown": round(max_dd, 4),
        "n_periods": len(rets),
    }


# ---------------------------------------------------------------------------
# Decile returns
# ---------------------------------------------------------------------------

def _decile_returns(df: pd.DataFrame) -> List[dict]:
    """Annualized mean return per prediction decile (1 = lowest score)."""
    lo = df["forward_10d"].quantile(0.01)
    hi = df["forward_10d"].quantile(0.99)
    df = df.copy()
    df["fwd"] = df["forward_10d"].clip(lo, hi)

    df["decile"] = df.groupby("date")["pred_score"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 10, labels=False) + 1
        if len(x) >= 10 else np.nan
    )
    df = df.dropna(subset=["decile"])
    df["decile"] = df["decile"].astype(int)

    result = []
    for d in range(1, 11):
        mean_period = float(df[df["decile"] == d]["fwd"].mean())
        ann = float((1 + mean_period) ** PERIODS_PER_YEAR - 1)
        result.append({"decile": d, "ann_return": round(ann, 4)})
    return result


# ---------------------------------------------------------------------------
# Feature importance (SHAP)
# ---------------------------------------------------------------------------

def _feature_importance(booster, df: pd.DataFrame, feat_cols: List[str]) -> List[dict]:
    """Mean |SHAP| across a sample of test rows. Falls back to split-based importance."""
    try:
        feat_cols = list(booster.model.feature_name_)
    except AttributeError:
        pass
    available = [c for c in feat_cols if c in df.columns]
    X = df[available].fillna(0).astype(np.float32)

    try:
        import shap
        sample_size = min(2000, len(X))
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(X), size=sample_size, replace=False)
        X_sample = X.iloc[sample_idx]

        # Pass the inner sklearn estimator (lgb.LGBMRegressor) to SHAP
        lgbm_estimator = booster.model if hasattr(booster, "model") else booster
        explainer = shap.TreeExplainer(lgbm_estimator)
        shap_vals = explainer.shap_values(X_sample)
        mean_abs = np.abs(shap_vals).mean(axis=0)
        importance = pd.Series(mean_abs, index=available)
    except Exception as exc:
        logger.warning("SHAP failed (%s) — falling back to gain-based importance", exc)
        lgbm_estimator = booster.model if hasattr(booster, "model") else booster
        feature_names = lgbm_estimator.feature_name_
        gain_vals = lgbm_estimator.booster_.feature_importance(importance_type="gain")
        importance = pd.Series(gain_vals, index=feature_names)
        # Normalize to same scale as SHAP
        total = importance.sum()
        if total > 0:
            importance = importance / total

    importance = importance.sort_values(ascending=False)
    return [
        {
            "feature": feat,
            "importance": round(float(val), 6),
            "category": _feature_category(feat),
        }
        for feat, val in importance.items()
    ]


# ---------------------------------------------------------------------------
# JSON writers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    console.print(f"  Wrote [dim]{path}[/dim]")


def _write_model_files(
    model_id: str,
    out_dir: Path,
    monthly_ic: pd.DataFrame,
    snapshot_ic: pd.DataFrame,
    portfolio: pd.DataFrame,
    decile_rets: List[dict],
    feature_imp: List[dict],
) -> None:
    model_dir = out_dir / model_id
    console.print(f"\n[bold]Writing {model_id} files → {model_dir}/[/bold]")

    # ic_over_time.json
    ic_time = [
        {"month": row["month"], "ic": round(float(row["ic"]), 4)}
        for _, row in monthly_ic.iterrows()
    ]
    _write_json(model_dir / "ic_over_time.json", ic_time)

    # cumulative_ic.json
    cumulative = float(0)
    cum_ic = []
    for row in ic_time:
        cumulative += row["ic"]
        cum_ic.append({"month": row["month"], "cumulative_ic": round(cumulative, 4)})
    _write_json(model_dir / "cumulative_ic.json", cum_ic)

    # decile_returns.json
    _write_json(model_dir / "decile_returns.json", decile_rets)

    # equity_curve.json
    equity = [
        {
            "date": row["date"].strftime("%Y-%m-%d"),
            "cumulative_return": round(float(row["cumulative_return"]), 6),
            "underwater": round(float(row["underwater"]), 6),
        }
        for _, row in portfolio.iterrows()
    ]
    _write_json(model_dir / "equity_curve.json", equity)

    # feature_importance.json (top 20)
    _write_json(model_dir / "feature_importance.json", feature_imp[:20])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command("backtest")
@click.option("--model", "model_id", default="lgbm_v4", show_default=True,
              help="Model checkpoint name (must exist in ~/.morningalpha/models/).")
@click.option("--out", "out_dir", default=str(DEFAULT_OUT), show_default=True,
              help="Output directory for JSON files.")
@click.option("--dataset", "dataset_path", default=str(DATASET_PATH), show_default=True,
              help="Path to training dataset parquet.")
def backtest(model_id: str, out_dir: str, dataset_path: str):
    """Run the full evaluation suite for a trained model and write dashboard JSON.

    \b
    Examples:
      alpha ml backtest
      alpha ml backtest --model lgbm_v4 --out morningalpha/web/public/data/backtest/
    """
    global DATASET_PATH
    DATASET_PATH = Path(dataset_path)
    out = Path(out_dir)

    # --- Load model + data ---
    console.print(f"\n[bold cyan]Loading model: {model_id}[/bold cyan]")
    booster, config = _load_model_and_config(model_id)
    feat_cols = config["feature_columns"]
    persistence_ic = config.get("persistence_ic", float("nan"))

    console.print(f"[bold cyan]Loading dataset: {DATASET_PATH}[/bold cyan]")
    df = _load_test_data(config)

    # --- Inference ---
    console.print("\n[bold cyan]Running inference...[/bold cyan]")
    df = _run_inference(booster, df, feat_cols)

    # --- IC metrics ---
    console.print("\n[bold cyan]Computing IC metrics...[/bold cyan]")
    snap_ic = _compute_snapshot_ic(df)
    monthly = _monthly_ic(snap_ic)
    ic_stats = _ic_summary(monthly, snap_ic)

    # --- L/S portfolio ---
    console.print("[bold cyan]Building long-short portfolio...[/bold cyan]")
    portfolio = _build_ls_portfolio(df)
    ls_stats = _ls_summary(portfolio)

    # --- Decile returns ---
    console.print("[bold cyan]Computing decile returns...[/bold cyan]")
    decile_rets = _decile_returns(df)

    # --- Feature importance ---
    console.print("[bold cyan]Computing SHAP feature importance...[/bold cyan]")
    feat_imp = _feature_importance(booster, df, feat_cols)

    # --- Summary table ---
    table = Table(title=f"Backtest Results — {model_id}", show_header=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("IC Mean (monthly)", f"{ic_stats['ic_mean']:.4f}")
    table.add_row("IC Std", f"{ic_stats['ic_std']:.4f}")
    table.add_row("ICIR", f"{ic_stats['icir']:.3f}")
    table.add_row("IC Hit Rate", f"{ic_stats['ic_hit_rate']:.1%}")
    table.add_row("IC t-stat", f"{ic_stats['ic_tstat']:.2f}")
    table.add_row("L/S Sharpe", f"{ls_stats['ls_sharpe']:.3f}")
    table.add_row("L/S Ann. Return", f"{ls_stats['ls_ann_return']:.2%}")
    table.add_row("L/S Max Drawdown", f"{ls_stats['ls_max_drawdown']:.2%}")
    table.add_row("Persistence IC (test)", f"{persistence_ic:.4f}")
    console.print(table)

    # --- Write JSON ---
    _write_model_files(
        model_id, out, monthly, snap_ic, portfolio, decile_rets, feat_imp
    )

    # --- leaderboard.json ---
    leaderboard_path = out / "leaderboard.json"
    existing: List[dict] = []
    if leaderboard_path.exists():
        try:
            with open(leaderboard_path) as f:
                existing = json.load(f)
        except Exception:
            pass

    # Upsert this model's entry
    entry = {
        "model_id": model_id,
        "model_type": config.get("model_type", "lgbm").upper(),
        "n_features": len(feat_cols),
        "test_period": {
            "start": df["date"].min().strftime("%Y-%m-%d"),
            "end": df["date"].max().strftime("%Y-%m-%d"),
        },
        **ic_stats,
        **ls_stats,
        "persistence_ic": round(float(persistence_ic), 4),
        "n_test_rows": len(df),
        "n_test_dates": int(df["date"].nunique()),
    }
    existing = [e for e in existing if e.get("model_id") != model_id]
    existing.append(entry)
    _write_json(leaderboard_path, existing)

    console.print(f"\n[bold green]Backtest complete → {out}/[/bold green]")
