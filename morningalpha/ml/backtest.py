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

    missing = set(feat_cols) - set(df.columns)
    if missing:
        logger.warning("%d feature columns missing from dataset: %s", len(missing), missing)

    # Build full feature matrix — fill missing columns with 0 (same as inference)
    X = df.reindex(columns=feat_cols, fill_value=0.0).fillna(0).astype(np.float32)
    df = df.copy()
    df["pred_score"] = booster.predict(X)
    return df


# ---------------------------------------------------------------------------
# IC metrics
# ---------------------------------------------------------------------------

def _compute_snapshot_ic(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Compute cross-sectional rank IC per snapshot date against the training target."""
    if target_col not in df.columns:
        return pd.DataFrame(columns=["date", "ic"])
    rows = []
    for date, grp in df.groupby("date"):
        if len(grp) < 10:
            continue
        ic, _ = spearmanr(grp["pred_score"], grp[target_col])
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

def _build_ls_portfolio(df: pd.DataFrame, fwd_col: str, horizon: int) -> pd.DataFrame:
    """
    Equal-weighted long top decile / short bottom decile, rebalanced each snapshot.
    Only uses non-overlapping snapshots (strided by horizon days) so each period's
    forward return window is independent — avoids compounding overlapping positions.
    Returns a DataFrame with columns: date, period_return, cumulative_return, underwater.
    The realized return column is winsorized at 1st/99th percentile to remove data errors.
    """
    periods_per_year = 252 / horizon
    lo = df[fwd_col].quantile(0.01)
    hi = df[fwd_col].quantile(0.99)
    df = df.copy()
    df["fwd"] = df[fwd_col].clip(lo, hi)

    # Stride snapshot dates by horizon to ensure non-overlapping return windows
    all_dates = sorted(df["date"].unique())
    selected_dates = [all_dates[0]]
    for d in all_dates[1:]:
        gap = (pd.Timestamp(d) - pd.Timestamp(selected_dates[-1])).days
        if gap >= horizon:
            selected_dates.append(d)

    rows = []
    for date in selected_dates:
        grp = df[df["date"] == date]
        if len(grp) < 20:
            continue
        grp = grp.sort_values("pred_score")
        n = len(grp)
        n_decile = max(1, n // 10)

        short_ret = float(grp["fwd"].iloc[:n_decile].mean())
        long_ret = float(grp["fwd"].iloc[-n_decile:].mean())

        period_ret = long_ret - short_ret - TX_COST
        rows.append({"date": pd.Timestamp(date), "period_return": period_ret})

    portfolio = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    portfolio["cumulative_return"] = (1 + portfolio["period_return"]).cumprod()
    portfolio["_periods_per_year"] = periods_per_year

    running_max = portfolio["cumulative_return"].cummax()
    portfolio["underwater"] = (portfolio["cumulative_return"] - running_max) / running_max

    return portfolio


def _ls_summary(portfolio: pd.DataFrame) -> dict:
    """Compute annualized Sharpe, return, and max drawdown from the L/S portfolio."""
    rets = portfolio["period_return"].values
    periods_per_year = float(portfolio["_periods_per_year"].iloc[0])
    mean_r = float(np.mean(rets))
    std_r = float(np.std(rets, ddof=1)) if len(rets) > 1 else float("nan")
    sharpe = (mean_r / std_r) * np.sqrt(periods_per_year) if std_r > 0 else float("nan")
    ann_return = float((1 + mean_r) ** periods_per_year - 1)
    max_dd = float(portfolio["underwater"].min())
    return {
        "ls_sharpe": round(sharpe, 3),
        "ls_ann_return": round(ann_return, 4),
        "ls_max_drawdown": round(max_dd, 4),
        "n_periods": len(rets),
    }


def _top_decile_quality(df: pd.DataFrame, fwd_col: str, horizon: int) -> dict:
    """Quality metrics for the top-decile long book vs. the full universe.

    These are the metrics that matter for a quality-growth model:
    - Does the top decile deliver better Sharpe than average?
    - Is the upside consistent (high % positive periods)?
    - How does top-decile average return compare to the bottom decile and the universe?
    """
    periods_per_year = 252 / horizon
    lo = df[fwd_col].quantile(0.01)
    hi = df[fwd_col].quantile(0.99)
    df = df.copy()
    df["fwd"] = df[fwd_col].clip(lo, hi)

    top_rets, bottom_rets, all_rets = [], [], []
    for date, grp in df.groupby("date"):
        if len(grp) < 20:
            continue
        grp = grp.sort_values("pred_score")
        n = len(grp)
        n_decile = max(1, n // 10)
        top_rets.append(float(grp["fwd"].iloc[-n_decile:].mean()))
        bottom_rets.append(float(grp["fwd"].iloc[:n_decile].mean()))
        all_rets.append(float(grp["fwd"].mean()))

    def _stats(rets):
        a = np.array(rets)
        mean_r = float(np.mean(a))
        std_r = float(np.std(a, ddof=1)) if len(a) > 1 else float("nan")
        sharpe = (mean_r / std_r) * np.sqrt(periods_per_year) if std_r and std_r > 0 else float("nan")
        consistency = float((a > 0).mean())
        ann = float((1 + mean_r) ** periods_per_year - 1)
        return {"ann_return": round(ann, 4), "sharpe": round(sharpe, 3), "consistency": round(consistency, 3)}

    return {
        "top_decile": _stats(top_rets),
        "bottom_decile": _stats(bottom_rets),
        "universe": _stats(all_rets),
    }


# ---------------------------------------------------------------------------
# Decile returns
# ---------------------------------------------------------------------------

def _decile_returns(df: pd.DataFrame, fwd_col: str, horizon: int) -> List[dict]:
    """Annualized mean return per prediction decile (1 = lowest score)."""
    periods_per_year = 252 / horizon
    lo = df[fwd_col].quantile(0.01)
    hi = df[fwd_col].quantile(0.99)
    df = df.copy()
    df["fwd"] = df[fwd_col].clip(lo, hi)

    df["decile"] = df.groupby("date")["pred_score"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 10, labels=False) + 1
        if len(x) >= 10 else np.nan
    )
    df = df.dropna(subset=["decile"])
    df["decile"] = df["decile"].astype(int)

    result = []
    for d in range(1, 11):
        mean_period = float(df[df["decile"] == d]["fwd"].mean())
        ann = float((1 + mean_period) ** periods_per_year - 1)
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
    quality: dict,
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

    # top_decile_quality.json — primary signal for quality-growth models
    _write_json(model_dir / "top_decile_quality.json", quality)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command("backtest")
@click.option("--model", "model_id", default="lgbm_composite_63d", show_default=True,
              help="Model checkpoint name (must exist in models/).")
@click.option("--out", "out_dir", default=str(DEFAULT_OUT), show_default=True,
              help="Output directory for JSON files.")
@click.option("--dataset", "dataset_path", default=str(DATASET_PATH), show_default=True,
              help="Path to training dataset parquet.")
@click.option("--target", "target_col", default=None,
              help="Training target column (auto-detected from feature_config.json if omitted).")
@click.option("--horizon", default=None, type=int,
              help="Forward return horizon in trading days (auto-detected from target name if omitted).")
def backtest(model_id: str, out_dir: str, dataset_path: str, target_col: Optional[str], horizon: Optional[int]):
    """Run the full evaluation suite for a trained model and write dashboard JSON.

    \b
    Examples:
      alpha ml backtest
      alpha ml backtest --model lgbm_composite_63d
      alpha ml backtest --model lgbm_v4 --target forward_10d_rank --horizon 10
    """
    import re

    global DATASET_PATH
    DATASET_PATH = Path(dataset_path)
    out = Path(out_dir)

    # --- Load model + data ---
    console.print(f"\n[bold cyan]Loading model: {model_id}[/bold cyan]")
    booster, config = _load_model_and_config(model_id)
    feat_cols = config["feature_columns"]
    persistence_ic = config.get("persistence_ic", float("nan"))

    # Resolve target and horizon
    if target_col is None:
        target_col = config.get("target", "forward_10d_rank")
    if horizon is None:
        # Auto-detect from target name: forward_63d_* → 63
        m = re.search(r"forward_(\d+)d", target_col)
        horizon = int(m.group(1)) if m else 10
    fwd_col = f"forward_{horizon}d"

    console.print(f"  Target: [cyan]{target_col}[/cyan]  Horizon: [cyan]{horizon}d[/cyan]  Realized: [cyan]{fwd_col}[/cyan]")

    console.print(f"[bold cyan]Loading dataset: {DATASET_PATH}[/bold cyan]")
    df = _load_test_data(config)

    if fwd_col not in df.columns:
        console.print(f"[red]Realized return column '{fwd_col}' not found in dataset. "
                      f"Rebuild dataset with --horizons including {horizon}.[/red]")
        raise SystemExit(1)

    # --- Inference ---
    console.print("\n[bold cyan]Running inference...[/bold cyan]")
    df = _run_inference(booster, df, feat_cols)

    # --- IC vs training target (secondary signal — noisy for momentum plays) ---
    console.print("\n[bold cyan]Computing IC vs training target...[/bold cyan]")
    snap_ic = _compute_snapshot_ic(df, target_col)
    monthly = _monthly_ic(snap_ic)
    ic_stats = _ic_summary(monthly, snap_ic)

    # --- L/S portfolio (what actually matters) ---
    console.print("[bold cyan]Building long-short portfolio...[/bold cyan]")
    portfolio = _build_ls_portfolio(df, fwd_col, horizon)
    ls_stats = _ls_summary(portfolio)

    # --- Top decile quality (the primary signal for quality-growth model) ---
    console.print("[bold cyan]Computing top-decile quality metrics...[/bold cyan]")
    quality = _top_decile_quality(df, fwd_col, horizon)

    # --- Decile returns ---
    console.print("[bold cyan]Computing decile returns...[/bold cyan]")
    decile_rets = _decile_returns(df, fwd_col, horizon)

    # --- Feature importance ---
    console.print("[bold cyan]Computing SHAP feature importance...[/bold cyan]")
    feat_imp = _feature_importance(booster, df, feat_cols)

    # --- Summary table ---
    table = Table(title=f"Backtest Results — {model_id}", show_header=True, show_lines=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("[bold cyan]Top Decile[/bold cyan]", "")
    table.add_row("  Ann. Return", f"{quality['top_decile']['ann_return']:.2%}")
    table.add_row("  Sharpe", f"{quality['top_decile']['sharpe']:.3f}")
    table.add_row("  Consistency (% positive periods)", f"{quality['top_decile']['consistency']:.1%}")
    table.add_row("[bold]Universe[/bold]", "")
    table.add_row("  Ann. Return", f"{quality['universe']['ann_return']:.2%}")
    table.add_row("  Sharpe", f"{quality['universe']['sharpe']:.3f}")
    table.add_row("  Consistency", f"{quality['universe']['consistency']:.1%}")
    table.add_row("[bold]Bottom Decile[/bold]", "")
    table.add_row("  Ann. Return", f"{quality['bottom_decile']['ann_return']:.2%}")
    table.add_row("  Sharpe", f"{quality['bottom_decile']['sharpe']:.3f}")
    table.add_row("  Consistency", f"{quality['bottom_decile']['consistency']:.1%}")
    table.add_row("[bold]Long-Short (non-overlapping)[/bold]", "")
    table.add_row("  L/S Sharpe", f"{ls_stats['ls_sharpe']:.3f}")
    table.add_row("  L/S Ann. Return", f"{ls_stats['ls_ann_return']:.2%}")
    table.add_row("  L/S Max Drawdown", f"{ls_stats['ls_max_drawdown']:.2%}")
    table.add_row("[dim]IC Mean (monthly)[/dim]", f"[dim]{ic_stats['ic_mean']:.4f}[/dim]")
    table.add_row("[dim]ICIR[/dim]", f"[dim]{ic_stats['icir']:.3f}[/dim]")
    console.print(table)

    # --- Write JSON ---
    _write_model_files(
        model_id, out, monthly, snap_ic, portfolio, decile_rets, feat_imp, quality
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
        "target": target_col,
        "horizon": horizon,
        "n_features": len(feat_cols),
        "test_period": {
            "start": df["date"].min().strftime("%Y-%m-%d"),
            "end": df["date"].max().strftime("%Y-%m-%d"),
        },
        "top_decile": quality["top_decile"],
        "universe": quality["universe"],
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
