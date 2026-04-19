"""alpha ml train — Train baseline and LightGBM models on the labeled dataset."""
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

from morningalpha.ml.features import FEATURE_COLUMNS, FLOAT_FEATURES

# ---------------------------------------------------------------------------
# Experiment helpers — feature sets, monotone constraints, objective prep
# ---------------------------------------------------------------------------

# Value/fundamental features to drop in momentum-only experiments.
# These taught the model that cheap stocks outperform — wrong lesson during
# the Feb-Mar 2026 tariff crash which contaminated recent training labels.
_VALUE_FEATURES = frozenset({
    "earnings_yield", "book_to_market", "sales_to_price", "roe",
    "debt_to_equity", "revenue_growth", "profit_margin", "current_ratio",
    "short_pct_float", "earnings_yield_vs_sector", "book_to_market_vs_sector",
    "earnings_yield_quality", "value_x_momentum",
})

# Monotone constraint signs: +1 = higher is better, -1 = lower is better.
# Applied only when --monotone-constraints flag is set.
_MONOTONE_POSITIVE = frozenset({
    "moving_average_alignment", "rs_rating", "rs_rating_delta_21d",
    "norm_momentum_5d", "norm_momentum_21d", "norm_momentum_63d",
    "momentum_accel_5_21", "momentum_accel_21_63",
    "days_consecutive_above_sma20", "up_down_volume_ratio_63d",
    "momentum_12_1", "log_momentum_12_1",
})
_MONOTONE_NEGATIVE = frozenset({
    "days_since_52wk_high", "price_vs_5yr_high", "price_vs_52wk_high",
})

# Threshold for "explosive" breakout: top 10% by cross-sectional market-excess rank.
_EXPLOSIVE_THRESHOLD = 0.8


def _momentum_cohort_mask(
    df: pd.DataFrame,
    mom_rank_min: float = 0.10,
    sma_rank_min: float = 0.0,
    return_pct_min: float = 0.20,
    mom_rank_max: float = 50.0,
) -> pd.Series:
    """Boolean mask selecting rows that look like ongoing momentum leaders.

    Thresholds operate on the *rank-normalized* columns in dataset.parquet
    (``momentum_12_1`` and ``price_to_sma200`` are in [-1, +1]; rank 0 = median,
    rank 0.5 ≈ 75th percentile). ``return_pct`` is in the raw percent scale.

    Default thresholds match the legacy ``--momentum-universe`` filter:
      - momentum_12_1 > 0.10  (above ~55th percentile of 12-1 momentum)
      - price_to_sma200 > 0   (above median — stock above its 200-day MA)
      - return_pct > 0.20     (stock up at least 0.20% this snapshot — trivial,
                                left in for backward compat; raise it to tighten)

    For a strict leaders cohort (top-quartile momentum, above SMA, near 52wk
    high) pass ``mom_rank_min=0.5``.
    """
    mom = df.get("momentum_12_1", pd.Series(0, index=df.index))
    sma = df.get("price_to_sma200", pd.Series(0, index=df.index))
    ret = df.get("return_pct", pd.Series(0, index=df.index))
    return (mom > mom_rank_min) & (mom < mom_rank_max) & (sma > sma_rank_min) & (ret > return_pct_min)


def _build_monotone_constraints(feat_cols: List[str]) -> List[int]:
    """Return a LightGBM monotone_constraints list aligned with feat_cols."""
    return [
        1 if f in _MONOTONE_POSITIVE else (-1 if f in _MONOTONE_NEGATIVE else 0)
        for f in feat_cols
    ]


def _make_relevance_labels(y: np.ndarray, n_bins: int = 4) -> np.ndarray:
    """Convert continuous rank targets [-1,1] → integer relevance labels [0..n_bins-1].

    Used for LambdaRank objective. qcut creates equal-frequency bins, so each
    query date contributes equally to the NDCG gradient.
    """
    s = pd.Series(y)
    try:
        labels = pd.qcut(s, n_bins, labels=False, duplicates="drop").fillna(0).astype(int)
    except Exception:
        labels = pd.cut(
            s, bins=[-1.001, -0.5, 0.0, 0.5, 1.001],
            labels=[0, 1, 2, 3], include_lowest=True,
        ).fillna(1).astype(int)
    return labels.values

logger = logging.getLogger(__name__)
console = Console()

_REPO_MODELS = Path(__file__).parents[2] / "models"
_HOME_MODELS = Path.home() / ".morningalpha" / "models"
MODEL_DIR = _REPO_MODELS if _REPO_MODELS.exists() else _HOME_MODELS

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rank_ic(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Spearman rank IC between predictions and actual returns."""
    mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    if mask.sum() < 2:
        return float("nan")
    return float(spearmanr(y_pred[mask], y_true[mask]).correlation)


def hit_rate(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Fraction of correct directional predictions."""
    mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    if mask.sum() < 2:
        return float("nan")
    return float((np.sign(y_pred[mask]) == np.sign(y_true[mask])).mean())


def cross_sectional_rank_ic(df: pd.DataFrame, pred_col: str, target_col: str = "forward_10d") -> float:
    """Compute rank IC cross-sectionally per date, then average."""
    ics = []
    for _, grp in df.groupby("date"):
        if len(grp) < 5:
            continue
        ic = spearmanr(grp[pred_col], grp[target_col]).correlation
        if not np.isnan(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else float("nan")


# ---------------------------------------------------------------------------
# Purged K-Fold
# ---------------------------------------------------------------------------

def purged_kfold_splits(
    dates: pd.Series,
    n_splits: int = 5,
    embargo_days: int = 10,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Purged k-fold splits with embargo for time series data.

    Splits the date range into n_splits folds. Training excludes the validation
    fold plus an embargo window after it to prevent label leakage.
    """
    unique_dates = sorted(dates.unique())
    n = len(unique_dates)
    fold_size = n // n_splits
    splits = []

    for i in range(n_splits):
        val_start = unique_dates[i * fold_size]
        val_end = unique_dates[min((i + 1) * fold_size - 1, n - 1)]
        embargo_cutoff = pd.Timestamp(val_end) + pd.Timedelta(days=embargo_days)

        train_mask = (dates < val_start) | (dates > embargo_cutoff)
        val_mask = (dates >= val_start) & (dates <= val_end)

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))

    return splits


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(dataset_path: str, target: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load and clean the dataset. Returns (df, feat_cols).

    Downstream callers slice by date or split column as needed.

    Daily datasets are automatically downsampled to weekly (every 5th row per
    ticker) so LGBM sees non-overlapping snapshots without overlapping forward
    return labels. Weekly/staggered datasets are left as-is.
    """
    df = pd.read_parquet(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    console.print(f"Loaded {len(df):,} rows from {dataset_path}")

    # Detect daily vs weekly by median gap across all tickers on a sample date
    sample_ticker = df["ticker"].iloc[0]
    sample_gaps = (
        df[df["ticker"] == sample_ticker]
        .sort_values("date")["date"]
        .diff().dt.days.dropna()
    )
    is_daily = sample_gaps.median() <= 2

    if is_daily:
        # Downsample to every 5th row per ticker (non-overlapping weekly equiv).
        # Keeps the most recent anchor in each window so forward labels don't overlap.
        n_before = len(df)
        df = (
            df.sort_values(["ticker", "date"])
            .groupby("ticker", sort=False)
            .apply(lambda g: g.iloc[::5])
            .reset_index(drop=True)
        )
        console.print(
            f"  Daily dataset detected — downsampled to weekly: "
            f"{len(df):,}/{n_before:,} rows ({len(df)/n_before:.1%})"
        )
    elif "is_anchor" in df.columns:
        n_before = len(df)
        df = df[df["is_anchor"] == True].copy()
        console.print(f"  is_anchor filter: {len(df):,}/{n_before:,} rows ({len(df)/n_before:.1%})")

    feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in dataset.")

    missing_thresh = 0.30 * len(feat_cols)
    df = df[df[feat_cols].isna().sum(axis=1) <= missing_thresh].copy()
    df[feat_cols] = df[feat_cols].fillna(0)
    df["date"] = pd.to_datetime(df["date"])
    console.print(f"After missing-value filter: {len(df):,} rows  ({df['date'].min().date()} → {df['date'].max().date()})")
    return df, feat_cols


def _xy(df: pd.DataFrame, feat_cols: List[str], target: str):
    """Extract (X, y, dates, df) arrays from a pre-filtered DataFrame slice."""
    X = df[feat_cols].astype(np.float32)
    y = df[target].values.astype(np.float32)
    return X, y, df["date"], df


def load_splits(dataset_path: str, target: str = "forward_10d"):
    """Load dataset and return (X, y, dates) for each static split (backward compat)."""
    df, feat_cols = load_data(dataset_path, target)

    X_tr, y_tr, d_tr, df_tr = _xy(df[df["split"] == "train"], feat_cols, target)
    X_va, y_va, d_va, df_va = _xy(df[df["split"] == "val"],   feat_cols, target)
    X_te, y_te, d_te, df_te = _xy(df[df["split"] == "test"],  feat_cols, target)

    console.print(f"  train={len(X_tr):,}  val={len(X_va):,}  test={len(X_te):,}")
    return (X_tr, y_tr, d_tr, df_tr), (X_va, y_va, d_va, df_va), (X_te, y_te, d_te, df_te), feat_cols


def load_splits_walk_forward(
    dataset_path: str,
    target: str,
    embargo_days: int = 10,
) -> Tuple[pd.DataFrame, List[str], dict]:
    """Load dataset and derive the final production splits from walk-forward folds.

    Final training window = all data before the last fold's test start (minus embargo).
    Final val window      = 90 days before that cutoff (for early stopping).
    Final test window     = last fold's test rows.

    Returns (df, feat_cols, split_dates) where split_dates has keys:
      train_end, val_start, test_start, test_end, n_folds
    """
    df, feat_cols = load_data(dataset_path, target)

    if "test_fold" not in df.columns:
        raise ValueError("Dataset has no 'test_fold' column. Re-run `alpha ml dataset`.")

    n_folds = int(df["test_fold"].max())
    if n_folds == 0:
        raise ValueError("No walk-forward folds in dataset.")

    # Walk back from the last fold to find the most recent one whose test rows
    # (after weekly downsampling) have resolved targets.  Folds overlap by
    # design (step=1 month, window=3 months), so the last fold's downsampled
    # rows often fall entirely in the null zone for long horizons.
    test_fold_k = n_folds
    for fold_k in range(n_folds, 0, -1):
        fold_rows = df[df["test_fold"] == fold_k]
        if fold_rows[target].notna().any():
            test_fold_k = fold_k
            break

    fold_rows  = df[df["test_fold"] == test_fold_k]
    resolved   = fold_rows[fold_rows[target].notna()]
    test_start = fold_rows["date"].min()
    test_end   = resolved["date"].max()
    train_end  = test_start - pd.Timedelta(days=embargo_days)
    val_start  = train_end  - pd.Timedelta(days=90)

    split_dates = {
        "train_end":   train_end,
        "val_start":   val_start,
        "test_start":  test_start,
        "test_end":    test_end,
        "n_folds":     n_folds,
        "test_fold_k": test_fold_k,   # actual fold used for test set
    }
    return df, feat_cols, split_dates


# ---------------------------------------------------------------------------
# Persistence baseline
# ---------------------------------------------------------------------------

def run_persistence_baseline(df_train, df_test, target="forward_10d", feature="return_pct"):
    """Rank IC of lagged return_pct as predictor (momentum persistence)."""
    tr_ic = cross_sectional_rank_ic(df_train.assign(**{f"_pred": df_train[feature]}), "_pred", target) if feature in df_train.columns else float("nan")
    te_ic = cross_sectional_rank_ic(df_test.assign(**{f"_pred": df_test[feature]}), "_pred", target) if feature in df_test.columns else float("nan")
    return {"train_ic": tr_ic, "test_ic": te_ic}


# ---------------------------------------------------------------------------
# Ridge
# ---------------------------------------------------------------------------

def run_ridge(X_tr, y_tr, X_va, y_va, X_te, y_te, feat_cols, alpha=1.0):
    from morningalpha.ml.baselines import RidgeModel
    model = RidgeModel(alpha=alpha)
    model.fit(X_tr, y_tr, feature_names=feat_cols)

    results = {
        "train_ic": rank_ic(model.predict(X_tr), y_tr),
        "val_ic": rank_ic(model.predict(X_va), y_va),
        "test_ic": rank_ic(model.predict(X_te), y_te),
    }
    importances = model.feature_importance_series()
    return model, results, importances


# ---------------------------------------------------------------------------
# LightGBM + Optuna
# ---------------------------------------------------------------------------

def tune_lgbm(X_tr, y_tr, d_tr, X_va, y_va, n_trials: int = 30, finetune_model=None):
    """Optuna-based hyperparameter search for LightGBM using walk-forward CV.

    Uses purged k-fold on the training set so Optuna sees multiple market
    regimes rather than overfitting to a single val period.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        console.print("[yellow]optuna not installed — using default LightGBM params[/yellow]")
        return {}

    import lightgbm as lgb

    # Pre-compute walk-forward splits on training data (positional indices)
    wf_splits = purged_kfold_splits(d_tr.reset_index(drop=True), n_splits=5, embargo_days=10)
    if not wf_splits:
        console.print("[yellow]Walk-forward splits unavailable — falling back to single val fold[/yellow]")
        wf_splits = None

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 0.8),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.9),
            "bagging_freq": 5,
            "min_child_samples": trial.suggest_int("min_child_samples", 100, 500),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.1, 100.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.1, 100.0, log=True),
            "n_estimators": 500,
            "verbose": -1,
            "objective": "regression",
            "metric": "rmse",
        }
        init_model = finetune_model.model.booster_ if finetune_model else None

        if wf_splits:
            fold_ics = []
            for tr_idx, va_idx in wf_splits:
                X_f_tr = X_tr.iloc[tr_idx]
                y_f_tr = y_tr[tr_idx]
                X_f_va = X_tr.iloc[va_idx]
                y_f_va = y_tr[va_idx]
                m = lgb.LGBMRegressor(**params)
                m.fit(
                    X_f_tr, y_f_tr,
                    eval_set=[(X_f_va, y_f_va)],
                    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
                    init_model=init_model,
                )
                ic = rank_ic(m.predict(X_f_va), y_f_va)
                if not np.isnan(ic):
                    fold_ics.append(ic)
            return -np.mean(fold_ics) if fold_ics else 0.0
        else:
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
                init_model=init_model,
            )
            return -rank_ic(model.predict(X_va), y_va)

    study = optuna.create_study(direction="minimize")
    with console.status(f"[bold]Optuna walk-forward CV ({n_trials} trials, 5 folds)...[/bold]"):
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    console.print(f"Best trial CV IC: {-study.best_value:.4f}")
    return study.best_params


def train_lgbm(X_tr, y_tr, X_va, y_va, X_te, y_te, best_params: dict, finetune_model=None):
    """Train final LightGBM model with best params."""
    from morningalpha.ml.baselines import LightGBMModel

    params = {
        "n_estimators": 1000,
        "verbose": -1,
        **best_params,
    }
    model = LightGBMModel(params=params)

    console.print("[bold]Training final LightGBM model...[/bold]")
    model.fit(X_tr, y_tr, X_va, y_va)

    results = {
        "train_ic": rank_ic(model.predict(X_tr), y_tr),
        "val_ic": rank_ic(model.predict(X_va), y_va),
        "test_ic": rank_ic(model.predict(X_te), y_te),
        "train_hit": hit_rate(model.predict(X_tr), y_tr),
        "val_hit": hit_rate(model.predict(X_va), y_va),
        "test_hit": hit_rate(model.predict(X_te), y_te),
    }
    return model, results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def generate_plots(model, X_tr, y_tr, X_va, y_va, X_te, y_te, feat_cols, plot_dir: Path):
    """Generate training diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[yellow]matplotlib not installed — skipping plots[/yellow]")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    # --- Feature importance (mean |SHAP|) ---
    try:
        sv = model.shap_values(X_te.iloc[:min(500, len(X_te))])
        mean_abs_shap = np.abs(sv).mean(axis=0)
        importance = pd.Series(mean_abs_shap, index=feat_cols).sort_values()

        fig, ax = plt.subplots(figsize=(8, 6))
        importance.plot(kind="barh", ax=ax, color="#4C72B0")
        ax.set_title("Feature Importance (mean |SHAP|)")
        ax.set_xlabel("Mean |SHAP value|")
        plt.tight_layout()
        fig.savefig(plot_dir / "feature_importance.png", dpi=120)
        plt.close(fig)
        console.print(f"  Saved feature_importance.png")
    except Exception as e:
        console.print(f"  [yellow]SHAP plot skipped: {e}[/yellow]")

    # --- Learning curve (LightGBM eval history) ---
    try:
        evals = model.model.evals_result_
        if evals:
            val_key = list(evals.keys())[-1]
            metric = list(evals[val_key].keys())[0]
            train_key = list(evals.keys())[0]
            tr_scores = evals[train_key][metric]
            va_scores = evals[val_key][metric]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(tr_scores, label="train", alpha=0.8)
            ax.plot(va_scores, label="val", alpha=0.8)
            best_round = model.model.best_iteration_
            ax.axvline(best_round, color="red", linestyle="--", label=f"best @ {best_round}")
            ax.set_xlabel("Boosting Round")
            ax.set_ylabel(metric.upper())
            ax.set_title("Learning Curve")
            ax.legend()
            plt.tight_layout()
            fig.savefig(plot_dir / "learning_curve.png", dpi=120)
            plt.close(fig)
            console.print(f"  Saved learning_curve.png")
    except Exception as e:
        console.print(f"  [yellow]Learning curve skipped: {e}[/yellow]")

    # --- Rank IC by fold (purged k-fold on val set using time-ordered slices) ---
    try:
        # Simple 5 folds on training data
        n = len(X_tr)
        fold_size = n // 5
        fold_ics = []
        for i in range(5):
            fold_val_idx = np.arange(i * fold_size, min((i + 1) * fold_size, n))
            preds = model.predict(X_tr.iloc[fold_val_idx])
            ic = rank_ic(preds, y_tr[fold_val_idx])
            fold_ics.append(ic)

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#4C72B0" if ic > 0 else "#DD8452" for ic in fold_ics]
        ax.bar(range(1, 6), fold_ics, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(np.mean(fold_ics), color="red", linestyle="--", label=f"mean={np.mean(fold_ics):.4f}")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Rank IC")
        ax.set_title("Rank IC by Training Fold")
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "ic_by_fold.png", dpi=120)
        plt.close(fig)
        console.print(f"  Saved ic_by_fold.png")
    except Exception as e:
        console.print(f"  [yellow]IC by fold plot skipped: {e}[/yellow]")


def _upsert_model_config(model_dir: Path, name: str, model_type: str, test_ic: float | None) -> None:
    """Add or update the model entry in config.json. Does not change the champion."""
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {"champion": name, "models": []}

    ic_str = f"{test_ic:.3f}" if test_ic is not None else "n/a"
    entry = {
        "id": name,
        "type": model_type,
        "description": f"{model_type.upper()} — test IC {ic_str}",
        "status": "candidate",
    }

    existing_ids = [m["id"] for m in config["models"]]
    if name in existing_ids:
        config["models"] = [entry if m["id"] == name else m for m in config["models"]]
    else:
        config["models"].append(entry)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    console.print(f"Config updated → {config_path}")


# ---------------------------------------------------------------------------
# Walk-forward CV
# ---------------------------------------------------------------------------

def walk_forward_cv(
    df: pd.DataFrame,
    feat_cols: List[str],
    best_params: dict,
    target: str = "forward_63d_composite_rank",
    objective: str = "regression",
    monotone_constraints_list: Optional[List[int]] = None,
    embargo_days: int = 10,
    lookback_years: float = 5.0,
) -> pd.DataFrame:
    """Expanding-window walk-forward CV using pre-assigned test_fold labels.

    Accepts an already-loaded, cleaned DataFrame and tuned best_params.
    For each fold N:
      - Train : all rows where date < fold_N_test_start (minus embargo)
      - Val   : last 90 calendar days of that window (early stopping)
      - Test  : rows tagged test_fold == N

    objective : "regression" (default), "lambdarank", or "binary".
      - lambdarank: converts target to integer relevance labels, uses LGBMRanker
      - binary: converts target >= 0.8 → top-10% label, uses LGBMClassifier
      In all cases, WFCV IC is computed as Spearman rank IC between predicted
      scores and the original *continuous* target (forward_63d_market_excess_rank).

    monotone_constraints_list : if provided, added to LightGBM params to
      prevent anti-momentum splits.

    lookback_years : evaluate this many years of recent folds (default 5.0).
                     Converted to max_folds via years * 12 since fold_step=1mo.
                     Use 0 to evaluate all folds.

    Returns a DataFrame with per-fold IC, hit rate, n_train, n_test.
    """
    from morningalpha.ml.baselines import LightGBMModel
    import lightgbm as lgb

    lgbm_params = {"n_estimators": 1000, "verbose": -1, **best_params}
    if monotone_constraints_list is not None:
        lgbm_params["monotone_constraints"] = monotone_constraints_list
    n_folds = int(df["test_fold"].max())

    max_folds = round(lookback_years * 12) if lookback_years > 0 else None
    first_fold = (n_folds - max_folds + 1) if (max_folds and n_folds > max_folds) else 1
    fold_results = []

    MIN_RESOLUTION = 0.10  # skip fold if < 10% of val or test targets are non-null

    for fold_n in range(first_fold, n_folds + 1):
        fold_test = df[df["test_fold"] == fold_n]
        if len(fold_test) < 10:
            continue

        fold_test_start = fold_test["date"].min()
        embargo_cutoff  = fold_test_start - pd.Timedelta(days=embargo_days)
        fold_train_all  = df[df["date"] < embargo_cutoff]
        if len(fold_train_all) < 100:
            continue

        val_cutoff = embargo_cutoff - pd.Timedelta(days=90)
        fold_val = fold_train_all[fold_train_all["date"] >= val_cutoff]
        fold_tr  = fold_train_all[fold_train_all["date"] <  val_cutoff]

        if len(fold_tr) < 50 or len(fold_val) < 10:
            fold_tr  = fold_train_all
            fold_val = fold_train_all.tail(max(10, len(fold_train_all) // 10))

        # Skip folds where test or val targets are insufficiently resolved.
        # Overlapping folds near the present land in the null zone — training
        # them wastes compute and produces misleading val loss / NaN test IC.
        test_res = fold_test[target].notna().mean()
        val_res  = fold_val[target].notna().mean()
        if test_res < MIN_RESOLUTION:
            console.print(
                f"  Fold {fold_n:>3}  "
                f"{str(fold_test_start.date()):>12} → {str(fold_test['date'].max().date()):<12}  "
                f"SKIPPED — test resolution {test_res:.0%} < {MIN_RESOLUTION:.0%}"
            )
            continue
        if val_res < MIN_RESOLUTION:
            console.print(
                f"  Fold {fold_n:>3}  "
                f"{str(fold_test_start.date()):>12} → {str(fold_test['date'].max().date()):<12}  "
                f"SKIPPED — val resolution {val_res:.0%} < {MIN_RESOLUTION:.0%}"
            )
            continue

        X_tr, y_tr, _, _ = _xy(fold_tr,   feat_cols, target)
        X_va, y_va, _, _ = _xy(fold_val,  feat_cols, target)
        X_te, y_te, _, _ = _xy(fold_test, feat_cols, target)

        if objective == "regression":
            model = LightGBMModel(params=lgbm_params)
            model.fit(X_tr, y_tr, X_va, y_va)
            preds = model.predict(X_te)

        elif objective == "lambdarank":
            # Sort by date so query groups are contiguous
            tr_s = fold_tr.sort_values("date")
            va_s = fold_val.sort_values("date")
            te_s = fold_test.sort_values("date")
            X_tr_s = tr_s[feat_cols].fillna(0).astype(np.float32)
            X_va_s = va_s[feat_cols].fillna(0).astype(np.float32)
            X_te_s = te_s[feat_cols].fillna(0).astype(np.float32)
            y_tr_int = _make_relevance_labels(tr_s[target].fillna(0).values)
            y_va_int = _make_relevance_labels(va_s[target].fillna(0).values)
            y_te     = te_s[target].fillna(0).values.astype(np.float32)  # continuous for IC
            group_tr = tr_s.groupby("date", sort=True).size().values
            group_va = va_s.groupby("date", sort=True).size().values
            ltr_params = {**lgbm_params, "objective": "lambdarank", "metric": "ndcg",
                          "ndcg_eval_at": [5, 10]}
            ranker = lgb.LGBMRanker(**ltr_params)
            ranker.fit(
                X_tr_s, y_tr_int, group=group_tr,
                eval_set=[(X_va_s, y_va_int)], eval_group=[group_va],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
            )
            preds = ranker.predict(X_te_s)

        elif objective == "binary":
            y_tr_bin = (y_tr >= _EXPLOSIVE_THRESHOLD).astype(np.float32)
            y_va_bin = (y_va >= _EXPLOSIVE_THRESHOLD).astype(np.float32)
            bin_params = {**lgbm_params, "objective": "binary", "metric": "auc"}
            clf = lgb.LGBMClassifier(**bin_params)
            clf.fit(
                X_tr, y_tr_bin,
                eval_set=[(X_va, y_va_bin)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
            )
            preds = clf.predict_proba(X_te)[:, 1]

        else:
            raise ValueError(f"Unknown objective: {objective!r}")

        ic = rank_ic(preds, y_te)
        hr = hit_rate(preds, y_te)

        fold_results.append({
            "fold":           fold_n,
            "test_start":     fold_test_start.date(),
            "test_end":       fold_test["date"].max().date(),
            "val_start":      fold_val["date"].min().date(),
            "val_end":        fold_val["date"].max().date(),
            "n_train":        len(fold_tr),
            "n_test":         len(fold_test),
            "ic":             round(ic, 4),
            "hit_rate":       round(hr, 4),
        })

        console.print(
            f"  Fold {fold_n:>3}  "
            f"{str(fold_test_start.date()):>12} → {str(fold_test['date'].max().date()):<12}  "
            f"IC={ic:+.4f}  hit={hr:.3f}  train={len(fold_tr):,}"
        )

    return pd.DataFrame(fold_results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command("train")
@click.option("--dataset", default="data/training/dataset.parquet", show_default=True, help="Path to dataset parquet.")
@click.option("--model", "model_type", default="lgbm", type=click.Choice(["lgbm", "ridge"]), show_default=True, help="Model type.")
@click.option("--target", default="forward_63d_composite_rank", show_default=True, help="Target label column. forward_63d_composite_rank (default) blends return/Sharpe/consistency/drawdown to predict quality-adjusted growth. forward_10d_rank for pure short-term return ranking. forward_63d_market_excess_rank for raw excess-return ranking.")
@click.option("--name", default=None, help="Model checkpoint name (default: {model_type}_v1).")
@click.option("--output", default=None, help="Path to save checkpoint (default: ~/.morningalpha/models/{name}.pkl).")
@click.option("--n-trials", "n_trials", default=30, show_default=True, help="Optuna hyperparameter search trials (lgbm only). Use 0 to skip tuning and run default params.")
@click.option("--finetune", is_flag=True, default=False, help="Warm-start from existing checkpoint.")
@click.option("--checkpoint", default=None, help="Source checkpoint path for fine-tuning.")
@click.option("--no-plots", "no_plots", is_flag=True, default=False, help="Skip diagnostic plot generation.")
@click.option("--exclude-features", "exclude_features", default=None, help="Comma-separated feature names to exclude (for ablation experiments).")
@click.option("--momentum-universe", "momentum_universe", is_flag=True, default=False,
              help="Restrict training rows to a momentum cohort (rank-normalized "
                   "momentum_12_1 > 0.10 AND price_to_sma200 > 0) and drop pure-value "
                   "features (earnings_yield, book_to_market, etc.). Applied to BOTH "
                   "walk-forward CV folds and the final expanding-window model, so "
                   "reported IC reflects cohort-only performance.")
@click.option("--no-walk-forward", "no_walk_forward", is_flag=True, default=False,
              help="Skip walk-forward CV and train on static splits only (faster, less rigorous).")
@click.option("--wfcv-years", "wfcv_years", default=5.0, show_default=True,
              help="Years of recent history to evaluate in walk-forward CV (default 5.0 ≈ 60 folds). Use 0 for all folds.")
@click.option("--objective", default="regression",
              type=click.Choice(["regression", "lambdarank", "binary"]), show_default=True,
              help="Training objective. 'regression': rank target (default). 'lambdarank': LTR with NDCG@K, "
                   "converts target to 4-level relevance labels. 'binary': explosive breakout classifier "
                   "(top-10%% cross-sectional excess return). All objectives report Spearman rank IC for comparison.")
@click.option("--feature-set", "feature_set", default="full",
              type=click.Choice(["full", "momentum"]), show_default=True,
              help="Feature set. 'full': all features (default). 'momentum': drop the 13 value/fundamental "
                   "features (earnings_yield, book_to_market, etc.) that bias the model toward value stocks.")
@click.option("--monotone-constraints", "monotone_constraints", is_flag=True, default=False,
              help="Apply monotone constraints to prevent anti-momentum tree splits. "
                   "Positive: moving_average_alignment, rs_rating, norm_momentum_*, momentum_accel_*, etc. "
                   "Negative: days_since_52wk_high, price_vs_5yr_high, price_vs_52wk_high.")
def train(dataset, model_type, target, name, output, n_trials, finetune, checkpoint, no_plots,
          exclude_features, momentum_universe, no_walk_forward, wfcv_years,
          objective, feature_set, monotone_constraints):
    """Train a model on the labeled dataset.

    \b
    Examples:
      alpha ml train --tickers-from data/latest/stocks_3m.csv
      alpha ml train --model lgbm --n-trials 50 --name lgbm_v2
      alpha ml train --finetune --checkpoint ~/.morningalpha/models/lgbm_v1.pkl
    """
    # Resolve checkpoint name and output path
    if name is None:
        name = f"{model_type}_v1"
    if output is None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        output = str(MODEL_DIR / f"{name}.pkl")

    use_walk_forward = (not no_walk_forward) and (model_type == "lgbm")

    # Load data — walk-forward uses date-based final splits; static uses split column
    if use_walk_forward:
        df_full, feat_cols, split_dates = load_splits_walk_forward(dataset, target=target)
        console.print(
            f"\n[bold]Walk-forward mode:[/bold] "
            f"final train → {split_dates['train_end'].date()}  "
            f"val {split_dates['val_start'].date()} → {split_dates['train_end'].date()}  "
            f"test {split_dates['test_start'].date()} → {split_dates['test_end'].date()}  "
            f"({split_dates['n_folds']} folds)"
        )

        # Apply momentum-cohort row filter to df_full before derivatives so
        # both the final expanding-window splits and the walk-forward CV see
        # the same (filtered) universe.  Without this the WFCV trains/tests on
        # the full universe while the final model trains on the cohort — IC
        # metrics become uncomparable.
        if momentum_universe:
            n_before = len(df_full)
            df_full = df_full[_momentum_cohort_mask(df_full)].reset_index(drop=True)
            console.print(
                f"[cyan]Momentum cohort filter (pre-split): "
                f"{n_before:,} → {len(df_full):,} rows ({len(df_full)/max(n_before,1):.1%} kept). "
                f"Applied to both WFCV folds and final splits.[/cyan]"
            )

        df_tr_final = df_full[df_full["date"] <  split_dates["val_start"]]
        df_va_final = df_full[(df_full["date"] >= split_dates["val_start"]) &
                              (df_full["date"] <  split_dates["train_end"])]
        df_te_final = df_full[
            (df_full["test_fold"] == split_dates["test_fold_k"]) &
            (df_full["date"] <= split_dates["test_end"])
        ]
        X_tr, y_tr, d_tr, df_tr = _xy(df_tr_final, feat_cols, target)
        X_va, y_va, d_va, df_va = _xy(df_va_final, feat_cols, target)
        X_te, y_te, d_te, df_te = _xy(df_te_final, feat_cols, target)
    else:
        (X_tr, y_tr, d_tr, df_tr), (X_va, y_va, d_va, df_va), (X_te, y_te, d_te, df_te), feat_cols = \
            load_splits(dataset, target=target)
        df_full = None

    # Apply feature exclusions (ablation experiments)
    if exclude_features:
        excluded = {f.strip() for f in exclude_features.split(",")}
        feat_cols = [f for f in feat_cols if f not in excluded]
        X_tr = X_tr[feat_cols]
        X_va = X_va[feat_cols]
        X_te = X_te[feat_cols]
        console.print(f"[yellow]Excluding {len(excluded)} features: {', '.join(sorted(excluded))}[/yellow]")

    # --feature-set momentum: drop all 13 value/fundamental features.
    # Cleaner than --momentum-universe (no row filter), just feature ablation.
    if feature_set == "momentum":
        n_before = len(feat_cols)
        feat_cols = [f for f in feat_cols if f not in _VALUE_FEATURES]
        dropped = n_before - len(feat_cols)
        X_tr = X_tr[feat_cols]; X_va = X_va[feat_cols]; X_te = X_te[feat_cols]
        if use_walk_forward and df_full is not None:
            pass  # df_full kept as-is; walk_forward_cv filters feat_cols per fold
        console.print(
            f"[cyan]Feature set = momentum: dropped {dropped} value features "
            f"→ {len(feat_cols)} features remaining[/cyan]"
        )

    # Momentum-universe mode: drop pure value features (row filter already
    # applied to df_full above for walk-forward path; handle static-split path
    # and feature exclusion here).
    if momentum_universe:
        _MOM_VALUE_FEATURES = {
            "earnings_yield", "book_to_market", "sales_to_price",
            "earnings_yield_vs_sector", "book_to_market_vs_sector",
            "earnings_yield_quality", "debt_to_equity", "current_ratio",
        }
        # Static-split path: df_full wasn't filtered earlier, so filter the
        # train/val/test dataframes here.
        if not use_walk_forward:
            tr_mask = _momentum_cohort_mask(df_tr)
            va_mask = _momentum_cohort_mask(df_va)
            te_mask = _momentum_cohort_mask(df_te)
            before = len(X_tr)
            X_tr, y_tr, d_tr, df_tr = X_tr[tr_mask], y_tr[tr_mask], d_tr[tr_mask], df_tr[tr_mask]
            X_va, y_va, d_va, df_va = X_va[va_mask], y_va[va_mask], d_va[va_mask], df_va[va_mask]
            X_te, y_te, d_te, df_te = X_te[te_mask], y_te[te_mask], d_te[te_mask], df_te[te_mask]
            console.print(
                f"[cyan]Momentum cohort filter (static splits): "
                f"{before:,} → {len(X_tr):,} train rows[/cyan]"
            )
        # Drop value features — they penalise growth stocks
        feat_cols = [f for f in feat_cols if f not in _MOM_VALUE_FEATURES]
        X_tr = X_tr[feat_cols]; X_va = X_va[feat_cols]; X_te = X_te[feat_cols]
        console.print(f"[cyan]Dropped {len(_MOM_VALUE_FEATURES)} value features — using {len(feat_cols)} momentum features[/cyan]")

    # Compute monotone constraint list after feature set is finalized
    monotone_constraints_list = None
    if monotone_constraints:
        monotone_constraints_list = _build_monotone_constraints(feat_cols)
        n_constrained = sum(1 for c in monotone_constraints_list if c != 0)
        pos = sum(1 for c in monotone_constraints_list if c == 1)
        neg = sum(1 for c in monotone_constraints_list if c == -1)
        console.print(
            f"[cyan]Monotone constraints: {n_constrained} features constrained "
            f"(+{pos} positive, -{neg} negative)[/cyan]"
        )

    # Load finetune checkpoint if requested
    finetune_model = None
    if finetune and checkpoint:
        console.print(f"[bold]Loading checkpoint for fine-tuning: {checkpoint}[/bold]")
        from morningalpha.ml.baselines import LightGBMModel
        finetune_model = LightGBMModel.load(checkpoint)

    # --- Persistence baseline ---
    console.print("\n[bold cyan]--- Persistence Baseline ---[/bold cyan]")
    persist = run_persistence_baseline(df_tr, df_te, target=target)
    console.print(f"  return_pct → {target}  train IC={persist['train_ic']:.4f}  test IC={persist['test_ic']:.4f}")
    persist_ic = persist["test_ic"]

    # --- Ridge ---
    console.print("\n[bold cyan]--- Ridge Regression ---[/bold cyan]")
    ridge_model, ridge_results, ridge_importances = run_ridge(X_tr, y_tr, X_va, y_va, X_te, y_te, feat_cols)
    console.print(f"  train IC={ridge_results['train_ic']:.4f}  val IC={ridge_results['val_ic']:.4f}  test IC={ridge_results['test_ic']:.4f}")
    console.print("  Top features by |coefficient|:")
    for feat, coef in ridge_importances.head(5).items():
        console.print(f"    {feat:25s}  {coef:+.4f}")

    # --- LightGBM ---
    if model_type == "lgbm":
        console.print(f"\n[bold cyan]--- LightGBM: Hyperparameter Tuning (objective={objective}) ---[/bold cyan]")
        # Optuna only supports regression objective; skip for lambdarank / binary.
        if objective == "regression" and n_trials > 0:
            best_params = tune_lgbm(X_tr, y_tr, d_tr, X_va, y_va, n_trials=n_trials, finetune_model=finetune_model)
        else:
            if n_trials > 0:
                console.print(f"[yellow]Optuna skipped for objective='{objective}' — using default LightGBM params[/yellow]")
            else:
                console.print("[yellow]n_trials=0 — using default LightGBM params (no Optuna)[/yellow]")
            best_params = {}

        # Persist best params so wfcv / future runs can reuse them
        params_path = MODEL_DIR / f"{name}_params.json"
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        console.print(f"Best params saved → {params_path}")

        # --- Walk-forward CV across all historical folds ---
        if use_walk_forward:
            console.print(f"\n[bold cyan]--- Walk-Forward CV ({split_dates['n_folds']} folds, objective={objective}) ---[/bold cyan]")
            wf_results = walk_forward_cv(
                df_full, feat_cols, best_params, target=target,
                objective=objective,
                monotone_constraints_list=monotone_constraints_list,
                lookback_years=wfcv_years,
            )

            if len(wf_results) > 0:
                mean_ic   = wf_results["ic"].mean()
                std_ic    = wf_results["ic"].std()
                pos_folds = (wf_results["ic"] > 0).sum()
                mean_hr   = wf_results["hit_rate"].mean()

                wf_table = Table(title="Walk-Forward CV Summary", show_header=True)
                wf_table.add_column("Metric"); wf_table.add_column("Value", style="bold")
                wf_table.add_row("Folds completed",  str(len(wf_results)))
                wf_table.add_row("Mean IC",          f"{mean_ic:+.4f}")
                wf_table.add_row("Std IC",           f"{std_ic:.4f}")
                wf_table.add_row("IC > 0",           f"{pos_folds}/{len(wf_results)} ({pos_folds/len(wf_results):.0%})")
                wf_table.add_row("Mean Hit Rate",    f"{mean_hr:.3f}")
                wf_table.add_row("Date range",       f"{wf_results['test_start'].min()} → {wf_results['test_end'].max()}")
                console.print(wf_table)

                wf_csv = Path("results") / f"{name}_wfcv.csv"
                wf_csv.parent.mkdir(parents=True, exist_ok=True)
                wf_results.to_csv(wf_csv, index=False)
                console.print(f"Per-fold results → {wf_csv}")

        # --- Final model on largest expanding window ---
        # lambdarank/binary: WFCV is the primary result; final model save is
        # regression-only until inference pipeline supports those objectives.
        console.print("\n[bold cyan]--- Final Model (expanding window) ---[/bold cyan]")
        if objective == "regression":
            final_params = {**best_params}
            if monotone_constraints_list is not None:
                final_params["monotone_constraints"] = monotone_constraints_list
            final_model, lgbm_results = train_lgbm(
                X_tr, y_tr, X_va, y_va, X_te, y_te, final_params, finetune_model
            )

            # Summary table
            table = Table(title="Final Model Results", show_header=True)
            table.add_column("Split")
            table.add_column("Rank IC", justify="right")
            table.add_column("Hit Rate", justify="right")
            table.add_row("Train", f"{lgbm_results['train_ic']:.4f}", f"{lgbm_results['train_hit']:.1%}")
            table.add_row("Val",   f"{lgbm_results['val_ic']:.4f}",   f"{lgbm_results['val_hit']:.1%}")
            table.add_row("Test",  f"{lgbm_results['test_ic']:.4f}",  f"{lgbm_results['test_hit']:.1%}")
            console.print(table)

            if lgbm_results["test_ic"] < 0.03:
                console.print("[bold yellow]WARNING: test IC < 0.03 — check dataset and features.[/bold yellow]")
            elif lgbm_results["test_ic"] >= 0.05:
                console.print("[bold green]IC >= 0.05 — good signal.[/bold green]")

            # SHAP top-15 summary
            try:
                sv = final_model.shap_values(X_te.iloc[:min(1000, len(X_te))])
                mean_abs_shap = np.abs(sv).mean(axis=0)
                shap_series = pd.Series(mean_abs_shap, index=feat_cols).sort_values(ascending=False)
                shap_table = Table(title="SHAP Top 15 Features (mean |SHAP| on test set)", show_header=True)
                shap_table.add_column("Rank", justify="right")
                shap_table.add_column("Feature")
                shap_table.add_column("Mean |SHAP|", justify="right")
                for rank, (feat, val) in enumerate(shap_series.head(15).items(), 1):
                    shap_table.add_row(str(rank), feat, f"{val:.6f}")
                console.print(shap_table)
            except Exception as e:
                console.print(f"[yellow]SHAP summary skipped: {e}[/yellow]")

            # Save model
            final_model.save(output)
            console.print(f"\n[bold green]Model saved → {output}[/bold green]")

            # Save feature config
            feat_config = {
                "model_name": name,
                "model_type": model_type,
                "objective": objective,
                "feature_set": feature_set,
                "monotone_constraints": monotone_constraints,
                "feature_columns": feat_cols,
                "target": target,
                "best_params": best_params,
                "results": lgbm_results,
                "persistence_ic": persist_ic,
                "train_cutoff": str(split_dates["train_end"].date()) if use_walk_forward else None,
                "test_start":   str(split_dates["test_start"].date()) if use_walk_forward else None,
                "test_end":     str(split_dates["test_end"].date()) if use_walk_forward else None,
            }
            feat_config_path = MODEL_DIR / f"{name}_feature_config.json"
            with open(feat_config_path, "w") as f:
                json.dump(feat_config, f, indent=2, default=str)
            shared_config_path = MODEL_DIR / "feature_config.json"
            with open(shared_config_path, "w") as f:
                json.dump(feat_config, f, indent=2, default=str)
            console.print(f"Feature config saved → {feat_config_path}")

            _upsert_model_config(MODEL_DIR, name, model_type, lgbm_results.get("test_ic"))

            if not no_plots:
                plot_dir = Path("data/training/plots") / name
                console.print(f"\n[bold]Generating plots → {plot_dir}/[/bold]")
                generate_plots(final_model, X_tr, y_tr, X_va, y_va, X_te, y_te, feat_cols, plot_dir)

        else:
            console.print(
                f"[yellow]Final model checkpoint skipped for objective='{objective}'. "
                f"WFCV IC is the primary result — see results/{name}_wfcv.csv[/yellow]"
            )

    elif model_type == "ridge":
        ridge_model.save(output)
        console.print(f"\n[bold green]Ridge model saved → {output}[/bold green]")


# ---------------------------------------------------------------------------
# Walk-forward CV CLI
# ---------------------------------------------------------------------------

@click.command("wfcv")
@click.option("--model-id", "model_id", required=True, help="Model ID to evaluate (e.g. lgbm_breakout_v5).")
@click.option("--dataset", default="data/training/dataset.parquet", show_default=True, help="Path to dataset parquet.")
@click.option("--target", default="forward_63d_composite_rank", show_default=True, help="Target label column.")
@click.option("--embargo", default=10, show_default=True, help="Embargo gap in calendar days between train and test.")
@click.option("--wfcv-years", "wfcv_years", default=5.0, show_default=True,
              help="Years of recent history to evaluate (default 5.0 ≈ 60 folds). Use 0 for all folds.")
@click.option("--output", default=None, help="Optional CSV path to save fold results.")
def wfcv(model_id, dataset, target, embargo, wfcv_years, output):
    """Expanding-window walk-forward CV across all pre-assigned test folds.

    \b
    For each fold N the model is retrained on all data before that fold's
    test window (minus an embargo gap) and evaluated on the 63-day test
    window. Reports per-fold IC, hit rate, and summary statistics.

    \b
    Examples:
      alpha ml wfcv --model-id lgbm_breakout_v5
      alpha ml wfcv --model-id lgbm_composite_v6 --target forward_63d_composite_rank
      alpha ml wfcv --model-id lgbm_breakout_v5 --output results/wfcv_breakout.csv
    """
    console.print(f"\n[bold cyan]Walk-Forward CV — {model_id}[/bold cyan]")
    console.print(f"Dataset : {dataset}")
    console.print(f"Target  : {target}")
    console.print(f"Embargo : {embargo} days\n")

    df, feat_cols, _ = load_splits_walk_forward(dataset, target=target, embargo_days=embargo)
    params_path = MODEL_DIR / f"{model_id}_params.json"
    if params_path.exists():
        with open(params_path) as f:
            best_params = json.load(f)
        console.print(f"Loaded params from {params_path}")
    else:
        console.print(f"[yellow]No params file at {params_path} — using defaults[/yellow]")
        best_params = {}

    results_df = walk_forward_cv(df, feat_cols, best_params, target=target, embargo_days=embargo,
                                 lookback_years=wfcv_years)

    if len(results_df) == 0:
        console.print("[red]No folds completed — check dataset has test_fold column.[/red]")
        return

    mean_ic  = results_df["ic"].mean()
    std_ic   = results_df["ic"].std()
    pos_folds = (results_df["ic"] > 0).sum()
    mean_hr  = results_df["hit_rate"].mean()

    table = Table(title=f"Walk-Forward CV Summary — {model_id}", show_header=True)
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="bold")
    table.add_row("Folds completed", str(len(results_df)))
    table.add_row("Mean IC",  f"{mean_ic:+.4f}")
    table.add_row("Std IC",   f"{std_ic:.4f}")
    table.add_row("IC > 0",   f"{pos_folds}/{len(results_df)} ({pos_folds/len(results_df):.0%})")
    table.add_row("Mean Hit Rate", f"{mean_hr:.3f}")
    table.add_row(
        "Date range",
        f"{results_df['test_start'].min()} → {results_df['test_end'].max()}"
    )
    console.print(table)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_path, index=False)
        console.print(f"\n[green]Results saved → {out_path}[/green]")

