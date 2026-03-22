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

def load_splits(dataset_path: str, target: str = "forward_10d"):
    """Load dataset and return (X, y, dates) for each split."""
    df = pd.read_parquet(dataset_path)
    console.print(f"Loaded {len(df):,} rows from {dataset_path}")

    feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    # Drop rows with >20% features missing
    missing_thresh = 0.30 * len(feat_cols)
    df = df[df[feat_cols].isna().sum(axis=1) <= missing_thresh].copy()
    console.print(f"After missing-value filter: {len(df):,} rows")

    # Fill remaining NaN with 0 (post rank-normalize, 0 = median)
    df[feat_cols] = df[feat_cols].fillna(0)

    def _split(name):
        sub = df[df["split"] == name]
        X = sub[feat_cols].astype(np.float32)  # keep as DataFrame — preserves feature names for LightGBM
        y = sub[target].values.astype(np.float32)
        return X, y, sub["date"], sub

    X_tr, y_tr, d_tr, df_tr = _split("train")
    X_va, y_va, d_va, df_va = _split("val")
    X_te, y_te, d_te, df_te = _split("test")

    console.print(f"  train={len(X_tr):,}  val={len(X_va):,}  test={len(X_te):,}")
    return (X_tr, y_tr, d_tr, df_tr), (X_va, y_va, d_va, df_va), (X_te, y_te, d_te, df_te), feat_cols


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
# CLI
# ---------------------------------------------------------------------------

@click.command("train")
@click.option("--dataset", default="data/training/dataset.parquet", show_default=True, help="Path to dataset parquet.")
@click.option("--model", "model_type", default="lgbm", type=click.Choice(["lgbm", "ridge"]), show_default=True, help="Model type.")
@click.option("--target", default="forward_10d_rank", show_default=True, help="Target label column. Use forward_10d_rank (default) for cross-sectional ranking, or forward_10d for raw return regression.")
@click.option("--name", default=None, help="Model checkpoint name (default: {model_type}_v1).")
@click.option("--output", default=None, help="Path to save checkpoint (default: ~/.morningalpha/models/{name}.pkl).")
@click.option("--n-trials", "n_trials", default=30, show_default=True, help="Optuna hyperparameter search trials (lgbm only).")
@click.option("--finetune", is_flag=True, default=False, help="Warm-start from existing checkpoint.")
@click.option("--checkpoint", default=None, help="Source checkpoint path for fine-tuning.")
@click.option("--no-plots", "no_plots", is_flag=True, default=False, help="Skip diagnostic plot generation.")
@click.option("--exclude-features", "exclude_features", default=None, help="Comma-separated feature names to exclude (for ablation experiments).")
def train(dataset, model_type, target, name, output, n_trials, finetune, checkpoint, no_plots, exclude_features):
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

    # Load data
    (X_tr, y_tr, d_tr, df_tr), (X_va, y_va, d_va, df_va), (X_te, y_te, d_te, df_te), feat_cols = \
        load_splits(dataset, target=target)

    # Apply feature exclusions (ablation experiments)
    if exclude_features:
        excluded = {f.strip() for f in exclude_features.split(",")}
        feat_cols = [f for f in feat_cols if f not in excluded]
        X_tr = X_tr[feat_cols]
        X_va = X_va[feat_cols]
        X_te = X_te[feat_cols]
        console.print(f"[yellow]Excluding {len(excluded)} features: {', '.join(sorted(excluded))}[/yellow]")

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
        console.print("\n[bold cyan]--- LightGBM ---[/bold cyan]")
        best_params = tune_lgbm(X_tr, y_tr, d_tr, X_va, y_va, n_trials=n_trials, finetune_model=finetune_model)
        final_model, lgbm_results = train_lgbm(X_tr, y_tr, X_va, y_va, X_te, y_te, best_params, finetune_model)

        # Summary table
        table = Table(title="LightGBM Results", show_header=True)
        table.add_column("Split")
        table.add_column("Rank IC", justify="right")
        table.add_column("Hit Rate", justify="right")
        table.add_row("Train", f"{lgbm_results['train_ic']:.4f}", f"{lgbm_results['train_hit']:.1%}")
        table.add_row("Val",   f"{lgbm_results['val_ic']:.4f}",   f"{lgbm_results['val_hit']:.1%}")
        table.add_row("Test",  f"{lgbm_results['test_ic']:.4f}",  f"{lgbm_results['test_hit']:.1%}")
        console.print(table)

        if lgbm_results["test_ic"] < 0.03:
            console.print("[bold yellow]WARNING: test IC < 0.03 — fix dataset before Phase 3.[/bold yellow]")
        elif lgbm_results["test_ic"] >= 0.05:
            console.print("[bold green]IC >= 0.05 — skip Phase 3 (LSTM), go straight to Phase 4 (Set Transformer).[/bold green]")

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
            "feature_columns": feat_cols,
            "target": target,
            "best_params": best_params,
            "results": lgbm_results,
            "persistence_ic": persist_ic,
        }
        feat_config_path = MODEL_DIR / "feature_config.json"
        with open(feat_config_path, "w") as f:
            json.dump(feat_config, f, indent=2, default=str)
        console.print(f"Feature config saved → {feat_config_path}")

        # Upsert model into config.json
        _upsert_model_config(MODEL_DIR, name, model_type, lgbm_results.get("test_ic"))

        # Plots
        if not no_plots:
            plot_dir = Path("data/training/plots") / name
            console.print(f"\n[bold]Generating plots → {plot_dir}/[/bold]")
            generate_plots(final_model, X_tr, y_tr, X_va, y_va, X_te, y_te, feat_cols, plot_dir)

    elif model_type == "ridge":
        ridge_model.save(output)
        console.print(f"\n[bold green]Ridge model saved → {output}[/bold green]")
