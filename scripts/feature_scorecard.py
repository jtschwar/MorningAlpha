"""Leave-one-out marginal IC feature scorecard (Experiment 6).

Trains LightGBM with fixed hyperparameters (from feature_config.json) once
per feature — each time dropping one feature — and measures the IC delta.

Usage:
    python scripts/feature_scorecard.py \
        --dataset data/training/dataset.parquet \
        --params-from models/feature_config.json \
        --target forward_10d_rank \
        --output data/training/feature_scorecard.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).parents[1]))

from morningalpha.ml.features import FEATURE_COLUMNS
from morningalpha.ml.train import rank_ic

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(dataset_path: str, target: str):
    df = pd.read_parquet(dataset_path)
    feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in dataset. Available: {[c for c in df.columns if 'forward' in c]}")

    missing_thresh = 0.30 * len(feat_cols)
    df = df[df[feat_cols].isna().sum(axis=1) <= missing_thresh].copy()
    df[feat_cols] = df[feat_cols].fillna(0)

    def _split(name):
        sub = df[df["split"] == name]
        return sub[feat_cols].astype(np.float32), sub[target].values.astype(np.float32)

    X_tr, y_tr = _split("train")
    X_va, y_va = _split("val")
    X_te, y_te = _split("test")
    return X_tr, y_tr, X_va, y_va, X_te, y_te, feat_cols


def train_lgbm_fixed(X_tr, y_tr, X_va, y_va, X_te, y_te, params: dict):
    import lightgbm as lgb
    full_params = {
        "n_estimators": 1000,
        "verbose": -1,
        "objective": "regression",
        "metric": "rmse",
        **params,
    }
    model = lgb.LGBMRegressor(**full_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    return rank_ic(model.predict(X_te), y_te)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Leave-one-out feature scorecard")
    parser.add_argument("--dataset", default="data/training/dataset.parquet")
    parser.add_argument("--params-from", default="models/feature_config.json", dest="params_from")
    parser.add_argument("--target", default="forward_10d_rank")
    parser.add_argument("--output", default="data/training/feature_scorecard.csv")
    args = parser.parse_args()

    # Load best hyperparams
    with open(args.params_from) as f:
        config = json.load(f)
    best_params = config.get("best_params", {})
    print(f"Using hyperparams from {args.params_from}: {best_params}")

    # Load dataset
    X_tr, y_tr, X_va, y_va, X_te, y_te, feat_cols = load_data(args.dataset, args.target)
    print(f"Dataset: {len(X_tr)} train / {len(X_va)} val / {len(X_te)} test, {len(feat_cols)} features")

    # Full model baseline
    print("\nTraining full model (baseline)...")
    full_ic = train_lgbm_fixed(X_tr, y_tr, X_va, y_va, X_te, y_te, best_params)
    print(f"Full model test IC: {full_ic:.4f}")

    # Leave-one-out
    results = []
    total = len(feat_cols)
    for i, feature in enumerate(feat_cols):
        reduced = [f for f in feat_cols if f != feature]
        ic = train_lgbm_fixed(
            X_tr[reduced], y_tr,
            X_va[reduced], y_va,
            X_te[reduced], y_te,
            best_params,
        )
        delta = full_ic - ic
        keep = delta >= 0.001
        results.append({
            "feature": feature,
            "full_ic": round(full_ic, 5),
            "reduced_ic": round(ic, 5),
            "delta_ic": round(delta, 5),
            "keep": keep,
        })
        print(f"  [{i+1:3d}/{total}] {feature:<35s}  reduced_ic={ic:.4f}  delta={delta:+.4f}  {'KEEP' if keep else 'drop'}")

    scorecard = pd.DataFrame(results).sort_values("delta_ic", ascending=False)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    scorecard.to_csv(args.output, index=False)
    print(f"\nScorecard saved → {args.output}")

    # Summary
    keep_count = scorecard["keep"].sum()
    drop_count = (~scorecard["keep"]).sum()
    hurts = (scorecard["delta_ic"] < 0).sum()
    print(f"\nSummary: {keep_count} keep  |  {drop_count} marginal/drop  |  {hurts} hurt IC when dropped")
    print("\nTop 15 features by marginal IC:")
    print(scorecard.head(15).to_string(index=False))
    print("\nBottom 10 features (candidates for removal):")
    print(scorecard.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
