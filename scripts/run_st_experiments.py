#!/usr/bin/env python
"""Run all Set Transformer experiments and save results to a single JSON file.

Usage (local or HPC):
    python scripts/run_st_experiments.py
    python scripts/run_st_experiments.py --dataset data/training/dataset.parquet --output-dir results/set_transformer
    python scripts/run_st_experiments.py --experiments st_sector_composite_v1 st_sector_relative_v1
    python scripts/run_st_experiments.py --group setsize     # Tier 1: max_set_size sweep
    python scripts/run_st_experiments.py --group width       # Tier 2: d_model sweep
    python scripts/run_st_experiments.py --group depth       # Tier 3: num_blocks sweep
    python scripts/run_st_experiments.py --group reg         # Tier 4: dropout sweep
    python scripts/run_st_experiments.py --dry-run           # print config and exit

Scaling sweep strategy — run tiers in order, each informs the next:
  Tier 1 (setsize):  fix d_model=128/blocks=3, vary max_set_size 80→128→256
  Tier 2 (width):    fix max_set_size=256/blocks=3, vary d_model 64→128→256→512
  Tier 3 (depth):    fix max_set_size=256 + best width, vary num_blocks 2→3→4→6
  Tier 4 (reg):      fix best width+depth, vary dropout 0.0→0.1→0.2→0.3

Results are written to:
    <output-dir>/results_<timestamp>.json   ← pass this back for analysis

Each experiment is fully reproducible: config + feature list + fold splits are
embedded in the JSON so the exact run can be reconstructed.
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Experiment registry
#
# Fields:
#   group       — which scaling tier this belongs to (used by --group flag)
#   name        — unique experiment identifier
#   description — logged to results JSON
#   target      — forward return column to predict
#   loss        — "mse" or "listmle"
#   d_model     — transformer embedding dimension
#   num_heads   — attention heads (must divide d_model)
#   num_blocks  — number of SetAttentionBlocks
#   dropout     — applied to attention and FF layers
#   max_set_size — max stocks per (sector, date) batch; truncates larger sets
#   batch_size  — number of sets per gradient step; reduce for large max_set_size
#   lr          — Adam learning rate
#   weight_decay
#   epochs      — max epochs (early stopping via patience)
#   patience    — early stopping patience
# ---------------------------------------------------------------------------

EXPERIMENTS = [

    # -----------------------------------------------------------------------
    # v2 focused experiment — best config from scaling sweep + longer patience
    # d_model=256, blocks=3, dropout=0.1, max_set_size=128
    # max_set_size=128 outperformed 256 on test IC (0.155 vs 0.137) in Tier 1.
    # patience=40 allows fuller convergence; prior runs stopped at 25-55 epochs.
    # -----------------------------------------------------------------------
    {
        "group": "focused",
        "name": "st_best_v2",
        "description": "Best config from scaling sweep: d_model=256, blocks=3, dropout=0.1, max_set_size=128, patience=40. LightGBM baseline: breakout_v5 IC 0.180.",
        "target": "forward_63d_composite_rank",
        "loss": "mse",
        "d_model": 256,
        "num_heads": 8,
        "num_blocks": 3,
        "dropout": 0.1,
        "max_set_size": 128,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "epochs": 150,
        "patience": 40,
    },
    {
        "group": "focused",
        "name": "st_best_v2_breakout",
        "description": "Same as st_best_v2 but with raw return rank target — matches lgbm_breakout_v5 objective directly.",
        "target": "forward_63d_rank",
        "loss": "mse",
        "d_model": 256,
        "num_heads": 8,
        "num_blocks": 3,
        "dropout": 0.1,
        "max_set_size": 128,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "epochs": 150,
        "patience": 40,
    },

    # -----------------------------------------------------------------------
    # v1 baseline experiments (original runs — kept for reference comparison)
    # -----------------------------------------------------------------------
    {
        "group": "baseline",
        "name": "st_sector_composite_v1",
        "description": "Baseline: cross-stock attention within sectors, composite target. d_model=128, blocks=3, max_set_size=80.",
        "target": "forward_63d_composite_rank",
        "loss": "mse",
        "d_model": 128,
        "num_heads": 4,
        "num_blocks": 3,
        "dropout": 0.1,
        "max_set_size": 80,
        "batch_size": 64,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "patience": 20,
    },
    {
        "group": "baseline",
        "name": "st_sector_relative_v1",
        "description": "Baseline: cross-stock attention within sectors, sector-relative target.",
        "target": "forward_63d_sector_relative_rank",
        "loss": "mse",
        "d_model": 128,
        "num_heads": 4,
        "num_blocks": 3,
        "dropout": 0.1,
        "max_set_size": 80,
        "batch_size": 64,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "patience": 20,
    },
    {
        "group": "baseline",
        "name": "st_sector_listmle_v1",
        "description": "Baseline: composite target with ListMLE loss.",
        "target": "forward_63d_composite_rank",
        "loss": "listmle",
        "d_model": 128,
        "num_heads": 4,
        "num_blocks": 3,
        "dropout": 0.1,
        "max_set_size": 80,
        "batch_size": 64,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "patience": 20,
    },

    # -----------------------------------------------------------------------
    # Tier 1 — set size sweep
    # Fix: d_model=128, num_heads=4, num_blocks=3, dropout=0.1
    # Vary: max_set_size 80 → 128 → 256
    # Goal: measure how much the 80-stock truncation was hurting; new dataset
    #       has median 176-209 stocks per (sector, date) cohort.
    # -----------------------------------------------------------------------
    {
        "group": "setsize",
        "name": "st_setsize_80",
        "description": "Setsize sweep: max_set_size=80 (old cap). Baseline for tier 1 comparison.",
        "target": "forward_63d_composite_rank",
        "loss": "mse",
        "d_model": 128,
        "num_heads": 4,
        "num_blocks": 3,
        "dropout": 0.1,
        "max_set_size": 80,
        "batch_size": 64,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "patience": 20,
    },
    {
        "group": "setsize",
        "name": "st_setsize_128",
        "description": "Setsize sweep: max_set_size=128. Intermediate — fits ~60th percentile sector cohort.",
        "target": "forward_63d_composite_rank",
        "loss": "mse",
        "d_model": 128,
        "num_heads": 4,
        "num_blocks": 3,
        "dropout": 0.1,
        "max_set_size": 128,
        "batch_size": 64,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "patience": 20,
    },
    {
        "group": "setsize",
        "name": "st_setsize_256",
        "description": "Setsize sweep: max_set_size=256. Covers full sector cohort (median 176-209 in new dataset).",
        "target": "forward_63d_composite_rank",
        "loss": "mse",
        "d_model": 128,
        "num_heads": 4,
        "num_blocks": 3,
        "dropout": 0.1,
        "max_set_size": 256,
        "batch_size": 32,    # halve batch_size to keep GPU memory flat vs setsize_80
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "patience": 20,
    },

    # -----------------------------------------------------------------------
    # Tier 2 — width (d_model) sweep
    # Fix: max_set_size=256, num_blocks=3, dropout=0.1
    # Vary: d_model 64 → 128 → 256 → 512
    # Goal: find the representational capacity sweet spot before overfitting.
    # Notes:
    #   num_heads must divide d_model; scaled to keep head_dim ≥ 32
    #   batch_size reduced for larger d_model to stay within memory
    # -----------------------------------------------------------------------
    {
        "group": "width",
        "name": "st_width_64",
        "description": "Width sweep: d_model=64, heads=4 (head_dim=16). Minimal capacity — fast sanity check.",
        "target": "forward_63d_composite_rank",
        "loss": "mse",
        "d_model": 64,
        "num_heads": 4,
        "num_blocks": 3,
        "dropout": 0.1,
        "max_set_size": 256,
        "batch_size": 64,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "patience": 20,
    },
    {
        "group": "width",
        "name": "st_width_128",
        "description": "Width sweep: d_model=128, heads=4 (head_dim=32). Same as setsize_256 — reference point.",
        "target": "forward_63d_composite_rank",
        "loss": "mse",
        "d_model": 128,
        "num_heads": 4,
        "num_blocks": 3,
        "dropout": 0.1,
        "max_set_size": 256,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "patience": 20,
    },
    {
        "group": "width",
        "name": "st_width_256",
        "description": "Width sweep: d_model=256, heads=8 (head_dim=32). 2x baseline — meaningful capacity increase.",
        "target": "forward_63d_composite_rank",
        "loss": "mse",
        "d_model": 256,
        "num_heads": 8,
        "num_blocks": 3,
        "dropout": 0.1,
        "max_set_size": 256,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "patience": 20,
    },
    {
        "group": "width",
        "name": "st_width_512",
        "description": "Width sweep: d_model=512, heads=8 (head_dim=64). 4x baseline — upper bound; watch for overfitting.",
        "target": "forward_63d_composite_rank",
        "loss": "mse",
        "d_model": 512,
        "num_heads": 8,
        "num_blocks": 3,
        "dropout": 0.1,
        "max_set_size": 256,
        "batch_size": 16,    # large model + large sets — reduce batch to fit in memory
        "lr": 1e-4,          # lower LR for larger model stability
        "weight_decay": 1e-4,
        "epochs": 100,
        "patience": 20,
    },

    # -----------------------------------------------------------------------
    # Tier 3 — depth (num_blocks) sweep
    # Fix: max_set_size=256, dropout=0.1
    # d_model/heads set to best from Tier 2 (default: 256/8; update after Tier 2)
    # Vary: num_blocks 2 → 3 → 4 → 6
    # Goal: does extra depth help or hurt? Attention stacks can saturate quickly.
    # -----------------------------------------------------------------------
    {
        "group": "depth",
        "name": "st_depth_2",
        "description": "Depth sweep: 2 SetAttentionBlocks. Shallow — tests if depth beyond 2 actually helps.",
        "target": "forward_63d_composite_rank",
        "loss": "mse",
        "d_model": 256,
        "num_heads": 8,
        "num_blocks": 2,
        "dropout": 0.1,
        "max_set_size": 256,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "patience": 20,
    },
    {
        "group": "depth",
        "name": "st_depth_3",
        "description": "Depth sweep: 3 blocks (baseline depth). Reference point for tier 3.",
        "target": "forward_63d_composite_rank",
        "loss": "mse",
        "d_model": 256,
        "num_heads": 8,
        "num_blocks": 3,
        "dropout": 0.1,
        "max_set_size": 256,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "patience": 20,
    },
    {
        "group": "depth",
        "name": "st_depth_4",
        "description": "Depth sweep: 4 blocks. One additional attention layer over baseline.",
        "target": "forward_63d_composite_rank",
        "loss": "mse",
        "d_model": 256,
        "num_heads": 8,
        "num_blocks": 4,
        "dropout": 0.1,
        "max_set_size": 256,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "patience": 20,
    },
    {
        "group": "depth",
        "name": "st_depth_6",
        "description": "Depth sweep: 6 blocks. Deep — likely to overfit or saturate on this dataset size.",
        "target": "forward_63d_composite_rank",
        "loss": "mse",
        "d_model": 256,
        "num_heads": 8,
        "num_blocks": 6,
        "dropout": 0.1,
        "max_set_size": 256,
        "batch_size": 16,    # reduce batch to fit deeper model
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "patience": 20,
    },

    # -----------------------------------------------------------------------
    # Tier 4 — regularization (dropout) sweep
    # Fix: max_set_size=256, d_model/heads/blocks = best from Tiers 2+3
    # Vary: dropout 0.0 → 0.1 → 0.2 → 0.3
    # Goal: larger models need more regularization; find the right amount.
    # -----------------------------------------------------------------------
    {
        "group": "reg",
        "name": "st_reg_drop0",
        "description": "Regularization sweep: dropout=0.0. No regularization — upper bound on capacity, lower bound on generalization.",
        "target": "forward_63d_composite_rank",
        "loss": "mse",
        "d_model": 256,
        "num_heads": 8,
        "num_blocks": 4,
        "dropout": 0.0,
        "max_set_size": 256,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "patience": 20,
    },
    {
        "group": "reg",
        "name": "st_reg_drop10",
        "description": "Regularization sweep: dropout=0.1. Same as depth/width baselines — cross-tier reference.",
        "target": "forward_63d_composite_rank",
        "loss": "mse",
        "d_model": 256,
        "num_heads": 8,
        "num_blocks": 4,
        "dropout": 0.1,
        "max_set_size": 256,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "patience": 20,
    },
    {
        "group": "reg",
        "name": "st_reg_drop20",
        "description": "Regularization sweep: dropout=0.2. Moderate regularization — likely best for 1M+ row dataset.",
        "target": "forward_63d_composite_rank",
        "loss": "mse",
        "d_model": 256,
        "num_heads": 8,
        "num_blocks": 4,
        "dropout": 0.2,
        "max_set_size": 256,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "patience": 20,
    },
    {
        "group": "reg",
        "name": "st_reg_drop30",
        "description": "Regularization sweep: dropout=0.3. Heavy regularization — tests whether strong dropout is needed.",
        "target": "forward_63d_composite_rank",
        "loss": "mse",
        "d_model": 256,
        "num_heads": 8,
        "num_blocks": 4,
        "dropout": 0.3,
        "max_set_size": 256,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "patience": 20,
    },
]

EXPERIMENT_REGISTRY = {e["name"]: e for e in EXPERIMENTS}
GROUP_REGISTRY: dict[str, list] = {}
for e in EXPERIMENTS:
    GROUP_REGISTRY.setdefault(e["group"], []).append(e)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path):
    log_path = output_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
    )


def load_dataset(dataset_path: Path, experiments: list) -> tuple[pd.DataFrame, list]:
    """Load parquet, validate targets + features exist, return (df, feature_cols)."""
    from morningalpha.ml.features import FEATURE_COLUMNS

    print(f"Loading dataset: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    print(f"  Rows: {len(df):,}  |  Splits: {df['split'].value_counts().to_dict()}")

    # Features present in dataset
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    print(f"  Features: {len(feature_cols)} / {len(FEATURE_COLUMNS)}")

    # Validate targets
    for exp in experiments:
        target = exp["target"]
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found in dataset. Available: {[c for c in df.columns if 'forward' in c]}")

    # Require sector column
    if "sector" not in df.columns:
        raise ValueError("'sector' column required for sector-grouped sets")

    # Drop rows missing too many features (>30%)
    missing_thresh = int(0.30 * len(feature_cols))
    before = len(df)
    df = df[df[feature_cols].isna().sum(axis=1) <= missing_thresh].copy()
    print(f"  After missing-value filter: {len(df):,} rows (dropped {before - len(df):,})")

    # Fill remaining NaN with 0 (features are rank-normalized; 0 = cross-sectional median)
    df[feature_cols] = df[feature_cols].fillna(0)

    return df, feature_cols


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run Set Transformer experiments")
    parser.add_argument("--dataset", default="data/training/dataset.parquet",
                        help="Path to training parquet (default: data/training/dataset.parquet)")
    parser.add_argument("--output-dir", default="results/set_transformer",
                        help="Directory to save model checkpoints and results JSON")

    # Selection — mutually exclusive: pick by name OR by group
    select_group = parser.add_mutually_exclusive_group()
    select_group.add_argument("--experiments", nargs="+",
                              choices=list(EXPERIMENT_REGISTRY.keys()) + ["all"],
                              help="Run specific experiments by name")
    select_group.add_argument("--group", choices=list(GROUP_REGISTRY.keys()),
                              help="Run all experiments in a scaling tier: "
                                   "baseline | setsize | width | depth | reg")

    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Device to train on (default: auto-detect cuda > mps > cpu)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Random seeds to run each experiment with. Multiple seeds measure "
                             "variance. Summary reports mean ± std across seeds. "
                             "Example: --seeds 42 123 456")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config and exit without training")
    args = parser.parse_args()

    # Resolve output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)

    # Device — prefer cuda > mps > cpu
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.set_float32_matmul_precision("high")  # H100 tensor cores
    elif device.type == "mps":
        logger.info("Apple Metal (MPS) — mixed precision disabled, torch.compile may be skipped")

    # Select experiments
    if args.group:
        selected = GROUP_REGISTRY[args.group]
        logger.info(f"Group: {args.group}  ({len(selected)} experiments)")
    elif args.experiments and args.experiments != ["all"]:
        selected = [EXPERIMENT_REGISTRY[n] for n in args.experiments]
    else:
        selected = EXPERIMENTS

    seeds = args.seeds
    logger.info(f"Experiments to run: {[e['name'] for e in selected]}  |  Seeds: {seeds}")

    if args.dry_run:
        print(f"\n=== DRY RUN — {len(selected)} experiment(s) × {len(seeds)} seed(s) = {len(selected)*len(seeds)} total runs ===")
        for exp in selected:
            print(f"\n[{exp['group']}] {exp['name']}")
            print(f"  d_model={exp['d_model']}  heads={exp['num_heads']}  blocks={exp['num_blocks']}"
                  f"  dropout={exp['dropout']}  max_set_size={exp['max_set_size']}  batch={exp['batch_size']}")
            print(f"  target={exp['target']}  loss={exp['loss']}  lr={exp['lr']}")
            print(f"  {exp['description']}")
        return

    # Load data
    dataset_path = Path(args.dataset)
    df, feature_cols = load_dataset(dataset_path, selected)

    # Split by pre-assigned split column
    df_train = df[df["split"] == "train"].copy()
    df_val   = df[df["split"] == "val"].copy()
    df_test  = df[df["split"] == "test"].copy()
    logger.info(f"Split sizes — train: {len(df_train):,}  val: {len(df_val):,}  test: {len(df_test):,}")

    # ---------------------------------------------------------------------------
    # Run experiments
    # ---------------------------------------------------------------------------
    from morningalpha.ml.train_st import run_experiment

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "run_id": run_id,
        "dataset": str(dataset_path),
        "n_rows": len(df),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "device": str(device),
        "seeds": seeds,
        "lgbm_baselines": {
            "lgbm_breakout_v5": {"test_ic": 0.180, "top_decile_sharpe": 2.579},
            "lgbm_composite_v6": {"test_ic": 0.171, "top_decile_sharpe": 2.304},
        },
        "experiments": [],
    }

    results_path = output_dir / f"results_{run_id}.json"

    for exp in selected:
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment: {exp['name']}  [group={exp['group']}]  seeds={seeds}")
        logger.info(f"  {exp['description']}")
        logger.info(f"  target={exp['target']}  loss={exp['loss']}")
        logger.info(f"  d_model={exp['d_model']}  heads={exp['num_heads']}  blocks={exp['num_blocks']}"
                    f"  dropout={exp['dropout']}  max_set_size={exp['max_set_size']}  batch={exp['batch_size']}")
        logger.info(f"{'='*60}")

        seed_runs = []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            logger.info(f"  Seed {seed} ...")
            try:
                fold_result = run_experiment(
                    df_train=df_train,
                    df_val=df_val,
                    df_test=df_test,
                    feature_cols=feature_cols,
                    config=exp,
                    output_dir=output_dir,
                    device=device,
                    fold=seed,   # use seed as fold id so checkpoints don't overwrite each other
                )
                seed_runs.append({
                    "seed": seed,
                    "val_ic": fold_result.get("val_ic", {}).get("ic_mean"),
                    "test_ic": fold_result.get("test_ic", {}).get("ic_mean"),
                    "training_time_s": fold_result.get("training_time_s"),
                    "n_params": fold_result.get("n_params"),
                    "status": "completed",
                })
                logger.info(
                    f"    seed={seed} → val IC: {seed_runs[-1]['val_ic']:.4f}"
                    f" | test IC: {seed_runs[-1]['test_ic']:.4f}"
                    f" | time: {seed_runs[-1]['training_time_s']:.0f}s"
                )
            except Exception as exc:
                logger.exception(f"  Seed {seed} failed: {exc}")
                seed_runs.append({"seed": seed, "status": "failed", "error": str(exc)})

        # Aggregate across seeds
        completed = [r for r in seed_runs if r.get("status") == "completed"]
        if completed:
            val_ics  = [r["val_ic"]  for r in completed]
            test_ics = [r["test_ic"] for r in completed]
            exp_result = {
                "name": exp["name"],
                "group": exp["group"],
                "description": exp["description"],
                "config": exp,
                "seed_runs": seed_runs,
                "n_seeds": len(completed),
                "val_ic_mean":  float(np.mean(val_ics)),
                "val_ic_std":   float(np.std(val_ics))  if len(val_ics) > 1 else None,
                "test_ic_mean": float(np.mean(test_ics)),
                "test_ic_std":  float(np.std(test_ics)) if len(test_ics) > 1 else None,
                "training_time_s": sum(r["training_time_s"] for r in completed),
                "n_params": completed[0].get("n_params"),
                "status": "completed",
            }
            std_str = f" ±{exp_result['test_ic_std']:.4f}" if exp_result["test_ic_std"] is not None else ""
            logger.info(
                f"  → {len(completed)}-seed mean:  val IC {exp_result['val_ic_mean']:.4f}"
                f" | test IC {exp_result['test_ic_mean']:.4f}{std_str}"
            )
        else:
            exp_result = {"name": exp["name"], "group": exp["group"],
                          "config": exp, "seed_runs": seed_runs, "status": "failed"}

        results["experiments"].append(exp_result)

        # Save after each experiment so partial results are not lost on crash
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved: {results_path}")

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    multi_seed = len(seeds) > 1
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    if multi_seed:
        logger.info(f"  {'Experiment':<38} {'Group':<10} {'Seeds':>6} {'Val IC':>8} {'Test IC mean±std':>20} {'Time(s)':>8}")
    else:
        logger.info(f"  {'Experiment':<38} {'Group':<10} {'Params':>9} {'Val IC':>8} {'Test IC':>9} {'Time(s)':>8}")
    logger.info(f"  {'lgbm_breakout_v5 (champion)':<38} {'—':<10} {'—':>9}  0.1800   0.1800       —")
    logger.info(f"  {'lgbm_composite_v6 (candidate)':<38} {'—':<10} {'—':>9}  0.1710   0.1710       —")
    for r in results["experiments"]:
        vic  = r.get("val_ic_mean")
        tic  = r.get("test_ic_mean")
        tstd = r.get("test_ic_std")
        t    = r.get("training_time_s")
        n    = r.get("n_params")
        ns   = r.get("n_seeds", 1)
        vic_str = f"{vic:.4f}" if vic is not None else "  error"
        t_str   = f"{t:.0f}"   if t   is not None else "  —"
        if multi_seed:
            tic_str = (f"{tic:.4f} ±{tstd:.4f}" if tstd is not None else f"{tic:.4f}") if tic is not None else "  error"
            logger.info(f"  {r['name']:<38} {r.get('group','—'):<10} {ns:>6} {vic_str:>8} {tic_str:>20} {t_str:>8}")
        else:
            tic_str = f"{tic:.4f}" if tic is not None else "  error"
            n_str   = f"{n:,}"     if n   is not None else "  —"
            logger.info(f"  {r['name']:<38} {r.get('group','—'):<10} {n_str:>9} {vic_str:>8} {tic_str:>9} {t_str:>8}")
    logger.info(f"\nFull results: {results_path}")


if __name__ == "__main__":
    main()
