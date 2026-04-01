#!/bin/bash
# ============================================================
# SLURM job script — MorningAlpha Set Transformer experiments
# Adjust the #SBATCH directives to match your HPC cluster.
# ============================================================
#SBATCH --job-name=ma_set_transformer
#SBATCH --output=results/set_transformer/slurm_%j.log
#SBATCH --error=results/set_transformer/slurm_%j.log
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00

# ---- Environment setup ----
# Adjust these to match your HPC module/conda setup:
# module load cuda/12.1
# module load python/3.11
# conda activate morningalpha
# -- OR --
# source /path/to/venv/bin/activate

# Ensure the package is installed (runs pip install -e . if needed)
pip install -e . --quiet

# ---- Create output directory ----
mkdir -p results/set_transformer

# ---- Run all experiments ----
# To run a subset: --experiments st_sector_composite_v1 st_sector_relative_v1
python scripts/run_st_experiments.py \
    --dataset data/training/dataset.parquet \
    --output-dir results/set_transformer \
    --device cuda \
    --seed 42

echo "Job complete. Results in results/set_transformer/"
