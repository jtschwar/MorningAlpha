#!/bin/bash
#SBATCH --job-name=lstm_ablation
#SBATCH --output=results/lstm_ablation/slurm_%j.log
#SBATCH --error=results/lstm_ablation/slurm_%j.log
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# ---- Environment ----
source /hpc/projects/group.czii/jonathan.schwartz/tomotv/alpha/bin/activate
cd /hpc/projects/group.czii/jonathan.schwartz/tomotv/alpha/MorningAlpha

mkdir -p results/lstm_ablation

echo "=== [$(date)] Starting LSTM ablation study ==="
echo "    Host: $(hostname)"
echo "    GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"

python scripts/lstm_ablation.py \
    --dataset data/training/dataset.parquet \
    --output-dir results/lstm_ablation \
    --n-folds 6 \
    --epochs 50 \
    --hidden 128 \
    --layers 2 \
    --workers 4

echo "=== [$(date)] Done ==="
