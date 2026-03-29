"""Set Transformer training loop.

Handles preprocessing, dataset construction, training, and IC evaluation
for SectorSetRanker experiments. Called by scripts/run_st_experiments.py.
"""
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from morningalpha.ml.features import FEATURE_COLUMNS
from morningalpha.ml.sector_dataset import SectorSetDataset
from morningalpha.ml.set_transformer import SectorSetRanker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def mse_loss_masked(scores: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE on composite rank, applied only to real stocks (mask=True)."""
    loss = (scores - targets) ** 2
    return (loss * mask.float()).sum() / mask.float().sum().clamp(min=1)


def listmle_loss(scores: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    ListMLE — maximizes log-likelihood of the correct stock ordering within each set.

    Sorts stocks by target rank (descending), then computes:
        L = -sum_i [ s_i - log( sum_{j>=i} exp(s_j) ) ]

    Padding positions are excluded via the mask before sorting.
    """
    B, N = scores.shape

    # Mask padding: set padding scores/targets to very low values so they sort last
    INF = 1e9
    scores_m = scores.masked_fill(~mask, -INF)
    targets_m = targets.masked_fill(~mask, -INF)

    # Sort by target descending — defines the "correct" order
    _, sorted_idx = targets_m.sort(dim=1, descending=True)
    sorted_scores = scores_m.gather(1, sorted_idx)
    sorted_mask = mask.gather(1, sorted_idx)

    # Numerically stable log-sum-exp from right (denominator for each position)
    max_s = sorted_scores.max(dim=1, keepdim=True).values
    exp_s = torch.exp(sorted_scores - max_s)
    # cumsum from right: sum_{j>=i} exp(s_j)
    cum_exp = exp_s.flip(1).cumsum(1).flip(1)
    log_denom = torch.log(cum_exp.clamp(min=1e-9)) + max_s

    # Loss per position (only real stocks contribute)
    position_loss = -(sorted_scores - log_denom) * sorted_mask.float()
    # Normalize by number of real stocks per set
    n_real = sorted_mask.float().sum(dim=1).clamp(min=1)
    return (position_loss.sum(dim=1) / n_real).mean()


LOSS_FNS = {
    "mse": mse_loss_masked,
    "listmle": listmle_loss,
}


# ---------------------------------------------------------------------------
# IC computation
# ---------------------------------------------------------------------------

def compute_ic(
    model: SectorSetRanker,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate IC (Spearman rank correlation between predicted score and target)
    aggregated across all stocks, and broken down by month.

    Returns dict with ic_mean, ic_std, ic_monthly (list of floats).
    """
    model.eval()
    all_scores, all_targets, all_dates = [], [], []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            scores = model(features, mask)

            # Unpack only real stocks from each set
            for i in range(len(batch["n"])):
                n = int(batch["n"][i])
                date = batch["date"][i]
                all_scores.extend(scores[i, :n].cpu().numpy().tolist())
                all_targets.extend(targets[i, :n].cpu().numpy().tolist())
                all_dates.extend([date] * n)

    df = pd.DataFrame({"score": all_scores, "target": all_targets, "date": pd.to_datetime(all_dates)})
    df = df.dropna()

    if len(df) < 10:
        return {"ic_mean": float("nan"), "ic_std": float("nan"), "ic_monthly": []}

    # Overall IC
    ic_overall, _ = spearmanr(df["score"], df["target"])

    # Monthly IC: compute IC within each calendar month, then average
    df["month"] = df["date"].dt.to_period("M")
    monthly_ics = []
    for _, grp in df.groupby("month"):
        if len(grp) < 5:
            continue
        ic, _ = spearmanr(grp["score"], grp["target"])
        monthly_ics.append(float(ic))

    return {
        "ic_mean": float(ic_overall),
        "ic_std": float(np.std(monthly_ics)) if monthly_ics else float("nan"),
        "ic_monthly": monthly_ics,
        "ic_monthly_mean": float(np.mean(monthly_ics)) if monthly_ics else float("nan"),
        "n_months": len(monthly_ics),
        "n_stocks": len(df),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: SectorSetRanker,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                scores = model(features, mask)
                loss = loss_fn(scores, targets, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        else:
            scores = model(features, mask)
            loss = loss_fn(scores, targets, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    config: dict,
    output_dir: Path,
    device: torch.device,
    fold: int = 0,
) -> dict:
    """
    Train and evaluate one Set Transformer experiment on a single train/val/test split.

    Returns a results dict with IC metrics, loss curves, and model path.
    """
    name = config["name"]
    target_col = config["target"]
    loss_name = config.get("loss", "mse")
    d_model = config.get("d_model", 128)
    num_heads = config.get("num_heads", 4)
    num_blocks = config.get("num_blocks", 3)
    dropout = config.get("dropout", 0.1)
    lr = config.get("lr", 3e-4)
    weight_decay = config.get("weight_decay", 1e-4)
    epochs = config.get("epochs", 80)
    batch_size = config.get("batch_size", 64)
    max_set_size = config.get("max_set_size", 80)
    patience = config.get("patience", 15)

    loss_fn = LOSS_FNS[loss_name]

    # Build datasets
    train_ds = SectorSetDataset(df_train, feature_cols, target_col, max_set_size)
    val_ds = SectorSetDataset(df_val, feature_cols, target_col, max_set_size)
    test_ds = SectorSetDataset(df_test, feature_cols, target_col, max_set_size)

    logger.info(f"[{name}/fold{fold}] train={train_ds.summary()}, val={val_ds.summary()['n_sets']} sets")

    if len(train_ds) == 0 or len(val_ds) == 0:
        return {"status": "error", "message": "Empty dataset after grouping"}

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=pin)

    # Model
    model = SectorSetRanker(
        dim_input=len(feature_cols),
        d_model=d_model,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout=dropout,
    ).to(device)

    n_params = model.count_parameters()
    logger.info(f"[{name}/fold{fold}] model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 20)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # torch.compile only on CUDA — inductor backend has no MPS support
    if device.type == "cuda":
        try:
            model = torch.compile(model)
            logger.info(f"[{name}/fold{fold}] torch.compile enabled")
        except Exception:
            logger.info(f"[{name}/fold{fold}] torch.compile unavailable, running eager")

    # Training
    t0 = time.time()
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_val_ic = float("-inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        scheduler.step()

        # Evaluate val loss
        model.eval()
        v_losses = []
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                targets = batch["targets"].to(device)
                mask = batch["mask"].to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    scores = model(features, mask)
                    loss = loss_fn(scores, targets, mask)
                v_losses.append(loss.item())
        val_loss = float(np.mean(v_losses))

        train_losses.append(round(train_loss, 5))
        val_losses.append(round(val_loss, 5))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save state (handle compiled model)
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            best_state = {k: v.cpu().clone() for k, v in raw.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"[{name}/fold{fold}] epoch {epoch+1}/{epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  patience={patience_counter}"
            )

        if patience_counter >= patience:
            logger.info(f"[{name}/fold{fold}] early stop at epoch {epoch+1}")
            break

    training_time = time.time() - t0

    # Restore best weights for evaluation
    if best_state is not None:
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw.load_state_dict(best_state)

    # Evaluate IC on val and test
    val_ic = compute_ic(model, val_loader, device)
    test_ic = compute_ic(model, test_loader, device)

    # Save model checkpoint
    model_path = output_dir / f"{name}_fold{fold}.pt"
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "model_state_dict": raw.state_dict(),
        "config": config,
        "feature_cols": feature_cols,
        "n_params": n_params,
        "val_ic": val_ic,
        "test_ic": test_ic,
    }, model_path)

    return {
        "fold": fold,
        "n_params": n_params,
        "train_dataset": train_ds.summary(),
        "val_dataset": val_ds.summary(),
        "test_dataset": test_ds.summary(),
        "epochs_trained": len(train_losses),
        "best_val_loss": round(best_val_loss, 6),
        "train_losses": train_losses,    # full curve — one value per epoch
        "val_losses": val_losses,
        "val_ic": val_ic,
        "test_ic": test_ic,
        "training_time_s": round(training_time, 1),
        "model_path": str(model_path),
        "status": "completed",
    }
