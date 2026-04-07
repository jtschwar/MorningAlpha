"""alpha ml train lstm — Train the StockPriceLSTM on the unified daily dataset.

Usage
-----
    alpha ml train lstm
    alpha ml train lstm --epochs 100 --hidden 512
    alpha ml train lstm --target-mode clip --name lstm_clip_v1
    alpha ml train lstm --target-mode rank --name lstm_rank_v1
    alpha ml train lstm --spike-threshold 0.5 --name lstm_spike_cls_v1

Target modes
------------
    log   (default) — log1p(clip(r, -0.99, 5.0)) — preserves spike direction,
                      compresses magnitude, upper-clipped to prevent extreme
                      outliers (964x → capped before log)
    clip  — raw returns clipped to ±2.0 — simple, interpretable scale
    rank  — cross-sectional rank targets (forward_Nd_rank columns already in
            dataset) — normalized to [-1, 1], immune to outliers; best for
            comparing relative outperformance across stocks

Spike classifier
----------------
    --spike-threshold 0.5  trains a binary classifier: did |forward_5d| > 0.5?
    Uses BCE loss and reports AUC instead of rank IC. Answers whether historical
    features contain pre-spike signal before investing in a full ensemble.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rich_click as click
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from torch.utils.data import DataLoader, Dataset

from morningalpha.ml.lstm_model import LSTM_HORIZONS, StockPriceLSTM
from morningalpha.ml.features import FEATURE_COLUMNS, MARKET_CONTEXT_COLUMNS
from morningalpha.ml.lstm_wfcv import (
    make_wfcv_folds, LSTMDateRangeDataset, make_ema_sampler,
)

logger = logging.getLogger(__name__)
console = Console()

_REPO_MODELS = Path(__file__).parents[2] / "models"
_HOME_MODELS = Path.home() / ".morningalpha" / "models"
MODEL_DIR = _REPO_MODELS if _REPO_MODELS.exists() else _HOME_MODELS

DATASET_PATH = Path("data/training/dataset.parquet")
DEFAULT_OUTPUT = MODEL_DIR / "lstm_price_v1.pt"

# Horizon columns that must be present as targets
TARGET_COLS = [f"forward_{h}d" for h in LSTM_HORIZONS]  # forward_1d … forward_63d

# Rank-target columns corresponding to each LSTM horizon (already in dataset)
RANK_TARGET_COLS = [f"forward_{h}d_rank" for h in LSTM_HORIZONS]

# Columns that are NOT cross-sectionally rank-normalized in the dataset
# (market context is constant per date; categoricals are ordinal integers).
# These need StandardScaler normalization before feeding to the LSTM.
COLS_TO_SCALE = MARKET_CONTEXT_COLUMNS + ["sector", "market_cap_cat"]


# ---------------------------------------------------------------------------
# Feature scaler
# ---------------------------------------------------------------------------

def fit_feature_scaler(
    df_train: pd.DataFrame,
    cols: List[str],
) -> Tuple[StandardScaler, List[str]]:
    """Fit a StandardScaler on train rows for columns that need it.

    Skips columns missing from df_train or with zero variance (constant).
    Returns (fitted_scaler, cols_actually_scaled).
    """
    present = [c for c in cols if c in df_train.columns]
    # Drop constant columns — StandardScaler would produce NaN for std=0
    non_constant = [c for c in present if df_train[c].std() > 0]
    dropped = set(present) - set(non_constant)
    if dropped:
        logger.info("Skipping constant columns from scaler: %s", dropped)

    scaler = StandardScaler()
    scaler.fit(df_train[non_constant].fillna(0).values)
    return scaler, non_constant


def apply_feature_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler,
    cols: List[str],
) -> pd.DataFrame:
    """Apply a fitted scaler to a DataFrame in-place (copy returned)."""
    df = df.copy()
    present = [c for c in cols if c in df.columns]
    df[present] = scaler.transform(df[present].fillna(0).values)
    return df


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LSTMSequenceDataset(Dataset):
    """Builds per-ticker time-ordered sequences from the unified parquet.

    Each sample: (x, y)
      x : float32 [lookback, n_features]  — the feature window
      y : float32 [n_horizons]            — targets at each horizon

    Target modes
    ------------
    log   : log1p(clip(r, -0.99, 5.0)) — default; compresses outliers
    clip  : raw returns clipped to ±2.0
    rank  : forward_Nd_rank columns (pre-normalized cross-sectional ranks)

    Spike mode (spike_threshold > 0)
    ---------------------------------
    Overrides target_mode. Binary target: 1 if |forward_5d| > threshold else 0.
    Only one output per sample (single horizon).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feat_cols: List[str],
        lookback: int = 60,
        stride: int = 3,
        split: str = "train",
        target_mode: str = "log",
        spike_threshold: float = 0.0,
    ) -> None:
        self.feat_cols = feat_cols
        self.lookback = lookback
        self.n_features = len(feat_cols)
        self.target_mode = target_mode
        self.spike_threshold = spike_threshold
        self.is_spike_mode = spike_threshold > 0.0

        split_df = df[df["split"] == split].copy()

        self._xs: List[np.ndarray] = []
        self._ys: List[np.ndarray] = []
        self._masks: List[np.ndarray] = []

        n_tickers = 0
        for ticker, grp in split_df.groupby("ticker"):
            grp = grp.sort_values("date")
            n = len(grp)
            if n < lookback + 1:
                continue

            X = grp[feat_cols].values.astype(np.float32)  # [T, F]

            if self.is_spike_mode:
                # Binary: 1 if |forward_5d| > threshold else 0
                fwd = grp["forward_5d"].values.astype(np.float32)
                Y = (np.abs(fwd) > spike_threshold).astype(np.float32).reshape(-1, 1)
            elif target_mode == "rank":
                # Use pre-computed cross-sectional rank columns
                rank_cols = [c for c in RANK_TARGET_COLS if c in grp.columns]
                Y = grp[rank_cols].values.astype(np.float32)
            elif target_mode == "combo":
                # Rank targets (cross-sectional ranking, [-1,1]) || clip targets (±2.0)
                # Concatenated → Y shape [T, 2*n_horizons]
                rank_cols = [c for c in RANK_TARGET_COLS if c in grp.columns]
                Y_rank = grp[rank_cols].values.astype(np.float32)
                Y_clip = np.clip(grp[TARGET_COLS].values.astype(np.float32), -2.0, 2.0)
                Y = np.concatenate([Y_rank, Y_clip], axis=1)
            elif target_mode == "clip":
                Y_raw = grp[TARGET_COLS].values.astype(np.float32)
                Y = np.clip(Y_raw, -2.0, 2.0)
            else:  # "log" — default
                Y_raw = grp[TARGET_COLS].values.astype(np.float32)
                Y = Y_raw.copy()
                # forward_1d is already a log return; Nd are simple returns → log1p
                # Clip upper end at 5.0 before log to prevent 964x outliers (log1p(5)≈1.79)
                Y[:, 1:] = np.log1p(np.clip(Y_raw[:, 1:], -0.99, 5.0))

            valid_windows = 0
            for start in range(0, n - lookback, stride):
                end = start + lookback
                x = X[start:end]       # [lookback, F]
                y = Y[end - 1]         # label at last step of window

                if np.any(np.isnan(x)):
                    continue
                # Allow partial targets — mask tracks which horizons are available
                target_mask = ~np.isnan(y)
                if not target_mask.any():
                    continue
                y = np.where(target_mask, y, 0.0)  # zero-fill NaN (masked out in loss)

                self._xs.append(x)
                self._ys.append(y)
                self._masks.append(target_mask)
                valid_windows += 1

            if valid_windows > 0:
                n_tickers += 1

        logger.info(
            "LSTMSequenceDataset [%s, mode=%s]: %d samples from %d tickers",
            split, "spike" if self.is_spike_mode else target_mode,
            len(self._xs), n_tickers,
        )

    def __len__(self) -> int:
        return len(self._xs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self._xs[idx]),
            torch.from_numpy(self._ys[idx]),
            torch.from_numpy(self._masks[idx]),
        )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        console.print("[dim]Using Apple MPS[/dim]")
        return torch.device("mps")
    if torch.cuda.is_available():
        console.print(f"[dim]Using CUDA: {torch.cuda.get_device_name(0)}[/dim]")
        return torch.device("cuda")
    console.print("[dim]Using CPU[/dim]")
    return torch.device("cpu")


def _rank_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """Spearman rank IC between flat arrays, ignoring NaN."""
    mask = ~(np.isnan(pred) | np.isnan(actual))
    if mask.sum() < 10:
        return float("nan")
    return float(spearmanr(pred[mask], actual[mask]).correlation)


class MaskedHuberLoss(nn.Module):
    """Drop-in for nn.HuberLoss that accepts an optional bool mask tensor."""
    def __init__(self, delta: float = 0.05):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return _masked_huber(pred, target, mask, self.delta)


def _masked_huber(pred: torch.Tensor, target: torch.Tensor,
                   mask: Optional[torch.Tensor], delta: float = 0.05) -> torch.Tensor:
    """Huber loss that ignores positions where mask is False (unresolved targets)."""
    err = target - pred
    loss = torch.where(err.abs() <= delta,
                       0.5 * err ** 2,
                       delta * (err.abs() - 0.5 * delta))
    if mask is not None:
        m = mask.float()
        denom = m.sum()
        if denom == 0:
            return loss.mean()
        return (loss * m).sum() / denom
    return loss.mean()


class ComboLoss(nn.Module):
    """Weighted sum of Huber losses on rank and clip output halves.

    pred / target shape: [B, 2*n_horizons]
    First half  → rank predictions vs rank targets
    Second half → clip predictions vs clip targets
    mask (optional): [B, 2*n_horizons] bool — False where target is unresolved
    """
    def __init__(self, n_horizons: int, rank_weight: float = 0.5, delta: float = 0.05):
        super().__init__()
        self.n = n_horizons
        self.rank_w = rank_weight
        self.clip_w = 1.0 - rank_weight
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        rank_mask = mask[:, :self.n] if mask is not None else None
        clip_mask = mask[:, self.n:] if mask is not None else None
        rank_loss = _masked_huber(pred[:, :self.n], target[:, :self.n], rank_mask, self.delta)
        clip_loss = _masked_huber(pred[:, self.n:], target[:, self.n:], clip_mask, self.delta)
        return self.rank_w * rank_loss + self.clip_w * clip_loss


def _evaluate(
    model: StockPriceLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_spike_mode: bool = False,
    is_combo_mode: bool = False,
) -> Tuple[float, List[float]]:
    """Return (mean_loss, [IC per horizon]) or (mean_loss, [AUC]) for spike mode."""
    model.eval()
    total_loss = 0.0
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            x, y, mask = batch if len(batch) == 3 else (*batch, None)
            x, y = x.to(device), y.to(device)
            mask = mask.to(device) if mask is not None else None
            pred = model(x)
            total_loss += criterion(pred, y, mask).item() * len(x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)    # [N, n_out]
    targets = np.concatenate(all_targets)

    if is_spike_mode:
        try:
            probs = 1.0 / (1.0 + np.exp(-preds[:, 0]))
            auc = float(roc_auc_score(targets[:, 0].astype(int), probs))
        except ValueError:
            auc = float("nan")
        metrics = [auc]
    elif is_combo_mode:
        # IC on rank half only (first n_horizons outputs vs first n_horizons targets)
        n = preds.shape[1] // 2
        metrics = [_rank_ic(preds[:, i], targets[:, i]) for i in range(n)]
    else:
        metrics = [_rank_ic(preds[:, i], targets[:, i]) for i in range(preds.shape[1])]

    return total_loss / len(loader.dataset), metrics


# ---------------------------------------------------------------------------
# Training loop (shared by WFCV folds and final model)
# ---------------------------------------------------------------------------

def _run_training_loop(
    model: StockPriceLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    patience: int = 10,
    criterion: Optional[nn.Module] = None,
    label: str = "Training",
    is_combo_mode: bool = False,
) -> Tuple[Optional[dict], float, List[dict]]:
    """Train with early stopping. Returns (best_state_dict, best_val_loss, history)."""
    if criterion is None:
        criterion = MaskedHuberLoss(delta=0.05)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_loss = float("inf")
    best_state: Optional[dict] = None
    patience_count = 0
    history: List[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(label, total=epochs)

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                x, y, mask = batch if len(batch) == 3 else (*batch, None)
                x, y = x.to(device), y.to(device)
                mask = mask.to(device) if mask is not None else None
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y, mask)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(x)
            train_loss /= len(train_loader.dataset)

            val_loss, val_metrics = _evaluate(model, val_loader, criterion, device, is_combo_mode=is_combo_mode)
            scheduler.step()

            mean_ic = float(np.nanmean(val_metrics))
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_ic": mean_ic})

            progress.advance(task)
            progress.update(
                task,
                description=f"{label}  ep={epoch}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}  IC={mean_ic:.4f}",
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    console.print(f"[yellow]  Early stopping at epoch {epoch}[/yellow]")
                    break

    return best_state, best_val_loss, history


# ---------------------------------------------------------------------------
# Walk-forward fine-tuning helpers
# ---------------------------------------------------------------------------

def _compute_wf_anchors(
    dataset_start: pd.Timestamp,
    dataset_end: pd.Timestamp,
    anchor_spacing_days: int = 63,
    min_train_days: int = 252,
    val_days: int = 90,
    embargo_days: int = 10,
) -> List[pd.Timestamp]:
    """Return quarterly anchor dates for walk-forward fine-tuning.

    Each anchor defines one fine-tuning step:
      train = dataset_start → (anchor - val_days - 2*embargo_days)
      val   = (anchor - val_days - embargo_days) → (anchor - embargo_days)

    The final anchor is always dataset_end so the model is trained through present.
    """
    # Earliest viable anchor: enough history for initial training + val window
    min_anchor = dataset_start + pd.Timedelta(days=min_train_days + val_days + 2 * embargo_days)
    anchors: List[pd.Timestamp] = []
    current = min_anchor
    while current < dataset_end:
        anchors.append(current)
        current += pd.Timedelta(days=anchor_spacing_days)
    if not anchors or anchors[-1] < dataset_end:
        anchors.append(dataset_end)
    return anchors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command("train-lstm")
@click.option("--dataset", "dataset_path", default=str(DATASET_PATH), show_default=True)
@click.option("--output", default=str(DEFAULT_OUTPUT), show_default=True)
@click.option("--lookback", default=60, show_default=True, help="Input sequence length (trading days).")
@click.option("--hidden", default=256, show_default=True, help="LSTM hidden dimension.")
@click.option("--layers", default=2, show_default=True, help="Number of LSTM layers.")
@click.option("--dropout", default=0.3, show_default=True, help="Dropout rate (also used for MC inference).")
@click.option("--epochs", default=50, show_default=True)
@click.option("--batch-size", "batch_size", default=512, show_default=True)
@click.option("--lr", default=1e-3, show_default=True)
@click.option("--patience", default=10, show_default=True, help="Early stopping patience (epochs).")
@click.option("--stride", default=3, show_default=True, help="Step between training windows per ticker.")
@click.option("--workers", default=0, show_default=True, help="DataLoader num_workers.")
@click.option(
    "--target-mode", "target_mode",
    default="log",
    show_default=True,
    type=click.Choice(["log", "clip", "rank", "combo"]),
    help=(
        "Target encoding: log=log1p(clip(r,-0.99,5)) [default], "
        "clip=raw returns ±2.0, rank=cross-sectional rank [-1,1], "
        "combo=rank+clip dual-head (rank fixes bias, clip preserves magnitude for fan chart)."
    ),
)
@click.option(
    "--spike-threshold", "spike_threshold",
    default=0.0,
    show_default=True,
    type=float,
    help=(
        "When >0, trains a binary spike classifier: target=1 if |forward_5d|>threshold. "
        "Overrides --target-mode. Reports AUC instead of rank IC. "
        "Example: --spike-threshold 0.5 for >50%% 5-day moves."
    ),
)
@click.option("--name", default=None, help="Model name suffix for checkpoint (e.g. 'clip_v1' → lstm_clip_v1.pt).")
@click.option("--walk-forward/--no-walk-forward", "walk_forward", default=True, show_default=True,
              help="Use expanding-window WFCV for evaluation then train final model on full window.")
@click.option("--n-folds", "n_folds", default=6, show_default=True,
              help="Number of WFCV folds (walk-forward mode only).")
@click.option("--ema-halflife", "ema_halflife", default=180, show_default=True,
              help="EMA sample weighting half-life in days (0 = uniform). Ablation best: 180.")
@click.option("--wf-finetune/--no-wf-finetune", "wf_finetune", default=False, show_default=True,
              help="Walk-forward fine-tuning for final model: sequentially warm-start through quarterly anchors "
                   "up to dataset_end. More compute but ensures model has seen the most recent regime.")
@click.option("--anchor-spacing", "anchor_spacing", default=63, show_default=True,
              help="Trading days between walk-forward fine-tuning anchors (~quarterly).")
@click.option("--finetune-lr-decay", "finetune_lr_decay", default=0.5, show_default=True,
              help="LR multiplier applied at each anchor step during walk-forward fine-tuning.")
def train_lstm(
    dataset_path: str,
    output: str,
    lookback: int,
    hidden: int,
    layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    stride: int,
    workers: int,
    target_mode: str,
    spike_threshold: float,
    name: Optional[str],
    walk_forward: bool,
    n_folds: int,
    ema_halflife: int,
    patience: int,
    wf_finetune: bool,
    anchor_spacing: int,
    finetune_lr_decay: float,
) -> None:
    """Train the StockPriceLSTM on the unified daily dataset.

    \b
    Examples:
      alpha ml train lstm --target-mode clip --name clip_v1   # recommended (ablation winner)
      alpha ml train lstm --target-mode log --name log_v1
      alpha ml train lstm --no-walk-forward --name simple_v1  # simple split, faster
      alpha ml train lstm --spike-threshold 0.5 --name spike_cls_v1
      alpha ml train lstm --epochs 100 --hidden 256
    """
    is_spike_mode = spike_threshold > 0.0
    is_combo_mode = target_mode == "combo" and not is_spike_mode
    use_wfcv = walk_forward and not is_spike_mode  # WFCV not applicable to spike classifier

    # Resolve output path
    if name:
        output = str(MODEL_DIR / f"lstm_{name}.pt")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ema_label = f"ema={ema_halflife}d" if ema_halflife > 0 else "ema=none"
    mode_label = f"spike(>{spike_threshold})" if is_spike_mode else target_mode
    wfcv_label = f"walk-forward ({n_folds} folds, {ema_label})" if use_wfcv else "simple split"
    console.print(f"[bold cyan]LSTM training — target={mode_label}  split={wfcv_label}[/bold cyan]")

    # --- Load dataset ---
    console.print(f"[bold cyan]Loading dataset: {dataset_path}[/bold cyan]")
    df = pd.read_parquet(dataset_path)
    df["date"] = pd.to_datetime(df["date"])

    console.print(
        f"  Total rows: [bold]{len(df):,}[/bold]  "
        f"Dates: {df['date'].min().date()} → {df['date'].max().date()}  "
        f"Tickers: {df['ticker'].nunique():,}"
    )

    # Check for daily data
    # Detect daily data by median date gap across a sample of tickers, not by
    # is_anchor (which is True for all rows when built with --no-overlap, even
    # on daily datasets).
    sample_ticker = df["ticker"].iloc[0]
    sample_gaps = df[df["ticker"] == sample_ticker].sort_values("date")["date"].diff().dt.days.dropna()
    is_daily = sample_gaps.median() <= 2
    if not is_daily:
        console.print(
            "[yellow]Dataset appears to be weekly/staggered (median gap "
            f"{sample_gaps.median():.0f} days). "
            "Rebuild with --snapshot-freq daily for best LSTM performance.[/yellow]"
        )

    # Ensure forward_1d is present
    if "forward_1d" not in df.columns:
        console.print(
            "[yellow]forward_1d not found — computing from forward_5d as proxy. "
            "Rebuild dataset with --snapshot-freq daily for accurate 1d targets.[/yellow]"
        )
        df["forward_1d"] = df.get("forward_5d", np.nan)

    # --- Feature columns ---
    feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    # Drop zero-variance features — constant cols add noise and break StandardScaler
    df_train = df[df["split"] == "train"]
    zero_var = [c for c in feat_cols if df_train[c].std() == 0]
    if zero_var:
        console.print(f"  [yellow]Dropping {len(zero_var)} constant feature(s): {zero_var}[/yellow]")
        feat_cols = [c for c in feat_cols if c not in zero_var]

    console.print(f"  Features: [bold]{len(feat_cols)}[/bold]")

    # --- Feature scaling ---
    # MARKET_CONTEXT_COLUMNS + categoricals are not rank-normalized in the dataset.
    # Fit StandardScaler on train split, apply to all splits.
    scale_cols = [c for c in COLS_TO_SCALE if c in feat_cols]
    console.print(f"  Scaling {len(scale_cols)} market-context/categorical columns with StandardScaler")
    scaler, scaled_cols = fit_feature_scaler(df_train, scale_cols)
    df = apply_feature_scaler(df, scaler, scaled_cols)

    # --- Validate rank target columns for rank/combo mode ---
    if not is_spike_mode and target_mode in ("rank", "combo"):
        missing_rank = [c for c in RANK_TARGET_COLS if c not in df.columns]
        if missing_rank:
            console.print(f"[bold red]Rank target columns missing: {missing_rank}[/bold red]")
            raise SystemExit(1)
    if is_combo_mode:
        console.print(f"  Targets: rank={RANK_TARGET_COLS} + clip={TARGET_COLS}  (mode=combo)")
    elif target_mode == "rank":
        console.print(f"  Targets: {RANK_TARGET_COLS}")
    elif is_spike_mode:
        console.print(f"  Target: binary spike — |forward_5d| > {spike_threshold}")
    else:
        console.print(f"  Targets: {TARGET_COLS}  (mode={target_mode})")

    horizon_days = [5] if is_spike_mode else LSTM_HORIZONS
    device = _select_device()
    criterion: nn.Module = (
        ComboLoss(n_horizons=len(horizon_days), rank_weight=0.5)
        if is_combo_mode else MaskedHuberLoss(delta=0.05)
    )

    ds_range_kwargs = dict(lookback=lookback, stride=stride, target_mode=target_mode)
    ds_split_kwargs = dict(feat_cols=feat_cols, lookback=lookback, stride=stride,
                           target_mode=target_mode, spike_threshold=spike_threshold)

    def _make_model() -> StockPriceLSTM:
        return StockPriceLSTM(
            n_features=len(feat_cols),
            hidden_dim=hidden,
            num_layers=layers,
            dropout=dropout,
            horizon_days=horizon_days,
            combo=is_combo_mode,
        ).to(device)

    def _make_loader(ds, shuffle: bool, sampler=None):
        if sampler:
            return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=workers, pin_memory=True)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True)

    halflife = ema_halflife if ema_halflife > 0 else None
    wfcv_fold_ics: List[dict] = []
    best_state: Optional[dict] = None
    best_val_loss = float("inf")
    history: List[dict] = []
    t0 = time.time()

    if use_wfcv:
        # --- Walk-forward CV evaluation + final model ---
        console.print("\n[bold cyan]Building walk-forward folds...[/bold cyan]")
        folds = make_wfcv_folds(df, n_folds=n_folds, embargo_days=10)
        console.print(f"  {len(folds)} folds  ({str(folds[0]['train_start'])[:10]} → {str(folds[-1]['val_end'])[:10]})")

        for fold in folds:
            console.print(
                f"\n[bold]Fold {fold['fold']}/{len(folds)}[/bold]  "
                f"train → {str(fold['train_end'])[:10]}  "
                f"val {str(fold['val_start'])[:10]} → {str(fold['val_end'])[:10]}"
            )
            tr_ds = LSTMDateRangeDataset(df, feat_cols, fold["train_start"], fold["train_end"], **ds_range_kwargs)
            va_ds = LSTMDateRangeDataset(df, feat_cols, fold["val_start"],   fold["val_end"],   **ds_range_kwargs)

            if len(tr_ds) == 0 or len(va_ds) == 0:
                console.print("  [yellow]Empty fold — skipping[/yellow]")
                continue

            console.print(f"  train={len(tr_ds):,}  val={len(va_ds):,}")
            sampler = make_ema_sampler(tr_ds, halflife)
            tr_loader = _make_loader(tr_ds, shuffle=True,  sampler=sampler)
            va_loader = _make_loader(va_ds, shuffle=False)

            fold_model = _make_model()
            fold_state, _, _ = _run_training_loop(
                fold_model, tr_loader, va_loader, epochs, lr, device,
                patience=patience, criterion=criterion, label=f"Fold {fold['fold']}",
                is_combo_mode=is_combo_mode,
            )
            if fold_state:
                fold_model.load_state_dict(fold_state)
            _, fold_metrics = _evaluate(fold_model, va_loader, criterion, device, is_combo_mode=is_combo_mode)
            ics = {f"ic_{h}d": round(m, 4) for h, m in zip(LSTM_HORIZONS, fold_metrics)}
            wfcv_fold_ics.append(ics)
            console.print(
                f"  IC-5d={ics.get('ic_5d', float('nan')):.4f}  "
                f"IC-63d={ics.get('ic_63d', float('nan')):.4f}"
            )

        # WFCV summary table
        if wfcv_fold_ics:
            wf_table = Table(title="WFCV Summary", show_header=True)
            wf_table.add_column("Metric")
            wf_table.add_column("Value", style="bold")
            for h in LSTM_HORIZONS:
                vals = [f[f"ic_{h}d"] for f in wfcv_fold_ics if not np.isnan(f.get(f"ic_{h}d", float("nan")))]
                if vals:
                    wf_table.add_row(f"Mean IC-{h}d", f"{np.mean(vals):+.4f}  (std {np.std(vals):.4f})")
            console.print(wf_table)

        dataset_end   = df["date"].max()
        embargo_td    = pd.Timedelta(days=10)
        val_td        = pd.Timedelta(days=90)

        if wf_finetune:
            # Walk-forward fine-tuning: sequentially warm-start through quarterly
            # anchors up to dataset_end so the final model has seen the full regime.
            anchors = _compute_wf_anchors(
                df["date"].min(), dataset_end,
                anchor_spacing_days=anchor_spacing,
            )
            console.print(
                f"\n[bold cyan]Walk-forward fine-tuning — {len(anchors)} anchors  "
                f"({str(anchors[0])[:10]} → {str(anchors[-1])[:10]})[/bold cyan]"
            )

            model     = _make_model()
            # Use a fixed reduced LR for all anchor steps — do NOT compound the
            # decay per anchor.  With 50+ anchors, 0.5x/step → lr ≈ 1e-19 by
            # the end, which is effectively zero and kills learning.
            cur_lr    = lr * finetune_lr_decay
            wf_state: Optional[dict] = None

            for i, anchor in enumerate(anchors):
                is_final_anchor = (i == len(anchors) - 1)

                # val window: 90 days ending (anchor - embargo)
                anc_val_end   = anchor   - embargo_td
                anc_val_start = anc_val_end - val_td
                # train window: start → val_start - embargo (no target overlap)
                anc_train_end = anc_val_start - embargo_td

                tr_ds = LSTMDateRangeDataset(
                    df, feat_cols, df["date"].min(), anc_train_end, **ds_range_kwargs
                )
                va_ds = LSTMDateRangeDataset(
                    df, feat_cols, anc_val_start, anc_val_end, **ds_range_kwargs
                )

                if len(tr_ds) == 0 or len(va_ds) == 0:
                    console.print(f"  [yellow]Anchor {i+1} empty — skipping[/yellow]")
                    continue

                # Warm-start from previous checkpoint (or fresh for anchor 0)
                if wf_state is not None:
                    model.load_state_dict(wf_state)

                console.print(
                    f"\n[bold]Anchor {i+1}/{len(anchors)}[/bold]  "
                    f"train → {str(anc_train_end)[:10]}  "
                    f"val {str(anc_val_start)[:10]} → {str(anc_val_end)[:10]}  "
                    f"lr={cur_lr:.2e}"
                )
                console.print(f"  train={len(tr_ds):,}  val={len(va_ds):,}")

                sampler    = make_ema_sampler(tr_ds, halflife)
                tr_loader  = _make_loader(tr_ds, shuffle=True, sampler=sampler)
                va_loader  = _make_loader(va_ds, shuffle=False)

                wf_state, anchor_val_loss, anchor_history = _run_training_loop(
                    model, tr_loader, va_loader, epochs, cur_lr, device,
                    patience=patience, criterion=criterion,
                    label=f"Anchor {i+1}/{len(anchors)}",
                    is_combo_mode=is_combo_mode,
                )
                history.extend(anchor_history)
                if wf_state:
                    best_val_loss = min(best_val_loss, anchor_val_loss)

                # LR is fixed across all anchors (set once above as lr * finetune_lr_decay)

            # Final anchor: extend training through dataset_end with masked loss.
            # Uses the most recent 90d as val for early stopping.
            if wf_state:
                model.load_state_dict(wf_state)
            final_val_end   = dataset_end
            final_val_start = dataset_end - val_td
            final_train_end = final_val_start - embargo_td

            console.print(
                f"\n[bold cyan]Final anchor — train through present  "
                f"val {str(final_val_start)[:10]} → {str(final_val_end)[:10]}[/bold cyan]"
            )
            train_ds = LSTMDateRangeDataset(
                df, feat_cols, df["date"].min(), final_train_end, **ds_range_kwargs
            )
            val_ds = LSTMDateRangeDataset(
                df, feat_cols, final_val_start, final_val_end, **ds_range_kwargs
            )

        else:
            # Simple: train on full window up to 90 days before end, val on last 90 days.
            val_cutoff = dataset_end - val_td
            console.print(
                f"\n[bold cyan]Final model — full window → {str(dataset_end)[:10]}  "
                f"(val from {str(val_cutoff)[:10]})[/bold cyan]"
            )
            train_ds = LSTMDateRangeDataset(
                df, feat_cols, df["date"].min(), val_cutoff, **ds_range_kwargs
            )
            val_ds = LSTMDateRangeDataset(
                df, feat_cols, val_cutoff, dataset_end, **ds_range_kwargs
            )

        console.print(f"  train={len(train_ds):,}  val={len(val_ds):,}")
        sampler      = make_ema_sampler(train_ds, halflife)
        train_loader = _make_loader(train_ds, shuffle=True,  sampler=sampler)
        val_loader   = _make_loader(val_ds,   shuffle=False)
        test_loader  = None  # val_end is the end of the dataset in WFCV mode

    else:
        # --- Simple train/val/test split ---
        console.print("\n[bold cyan]Building sequence datasets...[/bold cyan]")
        train_ds = LSTMSequenceDataset(df, split="train", **ds_split_kwargs)
        val_ds   = LSTMSequenceDataset(df, split="val",   **ds_split_kwargs)
        test_ds  = LSTMSequenceDataset(df, split="test",  **ds_split_kwargs)
        console.print(f"  train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")

        if len(train_ds) == 0:
            console.print("[bold red]No training samples. Check dataset splits.[/bold red]")
            raise SystemExit(1)

        if is_spike_mode:
            n_pos = sum(train_ds._ys[i][0] for i in range(len(train_ds)))
            n_neg = len(train_ds) - n_pos
            pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            console.print(f"  Spike prevalence: {n_pos/len(train_ds)*100:.1f}%  pos_weight={pos_weight.item():.2f}")

        train_loader = _make_loader(train_ds, shuffle=True)
        val_loader   = _make_loader(val_ds,   shuffle=False)
        test_loader  = _make_loader(test_ds,  shuffle=False)

    # --- Train final / only model ---
    # In wf_finetune mode the model is already warm from the anchor loop above;
    # otherwise initialise fresh here.
    if not (use_wfcv and wf_finetune):
        model = _make_model()
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"\n[bold]StockPriceLSTM[/bold]: {n_params:,} parameters  horizons={horizon_days}")
    final_lr = (lr * finetune_lr_decay) if (use_wfcv and wf_finetune) else lr
    console.print(f"[bold cyan]Training final model for up to {epochs} epochs...  lr={final_lr:.2e}[/bold cyan]")

    best_state, best_val_loss, history = _run_training_loop(
        model, train_loader, val_loader, epochs, final_lr, device,
        patience=patience, criterion=criterion, label="Final model",
        is_combo_mode=is_combo_mode,
    )

    elapsed = time.time() - t0
    console.print(f"Done in {elapsed:.0f}s. Best val loss: {best_val_loss:.6f}")

    if best_state:
        model.load_state_dict(best_state)

    # --- Evaluate on test set (simple split only) ---
    if test_loader is not None:
        _, test_metrics = _evaluate(model, test_loader, criterion, device, is_spike_mode, is_combo_mode=is_combo_mode)
        if is_spike_mode:
            table = Table(title="Test AUC — spike classifier", show_header=True)
            table.add_column("Horizon"); table.add_column("AUC", justify="right")
            table.add_row("5d spike", f"{test_metrics[0]:.4f}" if not np.isnan(test_metrics[0]) else "n/a")
        else:
            table = Table(title=f"Test IC (mode={target_mode})", show_header=True)
            table.add_column("Horizon"); table.add_column("IC", justify="right")
            for h, ic in zip(horizon_days, test_metrics):
                table.add_row(f"{h}d", f"{ic:.4f}" if not np.isnan(ic) else "n/a")
        console.print(table)
        final_test_metrics = test_metrics
    else:
        final_test_metrics = []

    # --- Save checkpoint ---
    checkpoint = {
        "model_state_dict": best_state or model.state_dict(),
        "config": model.config(),
        "feature_cols": feat_cols,
        "horizon_days": horizon_days,
        "lookback": lookback,
        "target_mode": "spike" if is_spike_mode else target_mode,
        "spike_threshold": spike_threshold,
        "val_loss": best_val_loss,
        "test_metrics": {
            str(h): round(float(m), 4)
            for h, m in zip(horizon_days, final_test_metrics)
        } if final_test_metrics else {},
        "wfcv": {
            "n_folds": len(wfcv_fold_ics),
            "ema_halflife": ema_halflife,
            "fold_ics": wfcv_fold_ics,
            "mean_ic": {
                f"ic_{h}d": round(float(np.nanmean([f.get(f"ic_{h}d", float("nan")) for f in wfcv_fold_ics])), 4)
                for h in LSTM_HORIZONS
            } if wfcv_fold_ics else {},
        } if use_wfcv else None,
        "history": history,
        "feature_scaler": {
            "cols": scaled_cols,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
    }
    torch.save(checkpoint, output_path)
    console.print(f"\n[bold green]Checkpoint saved → {output_path}[/bold green]")
