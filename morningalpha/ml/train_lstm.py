"""alpha ml train-lstm — Train the StockPriceLSTM on the unified daily dataset.

Usage
-----
    alpha ml train-lstm
    alpha ml train-lstm --epochs 100 --hidden 512
    alpha ml train-lstm --dataset data/training/dataset.parquet --output models/lstm_price_v1.pt

The dataset must have been built with --snapshot-freq daily (includes is_anchor,
forward_1d, and daily rows for all tickers). LGBM-only datasets (weekly/staggered)
still work but the LSTM will have far fewer sequence steps per ticker.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import rich_click as click
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from torch.utils.data import DataLoader, Dataset

from morningalpha.ml.lstm_model import LSTM_HORIZONS, StockPriceLSTM
from morningalpha.ml.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)
console = Console()

_REPO_MODELS = Path(__file__).parents[2] / "models"
_HOME_MODELS = Path.home() / ".morningalpha" / "models"
MODEL_DIR = _REPO_MODELS if _REPO_MODELS.exists() else _HOME_MODELS

DATASET_PATH = Path("data/training/dataset.parquet")
DEFAULT_OUTPUT = MODEL_DIR / "lstm_price_v1.pt"

# Horizon columns that must be present as targets
TARGET_COLS = [f"forward_{h}d" for h in LSTM_HORIZONS]  # forward_1d … forward_63d


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LSTMSequenceDataset(Dataset):
    """Builds per-ticker time-ordered sequences from the unified parquet.

    Each sample: (x, y)
      x : float32 [lookback, n_features]  — the feature window
      y : float32 [n_horizons]            — log-return targets at each horizon

    Only uses rows where all targets are non-NaN and the ticker has enough
    consecutive dates to fill a full lookback window.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feat_cols: List[str],
        lookback: int = 60,
        stride: int = 3,
        split: str = "train",
    ) -> None:
        self.feat_cols = feat_cols
        self.lookback = lookback
        self.n_features = len(feat_cols)

        split_df = df[df["split"] == split].copy()

        self._xs: List[np.ndarray] = []
        self._ys: List[np.ndarray] = []

        n_tickers = 0
        for ticker, grp in split_df.groupby("ticker"):
            grp = grp.sort_values("date")

            # Feature and target arrays for this ticker
            X = grp[feat_cols].values.astype(np.float32)      # [T, F]
            # forward_1d is a log return; the others are simple returns — convert all to log
            Y_raw = grp[TARGET_COLS].values.astype(np.float32) # [T, n_horizons]
            # forward_1d already log; forward_Xd are simple returns → log(1+r)
            Y = Y_raw.copy()
            Y[:, 1:] = np.log1p(np.clip(Y_raw[:, 1:], -0.99, None))

            n = len(grp)
            if n < lookback + 1:
                continue

            valid_windows = 0
            for start in range(0, n - lookback, stride):
                end = start + lookback
                x = X[start:end]                   # [lookback, F]
                y = Y[end - 1]                      # label at last step of window

                if np.any(np.isnan(x)) or np.any(np.isnan(y)):
                    continue

                self._xs.append(x)
                self._ys.append(y)
                valid_windows += 1

            if valid_windows > 0:
                n_tickers += 1

        logger.info(
            "LSTMSequenceDataset [%s]: %d samples from %d tickers",
            split, len(self._xs), n_tickers,
        )

    def __len__(self) -> int:
        return len(self._xs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self._xs[idx]),
            torch.from_numpy(self._ys[idx]),
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


def _evaluate(
    model: StockPriceLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, List[float]]:
    """Return (mean_loss, [IC per horizon])."""
    model.eval()
    total_loss = 0.0
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += criterion(pred, y).item() * len(x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)    # [N, n_horizons]
    targets = np.concatenate(all_targets)

    ics = [_rank_ic(preds[:, i], targets[:, i]) for i in range(len(LSTM_HORIZONS))]
    return total_loss / len(loader.dataset), ics


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
@click.option("--stride", default=3, show_default=True, help="Step between training windows per ticker.")
@click.option("--workers", default=0, show_default=True, help="DataLoader num_workers.")
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
) -> None:
    """Train the StockPriceLSTM on the unified daily dataset.

    \b
    Examples:
      alpha ml train-lstm
      alpha ml train-lstm --epochs 100 --hidden 512
      alpha ml train-lstm --dataset data/training/dataset.parquet
    """
    device = _select_device()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    is_daily = "is_anchor" in df.columns and (df["is_anchor"] == False).any()
    if not is_daily:
        console.print(
            "[yellow]Dataset appears to be weekly/staggered (no daily rows). "
            "Rebuild with --snapshot-freq daily for best LSTM performance.[/yellow]"
        )

    # Ensure forward_1d is present
    if "forward_1d" not in df.columns:
        console.print(
            "[yellow]forward_1d not found — computing from forward_5d as proxy. "
            "Rebuild dataset with --snapshot-freq daily for accurate 1d targets.[/yellow]"
        )
        df["forward_1d"] = df.get("forward_5d", np.nan)

    # Feature columns: same as LGBM (already rank-normalised in the parquet)
    feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    console.print(f"  Features: [bold]{len(feat_cols)}[/bold]")
    console.print(f"  Targets:  {TARGET_COLS}")

    # --- Build datasets ---
    console.print("\n[bold cyan]Building sequence datasets...[/bold cyan]")
    train_ds = LSTMSequenceDataset(df, feat_cols, lookback=lookback, stride=stride, split="train")
    val_ds   = LSTMSequenceDataset(df, feat_cols, lookback=lookback, stride=stride, split="val")
    test_ds  = LSTMSequenceDataset(df, feat_cols, lookback=lookback, stride=stride, split="test")

    console.print(
        f"  train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}"
    )

    if len(train_ds) == 0:
        console.print("[bold red]No training samples. Check dataset splits.[/bold red]")
        raise SystemExit(1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    # --- Build model ---
    model = StockPriceLSTM(
        n_features=len(feat_cols),
        hidden_dim=hidden,
        num_layers=layers,
        dropout=dropout,
        horizon_days=LSTM_HORIZONS,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"\n[bold]StockPriceLSTM[/bold]: {n_params:,} parameters")

    criterion = nn.HuberLoss(delta=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # --- Training loop ---
    console.print(f"\n[bold cyan]Training for {epochs} epochs...[/bold cyan]")
    best_val_loss = float("inf")
    best_state: Optional[dict] = None
    patience_count = 0
    patience = 10

    history = []
    t0 = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Epochs", total=epochs)

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(x)
            train_loss /= len(train_loader.dataset)

            val_loss, val_ics = _evaluate(model, val_loader, criterion, device)
            scheduler.step()

            mean_val_ic = float(np.nanmean(val_ics))
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_ic": mean_val_ic})

            progress.advance(task)
            progress.update(
                task,
                description=f"Epoch {epoch}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}  IC={mean_val_ic:.4f}",
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    console.print(f"[yellow]Early stopping at epoch {epoch}[/yellow]")
                    break

    elapsed = time.time() - t0
    console.print(f"Training complete in {elapsed:.0f}s. Best val loss: {best_val_loss:.6f}")

    # --- Evaluate best checkpoint on test set ---
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    _, test_ics = _evaluate(model, test_loader, criterion, device)

    table = Table(title="Test IC per horizon", show_header=True)
    table.add_column("Horizon")
    table.add_column("IC", justify="right")
    for h, ic in zip(LSTM_HORIZONS, test_ics):
        table.add_row(f"{h}d", f"{ic:.4f}" if not np.isnan(ic) else "n/a")
    console.print(table)

    # --- Save checkpoint ---
    checkpoint = {
        "model_state_dict": best_state or model.state_dict(),
        "config": model.config(),
        "feature_cols": feat_cols,
        "horizon_days": LSTM_HORIZONS,
        "lookback": lookback,
        "val_loss": best_val_loss,
        "test_ics": {str(h): round(float(ic), 4) for h, ic in zip(LSTM_HORIZONS, test_ics)},
        "history": history,
    }
    torch.save(checkpoint, output_path)
    console.print(f"\n[bold green]Checkpoint saved → {output_path}[/bold green]")
