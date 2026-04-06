"""Generate MC-dropout forecast fan charts for a list of tickers using the LSTM model.

Usage
-----
    python scripts/lstm_forecast_plot.py IBRX TOYO STRO PTGX
    python scripts/lstm_forecast_plot.py AAPL MSFT --lookback-months 12 --n-paths 10 --out forecast.png

For each ticker:
  - Historical close prices (last 6 months) shown as a thick black line
  - N MC-dropout forecast paths shown as thin colored lines extending from today
  - Forecast horizons: 1d, 5d, 10d, 21d, 63d (connected as a line per path)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parents[1]
_MODEL_DIR  = _REPO_ROOT / "models"
_DATASET    = _REPO_ROOT / "data" / "training" / "dataset.parquet"

LSTM_HORIZONS = [1, 5, 10, 21, 63]
FORECAST_COLORS = ["#4e8df5", "#f5934e", "#4ef574", "#f54e4e", "#b04ef5", "#f5e04e",
                   "#4ef5e8", "#f54eb0", "#9af54e", "#f5c84e"]


def load_model(model_path: Path):
    import torch
    from morningalpha.ml.lstm_model import StockPriceLSTM
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    cfg = dict(ckpt["config"])
    cfg["dropout"] = 0.3  # keep dropout active for MC sampling
    model = StockPriceLSTM.from_config(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train()  # dropout active for MC paths
    return model, ckpt


def build_sequence(ticker: str, ckpt: dict, ds: pd.DataFrame) -> np.ndarray | None:
    """Return [1, lookback, n_features] float32 array for the ticker, or None."""
    feat_cols   = ckpt["feature_cols"]
    lookback    = ckpt.get("lookback", 60)
    scaler_info = ckpt.get("feature_scaler", {})

    grp = ds[ds["ticker"] == ticker].sort_values("date")
    if len(grp) < lookback:
        print(f"  [!] {ticker}: only {len(grp)} rows in dataset (need {lookback}) — skipping")
        return None

    # Apply stored scaler
    if scaler_info and "cols" in scaler_info:
        sc_cols = [c for c in scaler_info["cols"] if c in grp.columns]
        full_cols = scaler_info["cols"]
        mean_map  = dict(zip(full_cols, scaler_info["mean"]))
        scale_map = dict(zip(full_cols, scaler_info["scale"]))
        grp = grp.copy()
        for col in sc_cols:
            grp[col] = (grp[col].fillna(0) - mean_map[col]) / max(scale_map[col], 1e-8)

    missing = [c for c in feat_cols if c not in grp.columns]
    for c in missing:
        grp = grp.copy()
        grp[c] = 0.0

    seq = grp[feat_cols].tail(lookback).values.astype(np.float32)
    seq = np.nan_to_num(seq, nan=0.0)
    return seq[np.newaxis]  # [1, lookback, F]


def fetch_history(ticker: str, months: int = 6) -> pd.Series | None:
    """Download recent close prices via yfinance."""
    try:
        import yfinance as yf
        period = f"{months}mo"
        raw = yf.download(ticker, period=period, interval="1d",
                          progress=False, auto_adjust=True)
        if raw.empty:
            return None
        if hasattr(raw.columns, "levels"):
            raw.columns = raw.columns.get_level_values(0)
        closes = raw["Close"].dropna()
        closes.index = pd.to_datetime(closes.index).tz_localize(None)
        return closes
    except Exception as exc:
        print(f"  [!] {ticker}: yfinance failed ({exc})")
        return None


def generate_forecast_dates(last_date: pd.Timestamp) -> list[pd.Timestamp]:
    """Convert LSTM horizon trading days to calendar dates using business day offset."""
    return [last_date + pd.tseries.offsets.BDay(h) for h in LSTM_HORIZONS]


def plot_ticker(ax, ticker: str, history: pd.Series,
                paths: np.ndarray, last_price: float):
    """Plot historical close + MC forecast paths on a single axes."""
    import matplotlib.dates as mdates

    # Historical price
    ax.plot(history.index, history.values, color="black", linewidth=1.8,
            label="true trend", zorder=10)

    last_date = history.index[-1]
    forecast_dates = [last_date] + generate_forecast_dates(last_date)

    # Each path: connect last_price → predicted prices at each horizon
    n_paths = paths.shape[0]
    for i in range(n_paths):
        # paths[i] are cumulative log-returns at each horizon
        # Convert to price levels: P_t = last_price * exp(log_ret)
        prices = [last_price] + [last_price * float(np.exp(paths[i, j]))
                                  for j in range(len(LSTM_HORIZONS))]
        color = FORECAST_COLORS[i % len(FORECAST_COLORS)]
        ax.plot(forecast_dates, prices, color=color, linewidth=1.0,
                alpha=0.85, label=f"forecast {i+1}")

    # Vertical line at forecast start
    ax.axvline(last_date, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_title(ticker, fontsize=11, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.tick_params(axis="x", rotation=25, labelsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#eaeaf2")


def main():
    parser = argparse.ArgumentParser(description="LSTM MC-dropout forecast fan chart")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols")
    parser.add_argument("--model", default=None, help="Path to .pt checkpoint")
    parser.add_argument("--lookback-months", type=int, default=6,
                        help="Months of history to show (default: 6)")
    parser.add_argument("--n-paths", type=int, default=6,
                        help="Number of MC dropout paths (default: 6)")
    parser.add_argument("--out", default=None,
                        help="Save plot to file instead of displaying")
    args = parser.parse_args()

    import torch
    import matplotlib
    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Resolve model path
    model_path = Path(args.model) if args.model else _MODEL_DIR / "lstm_clip_v1.pt"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model, ckpt = load_model(model_path)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)

    print(f"Loading dataset: {_DATASET}")
    if not _DATASET.exists():
        print(f"Dataset not found: {_DATASET}")
        sys.exit(1)
    ds = pd.read_parquet(_DATASET)

    tickers = [t.upper() for t in args.tickers]
    n = len(tickers)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 5 * nrows))
    fig.patch.set_facecolor("#f0f0f0")
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax_idx, ticker in enumerate(tickers):
        ax = axes_flat[ax_idx]
        print(f"\n{ticker}")

        # Fetch historical prices
        history = fetch_history(ticker, months=args.lookback_months)
        if history is None or history.empty:
            ax.set_title(f"{ticker} — no price data", fontsize=11)
            ax.set_visible(False)
            continue

        last_price = float(history.iloc[-1])
        print(f"  Last price: ${last_price:.2f}  ({len(history)} trading days of history)")

        # Build sequence for LSTM
        seq = build_sequence(ticker, ckpt, ds)
        if seq is None:
            ax.set_title(f"{ticker} — insufficient history in dataset", fontsize=11)
            continue

        # Generate MC dropout paths
        x = torch.tensor(seq).to(device)
        with torch.no_grad():
            paths = model.predict_paths(x, n_paths=args.n_paths).cpu().numpy()
        # paths: [n_paths, n_horizons]

        print(f"  63d forecast paths (log-ret): {paths[:, -1].round(3)}")
        print(f"  63d implied prices:           {[round(last_price * float(np.exp(p)), 2) for p in paths[:, -1]]}")

        plot_ticker(ax, ticker, history, paths, last_price)

    # Shared legend on first axis
    if axes_flat[0].lines:
        handles, labels = axes_flat[0].get_legend_handles_labels()
        axes_flat[0].legend(handles, labels, fontsize=7, loc="upper left",
                            framealpha=0.8)

    # Hide unused axes
    for i in range(len(tickers), len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(f"LSTM MC-Dropout Forecast — {', '.join(tickers)}", fontsize=13, y=1.01)
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"\nSaved → {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
