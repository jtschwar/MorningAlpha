"""LSTM model for individual stock price path prediction.

Architecture
------------
- Learned input gate (soft feature selector)
- 2-layer LSTM backbone with inter-layer dropout
- MC Dropout head: call predict_paths() at inference to get N stochastic paths
- Multi-task output: [forward_1d, forward_5d, forward_10d, forward_21d, forward_63d]

Usage
-----
    model = StockPriceLSTM(n_features=77, horizon_days=[1, 5, 10, 21, 63])
    paths = model.predict_paths(x, n_paths=6)  # [6, 5] cumulative log-returns
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# The five prediction targets — must match the order used in LSTMSequenceDataset
LSTM_HORIZONS: List[int] = [1, 5, 10, 21, 63]


class StockPriceLSTM(nn.Module):
    """Multi-step LSTM for individual stock price path prediction.

    Parameters
    ----------
    n_features  : number of input features per timestep
    hidden_dim  : LSTM hidden size
    num_layers  : number of stacked LSTM layers
    dropout     : dropout rate — used inside LSTM and in the output head;
                  also active during predict_paths() for MC sampling
    horizon_days: ordered list of forward horizons to predict
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        horizon_days: List[int] = LSTM_HORIZONS,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.horizon_days = horizon_days
        n_out = len(horizon_days)

        # Soft feature selector: learns to suppress low-signal inputs
        self.input_gate = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.Sigmoid(),
        )

        # LSTM backbone
        self.lstm = nn.LSTM(
            n_features,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # MC dropout — kept active at inference via predict_paths()
        self.mc_dropout = nn.Dropout(p=dropout)

        # Output head predicts all horizons at once
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : [batch, lookback, n_features]

        Returns
        -------
        [batch, n_horizons] — predicted cumulative log-returns
        """
        # Apply input gate per timestep
        g = self.input_gate(x)          # [B, T, F]
        h = x * g                        # [B, T, F]

        h, _ = self.lstm(h)             # [B, T, D]
        h = self.mc_dropout(h[:, -1])   # [B, D]
        return self.head(h)             # [B, n_horizons]

    def predict_paths(
        self,
        x: torch.Tensor,
        n_paths: int = 6,
    ) -> torch.Tensor:
        """MC-dropout inference — generate N independent forecast paths.

        Dropout stays active so each forward pass samples a different mask,
        producing the fan spread seen in the chart.

        Parameters
        ----------
        x       : [1, lookback, n_features]
        n_paths : number of stochastic realizations

        Returns
        -------
        [n_paths, n_horizons] — cumulative log-returns per path
        """
        was_training = self.training
        self.train()  # activate dropout for MC sampling
        with torch.no_grad():
            paths = torch.stack([self(x).squeeze(0) for _ in range(n_paths)])
        if not was_training:
            self.eval()
        return paths  # [n_paths, n_horizons]

    def config(self) -> dict:
        return {
            "n_features": self.n_features,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
            "horizon_days": self.horizon_days,
        }

    @classmethod
    def from_config(cls, cfg: dict) -> StockPriceLSTM:
        return cls(**cfg)
