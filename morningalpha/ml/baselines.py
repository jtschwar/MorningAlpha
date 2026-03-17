"""
Baseline models for stock prediction.
Simple implementations to compare against set transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class LinearBaseline(nn.Module):
    """
    Simple linear regression baseline.
    Uses only scalar features (no time-series).
    """
    
    def __init__(self, input_dim: int = 20):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, scalar_features):
        """
        Args:
            scalar_features: [batch_size, input_dim]
        
        Returns:
            Predicted return: [batch_size, 1]
        """
        return self.linear(scalar_features)


class MLPBaseline(nn.Module):
    """
    Multi-layer perceptron baseline.
    Uses scalar features with non-linear transformations.
    """
    
    def __init__(self, input_dim: int = 20, hidden_dims: list = [128, 64], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, scalar_features):
        """
        Args:
            scalar_features: [batch_size, input_dim]
        
        Returns:
            Predicted return: [batch_size, 1]
        """
        return self.net(scalar_features)


class CNN1DBaseline(nn.Module):
    """
    1D Convolutional Neural Network baseline.
    Processes time-series data with convolutional layers.
    """
    
    def __init__(self, input_channels: int = 5, num_filters: list = [32, 64], kernel_size: int = 3):
        super().__init__()
        self.convs = nn.ModuleList()
        prev_channels = input_channels
        
        for out_channels in num_filters:
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(prev_channels, out_channels, kernel_size, padding=kernel_size//2),
                    nn.ReLU(),
                    nn.MaxPool1d(2)
                )
            )
            prev_channels = out_channels
        
        # Calculate output size (depends on input length)
        # For 60-day sequence with 2 pooling layers: 60 -> 30 -> 15
        self.fc = nn.Linear(num_filters[-1] * 15, 1)
    
    def forward(self, time_series):
        """
        Args:
            time_series: [batch_size, time_steps, features] -> [batch_size, features, time_steps]
        
        Returns:
            Predicted return: [batch_size, 1]
        """
        # Convert to [batch, channels, time] format
        x = time_series.transpose(1, 2)
        
        # Apply convolutions
        for conv in self.convs:
            x = conv(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        return self.fc(x)


class LSTMBaseline(nn.Module):
    """
    LSTM baseline for time-series prediction.
    Standard recurrent neural network.
    """
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, time_series):
        """
        Args:
            time_series: [batch_size, time_steps, features]
        
        Returns:
            Predicted return: [batch_size, 1]
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(time_series)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Predict
        return self.fc(last_hidden)


class GRUBaseline(nn.Module):
    """
    GRU baseline - simpler alternative to LSTM.
    """
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, time_series):
        """
        Args:
            time_series: [batch_size, time_steps, features]
        
        Returns:
            Predicted return: [batch_size, 1]
        """
        gru_out, h_n = self.gru(time_series)
        last_hidden = gru_out[:, -1, :]
        return self.fc(last_hidden)


class CNNLSTMHybridBaseline(nn.Module):
    """
    Hybrid CNN + LSTM baseline.
    CNN extracts features, LSTM models sequences.
    """
    
    def __init__(self, input_channels: int = 5, cnn_filters: int = 32, lstm_hidden: int = 128, num_layers: int = 2):
        super().__init__()
        # CNN feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, cnn_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # LSTM sequence modeling
        self.lstm = nn.LSTM(
            cnn_filters,
            lstm_hidden,
            num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(lstm_hidden, 1)
    
    def forward(self, time_series):
        """
        Args:
            time_series: [batch_size, time_steps, features]
        
        Returns:
            Predicted return: [batch_size, 1]
        """
        # CNN: [batch, time, features] -> [batch, features, time]
        x = time_series.transpose(1, 2)
        x = self.cnn(x)  # [batch, filters, time/2]
        
        # Back to [batch, time, features] for LSTM
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        
        # Predict
        return self.fc(last_hidden)


class TransformerBaseline(nn.Module):
    """
    Standard Transformer baseline (with positional encoding).
    Compares against set transformer (which doesn't use positional encoding).
    """
    
    def __init__(self, input_dim: int = 5, d_model: int = 128, nhead: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, time_series):
        """
        Args:
            time_series: [batch_size, time_steps, features]
        
        Returns:
            Predicted return: [batch_size, 1]
        """
        # Project input
        x = self.input_proj(time_series) * np.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Pool (use mean of all time steps)
        x = x.mean(dim=1)
        
        # Predict
        return self.fc(x)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiTaskBaseline(nn.Module):
    """
    Multi-task baseline that predicts multiple outputs.
    Can be used with any of the above architectures.
    """
    
    def __init__(self, base_model: nn.Module, output_dim: int = 128):
        super().__init__()
        self.base_model = base_model
        
        # Multiple output heads
        self.return_head = nn.Linear(output_dim, 1)
        self.entry_head = nn.Linear(output_dim, 1)
        self.risk_head = nn.Linear(output_dim, 1)
    
    def forward(self, *args, **kwargs):
        # Get base model output
        features = self.base_model(*args, **kwargs)
        
        # Multiple predictions
        return {
            'return_3m': self.return_head(features),
            'entry_score': torch.sigmoid(self.entry_head(features)) * 100,
            'max_drawdown': self.risk_head(features)
        }


# ---------------------------------------------------------------------------
# Scikit-learn / LightGBM models (tabular — no PyTorch dependency)
# ---------------------------------------------------------------------------

try:
    import pickle
    import lightgbm as lgb
    import shap as shap_lib
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge

    class LightGBMModel:
        """LightGBM regressor wrapper with SHAP support."""

        DEFAULT_PARAMS = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "n_estimators": 1000,
            "verbose": -1,
        }

        def __init__(self, params=None):
            self.params = {**self.DEFAULT_PARAMS, **(params or {})}
            self.model: lgb.LGBMRegressor = None
            self._shap_explainer = None

        def fit(self, X_train, y_train, X_val, y_val):
            self.model = lgb.LGBMRegressor(**self.params)
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
            )
            self._shap_explainer = None  # reset on refit
            return self

        def predict(self, X):
            return self.model.predict(X)

        def shap_values(self, X):
            if self._shap_explainer is None:
                self._shap_explainer = shap_lib.TreeExplainer(self.model)
            return self._shap_explainer.shap_values(X)

        def rank_ic(self, X, y_true):
            preds = self.predict(X)
            return float(spearmanr(preds, y_true).correlation)

        def save(self, path):
            with open(path, "wb") as f:
                pickle.dump(self, f)

        @classmethod
        def load(cls, path):
            with open(path, "rb") as f:
                return pickle.load(f)

        @property
        def feature_importances_(self):
            return self.model.feature_importances_


    class RidgeModel:
        """Ridge regression wrapper for feature importance analysis."""

        def __init__(self, alpha: float = 1.0):
            self.alpha = alpha
            self.model = Ridge(alpha=alpha)
            self.feature_names_ = None

        def fit(self, X_train, y_train, feature_names=None):
            self.model.fit(X_train, y_train)
            self.feature_names_ = feature_names
            return self

        def predict(self, X):
            return self.model.predict(X)

        def rank_ic(self, X, y_true):
            preds = self.predict(X)
            return float(spearmanr(preds, y_true).correlation)

        def feature_importance_series(self):
            import pandas as pd
            names = self.feature_names_ or [f"f{i}" for i in range(len(self.model.coef_))]
            return (
                pd.Series(self.model.coef_, index=names)
                .reindex(pd.Series(self.model.coef_, index=names).abs().sort_values(ascending=False).index)
            )

        def save(self, path):
            import pickle
            with open(path, "wb") as f:
                pickle.dump(self, f)

        @classmethod
        def load(cls, path):
            import pickle
            with open(path, "rb") as f:
                return pickle.load(f)

except ImportError:
    pass  # lightgbm/sklearn/shap not installed — tabular models unavailable


# Example usage and comparison
if __name__ == "__main__":
    batch_size = 32
    time_steps = 60
    scalar_features_dim = 20
    
    # Create dummy data
    time_series = torch.randn(batch_size, time_steps, 5)
    scalar_features = torch.randn(batch_size, scalar_features_dim)
    
    # Test each baseline
    baselines = {
        'Linear': LinearBaseline(scalar_features_dim),
        'MLP': MLPBaseline(scalar_features_dim),
        'CNN1D': CNN1DBaseline(input_channels=5),
        'LSTM': LSTMBaseline(input_dim=5),
        'GRU': GRUBaseline(input_dim=5),
        'CNN+LSTM': CNNLSTMHybridBaseline(input_channels=5),
        'Transformer': TransformerBaseline(input_dim=5)
    }
    
    print("Testing baselines:")
    print("-" * 50)
    
    for name, model in baselines.items():
        try:
            if name in ['Linear', 'MLP']:
                output = model(scalar_features)
            else:
                output = model(time_series)
            
            print(f"{name:15} Output shape: {output.shape}")
            
            # Count parameters
            params = sum(p.numel() for p in model.parameters())
            print(f"{'':15} Parameters: {params:,}")
            print()
        except Exception as e:
            print(f"{name:15} Error: {e}")
            print()

