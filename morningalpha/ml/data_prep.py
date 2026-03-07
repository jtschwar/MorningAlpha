"""
Data preparation for two-tower set transformer model.
Converts stock data into format suitable for training.
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta


def prepare_stock_features(
    prices: pd.Series,
    volumes: pd.Series,
    window_size: int = 60
) -> Dict[str, np.ndarray]:
    """
    Prepare stock-specific time-series features.
    
    Args:
        prices: Series of closing prices
        volumes: Series of trading volumes
        window_size: Number of days to include
    
    Returns:
        Dictionary with time-series features
    """
    # Get recent window
    recent_prices = prices.iloc[-window_size:].values
    recent_volumes = volumes.iloc[-window_size:].values
    
    # Calculate returns
    returns = np.diff(recent_prices) / recent_prices[:-1]
    returns = np.concatenate([[0], returns])  # First day has no return
    
    # Get highs and lows (if available, otherwise use close price)
    # For simplicity, assume we only have close prices
    highs = recent_prices  # Could be replaced with actual high data
    lows = recent_prices   # Could be replaced with actual low data
    
    # Normalize volumes (log scale to handle large variations)
    log_volumes = np.log1p(recent_volumes)
    
    # Stack features: [price, volume, returns, high, low]
    # Normalize prices by dividing by first price (relative prices)
    normalized_prices = recent_prices / recent_prices[0]
    
    time_series = np.stack([
        normalized_prices,
        log_volumes / log_volumes.max() if log_volumes.max() > 0 else log_volumes,
        returns,
        highs / recent_prices[0],
        lows / recent_prices[0]
    ], axis=1)
    
    return {
        'time_series': time_series.astype(np.float32)
    }


def prepare_scalar_features(
    metrics: Dict[str, float],
    market_cap_category: Optional[str] = None,
    sector: Optional[str] = None
) -> np.ndarray:
    """
    Prepare scalar features (current metrics, metadata).
    
    Args:
        metrics: Dictionary of calculated metrics (RSI, Entry Score, etc.)
        market_cap_category: Market cap category (Large/Mid/Small)
        sector: Sector name
    
    Returns:
        Array of scalar features
    """
    features = []
    
    # Normalize metrics to 0-1 range
    rsi = metrics.get('rsi', 50) / 100.0  # RSI is 0-100
    entry_score = metrics.get('entry_score', 50) / 100.0
    quality_score = metrics.get('quality_score', 50) / 100.0
    sharpe = (metrics.get('sharpe_ratio', 0) + 2) / 4.0  # Normalize -2 to 2 range
    consistency = metrics.get('consistency_score', 50) / 100.0
    momentum_accel = metrics.get('momentum_acceleration', 1.0) / 2.0  # Normalize
    volume_surge = metrics.get('volume_surge', 100) / 200.0  # Normalize
    price_vs_high = metrics.get('price_vs_20d_high', 90) / 100.0
    max_drawdown = (metrics.get('max_drawdown', 0) + 50) / 50.0  # Normalize -50 to 0
    
    features.extend([
        rsi,
        entry_score,
        quality_score,
        sharpe,
        consistency,
        momentum_accel,
        volume_surge,
        price_vs_high,
        max_drawdown,
        0.0  # Placeholder for 10th feature
    ])
    
    return np.array(features, dtype=np.float32)


def prepare_market_features(
    spy_returns: pd.Series,
    sector_returns: Optional[pd.Series] = None,
    vix_levels: Optional[pd.Series] = None,
    window_size: int = 60
) -> Dict[str, np.ndarray]:
    """
    Prepare market context time-series features.
    
    Args:
        spy_returns: S&P 500 daily returns
        sector_returns: Sector daily returns (optional)
        vix_levels: VIX levels (optional)
        window_size: Number of days to include
    
    Returns:
        Dictionary with market time-series features
    """
    # Get recent window
    recent_spy = spy_returns.iloc[-window_size:].values
    
    # Use sector returns if available, otherwise use SPY
    if sector_returns is not None:
        recent_sector = sector_returns.iloc[-window_size:].values
    else:
        recent_sector = recent_spy
    
    # Use VIX if available, otherwise use SPY volatility as proxy
    if vix_levels is not None:
        recent_vix = vix_levels.iloc[-window_size:].values / 30.0  # Normalize
    else:
        # Use rolling volatility as proxy
        recent_vix = np.abs(recent_spy) * 10  # Rough proxy
    
    # Stack: [market_return, sector_return, vix]
    market_series = np.stack([
        recent_spy,
        recent_sector,
        recent_vix
    ], axis=1)
    
    return {
        'market_series': market_series.astype(np.float32)
    }


def prepare_regime_features(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    window_size: int = 60
) -> np.ndarray:
    """
    Prepare market regime features (beta, relative strength, etc.).
    
    Args:
        stock_returns: Stock daily returns
        market_returns: Market daily returns
        window_size: Number of days for calculation
    
    Returns:
        Array of regime features
    """
    # Get recent returns
    stock_recent = stock_returns.iloc[-window_size:].values
    market_recent = market_returns.iloc[-window_size:].values
    
    # Calculate beta (correlation * (stock_std / market_std))
    if len(stock_recent) > 1 and len(market_recent) > 1:
        correlation = np.corrcoef(stock_recent, market_recent)[0, 1]
        stock_std = np.std(stock_recent)
        market_std = np.std(market_recent)
        beta = correlation * (stock_std / market_std) if market_std > 0 else 1.0
    else:
        beta = 1.0
    
    # Relative strength (stock return / market return)
    stock_total_return = (1 + stock_recent).prod() - 1
    market_total_return = (1 + market_recent).prod() - 1
    relative_strength = stock_total_return / market_total_return if market_total_return != 0 else 1.0
    
    # Market trend (positive = uptrend)
    market_trend = np.mean(market_recent)
    
    # Volatility ratio
    stock_vol = np.std(stock_recent)
    market_vol = np.std(market_recent)
    vol_ratio = stock_vol / market_vol if market_vol > 0 else 1.0
    
    # Market regime (bull/bear indicator)
    market_regime = 1.0 if market_trend > 0 else -1.0
    
    features = np.array([
        beta,
        relative_strength,
        market_trend,
        vol_ratio,
        market_regime
    ], dtype=np.float32)
    
    return features


def create_labels(
    prices: pd.Series,
    entry_date: datetime,
    lookahead_days: int = 90
) -> Dict[str, float]:
    """
    Create labels for training (forward returns, drawdowns, etc.).
    
    Args:
        prices: Historical price series
        entry_date: Date of entry signal
        lookahead_days: Number of days to look ahead
    
    Returns:
        Dictionary with labels
    """
    # Find entry date in price series
    entry_idx = prices.index.get_indexer([entry_date], method='nearest')[0]
    
    if entry_idx < 0 or entry_idx >= len(prices):
        return None
    
    entry_price = prices.iloc[entry_idx]
    
    # Get future prices
    future_prices = prices.iloc[entry_idx:entry_idx + lookahead_days]
    
    if len(future_prices) < 2:
        return None
    
    # Calculate returns
    return_1m = (future_prices.iloc[min(21, len(future_prices)-1)] / entry_price - 1) * 100
    return_3m = (future_prices.iloc[-1] / entry_price - 1) * 100
    
    # Calculate max drawdown
    rolling_max = future_prices.expanding().max()
    drawdowns = (future_prices - rolling_max) / rolling_max * 100
    max_drawdown = drawdowns.min()
    
    # Binary labels
    entry_signal = 1.0 if return_3m > 10.0 else 0.0  # >10% return = good entry
    hit_target = 1.0 if return_3m > 15.0 else 0.0     # >15% return = hit target
    
    return {
        'return_1m': float(return_1m),
        'return_3m': float(return_3m),
        'max_drawdown_3m': float(max_drawdown),
        'entry_signal': entry_signal,
        'hit_target': hit_target
    }


class StockDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for stock prediction.
    """
    
    def __init__(
        self,
        stock_data: List[Dict],
        window_size: int = 60,
        lookahead_days: int = 90
    ):
        """
        Args:
            stock_data: List of dictionaries, each containing:
                - 'ticker': Stock ticker
                - 'prices': Price series
                - 'volumes': Volume series
                - 'metrics': Calculated metrics dict
                - 'spy_returns': S&P 500 returns
                - 'entry_date': Date of entry signal
            window_size: Number of days for time-series features
            lookahead_days: Number of days for label calculation
        """
        self.stock_data = stock_data
        self.window_size = window_size
        self.lookahead_days = lookahead_days
        
        # Filter out stocks with insufficient data
        self.valid_indices = []
        for i, data in enumerate(stock_data):
            if len(data['prices']) >= window_size + lookahead_days:
                self.valid_indices.append(i)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        data_idx = self.valid_indices[idx]
        data = self.stock_data[data_idx]
        
        # Prepare features
        stock_features = prepare_stock_features(
            data['prices'],
            data['volumes'],
            self.window_size
        )
        
        scalar_features = prepare_scalar_features(
            data['metrics']
        )
        
        market_features = prepare_market_features(
            data['spy_returns'],
            data.get('sector_returns'),
            data.get('vix_levels'),
            self.window_size
        )
        
        regime_features = prepare_regime_features(
            data['prices'].pct_change().dropna(),
            data['spy_returns'],
            self.window_size
        )
        
        # Create labels
        labels = create_labels(
            data['prices'],
            data['entry_date'],
            self.lookahead_days
        )
        
        if labels is None:
            # Return first valid item if this one fails
            return self.__getitem__(0)
        
        return {
            'stock_time_series': torch.FloatTensor(stock_features['time_series']),
            'stock_scalar': torch.FloatTensor(scalar_features),
            'market_series': torch.FloatTensor(market_features['market_series']),
            'market_regime': torch.FloatTensor(regime_features),
            'return_3m': torch.FloatTensor([labels['return_3m']]),
            'entry_signal': torch.FloatTensor([labels['entry_signal']]),
            'max_drawdown': torch.FloatTensor([labels['max_drawdown_3m']]),
            'ticker': data['ticker']
        }


# Example usage
if __name__ == "__main__":
    # Example: Create dummy data
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    
    # Dummy stock data
    prices = pd.Series(np.random.randn(200).cumsum() + 100, index=dates)
    volumes = pd.Series(np.random.randint(1000000, 10000000, 200), index=dates)
    spy_returns = pd.Series(np.random.randn(200) * 0.01, index=dates)
    
    metrics = {
        'rsi': 45.0,
        'entry_score': 75.0,
        'quality_score': 80.0,
        'sharpe_ratio': 1.5,
        'consistency_score': 60.0,
        'momentum_acceleration': 1.2,
        'volume_surge': 120.0,
        'price_vs_20d_high': 90.0,
        'max_drawdown': -15.0
    }
    
    stock_data = [{
        'ticker': 'AAPL',
        'prices': prices,
        'volumes': volumes,
        'metrics': metrics,
        'spy_returns': spy_returns,
        'entry_date': dates[100]
    }]
    
    # Create dataset
    dataset = StockDataset(stock_data, window_size=60, lookahead_days=90)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Test
    for batch in dataloader:
        print("Batch keys:", batch.keys())
        print("Stock time series shape:", batch['stock_time_series'].shape)
        print("Stock scalar shape:", batch['stock_scalar'].shape)
        print("Market series shape:", batch['market_series'].shape)
        print("Market regime shape:", batch['market_regime'].shape)
        print("Return 3m:", batch['return_3m'].item())
        break

