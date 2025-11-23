# morningalpha/spread/metrics.py
"""
Enhanced metrics for stock analysis.
All functions accept price/volume series and return scalar values or dicts.
"""
import numpy as np
import pandas as pd


def calculate_sharpe_ratio(returns_series, risk_free_rate=0.05):
    """Annualized Sharpe ratio (assuming 252 trading days)"""
    if len(returns_series) < 2:
        return np.nan
    
    excess_returns = returns_series - (risk_free_rate / 252)
    if excess_returns.std() == 0:
        return np.nan
    
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def calculate_sortino_ratio(returns_series, risk_free_rate=0.05):
    """Like Sharpe but only penalizes downside volatility"""
    if len(returns_series) < 2:
        return np.nan
    
    excess_returns = returns_series - (risk_free_rate / 252)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.nan
    
    downside_std = np.sqrt(np.mean(downside_returns**2))
    if downside_std == 0:
        return np.nan
    
    return np.sqrt(252) * excess_returns.mean() / downside_std


def calculate_momentum_score(prices):
    """Weighted momentum across multiple timeframes"""
    if len(prices) < 20:
        return np.nan
    
    # Handle cases where we don't have enough data for longer periods
    returns_1m = (prices[-20:].iloc[-1] / prices[-20:].iloc[0] - 1) * 100 if len(prices) >= 20 else np.nan
    returns_3m = (prices[-60:].iloc[-1] / prices[-60:].iloc[0] - 1) * 100 if len(prices) >= 60 else returns_1m
    returns_6m = (prices.iloc[-1] / prices.iloc[0] - 1) * 100 if len(prices) >= 120 else returns_3m
    
    # Weight recent performance higher
    momentum = (returns_1m * 0.5 + returns_3m * 0.3 + returns_6m * 0.2)
    return momentum


def calculate_consistency_score(returns_series):
    """Percentage of positive return days"""
    if len(returns_series) == 0:
        return np.nan
    return (returns_series > 0).sum() / len(returns_series) * 100


def calculate_volume_metrics(volumes):
    """Volume consistency and trends"""
    if len(volumes) < 20:
        return {
            'avg_volume': np.nan,
            'volume_trend': np.nan,
            'volume_consistency': np.nan
        }
    
    avg_volume = volumes.mean()
    recent_avg = volumes[-20:].mean()
    volume_trend = (recent_avg / avg_volume - 1) * 100 if avg_volume > 0 else np.nan
    volume_consistency = 1 - (volumes.std() / avg_volume) if avg_volume > 0 else np.nan
    
    return {
        'avg_volume': avg_volume,
        'volume_trend': volume_trend,
        'volume_consistency': volume_consistency
    }


def calculate_drawdown_metrics(prices):
    """Multiple drawdown metrics"""
    if len(prices) < 2:
        return {
            'max_drawdown': np.nan,
            'avg_drawdown': np.nan,
            'recovery_days': None
        }
    
    rolling_max = prices.expanding().max()
    drawdowns = (prices - rolling_max) / rolling_max * 100
    
    max_drawdown = drawdowns.min()
    avg_drawdown = drawdowns[drawdowns < 0].mean() if (drawdowns < 0).any() else 0
    
    # Recovery time from max drawdown
    try:
        max_dd_idx = drawdowns.idxmin()
        recovery_idx = prices[max_dd_idx:][prices[max_dd_idx:] >= rolling_max[max_dd_idx]].index
        recovery_days = len(prices[max_dd_idx:recovery_idx[0]]) if len(recovery_idx) > 0 else None
    except:
        recovery_days = None
    
    return {
        'max_drawdown': max_drawdown,
        'avg_drawdown': avg_drawdown,
        'recovery_days': recovery_days
    }


def normalize_to_100(value, min_val, max_val):
    """Normalize a value to 0-100 scale"""
    if pd.isna(value) or pd.isna(min_val) or pd.isna(max_val):
        return 50.0  # Default to middle if any value is NaN
    if max_val == min_val:
        return 50.0
    normalized = ((value - min_val) / (max_val - min_val)) * 100
    return np.clip(normalized, 0, 100)


def calculate_quality_score(return_pct, sharpe, consistency, volume_trend, max_drawdown):
    """
    Composite score (0-100) weighing multiple factors:
    - Return (30%)
    - Risk-adjusted return/Sharpe (25%)
    - Momentum consistency (20%)
    - Volume quality (15%)
    - Drawdown resilience (10%)
    
    Args:
        return_pct: Total return percentage
        sharpe: Sharpe ratio
        consistency: Consistency score (0-100)
        volume_trend: Volume trend percentage
        max_drawdown: Maximum drawdown (negative number)
    """
    weights = {
        'return': 0.30,
        'sharpe': 0.25,
        'consistency': 0.20,
        'volume': 0.15,
        'drawdown': 0.10
    }
    
    # Handle NaN values - if any metric is NaN, return NaN for quality score
    if any(pd.isna(x) for x in [return_pct, sharpe, consistency, volume_trend, max_drawdown]):
        return np.nan
    
    # Normalize each metric to 0-100 scale using reasonable ranges
    return_score = normalize_to_100(return_pct, 0, 200)  # 0-200% return
    sharpe_score = normalize_to_100(sharpe, -1, 3)  # -1 to 3 Sharpe
    consistency_score = consistency  # Already 0-100
    volume_score = normalize_to_100(volume_trend, -50, 200)  # -50% to +200%
    drawdown_score = normalize_to_100(max_drawdown, -80, 0)  # -80% to 0% (less negative is better)
    
    # Weighted average
    quality = (
        weights['return'] * return_score +
        weights['sharpe'] * sharpe_score +
        weights['consistency'] * consistency_score +
        weights['volume'] * volume_score +
        weights['drawdown'] * drawdown_score
    )
    
    return np.clip(quality, 0, 100)


def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index (RSI) - entry timing indicator"""
    if len(prices) < period + 1:
        return np.nan
    
    deltas = prices.diff()
    gains = deltas.where(deltas > 0, 0)
    losses = -deltas.where(deltas < 0, 0)
    
    avg_gain = gains.rolling(window=period).mean()
    avg_loss = losses.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1] if not rsi.empty else np.nan


def calculate_momentum_acceleration(prices):
    """Compare recent momentum (5 days) vs medium-term (20 days)"""
    if len(prices) < 20:
        return np.nan
    
    return_5d = (prices.iloc[-1] / prices.iloc[-5] - 1) * 100 if len(prices) >= 5 else np.nan
    return_20d = (prices.iloc[-1] / prices.iloc[-20] - 1) * 100 if len(prices) >= 20 else np.nan
    
    if pd.isna(return_5d) or pd.isna(return_20d) or return_20d == 0:
        return np.nan
    
    # Acceleration ratio: > 1 means recent momentum is stronger
    acceleration = return_5d / return_20d if return_20d != 0 else np.nan
    return acceleration


def calculate_price_position(prices):
    """Calculate price position relative to 20-day high and low"""
    if len(prices) < 20:
        return {
            'price_vs_20d_high': np.nan,
            'price_vs_20d_low': np.nan,
            'distance_from_high': np.nan
        }
    
    recent_20d = prices.iloc[-20:]
    current_price = prices.iloc[-1]
    high_20d = recent_20d.max()
    low_20d = recent_20d.min()
    
    price_vs_high = (current_price / high_20d) * 100 if high_20d > 0 else np.nan
    price_vs_low = (current_price / low_20d) * 100 if low_20d > 0 else np.nan
    distance_from_high = ((current_price - high_20d) / high_20d) * 100 if high_20d > 0 else np.nan
    
    return {
        'price_vs_20d_high': price_vs_high,
        'price_vs_20d_low': price_vs_low,
        'distance_from_high': distance_from_high
    }


def calculate_volume_surge(volumes):
    """Detect recent volume surge (last 5 days vs 20-day average)"""
    if len(volumes) < 20:
        return np.nan
    
    recent_5d_avg = volumes.iloc[-5:].mean()
    avg_20d = volumes.iloc[-20:].mean()
    
    if avg_20d == 0:
        return np.nan
    
    surge_ratio = (recent_5d_avg / avg_20d) * 100  # Percentage of average
    return surge_ratio


def calculate_short_term_volatility(returns):
    """Calculate short-term volatility (20-day) vs medium-term (60-day)"""
    if len(returns) < 20:
        return {
            'volatility_20d': np.nan,
            'volatility_60d': np.nan,
            'volatility_ratio': np.nan
        }
    
    vol_20d = returns.iloc[-20:].std() * np.sqrt(252) * 100  # Annualized %
    
    if len(returns) >= 60:
        vol_60d = returns.iloc[-60:].std() * np.sqrt(252) * 100
    else:
        vol_60d = vol_20d  # Use 20d if not enough data
    
    vol_ratio = vol_20d / vol_60d if vol_60d > 0 else np.nan
    
    return {
        'volatility_20d': vol_20d,
        'volatility_60d': vol_60d,
        'volatility_ratio': vol_ratio
    }


def calculate_entry_score(rsi, price_vs_high, momentum_accel, volume_surge):
    """
    Calculate entry timing score (0-100) for 1-3 month holds.
    Higher score = better entry point.
    """
    score = 0
    
    # RSI component (40-60 is ideal, not overbought)
    if not pd.isna(rsi):
        if 40 <= rsi <= 60:
            score += 30  # Perfect range
        elif 30 <= rsi < 40 or 60 < rsi <= 70:
            score += 20  # Acceptable
        elif rsi < 30:
            score += 15  # Oversold (could bounce)
        elif rsi > 70:
            score += 5  # Overbought (avoid)
    
    # Price position (85-95% of high = good entry, not at peak)
    if not pd.isna(price_vs_high):
        if 85 <= price_vs_high <= 95:
            score += 25  # Good entry zone
        elif 80 <= price_vs_high < 85 or 95 < price_vs_high <= 98:
            score += 15
        elif price_vs_high > 98:
            score += 5  # Too close to high
    
    # Momentum acceleration (recent momentum stronger = good)
    if not pd.isna(momentum_accel):
        if momentum_accel > 1.2:
            score += 25  # Strong acceleration
        elif momentum_accel > 1.0:
            score += 20  # Positive acceleration
        elif momentum_accel > 0.8:
            score += 10  # Slowing but positive
    
    # Volume surge (increased interest)
    if not pd.isna(volume_surge):
        if volume_surge > 150:
            score += 20  # Strong volume surge
        elif volume_surge > 120:
            score += 15  # Moderate surge
        elif volume_surge > 100:
            score += 10  # Slight increase
    
    return min(100, score)


def calculate_all_metrics(prices, volumes, returns):
    """
    One-stop function to calculate all metrics for a stock.
    
    Args:
        prices: Pandas Series of closing prices
        volumes: Pandas Series of trading volumes
        returns: Pandas Series of daily returns (percentage)
    
    Returns:
        Dict with all calculated metrics
    """
    # Calculate individual metrics
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    consistency = calculate_consistency_score(returns)
    volume_metrics = calculate_volume_metrics(volumes)
    drawdown_metrics = calculate_drawdown_metrics(prices)
    
    # Calculate return percentage
    return_pct = ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100 if len(prices) > 0 else np.nan
    
    # Calculate quality score using the computed metrics
    quality = calculate_quality_score(
        return_pct=return_pct,
        sharpe=sharpe,
        consistency=consistency,
        volume_trend=volume_metrics['volume_trend'],
        max_drawdown=drawdown_metrics['max_drawdown']
    )
    
    # NEW: Short-term holding metrics (for 1-3 month holds)
    rsi = calculate_rsi(prices)
    momentum_accel = calculate_momentum_acceleration(prices)
    price_position = calculate_price_position(prices)
    volume_surge = calculate_volume_surge(volumes)
    volatility_metrics = calculate_short_term_volatility(returns)
    
    # Calculate entry score
    entry_score = calculate_entry_score(
        rsi=rsi,
        price_vs_high=price_position.get('price_vs_20d_high'),
        momentum_accel=momentum_accel,
        volume_surge=volume_surge
    )
    
    return {
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': drawdown_metrics['max_drawdown'],
        'avg_drawdown': drawdown_metrics['avg_drawdown'],
        'recovery_days': drawdown_metrics['recovery_days'],
        'consistency_score': consistency,
        'volume_trend': volume_metrics['volume_trend'],
        'volume_consistency': volume_metrics['volume_consistency'],
        'quality_score': quality,
        # NEW: Short-term metrics
        'rsi': rsi,
        'momentum_acceleration': momentum_accel,
        'price_vs_20d_high': price_position.get('price_vs_20d_high'),
        'distance_from_high': price_position.get('distance_from_high'),
        'volume_surge': volume_surge,
        'volatility_20d': volatility_metrics.get('volatility_20d'),
        'volatility_ratio': volatility_metrics.get('volatility_ratio'),
        'entry_score': entry_score,  # Combined entry timing score
    }