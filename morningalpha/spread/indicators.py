"""
Technical indicator computations for the MorningAlpha daily pipeline.
All functions accept pandas Series/DataFrames and return scalar values.
"""

import numpy as np
import pandas as pd


def _compute_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Compute EMA using standard formula: multiplier = 2/(period+1), seed = SMA.
    Returns a Series aligned with the input index.
    """
    if len(series) < period:
        return pd.Series(np.nan, index=series.index)
    multiplier = 2.0 / (period + 1)
    ema_values = np.full(len(series), np.nan)
    # Seed with first SMA
    ema_values[period - 1] = series.iloc[:period].mean()
    for i in range(period, len(series)):
        ema_values[i] = series.iloc[i] * multiplier + ema_values[i - 1] * (1 - multiplier)
    return pd.Series(ema_values, index=series.index)


def _compute_rsi_series(prices: pd.Series, period: int) -> pd.Series:
    """
    Compute RSI series using Wilder smoothing.
    Returns a Series of RSI values aligned with input index.
    """
    if len(prices) < period + 1:
        return pd.Series(np.nan, index=prices.index)

    deltas = prices.diff()
    gains = deltas.where(deltas > 0, 0.0)
    losses = -deltas.where(deltas < 0, 0.0)

    # Wilder smoothing: seed with simple average, then rolling Wilder
    avg_gain = np.full(len(prices), np.nan)
    avg_loss = np.full(len(prices), np.nan)

    # Seed at position `period`
    avg_gain[period] = gains.iloc[1:period + 1].mean()
    avg_loss[period] = losses.iloc[1:period + 1].mean()

    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains.iloc[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses.iloc[i]) / period

    avg_gain_s = pd.Series(avg_gain, index=prices.index)
    avg_loss_s = pd.Series(avg_loss, index=prices.index)

    rs = avg_gain_s / avg_loss_s
    rsi = 100 - (100 / (1 + rs))
    # When avg_loss is 0, RSI should be 100
    rsi = rsi.where(avg_loss_s != 0, 100.0)
    return rsi


def compute_all_indicators(ohlcv_df: pd.DataFrame) -> dict:
    """
    Compute all technical indicators from OHLCV data.

    Args:
        ohlcv_df: DataFrame with columns Close, High, Low, Volume.
                  If MultiIndex columns are present, pass a pre-sliced single-ticker frame.

    Returns:
        Flat dict with all indicator scalar values. Returns np.nan for any indicator
        that cannot be computed due to insufficient data or errors.
    """
    result: dict = {}

    # --- Extract series ---
    try:
        close = ohlcv_df["Close"].dropna()
        high = ohlcv_df["High"].dropna()
        low = ohlcv_df["Low"].dropna()
        volume = ohlcv_df["Volume"].dropna()
    except Exception:
        # Return all-nan dict
        nan_keys = [
            "SMA7", "SMA20", "SMA50", "SMA200",
            "PriceToSMA20Pct", "PriceToSMA50Pct", "PriceToSMA200Pct",
            "EMA7", "EMA200",
            "MACD", "MACDSignal", "MACDHist",
            "RSI7", "RSI21",
            "StochK", "StochD",
            "ROC5", "ROC10", "ROC21",
            "ATR14",
            "BollingerPctB", "BollingerBandwidth",
            "AnnualizedVol",
            "OBV",
            "RelativeVolume", "VolumeROC",
        ]
        return {k: np.nan for k in nan_keys}

    n = len(close)
    last_close = close.iloc[-1] if n > 0 else np.nan

    # -------------------------------------------------------------------------
    # TREND
    # -------------------------------------------------------------------------

    # SMA7
    try:
        result["SMA7"] = close.iloc[-7:].mean() if n >= 7 else np.nan
    except Exception:
        result["SMA7"] = np.nan

    # SMA20
    try:
        result["SMA20"] = close.iloc[-20:].mean() if n >= 20 else np.nan
    except Exception:
        result["SMA20"] = np.nan

    # SMA50
    try:
        result["SMA50"] = close.iloc[-50:].mean() if n >= 50 else np.nan
    except Exception:
        result["SMA50"] = np.nan

    # SMA200
    try:
        result["SMA200"] = close.iloc[-200:].mean() if n >= 200 else np.nan
    except Exception:
        result["SMA200"] = np.nan

    # PriceToSMA20Pct
    try:
        sma20 = result["SMA20"]
        result["PriceToSMA20Pct"] = (last_close - sma20) / sma20 * 100 if (not pd.isna(sma20) and sma20 != 0) else np.nan
    except Exception:
        result["PriceToSMA20Pct"] = np.nan

    # PriceToSMA50Pct
    try:
        sma50 = result["SMA50"]
        result["PriceToSMA50Pct"] = (last_close - sma50) / sma50 * 100 if (not pd.isna(sma50) and sma50 != 0) else np.nan
    except Exception:
        result["PriceToSMA50Pct"] = np.nan

    # PriceToSMA200Pct
    try:
        sma200 = result["SMA200"]
        result["PriceToSMA200Pct"] = (last_close - sma200) / sma200 * 100 if (not pd.isna(sma200) and sma200 != 0) else np.nan
    except Exception:
        result["PriceToSMA200Pct"] = np.nan

    # EMA7
    try:
        ema7_series = _compute_ema(close, 7)
        result["EMA7"] = ema7_series.iloc[-1] if n >= 7 else np.nan
    except Exception:
        result["EMA7"] = np.nan

    # EMA200
    try:
        ema200_series = _compute_ema(close, 200)
        result["EMA200"] = ema200_series.iloc[-1] if n >= 200 else np.nan
    except Exception:
        result["EMA200"] = np.nan

    # MACD = EMA12 - EMA26
    try:
        if n >= 26:
            ema12 = _compute_ema(close, 12)
            ema26 = _compute_ema(close, 26)
            macd_line = ema12 - ema26
            macd_val = macd_line.iloc[-1]
            result["MACD"] = macd_val
            # MACDSignal = EMA9 of MACD line (need at least 26 + 9 - 1 = 34 points)
            if n >= 34:
                signal_series = _compute_ema(macd_line.dropna(), 9)
                result["MACDSignal"] = signal_series.iloc[-1]
                result["MACDHist"] = macd_val - signal_series.iloc[-1]
            else:
                result["MACDSignal"] = np.nan
                result["MACDHist"] = np.nan
        else:
            result["MACD"] = np.nan
            result["MACDSignal"] = np.nan
            result["MACDHist"] = np.nan
    except Exception:
        result["MACD"] = np.nan
        result["MACDSignal"] = np.nan
        result["MACDHist"] = np.nan

    # -------------------------------------------------------------------------
    # MOMENTUM
    # -------------------------------------------------------------------------

    # RSI7
    try:
        rsi7_series = _compute_rsi_series(close, 7)
        result["RSI7"] = rsi7_series.iloc[-1] if n >= 8 else np.nan
    except Exception:
        result["RSI7"] = np.nan

    # RSI21
    try:
        rsi21_series = _compute_rsi_series(close, 21)
        result["RSI21"] = rsi21_series.iloc[-1] if n >= 22 else np.nan
    except Exception:
        result["RSI21"] = np.nan

    # StochK and StochD (using 14-period)
    try:
        if n >= 14 and len(high) >= 14 and len(low) >= 14:
            # Align all series to the same index via the close index
            common_idx = close.index
            high_aligned = high.reindex(common_idx)
            low_aligned = low.reindex(common_idx)

            lowest_low = low_aligned.rolling(14).min()
            highest_high = high_aligned.rolling(14).max()

            denom = highest_high - lowest_low
            stoch_k_series = (close - lowest_low) / denom * 100
            stoch_k_series = stoch_k_series.where(denom != 0, np.nan)

            result["StochK"] = stoch_k_series.iloc[-1]
            if n >= 16:  # need at least 3 more for SMA3
                stoch_d = stoch_k_series.rolling(3).mean()
                result["StochD"] = stoch_d.iloc[-1]
            else:
                result["StochD"] = np.nan
        else:
            result["StochK"] = np.nan
            result["StochD"] = np.nan
    except Exception:
        result["StochK"] = np.nan
        result["StochD"] = np.nan

    # ROC5
    try:
        result["ROC5"] = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100 if n >= 6 else np.nan
    except Exception:
        result["ROC5"] = np.nan

    # ROC10
    try:
        result["ROC10"] = (close.iloc[-1] - close.iloc[-11]) / close.iloc[-11] * 100 if n >= 11 else np.nan
    except Exception:
        result["ROC10"] = np.nan

    # ROC21
    try:
        result["ROC21"] = (close.iloc[-1] - close.iloc[-22]) / close.iloc[-22] * 100 if n >= 22 else np.nan
    except Exception:
        result["ROC21"] = np.nan

    # -------------------------------------------------------------------------
    # VOLATILITY
    # -------------------------------------------------------------------------

    # ATR14 — EMA_14 of True Range
    try:
        if n >= 15 and len(high) >= 15 and len(low) >= 15:
            common_idx = close.index
            high_aligned = high.reindex(common_idx)
            low_aligned = low.reindex(common_idx)

            prev_close = close.shift(1)
            tr = pd.concat([
                high_aligned - low_aligned,
                (high_aligned - prev_close).abs(),
                (low_aligned - prev_close).abs(),
            ], axis=1).max(axis=1)

            atr_series = _compute_ema(tr.dropna(), 14)
            result["ATR14"] = atr_series.iloc[-1]
        else:
            result["ATR14"] = np.nan
    except Exception:
        result["ATR14"] = np.nan

    # Bollinger Bands (20-period, 2 std)
    try:
        if n >= 20:
            sma20_series = close.rolling(20).mean()
            std20_series = close.rolling(20).std(ddof=1)
            upper = sma20_series + 2 * std20_series
            lower = sma20_series - 2 * std20_series

            band_width = upper.iloc[-1] - lower.iloc[-1]
            sma20_last = sma20_series.iloc[-1]

            if band_width != 0:
                result["BollingerPctB"] = (last_close - lower.iloc[-1]) / band_width
            else:
                result["BollingerPctB"] = np.nan

            result["BollingerBandwidth"] = (band_width / sma20_last * 100) if sma20_last != 0 else np.nan
        else:
            result["BollingerPctB"] = np.nan
            result["BollingerBandwidth"] = np.nan
    except Exception:
        result["BollingerPctB"] = np.nan
        result["BollingerBandwidth"] = np.nan

    # AnnualizedVol
    try:
        if n >= 2:
            log_returns = np.log(close / close.shift(1)).dropna()
            result["AnnualizedVol"] = log_returns.std() * np.sqrt(252) * 100
        else:
            result["AnnualizedVol"] = np.nan
    except Exception:
        result["AnnualizedVol"] = np.nan

    # -------------------------------------------------------------------------
    # VOLUME
    # -------------------------------------------------------------------------

    # OBV
    try:
        if n >= 2 and len(volume) >= 2:
            common_idx = close.index
            vol_aligned = volume.reindex(common_idx).fillna(0)
            price_change = close.diff()
            obv_changes = np.where(price_change > 0, vol_aligned, np.where(price_change < 0, -vol_aligned, 0))
            obv_series = pd.Series(obv_changes, index=close.index).cumsum()
            result["OBV"] = obv_series.iloc[-1]
        else:
            result["OBV"] = np.nan
    except Exception:
        result["OBV"] = np.nan

    # RelativeVolume
    try:
        if len(volume) >= 20:
            sma20_vol = volume.iloc[-20:].mean()
            result["RelativeVolume"] = volume.iloc[-1] / sma20_vol if sma20_vol != 0 else np.nan
        else:
            result["RelativeVolume"] = np.nan
    except Exception:
        result["RelativeVolume"] = np.nan

    # VolumeROC
    try:
        if len(volume) >= 22:
            vol_21_ago = volume.iloc[-22]
            result["VolumeROC"] = (volume.iloc[-1] - vol_21_ago) / vol_21_ago * 100 if vol_21_ago != 0 else np.nan
        else:
            result["VolumeROC"] = np.nan
    except Exception:
        result["VolumeROC"] = np.nan

    return result
