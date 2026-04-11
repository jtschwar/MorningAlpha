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
            "PriceVs52wkHighPct",
            "PriceVs5yrHighPct",
            "PctDaysPositive21d",
            "Momentum12_1",
            "MomentumIntermediate",
            "MomentumAccelLong",
            "VolumeUpDnRatio",
            "MovingAvgAlignment",
            "DaysAboveSMA20",
            "UpDnVolumeRatio63d",
            "ROC63",
            "VolCompression5d63d",
            "ConsolidationTightness10d",
            "MaxSingleDayReturn21d",
            "GapUpMagnitude10d",
            "TrendLinearity63d",
            "DaysSince52wkHigh",
            "NormMomentum5d", "NormMomentum21d", "NormMomentum63d",
            "MomentumAccel5_21", "MomentumAccel21_63",
            "VolumeConfirmedMomentum",
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

    # PriceVs52wkHighPct — % below 52-week high (0 = at high, negative = below)
    try:
        high_52wk = close.iloc[-252:].max() if n >= 252 else close.max()
        result["PriceVs52wkHighPct"] = (last_close - high_52wk) / high_52wk * 100 if high_52wk > 0 else np.nan
    except Exception:
        result["PriceVs52wkHighPct"] = np.nan

    # PriceVs5yrHighPct — fraction below 5-year high (0 = at peak, -0.96 = 96% below peak).
    # Stored as a fraction (not ×100) to match the training dataset convention in dataset.py.
    # Falls back to full available history for tickers with < 5 years of data.
    try:
        high_5yr = close.iloc[-1260:].max() if n >= 1260 else close.max()
        result["PriceVs5yrHighPct"] = (last_close - high_5yr) / high_5yr if high_5yr > 0 else np.nan
    except Exception:
        result["PriceVs5yrHighPct"] = np.nan

    # PctDaysPositive21d — % of last 21 trading days that closed up
    try:
        daily_rets = close.pct_change().dropna()
        recent = daily_rets.iloc[-21:] if len(daily_rets) >= 21 else daily_rets
        result["PctDaysPositive21d"] = float((recent > 0).mean()) if len(recent) > 0 else np.nan
    except Exception:
        result["PctDaysPositive21d"] = np.nan

    # VolumeUpDnRatio — up-day volume / down-day volume over last 21 days.
    # >1 means more volume on up days (healthy trend); <1 = distribution (warning signal).
    try:
        if n >= 22 and len(volume) >= 22:
            rets_21 = close.pct_change().dropna().iloc[-21:]
            vol_21  = volume.iloc[-len(rets_21):]
            up_vol  = vol_21[rets_21.values > 0].sum()
            dn_vol  = vol_21[rets_21.values <= 0].sum()
            result["VolumeUpDnRatio"] = float(up_vol / dn_vol) if dn_vol > 0 else 2.0
        else:
            result["VolumeUpDnRatio"] = np.nan
    except Exception:
        result["VolumeUpDnRatio"] = np.nan

    # MovingAvgAlignment — ordinal 0-3 trend health (Price > SMA20 > SMA50 > SMA200)
    # 3 = full bull trend, 0 = broken/below all MAs. Encodes the ordering relationship
    # between MAs that price_to_sma* ratios cannot capture.
    try:
        sma20_v = result.get("SMA20", np.nan)
        sma50_v = result.get("SMA50", np.nan)
        sma200_v = result.get("SMA200", np.nan)
        align = 0
        if not pd.isna(sma20_v) and last_close > sma20_v:
            align = 1
            if not pd.isna(sma50_v) and sma20_v > sma50_v:
                align = 2
                if not pd.isna(sma200_v) and sma50_v > sma200_v:
                    align = 3
        result["MovingAvgAlignment"] = float(align)
    except Exception:
        result["MovingAvgAlignment"] = np.nan

    # DaysAboveSMA20 — consecutive trading days price has closed above 20-day SMA.
    # A 45-day run is a different regime than a 2-day cross, even if price_to_sma20 looks the same.
    try:
        if n >= 20:
            sma20_series = close.rolling(20).mean()
            above = (close > sma20_series).values
            count = 0
            for i in range(len(above) - 1, -1, -1):
                if above[i]:
                    count += 1
                else:
                    break
            result["DaysAboveSMA20"] = float(count)
        else:
            result["DaysAboveSMA20"] = np.nan
    except Exception:
        result["DaysAboveSMA20"] = np.nan

    # UpDnVolumeRatio63d — up-day vol / down-day vol over trailing 63 trading days.
    # 63d window captures institutional accumulation patterns that the 21d version misses.
    try:
        if n >= 64 and len(volume) >= 64:
            rets_63 = close.pct_change().dropna().iloc[-63:]
            vol_63 = volume.iloc[-len(rets_63):]
            up_vol = vol_63[rets_63.values > 0].sum()
            dn_vol = vol_63[rets_63.values <= 0].sum()
            result["UpDnVolumeRatio63d"] = float(up_vol / dn_vol) if dn_vol > 0 else 2.0
        else:
            result["UpDnVolumeRatio63d"] = np.nan
    except Exception:
        result["UpDnVolumeRatio63d"] = np.nan

    # ROC63 — 63-day rate of change (needed for norm_momentum_63d, momentum_accel_21_63)
    try:
        result["ROC63"] = (close.iloc[-1] - close.iloc[-64]) / close.iloc[-64] * 100 if n >= 64 else np.nan
    except Exception:
        result["ROC63"] = np.nan

    # VolCompression5d63d — 5d vol / 63d vol; < 0.5 signals a volatility squeeze
    try:
        log_rets = np.log(close / close.shift(1)).dropna()
        _v5  = float(log_rets.iloc[-5:].std()  * np.sqrt(252)) if len(log_rets) >= 5  else np.nan
        _v63 = float(log_rets.iloc[-63:].std() * np.sqrt(252)) if len(log_rets) >= 63 else np.nan
        result["VolCompression5d63d"] = (_v5 / _v63) if (not pd.isna(_v63) and _v63 > 0) else np.nan
    except Exception:
        result["VolCompression5d63d"] = np.nan

    # ConsolidationTightness10d — (10d high - 10d low) / close; lower = tighter base
    try:
        if n >= 10 and len(high) >= 10 and len(low) >= 10:
            common_idx = close.index
            h10 = high.reindex(common_idx).iloc[-10:].max()
            l10 = low.reindex(common_idx).iloc[-10:].min()
            result["ConsolidationTightness10d"] = float((h10 - l10) / last_close) if last_close > 0 else np.nan
        else:
            result["ConsolidationTightness10d"] = np.nan
    except Exception:
        result["ConsolidationTightness10d"] = np.nan

    # MaxSingleDayReturn21d — largest single-day % gain in trailing 21 days (catalyst detector)
    try:
        daily_rets = close.pct_change().dropna()
        result["MaxSingleDayReturn21d"] = float(daily_rets.iloc[-21:].max()) if len(daily_rets) >= 21 else np.nan
    except Exception:
        result["MaxSingleDayReturn21d"] = np.nan

    # GapUpMagnitude10d — largest gap-up (open above prior day's high) in trailing 10 days
    try:
        open_prices = ohlcv_df["Open"].dropna()
        if len(open_prices) >= 11 and len(high) >= 11:
            common_idx = close.index
            open_aligned = open_prices.reindex(common_idx)
            high_aligned = high.reindex(common_idx)
            gaps = (open_aligned - high_aligned.shift(1)) / high_aligned.shift(1)
            result["GapUpMagnitude10d"] = float(gaps.iloc[-10:].max())
        else:
            result["GapUpMagnitude10d"] = np.nan
    except Exception:
        result["GapUpMagnitude10d"] = np.nan

    # TrendLinearity63d — R² of 63-day linear price trend; near 1 = smooth institutional accumulation
    try:
        if n >= 63:
            window = close.iloc[-63:].values.astype(np.float64)
            t = np.arange(63, dtype=np.float64)
            r = np.corrcoef(window, t)[0, 1]
            result["TrendLinearity63d"] = float(r ** 2) if not np.isnan(r) else np.nan
        else:
            result["TrendLinearity63d"] = np.nan
    except Exception:
        result["TrendLinearity63d"] = np.nan

    # DaysSince52wkHigh — trading days since 52-week high, normalized by 252
    try:
        window_c = close.iloc[-252:] if n >= 252 else close
        argmax_pos = int(window_c.values.argmax())
        result["DaysSince52wkHigh"] = float(len(window_c) - 1 - argmax_pos) / 252.0
    except Exception:
        result["DaysSince52wkHigh"] = np.nan

    # Derived momentum features — computed from already-computed indicators above
    _roc5  = result.get("ROC5",  np.nan)
    _roc21 = result.get("ROC21", np.nan)
    _roc63 = result.get("ROC63", np.nan)
    _vol20 = result.get("AnnualizedVol", np.nan)

    # NormMomentum*d — return / volatility (risk-adjusted momentum, cross-sectionally comparable)
    result["NormMomentum5d"]  = (_roc5  / _vol20) if (not pd.isna(_vol20) and _vol20 > 0) else np.nan
    result["NormMomentum21d"] = (_roc21 / _vol20) if (not pd.isna(_vol20) and _vol20 > 0) else np.nan
    result["NormMomentum63d"] = (_roc63 / _vol20) if (not pd.isna(_vol20) and _vol20 > 0) else np.nan

    # MomentumAccel5_21 / 21_63 — short/medium ratio; > 1 = accelerating momentum
    def _accel(num, denom):
        if pd.isna(num) or pd.isna(denom) or abs(denom) < 0.01:
            return np.nan
        return float(np.clip(num / denom, -5.0, 5.0))

    result["MomentumAccel5_21"]  = _accel(_roc5,  _roc21)
    result["MomentumAccel21_63"] = _accel(_roc21, _roc63)

    # VolumeConfirmedMomentum — roc_21 × volume_surge (only counts if volume is elevated)
    try:
        if n >= 21 and len(volume) >= 21:
            avg_5d  = float(volume.iloc[-5:].mean())
            avg_20d = float(volume.iloc[-20:].mean())
            vol_surge_local = (avg_5d / avg_20d * 100) if avg_20d > 0 else np.nan
            result["VolumeConfirmedMomentum"] = (
                float(_roc21 * vol_surge_local)
                if (not pd.isna(_roc21) and not pd.isna(vol_surge_local))
                else np.nan
            )
        else:
            result["VolumeConfirmedMomentum"] = np.nan
    except Exception:
        result["VolumeConfirmedMomentum"] = np.nan

    # Long-horizon momentum (academic factors — require 252 days of history)
    # Momentum12_1: Jegadeesh-Titman — return from month -12 to -1 (skip last month)
    # MomentumIntermediate: Novy-Marx — return from month -12 to -7
    # MomentumAccelLong: 3-month ROC minus Momentum12_1 (acceleration vs trend)
    try:
        if n >= 252:
            p_252 = float(close.iloc[-252])   # ~12 months ago
            p_21  = float(close.iloc[-21])    # ~1 month ago (skip)
            p_147 = float(close.iloc[-147])   # ~7 months ago
            result["Momentum12_1"] = (p_21 / p_252 - 1) * 100 if p_252 > 0 and p_21 > 0 else np.nan
            result["MomentumIntermediate"] = (p_147 / p_252 - 1) * 100 if p_252 > 0 and p_147 > 0 else np.nan
            if n >= 63:
                p_63 = float(close.iloc[-63])
                roc_63 = (last_close / p_63 - 1) * 100 if p_63 > 0 else np.nan
                mom = result.get("Momentum12_1", np.nan)
                result["MomentumAccelLong"] = (roc_63 - mom) if (not pd.isna(roc_63) and not pd.isna(mom)) else np.nan
            else:
                result["MomentumAccelLong"] = np.nan
        else:
            result["Momentum12_1"] = np.nan
            result["MomentumIntermediate"] = np.nan
            result["MomentumAccelLong"] = np.nan
    except Exception:
        result["Momentum12_1"] = np.nan
        result["MomentumIntermediate"] = np.nan
        result["MomentumAccelLong"] = np.nan

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
