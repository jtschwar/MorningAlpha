#!/usr/bin/env python3
"""
API functions for stock analysis - separated from CLI.
These functions can be imported and used programmatically.
"""

from morningalpha.spread.metrics import calculate_all_metrics
from morningalpha.spread.indicators import compute_all_indicators
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Tuple, Optional
from datetime import date
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io, time

WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

_NASDAQ_ENDPOINTS = [
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt",
    "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt",
]

_NYSE_ENDPOINTS = [
    "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
]


def _get_with_retries(url: str, attempts: int = 4, timeout: int = 30) -> str:
    """Fetch URL with retries."""
    last = None
    for i in range(attempts):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            txt = r.text
            if "Security Name" in txt and ("Symbol|" in txt or "ACT Symbol|" in txt):
                return txt
        except Exception as e:
            last = e
            time.sleep(0.5 * (2 ** i))
    if last:
        raise last


def _get_html_with_retries(url: str, attempts: int = 4, timeout: int = 30) -> str:
    """Fetch HTML with retries."""
    last = None
    for i in range(attempts):
        try:
            r = requests.get(
                url,
                timeout=timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                                   "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
                },
            )
            r.raise_for_status()
            if "<html" in r.text.lower():
                return r.text
        except Exception as e:
            last = e
            time.sleep(0.5 * (2 ** i))
    if last:
        raise last


def read_nasdaq() -> pd.DataFrame:
    """
    Fetch NASDAQ stock listings.
    
    Returns:
        DataFrame with columns: Ticker, Name, Exchange
    """
    content = None
    for url in _NASDAQ_ENDPOINTS:
        try:
            content = _get_with_retries(url)
            break
        except Exception:
            continue
    if content is None:
        raise RuntimeError("Could not download NASDAQ symbol directory from any endpoint.")

    df = pd.read_csv(io.StringIO(content), sep="|")
    sym_col = "Symbol" if "Symbol" in df.columns else "ACT Symbol"
    name_col = "Security Name"

    df = df[df["Test Issue"] == "N"].copy()
    df["Ticker"] = df[sym_col].astype(str).str.strip().str.replace(".", "-", regex=False)
    df["Name"] = df[name_col].astype(str).str.strip()
    df["Exchange"] = "NASDAQ"

    df = df[df["Ticker"].str.fullmatch(r"[A-Z0-9\-]+", na=False)]
    return df[["Ticker", "Name", "Exchange"]]


def read_nyse() -> pd.DataFrame:
    """
    Fetch NYSE stock listings.
    
    Returns:
        DataFrame with columns: Ticker, Name, Exchange
    """
    content = None
    for url in _NYSE_ENDPOINTS:
        try:
            content = _get_with_retries(url)
            break
        except Exception:
            continue
    if content is None:
        raise RuntimeError("Could not download NYSE symbol directory from any endpoint.")

    df = pd.read_csv(io.StringIO(content), sep="|")
    sym_col = "ACT Symbol" if "ACT Symbol" in df.columns else "Symbol"
    name_col = "Security Name"

    # Filter for NYSE stocks only (Exchange column should be 'N')
    df = df[df["Exchange"] == "N"].copy()
    df = df[df["Test Issue"] == "N"].copy()
    
    df["Ticker"] = df[sym_col].astype(str).str.strip().str.replace(".", "-", regex=False)
    df["Name"] = df[name_col].astype(str).str.strip()
    df["Exchange"] = "NYSE"

    df = df[df["Ticker"].str.fullmatch(r"[A-Z0-9\-]+", na=False)]
    return df[["Ticker", "Name", "Exchange"]]


def read_sp500() -> pd.DataFrame:
    """
    Fetch S&P 500 stock listings from Wikipedia.
    
    Returns:
        DataFrame with columns: Ticker, Name, Exchange
    """
    html = _get_html_with_retries(WIKI_SP500)
    # Try available parsers — lxml is fastest but html5lib is more lenient
    tables = None
    for flavor in ["lxml", "html5lib", None]:
        try:
            tables = pd.read_html(io.StringIO(html), flavor=flavor)
            break
        except Exception:
            continue
    if tables is None:
        raise RuntimeError("pd.read_html failed with all available parsers for S&P 500 page.")

    tbl = None
    for t in tables:
        cols = {c.strip() for c in t.columns.astype(str)}
        if ({"Symbol", "Security"} <= cols) or ({"Ticker symbol", "Security"} <= cols):
            tbl = t
            break
    if tbl is None:
        raise RuntimeError("Could not find S&P 500 table on the Wikipedia page.")

    if "Ticker symbol" in tbl.columns:
        tbl = tbl.rename(columns={"Ticker symbol": "Symbol"})
    out = tbl.rename(columns={"Symbol": "Ticker", "Security": "Name"}).copy()

    out["Ticker"] = (
        out["Ticker"].astype(str).str.replace(".", "-", regex=False).str.strip()
    )
    out["Name"] = out["Name"].astype(str).str.strip()
    out["Exchange"] = "S&P500"
    return out[["Ticker", "Name", "Exchange"]]


def make_universe(
    include_nasdaq: bool = True, 
    include_nyse: bool = False,
    include_sp500: bool = True
) -> pd.DataFrame:
    """
    Build stock universe from NASDAQ, NYSE, and/or S&P 500.
    
    Args:
        include_nasdaq: Include NASDAQ stocks
        include_nyse: Include NYSE stocks
        include_sp500: Include S&P 500 stocks
    
    Returns:
        DataFrame with unique stocks
    """
    frames = []

    # Priority order matters for dedup: S&P500 label > NYSE label > NASDAQ label.
    # A stock in both S&P500 and NASDAQ keeps the "S&P500" exchange tag, which
    # lets the frontend exchange filter work correctly.
    if include_sp500:
        frames.append(read_sp500())

    if include_nyse:
        frames.append(read_nyse())

    if include_nasdaq:
        frames.append(read_nasdaq())
    
    if not frames:
        raise ValueError("At least one universe must be included")
    
    uni = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Ticker"])
    uni = uni[~uni["Ticker"].str.contains(r"[\^=]", regex=True)]
    return uni.reset_index(drop=True)


def period_bounds(kind: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Calculate start and end dates for a period.
    
    Args:
        kind: One of '1wk', '2wk', '1m', '3m', '6m', 'ytd'
    
    Returns:
        Tuple of (start_date, end_date) as pd.Timestamp
    """
    today = date.today()
    
    if kind.lower() == '1wk':
        start = today - relativedelta(weeks=1)
    elif kind.lower() == '2wk':
        start = today - relativedelta(weeks=2)
    elif kind.lower() == '1m':
        start = today - relativedelta(months=1)
    elif kind.lower() == "3m":
        start = today - relativedelta(months=3)
    elif kind.lower() == "6m":
        start = today - relativedelta(months=6)
    elif kind.lower() == "ytd":
        start = date(today.year, 1, 1)
    else:
        raise ValueError("metric must be one of: 1wk, 2wk, 1m, 3m, 6m, ytd")
    return pd.Timestamp(start), pd.Timestamp(today)


def percent_change(series: pd.Series) -> float:
    """Calculate percentage change from first to last value."""
    s = series.dropna()
    if len(s) < 2:
        return float("nan")
    first = s.iloc[0]
    last = s.iloc[-1]
    if first == 0 or pd.isna(first) or pd.isna(last):
        return float("nan")
    return (last / first - 1.0) * 100.0


def batched(iterable, n: int):
    """Yield successive n-sized batches from iterable."""
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def get_market_cap(ticker: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Fetch market cap for a ticker and categorize it.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Tuple of (market_cap_value, market_cap_category)
        market_cap_value: Market cap in dollars (float) or None if unavailable
        market_cap_category: 'Large', 'Mid', 'Small', or None
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Market cap is typically in 'marketCap' field
        market_cap = info.get('marketCap')
        
        if market_cap is None:
            # Try alternative fields
            market_cap = info.get('enterpriseValue')  # Fallback
            if market_cap is None:
                return None, None
        
        # Categorize market cap
        # Large cap: > $10 billion
        # Mid cap: $2 billion - $10 billion
        # Small cap: < $2 billion
        if market_cap >= 10_000_000_000:
            category = 'Large'
        elif market_cap >= 2_000_000_000:
            category = 'Mid'
        else:
            category = 'Small'
        
        return float(market_cap), category
        
    except Exception as e:
        # Silently fail - market cap is optional
        return None, None


def fetch_market_caps_batch(tickers: List[str], pause: float = 0.1) -> dict:
    """
    Fetch market caps for a batch of tickers.
    
    Args:
        tickers: List of ticker symbols
        pause: Seconds to pause between requests
    
    Returns:
        Dict mapping ticker -> (market_cap, category)
    """
    results = {}
    
    for ticker in tickers:
        market_cap, category = get_market_cap(ticker)
        results[ticker] = (market_cap, category)
        time.sleep(pause)  # Rate limiting
    
    return results


def fetch_returns_with_metrics(
    tickers: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    pause: float = 0.8,
    batch_size: int = 200,
    progress_callback: Optional[callable] = None,
    fetch_market_cap: bool = False
) -> pd.DataFrame:
    """
    Fetch stock returns AND calculate metrics for each stock.
    
    Args:
        tickers: List of ticker symbols
        start: Start date
        end: End date
        pause: Seconds to pause between batches
        batch_size: Tickers per batch
        progress_callback: Optional callback for progress updates
        fetch_market_cap: Whether to fetch market cap data (slower but more complete)
                        NOTE: Market cap fetching is slow - only use for final top stocks
    """
    results = []
    
    # Cap batch size to avoid rate limits
    batch_size = min(batch_size, 512)
    
    batches = list(batched(tickers, batch_size))
    
    # Market caps will be fetched AFTER we have results (only for top stocks)
    market_caps = {}
    
    # For 3-month filter, we need at least 3 months of data to check trading history
    # So extend start date back 3 months if the requested period is shorter
    min_start_date = start - pd.Timedelta(days=90)
    actual_start = min(start, min_start_date)
    
    for i, batch in enumerate(batches):
        data = yf.download(
            tickers=" ".join(batch),
            start=actual_start.tz_localize(None),
            end=end.tz_localize(None) + pd.Timedelta(days=1),
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        
        if isinstance(data.columns, pd.MultiIndex):
            close_cols = data.xs("Close", axis=1, level=1, drop_level=False)
            volume_cols = data.xs("Volume", axis=1, level=1, drop_level=False)
            high_cols = data.xs("High", axis=1, level=1, drop_level=False) if "High" in data.columns.get_level_values(1) else None
            low_cols = data.xs("Low", axis=1, level=1, drop_level=False) if "Low" in data.columns.get_level_values(1) else None

            for t in batch:
                try:
                    prices = close_cols[(t, "Close")]
                    volumes = volume_cols[(t, "Volume")]
                    
                    # Filter out stocks with insufficient data
                    if len(prices) < 2:
                        continue
                    
                    # Filter out stocks with less than 3 months of trading history
                    # Check date span - need at least 90 calendar days (~60 trading days)
                    price_dates = prices.index
                    if len(price_dates) > 0:
                        days_span = (price_dates[-1] - price_dates[0]).days
                        if days_span < 90:
                            continue
                    
                    # Filter to only the requested period for calculations
                    period_prices = prices[prices.index >= start]
                    period_volumes = volumes[volumes.index >= start]
                    
                    if len(period_prices) < 2:
                        continue
                    
                    # Calculate basic return for the requested period
                    return_pct = percent_change(period_prices)
                    
                    # Calculate daily returns for metrics (use full period for better metrics)
                    daily_returns = prices.pct_change(fill_method=None).dropna()
                    
                    # Calculate all metrics (use full available data for better accuracy)
                    metrics = calculate_all_metrics(prices, volumes, daily_returns)

                    # Compute technical indicators from OHLCV data
                    try:
                        ohlcv = pd.DataFrame({
                            'Close': prices,
                            'High': high_cols[(t, "High")] if high_cols is not None else pd.Series(np.nan, index=prices.index),
                            'Low': low_cols[(t, "Low")] if low_cols is not None else pd.Series(np.nan, index=prices.index),
                            'Volume': volumes,
                        })
                        indicator_vals = compute_all_indicators(ohlcv)
                    except Exception:
                        indicator_vals = {}

                    # Get market cap if available
                    market_cap, market_cap_category = market_caps.get(t, (None, None))

                    # Store everything
                    results.append({
                        "Ticker": t,
                        "ReturnPct": return_pct,
                        "SharpeRatio": metrics.get('sharpe_ratio', np.nan),
                        "SortinoRatio": metrics.get('sortino_ratio', np.nan),
                        "MaxDrawdown": metrics.get('max_drawdown', np.nan),
                        "ConsistencyScore": metrics.get('consistency_score', np.nan),
                        "VolumeTrend": metrics.get('volume_trend', np.nan),
                        "QualityScore": metrics.get('quality_score', np.nan),
                        # NEW: Short-term holding metrics
                        "RSI": metrics.get('rsi', np.nan),
                        "MomentumAccel": metrics.get('momentum_acceleration', np.nan),
                        "PriceVs20dHigh": metrics.get('price_vs_20d_high', np.nan),
                        "VolumeSurge": metrics.get('volume_surge', np.nan),
                        "EntryScore": metrics.get('entry_score', np.nan),
                        "MarketCap": market_cap,
                        "MarketCapCategory": market_cap_category,
                        **indicator_vals,
                    })
                except KeyError:
                    continue
        else:
            # Handle single ticker case
            prices = data.get("Close")
            volumes = data.get("Volume")
            if prices is not None:
                # Filter out stocks with insufficient data
                if len(prices) < 2:
                    continue
                
                # Filter out stocks with less than 3 months of trading history
                price_dates = prices.index
                if len(price_dates) > 0:
                    days_span = (price_dates[-1] - price_dates[0]).days
                    if days_span < 90:
                        continue
                
                # Filter to only the requested period for return calculation
                period_prices = prices[prices.index >= start]
                if len(period_prices) < 2:
                    continue
                
                return_pct = percent_change(period_prices)
                daily_returns = prices.pct_change().dropna()
                metrics = calculate_all_metrics(prices, volumes, daily_returns)

                # Compute technical indicators from OHLCV data
                try:
                    ohlcv = pd.DataFrame({
                        'Close': prices,
                        'High': data.get("High", pd.Series(np.nan, index=prices.index)),
                        'Low': data.get("Low", pd.Series(np.nan, index=prices.index)),
                        'Volume': volumes,
                    })
                    indicator_vals = compute_all_indicators(ohlcv)
                except Exception:
                    indicator_vals = {}

                # Get market cap if available
                market_cap, market_cap_category = market_caps.get(batch[0], (None, None))

                results.append({
                    "Ticker": batch[0],
                    "ReturnPct": return_pct,
                    "SharpeRatio": metrics.get('sharpe_ratio', np.nan),
                    "SortinoRatio": metrics.get('sortino_ratio', np.nan),
                    "MaxDrawdown": metrics.get('max_drawdown', np.nan),
                    "ConsistencyScore": metrics.get('consistency_score', np.nan),
                    "VolumeTrend": metrics.get('volume_trend', np.nan),
                    "QualityScore": metrics.get('quality_score', np.nan),
                    # NEW: Short-term holding metrics
                    "RSI": metrics.get('rsi', np.nan),
                    "MomentumAccel": metrics.get('momentum_acceleration', np.nan),
                    "PriceVs20dHigh": metrics.get('price_vs_20d_high', np.nan),
                    "VolumeSurge": metrics.get('volume_surge', np.nan),
                    "EntryScore": metrics.get('entry_score', np.nan),
                    "MarketCap": market_cap,
                    "MarketCapCategory": market_cap_category,
                    **indicator_vals,
                })
        
        if progress_callback:
            progress_callback(i + 1, len(batches), "Downloading stock data...")
        
        time.sleep(pause)
    
    return pd.DataFrame(results)


def analyze_stocks(
    universe: List[str] = None,
    include_nasdaq: bool = True,
    include_nyse: bool = True,
    include_sp500: bool = True,
    metric: str = '3m',
    top: int = 200,
    top_per_exchange: Optional[Dict[str, int]] = None,
    batch_size: int = 200,
    pause: float = 0.8,
    progress_callback: Optional[callable] = None,
    fetch_market_cap: bool = True
) -> pd.DataFrame:
    """
    High-level API to analyze top stock gainers WITH metrics.
    
    Args:
        universe: Optional list of specific tickers to analyze
        include_nasdaq: Include NASDAQ stocks
        include_nyse: Include NYSE stocks
        include_sp500: Include S&P 500 stocks
        metric: Time period ('1wk', '2wk', '1m', '3m', '6m', 'ytd')
        top: Number of top gainers to return
        batch_size: Tickers per batch
        pause: Seconds between batches
        progress_callback: Optional callback(current, total, message) for progress
        fetch_market_cap: Whether to fetch market cap data (only for top stocks, slower)
    
    Returns:
        DataFrame with top gainers, sorted by return
    """
    # Build universe
    if universe:
        uni = pd.DataFrame({
            'Ticker': universe,
            'Name': [''] * len(universe),
            'Exchange': ['Custom'] * len(universe)
        })
    else:
        if progress_callback:
            progress_callback(0, 0, "Building stock universe...")
        uni = make_universe(include_nasdaq, include_nyse, include_sp500)
    
    # Get date range
    start, end = period_bounds(metric)
    
    # Fetch returns WITH METRICS
    if progress_callback:
        progress_callback(0, 0, "Downloading stock data and calculating metrics...")
    
    rets = fetch_returns_with_metrics(
        uni["Ticker"].tolist(),
        start,
        end,
        pause=pause,
        batch_size=batch_size,
        progress_callback=lambda cur, tot, msg=None: progress_callback(cur, tot, msg or "Analyzing...") if progress_callback else None,
        fetch_market_cap=False  # Don't fetch market cap during initial analysis
    )
    
    # Process results
    # Ensure rets has the expected columns even if empty
    if rets.empty or "Ticker" not in rets.columns:
        # Create empty DataFrame with expected columns
        rets = pd.DataFrame(columns=["Ticker", "ReturnPct", "SharpeRatio", "SortinoRatio", 
                                     "MaxDrawdown", "ConsistencyScore", "VolumeTrend", 
                                     "QualityScore", "MarketCap", "MarketCapCategory"])
    
    tmp = uni.merge(rets, on="Ticker", how="left").dropna(subset=["ReturnPct"])

    if tmp.empty:
        raise ValueError("No stocks found with sufficient data (need at least 3 months of trading history)")

    # Log per-exchange breakdown so workflow output is diagnosable
    for exch, count in tmp["Exchange"].value_counts().items():
        print(f"  {exch}: {count} stocks with valid data")

    if top_per_exchange:
        # Take top N per exchange so every exchange has meaningful representation.
        parts = [
            tmp[tmp["Exchange"] == exch].nlargest(n, "ReturnPct")
            for exch, n in top_per_exchange.items()
        ]
        parts = [p for p in parts if not p.empty]
        result = (
            pd.concat(parts) if parts else tmp.head(0)
        ).drop_duplicates(subset=["Ticker"]).sort_values("ReturnPct", ascending=False).reset_index(drop=True)
    else:
        result = tmp.nlargest(top, "ReturnPct").sort_values("ReturnPct", ascending=False).reset_index(drop=True)
    
    # NOW fetch market caps only for the top stocks (much faster!)
    if fetch_market_cap:
        if progress_callback:
            progress_callback(0, 0, f"Fetching market cap for top {len(result)} stocks...")
        
        market_caps = {}
        top_tickers = result["Ticker"].tolist()
        market_cap_batch_size = 50
        
        for i in range(0, len(top_tickers), market_cap_batch_size):
            batch = top_tickers[i:i + market_cap_batch_size]
            batch_caps = fetch_market_caps_batch(batch, pause=0.05)  # Faster pause
            market_caps.update(batch_caps)
            
            if progress_callback:
                progress_callback(i + len(batch), len(top_tickers), f"Fetching market cap... ({i + len(batch)}/{len(top_tickers)})")
        
        # Add market cap data to results
        result["MarketCap"] = result["Ticker"].map(lambda t: market_caps.get(t, (None, None))[0])
        result["MarketCapCategory"] = result["Ticker"].map(lambda t: market_caps.get(t, (None, None))[1])
    else:
        # No market cap data
        result["MarketCap"] = None
        result["MarketCapCategory"] = None
    
    # Rename the return column
    result = result.rename(columns={"ReturnPct": f"Return_{metric.upper()}_%"})
    result.index = result.index + 1
    
    return result