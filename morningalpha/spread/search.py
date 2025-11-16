#!/usr/bin/env python3
"""
API functions for stock analysis - separated from CLI.
These functions can be imported and used programmatically.
"""
import time
from datetime import date
from dateutil.relativedelta import relativedelta
import io
import pandas as pd
import requests
import yfinance as yf
from typing import List, Tuple, Optional

WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

_NASDAQ_ENDPOINTS = [
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt",
    "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt",
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


def read_sp500() -> pd.DataFrame:
    """
    Fetch S&P 500 stock listings from Wikipedia.
    
    Returns:
        DataFrame with columns: Ticker, Name, Exchange
    """
    html = _get_html_with_retries(WIKI_SP500)
    tables = pd.read_html(html, flavor="lxml")

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


def make_universe(include_nasdaq: bool = True, include_sp500: bool = True) -> pd.DataFrame:
    """
    Build stock universe from NASDAQ and/or S&P 500.
    
    Args:
        include_nasdaq: Include NASDAQ stocks
        include_sp500: Include S&P 500 stocks
    
    Returns:
        DataFrame with unique stocks
    """
    frames = []
    
    if include_nasdaq:
        frames.append(read_nasdaq())
    
    if include_sp500:
        frames.append(read_sp500())
    
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


def fetch_returns(
    tickers: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    pause: float = 0.8,
    batch_size: int = 200,
    progress_callback: Optional[callable] = None
) -> pd.DataFrame:
    """
    Fetch stock returns for a list of tickers.
    
    Args:
        tickers: List of stock tickers
        start: Start date
        end: End date
        pause: Seconds to pause between batches
        batch_size: Number of tickers per batch
        progress_callback: Optional callback(current, total) for progress
    
    Returns:
        DataFrame with columns: Ticker, ReturnPct
    """
    results = []
    batches = list(batched(tickers, batch_size))
    
    for i, batch in enumerate(batches):
        data = yf.download(
            tickers=" ".join(batch),
            start=start.tz_localize(None),
            end=end.tz_localize(None) + pd.Timedelta(days=1),
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        
        if isinstance(data.columns, pd.MultiIndex):
            close_cols = data.xs("Close", axis=1, level=1, drop_level=False)
            for t in batch:
                try:
                    ser = close_cols[(t, "Close")]
                except KeyError:
                    continue
                results.append({"Ticker": t, "ReturnPct": percent_change(ser)})
        else:
            ser = data.get("Close")
            if ser is not None:
                results.append({"Ticker": batch[0], "ReturnPct": percent_change(ser)})
        
        if progress_callback:
            progress_callback(i + 1, len(batches))
        
        time.sleep(pause)
    
    return pd.DataFrame(results)


def analyze_stocks(
    universe: List[str] = None,
    include_nasdaq: bool = True,
    include_sp500: bool = True,
    metric: str = '3m',
    top: int = 200,
    batch_size: int = 200,
    pause: float = 0.8,
    progress_callback: Optional[callable] = None
) -> pd.DataFrame:
    """
    High-level API to analyze top stock gainers.
    
    Args:
        universe: Optional list of specific tickers to analyze
        include_nasdaq: Include NASDAQ stocks
        include_sp500: Include S&P 500 stocks
        metric: Time period ('1wk', '2wk', '1m', '3m', '6m', 'ytd')
        top: Number of top gainers to return
        batch_size: Tickers per batch
        pause: Seconds between batches
        progress_callback: Optional callback(current, total, message) for progress
    
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
        uni = make_universe(include_nasdaq, include_sp500)
    
    # Get date range
    start, end = period_bounds(metric)
    
    # Fetch returns
    if progress_callback:
        progress_callback(0, 0, "Downloading stock data...")
    
    rets = fetch_returns(
        uni["Ticker"].tolist(),
        start,
        end,
        pause=pause,
        batch_size=batch_size,
        progress_callback=lambda cur, tot: progress_callback(cur, tot, "Downloading stock data...") if progress_callback else None
    )
    
    # Process results
    tmp = uni.merge(rets, on="Ticker", how="left").dropna(subset=["ReturnPct"])
    topn = tmp.nlargest(top, "ReturnPct")
    result = topn.sort_values("ReturnPct", ascending=False).reset_index(drop=True)
    
    result = result.rename(columns={"ReturnPct": f"Return_{metric.upper()}_%"})
    result.index = result.index + 1
    
    return result

