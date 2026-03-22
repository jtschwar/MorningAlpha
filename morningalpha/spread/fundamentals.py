"""
Fetches fundamental metrics for the stock universe via yfinance.Ticker.info.
"""
import click
import yfinance as yf
import pandas as pd
import numpy as np
import time
from typing import List

FUNDAMENTAL_FIELDS = {
    'Sector': ('sector', str),
    'Industry': ('industry', str),
    'MarketCap': ('marketCap', float),
    'PE': ('trailingPE', float),
    'ForwardPE': ('forwardPE', float),
    'PB': ('priceToBook', float),
    'PS': ('priceToSalesTrailing12Months', float),
    'PEG': ('pegRatio', float),
    'EPS': ('trailingEps', float),
    'RevenueGrowth': ('revenueGrowth', float),
    'EarningsGrowth': ('earningsGrowth', float),
    'ROE': ('returnOnEquity', float),
    'ROA': ('returnOnAssets', float),
    'GrossMargin': ('grossMargins', float),
    'OperatingMargin': ('operatingMargins', float),
    'NetMargin': ('profitMargins', float),
    'DebtEquity': ('debtToEquity', float),
    'CurrentRatio': ('currentRatio', float),
    'DivYield': ('dividendYield', float),
    'Beta': ('beta', float),
    'ShortFloat': ('shortPercentOfFloat', float),
    'InstOwnership': ('heldPercentInstitutions', float),
}


def fetch_fundamentals(
    tickers: List[str],
    out_path: str,
    batch_size: int = 50,
    pause: float = 1.0,
) -> pd.DataFrame:
    """
    Fetch fundamental metrics for a list of tickers and save to CSV.

    Args:
        tickers: List of ticker symbols.
        out_path: File path for the output CSV.
        batch_size: Number of tickers to process before pausing.
        pause: Seconds to sleep between batches.

    Returns:
        DataFrame with one row per ticker and columns matching FUNDAMENTAL_FIELDS.
    """
    rows = []

    for i, t in enumerate(tickers):
        try:
            info = yf.Ticker(t).info
            row = {"Ticker": t}
            for col_name, (info_key, cast) in FUNDAMENTAL_FIELDS.items():
                raw = info.get(info_key)
                if raw is None:
                    row[col_name] = np.nan if cast is float else None
                else:
                    try:
                        row[col_name] = cast(raw)
                    except (ValueError, TypeError):
                        row[col_name] = np.nan if cast is float else None
            rows.append(row)
        except Exception as e:
            print(f"  [WARN] Failed to fetch {t}: {e}")
            continue

        # Pause between batches
        if (i + 1) % batch_size == 0:
            time.sleep(pause)

    col_order = ["Ticker"] + list(FUNDAMENTAL_FIELDS.keys())
    df = pd.DataFrame(rows, columns=col_order) if rows else pd.DataFrame(columns=col_order)

    df.to_csv(out_path, index=False)
    return df


@click.command('fundamentals')
@click.option('--out', default='data/latest/fundamentals.csv', show_default=True,
              help='Output CSV path')
@click.option('--batch-size', default=50, show_default=True,
              help='Number of tickers per batch before pausing')
@click.option('--pause', default=1.0, show_default=True,
              help='Seconds to sleep between batches')
@click.option('--universe', multiple=True, default=['nasdaq', 'nyse'], show_default=True,
              type=click.Choice(['nasdaq', 'nyse', 'sp500'], case_sensitive=False))
def fundamentals(out, batch_size, pause, universe):
    """Fetch fundamental metrics for the stock universe."""
    from morningalpha.spread.search import make_universe

    include_nasdaq = 'nasdaq' in [u.lower() for u in universe]
    include_nyse = 'nyse' in [u.lower() for u in universe]
    include_sp500 = 'sp500' in [u.lower() for u in universe]

    uni = make_universe(
        include_nasdaq=include_nasdaq,
        include_nyse=include_nyse,
        include_sp500=include_sp500,
    )

    tickers = uni["Ticker"].tolist()
    print(f"Fetching fundamentals for {len(tickers)} tickers...")

    df = fetch_fundamentals(tickers, out_path=out, batch_size=batch_size, pause=pause)

    print(f"Saved {len(df)} rows to {out}")
