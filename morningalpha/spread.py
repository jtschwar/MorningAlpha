#!/usr/bin/env python3
"""
Stock analyzer that generates CSV data for web visualization.
Usage: python stock_analyzer.py --metric 3m --top 200 --out stock_data.csv
"""
import sys
import time
from datetime import date
from dateutil.relativedelta import relativedelta

from rich.console import Console
import rich_click as click

console = Console()

WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

_NASDAQ_ENDPOINTS = [
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt",
    "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt",
]

# Rich-click configuration
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.STYLE_ERRORS_SUGGESTION = "magenta italic"

# Lazy imports - only loaded when command runs
def _lazy_imports():
    import io
    import pandas as pd
    import requests
    import yfinance as yf
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
    from rich.panel import Panel
    from rich.table import Table
    return io, pd, requests, yf, Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, Panel, Table

def _get_with_retries(url, attempts=4, timeout=30):
    _, _, requests, _, _, _, _, _, _, _, _ = _lazy_imports()
    last = None
    for i in range(attempts):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
            r.raise_for_status()
            txt = r.text
            if "Security Name" in txt and ("Symbol|" in txt or "ACT Symbol|" in txt):
                return txt
        except Exception as e:
            last = e
            time.sleep(0.5 * (2 ** i))
    if last:
        raise last

def read_nasdaq():
    io, pd, _, _, _, _, _, _, _, _, _ = _lazy_imports()
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
    return df[["Ticker","Name","Exchange"]]

def _get_html_with_retries(url, attempts=4, timeout=30):
    _, _, requests, _, _, _, _, _, _, _, _ = _lazy_imports()
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

def read_sp500():
    _, pd, _, _, _, _, _, _, _, _, _ = _lazy_imports()
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

def make_universe(include_nasdaq: bool, include_sp500: bool):
    _, pd, _, _, _, _, _, _, _, _, _ = _lazy_imports()
    frames = []
    
    with console.status("[bold green]Building stock universe...") as status:
        if include_nasdaq:
            status.update("[bold cyan]Fetching NASDAQ listings...")
            frames.append(read_nasdaq())
            console.log(f"✓ Loaded {len(frames[-1])} NASDAQ stocks")
        
        if include_sp500:
            status.update("[bold cyan]Fetching S&P 500 listings...")
            frames.append(read_sp500())
            console.log(f"✓ Loaded {len(frames[-1])} S&P 500 stocks")
    
    uni = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Ticker"])
    uni = uni[~uni["Ticker"].str.contains(r"[\^=]", regex=True)]
    return uni.reset_index(drop=True)

def period_bounds(kind: str):
    _, pd, _, _, _, _, _, _, _, _, _ = _lazy_imports()
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

def percent_change(series):
    _, pd, _, _, _, _, _, _, _, _, _ = _lazy_imports()
    s = series.dropna()
    if len(s) < 2:
        return float("nan")
    first = s.iloc[0]
    last = s.iloc[-1]
    if first == 0 or pd.isna(first) or pd.isna(last):
        return float("nan")
    return (last / first - 1.0) * 100.0

def batched(iterable, n):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

def fetch_returns(tickers, start, end, pause=0.8, batch_size=200):
    _, pd, _, yf, Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, _, _ = _lazy_imports()
    results = []
    batches = list(batched(tickers, batch_size))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total} batches)"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        
        task = progress.add_task(
            "[cyan]Downloading stock data...",
            total=len(batches)
        )
        
        for batch in batches:
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
            
            progress.update(task, advance=1)
            time.sleep(pause)
    
    return pd.DataFrame(results)

@click.command(name='spread')
@click.option(
    '-u', '--universe',
    multiple=True,
    type=click.Choice(['nasdaq', 'sp500'], case_sensitive=False),
    default=['nasdaq', 'sp500'],
    help='Stock universes to include (can specify multiple)',
    show_default=True
)
@click.option(
    '-m', '--metric',
    type=click.Choice(['1wk', '2wk', '1m', '3m', '6m', 'ytd'], case_sensitive=False),
    default='3m',
    help='Return window to rank stocks',
    show_default=True
)
@click.option(
    '-t', '--top',
    type=int,
    default=200,
    help='Number of top gainers to save',
    show_default=True
)
@click.option(
    '-o', '--out',
    type=click.Path(),
    default='top_gainers.csv',
    help='Output CSV filename',
    show_default=True
)
@click.option(
    '-bs', '--batch-size',
    type=int,
    default=200,
    help='Tickers per yfinance batch',
    show_default=True
)
@click.option(
    '-p', '--pause',
    type=float,
    default=0.8,
    help='Seconds to sleep between batches',
    show_default=True
)
def main(universe, metric, top, out, batch_size, pause):
    """
    [bold cyan]📈 Stock Gainers Analyzer[/bold cyan]
    
    Analyze top stock gainers from NASDAQ and S&P 500.
    Outputs a CSV file ready for web visualization.
    
    [bold]Examples:[/bold]
    
      $ python stock_analyzer.py --metric 3m --top 50
      
      $ python stock_analyzer.py --universe nasdaq --metric ytd --top 100
      
      $ python stock_analyzer.py --batch-size 100 --pause 1.0 --out my_stocks.csv
    """
    _, _, _, _, _, _, _, _, _, Panel, Table = _lazy_imports()
    
    console.print(Panel.fit(
        "[bold cyan]📊 Stock Gainers Analyzer[/bold cyan]\n"
        "[dim]Powered by yfinance[/dim]",
        border_style="cyan"
    ))
    
    universe_list = list(universe) if universe else ['nasdaq', 'sp500']
    include_nasdaq = 'nasdaq' in universe_list
    include_sp500 = 'sp500' in universe_list

    uni = make_universe(include_nasdaq, include_sp500)
    console.print(f"\n[bold green]✓ Built universe: {len(uni)} stocks[/bold green]")

    start, end = period_bounds(metric)
    console.print(f"[bold]Period:[/bold] {start.date()} to {end.date()} ({metric.upper()})\n")

    rets = fetch_returns(
        uni["Ticker"].tolist(), 
        start, 
        end,
        pause=pause, 
        batch_size=batch_size
    )

    with console.status("[bold cyan]Processing results..."):
        tmp = uni.merge(rets, on="Ticker", how="left").dropna(subset=["ReturnPct"])
        topn = tmp.nlargest(top, "ReturnPct")
        result = topn.sort_values("ReturnPct", ascending=False).reset_index(drop=True)
        
        result = result.rename(columns={"ReturnPct": f"Return_{metric.upper()}_%"})
        result.index = result.index + 1
        result.to_csv(out, index_label="Rank")
    
    console.print(f"[bold green]✓ Saved {len(result)} stocks to {out}[/bold green]\n")
    
    table = Table(title="Top 5 Gainers", show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="dim")
    table.add_column("Ticker", style="cyan")
    table.add_column("Company", style="white")
    table.add_column("Return %", justify="right", style="green")
    
    for idx, row in result.head().iterrows():
        table.add_row(
            str(idx),
            row['Ticker'],
            row['Name'][:40] + "..." if len(row['Name']) > 40 else row['Name'],
            f"{row[f'Return_{metric.upper()}_%']:.2f}%"
        )
    
    console.print(table)
    
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  • Total analyzed: {len(uni)} stocks")
    console.print(f"  • Valid data: {len(tmp)} stocks")
    console.print(f"  • Top {len(result)} saved to [cyan]{out}[/cyan]")
    console.print(f"  • Average return (top {len(result)}): [green]{result[f'Return_{metric.upper()}_%'].mean():.2f}%[/green]\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]ERROR: {e}[/bold red]")
        import traceback
        console.print("[dim]" + traceback.format_exc() + "[/dim]")
        sys.exit(1)