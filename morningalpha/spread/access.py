#!/usr/bin/env python3
"""
CLI command for stock spread analysis.
"""

from morningalpha import cli_context
import rich_click as click

# Rich-click configuration
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.STYLE_ERRORS_SUGGESTION = "magenta italic"

@click.command(name='spread', context_settings=cli_context)
@click.option(
    '-u', '--universe',
    multiple=True,
    type=click.Choice(['nasdaq', 'nyse', 'sp500'], case_sensitive=False),
    default=['nasdaq', 'nyse', 'sp500'],
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
    '--output-dir',
    type=click.Path(),
    default=None,
    help='Output directory — runs all 4 periods and saves stocks_2w/1m/3m/6m.csv',
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
@click.option(
    '--no-market-cap',
    is_flag=True,
    default=False,
    help='Skip market cap fetching (faster, but no market cap data)',
    show_default=True
)
@click.option('--top-nasdaq', type=int, default=250, show_default=True,
              help='Max NASDAQ stocks when using --output-dir')
@click.option('--top-nyse', type=int, default=100, show_default=True,
              help='Max NYSE stocks when using --output-dir')
@click.option('--top-sp500', type=int, default=100, show_default=True,
              help='Max S&P500 stocks when using --output-dir')
def spread(universe, metric, top, out, batch_size, pause, no_market_cap, output_dir,
           top_nasdaq, top_nyse, top_sp500):
    """
    [bold cyan]📈 Stock Gainers Analyzer[/bold cyan]

    Analyze top stock gainers from NASDAQ, NYSE, and S&P 500.
    Outputs a CSV file ready for web visualization.

    [bold]Examples:[/bold]

      $ morningalpha spread --metric 3m --top 50

      $ morningalpha spread --universe nasdaq --universe nyse --metric ytd --top 100

      $ morningalpha spread --universe sp500 --metric 6m --top 200

      $ morningalpha spread --batch-size 100 --pause 1.0 --out my_stocks.csv

      $ morningalpha spread --output-dir data/latest  # Run all 4 periods
    """
    if output_dir:
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        per_exchange = {'NASDAQ': top_nasdaq, 'NYSE': top_nyse, 'S&P500': top_sp500}
        period_map = [('2wk', '2w'), ('1m', '1m'), ('3m', '3m'), ('6m', '6m')]
        for metric_code, file_suffix in period_map:
            dest = str(Path(output_dir) / f'stocks_{file_suffix}.csv')
            get_spread(universe, metric_code, top, dest, batch_size, pause, no_market_cap,
                       top_per_exchange=per_exchange)
    else:
        get_spread(universe, metric, top, out, batch_size, pause, no_market_cap)

def get_spread(universe, metric, top, out, batch_size, pause, no_market_cap, top_per_exchange=None):
    """
    Execute the spread analysis.
    """
    from morningalpha.spread.search import analyze_stocks 
    from morningalpha.spread.search import make_universe
    from morningalpha.spread.search import period_bounds
    
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    
    console.print(Panel.fit(
        "[bold cyan]📊 Stock Gainers Analyzer[/bold cyan]\n"
        "[dim]Powered by yfinance[/dim]",
        border_style="cyan"
    ))
    
    universe_list = list(universe) if universe else ['nasdaq', 'sp500']
    include_nasdaq = 'nasdaq' in universe_list
    include_nyse = 'nyse' in universe_list
    include_sp500 = 'sp500' in universe_list

    # Build universe with progress
    with console.status("[bold green]Building stock universe...") as status:
        if include_nasdaq:
            status.update("[bold cyan]Fetching NASDAQ listings...")
        if include_nyse:
            status.update("[bold cyan]Fetching NYSE listings...")
        if include_sp500:
            status.update("[bold cyan]Fetching S&P 500 listings...")
        
        uni = make_universe(include_nasdaq, include_nyse, include_sp500)
        console.log(f"✓ Built universe: {len(uni)} stocks")

    start, end = period_bounds(metric)
    console.print(f"[bold]Period:[/bold] {start.date()} to {end.date()} ({metric.upper()})\n")

    # Progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total} batches)"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        
        task = progress.add_task("[cyan]Downloading stock data...", total=100)
        
        def progress_callback(current, total, message):
            if total > 0:
                progress.update(task, completed=current, total=total, description=f"[cyan]{message}")
        
        # Use the API function
        result = analyze_stocks(
            include_nasdaq=include_nasdaq,
            include_nyse=include_nyse,
            include_sp500=include_sp500,
            metric=metric,
            top=top,
            top_per_exchange=top_per_exchange,
            batch_size=batch_size,
            pause=pause,
            progress_callback=progress_callback,
            fetch_market_cap=not no_market_cap
        )

    # Save results
    with console.status("[bold cyan]Saving results..."):
        result.to_csv(out, index_label="Rank")
    
    console.print(f"[bold green]✓ Saved {len(result)} stocks to {out}[/bold green]\n")
    
    # Display preview table
    table = Table(title="Top 5 Gainers", show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="dim")
    table.add_column("Ticker", style="cyan")
    table.add_column("Company", style="white")
    table.add_column("Return %", justify="right", style="green")
    
    return_col = f"Return_{metric.upper()}_%"
    for idx, row in result.head().iterrows():
        table.add_row(
            str(idx),
            row['Ticker'],
            row['Name'][:40] + "..." if len(row['Name']) > 40 else row['Name'],
            f"{row[return_col]:.2f}%"
        )
    
    console.print(table)
    
    # Summary statistics
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  • Total analyzed: {len(uni)} stocks")
    console.print(f"  • Valid data: {len(result)} stocks")
    console.print(f"  • Top {len(result)} saved to [cyan]{out}[/cyan]")
    console.print(f"  • Average return (top {len(result)}): [green]{result[return_col].mean():.2f}%[/green]\n")