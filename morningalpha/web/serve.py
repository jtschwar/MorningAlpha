#!/usr/bin/env python3
"""
CLI command to launch the morningalpha proxy server and web app.
"""
from pathlib import Path
from morningalpha import cli_context
import rich_click as click

WEBAPP_DIR = Path(__file__).parent
DATA_DIR = WEBAPP_DIR / "public" / "data" / "latest"
# Repo-root data/latest/ — populated by `alpha spread` or `git pull` after daily action runs
LOCAL_DATA_DIR = WEBAPP_DIR.parent.parent / "data" / "latest"
PAGES_BASE = "https://jtschwar.github.io/MorningAlpha"
CSV_PERIODS = ["2w", "1m", "3m", "6m"]


def _last_market_close():
    """Return the datetime of the most recent weekday 4:15pm ET."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    ET = ZoneInfo("America/New_York")
    now = datetime.now(ET)
    d = now.date()
    # Walk back to the most recent weekday
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    close = datetime(d.year, d.month, d.day, 16, 15, tzinfo=ET)
    # If that close is still in the future (e.g. running before 4:15pm today),
    # step back one more weekday
    if close > now:
        d -= timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        close = datetime(d.year, d.month, d.day, 16, 15, tzinfo=ET)
    return close


def _is_current(path: Path) -> bool:
    """Return True if path exists and was written after the last market close."""
    if not path.exists():
        return False
    from datetime import datetime
    from zoneinfo import ZoneInfo
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=ZoneInfo("America/New_York"))
    return mtime >= _last_market_close()


def sync_data(console):
    """Ensure public/data/latest/ has current CSVs.

    1. Check the timestamp on the existing stocks_3m.csv against the last
       market close (weekday 4:15pm ET). If it's already current, skip.
    2. Otherwise try to download all 4 CSVs from GitHub Pages.
    3. If GitHub Pages is unreachable, fall back to the local repo's
       data/latest/ (populated by `git pull` after the daily action runs).
    """
    import urllib.request
    import urllib.error
    import shutil
    from datetime import datetime
    from zoneinfo import ZoneInfo

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sentinel = DATA_DIR / "stocks_3m.csv"

    if _is_current(sentinel):
        mtime = datetime.fromtimestamp(sentinel.stat().st_mtime, tz=ZoneInfo("America/New_York"))
        console.print(f"[green]✓[/green] Data is current (generated {mtime.strftime('%b %d %I:%M %p ET')})")
        return

    console.print("[dim]Data is stale — fetching latest...[/dim]")
    pages_synced = 0

    for period in CSV_PERIODS:
        filename = f"stocks_{period}.csv"
        url = f"{PAGES_BASE}/data/latest/{filename}"
        dest = DATA_DIR / filename
        try:
            urllib.request.urlretrieve(url, dest)
            pages_synced += 1
        except urllib.error.HTTPError as e:
            console.print(f"[yellow]⚠[/yellow] {filename}: HTTP {e.code} — skipped")
        except Exception:
            pass  # fall back to local below

    if pages_synced:
        console.print(f"[green]✓[/green] Synced {pages_synced}/{len(CSV_PERIODS)} CSVs from GitHub Pages")
    else:
        console.print("[yellow]⚠[/yellow] GitHub Pages unreachable — checking local data/latest/")

    # Fall back: copy any still-missing CSVs from the local repo data/latest/
    local_copied = 0
    for period in CSV_PERIODS:
        filename = f"stocks_{period}.csv"
        dest = DATA_DIR / filename
        if not dest.exists():
            local_src = LOCAL_DATA_DIR / filename
            if local_src.exists():
                shutil.copy2(local_src, dest)
                local_copied += 1

    if local_copied:
        console.print(f"[green]✓[/green] Loaded {local_copied} CSV(s) from local data/latest/")
    elif pages_synced == 0:
        console.print("[dim]No data found — run `git pull` to get latest, or `alpha spread` to generate[/dim]")


def _npm():
    """Return the correct npm executable name for the current platform."""
    import platform
    return "npm.cmd" if platform.system() == "Windows" else "npm"


def launch_vite(console):
    """Start the Vite dev server as a subprocess. Returns the Popen handle."""
    import subprocess

    if not WEBAPP_DIR.exists():
        console.print("[yellow]⚠[/yellow] webapp/ not found — skipping Vite dev server")
        return None

    npm = _npm()
    node_modules = WEBAPP_DIR / "node_modules"
    if not node_modules.exists():
        console.print("[dim]Installing npm dependencies...[/dim]")
        result = subprocess.run([npm, "install"], cwd=WEBAPP_DIR)
        if result.returncode != 0:
            console.print("[red]npm install failed — web app will not start[/red]")
            return None

    console.print("[green]✓[/green] Starting Vite dev server on [cyan]http://localhost:5173[/cyan]")
    proc = subprocess.Popen([npm, "run", "dev"], cwd=WEBAPP_DIR)
    return proc


def open_browser_delayed(url="http://localhost:5173", delay=2.0):
    from threading import Timer
    import webbrowser
    Timer(delay, lambda: webbrowser.open(url)).start()


@click.command(context_settings=cli_context, name='launch')
@click.option('-p', '--port', default=5050, help='Proxy server port', show_default=True)
@click.option('-nb', '--no-browser', is_flag=True, default=False, help="Don't open browser")
@click.option('--no-sync', is_flag=True, default=False, help="Skip downloading latest CSVs")
def serve(port, no_browser, no_sync):
    """
    Launch the proxy server and web app together.

    Starts the Flask proxy on :5050 and the Vite dev server on :5173.
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print("[bold cyan]morningalpha[/bold cyan] [dim]launch[/dim]\n")

    console.print(Panel(
        f"[green]Proxy[/green]  [bold cyan]http://localhost:{port}[/bold cyan]\n"
        f"[green]Web app[/green] [bold cyan]http://localhost:5173[/bold cyan]\n\n"
        f"[dim]• Powered by yfinance · no API key required · 4h cache\n"
        f"• Press Ctrl+C to stop both[/dim]",
        title="morningalpha",
        border_style="cyan"
    ))

    # Sync latest CSVs from GitHub Pages
    if no_sync:
        console.print("[dim]Skipping CSV sync (--no-sync)[/dim]")
    else:
        console.print("[dim]Syncing latest data from GitHub Pages...[/dim]")
        sync_data(console)
    console.print()

    # Start Vite
    vite_proc = launch_vite(console)
    console.print()

    # Open browser
    if not no_browser:
        open_browser_delayed()

    # Start Flask proxy (blocking) — clean up Vite on exit
    try:
        from morningalpha.web.proxy_server import run_server
        run_server(port=port, debug=False)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Proxy server error:[/red] {e}")
    finally:
        if vite_proc and vite_proc.poll() is None:
            vite_proc.terminate()
            console.print("[dim]Vite dev server stopped[/dim]")


if __name__ == '__main__':
    serve()
