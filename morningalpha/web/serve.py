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
RAW_BASE   = "https://raw.githubusercontent.com/jtschwar/MorningAlpha/main/data/latest"
CSV_PERIODS = ["2w", "1m", "3m", "6m"]


def sync_data(console):
    """Download the latest CSVs into public/data/latest/ on every launch.

    Sources tried in order for each file:
      1. raw.githubusercontent.com  — available immediately after daily-data.yml commits
      2. GitHub Pages                — available after pages.yml deploys (~5 min lag)
      3. Local data/latest/          — last resort (repo clone, no network needed)

    The timestamp check is intentionally removed: it caused syncs to be skipped
    whenever local files appeared "fresh enough", missing workflow runs triggered
    manually or out-of-band.
    """
    import urllib.request
    import urllib.error
    import shutil

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    synced = 0
    failed = []

    for period in CSV_PERIODS:
        filename = f"stocks_{period}.csv"
        dest = DATA_DIR / filename
        downloaded = False

        for base in [RAW_BASE, f"{PAGES_BASE}/data/latest"]:
            url = f"{base}/{filename}"
            tmp = dest.with_suffix('.tmp')
            try:
                urllib.request.urlretrieve(url, tmp)
                tmp.replace(dest)
                downloaded = True
                synced += 1
                break
            except Exception:
                if tmp.exists():
                    tmp.unlink()

        if not downloaded:
            local_src = LOCAL_DATA_DIR / filename
            if local_src.exists():
                shutil.copy2(local_src, dest)
                synced += 1
                console.print(f"[yellow]⚠[/yellow] {filename}: network unavailable — using local copy")
            else:
                failed.append(filename)

    if synced == len(CSV_PERIODS):
        console.print(f"[green]✓[/green] Data synced ({synced}/{len(CSV_PERIODS)} files)")
    elif synced > 0:
        console.print(f"[yellow]⚠[/yellow] Partial sync ({synced}/{len(CSV_PERIODS)} files)")
    if failed:
        console.print(f"[red]✗[/red] Missing: {', '.join(failed)} — run `alpha spread` to generate")


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
