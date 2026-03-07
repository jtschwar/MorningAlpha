#!/usr/bin/env python3
"""
CLI command to launch the morningalpha proxy server and web app.
"""
from pathlib import Path
from morningalpha import cli_context
import rich_click as click

API_KEY_URL = "https://www.alphavantage.co/support/#api-key"
WEBAPP_DIR = Path(__file__).parent
DATA_DIR = WEBAPP_DIR / "public" / "data" / "latest"
PAGES_BASE = "https://jtschwar.github.io/MorningAlpha"
CSV_PERIODS = ["2w", "1m", "3m", "6m"]


def sync_data(console):
    """Download the latest CSVs from GitHub Pages into public/data/latest/."""
    import urllib.request
    import urllib.error

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    updated = 0

    for period in CSV_PERIODS:
        filename = f"stocks_{period}.csv"
        url = f"{PAGES_BASE}/data/latest/{filename}"
        dest = DATA_DIR / filename
        try:
            urllib.request.urlretrieve(url, dest)
            updated += 1
        except urllib.error.HTTPError as e:
            console.print(f"[yellow]⚠[/yellow] {filename}: HTTP {e.code} — skipped")
        except Exception:
            console.print(f"[yellow]⚠[/yellow] {filename}: download failed — skipped")

    if updated:
        console.print(f"[green]✓[/green] Synced {updated}/{len(CSV_PERIODS)} CSVs from GitHub Pages")
    else:
        console.print("[yellow]⚠[/yellow] Could not sync CSVs — using local data if available")


def prompt_for_api_key():
    import sys
    import webbrowser
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt

    console = Console()
    console.print(Panel(
        f"[yellow]No Alpha Vantage API key found![/yellow]\n\n"
        f"Get your free API key at:\n[cyan]{API_KEY_URL}[/cyan]\n\n"
        f"[dim]• Free tier: 25 calls/day\n"
        f"• No credit card required\n"
        f"• Instant activation[/dim]",
        title="API Key Required",
        border_style="yellow"
    ))
    if click.confirm("Open API key page in browser?", default=True):
        webbrowser.open(API_KEY_URL)
    console.print()
    api_key = Prompt.ask("[bold]Enter your API key[/bold]")
    if not api_key or len(api_key) < 10:
        console.print("[red]Invalid API key. Exiting.[/red]")
        sys.exit(1)
    return api_key


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
    import sys
    from rich.console import Console
    from rich.panel import Panel
    from morningalpha import keys

    console = Console()
    console.print("[bold cyan]morningalpha[/bold cyan] [dim]launch[/dim]\n")

    # Ensure API key
    if not keys.has_alpha_vantage_key():
        api_key = prompt_for_api_key()
        try:
            keys.set_alpha_vantage_key(api_key)
            console.print(Panel(
                "[green]✓[/green] API key saved successfully!",
                title="Success", border_style="green"
            ))
        except Exception as e:
            console.print(f"[red]Failed to save API key:[/red] {e}")
            sys.exit(1)
        console.print()
    else:
        api_key = keys.get_alpha_vantage_key()
        console.print(f"[green]✓[/green] API key: [dim]{api_key[:10]}...[/dim]")

    console.print(Panel(
        f"[green]Proxy[/green]  [bold cyan]http://localhost:{port}[/bold cyan]\n"
        f"[green]Web app[/green] [bold cyan]http://localhost:5173[/bold cyan]\n\n"
        f"[dim]• {api_key[:10]}... · 25 calls/day · 4h cache\n"
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
