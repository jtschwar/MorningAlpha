#!/usr/bin/env python3
"""
CLI command to launch the morningalpha proxy server with rich interface.
"""
from morningalpha import cli_context
import rich_click as click

API_KEY_URL = "https://www.alphavantage.co/support/#api-key"


def prompt_for_api_key():
    """Prompt user to enter API key."""
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
    
    # Open browser to API key page
    if click.confirm("Open API key page in browser?", default=True):
        webbrowser.open(API_KEY_URL)
    
    console.print()
    api_key = Prompt.ask("[bold]Enter your API key[/bold]")
    
    if not api_key or len(api_key) < 10:
        console.print("[red]Invalid API key. Exiting.[/red]")
        sys.exit(1)
    
    return api_key


def find_index_html():
    """Find the index.html file in the web directory."""
    from pathlib import Path
    
    # Look for index.html relative to package location
    package_dir = Path(__file__).parent
    possible_paths = [
        package_dir / "index.html",
        package_dir / "web" / "index.html",
        package_dir.parent / "web" / "index.html",
        Path("morningalpha/web/index.html"),
        Path("web/index.html"),
        Path("index.html"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return path.resolve()
    
    return None


def open_browser_delayed(index_path, port, delay=1.5):
    """Open browser to index.html and server after a delay."""
    from threading import Timer
    import webbrowser
    
    def _open():
        # Open index.html first
        webbrowser.open(f"file://{index_path}")
    Timer(delay, _open).start()


@click.command(context_settings=cli_context, name='launch')
@click.option(
    '-p', '--port',
    default=5050,
    help='Port to run the server on',
    show_default=True
)
@click.option(
    '-nb', '--no-browser',
    type=bool,
    default=False,
    help='Don\'t automatically open browser',
    show_default=True
)
def serve(port, no_browser):
    """
    Launch the morningalpha proxy server.
    
    Starts the Flask server and opens the dashboard in your browser.
    """
    import sys
    from rich.console import Console
    from rich.panel import Panel
    from morningalpha import keys
    
    console = Console()
    
    console.print("[bold cyan]morningalpha[/bold cyan] [dim]proxy server[/dim]\n")
    
    # Check for API key - if missing, prompt and save it
    if not keys.has_alpha_vantage_key():
        api_key = prompt_for_api_key()
        try:
            keys.set_alpha_vantage_key(api_key)
            console.print(Panel(
                "[green]✓[/green] API key saved successfully!\n"
                "[dim]You won't need to enter it again.[/dim]",
                title="Success",
                border_style="green"
            ))
        except Exception as e:
            console.print(f"[red]Failed to save API key:[/red] {e}")
            sys.exit(1)
        console.print()
    else:
        api_key = keys.get_alpha_vantage_key()
        console.print(f"[green]✓[/green] API key found: [dim]{api_key[:10]}...[/dim]")
    
    # Find index.html
    index_path = find_index_html()
    if index_path:
        console.print(f"[green]✓[/green] Found dashboard: [dim]{index_path}[/dim]")
    else:
        console.print("[yellow]⚠[/yellow] index.html not found - dashboard may not load")
    
    console.print()
    
    # Display server info
    console.print(Panel(
        f"[green]Starting server on[/green] [bold cyan]http://localhost:{port}[/bold cyan]\n\n"
        f"[dim]• API Key: {api_key[:10]}...\n"
        f"• Free tier: 25 calls/day\n"
        f"• Cache: 4 hours\n"
        f"• Press Ctrl+C to stop[/dim]",
        title="🚀 morningalpha server",
        border_style="cyan"
    ))
    
    # Open browser if requested
    if not no_browser and index_path:
        open_browser_delayed(index_path, port)
    
    console.print("[dim]Launching Flask server...[/dim]\n")
    
    try:
        # Import and run the server - it will get the API key itself
        from morningalpha.web.proxy_server import run_server
        run_server(port=port, debug=False)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped[/yellow]")
    except ImportError as e:
        console.print(f"\n[red]Failed to import proxy_server:[/red] {e}")
        console.print("[dim]Make sure morningalpha/web/proxy_server.py exists[/dim]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Server error:[/red] {e}")
        sys.exit(1)


if __name__ == '__main__':
    serve()