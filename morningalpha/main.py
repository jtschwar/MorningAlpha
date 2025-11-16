from morningalpha.spread.access import spread
from morningalpha.web.serve import serve
from morningalpha import cli_context
import rich_click as click

@click.group(context_settings=cli_context)
def routines():
    """MorningAlpha -- Your Morning Insight to New Market Trends."""
    pass


# Add Subcommands to the group
routines.add_command(spread)
routines.add_command(serve)