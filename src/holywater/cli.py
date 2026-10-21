"""Command-line entry point.

Commands are wired up stage by stage (see docs/IMPLEMENTATION_PLAN.md).
"""

import typer

from holywater import __version__

app = typer.Typer(
    help="HOLYWATER PoC: idea -> video with a verbatim spoken line.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Root command group; subcommands are registered below."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)
