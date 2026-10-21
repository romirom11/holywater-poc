"""Command-line entry point: `hw generate`, `hw inspect`, `hw eval` (see docs/ARCHITECTURE.md)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from holywater import __version__

app = typer.Typer(
    help="HOLYWATER PoC: idea -> video with a verbatim spoken line.",
    no_args_is_help=True,
)

# Prefect needs these before it is imported; harmless if the user already set them.
os.environ.setdefault("PREFECT_SERVER_ALLOW_EPHEMERAL_MODE", "true")
os.environ.setdefault("PREFECT_LOGGING_LEVEL", "WARNING")


@app.callback()
def main() -> None:
    """Root command group; subcommands are registered below."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def generate(
    idea: Annotated[str, typer.Argument(help="The idea, 1-3 sentences; put the line in quotes.")],
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="pipeline.yaml path")
    ] = None,
    tag: Annotated[str | None, typer.Option(help="Label for eval/compare grouping.")] = None,
    run_id: Annotated[str | None, typer.Option(help="Reuse a run id (resume/inspect).")] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Fake providers: no network, no cost.")
    ] = False,
    log_level: Annotated[str, typer.Option(envvar="HW_LOG_LEVEL")] = "INFO",
) -> None:
    """Turn an idea into a vertical video with the quoted line spoken verbatim."""
    from holywater.config import load_config, load_secrets
    from holywater.domain.idea import IdeaInput
    from holywater.flows.generate_video_flow import generate_video_flow
    from holywater.flows.wiring import build_deps
    from holywater.observability.errors import PipelineError
    from holywater.observability.logging import configure_logging
    from holywater.storage import new_run_id

    configure_logging(log_level)
    try:
        cfg = load_config(config)
        deps = build_deps(cfg, load_secrets(), dry_run=dry_run)
        outcome = generate_video_flow(
            IdeaInput(text=idea), deps, run_id=run_id or new_run_id(), tag=tag
        )
    except PipelineError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(outcome.final.path)
    typer.secho(
        f"run {outcome.run_id}: {outcome.final.duration_s:.1f}s, strategy={outcome.strategy_used}, "
        f"cost=${outcome.cost_usd:.3f}, report={outcome.report_path}",
        err=True,
    )


@app.command()
def inspect(
    run_id: Annotated[str, typer.Argument(help="Run id to inspect.")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """Print the stored report of a run."""
    from holywater.config import load_config

    cfg = load_config(config)
    report_path = cfg.storage.runs_dir / run_id / "report.json"
    if not report_path.is_file():
        typer.secho(f"no report at {report_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(report_path.read_text(encoding="utf-8"))
