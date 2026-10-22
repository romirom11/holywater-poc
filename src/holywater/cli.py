"""Command-line entry point: `hw generate`, `hw inspect`, `hw eval` (see docs/ARCHITECTURE.md)."""

from __future__ import annotations

import os
import sys
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NoReturn

import typer

from holywater import __version__

if TYPE_CHECKING:
    from holywater.config import PipelineConfig
    from holywater.domain.idea import IdeaInput
    from holywater.flows.generate_video_flow import RunOutcome
    from holywater.storage import RunRepository

app = typer.Typer(
    help="HOLYWATER PoC: idea -> video with a verbatim spoken line.",
    no_args_is_help=True,
)
eval_app = typer.Typer(
    help="Run the eval suite and compare pipeline versions.", no_args_is_help=True
)
app.add_typer(eval_app, name="eval")

# Prefect reads these when it is imported (lazily, inside the commands); values the user already
# set win. Telemetry is off: the ephemeral Prefect server would otherwise POST an anonymous
# heartbeat to sens-o-matic.prefect.io and the SDK would emit onboarding events in a terminal.
os.environ.setdefault("PREFECT_SERVER_ALLOW_EPHEMERAL_MODE", "true")
os.environ.setdefault("PREFECT_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
os.environ.setdefault("DO_NOT_TRACK", "1")

_IDEA_HELP = (
    "The idea: 1-3 sentences (10-1000 characters) with the spoken line in quotes. "
    "'-' or nothing with input piped in: read it from stdin (no shell quoting of apostrophes)."
)


@app.callback()
def main() -> None:
    """Root command group; subcommands are registered below."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def generate(
    idea: Annotated[str | None, typer.Argument(help=_IDEA_HELP, show_default=False)] = None,
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="pipeline.yaml path")
    ] = None,
    tag: Annotated[
        str | None, typer.Option(help="Label for eval/compare grouping (never mixes dry and real).")
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option(help="Resume a run (same idea) or name a new one: letters, digits, . _ -"),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Fake providers: no vendor calls, no cost.")
    ] = False,
    log_level: Annotated[str, typer.Option(envvar="HW_LOG_LEVEL")] = "INFO",
) -> None:
    """Turn an idea into a vertical video with the quoted line spoken verbatim."""
    from holywater.config import load_config, load_secrets
    from holywater.flows.generate_video_flow import generate_video_flow
    from holywater.flows.wiring import build_deps
    from holywater.observability.errors import PipelineError
    from holywater.observability.logging import configure_logging
    from holywater.storage import new_run_id, validate_run_id

    configure_logging(log_level)
    idea_input = _parse_idea(_read_idea(idea))  # before any setup: bad input costs nothing
    rid = run_id or new_run_id()
    cfg: PipelineConfig | None = None
    repo: RunRepository | None = None
    try:
        validate_run_id(rid)
        cfg = load_config(config)
        deps = build_deps(cfg, load_secrets(), dry_run=dry_run)
        repo = deps.repo
        repo.ensure_tag_mode(tag, dry_run=dry_run)
        repo.ensure_run_mode(rid, dry_run=dry_run)
        outcome = generate_video_flow(idea_input, deps, run_id=rid, tag=tag)
    except PipelineError as exc:
        _fail(exc, run_id=rid, cfg=cfg, repo=repo)
    except Exception as exc:  # untyped = a bug: full traceback to the run log, one line here
        _fail(exc, run_id=rid, cfg=cfg, repo=repo)
    finally:
        if repo is not None:
            repo.close()
    _print_summary(outcome)


def _read_idea(arg: str | None) -> str:
    """The idea from the argument, or from stdin for `-` / no argument with input piped in."""
    if arg is not None and arg != "-":
        return arg
    if arg is None and sys.stdin.isatty():
        typer.secho(
            "error: no idea given: pass it as an argument, or pipe it in "
            "(hw generate - < idea.txt)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    return sys.stdin.read()


def _parse_idea(text: str) -> IdeaInput:
    """Validate the idea: a readable error (exit 1) instead of a traceback, and a warning when it
    is longer than the 1-3 sentences the brief asks for."""
    from pydantic import ValidationError

    from holywater.domain.idea import MAX_IDEA_SENTENCES, IdeaInput
    from holywater.observability.logging import get_logger

    try:
        idea = IdeaInput(text=text)
    except ValidationError as exc:
        typer.secho(f"error: {exc.errors()[0]['msg']}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    if idea.sentence_count > MAX_IDEA_SENTENCES:
        get_logger(__name__).warning(
            "idea.too_many_sentences", sentences=idea.sentence_count, max=MAX_IDEA_SENTENCES
        )
    return idea


def _fail(
    exc: Exception,
    *,
    run_id: str,
    cfg: PipelineConfig | None,
    repo: RunRepository | None,
) -> NoReturn:
    """One error line; for a run that started, its id, report and how to resume it."""
    from holywater.observability.errors import PipelineError

    run_dir = cfg.storage.runs_dir / run_id if cfg is not None else None
    if not isinstance(exc, PipelineError):
        _log_unexpected(exc, run_id=run_id, run_dir=run_dir)
        typer.secho(f"error: unexpected {type(exc).__name__}: {exc}", fg=typer.colors.RED, err=True)
    else:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
    if run_dir is not None and _run_exists(repo, run_id):
        report = run_dir / "report.json"
        typer.secho(
            f"run {run_id} failed; report: {report if report.is_file() else 'not written'}; "
            f"logs: {run_dir / 'logs.jsonl'}; resume: same idea with --run-id {run_id}",
            err=True,
        )
    raise typer.Exit(code=1) from exc


def _log_unexpected(exc: Exception, *, run_id: str, run_dir: Path | None) -> None:
    from holywater.observability.logging import get_logger, log_context, run_log_file

    log_file = run_dir / "logs.jsonl" if run_dir is not None and run_dir.is_dir() else None
    sink: AbstractContextManager[None] = run_log_file(log_file) if log_file else nullcontext()
    with sink, log_context(run_id=run_id):
        get_logger(__name__).error(
            "cli.unexpected_error", error=type(exc).__name__, message=str(exc), exc_info=exc
        )


def _run_exists(repo: RunRepository | None, run_id: str) -> bool:
    if repo is None:
        return False
    try:
        return repo.get_run(run_id) is not None
    except Exception as exc:  # the error line above matters more than this lookup
        from holywater.observability.logging import get_logger

        get_logger(__name__).warning("cli.run_lookup_failed", run_id=run_id, error=str(exc))
        return False


def _print_summary(outcome: RunOutcome) -> None:
    """stdout: the final path only (scriptable). stderr: the summary, and a warning when the
    spoken line is not evidenced by STT (report.json `line_verified` is false)."""
    verified = _line_verified(outcome.report_path)
    typer.echo(outcome.final.path)
    typer.secho(
        f"run {outcome.run_id}: {outcome.final.duration_s:.1f}s, strategy={outcome.strategy_used}, "
        f"line_verified={'yes' if verified else 'no'}, cost=${outcome.cost_usd:.3f}, "
        f"report={outcome.report_path}",
        err=True,
    )
    if not verified:
        typer.secho(
            "WARNING: the spoken line was not verified by STT (line_verified=false in "
            "report.json): the video exists, but the verbatim line is not evidenced.",
            fg=typer.colors.YELLOW,
            err=True,
        )


def _line_verified(report_path: Path) -> bool:
    from pydantic import ValidationError

    from holywater.domain.report import RunReport
    from holywater.observability.logging import get_logger

    try:
        return RunReport.model_validate_json(report_path.read_text(encoding="utf-8")).line_verified
    except (OSError, ValidationError) as exc:
        get_logger(__name__).warning(
            "cli.report_unreadable", path=str(report_path), error=str(exc), fallback=False
        )
        return False


@app.command()
def inspect(
    run_id: Annotated[str, typer.Argument(help="Run id to inspect.")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """Print the stored report of a run."""
    from holywater.config import load_config
    from holywater.observability.errors import PipelineError
    from holywater.storage import validate_run_id

    try:
        validate_run_id(run_id)
        cfg = load_config(config)
    except PipelineError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    report_path = cfg.storage.runs_dir / run_id / "report.json"
    if not report_path.is_file():
        typer.secho(f"no report at {report_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(report_path.read_text(encoding="utf-8"))


@eval_app.command("run")
def eval_run(
    tag: Annotated[str, typer.Option(help="Label for this pipeline version, e.g. v0.2.")],
    cases: Annotated[Path, typer.Option(help="YAML file with eval cases.")] = Path(
        "evals/cases.yaml"
    ),
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Fake providers: no vendor calls, no cost.")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Start a paid suite without asking.")
    ] = False,
    log_level: Annotated[str, typer.Option(envvar="HW_LOG_LEVEL")] = "WARNING",
) -> None:
    """Run every case under one tag and print the per-case outcome plus the tag summary.

    Without --dry-run every case is a paid generation: the command shows the count and a rough
    cost and asks before starting (--yes skips the question).
    """
    from holywater.config import load_config, load_secrets
    from holywater.evals import load_cases, render_stats, run_suite
    from holywater.flows.generate_video_flow import generate_video_flow
    from holywater.flows.wiring import build_deps
    from holywater.observability.errors import PipelineError
    from holywater.observability.logging import configure_logging

    configure_logging(log_level)
    repo: RunRepository | None = None
    try:
        cfg = load_config(config)
        suite = load_cases(cases)
        deps = build_deps(cfg, load_secrets(), dry_run=dry_run)
        repo = deps.repo
        repo.ensure_tag_mode(tag, dry_run=dry_run)
    except PipelineError as exc:
        if repo is not None:
            repo.close()
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    try:
        if not dry_run:
            _confirm_paid_suite(cfg, len(suite.cases), yes=yes)

        def run_one(idea: IdeaInput, run_tag: str, case_run_id: str) -> None:
            generate_video_flow(idea, deps, run_id=case_run_id, tag=run_tag)

        outcomes = run_suite(suite, tag=tag, run_one=run_one)
        for o in outcomes:
            mark = "ok " if o.success else "FAIL"
            typer.echo(f"{mark} {o.case_id:<16} {o.run_id} {o.error or ''}")
        typer.echo()
        typer.echo(render_stats([deps.repo.tag_stats(tag)]))
    finally:
        deps.repo.close()
    if not all(o.success for o in outcomes):
        raise typer.Exit(code=2)


def _confirm_paid_suite(cfg: PipelineConfig, n_cases: int, *, yes: bool) -> None:
    """Show what a real suite will spend and ask; `--yes` (or a 'y') continues."""
    typer.secho(
        f"{n_cases} PAID runs: video {cfg.video.provider}:{cfg.video.model} at "
        f"{cfg.video.resolution}; {_suite_estimate(cfg, n_cases)}",
        fg=typer.colors.YELLOW,
        err=True,
    )
    if yes:
        return
    if not typer.confirm(f"Start {n_cases} paid runs?", default=False, err=True):
        typer.secho("aborted: no run started", err=True)
        raise typer.Exit(code=1)


def _suite_estimate(cfg: PipelineConfig, n_cases: int) -> str:
    """A rough range from the configured per-second video price and the 8-30 s duration bounds;
    the LLM, STT and judge add cents, regenerations and fallbacks add whole shots."""
    per_s = (
        cfg.providers.openrouter.video_usd_per_s.get(cfg.video.resolution)
        if cfg.video.provider == "openrouter"
        else None
    )
    if per_s is None:
        return f"no per-second video price is configured for {cfg.video.provider}"
    lo, hi = cfg.quality.duration_s.min, cfg.quality.duration_s.max
    return (
        f"about ${n_cases * lo * per_s:.0f}-{n_cases * hi * per_s:.0f} for video alone "
        f"({lo}-{hi} s per run at ${per_s:.3f}/s); regenerations and fallbacks cost more"
    )


@eval_app.command("compare")
def eval_compare(
    tags: Annotated[list[str], typer.Argument(help="Two or more tags to compare.")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """Print a side-by-side table of aggregate metrics for the given tags."""
    from holywater.config import load_config
    from holywater.evals import compare_table
    from holywater.storage import RunRepository, init_db, make_engine

    cfg = load_config(config)
    engine = make_engine(cfg.storage.db_path)
    try:
        init_db(engine)
        repo = RunRepository(engine)
        typer.echo(compare_table(repo, tags))
        unknown = [t for t in dict.fromkeys(tags) if not repo.list_runs(tag=t, limit=1)]
    finally:
        engine.dispose()
    if unknown:
        typer.secho(f"no runs under tag(s): {', '.join(unknown)}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command()
def costs(
    run_id: Annotated[str, typer.Argument(help="Run id to break down.")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """Where the money went for one run: by provider, model, stage and generation cycle."""
    from holywater.config import load_config
    from holywater.storage import RunRepository, init_db, make_engine

    cfg = load_config(config)
    engine = make_engine(cfg.storage.db_path)
    try:
        init_db(engine)
        repo = RunRepository(engine)
        if repo.get_run(run_id) is None:
            typer.secho(f"unknown run {run_id}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        b = repo.cost_breakdown(run_id)
    finally:
        engine.dispose()
    typer.echo(f"run {run_id}")
    typer.echo(
        f"total ${b.total_usd:.4f}  (vendor-reported ${b.actual_usd:.4f}, "
        f"estimated ${b.estimated_usd:.4f})"
    )
    for title, table in (
        ("by provider", b.by_provider),
        ("by model", b.by_model),
        ("by stage", b.by_stage),
    ):
        typer.echo(f"\n{title}")
        for name, amount in table.items():
            typer.echo(f"  {name:<48} ${amount:.4f}")
    if b.by_cycle:
        typer.echo("\nby cycle (shot/attempt)")
        for c in b.by_cycle:
            typer.echo(
                f"  shot {c.shot_id} attempt {c.attempt:<3} ${c.cost_usd:.4f}  ({c.calls} calls)"
            )
