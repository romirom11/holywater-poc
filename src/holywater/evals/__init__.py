"""Eval runner and tag comparison (docs/QUALITY_METRICS.md §3)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from holywater.domain.idea import IdeaInput
from holywater.observability.errors import ConfigError, PipelineError
from holywater.observability.logging import get_logger, log_context
from holywater.storage.paths import new_run_id
from holywater.storage.repo import RunRepository, TagStats

log = get_logger(__name__)

# (idea, tag, run_id) -> anything; raises on failure. The suite allocates the run id up front, so
# a failed case keeps it even when the error does not carry one.
RunOneFn = Callable[[IdeaInput, str, str], object]
MakeRunIdFn = Callable[[str], str]


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Part of the run id (`<new_run_id>-<case id>`), i.e. a directory name under runs/.
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,40}$")
    text: str = Field(min_length=10)


class EvalSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cases: list[EvalCase] = Field(min_length=1)


def load_cases(path: Path) -> EvalSuite:
    if not path.is_file():
        msg = f"eval cases file not found: {path}"
        raise ConfigError(msg, context={"path": str(path)})
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return EvalSuite.model_validate(raw)
    except (yaml.YAMLError, ValueError) as exc:
        msg = f"invalid eval cases file: {path}"
        raise ConfigError(msg, context={"path": str(path), "error": str(exc)}) from exc


@dataclass(frozen=True)
class CaseOutcome:
    case_id: str
    run_id: str
    success: bool
    error: str | None = None


def case_run_id(case_id: str) -> str:
    """`<new_run_id>-<case id>`: sortable, and the case is visible in `ls runs/`."""
    return f"{new_run_id()}-{case_id}"


def run_suite(
    suite: EvalSuite,
    *,
    tag: str,
    run_one: RunOneFn,
    make_run_id: MakeRunIdFn = case_run_id,
) -> list[CaseOutcome]:
    """Run every case under one tag; a failing case never stops the suite.

    Typed pipeline errors are expected outcomes. Anything else is a bug: logged with its
    traceback and recorded as a failed case, and the suite moves on.
    """
    outcomes: list[CaseOutcome] = []
    for case in suite.cases:
        run_id = make_run_id(case.id)
        with log_context(case_id=case.id):
            log.info("eval.case.start", tag=tag, case_run_id=run_id)
            try:
                run_one(IdeaInput(text=case.text), tag, run_id)
                outcomes.append(CaseOutcome(case.id, run_id, True))
            except PipelineError as exc:
                log.error("eval.case.failed", case_run_id=run_id, error=type(exc).__name__)
                outcomes.append(CaseOutcome(case.id, run_id, False, str(exc)))
            except Exception as exc:  # a bug must not cost the remaining (paid) cases
                log.error(
                    "eval.case.crashed", case_run_id=run_id, error=type(exc).__name__, exc_info=exc
                )
                outcomes.append(CaseOutcome(case.id, run_id, False, f"{type(exc).__name__}: {exc}"))
    return outcomes


def compare_table(repo: RunRepository, tags: list[str]) -> str:
    stats = [repo.tag_stats(t) for t in tags]
    for s in stats:
        if s.runs == 0:
            log.warning("eval.unknown_tag", tag=s.tag)
    return render_stats(stats)


def render_stats(stats: list[TagStats]) -> str:
    """One column per tag. Every run under a tag is pooled (`RunRepository.tag_stats`);
    `wer_samples` is the n behind mean/p95 WER (p95 is nearest-rank: the max for n < 20). A tag
    without runs shows `-` instead of rates and costs, so it never looks like a tag whose runs
    all failed."""
    rows: list[tuple[str, Callable[[TagStats], str]]] = [
        ("dry_run", _mode),
        ("runs", lambda s: str(s.runs)),
        ("success_rate", _if_runs(lambda s: _pct(s.success_rate))),
        ("line_verified_rate", _if_runs(lambda s: _pct(s.line_verified_rate))),
        ("fallback_rate", _if_runs(lambda s: _pct(s.fallback_rate))),
        ("mean_wer", lambda s: _num(s.mean_wer, 3)),
        ("p95_wer", lambda s: _num(s.p95_wer, 3)),
        ("wer_samples", lambda s: str(s.wer_samples)),
        ("mean_cost_usd", _if_runs(lambda s: _num(s.mean_cost_usd, 3))),
        ("mean_shot_duration_s", lambda s: _num(s.mean_duration_s, 1)),
    ]
    for provider in sorted({p for s in stats for p in s.mean_cost_by_provider}):
        rows.append((f"mean_cost_usd:{provider}", _provider_cost_cell(provider)))
    for kind in sorted({k for s in stats for k in s.failure_kinds}):
        rows.append((f"failure_kind:{kind}", _failure_kind_cell(kind)))
    name_width = max(len(name) for name, _ in rows) + 2
    width = max([len(s.tag) for s in stats] + [10])
    header = f"{'metric':<{name_width}}" + "".join(f"{s.tag:>{width + 2}}" for s in stats)
    lines = [header, "-" * len(header)]
    for name, fn in rows:
        lines.append(f"{name:<{name_width}}" + "".join(f"{fn(s):>{width + 2}}" for s in stats))
    return "\n".join(lines)


def _if_runs(cell: Callable[[TagStats], str]) -> Callable[[TagStats], str]:
    return lambda s: cell(s) if s.runs else "-"


def _mode(s: TagStats) -> str:
    if s.runs == 0:
        return "-"
    if s.dry_runs == 0:
        return "no"
    return "yes" if s.dry_runs == s.runs else f"mixed {s.dry_runs}/{s.runs}"


def _provider_cost_cell(provider: str) -> Callable[[TagStats], str]:
    return lambda s: _num(s.mean_cost_by_provider.get(provider), 3)


def _failure_kind_cell(kind: str) -> Callable[[TagStats], str]:
    return lambda s: str(s.failure_kinds.get(kind, 0))


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _num(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"
