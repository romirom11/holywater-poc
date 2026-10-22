"""Eval runner and tag comparison (docs/QUALITY_METRICS.md §3)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from holywater.domain.idea import IdeaInput
from holywater.observability.errors import ConfigError, PipelineError
from holywater.observability.logging import get_logger
from holywater.storage.repo import RunRepository, TagStats

log = get_logger(__name__)


RunOneFn = Callable[[IdeaInput, str, str], str]


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
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


def run_suite(
    suite: EvalSuite,
    *,
    tag: str,
    run_one: RunOneFn,
) -> list[CaseOutcome]:
    """Run every case under one tag; a failing case never stops the suite."""
    outcomes: list[CaseOutcome] = []
    for case in suite.cases:
        log.info("eval.case.start", case_id=case.id, tag=tag)
        try:
            run_id = run_one(IdeaInput(text=case.text), tag, case.id)
            outcomes.append(CaseOutcome(case.id, run_id, True))
        except PipelineError as exc:
            log.error("eval.case.failed", case_id=case.id, error=type(exc).__name__)
            outcomes.append(CaseOutcome(case.id, exc.run_id or "", False, str(exc)))
    return outcomes


def compare_table(repo: RunRepository, tags: list[str]) -> str:
    stats = [repo.tag_stats(t) for t in tags]
    return render_stats(stats)


def render_stats(stats: list[TagStats]) -> str:
    rows: list[tuple[str, Callable[[TagStats], str]]] = [
        ("runs", lambda s: str(s.runs)),
        ("success_rate", lambda s: _pct(s.success_rate)),
        ("fallback_rate", lambda s: _pct(s.fallback_rate)),
        ("mean_wer", lambda s: _num(s.mean_wer, 3)),
        ("p95_wer", lambda s: _num(s.p95_wer, 3)),
        ("mean_cost_usd", lambda s: _num(s.mean_cost_usd, 3)),
        ("mean_shot_duration_s", lambda s: _num(s.mean_duration_s, 1)),
    ]
    for provider in sorted({p for s in stats for p in s.mean_cost_by_provider}):
        rows.append((f"mean_cost_usd:{provider}", _provider_cost_cell(provider)))
    width = max([len(s.tag) for s in stats] + [10])
    header = f"{'metric':<22}" + "".join(f"{s.tag:>{width + 2}}" for s in stats)
    lines = [header, "-" * len(header)]
    for name, fn in rows:
        lines.append(f"{name:<22}" + "".join(f"{fn(s):>{width + 2}}" for s in stats))
    return "\n".join(lines)


def _provider_cost_cell(provider: str) -> Callable[[TagStats], str]:
    return lambda s: _num(s.mean_cost_by_provider.get(provider), 3)


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _num(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"
