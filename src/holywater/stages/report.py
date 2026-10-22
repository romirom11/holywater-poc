"""Stage 6: export `report.json` from the DB. Runs even after a failure (partial metrics matter)."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from holywater.config.schema import PipelineConfig
from holywater.domain.artifacts import FinalVideo
from holywater.domain.enums import RunStatus, StageName, StageStatus
from holywater.domain.report import (
    CostLine,
    CostSummary,
    CycleCostMetrics,
    FinalMetrics,
    RunReport,
    ShotMetrics,
    StageMetrics,
)
from holywater.observability.logging import get_logger
from holywater.storage import RunRepository
from holywater.storage.models import Run, ShotAttempt, SpeechVerification, Stage
from holywater.storage.repo import STRAY_SPEECH_EXPECTED, decisive_line_verification

log = get_logger(__name__)

_FAKE_PREFIX = "fake/"


def build_report(
    repo: RunRepository,
    run_id: str,
    config: PipelineConfig,
    *,
    dialogue_line: str | None,
    final: FinalVideo | None,
    preflight_rewrites: int = 0,
    judge_model: str | None = None,
    prompt_versions: Sequence[str] = (),
    failure_stage: str | None = None,
    dry_run: bool | None = None,
    invocation_started_at: datetime | None = None,
) -> RunReport:
    """Assemble `report.json` from the DB.

    - `judge_model`: the judge that actually answered (`JudgeProvider.model`, `"noop"` when Jev
      was unavailable); None falls back to the configured model.
    - `prompt_versions`: versions of the prompts the flow rendered outside shot attempts (the
      scriptwriter); merged with the per-attempt video prompt versions from the DB.
    - `failure_stage`: the innermost stage that failed, as the flow saw it. None falls back to
      the failed stage that started last (nested stages start after their parent), then to the
      `stage` of the stored error.
    - `invocation_started_at`: when this invocation of the run started; the fallback above then
      ignores stage rows that failed in an earlier invocation (a resumed run).
    - `dry_run`: None reads `runs.dry_run`.
    - `models.video` names the model of the accepted clips (a resume may reuse clips made with
      another `video.model`); the configured one when no clip was accepted.
    """
    run = repo.get_run(run_id)
    if run is None:
        msg = f"run {run_id} not found"
        raise LookupError(msg)
    is_dry = run.dry_run if dry_run is None else dry_run
    success = run.status == RunStatus.DONE.value
    # The report cannot observe its own completion, so its stage row (always `running` at this
    # point) stays out of the export; `hw inspect` reads the final status from the DB.
    stages = [s for s in repo.get_stages(run_id) if s.name != StageName.REPORT.value]
    attempts = repo.get_attempts(run_id)
    verifications = repo.get_speech_verifications(run_id)
    line_checks = [v for v in verifications if v.expected != STRAY_SPEECH_EXPECTED]
    stray_checks = [v for v in verifications if v.expected == STRAY_SPEECH_EXPECTED]
    planned = {sh.shot_id: sh for sh in repo.get_shots(run_id)}

    shots: list[ShotMetrics] = []
    for shot_id in sorted(set(planned) | {a.shot_id for a in attempts}):
        shot_attempts = [a for a in attempts if a.shot_id == shot_id]
        attempt_ids = {a.id for a in shot_attempts}
        plan = planned.get(shot_id)
        final_id = plan.final_attempt_id if plan else None
        final_attempt = next((a for a in shot_attempts if a.id == final_id), None)
        final_v = _latest(v for v in line_checks if final_id and v.shot_attempt_id == final_id)
        shots.append(
            ShotMetrics(
                shot_id=shot_id,
                has_dialogue=plan.has_dialogue if plan else False,
                attempts=len(shot_attempts),
                final_attempt=final_attempt.attempt if final_attempt else None,
                final_strategy=final_attempt.strategy if final_attempt else None,
                final_wer=final_v.wer if final_v else None,
                verified=final_v.verified if final_v else None,
                failure_kinds=[
                    v.failure_kind
                    for v in line_checks
                    if v.shot_attempt_id in attempt_ids and v.failure_kind
                ],
                stray_speech_words=_stray_words(
                    [v for v in stray_checks if v.shot_attempt_id in attempt_ids], final_id
                ),
                cost_usd=sum(a.cost_usd for a in shot_attempts),
            )
        )

    stage_metrics = [
        StageMetrics(
            name=s.name,
            status=s.status,
            entries=s.attempt,
            retries=s.retries,
            # Rows written before `total_duration_s` existed only know their last entry.
            duration_s=s.total_duration_s or _last_entry_s(s),
            last_duration_s=_last_entry_s(s),
            error_type=s.error_type,
        )
        for s in stages
    ]
    versions = set(prompt_versions) | {a.prompt_version for a in attempts if a.prompt_version}
    breakdown = repo.cost_breakdown(run_id)
    line = decisive_line_verification(list(planned.values()), verifications)
    report = RunReport(
        run_id=run.id,
        tag=run.tag,
        dry_run=is_dry,
        success=success,
        status=run.status,
        failure_stage=(
            None
            if success
            else (failure_stage or _failed_stage(stages, run, since=invocation_started_at))
        ),
        failure_type=run.error_type,
        input_text=run.input_text,
        dialogue_line=dialogue_line,
        config_hash=run.config_hash,
        git_sha=run.git_sha,
        models=_models(
            config,
            judge_model=judge_model,
            dry_run=is_dry,
            accepted=repo.get_final_attempts(run_id),
        ),
        prompt_versions=sorted(versions),
        audio_strategy_requested=run.audio_strategy_requested,
        audio_strategy_used=run.audio_strategy_used,
        fallback_reason=run.fallback_reason,
        started_at=run.started_at,
        finished_at=run.finished_at,
        latency_s=((run.finished_at - run.started_at).total_seconds() if run.finished_at else None),
        stages=stage_metrics,
        shots=shots,
        line_verified=line is not None and line.verified and line.passed,
        shots_regenerated=sum(max(0, s.attempts - 1) for s in shots),
        failure_kinds=dict(
            sorted(Counter(v.failure_kind for v in verifications if v.failure_kind).items())
        ),
        retries_total=sum(s.retries for s in stages),
        judge_decisions=len(repo.get_judge_decisions(run_id)),
        preflight_rewrites=preflight_rewrites,
        costs=[
            CostLine(
                stage=c.stage,
                provider=c.provider,
                model=c.model,
                cost_usd=c.cost_usd,
                is_estimate=c.is_estimate,
                shot_id=c.shot_id,
                attempt=c.attempt,
            )
            for c in repo.get_costs(run_id)
        ],
        cost_usd_total=run.cost_usd_total,
        cost_summary=CostSummary(
            total_usd=breakdown.total_usd,
            actual_usd=breakdown.actual_usd,
            estimated_usd=breakdown.estimated_usd,
            by_provider=breakdown.by_provider,
            by_model=breakdown.by_model,
            by_stage=breakdown.by_stage,
            by_cycle=[
                CycleCostMetrics(
                    shot_id=c.shot_id, attempt=c.attempt, cost_usd=c.cost_usd, calls=c.calls
                )
                for c in breakdown.by_cycle
            ],
        ),
        final=(
            FinalMetrics(
                path=str(final.path),
                duration_s=final.duration_s,
                width=final.width,
                height=final.height,
                fps=final.fps,
                has_audio=final.has_audio,
            )
            if final
            else None
        ),
    )
    return report


def _models(
    config: PipelineConfig,
    *,
    judge_model: str | None,
    dry_run: bool,
    accepted: Sequence[ShotAttempt] = (),
) -> dict[str, str]:
    """Models that served the run. Dry runs name the fakes (`fake/<model>`), as their costs do.
    The video model comes from the accepted clips when there are any (joined with `+` if they
    differ), so a resumed run names the model that made the clips it ships."""
    video_models = sorted({a.model for a in accepted if a.model})
    models = {
        "llm": config.llm.model,
        "video": " + ".join(video_models) if video_models else config.video.model,
        "tts": config.audio.tts.model,
        "stt": config.audio.stt.model,
        "judge": (judge_model or config.judge.model) if config.judge.enabled else "disabled",
    }
    if not dry_run:
        return models
    return {
        k: v if v.startswith(_FAKE_PREFIX) or v in {"disabled", "noop"} else _FAKE_PREFIX + v
        for k, v in models.items()
    }


def _last_entry_s(stage: Stage) -> float | None:
    if stage.started_at is None or stage.finished_at is None:
        return None
    seconds = (stage.finished_at - stage.started_at).total_seconds()
    return seconds if seconds >= 0 else None


def _failed_stage(stages: list[Stage], run: Run, *, since: datetime | None = None) -> str | None:
    """Innermost failed stage: of the failed rows, the one that started last (a nested stage
    starts after its parent, and both are marked failed); with `since`, only rows that started
    in this invocation. Else the stored error's `stage`."""
    failed = [
        s
        for s in stages
        if s.status == StageStatus.FAILED.value
        and s.started_at
        and (since is None or _utc(s.started_at) >= _utc(since))
    ]
    if failed:
        return max(failed, key=lambda s: (s.started_at, s.id or 0)).name
    if run.error_json:
        try:
            stage = json.loads(run.error_json).get("stage")
        except (json.JSONDecodeError, AttributeError) as exc:
            log.warning("report.error_json_unreadable", run_id=run.id, error=str(exc))
            return None
        return stage if isinstance(stage, str) else None
    return None


def _utc(value: datetime) -> datetime:
    """SQLite hands datetimes back naive; they were written in UTC."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _latest(rows: Iterable[SpeechVerification]) -> SpeechVerification | None:
    return max(rows, key=lambda v: v.id or 0, default=None)


def _stray_words(checks: list[SpeechVerification], final_id: int | None) -> int | None:
    """Words heard on a no-dialogue shot: the latest check of its accepted attempt (else of any
    attempt). None when the shot was not checked or STT was unavailable."""
    row = _latest(v for v in checks if v.shot_attempt_id == final_id) or _latest(checks)
    if row is None or not row.verified:
        return None
    return len(row.transcript.split())


def write_report(report: RunReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    log.info(
        "report.written",
        path=str(path),
        success=report.success,
        cost_usd=round(report.cost_usd_total, 4),
        cost_by_provider=report.cost_summary.by_provider if report.cost_summary else {},
        latency_s=report.latency_s,
    )
    return path


__all__ = ["StageName", "build_report", "write_report"]
