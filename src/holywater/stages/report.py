"""Stage 6: export `report.json` from the DB. Runs even after a failure (partial metrics matter)."""

from __future__ import annotations

from pathlib import Path

from holywater.config.schema import PipelineConfig
from holywater.domain.artifacts import FinalVideo
from holywater.domain.enums import RunStatus, StageName
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

log = get_logger(__name__)


def build_report(
    repo: RunRepository,
    run_id: str,
    config: PipelineConfig,
    *,
    dialogue_line: str | None,
    final: FinalVideo | None,
    preflight_rewrites: int = 0,
) -> RunReport:
    run = repo.get_run(run_id)
    if run is None:
        msg = f"run {run_id} not found"
        raise LookupError(msg)
    stages = repo.get_stages(run_id)
    attempts = repo.get_attempts(run_id)
    verifications = repo.get_speech_verifications(run_id)
    finals = {a.shot_id: a for a in repo.get_final_attempts(run_id)}
    by_attempt = {v.shot_attempt_id: v for v in verifications}

    shot_ids = sorted({a.shot_id for a in attempts})
    shots: list[ShotMetrics] = []
    for shot_id in shot_ids:
        shot_attempts = [a for a in attempts if a.shot_id == shot_id]
        final_attempt = finals.get(shot_id)
        final_v = by_attempt.get(final_attempt.id) if final_attempt and final_attempt.id else None
        kinds = [
            v.failure_kind
            for a in shot_attempts
            if a.id and (v := by_attempt.get(a.id)) and v.failure_kind
        ]
        shots.append(
            ShotMetrics(
                shot_id=shot_id,
                has_dialogue=any(
                    v.shot_attempt_id == a.id for a in shot_attempts for v in verifications
                ),
                attempts=len(shot_attempts),
                final_attempt=final_attempt.attempt if final_attempt else None,
                final_strategy=final_attempt.strategy if final_attempt else None,
                final_wer=final_v.wer if final_v else None,
                verified=final_v.verified if final_v else None,
                failure_kinds=kinds,
                cost_usd=sum(a.cost_usd for a in shot_attempts),
            )
        )

    failed_stage = next((s for s in stages if s.status == "failed"), None)
    stage_metrics = [
        StageMetrics(
            name=s.name,
            status=s.status,
            attempts=s.attempt,
            duration_s=(
                (s.finished_at - s.started_at).total_seconds()
                if s.started_at and s.finished_at
                else None
            ),
            error_type=s.error_type,
        )
        for s in stages
    ]
    prompt_versions = sorted({a.prompt_version for a in attempts})
    breakdown = repo.cost_breakdown(run_id)
    report = RunReport(
        run_id=run.id,
        tag=run.tag,
        success=run.status == RunStatus.DONE.value,
        status=run.status,
        failure_stage=failed_stage.name if failed_stage else None,
        failure_type=run.error_type,
        input_text=run.input_text,
        dialogue_line=dialogue_line,
        config_hash=run.config_hash,
        git_sha=run.git_sha,
        models={
            "llm": config.llm.model,
            "video": config.video.model,
            "tts": config.audio.tts.model,
            "stt": config.audio.stt.model,
            "judge": config.judge.model if config.judge.enabled else "disabled",
        },
        prompt_versions=prompt_versions,
        audio_strategy_requested=run.audio_strategy_requested,
        audio_strategy_used=run.audio_strategy_used,
        fallback_reason=run.fallback_reason,
        started_at=run.started_at,
        finished_at=run.finished_at,
        latency_s=((run.finished_at - run.started_at).total_seconds() if run.finished_at else None),
        stages=stage_metrics,
        shots=shots,
        shots_regenerated=sum(max(0, s.attempts - 1) for s in shots),
        retries_total=sum(max(0, s.attempt - 1) for s in stages),
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
