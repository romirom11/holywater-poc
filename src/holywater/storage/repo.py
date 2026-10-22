"""All reads/writes of domain state go through `RunRepository`. Stages never touch SQL."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine
from sqlmodel import Session, col, select

from holywater.domain import artifacts as art
from holywater.domain.enums import AudioStrategy, RunStatus, StageName, StageStatus
from holywater.domain.idea import IdeaInput
from holywater.observability.errors import PipelineError
from holywater.storage.models import (
    Cost,
    JudgeDecision,
    Run,
    Shot,
    ShotAttempt,
    SpeechVerification,
    Stage,
)


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class TagStats:
    """Aggregates used by `hw eval compare`."""

    tag: str
    runs: int
    success_rate: float
    mean_cost_usd: float
    mean_wer: float | None
    p95_wer: float | None
    fallback_rate: float
    mean_duration_s: float | None
    mean_cost_by_provider: dict[str, float] = field(default_factory=dict)
    line_verified_rate: float = 0.0  # runs whose final line audio passed the STT/WER gate


@dataclass(frozen=True)
class CycleCost:
    """Money spent on one generation cycle (shot attempt), across all its provider calls."""

    shot_id: int
    attempt: int
    cost_usd: float
    calls: int


@dataclass(frozen=True)
class CostBreakdown:
    total_usd: float
    actual_usd: float
    estimated_usd: float
    by_provider: dict[str, float]
    by_model: dict[str, float]
    by_stage: dict[str, float]
    by_cycle: list[CycleCost]


class RunRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def _session(self) -> Session:
        return Session(self._engine, expire_on_commit=False)

    # -- runs ----------------------------------------------------------------------------------

    def create_run(
        self,
        run_id: str,
        idea: IdeaInput,
        *,
        config_hash: str,
        audio_strategy: AudioStrategy,
        tag: str | None = None,
        git_sha: str | None = None,
    ) -> Run:
        run = Run(
            id=run_id,
            tag=tag,
            input_text=idea.text,
            language=idea.language,
            config_hash=config_hash,
            git_sha=git_sha,
            status=RunStatus.RUNNING.value,
            audio_strategy_requested=audio_strategy.value,
        )
        with self._session() as s:
            s.add(run)
            s.commit()
        return run

    def get_run(self, run_id: str) -> Run | None:
        with self._session() as s:
            return s.get(Run, run_id)

    def reopen_run(self, run_id: str, *, config_hash: str, git_sha: str | None) -> Run:
        """Resume: mark an existing run as running again; costs and accepted attempts are kept."""
        with self._session() as s:
            run = self._require_run(s, run_id)
            run.status = RunStatus.RUNNING.value
            run.finished_at = None
            run.error_type = None
            run.error_json = None
            run.config_hash = config_hash
            run.git_sha = git_sha
            s.add(run)
            s.commit()
            return run

    def set_audio_strategy_used(
        self, run_id: str, strategy: AudioStrategy, *, fallback_reason: str | None = None
    ) -> None:
        with self._session() as s:
            run = self._require_run(s, run_id)
            run.audio_strategy_used = strategy.value
            if fallback_reason:
                run.fallback_reason = fallback_reason
            s.add(run)
            s.commit()

    def finish_run(
        self,
        run_id: str,
        status: RunStatus,
        *,
        final_path: str | None = None,
        error: BaseException | None = None,
    ) -> Run:
        with self._session() as s:
            run = self._require_run(s, run_id)
            run.status = status.value
            run.finished_at = _now()
            if final_path is not None:
                run.final_path = final_path
            if error is not None:
                run.error_type, run.error_json = _serialize_error(error)
            s.add(run)
            s.commit()
            return run

    # -- stages --------------------------------------------------------------------------------

    def start_stage(self, run_id: str, name: StageName, *, input_hash: str | None = None) -> Stage:
        """Create or re-open a stage record; each call increments `attempt`."""
        with self._session() as s:
            stage = self._get_stage(s, run_id, name)
            if stage is None:
                stage = Stage(run_id=run_id, name=name.value, status=StageStatus.PENDING.value)
            stage.attempt += 1
            stage.status = StageStatus.RUNNING.value
            stage.input_hash = input_hash
            stage.started_at = _now()
            stage.finished_at = None
            stage.error_type = None
            stage.error_json = None
            s.add(stage)
            s.commit()
            return stage

    def finish_stage(
        self,
        run_id: str,
        name: StageName,
        status: StageStatus,
        *,
        error: BaseException | None = None,
    ) -> Stage:
        with self._session() as s:
            stage = self._get_stage(s, run_id, name)
            if stage is None:
                stage = Stage(run_id=run_id, name=name.value, status=status.value)
            stage.status = status.value
            stage.finished_at = _now()
            if error is not None:
                stage.error_type, stage.error_json = _serialize_error(error)
            s.add(stage)
            s.commit()
            return stage

    def skip_stage(self, run_id: str, name: StageName) -> Stage:
        """Record that a stage was not needed (e.g. TTS under the native strategy).

        Unlike `start_stage` this does not count as an entry, so a later real run of the stage
        (after a fallback) is still entry #1.
        """
        with self._session() as s:
            stage = self._get_stage(s, run_id, name)
            if stage is None:
                stage = Stage(run_id=run_id, name=name.value, status=StageStatus.SKIPPED.value)
            stage.status = StageStatus.SKIPPED.value
            stage.finished_at = _now()
            s.add(stage)
            s.commit()
            return stage

    def record_stage_retry(self, run_id: str, name: StageName) -> int:
        """Count one provider-level retry (a Prefect task retry) against the stage it ran in."""
        with self._session() as s:
            stage = self._get_stage(s, run_id, name)
            if stage is None:
                msg = f"stage {name.value} of run {run_id} not started"
                raise LookupError(msg)
            stage.retries += 1
            s.add(stage)
            s.commit()
            return stage.retries

    def stage_is_done(self, run_id: str, name: StageName, *, input_hash: str | None) -> bool:
        """Idempotency check: the stage finished and its inputs have not changed."""
        with self._session() as s:
            stage = self._get_stage(s, run_id, name)
            return (
                stage is not None
                and stage.status == StageStatus.DONE.value
                and stage.input_hash == input_hash
            )

    def get_stages(self, run_id: str) -> list[Stage]:
        with self._session() as s:
            return list(s.exec(select(Stage).where(Stage.run_id == run_id).order_by(col(Stage.id))))

    # -- shots & attempts ----------------------------------------------------------------------

    def upsert_shot(
        self, run_id: str, shot_id: int, *, planned_duration_s: int, has_dialogue: bool
    ) -> Shot:
        with self._session() as s:
            shot = s.exec(
                select(Shot).where(Shot.run_id == run_id, Shot.shot_id == shot_id)
            ).first()
            if shot is None:
                shot = Shot(
                    run_id=run_id,
                    shot_id=shot_id,
                    planned_duration_s=planned_duration_s,
                    has_dialogue=has_dialogue,
                )
            else:
                shot.planned_duration_s = planned_duration_s
                shot.has_dialogue = has_dialogue
            s.add(shot)
            s.commit()
            return shot

    def add_shot_attempt(self, run_id: str, video: art.VideoArtifact) -> ShotAttempt:
        with self._session() as s:
            shot = s.exec(
                select(Shot).where(Shot.run_id == run_id, Shot.shot_id == video.shot_id)
            ).one()
            assert shot.id is not None
            row = ShotAttempt(
                shot_pk=shot.id,
                run_id=run_id,
                shot_id=video.shot_id,
                attempt=video.attempt,
                strategy=video.strategy.value,
                model=video.model,
                prompt_version=video.prompt_version,
                prompt_text=video.prompt_text,
                seed=video.seed,
                provider_request_id=video.provider_request_id,
                path=str(video.path),
                duration_s=video.duration_s,
                resolution=video.resolution.value,
                cost_usd=video.cost_usd,
            )
            s.add(row)
            s.commit()
            return row

    def mark_attempt(self, attempt_id: int, status: str) -> None:
        with self._session() as s:
            row = s.get(ShotAttempt, attempt_id)
            if row is None:
                msg = f"shot attempt {attempt_id} not found"
                raise LookupError(msg)
            row.status = status
            s.add(row)
            s.commit()

    def set_final_attempt(self, run_id: str, shot_id: int, attempt_id: int) -> None:
        with self._session() as s:
            shot = s.exec(select(Shot).where(Shot.run_id == run_id, Shot.shot_id == shot_id)).one()
            shot.final_attempt_id = attempt_id
            s.add(shot)
            s.commit()

    def get_final_attempts(self, run_id: str) -> list[ShotAttempt]:
        """Final (accepted) attempt per shot, ordered by shot id. Shots without one are omitted."""
        with self._session() as s:
            shots = s.exec(
                select(Shot).where(Shot.run_id == run_id).order_by(col(Shot.shot_id))
            ).all()
            rows: list[ShotAttempt] = []
            for shot in shots:
                if shot.final_attempt_id is not None:
                    attempt = s.get(ShotAttempt, shot.final_attempt_id)
                    if attempt is not None:
                        rows.append(attempt)
            return rows

    def get_attempts(self, run_id: str, shot_id: int | None = None) -> list[ShotAttempt]:
        with self._session() as s:
            stmt = select(ShotAttempt).where(ShotAttempt.run_id == run_id)
            if shot_id is not None:
                stmt = stmt.where(ShotAttempt.shot_id == shot_id)
            return list(s.exec(stmt.order_by(col(ShotAttempt.id))))

    # -- verifications, judge, costs -----------------------------------------------------------

    def add_speech_verification(
        self, run_id: str, attempt_id: int, verification: art.SpeechVerification
    ) -> SpeechVerification:
        row = SpeechVerification(
            shot_attempt_id=attempt_id,
            run_id=run_id,
            expected=verification.expected,
            transcript=verification.transcript,
            wer=verification.wer,
            passed=verification.passed,
            verified=verification.verified,
            stt_model=verification.stt_model,
            failure_kind=verification.failure_kind.value if verification.failure_kind else None,
        )
        with self._session() as s:
            s.add(row)
            s.commit()
        return row

    def get_speech_verifications(self, run_id: str) -> list[SpeechVerification]:
        with self._session() as s:
            return list(
                s.exec(
                    select(SpeechVerification)
                    .where(SpeechVerification.run_id == run_id)
                    .order_by(col(SpeechVerification.id))
                )
            )

    def add_judge_decision(
        self,
        run_id: str,
        *,
        stage: StageName,
        question: str,
        answer: str,
        probabilities: dict[str, float],
        model_version: str | None = None,
        latency_ms: int | None = None,
    ) -> JudgeDecision:
        row = JudgeDecision(
            run_id=run_id,
            stage=stage.value,
            question=question,
            answer=answer,
            probabilities_json=json.dumps(probabilities, sort_keys=True),
            model_version=model_version,
            latency_ms=latency_ms,
        )
        with self._session() as s:
            s.add(row)
            s.commit()
        return row

    def get_judge_decisions(self, run_id: str) -> list[JudgeDecision]:
        with self._session() as s:
            return list(
                s.exec(
                    select(JudgeDecision)
                    .where(JudgeDecision.run_id == run_id)
                    .order_by(col(JudgeDecision.id))
                )
            )

    def add_cost(
        self,
        run_id: str,
        *,
        stage: StageName,
        provider: str,
        model: str,
        units: float,
        unit_price: float,
        cost_usd: float | None = None,
        is_estimate: bool = True,
        shot_id: int | None = None,
        attempt: int | None = None,
    ) -> Cost:
        """Record a cost line and bump the run total in the same transaction."""
        amount = cost_usd if cost_usd is not None else units * unit_price
        row = Cost(
            run_id=run_id,
            stage=stage.value,
            provider=provider,
            model=model,
            units=units,
            unit_price=unit_price,
            cost_usd=amount,
            is_estimate=is_estimate,
            shot_id=shot_id,
            attempt=attempt,
        )
        with self._session() as s:
            run = self._require_run(s, run_id)
            run.cost_usd_total += amount
            s.add(run)
            s.add(row)
            s.commit()
        return row

    def get_costs(self, run_id: str) -> list[Cost]:
        with self._session() as s:
            return list(s.exec(select(Cost).where(Cost.run_id == run_id).order_by(col(Cost.id))))

    def cost_breakdown(self, run_id: str) -> CostBreakdown:
        """Totals by provider, model, stage and generation cycle for one run."""
        return _breakdown(self.get_costs(run_id))

    # -- aggregates ----------------------------------------------------------------------------

    def list_runs(self, *, tag: str | None = None, limit: int = 100) -> list[Run]:
        with self._session() as s:
            stmt = select(Run)
            if tag is not None:
                stmt = stmt.where(Run.tag == tag)
            return list(s.exec(stmt.order_by(col(Run.started_at).desc()).limit(limit)))

    def tag_stats(self, tag: str) -> TagStats:
        runs = self.list_runs(tag=tag, limit=10_000)
        if not runs:
            return TagStats(tag, 0, 0.0, 0.0, None, None, 0.0, None, {}, 0.0)
        done = [r for r in runs if r.status == RunStatus.DONE.value]
        fallbacks = [
            r
            for r in runs
            if r.audio_strategy_used and r.audio_strategy_used != r.audio_strategy_requested
        ]
        wers: list[float] = []
        durations: list[float] = []
        by_provider: dict[str, float] = {}
        verified_runs = 0
        with self._session() as s:
            for run in runs:
                for provider, amount in self.cost_breakdown(run.id).by_provider.items():
                    by_provider[provider] = by_provider.get(provider, 0.0) + amount
                finals = self.get_final_attempts(run.id)
                for attempt in finals:
                    if attempt.duration_s is not None:
                        durations.append(attempt.duration_s)
                    assert attempt.id is not None
                    v = s.exec(
                        select(SpeechVerification)
                        .where(SpeechVerification.shot_attempt_id == attempt.id)
                        .order_by(col(SpeechVerification.id).desc())
                    ).first()
                    if v is not None and v.verified:
                        wers.append(v.wer)
                        if v.passed and run.status == RunStatus.DONE.value:
                            verified_runs += 1
        return TagStats(
            tag=tag,
            runs=len(runs),
            success_rate=len(done) / len(runs),
            mean_cost_usd=statistics.fmean(r.cost_usd_total for r in runs),
            mean_wer=statistics.fmean(wers) if wers else None,
            p95_wer=_percentile(wers, 0.95) if wers else None,
            fallback_rate=len(fallbacks) / len(runs),
            mean_duration_s=statistics.fmean(durations) if durations else None,
            mean_cost_by_provider={k: v / len(runs) for k, v in sorted(by_provider.items())},
            line_verified_rate=verified_runs / len(runs),
        )

    # -- helpers -------------------------------------------------------------------------------

    @staticmethod
    def _require_run(s: Session, run_id: str) -> Run:
        run = s.get(Run, run_id)
        if run is None:
            msg = f"run {run_id} not found"
            raise LookupError(msg)
        return run

    @staticmethod
    def _get_stage(s: Session, run_id: str, name: StageName) -> Stage | None:
        return s.exec(select(Stage).where(Stage.run_id == run_id, Stage.name == name.value)).first()


def _serialize_error(error: BaseException) -> tuple[str, str]:
    payload: dict[str, Any] = (
        error.to_dict() if isinstance(error, PipelineError) else {"message": str(error)}
    )
    return type(error).__name__, json.dumps(payload, default=str)


def _breakdown(rows: list[Cost]) -> CostBreakdown:
    by_provider: dict[str, float] = {}
    by_model: dict[str, float] = {}
    by_stage: dict[str, float] = {}
    cycles: dict[tuple[int, int], list[float]] = {}
    actual = estimated = 0.0
    for c in rows:
        by_provider[c.provider] = by_provider.get(c.provider, 0.0) + c.cost_usd
        by_model[c.model] = by_model.get(c.model, 0.0) + c.cost_usd
        by_stage[c.stage] = by_stage.get(c.stage, 0.0) + c.cost_usd
        if c.is_estimate:
            estimated += c.cost_usd
        else:
            actual += c.cost_usd
        if c.shot_id is not None and c.attempt is not None:
            cycles.setdefault((c.shot_id, c.attempt), []).append(c.cost_usd)
    return CostBreakdown(
        total_usd=round(actual + estimated, 6),
        actual_usd=round(actual, 6),
        estimated_usd=round(estimated, 6),
        by_provider={k: round(v, 6) for k, v in sorted(by_provider.items())},
        by_model={k: round(v, 6) for k, v in sorted(by_model.items())},
        by_stage={k: round(v, 6) for k, v in sorted(by_stage.items())},
        by_cycle=[
            CycleCost(shot_id=k[0], attempt=k[1], cost_usd=round(sum(v), 6), calls=len(v))
            for k, v in sorted(cycles.items())
        ],
    )


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(q * (len(ordered) - 1))))
    return ordered[index]
