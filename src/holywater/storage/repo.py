"""All reads/writes of domain state go through `RunRepository`. Stages never touch SQL."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Engine
from sqlmodel import Session, col, select

from holywater.domain import artifacts as art
from holywater.domain.enums import (
    AttemptStatus,
    AudioStrategy,
    Resolution,
    RunStatus,
    StageName,
    StageStatus,
)
from holywater.domain.idea import IdeaInput
from holywater.observability.errors import ConfigError, PipelineError
from holywater.storage.models import (
    Cost,
    JudgeDecision,
    Run,
    Shot,
    ShotAttempt,
    SpeechVerification,
    Stage,
)

# `speech_verifications.expected` of a stray-speech check on a shot without dialogue: STT on a
# clip that should be silent of speech. Such rows never count as the line's verification.
STRAY_SPEECH_EXPECTED = ""

_HASH_CHUNK = 1 << 20


def _now() -> datetime:
    return datetime.now(UTC)


def _aware(value: datetime) -> datetime:
    """SQLite hands datetimes back naive; they were written in UTC."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


@dataclass(frozen=True)
class TagStats:
    """Aggregates used by `hw eval compare`, pooled over every run recorded under the tag."""

    tag: str
    runs: int
    success_rate: float
    mean_cost_usd: float
    mean_wer: float | None
    p95_wer: float | None
    fallback_rate: float
    mean_duration_s: float | None
    mean_cost_by_provider: dict[str, float] = field(default_factory=dict)
    line_verified_rate: float = 0.0  # runs that finished with the line passing the STT/WER gate
    dry_runs: int = 0  # runs made with fake providers (a tag is either all dry or all real)
    wer_samples: int = 0  # runs with a verified line WER (the n behind mean/p95)
    failure_kinds: dict[str, int] = field(default_factory=dict)  # over all speech checks


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

    def close(self) -> None:
        """Release every pooled connection (call once the CLI command is done with the DB)."""
        self._engine.dispose()

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
        dry_run: bool = False,
    ) -> Run:
        """Insert a new run. Raises `ConfigError` if `tag` already holds runs of the other mode."""
        self.ensure_tag_mode(tag, dry_run=dry_run)
        run = Run(
            id=run_id,
            tag=tag,
            dry_run=dry_run,
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

    def reopen_run(
        self,
        run_id: str,
        *,
        config_hash: str,
        git_sha: str | None,
        dry_run: bool | None = None,
    ) -> Run:
        """Resume: mark an existing run as running again; costs and accepted attempts are kept.

        With `dry_run` given, a dry run cannot be resumed as a real one (its clips are test
        patterns) and vice versa: `ConfigError`.
        """
        if dry_run is not None:
            self.ensure_run_mode(run_id, dry_run=dry_run)
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

    def ensure_run_mode(self, run_id: str, *, dry_run: bool) -> None:
        """An existing run resumes only in its own mode (a dry run's clips are test patterns):
        `ConfigError` otherwise. Unknown run ids pass (they are new runs)."""
        run = self.get_run(run_id)
        if run is None or run.dry_run == dry_run:
            return
        was = "a dry run (fake providers)" if run.dry_run else "a real run"
        msg = (
            f"run {run_id} is {was}; resume it "
            f"{'with' if run.dry_run else 'without'} --dry-run or pick a new run id"
        )
        raise ConfigError(msg, run_id=run_id, context={"stored_dry_run": run.dry_run})

    def tag_modes(self, tag: str) -> set[bool]:
        """`dry_run` values of the runs already recorded under `tag` (empty: a new tag)."""
        with self._session() as s:
            rows = s.exec(select(Run.dry_run).where(Run.tag == tag).distinct()).all()
            return {bool(v) for v in rows}

    def ensure_tag_mode(self, tag: str | None, *, dry_run: bool) -> None:
        """A tag is homogeneous: `ConfigError` if it already holds runs of the other mode."""
        if tag is None:
            return
        modes = self.tag_modes(tag)
        if modes and modes != {dry_run}:
            held = "dry-run" if True in modes else "real"
            wanted = "dry-run" if dry_run else "real"
            msg = (
                f"tag {tag!r} already holds {held} runs; a {wanted} run cannot join it "
                "(hw eval compare would pool fake and real results); use another tag"
            )
            raise ConfigError(msg, context={"tag": tag, "dry_run": dry_run})

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
        """Close the current entry; its time is added to `total_duration_s` once (a second
        finish of the same entry, e.g. a gate verdict after the stage block, only sets status)."""
        with self._session() as s:
            stage = self._get_stage(s, run_id, name)
            if stage is None:
                stage = Stage(run_id=run_id, name=name.value, status=status.value)
            now = _now()
            if stage.started_at is not None and stage.finished_at is None:
                elapsed = (now - _aware(stage.started_at)).total_seconds()
                stage.total_duration_s += max(0.0, elapsed)
            stage.status = status.value
            stage.finished_at = now
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

    def get_shots(self, run_id: str) -> list[Shot]:
        """Planned shots of the latest script, ordered by shot id."""
        with self._session() as s:
            return list(
                s.exec(select(Shot).where(Shot.run_id == run_id).order_by(col(Shot.shot_id)))
            )

    def add_shot_attempt(self, run_id: str, video: art.VideoArtifact) -> ShotAttempt:
        """Record a generated clip with its size and sha256, so a resume can tell an intact file
        from a truncated or replaced one (`attempt_file_ok`)."""
        size_bytes, sha256 = _file_fingerprint(video.path)
        with self._session() as s:
            shot_pk = self._require_shot_pk(s, run_id, video.shot_id)
            row = ShotAttempt(
                shot_pk=shot_pk,
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
                cost_is_estimate=video.cost_is_estimate,
                size_bytes=size_bytes,
                sha256=sha256,
            )
            s.add(row)
            s.commit()
            return row

    def add_abandoned_attempt(
        self,
        run_id: str,
        *,
        shot_id: int,
        attempt: int,
        strategy: AudioStrategy,
        model: str,
        provider_request_id: str | None,
        cost_usd: float,
        is_estimate: bool,
        reason: str,
        prompt_version: str = "",
        prompt_text: str = "",
        resolution: Resolution | None = None,
        seed: int | None = None,
    ) -> ShotAttempt:
        """A billed video job whose clip was never collected (`BilledJobError`): status
        `abandoned`, no path. The matching `costs` row is the flow's to add (`add_cost`)."""
        with self._session() as s:
            shot_pk = self._require_shot_pk(s, run_id, shot_id)
            row = ShotAttempt(
                shot_pk=shot_pk,
                run_id=run_id,
                shot_id=shot_id,
                attempt=attempt,
                strategy=strategy.value,
                model=model,
                prompt_version=prompt_version,
                prompt_text=prompt_text,
                seed=seed,
                provider_request_id=provider_request_id,
                path=None,
                duration_s=None,
                resolution=resolution.value if resolution is not None else "",
                cost_usd=cost_usd,
                cost_is_estimate=is_estimate,
                status=AttemptStatus.ABANDONED.value,
                reason=reason,
            )
            s.add(row)
            s.commit()
            return row

    def mark_attempt(self, attempt_id: int, status: AttemptStatus | str) -> None:
        """Set an attempt's status; an unknown value raises `ValueError` (stored as `.value`)."""
        value = AttemptStatus(status).value
        with self._session() as s:
            row = s.get(ShotAttempt, attempt_id)
            if row is None:
                msg = f"shot attempt {attempt_id} not found"
                raise LookupError(msg)
            row.status = value
            s.add(row)
            s.commit()

    @staticmethod
    def attempt_file_ok(row: ShotAttempt) -> bool:
        """The attempt's clip is on disk and, when recorded, has the same size and sha256."""
        if row.path is None:
            return False
        path = Path(row.path)
        if not path.is_file():
            return False
        if row.size_bytes is not None and path.stat().st_size != row.size_bytes:
            return False
        return row.sha256 is None or _sha256(path) == row.sha256

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

    def line_verification(self, run_id: str) -> SpeechVerification | None:
        """The check `line_verified` is read from (see `decisive_line_verification`)."""
        return decisive_line_verification(
            self.get_shots(run_id), self.get_speech_verifications(run_id)
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
        shot_id: int | None = None,
        decision: str | None = None,
    ) -> JudgeDecision:
        row = JudgeDecision(
            run_id=run_id,
            stage=stage.value,
            shot_id=shot_id,
            question=question,
            answer=answer,
            decision=decision,
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

    def list_runs(self, *, tag: str | None = None, limit: int | None = 100) -> list[Run]:
        with self._session() as s:
            stmt = select(Run)
            if tag is not None:
                stmt = stmt.where(Run.tag == tag)
            stmt = stmt.order_by(col(Run.started_at).desc())
            if limit is not None:
                stmt = stmt.limit(limit)
            return list(s.exec(stmt))

    def tag_stats(self, tag: str) -> TagStats:
        """Aggregates over *every* run recorded under `tag`: failed, resumed and repeated runs
        (the same case run twice) are all pooled. A tag never mixes dry-run and real runs
        (`ensure_tag_mode`); to compare pipeline versions, give each version its own tag.

        Per run: success (status done), fallback (strategy used != requested), the line check
        (`decisive_line_verification`: its WER when verified; `line_verified` when it passed and
        the run finished), the final clip duration of every shot, and costs by provider.
        """
        runs = self.list_runs(tag=tag, limit=None)
        if not runs:
            return TagStats(tag, 0, 0.0, 0.0, None, None, 0.0, None)
        run_ids = [r.id for r in runs]
        with self._session() as s:
            costs = s.exec(select(Cost).where(col(Cost.run_id).in_(run_ids))).all()
            shots = s.exec(select(Shot).where(col(Shot.run_id).in_(run_ids))).all()
            final_ids = [sh.final_attempt_id for sh in shots if sh.final_attempt_id is not None]
            finals = s.exec(select(ShotAttempt).where(col(ShotAttempt.id).in_(final_ids))).all()
            checks = s.exec(
                select(SpeechVerification)
                .where(col(SpeechVerification.run_id).in_(run_ids))
                .order_by(col(SpeechVerification.id))
            ).all()

        by_provider: dict[str, float] = {}
        for c in costs:
            by_provider[c.provider] = by_provider.get(c.provider, 0.0) + c.cost_usd
        durations = [a.duration_s for a in finals if a.duration_s is not None]
        wers: list[float] = []
        verified_runs = 0
        for run in runs:
            line = decisive_line_verification(
                [sh for sh in shots if sh.run_id == run.id],
                [v for v in checks if v.run_id == run.id],
            )
            if line is not None and line.verified:
                wers.append(line.wer)
                if line.passed and run.status == RunStatus.DONE.value:
                    verified_runs += 1
        n = len(runs)
        return TagStats(
            tag=tag,
            runs=n,
            success_rate=sum(r.status == RunStatus.DONE.value for r in runs) / n,
            mean_cost_usd=statistics.fmean(r.cost_usd_total for r in runs),
            mean_wer=statistics.fmean(wers) if wers else None,
            p95_wer=percentile(wers, 0.95) if wers else None,
            fallback_rate=sum(
                bool(r.audio_strategy_used) and r.audio_strategy_used != r.audio_strategy_requested
                for r in runs
            )
            / n,
            mean_duration_s=statistics.fmean(durations) if durations else None,
            mean_cost_by_provider={k: v / n for k, v in sorted(by_provider.items())},
            line_verified_rate=verified_runs / n,
            dry_runs=sum(r.dry_run for r in runs),
            wer_samples=len(wers),
            failure_kinds=dict(
                sorted(Counter(v.failure_kind for v in checks if v.failure_kind).items())
            ),
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
    def _require_shot_pk(s: Session, run_id: str, shot_id: int) -> int:
        shot = s.exec(select(Shot).where(Shot.run_id == run_id, Shot.shot_id == shot_id)).one()
        assert shot.id is not None
        return shot.id

    @staticmethod
    def _get_stage(s: Session, run_id: str, name: StageName) -> Stage | None:
        return s.exec(select(Stage).where(Stage.run_id == run_id, Stage.name == name.value)).first()


def decisive_line_verification(
    shots: Sequence[Shot], verifications: Sequence[SpeechVerification]
) -> SpeechVerification | None:
    """The latest line check (not a stray-speech check) on the dialogue shot's accepted attempt:
    for native speech the clip's STT gate, for tts_overlay the check on the assembled final.
    `line_verified` (report.json, `hw eval compare`) is this row being verified and passed."""
    dialogue = next((sh for sh in shots if sh.has_dialogue), None)
    if dialogue is None or dialogue.final_attempt_id is None:
        return None
    rows = [
        v
        for v in verifications
        if v.shot_attempt_id == dialogue.final_attempt_id and v.expected != STRAY_SPEECH_EXPECTED
    ]
    return max(rows, key=lambda v: v.id or 0, default=None)


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile: the smallest value with at least `q` of the samples at or below
    it. Never interpolates, so it is always an observed value; for n < 20 the p95 is the max."""
    if not values:
        msg = "percentile of no values"
        raise ValueError(msg)
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _serialize_error(error: BaseException) -> tuple[str, str]:
    payload: dict[str, Any] = (
        error.to_dict() if isinstance(error, PipelineError) else {"message": str(error)}
    )
    return type(error).__name__, json.dumps(payload, default=str)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _file_fingerprint(path: Path) -> tuple[int | None, str | None]:
    """(size, sha256) of a file on disk; (None, None) when it is not there (yet)."""
    if not path.is_file():
        return None, None
    return path.stat().st_size, _sha256(path)


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
