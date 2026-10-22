"""The pipeline as a Prefect flow (docs/DATA_FLOW.md).

Tasks are thin wrappers around stages; retries (typed, with backoff) live on the tasks, while
quality-gate decisions — regenerate a shot, soften a prompt, widen the line window, switch audio
strategy, stop — live in the flow body (docs/DATA_FLOW.md §6).

Every charge goes through `_record_cost`: stages report the charges they cause through a cost
sink (`ProviderCost`) as soon as they are known, so a call that is billed and then fails — or is
retried by Prefect — still gets its costs row.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn, cast

from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from holywater.config.schema import PipelineConfig
from holywater.domain.artifacts import AudioArtifact, FinalVideo, ProviderCost, VideoArtifact
from holywater.domain.enums import (
    AttemptStatus,
    AudioStrategy,
    FailureKind,
    Resolution,
    RunStatus,
    StageName,
    StageStatus,
)
from holywater.domain.idea import IdeaInput
from holywater.domain.script import MAX_SHOT_S, MAX_TOTAL_S, Script, Shot
from holywater.flows.wiring import PipelineDeps
from holywater.observability.errors import (
    BilledJobError,
    ConfigError,
    FatalProviderError,
    MediaError,
    PipelineError,
    PromptRejectedError,
    ProviderError,
    QualityGateError,
    RetryableProviderError,
    ScriptError,
    is_retryable,
)
from holywater.observability.logging import get_logger, log_context, run_log_file
from holywater.prompts import load_prompt
from holywater.providers.base import JudgeAnswer
from holywater.quality.gates import DurationGate, GateResult
from holywater.stages import assemble as assemble_stage
from holywater.stages import generate_audio, generate_video, preflight_script, report
from holywater.stages import verify_speech as verify_stage
from holywater.stages import write_script as write_stage
from holywater.storage import RunPaths, RunRepository
from holywater.storage.models import ShotAttempt
from holywater.storage.repo import STRAY_SPEECH_EXPECTED

log = get_logger(__name__)

CostSink = Callable[[ProviderCost], None]

# The stage currently open in this run, so a task retry can be charged to it. Set by `_stage`;
# Prefect runs sync tasks in the flow's context, so the retry predicate below sees the same value.
_active_stage: ContextVar[tuple[RunContext, StageName] | None] = ContextVar(
    "holywater_active_stage", default=None
)

# `costs.provider` for Jev calls: the judge has no `provider:` key in config (one vendor).
_JUDGE_PROVIDER = "typesafe"
# TTS statuses that mean "this voice id is not accepted" -> one retry with `audio.tts.voice`.
_VOICE_REJECTED_STATUS = frozenset({400, 422})
# A billed job whose job endpoints answered one of these: the next job would fail the same way,
# so the run stops instead of paying for another attempt.
_JOB_UNREACHABLE_STATUS = frozenset({401, 403, 404})
# STT statuses that no later call can get past (bad key, no credits): fatal even where the STT
# check itself is only advisory.
_STT_AUTH_STATUS = frozenset({401, 402, 403})
# Silence kept inside the dialogue shot after the TTS voice ends (tts_overlay line window).
_VOICE_TAIL_S = 0.3
# A paid clip from an earlier invocation in one of these states was never accepted: resume
# gates it instead of paying for a new one.
_PENDING_STATUSES = frozenset(
    {AttemptStatus.GENERATED.value, AttemptStatus.PASSED.value, AttemptStatus.UNVERIFIED.value}
)


# -- tasks (one per stage; retries only on typed retryable errors) ----------------------------


def _retry_on_typed_error(_task: Any, _task_run: Any, state: Any) -> bool:
    """Prefect consults this only while retries remain, so a `True` here is one real retry."""
    try:
        state.result()
    except BaseException as exc:
        retry = is_retryable(exc)
        active = _active_stage.get()
        retries = (
            active[0].repo.record_stage_retry(active[0].run_id, active[1])
            if retry and active is not None
            else None
        )
        log.warning(
            "task.retry_decision", error=type(exc).__name__, retry=retry, stage_retries=retries
        )
        return retry
    return False


@task(name="write_script", cache_policy=NO_CACHE)
def t_write_script(
    idea: IdeaInput, deps: PipelineDeps, on_cost: CostSink
) -> write_stage.ScriptResult:
    return write_stage.write_script(idea, _writer_deps(deps, on_cost))


@task(name="rewrite_script", cache_policy=NO_CACHE)
def t_rewrite_script(
    script: Script, issues: list[str], idea: IdeaInput, deps: PipelineDeps, on_cost: CostSink
) -> write_stage.ScriptResult:
    return write_stage.rewrite_script(script, issues, idea=idea, deps=_writer_deps(deps, on_cost))


@task(name="soften_shot", cache_policy=NO_CACHE)
def t_soften_shot(
    script: Script,
    shot_id: int,
    reason: str,
    idea: IdeaInput,
    deps: PipelineDeps,
    on_cost: CostSink,
) -> write_stage.ScriptResult:
    return write_stage.soften_shot(
        script, shot_id, reason=reason, idea=idea, deps=_writer_deps(deps, on_cost)
    )


def _writer_deps(deps: PipelineDeps, on_cost: CostSink) -> write_stage.ScriptWriterDeps:
    return write_stage.ScriptWriterDeps(
        llm=deps.llm,
        config=deps.config.llm,
        chars_per_s=deps.config.quality.speech_chars_per_s,
        on_cost=on_cost,
    )


@task(name="preflight_script", cache_policy=NO_CACHE)
def t_preflight(script: Script, deps: PipelineDeps) -> preflight_script.PreflightResult:
    return preflight_script.preflight_script(
        script, preflight_script.PreflightDeps(judge=deps.judge, config=deps.config.judge)
    )


@task(name="generate_audio", cache_policy=NO_CACHE)
def t_generate_audio(
    script: Script, deps: PipelineDeps, dest: Path, voice: str | None, on_cost: CostSink
) -> AudioArtifact:
    return generate_audio.generate_line_audio(
        script,
        generate_audio.AudioDeps(tts=deps.tts, ffmpeg=deps.ffmpeg, config=deps.config.audio.tts),
        dest=dest,
        voice=voice,
        on_cost=on_cost,
    )


@task(name="generate_video", cache_policy=NO_CACHE)
def t_generate_shot(
    script: Script,
    shot: Shot,
    deps: PipelineDeps,
    dest: Path,
    attempt: int,
    strategy: AudioStrategy,
    strict: bool,
    on_cost: CostSink,
) -> VideoArtifact:
    return generate_video.generate_shot(
        script,
        shot,
        generate_video.VideoDeps(video=deps.video, ffmpeg=deps.ffmpeg, config=deps.config.video),
        dest=dest,
        attempt=attempt,
        strategy=strategy,
        strict=strict,
        on_cost=on_cost,
    )


def _verify_deps(deps: PipelineDeps) -> verify_stage.VerifyDeps:
    return verify_stage.VerifyDeps(
        stt=deps.stt,
        judge=deps.judge,
        ffmpeg=deps.ffmpeg,
        stt_config=deps.config.audio.stt,
        quality=deps.config.quality,
        judge_config=deps.config.judge,
    )


@task(name="verify_speech", cache_policy=NO_CACHE)
def t_verify_speech(
    media_path: Path,
    shot_id: int,
    attempt: int,
    expected: str,
    language: str,
    deps: PipelineDeps,
    work_dir: Path,
    window: tuple[float, float] | None,
    has_audio: bool,
) -> verify_stage.VerifyResult:
    return verify_stage.verify_speech(
        media_path,
        shot_id=shot_id,
        attempt=attempt,
        expected=expected,
        language=language,
        deps=_verify_deps(deps),
        work_dir=work_dir,
        window=window,
        has_audio=has_audio,
    )


@task(name="check_stray_speech", cache_policy=NO_CACHE)
def t_check_stray_speech(
    media_path: Path,
    shot_id: int,
    attempt: int,
    language: str,
    deps: PipelineDeps,
    work_dir: Path,
) -> verify_stage.StrayResult:
    return verify_stage.check_stray_speech(
        media_path,
        shot_id=shot_id,
        attempt=attempt,
        language=language,
        max_words=deps.config.quality.stray_speech.max_words,
        deps=_verify_deps(deps),
        work_dir=work_dir,
    )


@task(name="assemble", cache_policy=NO_CACHE)
def t_assemble(
    shots: list[VideoArtifact],
    deps: PipelineDeps,
    work_dir: Path,
    dest: Path,
    strategy_used: AudioStrategy,
    voice: assemble_stage.VoiceOverlay | None,
    protect_until_s: float | None,
) -> FinalVideo:
    return assemble_stage.assemble(
        shots,
        assemble_stage.AssembleDeps(
            ffmpeg=deps.ffmpeg,
            quality=deps.config.quality,
            resolution=deps.config.video.resolution,
            lipsync=deps.lipsync,
            lipsync_config=deps.config.audio.lipsync,
            min_audio_peak_db=deps.config.quality.min_audio_peak_db,
        ),
        work_dir=work_dir,
        dest=dest,
        strategy_used=strategy_used,
        voice=voice,
        protect_until_s=protect_until_s,
    )


# -- flow -----------------------------------------------------------------------------------


@dataclass
class RunContext:
    run_id: str
    idea: IdeaInput
    deps: PipelineDeps
    paths: RunPaths
    tag: str | None = None
    resumed: bool = False
    script: Script | None = None
    strategy_used: AudioStrategy | None = None
    fallback_reason: str | None = None
    line_audio: AudioArtifact | None = None
    preflight_rewrites: int = 0
    softened_shots: set[int] = field(default_factory=set)
    shot_artifacts: list[VideoArtifact] = field(default_factory=list)
    shot_attempt_ids: dict[tuple[int, int], int] = field(default_factory=dict)
    # The innermost stage that failed first in this invocation (report.failure_stage).
    failure_stage: StageName | None = None
    # Scriptwriter prompt versions used (video prompt versions live on the attempt rows).
    prompt_versions: set[str] = field(default_factory=set)
    # When this invocation started: the report ignores stage rows failed by an earlier one.
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def config(self) -> PipelineConfig:
        return self.deps.config

    @property
    def repo(self) -> RunRepository:
        return self.deps.repo

    @property
    def current_script(self) -> Script:
        assert self.script is not None, "script stage has not run"
        return self.script

    def shot(self, shot_id: int) -> Shot:
        return next(s for s in self.current_script.shots if s.id == shot_id)

    def accepted(self, shot_id: int) -> VideoArtifact | None:
        return next((a for a in self.shot_artifacts if a.shot_id == shot_id), None)

    def shot_offset_s(self, shot_id: int) -> float:
        """Where a shot starts on the final timeline (sum of the accepted clips before it)."""
        return sum(a.duration_s for a in self.shot_artifacts if a.shot_id < shot_id)


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    final: FinalVideo
    report_path: Path
    strategy_used: AudioStrategy
    cost_usd: float


@flow(name="holywater-generate-video", validate_parameters=False, log_prints=False)
def generate_video_flow(
    idea: IdeaInput, deps: PipelineDeps, *, run_id: str, tag: str | None = None
) -> RunOutcome:
    cfg = deps.config
    paths = RunPaths(cfg.storage.runs_dir, run_id, cfg.storage.final_name)
    ctx = RunContext(run_id=run_id, idea=idea, deps=deps, paths=paths, tag=tag)
    _open_run(ctx)  # a refusal raises here, before any directory or report exists
    paths.ensure()
    final: FinalVideo | None = None
    with run_log_file(paths.logs_jsonl), log_context(run_id=run_id):
        log.info(
            "run.start",
            idea=idea.text,
            dry_run=deps.dry_run,
            strategy=cfg.audio.strategy,
            resumed=ctx.resumed,
            judge=deps.judge.model if cfg.judge.enabled else "disabled",
        )
        try:
            ctx.script = _stage_script(ctx)
            _hitl(ctx, "after_script", title=ctx.script.title)
            if ctx.strategy_used is None:
                ctx.strategy_used = cfg.audio.strategy
            if ctx.strategy_used is AudioStrategy.TTS_OVERLAY:
                _prepare_tts_line(ctx)
            else:
                _skip_audio_stage(ctx)
            for shot_id in [s.id for s in ctx.current_script.shots]:
                ctx.shot_artifacts.append(_produce_shot(ctx, shot_id))
            final = _stage_assemble(ctx)
            _verify_final(ctx, final)
            deps.repo.finish_run(run_id, RunStatus.DONE, final_path=str(final.path))
        except BaseException as exc:
            log.error("run.failed", error=type(exc).__name__, message=str(exc), exc_info=True)
            deps.repo.finish_run(run_id, RunStatus.FAILED, error=exc)
            _write_report(ctx, final)
            raise
        report_path = _write_report(ctx, final)
        run = deps.repo.get_run(run_id)
        assert run is not None and ctx.strategy_used is not None
        breakdown = deps.repo.cost_breakdown(run_id)
        log.info(
            "run.done",
            final=str(final.path),
            duration_s=round(final.duration_s, 2),
            strategy_used=ctx.strategy_used.value,
            cost_usd=round(run.cost_usd_total, 4),
            cost_actual_usd=breakdown.actual_usd,
            cost_estimated_usd=breakdown.estimated_usd,
            cost_by_provider=breakdown.by_provider,
            cost_by_model=breakdown.by_model,
            cost_by_cycle={f"shot{c.shot_id}/a{c.attempt}": c.cost_usd for c in breakdown.by_cycle},
        )
        return RunOutcome(
            run_id=run_id,
            final=final,
            report_path=report_path,
            strategy_used=ctx.strategy_used,
            cost_usd=run.cost_usd_total,
        )


# -- run bookkeeping ----------------------------------------------------------------------------


def _open_run(ctx: RunContext) -> None:
    """Create the run row, or reopen an existing one (`hw generate --run-id X`) to resume it.

    Every refusal (an enabled HITL point, a different idea, a dry run resumed for real or the
    other way round) is a `ConfigError` raised before any paid call.
    """
    cfg, repo = ctx.config, ctx.repo
    for point, enabled in (
        ("after_script", cfg.hitl.after_script.enabled),
        ("on_unverified_speech", cfg.hitl.on_unverified_speech.enabled),
    ):
        if enabled:  # fail before any paid call rather than silently skipping a review
            msg = (
                f"hitl.{point}.enabled=true but the PoC wires no reviewer (docs/DATA_FLOW.md §7); "
                "disable it or implement the hook"
            )
            raise ConfigError(msg, run_id=ctx.run_id, context={"point": point})
    existing = repo.get_run(ctx.run_id)
    if existing is None:
        repo.create_run(
            ctx.run_id,
            ctx.idea,
            config_hash=cfg.config_hash(),
            audio_strategy=cfg.audio.strategy,
            tag=ctx.tag,
            git_sha=ctx.deps.git_sha,
            dry_run=ctx.deps.dry_run,
        )
        return
    if existing.input_text != ctx.idea.text:
        msg = f"run {ctx.run_id} exists with a different idea; pick a new run id to start over"
        raise ConfigError(msg, run_id=ctx.run_id)
    repo.reopen_run(
        ctx.run_id,
        config_hash=cfg.config_hash(),
        git_sha=ctx.deps.git_sha,
        dry_run=ctx.deps.dry_run,
    )
    ctx.resumed = True
    if existing.audio_strategy_used:
        ctx.strategy_used = AudioStrategy(existing.audio_strategy_used)
        ctx.fallback_reason = existing.fallback_reason
    log.info(
        "run.resume",
        previous_status=existing.status,
        previous_error=existing.error_type,
        strategy_used=existing.audio_strategy_used,
        dry_run=ctx.deps.dry_run,
    )


def _hitl(ctx: RunContext, point: str, **payload: Any) -> None:
    """Documented human-in-the-loop points (docs/DATA_FLOW.md §7). Logged no-ops in the PoC;
    `_open_run` refuses to start when one is enabled, so a review is never silently skipped."""
    log.info("hitl.checkpoint", point=point, enabled=False, **payload)


# -- stage 1: script --------------------------------------------------------------------------


def _stage_script(ctx: RunContext) -> Script:
    """Write the script, then pre-flight it; each failed pre-flight round asks the LLM to revise
    its previous script with the judge's issues, up to `llm.max_attempts` rounds in total."""
    cfg, repo = ctx.config, ctx.repo
    input_hash = _script_input_hash(ctx)
    if (
        ctx.resumed
        and repo.stage_is_done(ctx.run_id, StageName.WRITE_SCRIPT, input_hash=input_hash)
        and ctx.paths.script_json.is_file()
    ):
        script = Script.model_validate_json(ctx.paths.script_json.read_text(encoding="utf-8"))
        # The input hash covers the prompt version, so the reused script was written with it.
        ctx.prompt_versions.add(load_prompt("scriptwriter").version_tag)
        log.info("stage.resumed", stage=StageName.WRITE_SCRIPT.value, line=script.dialogue.text)
        return script
    sink = _cost_sink(ctx, StageName.WRITE_SCRIPT)
    with _stage(ctx, StageName.WRITE_SCRIPT, input_hash=input_hash):
        result = cast(
            "write_stage.ScriptResult",
            _with_retries(t_write_script, cfg)(ctx.idea, ctx.deps, sink),
        )
        ctx.prompt_versions.add(result.prompt_version)
        script = result.script
        rounds = cfg.llm.max_attempts
        for round_no in range(1, rounds + 1):
            pre = _stage_preflight(ctx, script)
            if pre.ok:
                break
            if round_no == rounds:
                msg = f"script failed pre-flight after {rounds} round(s)"
                raise ScriptError(
                    msg,
                    stage=StageName.WRITE_SCRIPT.value,
                    run_id=ctx.run_id,
                    context={"issues": pre.issues, "rewrites": ctx.preflight_rewrites},
                )
            log.warning("script.rewrite_requested", round=round_no, issues=pre.issues)
            result = cast(
                "write_stage.ScriptResult",
                _with_retries(t_rewrite_script, cfg)(script, pre.issues, ctx.idea, ctx.deps, sink),
            )
            ctx.preflight_rewrites += 1
            ctx.prompt_versions.add(result.prompt_version)
            script = result.script
        _persist_script(ctx, script)
    return script


def _script_input_hash(ctx: RunContext) -> str:
    """What the script depends on: the idea, the LLM and judge settings, the speech rate the line
    window is checked at, and the scriptwriter prompt version. Other config edits (resolution,
    downstream thresholds) keep the script on resume; the shots are then re-checked against it."""
    cfg = ctx.config
    inputs = {
        "llm": cfg.llm.model_dump(mode="json"),
        "judge": cfg.judge.model_dump(mode="json"),
        "speech_chars_per_s": cfg.quality.speech_chars_per_s,
        "prompt": load_prompt("scriptwriter").version_tag,
    }
    return _hash(ctx.idea.text, json.dumps(inputs, sort_keys=True))


def _stage_preflight(ctx: RunContext, script: Script) -> preflight_script.PreflightResult:
    """Jev checks before any paid video; every answer (or its absence) is a judge_decisions row."""
    with _stage(ctx, StageName.PREFLIGHT_SCRIPT):
        result = t_preflight(script, ctx.deps)
        _record_judge_costs(ctx, StageName.PREFLIGHT_SCRIPT, result.decisions)
        if not result.checks:  # nothing was asked: the judge is disabled or unavailable
            ctx.repo.add_judge_decision(
                ctx.run_id,
                stage=StageName.PREFLIGHT_SCRIPT,
                question="preflight",
                answer="unknown",
                probabilities={},
                model_version=result.skip_reason,
                decision=preflight_script.DECISION_UNAVAILABLE,
            )
        for check in result.checks:
            ctx.repo.add_judge_decision(
                ctx.run_id,
                stage=StageName.PREFLIGHT_SCRIPT,
                question=check.question,
                answer=check.answer.answer,
                probabilities=check.answer.probabilities,
                model_version=check.answer.model_version,
                latency_ms=check.answer.latency_ms,
                shot_id=check.shot_id,
                decision=check.decision,
            )
        if result.skipped:
            ctx.repo.finish_stage(ctx.run_id, StageName.PREFLIGHT_SCRIPT, StageStatus.SKIPPED)
    return result


# -- stage 2: TTS line ------------------------------------------------------------------------


def _skip_audio_stage(ctx: RunContext) -> None:
    """Native strategy: no TTS unless a shot falls back later (then the stage runs for real)."""
    ctx.repo.skip_stage(ctx.run_id, StageName.GENERATE_AUDIO)
    log.info("stage.skipped", stage=StageName.GENERATE_AUDIO.value, reason="strategy=native")


def _prepare_tts_line(ctx: RunContext) -> AudioArtifact:
    """tts_overlay: synthesize the line, then size the line window (and the dialogue shot) from
    the measured voice before the dialogue clip is generated."""
    audio = _stage_audio(ctx)
    _fit_line_to_voice(ctx, audio)
    return audio


def _stage_audio(ctx: RunContext) -> AudioArtifact:
    if ctx.line_audio is not None:
        return ctx.line_audio
    cfg = ctx.config
    d = ctx.current_script.dialogue
    voice = cfg.audio.tts.voice_for(d.voice_gender)
    dest = ctx.paths.audio_line(d.shot_id)
    input_hash = _hash(d.text, d.language, cfg.audio.tts.provider, cfg.audio.tts.model, voice or "")
    if (
        ctx.resumed
        and dest.is_file()
        and ctx.repo.stage_is_done(ctx.run_id, StageName.GENERATE_AUDIO, input_hash=input_hash)
    ):
        reused = _reusable_line_audio(ctx, dest, voice)
        if reused is not None:
            ctx.line_audio = reused
            return reused
    with _stage(ctx, StageName.GENERATE_AUDIO, input_hash=input_hash):
        audio = _synthesize_line(ctx, dest, voice)
    ctx.line_audio = audio
    return audio


def _reusable_line_audio(ctx: RunContext, dest: Path, voice: str | None) -> AudioArtifact | None:
    """The TTS line of an earlier invocation, if ffprobe can still read it; else None (it is
    synthesized again, like a damaged clip is regenerated)."""
    d = ctx.current_script.dialogue
    try:
        info = ctx.deps.ffmpeg.probe(dest)
    except MediaError as exc:
        log.warning(
            "stage.resume_mismatch",
            stage=StageName.GENERATE_AUDIO.value,
            path=str(dest),
            reason=f"audio unreadable: {exc.message}",
            action="synthesize",
        )
        return None
    if not info.has_audio or info.duration_s <= 0:
        log.warning(
            "stage.resume_mismatch",
            stage=StageName.GENERATE_AUDIO.value,
            path=str(dest),
            reason="no playable audio",
            action="synthesize",
        )
        return None
    log.info("stage.resumed", stage=StageName.GENERATE_AUDIO.value, path=str(dest))
    return AudioArtifact(
        shot_id=d.shot_id,
        path=dest,
        duration_s=info.duration_s,
        model=ctx.config.audio.tts.model,
        voice=voice,
        text=d.text,
    )


def _synthesize_line(ctx: RunContext, dest: Path, voice: str | None) -> AudioArtifact:
    """TTS with the speaker's voice; a vendor that refuses that voice id (400/422) gets one retry
    with the default `audio.tts.voice`."""
    default = ctx.config.audio.tts.voice
    try:
        return _call_tts(ctx, dest, voice)
    except FatalProviderError as exc:
        if voice == default or exc.status_code not in _VOICE_REJECTED_STATUS:
            raise
        log.warning(
            "tts.voice_rejected",
            voice=voice,
            fallback_voice=default,
            status_code=exc.status_code,
            message=exc.message,
        )
        return _call_tts(ctx, dest, default)


def _call_tts(ctx: RunContext, dest: Path, voice: str | None) -> AudioArtifact:
    """One TTS task; the stage reports every charge (also of an unusable file) to the sink."""
    sink = _cost_sink(ctx, StageName.GENERATE_AUDIO)
    return cast(
        "AudioArtifact",
        _with_retries(t_generate_audio, ctx.config)(
            ctx.current_script, ctx.deps, dest, voice, sink
        ),
    )


def _fit_line_to_voice(ctx: RunContext, audio: AudioArtifact) -> None:
    """The TTS voice is mixed in at `dialogue.start_s` and must end inside the dialogue shot.

    The line window becomes the measured voice plus a short tail. If that runs past the shot, the
    shot is lengthened (at most 30 s, the total kept within `quality.duration_s.max`) before its
    clip is generated; only when it cannot grow enough does the line start earlier. The script is
    persisted, so the clip prompt, the final check and a resume all see the change.
    """
    script = ctx.current_script
    d, shot = script.dialogue, script.dialogue_shot
    voice_s = audio.duration_s
    start_s, end_s = d.start_s, d.start_s + voice_s + _VOICE_TAIL_S
    duration = shot.duration_s
    if end_s > duration:
        others = script.total_duration_s - shot.duration_s
        cap = min(MAX_SHOT_S, min(MAX_TOTAL_S, ctx.config.quality.duration_s.max) - others)
        duration = max(duration, min(cap, math.ceil(end_s)))
        if end_s > duration:
            start_s = max(0.0, duration - voice_s - _VOICE_TAIL_S)
            end_s = start_s + voice_s + _VOICE_TAIL_S
        if start_s + voice_s > duration:
            msg = (
                f"the TTS line ({voice_s:.2f}s) does not fit in shot {shot.id} even at {duration}s"
            )
            exc = QualityGateError(
                msg,
                gate="voice_fit",
                measured=round(voice_s, 3),
                threshold=duration,
                stage=StageName.GENERATE_AUDIO.value,
                run_id=ctx.run_id,
                context={"shot_id": shot.id, "start_s": start_s},
            )
            _fail_stage(ctx, StageName.GENERATE_AUDIO, exc)
            raise exc
    start_s, end_s = round(start_s, 3), round(min(end_s, duration), 3)
    if (duration, start_s, end_s) == (shot.duration_s, d.start_s, d.end_s):
        return
    updated = (
        script if duration == shot.duration_s else script.with_shot_duration(shot.id, duration)
    )
    updated = updated.with_dialogue_window(start_s, end_s)
    fields: dict[str, Any] = {
        "shot_id": shot.id,
        "voice_s": round(voice_s, 3),
        "old_duration_s": shot.duration_s,
        "new_duration_s": duration,
        "old_window": (d.start_s, d.end_s),
        "new_window": (start_s, end_s),
    }
    if duration != shot.duration_s:  # a longer clip costs more: worth a WARNING
        log.warning("shot.extended_for_tts", **fields)
    else:
        log.info("shot.line_window_fitted", **fields)
    _persist_script(ctx, updated, reason="fit_tts_line")


# -- stage 3+4: shots -------------------------------------------------------------------------


@dataclass
class _Tally:
    """Why the attempts of one shot in this cycle failed, to pick the outcome once they run out."""

    attempts: int = 0
    duration: int = 0
    speech: int = 0
    billed: int = 0
    stray: int = 0
    last_kind: FailureKind | None = None
    last_gate: GateResult | None = None
    last_billed: BilledJobError | None = None

    def summary(self) -> dict[str, int]:
        return {
            "attempts": self.attempts,
            "duration": self.duration,
            "speech": self.speech,
            "billed": self.billed,
            "stray": self.stray,
        }

    def count_billed(self, exc: BilledJobError) -> None:
        """A paid job whose clip never arrived costs one attempt. When its job endpoint answered
        401/403/404, the next job would fail the same way: raise instead of paying again."""
        if exc.status_code in _JOB_UNREACHABLE_STATUS:
            log.error(
                "shot.job_unreachable",
                job_id=exc.job_id,
                status_code=exc.status_code,
                action="stop",
                **self.summary(),
            )
            raise exc
        self.billed += 1
        self.last_billed = exc


@dataclass(frozen=True)
class _Verdict:
    passed: bool
    verified: bool
    kind: FailureKind | None
    reason: str = ""
    transcript: str = ""
    wer: float | None = None


def _produce_shot(ctx: RunContext, shot_id: int, *, fresh: bool = False) -> VideoArtifact:
    """Generate one shot until it passes its gates (docs/DATA_FLOW.md §6).

    A native dialogue shot is gated on clip duration and verbatim speech, and falls back to TTS
    overlay when its speech cannot be made verbatim; every other clip must be speechless and is
    gated on duration plus a stray-speech check. Only this shot is regenerated; others are
    untouched. On resume, a clip that still matches the current script is reused, and a paid clip
    that was never gated is gated instead of paying for a new one.
    """
    pending: VideoArtifact | None = None
    if ctx.resumed and not fresh:
        reused = _resume_accepted(ctx, shot_id)
        if reused is not None:
            return reused
        pending = _resume_pending(ctx, shot_id)
    assert ctx.strategy_used is not None
    with _stage(ctx, StageName.GENERATE_VIDEO), log_context(shot_id=shot_id):
        if ctx.shot(shot_id).has_dialogue and ctx.strategy_used is AudioStrategy.NATIVE:
            outcome = _native_line_attempts(ctx, shot_id, pending)
            if isinstance(outcome, VideoArtifact):
                return outcome
            return _fallback_to_tts(ctx, shot_id, reason=outcome)
        return _speechless_attempts(ctx, shot_id, pending)


def _native_line_attempts(
    ctx: RunContext, shot_id: int, pending: VideoArtifact | None
) -> VideoArtifact | str:
    """Up to `audio.native_attempts` clips that speak the line. Returns the accepted clip, or the
    reason the shot falls back to tts_overlay. Raises only when no attempt failed on speech."""
    tally = _Tally()
    strict = pending is not None and generate_video.is_strict(pending.prompt_version)
    number = _last_attempt_no(ctx, shot_id)
    while tally.attempts < ctx.config.audio.native_attempts:
        tally.attempts += 1
        if pending is not None:
            artifact, pending = pending, None
        else:
            number += 1
            try:
                artifact = _generate_attempt(ctx, shot_id, number, AudioStrategy.NATIVE, strict)
            except PromptRejectedError as exc:
                # Rejected again after its one soften: the TTS clip's prompt omits the line.
                log.warning("shot.line_prompt_rejected", attempt=number, message=exc.message)
                return "video prompt rejected"
            except BilledJobError as exc:
                tally.count_billed(exc)
                continue
        if not _clip_duration_ok(ctx, artifact, tally):
            continue
        verdict = _verify_attempt(ctx, artifact)
        if not verdict.passed and verdict.verified and verdict.kind is FailureKind.STT_NOISE:
            # The transcriber, not the actor, is the suspect: listen again, keep the clip.
            log.warning("shot.stt_rerun", attempt=artifact.attempt, reason="failure_kind=stt_noise")
            verdict = _verify_attempt(ctx, artifact)
        if verdict.passed:
            _accept(ctx, artifact)
            return artifact
        if not verdict.verified:
            _hitl(ctx, "on_unverified_speech", shot_id=shot_id, attempt=artifact.attempt)
            return verdict.reason  # STT is down; TTS is the only path that does not need it
        tally.speech += 1
        tally.last_kind = verdict.kind
        if verdict.kind is FailureKind.WRONG_LANGUAGE:
            return "wrong_language"  # retrying the same model will not fix this
        if verdict.kind is FailureKind.TRUNCATED and _widen_dialogue_window(ctx):
            continue  # same prompt with a longer window for the line
        strict = True  # paraphrased / extra_words / silence / unknown -> stricter prompt
    if tally.speech:
        kind = tally.last_kind.value if tally.last_kind else FailureKind.UNKNOWN.value
        return f"exhausted {tally.attempts} native attempts (last failure: {kind})"
    _raise_exhausted(ctx, shot_id, tally)


def _speechless_attempts(
    ctx: RunContext, shot_id: int, pending: VideoArtifact | None
) -> VideoArtifact:
    """Up to `video.max_attempts` clips that must carry no speech (a non-dialogue shot, or the
    dialogue shot under a TTS voice), gated on duration, then checked for stray speech (warned or
    regenerated per `quality.stray_speech.action`; always regenerated under the TTS voice)."""
    assert ctx.strategy_used is not None
    strategy = ctx.strategy_used
    stray = ctx.config.quality.stray_speech
    # Words in the clip under the TTS voice land in the line window and fail the final check, so
    # that clip is regenerated whatever `stray_speech.action` says.
    regenerate = stray.action == "regenerate" or (
        ctx.shot(shot_id).has_dialogue and strategy is AudioStrategy.TTS_OVERLAY
    )
    tally = _Tally()
    number = _last_attempt_no(ctx, shot_id)
    least_stray: tuple[int, VideoArtifact] | None = None
    while tally.attempts < ctx.config.video.max_attempts:
        tally.attempts += 1
        if pending is not None:
            artifact, pending = pending, None
        else:
            number += 1
            try:
                artifact = _generate_attempt(ctx, shot_id, number, strategy, strict=False)
            except BilledJobError as exc:
                tally.count_billed(exc)
                continue
        if not _clip_duration_ok(ctx, artifact, tally):
            continue
        words = _check_stray_speech(ctx, artifact)
        if words is None or words <= stray.max_words:
            _accept(ctx, artifact)
            return artifact
        if not regenerate:
            log.warning(
                "shot.stray_speech", attempt=artifact.attempt, words=words, action="accepted"
            )
            _accept(ctx, artifact)
            return artifact
        tally.stray += 1
        ctx.repo.mark_attempt(_attempt_id(ctx, artifact), AttemptStatus.FAILED)
        log.warning("shot.stray_speech", attempt=artifact.attempt, words=words, action="regenerate")
        if least_stray is None or words < least_stray[0]:
            least_stray = (words, artifact)
    if least_stray is not None:
        words, artifact = least_stray
        log.warning(
            "shot.stray_speech_accepted",
            attempt=artifact.attempt,
            words=words,
            reason=f"video.max_attempts={ctx.config.video.max_attempts} used up",
        )
        _accept(ctx, artifact)
        return artifact
    _raise_exhausted(ctx, shot_id, tally)


def _fallback_to_tts(ctx: RunContext, shot_id: int, *, reason: str) -> VideoArtifact:
    """Native speech could not be made verbatim: switch the run to TTS overlay, synthesize and
    fit the line, then produce the dialogue clip without speech (up to `video.max_attempts`).
    The TTS line is mixed in at assemble time."""
    log.warning("shot.fallback_to_tts", shot_id=shot_id, reason=reason)
    ctx.strategy_used = AudioStrategy.TTS_OVERLAY
    ctx.fallback_reason = reason
    ctx.repo.set_audio_strategy_used(ctx.run_id, AudioStrategy.TTS_OVERLAY, fallback_reason=reason)
    _prepare_tts_line(ctx)
    return _speechless_attempts(ctx, shot_id, None)


def _raise_exhausted(ctx: RunContext, shot_id: int, tally: _Tally) -> NoReturn:
    """No attempt passed and none failed on speech. The duration error is raised only when every
    attempt missed the length; otherwise the last billed-but-uncollected job is the cause."""
    if tally.duration == tally.attempts and tally.last_gate is not None:
        gate = tally.last_gate
        msg = (
            f"shot {shot_id}: clip length stayed outside tolerance in all "
            f"{tally.attempts} attempt(s)"
        )
        raise QualityGateError(
            msg,
            gate=gate.gate,
            measured=gate.measured,
            threshold=gate.threshold,
            stage=StageName.GENERATE_VIDEO.value,
            run_id=ctx.run_id,
            context={"shot_id": shot_id, **tally.summary()},
        )
    if tally.last_billed is not None:
        log.error("shot.exhausted", shot_id=shot_id, **tally.summary())
        raise tally.last_billed
    msg = f"shot {shot_id}: no usable clip after {tally.attempts} attempt(s)"
    raise PipelineError(
        msg,
        stage=StageName.GENERATE_VIDEO.value,
        run_id=ctx.run_id,
        context={"shot_id": shot_id, **tally.summary()},
    )


def _generate_attempt(
    ctx: RunContext,
    shot_id: int,
    attempt: int,
    strategy: AudioStrategy,
    strict: bool,
) -> VideoArtifact:
    """One paid video call; its charges are recorded by the task as soon as they are known.

    Only a prompt the model refused (`PromptRejectedError`) gets one LLM-softened rewrite per
    shot; a second rejection propagates. A job that was billed but whose clip never arrived
    (`BilledJobError`) becomes an `abandoned` attempt row before it propagates.
    """
    dest = ctx.paths.shot_video(shot_id, attempt)
    generate = _with_retries(t_generate_shot, ctx.config)
    sink = _cost_sink(ctx, StageName.GENERATE_VIDEO, attempt=attempt)

    def call() -> VideoArtifact:
        return cast(
            "VideoArtifact",
            generate(
                ctx.current_script,
                ctx.shot(shot_id),
                ctx.deps,
                dest,
                attempt,
                strategy,
                strict,
                sink,
            ),
        )

    try:
        try:
            artifact = call()
        except PromptRejectedError as exc:
            if shot_id in ctx.softened_shots:
                raise
            _soften_shot(ctx, shot_id, attempt, exc)
            artifact = call()
    except BilledJobError as exc:
        _record_abandoned(ctx, shot_id, attempt, strategy, strict, exc)
        raise
    row = ctx.repo.add_shot_attempt(ctx.run_id, artifact)
    assert row.id is not None
    ctx.shot_attempt_ids[(shot_id, attempt)] = row.id
    return artifact


def _record_abandoned(
    ctx: RunContext,
    shot_id: int,
    attempt: int,
    strategy: AudioStrategy,
    strict: bool,
    exc: BilledJobError,
) -> None:
    """The job was billed (its costs row is already written) but no clip arrived."""
    rendered = generate_video.build_shot_prompt(
        ctx.current_script,
        ctx.shot(shot_id),
        strategy=strategy,
        prompt=load_prompt("seedance_shot"),
        strict=strict,
    )
    ctx.repo.add_abandoned_attempt(
        ctx.run_id,
        shot_id=shot_id,
        attempt=attempt,
        strategy=strategy,
        model=exc.model or ctx.config.video.model,
        provider_request_id=exc.job_id,
        cost_usd=exc.spent_usd or 0.0,
        is_estimate=exc.spent_is_estimate,
        reason=exc.message,
        prompt_version=rendered.version,
        prompt_text=rendered.text,
        resolution=ctx.config.video.resolution,
    )
    log.warning(
        "shot.job_abandoned",
        shot_id=shot_id,
        attempt=attempt,
        job_id=exc.job_id,
        spent_usd=exc.spent_usd,
        message=exc.message,
    )


def _soften_shot(ctx: RunContext, shot_id: int, attempt: int, exc: PromptRejectedError) -> None:
    log.warning(
        "shot.prompt_rejected",
        shot_id=shot_id,
        error=type(exc).__name__,
        message=exc.message,
        status_code=exc.status_code,
        tactic="soften prompt via LLM, retry once",
    )
    sink = _cost_sink(ctx, StageName.WRITE_SCRIPT, shot_id=shot_id, attempt=attempt)
    result = cast(
        "write_stage.ScriptResult",
        _with_retries(t_soften_shot, ctx.config)(
            ctx.current_script, shot_id, exc.message, ctx.idea, ctx.deps, sink
        ),
    )
    ctx.prompt_versions.add(result.prompt_version)
    ctx.softened_shots.add(shot_id)
    _persist_script(ctx, result.script, reason="soften")


def _clip_duration_gate(ctx: RunContext, shot_id: int, strategy: AudioStrategy) -> DurationGate:
    """Per-clip DurationGate: the requested length ± tolerance; a clip under a TTS voice must
    also be long enough for the voice placed at `dialogue.start_s`."""
    shot = ctx.shot(shot_id)
    tol = ctx.config.quality.shot_duration_tolerance_s
    min_s = shot.duration_s - tol
    if shot.has_dialogue and strategy is AudioStrategy.TTS_OVERLAY and ctx.line_audio is not None:
        min_s = max(min_s, ctx.current_script.dialogue.start_s + ctx.line_audio.duration_s)
    return DurationGate(min_s=round(min_s, 3), max_s=shot.duration_s + tol)


def _clip_duration_ok(ctx: RunContext, artifact: VideoArtifact, tally: _Tally) -> bool:
    result = _clip_duration_gate(ctx, artifact.shot_id, artifact.strategy).check(
        artifact.duration_s
    )
    if result.passed:
        return True
    tally.duration += 1
    tally.last_gate = result
    ctx.repo.mark_attempt(_attempt_id(ctx, artifact), AttemptStatus.FAILED)
    log.warning(
        "shot.duration_gate_failed",
        shot_id=artifact.shot_id,
        attempt=artifact.attempt,
        measured=result.measured,
        threshold=result.threshold,
    )
    return False


def _widen_dialogue_window(ctx: RunContext) -> bool:
    """`truncated` tactic: give the line more time inside the shot; False when it cannot grow."""
    script = ctx.current_script
    d, shot = script.dialogue, script.dialogue_shot
    widen = ctx.config.quality.truncated_window_widen_s
    start_s = max(0.0, d.start_s - widen)
    end_s = min(float(shot.duration_s), d.end_s + widen)
    if widen <= 0 or (start_s, end_s) == (d.start_s, d.end_s):
        log.warning("shot.window_cannot_widen", start_s=d.start_s, end_s=d.end_s)
        return False
    log.warning("shot.window_widened", old=(d.start_s, d.end_s), new=(start_s, end_s))
    _persist_script(ctx, script.with_dialogue_window(start_s, end_s), reason="widen_window")
    return True


def _verify_attempt(ctx: RunContext, artifact: VideoArtifact) -> _Verdict:
    return _run_verification(
        ctx,
        media_path=artifact.path,
        shot_id=artifact.shot_id,
        attempt=artifact.attempt,
        duration_s=artifact.duration_s,
        window=None,
        has_audio=artifact.has_native_audio,
    )


def _run_verification(
    ctx: RunContext,
    *,
    media_path: Path,
    shot_id: int,
    attempt: int,
    duration_s: float,
    window: tuple[float, float] | None,
    has_audio: bool = True,
) -> _Verdict:
    """STT + WER gate on a clip (or a window of the final); every outcome lands in the DB.

    Only an STT that stays unavailable through the task retries (`RetryableProviderError`) makes
    the gate `unverified`; a fatal STT error (bad key, unknown model) fails the run, because no
    later clip could be verified either.
    """
    attempt_id = ctx.shot_attempt_ids[(shot_id, attempt)]
    script = ctx.current_script
    expected, language = script.dialogue.text, script.dialogue.language
    with _stage(ctx, StageName.VERIFY_SPEECH):
        try:
            result = cast(
                "verify_stage.VerifyResult",
                _with_retries(t_verify_speech, ctx.config)(
                    media_path,
                    shot_id,
                    attempt,
                    expected,
                    language,
                    ctx.deps,
                    ctx.paths.work_dir,
                    window,
                    has_audio,
                ),
            )
        except RetryableProviderError as exc:
            verification = verify_stage.unverified(
                shot_id=shot_id,
                attempt=attempt,
                expected=expected,
                reason=f"STT unavailable: {exc}",
            )
            ctx.repo.add_speech_verification(ctx.run_id, attempt_id, verification)
            if window is None:  # the final check must not downgrade an already accepted attempt
                ctx.repo.mark_attempt(attempt_id, AttemptStatus.UNVERIFIED)
            return _Verdict(
                passed=False,
                verified=False,
                kind=FailureKind.UNKNOWN,
                reason=f"stt unavailable ({exc.message[:200]})",
            )
        v = result.verification
        ctx.repo.add_speech_verification(ctx.run_id, attempt_id, v)
        if v.stt_model is not None:  # a clip without audio is judged without an STT call
            _record_cost(
                ctx,
                stage=StageName.VERIFY_SPEECH,
                provider=ctx.config.audio.stt.provider,
                model=v.stt_model,
                units=window[1] if window else duration_s,
                cost_usd=result.cost_usd,
                is_estimate=v.cost_is_estimate,
                shot_id=shot_id,
                attempt=attempt,
            )
        _record_classifications(ctx, result.classifications, shot_id=shot_id, attempt=attempt)
        if window is None:  # the final check must not downgrade an already accepted attempt
            status = AttemptStatus.PASSED if v.passed else AttemptStatus.FAILED
            ctx.repo.mark_attempt(attempt_id, status)
    return _Verdict(
        passed=v.passed,
        verified=v.verified,
        kind=v.failure_kind,
        transcript=v.transcript,
        wer=v.wer,
    )


def _check_stray_speech(ctx: RunContext, artifact: VideoArtifact) -> int | None:
    """Words heard in a clip that must carry no speech, recorded as a speech_verifications row
    with `expected=""`. None when there is nothing to check (no audio track) or the advisory check
    could not run (any STT/media error after the task retries: recorded as unverified, logged at
    WARNING); an STT 401/402/403 is raised, as every later STT call would fail too."""
    if not artifact.has_native_audio:
        log.info("shot.stray_check_skipped", attempt=artifact.attempt, reason="no audio track")
        return None
    attempt_id = _attempt_id(ctx, artifact)
    language = ctx.current_script.dialogue.language
    with _stage(ctx, StageName.VERIFY_SPEECH):
        try:
            result = cast(
                "verify_stage.StrayResult",
                _with_retries(t_check_stray_speech, ctx.config)(
                    artifact.path,
                    artifact.shot_id,
                    artifact.attempt,
                    language,
                    ctx.deps,
                    ctx.paths.work_dir,
                ),
            )
        except (ProviderError, MediaError) as exc:
            # The check is advisory: whatever kept STT from answering on this clip, the clip is
            # accepted unverified. Only a bad key / no credits stops the run, since every later
            # STT call (the line check) would fail the same way.
            status_code = exc.status_code if isinstance(exc, ProviderError) else None
            if status_code in _STT_AUTH_STATUS:
                raise
            verification = verify_stage.unverified(
                shot_id=artifact.shot_id,
                attempt=artifact.attempt,
                expected=STRAY_SPEECH_EXPECTED,
                reason=f"STT unavailable: {exc}",
            )
            ctx.repo.add_speech_verification(ctx.run_id, attempt_id, verification)
            log.warning(
                "shot.stray_check_unavailable",
                attempt=artifact.attempt,
                error=type(exc).__name__,
                status_code=status_code,
                message=exc.message,
            )
            return None
        v = result.verification
        ctx.repo.add_speech_verification(ctx.run_id, attempt_id, v)
        _record_cost(
            ctx,
            stage=StageName.VERIFY_SPEECH,
            provider=ctx.config.audio.stt.provider,
            model=v.stt_model or ctx.config.audio.stt.model,
            units=artifact.duration_s,
            cost_usd=v.cost_usd,
            is_estimate=v.cost_is_estimate,
            shot_id=artifact.shot_id,
            attempt=artifact.attempt,
        )
    return result.words


def _record_classifications(
    ctx: RunContext,
    classifications: list[verify_stage.Classification],
    *,
    shot_id: int,
    attempt: int,
) -> None:
    """Every answer behind a failure_kind (Jev's, or the heuristic's when Jev was off or gave
    nothing usable) is a judge_decisions row; Jev calls also get a costs row."""
    _record_judge_costs(
        ctx,
        StageName.VERIFY_SPEECH,
        [c.answer for c in classifications],
        shot_id=shot_id,
        attempt=attempt,
    )
    for c in classifications:
        ctx.repo.add_judge_decision(
            ctx.run_id,
            stage=StageName.VERIFY_SPEECH,
            question=verify_stage.Q_FAILURE_KIND,
            answer=c.answer.answer,
            probabilities=c.answer.probabilities,
            model_version=c.answer.model_version,
            latency_ms=c.answer.latency_ms,
            shot_id=shot_id,
            decision=c.decision,
        )


def _accept(ctx: RunContext, artifact: VideoArtifact) -> None:
    attempt_id = _attempt_id(ctx, artifact)
    ctx.repo.mark_attempt(attempt_id, AttemptStatus.ACCEPTED)
    ctx.repo.set_final_attempt(ctx.run_id, artifact.shot_id, attempt_id)
    if ctx.strategy_used is not None:
        ctx.repo.set_audio_strategy_used(ctx.run_id, ctx.strategy_used)


def _attempt_id(ctx: RunContext, artifact: VideoArtifact) -> int:
    return ctx.shot_attempt_ids[(artifact.shot_id, artifact.attempt)]


def _last_attempt_no(ctx: RunContext, shot_id: int) -> int:
    """Attempt numbers keep growing per shot across cycles and invocations, so no clip or cost
    row is overwritten (a rejected job leaves no attempt row but may leave a costs row)."""
    rows = [a.attempt for a in ctx.repo.get_attempts(ctx.run_id, shot_id)]
    costs = [
        c.attempt
        for c in ctx.repo.get_costs(ctx.run_id)
        if c.shot_id == shot_id and c.attempt is not None
    ]
    return max([*rows, *costs], default=0)


# -- resume -----------------------------------------------------------------------------------


def _resume_accepted(ctx: RunContext, shot_id: int) -> VideoArtifact | None:
    """A shot accepted by an earlier invocation is reused if its clip is still on disk and still
    matches what the current script would generate for it."""
    row = next((a for a in ctx.repo.get_final_attempts(ctx.run_id) if a.shot_id == shot_id), None)
    if row is None:
        return None
    artifact = _reusable_clip(ctx, row)
    if artifact is not None:
        log.info("shot.resumed", shot_id=shot_id, attempt=row.attempt, path=row.path)
    return artifact


def _resume_pending(ctx: RunContext, shot_id: int) -> VideoArtifact | None:
    """The latest attempt of the shot, if it is a paid clip that was never accepted (the previous
    invocation stopped between paying and gating): gate it instead of paying again."""
    attempts = ctx.repo.get_attempts(ctx.run_id, shot_id)
    if not attempts or attempts[-1].status not in _PENDING_STATUSES:
        return None
    row = attempts[-1]
    artifact = _reusable_clip(ctx, row)
    if artifact is not None:
        log.info(
            "shot.resume_pending",
            shot_id=shot_id,
            attempt=row.attempt,
            status=row.status,
            path=row.path,
        )
    return artifact


def _reusable_clip(ctx: RunContext, row: ShotAttempt) -> VideoArtifact | None:
    if row.id is None or row.path is None:
        return None
    mismatch = (
        _resume_mismatch(ctx, row)
        if RunRepository.attempt_file_ok(row)
        else "clip missing or changed on disk (size/sha256)"
    )
    if mismatch is None:
        try:
            info = ctx.deps.ffmpeg.probe(Path(row.path))
        except MediaError as exc:
            mismatch = f"clip unreadable: {exc.message}"
        else:
            gate = _clip_duration_gate(ctx, row.shot_id, AudioStrategy(row.strategy)).check(
                info.duration_s
            )
            if not gate.passed:
                mismatch = f"clip is {gate.measured}s, the shot now needs {gate.threshold}s"
    if mismatch is not None:
        log.warning(
            "shot.resume_mismatch",
            shot_id=row.shot_id,
            attempt=row.attempt,
            reason=mismatch,
            action="regenerate",
        )
        return None
    artifact = VideoArtifact(
        shot_id=row.shot_id,
        attempt=row.attempt,
        strategy=AudioStrategy(row.strategy),
        path=Path(row.path),
        seed=row.seed,
        duration_s=info.duration_s,
        resolution=Resolution(row.resolution),
        has_native_audio=info.has_audio,
        prompt_version=row.prompt_version,
        prompt_text=row.prompt_text,
        provider_request_id=row.provider_request_id,
        model=row.model,
        cost_usd=row.cost_usd,
    )
    ctx.shot_attempt_ids[(row.shot_id, row.attempt)] = row.id
    return artifact


def _resume_mismatch(ctx: RunContext, row: ShotAttempt) -> str | None:
    """Why a stored clip no longer fits the current script/config (None when it still does): the
    prompt the current script renders for it, the strategy of the dialogue shot, the resolution."""
    shot = ctx.shot(row.shot_id)
    strategy = AudioStrategy(row.strategy)
    if shot.has_dialogue and strategy is not ctx.strategy_used:
        return f"strategy {strategy.value} but the run now uses {ctx.strategy_used}"
    if row.resolution != ctx.config.video.resolution.value:
        return f"resolution {row.resolution} but config asks {ctx.config.video.resolution.value}"
    rendered = generate_video.build_shot_prompt(
        ctx.current_script,
        shot,
        strategy=strategy,
        prompt=load_prompt("seedance_shot"),
        strict=generate_video.is_strict(row.prompt_version),
    )
    if (rendered.version, rendered.text) != (row.prompt_version, row.prompt_text):
        return "the current script renders a different prompt for this shot"
    return None


# -- stage 5: assemble ------------------------------------------------------------------------


def _stage_assemble(ctx: RunContext) -> FinalVideo:
    """Concat + loudnorm + gates. A final that is too short gets one shot extended once.

    Paid calls made inside assemble (lipsync) are recorded by the assemble that returns; when
    assemble fails instead, the ones no final reported are recorded before the error leaves.
    """
    cfg = ctx.config
    with _stage(ctx, StageName.ASSEMBLE):
        try:
            try:
                return _assemble_once(ctx)
            except QualityGateError as exc:
                measured = exc.measured if isinstance(exc.measured, (int, float)) else None
                min_s = cfg.quality.duration_s.min
                if exc.gate != "duration" or measured is None or measured >= min_s:
                    raise
                if not _extend_one_shot(ctx, deficit_s=min_s - measured):
                    raise
                return _assemble_once(ctx)
        except BaseException:
            _record_assemble_costs(ctx, assemble_stage.collect_unreported_costs(ctx.paths.work_dir))
            raise


def _assemble_once(ctx: RunContext) -> FinalVideo:
    assert ctx.strategy_used is not None
    script = ctx.current_script
    voice = None
    if ctx.strategy_used is AudioStrategy.TTS_OVERLAY and ctx.line_audio is not None:
        voice = assemble_stage.VoiceOverlay(audio=ctx.line_audio, offset_s=script.dialogue.start_s)
    protect_until_s = ctx.shot_offset_s(script.dialogue.shot_id) + script.dialogue.end_s
    final = cast(
        "FinalVideo",
        _with_retries(t_assemble, ctx.config)(
            sorted(ctx.shot_artifacts, key=lambda a: a.shot_id),
            ctx.deps,
            ctx.paths.work_dir,
            ctx.paths.final_video,
            ctx.strategy_used,
            voice,
            protect_until_s,
        ),
    )
    # Paid calls made inside assemble (lipsync) are reported once, by the assemble that returns;
    # record them now, before anything else can raise.
    _record_assemble_costs(ctx, final.provider_costs)
    return final


def _record_assemble_costs(ctx: RunContext, costs: Iterable[ProviderCost]) -> None:
    sink = _cost_sink(ctx, StageName.ASSEMBLE)
    for cost in costs:
        sink(cost)


def _extend_one_shot(ctx: RunContext, *, deficit_s: float) -> bool:
    """Final too short: request a longer version of one shot and regenerate it. The shortest shot
    that does not carry an already verified native line is picked first (regenerating that one
    would pay for the line, and its verification, again)."""
    script = ctx.current_script

    def cost_of_redoing(shot: Shot) -> tuple[bool, int]:
        native_line = shot.has_dialogue and ctx.strategy_used is AudioStrategy.NATIVE
        return native_line, shot.duration_s

    target = min(
        (s for s in script.shots if s.duration_s < MAX_SHOT_S),
        key=cost_of_redoing,
        default=None,
    )
    if target is None:
        log.warning("assemble.cannot_extend", shots=[s.id for s in script.shots])
        return False
    new_duration = min(MAX_SHOT_S, target.duration_s + math.ceil(deficit_s) + 1)
    log.warning(
        "assemble.too_short",
        deficit_s=round(deficit_s, 2),
        shot_id=target.id,
        old_duration_s=target.duration_s,
        new_duration_s=new_duration,
    )
    previous = ctx.accepted(target.id)
    _persist_script(ctx, script.with_shot_duration(target.id, new_duration), reason="extend_shot")
    regenerated = _produce_shot(ctx, target.id, fresh=True)
    if previous is not None and previous.attempt != regenerated.attempt:
        ctx.repo.mark_attempt(_attempt_id(ctx, previous), AttemptStatus.SUPERSEDED)
    ctx.shot_artifacts = [regenerated if a.shot_id == target.id else a for a in ctx.shot_artifacts]
    return True


def _verify_final(ctx: RunContext, final: FinalVideo) -> None:
    """tts_overlay sanity check: STT on the line window of the assembled final confirms the mux
    (and lipsync/loudnorm) did not break the audio. Native clips were already verified."""
    if ctx.strategy_used is not AudioStrategy.TTS_OVERLAY:
        return
    d = ctx.current_script.dialogue
    accepted = ctx.accepted(d.shot_id)
    assert accepted is not None
    # The voice is mixed in at `d.start_s`; listen to all of it, with a margin on each side.
    margin = ctx.config.quality.final_check_margin_s
    line_s = ctx.line_audio.duration_s if ctx.line_audio is not None else d.window_s
    start_s = ctx.shot_offset_s(d.shot_id) + d.start_s - margin
    window = (round(max(0.0, start_s), 3), round(line_s + 2 * margin, 3))
    log.info("verify_final.start", path=str(final.path), window=window)
    with log_context(shot_id=d.shot_id):
        verdict = _run_verification(
            ctx,
            media_path=final.path,
            shot_id=d.shot_id,
            attempt=accepted.attempt,
            duration_s=final.duration_s,
            window=window,
        )
    if not verdict.verified:
        _hitl(ctx, "on_unverified_speech", shot_id=d.shot_id, final=True)
        return  # recorded as unverified: report.line_verified stays false
    if not verdict.passed:
        msg = (
            f"the assembled final does not carry the line verbatim: heard {verdict.transcript!r} "
            "in the line window (speech in the dialogue clip under the voice, or the "
            "mix/lipsync/loudnorm damaged it)"
        )
        exc = QualityGateError(
            msg,
            gate="speech",
            measured=verdict.wer,
            threshold=ctx.config.quality.max_wer,
            stage=StageName.VERIFY_SPEECH.value,
            run_id=ctx.run_id,
            context={
                "path": str(final.path),
                "window": window,
                "transcript": verdict.transcript,
                "expected": d.text,
            },
        )
        # The verification itself ran fine; the *gate* failed, which is this stage's verdict.
        _fail_stage(ctx, StageName.VERIFY_SPEECH, exc)
        raise exc


# -- stage 6: report --------------------------------------------------------------------------


def _write_report(ctx: RunContext, final: FinalVideo | None) -> Path:
    with _stage(ctx, StageName.REPORT):
        rep = report.build_report(
            ctx.repo,
            ctx.run_id,
            ctx.config,
            dialogue_line=ctx.script.dialogue.text if ctx.script else None,
            final=final,
            preflight_rewrites=ctx.preflight_rewrites,
            judge_model=ctx.deps.judge.model,
            prompt_versions=sorted(ctx.prompt_versions),
            failure_stage=ctx.failure_stage.value if ctx.failure_stage else None,
            dry_run=ctx.deps.dry_run,
            invocation_started_at=ctx.started_at,
        )
        return report.write_report(rep, ctx.paths.report_json)


# -- helpers ----------------------------------------------------------------------------------


def _persist_script(ctx: RunContext, script: Script, *, reason: str | None = None) -> None:
    """The script on disk and the `shots` rows always reflect the latest accepted version."""
    ctx.script = script
    ctx.paths.script_json.write_text(script.model_dump_json(indent=2), encoding="utf-8")
    for shot in script.shots:
        ctx.repo.upsert_shot(
            ctx.run_id,
            shot.id,
            planned_duration_s=shot.duration_s,
            has_dialogue=shot.has_dialogue,
        )
    if reason:
        log.info("script.updated", reason=reason, path=str(ctx.paths.script_json))


def _cost_sink(
    ctx: RunContext, stage: StageName, *, shot_id: int | None = None, attempt: int | None = None
) -> CostSink:
    """A callback for stages: each `ProviderCost` they report becomes one costs row, attributed
    to `stage` and, when it belongs to a generation cycle, to the shot and attempt."""

    def record(cost: ProviderCost) -> None:
        _record_cost(
            ctx,
            stage=stage,
            provider=_vendor_of(ctx, cost.capability),
            model=cost.model,
            units=cost.units,
            cost_usd=cost.cost_usd,
            is_estimate=cost.is_estimate,
            shot_id=cost.shot_id if cost.shot_id is not None else shot_id,
            attempt=attempt,
        )

    return record


def _vendor_of(ctx: RunContext, capability: str) -> str:
    """`costs.provider` for a capability: the vendor config names for it."""
    cfg = ctx.config
    vendors = {
        write_stage.LLM_CAPABILITY: cfg.llm.provider,
        generate_video.VIDEO_CAPABILITY: cfg.video.provider,
        generate_audio.TTS_CAPABILITY: cfg.audio.tts.provider,
        assemble_stage.LIPSYNC_CAPABILITY: cfg.audio.lipsync.provider,
    }
    return vendors.get(capability, capability)


def _record_judge_costs(
    ctx: RunContext,
    stage: StageName,
    answers: Iterable[JudgeAnswer],
    *,
    shot_id: int | None = None,
    attempt: int | None = None,
) -> None:
    """One costs row per Jev call that reached the vendor (NoopJudge answers cost nothing)."""
    for answer in answers:
        if not answer.available and answer.cost_usd == 0:
            continue
        _record_cost(
            ctx,
            stage=stage,
            provider=_JUDGE_PROVIDER,
            model=ctx.deps.judge.model,
            units=1,
            cost_usd=answer.cost_usd,
            is_estimate=answer.cost_is_estimate,
            shot_id=shot_id,
            attempt=attempt,
        )


def _record_cost(
    ctx: RunContext,
    *,
    stage: StageName,
    provider: str,
    model: str,
    units: float,
    cost_usd: float,
    is_estimate: bool,
    shot_id: int | None = None,
    attempt: int | None = None,
) -> None:
    """Persist one provider call's cost and log it with the running total (one event per spend)."""
    ctx.repo.add_cost(
        ctx.run_id,
        stage=stage,
        provider=provider,
        model=model,
        units=units,
        unit_price=0.0,
        cost_usd=cost_usd,
        is_estimate=is_estimate,
        shot_id=shot_id,
        attempt=attempt,
    )
    run = ctx.repo.get_run(ctx.run_id)
    log.info(
        "cost.recorded",
        stage=stage.value,
        provider=provider,
        model=model,
        shot_id=shot_id,
        attempt=attempt,
        units=round(units, 3),
        cost_usd=round(cost_usd, 6),
        estimate=is_estimate,
        run_total_usd=round(run.cost_usd_total, 6) if run else None,
    )


def _fail_stage(ctx: RunContext, name: StageName, exc: BaseException) -> None:
    """Mark a stage failed outside its `_stage` block (a gate verdict reached after it closed)."""
    ctx.repo.finish_stage(ctx.run_id, name, StageStatus.FAILED, error=exc)
    if ctx.failure_stage is None:
        ctx.failure_stage = name


class _stage:  # noqa: N801 - context manager used like a function
    """Marks stage start/finish in the DB and binds the stage name to log lines."""

    def __init__(self, ctx: RunContext, name: StageName, *, input_hash: str | None = None) -> None:
        self.ctx, self.name, self.input_hash = ctx, name, input_hash
        self._log_ctx = log_context(stage=name.value)
        self._token: Token[tuple[RunContext, StageName] | None] | None = None

    def __enter__(self) -> None:
        self.ctx.repo.start_stage(self.ctx.run_id, self.name, input_hash=self.input_hash)
        self._token = _active_stage.set((self.ctx, self.name))
        self._log_ctx.__enter__()
        log.info("stage.start")

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        if self._token is not None:
            _active_stage.reset(self._token)
        if exc is None:
            current = next(
                (s for s in self.ctx.repo.get_stages(self.ctx.run_id) if s.name == self.name.value),
                None,
            )
            if current is None or current.status != StageStatus.SKIPPED.value:
                self.ctx.repo.finish_stage(self.ctx.run_id, self.name, StageStatus.DONE)
            log.info("stage.done")
        else:
            self.ctx.repo.finish_stage(self.ctx.run_id, self.name, StageStatus.FAILED, error=exc)
            if self.ctx.failure_stage is None:  # nested stages exit first: the innermost wins
                self.ctx.failure_stage = self.name
            log.error("stage.failed", error=type(exc).__name__, message=str(exc))
        self._log_ctx.__exit__(exc_type, exc, tb)


def _with_retries(task_fn: Any, cfg: PipelineConfig) -> Any:
    retries = max(0, cfg.retries.provider.attempts - 1)
    delays = list(cfg.retries.provider.backoff_s[:retries]) or [0]
    return task_fn.with_options(
        retries=retries,
        retry_delay_seconds=delays if retries else None,
        retry_condition_fn=_retry_on_typed_error,
    )


def _hash(*parts: str) -> str:
    return hashlib.sha256("\u0000".join(parts).encode()).hexdigest()[:16]
