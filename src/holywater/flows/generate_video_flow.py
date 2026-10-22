"""The pipeline as a Prefect flow (docs/DATA_FLOW.md).

Tasks are thin wrappers around stages; retries (typed, with backoff) live on the tasks, while
quality-gate decisions — regenerate a shot, soften a prompt, widen the line window, switch audio
strategy, stop — live in the flow body (docs/DATA_FLOW.md §6).
"""

from __future__ import annotations

import hashlib
import math
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from holywater.config.schema import PipelineConfig
from holywater.domain.artifacts import AudioArtifact, FinalVideo, VideoArtifact
from holywater.domain.enums import (
    AudioStrategy,
    FailureKind,
    Resolution,
    RunStatus,
    StageName,
    StageStatus,
)
from holywater.domain.idea import IdeaInput
from holywater.domain.script import MAX_SHOT_S, Script, Shot
from holywater.flows.wiring import PipelineDeps
from holywater.observability.errors import (
    ConfigError,
    FatalProviderError,
    PipelineError,
    ProviderError,
    QualityGateError,
    ScriptError,
    is_retryable,
)
from holywater.observability.logging import get_logger, log_context, run_log_file
from holywater.quality.gates import DurationGate, GateResult
from holywater.stages import assemble as assemble_stage
from holywater.stages import generate_audio, generate_video, preflight_script, report
from holywater.stages import verify_speech as verify_stage
from holywater.stages import write_script as write_stage
from holywater.storage import RunPaths, RunRepository

log = get_logger(__name__)

# The stage currently open in this run, so a task retry can be charged to it. Set by `_stage`;
# Prefect runs sync tasks in the flow's context, so the retry predicate below sees the same value.
_active_stage: ContextVar[tuple[RunContext, StageName] | None] = ContextVar(
    "holywater_active_stage", default=None
)

# A fatal video error with one of these statuses is about credentials/billing/endpoint, not about
# the prompt — softening the prompt would not help.
_NO_SOFTEN_STATUS = frozenset({401, 402, 403, 404})
# Extra audio on each side of the line window when verifying the assembled final.
_FINAL_CHECK_MARGIN_S = 0.5


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
    idea: IdeaInput, deps: PipelineDeps, feedback: list[str]
) -> write_stage.ScriptResult:
    return write_stage.write_script(
        idea,
        write_stage.ScriptWriterDeps(llm=deps.llm, config=deps.config.llm),
        feedback=feedback,
    )


@task(name="soften_shot", cache_policy=NO_CACHE)
def t_soften_shot(
    script: Script, shot_id: int, reason: str, idea: IdeaInput, deps: PipelineDeps
) -> write_stage.ScriptResult:
    return write_stage.soften_shot(
        script,
        shot_id,
        reason=reason,
        idea=idea,
        deps=write_stage.ScriptWriterDeps(llm=deps.llm, config=deps.config.llm),
    )


@task(name="preflight_script", cache_policy=NO_CACHE)
def t_preflight(script: Script, deps: PipelineDeps) -> preflight_script.PreflightResult:
    return preflight_script.preflight_script(
        script, preflight_script.PreflightDeps(judge=deps.judge, config=deps.config.judge)
    )


@task(name="generate_audio", cache_policy=NO_CACHE)
def t_generate_audio(script: Script, deps: PipelineDeps, dest: Path) -> AudioArtifact:
    return generate_audio.generate_line_audio(
        script,
        generate_audio.AudioDeps(tts=deps.tts, ffmpeg=deps.ffmpeg, config=deps.config.audio.tts),
        dest=dest,
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
) -> VideoArtifact:
    return generate_video.generate_shot(
        script,
        shot,
        generate_video.VideoDeps(video=deps.video, ffmpeg=deps.ffmpeg, config=deps.config.video),
        dest=dest,
        attempt=attempt,
        strategy=strategy,
        strict=strict,
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
) -> verify_stage.VerifyResult:
    return verify_stage.verify_speech(
        media_path,
        shot_id=shot_id,
        attempt=attempt,
        expected=expected,
        language=language,
        deps=verify_stage.VerifyDeps(
            stt=deps.stt,
            judge=deps.judge,
            ffmpeg=deps.ffmpeg,
            stt_config=deps.config.audio.stt,
            quality=deps.config.quality,
            judge_config=deps.config.judge,
        ),
        work_dir=work_dir,
        window=window,
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
    last_gate: GateResult | None = None

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
    paths = RunPaths(cfg.storage.runs_dir, run_id, cfg.storage.final_name).ensure()
    ctx = RunContext(run_id=run_id, idea=idea, deps=deps, paths=paths, tag=tag)
    _open_run(ctx)
    final: FinalVideo | None = None
    with run_log_file(paths.logs_jsonl), log_context(run_id=run_id):
        log.info(
            "run.start",
            idea=idea.text,
            dry_run=deps.dry_run,
            strategy=cfg.audio.strategy,
            resumed=ctx.resumed,
        )
        try:
            ctx.script = _stage_script(ctx)
            _hitl(ctx, "after_script", title=ctx.script.title)
            if ctx.strategy_used is None:
                ctx.strategy_used = cfg.audio.strategy
            if ctx.strategy_used is AudioStrategy.TTS_OVERLAY:
                ctx.line_audio = _stage_audio(ctx)
            else:
                _skip_audio_stage(ctx)
            for shot in ctx.script.shots:
                ctx.shot_artifacts.append(_produce_shot(ctx, shot.id))
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
    """Create the run row, or reopen an existing one (`hw generate --run-id X`) to resume it."""
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
        )
        return
    if existing.input_text != ctx.idea.text:
        msg = f"run {ctx.run_id} exists with a different idea; pick a new run id to start over"
        raise ConfigError(msg, run_id=ctx.run_id)
    repo.reopen_run(ctx.run_id, config_hash=cfg.config_hash(), git_sha=ctx.deps.git_sha)
    ctx.resumed = True
    if existing.audio_strategy_used:
        ctx.strategy_used = AudioStrategy(existing.audio_strategy_used)
        ctx.fallback_reason = existing.fallback_reason
    log.info(
        "run.resume",
        previous_status=existing.status,
        previous_error=existing.error_type,
        strategy_used=existing.audio_strategy_used,
    )


def _hitl(ctx: RunContext, point: str, **payload: Any) -> None:
    """Documented human-in-the-loop points (docs/DATA_FLOW.md §7). Logged no-ops in the PoC;
    `_open_run` refuses to start when one is enabled, so a review is never silently skipped."""
    log.info("hitl.checkpoint", point=point, enabled=False, **payload)


# -- stage orchestration ----------------------------------------------------------------------


def _stage_script(ctx: RunContext) -> Script:
    cfg, repo = ctx.config, ctx.repo
    feedback: list[str] = []
    input_hash = _hash(ctx.idea.text, cfg.config_hash())
    if (
        ctx.resumed
        and repo.stage_is_done(ctx.run_id, StageName.WRITE_SCRIPT, input_hash=input_hash)
        and ctx.paths.script_json.is_file()
    ):
        script = Script.model_validate_json(ctx.paths.script_json.read_text(encoding="utf-8"))
        log.info("stage.resumed", stage=StageName.WRITE_SCRIPT.value, line=script.dialogue.text)
        return script
    with _stage(ctx, StageName.WRITE_SCRIPT, input_hash=input_hash):
        for round_no in range(1, cfg.llm.max_attempts + 1):
            result = cast(
                "write_stage.ScriptResult",
                _with_retries(t_write_script, cfg)(ctx.idea, ctx.deps, feedback),
            )
            _record_llm_cost(ctx, result)
            script = result.script
            pre = _stage_preflight(ctx, script)
            if pre.ok:
                break
            ctx.preflight_rewrites += 1
            feedback = pre.issues
            log.warning("script.rewrite_requested", round=round_no, issues=pre.issues)
        else:
            msg = f"script failed pre-flight after {cfg.llm.max_attempts} rounds"
            raise ScriptError(
                msg, stage="preflight_script", run_id=ctx.run_id, context={"issues": feedback}
            )
        _persist_script(ctx, script)
    return script


def _stage_preflight(ctx: RunContext, script: Script) -> preflight_script.PreflightResult:
    with _stage(ctx, StageName.PREFLIGHT_SCRIPT):
        result = t_preflight(script, ctx.deps)
        for decision in result.decisions:
            if decision.available:
                ctx.repo.add_judge_decision(
                    ctx.run_id,
                    stage=StageName.PREFLIGHT_SCRIPT,
                    question=decision.question.split(":", 1)[0],
                    answer=decision.answer,
                    probabilities=decision.probabilities,
                    model_version=decision.model_version,
                    latency_ms=decision.latency_ms,
                )
        if result.skipped:
            ctx.repo.finish_stage(ctx.run_id, StageName.PREFLIGHT_SCRIPT, StageStatus.SKIPPED)
    return result


def _skip_audio_stage(ctx: RunContext) -> None:
    """Native strategy: no TTS unless a shot falls back later (then the stage runs for real)."""
    ctx.repo.skip_stage(ctx.run_id, StageName.GENERATE_AUDIO)
    log.info("stage.skipped", stage=StageName.GENERATE_AUDIO.value, reason="strategy=native")


def _stage_audio(ctx: RunContext) -> AudioArtifact:
    if ctx.line_audio is not None:
        return ctx.line_audio
    script = ctx.current_script
    dest = ctx.paths.audio_line(script.dialogue.shot_id)
    if ctx.resumed and dest.is_file():
        info = ctx.deps.ffmpeg.probe(dest)
        audio = AudioArtifact(
            shot_id=script.dialogue.shot_id,
            path=dest,
            duration_s=info.duration_s,
            model=ctx.config.audio.tts.model,
            voice=ctx.config.audio.tts.voice,
            text=script.dialogue.text,
        )
        log.info("stage.resumed", stage=StageName.GENERATE_AUDIO.value, path=str(dest))
        ctx.line_audio = audio
        return audio
    with _stage(ctx, StageName.GENERATE_AUDIO):
        audio = cast(
            "AudioArtifact",
            _with_retries(t_generate_audio, ctx.config)(script, ctx.deps, dest),
        )
        _record_cost(
            ctx,
            stage=StageName.GENERATE_AUDIO,
            provider=ctx.config.audio.tts.provider,
            model=audio.model,
            units=len(audio.text),
            cost_usd=audio.cost_usd,
            is_estimate=audio.cost_is_estimate,
            shot_id=audio.shot_id,
        )
    ctx.line_audio = audio
    return audio


def _produce_shot(ctx: RunContext, shot_id: int, *, fresh: bool = False) -> VideoArtifact:
    """Generate one shot until it passes its gates (clip duration, then verbatim speech).

    Tactics per failure kind are in docs/DATA_FLOW.md §6. Only this shot is regenerated; others
    are untouched. If native speech cannot be made verbatim, the run falls back to TTS overlay.
    """
    cfg, repo = ctx.config, ctx.repo
    if ctx.resumed and not fresh and (resumed := _resume_shot(ctx, shot_id)) is not None:
        return resumed
    assert ctx.strategy_used is not None
    strategy = ctx.strategy_used
    shot = ctx.shot(shot_id)
    needs_verify = shot.has_dialogue and strategy is AudioStrategy.NATIVE
    max_attempts = cfg.audio.native_attempts if needs_verify else cfg.video.max_attempts
    first = len(repo.get_attempts(ctx.run_id, shot_id))  # continue numbering after earlier cycles
    attempt = first
    strict = False
    last_failure = "none"
    with _stage(ctx, StageName.GENERATE_VIDEO), log_context(shot_id=shot_id):
        while attempt - first < max_attempts:
            attempt += 1
            artifact = _generate_attempt(ctx, shot_id, attempt, strategy, strict)
            if not _clip_duration_ok(ctx, artifact):
                last_failure = "duration"
                continue
            if not needs_verify:
                _accept(ctx, artifact)
                return artifact
            verdict = _verify_attempt(ctx, artifact)
            if not verdict.passed and verdict.verified and verdict.kind is FailureKind.STT_NOISE:
                # The transcriber, not the actor, is the suspect: listen again, keep the clip.
                log.warning("shot.stt_rerun", attempt=attempt, reason="failure_kind=stt_noise")
                verdict = _verify_attempt(ctx, artifact)
            if verdict.passed:
                _accept(ctx, artifact)
                return artifact
            last_failure = "speech"
            if not verdict.verified:
                _hitl(ctx, "on_unverified_speech", shot_id=shot_id, attempt=attempt)
                break  # STT is down; TTS is the only path that does not need it
            if verdict.kind is FailureKind.WRONG_LANGUAGE:
                break  # retrying the same model will not fix this; go to TTS
            if verdict.kind is FailureKind.TRUNCATED and _widen_dialogue_window(ctx):
                continue  # same prompt with a longer window for the line
            strict = True  # paraphrased / extra_words / silence / unknown → stricter prompt

        if last_failure == "duration":
            _raise_duration_exhausted(ctx, shot_id, attempt - first)
        if not needs_verify:
            msg = f"shot {shot_id} exhausted {max_attempts} generation attempts"
            raise PipelineError(msg, stage="generate_video", run_id=ctx.run_id)
        return _fallback_to_tts(ctx, shot_id, next_attempt=attempt + 1, tried=attempt - first)


def _fallback_to_tts(
    ctx: RunContext, shot_id: int, *, next_attempt: int, tried: int
) -> VideoArtifact:
    """Native speech could not be made verbatim: switch the run to TTS overlay and regenerate the
    dialogue shot once without speech. The TTS line is mixed in at assemble time."""
    reason = f"native speech failed verification after {tried} attempt(s)"
    log.warning("shot.fallback_to_tts", shot_id=shot_id, reason=reason)
    ctx.strategy_used = AudioStrategy.TTS_OVERLAY
    ctx.fallback_reason = reason
    ctx.repo.set_audio_strategy_used(ctx.run_id, AudioStrategy.TTS_OVERLAY, fallback_reason=reason)
    ctx.line_audio = _stage_audio(ctx)
    artifact = _generate_attempt(
        ctx, shot_id, next_attempt, AudioStrategy.TTS_OVERLAY, strict=False
    )
    if not _clip_duration_ok(ctx, artifact):
        _raise_duration_exhausted(ctx, shot_id, 1)
    _accept(ctx, artifact)
    return artifact


def _resume_shot(ctx: RunContext, shot_id: int) -> VideoArtifact | None:
    """A shot accepted by an earlier run of this run id is reused if its clip is still on disk."""
    row = next((a for a in ctx.repo.get_final_attempts(ctx.run_id) if a.shot_id == shot_id), None)
    if row is None or row.path is None or not Path(row.path).is_file() or row.id is None:
        return None
    artifact = VideoArtifact(
        shot_id=shot_id,
        attempt=row.attempt,
        strategy=AudioStrategy(row.strategy),
        path=Path(row.path),
        seed=row.seed,
        duration_s=row.duration_s or ctx.deps.ffmpeg.probe(Path(row.path)).duration_s,
        resolution=Resolution(row.resolution),
        has_native_audio=row.strategy == AudioStrategy.NATIVE.value,
        prompt_version=row.prompt_version,
        prompt_text=row.prompt_text,
        provider_request_id=row.provider_request_id,
        model=row.model,
        cost_usd=row.cost_usd,
    )
    ctx.shot_attempt_ids[(shot_id, row.attempt)] = row.id
    log.info("shot.resumed", shot_id=shot_id, attempt=row.attempt, path=row.path)
    return artifact


@dataclass(frozen=True)
class _Verdict:
    passed: bool
    verified: bool
    kind: FailureKind | None


def _generate_attempt(
    ctx: RunContext,
    shot_id: int,
    attempt: int,
    strategy: AudioStrategy,
    strict: bool,
) -> VideoArtifact:
    """One paid video call. A prompt the model rejects gets one LLM-softened rewrite."""
    dest = ctx.paths.shot_video(shot_id, attempt)
    generate = _with_retries(t_generate_shot, ctx.config)
    try:
        artifact = cast(
            "VideoArtifact",
            generate(
                ctx.current_script, ctx.shot(shot_id), ctx.deps, dest, attempt, strategy, strict
            ),
        )
    except FatalProviderError as exc:
        if shot_id in ctx.softened_shots or exc.status_code in _NO_SOFTEN_STATUS:
            raise
        _soften_shot(ctx, shot_id, exc)
        artifact = cast(
            "VideoArtifact",
            generate(
                ctx.current_script, ctx.shot(shot_id), ctx.deps, dest, attempt, strategy, strict
            ),
        )
    row = ctx.repo.add_shot_attempt(ctx.run_id, artifact)
    _record_cost(
        ctx,
        stage=StageName.GENERATE_VIDEO,
        provider=ctx.config.video.provider,
        model=artifact.model,
        units=artifact.duration_s,
        cost_usd=artifact.cost_usd,
        is_estimate=artifact.cost_is_estimate,
        shot_id=shot_id,
        attempt=attempt,
    )
    assert row.id is not None
    ctx.shot_attempt_ids[(shot_id, attempt)] = row.id
    return artifact


def _soften_shot(ctx: RunContext, shot_id: int, exc: FatalProviderError) -> None:
    log.warning(
        "shot.prompt_rejected",
        shot_id=shot_id,
        error=type(exc).__name__,
        message=exc.message,
        status_code=exc.status_code,
        tactic="soften prompt via LLM, retry once",
    )
    result = cast(
        "write_stage.ScriptResult",
        _with_retries(t_soften_shot, ctx.config)(
            ctx.current_script, shot_id, exc.message, ctx.idea, ctx.deps
        ),
    )
    _record_llm_cost(ctx, result)
    ctx.softened_shots.add(shot_id)
    _persist_script(ctx, result.script, reason="soften")


def _clip_duration_ok(ctx: RunContext, artifact: VideoArtifact) -> bool:
    """DurationGate per clip: the model must deliver roughly the requested length."""
    shot = ctx.shot(artifact.shot_id)
    tol = ctx.config.quality.shot_duration_tolerance_s
    result = DurationGate(min_s=shot.duration_s - tol, max_s=shot.duration_s + tol).check(
        artifact.duration_s
    )
    if result.passed:
        return True
    ctx.last_gate = result
    ctx.repo.mark_attempt(ctx.shot_attempt_ids[(artifact.shot_id, artifact.attempt)], "failed")
    log.warning(
        "shot.duration_gate_failed",
        shot_id=artifact.shot_id,
        attempt=artifact.attempt,
        measured=result.measured,
        threshold=result.threshold,
    )
    return False


def _raise_duration_exhausted(ctx: RunContext, shot_id: int, attempts: int) -> None:
    gate = ctx.last_gate
    assert gate is not None
    msg = f"shot {shot_id}: clip length stayed outside tolerance in {attempts} attempt(s)"
    raise QualityGateError(
        msg,
        gate=gate.gate,
        measured=gate.measured,
        threshold=gate.threshold,
        stage="generate_video",
        run_id=ctx.run_id,
        context={"shot_id": shot_id},
    )


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
    )


def _run_verification(
    ctx: RunContext,
    *,
    media_path: Path,
    shot_id: int,
    attempt: int,
    duration_s: float,
    window: tuple[float, float] | None,
) -> _Verdict:
    """STT + WER gate on a clip (or a window of the final); every outcome lands in the DB."""
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
                ),
            )
        except ProviderError as exc:
            verification = verify_stage.unverified(
                shot_id=shot_id,
                attempt=attempt,
                expected=expected,
                reason=f"STT unavailable: {exc}",
            )
            ctx.repo.add_speech_verification(ctx.run_id, attempt_id, verification)
            ctx.repo.mark_attempt(attempt_id, "unverified")
            return _Verdict(passed=False, verified=False, kind=FailureKind.UNKNOWN)
        v = result.verification
        ctx.repo.add_speech_verification(ctx.run_id, attempt_id, v)
        _record_cost(
            ctx,
            stage=StageName.VERIFY_SPEECH,
            provider=ctx.config.audio.stt.provider,
            model=v.stt_model or "stt",
            units=window[1] if window else duration_s,
            cost_usd=result.cost_usd,
            is_estimate=v.cost_is_estimate,
            shot_id=shot_id,
            attempt=attempt,
        )
        if result.decision is not None:
            ctx.repo.add_judge_decision(
                ctx.run_id,
                stage=StageName.VERIFY_SPEECH,
                question=verify_stage.Q_FAILURE_KIND,
                answer=result.decision.answer,
                probabilities=result.decision.probabilities,
                model_version=result.decision.model_version,
                latency_ms=result.decision.latency_ms,
            )
        if window is None:  # the final check must not downgrade an already accepted attempt
            ctx.repo.mark_attempt(attempt_id, "passed" if v.passed else "failed")
    return _Verdict(passed=v.passed, verified=True, kind=v.failure_kind)


def _accept(ctx: RunContext, artifact: VideoArtifact) -> None:
    attempt_id = ctx.shot_attempt_ids[(artifact.shot_id, artifact.attempt)]
    ctx.repo.mark_attempt(attempt_id, "accepted")
    ctx.repo.set_final_attempt(ctx.run_id, artifact.shot_id, attempt_id)
    if ctx.strategy_used is not None:
        ctx.repo.set_audio_strategy_used(ctx.run_id, ctx.strategy_used)


def _stage_assemble(ctx: RunContext) -> FinalVideo:
    """Concat + loudnorm + gates. A final that is too short gets the shortest shot extended once."""
    cfg = ctx.config
    with _stage(ctx, StageName.ASSEMBLE):
        try:
            return _assemble_once(ctx)
        except QualityGateError as exc:
            measured = exc.measured if isinstance(exc.measured, (int, float)) else None
            if exc.gate != "duration" or measured is None or measured >= cfg.quality.duration_s.min:
                raise
            if not _extend_shortest_shot(ctx, deficit_s=cfg.quality.duration_s.min - measured):
                raise
            return _assemble_once(ctx)


def _assemble_once(ctx: RunContext) -> FinalVideo:
    assert ctx.strategy_used is not None
    script = ctx.current_script
    voice = None
    if ctx.strategy_used is AudioStrategy.TTS_OVERLAY and ctx.line_audio is not None:
        voice = assemble_stage.VoiceOverlay(audio=ctx.line_audio, offset_s=script.dialogue.start_s)
    protect_until_s = ctx.shot_offset_s(script.dialogue.shot_id) + script.dialogue.end_s
    return cast(
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


def _extend_shortest_shot(ctx: RunContext, *, deficit_s: float) -> bool:
    """Final too short: request a longer version of the shortest shot and regenerate it."""
    script = ctx.current_script
    shortest = min(script.shots, key=lambda s: s.duration_s)
    new_duration = min(MAX_SHOT_S, shortest.duration_s + math.ceil(deficit_s) + 1)
    if new_duration == shortest.duration_s:
        log.warning("assemble.cannot_extend", shot_id=shortest.id, duration_s=shortest.duration_s)
        return False
    log.warning(
        "assemble.too_short",
        deficit_s=round(deficit_s, 2),
        shot_id=shortest.id,
        old_duration_s=shortest.duration_s,
        new_duration_s=new_duration,
    )
    _persist_script(
        ctx, script.with_shot_duration(shortest.id, new_duration), reason="extend_shortest_shot"
    )
    regenerated = _produce_shot(ctx, shortest.id, fresh=True)
    ctx.shot_artifacts = [
        regenerated if a.shot_id == shortest.id else a for a in ctx.shot_artifacts
    ]
    return True


def _verify_final(ctx: RunContext, final: FinalVideo) -> None:
    """tts_overlay sanity check: STT on the line window of the assembled final confirms the mux
    (and lipsync/loudnorm) did not break the audio. Native clips were already verified."""
    if ctx.strategy_used is not AudioStrategy.TTS_OVERLAY:
        return
    d = ctx.current_script.dialogue
    accepted = ctx.accepted(d.shot_id)
    assert accepted is not None
    start_s = ctx.shot_offset_s(d.shot_id) + d.start_s - _FINAL_CHECK_MARGIN_S
    window = (max(0.0, start_s), d.window_s + 2 * _FINAL_CHECK_MARGIN_S)
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
        msg = "the assembled final does not carry the line verbatim (mux/lipsync/loudnorm broke it)"
        exc = QualityGateError(
            msg,
            gate="speech",
            measured=None,
            threshold=ctx.config.quality.max_wer,
            stage="verify_speech",
            run_id=ctx.run_id,
            context={"path": str(final.path), "window": window},
        )
        # The verification itself ran fine; the *gate* failed, which is this stage's verdict.
        ctx.repo.finish_stage(ctx.run_id, StageName.VERIFY_SPEECH, StageStatus.FAILED, error=exc)
        raise exc


def _write_report(ctx: RunContext, final: FinalVideo | None) -> Path:
    with _stage(ctx, StageName.REPORT):
        rep = report.build_report(
            ctx.repo,
            ctx.run_id,
            ctx.config,
            dialogue_line=ctx.script.dialogue.text if ctx.script else None,
            final=final,
            preflight_rewrites=ctx.preflight_rewrites,
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


def _record_llm_cost(ctx: RunContext, result: write_stage.ScriptResult) -> None:
    _record_cost(
        ctx,
        stage=StageName.WRITE_SCRIPT,
        provider=ctx.config.llm.provider,
        model=result.model,
        units=result.input_tokens + result.output_tokens,
        cost_usd=result.cost_usd,
        is_estimate=result.cost_is_estimate,
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
        provider=provider,
        model=model,
        shot_id=shot_id,
        attempt=attempt,
        units=round(units, 3),
        cost_usd=round(cost_usd, 6),
        estimate=is_estimate,
        run_total_usd=round(run.cost_usd_total, 6) if run else None,
    )


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
