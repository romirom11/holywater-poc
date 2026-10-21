"""The pipeline as a Prefect flow (docs/DATA_FLOW.md).

Tasks are thin wrappers around stages; retries (typed, with backoff) live on the tasks, while
quality-gate decisions — regenerate a shot, switch audio strategy, stop — live in the flow.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from holywater.config.schema import PipelineConfig
from holywater.domain.artifacts import AudioArtifact, FinalVideo, VideoArtifact
from holywater.domain.enums import AudioStrategy, FailureKind, RunStatus, StageName, StageStatus
from holywater.domain.idea import IdeaInput
from holywater.domain.script import Script, Shot
from holywater.flows.wiring import PipelineDeps
from holywater.observability.errors import (
    PipelineError,
    ProviderError,
    ScriptError,
    is_retryable,
)
from holywater.observability.logging import get_logger, log_context, run_log_file
from holywater.stages import assemble as assemble_stage
from holywater.stages import generate_audio, generate_video, preflight_script, report
from holywater.stages import verify_speech as verify_stage
from holywater.stages import write_script as write_stage
from holywater.storage import RunPaths, RunRepository

log = get_logger(__name__)


# -- tasks (one per stage; retries only on typed retryable errors) ----------------------------


def _retry_on_typed_error(_task: Any, _task_run: Any, state: Any) -> bool:
    try:
        state.result()
    except BaseException as exc:
        retry = is_retryable(exc)
        log.warning("task.retry_decision", error=type(exc).__name__, retry=retry)
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
    media_path: Path, shot_id: int, attempt: int, expected: str, language: str, deps: PipelineDeps
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
            quality=deps.config.quality,
            judge_config=deps.config.judge,
        ),
    )


@task(name="assemble", cache_policy=NO_CACHE)
def t_assemble(
    shots: list[VideoArtifact],
    deps: PipelineDeps,
    work_dir: Path,
    dest: Path,
    strategy_used: AudioStrategy,
    voice: assemble_stage.VoiceOverlay | None,
) -> FinalVideo:
    return assemble_stage.assemble(
        shots,
        assemble_stage.AssembleDeps(
            ffmpeg=deps.ffmpeg,
            quality=deps.config.quality,
            resolution=deps.config.video.resolution,
            lipsync=deps.lipsync,
        ),
        work_dir=work_dir,
        dest=dest,
        strategy_used=strategy_used,
        voice=voice,
    )


# -- flow -----------------------------------------------------------------------------------


@dataclass
class RunContext:
    run_id: str
    idea: IdeaInput
    deps: PipelineDeps
    paths: RunPaths
    tag: str | None = None
    strategy_used: AudioStrategy | None = None
    fallback_reason: str | None = None
    line_audio: AudioArtifact | None = None
    preflight_rewrites: int = 0
    shot_artifacts: list[VideoArtifact] = field(default_factory=list)
    shot_attempt_ids: dict[tuple[int, int], int] = field(default_factory=dict)

    @property
    def config(self) -> PipelineConfig:
        return self.deps.config

    @property
    def repo(self) -> RunRepository:
        return self.deps.repo


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
    deps.repo.create_run(
        run_id,
        idea,
        config_hash=cfg.config_hash(),
        audio_strategy=cfg.audio.strategy,
        tag=tag,
        git_sha=deps.git_sha,
    )
    final: FinalVideo | None = None
    script: Script | None = None
    with run_log_file(paths.logs_jsonl), log_context(run_id=run_id):
        log.info("run.start", idea=idea.text, dry_run=deps.dry_run, strategy=cfg.audio.strategy)
        try:
            script = _stage_script(ctx)
            ctx.strategy_used = cfg.audio.strategy
            if cfg.audio.strategy is AudioStrategy.TTS_OVERLAY:
                ctx.line_audio = _stage_audio(ctx, script)
            for shot in script.shots:
                ctx.shot_artifacts.append(_produce_shot(ctx, script, shot))
            final = _stage_assemble(ctx, script)
            deps.repo.finish_run(run_id, RunStatus.DONE, final_path=str(final.path))
        except BaseException as exc:
            log.error("run.failed", error=type(exc).__name__, message=str(exc), exc_info=True)
            deps.repo.finish_run(run_id, RunStatus.FAILED, error=exc)
            _write_report(ctx, script, final)
            raise
        report_path = _write_report(ctx, script, final)
        run = deps.repo.get_run(run_id)
        assert run is not None and ctx.strategy_used is not None
        log.info(
            "run.done",
            final=str(final.path),
            duration_s=round(final.duration_s, 2),
            strategy_used=ctx.strategy_used.value,
            cost_usd=round(run.cost_usd_total, 4),
        )
        return RunOutcome(
            run_id=run_id,
            final=final,
            report_path=report_path,
            strategy_used=ctx.strategy_used,
            cost_usd=run.cost_usd_total,
        )


# -- stage orchestration ----------------------------------------------------------------------


def _stage_script(ctx: RunContext) -> Script:
    cfg, repo = ctx.config, ctx.repo
    feedback: list[str] = []
    input_hash = _hash(ctx.idea.text, cfg.config_hash())
    with _stage(ctx, StageName.WRITE_SCRIPT, input_hash=input_hash):
        for round_no in range(1, cfg.llm.max_attempts + 1):
            result = cast(
                "write_stage.ScriptResult",
                _with_retries(t_write_script, cfg)(ctx.idea, ctx.deps, feedback),
            )
            repo.add_cost(
                ctx.run_id,
                stage=StageName.WRITE_SCRIPT,
                provider="openrouter",
                model=result.model,
                units=result.input_tokens + result.output_tokens,
                unit_price=0.0,
                cost_usd=result.cost_usd,
            )
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
        ctx.paths.script_json.write_text(script.model_dump_json(indent=2), encoding="utf-8")
        for shot in script.shots:
            repo.upsert_shot(
                ctx.run_id,
                shot.id,
                planned_duration_s=shot.duration_s,
                has_dialogue=shot.has_dialogue,
            )
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


def _stage_audio(ctx: RunContext, script: Script) -> AudioArtifact:
    if ctx.line_audio is not None:
        return ctx.line_audio
    with _stage(ctx, StageName.GENERATE_AUDIO):
        audio = cast(
            "AudioArtifact",
            _with_retries(t_generate_audio, ctx.config)(
                script, ctx.deps, ctx.paths.audio_line(script.dialogue.shot_id)
            ),
        )
        ctx.repo.add_cost(
            ctx.run_id,
            stage=StageName.GENERATE_AUDIO,
            provider="fal",
            model=audio.model,
            units=len(audio.text),
            unit_price=0.0,
            cost_usd=audio.cost_usd,
        )
    ctx.line_audio = audio
    return audio


def _produce_shot(ctx: RunContext, script: Script, shot: Shot) -> VideoArtifact:
    """Generate one shot until it passes its gates; fall back to TTS overlay if native speech
    cannot be made verbatim. Only this shot is regenerated — others are untouched."""
    cfg, repo = ctx.config, ctx.repo
    assert ctx.strategy_used is not None
    strategy = ctx.strategy_used
    needs_verify = shot.has_dialogue and strategy is AudioStrategy.NATIVE
    max_attempts = cfg.audio.native_attempts if needs_verify else cfg.video.max_attempts
    strict = False
    attempt = 0
    with _stage(ctx, StageName.GENERATE_VIDEO), log_context(shot_id=shot.id):
        while attempt < max_attempts:
            attempt += 1
            artifact = _generate_attempt(ctx, script, shot, attempt, strategy, strict)
            if not needs_verify:
                _accept(ctx, artifact)
                return artifact
            verdict = _verify_attempt(ctx, script, artifact)
            if verdict.passed:
                _accept(ctx, artifact)
                return artifact
            if verdict.kind is FailureKind.WRONG_LANGUAGE or not verdict.verified:
                break  # retrying the same model will not fix this; go to TTS
            strict = True  # paraphrased / extra_words / truncated / unknown → stricter prompt

        reason = f"native speech failed verification after {attempt} attempt(s)"
        if not needs_verify:
            msg = f"shot {shot.id} exhausted {max_attempts} generation attempts"
            raise PipelineError(msg, stage="generate_video", run_id=ctx.run_id)
        log.warning("shot.fallback_to_tts", shot_id=shot.id, reason=reason)
        ctx.strategy_used = AudioStrategy.TTS_OVERLAY
        ctx.fallback_reason = reason
        repo.set_audio_strategy_used(ctx.run_id, AudioStrategy.TTS_OVERLAY, fallback_reason=reason)
        ctx.line_audio = _stage_audio(ctx, script)
        artifact = _generate_attempt(
            ctx, script, shot, attempt + 1, AudioStrategy.TTS_OVERLAY, strict=False
        )
        _accept(ctx, artifact)
        return artifact


@dataclass(frozen=True)
class _Verdict:
    passed: bool
    verified: bool
    kind: FailureKind | None


def _generate_attempt(
    ctx: RunContext,
    script: Script,
    shot: Shot,
    attempt: int,
    strategy: AudioStrategy,
    strict: bool,
) -> VideoArtifact:
    dest = ctx.paths.shot_video(shot.id, attempt)
    artifact = cast(
        "VideoArtifact",
        _with_retries(t_generate_shot, ctx.config)(
            script, shot, ctx.deps, dest, attempt, strategy, strict
        ),
    )
    row = ctx.repo.add_shot_attempt(ctx.run_id, artifact)
    ctx.repo.add_cost(
        ctx.run_id,
        stage=StageName.GENERATE_VIDEO,
        provider="fal",
        model=artifact.model,
        units=artifact.duration_s,
        unit_price=0.0,
        cost_usd=artifact.cost_usd,
    )
    assert row.id is not None
    ctx.shot_attempt_ids[(shot.id, attempt)] = row.id
    return artifact


def _verify_attempt(ctx: RunContext, script: Script, artifact: VideoArtifact) -> _Verdict:
    attempt_id = ctx.shot_attempt_ids[(artifact.shot_id, artifact.attempt)]
    expected, language = script.dialogue.text, script.dialogue.language
    with _stage(ctx, StageName.VERIFY_SPEECH):
        try:
            result = cast(
                "verify_stage.VerifyResult",
                _with_retries(t_verify_speech, ctx.config)(
                    artifact.path, artifact.shot_id, artifact.attempt, expected, language, ctx.deps
                ),
            )
        except ProviderError as exc:
            verification = verify_stage.unverified(
                shot_id=artifact.shot_id,
                attempt=artifact.attempt,
                expected=expected,
                reason=f"STT unavailable: {exc}",
            )
            ctx.repo.add_speech_verification(ctx.run_id, attempt_id, verification)
            ctx.repo.mark_attempt(attempt_id, "unverified")
            return _Verdict(passed=False, verified=False, kind=FailureKind.UNKNOWN)
        v = result.verification
        ctx.repo.add_speech_verification(ctx.run_id, attempt_id, v)
        ctx.repo.add_cost(
            ctx.run_id,
            stage=StageName.VERIFY_SPEECH,
            provider="fal",
            model=v.stt_model or "stt",
            units=artifact.duration_s,
            unit_price=0.0,
            cost_usd=result.cost_usd,
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
        ctx.repo.mark_attempt(attempt_id, "passed" if v.passed else "failed")
    return _Verdict(passed=v.passed, verified=True, kind=v.failure_kind)


def _accept(ctx: RunContext, artifact: VideoArtifact) -> None:
    attempt_id = ctx.shot_attempt_ids[(artifact.shot_id, artifact.attempt)]
    ctx.repo.mark_attempt(attempt_id, "accepted")
    ctx.repo.set_final_attempt(ctx.run_id, artifact.shot_id, attempt_id)
    if ctx.strategy_used is not None:
        ctx.repo.set_audio_strategy_used(ctx.run_id, ctx.strategy_used)


def _stage_assemble(ctx: RunContext, script: Script) -> FinalVideo:
    assert ctx.strategy_used is not None
    voice = None
    if ctx.strategy_used is AudioStrategy.TTS_OVERLAY and ctx.line_audio is not None:
        voice = assemble_stage.VoiceOverlay(audio=ctx.line_audio, offset_s=script.dialogue.start_s)
    with _stage(ctx, StageName.ASSEMBLE):
        return cast(
            "FinalVideo",
            _with_retries(t_assemble, ctx.config)(
                ctx.shot_artifacts,
                ctx.deps,
                ctx.paths.work_dir,
                ctx.paths.final_video,
                ctx.strategy_used,
                voice,
            ),
        )


def _write_report(ctx: RunContext, script: Script | None, final: FinalVideo | None) -> Path:
    with _stage(ctx, StageName.REPORT):
        rep = report.build_report(
            ctx.repo,
            ctx.run_id,
            ctx.config,
            dialogue_line=script.dialogue.text if script else None,
            final=final,
            preflight_rewrites=ctx.preflight_rewrites,
        )
        return report.write_report(rep, ctx.paths.report_json)


# -- helpers ----------------------------------------------------------------------------------


class _stage:  # noqa: N801 - context manager used like a function
    """Marks stage start/finish in the DB and binds the stage name to log lines."""

    def __init__(self, ctx: RunContext, name: StageName, *, input_hash: str | None = None) -> None:
        self.ctx, self.name, self.input_hash = ctx, name, input_hash
        self._log_ctx = log_context(stage=name.value)

    def __enter__(self) -> None:
        self.ctx.repo.start_stage(self.ctx.run_id, self.name, input_hash=self.input_hash)
        self._log_ctx.__enter__()
        log.info("stage.start")

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
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
