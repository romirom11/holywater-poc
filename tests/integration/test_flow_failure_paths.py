"""Failure paths of the flow (docs/DATA_FLOW.md §6, resume, costs) with fakes and the Prefect
harness. Each test names the tactic it pins; the happy path lives in test_flow_dry_run.py."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from structlog.testing import capture_logs

from holywater.config import PipelineConfig, Secrets, load_config
from holywater.domain.enums import AudioStrategy
from holywater.flows.generate_video_flow import generate_video_flow
from holywater.flows.wiring import PipelineDeps, build_deps
from holywater.media import FFmpeg
from holywater.observability.errors import (
    ConfigError,
    FatalProviderError,
    MediaError,
    RetryableProviderError,
    ScriptError,
)
from holywater.providers.base import JudgeAnswer, LLMResult, STTRequest, STTResult
from holywater.providers.fake import (
    FakeJudge,
    FakeLipsync,
    FakeLLM,
    FakeSTT,
    FakeTTS,
    FakeVideo,
    fake_idea,
    paraphrase_first_n,
)
from holywater.storage import new_run_id

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not FFmpeg.is_available(), reason="ffmpeg/ffprobe not on PATH"),
]

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config" / "pipeline.yaml"
LINE = "Are you sure this is the address?"


@pytest.fixture
def config(tmp_path: Path) -> PipelineConfig:
    raw = load_config(REPO_CONFIG).model_dump(mode="json")
    raw["storage"]["runs_dir"] = str(tmp_path / "runs")
    raw["storage"]["db_path"] = str(tmp_path / "runs" / "test.db")
    raw["retries"]["provider"] = {"attempts": 3, "backoff_s": [0, 0]}
    return PipelineConfig.model_validate(raw)


def _with(config: PipelineConfig, **sections: dict[str, Any]) -> PipelineConfig:
    """`config` with some keys of some sections replaced, e.g. `audio={"strategy": ...}`."""
    raw = config.model_dump(mode="json")
    for section, values in sections.items():
        raw[section].update(values)
    return PipelineConfig.model_validate(raw)


def _deps(config: PipelineConfig) -> PipelineDeps:
    return build_deps(config, Secrets(_env_file=None), dry_run=True)  # type: ignore[call-arg]


def _report(config: PipelineConfig, run_id: str) -> dict[str, Any]:
    return json.loads((config.storage.runs_dir / run_id / "report.json").read_text())  # type: ignore[no-any-return]


def _script(config: PipelineConfig, run_id: str) -> dict[str, Any]:
    return json.loads((config.storage.runs_dir / run_id / "script.json").read_text())  # type: ignore[no-any-return]


def _line_checks(deps: PipelineDeps, run_id: str) -> list[Any]:
    return [v for v in deps.repo.get_speech_verifications(run_id) if v.expected]


def _attempts(deps: PipelineDeps, run_id: str, shot_id: int) -> list[tuple[str, str]]:
    return [(a.strategy, a.status) for a in deps.repo.get_attempts(run_id, shot_id)]


# -- 1b preflight -------------------------------------------------------------------------------


class _FlagOnce(FakeJudge):
    """Flags the first safety question only: one rewrite fixes the script."""

    def ask_bool(self, state: str, question: str) -> JudgeAnswer:
        flagged = "safety_risk" in question and not any("safety_risk" in q for q in self.calls)
        answer = super().ask_bool(state, question)
        if not flagged:
            return answer
        return dataclasses.replace(answer, answer="yes", probabilities={"yes": 0.9, "no": 0.1})


def test_preflight_rewrite_revises_the_previous_script(config: PipelineConfig) -> None:
    deps = _deps(config)
    deps.judge = _FlagOnce()
    run_id = new_run_id()
    generate_video_flow(fake_idea(), deps, run_id=run_id)
    llm = deps.llm
    assert isinstance(llm, FakeLLM) and llm.calls == 2
    assert "pre-flight review of your previous script" in llm.users[1]
    assert '"visual_prompt": "Establishing shot derived from the idea."' in llm.users[1]
    report = _report(config, run_id)
    assert report["preflight_rewrites"] == 1 and report["success"] is True
    decisions = [(d.question, d.shot_id, d.decision) for d in deps.repo.get_judge_decisions(run_id)]
    assert decisions[0] == ("safety_risk", 1, "rewrite")
    assert decisions.count(("safety_risk", 1, "pass")) == 1  # the second round passed


def test_preflight_rejecting_every_round_is_a_script_error(config: PipelineConfig) -> None:
    deps = _deps(config)
    deps.judge = FakeJudge(bool_answers={"safety_risk": 0.9})
    run_id = new_run_id()
    with pytest.raises(ScriptError, match="pre-flight"):
        generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert isinstance(deps.llm, FakeLLM) and deps.llm.calls == config.llm.max_attempts
    assert isinstance(deps.video, FakeVideo) and deps.video.calls == 0
    report = _report(config, run_id)
    assert report["preflight_rewrites"] == config.llm.max_attempts - 1  # rewrites performed
    assert report["failure_stage"] == "write_script"
    rows = deps.repo.get_judge_decisions(run_id)
    assert len(rows) == 5 * config.llm.max_attempts
    assert {r.decision for r in rows if r.question == "safety_risk"} == {"rewrite"}


def test_unavailable_judge_is_recorded(config: PipelineConfig) -> None:
    """Jev down: preflight rows say `unavailable`; the classifier falls back to heuristics and
    that is recorded too (model_version "heuristic")."""
    deps = _deps(config)
    deps.judge = FakeJudge(available=False)
    deps.video = FakeVideo(ffmpeg=deps.ffmpeg, speech_mutator=paraphrase_first_n(2))
    run_id = new_run_id()
    generate_video_flow(fake_idea(), deps, run_id=run_id)
    rows = deps.repo.get_judge_decisions(run_id)
    pre = [r for r in rows if r.stage == "preflight_script"]
    assert len(pre) == 5 and {(r.answer, r.decision) for r in pre} == {("unknown", "unavailable")}
    verify = [(r.answer, r.decision, r.model_version, r.shot_id) for r in rows[5:]]
    assert verify == [
        ("unknown", "unavailable", None, 2),
        ("extra_words", "heuristic", "heuristic", 2),
    ]
    stages = {s.name: s.status for s in deps.repo.get_stages(run_id)}
    assert stages["preflight_script"] == "skipped"


def test_disabled_judge_leaves_one_unavailable_row(config: PipelineConfig) -> None:
    from holywater.providers.base import NoopJudge

    deps = _deps(config)
    deps.judge = NoopJudge()
    run_id = new_run_id()
    generate_video_flow(fake_idea(), deps, run_id=run_id)
    rows = [
        (r.question, r.answer, r.decision, r.model_version)
        for r in deps.repo.get_judge_decisions(run_id)
    ]
    assert rows == [("preflight", "unknown", "unavailable", "noop")]


# -- 1 script: costs of every billed LLM call -------------------------------------------------


def test_transient_llm_error_is_retried_with_backoff(config: PipelineConfig) -> None:
    deps = _deps(config)
    deps.llm = FakeLLM(fail_times=2)
    run_id = new_run_id()
    generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert deps.llm.calls == 3
    stage = next(s for s in _report(config, run_id)["stages"] if s["name"] == "write_script")
    assert (stage["status"], stage["retries"]) == ("done", 2)


def test_llm_cost_of_invalid_output_is_recorded(config: PipelineConfig) -> None:
    deps = _deps(config)
    deps.llm = FakeLLM(invalid_times=1, cost_usd=0.01)
    run_id = new_run_id()
    generate_video_flow(fake_idea(), deps, run_id=run_id)
    llm = [c for c in _report(config, run_id)["costs"] if c["stage"] == "write_script"]
    assert [c["cost_usd"] for c in llm] == [0.01, 0.01]  # the unusable answer was billed too
    assert all(c["provider"] == config.llm.provider for c in llm)


def test_llm_cost_is_recorded_when_the_script_stage_fails(config: PipelineConfig) -> None:
    deps = _deps(config)
    deps.llm = FakeLLM(line_override="Is this really the place?", override_times=99, cost_usd=0.01)
    run_id = new_run_id()
    with pytest.raises(ScriptError):
        generate_video_flow(fake_idea(), deps, run_id=run_id)
    report = _report(config, run_id)
    assert report["cost_usd_total"] == pytest.approx(0.01 * config.llm.max_attempts)
    assert len([c for c in report["costs"] if c["stage"] == "write_script"]) == 3


# -- 3 video: rejection -> soften -> TTS fallback --------------------------------------------


def test_dialogue_shot_rejected_after_soften_falls_back_to_tts(config: PipelineConfig) -> None:
    """Moderation that objects to the line itself: softening the visuals cannot help, but the
    tts_overlay clip's prompt does not contain the line."""
    deps = _deps(config)
    deps.llm = FakeLLM(cost_usd=0.01)
    deps.video = FakeVideo(ffmpeg=deps.ffmpeg, reject_if_contains=f'says "{LINE}"')
    run_id = new_run_id()
    outcome = generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert outcome.strategy_used is AudioStrategy.TTS_OVERLAY
    assert deps.repo.get_run(run_id).fallback_reason == "video prompt rejected"  # type: ignore[union-attr]
    assert deps.llm.calls == 2  # script + one soften
    assert _attempts(deps, run_id, 2) == [("tts_overlay", "accepted")]
    report = _report(config, run_id)
    assert report["success"] is True and report["line_verified"] is True
    soften = [c for c in report["costs"] if c["stage"] == "write_script" and c["shot_id"]]
    assert [(c["shot_id"], c["attempt"]) for c in soften] == [(2, 1)]  # charged to the cycle


def test_non_dialogue_shot_rejected_after_soften_is_fatal(config: PipelineConfig) -> None:
    from holywater.observability.errors import PromptRejectedError

    deps = _deps(config)
    deps.video = FakeVideo(ffmpeg=deps.ffmpeg, reject_if_contains="No one speaks")
    run_id = new_run_id()
    with pytest.raises(PromptRejectedError):
        generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert isinstance(deps.llm, FakeLLM) and deps.llm.calls == 2
    assert _report(config, run_id)["failure_stage"] == "generate_video"


# -- 3 video: duration gate ---------------------------------------------------------------------


def test_native_shot_falls_back_when_any_attempt_failed_on_speech(config: PipelineConfig) -> None:
    """a1 paraphrased, a2 the wrong length: the outcome is the TTS fallback, not a duration
    error (the order of failures no longer decides)."""
    deps = _deps(config)
    deps.video = FakeVideo(
        ffmpeg=deps.ffmpeg,
        speech_mutator=paraphrase_first_n(2),
        duration_offset_s=3.0,
        duration_offset_calls={3},
    )
    deps.judge = FakeJudge(choice_answers={"failure_kind": "paraphrased"})
    run_id = new_run_id()
    outcome = generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert outcome.strategy_used is AudioStrategy.TTS_OVERLAY
    assert _attempts(deps, run_id, 2) == [
        ("native", "failed"),
        ("native", "failed"),
        ("tts_overlay", "accepted"),
    ]
    assert deps.repo.get_run(run_id).fallback_reason == (  # type: ignore[union-attr]
        "exhausted 2 native attempts (last failure: paraphrased)"
    )


def test_fallback_clip_is_regenerated_on_a_duration_miss(config: PipelineConfig) -> None:
    deps = _deps(config)
    deps.video = FakeVideo(
        ffmpeg=deps.ffmpeg,
        speech_mutator=paraphrase_first_n(3),
        duration_offset_s=3.0,
        duration_offset_calls={4},  # the first TTS-fallback clip
    )
    deps.judge = FakeJudge(choice_answers={"failure_kind": "paraphrased"})
    run_id = new_run_id()
    outcome = generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert outcome.strategy_used is AudioStrategy.TTS_OVERLAY
    assert _attempts(deps, run_id, 2) == [
        ("native", "failed"),
        ("native", "failed"),
        ("tts_overlay", "failed"),
        ("tts_overlay", "accepted"),
    ]


def test_duration_error_counts_attempts_accurately(config: PipelineConfig) -> None:
    from holywater.observability.errors import QualityGateError

    deps = _deps(config)
    deps.video = FakeVideo(ffmpeg=deps.ffmpeg, duration_offset_s=3.0)
    with pytest.raises(QualityGateError, match="in all 2 attempt") as exc_info:
        generate_video_flow(fake_idea(), deps, run_id=new_run_id())
    assert exc_info.value.context["duration"] == 2 and exc_info.value.context["attempts"] == 2


# -- 4 verify ---------------------------------------------------------------------------------


def test_mute_native_clip_is_silence_not_a_media_error(config: PipelineConfig) -> None:
    deps = _deps(config)
    deps.video = FakeVideo(ffmpeg=deps.ffmpeg, mute_calls={2})  # shot 2, attempt 1
    run_id = new_run_id()
    outcome = generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert outcome.strategy_used is AudioStrategy.NATIVE
    checks = _line_checks(deps, run_id)
    assert [(v.transcript, v.failure_kind, v.verified) for v in checks] == [
        ("", "silence", True),
        (LINE, None, True),
    ]
    attempts = deps.repo.get_attempts(run_id, shot_id=2)
    assert [a.status for a in attempts] == ["failed", "accepted"]
    assert attempts[1].prompt_version.endswith("/dialogue_strict")  # the silence tactic
    stages = {s.name: (s.status, s.retries) for s in deps.repo.get_stages(run_id)}
    assert stages["verify_speech"] == ("done", 0)  # nothing retried on the mute file


def test_fatal_stt_error_fails_the_run_before_more_spending(config: PipelineConfig) -> None:
    """A 401/404 from STT is misconfiguration: no clip can ever be verified, so the run stops
    (typed error in stages.error) instead of paying for TTS and another clip."""
    deps = _deps(config)
    deps.stt = FakeSTT(fatal_status=401)
    run_id = new_run_id()
    with pytest.raises(FatalProviderError, match="HTTP 401"):
        generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert isinstance(deps.video, FakeVideo) and deps.video.calls == 1  # shot 1 only
    assert isinstance(deps.tts, FakeTTS) and deps.tts.calls == 0
    report = _report(config, run_id)
    assert report["failure_stage"] == "verify_speech" and report["success"] is False
    stage = next(s for s in deps.repo.get_stages(run_id) if s.name == "verify_speech")
    assert (stage.status, stage.error_type) == ("failed", "FatalProviderError")


def test_truncated_without_room_to_widen_uses_the_strict_prompt(config: PipelineConfig) -> None:
    cfg = _with(config, quality={"truncated_window_widen_s": 0.0})
    deps = _deps(cfg)
    deps.video = FakeVideo(
        ffmpeg=deps.ffmpeg,
        speech_mutator=lambda line, i: " ".join(line.split()[:3]) if i == 2 else line,
    )
    deps.judge = FakeJudge(choice_answers={"failure_kind": "truncated"})
    run_id = new_run_id()
    with capture_logs() as logs:
        generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert "shot.window_cannot_widen" in [e["event"] for e in logs]
    attempts = deps.repo.get_attempts(run_id, shot_id=2)
    assert attempts[1].prompt_version.endswith("/dialogue_strict")


# -- tts_overlay: the voice decides the window -------------------------------------------------


def test_slow_tts_extends_the_dialogue_shot_before_its_clip(config: PipelineConfig) -> None:
    """A voice longer than the shot leaves room for: the window becomes the voice, the shot
    grows before its clip is generated, and the final check listens to the whole voice."""
    cfg = _with(config, audio={"strategy": "tts_overlay"})
    deps = _deps(cfg)
    deps.tts = FakeTTS(ffmpeg=deps.ffmpeg, chars_per_s=4.0)  # 33 chars -> 8.25 s, from 1.0 s
    run_id = new_run_id()
    with capture_logs() as logs:
        outcome = generate_video_flow(fake_idea(), deps, run_id=run_id)
    script = _script(cfg, run_id)
    assert script["shots"][1]["duration_s"] == 10  # ceil(1.0 + 8.25 + 0.3)
    assert (script["dialogue"]["start_s"], script["dialogue"]["end_s"]) == (1.0, 9.55)
    clip = deps.repo.get_attempts(run_id, shot_id=2)[0]
    assert "0–10s" in clip.prompt_text  # the clip was requested at the new length
    assert "shot.extended_for_tts" in [e["event"] for e in logs]
    start = next(e for e in logs if e["event"] == "verify_final.start")
    assert start["window"] == [5.5, 9.25] or start["window"] == (5.5, 9.25)
    report = json.loads(outcome.report_path.read_text())
    assert report["line_verified"] is True and 8 <= outcome.final.duration_s <= 30


def test_tts_that_cannot_grow_the_shot_starts_the_line_earlier(config: PipelineConfig) -> None:
    cfg = _with(
        config, audio={"strategy": "tts_overlay"}, quality={"duration_s": {"min": 8, "max": 14}}
    )
    deps = _deps(cfg)
    deps.tts = FakeTTS(ffmpeg=deps.ffmpeg, chars_per_s=4.0)  # 8.25 s; shot 2 may grow to 9 s
    run_id = new_run_id()
    outcome = generate_video_flow(fake_idea(), deps, run_id=run_id)
    script = _script(cfg, run_id)
    assert script["shots"][1]["duration_s"] == 9
    assert (script["dialogue"]["start_s"], script["dialogue"]["end_s"]) == (0.45, 9.0)
    assert json.loads(outcome.report_path.read_text())["line_verified"] is True


def test_tts_line_that_cannot_fit_fails_before_any_clip(config: PipelineConfig) -> None:
    from holywater.observability.errors import QualityGateError

    cfg = _with(config, audio={"strategy": "tts_overlay"})
    deps = _deps(cfg)
    deps.tts = FakeTTS(ffmpeg=deps.ffmpeg, chars_per_s=1.0)  # a 33 s voice; shot 2 can reach 25 s
    run_id = new_run_id()
    with pytest.raises(QualityGateError) as exc_info:
        generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert exc_info.value.gate == "voice_fit"
    assert isinstance(deps.video, FakeVideo) and deps.video.calls == 0  # nothing paid for video
    assert _report(cfg, run_id)["failure_stage"] == "generate_audio"


def test_final_check_hears_only_its_window(config: PipelineConfig) -> None:
    """The window-aware fake STT really cuts: a window that misses the voice's tail fails the
    final check (guards the sizing above against regressions)."""
    from holywater.observability.errors import QualityGateError

    cfg = _with(config, audio={"strategy": "tts_overlay"}, quality={"final_check_margin_s": 0.0})
    deps = _deps(cfg)

    class _VoiceLongerThanItsFile(FakeTTS):
        """Claims a shorter audio file than the speech it carries (a mis-measured voice)."""

        def synthesize(self, request: Any, *, dest: Path) -> Any:
            result = super().synthesize(request, dest=dest)
            from holywater.providers.fake import write_sidecar

            write_sidecar(dest, request.text, start_s=0.0, end_s=6.0)
            return result

    deps.tts = _VoiceLongerThanItsFile(ffmpeg=deps.ffmpeg)
    with pytest.raises(QualityGateError, match="does not carry the line"):
        generate_video_flow(fake_idea(), deps, run_id=new_run_id())


def test_male_voice_is_picked_and_a_rejected_voice_falls_back(config: PipelineConfig) -> None:
    cfg = _with(config, audio={"strategy": "tts_overlay"})
    deps = _deps(cfg)
    deps.llm = FakeLLM(voice_gender="male")
    deps.tts = FakeTTS(ffmpeg=deps.ffmpeg, reject_voices={"george"})
    with capture_logs() as logs:
        generate_video_flow(fake_idea(), deps, run_id=new_run_id())
    assert deps.tts.voices == ["george", cfg.audio.tts.voice]
    rejected = next(e for e in logs if e["event"] == "tts.voice_rejected")
    assert rejected["log_level"] == "warning" and rejected["fallback_voice"] == "sarah"


# -- stray speech -------------------------------------------------------------------------------


def test_stray_speech_is_recorded_and_warned_by_default(config: PipelineConfig) -> None:
    deps = _deps(config)
    deps.video = FakeVideo(ffmpeg=deps.ffmpeg, stray_speech="hey you over there")
    run_id = new_run_id()
    with capture_logs() as logs:
        outcome = generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert deps.video.calls == 2  # warn: nothing regenerated
    stray = [v for v in deps.repo.get_speech_verifications(run_id) if not v.expected]
    assert [(v.transcript, v.passed, v.failure_kind) for v in stray] == [
        ("hey you over there", False, "extra_words")
    ]
    warned = next(e for e in logs if e["event"] == "shot.stray_speech")
    assert warned["log_level"] == "warning" and warned["words"] == 4
    assert json.loads(outcome.report_path.read_text())["line_verified"] is True


def test_stray_speech_regenerate_then_accept_with_warning(config: PipelineConfig) -> None:
    cfg = _with(config, quality={"stray_speech": {"action": "regenerate", "max_words": 2}})
    deps = _deps(cfg)
    deps.video = FakeVideo(ffmpeg=deps.ffmpeg, stray_speech="hey you there", stray_speech_times=1)
    run_id = new_run_id()
    generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert _attempts(deps, run_id, 1) == [("native", "failed"), ("native", "accepted")]

    deps = _deps(cfg)
    deps.video = FakeVideo(ffmpeg=deps.ffmpeg, stray_speech="hey you there")
    run_id = new_run_id()
    with capture_logs() as logs:
        generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert _attempts(deps, run_id, 1) == [("native", "accepted"), ("native", "failed")]
    assert "shot.stray_speech_accepted" in [e["event"] for e in logs]


# -- 5 assemble ---------------------------------------------------------------------------------


class _FlakyFFmpeg(FFmpeg):
    """loudnorm fails the first `failures` times (a transient ffmpeg error)."""

    def __init__(self, failures: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.failures = failures

    def loudnorm(self, src: Path, dst: Path, *, target_lufs: float = -16.0) -> Any:
        if self.failures > 0:
            self.failures -= 1
            raise MediaError("simulated loudnorm crash")
        return super().loudnorm(src, dst, target_lufs=target_lufs)


def test_assemble_media_error_is_retried_then_fatal(config: PipelineConfig) -> None:
    deps = _deps(config)
    deps.ffmpeg = _FlakyFFmpeg(1, timeout_s=config.retries.timeouts_s.ffmpeg)
    run_id = new_run_id()
    generate_video_flow(fake_idea(), deps, run_id=run_id)
    stage = next(s for s in deps.repo.get_stages(run_id) if s.name == "assemble")
    assert (stage.status, stage.retries) == ("done", 1)

    deps = _deps(config)
    deps.ffmpeg = _FlakyFFmpeg(99, timeout_s=config.retries.timeouts_s.ffmpeg)
    run_id = new_run_id()
    with pytest.raises(MediaError):
        generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert _report(config, run_id)["failure_stage"] == "assemble"


def test_long_final_is_trimmed_to_the_maximum(config: PipelineConfig) -> None:
    cfg = _with(config, quality={"duration_s": {"min": 8, "max": 12}})  # 13 s of clips
    outcome = generate_video_flow(fake_idea(), _deps(cfg), run_id=new_run_id())
    assert outcome.final.duration_s == pytest.approx(12.0, abs=0.05)  # the line ends at 8.7 s


# -- resume -------------------------------------------------------------------------------------


class _NewShot2LLM(FakeLLM):
    """Writes the same script except for a different shot-2 picture."""

    def complete_structured[T: BaseModel](self, **kwargs: Any) -> LLMResult[T]:
        result: LLMResult[Any] = super().complete_structured(**kwargs)
        script = result.data
        shots = [
            s.model_copy(update={"visual_prompt": "A different close-up of the speaker."})
            if s.id == 2
            else s
            for s in script.shots
        ]
        return dataclasses.replace(result, data=script.model_copy(update={"shots": shots}))


def test_resume_after_a_script_change_regenerates_only_mismatched_shots(
    config: PipelineConfig,
) -> None:
    run_id = new_run_id()
    generate_video_flow(fake_idea(), _deps(config), run_id=run_id)

    cfg = _with(config, llm={"model": "anthropic/another-model"})  # the script is rewritten
    deps = _deps(cfg)
    deps.llm = _NewShot2LLM()
    with capture_logs() as logs:
        outcome = generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert deps.llm.calls == 1
    assert isinstance(deps.video, FakeVideo) and deps.video.calls == 1  # shot 2 only
    mismatch = [e for e in logs if e["event"] == "shot.resume_mismatch"]
    assert [e["shot_id"] for e in mismatch] == [2]
    final_attempts = {a.shot_id: a.attempt for a in deps.repo.get_final_attempts(run_id)}
    assert final_attempts == {1: 1, 2: 2}
    assert "A different close-up" in deps.repo.get_attempts(run_id, 2)[-1].prompt_text
    assert json.loads(outcome.report_path.read_text())["line_verified"] is True


def test_resume_regenerates_a_clip_changed_on_disk(config: PipelineConfig) -> None:
    run_id = new_run_id()
    first = _deps(config)
    generate_video_flow(fake_idea(), first, run_id=run_id)
    clip = Path(first.repo.get_attempts(run_id, 1)[-1].path or "")
    clip.write_bytes(clip.read_bytes()[:-1024])  # truncated after it was recorded

    deps = _deps(config)
    with capture_logs() as logs:
        generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert isinstance(deps.video, FakeVideo) and deps.video.calls == 1  # shot 1 only
    mismatch = [e for e in logs if e["event"] == "shot.resume_mismatch"]
    assert [(e["shot_id"], e["log_level"]) for e in mismatch] == [(1, "warning")]
    assert "changed on disk" in mismatch[0]["reason"]
    final_attempts = {a.shot_id: a.attempt for a in deps.repo.get_final_attempts(run_id)}
    assert final_attempts == {1: 2, 2: 1}


def test_resume_after_a_resolution_change_keeps_the_script(config: PipelineConfig) -> None:
    run_id = new_run_id()
    generate_video_flow(fake_idea(), _deps(config), run_id=run_id)
    deps = _deps(_with(config, video={"resolution": "720p"}))
    outcome = generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert isinstance(deps.llm, FakeLLM) and deps.llm.calls == 0  # no paid rewrite
    assert isinstance(deps.video, FakeVideo) and deps.video.calls == 2  # 480p clips not reused
    assert outcome.final.height == 1280


class _CrashOnSecondCall(FakeSTT):
    """The process dies while the dialogue clip is being verified (after it was paid)."""

    def transcribe(self, request: STTRequest) -> STTResult:
        if self.calls == 1:
            self.calls += 1
            raise FatalProviderError("simulated crash", provider="fake", status_code=401)
        return super().transcribe(request)


def test_resume_gates_a_paid_clip_instead_of_paying_again(config: PipelineConfig) -> None:
    deps = _deps(config)
    deps.stt = _CrashOnSecondCall()
    run_id = new_run_id()
    with pytest.raises(FatalProviderError):
        generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert _attempts(deps, run_id, 2) == [("native", "generated")]

    deps = _deps(config)
    with capture_logs() as logs:
        outcome = generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert isinstance(deps.video, FakeVideo) and deps.video.calls == 0  # nothing paid twice
    assert _attempts(deps, run_id, 2) == [("native", "accepted")]
    assert "shot.resume_pending" in [e["event"] for e in logs]
    assert json.loads(outcome.report_path.read_text())["line_verified"] is True


def test_resume_reuses_the_tts_line(config: PipelineConfig) -> None:
    cfg = _with(config, audio={"strategy": "tts_overlay"})
    deps = _deps(cfg)
    deps.video = FakeVideo(ffmpeg=deps.ffmpeg, fail_from_call=2)
    run_id = new_run_id()
    with pytest.raises(RetryableProviderError):
        generate_video_flow(fake_idea(), deps, run_id=run_id)
    deps = _deps(cfg)
    generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert isinstance(deps.tts, FakeTTS) and deps.tts.calls == 0
    assert isinstance(deps.video, FakeVideo) and deps.video.calls == 1


def test_dry_run_and_real_runs_do_not_resume_each_other(config: PipelineConfig) -> None:
    deps = _deps(config)
    run_id = new_run_id()
    generate_video_flow(fake_idea(), deps, run_id=run_id)
    real = dataclasses.replace(deps, dry_run=False, llm=FakeLLM())
    with pytest.raises(ConfigError, match="dry run"):
        generate_video_flow(fake_idea(), real, run_id=run_id)
    assert isinstance(real.llm, FakeLLM) and real.llm.calls == 0  # refused before any call
    assert deps.repo.get_run(run_id).status == "done"  # type: ignore[union-attr]


# -- money: paid jobs are never bought twice by a task retry ------------------------------------


def test_unreadable_paid_clip_is_abandoned_not_bought_again_by_the_task(
    config: PipelineConfig,
) -> None:
    """A paid clip ffprobe cannot read is a BilledJobError: one abandoned attempt and one costs
    row, then the flow's attempt budget (not a Prefect retry) buys the next clip."""
    deps = _deps(config)
    deps.video = FakeVideo(ffmpeg=deps.ffmpeg, unreadable_calls={1}, usd_per_s=0.1)
    run_id = new_run_id()
    outcome = generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert deps.video.calls == 3  # 1 unreadable + 2 shots, no task retry of the paid call
    assert _attempts(deps, run_id, 1) == [("native", "abandoned"), ("native", "accepted")]
    video = [
        (c.shot_id, c.attempt) for c in deps.repo.get_costs(run_id) if c.stage == "generate_video"
    ]
    assert video == [(1, 1), (1, 2), (2, 1)]  # one row per paid job
    stage = next(s for s in deps.repo.get_stages(run_id) if s.name == "generate_video")
    assert stage.retries == 0
    assert json.loads(outcome.report_path.read_text())["success"] is True


def test_unreachable_job_stops_the_run_instead_of_buying_another(config: PipelineConfig) -> None:
    """A billed job whose endpoints answer 404 (or 401/403): the next job would fail the same
    way, so no second clip is bought."""
    from holywater.observability.errors import BilledJobError

    deps = _deps(config)
    deps.video = FakeVideo(
        ffmpeg=deps.ffmpeg, billed_times=9, billed_spent_usd=0.62, billed_status_code=404
    )
    run_id = new_run_id()
    with capture_logs() as logs, pytest.raises(BilledJobError):
        generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert isinstance(deps.video, FakeVideo) and deps.video.calls == 1
    assert _attempts(deps, run_id, 1) == [("native", "abandoned")]
    assert "shot.job_unreachable" in [e["event"] for e in logs]
    report = _report(config, run_id)
    assert report["cost_usd_total"] == pytest.approx(0.62)
    assert report["failure_stage"] == "generate_video"


# -- stray speech: advisory check, and the clip under the TTS voice -----------------------------


class _FirstCallFails(FakeSTT):
    """The first STT call (shot 1's stray check) fails with `status`; the rest work."""

    def __init__(self, status: int) -> None:
        super().__init__()
        self.status = status

    def transcribe(self, request: STTRequest) -> STTResult:
        if self.calls == 0:
            self.calls += 1
            msg = f"simulated STT error (HTTP {self.status})"
            raise FatalProviderError(msg, provider="fake", status_code=self.status)
        return super().transcribe(request)


def test_fatal_stt_error_on_the_advisory_stray_check_keeps_the_paid_clip(
    config: PipelineConfig,
) -> None:
    deps = _deps(config)
    deps.stt = _FirstCallFails(400)  # e.g. "audio contains no speech" on an ambient-only clip
    run_id = new_run_id()
    with capture_logs() as logs:
        outcome = generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert isinstance(deps.video, FakeVideo) and deps.video.calls == 2
    stray = [v for v in deps.repo.get_speech_verifications(run_id) if not v.expected]
    assert [(v.verified, v.passed) for v in stray] == [(False, False)]
    unavailable = next(e for e in logs if e["event"] == "shot.stray_check_unavailable")
    assert unavailable["log_level"] == "warning"
    assert (unavailable["error"], unavailable["status_code"]) == ("FatalProviderError", 400)
    assert json.loads(outcome.report_path.read_text())["line_verified"] is True


def test_stt_auth_error_on_the_stray_check_still_stops_the_run(config: PipelineConfig) -> None:
    deps = _deps(config)
    deps.stt = _FirstCallFails(401)  # every later STT call (the line check) would fail too
    run_id = new_run_id()
    with pytest.raises(FatalProviderError):
        generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert isinstance(deps.video, FakeVideo) and deps.video.calls == 1


def test_stray_speech_under_the_tts_voice_is_regenerated_even_with_warn(
    config: PipelineConfig,
) -> None:
    """Words in the clip under the TTS voice land in the line window and would fail the final
    check, so that clip is regenerated although `stray_speech.action` is `warn`."""
    cfg = _with(config, audio={"strategy": "tts_overlay"})
    assert cfg.quality.stray_speech.action == "warn"
    deps = _deps(cfg)
    deps.video = FakeVideo(ffmpeg=deps.ffmpeg, stray_speech="hey hey you", stray_speech_times=2)
    run_id = new_run_id()
    outcome = generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert _attempts(deps, run_id, 1) == [("tts_overlay", "accepted")]  # warn: kept
    assert _attempts(deps, run_id, 2) == [("tts_overlay", "failed"), ("tts_overlay", "accepted")]
    report = json.loads(outcome.report_path.read_text())
    assert report["success"] is True and report["line_verified"] is True


def test_final_check_failure_names_what_was_heard(config: PipelineConfig) -> None:
    from holywater.observability.errors import QualityGateError

    cfg = _with(config, audio={"strategy": "tts_overlay"})
    deps = _deps(cfg)
    deps.stt = FakeSTT(queue=["", "", "something else entirely"])
    with pytest.raises(QualityGateError, match="heard 'something else entirely'") as exc_info:
        generate_video_flow(fake_idea(), deps, run_id=new_run_id())
    assert exc_info.value.context["transcript"] == "something else entirely"
    assert exc_info.value.measured == pytest.approx(1.0)


# -- TTS and lipsync charges survive a failure after the paid call ------------------------------


class _CorruptFirstTTS(FakeTTS):
    """Charges $0.05 per call; the first file is unreadable (the task retries the call)."""

    def synthesize(self, request: Any, *, dest: Path) -> Any:
        from holywater.providers.base import TTSResult

        result = super().synthesize(request, dest=dest)
        if self.calls == 1:
            dest.write_bytes(b"\x00" * 512)
        return TTSResult(path=result.path, model=result.model, cost_usd=0.05)


def test_tts_charge_is_recorded_when_its_file_fails_the_probe(config: PipelineConfig) -> None:
    cfg = _with(config, audio={"strategy": "tts_overlay"})
    deps = _deps(cfg)
    deps.tts = _CorruptFirstTTS(ffmpeg=deps.ffmpeg)
    run_id = new_run_id()
    generate_video_flow(fake_idea(), deps, run_id=run_id)
    tts = [c for c in deps.repo.get_costs(run_id) if c.stage == "generate_audio"]
    assert [(c.provider, c.cost_usd, c.shot_id) for c in tts] == [("openrouter", 0.05, 2)] * 2


class _ShortLipsync(FakeLipsync):
    """Charges $0.40 and returns a clip 2 s shorter than its input (a broken lipsync result)."""

    def sync(self, request: Any, *, dest: Path) -> Any:
        from holywater.providers.base import LipsyncResult

        self.calls += 1
        info = self.ffmpeg.probe(request.video_path)
        self.ffmpeg.synth_clip(
            dest,
            duration_s=info.duration_s - 2.0,
            width=info.width,
            height=info.height,
            with_audio=True,
        )
        return LipsyncResult(path=dest, model="fake/lipsync", cost_usd=0.40, cost_is_estimate=False)


def test_lipsync_charge_is_recorded_when_assemble_fails(config: PipelineConfig) -> None:
    from holywater.observability.errors import QualityGateError

    cfg = _with(config, audio={"strategy": "tts_overlay", "lipsync": {"enabled": True}})
    deps = _deps(cfg)
    deps.lipsync = _ShortLipsync(ffmpeg=deps.ffmpeg)
    run_id = new_run_id()
    with pytest.raises(QualityGateError, match="lipsync_duration"):
        generate_video_flow(fake_idea(), deps, run_id=run_id)
    rows = [(c.stage, c.cost_usd) for c in deps.repo.get_costs(run_id) if c.stage == "assemble"]
    assert rows == [("assemble", 0.40)]
    assert _report(cfg, run_id)["cost_usd_total"] == pytest.approx(0.40)


# -- a short final extends a shot that is cheap to redo -----------------------------------------


class _ShortDialogueShotLLM(FakeLLM):
    """The dialogue shot is the shortest one (6 s + 4 s)."""

    def complete_structured(self, **kwargs: Any) -> Any:
        result = super().complete_structured(**kwargs)
        script = result.data
        shots = [
            script.shots[0].model_copy(update={"duration_s": 6}),
            script.shots[1].model_copy(update={"duration_s": 4}),
        ]
        dialogue = script.dialogue.model_copy(update={"start_s": 0.5, "end_s": 3.5})
        data = script.model_copy(update={"shots": shots, "dialogue": dialogue})
        return dataclasses.replace(result, data=data)


def test_short_final_extends_a_shot_without_the_verified_line(config: PipelineConfig) -> None:
    cfg = _with(
        config, quality={"duration_s": {"min": 10, "max": 30}, "shot_duration_tolerance_s": 3.0}
    )
    deps = _deps(cfg)
    deps.llm = _ShortDialogueShotLLM()
    deps.video = FakeVideo(ffmpeg=deps.ffmpeg, duration_offset_s=-1.0)
    run_id = new_run_id()
    generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert _attempts(deps, run_id, 1) == [("native", "superseded"), ("native", "accepted")]
    assert _attempts(deps, run_id, 2) == [("native", "accepted")]  # the verified line is kept
    assert len(_line_checks(deps, run_id)) == 1
    assert _script(cfg, run_id)["shots"][0]["duration_s"] > 6


# -- resume and refusals ------------------------------------------------------------------------


def test_resume_resynthesizes_a_damaged_tts_line(config: PipelineConfig) -> None:
    cfg = _with(config, audio={"strategy": "tts_overlay"})
    deps = _deps(cfg)
    deps.video = FakeVideo(ffmpeg=deps.ffmpeg, fail_from_call=2)
    run_id = new_run_id()
    with pytest.raises(RetryableProviderError):
        generate_video_flow(fake_idea(), deps, run_id=run_id)
    (line,) = (cfg.storage.runs_dir / run_id / "audio").glob("*.mp3")
    line.write_bytes(line.read_bytes()[:64])  # damaged on disk
    deps = _deps(cfg)
    with capture_logs() as logs:
        outcome = generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert isinstance(deps.tts, FakeTTS) and deps.tts.calls == 1
    assert "stage.resume_mismatch" in [e["event"] for e in logs]
    assert json.loads(outcome.report_path.read_text())["success"] is True


def test_resume_failure_is_not_blamed_on_a_stage_of_the_previous_invocation(
    config: PipelineConfig,
) -> None:
    from pydantic import ValidationError

    deps = _deps(config)
    deps.video = FakeVideo(ffmpeg=deps.ffmpeg, fail_from_call=2)
    run_id = new_run_id()
    with pytest.raises(RetryableProviderError):
        generate_video_flow(fake_idea(), deps, run_id=run_id)
    assert _report(config, run_id)["failure_stage"] == "generate_video"
    (config.storage.runs_dir / run_id / "script.json").write_text("{}")  # fails validation
    with pytest.raises(ValidationError):
        generate_video_flow(fake_idea(), _deps(config), run_id=run_id)
    report = _report(config, run_id)
    assert report["failure_type"] == "ValidationError"
    assert report["failure_stage"] is None  # unknown, not the stale generate_video row


def test_refused_start_leaves_no_run_directory(config: PipelineConfig) -> None:
    cfg = _with(config, hitl={"after_script": {"enabled": True}})
    run_id = new_run_id()
    with pytest.raises(ConfigError, match="after_script"):
        generate_video_flow(fake_idea(), _deps(cfg), run_id=run_id)
    assert not (cfg.storage.runs_dir / run_id).exists()
