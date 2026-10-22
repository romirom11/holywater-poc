"""End-to-end runs of the Prefect flow with fake providers (no network, no cost)."""

import json
from pathlib import Path

import pytest

from holywater.config import PipelineConfig, Secrets, load_config
from holywater.domain.enums import AudioStrategy, FailureKind, StageName
from holywater.flows.generate_video_flow import generate_video_flow
from holywater.flows.wiring import build_deps
from holywater.media import FFmpeg
from holywater.providers.fake import FakeJudge, FakeSTT, FakeVideo, fake_idea, paraphrase_first_n
from holywater.storage import new_run_id

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not FFmpeg.is_available(), reason="ffmpeg/ffprobe not on PATH"),
]

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config" / "pipeline.yaml"


@pytest.fixture
def config(tmp_path: Path) -> PipelineConfig:
    raw = load_config(REPO_CONFIG).model_dump(mode="json")
    raw["storage"]["runs_dir"] = str(tmp_path / "runs")
    raw["storage"]["db_path"] = str(tmp_path / "runs" / "test.db")
    raw["retries"]["provider"] = {"attempts": 3, "backoff_s": [0, 0]}
    raw["audio"]["native_attempts"] = 2
    return PipelineConfig.model_validate(raw)


def _deps(config: PipelineConfig) -> object:
    return build_deps(config, Secrets(_env_file=None), dry_run=True)  # type: ignore[call-arg]


def test_happy_path_native(config: PipelineConfig) -> None:
    deps = _deps(config)
    run_id = new_run_id()
    outcome = generate_video_flow(fake_idea(), deps, run_id=run_id, tag="t")  # type: ignore[arg-type]

    assert outcome.final.path.is_file()
    assert 8 <= outcome.final.duration_s <= 30.5
    assert outcome.final.has_audio and outcome.final.height > outcome.final.width
    assert outcome.strategy_used is AudioStrategy.NATIVE

    report = json.loads(outcome.report_path.read_text())
    assert report["success"] is True
    assert report["dialogue_line"] == "Are you sure this is the address?"
    assert report["audio_strategy_used"] == "native"
    assert [s["name"] for s in report["stages"]][:2] == ["write_script", "preflight_script"]
    assert report["judge_decisions"] == 5
    assert report["shots_regenerated"] == 0
    assert report["retries_total"] == 0
    by_name = {s["name"]: s for s in report["stages"]}
    assert "report" not in by_name  # cannot observe its own completion; see DB for its status
    assert all(s["status"] in {"done", "skipped"} for s in by_name.values())
    assert by_name["generate_video"]["entries"] == 2  # one entry per shot, not a retry

    run_dir = config.storage.runs_dir / run_id
    assert (run_dir / "script.json").is_file()
    assert (run_dir / "logs.jsonl").stat().st_size > 0
    assert report["line_verified"] is True
    stages = {s.name: s.status for s in deps.repo.get_stages(run_id)}  # type: ignore[attr-defined]
    assert stages[StageName.GENERATE_AUDIO.value] == "skipped"  # native strategy: no TTS
    assert stages["assemble"] == "done" and stages["report"] == "done"


def test_wer_failure_retries_then_falls_back_to_tts(config: PipelineConfig) -> None:
    deps = _deps(config)
    deps.video = FakeVideo(ffmpeg=deps.ffmpeg, speech_mutator=paraphrase_first_n(10))  # type: ignore[attr-defined]
    deps.judge = FakeJudge(choice_answers={"failure_kind": "paraphrased"})  # type: ignore[attr-defined]
    run_id = new_run_id()
    outcome = generate_video_flow(fake_idea(), deps, run_id=run_id)  # type: ignore[arg-type]

    assert outcome.strategy_used is AudioStrategy.TTS_OVERLAY
    repo = deps.repo  # type: ignore[attr-defined]
    run = repo.get_run(run_id)
    assert run.audio_strategy_used == "tts_overlay"
    assert "failed verification" in run.fallback_reason
    attempts = repo.get_attempts(run_id, shot_id=2)
    assert [a.strategy for a in attempts] == ["native", "native", "tts_overlay"]
    assert [a.status for a in attempts] == ["failed", "failed", "accepted"]
    assert attempts[1].prompt_version.endswith("/dialogue_strict")
    verifications = repo.get_speech_verifications(run_id)
    assert [v.failure_kind for v in verifications] == ["paraphrased", "paraphrased", None]
    assert verifications[-1].passed and verifications[-1].shot_attempt_id == attempts[2].id
    assert (config.storage.runs_dir / run_id / "audio" / "line_2.mp3").is_file()
    report = json.loads(outcome.report_path.read_text())
    assert report["shots_regenerated"] == 2
    assert report["success"] is True and report["line_verified"] is True
    stages = {s.name: s.status for s in repo.get_stages(run_id)}
    assert stages["generate_audio"] == "done"  # skipped first, then ran for the fallback


def test_wrong_language_falls_back_immediately(config: PipelineConfig) -> None:
    deps = _deps(config)
    deps.video = FakeVideo(ffmpeg=deps.ffmpeg, speech_mutator=lambda _line, _i: "зовсім інше")  # type: ignore[attr-defined]
    deps.judge = FakeJudge(choice_answers={"failure_kind": "wrong_language"})  # type: ignore[attr-defined]
    outcome = generate_video_flow(fake_idea(), deps, run_id=new_run_id())  # type: ignore[arg-type]
    attempts = deps.repo.get_attempts(outcome.run_id, shot_id=2)  # type: ignore[attr-defined]
    assert [a.strategy for a in attempts] == ["native", "tts_overlay"]


def test_transient_provider_error_is_retried(config: PipelineConfig) -> None:
    deps = _deps(config)
    deps.video = FakeVideo(ffmpeg=deps.ffmpeg, fail_times=2)  # type: ignore[attr-defined]
    outcome = generate_video_flow(fake_idea(), deps, run_id=new_run_id())  # type: ignore[arg-type]
    assert outcome.final.path.is_file()
    assert deps.video.calls == 4  # 2 failures + 2 shots  # type: ignore[attr-defined]
    report = json.loads(outcome.report_path.read_text())
    assert report["success"] is True
    assert report["shots_regenerated"] == 0
    assert report["retries_total"] == 2
    by_name = {s["name"]: s for s in report["stages"]}
    assert (by_name["generate_video"]["retries"], by_name["generate_video"]["entries"]) == (2, 2)
    assert by_name["write_script"]["retries"] == 0


def test_stt_outage_marks_unverified_and_falls_back(config: PipelineConfig) -> None:
    deps = _deps(config)
    deps.stt = FakeSTT(fail_times=10)  # type: ignore[attr-defined]
    outcome = generate_video_flow(fake_idea(), deps, run_id=new_run_id())  # type: ignore[arg-type]
    assert outcome.strategy_used is AudioStrategy.TTS_OVERLAY
    v = deps.repo.get_speech_verifications(outcome.run_id)  # type: ignore[attr-defined]
    assert len(v) == 2  # the native clip, then the assembled final: both unverified
    assert all(not x.verified and x.failure_kind == FailureKind.UNKNOWN for x in v)
    report = json.loads(outcome.report_path.read_text())
    assert report["success"] is True and report["line_verified"] is False  # R4 not evidenced


def test_tts_overlay_strategy_from_config(config: PipelineConfig) -> None:
    raw = config.model_dump(mode="json")
    raw["audio"]["strategy"] = "tts_overlay"
    raw["audio"]["lipsync"]["enabled"] = True
    cfg = PipelineConfig.model_validate(raw)
    deps = _deps(cfg)
    outcome = generate_video_flow(fake_idea(), deps, run_id=new_run_id())  # type: ignore[arg-type]
    assert outcome.strategy_used is AudioStrategy.TTS_OVERLAY
    assert deps.lipsync is not None and deps.lipsync.calls == 1  # type: ignore[attr-defined]
    v = deps.repo.get_speech_verifications(outcome.run_id)  # type: ignore[attr-defined]
    assert len(v) == 1 and v[0].passed  # sanity check on the assembled final, not on a clip
    attempts = deps.repo.get_attempts(outcome.run_id, shot_id=2)  # type: ignore[attr-defined]
    assert attempts[0].status == "accepted" and v[0].shot_attempt_id == attempts[0].id
    report = json.loads(outcome.report_path.read_text())
    assert report["line_verified"] is True
    stages = {s.name: s.status for s in deps.repo.get_stages(outcome.run_id)}  # type: ignore[attr-defined]
    assert stages["generate_audio"] == "done"


def test_tts_overlay_broken_mux_fails_the_run(config: PipelineConfig) -> None:
    from holywater.observability.errors import QualityGateError

    raw = config.model_dump(mode="json")
    raw["audio"]["strategy"] = "tts_overlay"
    cfg = PipelineConfig.model_validate(raw)
    deps = _deps(cfg)
    deps.stt = FakeSTT(queue=["something completely different was heard"])  # type: ignore[attr-defined]
    run_id = new_run_id()
    with pytest.raises(QualityGateError, match="does not carry the line"):
        generate_video_flow(fake_idea(), deps, run_id=run_id)  # type: ignore[arg-type]
    report = json.loads((cfg.storage.runs_dir / run_id / "report.json").read_text())
    assert report["success"] is False and report["failure_stage"] == "verify_speech"
    assert report["line_verified"] is False


def test_script_failure_writes_failed_report(config: PipelineConfig) -> None:
    from holywater.observability.errors import ScriptError
    from holywater.providers.fake import FakeLLM

    deps = _deps(config)
    deps.llm = FakeLLM(line_override="not in the input", override_times=99)  # type: ignore[attr-defined]
    run_id = new_run_id()
    with pytest.raises(ScriptError):
        generate_video_flow(fake_idea(), deps, run_id=run_id)  # type: ignore[arg-type]
    report = json.loads((config.storage.runs_dir / run_id / "report.json").read_text())
    assert report["success"] is False
    assert report["failure_stage"] == "write_script"
    assert report["failure_type"] == "ScriptError"


# -- docs/DATA_FLOW.md §6 tactics -------------------------------------------------------------


def test_clip_duration_gate_regenerates_then_gives_up(config: PipelineConfig) -> None:
    from holywater.observability.errors import QualityGateError

    deps = _deps(config)
    # First clip comes back 3 s too long -> rejected and regenerated; the rest are fine.
    deps.video = FakeVideo(ffmpeg=deps.ffmpeg, duration_offset_s=3.0, duration_offset_times=1)  # type: ignore[attr-defined]
    outcome = generate_video_flow(fake_idea(), deps, run_id=new_run_id())  # type: ignore[arg-type]
    attempts = deps.repo.get_attempts(outcome.run_id, shot_id=1)  # type: ignore[attr-defined]
    assert [a.status for a in attempts] == ["failed", "accepted"]
    report = json.loads(outcome.report_path.read_text())
    assert report["shots_regenerated"] == 1 and report["success"] is True

    deps = _deps(config)
    deps.video = FakeVideo(ffmpeg=deps.ffmpeg, duration_offset_s=3.0)  # type: ignore[attr-defined]
    run_id = new_run_id()
    with pytest.raises(QualityGateError) as exc_info:
        generate_video_flow(fake_idea(), deps, run_id=run_id)  # type: ignore[arg-type]
    assert exc_info.value.gate == "duration"
    assert deps.video.calls == config.video.max_attempts  # type: ignore[attr-defined]
    report = json.loads((config.storage.runs_dir / run_id / "report.json").read_text())
    assert report["failure_stage"] == "generate_video"


def test_rejected_prompt_is_softened_once(config: PipelineConfig) -> None:
    from holywater.observability.errors import FatalProviderError

    deps = _deps(config)
    deps.video = FakeVideo(ffmpeg=deps.ffmpeg, reject_times=1)  # type: ignore[attr-defined]
    outcome = generate_video_flow(fake_idea(), deps, run_id=new_run_id())  # type: ignore[arg-type]
    assert outcome.final.path.is_file()
    assert deps.llm.calls == 2  # script + one softening rewrite  # type: ignore[attr-defined]
    assert deps.video.calls == 3  # rejected, retried, shot 2  # type: ignore[attr-defined]
    report = json.loads(outcome.report_path.read_text())
    assert report["shots_regenerated"] == 0  # a rejection produces no clip, so no attempt row
    assert report["cost_summary"]["by_stage"].keys() >= {"write_script", "generate_video"}

    deps = _deps(config)
    deps.video = FakeVideo(ffmpeg=deps.ffmpeg, reject_times=2)  # type: ignore[attr-defined]
    with pytest.raises(FatalProviderError):
        generate_video_flow(fake_idea(), deps, run_id=new_run_id())  # type: ignore[arg-type]
    assert deps.llm.calls == 2  # only one softening attempt, then fatal  # type: ignore[attr-defined]


def test_truncated_line_widens_the_window_instead_of_strict_prompt(
    config: PipelineConfig,
) -> None:
    deps = _deps(config)
    # Call 2 is shot 2's first attempt: the "actor" drops the end of the line once.
    deps.video = FakeVideo(  # type: ignore[attr-defined]
        ffmpeg=deps.ffmpeg,
        speech_mutator=lambda line, i: " ".join(line.split()[:3]) if i == 2 else line,
    )
    deps.judge = FakeJudge(choice_answers={"failure_kind": "truncated"})  # type: ignore[attr-defined]
    run_id = new_run_id()
    outcome = generate_video_flow(fake_idea(), deps, run_id=run_id)  # type: ignore[arg-type]
    assert outcome.strategy_used is AudioStrategy.NATIVE
    attempts = deps.repo.get_attempts(run_id, shot_id=2)  # type: ignore[attr-defined]
    assert [a.status for a in attempts] == ["failed", "accepted"]
    assert all(a.prompt_version.endswith("/dialogue") for a in attempts)  # not strict
    script = json.loads((config.storage.runs_dir / run_id / "script.json").read_text())
    widen = config.quality.truncated_window_widen_s
    assert script["dialogue"]["start_s"] == pytest.approx(1.0 - widen)
    assert f"{script['dialogue']['start_s']:g}" in attempts[1].prompt_text


def test_stt_noise_reruns_stt_without_regenerating_video(config: PipelineConfig) -> None:
    deps = _deps(config)
    deps.stt = FakeSTT(queue=["are you sure this is the adress"])  # type: ignore[attr-defined]
    deps.judge = FakeJudge(choice_answers={"failure_kind": "stt_noise"})  # type: ignore[attr-defined]
    run_id = new_run_id()
    outcome = generate_video_flow(fake_idea(), deps, run_id=run_id)  # type: ignore[arg-type]
    assert outcome.strategy_used is AudioStrategy.NATIVE
    assert deps.video.calls == 2  # no extra clip was paid for  # type: ignore[attr-defined]
    v = deps.repo.get_speech_verifications(run_id)  # type: ignore[attr-defined]
    assert [x.passed for x in v] == [False, True] and {x.shot_attempt_id for x in v} == {
        v[0].shot_attempt_id
    }
    report = json.loads(outcome.report_path.read_text())
    assert report["shots_regenerated"] == 0 and report["line_verified"] is True


def test_short_final_extends_the_shortest_shot(config: PipelineConfig) -> None:
    raw = config.model_dump(mode="json")
    raw["quality"]["duration_s"] = {"min": 10, "max": 30}
    raw["quality"]["shot_duration_tolerance_s"] = 3.0  # let 2 s short clips through the clip gate
    cfg = PipelineConfig.model_validate(raw)
    deps = _deps(cfg)
    deps.video = FakeVideo(ffmpeg=deps.ffmpeg, duration_offset_s=-2.0)  # type: ignore[attr-defined]
    run_id = new_run_id()
    outcome = generate_video_flow(fake_idea(), deps, run_id=run_id)  # type: ignore[arg-type]
    assert outcome.final.duration_s >= 10 - 0.5
    script = json.loads((cfg.storage.runs_dir / run_id / "script.json").read_text())
    assert script["shots"][0]["duration_s"] == 7  # 5 s shot, 1 s deficit + 1 s margin
    attempts = deps.repo.get_attempts(run_id, shot_id=1)  # type: ignore[attr-defined]
    assert [a.attempt for a in attempts] == [1, 2] and attempts[1].status == "accepted"
    assert deps.repo.get_attempts(run_id, shot_id=2)[0].status == "accepted"  # untouched  # type: ignore[attr-defined]


def test_trim_never_cuts_the_line(config: PipelineConfig) -> None:
    from holywater.observability.errors import QualityGateError

    raw = config.model_dump(mode="json")
    raw["quality"]["duration_s"] = {"min": 8, "max": 8}  # 13 s final must be trimmed to 8 s
    cfg = PipelineConfig.model_validate(raw)
    deps = _deps(cfg)
    run_id = new_run_id()
    with pytest.raises(QualityGateError, match="would cut it"):
        generate_video_flow(fake_idea(), deps, run_id=run_id)  # type: ignore[arg-type]


def test_enabled_hitl_point_fails_before_spending(config: PipelineConfig) -> None:
    from holywater.observability.errors import ConfigError

    raw = config.model_dump(mode="json")
    raw["hitl"]["after_script"] = {"enabled": True}
    cfg = PipelineConfig.model_validate(raw)
    deps = _deps(cfg)
    with pytest.raises(ConfigError, match="after_script"):
        generate_video_flow(fake_idea(), deps, run_id=new_run_id())  # type: ignore[arg-type]
    assert deps.llm.calls == 0 and deps.video.calls == 0  # type: ignore[attr-defined]


def test_resume_reuses_script_and_accepted_shots(config: PipelineConfig) -> None:
    from holywater.observability.errors import ConfigError, RetryableProviderError

    deps = _deps(config)
    # Shot 1 succeeds, shot 2 keeps failing transiently until the retries are exhausted.
    deps.video = FakeVideo(ffmpeg=deps.ffmpeg, fail_from_call=2)  # type: ignore[attr-defined]
    run_id = new_run_id()
    with pytest.raises(RetryableProviderError):
        generate_video_flow(fake_idea(), deps, run_id=run_id)  # type: ignore[arg-type]
    assert deps.repo.get_run(run_id).status == "failed"  # type: ignore[attr-defined]

    deps.video = FakeVideo(ffmpeg=deps.ffmpeg)  # provider is healthy again  # type: ignore[attr-defined]
    outcome = generate_video_flow(fake_idea(), deps, run_id=run_id)  # type: ignore[arg-type]
    assert outcome.final.path.is_file()
    assert deps.llm.calls == 1  # script was not rewritten  # type: ignore[attr-defined]
    assert deps.video.calls == 1  # only shot 2 was generated  # type: ignore[attr-defined]
    assert [a.attempt for a in deps.repo.get_attempts(run_id, shot_id=1)] == [1]  # type: ignore[attr-defined]
    report = json.loads(outcome.report_path.read_text())
    assert report["success"] is True and report["retries_total"] == 2
    assert len(report["costs"]) == 4  # llm, shot 1, shot 2, stt: nothing paid twice

    with pytest.raises(ConfigError, match="different idea"):
        generate_video_flow(fake_idea('Other idea: "Hello there"'), deps, run_id=run_id)  # type: ignore[arg-type]
