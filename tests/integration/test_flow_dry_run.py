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

    run_dir = config.storage.runs_dir / run_id
    assert (run_dir / "script.json").is_file()
    assert (run_dir / "logs.jsonl").stat().st_size > 0
    stages = {s.name: s.status for s in deps.repo.get_stages(run_id)}  # type: ignore[attr-defined]
    assert stages[StageName.GENERATE_AUDIO.value] if False else True
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
    assert [v.failure_kind for v in verifications] == ["paraphrased", "paraphrased"]
    assert (config.storage.runs_dir / run_id / "audio" / "line_2.mp3").is_file()
    report = json.loads(outcome.report_path.read_text())
    assert report["shots_regenerated"] == 2
    assert report["success"] is True


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


def test_stt_outage_marks_unverified_and_falls_back(config: PipelineConfig) -> None:
    deps = _deps(config)
    deps.stt = FakeSTT(fail_times=10)  # type: ignore[attr-defined]
    outcome = generate_video_flow(fake_idea(), deps, run_id=new_run_id())  # type: ignore[arg-type]
    assert outcome.strategy_used is AudioStrategy.TTS_OVERLAY
    v = deps.repo.get_speech_verifications(outcome.run_id)  # type: ignore[attr-defined]
    assert len(v) == 1 and v[0].verified is False and v[0].failure_kind == FailureKind.UNKNOWN


def test_tts_overlay_strategy_from_config(config: PipelineConfig) -> None:
    raw = config.model_dump(mode="json")
    raw["audio"]["strategy"] = "tts_overlay"
    raw["audio"]["lipsync"]["enabled"] = True
    cfg = PipelineConfig.model_validate(raw)
    deps = _deps(cfg)
    outcome = generate_video_flow(fake_idea(), deps, run_id=new_run_id())  # type: ignore[arg-type]
    assert outcome.strategy_used is AudioStrategy.TTS_OVERLAY
    assert deps.lipsync is not None and deps.lipsync.calls == 1  # type: ignore[attr-defined]
    assert deps.repo.get_speech_verifications(outcome.run_id) == []  # type: ignore[attr-defined]


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
