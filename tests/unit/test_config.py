from pathlib import Path

import pytest

from holywater.config import PipelineConfig, Secrets, load_config
from holywater.config.settings import resolve_config_path
from holywater.domain.enums import AudioStrategy, Resolution
from holywater.observability.errors import ConfigError

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config" / "pipeline.yaml"


def test_repo_config_loads() -> None:
    cfg = load_config(REPO_CONFIG)
    assert cfg.video.resolution is Resolution.P480
    assert cfg.audio.strategy is AudioStrategy.NATIVE
    assert cfg.llm.model.startswith("anthropic/")
    assert len(cfg.config_hash()) == 16


def test_config_hash_changes_with_resolution() -> None:
    cfg = load_config(REPO_CONFIG)
    raw = cfg.model_dump(mode="json")
    raw["video"]["resolution"] = "720p"
    assert cfg.config_hash() != PipelineConfig.model_validate(raw).config_hash()


def test_missing_file_is_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml_is_config_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("llm: [unclosed")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path)


def test_unknown_key_is_config_error(tmp_path: Path) -> None:
    path = tmp_path / "extra.yaml"
    path.write_text(REPO_CONFIG.read_text() + "\nunknown_section: 1\n")
    with pytest.raises(ConfigError) as exc_info:
        load_config(path)
    assert exc_info.value.context["errors"][0]["loc"] == ("unknown_section",)


def test_out_of_range_value_is_config_error(tmp_path: Path) -> None:
    path = tmp_path / "range.yaml"
    path.write_text(REPO_CONFIG.read_text().replace("max_wer: 0.10", "max_wer: 1.5"))
    with pytest.raises(ConfigError, match="validation failed"):
        load_config(path)


def test_range_min_le_max(tmp_path: Path) -> None:
    path = tmp_path / "range.yaml"
    path.write_text(REPO_CONFIG.read_text().replace("{min: 8, max: 30}", "{min: 31, max: 30}"))
    with pytest.raises(ConfigError, match="validation failed"):
        load_config(path)


def test_config_path_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HW_CONFIG", str(tmp_path / "x.yaml"))
    assert resolve_config_path() == tmp_path / "x.yaml"
    assert resolve_config_path(Path("explicit.yaml")) == Path("explicit.yaml")


def test_secrets_require_lists_all_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("FAL_KEY", "OPENROUTER_API_KEY", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    secrets = Secrets(_env_file=None)  # type: ignore[call-arg]
    with pytest.raises(ConfigError) as exc_info:
        secrets.require("fal_key", "openrouter_api_key")
    assert exc_info.value.context["missing"] == ["fal_key", "openrouter_api_key"]
    assert "FAL_KEY" in str(exc_info.value)


def test_secrets_are_not_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAL_KEY", "super-secret")
    secrets = Secrets(_env_file=None)  # type: ignore[call-arg]
    assert "super-secret" not in repr(secrets)
    assert secrets.fal_key is not None
    assert secrets.fal_key.get_secret_value() == "super-secret"


def test_quality_tactic_knobs_have_defaults_and_bounds() -> None:
    from holywater.config.schema import QualityConfig

    q = QualityConfig()
    assert (q.shot_duration_tolerance_s, q.truncated_window_widen_s) == (1.0, 1.0)
    with pytest.raises(ValueError):
        QualityConfig(shot_duration_tolerance_s=-1)


def test_speech_rate_and_audio_peak_knobs(tmp_path: Path) -> None:
    from holywater.config.schema import QualityConfig

    q = load_config(REPO_CONFIG).quality
    assert (q.speech_chars_per_s, q.min_audio_peak_db) == (15.0, -50.0)
    assert (QualityConfig().speech_chars_per_s, QualityConfig().min_audio_peak_db) == (15.0, -50.0)
    for bad in ("speech_chars_per_s: 0", "min_audio_peak_db: 3"):
        path = tmp_path / "bad.yaml"
        key = bad.split(":")[0]
        text = REPO_CONFIG.read_text()
        line = next(ln for ln in text.splitlines() if ln.strip().startswith(f"{key}:"))
        path.write_text(text.replace(line, f"  {bad}"))
        with pytest.raises(ConfigError, match="validation failed"):
            load_config(path)


def test_final_check_margin_and_stray_speech_knobs(tmp_path: Path) -> None:
    from holywater.config.schema import QualityConfig, StraySpeechConfig

    q = load_config(REPO_CONFIG).quality
    assert q.final_check_margin_s == 0.5
    assert q.stray_speech == StraySpeechConfig(action="warn", max_words=2)
    assert QualityConfig().stray_speech.action == "warn"  # Whisper hears words in ambience
    for bad in (
        "final_check_margin_s: -0.1",
        "stray_speech: {action: drop, max_words: 2}",
        "stray_speech: {action: warn, max_words: -1}",
    ):
        key = bad.split(":")[0]
        text = REPO_CONFIG.read_text()
        line = next(ln for ln in text.splitlines() if ln.strip().startswith(f"{key}:"))
        path = tmp_path / "bad.yaml"
        path.write_text(text.replace(line, f"  {bad}"))
        with pytest.raises(ConfigError, match="validation failed"):
            load_config(path)


def test_tts_voices_by_gender() -> None:
    from holywater.config.schema import TTSConfig
    from holywater.domain.enums import VoiceGender

    tts = load_config(REPO_CONFIG).audio.tts
    assert tts.voices == {VoiceGender.FEMALE: "sarah", VoiceGender.MALE: "george"}
    assert tts.voice_for(VoiceGender.MALE) == "george"
    only_default = TTSConfig(model="m", voice="sarah")
    assert only_default.voice_for(VoiceGender.MALE) == "sarah"  # no entry -> the default voice
    with pytest.raises(ValueError):
        TTSConfig(model="m", voices={"robot": "x"})  # type: ignore[dict-item]


def test_quality_shots_knob_is_gone_and_duration_stays_within_task_bounds(
    tmp_path: Path,
) -> None:
    """Shot count 1..2 is a task invariant (domain), not a knob; the final length knob may only
    narrow the task's 8..30 s."""
    text = REPO_CONFIG.read_text()
    with_shots = tmp_path / "shots.yaml"
    with_shots.write_text(text.replace("quality:\n", "quality:\n  shots: {min: 1, max: 2}\n"))
    with pytest.raises(ConfigError) as exc_info:
        load_config(with_shots)
    assert exc_info.value.context["errors"][0]["loc"] == ("quality", "shots")
    for bounds in ("{min: 5, max: 30}", "{min: 8, max: 40}"):
        path = tmp_path / "bounds.yaml"
        path.write_text(text.replace("{min: 8, max: 30}", bounds))
        with pytest.raises(ConfigError, match="validation failed"):
            load_config(path)
    narrower = tmp_path / "narrow.yaml"
    narrower.write_text(text.replace("{min: 8, max: 30}", "{min: 10, max: 20}"))
    assert load_config(narrower).quality.duration_s.max == 20


def test_ffmpeg_timeout_knob() -> None:
    from holywater.config.schema import TimeoutsConfig

    assert load_config(REPO_CONFIG).retries.timeouts_s.ffmpeg == 600
    with pytest.raises(ValueError):
        TimeoutsConfig(ffmpeg=0)
