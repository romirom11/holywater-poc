"""generate_line_audio without ffmpeg or network: voice choice, measured length, the tts gate."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from holywater.config.schema import TTSConfig
from holywater.domain.artifacts import ProviderCost
from holywater.domain.media import MediaInfo
from holywater.domain.script import Script
from holywater.observability.errors import FatalProviderError, MediaError, QualityGateError
from holywater.providers.base import TTSRequest, TTSResult
from holywater.stages.generate_audio import TTS_CAPABILITY, AudioDeps, generate_line_audio
from tests.unit.test_domain import make_script


@dataclass
class _RecordingTTS:
    requests: list[TTSRequest] = field(default_factory=list)

    def synthesize(self, request: TTSRequest, *, dest: Path) -> TTSResult:
        self.requests.append(request)
        dest.write_bytes(b"mp3")
        return TTSResult(path=dest, model=request.model, cost_usd=0.01, cost_is_estimate=False)


@dataclass
class _StubProbe:
    duration_s: float = 2.4
    has_audio: bool = True

    def probe(self, path: Path) -> MediaInfo:
        return MediaInfo(
            duration_s=self.duration_s,
            width=0,
            height=0,
            fps=0.0,
            has_video=False,
            has_audio=self.has_audio,
            size_bytes=3,
        )


def _deps(tts: _RecordingTTS, probe: _StubProbe | None = None) -> AudioDeps:
    return AudioDeps(
        tts=tts,
        ffmpeg=probe or _StubProbe(),  # type: ignore[arg-type]
        config=TTSConfig(model="elevenlabs/eleven-v3", voice="sarah"),
    )


def test_config_voice_and_measured_duration(tmp_path: Path) -> None:
    tts = _RecordingTTS()
    audio = generate_line_audio(
        Script.model_validate(make_script()), _deps(tts), dest=tmp_path / "line.mp3"
    )
    assert tts.requests[0].voice == "sarah"
    assert tts.requests[0].text == "Are you sure this is the address?"
    assert audio.voice == "sarah"
    assert audio.duration_s == 2.4  # measured, so the flow can fit it into the shot
    assert (audio.cost_usd, audio.cost_is_estimate) == (0.01, False)


def test_voice_keyword_overrides_config(tmp_path: Path) -> None:
    tts = _RecordingTTS()
    audio = generate_line_audio(
        Script.model_validate(make_script()), _deps(tts), dest=tmp_path / "l.mp3", voice="george"
    )
    assert tts.requests[0].voice == "george"
    assert audio.voice == "george"


def test_tts_without_audio_fails_the_tts_gate(tmp_path: Path) -> None:
    tts = _RecordingTTS()
    with pytest.raises(QualityGateError) as exc_info:
        generate_line_audio(
            Script.model_validate(make_script()),
            _deps(tts, _StubProbe(duration_s=0.0, has_audio=False)),
            dest=tmp_path / "line.mp3",
        )
    assert exc_info.value.gate == "tts_audio"
    assert exc_info.value.stage == "generate_audio"
    assert exc_info.value.context["cost_usd"] == 0.01


class _BrokenProbe:
    def probe(self, path: Path) -> MediaInfo:
        raise MediaError("probe failed with exit code 1")


@dataclass
class _BilledFailingTTS:
    def synthesize(self, request: TTSRequest, *, dest: Path) -> TTSResult:
        msg = "upstream failed after billing"
        raise FatalProviderError(msg, provider="openrouter", model=request.model, spent_usd=0.02)


def test_charge_is_reported_before_the_probe(tmp_path: Path) -> None:
    """A billed TTS file that ffprobe cannot read: the MediaError is retried by the task (a new
    TTS call), so this call's charge must already be in the sink."""
    seen: list[ProviderCost] = []
    deps = AudioDeps(
        tts=_RecordingTTS(),
        ffmpeg=_BrokenProbe(),  # type: ignore[arg-type]
        config=TTSConfig(model="elevenlabs/eleven-v3", voice="sarah"),
    )
    with pytest.raises(MediaError):
        generate_line_audio(
            Script.model_validate(make_script()),
            deps,
            dest=tmp_path / "line.mp3",
            on_cost=seen.append,
        )
    assert [(c.capability, c.model, c.cost_usd, c.is_estimate, c.shot_id) for c in seen] == [
        (TTS_CAPABILITY, "elevenlabs/eleven-v3", 0.01, False, 2)
    ]


def test_charge_of_a_good_file_is_reported_once(tmp_path: Path) -> None:
    seen: list[ProviderCost] = []
    generate_line_audio(
        Script.model_validate(make_script()),
        _deps(_RecordingTTS()),
        dest=tmp_path / "line.mp3",
        on_cost=seen.append,
    )
    assert [(c.cost_usd, c.units) for c in seen] == [
        (0.01, len("Are you sure this is the address?"))
    ]


def test_spend_carried_by_a_provider_error_is_reported(tmp_path: Path) -> None:
    seen: list[ProviderCost] = []
    deps = AudioDeps(
        tts=_BilledFailingTTS(),
        ffmpeg=_StubProbe(),  # type: ignore[arg-type]
        config=TTSConfig(model="elevenlabs/eleven-v3", voice="sarah"),
    )
    with pytest.raises(FatalProviderError):
        generate_line_audio(
            Script.model_validate(make_script()),
            deps,
            dest=tmp_path / "line.mp3",
            on_cost=seen.append,
        )
    assert [(c.cost_usd, c.is_estimate) for c in seen] == [(0.02, True)]
