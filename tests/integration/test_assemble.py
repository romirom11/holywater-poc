"""assemble() on real ffmpeg output: audio format, final length, integrity, voice, lipsync."""

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from holywater.config.schema import LipsyncConfig, QualityConfig
from holywater.domain.artifacts import AudioArtifact, FinalVideo, ProviderCost, VideoArtifact
from holywater.domain.enums import AudioStrategy, Resolution
from holywater.media import FFmpeg
from holywater.observability.errors import QualityGateError
from holywater.providers.base import LipsyncRequest, LipsyncResult
from holywater.stages.assemble import AssembleDeps, VoiceOverlay, assemble

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not FFmpeg.is_available(), reason="ffmpeg/ffprobe not on PATH"),
]

SILENT_DB = -60.0  # anything below is silence for these assertions
AUDIBLE_DB = -30.0


@pytest.fixture(scope="module")
def ff() -> FFmpeg:
    return FFmpeg(timeout_s=120)


def _deps(ff: FFmpeg, **kwargs: object) -> AssembleDeps:
    return AssembleDeps(
        ffmpeg=ff,
        quality=QualityConfig(),  # duration_s 8..30
        resolution=Resolution.P480,
        **kwargs,  # type: ignore[arg-type]
    )


def _shot(path: Path, shot_id: int, duration_s: float) -> VideoArtifact:
    return VideoArtifact(
        shot_id=shot_id,
        attempt=1,
        strategy=AudioStrategy.NATIVE,
        path=path,
        duration_s=duration_s,
        resolution=Resolution.P480,
        has_native_audio=True,
        prompt_version="test",
        prompt_text="test",
        model="test/video",
    )


def _stereo_noise_clip(dst: Path, *, duration_s: float, sample_rate: int = 32000) -> Path:
    """Seedance-like clip: stereo AAC at 32 kHz with audible noise."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=240x426:rate=24",
            "-f",
            "lavfi",
            "-i",
            f"anoisesrc=color=pink:sample_rate={sample_rate}:amplitude=0.2",
            "-t",
            f"{duration_s:.3f}",
            "-ac",
            "2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            str(sample_rate),
            str(dst),
        ],
        check=True,
    )
    return dst


def _clip(ff: FFmpeg, dst: Path, duration_s: float, *, with_audio: bool = True) -> Path:
    """synth_clip audio is a mono 48 kHz tone."""
    ff.synth_clip(dst, duration_s=duration_s, width=240, height=426, with_audio=with_audio)
    return dst


def _voice(ff: FFmpeg, dst: Path, *, shot_id: int, duration_s: float) -> AudioArtifact:
    info = ff.synth_audio(dst, duration_s=duration_s)
    return AudioArtifact(
        shot_id=shot_id, path=dst, duration_s=info.duration_s, model="test/tts", text="a line"
    )


# -- C1: one audio format everywhere ----------------------------------------------------------


def test_mono_then_stereo_keeps_audio_at_the_boundary(ff: FFmpeg, tmp_path: Path) -> None:
    mono = _clip(ff, tmp_path / "mono.mp4", 5.0)  # tone 0..5 s
    stereo = _stereo_noise_clip(tmp_path / "stereo.mp4", duration_s=4.0)
    final = assemble(
        [_shot(mono, 1, 5.0), _shot(stereo, 2, 4.0)],
        _deps(ff),
        work_dir=tmp_path / "work",
        dest=tmp_path / "final.mp4",
        strategy_used=AudioStrategy.NATIVE,
    )
    info = ff.probe(final.path)
    assert (info.audio_sample_rate, info.audio_channels) == (48000, 2)
    # The bug: loudnorm reset at the mono->stereo switch and the tone vanished from 2.1 to 5.0 s.
    tail_of_first_clip = ff.audio_levels(final.path, start_s=2.6, duration_s=2.2)
    assert tail_of_first_clip.max_volume_db > AUDIBLE_DB
    assert final.duration_s == pytest.approx(9.0, abs=0.2)


# -- C3: the final lasts quality.duration_s exactly -------------------------------------------


@pytest.fixture(scope="module")
def long_clips(ff: FFmpeg, tmp_path_factory: pytest.TempPathFactory) -> list[VideoArtifact]:
    d = tmp_path_factory.mktemp("long")
    return [_shot(_clip(ff, d / f"s{i}.mp4", 15.2), i, 15.2) for i in (1, 2)]  # 30.4 s


def test_final_of_7_6_s_fails_the_duration_gate(ff: FFmpeg, tmp_path: Path) -> None:
    shots = [_shot(_clip(ff, tmp_path / f"s{i}.mp4", 3.8), i, 3.8) for i in (1, 2)]
    with pytest.raises(QualityGateError) as exc_info:
        assemble(
            shots,
            _deps(ff),
            work_dir=tmp_path / "work",
            dest=tmp_path / "final.mp4",
            strategy_used=AudioStrategy.NATIVE,
        )
    assert exc_info.value.gate == "duration"
    assert isinstance(exc_info.value.measured, float)
    assert exc_info.value.measured == pytest.approx(7.6, abs=0.1)
    assert exc_info.value.threshold == "8..30"


def test_final_of_30_4_s_is_trimmed_to_30(
    ff: FFmpeg, tmp_path: Path, long_clips: list[VideoArtifact]
) -> None:
    final = assemble(
        long_clips,
        _deps(ff),
        work_dir=tmp_path / "work",
        dest=tmp_path / "final.mp4",
        strategy_used=AudioStrategy.NATIVE,
        protect_until_s=20.0,
    )
    assert 29.9 <= final.duration_s <= 30.0
    assert ff.probe(final.path).audio_sample_rate == 48000


def test_trim_that_would_cut_the_line_raises(
    ff: FFmpeg, tmp_path: Path, long_clips: list[VideoArtifact]
) -> None:
    with pytest.raises(QualityGateError) as exc_info:
        assemble(
            long_clips,
            _deps(ff),
            work_dir=tmp_path / "work",
            dest=tmp_path / "final.mp4",
            strategy_used=AudioStrategy.NATIVE,
            protect_until_s=30.2,
        )
    assert exc_info.value.gate == "duration"
    assert exc_info.value.context["protect_until_s"] == 30.2


def test_trim_never_cuts_the_mixed_tts_voice(
    ff: FFmpeg, tmp_path: Path, long_clips: list[VideoArtifact]
) -> None:
    """The script window may end earlier than the TTS: the voice end is protected too."""
    voice = _voice(ff, tmp_path / "line.mp3", shot_id=2, duration_s=1.0)
    with pytest.raises(QualityGateError) as exc_info:
        assemble(
            long_clips,
            _deps(ff),
            work_dir=tmp_path / "work",
            dest=tmp_path / "final.mp4",
            strategy_used=AudioStrategy.TTS_OVERLAY,
            voice=VoiceOverlay(audio=voice, offset_s=14.0),  # ends at 15.2 + 15.0 = 30.2 s
            protect_until_s=29.0,
        )
    assert exc_info.value.gate == "duration"
    assert exc_info.value.context["protect_until_s"] == pytest.approx(30.2, abs=0.1)


# -- C2: the integrity gate can fail ----------------------------------------------------------


def test_silent_final_fails_the_integrity_gate(ff: FFmpeg, tmp_path: Path) -> None:
    shots = [
        _shot(_clip(ff, tmp_path / f"s{i}.mp4", 4.5, with_audio=False), i, 4.5) for i in (1, 2)
    ]
    with pytest.raises(QualityGateError) as exc_info:
        assemble(
            shots,
            _deps(ff),
            work_dir=tmp_path / "work",
            dest=tmp_path / "final.mp4",
            strategy_used=AudioStrategy.NATIVE,
        )
    assert exc_info.value.gate == "media_integrity"
    assert "silent audio" in str(exc_info.value.measured)
    assert exc_info.value.context["audio_levels"]["max_volume_db"] < SILENT_DB


# -- C6: the voice is never cut silently ------------------------------------------------------


def test_voice_longer_than_the_rest_of_the_shot_raises_voice_fit(
    ff: FFmpeg, tmp_path: Path
) -> None:
    shots = [_shot(_clip(ff, tmp_path / "s1.mp4", 8.0), 1, 8.0)]
    voice = _voice(ff, tmp_path / "line.mp3", shot_id=1, duration_s=7.5)
    with pytest.raises(QualityGateError) as exc_info:
        assemble(
            shots,
            _deps(ff),
            work_dir=tmp_path / "work",
            dest=tmp_path / "final.mp4",
            strategy_used=AudioStrategy.TTS_OVERLAY,
            voice=VoiceOverlay(audio=voice, offset_s=1.0),
        )
    assert exc_info.value.gate == "voice_fit"
    assert exc_info.value.measured == pytest.approx(8.5, abs=0.1)
    assert exc_info.value.threshold == pytest.approx(8.0, abs=0.1)


# -- C4/C5: lipsync gets the voice as mixed, is paid once, and its cost is reported once ------


@dataclass
class _RecordingLipsync:
    keep_dir: Path
    calls: int = 0
    audio_inputs: list[Path] = field(default_factory=list)

    def sync(self, request: LipsyncRequest, *, dest: Path) -> LipsyncResult:
        self.calls += 1
        kept = self.keep_dir / f"audio_input_{self.calls}.wav"
        shutil.copyfile(request.audio_path, kept)
        self.audio_inputs.append(kept)
        shutil.copyfile(request.video_path, dest)
        return LipsyncResult(
            path=dest, model=f"fake/{request.model}", cost_usd=0.4, cost_is_estimate=False
        )


def test_lipsync_gets_the_positioned_voice_and_is_paid_and_reported_once(
    ff: FFmpeg, tmp_path: Path
) -> None:
    lipsync = _RecordingLipsync(keep_dir=tmp_path)
    deps = _deps(ff, lipsync=lipsync, lipsync_config=LipsyncConfig(enabled=True))
    dialogue = _shot(_clip(ff, tmp_path / "s2.mp4", 4.0, with_audio=False), 2, 4.0)
    voice = VoiceOverlay(
        audio=_voice(ff, tmp_path / "line.mp3", shot_id=2, duration_s=1.5), offset_s=1.0
    )
    work_dir = tmp_path / "work"

    def run(first_shot_s: float) -> FinalVideo:
        first = _shot(_clip(ff, tmp_path / f"s1_{first_shot_s}.mp4", first_shot_s), 1, first_shot_s)
        return assemble(
            [first, dialogue],
            deps,
            work_dir=work_dir,
            dest=tmp_path / "final.mp4",
            strategy_used=AudioStrategy.TTS_OVERLAY,
            voice=voice,
        )

    # 1) Too short: lipsync was paid, the final fails the duration gate, nothing reported yet.
    with pytest.raises(QualityGateError, match="duration"):
        run(3.0)
    assert lipsync.calls == 1
    sent = lipsync.audio_inputs[0]
    assert ff.probe(sent).duration_s == pytest.approx(4.0, abs=0.05)  # the clip length
    assert ff.audio_levels(sent, start_s=0.0, duration_s=0.9).max_volume_db < SILENT_DB
    assert ff.audio_levels(sent, start_s=1.1, duration_s=1.3).max_volume_db > AUDIBLE_DB
    assert ff.audio_levels(sent, start_s=2.6, duration_s=1.3).max_volume_db < SILENT_DB

    # 2) The flow extends the other shot: the cached lipsync is reused and reported now.
    final = run(5.0)
    assert lipsync.calls == 1
    [cost] = final.provider_costs
    assert cost.units == pytest.approx(4.0, abs=0.1)
    assert cost == ProviderCost(
        capability="lipsync",
        model="fake/fal-ai/sync-lipsync/v2",
        units=cost.units,
        cost_usd=0.4,
        is_estimate=False,
        shot_id=2,
    )
    assert final.duration_s == pytest.approx(9.0, abs=0.2)
    assert final.has_audio

    # 3) A resume or retry of a finished assemble: no new call, nothing reported twice.
    again = run(5.0)
    assert lipsync.calls == 1
    assert again.provider_costs == []
