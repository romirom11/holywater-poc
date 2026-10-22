from pathlib import Path

import pytest

from holywater.media import FFmpeg
from holywater.observability.errors import MediaError, QualityGateError

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not FFmpeg.is_available(), reason="ffmpeg/ffprobe not on PATH"),
]


@pytest.fixture
def ff() -> FFmpeg:
    return FFmpeg(timeout_s=120)


def test_synth_and_probe(ff: FFmpeg, tmp_path: Path) -> None:
    info = ff.synth_clip(tmp_path / "a.mp4", duration_s=2, width=480, height=854)
    assert info.has_video and info.has_audio
    assert info.width == 480 and info.height == 854 and info.is_vertical
    assert info.duration_s == pytest.approx(2.0, abs=0.15)
    assert info.fps == pytest.approx(24.0)
    assert info.video_codec == "h264" and info.audio_codec == "aac"


def test_probe_missing_file_raises(ff: FFmpeg, tmp_path: Path) -> None:
    with pytest.raises(MediaError, match="not found"):
        ff.probe(tmp_path / "missing.mp4")


def test_broken_file_raises_media_error_with_stderr(ff: FFmpeg, tmp_path: Path) -> None:
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    with pytest.raises(MediaError) as exc_info:
        ff.probe(broken)
    assert "stderr" in exc_info.value.context


def test_normalize_adds_silent_audio_and_letterboxes(ff: FFmpeg, tmp_path: Path) -> None:
    src = tmp_path / "landscape.mp4"
    ff.synth_clip(src, duration_s=1.5, width=640, height=360, with_audio=False)
    info = ff.normalize_clip(src, tmp_path / "norm.mp4", width=480, height=854)
    assert info.has_audio
    assert (info.width, info.height) == (480, 854)
    assert info.duration_s == pytest.approx(1.5, abs=0.15)
    assert (info.audio_sample_rate, info.audio_channels) == (48000, 2)
    assert ff.audio_levels(tmp_path / "norm.mp4").max_volume_db < -80


def test_normalize_forces_48k_stereo_audio(ff: FFmpeg, tmp_path: Path) -> None:
    src = ff.synth_clip(tmp_path / "mono.mp4", duration_s=1.5, width=480, height=854)
    assert src.audio_channels == 1  # synth_clip's tone is mono
    info = ff.normalize_clip(tmp_path / "mono.mp4", tmp_path / "norm.mp4", width=480, height=854)
    assert (info.audio_sample_rate, info.audio_channels) == (48000, 2)
    assert ff.audio_levels(tmp_path / "norm.mp4").max_volume_db > -30


def test_normalize_can_take_audio_from_another_file(ff: FFmpeg, tmp_path: Path) -> None:
    ff.synth_clip(tmp_path / "picture.mp4", duration_s=2, width=480, height=854, with_audio=False)
    ff.synth_audio(tmp_path / "sound.mp3", duration_s=3)
    info = ff.normalize_clip(
        tmp_path / "picture.mp4",
        tmp_path / "out.mp4",
        width=480,
        height=854,
        audio_src=tmp_path / "sound.mp3",
    )
    assert info.duration_s == pytest.approx(2.0, abs=0.15)  # the shorter input
    assert (info.audio_sample_rate, info.audio_channels) == (48000, 2)
    assert ff.audio_levels(tmp_path / "out.mp4").max_volume_db > -30


def test_concat_two_normalized_clips(ff: FFmpeg, tmp_path: Path) -> None:
    ff.synth_clip(tmp_path / "a_raw.mp4", duration_s=2, width=480, height=854)
    a = ff.normalize_clip(tmp_path / "a_raw.mp4", tmp_path / "a.mp4", width=480, height=854)
    ff.synth_clip(tmp_path / "b_raw.mp4", duration_s=3, width=720, height=1280, with_audio=False)
    b = ff.normalize_clip(tmp_path / "b_raw.mp4", tmp_path / "b.mp4", width=480, height=854)
    info = ff.concat([tmp_path / "a.mp4", tmp_path / "b.mp4"], tmp_path / "out.mp4")
    assert info.duration_s == pytest.approx(a.duration_s + b.duration_s, abs=0.3)
    assert info.has_audio and info.is_vertical
    assert (info.audio_sample_rate, info.audio_channels) == (48000, 2)


def test_concat_refuses_clips_with_different_audio_layouts(ff: FFmpeg, tmp_path: Path) -> None:
    ff.synth_clip(tmp_path / "mono.mp4", duration_s=1, width=480, height=854)
    ff.synth_clip(tmp_path / "b_raw.mp4", duration_s=1, width=480, height=854, with_audio=False)
    ff.normalize_clip(tmp_path / "b_raw.mp4", tmp_path / "stereo.mp4", width=480, height=854)
    with pytest.raises(MediaError, match="differ") as exc_info:
        ff.concat([tmp_path / "mono.mp4", tmp_path / "stereo.mp4"], tmp_path / "out.mp4")
    assert len(exc_info.value.context["formats"]) == 2


def test_concat_requires_clips(ff: FFmpeg, tmp_path: Path) -> None:
    with pytest.raises(MediaError, match="at least one"):
        ff.concat([], tmp_path / "out.mp4")


def test_overlay_voice_keeps_duration(ff: FFmpeg, tmp_path: Path) -> None:
    ff.synth_clip(tmp_path / "v.mp4", duration_s=4, width=480, height=854)
    ff.synth_audio(tmp_path / "line.mp3", duration_s=1.5)
    info = ff.overlay_voice(
        tmp_path / "v.mp4", tmp_path / "line.mp3", tmp_path / "mixed.mp4", offset_s=1.0
    )
    assert info.has_audio
    assert info.duration_s == pytest.approx(4.0, abs=0.2)
    assert (info.audio_sample_rate, info.audio_channels) == (48000, 2)


def test_overlay_voice_refuses_to_cut_the_voice(ff: FFmpeg, tmp_path: Path) -> None:
    ff.synth_clip(tmp_path / "v.mp4", duration_s=4, width=480, height=854)
    ff.synth_audio(tmp_path / "line.mp3", duration_s=3.5)
    with pytest.raises(QualityGateError) as exc_info:
        ff.overlay_voice(
            tmp_path / "v.mp4", tmp_path / "line.mp3", tmp_path / "mixed.mp4", offset_s=1.0
        )
    err = exc_info.value
    assert err.gate == "voice_fit"
    assert err.measured == pytest.approx(4.5, abs=0.1)
    assert err.threshold == pytest.approx(4.0, abs=0.1)
    assert not (tmp_path / "mixed.mp4").exists()


def test_place_voice_matches_the_mix(ff: FFmpeg, tmp_path: Path) -> None:
    ff.synth_audio(tmp_path / "line.mp3", duration_s=1.5)
    placed = tmp_path / "placed.wav"
    info = ff.place_voice(tmp_path / "line.mp3", placed, offset_s=1.0, duration_s=4.0)
    assert info.duration_s == pytest.approx(4.0, abs=0.02)
    assert (info.audio_sample_rate, info.audio_channels) == (48000, 2)
    assert ff.audio_levels(placed, start_s=0.0, duration_s=0.9).max_volume_db < -60
    assert ff.audio_levels(placed, start_s=1.1, duration_s=1.3).max_volume_db > -30
    assert ff.audio_levels(placed, start_s=2.6, duration_s=1.3).max_volume_db < -60


def test_place_voice_refuses_to_cut_the_voice(ff: FFmpeg, tmp_path: Path) -> None:
    ff.synth_audio(tmp_path / "line.mp3", duration_s=3.5)
    with pytest.raises(QualityGateError, match="would be cut"):
        ff.place_voice(tmp_path / "line.mp3", tmp_path / "p.wav", offset_s=1.0, duration_s=4.0)


def test_loudnorm_and_trim(ff: FFmpeg, tmp_path: Path) -> None:
    ff.synth_clip(tmp_path / "v.mp4", duration_s=5, width=480, height=854)
    norm = ff.loudnorm(tmp_path / "v.mp4", tmp_path / "ln.mp4")
    assert norm.has_audio
    assert norm.audio_sample_rate == 48000  # loudnorm upsamples to 192 kHz internally
    trimmed = ff.trim(tmp_path / "ln.mp4", tmp_path / "t.mp4", max_s=2.0)
    assert trimmed.duration_s == pytest.approx(2.0, abs=0.15)
    assert trimmed.audio_sample_rate == 48000


def test_audio_levels_tone_silence_and_a_window_without_samples(ff: FFmpeg, tmp_path: Path) -> None:
    ff.synth_clip(tmp_path / "tone.mp4", duration_s=2, width=480, height=854)
    loud = ff.audio_levels(tmp_path / "tone.mp4")
    assert -30 < loud.max_volume_db <= 0 and loud.mean_volume_db <= loud.max_volume_db
    ff.synth_audio(tmp_path / "short.mp3", duration_s=1.0)
    past_the_end = ff.audio_levels(tmp_path / "short.mp3", start_s=5.0, duration_s=1.0)
    assert past_the_end.max_volume_db < -100


def test_audio_levels_without_audio_raises(ff: FFmpeg, tmp_path: Path) -> None:
    ff.synth_clip(tmp_path / "mute.mp4", duration_s=1, width=480, height=854, with_audio=False)
    with pytest.raises(MediaError):
        ff.audio_levels(tmp_path / "mute.mp4")
