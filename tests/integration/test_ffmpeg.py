from pathlib import Path

import pytest

from holywater.media import FFmpeg
from holywater.observability.errors import MediaError

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


def test_concat_two_normalized_clips(ff: FFmpeg, tmp_path: Path) -> None:
    a = ff.synth_clip(tmp_path / "a.mp4", duration_s=2, width=480, height=854)
    ff.synth_clip(tmp_path / "b_raw.mp4", duration_s=3, width=720, height=1280, with_audio=False)
    b = ff.normalize_clip(tmp_path / "b_raw.mp4", tmp_path / "b.mp4", width=480, height=854)
    info = ff.concat([tmp_path / "a.mp4", tmp_path / "b.mp4"], tmp_path / "out.mp4")
    assert info.duration_s == pytest.approx(a.duration_s + b.duration_s, abs=0.3)
    assert info.has_audio and info.is_vertical


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


def test_loudnorm_and_trim(ff: FFmpeg, tmp_path: Path) -> None:
    ff.synth_clip(tmp_path / "v.mp4", duration_s=5, width=480, height=854)
    norm = ff.loudnorm(tmp_path / "v.mp4", tmp_path / "ln.mp4")
    assert norm.has_audio
    trimmed = ff.trim(tmp_path / "ln.mp4", tmp_path / "t.mp4", max_s=2.0)
    assert trimmed.duration_s == pytest.approx(2.0, abs=0.15)
