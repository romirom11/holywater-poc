from pathlib import Path

import pytest

from holywater.media.ffmpeg import (
    SILENCE_FLOOR_DB,
    _parse_probe,
    _parse_rate,
    _parse_volumedetect,
)
from holywater.observability.errors import MediaError


def test_parse_rate_variants() -> None:
    assert _parse_rate("24/1") == 24.0
    assert _parse_rate("30000/1001") > 29.9
    assert _parse_rate("0/0") == 0.0
    assert _parse_rate(None) == 0.0
    assert _parse_rate("garbage") == 0.0
    assert _parse_rate("25") == 25.0


def test_parse_probe_video_only(tmp_path: Path) -> None:
    f = tmp_path / "x.mp4"
    f.write_bytes(b"123")
    data = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 720,
                "height": 1280,
                "avg_frame_rate": "24/1",
            }
        ],
        "format": {"duration": "8.04"},
    }
    info = _parse_probe(data, f)
    assert info.has_video and not info.has_audio
    assert info.size_bytes == 3
    assert info.duration_s == 8.04
    assert info.is_vertical


def test_parse_probe_audio_format_and_rate_fallback(tmp_path: Path) -> None:
    f = tmp_path / "x.mp4"
    f.write_bytes(b"123")
    data = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 480,
                "height": 854,
                "avg_frame_rate": "0/0",
                "r_frame_rate": "24/1",
            },
            {"codec_type": "audio", "codec_name": "aac", "sample_rate": "32000", "channels": 2},
        ],
        "format": {"duration": "6.08", "size": "1000"},
    }
    info = _parse_probe(data, f)
    assert info.fps == 24.0  # avg_frame_rate "0/0" falls back to r_frame_rate
    assert (info.audio_sample_rate, info.audio_channels) == (32000, 2)
    assert info.aspect_ratio == 480 / 854


def test_parse_probe_audio_without_format_fields(tmp_path: Path) -> None:
    f = tmp_path / "x.wav"
    f.write_bytes(b"1")
    info = _parse_probe({"streams": [{"codec_type": "audio", "sample_rate": "N/A"}]}, f)
    assert info.has_audio and info.audio_sample_rate is None and info.audio_channels is None


_VOLUMEDETECT_STDERR = """\
[Parsed_volumedetect_0 @ 0x1] n_samples: 0
[Parsed_volumedetect_0 @ 0x2] n_samples: 48000
[Parsed_volumedetect_0 @ 0x2] mean_volume: -21.1 dB
[Parsed_volumedetect_0 @ 0x2] max_volume: -17.7 dB
[Parsed_volumedetect_0 @ 0x2] histogram_17db: 1180
"""


def test_parse_volumedetect_levels() -> None:
    levels = _parse_volumedetect(_VOLUMEDETECT_STDERR, Path("x.mp4"))
    assert (levels.mean_volume_db, levels.max_volume_db) == (-21.1, -17.7)


def test_parse_volumedetect_uses_the_last_report_and_floors_minus_inf() -> None:
    stderr = (
        "mean_volume: -5.0 dB\nmax_volume: -1.0 dB\nmean_volume: -inf dB\nmax_volume: -inf dB\n"
    )
    levels = _parse_volumedetect(stderr, Path("x.mp4"))
    assert levels.max_volume_db == SILENCE_FLOOR_DB == levels.mean_volume_db


def test_parse_volumedetect_window_without_samples_is_silence() -> None:
    levels = _parse_volumedetect("[Parsed_volumedetect_0 @ 0x1] n_samples: 0\n", Path("x.mp4"))
    assert levels.max_volume_db == SILENCE_FLOOR_DB


def test_parse_volumedetect_garbage_raises_media_error() -> None:
    with pytest.raises(MediaError) as exc_info:
        _parse_volumedetect("Stream map '0:a' matches no streams.", Path("x.mp4"))
    assert exc_info.value.context["op"] == "audio_levels"
    assert "matches no streams" in exc_info.value.context["stderr"]
