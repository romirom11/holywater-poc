from pathlib import Path

from holywater.media.ffmpeg import _parse_probe, _parse_rate


def test_parse_rate_variants() -> None:
    assert _parse_rate("24/1") == 24.0
    assert _parse_rate("30000/1001") > 29.9
    assert _parse_rate("0/0") == 0.0
    assert _parse_rate(None) == 0.0
    assert _parse_rate("garbage") == 0.0


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
