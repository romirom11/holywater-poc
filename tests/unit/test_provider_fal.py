"""Error mapping and payload handling of the fal vendor, with the SDK mocked out."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fal_client.client import FalClientHTTPError, FalClientTimeoutError, SyncClient

from holywater.domain.enums import Resolution
from holywater.observability.errors import FatalProviderError, RetryableProviderError
from holywater.providers.base import (
    LipsyncProvider,
    LLMProvider,
    STTProvider,
    STTRequest,
    TTSProvider,
    TTSRequest,
    VideoProvider,
    VideoRequest,
)
from holywater.providers.fal import FalProvider, _file_url


def _http_error(status: int) -> FalClientHTTPError:
    return FalClientHTTPError(
        f"status {status}", status_code=status, response_headers={}, response=MagicMock()
    )


def _video_request(**overrides: Any) -> VideoRequest:
    base: dict[str, Any] = {
        "model": "bytedance/seedance-2.5/text-to-video",
        "prompt": "p",
        "duration_s": 8,
        "resolution": Resolution.P480,
        "aspect_ratio": "9:16",
        "generate_audio": True,
    }
    base.update(overrides)
    return VideoRequest(**base)


def test_fal_implements_media_capabilities_but_not_llm() -> None:
    fal = FalProvider(api_key="k")
    assert isinstance(fal, VideoProvider)
    assert isinstance(fal, TTSProvider)
    assert isinstance(fal, STTProvider)
    assert isinstance(fal, LipsyncProvider)
    assert not isinstance(fal, LLMProvider)


@pytest.mark.parametrize(
    ("status", "retryable"), [(429, True), (503, True), (400, False), (401, False)]
)
def test_fal_http_errors_are_classified(status: int, retryable: bool, tmp_path: Path) -> None:
    fal = FalProvider(api_key="k")
    expected = RetryableProviderError if retryable else FatalProviderError
    with (
        patch.object(SyncClient, "subscribe", side_effect=_http_error(status)),
        pytest.raises(expected) as exc_info,
    ):
        fal.generate(_video_request(), dest=tmp_path / "v.mp4")
    assert exc_info.value.status_code == status
    assert exc_info.value.provider == "fal"


def test_fal_timeout_and_transport_are_retryable(tmp_path: Path) -> None:
    fal = FalProvider(api_key="k", timeout_s=5)
    with (
        patch.object(SyncClient, "subscribe", side_effect=FalClientTimeoutError("slow")),
        pytest.raises(RetryableProviderError, match="timed out"),
    ):
        fal.generate(_video_request(), dest=tmp_path / "v.mp4")
    with (
        patch.object(SyncClient, "subscribe", side_effect=httpx.ConnectError("down")),
        pytest.raises(RetryableProviderError, match="transport"),
    ):
        fal.generate(_video_request(), dest=tmp_path / "v.mp4")


def test_fal_non_object_result_is_fatal(tmp_path: Path) -> None:
    fal = FalProvider(api_key="k")
    with (
        patch.object(SyncClient, "subscribe", return_value=["nope"]),
        pytest.raises(FatalProviderError, match="non-object"),
    ):
        fal.generate(_video_request(), dest=tmp_path / "v.mp4")


def test_file_url_extraction() -> None:
    assert _file_url({"video": {"url": "https://x/v.mp4"}}, "video") == "https://x/v.mp4"
    with pytest.raises(FatalProviderError, match=r"video\.url"):
        _file_url({"other": 1}, "video")


def test_fal_video_builds_arguments_and_downloads(tmp_path: Path) -> None:
    fal = FalProvider(api_key="k")
    captured: dict[str, Any] = {}

    def fake_subscribe(model: str, arguments: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        captured["model"] = model
        captured.update(arguments)
        kwargs["on_enqueue"]("req-1")
        return {"video": {"url": "https://x/v.mp4"}, "seed": 77}

    dest = tmp_path / "shot.mp4"
    with (
        patch.object(SyncClient, "subscribe", side_effect=fake_subscribe),
        patch.object(FalProvider, "_download", return_value=dest) as dl,
    ):
        result = fal.generate(_video_request(seed=5), dest=dest)
    assert captured == {
        "model": "bytedance/seedance-2.5/text-to-video",
        "prompt": "p",
        "duration": "8",
        "resolution": "480p",
        "aspect_ratio": "9:16",
        "generate_audio": True,
    }  # no `seed`: it is output-only in the fal Seedance schema
    dl.assert_called_once()
    assert result.seed == 77 and result.request_id == "req-1"
    assert result.cost_usd == pytest.approx(0.2205 * 8, abs=1e-3)


def test_fal_tts_elevenlabs_disables_normalization(tmp_path: Path) -> None:
    fal = FalProvider(api_key="k")
    captured: dict[str, Any] = {}

    def fake_subscribe(model: str, arguments: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        captured.update(arguments)
        return {"audio": {"url": "https://x/a.mp3"}}

    with (
        patch.object(SyncClient, "subscribe", side_effect=fake_subscribe),
        patch.object(FalProvider, "_download"),
    ):
        result = fal.synthesize(
            TTSRequest(
                model="fal-ai/elevenlabs/tts/eleven-v3",
                text="Meet me at 7.",
                language="en",
                voice="Rachel",
            ),
            dest=tmp_path / "a.mp3",
        )
    assert captured["apply_text_normalization"] == "off"
    assert captured["language_code"] == "en" and captured["voice"] == "Rachel"
    assert result.cost_usd > 0


def test_fal_stt_parses_text_and_language(tmp_path: Path) -> None:
    fal = FalProvider(api_key="k")
    request = STTRequest(model="fal-ai/whisper", audio_path=tmp_path / "in.wav", language="auto")
    with (
        patch.object(FalProvider, "_upload", return_value="https://x/in.wav"),
        patch.object(
            SyncClient,
            "subscribe",
            return_value={"text": "  hello there ", "inferred_languages": ["en"]},
        ),
    ):
        result = fal.transcribe(request)
    assert result.text == "hello there" and result.language == "en"
    with (
        patch.object(FalProvider, "_upload", return_value="https://x/in.wav"),
        patch.object(SyncClient, "subscribe", return_value={"chunks": []}),
        pytest.raises(FatalProviderError, match="no text"),
    ):
        fal.transcribe(request)
