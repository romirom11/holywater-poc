"""Error mapping and payload handling of the fal vendor, with the SDK mocked out and downloads
served by `httpx.MockTransport`."""

from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import fal_client
import httpx
import pytest
from fal_client.client import FalClientHTTPError, FalClientTimeoutError, SyncClient
from structlog.testing import capture_logs

from holywater.domain.enums import Resolution
from holywater.observability.errors import (
    BilledJobError,
    FatalProviderError,
    PromptRejectedError,
    RetryableProviderError,
)
from holywater.providers import fal as fal_module
from holywater.providers.base import (
    LipsyncProvider,
    LipsyncRequest,
    LLMProvider,
    STTProvider,
    STTRequest,
    TTSProvider,
    TTSRequest,
    VideoProvider,
    VideoRequest,
)
from holywater.providers.fal import FalProvider, _file_url

VIDEO_URL = "https://v3.fal.media/files/x/v.mp4"


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fal_module.time, "sleep", lambda _s: None)


def _http_error(
    status: int, message: str = "", error_type: str | None = None
) -> FalClientHTTPError:
    return FalClientHTTPError(
        message or f"status {status}",
        status_code=status,
        response_headers={},
        response=MagicMock(),
        error_type=error_type,
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


def _with_downloads(fal: FalProvider, *responses: httpx.Response) -> list[httpx.Request]:
    """Serve result downloads from a queue (the last response repeats); return the requests."""
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    fal._http = httpx.Client(transport=httpx.MockTransport(handler))
    return seen


def _mp4(content: bytes = b"mp4") -> httpx.Response:
    return httpx.Response(200, content=content, headers={"content-type": "video/mp4"})


def _subscribe_returning(
    result: dict[str, Any], captured: dict[str, Any] | None = None
) -> Callable[..., dict[str, Any]]:
    def fake_subscribe(model: str, arguments: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        if captured is not None:
            captured.update(model=model, arguments=arguments, **kwargs)
        kwargs["on_enqueue"]("req-1")
        return result

    return fake_subscribe


def test_fal_implements_media_capabilities_but_not_llm() -> None:
    fal = FalProvider(api_key="k")
    assert isinstance(fal, VideoProvider)
    assert isinstance(fal, TTSProvider)
    assert isinstance(fal, STTProvider)
    assert isinstance(fal, LipsyncProvider)
    assert not isinstance(fal, LLMProvider)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (429, RetryableProviderError),
        (503, RetryableProviderError),
        (400, PromptRejectedError),  # 400 on a video submit: the prompt is refused
        (401, FatalProviderError),
        (422, FatalProviderError),
    ],
)
def test_fal_http_errors_are_classified(
    status: int, expected: type[Exception], tmp_path: Path
) -> None:
    fal = FalProvider(api_key="k")
    with (
        patch.object(SyncClient, "subscribe", side_effect=_http_error(status)),
        pytest.raises(expected) as exc_info,
    ):
        fal.generate(_video_request(), dest=tmp_path / "v.mp4")
    assert type(exc_info.value) is expected
    assert exc_info.value.status_code == status
    assert exc_info.value.provider == "fal"


def test_fal_content_policy_wording_is_a_prompt_rejection(tmp_path: Path) -> None:
    fal = FalProvider(api_key="k")
    error = _http_error(422, "Prompt was flagged", error_type="content_policy_violation")
    with (
        patch.object(SyncClient, "subscribe", side_effect=error),
        pytest.raises(PromptRejectedError),
    ):
        fal.synthesize(
            TTSRequest(model="fal-ai/minimax/speech-02-hd", text="x", language="en"),
            dest=tmp_path / "a.mp3",
        )
    # A plain 400 outside a video submit stays fatal.
    with (
        patch.object(SyncClient, "subscribe", side_effect=_http_error(400)),
        pytest.raises(FatalProviderError) as exc_info,
    ):
        fal.synthesize(
            TTSRequest(model="fal-ai/minimax/speech-02-hd", text="x", language="en"),
            dest=tmp_path / "a.mp3",
        )
    assert not isinstance(exc_info.value, PromptRejectedError)


def test_fal_timeout_and_transport_are_retryable(tmp_path: Path) -> None:
    fal = FalProvider(api_key="k", timeout_s=5)
    with (
        patch.object(SyncClient, "subscribe", side_effect=FalClientTimeoutError(5)),
        pytest.raises(RetryableProviderError, match="timed out after 5s"),
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


def test_finished_video_without_url_is_fatal_and_carries_its_cost(tmp_path: Path) -> None:
    fal = FalProvider(api_key="k")
    with (
        patch.object(SyncClient, "subscribe", side_effect=_subscribe_returning({"seed": 1})),
        pytest.raises(FatalProviderError, match=r"video\.url") as exc_info,
    ):
        fal.generate(_video_request(), dest=tmp_path / "v.mp4")
    assert exc_info.value.request_id == "req-1"
    assert exc_info.value.spent_usd == pytest.approx(0.2205 * 8, abs=1e-3)


def test_fal_video_builds_arguments_downloads_and_uses_video_timeout(tmp_path: Path) -> None:
    fal = FalProvider(api_key="k", timeout_s=321)
    captured: dict[str, Any] = {}
    downloads = _with_downloads(fal, _mp4())
    dest = tmp_path / "shot.mp4"
    with patch.object(
        SyncClient,
        "subscribe",
        side_effect=_subscribe_returning({"video": {"url": VIDEO_URL}, "seed": 77}, captured),
    ):
        result = fal.generate(_video_request(seed=5), dest=dest)
    assert captured["model"] == "bytedance/seedance-2.5/text-to-video"
    assert captured["arguments"] == {
        "prompt": "p",
        "duration": "8",
        "resolution": "480p",
        "aspect_ratio": "9:16",
        "generate_audio": True,
    }  # no `seed`: it is output-only in the fal Seedance schema
    assert captured["client_timeout"] == 321
    assert [str(r.url) for r in downloads] == [VIDEO_URL]
    assert downloads[0].extensions["timeout"]["read"] == 321
    assert "authorization" not in downloads[0].headers  # public CDN link, no key leaves us
    assert dest.read_bytes() == b"mp4"
    assert result.seed == 77 and result.request_id == "req-1" and result.source_url == VIDEO_URL
    assert result.cost_usd == pytest.approx(0.2205 * 8, abs=1e-3)


def test_fal_download_is_retried_then_billed(tmp_path: Path) -> None:
    fal = FalProvider(api_key="k", download_attempts=3)
    downloads = _with_downloads(fal, httpx.Response(502), httpx.Response(200), _mp4(b"ok"))
    with patch.object(
        SyncClient, "subscribe", side_effect=_subscribe_returning({"video": {"url": VIDEO_URL}})
    ) as subscribe:
        fal.generate(_video_request(), dest=tmp_path / "v.mp4")
    assert len(downloads) == 3 and subscribe.call_count == 1  # 502, empty, ok
    assert (tmp_path / "v.mp4").read_bytes() == b"ok"

    fal = FalProvider(api_key="k", download_attempts=2)
    downloads = _with_downloads(fal, httpx.Response(503))
    with (
        patch.object(
            SyncClient,
            "subscribe",
            side_effect=_subscribe_returning({"video": {"url": VIDEO_URL}}),
        ) as subscribe,
        capture_logs() as logs,
        pytest.raises(BilledJobError, match="could not be downloaded") as exc_info,
    ):
        fal.generate(_video_request(), dest=tmp_path / "w.mp4")
    assert len(downloads) == 2 and subscribe.call_count == 1
    assert exc_info.value.job_id == "req-1" and exc_info.value.status_code == 503
    assert exc_info.value.spent_usd == pytest.approx(0.2205 * 8, abs=1e-3)
    assert [e["event"] for e in logs].count("fal.download_error") == 2
    assert not (tmp_path / "w.mp4").exists() and not (tmp_path / "w.mp4.part").exists()


def test_fal_download_not_found_is_billed_without_retrying(tmp_path: Path) -> None:
    fal = FalProvider(api_key="k")
    downloads = _with_downloads(fal, httpx.Response(404))
    with (
        patch.object(
            SyncClient,
            "subscribe",
            side_effect=_subscribe_returning({"video": {"url": VIDEO_URL}}),
        ),
        pytest.raises(BilledJobError),
    ):
        fal.generate(_video_request(), dest=tmp_path / "v.mp4")
    assert len(downloads) == 1


def test_fal_download_size_cap_is_enforced_while_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fal_module, "_MAX_DOWNLOAD_BYTES", 4)
    fal = FalProvider(api_key="k")
    downloads = _with_downloads(fal, _mp4(b"0123456789"))
    with (
        patch.object(
            SyncClient,
            "subscribe",
            side_effect=_subscribe_returning({"video": {"url": VIDEO_URL}}),
        ),
        pytest.raises(BilledJobError, match="size limit"),
    ):
        fal.generate(_video_request(), dest=tmp_path / "v.mp4")
    assert len(downloads) == 1
    assert not (tmp_path / "v.mp4").exists() and not (tmp_path / "v.mp4.part").exists()


def test_fal_unexpected_content_type_only_warns(tmp_path: Path) -> None:
    fal = FalProvider(api_key="k")
    _with_downloads(
        fal, httpx.Response(200, content=b"a", headers={"content-type": "application/octet-stream"})
    )
    with (
        patch.object(
            SyncClient,
            "subscribe",
            side_effect=_subscribe_returning({"audio": {"url": "https://x/a.mp3"}}),
        ),
        capture_logs() as logs,
    ):
        fal.synthesize(
            TTSRequest(model="fal-ai/minimax/speech-02-hd", text="x", language="en"),
            dest=tmp_path / "a.mp3",
        )
    assert (tmp_path / "a.mp3").read_bytes() == b"a"
    assert any(e["event"] == "fal.download.unexpected_type" for e in logs)


def test_fal_tts_elevenlabs_arguments_and_tts_timeout(tmp_path: Path) -> None:
    fal = FalProvider(api_key="k", tts_timeout_s=42)
    captured: dict[str, Any] = {}
    downloads = _with_downloads(
        fal, httpx.Response(200, content=b"ID3", headers={"content-type": "audio/mpeg"})
    )
    with patch.object(
        SyncClient,
        "subscribe",
        side_effect=_subscribe_returning({"audio": {"url": "https://x/a.mp3"}}, captured),
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
    assert captured["arguments"] == {
        "text": "Meet me at 7.",
        "voice": "Rachel",
        "language_code": "en",
        "apply_text_normalization": "off",
    }
    assert captured["client_timeout"] == 42
    assert downloads[0].extensions["timeout"]["read"] == 42
    assert result.cost_usd == pytest.approx(0.10 * len("Meet me at 7.") / 1000)


def test_fal_tts_unknown_model_price_warns(tmp_path: Path) -> None:
    fal = FalProvider(api_key="k")
    _with_downloads(fal, httpx.Response(200, content=b"a", headers={"content-type": "audio/mpeg"}))
    with (
        patch.object(
            SyncClient,
            "subscribe",
            side_effect=_subscribe_returning({"audio": {"url": "https://x/a.mp3"}}),
        ),
        capture_logs() as logs,
    ):
        result = fal.synthesize(
            TTSRequest(model="fal-ai/new-tts", text="abcd", language="en"), dest=tmp_path / "a"
        )
    assert result.cost_usd == pytest.approx(0.10 * 4 / 1000)
    assert any(e["event"] == "cost.unknown_price" and e["log_level"] == "warning" for e in logs)


def test_fal_stt_arguments_timeout_and_parsing(tmp_path: Path) -> None:
    fal = FalProvider(api_key="k", stt_timeout_s=17)
    captured: dict[str, Any] = {}
    request = STTRequest(model="fal-ai/whisper", audio_path=tmp_path / "in.wav", language="auto")
    with (
        patch.object(FalProvider, "_upload", return_value="https://x/in.wav"),
        patch.object(
            SyncClient,
            "subscribe",
            side_effect=_subscribe_returning(
                {"text": "  hello there ", "inferred_languages": ["en"]}, captured
            ),
        ),
    ):
        result = fal.transcribe(request)
        assert captured["arguments"] == {
            "audio_url": "https://x/in.wav",
            "task": "transcribe",
            "chunk_level": "none",
        }  # "auto" is never sent
        assert captured["client_timeout"] == 17
        fal.transcribe(
            STTRequest(model="fal-ai/whisper", audio_path=tmp_path / "in.wav", language="uk")
        )
        assert captured["arguments"]["language"] == "uk"
    assert result.text == "hello there" and result.language == "en"
    with (
        patch.object(FalProvider, "_upload", return_value="https://x/in.wav"),
        patch.object(SyncClient, "subscribe", return_value={"chunks": []}),
        pytest.raises(FatalProviderError, match="no text"),
    ):
        fal.transcribe(request)


def test_upload_goes_through_the_keyed_client(tmp_path: Path) -> None:
    """FAL_KEY lives in .env, not os.environ: the module-level fal client would not see it."""
    fal = FalProvider(api_key="key-from-dotenv")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    with (
        patch.object(fal_client, "upload_file", side_effect=AssertionError("unkeyed client")),
        patch.object(
            SyncClient, "upload_file", autospec=True, return_value="https://x/a.wav"
        ) as up,
    ):
        assert fal._upload(audio, model="fal-ai/whisper") == "https://x/a.wav"
    client, path = up.call_args.args
    assert client is fal._client and client.key == "key-from-dotenv" and path == audio
    assert fal._client.default_timeout == fal.stt_timeout_s


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_http_error(503), RetryableProviderError),
        (_http_error(401), FatalProviderError),
        (httpx.ConnectError("down"), RetryableProviderError),
    ],
)
def test_upload_errors_are_typed(tmp_path: Path, error: Exception, expected: type) -> None:
    fal = FalProvider(api_key="k")
    with (
        patch.object(SyncClient, "upload_file", side_effect=error),
        pytest.raises(expected) as exc_info,
    ):
        fal._upload(tmp_path / "a.wav", model="fal-ai/whisper")
    assert type(exc_info.value) is expected


def test_fal_lipsync_uploads_inputs_and_pins_arguments(tmp_path: Path) -> None:
    fal = FalProvider(api_key="k", lipsync_sync_mode="bounce", timeout_s=99)
    captured: dict[str, Any] = {}
    downloads = _with_downloads(fal, _mp4(b"synced"))
    uploads = iter(["https://x/video.mp4", "https://x/voice.wav"])
    with (
        patch.object(SyncClient, "upload_file", side_effect=lambda _p: next(uploads)),
        patch.object(
            SyncClient,
            "subscribe",
            side_effect=_subscribe_returning({"video": {"url": VIDEO_URL}}, captured),
        ),
        capture_logs() as logs,
    ):
        result = fal.sync(
            LipsyncRequest(
                model="fal-ai/sync-lipsync/v2",
                video_path=tmp_path / "v.mp4",
                audio_path=tmp_path / "a.wav",
            ),
            dest=tmp_path / "out.mp4",
        )
    assert captured["arguments"] == {
        "video_url": "https://x/video.mp4",
        "audio_url": "https://x/voice.wav",
        "sync_mode": "bounce",
    }
    assert captured["client_timeout"] == 99 and len(downloads) == 1
    assert (tmp_path / "out.mp4").read_bytes() == b"synced"
    assert result.cost_is_estimate is True
    assert any(e["event"] == "cost.unknown_price" for e in logs)
