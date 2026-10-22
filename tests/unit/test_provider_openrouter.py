"""OpenRouter vendor: chat structured output (SDK mocked) and media endpoints (httpx mocked)."""

import base64
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

from holywater.domain.enums import Resolution
from holywater.domain.script import Script
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
from holywater.providers.openrouter import OpenRouterProvider
from tests.unit.test_domain import make_script

MODEL = "anthropic/claude-sonnet-5.5"


def _provider(handler: Any | None = None) -> OpenRouterProvider:
    p = OpenRouterProvider(api_key="k", video_poll_interval_s=0.0)
    if handler is not None:
        p._http = httpx.Client(
            base_url="https://openrouter.ai/api/v1",
            transport=httpx.MockTransport(handler),
            headers={"Authorization": "Bearer k"},
        )
    return p


def test_openrouter_implements_all_but_lipsync() -> None:
    p = _provider()
    assert isinstance(p, LLMProvider)
    assert isinstance(p, VideoProvider)
    assert isinstance(p, TTSProvider)
    assert isinstance(p, STTProvider)
    assert not isinstance(p, LipsyncProvider)


# -- chat -----------------------------------------------------------------------------------------


def _completion(content: str | None, *, cost: float = 0.0123) -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens, usage.completion_tokens = 100, 50
    usage.model_extra = {"cost": cost}
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message, choice.finish_reason = message, "stop"
    completion = MagicMock()
    completion.choices, completion.usage, completion.model, completion.id = (
        [choice],
        usage,
        MODEL,
        "gen-1",
    )
    return completion


def test_structured_output_roundtrip() -> None:
    p = _provider()
    script_json = Script.model_validate(make_script()).model_dump_json()
    with patch.object(
        p._chat.chat.completions, "create", return_value=_completion(script_json)
    ) as create:
        result = p.complete_structured(model=MODEL, system="s", user="u", schema=Script)
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == MODEL
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["strict"] is True
    assert kwargs["extra_body"] == {"provider": {"require_parameters": True}}
    assert "temperature" not in kwargs
    assert result.data.dialogue.text == "Are you sure this is the address?"
    assert result.cost_usd == pytest.approx(0.0123)
    assert (result.input_tokens, result.output_tokens) == (100, 50)


def test_temperature_sent_only_when_set() -> None:
    p = _provider()
    script_json = Script.model_validate(make_script()).model_dump_json()
    with patch.object(
        p._chat.chat.completions, "create", return_value=_completion(script_json)
    ) as create:
        p.complete_structured(model=MODEL, system="s", user="u", schema=Script, temperature=0.3)
    assert create.call_args.kwargs["temperature"] == 0.3


def test_unwraps_single_key_envelope() -> None:
    p = _provider()
    wrapped = '{"Script": ' + Script.model_validate(make_script()).model_dump_json() + "}"
    with patch.object(p._chat.chat.completions, "create", return_value=_completion(wrapped)):
        result = p.complete_structured(model=MODEL, system="s", user="u", schema=Script)
    assert result.data.title == "Rain"


def test_empty_content_is_fatal() -> None:
    p = _provider()
    with (
        patch.object(p._chat.chat.completions, "create", return_value=_completion(None)),
        pytest.raises(FatalProviderError, match="no content"),
    ):
        p.complete_structured(model=MODEL, system="s", user="u", schema=Script)


def _status_error(status: int) -> openai.APIStatusError:
    response = httpx.Response(status, request=httpx.Request("POST", "https://x"))
    return openai.APIStatusError("boom", response=response, body=None)


def test_chat_error_mapping() -> None:
    p = _provider()
    rate_limited = openai.RateLimitError(
        "rl", response=httpx.Response(429, request=httpx.Request("POST", "https://x")), body=None
    )
    cases = [
        (rate_limited, RetryableProviderError),
        (
            openai.APITimeoutError(request=httpx.Request("POST", "https://x")),
            RetryableProviderError,
        ),
        (_status_error(502), RetryableProviderError),
        (_status_error(404), FatalProviderError),
        (_status_error(400), FatalProviderError),
    ]
    for error, expected in cases:
        with (
            patch.object(p._chat.chat.completions, "create", side_effect=error),
            pytest.raises(expected),
        ):
            p.complete_structured(model=MODEL, system="s", user="u", schema=Script)


# -- video ----------------------------------------------------------------------------------------


def _video_request() -> VideoRequest:
    return VideoRequest(
        model="bytedance/seedance-2.5",
        prompt="p",
        duration_s=8,
        resolution=Resolution.P480,
        aspect_ratio="9:16",
        generate_audio=True,
        seed=7,
    )


def test_video_submit_poll_download(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []
    polls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path.endswith("/videos"):
            body = json.loads(request.content)
            assert body["duration"] == 8 and body["resolution"] == "480p" and body["seed"] == 7
            return httpx.Response(202, json={"id": "job1", "status": "pending"})
        if request.url.path.endswith("/videos/job1"):
            polls["n"] += 1
            status = "in_progress" if polls["n"] < 3 else "completed"
            return httpx.Response(
                200,
                json={
                    "id": "job1",
                    "status": status,
                    "generation_id": "gen-9",
                    "usage": {"cost": 1.76},
                },
            )
        if request.url.path.endswith("/videos/job1/content"):
            return httpx.Response(200, content=b"mp4bytes", headers={"content-type": "video/mp4"})
        return httpx.Response(404)

    p = _provider(handler)
    result = p.generate(_video_request(), dest=tmp_path / "v.mp4")
    assert (tmp_path / "v.mp4").read_bytes() == b"mp4bytes"
    assert result.cost_usd == pytest.approx(1.76)
    assert result.request_id == "gen-9" and result.seed == 7
    assert polls["n"] == 3
    assert calls[0] == ("POST", "/api/v1/videos")


def test_video_failed_job_is_fatal(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"id": "job2", "status": "pending"})
        return httpx.Response(
            200, json={"id": "job2", "status": "failed", "error": {"message": "nsfw"}}
        )

    with pytest.raises(FatalProviderError, match="failed"):
        _provider(handler).generate(_video_request(), dest=tmp_path / "v.mp4")


def test_video_submit_http_errors(tmp_path: Path) -> None:
    for status, expected in ((400, FatalProviderError), (503, RetryableProviderError)):
        p = _provider(
            lambda _r, s=status: httpx.Response(s, json={"error": {"message": "x", "code": s}})
        )
        with pytest.raises(expected):
            p.generate(_video_request(), dest=tmp_path / "v.mp4")


def test_video_poll_timeout_is_retryable(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"id": "job3", "status": "pending"})
        return httpx.Response(200, json={"id": "job3", "status": "in_progress"})

    p = _provider(handler)
    p.timeout_s = 0
    with pytest.raises(RetryableProviderError, match="still"):
        p.generate(_video_request(), dest=tmp_path / "v.mp4")


# -- tts / stt ------------------------------------------------------------------------------------


def test_tts_writes_mp3_and_estimates_cost(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=b"ID3mp3", headers={"content-type": "audio/mpeg"})

    p = _provider(handler)
    result = p.synthesize(
        TTSRequest(model="elevenlabs/eleven-v3", text="Hello there.", language="en", voice="sarah"),
        dest=tmp_path / "line.mp3",
    )
    assert seen == {
        "model": "elevenlabs/eleven-v3",
        "input": "Hello there.",
        "response_format": "mp3",
        "voice": "sarah",
    }
    assert (tmp_path / "line.mp3").read_bytes() == b"ID3mp3"
    assert result.cost_usd == pytest.approx(0.00008 * len("Hello there."))


def test_stt_sends_base64_wav_and_reads_cost(tmp_path: Path) -> None:
    wav = tmp_path / "shot.stt.wav"
    wav.write_bytes(b"RIFFdata")
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"text": " hello ", "usage": {"cost": 0.002}})

    p = _provider(handler)
    result = p.transcribe(
        STTRequest(model="openai/whisper-large-v3", audio_path=wav, language="en")
    )
    assert seen["model"] == "openai/whisper-large-v3" and seen["language"] == "en"
    assert seen["input_audio"] == {
        "data": base64.b64encode(b"RIFFdata").decode(),
        "format": "wav",
    }
    assert result.text == "hello" and result.cost_usd == pytest.approx(0.002)


def test_stt_missing_text_is_fatal(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    p = _provider(lambda _r: httpx.Response(200, json={"usage": {}}))
    with pytest.raises(FatalProviderError, match="no text"):
        p.transcribe(STTRequest(model="m", audio_path=wav))


def test_transport_errors_are_retryable(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    with pytest.raises(RetryableProviderError, match="transport"):
        _provider(handler).transcribe(STTRequest(model="m", audio_path=wav))
