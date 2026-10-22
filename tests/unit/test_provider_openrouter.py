"""OpenRouter vendor over `httpx.MockTransport`: chat through the real OpenAI SDK, media endpoints
through the shared httpx client. Request bodies are pinned as in docs/PROVIDERS_API.md."""

import base64
import json
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest
from pydantic import BaseModel, ConfigDict
from structlog.testing import capture_logs

from holywater.domain.enums import Resolution
from holywater.observability.errors import (
    BilledJobError,
    FatalProviderError,
    PromptRejectedError,
    RetryableProviderError,
    StructuredOutputError,
)
from holywater.providers import openrouter as openrouter_module
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

MODEL = "anthropic/claude-sonnet-5.5"
BASE = "https://openrouter.ai/api/v1"
Handler = Callable[[httpx.Request], httpx.Response]


def _provider(handler: Handler | None = None, **kwargs: Any) -> OpenRouterProvider:
    p = OpenRouterProvider(api_key="k", video_poll_interval_s=0.0, **kwargs)
    if handler is not None:
        p._http = httpx.Client(
            base_url=BASE,
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


class Pick(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    count: int


GOOD = '{"title": "Rain", "count": 2}'


def _chat_body(
    content: str | None,
    *,
    finish: str | None = "stop",
    cost: float | None = 0.0123,
    refusal: str | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if refusal is not None:
        message["refusal"] = refusal
    usage: dict[str, Any] = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    if cost is not None:
        usage["cost"] = cost
    return {
        "id": "gen-1",
        "object": "chat.completion",
        "created": 1,
        "model": MODEL,
        "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        "usage": usage,
    }


def _chat_provider(
    *responses: httpx.Response | Exception, **kwargs: Any
) -> tuple[OpenRouterProvider, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []
    queue = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        item = next(queue)
        if isinstance(item, Exception):
            raise item
        return item

    p = OpenRouterProvider(api_key="k", **kwargs)
    p._chat = openai.OpenAI(
        api_key="k",
        base_url=BASE,
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return p, seen


def _ask(p: OpenRouterProvider, **kwargs: Any) -> Any:
    return p.complete_structured(model=MODEL, system="s", user="u", schema=Pick, **kwargs)


def test_chat_request_body_is_pinned_and_cost_is_actual() -> None:
    p, seen = _chat_provider(httpx.Response(200, json=_chat_body(GOOD)))
    result = _ask(p)
    assert seen == [
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
            ],
            "max_tokens": 4000,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "Pick", "strict": True, "schema": Pick.model_json_schema()},
            },
            "provider": {"require_parameters": True},
        }
    ]
    assert result.data == Pick(title="Rain", count=2)
    assert result.cost_usd == pytest.approx(0.0123) and result.cost_is_estimate is False
    assert (result.input_tokens, result.output_tokens) == (100, 50)
    assert result.raw_text == GOOD


def test_temperature_sent_only_when_set() -> None:
    p, seen = _chat_provider(httpx.Response(200, json=_chat_body(GOOD)))
    _ask(p, temperature=0.3)
    assert seen[0]["temperature"] == 0.3


def test_chat_uses_llm_timeout() -> None:
    assert OpenRouterProvider(api_key="k", llm_timeout_s=33)._chat.timeout == 33


def test_unwraps_single_key_envelope() -> None:
    p, _ = _chat_provider(httpx.Response(200, json=_chat_body('{"Pick": ' + GOOD + "}")))
    assert _ask(p).data.title == "Rain"


def test_missing_usage_cost_is_an_estimate_with_warning() -> None:
    p, _ = _chat_provider(httpx.Response(200, json=_chat_body(GOOD, cost=None)))
    with capture_logs() as logs:
        result = _ask(p)
    assert result.cost_is_estimate is True
    assert any(e["event"] == "cost.unknown_price" and e["log_level"] == "warning" for e in logs)


@pytest.mark.parametrize(
    ("body", "finish", "match"),
    [
        (GOOD[:12], "length", "truncated at max_tokens=4000"),
        (None, "stop", "no content"),
        ("not json", "stop", "does not match Pick"),
        ('{"title": "Rain"}', "stop", "does not match Pick"),
    ],
)
def test_unusable_output_is_billed_structured_output_error(
    body: str | None, finish: str, match: str
) -> None:
    p, _ = _chat_provider(httpx.Response(200, json=_chat_body(body, finish=finish)))
    with pytest.raises(StructuredOutputError, match=match) as exc_info:
        _ask(p)
    err = exc_info.value
    assert isinstance(err, ValueError)  # keeps the LLMProvider contract for write_script
    assert err.finish_reason == finish
    assert err.raw_text == (body or "")
    assert (err.spent_usd, err.spent_is_estimate) == (pytest.approx(0.0123), False)
    assert err.request_id == "gen-1" and not err.retryable


def test_refusal_and_content_filter_are_prompt_rejections() -> None:
    p, _ = _chat_provider(
        httpx.Response(200, json=_chat_body(None, refusal="I can't help with that.")),
        httpx.Response(200, json=_chat_body('{"title": "x"', finish="content_filter")),
    )
    with pytest.raises(PromptRejectedError, match="I can't help with that") as exc_info:
        _ask(p)
    assert exc_info.value.context["refusal"] == "I can't help with that."
    assert exc_info.value.spent_usd == pytest.approx(0.0123)
    with pytest.raises(PromptRejectedError, match="content_filter"):
        _ask(p)


def test_finish_reason_error_is_retryable_and_billed() -> None:
    p, _ = _chat_provider(httpx.Response(200, json=_chat_body(None, finish="error")))
    with pytest.raises(RetryableProviderError, match="finish_reason=error") as exc_info:
        _ask(p)
    assert exc_info.value.spent_usd == pytest.approx(0.0123)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ({"code": 502, "message": "upstream overloaded"}, RetryableProviderError),
        ({"code": "429", "message": "rate limited"}, RetryableProviderError),
        ({"code": 400, "message": "bad request"}, FatalProviderError),
        ({"code": 403, "message": "input was flagged by moderation"}, PromptRejectedError),
    ],
)
def test_error_object_in_a_200_body(error: dict[str, Any], expected: type[Exception]) -> None:
    p, _ = _chat_provider(httpx.Response(200, json={"error": error}))
    with pytest.raises(expected, match="error in a 200 body") as exc_info:
        _ask(p)
    assert type(exc_info.value) is expected


def test_empty_choices_is_fatal() -> None:
    body = _chat_body(GOOD)
    body["choices"] = []
    p, _ = _chat_provider(httpx.Response(200, json=body))
    with pytest.raises(FatalProviderError, match="no choices") as exc_info:
        _ask(p)
    assert not isinstance(exc_info.value, StructuredOutputError | PromptRejectedError)


def test_non_json_200_is_fatal() -> None:
    p, _ = _chat_provider(
        httpx.Response(200, text="<html>Bad gateway</html>", headers={"content-type": "text/html"})
    )
    with pytest.raises(FatalProviderError, match="non-JSON"):
        _ask(p)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            httpx.Response(429, json={"error": {"code": 429, "message": "rl"}}),
            RetryableProviderError,
        ),
        (
            httpx.Response(502, json={"error": {"code": 502, "message": "x"}}),
            RetryableProviderError,
        ),
        (httpx.Response(404, json={"error": {"code": 404, "message": "x"}}), FatalProviderError),
        (httpx.Response(400, json={"error": {"code": 400, "message": "x"}}), FatalProviderError),
        (
            httpx.Response(
                403,
                json={
                    "error": {
                        "code": 403,
                        "message": "Your input was flagged",
                        "metadata": {"reasons": ["violence"], "flagged_input": "..."},
                    }
                },
            ),
            PromptRejectedError,
        ),
        (httpx.ConnectError("down"), RetryableProviderError),
    ],
)
def test_chat_error_mapping(
    response: httpx.Response | Exception, expected: type[Exception]
) -> None:
    p, _ = _chat_provider(response)
    with pytest.raises(expected) as exc_info:
        _ask(p)
    assert type(exc_info.value) is expected


# -- video ----------------------------------------------------------------------------------------


def _video_request(**overrides: Any) -> VideoRequest:
    base: dict[str, Any] = {
        "model": "bytedance/seedance-2.5",
        "prompt": "p",
        "duration_s": 8,
        "resolution": Resolution.P480,
        "aspect_ratio": "9:16",
        "generate_audio": True,
        "seed": 7,
    }
    base.update(overrides)
    return VideoRequest(**base)


class VideoApi:
    """A scripted OpenRouter /videos API that records every request."""

    def __init__(
        self,
        *,
        job_id: str = "job-abc123",
        polling_url: str | None = "/api/v1/videos/job-abc123",
        statuses: tuple[str, ...] = ("in_progress", "in_progress", "completed"),
        final: dict[str, Any] | None = None,
        poll_faults: dict[int, httpx.Response | Exception] | None = None,
        content: list[httpx.Response] | None = None,
    ) -> None:
        self.job_id, self.polling_url, self.statuses = job_id, polling_url, statuses
        self.final = (
            final if final is not None else {"generation_id": "gen-9", "usage": {"cost": 1.76}}
        )
        self.poll_faults = poll_faults or {}
        self.content = content or [
            httpx.Response(200, content=b"mp4bytes", headers={"content-type": "video/mp4"})
        ]
        self.calls: list[tuple[str, str]] = []
        self.bodies: list[dict[str, Any]] = []
        self.polls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.calls.append((request.method, url))
        if request.method == "POST":
            self.bodies.append(json.loads(request.content))
            job: dict[str, Any] = {"id": self.job_id, "status": "pending"}
            if self.polling_url is not None:
                job["polling_url"] = self.polling_url
            return httpx.Response(202, json=job)
        if url.endswith("/content"):
            return self.content.pop(0) if len(self.content) > 1 else self.content[0]
        self.polls += 1
        fault = self.poll_faults.get(self.polls)
        if isinstance(fault, Exception):
            raise fault
        if fault is not None:
            return fault
        status = self.statuses[min(self.polls, len(self.statuses)) - 1]
        body: dict[str, Any] = {"id": self.job_id, "status": status}
        if status == "completed":
            body.update(self.final)
        return httpx.Response(200, json=body)

    def count(self, method: str, suffix: str = "") -> int:
        return sum(1 for m, u in self.calls if m == method and u.endswith(suffix))


def test_video_body_is_pinned_and_relative_polling_url_is_resolved(tmp_path: Path) -> None:
    api = VideoApi()  # the spec's own 202 example: "polling_url": "/api/v1/videos/job-abc123"
    result = _provider(api).generate(_video_request(), dest=tmp_path / "v.mp4")
    assert api.bodies == [
        {
            "model": "bytedance/seedance-2.5",
            "prompt": "p",
            "duration": 8,
            "resolution": "480p",
            "aspect_ratio": "9:16",
            "generate_audio": True,
            "seed": 7,
        }
    ]
    assert api.calls == [
        ("POST", f"{BASE}/videos"),
        ("GET", f"{BASE}/videos/job-abc123"),
        ("GET", f"{BASE}/videos/job-abc123"),
        ("GET", f"{BASE}/videos/job-abc123"),
        ("GET", f"{BASE}/videos/job-abc123/content"),
    ]
    assert (tmp_path / "v.mp4").read_bytes() == b"mp4bytes"
    assert not (tmp_path / "v.mp4.part").exists()
    assert result.cost_usd == pytest.approx(1.76) and result.cost_is_estimate is False
    assert result.request_id == "gen-9" and result.seed == 7


def test_video_body_without_seed(tmp_path: Path) -> None:
    api = VideoApi()
    _provider(api).generate(_video_request(seed=None), dest=tmp_path / "v.mp4")
    assert "seed" not in api.bodies[0]


@pytest.mark.parametrize(
    ("polling_url", "expected"),
    [
        ("https://openrouter.ai/api/v1/videos/job1", f"{BASE}/videos/job1"),
        (None, f"{BASE}/videos/job1"),
        ("", f"{BASE}/videos/job1"),
        ("https://evil.example/videos/job1", f"{BASE}/videos/job1"),  # never leak the key
        ("http://openrouter.ai/api/v1/videos/job1", f"{BASE}/videos/job1"),  # nor over http
        ("https://openrouter.ai:8443/api/v1/videos/job1", f"{BASE}/videos/job1"),  # other port
    ],
)
def test_polling_url_variants(tmp_path: Path, polling_url: str | None, expected: str) -> None:
    api = VideoApi(job_id="job1", polling_url=polling_url, statuses=("completed",))
    _provider(api).generate(_video_request(), dest=tmp_path / "v.mp4")
    assert api.calls[1] == ("GET", expected)
    assert all("evil" not in url for _, url in api.calls)


def test_transient_poll_errors_keep_polling_the_same_job(tmp_path: Path) -> None:
    api = VideoApi(
        poll_faults={
            1: httpx.Response(502, json={"error": {"message": "bad gateway"}}),
            2: httpx.ConnectError("blip"),
        },
        statuses=("in_progress", "in_progress", "in_progress", "completed"),
    )
    with capture_logs() as logs:
        result = _provider(api).generate(_video_request(), dest=tmp_path / "v.mp4")
    assert api.count("POST") == 1
    assert result.cost_usd == pytest.approx(1.76)
    warnings = [e for e in logs if e["event"] == "openrouter.video.poll_error"]
    assert len(warnings) == 2 and all(e["log_level"] == "warning" for e in warnings)


def test_poll_deadline_raises_billed_job_error(tmp_path: Path) -> None:
    api = VideoApi(statuses=("in_progress",))
    p = _provider(api, timeout_s=0)
    with pytest.raises(BilledJobError, match="still 'in_progress'") as exc_info:
        p.generate(_video_request(), dest=tmp_path / "v.mp4")
    err = exc_info.value
    assert api.count("POST") == 1
    assert err.job_id == "job-abc123" and err.request_id == "job-abc123"
    assert err.spent_usd == pytest.approx(0.1035 * 8) and err.spent_is_estimate is True
    assert not err.retryable and not isinstance(err, FatalProviderError)


def test_non_retryable_poll_failure_is_billed(tmp_path: Path) -> None:
    api = VideoApi(poll_faults={1: httpx.Response(401, json={"error": {"message": "no auth"}})})
    with pytest.raises(BilledJobError, match="polling failed") as exc_info:
        _provider(api).generate(_video_request(resolution=Resolution.P720), dest=tmp_path / "v")
    assert exc_info.value.status_code == 401
    assert exc_info.value.spent_usd == pytest.approx(0.233 * 8)
    assert api.count("POST") == 1 and api.polls == 1


def test_poll_not_found_is_tolerated_for_a_grace_window(tmp_path: Path) -> None:
    gone = httpx.Response(404, json={"error": {"message": "Resource not found"}})
    api = VideoApi(poll_faults={1: gone, 2: gone}, statuses=("completed",))
    with capture_logs() as logs:
        result = _provider(api).generate(_video_request(), dest=tmp_path / "v.mp4")
    assert result.cost_usd == pytest.approx(1.76)
    assert api.count("POST") == 1 and api.polls == 3
    assert [e["event"] for e in logs].count("openrouter.video.poll_not_found") == 2


def test_poll_not_found_past_the_grace_window_is_billed(tmp_path: Path) -> None:
    grace = openrouter_module._POLL_NOT_FOUND_GRACE
    gone = httpx.Response(404, json={"error": {"message": "Resource not found"}})
    api = VideoApi(poll_faults=dict.fromkeys(range(1, grace + 2), gone))
    with pytest.raises(BilledJobError, match="polling failed") as exc_info:
        _provider(api).generate(_video_request(), dest=tmp_path / "v.mp4")
    assert exc_info.value.status_code == 404
    assert api.count("POST") == 1 and api.polls == grace + 1


def test_failing_polling_url_falls_back_to_the_default_once(tmp_path: Path) -> None:
    """A `polling_url` that resolves to another path (here a doubled /api) and answers 4xx is
    replaced by `{base_url}/videos/{id}`, the URL the first real run polled."""
    calls: list[str] = []
    api = VideoApi(job_id="job1", polling_url="api/v1/videos/job1", statuses=("completed",))

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.method == "GET" and "/api/api/" in str(request.url):
            return httpx.Response(404, json={"error": {"message": "Not Found"}})
        return api(request)

    with capture_logs() as logs:
        _provider(handler).generate(_video_request(), dest=tmp_path / "v.mp4")
    assert calls[1] == "https://openrouter.ai/api/api/v1/videos/job1"
    assert calls[2] == f"{BASE}/videos/job1"
    assert sum(1 for u in calls if u.endswith("/videos")) == 1  # one paid submit
    assert any(e["event"] == "openrouter.video.polling_url_failed" for e in logs)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ({"message": "nsfw"}, PromptRejectedError),
        ("Output was flagged by the safety system", PromptRejectedError),
        (None, PromptRejectedError),  # no infrastructure cause given
        ("Generation failed", PromptRejectedError),
        ("Internal server error, please try again", RetryableProviderError),
        ("upstream timed out", RetryableProviderError),
    ],
)
def test_failed_job_classification(tmp_path: Path, error: Any, expected: type[Exception]) -> None:
    api = VideoApi(statuses=("failed",), final={})

    def handler(request: httpx.Request) -> httpx.Response:
        response = api(request)
        if request.method == "GET":
            return httpx.Response(
                200, json={"id": "job-abc123", "status": "failed", "error": error}
            )
        return response

    with pytest.raises(expected, match="failed") as exc_info:
        _provider(handler).generate(_video_request(), dest=tmp_path / "v.mp4")
    assert type(exc_info.value) is expected
    assert exc_info.value.request_id == "job-abc123"


@pytest.mark.parametrize("status", ["cancelled", "expired"])
def test_cancelled_or_expired_job_is_fatal_not_a_rejection(tmp_path: Path, status: str) -> None:
    api = VideoApi(statuses=(status,))
    with pytest.raises(FatalProviderError, match=status) as exc_info:
        _provider(api).generate(_video_request(), dest=tmp_path / "v.mp4")
    assert not isinstance(exc_info.value, PromptRejectedError)


@pytest.mark.parametrize(
    ("status", "message", "expected"),
    [
        (400, "Invalid request parameters", PromptRejectedError),
        (503, "overloaded", RetryableProviderError),
        (401, "Missing Authentication header", FatalProviderError),
        (402, "Insufficient credits", FatalProviderError),
        (403, "Only management keys can perform this operation", FatalProviderError),
        (404, "Model not found", FatalProviderError),
        (403, "Prompt violates the content policy", PromptRejectedError),
    ],
)
def test_video_submit_http_errors(
    tmp_path: Path, status: int, message: str, expected: type[Exception]
) -> None:
    p = _provider(
        lambda _r: httpx.Response(status, json={"error": {"code": status, "message": message}})
    )
    with pytest.raises(expected) as exc_info:
        p.generate(_video_request(), dest=tmp_path / "v.mp4")
    assert type(exc_info.value) is expected
    assert exc_info.value.status_code == status


def test_malformed_submit_responses_are_fatal(tmp_path: Path) -> None:
    no_id = _provider(lambda _r: httpx.Response(202, json={"status": "pending"}))
    with pytest.raises(FatalProviderError, match="has no id"):
        no_id.generate(_video_request(), dest=tmp_path / "v.mp4")
    not_json = _provider(lambda _r: httpx.Response(202, text="<html>"))
    with pytest.raises(FatalProviderError, match="invalid JSON") as exc_info:
        not_json.generate(_video_request(), dest=tmp_path / "v.mp4")
    assert not isinstance(exc_info.value, PromptRejectedError)


def test_download_is_retried_against_the_same_job(tmp_path: Path) -> None:
    api = VideoApi(
        statuses=("completed",),
        content=[
            httpx.Response(502, json={"error": {"message": "bad gateway"}}),
            httpx.Response(200, content=b"", headers={"content-type": "video/mp4"}),
            httpx.Response(200, content=b"mp4", headers={"content-type": "video/mp4"}),
        ],
    )
    with capture_logs() as logs:
        _provider(api).generate(_video_request(), dest=tmp_path / "v.mp4")
    assert api.count("POST") == 1 and api.count("GET", "/content") == 3
    assert (tmp_path / "v.mp4").read_bytes() == b"mp4"
    assert [e["event"] for e in logs].count("openrouter.video.download_error") == 2


def test_download_that_keeps_failing_is_billed_with_the_reported_cost(tmp_path: Path) -> None:
    api = VideoApi(
        statuses=("completed",),
        content=[httpx.Response(502, json={"error": {"message": "bad gateway"}})],
    )
    with pytest.raises(BilledJobError, match="could not be downloaded") as exc_info:
        _provider(api, download_attempts=3).generate(_video_request(), dest=tmp_path / "v.mp4")
    err = exc_info.value
    assert api.count("POST") == 1 and api.count("GET", "/content") == 3
    assert (err.spent_usd, err.spent_is_estimate) == (pytest.approx(1.76), False)
    assert err.status_code == 502
    assert not (tmp_path / "v.mp4").exists() and not (tmp_path / "v.mp4.part").exists()


def test_download_not_found_is_billed_without_retrying(tmp_path: Path) -> None:
    api = VideoApi(statuses=("completed",), content=[httpx.Response(404, json={})])
    with pytest.raises(BilledJobError):
        _provider(api).generate(_video_request(), dest=tmp_path / "v.mp4")
    assert api.count("GET", "/content") == 1


def test_oversize_download_is_aborted_while_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(openrouter_module, "_MAX_DOWNLOAD_BYTES", 4)
    api = VideoApi(
        statuses=("completed",),
        content=[httpx.Response(200, content=b"0123456789", headers={"content-type": "video/mp4"})],
    )
    with pytest.raises(BilledJobError, match="size limit"):
        _provider(api).generate(_video_request(), dest=tmp_path / "v.mp4")
    assert api.count("GET", "/content") == 1
    assert not (tmp_path / "v.mp4").exists() and not (tmp_path / "v.mp4.part").exists()


@pytest.mark.parametrize("ctype", ["application/json", "text/html; charset=utf-8"])
def test_json_or_text_body_on_a_video_download_is_billed_not_saved(
    tmp_path: Path, ctype: str
) -> None:
    """An error page sent with a 200 is never a clip: the job is billed and abandoned (the flow
    decides on a new attempt) instead of being handed to ffprobe and retried as a new job."""
    api = VideoApi(
        statuses=("completed",),
        content=[
            httpx.Response(200, content=b"<html>oops</html>", headers={"content-type": ctype})
        ],
    )
    with pytest.raises(BilledJobError, match="could not be downloaded") as exc_info:
        _provider(api).generate(_video_request(), dest=tmp_path / "v.mp4")
    assert api.count("POST") == 1 and api.count("GET", "/content") == 1
    assert exc_info.value.spent_usd == pytest.approx(1.76)
    assert not (tmp_path / "v.mp4").exists() and not (tmp_path / "v.mp4.part").exists()


def test_unexpected_content_type_only_warns(tmp_path: Path) -> None:
    api = VideoApi(
        statuses=("completed",),
        content=[
            httpx.Response(
                200, content=b"mp4", headers={"content-type": "application/octet-stream"}
            )
        ],
    )
    with capture_logs() as logs:
        _provider(api).generate(_video_request(), dest=tmp_path / "v.mp4")
    assert (tmp_path / "v.mp4").read_bytes() == b"mp4"
    assert any(e["event"] == "openrouter.download.unexpected_type" for e in logs)


def test_completed_job_without_usage_cost_is_estimated(tmp_path: Path) -> None:
    api = VideoApi(statuses=("completed",), final={})
    with capture_logs() as logs:
        result = _provider(api).generate(_video_request(), dest=tmp_path / "v.mp4")
    assert result.cost_usd == pytest.approx(0.1035 * 8) and result.cost_is_estimate is True
    assert any(e["event"] == "cost.unknown_price" and e["log_level"] == "warning" for e in logs)


def test_video_requests_use_the_video_timeout(tmp_path: Path) -> None:
    timeouts: list[tuple[str, float]] = []
    api = VideoApi(statuses=("completed",))

    def handler(request: httpx.Request) -> httpx.Response:
        timeouts.append((request.method, request.extensions["timeout"]["read"]))
        return api(request)

    _provider(handler, timeout_s=123).generate(_video_request(), dest=tmp_path / "v.mp4")
    submit, poll, download = timeouts
    assert submit == ("POST", 123) and download == ("GET", 123)
    assert 1.0 <= poll[1] <= 123  # bounded by what is left of the polling deadline


# -- tts / stt ------------------------------------------------------------------------------------


def test_tts_body_timeout_and_cost(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        seen["_timeout"] = request.extensions["timeout"]["read"]
        return httpx.Response(200, content=b"ID3mp3", headers={"content-type": "audio/mpeg"})

    p = _provider(handler, tts_timeout_s=7)
    result = p.synthesize(
        TTSRequest(model="elevenlabs/eleven-v3", text="Hello there.", language="en", voice="sarah"),
        dest=tmp_path / "line.mp3",
    )
    assert seen == {
        "model": "elevenlabs/eleven-v3",
        "input": "Hello there.",
        "response_format": "mp3",
        "voice": "sarah",
        "_timeout": 7,
    }
    assert (tmp_path / "line.mp3").read_bytes() == b"ID3mp3"
    assert result.cost_usd == pytest.approx(0.00008 * len("Hello there."))


def test_tts_unknown_model_price_warns_and_over_estimates(tmp_path: Path) -> None:
    p = _provider(
        lambda _r: httpx.Response(200, content=b"x", headers={"content-type": "audio/mpeg"})
    )
    with capture_logs() as logs:
        result = p.synthesize(
            TTSRequest(model="vendor/new-voice", text="abcd", language="en"),
            dest=tmp_path / "a.mp3",
        )
    assert result.cost_usd == pytest.approx(0.0001 * 4)  # most expensive known model
    assert any(e["event"] == "cost.unknown_price" for e in logs)


def _wav(path: Path, seconds: float, rate: int = 16000) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return path


def test_stt_body_timeout_and_reported_cost(tmp_path: Path) -> None:
    wav = _wav(tmp_path / "shot.stt.wav", 0.5)
    seen: list[tuple[dict[str, Any], float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((json.loads(request.content), request.extensions["timeout"]["read"]))
        return httpx.Response(200, json={"text": " hello ", "usage": {"cost": 0.002}})

    p = _provider(handler, stt_timeout_s=9)
    result = p.transcribe(
        STTRequest(model="openai/whisper-large-v3", audio_path=wav, language="en")
    )
    p.transcribe(STTRequest(model="openai/whisper-large-v3", audio_path=wav, language="auto"))
    audio = {"data": base64.b64encode(wav.read_bytes()).decode(), "format": "wav"}
    assert seen[0] == (
        {
            "model": "openai/whisper-large-v3",
            "input_audio": audio,
            "response_format": "json",
            "language": "en",
        },
        9,
    )
    assert "language" not in seen[1][0]  # "auto" is never sent
    assert result.text == "hello"
    assert result.cost_usd == pytest.approx(0.002) and result.cost_is_estimate is False


def test_stt_without_usage_cost_is_estimated_from_audio_length(tmp_path: Path) -> None:
    wav = _wav(tmp_path / "a.wav", 2.0)
    p = _provider(lambda _r: httpx.Response(200, json={"text": "hi"}))
    with capture_logs() as logs:
        result = p.transcribe(STTRequest(model="openai/whisper-large-v3", audio_path=wav))
    assert result.cost_usd == pytest.approx(0.000025 * 2.0) and result.cost_is_estimate is True
    assert any(e["event"] == "cost.unknown_price" for e in logs)


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
    with pytest.raises(RetryableProviderError, match="transport"):
        _provider(handler).synthesize(
            TTSRequest(model="elevenlabs/eleven-v3", text="x", language="en"),
            dest=tmp_path / "a.mp3",
        )
    assert not (tmp_path / "a.mp3").exists()
