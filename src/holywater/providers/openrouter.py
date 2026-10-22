"""OpenRouter vendor: LLM (chat, structured output), video (`/videos` async jobs),
TTS (`/audio/speech`) and STT (`/audio/transcriptions`) behind one API key.

Chat goes through the OpenAI SDK (schema-validated structured outputs, ADR-004); the media
endpoints are OpenRouter-specific and use one shared `httpx.Client`. All HTTP statuses are
classified by the same rule (`RETRYABLE_HTTP_STATUS`).
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import openai
from pydantic import BaseModel, ValidationError

from holywater.observability.errors import (
    FatalProviderError,
    MediaError,
    ProviderError,
    RetryableProviderError,
)
from holywater.observability.logging import get_logger
from holywater.providers.base import (
    RETRYABLE_HTTP_STATUS,
    LLMResult,
    STTRequest,
    STTResult,
    TTSRequest,
    TTSResult,
    VideoRequest,
    VideoResult,
)

log = get_logger(__name__)

NAME = "openrouter"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_HEADERS = {"HTTP-Referer": "https://github.com/holywater-poc", "X-Title": "holywater-poc"}
_TERMINAL_FAILURE = {"failed", "cancelled", "expired"}
_MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024

# USD per character (OpenRouter lists TTS price per input "token" = character). Estimate only.
TTS_PRICE_PER_CHAR_USD: dict[str, float] = {
    "elevenlabs/eleven-v3": 0.00008,
    "elevenlabs/eleven-turbo-v2.5": 0.00004,
    "elevenlabs/eleven-flash-v2.5": 0.00004,
    "minimax/speech-2.8-hd": 0.0001,
    "minimax/speech-2.8-turbo": 0.00006,
}


@dataclass
class OpenRouterProvider:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout_s: int = 600
    llm_timeout_s: int = 60
    llm_max_tokens: int = 4000
    video_poll_interval_s: float = 10.0
    _chat: openai.OpenAI = field(init=False, repr=False)
    _http: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # SDK-level retries are disabled: the pipeline owns retry policy and logging.
        self._chat = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.llm_timeout_s,
            max_retries=0,
            default_headers=_HEADERS,
        )
        self._http = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}", **_HEADERS},
            timeout=httpx.Timeout(120.0, connect=15.0),
            follow_redirects=True,
        )

    # -- LLMProvider -------------------------------------------------------------------------

    def complete_structured[T: BaseModel](
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type[T],
        temperature: float | None = None,
    ) -> LLMResult[T]:
        # Only send parameters we really need: with `require_parameters` every extra one narrows
        # the set of eligible endpoints (Claude 5.x, for example, has no `temperature`).
        optional: dict[str, Any] = {}
        if temperature is not None:
            optional["temperature"] = temperature
        started = time.monotonic()
        log.debug("llm.request", model=model, schema=schema.__name__, chars=len(user))
        try:
            completion = self._chat.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=self.llm_max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "strict": True,
                        "schema": schema.model_json_schema(),
                    },
                },
                # Only route to endpoints that honour json_schema; never silently degrade.
                extra_body={"provider": {"require_parameters": True}},
                **optional,
            )
        except openai.RateLimitError as exc:
            raise RetryableProviderError(
                f"openrouter rate limited: {exc}",
                provider=NAME,
                model=model,
                status_code=exc.status_code,
            ) from exc
        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            raise RetryableProviderError(
                f"openrouter connection error: {exc}", provider=NAME, model=model
            ) from exc
        except openai.APIStatusError as exc:
            raise _status_error(
                exc.status_code, f"openrouter HTTP {exc.status_code}: {exc.message}", model=model
            ) from exc

        choice = completion.choices[0] if completion.choices else None
        content = choice.message.content if choice and choice.message else None
        if not content:
            finish = getattr(choice, "finish_reason", None)
            msg = f"openrouter returned no content (finish_reason={finish})"
            raise FatalProviderError(msg, provider=NAME, model=model, request_id=completion.id)
        usage = completion.usage
        extra: dict[str, Any] = (
            dict(usage.model_extra or {}) if usage is not None and usage.model_extra else {}
        )
        cost = float(extra.get("cost") or 0.0)
        log.info(
            "llm.response",
            model=completion.model or model,
            request_id=completion.id,
            latency_s=round(time.monotonic() - started, 2),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            cost_usd=cost,
        )
        try:
            data = schema.model_validate_json(content)
        except ValidationError:
            # Some providers wrap the object; try one level of unwrapping before giving up.
            data = schema.model_validate(_unwrap(json.loads(content)))
        return LLMResult(
            data=data,
            model=completion.model or model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cost_usd=cost,
            cost_is_estimate="cost" not in extra,
            raw_text=content,
        )

    # -- VideoProvider -----------------------------------------------------------------------

    def generate(self, request: VideoRequest, *, dest: Path) -> VideoResult:
        body: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt,
            "duration": request.duration_s,
            "resolution": request.resolution.value,
            "aspect_ratio": request.aspect_ratio,
            "generate_audio": request.generate_audio,
        }
        if request.seed is not None:
            body["seed"] = request.seed
        job = self._json("POST", "/videos", json=body, op="video.submit", model=request.model)
        job_id = str(job.get("id") or "")
        if not job_id:
            msg = f"openrouter video job has no id (keys: {sorted(job)})"
            raise FatalProviderError(msg, provider=NAME, model=request.model)
        log.info("openrouter.video.submitted", job_id=job_id, model=request.model)

        final = self._poll_video(job_id, model=request.model)
        cost = _usage_cost(final)
        self._download(
            f"/videos/{job_id}/content", dest, model=request.model, expect_prefix="video/"
        )
        return VideoResult(
            path=dest,
            model=request.model,
            source_url=f"{self.base_url}/videos/{job_id}/content",
            seed=request.seed,
            request_id=str(final.get("generation_id") or job_id),
            cost_usd=round(cost, 4),
            cost_is_estimate=not _has_usage_cost(final),
        )

    def _poll_video(self, job_id: str, *, model: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_s
        while True:
            job = self._json("GET", f"/videos/{job_id}", op="video.poll", model=model)
            status = str(job.get("status") or "")
            if status == "completed":
                return job
            if status in _TERMINAL_FAILURE:
                error = job.get("error")
                msg = f"openrouter video job {job_id} {status}: {error}"
                raise FatalProviderError(
                    msg, provider=NAME, model=model, request_id=job_id, context={"error": error}
                )
            if time.monotonic() >= deadline:
                msg = f"openrouter video job {job_id} still {status!r} after {self.timeout_s}s"
                raise RetryableProviderError(msg, provider=NAME, model=model, request_id=job_id)
            log.debug("openrouter.video.poll", job_id=job_id, status=status)
            time.sleep(self.video_poll_interval_s)

    # -- TTSProvider -------------------------------------------------------------------------

    def synthesize(self, request: TTSRequest, *, dest: Path) -> TTSResult:
        body: dict[str, Any] = {
            "model": request.model,
            "input": request.text,
            "response_format": "mp3",
        }
        if request.voice:
            body["voice"] = request.voice
        self._download(
            "/audio/speech", dest, model=request.model, expect_prefix="audio/", json=body
        )
        price = TTS_PRICE_PER_CHAR_USD.get(request.model, 0.0)
        return TTSResult(
            path=dest, model=request.model, cost_usd=round(price * len(request.text), 5)
        )

    # -- STTProvider -------------------------------------------------------------------------

    def transcribe(self, request: STTRequest) -> STTResult:
        audio = request.audio_path
        body: dict[str, Any] = {
            "model": request.model,
            "input_audio": {
                "data": base64.b64encode(audio.read_bytes()).decode("ascii"),
                "format": audio.suffix.lstrip(".").lower() or "wav",
            },
            "response_format": "json",
        }
        if request.language and request.language != "auto":
            body["language"] = request.language
        result = self._json(
            "POST", "/audio/transcriptions", json=body, op="stt", model=request.model
        )
        text = result.get("text")
        if not isinstance(text, str):
            msg = f"openrouter STT result has no text (keys: {sorted(result)})"
            raise FatalProviderError(msg, provider=NAME, model=request.model)
        return STTResult(
            text=text.strip(),
            model=request.model,
            language=result.get("language") or request.language,
            cost_usd=_usage_cost(result),
            cost_is_estimate=not _has_usage_cost(result),
        )

    # -- shared internals --------------------------------------------------------------------

    def _json(
        self, method: str, path: str, *, op: str, model: str, json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        resp = self._request(method, path, op=op, model=model, json=json)
        try:
            payload = resp.json()
        except ValueError as exc:
            msg = f"openrouter {op} returned invalid JSON"
            raise FatalProviderError(msg, provider=NAME, model=model) from exc
        if not isinstance(payload, dict):
            msg = f"openrouter {op} returned a non-object result"
            raise FatalProviderError(msg, provider=NAME, model=model)
        return payload

    def _download(
        self,
        path: str,
        dest: Path,
        *,
        model: str,
        expect_prefix: str,
        json: dict[str, Any] | None = None,
    ) -> Path:
        method = "POST" if json is not None else "GET"
        resp = self._request(method, path, op="download", model=model, json=json)
        ctype = resp.headers.get("content-type", "")
        if not ctype.startswith(expect_prefix):
            log.warning("openrouter.download.unexpected_type", path=path, content_type=ctype)
        data = resp.content
        if not data:
            raise MediaError("downloaded file is empty", context={"path": path})
        if len(data) > _MAX_DOWNLOAD_BYTES:
            raise MediaError("downloaded file exceeds size limit", context={"path": path})
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        log.debug("openrouter.downloaded", path=str(dest), bytes=len(data))
        return dest

    def _request(
        self, method: str, path: str, *, op: str, model: str, json: dict[str, Any] | None
    ) -> httpx.Response:
        started = time.monotonic()
        try:
            resp = self._http.request(method, path, json=json)
        except httpx.TimeoutException as exc:
            msg = f"openrouter {op} timed out"
            raise RetryableProviderError(msg, provider=NAME, model=model) from exc
        except httpx.TransportError as exc:
            msg = f"openrouter {op} transport error: {exc}"
            raise RetryableProviderError(msg, provider=NAME, model=model) from exc
        if resp.status_code >= httpx.codes.BAD_REQUEST:
            raise _status_error(
                resp.status_code,
                f"openrouter {op}: HTTP {resp.status_code}: {_error_message(resp)}",
                model=model,
            )
        log.debug(
            "openrouter.http",
            op=op,
            status=resp.status_code,
            latency_s=round(time.monotonic() - started, 2),
        )
        return resp


def _status_error(status: int, message: str, *, model: str) -> ProviderError:
    cls = RetryableProviderError if status in RETRYABLE_HTTP_STATUS else FatalProviderError
    return cls(message, provider=NAME, model=model, status_code=status)


def _error_message(resp: httpx.Response) -> str:
    try:
        payload = resp.json()
    except ValueError:
        return resp.text[:300]
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or error)[:500]
    return str(payload)[:300]


def _has_usage_cost(payload: dict[str, Any]) -> bool:
    usage = payload.get("usage")
    return isinstance(usage, dict) and usage.get("cost") is not None


def _usage_cost(payload: dict[str, Any]) -> float:
    usage = payload.get("usage")
    if isinstance(usage, dict):
        return round(float(usage.get("cost") or 0.0), 6)
    return 0.0


def _unwrap(payload: Any) -> Any:
    if isinstance(payload, dict) and len(payload) == 1:
        (inner,) = payload.values()
        if isinstance(inner, dict):
            return inner
    return payload
