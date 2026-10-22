"""OpenRouter vendor: LLM (chat, structured output), video (`/videos` async jobs),
TTS (`/audio/speech`) and STT (`/audio/transcriptions`) behind one API key.

Chat goes through the OpenAI SDK (schema-validated structured outputs, ADR-004); the media
endpoints are OpenRouter-specific and use one shared `httpx.Client`. Every HTTP failure is
classified by one rule (`_classify`): `RETRYABLE_HTTP_STATUS` -> `RetryableProviderError`,
moderation / content-policy wording or a 400 on a video submit -> `PromptRejectedError`,
anything else -> `FatalProviderError`.

Idempotent reads (video job polls, the download of a finished job) are retried here against the
same job id. Anything that would start a new paid job is left to the Prefect task / flow: a job
that was accepted but cannot be collected raises `BilledJobError` with what it costs.
"""

from __future__ import annotations

import base64
import json
import re
import time
import wave
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import openai
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ValidationError

from holywater.observability.errors import (
    BilledJobError,
    FatalProviderError,
    MediaError,
    PipelineError,
    PromptRejectedError,
    ProviderError,
    RetryableProviderError,
    StructuredOutputError,
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
    looks_like_prompt_rejection,
    table_price,
)

log = get_logger(__name__)

NAME = "openrouter"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_HEADERS = {"HTTP-Referer": "https://github.com/holywater-poc", "X-Title": "holywater-poc"}
_MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024
_ERROR_DETAIL_CHARS = 1000
# Polls of a just-accepted job that may answer 404 before the job is visible (~30 s at 10 s).
_POLL_NOT_FOUND_GRACE = 3
# Content types that are never media: an error page or JSON sent with a 200 on a media download.
_NOT_MEDIA_TYPES = ("application/json", "text/")

# Wording that marks a vendor-side (infrastructure) cause of a failed video job.
_INFRA_RE = re.compile(
    r"time[d ]?\s?out|internal|server error|unavailable|overload|capacity|rate[ _-]?limit"
    r"|try again|gateway|network|connection|upstream",
    re.IGNORECASE,
)

# USD per character (OpenRouter lists TTS price per input "token" = character). Estimate only.
TTS_PRICE_PER_CHAR_USD: dict[str, float] = {
    "elevenlabs/eleven-v3": 0.00008,
    "elevenlabs/eleven-turbo-v2.5": 0.00004,
    "elevenlabs/eleven-flash-v2.5": 0.00004,
    "minimax/speech-2.8-hd": 0.0001,
    "minimax/speech-2.8-turbo": 0.00006,
}
# USD per second of input audio, only used when the response carries no `usage.cost`. Observed:
# $0.000152 for a 6.08 s clip; turbo scaled by the ratio of the listed `pricing.prompt` values.
STT_PRICE_PER_SECOND_USD: dict[str, float] = {
    "openai/whisper-large-v3": 0.000025,
    "openai/whisper-large-v3-turbo": 0.0000111,
}
# USD per output second of Seedance 2.5 (config: providers.openrouter.video_usd_per_s).
DEFAULT_VIDEO_USD_PER_S: dict[str, float] = {"480p": 0.1035, "720p": 0.233}


@dataclass
class OpenRouterProvider:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout_s: int = 600  # one video job: submit, polling deadline, each download
    llm_timeout_s: int = 60
    tts_timeout_s: int = 60
    stt_timeout_s: int = 120
    llm_max_tokens: int = 4000
    video_poll_interval_s: float = 10.0
    download_attempts: int = 3
    video_usd_per_s: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_VIDEO_USD_PER_S)
    )
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
        # Every request passes its own timeout (video/tts/stt); this default is a backstop.
        self._http = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}", **_HEADERS},
            timeout=httpx.Timeout(self.timeout_s),
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
            raise _classify(
                exc.status_code,
                f"openrouter chat: HTTP {exc.status_code}: {exc.message}",
                model=model,
                op="chat",
                detail=str(exc.body)[:_ERROR_DETAIL_CHARS],
            ) from exc
        except openai.OpenAIError as exc:
            msg = f"openrouter chat returned an unusable response: {exc}"
            raise FatalProviderError(msg, provider=NAME, model=model) from exc
        if not isinstance(completion, ChatCompletion):
            # e.g. an HTML page with status 200 from an edge proxy
            msg = f"openrouter chat returned a non-JSON body ({type(completion).__name__})"
            raise FatalProviderError(msg, provider=NAME, model=model)
        return self._parse_completion(completion, model=model, schema=schema, started=started)

    def _parse_completion[T: BaseModel](
        self, completion: ChatCompletion, *, model: str, schema: type[T], started: float
    ) -> LLMResult[T]:
        # The SDK builds the object without validation, so absent fields are read defensively.
        request_id = getattr(completion, "id", None)
        usage = getattr(completion, "usage", None)
        usage_extra: dict[str, Any] = dict(usage.model_extra or {}) if usage is not None else {}
        reported = usage_extra.get("cost") is not None
        cost = round(float(usage_extra.get("cost") or 0.0), 6)
        billed: dict[str, Any] = {
            "provider": NAME,
            "model": model,
            "request_id": request_id,
            "spent_usd": cost,
            "spent_is_estimate": not reported,
        }
        error = (completion.model_extra or {}).get("error")
        if isinstance(error, dict):
            msg = f"openrouter chat: error in a 200 body: {error.get('message') or error}"
            raise _classify(
                _as_status(error.get("code")),
                msg,
                model=model,
                op="chat",
                detail=json.dumps(error, default=str)[:_ERROR_DETAIL_CHARS],
                request_id=request_id,
                spent_usd=cost,
                spent_is_estimate=not reported,
            )
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise FatalProviderError("openrouter chat returned no choices", **billed)
        choice = choices[0]
        finish = getattr(choice, "finish_reason", None)
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        refusal = getattr(message, "refusal", None)
        log.info(
            "llm.response",
            model=getattr(completion, "model", None) or model,
            request_id=request_id,
            latency_s=round(time.monotonic() - started, 2),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            finish_reason=finish,
            cost_usd=cost,
        )
        if not reported:
            log.warning("cost.unknown_price", provider=NAME, model=model, reason="no usage.cost")
        if refusal or finish == "content_filter":
            msg = f"openrouter model refused the request (finish_reason={finish})"
            if refusal:
                msg = f"{msg}: {refusal}"
            raise PromptRejectedError(msg, context={"refusal": refusal}, **billed)
        if finish == "error":
            msg = "openrouter upstream failed mid-generation (finish_reason=error)"
            raise RetryableProviderError(msg, **billed)
        if finish == "length":
            msg = f"output truncated at max_tokens={self.llm_max_tokens} (finish_reason=length)"
            raise StructuredOutputError(msg, raw_text=content or "", finish_reason=finish, **billed)
        if not content:
            msg = f"openrouter returned no content (finish_reason={finish})"
            raise StructuredOutputError(msg, finish_reason=finish, **billed)
        try:
            data = schema.model_validate_json(content)
        except ValidationError as first:
            # Some providers wrap the object; try one level of unwrapping before giving up.
            try:
                data = schema.model_validate(_unwrap(json.loads(content)))
            except ValueError:  # JSONDecodeError and pydantic's ValidationError
                msg = f"output does not match {schema.__name__}: {str(first)[:1500]}"
                raise StructuredOutputError(
                    msg, raw_text=content, finish_reason=finish, **billed
                ) from first
        return LLMResult(
            data=data,
            model=getattr(completion, "model", None) or model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cost_usd=cost,
            cost_is_estimate=not reported,
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
        job = self._json(
            "POST",
            "/videos",
            json=body,
            op="video.submit",
            model=request.model,
            timeout=self.timeout_s,
        )
        job_id = str(job.get("id") or "")
        if not job_id:
            msg = f"openrouter video job has no id (keys: {sorted(job)})"
            raise FatalProviderError(msg, provider=NAME, model=request.model)
        poll_url = self._polling_url(job, job_id)
        log.info(
            "openrouter.video.submitted",
            job_id=job_id,
            model=request.model,
            status=job.get("status"),
            polling_url=poll_url,
        )

        final = self._poll_video(job_id, poll_url, request=request)
        reported = _has_usage_cost(final)
        cost = _usage_cost(final) if reported else self._estimate_video_usd(request)
        if not reported:
            log.warning(
                "cost.unknown_price",
                provider=NAME,
                model=request.model,
                reason="no usage.cost on the completed job",
                estimate_usd=cost,
            )
        self._download_job(job_id, dest, request=request, spent_usd=cost, reported=reported)
        return VideoResult(
            path=dest,
            model=request.model,
            source_url=f"{self.base_url}/videos/{job_id}/content",
            seed=request.seed,
            request_id=str(final.get("generation_id") or job_id),
            cost_usd=round(cost, 4),
            cost_is_estimate=not reported,
        )

    def _default_polling_url(self, job_id: str) -> str:
        return f"{self.base_url.rstrip('/')}/videos/{job_id}"

    def _polling_url(self, job: dict[str, Any], job_id: str) -> str:
        """`polling_url` from the 202, resolved against the origin (the spec's examples are
        origin-relative, `/api/v1/videos/<id>`); `{base_url}/videos/{id}` when absent."""
        default = self._default_polling_url(job_id)
        raw = job.get("polling_url")
        if not isinstance(raw, str) or not raw:
            return default
        base = httpx.URL(self.base_url)
        url = base.join(raw)
        if (url.scheme, url.host, url.port) != (base.scheme, base.host, base.port):
            # Never send the bearer key to another origin (another host, or plain http).
            log.warning("openrouter.video.foreign_polling_url", job_id=job_id, polling_url=raw)
            return default
        return str(url)

    def _poll_video(self, job_id: str, url: str, *, request: VideoRequest) -> dict[str, Any]:
        """Poll until the job finishes. A 4xx on a `polling_url` other than the default switches
        to `{base_url}/videos/{id}` once; a 404 is tolerated for the first `_POLL_NOT_FOUND_GRACE`
        polls (a job that was just accepted may not be visible yet). Any other non-retryable error
        abandons the job (`BilledJobError`): it was accepted and is billed."""
        model = request.model
        default = self._default_polling_url(job_id)
        deadline = time.monotonic() + self.timeout_s
        status = "unknown"
        not_found = 0
        while True:
            try:
                job = self._json(
                    "GET",
                    url,
                    op="video.poll",
                    model=model,
                    timeout=max(deadline - time.monotonic(), 1.0),
                )
            except RetryableProviderError as exc:
                # A poll is an idempotent read: a blip must not cost a new paid job.
                log.warning(
                    "openrouter.video.poll_error",
                    job_id=job_id,
                    error=exc.message,
                    status_code=exc.status_code,
                )
            except PromptRejectedError:
                raise
            except ProviderError as exc:
                code = exc.status_code
                if url != default and code is not None and _is_client_error(code):
                    log.warning(
                        "openrouter.video.polling_url_failed",
                        job_id=job_id,
                        polling_url=url,
                        status_code=code,
                        fallback=default,
                    )
                    url = default
                    continue
                if code == httpx.codes.NOT_FOUND and not_found < _POLL_NOT_FOUND_GRACE:
                    not_found += 1
                    log.warning(
                        "openrouter.video.poll_not_found",
                        job_id=job_id,
                        polls=not_found,
                        grace=_POLL_NOT_FOUND_GRACE,
                    )
                else:
                    msg = f"openrouter video job {job_id}: polling failed: {exc.message}"
                    raise self._billed(job_id, request, msg, status_code=code) from exc
            else:
                status = str(job.get("status") or "")
                if status == "completed":
                    return job
                if status == "failed":
                    raise _failed_job_error(job_id, job, model=model)
                if status in ("cancelled", "expired"):
                    msg = f"openrouter video job {job_id} {status}: {job.get('error')}"
                    raise FatalProviderError(
                        msg,
                        provider=NAME,
                        model=model,
                        request_id=job_id,
                        context={"error": job.get("error"), "status": status},
                        **_reported_spend(job),
                    )
                log.debug("openrouter.video.poll", job_id=job_id, status=status)
            if time.monotonic() >= deadline:
                msg = f"openrouter video job {job_id} still {status!r} after {self.timeout_s}s"
                raise self._billed(job_id, request, msg)
            time.sleep(self.video_poll_interval_s)

    def _download_job(
        self, job_id: str, dest: Path, *, request: VideoRequest, spent_usd: float, reported: bool
    ) -> None:
        """Download a finished (billed) job, retrying the same job id; never resubmits."""
        last: PipelineError | None = None
        for attempt in range(1, self.download_attempts + 1):
            try:
                self._download(
                    "GET",
                    f"/videos/{job_id}/content",
                    dest,
                    op="video.download",
                    model=request.model,
                    expect_prefix="video/",
                    timeout=self.timeout_s,
                )
            except RetryableProviderError as exc:
                last = exc
                log.warning(
                    "openrouter.video.download_error",
                    job_id=job_id,
                    attempt=attempt,
                    attempts=self.download_attempts,
                    error=exc.message,
                    status_code=exc.status_code,
                )
            except (ProviderError, MediaError) as exc:
                last = exc
                break
            else:
                return
            if attempt < self.download_attempts:
                time.sleep(self.video_poll_interval_s)
        assert last is not None
        msg = f"openrouter video job {job_id} completed but could not be downloaded: {last.message}"
        raise BilledJobError(
            msg,
            provider=NAME,
            model=request.model,
            job_id=job_id,
            spent_usd=spent_usd,
            spent_is_estimate=not reported,
            status_code=getattr(last, "status_code", None),
        ) from last

    def _billed(
        self, job_id: str, request: VideoRequest, message: str, *, status_code: int | None = None
    ) -> BilledJobError:
        return BilledJobError(
            message,
            provider=NAME,
            model=request.model,
            job_id=job_id,
            spent_usd=self._estimate_video_usd(request),
            spent_is_estimate=True,
            status_code=status_code,
        )

    def _estimate_video_usd(self, request: VideoRequest) -> float:
        per_s = table_price(
            self.video_usd_per_s,
            request.resolution.value,
            provider=NAME,
            model=request.model,
            unit="second",
        )
        return round(per_s * request.duration_s, 4)

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
            "POST",
            "/audio/speech",
            dest,
            op="tts",
            model=request.model,
            expect_prefix="audio/",
            timeout=self.tts_timeout_s,
            json=body,
        )
        price = table_price(
            TTS_PRICE_PER_CHAR_USD, request.model, provider=NAME, model=request.model, unit="char"
        )
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
            "POST",
            "/audio/transcriptions",
            json=body,
            op="stt",
            model=request.model,
            timeout=self.stt_timeout_s,
        )
        text = result.get("text")
        if not isinstance(text, str):
            msg = f"openrouter STT result has no text (keys: {sorted(result)})"
            raise FatalProviderError(msg, provider=NAME, model=request.model)
        reported = _has_usage_cost(result)
        if reported:
            cost = _usage_cost(result)
        else:
            per_s = table_price(
                STT_PRICE_PER_SECOND_USD,
                request.model,
                provider=NAME,
                model=request.model,
                unit="second",
            )
            cost = round(per_s * _wav_seconds(audio), 6)
            log.warning(
                "cost.unknown_price",
                provider=NAME,
                model=request.model,
                reason="no usage.cost in the STT response",
                estimate_usd=cost,
            )
        return STTResult(
            text=text.strip(),
            model=request.model,
            language=result.get("language") or request.language,
            cost_usd=cost,
            cost_is_estimate=not reported,
        )

    # -- shared internals --------------------------------------------------------------------

    def _json(
        self,
        method: str,
        path: str,
        *,
        op: str,
        model: str,
        timeout: float,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resp = self._request(method, path, op=op, model=model, json=json, timeout=timeout)
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
        method: str,
        path: str,
        dest: Path,
        *,
        op: str,
        model: str,
        expect_prefix: str,
        timeout: float,
        json: dict[str, Any] | None = None,
    ) -> Path:
        """Stream a media body to `dest` with a running size cap. A JSON or text body is a
        `FatalProviderError` (an error sent with a 200 is never media); any other content type
        than `expect_prefix` is only a WARNING (vendors sometimes send application/octet-stream),
        and ffprobe downstream is the real check. Nothing is left at `dest` on failure."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        size = 0
        try:
            with self._http.stream(method, path, json=json, timeout=timeout) as resp:
                if resp.status_code >= httpx.codes.BAD_REQUEST:
                    resp.read()
                    raise _classify(
                        resp.status_code,
                        f"openrouter {op}: HTTP {resp.status_code}: {_error_message(resp)}",
                        model=model,
                        op=op,
                        detail=resp.text[:_ERROR_DETAIL_CHARS],
                    )
                ctype = resp.headers.get("content-type", "")
                if ctype.lower().startswith(_NOT_MEDIA_TYPES):
                    resp.read()
                    msg = f"openrouter {op}: expected {expect_prefix}* but got {ctype!r}"
                    raise FatalProviderError(
                        msg,
                        provider=NAME,
                        model=model,
                        status_code=resp.status_code,
                        context={"detail": resp.text[:_ERROR_DETAIL_CHARS]},
                    )
                if not ctype.startswith(expect_prefix):
                    log.warning("openrouter.download.unexpected_type", op=op, content_type=ctype)
                with part.open("wb") as fh:
                    for chunk in resp.iter_bytes():
                        size += len(chunk)
                        if size > _MAX_DOWNLOAD_BYTES:
                            msg = "downloaded file exceeds size limit"
                            raise MediaError(
                                msg, context={"path": path, "limit": _MAX_DOWNLOAD_BYTES}
                            )
                        fh.write(chunk)
        except httpx.TimeoutException as exc:
            part.unlink(missing_ok=True)
            msg = f"openrouter {op} timed out"
            raise RetryableProviderError(msg, provider=NAME, model=model) from exc
        except httpx.TransportError as exc:
            part.unlink(missing_ok=True)
            msg = f"openrouter {op} transport error: {exc}"
            raise RetryableProviderError(msg, provider=NAME, model=model) from exc
        except PipelineError:
            part.unlink(missing_ok=True)
            raise
        if size == 0:
            part.unlink(missing_ok=True)
            msg = f"openrouter {op}: downloaded file is empty"
            raise RetryableProviderError(msg, provider=NAME, model=model)
        part.replace(dest)
        log.debug("openrouter.downloaded", op=op, path=str(dest), bytes=size)
        return dest

    def _request(
        self,
        method: str,
        path: str,
        *,
        op: str,
        model: str,
        json: dict[str, Any] | None,
        timeout: float,
    ) -> httpx.Response:
        started = time.monotonic()
        try:
            resp = self._http.request(method, path, json=json, timeout=timeout)
        except httpx.TimeoutException as exc:
            msg = f"openrouter {op} timed out"
            raise RetryableProviderError(msg, provider=NAME, model=model) from exc
        except httpx.TransportError as exc:
            msg = f"openrouter {op} transport error: {exc}"
            raise RetryableProviderError(msg, provider=NAME, model=model) from exc
        if resp.status_code >= httpx.codes.BAD_REQUEST:
            raise _classify(
                resp.status_code,
                f"openrouter {op}: HTTP {resp.status_code}: {_error_message(resp)}",
                model=model,
                op=op,
                detail=resp.text[:_ERROR_DETAIL_CHARS],
            )
        log.debug(
            "openrouter.http",
            op=op,
            status=resp.status_code,
            latency_s=round(time.monotonic() - started, 2),
        )
        return resp


def _classify(
    status: int | None, message: str, *, model: str, op: str, detail: str = "", **kwargs: Any
) -> ProviderError:
    """One rule for every OpenRouter failure (see the module docstring)."""
    rejected = looks_like_prompt_rejection(f"{message} {detail}") or (
        status == httpx.codes.BAD_REQUEST and op == "video.submit"
    )
    cls: type[ProviderError]
    if status in RETRYABLE_HTTP_STATUS:
        cls = RetryableProviderError
    elif rejected:
        cls = PromptRejectedError
    else:
        cls = FatalProviderError
    return cls(message, provider=NAME, model=model, status_code=status, **kwargs)


def _failed_job_error(job_id: str, job: dict[str, Any], *, model: str) -> ProviderError:
    """A `failed` job is a prompt rejection unless its error clearly names an infrastructure
    cause; then a new submission (decided by the Prefect task) is the cure."""
    error = job.get("error")
    text = str(error.get("message") or error) if isinstance(error, dict) else str(error or "")
    msg = f"openrouter video job {job_id} failed: {text or 'no error given'}"
    kwargs: dict[str, Any] = {
        "provider": NAME,
        "model": model,
        "request_id": job_id,
        "context": {"error": error, "status": "failed"},
        **_reported_spend(job),
    }
    if _INFRA_RE.search(text) and not looks_like_prompt_rejection(text):
        return RetryableProviderError(msg, **kwargs)
    return PromptRejectedError(msg, **kwargs)


def _is_client_error(status: int) -> bool:
    return httpx.codes.BAD_REQUEST <= status < httpx.codes.INTERNAL_SERVER_ERROR


def _wav_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes() / float(wav.getframerate() or 1)
    except (wave.Error, EOFError, OSError) as exc:
        log.warning("openrouter.stt.duration_unknown", path=str(path), error=str(exc))
        return 0.0


def _as_status(code: object) -> int | None:
    if isinstance(code, int) and not isinstance(code, bool):
        return code
    if isinstance(code, str) and code.isdigit():
        return int(code)
    return None


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


def _reported_spend(payload: dict[str, Any]) -> dict[str, Any]:
    """`spent_usd` kwargs for an error when the vendor reported a cost on the job."""
    if _has_usage_cost(payload):
        return {"spent_usd": _usage_cost(payload), "spent_is_estimate": False}
    return {}


def _unwrap(payload: Any) -> Any:
    if isinstance(payload, dict) and len(payload) == 1:
        (inner,) = payload.values()
        if isinstance(inner, dict):
            return inner
    return payload
