"""fal.ai vendor: video (Seedance), TTS (ElevenLabs/MiniMax), STT (Whisper), lipsync.

One authenticated `SyncClient` (submit/poll and uploads), one download path, one error mapping —
shared by every capability. Model ids are fal application ids, e.g.
`bytedance/seedance-2.5/text-to-video`.

Downloading a finished result is an idempotent read and is retried here; a result that cannot be
collected raises `BilledJobError`. Known limitation: the SDK's `subscribe` owns polling, so a
transient error after the job was enqueued surfaces as `RetryableProviderError` and a Prefect
retry submits a new job (fal is not the default video vendor).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fal_client
import httpx
from fal_client.client import FalClientError, FalClientHTTPError, FalClientTimeoutError

from holywater.observability.errors import (
    BilledJobError,
    FatalProviderError,
    MediaError,
    PipelineError,
    PromptRejectedError,
    ProviderError,
    RetryableProviderError,
)
from holywater.observability.logging import get_logger
from holywater.providers.base import (
    RETRYABLE_HTTP_STATUS,
    LipsyncRequest,
    LipsyncResult,
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

NAME = "fal"
_MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024
_DOWNLOAD_BACKOFF_S = 2.0  # x attempt number, between re-downloads of the same result
_LOG_VALUE_CHARS = 200

# Estimates only; fal bills Seedance per token (ADR-003). USD per output second.
VIDEO_PRICE_PER_SECOND_USD: dict[str, dict[str, float]] = {
    "bytedance/seedance-2.5/text-to-video": {"480p": 0.2205, "720p": 0.4730},
    "bytedance/seedance-2.0/text-to-video": {"480p": 0.14, "720p": 0.3034},
}
# USD per 1k characters.
TTS_PRICE_PER_1K_CHARS_USD: dict[str, float] = {
    "fal-ai/elevenlabs/tts/eleven-v3": 0.10,
    "fal-ai/elevenlabs/tts/turbo-v2.5": 0.05,
    "fal-ai/minimax/speech-02-hd": 0.10,
}
STT_PRICE_PER_CALL_USD = 0.01


@dataclass
class FalProvider:
    api_key: str
    timeout_s: int = 600  # video and lipsync jobs (submit..result) and their downloads
    tts_timeout_s: int = 60
    stt_timeout_s: int = 120
    lipsync_sync_mode: str = "cut_off"
    download_attempts: int = 3
    _client: fal_client.SyncClient = field(init=False, repr=False)
    _http: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # `default_timeout` caps each HTTP request the SDK makes (submit, status, result, upload);
        # every job as a whole is bounded by its own `client_timeout`. Uploads have no per-call
        # bound and feed STT (and lipsync), hence the STT timeout.
        self._client = fal_client.SyncClient(
            key=self.api_key, default_timeout=float(self.stt_timeout_s)
        )
        # Result URLs are public CDN links: no auth header on downloads.
        self._http = httpx.Client(timeout=float(self.timeout_s), follow_redirects=True)

    # -- VideoProvider -----------------------------------------------------------------------

    def generate(self, request: VideoRequest, *, dest: Path) -> VideoResult:
        arguments: dict[str, Any] = {
            "prompt": request.prompt,
            "duration": str(request.duration_s),
            "resolution": request.resolution.value,
            "aspect_ratio": request.aspect_ratio,
            "generate_audio": request.generate_audio,
        }
        if request.seed is not None:
            # Not part of the Seedance input schema on fal (seed is output-only); never send it.
            log.warning("fal.seed_unsupported", model=request.model, seed=request.seed)
        result, request_id = self._call(
            request.model, arguments, op="text_to_video", timeout_s=self.timeout_s
        )
        res = request.resolution.value
        per_second = {m: p.get(res, 0.0) for m, p in VIDEO_PRICE_PER_SECOND_USD.items()}
        price = table_price(
            per_second, request.model, provider=NAME, model=request.model, unit="second"
        )
        cost = round(price * request.duration_s, 4)
        url = self._collect(
            result,
            "video",
            dest,
            expect_prefix="video/",
            model=request.model,
            request_id=request_id,
            spent_usd=cost,
            timeout_s=self.timeout_s,
        )
        return VideoResult(
            path=dest,
            model=request.model,
            source_url=url,
            seed=_as_int(result.get("seed")),
            request_id=request_id,
            cost_usd=cost,
        )

    # -- TTSProvider -------------------------------------------------------------------------

    def synthesize(self, request: TTSRequest, *, dest: Path) -> TTSResult:
        arguments: dict[str, Any] = {"text": request.text}
        if request.voice:
            arguments["voice"] = request.voice
        if "elevenlabs" in request.model:
            arguments["language_code"] = request.language
            # The line must be read as written: never let the vendor "normalize" numbers etc.
            arguments["apply_text_normalization"] = "off"
        result, request_id = self._call(
            request.model, arguments, op="tts", timeout_s=self.tts_timeout_s
        )
        price = table_price(
            TTS_PRICE_PER_1K_CHARS_USD,
            request.model,
            provider=NAME,
            model=request.model,
            unit="1k chars",
        )
        cost = round(price * len(request.text) / 1000, 5)
        self._collect(
            result,
            "audio",
            dest,
            expect_prefix="audio/",
            model=request.model,
            request_id=request_id,
            spent_usd=cost,
            timeout_s=self.tts_timeout_s,
        )
        return TTSResult(path=dest, model=request.model, cost_usd=cost)

    # -- STTProvider -------------------------------------------------------------------------

    def transcribe(self, request: STTRequest) -> STTResult:
        url = self._upload(request.audio_path, model=request.model)
        arguments: dict[str, Any] = {"audio_url": url, "task": "transcribe", "chunk_level": "none"}
        if request.language and request.language != "auto":
            arguments["language"] = request.language
        result, request_id = self._call(
            request.model, arguments, op="stt", timeout_s=self.stt_timeout_s
        )
        text = result.get("text")
        if not isinstance(text, str):
            msg = f"fal STT result has no text (keys: {sorted(result)})"
            raise FatalProviderError(
                msg,
                provider=NAME,
                model=request.model,
                request_id=request_id,
                spent_usd=STT_PRICE_PER_CALL_USD,
            )
        inferred = result.get("inferred_languages")
        detected = inferred[0] if isinstance(inferred, list) and inferred else request.language
        return STTResult(
            text=text.strip(),
            model=request.model,
            language=str(detected) if detected else None,
            cost_usd=STT_PRICE_PER_CALL_USD,
        )

    # -- LipsyncProvider ---------------------------------------------------------------------

    def sync(self, request: LipsyncRequest, *, dest: Path) -> LipsyncResult:
        arguments = {
            "video_url": self._upload(request.video_path, model=request.model),
            "audio_url": self._upload(request.audio_path, model=request.model),
            "sync_mode": self.lipsync_sync_mode,
        }
        result, request_id = self._call(
            request.model, arguments, op="lipsync", timeout_s=self.timeout_s
        )
        # fal bills lipsync per output minute; we have no duration here, so the cost is unknown.
        log.warning("cost.unknown_price", provider=NAME, model=request.model, unit="minute")
        self._collect(
            result,
            "video",
            dest,
            expect_prefix="video/",
            model=request.model,
            request_id=request_id,
            spent_usd=0.0,
            timeout_s=self.timeout_s,
        )
        return LipsyncResult(path=dest, model=request.model)

    # -- shared internals --------------------------------------------------------------------

    def _call(
        self, model: str, arguments: dict[str, Any], *, op: str, timeout_s: int
    ) -> tuple[dict[str, Any], str | None]:
        """Submit to the fal queue and wait for the result. Returns `(result, request_id)`."""
        request_id: str | None = None

        def _on_enqueue(rid: str) -> None:
            nonlocal request_id
            request_id = rid
            log.info("fal.enqueued", op=op, model=model, request_id=rid)

        started = time.monotonic()
        log.debug("fal.submit", op=op, model=model, arguments=_redact(arguments))
        result: Any
        try:
            result = self._client.subscribe(
                model, arguments, on_enqueue=_on_enqueue, client_timeout=timeout_s
            )
        except FalClientHTTPError as exc:
            raise _map_http_error(exc, model=model, op=op, request_id=request_id) from exc
        except FalClientTimeoutError as exc:
            msg = f"fal {model} timed out after {timeout_s}s"
            raise RetryableProviderError(
                msg, provider=NAME, model=model, request_id=request_id
            ) from exc
        except (httpx.TransportError, FalClientError) as exc:
            msg = f"fal {model} transport error: {exc}"
            raise RetryableProviderError(
                msg, provider=NAME, model=model, request_id=request_id
            ) from exc
        if not isinstance(result, dict):
            msg = f"fal {model} returned a non-object result"
            raise FatalProviderError(msg, provider=NAME, model=model, request_id=request_id)
        log.info(
            "fal.done",
            op=op,
            model=model,
            request_id=request_id,
            latency_s=round(time.monotonic() - started, 2),
        )
        return result, request_id

    def _upload(self, path: Path, *, model: str) -> str:
        """Upload a local file to fal storage with the keyed client; return its URL."""
        try:
            return str(self._client.upload_file(path))
        except FalClientHTTPError as exc:
            raise _map_http_error(exc, model=model, op="upload", request_id=None) from exc
        except (httpx.TransportError, FalClientError) as exc:
            msg = f"fal upload failed: {exc}"
            raise RetryableProviderError(msg, provider=NAME, model=model) from exc

    def _collect(
        self,
        result: dict[str, Any],
        key: str,
        dest: Path,
        *,
        expect_prefix: str,
        model: str,
        request_id: str | None,
        spent_usd: float,
        timeout_s: int,
    ) -> str:
        """Download `result[key].url` of a finished (billed) job, retrying the same URL."""
        url = _file_url(result, key, model=model, request_id=request_id, spent_usd=spent_usd)
        last: PipelineError | None = None
        for attempt in range(1, self.download_attempts + 1):
            try:
                self._download(
                    url, dest, expect_prefix=expect_prefix, model=model, timeout_s=timeout_s
                )
            except RetryableProviderError as exc:
                last = exc
                log.warning(
                    "fal.download_error",
                    request_id=request_id,
                    attempt=attempt,
                    attempts=self.download_attempts,
                    error=exc.message,
                    status_code=exc.status_code,
                )
            except (ProviderError, MediaError) as exc:
                last = exc
                break
            else:
                return url
            if attempt < self.download_attempts:
                time.sleep(_DOWNLOAD_BACKOFF_S * attempt)
        assert last is not None
        msg = f"fal {model} result could not be downloaded: {last.message}"
        raise BilledJobError(
            msg,
            provider=NAME,
            model=model,
            job_id=request_id or url,
            spent_usd=spent_usd,
            status_code=getattr(last, "status_code", None),
        ) from last

    def _download(
        self, url: str, dest: Path, *, expect_prefix: str, model: str, timeout_s: int
    ) -> Path:
        """Stream a result file to disk with a running size cap. A content type other than
        `expect_prefix` is only a WARNING (CDNs sometimes send application/octet-stream);
        ffprobe downstream is the real check. Nothing is left at `dest` on failure."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        size = 0
        try:
            with self._http.stream("GET", url, timeout=float(timeout_s)) as resp:
                if resp.status_code != httpx.codes.OK:
                    status = resp.status_code
                    cls = (
                        RetryableProviderError
                        if status in RETRYABLE_HTTP_STATUS
                        else FatalProviderError
                    )
                    msg = f"fal download failed with HTTP {status}"
                    raise cls(msg, provider=NAME, model=model, status_code=status)
                ctype = resp.headers.get("content-type", "")
                if expect_prefix and not ctype.startswith(expect_prefix):
                    log.warning("fal.download.unexpected_type", url=url, content_type=ctype)
                with part.open("wb") as fh:
                    for chunk in resp.iter_bytes():
                        size += len(chunk)
                        if size > _MAX_DOWNLOAD_BYTES:
                            msg = "downloaded file exceeds size limit"
                            raise MediaError(msg, context={"url": url, "bytes": size})
                        fh.write(chunk)
        except httpx.TransportError as exc:
            part.unlink(missing_ok=True)
            msg = f"fal download transport error: {exc}"
            raise RetryableProviderError(msg, provider=NAME, model=model) from exc
        except PipelineError:
            part.unlink(missing_ok=True)
            raise
        if size == 0:
            part.unlink(missing_ok=True)
            msg = "fal download: file is empty"
            raise RetryableProviderError(msg, provider=NAME, model=model)
        part.replace(dest)
        log.debug("fal.downloaded", path=str(dest), bytes=size)
        return dest


def _map_http_error(
    exc: FalClientHTTPError, *, model: str, op: str, request_id: str | None
) -> ProviderError:
    status = exc.status_code
    error_type = getattr(exc, "error_type", None)
    kwargs: dict[str, Any] = {
        "provider": NAME,
        "model": model,
        "request_id": request_id,
        "status_code": status,
        "context": {"error_type": error_type},
    }
    rejected = looks_like_prompt_rejection(f"{exc} {error_type or ''}") or (
        status == httpx.codes.BAD_REQUEST and op == "text_to_video"
    )
    cls: type[ProviderError]
    if status in RETRYABLE_HTTP_STATUS:
        cls = RetryableProviderError
    elif rejected:
        cls = PromptRejectedError
    else:
        cls = FatalProviderError
    return cls(f"fal {model}: HTTP {status}: {exc}", **kwargs)


def _file_url(
    result: dict[str, Any],
    key: str,
    *,
    model: str | None = None,
    request_id: str | None = None,
    spent_usd: float | None = None,
) -> str:
    """Extract `result[key]["url"]` or raise a fatal error with the payload keys (the job ran,
    so the error carries what it cost)."""
    obj = result.get(key)
    url = obj.get("url") if isinstance(obj, dict) else None
    if not isinstance(url, str) or not url:
        msg = f"fal result has no {key}.url (keys: {sorted(result)})"
        raise FatalProviderError(
            msg, provider=NAME, model=model, request_id=request_id, spent_usd=spent_usd
        )
    return url


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return None


def _redact(arguments: dict[str, Any]) -> dict[str, Any]:
    """Keep log payloads short: long strings are truncated, URLs kept."""
    return {
        k: (v[:_LOG_VALUE_CHARS] + "…") if isinstance(v, str) and len(v) > _LOG_VALUE_CHARS else v
        for k, v in arguments.items()
    }
