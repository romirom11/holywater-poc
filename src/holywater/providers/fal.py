"""fal.ai vendor: video (Seedance), TTS (ElevenLabs/MiniMax), STT (Whisper), lipsync.

One `SyncClient`, one submit/poll path, one download path, one error mapping — shared by every
capability. Model ids are fal application ids, e.g. `bytedance/seedance-2.5/text-to-video`.
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
    FatalProviderError,
    MediaError,
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
)

log = get_logger(__name__)

NAME = "fal"
_MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024
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
    timeout_s: int = 600
    lipsync_sync_mode: str = "cut_off"
    _client: fal_client.SyncClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = fal_client.SyncClient(
            key=self.api_key, default_timeout=float(self.timeout_s)
        )

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
        result, request_id = self._call(request.model, arguments, op="text_to_video")
        url = _file_url(result, "video")
        self._download(url, dest, expect_prefix="video/", model=request.model)
        per_second = VIDEO_PRICE_PER_SECOND_USD.get(request.model, {})
        return VideoResult(
            path=dest,
            model=request.model,
            source_url=url,
            seed=_as_int(result.get("seed")),
            request_id=request_id,
            cost_usd=round(per_second.get(request.resolution.value, 0.0) * request.duration_s, 4),
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
        result, _ = self._call(request.model, arguments, op="tts")
        self._download(
            _file_url(result, "audio"), dest, expect_prefix="audio/", model=request.model
        )
        price = TTS_PRICE_PER_1K_CHARS_USD.get(request.model, 0.0)
        return TTSResult(
            path=dest, model=request.model, cost_usd=round(price * len(request.text) / 1000, 5)
        )

    # -- STTProvider -------------------------------------------------------------------------

    def transcribe(self, request: STTRequest) -> STTResult:
        url = self._upload(request.audio_path, model=request.model)
        arguments: dict[str, Any] = {"audio_url": url, "task": "transcribe", "chunk_level": "none"}
        if request.language and request.language != "auto":
            arguments["language"] = request.language
        result, _ = self._call(request.model, arguments, op="stt")
        text = result.get("text")
        if not isinstance(text, str):
            msg = f"fal STT result has no text (keys: {sorted(result)})"
            raise FatalProviderError(msg, provider=NAME, model=request.model)
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
        result, _ = self._call(request.model, arguments, op="lipsync")
        self._download(
            _file_url(result, "video"), dest, expect_prefix="video/", model=request.model
        )
        return LipsyncResult(path=dest, model=request.model)

    # -- shared internals --------------------------------------------------------------------

    def _call(
        self, model: str, arguments: dict[str, Any], *, op: str
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
                model, arguments, on_enqueue=_on_enqueue, client_timeout=self.timeout_s
            )
        except FalClientHTTPError as exc:
            raise self._map_http_error(exc, model=model, request_id=request_id) from exc
        except FalClientTimeoutError as exc:
            msg = f"fal {model} timed out after {self.timeout_s}s"
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
        """Upload a local file to fal storage and return its URL (inputs for STT/lipsync)."""
        try:
            return str(fal_client.upload_file(path))
        except FalClientHTTPError as exc:
            raise self._map_http_error(exc, model=model, request_id=None) from exc
        except (httpx.TransportError, FalClientError) as exc:
            msg = f"fal upload failed: {exc}"
            raise RetryableProviderError(msg, provider=NAME, model=model) from exc

    def _download(self, url: str, dest: Path, *, expect_prefix: str, model: str) -> Path:
        """Stream a result file to disk, checking content type and size."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        try:
            with (
                httpx.Client(timeout=120, follow_redirects=True) as client,
                client.stream("GET", url) as resp,
            ):
                if resp.status_code != httpx.codes.OK:
                    msg = f"download failed with HTTP {resp.status_code}"
                    raise RetryableProviderError(
                        msg, provider=NAME, model=model, status_code=resp.status_code
                    )
                ctype = resp.headers.get("content-type", "")
                if expect_prefix and not ctype.startswith(expect_prefix):
                    log.warning("fal.download.unexpected_type", url=url, content_type=ctype)
                with dest.open("wb") as fh:
                    for chunk in resp.iter_bytes():
                        size += len(chunk)
                        if size > _MAX_DOWNLOAD_BYTES:
                            msg = "downloaded file exceeds size limit"
                            raise MediaError(msg, context={"url": url, "bytes": size})
                        fh.write(chunk)
        except httpx.TransportError as exc:
            msg = f"download transport error: {exc}"
            raise RetryableProviderError(msg, provider=NAME, model=model) from exc
        if size == 0:
            msg = "downloaded file is empty"
            raise MediaError(msg, context={"url": url, "path": str(dest)})
        log.debug("fal.downloaded", path=str(dest), bytes=size)
        return dest

    @staticmethod
    def _map_http_error(
        exc: FalClientHTTPError, *, model: str, request_id: str | None
    ) -> ProviderError:
        status = exc.status_code
        kwargs: dict[str, Any] = {
            "provider": NAME,
            "model": model,
            "request_id": request_id,
            "status_code": status,
            "context": {"error_type": exc.error_type},
        }
        cls = RetryableProviderError if status in RETRYABLE_HTTP_STATUS else FatalProviderError
        return cls(f"fal {model}: HTTP {status}: {exc}", **kwargs)


def _file_url(result: dict[str, Any], key: str) -> str:
    """Extract `result[key]["url"]` or raise a fatal error with the payload keys."""
    obj = result.get(key)
    url = obj.get("url") if isinstance(obj, dict) else None
    if not isinstance(url, str) or not url:
        msg = f"fal result has no {key}.url (keys: {sorted(result)})"
        raise FatalProviderError(msg, provider=NAME)
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
