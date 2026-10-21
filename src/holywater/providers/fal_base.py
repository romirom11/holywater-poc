"""Single fal.ai client wrapper: submit/poll, download, timeouts, and error mapping.

Every fal adapter (video, TTS, STT, lipsync) goes through `FalBase.call()` so that retry
classification and logging happen in exactly one place.
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

log = get_logger(__name__)

PROVIDER = "fal"
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024


@dataclass
class FalBase:
    api_key: str
    model: str
    timeout_s: int = 600
    _client: fal_client.SyncClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = fal_client.SyncClient(
            key=self.api_key, default_timeout=float(self.timeout_s)
        )

    def call(self, arguments: dict[str, Any], *, op: str) -> tuple[dict[str, Any], str | None]:
        """Run the model and wait for the result. Returns `(result, request_id)`."""
        request_id: str | None = None

        def _on_enqueue(rid: str) -> None:
            nonlocal request_id
            request_id = rid
            log.info("fal.enqueued", op=op, model=self.model, request_id=rid)

        started = time.monotonic()
        log.debug("fal.submit", op=op, model=self.model, arguments=_redact(arguments))
        result: Any
        try:
            result = self._client.subscribe(
                self.model,
                arguments,
                on_enqueue=_on_enqueue,
                client_timeout=self.timeout_s,
            )
        except FalClientHTTPError as exc:
            raise self._map_http_error(exc, request_id) from exc
        except FalClientTimeoutError as exc:
            msg = f"fal {self.model} timed out after {self.timeout_s}s"
            raise RetryableProviderError(
                msg, provider=PROVIDER, model=self.model, request_id=request_id
            ) from exc
        except (httpx.TransportError, FalClientError) as exc:
            msg = f"fal {self.model} transport error: {exc}"
            raise RetryableProviderError(
                msg, provider=PROVIDER, model=self.model, request_id=request_id
            ) from exc
        if not isinstance(result, dict):
            msg = f"fal {self.model} returned a non-object result"
            raise FatalProviderError(
                msg, provider=PROVIDER, model=self.model, request_id=request_id
            )
        log.info(
            "fal.done",
            op=op,
            model=self.model,
            request_id=request_id,
            latency_s=round(time.monotonic() - started, 2),
        )
        return result, request_id

    def upload(self, path: Path) -> str:
        """Upload a local file to fal storage and return its URL (inputs for STT/lipsync)."""
        try:
            return str(fal_client.upload_file(path))
        except FalClientHTTPError as exc:
            raise self._map_http_error(exc, None) from exc
        except (httpx.TransportError, FalClientError) as exc:
            msg = f"fal upload failed: {exc}"
            raise RetryableProviderError(msg, provider=PROVIDER, model=self.model) from exc

    def download(self, url: str, dest: Path, *, expect_prefix: str) -> Path:
        """Stream a result file to disk, checking content type and size."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with (
                httpx.Client(timeout=120, follow_redirects=True) as client,
                client.stream("GET", url) as resp,
            ):
                if resp.status_code != httpx.codes.OK:
                    msg = f"download failed with HTTP {resp.status_code}"
                    raise RetryableProviderError(
                        msg, provider=PROVIDER, model=self.model, status_code=resp.status_code
                    )
                ctype = resp.headers.get("content-type", "")
                if expect_prefix and not ctype.startswith(expect_prefix):
                    log.warning("fal.download.unexpected_type", url=url, content_type=ctype)
                size = 0
                with dest.open("wb") as fh:
                    for chunk in resp.iter_bytes():
                        size += len(chunk)
                        if size > _MAX_DOWNLOAD_BYTES:
                            msg = "downloaded file exceeds size limit"
                            raise MediaError(msg, context={"url": url, "bytes": size})
                        fh.write(chunk)
        except httpx.TransportError as exc:
            msg = f"download transport error: {exc}"
            raise RetryableProviderError(msg, provider=PROVIDER, model=self.model) from exc
        if size == 0:
            msg = "downloaded file is empty"
            raise MediaError(msg, context={"url": url, "path": str(dest)})
        log.debug("fal.downloaded", path=str(dest), bytes=size)
        return dest

    def _map_http_error(self, exc: FalClientHTTPError, request_id: str | None) -> ProviderError:
        status = exc.status_code
        kwargs: dict[str, Any] = {
            "provider": PROVIDER,
            "model": self.model,
            "request_id": request_id,
            "status_code": status,
            "context": {"error_type": exc.error_type},
        }
        if status in _RETRYABLE_STATUS:
            return RetryableProviderError(f"fal {self.model}: HTTP {status}: {exc}", **kwargs)
        return FatalProviderError(f"fal {self.model}: HTTP {status}: {exc}", **kwargs)


def file_url(result: dict[str, Any], key: str) -> str:
    """Extract `result[key]["url"]` or raise a fatal error with the payload keys."""
    obj = result.get(key)
    url = obj.get("url") if isinstance(obj, dict) else None
    if not isinstance(url, str) or not url:
        msg = f"fal result has no {key}.url (keys: {sorted(result)})"
        raise FatalProviderError(msg, provider=PROVIDER)
    return url


def _redact(arguments: dict[str, Any]) -> dict[str, Any]:
    """Keep log payloads short: long strings are truncated, URLs kept."""
    out: dict[str, Any] = {}
    for k, v in arguments.items():
        out[k] = (v[:200] + "…") if isinstance(v, str) and len(v) > 200 else v  # noqa: PLR2004
    return out
