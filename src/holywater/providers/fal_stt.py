"""Speech-to-text on fal.ai (Whisper / Wizper: `audio_url` in, `text` out)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from holywater.observability.errors import FatalProviderError
from holywater.providers.base import STTResult
from holywater.providers.fal_base import PROVIDER, FalBase

# Flat estimate per request; fal bills Whisper per audio minute at a few cents.
PRICE_PER_CALL_USD = 0.01


@dataclass
class FalSTT(FalBase):
    def transcribe(self, audio_or_video: Path, *, language: str | None = None) -> STTResult:
        url = self.upload(audio_or_video)
        arguments: dict[str, Any] = {"audio_url": url, "task": "transcribe", "chunk_level": "none"}
        if language and language != "auto":
            arguments["language"] = language
        result, _ = self.call(arguments, op="stt")
        text = result.get("text")
        if not isinstance(text, str):
            msg = f"fal STT result has no text (keys: {sorted(result)})"
            raise FatalProviderError(msg, provider=PROVIDER, model=self.model)
        inferred = result.get("inferred_languages")
        detected = inferred[0] if isinstance(inferred, list) and inferred else language
        return STTResult(
            text=text.strip(),
            model=self.model,
            language=str(detected) if detected else None,
            cost_usd=PRICE_PER_CALL_USD,
        )
