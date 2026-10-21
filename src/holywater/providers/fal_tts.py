"""Text-to-speech on fal.ai (ElevenLabs / MiniMax share the `text` in, `audio.url` out shape)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from holywater.providers.base import TTSRequest, TTSResult
from holywater.providers.fal_base import FalBase, file_url

# USD per 1k characters, estimate only.
PRICE_PER_1K_CHARS_USD: dict[str, float] = {
    "fal-ai/elevenlabs/tts/eleven-v3": 0.10,
    "fal-ai/elevenlabs/tts/turbo-v2.5": 0.05,
    "fal-ai/minimax/speech-02-hd": 0.10,
}


@dataclass
class FalTTS(FalBase):
    def synthesize(self, request: TTSRequest, *, dest: Path) -> TTSResult:
        arguments: dict[str, Any] = {"text": request.text}
        if request.voice:
            arguments["voice"] = request.voice
        if "elevenlabs" in self.model:
            arguments["language_code"] = request.language
            # The line must be read as written: never let the vendor "normalize" numbers etc.
            arguments["apply_text_normalization"] = "off"
        result, _ = self.call(arguments, op="tts")
        url = file_url(result, "audio")
        self.download(url, dest, expect_prefix="audio/")
        price = PRICE_PER_1K_CHARS_USD.get(self.model, 0.0)
        return TTSResult(
            path=dest, model=self.model, cost_usd=round(price * len(request.text) / 1000, 5)
        )
