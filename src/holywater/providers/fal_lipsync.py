"""Lip-sync on fal.ai (`fal-ai/sync-lipsync/v2`): video + audio in, video out."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from holywater.providers.base import LipsyncResult
from holywater.providers.fal_base import FalBase, file_url

PRICE_PER_MINUTE_USD = 3.0


@dataclass
class FalLipsync(FalBase):
    sync_mode: str = "cut_off"

    def sync(self, video: Path, audio: Path, *, dest: Path) -> LipsyncResult:
        arguments = {
            "video_url": self.upload(video),
            "audio_url": self.upload(audio),
            "sync_mode": self.sync_mode,
        }
        result, _ = self.call(arguments, op="lipsync")
        self.download(file_url(result, "video"), dest, expect_prefix="video/")
        return LipsyncResult(path=dest, model=self.model, cost_usd=0.0)
