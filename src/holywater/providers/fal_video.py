"""Seedance (2.5 / 2.0) text-to-video on fal.ai."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from holywater.providers.base import VideoRequest, VideoResult
from holywater.providers.fal_base import FalBase, file_url

# USD per second of output, used for cost estimates only (fal bills per token; see ADR-003).
PRICE_PER_SECOND_USD: dict[str, dict[str, float]] = {
    "bytedance/seedance-2.5/text-to-video": {"480p": 0.2205, "720p": 0.4730},
    "bytedance/seedance-2.0/text-to-video": {"480p": 0.14, "720p": 0.3034},
}


@dataclass
class FalVideo(FalBase):
    def generate(self, request: VideoRequest, *, dest: Path) -> VideoResult:
        arguments: dict[str, Any] = {
            "prompt": request.prompt,
            "duration": str(request.duration_s),
            "resolution": request.resolution.value,
            "aspect_ratio": request.aspect_ratio,
            "generate_audio": request.generate_audio,
        }
        if request.seed is not None:
            arguments["seed"] = request.seed
        result, request_id = self.call(arguments, op="text_to_video")
        url = file_url(result, "video")
        self.download(url, dest, expect_prefix="video/")
        seed = result.get("seed")
        return VideoResult(
            path=dest,
            model=self.model,
            source_url=url,
            seed=int(seed)
            if isinstance(seed, int | str) and str(seed).lstrip("-").isdigit()
            else None,
            request_id=request_id,
            cost_usd=self.estimate_cost(request),
        )

    def estimate_cost(self, request: VideoRequest) -> float:
        table = PRICE_PER_SECOND_USD.get(self.model, {})
        return round(table.get(request.resolution.value, 0.0) * request.duration_s, 4)
