"""Typed view of `config/pipeline.yaml` (see docs/ARCHITECTURE.md §4).

Each capability section names its `provider` (vendor) and `model`; vendor-wide connection
settings live under `providers:`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from holywater.domain.enums import AudioStrategy, Resolution

Vendor = Literal["fal", "openrouter"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LLMConfig(_Strict):
    provider: Vendor = "openrouter"
    model: str
    # None = do not send the parameter (Claude 5.x endpoints reject `temperature`).
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_attempts: int = Field(default=3, ge=1, le=10)


class PreflightConfig(_Strict):
    enabled: bool = True
    safety_risk_threshold: float = Field(default=0.7, ge=0, le=1)
    min_filmability_score: int = Field(default=3, ge=1, le=5)


class FailureClassifierConfig(_Strict):
    enabled: bool = True


class JudgeConfig(_Strict):
    enabled: bool = True
    model: str = "jev-latest"
    preflight: PreflightConfig = PreflightConfig()
    failure_classifier: FailureClassifierConfig = FailureClassifierConfig()


class VideoConfig(_Strict):
    provider: Vendor = "openrouter"
    model: str
    resolution: Resolution = Resolution.P480
    aspect_ratio: str = "9:16"
    generate_audio: bool = True
    max_attempts: int = Field(default=2, ge=1, le=5)


class TTSConfig(_Strict):
    provider: Vendor = "openrouter"
    model: str
    voice: str | None = None


class LipsyncConfig(_Strict):
    provider: Vendor = "fal"
    enabled: bool = False
    model: str = "fal-ai/sync-lipsync/v2"


class STTConfig(_Strict):
    provider: Vendor = "openrouter"
    model: str
    language: str = "auto"


class AudioConfig(_Strict):
    strategy: AudioStrategy = AudioStrategy.NATIVE
    native_attempts: int = Field(default=2, ge=1, le=5)
    tts: TTSConfig
    lipsync: LipsyncConfig = LipsyncConfig()
    stt: STTConfig


class RangeConfig(_Strict):
    min: int = Field(ge=0)
    max: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> RangeConfig:
        if self.min > self.max:
            msg = f"min ({self.min}) must be <= max ({self.max})"
            raise ValueError(msg)
        return self


class QualityConfig(_Strict):
    max_wer: float = Field(default=0.10, ge=0, le=1)
    duration_s: RangeConfig = RangeConfig(min=8, max=30)
    shots: RangeConfig = RangeConfig(min=1, max=2)
    # Per-clip DurationGate: |clip - requested| must be within this, else the shot is regenerated.
    shot_duration_tolerance_s: float = Field(default=1.0, ge=0)
    # `truncated` tactic: how much the dialogue window grows on each side before regenerating.
    truncated_window_widen_s: float = Field(default=1.0, ge=0)


class ProviderRetryConfig(_Strict):
    attempts: int = Field(default=3, ge=1, le=10)
    backoff_s: list[int] = Field(default_factory=lambda: [5, 15, 45])


class TimeoutsConfig(_Strict):
    llm: int = Field(default=60, gt=0)
    video: int = Field(default=600, gt=0)
    tts: int = Field(default=60, gt=0)
    stt: int = Field(default=120, gt=0)
    judge: int = Field(default=10, gt=0)


class RetriesConfig(_Strict):
    provider: ProviderRetryConfig = ProviderRetryConfig()
    timeouts_s: TimeoutsConfig = TimeoutsConfig()


class OpenRouterSettings(_Strict):
    base_url: str = "https://openrouter.ai/api/v1"
    video_poll_interval_s: float = Field(default=10.0, gt=0)


class FalSettings(_Strict):
    lipsync_sync_mode: Literal["cut_off", "loop", "bounce", "silence", "remap"] = "cut_off"


class ProvidersConfig(_Strict):
    openrouter: OpenRouterSettings = OpenRouterSettings()
    fal: FalSettings = FalSettings()


class HitlPointConfig(_Strict):
    enabled: bool = False


class HitlConfig(_Strict):
    after_script: HitlPointConfig = HitlPointConfig()
    on_unverified_speech: HitlPointConfig = HitlPointConfig()


class StorageConfig(_Strict):
    runs_dir: Path = Path("runs")
    db_path: Path = Path("runs/holywater.db")
    final_name: str = "final.mp4"


class PipelineConfig(_Strict):
    version: Literal[1] = 1
    llm: LLMConfig
    judge: JudgeConfig = JudgeConfig()
    video: VideoConfig
    audio: AudioConfig
    quality: QualityConfig = QualityConfig()
    retries: RetriesConfig = RetriesConfig()
    providers: ProvidersConfig = ProvidersConfig()
    hitl: HitlConfig = HitlConfig()
    storage: StorageConfig = StorageConfig()

    def config_hash(self) -> str:
        """Stable hash of the effective config; stored per run to compare pipeline versions."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def vendors_in_use(self) -> set[str]:
        used: set[str] = {
            self.llm.provider,
            self.video.provider,
            self.audio.tts.provider,
            self.audio.stt.provider,
        }
        if self.audio.lipsync.enabled:
            used.add(self.audio.lipsync.provider)
        return used
