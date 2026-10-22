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

from holywater.domain.enums import AudioStrategy, Resolution, VoiceGender
from holywater.domain.script import MAX_TOTAL_S, MIN_TOTAL_S

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
    # Price used to estimate each Jev call (input tokens are billed, output tokens are free).
    usd_per_m_input_tokens: float = Field(default=0.042, ge=0)
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
    # Default voice: used when `voices` has no entry for the speaker, and as the retry voice when
    # the vendor rejects a gender-specific one.
    voice: str | None = None
    # Voice per `dialogue.voice_gender` (vendor/model specific ids).
    voices: dict[VoiceGender, str] = Field(default_factory=dict)

    def voice_for(self, gender: VoiceGender) -> str | None:
        return self.voices.get(gender, self.voice)


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


class StraySpeechConfig(_Strict):
    """STT on clips that must carry no speech (non-dialogue shots, the tts_overlay dialogue clip).

    `warn` accepts the clip and logs a WARNING; `regenerate` pays for another clip (within
    `video.max_attempts`, then accepts with a WARNING). Whisper hallucinates words on ambience,
    which is why the default only warns. The dialogue clip under a TTS voice is always handled as
    `regenerate`: its words would land in the line window and fail the final check.
    """

    action: Literal["warn", "regenerate"] = "warn"
    max_words: int = Field(default=2, ge=0)


class QualityConfig(_Strict):
    max_wer: float = Field(default=0.10, ge=0, le=1)
    # Final length bounds; must lie within the task's 8..30 s (the script invariant).
    duration_s: RangeConfig = RangeConfig(min=MIN_TOTAL_S, max=MAX_TOTAL_S)
    # Per-clip DurationGate: |clip - requested| must be within this, else the shot is regenerated.
    shot_duration_tolerance_s: float = Field(default=1.0, ge=0)
    # `truncated` tactic: how much the dialogue window grows on each side before regenerating.
    truncated_window_widen_s: float = Field(default=1.0, ge=0)
    # Script check: the dialogue window must hold the line at this speaking rate.
    speech_chars_per_s: float = Field(default=15.0, gt=0)
    # MediaIntegrityGate on the final: an audio peak below this (dBFS) counts as silent.
    min_audio_peak_db: float = Field(default=-50.0, le=0)
    # tts_overlay final check: audio kept on each side of the TTS voice when it is transcribed.
    final_check_margin_s: float = Field(default=0.5, ge=0)
    stray_speech: StraySpeechConfig = StraySpeechConfig()

    @model_validator(mode="after")
    def _duration_within_task_bounds(self) -> QualityConfig:
        bounds = self.duration_s
        if bounds.min < MIN_TOTAL_S or bounds.max > MAX_TOTAL_S:
            msg = (
                f"quality.duration_s must lie within {MIN_TOTAL_S}..{MAX_TOTAL_S}s (task bounds), "
                f"got {bounds.min}..{bounds.max}"
            )
            raise ValueError(msg)
        return self


class ProviderRetryConfig(_Strict):
    """Prefect task retries for retryable errors: `attempts` tries in total, so
    `attempts - 1` retries, each preceded by the matching `backoff_s` delay."""

    attempts: int = Field(default=3, ge=1, le=10)
    backoff_s: list[int] = Field(default_factory=lambda: [5, 15])

    @model_validator(mode="after")
    def _one_delay_per_retry(self) -> ProviderRetryConfig:
        if len(self.backoff_s) != self.attempts - 1:
            msg = (
                f"backoff_s needs one delay per retry: attempts={self.attempts} -> "
                f"{self.attempts - 1} values, got {len(self.backoff_s)}"
            )
            raise ValueError(msg)
        if any(d < 0 for d in self.backoff_s):
            raise ValueError("backoff_s delays must be >= 0")
        return self


class TimeoutsConfig(_Strict):
    llm: int = Field(default=60, gt=0)
    video: int = Field(default=600, gt=0)
    tts: int = Field(default=60, gt=0)
    stt: int = Field(default=120, gt=0)
    judge: int = Field(default=10, gt=0)
    ffmpeg: int = Field(default=600, gt=0)  # one ffmpeg/ffprobe subprocess


class RetriesConfig(_Strict):
    provider: ProviderRetryConfig = ProviderRetryConfig()
    timeouts_s: TimeoutsConfig = TimeoutsConfig()


class OpenRouterSettings(_Strict):
    base_url: str = "https://openrouter.ai/api/v1"
    video_poll_interval_s: float = Field(default=10.0, gt=0)
    # Downloads of a finished video job are idempotent reads, retried against the same job id.
    download_attempts: int = Field(default=3, ge=1, le=10)
    # USD per output second of `video.model` (default Seedance 2.5; docs/PROVIDERS_API.md). Only
    # used when the vendor does not report `usage.cost` (abandoned or cost-less jobs).
    video_usd_per_s: dict[Resolution, float] = Field(
        default_factory=lambda: {Resolution.P480: 0.1035, Resolution.P720: 0.233}
    )


class FalSettings(_Strict):
    lipsync_sync_mode: Literal["cut_off", "loop", "bounce", "silence", "remap"] = "cut_off"
    download_attempts: int = Field(default=3, ge=1, le=10)


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
