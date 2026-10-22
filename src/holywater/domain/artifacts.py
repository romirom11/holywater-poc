"""References to artifacts produced by stages. Files live on disk; these carry paths + metadata."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from holywater.domain.enums import AudioStrategy, FailureKind, Resolution


class AudioArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shot_id: int
    path: Path
    duration_s: float = Field(gt=0)
    model: str
    voice: str | None = None
    text: str
    cost_usd: float = Field(ge=0, default=0.0)
    cost_is_estimate: bool = True


class VideoArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shot_id: int
    attempt: int = Field(ge=1)
    strategy: AudioStrategy
    path: Path
    source_url: str | None = None
    seed: int | None = None
    duration_s: float = Field(gt=0)
    resolution: Resolution
    has_native_audio: bool
    prompt_version: str
    prompt_text: str
    provider_request_id: str | None = None
    model: str
    cost_usd: float = Field(ge=0, default=0.0)
    cost_is_estimate: bool = True


class SpeechVerification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shot_id: int
    attempt: int = Field(ge=1)
    expected: str
    transcript: str
    wer: float = Field(ge=0)
    passed: bool
    verified: bool = Field(
        default=True, description="False when STT was unavailable and the gate could not run."
    )
    stt_model: str | None = None
    failure_kind: FailureKind | None = None
    cost_usd: float = Field(ge=0, default=0.0)
    cost_is_estimate: bool = True


class ProviderCost(BaseModel):
    """A paid provider call made inside a stage; the flow records each one via `_record_cost`.

    `capability` names the config section the vendor comes from (e.g. "lipsync" ->
    `audio.lipsync.provider`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    capability: str
    model: str
    units: float = Field(ge=0)
    cost_usd: float = Field(ge=0)
    is_estimate: bool
    shot_id: int | None = None


class FinalVideo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    duration_s: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    has_audio: bool
    strategy_used: AudioStrategy
    shots: list[VideoArtifact] = Field(min_length=1)
    provider_costs: list[ProviderCost] = Field(
        default_factory=list,
        description="Paid calls made by assemble (lipsync) that are not recorded yet.",
    )
