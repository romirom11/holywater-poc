"""Per-run report exported from the DB (docs/QUALITY_METRICS.md §1)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class StageMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: str
    attempts: int
    duration_s: float | None = None
    error_type: str | None = None


class ShotMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shot_id: int
    has_dialogue: bool
    attempts: int
    final_attempt: int | None
    final_strategy: str | None
    final_wer: float | None
    verified: bool | None
    failure_kinds: list[str] = Field(default_factory=list)
    cost_usd: float = 0.0


class CostLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    provider: str
    model: str
    cost_usd: float


class FinalMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    duration_s: float
    width: int
    height: int
    fps: float
    has_audio: bool


class RunReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    tag: str | None
    success: bool
    status: str
    failure_stage: str | None = None
    failure_type: str | None = None
    input_text: str
    dialogue_line: str | None = None
    config_hash: str
    git_sha: str | None = None
    models: dict[str, str] = Field(default_factory=dict)
    prompt_versions: list[str] = Field(default_factory=list)
    audio_strategy_requested: str
    audio_strategy_used: str | None
    fallback_reason: str | None
    started_at: datetime
    finished_at: datetime | None
    latency_s: float | None
    stages: list[StageMetrics] = Field(default_factory=list)
    shots: list[ShotMetrics] = Field(default_factory=list)
    shots_regenerated: int = 0
    retries_total: int = 0
    judge_decisions: int = 0
    preflight_rewrites: int = 0
    costs: list[CostLine] = Field(default_factory=list)
    cost_usd_total: float = 0.0
    final: FinalMetrics | None = None
