"""SQLModel tables. Enum-valued columns are stored as plain strings (`StrEnum.value`)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(UTC)


class Run(SQLModel, table=True):
    __tablename__ = "runs"

    id: str = Field(primary_key=True)
    tag: str | None = Field(default=None, index=True)
    input_text: str
    language: str | None = None
    config_hash: str
    git_sha: str | None = None
    status: str = Field(index=True)
    audio_strategy_requested: str
    audio_strategy_used: str | None = None
    fallback_reason: str | None = None
    started_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None
    cost_usd_total: float = 0.0
    final_path: str | None = None
    error_type: str | None = None
    error_json: str | None = None


class Stage(SQLModel, table=True):
    __tablename__ = "stages"
    __table_args__ = (UniqueConstraint("run_id", "name", name="uq_stage_run_name"),)

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="runs.id", index=True)
    name: str
    status: str
    attempt: int = 0
    input_hash: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_type: str | None = None
    error_json: str | None = None


class Shot(SQLModel, table=True):
    __tablename__ = "shots"
    __table_args__ = (UniqueConstraint("run_id", "shot_id", name="uq_shot_run_shot"),)

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="runs.id", index=True)
    shot_id: int
    planned_duration_s: int
    has_dialogue: bool
    final_attempt_id: int | None = Field(default=None, foreign_key="shot_attempts.id")


class ShotAttempt(SQLModel, table=True):
    __tablename__ = "shot_attempts"

    id: int | None = Field(default=None, primary_key=True)
    shot_pk: int = Field(foreign_key="shots.id", index=True)
    run_id: str = Field(foreign_key="runs.id", index=True)
    shot_id: int
    attempt: int
    strategy: str
    model: str
    prompt_version: str
    prompt_text: str
    seed: int | None = None
    provider_request_id: str | None = None
    path: str | None = None
    duration_s: float | None = None
    resolution: str
    cost_usd: float = 0.0
    status: str = "generated"
    created_at: datetime = Field(default_factory=_now)


class SpeechVerification(SQLModel, table=True):
    __tablename__ = "speech_verifications"

    id: int | None = Field(default=None, primary_key=True)
    shot_attempt_id: int = Field(foreign_key="shot_attempts.id", index=True)
    run_id: str = Field(foreign_key="runs.id", index=True)
    expected: str
    transcript: str
    wer: float
    passed: bool
    verified: bool = True
    stt_model: str | None = None
    failure_kind: str | None = None
    created_at: datetime = Field(default_factory=_now)


class JudgeDecision(SQLModel, table=True):
    __tablename__ = "judge_decisions"

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="runs.id", index=True)
    stage: str
    question: str
    answer: str
    probabilities_json: str
    model_version: str | None = None
    latency_ms: int | None = None
    created_at: datetime = Field(default_factory=_now)


class Cost(SQLModel, table=True):
    __tablename__ = "costs"

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="runs.id", index=True)
    stage: str
    provider: str
    model: str
    units: float
    unit_price: float
    cost_usd: float
    is_estimate: bool = True
    shot_id: int | None = None
    attempt: int | None = None
    created_at: datetime = Field(default_factory=_now)
