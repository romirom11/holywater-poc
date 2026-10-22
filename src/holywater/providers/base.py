"""Capability contracts (ADR-008).

A *vendor* (fal, OpenRouter, TypeSafe) is one class that may implement several of these
capability Protocols. Stages depend on a capability, never on a vendor; `providers.registry`
picks the vendor named in config and checks it actually implements the capability.

Every request carries the `model` to use, so one vendor instance serves any number of models.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from holywater.domain.enums import Resolution

# -- LLM --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMResult[T: BaseModel]:
    data: T
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cost_is_estimate: bool = True
    raw_text: str = ""


@runtime_checkable
class LLMProvider(Protocol):
    def complete_structured[T: BaseModel](
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type[T],
        temperature: float | None = None,
    ) -> LLMResult[T]:
        """Return a schema-validated object. Raises `ProviderError` on API failure and
        `ValueError` when the model's output does not validate against `schema`."""
        ...


# -- Video ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoRequest:
    model: str
    prompt: str
    duration_s: int
    resolution: Resolution
    aspect_ratio: str
    generate_audio: bool
    seed: int | None = None


@dataclass(frozen=True)
class VideoResult:
    path: Path
    model: str
    source_url: str | None = None
    seed: int | None = None
    request_id: str | None = None
    cost_usd: float = 0.0
    cost_is_estimate: bool = True


@runtime_checkable
class VideoProvider(Protocol):
    def generate(self, request: VideoRequest, *, dest: Path) -> VideoResult:
        """Generate a clip and download it to `dest`."""
        ...


# -- TTS / STT / lipsync ----------------------------------------------------------------------


@dataclass(frozen=True)
class TTSRequest:
    model: str
    text: str
    language: str
    voice: str | None = None


@dataclass(frozen=True)
class TTSResult:
    path: Path
    model: str
    cost_usd: float = 0.0
    cost_is_estimate: bool = True


@runtime_checkable
class TTSProvider(Protocol):
    def synthesize(self, request: TTSRequest, *, dest: Path) -> TTSResult: ...


@dataclass(frozen=True)
class STTRequest:
    model: str
    audio_path: Path
    language: str | None = None
    source_path: Path | None = None
    """The media the audio was extracted from (traceability; fakes read their sidecar here)."""


@dataclass(frozen=True)
class STTResult:
    text: str
    model: str
    language: str | None = None
    cost_usd: float = 0.0
    cost_is_estimate: bool = True


@runtime_checkable
class STTProvider(Protocol):
    def transcribe(self, request: STTRequest) -> STTResult: ...


@dataclass(frozen=True)
class LipsyncRequest:
    model: str
    video_path: Path
    audio_path: Path


@dataclass(frozen=True)
class LipsyncResult:
    path: Path
    model: str
    cost_usd: float = 0.0
    cost_is_estimate: bool = True


@runtime_checkable
class LipsyncProvider(Protocol):
    def sync(self, request: LipsyncRequest, *, dest: Path) -> LipsyncResult: ...


# -- Typed judge (Jev) ------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeAnswer:
    question: str
    answer: str
    probabilities: dict[str, float] = field(default_factory=dict)
    model_version: str | None = None
    latency_ms: int | None = None
    available: bool = True

    @property
    def confidence(self) -> float:
        return self.probabilities.get(self.answer, 0.0)


@runtime_checkable
class JudgeProvider(Protocol):
    """Typed decisions: yes/no probability, one-of-N choice, or a score on a scale."""

    @property
    def enabled(self) -> bool: ...

    def ask_bool(self, state: str, question: str) -> JudgeAnswer:
        """`answer` is "yes" or "no"; `probabilities` holds both."""
        ...

    def ask_choice(self, state: str, question: str, options: Sequence[str]) -> JudgeAnswer:
        """`answer` is one of `options`."""
        ...

    def ask_score(self, state: str, question: str, *, levels: int) -> JudgeAnswer:
        """`answer` is a 1-based integer rendered as a string, 1..levels."""
        ...


class NoopJudge:
    """Used when Jev is disabled or unreachable: every answer is `unknown`/unavailable."""

    enabled = False

    def ask_bool(self, state: str, question: str) -> JudgeAnswer:
        return JudgeAnswer(question=question, answer="unknown", available=False)

    def ask_choice(self, state: str, question: str, options: Sequence[str]) -> JudgeAnswer:
        return JudgeAnswer(question=question, answer="unknown", available=False)

    def ask_score(self, state: str, question: str, *, levels: int) -> JudgeAnswer:
        return JudgeAnswer(question=question, answer="unknown", available=False)


# -- shared helpers for vendor adapters -------------------------------------------------------

# HTTP statuses worth retrying with backoff; everything else 4xx is the caller's fault.
RETRYABLE_HTTP_STATUS: frozenset[int] = frozenset(
    {408, 409, 425, 429, 500, 502, 503, 504, 524, 529}
)
