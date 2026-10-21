"""Quality gates. Each returns a `GateResult`; raising is the caller's decision."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from holywater.domain.media import MediaInfo
from holywater.observability.errors import QualityGateError
from holywater.quality.text_norm import wer


@dataclass(frozen=True)
class GateResult:
    gate: str
    passed: bool
    measured: float | str | None
    threshold: float | str | None
    details: dict[str, Any] = field(default_factory=dict)

    def raise_if_failed(self, *, stage: str | None = None, run_id: str | None = None) -> None:
        if not self.passed:
            msg = f"{self.gate} gate failed: measured={self.measured} threshold={self.threshold}"
            raise QualityGateError(
                msg,
                gate=self.gate,
                measured=self.measured,
                threshold=self.threshold,
                stage=stage,
                run_id=run_id,
                context=self.details,
            )


@dataclass(frozen=True)
class DurationGate:
    min_s: float
    max_s: float
    tolerance_s: float = 0.0
    name: str = "duration"

    def check(self, duration_s: float) -> GateResult:
        low, high = self.min_s - self.tolerance_s, self.max_s + self.tolerance_s
        return GateResult(
            gate=self.name,
            passed=low <= duration_s <= high,
            measured=round(duration_s, 3),
            threshold=f"{self.min_s}..{self.max_s}",
        )


@dataclass(frozen=True)
class SpeechGate:
    """WER between the expected line and the STT transcript."""

    max_wer: float
    name: str = "speech"

    def check(self, expected: str, transcript: str, language: str = "en") -> GateResult:
        measured = wer(expected, transcript, language)
        return GateResult(
            gate=self.name,
            passed=measured <= self.max_wer,
            measured=round(measured, 4),
            threshold=self.max_wer,
            details={"expected": expected, "transcript": transcript},
        )


@dataclass(frozen=True)
class MediaIntegrityGate:
    """The file is a playable vertical video with an audio track."""

    require_audio: bool = True
    require_vertical: bool = True
    name: str = "media_integrity"

    def check(self, info: MediaInfo) -> GateResult:
        problems: list[str] = []
        if not info.has_video or info.duration_s <= 0:
            problems.append("no video stream")
        if self.require_audio and not info.has_audio:
            problems.append("no audio stream")
        if self.require_vertical and not info.is_vertical:
            problems.append(f"not vertical ({info.width}x{info.height})")
        return GateResult(
            gate=self.name,
            passed=not problems,
            measured="; ".join(problems) or "ok",
            threshold="video+audio, vertical" if self.require_vertical else "video+audio",
            details=info.model_dump(),
        )
