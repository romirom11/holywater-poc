"""Quality gates. Each returns a `GateResult`; raising is the caller's decision."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from holywater.domain.media import AudioLevels, MediaInfo
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
        measured = round(duration_s, 3)  # millisecond precision: ffprobe float noise is not a miss
        return GateResult(
            gate=self.name,
            passed=low <= measured <= high,
            measured=measured,
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
    """The file is a playable 9:16 video whose audio track is present and not silent.

    The silence check needs measured `levels` (`FFmpeg.audio_levels`); without them only the
    presence of the track is checked. Digital silence measures about -91 dB peak.
    """

    require_audio: bool = True
    require_vertical: bool = True
    aspect_ratio: float = 9 / 16
    aspect_tolerance: float = 0.01  # 480x854 is 0.562 vs 0.5625
    min_peak_db: float | None = -50.0
    name: str = "media_integrity"

    def check(self, info: MediaInfo, *, levels: AudioLevels | None = None) -> GateResult:
        problems: list[str] = []
        if not info.has_video or info.duration_s <= 0:
            problems.append("no video stream")
        elif info.fps <= 0:
            problems.append("unknown frame rate")
        if self.require_audio and not info.has_audio:
            problems.append("no audio stream")
        elif self.require_audio and levels is not None and self._is_silent(levels):
            problems.append(f"silent audio (peak {levels.max_volume_db} dB)")
        if self.require_vertical and not info.is_vertical:
            problems.append(f"not vertical ({info.width}x{info.height})")
        elif self.require_vertical and abs(info.aspect_ratio - self.aspect_ratio) > (
            self.aspect_tolerance
        ):
            problems.append(f"not 9:16 ({info.width}x{info.height})")
        details: dict[str, Any] = info.model_dump()
        if levels is not None:
            details["audio_levels"] = levels.model_dump()
        return GateResult(
            gate=self.name,
            passed=not problems,
            measured="; ".join(problems) or "ok",
            threshold=self._threshold(),
            details=details,
        )

    def _is_silent(self, levels: AudioLevels) -> bool:
        return self.min_peak_db is not None and levels.max_volume_db < self.min_peak_db

    def _threshold(self) -> str:
        audio = "audio" if self.min_peak_db is None else f"audio peak >= {self.min_peak_db} dB"
        parts = ["video", audio] if self.require_audio else ["video"]
        if self.require_vertical:
            parts.append("9:16")
        return ", ".join(parts)
