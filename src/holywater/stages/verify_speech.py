"""Stage 4: STT on the dialogue clip, WER gate, and (optionally) failure classification by Jev.

The audio track is extracted to a mono 16 kHz WAV first so every STT vendor gets the same input;
a `<wav>.json` next to it records which media and which segment it was cut from.

`check_stray_speech` runs the same STT on clips that must carry no speech (non-dialogue shots,
the silent clip under a TTS voice) and counts the words it hears.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from holywater.config.schema import JudgeConfig, QualityConfig, STTConfig
from holywater.domain.artifacts import SpeechVerification
from holywater.domain.enums import FailureKind
from holywater.media import FFmpeg
from holywater.observability.logging import get_logger
from holywater.providers.base import (
    JudgeAnswer,
    JudgeProvider,
    STTProvider,
    STTRequest,
    STTResult,
)
from holywater.quality.gates import SpeechGate
from holywater.quality.text_norm import normalize

log = get_logger(__name__)

Q_FAILURE_KIND = "failure_kind"
# `judge_decisions.decision` / `model_version` values for the failure classifier.
DECISION_CLASSIFIED = "classified"  # Jev picked the failure kind
DECISION_UNAVAILABLE = "unavailable"  # Jev was asked but gave no usable answer
DECISION_HEURISTIC = "heuristic"  # word-count heuristics picked it (Jev off or unusable)
HEURISTIC_MODEL = "heuristic"

_TRUNCATED_RATIO = 0.6
_EXTRA_RATIO = 1.5
_CLASSIFIABLE = [
    FailureKind.PARAPHRASED,
    FailureKind.TRUNCATED,
    FailureKind.EXTRA_WORDS,
    FailureKind.WRONG_LANGUAGE,
    FailureKind.STT_NOISE,
    FailureKind.SILENCE,
]


@dataclass
class VerifyDeps:
    stt: STTProvider
    judge: JudgeProvider
    ffmpeg: FFmpeg
    stt_config: STTConfig
    quality: QualityConfig
    judge_config: JudgeConfig


@dataclass(frozen=True)
class Classification:
    """One answer behind a `failure_kind`, and how it was reached (a `judge_decisions` row)."""

    answer: JudgeAnswer
    decision: str


@dataclass(frozen=True)
class VerifyResult:
    verification: SpeechVerification
    classifications: list[Classification] = field(default_factory=list)
    cost_usd: float = 0.0

    @property
    def decision(self) -> JudgeAnswer | None:
        """The answer that decided `failure_kind` (Jev's or the heuristic's), if any."""
        return self.classifications[-1].answer if self.classifications else None


@dataclass(frozen=True)
class StrayResult:
    verification: SpeechVerification
    words: int

    @property
    def found(self) -> bool:
        return not self.verification.passed


def verify_speech(
    media_path: Path,
    *,
    shot_id: int,
    attempt: int,
    expected: str,
    language: str,
    deps: VerifyDeps,
    work_dir: Path | None = None,
    window: tuple[float, float] | None = None,
    has_audio: bool = True,
) -> VerifyResult:
    """Transcribe and compare. Provider errors propagate (the flow decides on retries).

    `window=(start_s, duration_s)` transcribes only that segment — used on the assembled final,
    where other shots carry ambient sound that would otherwise pollute the transcript.
    A clip without an audio track (`has_audio=False`) said nothing: it fails as `silence` with
    an empty transcript, without an STT call.
    """
    if not has_audio:
        log.warning("verify_speech.no_audio_track", shot_id=shot_id, attempt=attempt)
        return VerifyResult(
            verification=SpeechVerification(
                shot_id=shot_id,
                attempt=attempt,
                expected=expected,
                transcript="",
                wer=1.0,
                passed=False,
                verified=True,
                failure_kind=FailureKind.SILENCE,
            )
        )
    stt = _transcribe(media_path, language=language, deps=deps, work_dir=work_dir, window=window)
    result = SpeechGate(max_wer=deps.quality.max_wer).check(expected, stt.text, language)
    log.info(
        "verify_speech.result",
        shot_id=shot_id,
        attempt=attempt,
        provider=deps.stt_config.provider,
        model=stt.model,
        wer=result.measured,
        passed=result.passed,
        transcript=stt.text,
    )
    failure_kind: FailureKind | None = None
    classifications: list[Classification] = []
    if not result.passed:
        failure_kind, classifications = classify_failure(expected, stt.text, language, deps)
        log.warning(
            "verify_speech.failed",
            shot_id=shot_id,
            attempt=attempt,
            wer=result.measured,
            failure_kind=failure_kind.value,
            classified_by=classifications[-1].decision,
        )
    verification = SpeechVerification(
        shot_id=shot_id,
        attempt=attempt,
        expected=expected,
        transcript=stt.text,
        wer=float(result.measured or 0.0),
        passed=result.passed,
        verified=True,
        stt_model=stt.model,
        failure_kind=failure_kind,
        cost_usd=stt.cost_usd,
        cost_is_estimate=stt.cost_is_estimate,
    )
    return VerifyResult(
        verification=verification, classifications=classifications, cost_usd=stt.cost_usd
    )


def check_stray_speech(
    media_path: Path,
    *,
    shot_id: int,
    attempt: int,
    language: str,
    max_words: int,
    deps: VerifyDeps,
    work_dir: Path | None = None,
) -> StrayResult:
    """STT on a clip that must carry no speech. More than `max_words` words is stray speech:
    recorded as a failed verification with `expected=""` and `failure_kind=extra_words`."""
    stt = _transcribe(media_path, language=language, deps=deps, work_dir=work_dir, window=None)
    words = len(normalize(stt.text, language).split())
    passed = words <= max_words
    log.info(
        "verify_speech.stray_check",
        shot_id=shot_id,
        attempt=attempt,
        words=words,
        max_words=max_words,
        transcript=stt.text,
    )
    verification = SpeechVerification(
        shot_id=shot_id,
        attempt=attempt,
        expected="",
        transcript=stt.text,
        wer=0.0 if words == 0 else 1.0,
        passed=passed,
        verified=True,
        stt_model=stt.model,
        failure_kind=None if passed else FailureKind.EXTRA_WORDS,
        cost_usd=stt.cost_usd,
        cost_is_estimate=stt.cost_is_estimate,
    )
    return StrayResult(verification=verification, words=words)


def unverified(*, shot_id: int, attempt: int, expected: str, reason: str) -> SpeechVerification:
    """Record that the gate could not run (STT unavailable). Never counts as a pass."""
    log.warning("verify_speech.unverified", shot_id=shot_id, attempt=attempt, reason=reason)
    return SpeechVerification(
        shot_id=shot_id,
        attempt=attempt,
        expected=expected,
        transcript="",
        wer=1.0,
        passed=False,
        verified=False,
        failure_kind=FailureKind.UNKNOWN,
    )


def classify_failure(
    expected: str, transcript: str, language: str, deps: VerifyDeps
) -> tuple[FailureKind, list[Classification]]:
    """Pick the retry tactic. Without a usable judge answer, fall back to simple heuristics.

    Returns the kind and every answer behind it: Jev's (when asked) and the heuristic's (when it
    decided), so both land in `judge_decisions`.
    """
    asked: list[Classification] = []
    judge_on = (
        deps.judge_config.enabled
        and deps.judge_config.failure_classifier.enabled
        and deps.judge.enabled
    )
    if judge_on:
        state = (
            f"Expected line ({language}): {expected!r}\nSTT transcript: {transcript!r}\n"
            f"Expected words: {len(expected.split())}, transcript words: {len(transcript.split())}"
        )
        answer = deps.judge.ask_choice(
            state,
            f"{Q_FAILURE_KIND}: Why does the transcript differ from the expected line? "
            "paraphrased = same meaning, different words; truncated = the transcript stops early "
            "or drops words; extra_words = the expected line is there plus additional speech; "
            "wrong_language = spoken in another language; stt_noise = tiny spelling/accent "
            "differences the transcriber likely caused; silence = nothing intelligible was said.",
            [k.value for k in _CLASSIFIABLE],
        )
        if answer.available:
            try:
                return FailureKind(answer.answer), [Classification(answer, DECISION_CLASSIFIED)]
            except ValueError:
                log.warning("verify_speech.judge_unparsable", answer=answer.answer)
        else:
            log.warning("verify_speech.judge_unavailable", fallback=DECISION_HEURISTIC)
        asked.append(Classification(answer, DECISION_UNAVAILABLE))
    kind = _heuristic_failure_kind(expected, transcript)
    heuristic = JudgeAnswer(
        question=Q_FAILURE_KIND,
        answer=kind.value,
        model_version=HEURISTIC_MODEL,
        available=False,  # no judge answered; the cost is zero
    )
    return kind, [*asked, Classification(heuristic, DECISION_HEURISTIC)]


def _transcribe(
    media_path: Path,
    *,
    language: str,
    deps: VerifyDeps,
    work_dir: Path | None,
    window: tuple[float, float] | None,
) -> STTResult:
    wav = (work_dir or media_path.parent) / f"{media_path.stem}.stt.wav"
    start_s, duration_s = window if window else (None, None)
    deps.ffmpeg.extract_audio(media_path, wav, start_s=start_s, duration_s=duration_s)
    _write_segment(wav, media_path, start_s=start_s, duration_s=duration_s)
    stt_language = language if deps.stt_config.language == "auto" else deps.stt_config.language
    return deps.stt.transcribe(
        STTRequest(
            model=deps.stt_config.model,
            audio_path=wav,
            language=stt_language,
            source_path=media_path,
        )
    )


def _write_segment(
    wav: Path, source: Path, *, start_s: float | None, duration_s: float | None
) -> None:
    """Provenance of the STT input: which media and which segment of it the WAV holds (kept in
    the work dir for debugging; the dry-run `FakeSTT` reads it to honour the window)."""
    segment = {"source": str(source), "start_s": start_s or 0.0, "duration_s": duration_s}
    wav.with_name(wav.name + ".json").write_text(json.dumps(segment), encoding="utf-8")


def _heuristic_failure_kind(expected: str, transcript: str) -> FailureKind:
    exp_words, got_words = expected.split(), transcript.split()
    if not got_words:
        return FailureKind.SILENCE
    if len(got_words) < len(exp_words) * _TRUNCATED_RATIO:
        return FailureKind.TRUNCATED
    if len(got_words) > len(exp_words) * _EXTRA_RATIO:
        return FailureKind.EXTRA_WORDS
    return FailureKind.UNKNOWN
