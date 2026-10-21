"""Stage 4: STT on the dialogue clip, WER gate, and (optionally) failure classification by Jev."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from holywater.config.schema import JudgeConfig, QualityConfig
from holywater.domain.artifacts import SpeechVerification
from holywater.domain.enums import FailureKind
from holywater.observability.logging import get_logger
from holywater.providers.base import JudgeAnswer, JudgeProvider, STTProvider
from holywater.quality.gates import SpeechGate

log = get_logger(__name__)

Q_FAILURE_KIND = "failure_kind"
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
    quality: QualityConfig
    judge_config: JudgeConfig


@dataclass(frozen=True)
class VerifyResult:
    verification: SpeechVerification
    decision: JudgeAnswer | None = None
    cost_usd: float = 0.0


def verify_speech(
    media_path: Path,
    *,
    shot_id: int,
    attempt: int,
    expected: str,
    language: str,
    deps: VerifyDeps,
) -> VerifyResult:
    """Transcribe and compare. Provider errors propagate (the flow decides on retries)."""
    stt = deps.stt.transcribe(media_path, language=language)
    result = SpeechGate(max_wer=deps.quality.max_wer).check(expected, stt.text, language)
    log.info(
        "verify_speech.result",
        shot_id=shot_id,
        attempt=attempt,
        wer=result.measured,
        passed=result.passed,
        transcript=stt.text,
    )
    failure_kind: FailureKind | None = None
    decision: JudgeAnswer | None = None
    if not result.passed:
        failure_kind, decision = classify_failure(expected, stt.text, language, deps)
        log.warning(
            "verify_speech.failed",
            shot_id=shot_id,
            attempt=attempt,
            wer=result.measured,
            failure_kind=failure_kind.value,
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
    )
    return VerifyResult(verification=verification, decision=decision, cost_usd=stt.cost_usd)


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
) -> tuple[FailureKind, JudgeAnswer | None]:
    """Pick the retry tactic. Without a judge, fall back to simple heuristics."""
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
                return FailureKind(answer.answer), answer
            except ValueError:
                log.warning("verify_speech.judge_unparsable", answer=answer.answer)
    return _heuristic_failure_kind(expected, transcript), None


def _heuristic_failure_kind(expected: str, transcript: str) -> FailureKind:
    exp_words, got_words = expected.split(), transcript.split()
    if not got_words:
        return FailureKind.SILENCE
    if len(got_words) < len(exp_words) * _TRUNCATED_RATIO:
        return FailureKind.TRUNCATED
    if len(got_words) > len(exp_words) * _EXTRA_RATIO:
        return FailureKind.EXTRA_WORDS
    return FailureKind.UNKNOWN
