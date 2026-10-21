"""Stage 1b: cheap typed checks (Jev) before the first paid video call (ADR-007)."""

from __future__ import annotations

from dataclasses import dataclass, field

from holywater.config.schema import JudgeConfig
from holywater.domain.script import Script
from holywater.observability.logging import get_logger
from holywater.providers.base import JudgeAnswer, JudgeProvider

log = get_logger(__name__)

Q_SAFETY = "safety_risk"
Q_FILMABILITY = "filmability"
Q_LINE_FITS = "line_fits_window"

SCORE_LEVELS = 5
_LINE_FITS_MIN_P = 0.5


@dataclass
class PreflightDeps:
    judge: JudgeProvider
    config: JudgeConfig


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    skipped: bool
    issues: list[str] = field(default_factory=list)
    decisions: list[JudgeAnswer] = field(default_factory=list)


def preflight_script(script: Script, deps: PreflightDeps) -> PreflightResult:
    """Return issues phrased as feedback for the script writer; empty issues means go."""
    if not (deps.config.enabled and deps.config.preflight.enabled and deps.judge.enabled):
        log.info("preflight.skipped", reason="judge disabled")
        return PreflightResult(ok=True, skipped=True)

    issues: list[str] = []
    decisions: list[JudgeAnswer] = []
    for shot in script.shots:
        state = (
            f"Shot {shot.id} prompt for a text-to-video model.\n"
            f"Visual: {shot.visual_prompt}\nCamera: {shot.camera}\n"
            f"Continuity: {shot.continuity}\nStyle: {script.style}"
        )
        safety = deps.judge.ask_bool(
            state,
            f"{Q_SAFETY}: The prompt describes content likely to be rejected by a commercial "
            "video model's safety filter (graphic violence, gore, sexual content, real public "
            "figures, hate symbols, minors in unsafe situations).",
        )
        decisions.append(safety)
        if safety.available and safety.probabilities.get("yes", 0.0) > (
            deps.config.preflight.safety_risk_threshold
        ):
            issues.append(
                f"shot {shot.id}: the visual prompt is likely to trip the video model's safety "
                f"filter (p={safety.probabilities['yes']:.2f}); rewrite it to be clearly safe"
            )

        filmability = deps.judge.ask_score(
            state,
            f"{Q_FILMABILITY}: How concrete and filmable is this prompt as a single continuous "
            "shot: 1 = vague or several scenes/actions, 5 = one character, one action, one "
            "location, clearly described.",
            levels=SCORE_LEVELS,
        )
        decisions.append(filmability)
        if filmability.available and _as_int(filmability.answer) < (
            deps.config.preflight.min_filmability_score
        ):
            issues.append(
                f"shot {shot.id}: the visual prompt scored {filmability.answer}/{SCORE_LEVELS} "
                "for filmability; make it one character, one action, one location, concretely"
            )

    d = script.dialogue
    fits = deps.judge.ask_bool(
        f'Line: "{d.text}" ({len(d.text)} characters, language {d.language}).\n'
        f"Time window: {d.window_s:.1f} seconds. Delivery: {d.emotion}.",
        f"{Q_LINE_FITS}: The line can be spoken naturally within the time window.",
    )
    decisions.append(fits)
    if fits.available and fits.probabilities.get("yes", 1.0) < _LINE_FITS_MIN_P:
        issues.append(
            f"the dialogue window ({d.window_s:.1f}s) is too short for the line; widen it or "
            "lengthen the shot"
        )

    log.info("preflight.done", issues=len(issues), decisions=len(decisions))
    return PreflightResult(ok=not issues, skipped=False, issues=issues, decisions=decisions)


def _as_int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return SCORE_LEVELS  # unparsable answer must not block generation
