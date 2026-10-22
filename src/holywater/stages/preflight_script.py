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

# `judge_decisions.decision` values written for pre-flight answers.
DECISION_PASS = "pass"
DECISION_REWRITE = "rewrite"
DECISION_UNAVAILABLE = "unavailable"

SCORE_LEVELS = 5
_LINE_FITS_MIN_P = 0.5


@dataclass
class PreflightDeps:
    judge: JudgeProvider
    config: JudgeConfig


@dataclass(frozen=True)
class PreflightCheck:
    """One judge question about one shot (or the line, `shot_id=None`) and what it decided."""

    question: str
    answer: JudgeAnswer
    shot_id: int | None
    usable: bool  # the judge answered and the answer could be read
    flagged: bool  # the answer asks for a rewrite

    @property
    def decision(self) -> str:
        if not self.usable:
            return DECISION_UNAVAILABLE
        return DECISION_REWRITE if self.flagged else DECISION_PASS


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    skipped: bool
    issues: list[str] = field(default_factory=list)
    checks: list[PreflightCheck] = field(default_factory=list)
    # Why nothing was asked ("disabled" in config, or the judge that would answer, e.g. "noop").
    skip_reason: str | None = None

    @property
    def decisions(self) -> list[JudgeAnswer]:
        return [c.answer for c in self.checks]


def preflight_script(script: Script, deps: PreflightDeps) -> PreflightResult:
    """Return issues phrased as feedback for the script writer; empty issues means go."""
    if not (deps.config.enabled and deps.config.preflight.enabled):
        log.info("preflight.skipped", reason="disabled in config")
        return PreflightResult(ok=True, skipped=True, skip_reason="disabled")
    if not deps.judge.enabled:
        log.warning("preflight.skipped", reason="judge unavailable", judge=deps.judge.model)
        return PreflightResult(ok=True, skipped=True, skip_reason=deps.judge.model)

    issues: list[str] = []
    checks: list[PreflightCheck] = []
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
        risky = safety.available and safety.probabilities.get("yes", 0.0) > (
            deps.config.preflight.safety_risk_threshold
        )
        if risky:
            issues.append(
                f"shot {shot.id}: the visual prompt is likely to trip the video model's safety "
                f"filter (p={safety.probabilities['yes']:.2f}); rewrite it to be clearly safe"
            )
        checks.append(PreflightCheck(Q_SAFETY, safety, shot.id, safety.available, risky))

        filmability = deps.judge.ask_score(
            state,
            f"{Q_FILMABILITY}: How concrete and filmable is this prompt as a single continuous "
            "shot: 1 = vague or several scenes/actions, 5 = one character, one action, one "
            "location, clearly described.",
            levels=SCORE_LEVELS,
        )
        score = _score(filmability, shot_id=shot.id)
        vague = score is not None and score < deps.config.preflight.min_filmability_score
        if vague:
            issues.append(
                f"shot {shot.id}: the visual prompt scored {score}/{SCORE_LEVELS} "
                "for filmability; make it one character, one action, one location, concretely"
            )
        checks.append(PreflightCheck(Q_FILMABILITY, filmability, shot.id, score is not None, vague))

    d = script.dialogue
    fits = deps.judge.ask_bool(
        f'Line: "{d.text}" ({len(d.text)} characters, language {d.language}).\n'
        f"Time window: {d.window_s:.1f} seconds. Delivery: {d.emotion}.",
        f"{Q_LINE_FITS}: The line can be spoken naturally within the time window.",
    )
    too_short = fits.available and fits.probabilities.get("yes", 1.0) < _LINE_FITS_MIN_P
    if too_short:
        issues.append(
            f"the dialogue window ({d.window_s:.1f}s) is too short for the line; widen it or "
            "lengthen the shot"
        )
    checks.append(PreflightCheck(Q_LINE_FITS, fits, None, fits.available, too_short))

    if not any(c.usable for c in checks):
        log.warning("preflight.unavailable", reason="judge answered nothing; continuing without it")
        return PreflightResult(ok=True, skipped=True, checks=checks, skip_reason="unavailable")
    log.info("preflight.done", issues=len(issues), decisions=len(checks))
    return PreflightResult(ok=not issues, skipped=False, issues=issues, checks=checks)


def _score(answer: JudgeAnswer, *, shot_id: int) -> int | None:
    """The 1..levels score, or None when the judge gave none: an unreadable score is treated as
    no answer (it neither blocks generation nor counts as a pass)."""
    if not answer.available:
        return None
    try:
        return int(answer.answer)
    except ValueError:
        log.warning(
            "preflight.unparsable_score",
            shot_id=shot_id,
            question=Q_FILMABILITY,
            answer=answer.answer,
        )
        return None
