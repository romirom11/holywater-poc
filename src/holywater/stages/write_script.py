"""Stage 1: idea -> Script via the LLM, with a repair loop and the verbatim-line check."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import ValidationError

from holywater.config.schema import LLMConfig
from holywater.domain.artifacts import ProviderCost
from holywater.domain.idea import IdeaInput
from holywater.domain.script import CHARS_PER_SECOND, Script
from holywater.observability.errors import ProviderError, ScriptError, StructuredOutputError
from holywater.observability.logging import get_logger
from holywater.prompts import PromptTemplate, load_prompt
from holywater.providers.base import LLMProvider
from holywater.quality.script_checks import check_script_against_idea

log = get_logger(__name__)

# `ProviderCost.capability` of an LLM call (the flow maps it to `llm.provider`).
LLM_CAPABILITY = "llm"
# What a softened shot may change: its wording, not its timing (the dialogue window, a TTS voice
# fitted to the shot and the other shot's length all depend on `duration_s`).
_SOFTENED_FIELDS = ("visual_prompt", "camera", "continuity", "ambient_audio")


@dataclass
class ScriptWriterDeps:
    llm: LLMProvider
    config: LLMConfig
    prompt: PromptTemplate = field(default_factory=lambda: load_prompt("scriptwriter"))
    # Speaking rate the dialogue window must allow (config `quality.speech_chars_per_s`).
    chars_per_s: float = CHARS_PER_SECOND
    # Called once per billed LLM call — including answers that fail validation and failed calls
    # that carry `spent_usd` — as soon as the charge is known, so the flow records every one of
    # them even when the stage ends in `ScriptError` or a retried provider error.
    on_cost: Callable[[ProviderCost], None] | None = None


@dataclass(frozen=True)
class ScriptResult:
    script: Script
    attempts: int
    model: str
    prompt_version: str
    cost_usd: float = 0.0
    cost_is_estimate: bool = True
    input_tokens: int = 0
    output_tokens: int = 0


def write_script(
    idea: IdeaInput,
    deps: ScriptWriterDeps,
    *,
    feedback: list[str] | None = None,
    revision: str | None = None,
) -> ScriptResult:
    """Ask the LLM for a script; feed validation errors back until it fits or attempts run out.

    `revision` is a rendered `soften`/`rewrite` section (the previous script plus what to change);
    it is sent with every attempt, while `feedback` holds the reasons the last answer was rejected.
    """
    pending = list(feedback or [])
    total_cost = 0.0
    any_estimate = False
    tokens_in = tokens_out = 0
    system = deps.prompt.render("system", chars_per_s=f"{deps.chars_per_s:g}")
    for attempt in range(1, deps.config.max_attempts + 1):
        user = deps.prompt.render(
            "user", idea_text=idea.text, feedback_block=_feedback_block(revision, pending)
        )
        log.info(
            "write_script.attempt", attempt=attempt, feedback=len(pending), revision=bool(revision)
        )
        try:
            result = deps.llm.complete_structured(
                model=deps.config.model,
                system=system,
                user=user,
                schema=Script,
                temperature=deps.config.temperature,
            )
        except (ValidationError, ValueError) as exc:
            if isinstance(exc, StructuredOutputError) and exc.spent_usd is not None:
                # The unusable answer was billed: report it now and count it in the total.
                _report(deps, exc.model or deps.config.model, exc.spent_usd, exc.spent_is_estimate)
                total_cost += exc.spent_usd
                any_estimate = any_estimate or exc.spent_is_estimate
            pending = [_invalid_output_feedback(exc)]
            log.warning("write_script.invalid_output", attempt=attempt, error=str(exc)[:500])
            continue
        except ProviderError as exc:
            if exc.spent_usd is not None:  # billed although it failed; the task may retry
                _report(deps, exc.model or deps.config.model, exc.spent_usd, exc.spent_is_estimate)
            raise
        _report(
            deps,
            result.model,
            result.cost_usd,
            result.cost_is_estimate,
            units=result.input_tokens + result.output_tokens,
        )
        total_cost += result.cost_usd
        any_estimate = any_estimate or result.cost_is_estimate
        tokens_in += result.input_tokens
        tokens_out += result.output_tokens

        check = check_script_against_idea(result.data, idea, chars_per_s=deps.chars_per_s)
        if check.ok:
            log.info(
                "write_script.ok",
                attempt=attempt,
                shots=len(check.script.shots),
                total_s=check.script.total_duration_s,
                line=check.script.dialogue.text,
                voice_gender=check.script.dialogue.voice_gender.value,
            )
            return ScriptResult(
                script=check.script,
                attempts=attempt,
                model=result.model,
                prompt_version=deps.prompt.version_tag,
                cost_usd=total_cost,
                cost_is_estimate=any_estimate,
                input_tokens=tokens_in,
                output_tokens=tokens_out,
            )
        pending = check.violations
        log.warning("write_script.rejected", attempt=attempt, violations=pending)

    msg = f"could not produce a valid script after {deps.config.max_attempts} attempts"
    raise ScriptError(
        msg,
        stage="write_script",
        context={
            "last_feedback": pending,
            "attempts": deps.config.max_attempts,
            "cost_usd": round(total_cost, 6),
            "prompt_version": deps.prompt.version_tag,
        },
    )


def rewrite_script(
    script: Script, issues: list[str], *, idea: IdeaInput, deps: ScriptWriterDeps
) -> ScriptResult:
    """A pre-flight rewrite: the LLM gets its previous script plus the judge's issues and revises
    it (instead of writing a new script from scratch). The line is re-checked as always."""
    revision = deps.prompt.render(
        "rewrite",
        issues="\n".join(f"- {issue}" for issue in issues),
        previous=script.model_dump_json(indent=2),
    )
    log.warning("write_script.rewrite", issues=issues)
    return write_script(idea, deps, revision=revision)


def soften_shot(
    script: Script, shot_id: int, *, reason: str, idea: IdeaInput, deps: ScriptWriterDeps
) -> ScriptResult:
    """One LLM rewrite of a shot the video model rejected (content policy / bad input).

    Only that shot's wording (visual prompt, camera, continuity, ambience) is taken from the new
    answer; its timing and the rest of the script — including the verbatim line — are kept, and
    the merged script is re-validated.
    """
    revision = deps.prompt.render(
        "soften",
        shot_id=shot_id,
        reason=reason.replace('"', "'")[:300],
        previous=script.model_dump_json(indent=2),
    )
    log.warning("write_script.soften", shot_id=shot_id, reason=reason[:300])
    result = write_script(idea, deps, revision=revision)
    new_shot = next((s for s in result.script.shots if s.id == shot_id), None)
    old_shot = next((s for s in script.shots if s.id == shot_id), None)
    if new_shot is None or old_shot is None:
        msg = f"softened script has no shot {shot_id}"
        raise ScriptError(msg, stage="write_script", context={"shot_id": shot_id})
    softened = old_shot.model_copy(
        update={name: getattr(new_shot, name) for name in _SOFTENED_FIELDS}
    )
    try:
        merged = script.with_shot(softened)
    except ValueError as exc:
        msg = f"softened shot {shot_id} breaks the script invariants: {exc}"
        raise ScriptError(msg, stage="write_script", context={"shot_id": shot_id}) from exc
    return ScriptResult(
        script=merged,
        attempts=result.attempts,
        model=result.model,
        prompt_version=result.prompt_version,
        cost_usd=result.cost_usd,
        cost_is_estimate=result.cost_is_estimate,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


def _report(
    deps: ScriptWriterDeps, model: str, cost_usd: float, is_estimate: bool, *, units: float = 0
) -> None:
    if deps.on_cost is None:
        return
    deps.on_cost(
        ProviderCost(
            capability=LLM_CAPABILITY,
            model=model,
            units=units,
            cost_usd=cost_usd,
            is_estimate=is_estimate,
        )
    )


def _invalid_output_feedback(exc: ValueError) -> str:
    if isinstance(exc, StructuredOutputError) and exc.finish_reason == "length":
        return (
            "Your previous answer was cut off before the JSON was complete; keep every field "
            "concise so the whole script fits."
        )
    return f"Your previous answer did not match the schema: {exc}"


def _feedback_block(revision: str | None, feedback: list[str]) -> str:
    parts = [revision] if revision else []
    if feedback:
        bullets = "\n".join(f"- {item}" for item in feedback)
        parts.append(
            "Your previous answer was rejected for the following reasons. Fix every one of "
            f"them:\n{bullets}"
        )
    return "\n\n".join(parts)
