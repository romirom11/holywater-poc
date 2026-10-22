"""Stage 1: idea -> Script via the LLM, with a repair loop and the verbatim-line check."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from holywater.config.schema import LLMConfig
from holywater.domain.idea import IdeaInput
from holywater.domain.script import Script
from holywater.observability.errors import ScriptError
from holywater.observability.logging import get_logger
from holywater.prompts import PromptTemplate, load_prompt
from holywater.providers.base import LLMProvider
from holywater.quality.script_checks import check_script_against_idea

log = get_logger(__name__)


@dataclass
class ScriptWriterDeps:
    llm: LLMProvider
    config: LLMConfig
    prompt: PromptTemplate = field(default_factory=lambda: load_prompt("scriptwriter"))


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
    idea: IdeaInput, deps: ScriptWriterDeps, *, feedback: list[str] | None = None
) -> ScriptResult:
    """Ask the LLM for a script; feed validation errors back until it fits or attempts run out."""
    pending = list(feedback or [])
    total_cost = 0.0
    any_estimate = False
    tokens_in = tokens_out = 0
    for attempt in range(1, deps.config.max_attempts + 1):
        user = deps.prompt.render(
            "user", idea_text=idea.text, feedback_block=_feedback_block(pending)
        )
        log.info("write_script.attempt", attempt=attempt, feedback=len(pending))
        try:
            result = deps.llm.complete_structured(
                model=deps.config.model,
                system=deps.prompt.render("system"),
                user=user,
                schema=Script,
                temperature=deps.config.temperature,
            )
        except (ValidationError, ValueError) as exc:
            pending = [f"Your previous answer did not match the schema: {exc}"]
            log.warning("write_script.invalid_output", attempt=attempt, error=str(exc)[:500])
            continue
        total_cost += result.cost_usd
        any_estimate = any_estimate or result.cost_is_estimate
        tokens_in += result.input_tokens
        tokens_out += result.output_tokens

        check = check_script_against_idea(result.data, idea)
        if check.ok:
            log.info(
                "write_script.ok",
                attempt=attempt,
                shots=len(check.script.shots),
                total_s=check.script.total_duration_s,
                line=check.script.dialogue.text,
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
    raise ScriptError(msg, stage="write_script", context={"last_feedback": pending})


def soften_shot(
    script: Script, shot_id: int, *, reason: str, idea: IdeaInput, deps: ScriptWriterDeps
) -> ScriptResult:
    """One LLM rewrite of a shot the video model rejected (content policy / bad input).

    Only that shot is taken from the new answer; the rest of the script — including the verbatim
    line — is kept, and the merged script is re-validated.
    """
    feedback = deps.prompt.render(
        "soften",
        shot_id=shot_id,
        reason=reason.replace('"', "'")[:300],
        previous=script.model_dump_json(indent=2),
    )
    log.warning("write_script.soften", shot_id=shot_id, reason=reason[:300])
    result = write_script(idea, deps, feedback=[feedback])
    new_shot = next((s for s in result.script.shots if s.id == shot_id), None)
    if new_shot is None:
        msg = f"softened script has no shot {shot_id}"
        raise ScriptError(msg, stage="write_script", context={"shot_id": shot_id})
    try:
        merged = script.with_shot(new_shot)
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


def _feedback_block(feedback: list[str]) -> str:
    if not feedback:
        return ""
    bullets = "\n".join(f"- {item}" for item in feedback)
    return (
        "Your previous answer was rejected for the following reasons. Fix every one of them:\n"
        f"{bullets}"
    )
