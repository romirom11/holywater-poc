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
    input_tokens: int = 0
    output_tokens: int = 0


def write_script(
    idea: IdeaInput, deps: ScriptWriterDeps, *, feedback: list[str] | None = None
) -> ScriptResult:
    """Ask the LLM for a script; feed validation errors back until it fits or attempts run out."""
    pending = list(feedback or [])
    total_cost = 0.0
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
                input_tokens=tokens_in,
                output_tokens=tokens_out,
            )
        pending = check.violations
        log.warning("write_script.rejected", attempt=attempt, violations=pending)

    msg = f"could not produce a valid script after {deps.config.max_attempts} attempts"
    raise ScriptError(msg, stage="write_script", context={"last_feedback": pending})


def _feedback_block(feedback: list[str]) -> str:
    if not feedback:
        return ""
    bullets = "\n".join(f"- {item}" for item in feedback)
    return (
        "Your previous answer was rejected for the following reasons. Fix every one of them:\n"
        f"{bullets}"
    )
