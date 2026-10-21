"""Cross-checks between the LLM script and the user's input (ADR-005)."""

from __future__ import annotations

from dataclasses import dataclass

from holywater.domain.idea import IdeaInput
from holywater.domain.script import Script
from holywater.quality.text_norm import find_verbatim_span


@dataclass(frozen=True)
class ScriptCheck:
    script: Script
    violations: list[str]

    @property
    def ok(self) -> bool:
        return not self.violations


def check_script_against_idea(script: Script, idea: IdeaInput) -> ScriptCheck:
    """Replace the LLM's line with the verbatim span from the input, or report why it can't.

    The returned script (when `ok`) carries the original characters the user typed, so nothing
    downstream ever sees the model's spelling of the line.
    """
    match = find_verbatim_span(idea.text, script.dialogue.text, script.dialogue.language)
    if match is None:
        return ScriptCheck(
            script=script,
            violations=[
                "dialogue.text must be a verbatim fragment of the input idea; "
                f"got {script.dialogue.text!r}, which does not occur in the input"
            ],
        )
    return ScriptCheck(script=script.with_dialogue_text(match.text), violations=[])
