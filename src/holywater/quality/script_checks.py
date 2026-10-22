"""Cross-checks between the LLM script and the user's input (ADR-005).

Violations are phrased as instructions: the script writer feeds them back to the LLM on retry.
"""

from __future__ import annotations

from dataclasses import dataclass

from holywater.domain.idea import IdeaInput
from holywater.domain.script import CHARS_PER_SECOND, Dialogue, Script
from holywater.observability.logging import get_logger
from holywater.quality.text_norm import (
    VerbatimMatch,
    find_verbatim_span,
    normalize,
    quoted_spans,
    same_words,
)

log = get_logger(__name__)


@dataclass(frozen=True)
class ScriptCheck:
    script: Script
    violations: list[str]

    @property
    def ok(self) -> bool:
        return not self.violations


def check_script_against_idea(
    script: Script, idea: IdeaInput, *, chars_per_s: float = CHARS_PER_SECOND
) -> ScriptCheck:
    """Replace the LLM's line with the verbatim span from the input, or report why it can't.

    When the idea quotes something, the line must be one complete quoted fragment: a partial
    quote, a quote plus narration, or narration instead of the quote is a violation. A line the
    quote parser missed (a stray mark such as an inch sign) still passes when it sits in the idea
    directly between an opening and a closing quote mark. Without quotes any verbatim fragment of
    whole words is accepted. The returned script (when `ok`) carries the original characters the
    user typed, so nothing downstream sees the model's spelling of the line. `chars_per_s` is the
    speech rate used to check that the line fits its window.
    """
    line = script.dialogue.text
    language = script.dialogue.language
    quotes = [q for q in quoted_spans(idea.text) if normalize(q.text, language)]
    if quotes:
        match = _match_quoted(line, quotes, language) or _between_quote_marks(
            idea.text, line, language, quotes
        )
        if match is None:
            allowed = " or ".join(repr(q.text) for q in quotes)
            return ScriptCheck(
                script=script,
                violations=[
                    "dialogue.text must be exactly one complete quoted line from the input, "
                    "copied verbatim: not shortened, not merged with narration and without quote "
                    f"marks. Allowed: {allowed}; got {line!r}"
                ],
            )
    else:
        match = find_verbatim_span(idea.text, line, language)
        if match is None:
            return ScriptCheck(
                script=script,
                violations=[
                    "dialogue.text must be a verbatim fragment of the input idea; "
                    f"got {line!r}, which does not occur in the input"
                ],
            )
    checked = script.with_dialogue_text(match.text)
    return ScriptCheck(
        script=checked, violations=window_violations(checked.dialogue, chars_per_s=chars_per_s)
    )


def window_violations(dialogue: Dialogue, *, chars_per_s: float = CHARS_PER_SECOND) -> list[str]:
    """The line's time window must be long enough to say it at `chars_per_s`."""
    if chars_per_s <= 0:
        msg = f"chars_per_s must be positive, got {chars_per_s}"
        raise ValueError(msg)
    needed_s = len(dialogue.text) / chars_per_s
    if dialogue.window_s >= needed_s:
        return []
    return [
        f"dialogue window {dialogue.window_s:.1f}s is shorter than the estimated speech time "
        f"{needed_s:.1f}s for {len(dialogue.text)} characters at {chars_per_s:g} chars/s; "
        "widen start_s..end_s or lengthen the dialogue shot, never shorten the line"
    ]


def _match_quoted(line: str, quotes: list[VerbatimMatch], language: str) -> VerbatimMatch | None:
    for index, quote in enumerate(quotes):
        if not same_words(quote.text, line, language):
            continue
        inner = find_verbatim_span(quote.text, line, language)
        if inner is None:  # equal normalized text always maps back; guard for type narrowing
            continue
        if len(quotes) > 1:
            log.info("script_check.quote_chosen", index=index, quotes=len(quotes))
        return VerbatimMatch(
            text=inner.text, start=quote.start + inner.start, end=quote.start + inner.end
        )
    return None


_OPENING_MARKS = frozenset("\"“„«‘'")
_CLOSING_MARKS = frozenset("\"”“»’'")


def _between_quote_marks(
    idea: str, line: str, language: str, quotes: list[VerbatimMatch]
) -> VerbatimMatch | None:
    """The verbatim span of `line` when the idea has a quote mark right before it and right
    after it (whitespace and terminal punctuation aside) and it overlaps no parsed quote: a
    complete quote the parser missed. A fragment of a parsed quote never passes this way."""
    match = find_verbatim_span(idea, line, language)
    if match is None or any(q.start < match.end and match.start < q.end for q in quotes):
        return None
    before = idea[: match.start].rstrip()
    after = idea[match.end :].lstrip().lstrip(".!?…,")
    if before[-1:] in _OPENING_MARKS and after[:1] in _CLOSING_MARKS:
        log.info("script_check.quote_by_marks", start=match.start, end=match.end)
        return match
    return None
