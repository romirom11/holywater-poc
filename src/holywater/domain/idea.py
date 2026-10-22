"""The user's idea: 1-3 sentences, the single source of truth for the verbatim line."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import PydanticCustomError

MIN_IDEA_CHARS = 10
MAX_IDEA_CHARS = 1000
# The task asks for 1-3 sentences. More is a warning for the CLI, not a validation error.
MAX_IDEA_SENTENCES = 3

# Abbreviations that are followed by a capitalized word without ending the sentence
# ("Dr. Smith", "вул. Хрещатик"). Words that often end a sentence ("etc.", "грн.") are left out.
_ABBREVIATIONS = frozenset(
    {
        *("mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "e.g", "i.e", "approx"),
        *("вул", "просп", "пл", "ім", "д-р", "проф", "див", "напр", "т.д", "т.п"),
    }
)
_SENTENCE_END_RE = re.compile(r"[.!?…]+[\"'”’»“)\]]*(?=\s|$)")
_WORD_BEFORE_RE = re.compile(r"([^\W\d_][\w.\-]*)$")
_CLOSERS = "\"'”’»“)]"


class IdeaInput(BaseModel):
    """The user's idea. `text` is the single source of truth for the verbatim line."""

    model_config = ConfigDict(extra="forbid")

    text: str
    language: str | None = Field(default=None, description="BCP-47 hint, e.g. 'uk' or 'en'.")
    received_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("text")
    @classmethod
    def _strip_and_check_length(cls, value: str) -> str:
        value = value.strip()
        if len(value) < MIN_IDEA_CHARS:
            raise PydanticCustomError(
                "idea_too_short",
                "The idea is too short ({length} characters); write at least {minimum}: "
                "1-3 sentences with the spoken line in quotes.",
                {"length": len(value), "minimum": MIN_IDEA_CHARS},
            )
        if len(value) > MAX_IDEA_CHARS:
            raise PydanticCustomError(
                "idea_too_long",
                "The idea is too long ({length} characters); keep it within {maximum}: "
                "1-3 sentences with the spoken line in quotes.",
                {"length": len(value), "maximum": MAX_IDEA_CHARS},
            )
        return value

    @property
    def sentence_count(self) -> int:
        """Approximate number of sentences (the CLI warns above MAX_IDEA_SENTENCES)."""
        return count_sentences(self.text)


def count_sentences(text: str) -> int:
    """Count sentences with a simple splitter that skips abbreviations, initials and decimals.

    A sentence ends at . ! ? or … (optionally followed by closing quotes) when whitespace or the
    end of the text follows and the next word does not start in lower case.
    """
    count = 0
    start = 0
    for m in _SENTENCE_END_RE.finditer(text):
        if not _ends_sentence(text, m):
            continue
        if any(ch.isalnum() for ch in text[start : m.start()]):
            count += 1
        start = m.end()
    if any(ch.isalnum() for ch in text[start:]):
        count += 1
    return count


def _ends_sentence(text: str, m: re.Match[str]) -> bool:
    rest = text[m.end() :].lstrip()
    if rest and rest[0].islower():
        return False  # "etc. and", "Wait... no", '"Run!" he shouted'
    if m.group(0).rstrip(_CLOSERS) != ".":
        return True
    before = _WORD_BEFORE_RE.search(text[: m.start()])
    if before is None:
        return True
    word = before.group(1)
    if word.lower() in _ABBREVIATIONS:
        return False
    return not (len(word) == 1 and word.isupper())  # an initial: "J. K. Rowling"
