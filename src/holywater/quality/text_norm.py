"""Text normalization and WER for the verbatim-line gate (R4).

Normalization is deliberately lossy: case, punctuation, quote style and digit/word spelling of
numbers must not count as a mismatch between the input line and an STT transcript.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import jiwer
from num2words import num2words

_APOSTROPHES = "’ʼ‘`´"
_QUOTE_PAIRS: tuple[tuple[str, str], ...] = (
    ("“", "”"),
    ("„", "“"),
    ("«", "»"),
    ('"', '"'),
    ("‘", "’"),
    ("'", "'"),
)
_WORD_RE = re.compile(r"\S+")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_STRIP_RE = re.compile(r"[^\w'\s-]", flags=re.UNICODE)
_LANG_FALLBACK = "en"


def normalize(text: str, language: str = _LANG_FALLBACK) -> str:
    """Lowercase, unify apostrophes, spell out numbers, drop punctuation, collapse spaces."""
    text = unicodedata.normalize("NFKC", text)
    for ch in _APOSTROPHES:
        text = text.replace(ch, "'")
    text = _NUMBER_RE.sub(lambda m: _number_to_words(m.group(0), language), text)
    text = _STRIP_RE.sub(" ", text.lower())
    text = text.replace("-", " ")
    text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)  # quotes used as quotation marks, not apostrophes
    return " ".join(text.split())


def _number_to_words(token: str, language: str) -> str:
    value = token.replace(",", ".")
    try:
        number: float | int = float(value) if "." in value else int(value)
        return str(num2words(number, lang=language))
    except NotImplementedError:
        try:
            return str(num2words(number, lang=_LANG_FALLBACK))
        except (NotImplementedError, ValueError):
            return token
    except ValueError:
        return token


def wer(expected: str, actual: str, language: str = _LANG_FALLBACK) -> float:
    """Word error rate on normalized text. 0.0 = identical, 1.0 = everything wrong."""
    ref = normalize(expected, language)
    hyp = normalize(actual, language)
    if not ref:
        return 0.0 if not hyp else 1.0
    if not hyp:
        return 1.0
    return float(jiwer.wer(ref, hyp))


@dataclass(frozen=True)
class VerbatimMatch:
    """A span of the original input text that matches a candidate line after normalization."""

    text: str
    start: int
    end: int


def find_verbatim_span(
    source: str, candidate: str, language: str = _LANG_FALLBACK
) -> VerbatimMatch | None:
    """Locate `candidate` inside `source` ignoring normalization differences.

    Returns the *original* characters from `source` so the line downstream is exactly what the
    user typed (ADR-005). Surrounding quotation marks are not part of the returned span.
    """
    cand_tokens = normalize(candidate, language).split()
    if not cand_tokens:
        return None
    src_words = [(m.group(0), m.start(), m.end()) for m in _WORD_RE.finditer(source)]
    # A source word may normalize to several tokens (numbers) or none (pure punctuation).
    expanded: list[tuple[str, int, int]] = []
    for word, start, end in src_words:
        for tok in normalize(word, language).split():
            expanded.append((tok, start, end))
    src_tokens = [tok for tok, _, _ in expanded]
    n = len(cand_tokens)
    for i in range(len(src_tokens) - n + 1):
        if src_tokens[i : i + n] == cand_tokens:
            start = expanded[i][1]
            end = expanded[i + n - 1][2]
            return _trim_quotes(source, start, end)
    return None


def _trim_quotes(source: str, start: int, end: int) -> VerbatimMatch:
    span = source[start:end]
    opening = "".join(p[0] for p in _QUOTE_PAIRS) + "("
    closing = "".join(p[1] for p in _QUOTE_PAIRS) + ")"
    while span and span[0] in opening:
        span, start = span[1:], start + 1
    # Terminal punctuation stays (it is part of the line); only closing quotes are stripped.
    while span and span[-1] in closing:
        span, end = span[:-1], end - 1
    return VerbatimMatch(text=span, start=start, end=end)


def extract_quoted(text: str) -> list[str]:
    """Return quoted fragments in order of appearance (hints for the script writer)."""
    found: list[tuple[int, str]] = []
    for open_q, close_q in _QUOTE_PAIRS:
        pattern = (
            re.escape(open_q) + r"([^" + re.escape(open_q + close_q) + r"]+?)" + re.escape(close_q)
        )
        found.extend((m.start(), m.group(1).strip()) for m in re.finditer(pattern, text))
    seen: set[str] = set()
    result: list[str] = []
    for _, fragment in sorted(found):
        if fragment and fragment not in seen:
            seen.add(fragment)
            result.append(fragment)
    return result
