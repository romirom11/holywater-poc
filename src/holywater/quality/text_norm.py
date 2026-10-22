"""Text normalization and WER for the verbatim-line gate (R4).

Normalization is deliberately lossy and used for *comparison only*: case, punctuation, quote and
apostrophe style, and the spelling of numbers (digits vs words, thousands separators, decimals,
currency and percent signs, ordinals, times, Ukrainian case/gender forms of numerals) must not
count as a mismatch between the input line and an STT transcript. The line itself always keeps
the user's original characters (ADR-005): `find_verbatim_span` maps the comparison back to them.

Text is split into *units* (a word, or a number with its currency/percent sign and suffix), each
unit becomes zero or more canonical tokens, and a few cross-token rules run on the token list.
Every token remembers the character span of its unit, which is how a match is mapped back.

Leniency is asymmetric where symmetric folding would equate different words: a Ukrainian ordinal
("сьомій", "перший") or "процент" keeps its own canonical form and is read as the cardinal
("сім", "один") or "відсоток" only when the *other* text wrote that number (or "%") in digits.
So "О 7 ранку" == "О сьомій ранку", but "Ти перший" != "Ти один".
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal

import jiwer
from num2words import num2words

from holywater.observability.logging import get_logger

log = get_logger(__name__)

_LANG_FALLBACK = "en"
_LANG_ALIASES = {"ua": "uk"}
_APOSTROPHES = "'’ʼ‘`´"
_CURRENCY_SIGNS = "$€£₴"
_COMBINING = "\u0300-\u036f"
_LETTER = rf"(?:[^\W\d_]|[{_COMBINING}])"
_MAX_SPELLED = 10**15  # longer digit runs stay digits (ids, phone numbers)

_DIGITS = r"\d+(?:[.,\u00a0\u202f]\d+)*"
# Ukrainian (and most languages but English) also group thousands with a plain space: "1 000".
_SPACE_GROUPED = r"\d{1,3}(?:[ ]\d{3}(?!\d))+(?:,\d+)?"
_UNIT_TEMPLATE = rf"""
    (?P<number>
        (?:(?P<pre>[{_CURRENCY_SIGNS}])\s?)?
        (?:
            (?P<hh>\d{{1,2}}):(?P<mm>\d{{2}})(?!\d)
          | (?P<digits>DIGITS)
        )
        (?:\s?(?P<post>[{_CURRENCY_SIGNS}%])|-?(?P<suffix>{_LETTER}+))?
    )
    | (?P<word>{_LETTER}+(?:[{_APOSTROPHES}\-]{_LETTER}+)*)
"""
_UNIT_RE_EN = re.compile(_UNIT_TEMPLATE.replace("DIGITS", _DIGITS), flags=re.VERBOSE)
_UNIT_RE = re.compile(
    _UNIT_TEMPLATE.replace("DIGITS", f"{_SPACE_GROUPED}|{_DIGITS}"), flags=re.VERBOSE
)
_WORDS_ONLY_RE = re.compile(rf"{_LETTER}+(?:[{_APOSTROPHES}\-]{_LETTER}+)*")
# Terminal punctuation that belongs to the line when it directly follows the last word.
_TERMINAL_RE = re.compile(r"[.!?…]+")

_EN_ORDINAL_SUFFIXES = frozenset({"st", "nd", "rd", "th"})
_DOTTED_TIME_RE = re.compile(r"(\d{1,2})\.(\d{2})")
_HOURS_PER_DAY = 24
_MINUTES_PER_HOUR = 60
_MAX_GRAMMATICAL_SUFFIX = 3  # "7-й", "7-го", "1-ша": a case ending, not a word
_PERCENT = {"en": "percent", "uk": "відсоток"}
_CURRENCY_UNITS: dict[str, dict[str, tuple[str, str]]] = {
    "en": {
        "$": ("dollars", "cents"),
        "€": ("euros", "cents"),
        "£": ("pounds", "pence"),
        "₴": ("hryvnias", "kopiykas"),
    },
    "uk": {
        "$": ("долар", "цент"),
        "€": ("євро", "цент"),
        "£": ("фунт", "пенс"),
        "₴": ("гривня", "копійка"),
    },
}
_MAJOR_UNITS = frozenset(major for units in _CURRENCY_UNITS.values() for major, _ in units.values())
_MINOR_UNITS = frozenset(minor for units in _CURRENCY_UNITS.values() for _, minor in units.values())
_CONJUNCTIONS = frozenset({"and", "і", "й", "та"})
_CURRENCY_LOOKAHEAD = 4  # "dollars and seven cents": the minor unit follows within a few tokens
_GROUP_DIGITS = 3  # "1,000": thousands groups
_CENT_DIGITS = 2  # "$42.07"

_EN_SCALES = frozenset({"hundred", "thousand", "million", "billion", "trillion"})
_UK_SCALES = frozenset({"тисяча", "мільйон", "мільярд"})
_EN_NUMBER_WORDS = _EN_SCALES | frozenset(
    {
        *("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"),
        *("ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen"),
        *("eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy"),
        *("eighty", "ninety"),
    }
)
_EN_CANONICAL = {
    "dollar": "dollars",
    "cent": "cents",
    "euro": "euros",
    "pound": "pounds",
    "penny": "pence",
    "pennies": "pence",
    "ok": "okay",
    "alright": "all right",
}
# Whole hyphenated words with a common closed spelling ("e-mail" == "email").
_EN_JOINED = {"e-mail": "email"}


# Ukrainian numerals decline by case and gender, and ordinals are adjectives. Canonical form is
# the nominative cardinal that num2words produces for digits ("7" -> "сім"); speech and STT use
# whatever form the sentence needs ("о сьомій", "двадцять сьома"). Both sides fold to it.
# canonical -> irregular forms; currency and percent nouns fold the same way ("$5" vs
# "п'ять доларів").
_UK_FORMS: dict[str, str] = {
    "нуль": "нуля нулю нулем нулі нулів нулям нулями нулях",
    "один": "одна одне одно одного одної однієї одному одній одним одною однією одну одні одних "
    "одними",
    "два": "дві двох двом двома",
    "три": "трьох трьом трьома",
    "чотири": "чотирьох чотирьом чотирма чотирьома",
    "шість": "шести шістьох шістьом шістьма шістьома",
    "сім": "семи сімох сімом сьома сімома",
    "вісім": "восьми вісьмох вісьмом вісьма вісьмома",
    "сорок": "сорока",
    "дев'яносто": "дев'яноста",
    "сто": "ста",
    "двісті": "двохсот двомстам двомастами двохстах",
    "триста": "трьохсот трьомстам трьомастами трьохстах",
    "чотириста": "чотирьохсот чотирьомстам чотирмастами чотирьохстах",
    "п'ятсот": "п'ятисот п'ятистам п'ятистах п'ятьмастами",
    "шістсот": "шестисот шестистам шестистах шістьмастами",
    "сімсот": "семисот семистам семистах сімомастами",
    "вісімсот": "восьмисот восьмистам восьмистах вісьмомастами",
    "дев'ятсот": "дев'ятисот дев'ятистам дев'ятистах дев'ятьмастами",
    "тисяча": "тисячі тисяч тисячу тисячею тисячам тисячами тисячах",
    "мільйон": "мільйона мільйону мільйоном мільйоні мільйони мільйонів мільйонам мільйонами "
    "мільйонах",
    "мільярд": "мільярда мільярду мільярдом мільярді мільярди мільярдів мільярдам мільярдами "
    "мільярдах",
    "долар": "долара долару доларові доларом доларі долари доларів доларам доларами доларах",
    "цент": "цента центу центові центом центі центи центів центам центами центах",
    "гривня": "гривні гривню гривнею гривень гривням гривнями гривнях грн",
    "копійка": "копійки копійці копійку копійкою копійок копійкам копійками копійках коп",
    "відсоток": "відсотка відсотку відсоткові відсотком відсотки відсотків відсоткам відсотками "
    "відсотках",
    "процент": "процента проценту процентом проценті проценти процентів процентам процентами "
    "процентах",
}
# Ordinals whose forms are not stem + hard ending, by their masculine nominative.
_UK_IRREGULAR_ORDINALS = {
    "третій": "третя третє третього третьої третьому третім третьою третю треті третіх третіми",
}
# Numerals that decline like "п'ять" (stem + ending); the stem is also the ordinal stem.
_UK_LIKE_FIVE = (
    "п'ять",
    "дев'ять",
    "десять",
    "одинадцять",
    "дванадцять",
    "тринадцять",
    "чотирнадцять",
    "п'ятнадцять",
    "шістнадцять",
    "сімнадцять",
    "вісімнадцять",
    "дев'ятнадцять",
    "двадцять",
    "тридцять",
)
_UK_LIKE_FIFTY = ("п'ятдесят", "шістдесят", "сімдесят", "вісімдесят")
_UK_FIVE_ENDINGS = ("и", "ьох", "ьом", "ьма", "ьома")
# canonical -> ordinal stem taking the hard adjective endings ("сьом" + "ій" = "сьомій").
_UK_ORDINAL_STEMS = {
    "нуль": "нульов",
    "один": "перш",
    "два": "друг",
    "чотири": "четверт",
    "шість": "шост",
    "сім": "сьом",
    "вісім": "восьм",
    "сорок": "сороков",
    "дев'яносто": "дев'яност",
    "сто": "сот",
    "двісті": "двохсот",
    "триста": "трьохсот",
    "чотириста": "чотирьохсот",
    "п'ятсот": "п'ятисот",
    "шістсот": "шестисот",
    "сімсот": "семисот",
    "вісімсот": "восьмисот",
    "дев'ятсот": "дев'ятисот",
    "тисяча": "тисячн",
    "мільйон": "мільйонн",
    "мільярд": "мільярдн",
}
_UK_HARD_ENDINGS = (
    "ий",
    "а",
    "е",
    "ого",
    "ої",
    "ому",
    "ім",
    "ій",
    "им",
    "ою",
    "у",
    "і",
    "их",
    "ими",
)


def _build_uk_tables() -> tuple[dict[str, str], dict[str, str]]:
    """(form -> canonical, ordinal canonical -> cardinal canonical).

    Cardinal forms (any case or gender) fold to the nominative cardinal ("семи" -> "сім");
    ordinal forms fold to the masculine nominative ordinal ("сьомій" -> "сьомий").
    A form that is both ("сьома": "with seven" or "seventh") folds to the cardinal.
    """
    cardinals = {canonical: words.split() for canonical, words in _UK_FORMS.items()}
    stems = dict(_UK_ORDINAL_STEMS)
    for nominative in _UK_LIKE_FIVE:
        stems[nominative] = nominative[:-1]  # "п'ят"
        cardinals[nominative] = [nominative[:-1] + e for e in _UK_FIVE_ENDINGS]
    for nominative in _UK_LIKE_FIFTY:
        stems[nominative] = nominative
        cardinals[nominative] = [nominative + e for e in _UK_FIVE_ENDINGS]
    ordinals = {stem + "ий": [stem + e for e in _UK_HARD_ENDINGS] for stem in stems.values()}
    ordinals.update({k: v.split() for k, v in _UK_IRREGULAR_ORDINALS.items()})
    ordinal_of = {stem + "ий": cardinal for cardinal, stem in stems.items()}
    ordinal_of["третій"] = "три"
    table: dict[str, str] = {}
    for forms in (ordinals, cardinals):  # cardinals last: they win a shared form
        for canonical, words in forms.items():
            for word in (canonical, *words):
                table[word] = canonical
                table[word.replace("'", "")] = canonical  # STT may drop the apostrophe
    loose = {**ordinal_of, "процент": "відсоток"}
    return table, loose


_UK_CANONICAL, _UK_LOOSE = _build_uk_tables()


@dataclass(frozen=True)
class _Token:
    text: str
    start: int
    end: int
    unit: int
    from_digits: bool = False  # spelled from digits or a sign ("7", "$", "%"), not typed as words


def normalize(text: str, language: str = _LANG_FALLBACK) -> str:
    """Canonical comparison form: lowercase words, numbers spelled out, no punctuation."""
    return " ".join(tok.text for tok in _tokenize(text, language))


def wer(expected: str, actual: str, language: str = _LANG_FALLBACK) -> float:
    """Word error rate on normalized text. 0.0 = identical, 1.0 = everything wrong."""
    ref_tokens, hyp_tokens = _tokenize(expected, language), _tokenize(actual, language)
    ref = " ".join(_relaxed(ref_tokens, hyp_tokens))
    hyp = " ".join(_relaxed(hyp_tokens, ref_tokens))
    if not ref:
        return 0.0 if not hyp else 1.0
    if not hyp:
        return 1.0
    return float(jiwer.wer(ref, hyp))


def same_words(a: str, b: str, language: str = _LANG_FALLBACK) -> bool:
    """Whether two texts are the same words after normalization (WER 0 in both directions)."""
    a_tokens, b_tokens = _tokenize(a, language), _tokenize(b, language)
    return _relaxed(a_tokens, b_tokens) == _relaxed(b_tokens, a_tokens)


def _relaxed(tokens: list[_Token], other: Iterable[_Token]) -> list[str]:
    """Token texts, with an ordinal (or "процент") read as its cardinal (or "відсоток") when the
    other text wrote that number (or "%") in digits; see the module docstring."""
    digits = {t.text for t in other if t.from_digits}
    out: list[str] = []
    for tok in tokens:
        loose = None if tok.from_digits else _UK_LOOSE.get(tok.text)
        out.append(loose if loose is not None and loose in digits else tok.text)
    return out


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
    user typed (ADR-005). The span runs from the first matched word to the last matched word,
    plus terminal punctuation (. ? ! …) directly after it. Quote marks, commas after a closing
    quote and neighbouring narration words are never part of it. A match must cover whole
    source words: "two apples" does not match inside "42 apples".
    """
    candidate_tokens = _tokenize(candidate, language)
    if not candidate_tokens:
        return None
    tokens = _tokenize(source, language)
    wanted = _relaxed(candidate_tokens, tokens)
    texts = _relaxed(tokens, candidate_tokens)
    n = len(wanted)
    for i in range(len(tokens) - n + 1):
        first, last = tokens[i], tokens[i + n - 1]
        starts_unit = i == 0 or tokens[i - 1].unit != first.unit
        ends_unit = i + n == len(tokens) or tokens[i + n].unit != last.unit
        if starts_unit and ends_unit and texts[i : i + n] == wanted:
            end = _close_inner_quotes(source, first.start, last.end)
            terminal = _TERMINAL_RE.match(source, end)
            if terminal is not None:
                end = terminal.end()
            return VerbatimMatch(text=source[first.start : end], start=first.start, end=end)
    return None


# Opening mark -> closing marks, for quotes opened inside a line ('"Room 12' needs its '"').
# ASCII "'" is left out: inside a line it is far more often an apostrophe.
_INNER_QUOTES = {'"': '"', "“": "”", "„": "“”", "«": "»", "‘": "’"}


def _close_inner_quotes(source: str, start: int, end: int) -> int:
    """Extend `end` over closing marks of quotes the fragment `source[start:end]` opened but did
    not close, so the line keeps balanced quote marks (and then its terminal punctuation)."""
    fragment = source[start:end]
    for opener, closers in _INNER_QUOTES.items():
        if opener == closers:
            unclosed = fragment.count(opener) % 2
        else:
            unclosed = fragment.count(opener) - sum(fragment.count(c) for c in closers)
        if unclosed > 0 and end < len(source) and source[end] in closers:
            end += 1
    return end


# -- quotes -----------------------------------------------------------------------------------

# Double-style quotes: unambiguous pairs. A '"' right after a digit is an inch mark (a 10" knife).
_DOUBLE_QUOTE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"“([^“”]+?)”"),
    re.compile(r"„([^„“”]+?)[“”]"),
    re.compile(r"«([^«»]+?)»"),
    re.compile(r'(?<!\d)"([^"]+?)"'),
)
# Single quotes double as apostrophes ("It's", "players'", "'n'", "'90s"): (opening, closing).
_SINGLE_QUOTES = (("‘", "’"), ("'", "'"))
_WORD_CHAR = re.compile(r"\w")


def quoted_spans(text: str) -> list[VerbatimMatch]:
    """Quoted fragments with offsets in `text`, in order of appearance, each distinct one once.

    Offsets cover the fragment without the quote marks and without surrounding whitespace. A
    single quote is ambiguous, so every closing candidate is offered ("The players' bags stay
    here!" yields "The players" and the whole line); a fragment nested inside a double-style quote
    is not offered on its own (in "Mom said 'run' and I ran.", "run" is part of the line).
    """
    found: list[VerbatimMatch] = []
    for pattern in _DOUBLE_QUOTE_PATTERNS:
        doubles = [_span(text, m.start(1), m.end(1)) for m in pattern.finditer(text)]
        found.extend(span for span in doubles if span is not None)
    outer = list(found)
    for opener, closer in _SINGLE_QUOTES:
        for start, end in _single_quoted(text, opener, closer):
            span = _span(text, start, end)
            if span is not None and not any(_inside(span, o) for o in outer):
                found.append(span)
    found = [s for s in found if not any(_inside(s, o) for o in outer)]
    seen: set[str] = set()
    result: list[VerbatimMatch] = []
    for span in sorted(found, key=lambda s: (s.start, -s.end)):
        if span.text not in seen:
            seen.add(span.text)
            result.append(span)
    return result


def _span(text: str, start: int, end: int) -> VerbatimMatch | None:
    inner = text[start:end]
    stripped = inner.strip()
    if not stripped:
        return None
    start += len(inner) - len(inner.lstrip())
    return VerbatimMatch(text=stripped, start=start, end=start + len(stripped))


def _inside(span: VerbatimMatch, outer: VerbatimMatch) -> bool:
    """`span` lies within `outer` and is not `outer` itself."""
    return outer.start <= span.start and span.end <= outer.end and span != outer


def _single_quoted(text: str, opener: str, closer: str) -> list[tuple[int, int]]:
    """(start, end) of every fragment between an opening single quote and each of its closing
    candidates. An opener follows no word character and precedes neither a digit ("'90s") nor
    whitespace; a closer follows no whitespace and precedes no word character."""
    spans: list[tuple[int, int]] = []
    for o, ch in enumerate(text):
        if ch != opener or o + 1 >= len(text):
            continue
        before, after = text[o - 1] if o else " ", text[o + 1]
        if _WORD_CHAR.match(before) or after.isdigit() or after.isspace():
            continue
        for c in range(o + 2, len(text)):
            if text[c] != closer or text[c - 1].isspace():
                continue
            if c + 1 < len(text) and _WORD_CHAR.match(text[c + 1]):
                continue
            spans.append((o + 1, c))
    return spans


def extract_quoted(text: str) -> list[str]:
    """Return quoted fragments in order of appearance (hints for the script writer)."""
    return [span.text for span in quoted_spans(text)]


# -- tokenization -----------------------------------------------------------------------------


def _lang(language: str) -> str:
    base = language.strip().lower().replace("_", "-").split("-")[0] or _LANG_FALLBACK
    return _LANG_ALIASES.get(base, base)


def _tokenize(text: str, language: str) -> list[_Token]:
    lang = _lang(language)
    tokens: list[_Token] = []
    unit_re = _UNIT_RE_EN if lang == "en" else _UNIT_RE
    for unit, m in enumerate(unit_re.finditer(text)):
        is_number = bool(m.group("number"))
        words = _number_words(m, lang) if is_number else _word_tokens(m.group(0), lang)
        tokens.extend(_Token(word, m.start(), m.end(), unit, is_number) for word in words)
    return _apply_cross_token_rules(tokens, lang)


def _word_tokens(raw: str, lang: str) -> list[str]:
    text = unicodedata.normalize("NFKC", raw).lower()
    text = "".join(ch for ch in text if not unicodedata.combining(ch))  # stress marks
    for ch in _APOSTROPHES:
        text = text.replace(ch, "'")
    if lang == "en":
        text = _EN_JOINED.get(text, text)
    table = _UK_CANONICAL if lang == "uk" else _EN_CANONICAL if lang == "en" else {}
    return [word for part in text.split("-") if part for word in table.get(part, part).split()]


def _spelled(text: str, lang: str) -> list[str]:
    out: list[str] = []
    for word in _WORDS_ONLY_RE.findall(text):
        out.extend(_word_tokens(word, lang))
    return out


def _say(value: int | Decimal, lang: str, *, ordinal: bool = False) -> list[str]:
    """num2words in `lang` (falling back to English); digits as-is when it cannot spell them.

    Both sides of a comparison go through the same fallback, so it stays consistent.
    """
    if abs(int(value)) >= _MAX_SPELLED:
        return [str(value)]
    to = "ordinal" if ordinal else "cardinal"
    for code in dict.fromkeys((lang, _LANG_FALLBACK)):
        try:
            return _spelled(str(num2words(value, lang=code, to=to)), lang)
        except (NotImplementedError, LookupError, OverflowError, ValueError) as exc:
            log.warning(
                "text_norm.spell_fallback", value=str(value), language=code, error=repr(exc)
            )
    return [str(value)]


def _number_words(m: re.Match[str], lang: str) -> list[str]:
    time = _time_of(m, lang)
    if time is not None:
        hours, minutes = time
        return _say(hours, lang) + (_say(minutes, lang) if minutes else [])
    parsed = _parse_digits(m.group("digits"), lang)
    if parsed is None:  # ambiguous grouping such as "1,2,3": read each group
        groups = re.split(r"[.,\u00a0\u202f ]", m.group("digits"))
        return [w for g in groups for w in _say(int(g), lang)]
    integer, fraction = parsed
    sign = m.group("pre") or (m.group("post") if m.group("post") != "%" else None)
    if sign:
        return _money(integer, fraction, sign, lang)
    suffix = (m.group("suffix") or "").lower()
    if suffix and fraction is None:
        if lang == "en" and suffix in _EN_ORDINAL_SUFFIXES:
            return _say(integer, lang, ordinal=True)
        if lang != "en" and len(suffix) <= _MAX_GRAMMATICAL_SUFFIX:
            # "7-й", "7-ма", "21-ше": the ending only inflects; numerals fold to the cardinal.
            return _say(integer, lang)
    words = _say(integer if fraction is None else Decimal(f"{integer}.{fraction}"), lang)
    if m.group("post") == "%":
        words.append(_PERCENT.get(lang, _PERCENT[_LANG_FALLBACK]))
    elif suffix:
        words.extend(_word_tokens(suffix, lang))
    return words


def _time_of(m: re.Match[str], lang: str) -> tuple[int, int] | None:
    """(hours, minutes) of a time unit: "7:30" anywhere, and "7.30" outside English — where ","
    is the decimal mark, a bare h.mm is a time ("о 7.00 ранку"), not a decimal."""
    if m.group("hh") is not None:
        return int(m.group("hh")), int(m.group("mm"))
    dotted = _DOTTED_TIME_RE.fullmatch(m.group("digits"))
    if lang == "en" or dotted is None or m.group("pre") or m.group("post") or m.group("suffix"):
        return None
    hours, minutes = int(dotted.group(1)), int(dotted.group(2))
    return (hours, minutes) if hours < _HOURS_PER_DAY and minutes < _MINUTES_PER_HOUR else None


def _parse_digits(raw: str, lang: str) -> tuple[int, str | None] | None:
    """Split a digit run into (integer, fraction digits) using the language's separators."""
    raw = raw.replace(" ", "").replace("\u00a0", "").replace("\u202f", "")
    seps = [ch for ch in raw if ch in ".,"]
    groups = re.split(r"[.,]", raw)
    if not seps:
        return int(raw), None
    if "." in seps and "," in seps:  # "1,234.56" / "1.234,56": the last one is the decimal
        decimal_sep = raw[max(raw.rfind("."), raw.rfind(","))]
        head, _, fraction = raw.rpartition(decimal_sep)
        head_groups = re.split(r"[.,]", head)
        if any(len(g) != _GROUP_DIGITS for g in head_groups[1:]):
            return None
        return int("".join(head_groups)), fraction
    thousands_like = all(len(g) == _GROUP_DIGITS for g in groups[1:])
    if len(seps) > 1:
        return (int("".join(groups)), None) if thousands_like else None
    # One separator: "1,000" is a thousand in English; elsewhere "," is the decimal comma.
    if seps[0] == "," and lang == "en" and thousands_like:
        return int("".join(groups)), None
    return int(groups[0]), groups[1]


def _money(integer: int, fraction: str | None, sign: str, lang: str) -> list[str]:
    major, minor = _CURRENCY_UNITS.get(lang, _CURRENCY_UNITS[_LANG_FALLBACK])[sign]
    if fraction is not None and len(fraction) > _CENT_DIGITS:  # "$1.234": decimal, no cents
        return [*_say(Decimal(f"{integer}.{fraction}"), lang), major]
    cents = int(fraction.ljust(_CENT_DIGITS, "0")) if fraction else 0
    words: list[str] = []
    if integer or not cents:
        words += [*_say(integer, lang), major]
    if cents:
        words += [*_say(cents, lang), minor]
    return words


def _apply_cross_token_rules(tokens: list[_Token], lang: str) -> list[_Token]:
    """Rules that need neighbours: dropped tokens keep their unit's span covered by the others."""
    out: list[_Token] = []
    for i, tok in enumerate(tokens):
        prev = out[-1].text if out else ""
        nxt = tokens[i + 1].text if i + 1 < len(tokens) else ""
        if lang == "uk" and tok.text == "один" and nxt in _UK_SCALES:
            continue  # num2words says "одна тисяча", people say "тисяча"
        if (
            tok.text in _CONJUNCTIONS
            and prev in _MAJOR_UNITS
            and _any_in(tokens[i + 1 : i + 1 + _CURRENCY_LOOKAHEAD], _MINOR_UNITS)
        ):
            continue  # "42 dollars and 7 cents" == "$42.07"
        if lang == "en" and tok.text == "and" and prev in _EN_SCALES and nxt in _EN_NUMBER_WORDS:
            continue  # "two hundred and five" == "two hundred five"
        if lang == "en" and tok.text == "a" and nxt in _EN_SCALES:
            out.append(_Token("one", tok.start, tok.end, tok.unit))  # "a thousand" == "one ..."
            continue
        out.append(tok)
    return out


def _any_in(tokens: Iterable[_Token], words: frozenset[str]) -> bool:
    return any(tok.text in words for tok in tokens)
