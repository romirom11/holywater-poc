import pytest

from holywater.quality.text_norm import (
    extract_quoted,
    find_verbatim_span,
    normalize,
    quoted_spans,
    wer,
)


@pytest.mark.parametrize(
    ("raw", "lang", "expected"),
    [
        ("Hello, World!", "en", "hello world"),
        ("  double   spaces ", "en", "double spaces"),
        ("It’s fine", "en", "it's fine"),
        ("Itʼs fine", "en", "it's fine"),
        ("“Quoted” text", "en", "quoted text"),
        ("«Лапки» і текст", "uk", "лапки і текст"),
        ("Ти впевнений, що це та адреса?", "uk", "ти впевнений що це та адреса"),
        ("Room 7", "en", "room seven"),
        ("Кімната 7", "uk", "кімната сім"),
        ("well-known", "en", "well known"),
        ("'quoted'", "en", "quoted"),
        ("don't", "en", "don't"),
    ],
)
def test_normalize(raw: str, lang: str, expected: str) -> None:
    assert normalize(raw, lang) == expected


def test_wer_identical_after_normalization() -> None:
    assert wer("Are you sure this is the address?", "are you sure this is the address") == 0.0


def test_wer_numbers_as_words() -> None:
    assert wer("Meet me at 7", "meet me at seven") == 0.0


def test_wer_counts_substitutions() -> None:
    assert wer("one two three four", "one two tree four") == pytest.approx(0.25)


def test_wer_empty_cases() -> None:
    assert wer("", "") == 0.0
    assert wer("", "noise") == 1.0
    assert wer("something", "") == 1.0


def test_find_verbatim_span_returns_original_characters() -> None:
    source = 'Night, rain. The driver says: "Are you SURE this is the address?" Then silence.'
    match = find_verbatim_span(source, "are you sure this is the address")
    assert match is not None
    assert match.text == "Are you SURE this is the address?"
    assert source[match.start : match.end] == match.text


def test_find_verbatim_span_ukrainian_quotes() -> None:
    source = "Дощ. Таксист питає: «Ти впевнена, що це та адреса?» і мовчить."
    match = find_verbatim_span(source, "ти впевнена що це та адреса", "uk")
    assert match is not None
    assert match.text == "Ти впевнена, що це та адреса?"


def test_find_verbatim_span_rejects_paraphrase() -> None:
    source = 'The driver says: "Are you sure this is the address?"'
    assert find_verbatim_span(source, "are you certain this is the address") is None


def test_find_verbatim_span_number_spelled_differently() -> None:
    source = 'She whispers: "Meet me at 7."'
    match = find_verbatim_span(source, "meet me at seven")
    assert match is not None
    assert match.text == "Meet me at 7."


def test_extract_quoted_handles_mixed_quotes() -> None:
    text = 'He said "first line". Then «друга репліка». And “third”.'
    assert extract_quoted(text) == ["first line", "друга репліка", "third"]
    assert extract_quoted("no quotes here") == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("She says 'It's fine.' He nods.", ["It's fine."]),
        ("She says ‘It’s fine.’ He nods.", ["It’s fine."]),
        ("It's the dogs' day and the cats' night.", []),
        ("Вона каже: „Я повернуся о 7“.", ["Я повернуся о 7"]),
        ('She says "Hi". He answers "Go away".', ["Hi", "Go away"]),
        ('Empty "  " quotes and "real one"', ["real one"]),
    ],
)
def test_extract_quoted_apostrophes_and_styles(text: str, expected: list[str]) -> None:
    assert extract_quoted(text) == expected


def test_quoted_spans_point_at_the_original_characters() -> None:
    text = 'He says: "  Wait for me. " and «Не йди»'
    spans = quoted_spans(text)
    assert [s.text for s in spans] == ["Wait for me.", "Не йди"]
    for span in spans:
        assert text[span.start : span.end] == span.text


# -- B1: the span is exactly the line ----------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "candidate", "lang", "expected"),
    [
        (
            'He says "Just hire Roman already", and leaves.',
            "Just hire Roman already",
            "en",
            "Just hire Roman already",
        ),
        (
            "Таксист питає: «Ти впевнена, що це та адреса?», і чекає.",
            "Ти впевнена, що це та адреса?",
            "uk",
            "Ти впевнена, що це та адреса?",
        ),
        (
            'A taxi driver asks: "Are you sure this is the address?"',
            "Are you sure this is the address?",
            "en",
            "Are you sure this is the address?",
        ),
        (
            'The driver asks:"Are you sure this is the address?"',
            "Are you sure this is the address",
            "en",
            "Are you sure this is the address?",
        ),
        (
            'She says: "Cancel the other interviews, Roman gets the job."',
            "Cancel the other interviews, Roman gets the job.",
            "en",
            "Cancel the other interviews, Roman gets the job.",
        ),
        (
            "A lighthouse keeper watches the storm roll in. The light must not go out tonight.",
            "The light must not go out tonight",
            "en",
            "The light must not go out tonight.",
        ),
        ('He says "wait", then leaves the room quietly.', "wait", "en", "wait"),
        ('He says "wait," then leaves the room quietly.', "wait,", "en", "wait"),
        ("She whispers «Не йди», і зачиняє двері.", "Не йди", "uk", "Не йди"),
        ('He says "Run!"—and drops the bag.', "Run!", "en", "Run!"),
        ('He says "Run"! and drops the bag.', "Run", "en", "Run"),
        ("Вона каже: „Я повернуся о 7“.", "Я повернуся о 7", "uk", "Я повернуся о 7"),
        ('She says "Hi". He answers "Go away".', "Go away", "en", "Go away"),
        ('He mutters: "Wait..." and sits.', "wait", "en", "Wait..."),
        ('She asks: "Really?!" and laughs.', "really", "en", "Really?!"),
        (
            "Вона каже: «Пʼять хвилин — і будь-що станеться».",
            "п'ять хвилин і будь що станеться",
            "uk",
            "Пʼять хвилин — і будь-що станеться",
        ),
        (
            "Він каже: «Мамин обід о 7:30, не запізнюйся!»",
            "мамин обід о сьомій тридцять не запізнюйся",
            "uk",
            "Мамин обід о 7:30, не запізнюйся!",
        ),
        (
            'Жінка каже: "О 7 ранку його вже не буде вдома."',
            "О сьомій ранку його вже не буде вдома.",
            "uk",
            "О 7 ранку його вже не буде вдома.",
        ),
        (
            'He mutters: "Forty-two dollars and 7 cents, same as last time".',
            "42 dollars and seven cents, same as last time",
            "en",
            "Forty-two dollars and 7 cents, same as last time",
        ),
    ],
)
def test_find_verbatim_span_is_exactly_the_line(
    source: str, candidate: str, lang: str, expected: str
) -> None:
    match = find_verbatim_span(source, candidate, lang)
    assert match is not None
    assert match.text == expected
    assert source[match.start : match.end] == expected


def test_find_verbatim_span_needs_whole_source_words() -> None:
    assert find_verbatim_span('A man says "I have 42 apples" and smiles.', "two apples") is None
    assert find_verbatim_span("It is well-known.", "known") is None


# -- B3: number spelling ----------------------------------------------------------------------

_EN_EQUIVALENT = [
    ("$42.07", "42 dollars and seven cents"),
    ("$42.07", "Forty-two dollars and 7 cents"),
    ("42 dollars and seven cents", "Forty-two dollars and 7 cents"),
    ("$42 and 7 cents, same as last time.", "Forty-two dollars and seven cents. Same as last time"),
    ("It costs $1", "It costs one dollar"),
    ("$0.50 left", "fifty cents left"),
    ("1,000 reasons", "one thousand reasons"),
    ("1,000 reasons", "a thousand reasons"),
    ("1,234 steps", "one thousand two hundred thirty-four steps"),
    ("1,000,000 views", "one million views"),
    ("3.5 hours", "three point five hours"),
    ("50% off", "fifty percent off"),
    ("50 % off", "fifty percent off"),
    ("the 1st time", "the first time"),
    ("the 21st floor", "the twenty-first floor"),
    ("my 2nd and 3rd tries", "my second and third tries"),
    ("It's 9:30 already", "It's nine thirty already"),
    ("Meet me at 7:00", "Meet me at seven"),
    ("Room 7pm", "Room seven pm"),
]

_UK_EQUIVALENT = [
    ("О 7 ранку його вже не буде вдома.", "О сьомій ранку його вже не буде вдома."),
    ("О 7 ранку його вже не буде вдома.", "о сьомої ранку його вже не буде вдома"),
    ("О 7 ранку його вже не буде вдома.", "О сьома ранку його вже не буде вдома."),
    ("О 7 ранку його вже не буде вдома.", "О 7-й ранку його вже не буде вдома."),
    ("27 квітня", "двадцять сьома квітня"),
    ("27-го квітня", "двадцять сьомого квітня"),
    ("на 3-му поверсі", "на третьому поверсі"),
    ("1-ша спроба", "перша спроба"),
    ("Мені 21", "Мені двадцять один"),
    ("у 21-му столітті", "у двадцять першому столітті"),
    ("100 гривень", "сто гривень"),
    ("₴100", "сто гривень"),
    ("$5", "п'ять доларів"),
    ("1000 разів", "тисяча разів"),
    ("2000 кроків", "двох тисяч кроків"),
    ("50% знижки", "п'ятдесят відсотків знижки"),
    ("П'ять хвилин", "Пʼять хвилин"),
    ("П'ять хвилин", "Пять хвилин"),
    ("за 300 метрів", "за трьохсот метрів"),
    ("о 7:30", "о сьомій тридцять"),
]


@pytest.mark.parametrize(("a", "b"), _EN_EQUIVALENT)
def test_english_number_spellings_are_equivalent(a: str, b: str) -> None:
    assert wer(a, b, "en") == 0.0, (normalize(a, "en"), normalize(b, "en"))
    assert wer(b, a, "en") == 0.0


@pytest.mark.parametrize(("a", "b"), _UK_EQUIVALENT)
def test_ukrainian_numerals_in_any_form_are_equivalent(a: str, b: str) -> None:
    assert wer(a, b, "uk") == 0.0, (normalize(a, "uk"), normalize(b, "uk"))
    assert wer(b, a, "uk") == 0.0


@pytest.mark.parametrize(
    ("expected", "heard", "lang"),
    [
        ("Are you sure this is the address?", "Are you sure this is the right address?", "en"),
        ("Meet me at 7", "Meet me at 8", "en"),
        ("$42.07", "$42.70", "en"),
        ("Forty-two dollars and 7 cents", "Forty-two dollars and 70 cents", "en"),
        ("50% off", "15% off", "en"),
        ("the 1st time", "the 2nd time", "en"),
        ("1,000 reasons", "100 reasons", "en"),
        ("3.5 hours", "35 hours", "en"),
        ("О 7 ранку його вже не буде вдома.", "О восьмій ранку його вже не буде вдома.", "uk"),
        ("27 квітня", "двадцять шоста квітня", "uk"),
        ("Ти впевнена, що це та адреса?", "Ти впевнена, що це та вулиця?", "uk"),
    ],
)
def test_real_differences_fail_the_gate(expected: str, heard: str, lang: str) -> None:
    assert wer(expected, heard, lang) > 0.1


def test_ukrainian_folding_does_not_touch_english() -> None:
    assert normalize("Seven seconds, third time", "en") == "seven seconds third time"
    assert normalize("сьома", "en") == "сьома"
    assert normalize("сьома", "uk") == "сім"


def test_unknown_language_falls_back_to_english_numbers() -> None:
    assert normalize("O 7", "ua") == normalize("o 7", "uk") == "o сім"
    assert normalize("Room 7", "xx") == "room seven"


# -- quote parsing: apostrophes, inch marks, nested quotes -----------------------------------


@pytest.mark.parametrize(
    ("text", "line"),
    [
        (
            "The coach yells: 'The players' bags stay here!' and walks off.",
            "The players' bags stay here!",
        ),
        ("In the '90s, a DJ yells 'Don't stop!' at the crowd.", "Don't stop!"),
        ('A chef lifts a 10" knife and says: "Dinner is ready."', "Dinner is ready."),
        ("A fan shouts 'Rock 'n' roll never dies.' and leaves.", "Rock 'n' roll never dies."),
    ],
)
def test_single_quotes_and_inch_marks_offer_the_whole_line(text: str, line: str) -> None:
    """A word-final apostrophe ("players'"), an apostrophe-initial word ("'90s", "'n'") or an
    inch mark must not hide the real line; it is offered first (longest span at that start)."""
    quotes = extract_quoted(text)
    assert line in quotes
    assert "'90s" not in "".join(quotes) and "knife" not in "".join(quotes)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A boy whispers: \"Mom said 'run' and I ran.\" and cries.", ["Mom said 'run' and I ran."]),
        ("Вона читає: «Він сказав „ні“ і пішов».", ["Він сказав „ні“ і пішов"]),
    ],
)
def test_a_quote_nested_in_a_double_quote_is_not_a_line_of_its_own(
    text: str, expected: list[str]
) -> None:
    assert extract_quoted(text) == expected


@pytest.mark.parametrize(
    ("source", "candidate", "lang", "expected"),
    [
        (
            'Meet me at 7:30, "Room 12". Then',
            'Meet me at 7:30, "Room 12"',
            "en",
            'Meet me at 7:30, "Room 12".',
        ),
        ("Він сказав „ні“. І йде.", "Він сказав „ні", "uk", "Він сказав „ні“."),
        ("Він сказав «так» і пішов.", "Він сказав «так", "uk", "Він сказав «так»"),
    ],
)
def test_a_span_keeps_the_closing_mark_of_a_quote_it_opened(
    source: str, candidate: str, lang: str, expected: str
) -> None:
    match = find_verbatim_span(source, candidate, lang)
    assert match is not None and match.text == expected


# -- asymmetric leniency: ordinals and synonyms only against digits ---------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Ти перший, хто це сказав.", "Ти один, хто це сказав."),
        ("Я подзвоню другу.", "Я подзвоню двом."),  # "friend" (dat.), not "two"
        ("Це друга спроба.", "Це дві спроба."),
        ("Третя година.", "Три година."),
        ("Знижка п'ятдесят відсотків.", "Знижка п'ятдесят процентів."),
    ],
)
def test_ukrainian_words_that_only_look_alike_are_different(a: str, b: str) -> None:
    """An ordinal is not its cardinal and "процент" is not "відсоток" when both are typed as
    words: a substitution must not score WER 0."""
    assert wer(a, b, "uk") > 0.1 and wer(b, a, "uk") > 0.1


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("О 7.00 ранку його вже не буде вдома.", "О 7 ранку його вже не буде вдома."),
        ("О 7.00 ранку його вже не буде вдома.", "О сьомій ранку його вже не буде вдома."),
        ("о 7.30", "о сьомій тридцять"),
        ("50% знижки", "п'ятдесят процентів знижки"),
        ("Я подзвоню другу.", "я подзвоню другу"),
    ],
)
def test_ukrainian_digits_times_and_percent_match_their_spoken_forms(a: str, b: str) -> None:
    assert wer(a, b, "uk") == 0.0 and wer(b, a, "uk") == 0.0


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("It's OK.", "It's okay."),
        ("Alright, let's go.", "All right, let's go."),
        ("Send me an e-mail.", "Send me an email."),
    ],
)
def test_english_spelling_variants_are_equivalent(a: str, b: str) -> None:
    assert wer(a, b, "en") == 0.0 and wer(b, a, "en") == 0.0


def test_a_decimal_stays_a_decimal() -> None:
    assert normalize("3.5 години", "uk") != normalize("3:05 години", "uk")
    assert wer("Meet me at 7.30", "Meet me at 7:30", "en") > 0  # English "." is a decimal point
