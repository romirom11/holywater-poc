import pytest

from holywater.quality.text_norm import extract_quoted, find_verbatim_span, normalize, wer


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
