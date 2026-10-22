from typing import Any

import pytest
from pydantic import BaseModel

from holywater.config.schema import LLMConfig
from holywater.domain.idea import IdeaInput
from holywater.domain.script import Script
from holywater.observability.errors import ScriptError
from holywater.providers.base import LLMResult
from holywater.providers.fake import FakeLLM
from holywater.quality.script_checks import check_script_against_idea, window_violations
from holywater.stages.write_script import ScriptWriterDeps, write_script
from tests.unit.test_domain import make_script

SHOWCASE = (
    "In a glass meeting room, a stern hiring manager finishes an AI-made video and smiles for "
    'the first time today. She says: "Cancel the other interviews, Roman gets the job."'
)


def _script(line: str, language: str = "en", **dialogue: Any) -> Script:
    data = make_script()
    data["language"] = language
    data["dialogue"].update({"text": line, "language": language, "start_s": 1.0, "end_s": 7.5})
    data["dialogue"].update(dialogue)
    return Script.model_validate(data)


# -- B2: the line is one complete quoted fragment ---------------------------------------------


@pytest.mark.parametrize(
    ("idea", "line", "language", "expected"),
    [
        (
            SHOWCASE,
            "Cancel the other interviews, Roman gets the job.",
            "en",
            "Cancel the other interviews, Roman gets the job.",
        ),
        (
            SHOWCASE,
            "cancel the other interviews roman gets the job",
            "en",
            "Cancel the other interviews, Roman gets the job.",
        ),
        (
            SHOWCASE,
            '"Cancel the other interviews, Roman gets the job."',
            "en",
            "Cancel the other interviews, Roman gets the job.",
        ),
        (
            'He says "Just hire Roman already", and leaves.',
            "Just hire Roman already",
            "en",
            "Just hire Roman already",
        ),
        (
            "Таксист питає: «Ти впевнена, що це та адреса?», і чекає.",
            "Ти впевнена, що це та адреса",
            "uk",
            "Ти впевнена, що це та адреса?",
        ),
        (
            'Жінка на кухні дивиться на годинник і каже: "О 7 ранку його вже не буде вдома."',
            "О сьомій ранку його вже не буде вдома.",
            "uk",
            "О 7 ранку його вже не буде вдома.",
        ),
        ('She says "Hi". He answers "Go away".', "Go away", "en", "Go away"),
        ('She says "Hi". He answers "Go away".', "Hi", "en", "Hi"),
    ],
)
def test_complete_quote_passes_with_original_characters(
    idea: str, line: str, language: str, expected: str
) -> None:
    result = check_script_against_idea(_script(line, language), IdeaInput(text=idea))
    assert result.ok, result.violations
    assert result.script.dialogue.text == expected


@pytest.mark.parametrize(
    "line",
    [
        "Roman gets the job.",  # partial quote
        "Cancel the other interviews",  # partial quote
        "She says: Cancel the other interviews, Roman gets the job.",  # quote plus narration
        "smiles for the first time today",  # narration while quotes exist
        "Cancel the interviews, Roman gets the job.",  # paraphrase
    ],
)
def test_anything_but_a_complete_quote_is_a_violation(line: str) -> None:
    result = check_script_against_idea(_script(line), IdeaInput(text=SHOWCASE))
    assert not result.ok
    assert result.script.dialogue.text == line  # unchanged: nothing downstream uses it
    (message,) = result.violations
    assert "one complete quoted line" in message
    assert "verbatim" in message
    assert "'Cancel the other interviews, Roman gets the job.'" in message
    assert repr(line) in message


def test_violation_lists_every_quoted_fragment() -> None:
    idea = IdeaInput(text='She says "Hi". He answers "Go away, I am busy".')
    result = check_script_against_idea(_script("Go away"), idea)
    assert not result.ok
    assert "Allowed: 'Hi' or 'Go away, I am busy'; got 'Go away'" in result.violations[0]


def test_without_quotes_any_verbatim_fragment_passes() -> None:
    idea = IdeaInput(
        text="A lighthouse keeper watches the storm roll in. He has not slept in two days. "
        "The light must not go out tonight."
    )
    result = check_script_against_idea(_script("the light must not go out tonight"), idea)
    assert result.ok
    assert result.script.dialogue.text == "The light must not go out tonight."


def test_without_quotes_a_paraphrase_is_a_violation() -> None:
    idea = IdeaInput(text="A lighthouse keeper whispers that the light must not go out tonight.")
    result = check_script_against_idea(_script("The lamp must stay on tonight."), idea)
    assert not result.ok
    assert "verbatim fragment of the input idea" in result.violations[0]


class _RecordingLLM:
    """FakeLLM that keeps every user prompt, to see the feedback the LLM gets on retry."""

    def __init__(self, inner: FakeLLM) -> None:
        self.inner = inner
        self.prompts: list[str] = []

    def complete_structured[T: BaseModel](
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type[T],
        temperature: float | None = None,
    ) -> LLMResult[T]:
        self.prompts.append(user)
        return self.inner.complete_structured(
            model=model, system=system, user=user, schema=schema, temperature=temperature
        )


def test_write_script_feeds_the_allowed_quote_back_and_recovers() -> None:
    llm = _RecordingLLM(FakeLLM(line_override="Roman gets the job.", override_times=1))
    deps = ScriptWriterDeps(llm=llm, config=LLMConfig(model="m", max_attempts=3))
    result = write_script(IdeaInput(text=SHOWCASE), deps)
    assert result.attempts == 2
    assert result.script.dialogue.text == "Cancel the other interviews, Roman gets the job."
    assert "one complete quoted line" in llm.prompts[1]
    assert "Cancel the other interviews, Roman gets the job." in llm.prompts[1]


def test_write_script_never_accepts_a_partial_quote() -> None:
    llm = FakeLLM(line_override="Just give Roman the job.", override_times=99)
    idea = IdeaInput(
        text='Rooftop at sunset. A woman tears up a contract and says: "Forget the billionaire. '
        'Just give Roman the job."'
    )
    with pytest.raises(ScriptError) as exc_info:
        write_script(idea, ScriptWriterDeps(llm=llm, config=LLMConfig(model="m", max_attempts=2)))
    assert (
        "'Forget the billionaire. Just give Roman the job.'"
        in (exc_info.value.context["last_feedback"][0])
    )


# -- B5: the speech-rate check is configurable -------------------------------------------------


def test_window_check_uses_the_given_speech_rate() -> None:
    script = _script("Are you sure this is the address?")  # 33 chars in a 6.5 s window
    assert window_violations(script.dialogue) == []
    assert window_violations(script.dialogue, chars_per_s=15.0) == []
    (message,) = window_violations(script.dialogue, chars_per_s=5.0)
    assert "shorter than the estimated speech time 6.6s for 33 characters at 5 chars/s" in message
    assert "never shorten the line" in message


def test_script_check_reports_a_window_too_short_for_the_verbatim_line() -> None:
    idea = IdeaInput(text='Rainy night. The driver asks: "Are you sure this is the address?"')
    script = _script("Are you sure this is the address?", start_s=1.0, end_s=2.0)
    assert check_script_against_idea(script, idea, chars_per_s=40.0).ok
    result = check_script_against_idea(script, idea, chars_per_s=12.0)
    assert not result.ok
    assert "shorter than the estimated speech time" in result.violations[0]
    assert result.script.dialogue.text == "Are you sure this is the address?"


def test_window_check_rejects_a_non_positive_rate() -> None:
    script = _script("Are you sure this is the address?")
    with pytest.raises(ValueError, match="chars_per_s must be positive"):
        window_violations(script.dialogue, chars_per_s=0)


# -- quote parsing regressions (apostrophes, inch marks, nested quotes) ------------------------


@pytest.mark.parametrize(
    ("idea", "line", "expected"),
    [
        (
            "The coach yells: 'The players' bags stay here!' and walks off.",
            "The players' bags stay here!",
            "The players' bags stay here!",
        ),
        ("In the '90s, a DJ yells 'Don't stop!' at the crowd.", "Don't stop!", "Don't stop!"),
        (
            'A chef lifts a 10" knife and says: "Dinner is ready."',
            "Dinner is ready.",
            "Dinner is ready.",
        ),
        (
            "She reads the note aloud: 'Meet me at 7:30, \"Room 12\".' and frowns.",
            'Meet me at 7:30, "Room 12".',
            'Meet me at 7:30, "Room 12".',
        ),
        # The parser pairs the first two marks; the line between the last two still passes.
        ('The sign reads "OPEN and she says "Come in."', "Come in.", "Come in."),
    ],
)
def test_the_whole_line_passes_despite_ambiguous_marks(idea: str, line: str, expected: str) -> None:
    result = check_script_against_idea(_script(line), IdeaInput(text=idea))
    assert result.ok, result.violations
    assert result.script.dialogue.text == expected


@pytest.mark.parametrize(
    ("idea", "line", "language", "allowed"),
    [
        (
            "A boy whispers: \"Mom said 'run' and I ran.\" and cries.",
            "run",
            "en",
            "\"Mom said 'run' and I ran.\"",
        ),
        ("Вона читає: «Він сказав „ні“ і пішов».", "ні", "uk", "'Він сказав „ні“ і пішов'"),
    ],
)
def test_a_quote_nested_in_the_line_is_a_shortened_line(
    idea: str, line: str, language: str, allowed: str
) -> None:
    result = check_script_against_idea(_script(line, language), IdeaInput(text=idea))
    assert not result.ok
    assert f"Allowed: {allowed};" in result.violations[0]


def test_a_line_ending_in_an_inner_quote_keeps_its_closing_mark() -> None:
    idea = IdeaInput(text="Вона читає: «Він сказав „ні“.» і йде.")
    result = check_script_against_idea(_script("Він сказав „ні", "uk"), idea)
    assert result.ok, result.violations
    assert result.script.dialogue.text == "Він сказав „ні“."
