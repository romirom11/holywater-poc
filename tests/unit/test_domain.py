import json
from typing import Any

import pytest
from pydantic import ValidationError

from holywater.domain.enums import AttemptStatus, VoiceGender
from holywater.domain.idea import MAX_IDEA_CHARS, IdeaInput, count_sentences
from holywater.domain.script import Dialogue, Script, Shot


def make_script(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "title": "Rain",
        "logline": "A taxi driver doubts the address.",
        "language": "en",
        "style": "cinematic, moody",
        "shots": [
            {
                "id": 1,
                "duration_s": 6,
                "visual_prompt": "Night city, rain on a taxi windshield.",
                "camera": "slow push-in",
                "continuity": "yellow taxi, male driver 40s",
                "ambient_audio": "rain, engine idle",
                "has_dialogue": False,
            },
            {
                "id": 2,
                "duration_s": 8,
                "visual_prompt": "Driver glances in the mirror at the passenger.",
                "camera": "handheld close-up",
                "continuity": "yellow taxi, male driver 40s",
                "ambient_audio": "rain, engine idle",
                "has_dialogue": True,
            },
        ],
        "dialogue": {
            "shot_id": 2,
            "speaker": "the driver",
            "text": "Are you sure this is the address?",
            "language": "en",
            "emotion": "quiet, uneasy",
            "start_s": 2.0,
            "end_s": 6.0,
        },
    }
    base.update(overrides)
    return base


def test_valid_script_passes() -> None:
    script = Script.model_validate(make_script())
    assert script.total_duration_s == 14
    assert script.dialogue_shot.id == 2
    assert script.violations() == []


def test_total_duration_bounds() -> None:
    data = make_script()
    data["shots"][0]["duration_s"] = 30
    with pytest.raises(ValidationError, match="total duration"):
        Script.model_validate(data)


def test_shot_ids_must_be_sequential() -> None:
    data = make_script()
    data["shots"][1]["id"] = 1
    with pytest.raises(ValidationError, match="shot ids"):
        Script.model_validate(data)


def test_exactly_one_dialogue_shot() -> None:
    data = make_script()
    data["shots"][0]["has_dialogue"] = True
    with pytest.raises(ValidationError, match="exactly one shot"):
        Script.model_validate(data)


def test_dialogue_shot_id_must_match() -> None:
    data = make_script()
    data["dialogue"]["shot_id"] = 1
    with pytest.raises(ValidationError, match=r"dialogue\.shot_id"):
        Script.model_validate(data)


def test_dialogue_must_end_within_shot() -> None:
    data = make_script()
    data["dialogue"]["end_s"] = 9.0
    with pytest.raises(ValidationError, match="ends at"):
        Script.model_validate(data)


def test_speech_rate_is_not_a_domain_invariant() -> None:
    # The "window long enough for the line" check needs the configured speech rate; it lives in
    # quality.script_checks (see test_script_checks.py), so domain accepts the structure.
    data = make_script()
    data["dialogue"]["text"] = "x" * 120
    data["dialogue"]["start_s"] = 1.0
    data["dialogue"]["end_s"] = 3.0
    assert Script.model_validate(data).violations() == []


def test_dialogue_window_positive() -> None:
    with pytest.raises(ValidationError, match="positive"):
        Dialogue(
            shot_id=1,
            speaker="a",
            text="hi",
            language="en",
            emotion="x",
            start_s=3.0,
            end_s=3.0,
        )


def test_extra_fields_rejected() -> None:
    data = make_script()
    data["shots"][0]["unexpected"] = 1
    with pytest.raises(ValidationError):
        Script.model_validate(data)


def test_with_dialogue_text_returns_copy() -> None:
    script = Script.model_validate(make_script())
    updated = script.with_dialogue_text("Original line here")
    assert updated.dialogue.text == "Original line here"
    assert script.dialogue.text != updated.dialogue.text


def test_shot_duration_bounds() -> None:
    with pytest.raises(ValidationError):
        Shot(
            id=1,
            duration_s=3,
            visual_prompt="long enough prompt",
            camera="c",
            continuity="c",
            ambient_audio="a",
            has_dialogue=False,
        )


def test_idea_input_strips_and_validates() -> None:
    idea = IdeaInput(text='   A taxi driver asks: "Is this it?"   ')
    assert idea.text.startswith("A taxi")
    with pytest.raises(ValidationError):
        IdeaInput(text="   short   ")


def test_idea_input_errors_are_human_readable() -> None:
    with pytest.raises(ValidationError) as short:
        IdeaInput(text="   short   ")
    error = short.value.errors()[0]
    assert error["type"] == "idea_too_short"
    assert error["msg"] == (
        "The idea is too short (5 characters); write at least 10: "
        "1-3 sentences with the spoken line in quotes."
    )
    with pytest.raises(ValidationError) as long:
        IdeaInput(text="x" * (MAX_IDEA_CHARS + 1))
    assert long.value.errors()[0]["type"] == "idea_too_long"
    assert "too long (1001 characters)" in long.value.errors()[0]["msg"]
    assert len(IdeaInput(text="  " + "x" * MAX_IDEA_CHARS + "  ").text) == MAX_IDEA_CHARS


def test_idea_input_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra"):
        IdeaInput.model_validate({"text": "A long enough idea.", "lang": "uk"})


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("No punctuation at all", 1),
        ("One. Two! Three? Four…", 4),
        (
            "In a glass meeting room, a stern hiring manager finishes an AI-made video and "
            "smiles for the first time today. "
            'She says: "Cancel the other interviews, Roman gets the job."',
            2,
        ),
        ("Dr. Smith checks the chart at 3.5 hours past midnight. He sighs.", 2),
        ('"Run!" he shouted. Then silence.', 2),
        ("Wait... no. Yes!", 2),
        ("J. K. Rowling writes at 7 a.m. every day.", 1),
        ("Apples, pears, e.g. Fuji. Then rain.", 2),
        ("Дощ, нічне місто. Таксист питає: «Ти впевнена, що це та адреса?» і чекає.", 2),
        ("Вона йде на вул. Хрещатик. Там тихо.", 2),
        ("   ", 0),
    ],
)
def test_count_sentences(text: str, expected: int) -> None:
    assert count_sentences(text) == expected


def test_idea_sentence_count_property() -> None:
    idea = IdeaInput(text="One sentence. Two sentences. Three. Four sentences now.")
    assert idea.sentence_count == 4


def test_voice_gender_defaults_to_female_for_old_scripts() -> None:
    data = make_script()
    assert "voice_gender" not in data["dialogue"]
    script = Script.model_validate_json(json.dumps(data))
    assert script.dialogue.voice_gender is VoiceGender.FEMALE
    data["dialogue"]["voice_gender"] = "male"
    assert Script.model_validate(data).dialogue.voice_gender is VoiceGender.MALE
    data["dialogue"]["voice_gender"] = "robot"
    with pytest.raises(ValidationError, match="voice_gender"):
        Script.model_validate(data)


def test_voice_gender_is_required_in_the_llm_schema() -> None:
    # Strict structured outputs want every property required and no defaults in the schema.
    schema = Script.model_json_schema()
    dialogue = schema["$defs"]["Dialogue"]
    assert "voice_gender" in dialogue["required"]
    assert dialogue["properties"]["voice_gender"] == {"$ref": "#/$defs/VoiceGender"}
    assert schema["$defs"]["VoiceGender"]["enum"] == ["female", "male"]
    assert "default" not in json.dumps(schema)


def test_attempt_status_values() -> None:
    assert {s.value for s in AttemptStatus} == {
        "generated",
        "failed",
        "passed",
        "unverified",
        "accepted",
        "abandoned",
        "superseded",
    }


def test_script_window_and_shot_helpers_revalidate() -> None:
    script = Script.model_validate(make_script())
    wider = script.with_dialogue_window(1.0, 7.0)
    assert (wider.dialogue.start_s, wider.dialogue.end_s) == (1.0, 7.0)
    longer = script.with_shot_duration(1, 9)
    assert longer.shots[0].duration_s == 9 and longer.shots[1] == script.shots[1]
    with pytest.raises(ValueError, match="dialogue ends"):
        script.with_dialogue_window(2.0, 9.0)  # beyond the 8 s dialogue shot
    with pytest.raises(ValueError, match="total duration"):
        script.with_shot_duration(2, 30)  # 6 + 30 > 30


@pytest.mark.parametrize(
    ("given", "stored"),
    [("en", "en"), ("EN", "en"), ("en-US", "en"), ("uk_UA", "uk"), ("ua", "uk")],
)
def test_language_is_normalized_to_an_iso_code(given: str, stored: str) -> None:
    """The code picks the comparison rules and is sent as is to STT/TTS: "EN" or "en-US" are
    stored as "en" so every consumer sees the same value."""
    data = make_script(language=given)
    data["dialogue"]["language"] = given
    script = Script.model_validate(data)
    assert (script.language, script.dialogue.language) == (stored, stored)


@pytest.mark.parametrize("given", ["English", "eng", "e1", "  "])
def test_language_that_is_not_an_iso_code_is_rejected(given: str) -> None:
    """ "English" would silently switch the comparison to non-English rules and reach the STT and
    TTS vendors as is; as a validation error it goes back to the LLM instead."""
    data = make_script()
    data["dialogue"]["language"] = given
    with pytest.raises(ValidationError, match="ISO 639-1"):
        Script.model_validate(data)
