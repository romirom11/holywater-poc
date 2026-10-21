from typing import Any

import pytest
from pydantic import ValidationError

from holywater.domain.idea import IdeaInput
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


def test_dialogue_window_must_fit_speech() -> None:
    data = make_script()
    data["dialogue"]["text"] = "x" * 120
    data["dialogue"]["start_s"] = 1.0
    data["dialogue"]["end_s"] = 3.0
    with pytest.raises(ValidationError, match="shorter than"):
        Script.model_validate(data)


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
