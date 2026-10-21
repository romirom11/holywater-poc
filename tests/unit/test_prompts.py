import pytest

from holywater.observability.errors import ConfigError
from holywater.prompts import load_prompt


def test_scriptwriter_prompt_loads_and_renders() -> None:
    prompt = load_prompt("scriptwriter")
    assert prompt.version_tag == "scriptwriter@1"
    system = prompt.render("system")
    assert "VERBATIM" in system
    user = prompt.render("user", idea_text="An idea.", feedback_block="")
    assert "<idea>\nAn idea.\n</idea>" in user


def test_missing_placeholder_is_config_error() -> None:
    prompt = load_prompt("scriptwriter")
    with pytest.raises(ConfigError, match="placeholder"):
        prompt.render("user", idea_text="x")


def test_unknown_prompt_or_section() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_prompt("does_not_exist")
    with pytest.raises(ConfigError, match="no section"):
        load_prompt("scriptwriter").render("nope")


def test_seedance_prompt_has_all_sections() -> None:
    prompt = load_prompt("seedance_shot")
    assert set(prompt.sections) == {"dialogue", "no_dialogue", "strict"}
