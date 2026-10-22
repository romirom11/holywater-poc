import pytest

from holywater.observability.errors import ConfigError
from holywater.prompts import load_prompt


def test_scriptwriter_prompt_loads_and_renders() -> None:
    prompt = load_prompt("scriptwriter")
    assert prompt.version_tag == "scriptwriter@3"
    system = prompt.render("system", chars_per_s="15")
    assert "VERBATIM" in system
    user = prompt.render("user", idea_text="An idea.", feedback_block="")
    assert "<idea>\nAn idea.\n</idea>" in user


def test_scriptwriter_system_prompt_states_the_checked_rules() -> None:
    """The rules the code checks are spelled out, so the LLM does not burn paid retries on them:
    the speech rate comes from config, one complete quote, voice_gender."""
    system = load_prompt("scriptwriter").render("system", chars_per_s="13.5")
    assert "1 second per 13.5 characters" in system
    assert "12 characters" not in system  # the old hard-coded rate is gone
    assert "exactly one complete quoted fragment" in system
    assert "nothing cut" in system
    assert "`dialogue.voice_gender`" in system and '"female" or "male"' in system
    with pytest.raises(ConfigError, match="placeholder"):
        load_prompt("scriptwriter").render("system")


def test_scriptwriter_rewrite_section_carries_previous_script_and_issues() -> None:
    rewrite = load_prompt("scriptwriter").render(
        "rewrite", issues="- shot 1: too vague", previous='{"title": "T"}'
    )
    assert "- shot 1: too vague" in rewrite
    assert '{"title": "T"}' in rewrite
    assert "keep the\ndialogue line exactly as it is" in rewrite


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
    assert prompt.version_tag == "seedance_shot@2"
    assert set(prompt.sections) == {
        "dialogue",
        "lead_in",
        "lead_in_at_start",
        "tail",
        "no_dialogue",
        "strict",
    }
