import pytest

from holywater.domain.idea import IdeaInput
from holywater.domain.media import MediaInfo
from holywater.domain.script import Script
from holywater.observability.errors import QualityGateError
from holywater.quality.gates import DurationGate, MediaIntegrityGate, SpeechGate
from holywater.quality.script_checks import check_script_against_idea
from tests.unit.test_domain import make_script


def test_duration_gate_bounds_and_tolerance() -> None:
    gate = DurationGate(min_s=8, max_s=30, tolerance_s=0.5)
    assert gate.check(7.6).passed
    assert gate.check(30.4).passed
    assert not gate.check(7.4).passed
    assert not gate.check(31).passed


def test_speech_gate_pass_and_fail() -> None:
    gate = SpeechGate(max_wer=0.1)
    ok = gate.check("Are you sure this is the address?", "are you sure this is the address")
    assert ok.passed and ok.measured == 0.0
    bad = gate.check("Are you sure this is the address?", "are you certain that's the place")
    assert not bad.passed
    with pytest.raises(QualityGateError) as exc_info:
        bad.raise_if_failed(stage="verify_speech", run_id="r1")
    assert exc_info.value.gate == "speech"
    assert exc_info.value.context["transcript"].startswith("are you certain")


def _info(**overrides: object) -> MediaInfo:
    base: dict[str, object] = {
        "duration_s": 10.0,
        "width": 720,
        "height": 1280,
        "fps": 24.0,
        "has_video": True,
        "has_audio": True,
        "size_bytes": 1000,
    }
    base.update(overrides)
    return MediaInfo.model_validate(base)


def test_media_integrity_gate() -> None:
    gate = MediaIntegrityGate()
    assert gate.check(_info()).passed
    assert "no audio" in str(gate.check(_info(has_audio=False)).measured)
    assert "not vertical" in str(gate.check(_info(width=1280, height=720)).measured)
    assert MediaIntegrityGate(require_vertical=False).check(_info(width=1280, height=720)).passed


def test_script_check_replaces_line_with_original_text() -> None:
    idea = IdeaInput(text='Rainy night. The driver asks: "Are you SURE this is the address?"')
    script = Script.model_validate(make_script())
    assert script.dialogue.text == "Are you sure this is the address?"
    result = check_script_against_idea(script, idea)
    assert result.ok
    assert result.script.dialogue.text == "Are you SURE this is the address?"


def test_script_check_reports_paraphrase() -> None:
    idea = IdeaInput(text='Rainy night. The driver asks: "Is this really the place?"')
    script = Script.model_validate(make_script())
    result = check_script_against_idea(script, idea)
    assert not result.ok
    assert "verbatim" in result.violations[0]
