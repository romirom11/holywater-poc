from pathlib import Path

import pytest

from holywater.config.schema import (
    JudgeConfig,
    LLMConfig,
    PreflightConfig,
    QualityConfig,
    STTConfig,
)
from holywater.domain.enums import AudioStrategy, FailureKind
from holywater.domain.media import MediaInfo
from holywater.domain.script import Script
from holywater.observability.errors import ScriptError
from holywater.prompts import load_prompt
from holywater.providers.base import NoopJudge
from holywater.providers.fake import FakeJudge, FakeLLM, FakeSTT, fake_idea
from holywater.stages.generate_video import build_shot_prompt
from holywater.stages.preflight_script import PreflightDeps, preflight_script
from holywater.stages.verify_speech import VerifyDeps, classify_failure, verify_speech
from holywater.stages.write_script import ScriptWriterDeps, write_script
from tests.unit.test_domain import make_script

# -- write_script ------------------------------------------------------------------------------


def test_write_script_uses_verbatim_line_from_input() -> None:
    idea = fake_idea('Rainy night. A taxi driver asks: "Are you SURE this is the address?"')
    result = write_script(idea, ScriptWriterDeps(llm=FakeLLM(), config=LLMConfig(model="m")))
    assert result.attempts == 1
    assert result.script.dialogue.text == "Are you SURE this is the address?"
    assert result.prompt_version == "scriptwriter@1"


def test_write_script_retries_on_paraphrase_then_succeeds() -> None:
    idea = fake_idea()
    llm = FakeLLM(line_override="Is this really the place?", override_times=1)
    result = write_script(
        idea, ScriptWriterDeps(llm=llm, config=LLMConfig(model="m", max_attempts=3))
    )
    assert result.attempts == 2
    assert llm.calls == 2
    assert result.script.dialogue.text == "Are you sure this is the address?"


def test_write_script_exhausts_attempts() -> None:
    llm = FakeLLM(line_override="Is this really the place?", override_times=10)
    with pytest.raises(ScriptError) as exc_info:
        write_script(
            fake_idea(), ScriptWriterDeps(llm=llm, config=LLMConfig(model="m", max_attempts=2))
        )
    assert llm.calls == 2
    assert "verbatim" in exc_info.value.context["last_feedback"][0]


# -- preflight ---------------------------------------------------------------------------------


def _script() -> Script:
    return Script.model_validate(make_script())


def test_preflight_skipped_without_judge() -> None:
    result = preflight_script(_script(), PreflightDeps(judge=NoopJudge(), config=JudgeConfig()))
    assert result.ok and result.skipped


def test_preflight_passes_with_permissive_judge() -> None:
    judge = FakeJudge()
    result = preflight_script(_script(), PreflightDeps(judge=judge, config=JudgeConfig()))
    assert result.ok and not result.skipped
    assert len(result.decisions) == 2 * 2 + 1  # safety + filmability per shot, + line fits


def test_preflight_flags_safety_and_filmability() -> None:
    judge = FakeJudge(bool_answers={"safety_risk": 0.9}, score_answers={"filmability": 2})
    cfg = JudgeConfig(preflight=PreflightConfig(safety_risk_threshold=0.7, min_filmability_score=3))
    result = preflight_script(_script(), PreflightDeps(judge=judge, config=cfg))
    assert not result.ok
    assert any("safety" in i for i in result.issues)
    assert any("filmability" in i for i in result.issues)


# -- video prompt ------------------------------------------------------------------------------


def test_build_shot_prompt_variants() -> None:
    script = _script()
    prompt = load_prompt("seedance_shot")
    dialogue_shot = script.dialogue_shot
    native = build_shot_prompt(script, dialogue_shot, strategy=AudioStrategy.NATIVE, prompt=prompt)
    assert f'says "{script.dialogue.text}"' in native.text
    assert native.version.endswith("/dialogue")
    strict = build_shot_prompt(
        script, dialogue_shot, strategy=AudioStrategy.NATIVE, prompt=prompt, strict=True
    )
    assert "do not add, drop" in strict.text and strict.version.endswith("/dialogue_strict")
    overlay = build_shot_prompt(
        script, dialogue_shot, strategy=AudioStrategy.TTS_OVERLAY, prompt=prompt
    )
    assert "no dialogue" in overlay.text and script.dialogue.text not in overlay.text
    silent = build_shot_prompt(
        script, script.shots[0], strategy=AudioStrategy.NATIVE, prompt=prompt
    )
    assert "No one speaks" in silent.text


# -- verify_speech -----------------------------------------------------------------------------


class _StubFFmpeg:
    """Unit tests never spawn ffmpeg: audio extraction just creates an empty wav file."""

    def extract_audio(self, src: Path, dst: Path, *, sample_rate: int = 16000) -> MediaInfo:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"")
        return MediaInfo(
            duration_s=1.0,
            width=0,
            height=0,
            fps=0.0,
            has_video=False,
            has_audio=True,
            size_bytes=0,
        )


def _verify_deps(stt: FakeSTT, judge: FakeJudge | NoopJudge) -> VerifyDeps:
    return VerifyDeps(
        stt=stt,
        judge=judge,
        ffmpeg=_StubFFmpeg(),  # type: ignore[arg-type]
        stt_config=STTConfig(model="fake-stt"),
        quality=QualityConfig(max_wer=0.1),
        judge_config=JudgeConfig(),
    )


def test_verify_speech_pass(tmp_path: Path) -> None:
    deps = _verify_deps(FakeSTT(queue=["are you sure this is the address"]), NoopJudge())
    result = verify_speech(
        tmp_path / "x.mp4",
        shot_id=2,
        attempt=1,
        expected="Are you sure this is the address?",
        language="en",
        deps=deps,
    )
    assert result.verification.passed and result.verification.wer == 0.0
    assert result.verification.failure_kind is None


def test_verify_speech_fail_classified_by_judge(tmp_path: Path) -> None:
    judge = FakeJudge(choice_answers={"failure_kind": "wrong_language"})
    deps = _verify_deps(FakeSTT(queue=["ти впевнений що це та адреса"]), judge)
    result = verify_speech(
        tmp_path / "x.mp4",
        shot_id=2,
        attempt=1,
        expected="Are you sure this is the address?",
        language="en",
        deps=deps,
    )
    assert not result.verification.passed
    assert result.verification.failure_kind is FailureKind.WRONG_LANGUAGE
    assert result.decision is not None and result.decision.answer == "wrong_language"


def test_classify_failure_heuristics_without_judge() -> None:
    deps = _verify_deps(FakeSTT(), NoopJudge())
    assert classify_failure("one two three four", "", "en", deps)[0] is FailureKind.SILENCE
    assert classify_failure("one two three four", "one", "en", deps)[0] is FailureKind.TRUNCATED
    assert (
        classify_failure("one two", "one two three four five", "en", deps)[0]
        is FailureKind.EXTRA_WORDS
    )
    assert classify_failure("one two three", "one two tree", "en", deps)[0] is FailureKind.UNKNOWN
