import json
from pathlib import Path

import pytest
from pydantic import BaseModel
from structlog.testing import capture_logs

from holywater.config.schema import (
    FailureClassifierConfig,
    JudgeConfig,
    LLMConfig,
    PreflightConfig,
    QualityConfig,
    STTConfig,
    VideoConfig,
)
from holywater.domain.artifacts import ProviderCost
from holywater.domain.enums import AudioStrategy, FailureKind, Resolution
from holywater.domain.media import MediaInfo
from holywater.domain.script import Script
from holywater.observability.errors import (
    BilledJobError,
    RetryableProviderError,
    ScriptError,
    StructuredOutputError,
)
from holywater.prompts import load_prompt
from holywater.providers.base import LLMResult, NoopJudge, STTRequest, VideoRequest, VideoResult
from holywater.providers.fake import (
    FakeJudge,
    FakeLLM,
    FakeSTT,
    Speech,
    fake_idea,
    read_speech,
    write_sidecar,
)
from holywater.stages.generate_video import VideoDeps, build_shot_prompt, generate_shot, is_strict
from holywater.stages.preflight_script import PreflightDeps, preflight_script
from holywater.stages.verify_speech import (
    VerifyDeps,
    check_stray_speech,
    classify_failure,
    verify_speech,
)
from holywater.stages.write_script import (
    ScriptWriterDeps,
    rewrite_script,
    soften_shot,
    write_script,
)
from tests.unit.test_domain import make_script

# -- write_script ------------------------------------------------------------------------------


def test_write_script_uses_verbatim_line_from_input() -> None:
    idea = fake_idea('Rainy night. A taxi driver asks: "Are you SURE this is the address?"')
    result = write_script(idea, ScriptWriterDeps(llm=FakeLLM(), config=LLMConfig(model="m")))
    assert result.attempts == 1
    assert result.script.dialogue.text == "Are you SURE this is the address?"
    assert result.prompt_version == "scriptwriter@3"


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


class _BilledGarbageLLM:
    """Raises the queued billed `StructuredOutputError`s first, then answers like `FakeLLM`."""

    def __init__(self, *errors: StructuredOutputError) -> None:
        self.errors = list(errors)
        self.users: list[str] = []
        self.inner = FakeLLM()

    def complete_structured[T: BaseModel](
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type[T],
        temperature: float | None = None,
    ) -> LLMResult[T]:
        self.users.append(user)
        if self.errors:
            raise self.errors.pop(0)
        result = self.inner.complete_structured(
            model=model, system=system, user=user, schema=schema
        )
        return LLMResult(
            data=result.data, model=result.model, cost_usd=0.02, cost_is_estimate=False
        )


def test_write_script_counts_billed_unusable_answers() -> None:
    truncated = StructuredOutputError(
        "cut off",
        provider="openrouter",
        finish_reason="length",
        spent_usd=0.03,
        spent_is_estimate=False,
    )
    invalid = StructuredOutputError("not the schema", provider="openrouter", spent_usd=0.01)
    llm = _BilledGarbageLLM(truncated, invalid)
    result = write_script(
        fake_idea(), ScriptWriterDeps(llm=llm, config=LLMConfig(model="m", max_attempts=3))
    )
    assert result.attempts == 3
    assert result.cost_usd == pytest.approx(0.03 + 0.01 + 0.02)  # billed garbage is counted
    assert result.cost_is_estimate  # the second error carried an estimate
    assert "cut off before the JSON was complete" in llm.users[1]  # finish_reason=length feedback
    assert "did not match the schema: not the schema" in llm.users[2]


def test_write_script_checks_the_window_at_the_configured_speech_rate() -> None:
    # FakeLLM gives "Are you sure this is the address?" (33 chars) a 2.7 s window.
    slow = ScriptWriterDeps(
        llm=FakeLLM(), config=LLMConfig(model="m", max_attempts=1), chars_per_s=5
    )
    with pytest.raises(ScriptError) as exc_info:
        write_script(fake_idea(), slow)
    assert "at 5 chars/s" in exc_info.value.context["last_feedback"][0]
    default = ScriptWriterDeps(llm=FakeLLM(), config=LLMConfig(model="m", max_attempts=1))
    assert write_script(fake_idea(), default).attempts == 1


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


def test_preflight_skipped_when_judge_unavailable_mid_run() -> None:
    judge = FakeJudge(available=False)
    result = preflight_script(_script(), PreflightDeps(judge=judge, config=JudgeConfig()))
    assert result.ok and result.skipped and len(judge.calls) == 5


def test_soften_shot_replaces_only_the_rejected_shot() -> None:
    from holywater.stages.write_script import soften_shot

    script = _script()
    llm = FakeLLM()
    result = soften_shot(
        script,
        1,
        reason="content policy: weapons",
        idea=fake_idea(),
        deps=ScriptWriterDeps(llm=llm, config=LLMConfig(model="m")),
    )
    assert llm.calls == 1
    assert result.script.shots[0].visual_prompt == "Establishing shot derived from the idea."
    assert result.script.shots[1] == script.shots[1]  # the dialogue shot is untouched
    assert result.script.dialogue == script.dialogue  # and so is the verbatim line


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

    def extract_audio(
        self,
        src: Path,
        dst: Path,
        *,
        sample_rate: int = 16000,
        start_s: float | None = None,
        duration_s: float | None = None,
    ) -> MediaInfo:
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


# -- write_script: every billed call reaches the cost sink -------------------------------------


def test_write_script_reports_every_billed_call_even_when_it_ends_in_script_error() -> None:
    llm = FakeLLM(
        line_override="Is this really the place?", override_times=99, invalid_times=1, cost_usd=0.01
    )
    seen: list[ProviderCost] = []
    deps = ScriptWriterDeps(
        llm=llm, config=LLMConfig(model="m", max_attempts=3), on_cost=seen.append
    )
    with pytest.raises(ScriptError) as exc_info:
        write_script(fake_idea(), deps)
    # 1 unusable answer + 2 paraphrases: all billed, all reported, none lost with the error
    assert [c.cost_usd for c in seen] == [0.01, 0.01, 0.01]
    assert {c.capability for c in seen} == {"llm"}
    assert exc_info.value.context["cost_usd"] == pytest.approx(0.03)


class _BilledThenTransientLLM(_BilledGarbageLLM):
    def complete_structured[T: BaseModel](
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type[T],
        temperature: float | None = None,
    ) -> LLMResult[T]:
        raise RetryableProviderError(
            "finish_reason=error", provider="openrouter", model="m", spent_usd=0.004
        )


def test_write_script_reports_spend_of_a_failed_call_before_re_raising() -> None:
    """The task retries a RetryableProviderError; its spend must already be recorded by then."""
    seen: list[ProviderCost] = []
    deps = ScriptWriterDeps(
        llm=_BilledThenTransientLLM(), config=LLMConfig(model="m"), on_cost=seen.append
    )
    with pytest.raises(RetryableProviderError):
        write_script(fake_idea(), deps)
    assert [(c.cost_usd, c.model) for c in seen] == [(0.004, "m")]


def test_write_script_renders_the_configured_speech_rate_into_the_system_prompt() -> None:
    llm = _BilledGarbageLLM()
    systems: list[str] = []
    inner = llm.complete_structured

    def spy[T: BaseModel](**kwargs: object) -> LLMResult[T]:
        systems.append(str(kwargs["system"]))
        return inner(**kwargs)  # type: ignore[arg-type]

    llm.complete_structured = spy  # type: ignore[method-assign]
    write_script(
        fake_idea(),
        ScriptWriterDeps(llm=llm, config=LLMConfig(model="m"), chars_per_s=14.5),
    )
    assert "1 second per 14.5 characters" in systems[0]


def test_rewrite_sends_previous_script_and_issues_with_every_attempt() -> None:
    """A pre-flight rewrite revises the previous script; the revision survives inner retries."""
    script = Script.model_validate(make_script())
    llm = _BilledGarbageLLM(StructuredOutputError("not the schema", provider="openrouter"))
    rewrite_script(
        script,
        ["shot 1: the visual prompt scored 2/5 for filmability"],
        idea=fake_idea(),
        deps=ScriptWriterDeps(llm=llm, config=LLMConfig(model="m", max_attempts=2)),
    )
    assert len(llm.users) == 2
    for user in llm.users:
        assert "pre-flight review of your previous script" in user
        assert "- shot 1: the visual prompt scored 2/5 for filmability" in user
        assert '"title": "Rain"' in user  # the previous script itself
    assert "did not match the schema" in llm.users[1]  # plus the reason for the retry


def test_soften_keeps_the_shot_timing_and_only_takes_its_wording() -> None:
    script = Script.model_validate(make_script())  # shot 1 is 6 s; FakeLLM writes 5 s
    result = soften_shot(
        script,
        1,
        reason="content policy",
        idea=fake_idea(),
        deps=ScriptWriterDeps(llm=FakeLLM(), config=LLMConfig(model="m")),
    )
    shot = result.script.shots[0]
    assert shot.visual_prompt == "Establishing shot derived from the idea."  # new wording
    assert shot.duration_s == 6 and not shot.has_dialogue  # old timing


# -- preflight: decisions per shot, unknown scores ---------------------------------------------


def test_preflight_checks_carry_shot_and_decision() -> None:
    judge = FakeJudge(score_answers={"filmability": 2})
    result = preflight_script(_script(), PreflightDeps(judge=judge, config=JudgeConfig()))
    by_key = {(c.question, c.shot_id): c.decision for c in result.checks}
    assert by_key == {
        ("safety_risk", 1): "pass",
        ("filmability", 1): "rewrite",
        ("safety_risk", 2): "pass",
        ("filmability", 2): "rewrite",
        ("line_fits_window", None): "pass",
    }
    assert not result.ok


def test_preflight_unparsable_score_is_unknown_not_the_best_score() -> None:
    judge = FakeJudge(score_answers={"filmability": "unknown"})
    with capture_logs() as logs:
        result = preflight_script(_script(), PreflightDeps(judge=judge, config=JudgeConfig()))
    film = [c for c in result.checks if c.question == "filmability"]
    assert [c.decision for c in film] == ["unavailable", "unavailable"]
    assert result.ok and not result.skipped  # neither blocks nor counts as a pass
    assert [e["event"] for e in logs].count("preflight.unparsable_score") == 2
    assert all(
        e["log_level"] == "warning" for e in logs if e["event"] == "preflight.unparsable_score"
    )


def test_preflight_skip_reasons() -> None:
    disabled = JudgeConfig(preflight=PreflightConfig(enabled=False))
    off = preflight_script(_script(), PreflightDeps(judge=FakeJudge(), config=disabled))
    assert off.skipped and off.skip_reason == "disabled" and not off.checks
    noop = preflight_script(_script(), PreflightDeps(judge=NoopJudge(), config=JudgeConfig()))
    assert noop.skipped and noop.skip_reason == "noop"
    down = preflight_script(
        _script(), PreflightDeps(judge=FakeJudge(available=False), config=JudgeConfig())
    )
    assert down.skipped and down.skip_reason == "unavailable"
    assert {c.decision for c in down.checks} == {"unavailable"}


# -- video prompt: no empty timeline segments --------------------------------------------------


def test_dialogue_prompt_omits_empty_segments() -> None:
    script = Script.model_validate(make_script()).with_dialogue_window(0.0, 8.0)
    text = build_shot_prompt(
        script,
        script.dialogue_shot,
        strategy=AudioStrategy.NATIVE,
        prompt=load_prompt("seedance_shot"),
    ).text
    assert "0–0s" not in text and "8–8s" not in text
    assert (
        "[Timeline] Driver glances in the mirror at the passenger., handheld close-up. "
        '0–8s: the driver says "Are you sure this is the address?"'
    ) in text
    assert "stays silent" not in text


def test_dialogue_prompt_with_lead_in_and_tail_keeps_the_v1_wording() -> None:
    """start_s > 0 and end_s < duration render exactly what seedance_shot@1 sent."""
    script = Script.model_validate(make_script())
    text = build_shot_prompt(
        script,
        script.dialogue_shot,
        strategy=AudioStrategy.NATIVE,
        prompt=load_prompt("seedance_shot"),
    ).text
    assert (
        "[Timeline] 0–2s: Driver glances in the mirror at the passenger., handheld close-up. "
        '2–6s: the driver says "Are you sure this is the address?" (quiet, uneasy); the '
        "speaker's mouth moves only during this line, no other speech. 6–8s: the shot holds, "
        "the driver stays silent.\n"
    ) in text


def test_is_strict() -> None:
    assert is_strict("seedance_shot@2/dialogue_strict")
    assert not is_strict("seedance_shot@2/dialogue")


# -- generate_shot: charges reach the sink before anything can fail ---------------------------


class _StubVideo:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def generate(self, request: VideoRequest, *, dest: Path) -> VideoResult:
        if self.error is not None:
            raise self.error
        return VideoResult(path=dest, model="v", cost_usd=0.62, cost_is_estimate=False)


class _BrokenProbeFFmpeg:
    def probe(self, path: Path) -> MediaInfo:
        from holywater.observability.errors import MediaError

        raise MediaError("corrupt clip")


def test_generate_shot_reports_the_charge_before_probing(tmp_path: Path) -> None:
    from holywater.observability.errors import MediaError

    script = Script.model_validate(make_script())
    seen: list[ProviderCost] = []
    deps = VideoDeps(
        video=_StubVideo(),
        ffmpeg=_BrokenProbeFFmpeg(),  # type: ignore[arg-type]
        config=VideoConfig(model="v", resolution=Resolution.P480),
    )
    # A paid clip ffprobe cannot read is a BilledJobError (not retried by the task, which would
    # buy a new job): the flow abandons the attempt within its own budget.
    with pytest.raises(BilledJobError, match="unreadable") as exc_info:
        generate_shot(
            script,
            script.shots[0],
            deps,
            dest=tmp_path / "a.mp4",
            attempt=1,
            strategy=AudioStrategy.NATIVE,
            on_cost=seen.append,
        )
    err = exc_info.value
    assert isinstance(err.__cause__, MediaError) and not err.retryable
    assert (err.spent_usd, err.spent_is_estimate, err.model) == (0.62, False, "v")
    # The charge is reported once, before the probe; the error does not report it again.
    assert [(c.capability, c.cost_usd, c.is_estimate, c.shot_id) for c in seen] == [
        ("video", 0.62, False, 1)
    ]


def test_generate_shot_reports_spend_carried_by_an_error(tmp_path: Path) -> None:
    script = Script.model_validate(make_script())
    seen: list[ProviderCost] = []
    error = BilledJobError("poll deadline", provider="fake", job_id="j1", spent_usd=0.5)
    deps = VideoDeps(
        video=_StubVideo(error),
        ffmpeg=_BrokenProbeFFmpeg(),  # type: ignore[arg-type]
        config=VideoConfig(model="v"),
    )
    with pytest.raises(BilledJobError):
        generate_shot(
            script,
            script.shots[1],
            deps,
            dest=tmp_path / "b.mp4",
            attempt=2,
            strategy=AudioStrategy.NATIVE,
            on_cost=seen.append,
        )
    assert [(c.cost_usd, c.units, c.shot_id) for c in seen] == [(0.5, 8.0, 2)]
    unpaid = VideoDeps(
        video=_StubVideo(RetryableProviderError("503", provider="fake")),
        ffmpeg=_BrokenProbeFFmpeg(),  # type: ignore[arg-type]
        config=VideoConfig(model="v"),
    )
    with pytest.raises(RetryableProviderError):
        generate_shot(
            script,
            script.shots[1],
            unpaid,
            dest=tmp_path / "c.mp4",
            attempt=3,
            strategy=AudioStrategy.NATIVE,
            on_cost=seen.append,
        )
    assert len(seen) == 1  # nothing was spent, nothing reported


# -- verify_speech: mute clips, classifier provenance, stray speech ----------------------------


def test_verify_speech_on_a_clip_without_audio_is_silence_without_stt(tmp_path: Path) -> None:
    stt = FakeSTT()
    result = verify_speech(
        tmp_path / "mute.mp4",
        shot_id=2,
        attempt=1,
        expected="Are you sure this is the address?",
        language="en",
        deps=_verify_deps(stt, NoopJudge()),
        has_audio=False,
    )
    v = result.verification
    assert (v.passed, v.verified, v.failure_kind, v.transcript) == (
        False,
        True,
        FailureKind.SILENCE,
        "",
    )
    assert stt.calls == 0 and v.stt_model is None


def test_classify_failure_records_how_the_kind_was_reached() -> None:
    expected, heard = "one two three four", "one"
    jev = _verify_deps(FakeSTT(), FakeJudge(choice_answers={"failure_kind": "paraphrased"}))
    kind, steps = classify_failure(expected, heard, "en", jev)
    assert kind is FailureKind.PARAPHRASED and [s.decision for s in steps] == ["classified"]

    down = _verify_deps(FakeSTT(), FakeJudge(available=False))
    kind, steps = classify_failure(expected, heard, "en", down)
    assert kind is FailureKind.TRUNCATED
    assert [s.decision for s in steps] == ["unavailable", "heuristic"]
    assert steps[1].answer.model_version == "heuristic" and steps[1].answer.answer == "truncated"

    off = _verify_deps(FakeSTT(), FakeJudge())
    off.judge_config = JudgeConfig(failure_classifier=FailureClassifierConfig(enabled=False))
    kind, steps = classify_failure(expected, heard, "en", off)
    assert [s.decision for s in steps] == ["heuristic"]


def test_check_stray_speech_counts_words(tmp_path: Path) -> None:
    deps = _verify_deps(FakeSTT(queue=["", "Hey, you! Wait for me."]), NoopJudge())
    quiet = check_stray_speech(
        tmp_path / "a.mp4", shot_id=1, attempt=1, language="en", max_words=2, deps=deps
    )
    assert not quiet.found and quiet.words == 0 and quiet.verification.expected == ""
    talky = check_stray_speech(
        tmp_path / "a.mp4", shot_id=1, attempt=2, language="en", max_words=2, deps=deps
    )
    assert talky.found and talky.words == 5
    assert talky.verification.failure_kind is FailureKind.EXTRA_WORDS
    assert talky.verification.transcript == "Hey, you! Wait for me."


def test_verify_speech_records_the_transcribed_segment(tmp_path: Path) -> None:
    deps = _verify_deps(FakeSTT(queue=["x"]), NoopJudge())
    verify_speech(
        tmp_path / "final.mp4",
        shot_id=2,
        attempt=1,
        expected="x",
        language="en",
        deps=deps,
        window=(5.5, 4.0),
    )
    segment = json.loads((tmp_path / "final.stt.wav.json").read_text())
    assert segment == {"source": str(tmp_path / "final.mp4"), "start_s": 5.5, "duration_s": 4.0}


# -- fakes: STT honours the transcribed window ------------------------------------------------


def test_fake_stt_hears_only_the_words_inside_the_window(tmp_path: Path) -> None:
    media = tmp_path / "final.mp4"
    media.write_bytes(b"")
    write_sidecar(media, "one two three four", start_s=2.0, end_s=6.0)  # 1 s per word
    wav = tmp_path / "final.stt.wav"
    stt = FakeSTT()

    def heard(start_s: float, duration_s: float | None) -> str:
        wav.with_name(wav.name + ".json").write_text(
            json.dumps({"start_s": start_s, "duration_s": duration_s})
        )
        return stt.transcribe(STTRequest(model="m", audio_path=wav, source_path=media)).text

    assert heard(0.0, None) == "one two three four"
    assert heard(1.5, 5.0) == "one two three four"
    assert heard(1.5, 3.5) == "one two three"  # the window ends before the last word
    assert heard(4.0, 1.0) == "three"
    assert read_speech(media) == Speech("one two three four", 2.0, 6.0)


def test_unreported_lipsync_costs_are_collected_exactly_once(tmp_path: Path) -> None:
    """A paid lipsync result whose assemble failed: the flow collects its cost from the ledger,
    and neither a second collection nor a later final reports it again."""
    from holywater.stages.assemble import collect_unreported_costs

    cost = ProviderCost(
        capability="lipsync", model="m", units=6.0, cost_usd=0.4, is_estimate=False, shot_id=2
    )
    (tmp_path / "lipsync_abc.json").write_text(
        json.dumps({"cost": cost.model_dump(mode="json"), "reported": False})
    )
    (tmp_path / "lipsync_old.json").write_text(
        json.dumps({"cost": cost.model_dump(mode="json"), "reported": True})
    )
    assert collect_unreported_costs(tmp_path) == [cost]
    assert collect_unreported_costs(tmp_path) == []
    assert json.loads((tmp_path / "lipsync_abc.json").read_text())["reported"] is True
