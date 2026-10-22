"""build_report from DB rows only (no flow, no ffmpeg)."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlmodel import Session, select

from holywater.config import PipelineConfig, load_config
from holywater.domain.artifacts import SpeechVerification, VideoArtifact
from holywater.domain.enums import (
    AudioStrategy,
    FailureKind,
    Resolution,
    RunStatus,
    StageName,
    StageStatus,
)
from holywater.domain.idea import IdeaInput
from holywater.observability.errors import QualityGateError, ScriptError
from holywater.stages.report import build_report
from holywater.storage import RunRepository, init_db, make_engine, new_run_id
from holywater.storage.models import Stage
from holywater.storage.repo import STRAY_SPEECH_EXPECTED

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config" / "pipeline.yaml"
LINE = "Cancel the other interviews, Roman gets the job."


@pytest.fixture
def repo(tmp_path: Path) -> RunRepository:
    engine = make_engine(tmp_path / "report.db")
    init_db(engine)
    return RunRepository(engine)


@pytest.fixture
def config() -> PipelineConfig:
    return load_config(REPO_CONFIG)


def _run(repo: RunRepository, *, dry_run: bool = False, two_shots: bool = True) -> str:
    run_id = new_run_id()
    repo.create_run(
        run_id,
        IdeaInput(text=f'A hiring manager smiles. She says: "{LINE}"'),
        config_hash="h",
        audio_strategy=AudioStrategy.NATIVE,
        dry_run=dry_run,
    )
    if two_shots:
        repo.upsert_shot(run_id, 1, planned_duration_s=6, has_dialogue=False)
    repo.upsert_shot(run_id, 2, planned_duration_s=7, has_dialogue=True)
    return run_id


def _attempt(repo: RunRepository, run_id: str, shot_id: int, attempt: int) -> int:
    row = repo.add_shot_attempt(
        run_id,
        VideoArtifact(
            shot_id=shot_id,
            attempt=attempt,
            strategy=AudioStrategy.NATIVE,
            path=Path(f"runs/x/video/shot_{shot_id}_a{attempt}.mp4"),
            duration_s=6.0,
            resolution=Resolution.P480,
            has_native_audio=True,
            prompt_version="seedance_shot@2/dialogue" if shot_id == 2 else "seedance_shot@2/no",
            prompt_text="p",
            model="bytedance/seedance-2.5",
            cost_usd=0.62,
        ),
    )
    assert row.id is not None
    return row.id


def _check(
    repo: RunRepository,
    run_id: str,
    attempt_id: int,
    *,
    expected: str = LINE,
    transcript: str = LINE,
    wer: float = 0.0,
    passed: bool = True,
    verified: bool = True,
    kind: FailureKind | None = None,
) -> None:
    repo.add_speech_verification(
        run_id,
        attempt_id,
        SpeechVerification(
            shot_id=1,
            attempt=1,
            expected=expected,
            transcript=transcript,
            wer=wer,
            passed=passed,
            verified=verified,
            failure_kind=kind,
        ),
    )


def _backdate(repo: RunRepository, run_id: str, name: StageName, seconds: float) -> None:
    with Session(repo._engine) as s:
        row = s.exec(select(Stage).where(Stage.run_id == run_id, Stage.name == name.value)).one()
        row.started_at = datetime.now(UTC) - timedelta(seconds=seconds)
        s.add(row)
        s.commit()


def test_report_merges_scriptwriter_version_and_names_the_judge_that_ran(
    repo: RunRepository, config: PipelineConfig
) -> None:
    run_id = _run(repo)
    _attempt(repo, run_id, 2, 1)
    rep = build_report(
        repo,
        run_id,
        config,
        dialogue_line=LINE,
        final=None,
        judge_model="noop",
        prompt_versions=["scriptwriter@3"],
    )
    assert rep.prompt_versions == ["scriptwriter@3", "seedance_shot@2/dialogue"]
    assert rep.models["judge"] == "noop" and rep.models["video"] == config.video.model
    assert rep.dry_run is False


def test_dry_run_report_is_marked_and_names_the_fakes(
    repo: RunRepository, config: PipelineConfig
) -> None:
    run_id = _run(repo, dry_run=True)
    rep = build_report(repo, run_id, config, dialogue_line=None, final=None)
    assert rep.dry_run is True  # read from runs.dry_run
    assert rep.models["video"] == f"fake/{config.video.model}"
    assert rep.models["judge"] == f"fake/{config.judge.model}"
    passed_in = build_report(
        repo, run_id, config, dialogue_line=None, final=None, judge_model="fake/jev-latest"
    )
    assert passed_in.models["judge"] == "fake/jev-latest"  # never double-prefixed


def test_stage_duration_is_the_sum_of_all_entries(
    repo: RunRepository, config: PipelineConfig
) -> None:
    run_id = _run(repo)
    for seconds in (150.0, 160.0):  # one entry per shot
        repo.start_stage(run_id, StageName.GENERATE_VIDEO)
        _backdate(repo, run_id, StageName.GENERATE_VIDEO, seconds)
        repo.finish_stage(run_id, StageName.GENERATE_VIDEO, StageStatus.DONE)
    rep = build_report(repo, run_id, config, dialogue_line=LINE, final=None)
    video = next(s for s in rep.stages if s.name == "generate_video")
    assert video.entries == 2
    assert video.duration_s == pytest.approx(310.0, abs=1.0)
    assert video.last_duration_s == pytest.approx(160.0, abs=1.0)


def test_failure_stage_is_the_innermost_failed_stage(
    repo: RunRepository, config: PipelineConfig
) -> None:
    run_id = _run(repo)
    err = QualityGateError(
        "no audio", gate="speech", measured=None, threshold=0.1, stage="verify_speech"
    )
    repo.start_stage(run_id, StageName.GENERATE_VIDEO)
    repo.start_stage(run_id, StageName.VERIFY_SPEECH)  # nested: starts after its parent
    repo.finish_stage(run_id, StageName.VERIFY_SPEECH, StageStatus.FAILED, error=err)
    repo.finish_stage(run_id, StageName.GENERATE_VIDEO, StageStatus.FAILED, error=err)
    repo.finish_run(run_id, RunStatus.FAILED, error=err)

    rep = build_report(repo, run_id, config, dialogue_line=LINE, final=None)
    assert rep.failure_stage == "verify_speech" and rep.failure_type == "QualityGateError"
    told = build_report(
        repo, run_id, config, dialogue_line=LINE, final=None, failure_stage="generate_video"
    )
    assert told.failure_stage == "generate_video"  # the flow's own view wins


def test_failure_stage_falls_back_to_the_error_stage(
    repo: RunRepository, config: PipelineConfig
) -> None:
    run_id = _run(repo)
    repo.finish_run(run_id, RunStatus.FAILED, error=ScriptError("bad", stage="write_script"))
    rep = build_report(repo, run_id, config, dialogue_line=None, final=None)
    assert rep.failure_stage == "write_script"


def test_failure_stage_ignores_stage_rows_of_an_earlier_invocation(
    repo: RunRepository, config: PipelineConfig
) -> None:
    run_id = _run(repo)
    repo.start_stage(run_id, StageName.GENERATE_VIDEO)
    repo.finish_stage(run_id, StageName.GENERATE_VIDEO, StageStatus.FAILED)  # invocation 1
    resumed_at = datetime.now(UTC) + timedelta(seconds=1)
    repo.finish_run(run_id, RunStatus.FAILED, error=ScriptError("bad", stage="write_script"))
    rep = build_report(
        repo, run_id, config, dialogue_line=None, final=None, invocation_started_at=resumed_at
    )
    assert rep.failure_stage == "write_script"  # the current error, not the stale row
    stale = build_report(repo, run_id, config, dialogue_line=None, final=None)
    assert stale.failure_stage == "generate_video"  # without the cut-off: every failed row


def test_video_model_comes_from_the_accepted_clips(
    repo: RunRepository, config: PipelineConfig
) -> None:
    run_id = _run(repo)
    rep = build_report(repo, run_id, config, dialogue_line=LINE, final=None)
    assert rep.models["video"] == config.video.model  # nothing accepted: the configured model
    attempt_id = _attempt(repo, run_id, 1, 1)  # made with bytedance/seedance-2.5
    repo.set_final_attempt(run_id, 1, attempt_id)
    other = config.model_copy(
        update={"video": config.video.model_copy(update={"model": "bytedance/seedance-2.0"})}
    )
    rep = build_report(repo, run_id, other, dialogue_line=LINE, final=None)
    assert rep.models["video"] == "bytedance/seedance-2.5"  # a resumed clip keeps its model


def test_successful_run_has_no_failure_stage(repo: RunRepository, config: PipelineConfig) -> None:
    run_id = _run(repo)
    repo.start_stage(run_id, StageName.VERIFY_SPEECH)
    repo.finish_stage(run_id, StageName.VERIFY_SPEECH, StageStatus.FAILED)  # recovered later
    repo.finish_run(run_id, RunStatus.DONE)
    rep = build_report(repo, run_id, config, dialogue_line=LINE, final=None, failure_stage="x")
    assert rep.success and rep.failure_stage is None


def test_has_dialogue_comes_from_the_shots_table(
    repo: RunRepository, config: PipelineConfig
) -> None:
    # A tts_overlay run that failed before any check: no verification rows at all.
    run_id = _run(repo)
    _attempt(repo, run_id, 1, 1)
    _attempt(repo, run_id, 2, 1)
    rep = build_report(repo, run_id, config, dialogue_line=LINE, final=None)
    assert [(s.shot_id, s.has_dialogue) for s in rep.shots] == [(1, False), (2, True)]
    assert rep.line_verified is False


def test_planned_shot_without_attempts_is_listed(
    repo: RunRepository, config: PipelineConfig
) -> None:
    run_id = _run(repo)
    rep = build_report(repo, run_id, config, dialogue_line=LINE, final=None)
    assert [(s.shot_id, s.attempts, s.final_attempt) for s in rep.shots] == [
        (1, 0, None),
        (2, 0, None),
    ]


def test_line_verified_failure_kinds_and_stray_speech(
    repo: RunRepository, config: PipelineConfig
) -> None:
    run_id = _run(repo)
    quiet = _attempt(repo, run_id, 1, 1)
    first = _attempt(repo, run_id, 2, 1)
    second = _attempt(repo, run_id, 2, 2)
    _check(repo, run_id, first, wer=0.4, passed=False, kind=FailureKind.TRUNCATED)
    _check(repo, run_id, second)
    _check(
        repo,
        run_id,
        quiet,
        expected=STRAY_SPEECH_EXPECTED,
        transcript="hey you over there",
        wer=1.0,
        passed=False,
        kind=FailureKind.EXTRA_WORDS,
    )
    repo.set_final_attempt(run_id, 1, quiet)
    repo.set_final_attempt(run_id, 2, second)

    rep = build_report(repo, run_id, config, dialogue_line=LINE, final=None)
    assert rep.line_verified is True  # the stray check on shot 1 is not the line's check
    assert rep.failure_kinds == {"extra_words": 1, "truncated": 1}
    shots = {s.shot_id: s for s in rep.shots}
    assert shots[1].stray_speech_words == 4 and shots[1].final_wer is None
    assert shots[2].stray_speech_words is None
    assert (shots[2].final_wer, shots[2].verified, shots[2].failure_kinds) == (
        0.0,
        True,
        ["truncated"],
    )
    assert rep.shots_regenerated == 1


def test_line_not_verified_when_the_latest_check_is_unverified(
    repo: RunRepository, config: PipelineConfig
) -> None:
    run_id = _run(repo, two_shots=False)
    accepted = _attempt(repo, run_id, 2, 1)
    _check(repo, run_id, accepted)  # the clip passed ...
    _check(repo, run_id, accepted, transcript="", wer=1.0, passed=False, verified=False)
    repo.set_final_attempt(run_id, 2, accepted)  # ... but the final check had no STT
    rep = build_report(repo, run_id, config, dialogue_line=LINE, final=None)
    assert rep.line_verified is False
    assert rep.shots[0].verified is False
