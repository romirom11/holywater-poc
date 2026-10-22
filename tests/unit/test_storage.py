import json
from pathlib import Path

import pytest

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
from holywater.observability.errors import QualityGateError
from holywater.storage import RunPaths, RunRepository, init_db, make_engine, new_run_id


@pytest.fixture
def repo(tmp_path: Path) -> RunRepository:
    engine = make_engine(tmp_path / "test.db")
    init_db(engine)
    return RunRepository(engine)


def _idea() -> IdeaInput:
    return IdeaInput(text='Rainy night. The driver asks: "Are you sure this is the address?"')


def _video(shot_id: int = 1, attempt: int = 1, **overrides: object) -> VideoArtifact:
    base: dict[str, object] = {
        "shot_id": shot_id,
        "attempt": attempt,
        "strategy": AudioStrategy.NATIVE,
        "path": Path(f"runs/r/video/shot_{shot_id}_a{attempt}.mp4"),
        "seed": 42,
        "duration_s": 8.0,
        "resolution": Resolution.P480,
        "has_native_audio": True,
        "prompt_version": "v1",
        "prompt_text": "prompt",
        "model": "seedance",
        "cost_usd": 1.5,
    }
    base.update(overrides)
    return VideoArtifact.model_validate(base)


def _new_run(repo: RunRepository, tag: str | None = None) -> str:
    run_id = new_run_id()
    repo.create_run(
        run_id, _idea(), config_hash="abc", audio_strategy=AudioStrategy.NATIVE, tag=tag
    )
    return run_id


def test_run_id_is_sortable_and_unique() -> None:
    a, b = new_run_id(), new_run_id()
    assert a != b
    assert len(a) == len("20261021-143055-a1b2c3")


def test_run_paths_layout(tmp_path: Path) -> None:
    paths = RunPaths(tmp_path, "r1").ensure()
    assert paths.shot_video(2, 3) == tmp_path / "r1" / "video" / "shot_2_a3.mp4"
    assert paths.audio_line(1).parent.is_dir()
    assert paths.final_video.name == "final.mp4"


def test_stage_lifecycle_and_idempotency(repo: RunRepository) -> None:
    run_id = _new_run(repo)
    stage = repo.start_stage(run_id, StageName.WRITE_SCRIPT, input_hash="h1")
    assert stage.attempt == 1 and stage.status == "running"
    repo.finish_stage(run_id, StageName.WRITE_SCRIPT, StageStatus.DONE)
    assert repo.stage_is_done(run_id, StageName.WRITE_SCRIPT, input_hash="h1")
    assert not repo.stage_is_done(run_id, StageName.WRITE_SCRIPT, input_hash="h2")
    assert not repo.stage_is_done(run_id, StageName.ASSEMBLE, input_hash="h1")

    again = repo.start_stage(run_id, StageName.WRITE_SCRIPT, input_hash="h2")
    assert again.attempt == 2
    assert not repo.stage_is_done(run_id, StageName.WRITE_SCRIPT, input_hash="h1")


def test_stage_failure_persists_typed_error(repo: RunRepository) -> None:
    run_id = _new_run(repo)
    repo.start_stage(run_id, StageName.VERIFY_SPEECH)
    err = QualityGateError("wer", gate="speech", measured=0.5, threshold=0.1, run_id=run_id)
    stage = repo.finish_stage(run_id, StageName.VERIFY_SPEECH, StageStatus.FAILED, error=err)
    assert stage.error_type == "QualityGateError"
    assert stage.error_json is not None
    assert json.loads(stage.error_json)["context"]["gate"] == "speech"


def test_shots_attempts_and_final_selection(repo: RunRepository) -> None:
    run_id = _new_run(repo)
    repo.upsert_shot(run_id, 1, planned_duration_s=8, has_dialogue=True)
    repo.upsert_shot(run_id, 2, planned_duration_s=6, has_dialogue=False)
    a1 = repo.add_shot_attempt(run_id, _video(1, 1))
    a2 = repo.add_shot_attempt(run_id, _video(1, 2, seed=7))
    b1 = repo.add_shot_attempt(run_id, _video(2, 1, strategy=AudioStrategy.TTS_OVERLAY))
    assert a1.id and a2.id and b1.id
    repo.mark_attempt(a1.id, "failed")
    repo.set_final_attempt(run_id, 1, a2.id)
    repo.set_final_attempt(run_id, 2, b1.id)

    finals = repo.get_final_attempts(run_id)
    assert [f.shot_id for f in finals] == [1, 2]
    assert finals[0].seed == 7
    assert finals[1].strategy == "tts_overlay"
    assert len(repo.get_attempts(run_id, shot_id=1)) == 2


def test_verification_judge_and_cost_rows(repo: RunRepository) -> None:
    run_id = _new_run(repo)
    repo.upsert_shot(run_id, 1, planned_duration_s=8, has_dialogue=True)
    attempt = repo.add_shot_attempt(run_id, _video())
    assert attempt.id
    repo.add_speech_verification(
        run_id,
        attempt.id,
        SpeechVerification(
            shot_id=1,
            attempt=1,
            expected="a b c",
            transcript="a b",
            wer=0.33,
            passed=False,
            stt_model="whisper",
            failure_kind=FailureKind.TRUNCATED,
        ),
    )
    repo.add_judge_decision(
        run_id,
        stage=StageName.VERIFY_SPEECH,
        question="failure_kind",
        answer="truncated",
        probabilities={"truncated": 0.8, "paraphrased": 0.2},
        latency_ms=90,
    )
    repo.add_cost(
        run_id, stage=StageName.GENERATE_VIDEO, provider="fal", model="m", units=8, unit_price=0.22
    )
    repo.add_cost(
        run_id, stage=StageName.VERIFY_SPEECH, provider="fal", model="w", units=1, unit_price=0.01
    )

    assert repo.get_speech_verifications(run_id)[0].failure_kind == "truncated"
    decision = repo.get_judge_decisions(run_id)[0]
    assert json.loads(decision.probabilities_json)["truncated"] == 0.8
    run = repo.get_run(run_id)
    assert run is not None
    assert run.cost_usd_total == pytest.approx(8 * 0.22 + 0.01)
    assert len(repo.get_costs(run_id)) == 2


def test_finish_run_and_strategy_fallback(repo: RunRepository) -> None:
    run_id = _new_run(repo)
    repo.set_audio_strategy_used(run_id, AudioStrategy.TTS_OVERLAY, fallback_reason="wer exhausted")
    run = repo.finish_run(run_id, RunStatus.DONE, final_path="runs/r/final.mp4")
    assert run.status == "done" and run.finished_at is not None
    stored = repo.get_run(run_id)
    assert stored is not None
    assert stored.audio_strategy_used == "tts_overlay"
    assert stored.fallback_reason == "wer exhausted"

    failed = repo.finish_run(run_id, RunStatus.FAILED, error=RuntimeError("boom"))
    assert failed.error_type == "RuntimeError"


def test_tag_stats_aggregates(repo: RunRepository) -> None:
    for i in range(3):
        run_id = _new_run(repo, tag="v1")
        repo.upsert_shot(run_id, 1, planned_duration_s=8, has_dialogue=True)
        attempt = repo.add_shot_attempt(run_id, _video(duration_s=8.0 + i))
        assert attempt.id
        repo.set_final_attempt(run_id, 1, attempt.id)
        repo.add_speech_verification(
            run_id,
            attempt.id,
            SpeechVerification(
                shot_id=1, attempt=1, expected="x", transcript="x", wer=0.1 * i, passed=i < 2
            ),
        )
        repo.add_cost(
            run_id, stage=StageName.GENERATE_VIDEO, provider="fal", model="m", units=1, unit_price=2
        )
        if i == 2:
            repo.set_audio_strategy_used(run_id, AudioStrategy.TTS_OVERLAY, fallback_reason="x")
            repo.finish_run(run_id, RunStatus.FAILED)
        else:
            repo.set_audio_strategy_used(run_id, AudioStrategy.NATIVE)
            repo.finish_run(run_id, RunStatus.DONE)

    stats = repo.tag_stats("v1")
    assert stats.runs == 3
    assert stats.success_rate == pytest.approx(2 / 3)
    assert stats.mean_cost_usd == pytest.approx(2.0)
    assert stats.mean_wer == pytest.approx(0.1)
    assert stats.p95_wer == pytest.approx(0.2)
    assert stats.fallback_rate == pytest.approx(1 / 3)
    assert stats.mean_duration_s == pytest.approx(9.0)
    assert repo.tag_stats("missing").runs == 0


def test_in_memory_engine_works() -> None:
    engine = make_engine(":memory:")
    init_db(engine)
    repo = RunRepository(engine)
    run_id = _new_run(repo)
    assert repo.get_run(run_id) is not None


def test_cost_breakdown_by_provider_model_stage_and_cycle(repo: RunRepository) -> None:
    run_id = _new_run(repo)
    add = repo.add_cost
    add(
        run_id,
        stage=StageName.WRITE_SCRIPT,
        provider="openrouter",
        model="llm",
        units=1,
        unit_price=0,
        cost_usd=0.01,
        is_estimate=False,
    )
    add(
        run_id,
        stage=StageName.GENERATE_VIDEO,
        provider="openrouter",
        model="seedance",
        units=8,
        unit_price=0,
        cost_usd=1.5,
        is_estimate=False,
        shot_id=1,
        attempt=1,
    )
    add(
        run_id,
        stage=StageName.VERIFY_SPEECH,
        provider="fal",
        model="whisper",
        units=8,
        unit_price=0,
        cost_usd=0.01,
        is_estimate=True,
        shot_id=1,
        attempt=1,
    )
    add(
        run_id,
        stage=StageName.GENERATE_VIDEO,
        provider="openrouter",
        model="seedance",
        units=8,
        unit_price=0,
        cost_usd=1.6,
        is_estimate=False,
        shot_id=1,
        attempt=2,
    )
    b = repo.cost_breakdown(run_id)
    assert b.total_usd == pytest.approx(3.12)
    assert b.actual_usd == pytest.approx(3.11) and b.estimated_usd == pytest.approx(0.01)
    assert b.by_provider == {"fal": 0.01, "openrouter": pytest.approx(3.11)}
    assert b.by_model["seedance"] == pytest.approx(3.1)
    assert b.by_stage["generate_video"] == pytest.approx(3.1)
    assert [(c.shot_id, c.attempt, c.calls) for c in b.by_cycle] == [(1, 1, 2), (1, 2, 1)]
    assert b.by_cycle[0].cost_usd == pytest.approx(1.51)
    run = repo.get_run(run_id)
    assert run is not None and run.cost_usd_total == pytest.approx(3.12)
    assert repo.tag_stats("none").mean_cost_by_provider == {}
