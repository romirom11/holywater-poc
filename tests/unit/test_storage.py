import gc
import json
import sqlite3
import warnings
from contextlib import closing
from pathlib import Path

import pytest

from holywater.domain.artifacts import SpeechVerification, VideoArtifact
from holywater.domain.enums import (
    AttemptStatus,
    AudioStrategy,
    FailureKind,
    Resolution,
    RunStatus,
    StageName,
    StageStatus,
)
from holywater.domain.idea import IdeaInput
from holywater.observability.errors import ConfigError, QualityGateError
from holywater.storage import (
    RunPaths,
    RunRepository,
    init_db,
    make_engine,
    new_run_id,
    validate_run_id,
)
from holywater.storage.repo import STRAY_SPEECH_EXPECTED, percentile


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


def _new_run(repo: RunRepository, tag: str | None = None, *, dry_run: bool = False) -> str:
    run_id = new_run_id()
    repo.create_run(
        run_id,
        _idea(),
        config_hash="abc",
        audio_strategy=AudioStrategy.NATIVE,
        tag=tag,
        dry_run=dry_run,
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


def test_stage_retries_are_counted_separately_from_entries(repo: RunRepository) -> None:
    run_id = _new_run(repo)
    with pytest.raises(LookupError, match="not started"):
        repo.record_stage_retry(run_id, StageName.GENERATE_VIDEO)
    repo.start_stage(run_id, StageName.GENERATE_VIDEO)  # shot 1
    assert repo.record_stage_retry(run_id, StageName.GENERATE_VIDEO) == 1
    repo.finish_stage(run_id, StageName.GENERATE_VIDEO, StageStatus.DONE)
    stage = repo.start_stage(run_id, StageName.GENERATE_VIDEO)  # shot 2: re-entry, not a retry
    assert (stage.attempt, stage.retries) == (2, 1)


def test_init_db_adds_columns_missing_from_an_older_file(tmp_path: Path) -> None:
    from sqlalchemy import inspect, text

    db = tmp_path / "old.db"
    engine = make_engine(db)
    init_db(engine)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE stages DROP COLUMN retries"))
    assert "retries" not in {c["name"] for c in inspect(engine).get_columns("stages")}
    init_db(make_engine(db))
    repo = RunRepository(make_engine(db))
    run_id = _new_run(repo)
    assert repo.start_stage(run_id, StageName.WRITE_SCRIPT).retries == 0


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
    repo.mark_attempt(a1.id, AttemptStatus.FAILED)
    repo.set_final_attempt(run_id, 1, a2.id)
    repo.set_final_attempt(run_id, 2, b1.id)

    finals = repo.get_final_attempts(run_id)
    assert [f.shot_id for f in finals] == [1, 2]
    assert finals[0].seed == 7
    assert finals[1].strategy == "tts_overlay"
    assert len(repo.get_attempts(run_id, shot_id=1)) == 2
    statuses = {a.attempt: a.status for a in repo.get_attempts(run_id, shot_id=1)}
    assert statuses == {1: "failed", 2: "generated"}  # stored as plain `.value` strings
    repo.mark_attempt(a2.id, "passed")  # a plain value is converted at the boundary
    assert repo.get_attempts(run_id, shot_id=1)[1].status == AttemptStatus.PASSED.value
    with pytest.raises(ValueError):
        repo.mark_attempt(a2.id, "bogus")


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
    repo.close()  # the in-memory DB lives in its pooled connection; dispose closes it


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


# -- schema evolution -------------------------------------------------------------------------

# The tables as the first real run (commit edff939) wrote them: no stages.retries/total_duration_s,
# no runs.dry_run, no shot_attempts integrity columns, no judge_decisions.shot_id/decision.
_OLD_SCHEMA = """
CREATE TABLE runs (
    id VARCHAR NOT NULL PRIMARY KEY, tag VARCHAR, input_text VARCHAR NOT NULL, language VARCHAR,
    config_hash VARCHAR NOT NULL, git_sha VARCHAR, status VARCHAR NOT NULL,
    audio_strategy_requested VARCHAR NOT NULL, audio_strategy_used VARCHAR,
    fallback_reason VARCHAR, started_at DATETIME NOT NULL, finished_at DATETIME,
    cost_usd_total FLOAT NOT NULL, final_path VARCHAR, error_type VARCHAR, error_json VARCHAR
);
CREATE TABLE stages (
    id INTEGER NOT NULL PRIMARY KEY, run_id VARCHAR NOT NULL REFERENCES runs (id),
    name VARCHAR NOT NULL, status VARCHAR NOT NULL, attempt INTEGER NOT NULL,
    input_hash VARCHAR, started_at DATETIME, finished_at DATETIME, error_type VARCHAR,
    error_json VARCHAR, CONSTRAINT uq_stage_run_name UNIQUE (run_id, name)
);
CREATE TABLE shots (
    id INTEGER NOT NULL PRIMARY KEY, run_id VARCHAR NOT NULL REFERENCES runs (id),
    shot_id INTEGER NOT NULL, planned_duration_s INTEGER NOT NULL, has_dialogue BOOLEAN NOT NULL,
    final_attempt_id INTEGER REFERENCES shot_attempts (id),
    CONSTRAINT uq_shot_run_shot UNIQUE (run_id, shot_id)
);
CREATE TABLE shot_attempts (
    id INTEGER NOT NULL PRIMARY KEY, shot_pk INTEGER NOT NULL REFERENCES shots (id),
    run_id VARCHAR NOT NULL REFERENCES runs (id), shot_id INTEGER NOT NULL,
    attempt INTEGER NOT NULL, strategy VARCHAR NOT NULL, model VARCHAR NOT NULL,
    prompt_version VARCHAR NOT NULL, prompt_text VARCHAR NOT NULL, seed INTEGER,
    provider_request_id VARCHAR, path VARCHAR, duration_s FLOAT, resolution VARCHAR NOT NULL,
    cost_usd FLOAT NOT NULL, status VARCHAR NOT NULL, created_at DATETIME NOT NULL
);
CREATE TABLE judge_decisions (
    id INTEGER NOT NULL PRIMARY KEY, run_id VARCHAR NOT NULL REFERENCES runs (id),
    stage VARCHAR NOT NULL, question VARCHAR NOT NULL, answer VARCHAR NOT NULL,
    probabilities_json VARCHAR NOT NULL, model_version VARCHAR, latency_ms INTEGER,
    created_at DATETIME NOT NULL
);
CREATE TABLE costs (
    id INTEGER NOT NULL PRIMARY KEY, run_id VARCHAR NOT NULL REFERENCES runs (id),
    stage VARCHAR NOT NULL, provider VARCHAR NOT NULL, model VARCHAR NOT NULL,
    units FLOAT NOT NULL, unit_price FLOAT NOT NULL, cost_usd FLOAT NOT NULL,
    is_estimate BOOLEAN NOT NULL, shot_id INTEGER, attempt INTEGER, created_at DATETIME NOT NULL
);
INSERT INTO runs VALUES ('old-real', 'v0', 'idea "hi there"', NULL, 'h', NULL, 'done', 'native',
    'native', NULL, '2026-10-22 09:23:46.000000', '2026-10-22 09:29:12.000000', 1.26, NULL,
    NULL, NULL);
INSERT INTO runs VALUES ('old-dry', 'v0-dry', 'idea "hi there"', NULL, 'h', NULL, 'done',
    'native', 'native', NULL, '2026-10-22 10:00:00.000000', '2026-10-22 10:00:13.000000', 0.0,
    NULL, NULL, NULL);
INSERT INTO stages VALUES (1, 'old-real', 'generate_video', 'done', 2, NULL,
    '2026-10-22 09:26:32.000000', '2026-10-22 09:29:12.000000', NULL, NULL);
INSERT INTO costs VALUES (1, 'old-real', 'generate_video', 'openrouter', 'bytedance/seedance-2.5',
    6, 0, 0.62, 0, 1, 1, '2026-10-22 09:26:00.000000');
INSERT INTO costs VALUES (2, 'old-dry', 'generate_video', 'openrouter',
    'fake/bytedance/seedance-2.5', 6, 0, 0.0, 1, 1, 1, '2026-10-22 10:00:05.000000');
"""


def test_init_db_upgrades_a_db_written_with_the_old_schema(tmp_path: Path) -> None:
    from sqlalchemy import inspect

    db = tmp_path / "holywater.db"
    with closing(sqlite3.connect(db)) as conn:
        conn.executescript(_OLD_SCHEMA)
    engine = make_engine(db)
    init_db(engine)

    columns = {t: {c["name"] for c in inspect(engine).get_columns(t)} for t in ("runs", "stages")}
    assert {"dry_run"} <= columns["runs"] and {"retries", "total_duration_s"} <= columns["stages"]
    repo = RunRepository(engine)
    real, dry = repo.get_run("old-real"), repo.get_run("old-dry")
    assert real is not None and dry is not None
    # Backfill: a run whose every cost row names a fake model was a dry run.
    assert (real.dry_run, dry.dry_run) == (False, True)
    stage = repo.get_stages("old-real")[0]
    assert (stage.retries, stage.total_duration_s) == (0, 0.0)  # defaults on old rows

    # New writes work against the upgraded file, including the new columns.
    repo.upsert_shot("old-real", 1, planned_duration_s=6, has_dialogue=True)
    abandoned = repo.add_abandoned_attempt(
        "old-real",
        shot_id=1,
        attempt=1,
        strategy=AudioStrategy.NATIVE,
        model="seedance",
        provider_request_id="job-1",
        cost_usd=0.62,
        is_estimate=False,
        reason="download kept failing",
    )
    assert abandoned.status == "abandoned" and abandoned.size_bytes is None
    repo.add_judge_decision(
        "old-real",
        stage=StageName.PREFLIGHT_SCRIPT,
        question="q",
        answer="yes",
        probabilities={"yes": 1.0},
        shot_id=1,
        decision="rewrite",
    )
    assert (repo.get_judge_decisions("old-real")[0].shot_id, "rewrite") == (
        1,
        repo.get_judge_decisions("old-real")[0].decision,
    )
    init_db(engine)  # idempotent: nothing left to add, no second backfill
    assert repo.get_run("old-real").dry_run is False  # type: ignore[union-attr]


def test_init_db_refuses_a_not_null_column_without_default(tmp_path: Path) -> None:
    db = tmp_path / "broken.db"
    with closing(sqlite3.connect(db)) as conn:  # `input_text` (NOT NULL, no default) is missing
        conn.execute("CREATE TABLE runs (id VARCHAR NOT NULL PRIMARY KEY)")
    with pytest.raises(ConfigError, match=r"runs\.input_text.*storage\.db_path"):
        init_db(make_engine(db))


def test_file_engine_leaves_no_unclosed_connections(tmp_path: Path) -> None:
    gc.collect()  # objects left by earlier tests are not this test's business
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine = make_engine(tmp_path / "x.db")
        init_db(engine)
        repo = RunRepository(engine)
        run_id = _new_run(repo)
        assert repo.get_run(run_id) is not None
        repo.close()
        del repo, engine
        gc.collect()
    assert not [w for w in caught if issubclass(w.category, ResourceWarning)]


# -- run ids, tags, dry runs ------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["../../escaped", "a/b", ".hidden", "", "x" * 65, "a b"])
def test_run_ids_cannot_escape_the_runs_dir(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ConfigError, match="invalid run id"):
        validate_run_id(bad)
    with pytest.raises(ConfigError, match="invalid run id"):
        RunPaths(tmp_path, bad)
    assert validate_run_id("20261022-151616-02c98a-en_short") == "20261022-151616-02c98a-en_short"


def test_dry_run_is_stored_and_a_tag_never_mixes_modes(repo: RunRepository) -> None:
    dry = _new_run(repo, tag="smoke", dry_run=True)
    stored = repo.get_run(dry)
    assert stored is not None and stored.dry_run is True
    assert repo.tag_modes("smoke") == {True} and repo.tag_modes("new") == set()
    with pytest.raises(ConfigError, match="tag 'smoke' already holds dry-run runs"):
        _new_run(repo, tag="smoke", dry_run=False)
    _new_run(repo, tag="real", dry_run=False)
    with pytest.raises(ConfigError, match="already holds real runs"):
        repo.ensure_tag_mode("real", dry_run=True)
    repo.ensure_tag_mode(None, dry_run=True)  # untagged runs are never pooled
    assert repo.tag_stats("smoke").dry_runs == 1 and repo.tag_stats("real").dry_runs == 0


def test_a_dry_run_cannot_be_resumed_as_a_real_one(repo: RunRepository) -> None:
    run_id = _new_run(repo, dry_run=True)
    with pytest.raises(ConfigError, match="is a dry run"):
        repo.reopen_run(run_id, config_hash="abc", git_sha=None, dry_run=False)
    assert repo.reopen_run(run_id, config_hash="abc", git_sha=None, dry_run=True).dry_run
    assert repo.reopen_run(run_id, config_hash="abc", git_sha=None).status == "running"


# -- stage timing ------------------------------------------------------------------------------


def test_stage_duration_sums_every_entry(repo: RunRepository) -> None:
    from datetime import UTC, datetime, timedelta

    from sqlmodel import Session, select

    from holywater.storage.models import Stage

    run_id = _new_run(repo)

    def entry(seconds: float) -> None:
        repo.start_stage(run_id, StageName.GENERATE_VIDEO)
        with Session(repo._engine) as s:  # backdate the start instead of sleeping
            row = s.exec(select(Stage).where(Stage.run_id == run_id)).one()
            row.started_at = datetime.now(UTC) - timedelta(seconds=seconds)
            s.add(row)
            s.commit()
        repo.finish_stage(run_id, StageName.GENERATE_VIDEO, StageStatus.DONE)

    entry(150.0)  # shot 1
    entry(160.0)  # shot 2
    # A gate verdict recorded after the stage block closed must not count the entry twice.
    stage = repo.finish_stage(run_id, StageName.GENERATE_VIDEO, StageStatus.FAILED)
    assert stage.attempt == 2
    assert stage.total_duration_s == pytest.approx(310.0, abs=1.0)


# -- attempts: integrity and abandoned jobs -----------------------------------------------------


def test_attempt_records_size_and_sha256_and_resume_checks_them(
    repo: RunRepository, tmp_path: Path
) -> None:
    import hashlib

    clip = tmp_path / "shot_1_a1.mp4"
    clip.write_bytes(b"fake mp4 bytes")
    run_id = _new_run(repo)
    repo.upsert_shot(run_id, 1, planned_duration_s=8, has_dialogue=True)
    row = repo.add_shot_attempt(run_id, _video(path=clip, cost_is_estimate=False))
    assert row.size_bytes == len(b"fake mp4 bytes")
    assert row.sha256 == hashlib.sha256(b"fake mp4 bytes").hexdigest()
    assert row.cost_is_estimate is False
    assert repo.attempt_file_ok(row)
    clip.write_bytes(b"fake mp4 bytez")  # same size, other content
    assert not repo.attempt_file_ok(row)
    clip.write_bytes(b"short")
    assert not repo.attempt_file_ok(row)
    clip.unlink()
    assert not repo.attempt_file_ok(row)
    missing = repo.add_shot_attempt(run_id, _video(attempt=2))  # path not on disk (yet)
    assert (missing.size_bytes, missing.sha256) == (None, None)


def test_abandoned_attempt_keeps_numbering_and_cost(repo: RunRepository) -> None:
    run_id = _new_run(repo)
    repo.upsert_shot(run_id, 1, planned_duration_s=6, has_dialogue=True)
    row = repo.add_abandoned_attempt(
        run_id,
        shot_id=1,
        attempt=1,
        strategy=AudioStrategy.NATIVE,
        model="bytedance/seedance-2.5",
        provider_request_id="job-9",
        cost_usd=0.62,
        is_estimate=True,
        reason="BilledJobError: download kept failing",
        prompt_version="seedance_shot@1/dialogue",
        resolution=Resolution.P480,
    )
    assert (row.status, row.path, row.reason) == (
        AttemptStatus.ABANDONED.value,
        None,
        "BilledJobError: download kept failing",
    )
    assert (row.resolution, row.cost_usd, row.cost_is_estimate) == ("480p", 0.62, True)
    assert not repo.attempt_file_ok(row)
    assert [a.attempt for a in repo.get_attempts(run_id, shot_id=1)] == [1]


# -- line verification and tag statistics -------------------------------------------------------


def _check(
    expected: str, *, wer: float = 0.0, passed: bool = True, **kw: object
) -> SpeechVerification:
    return SpeechVerification.model_validate(
        {
            "shot_id": 1,
            "attempt": 1,
            "expected": expected,
            "transcript": "x",
            "wer": wer,
            "passed": passed,
            **kw,
        }
    )


def test_line_verification_ignores_stray_speech_checks(repo: RunRepository) -> None:
    run_id = _new_run(repo)
    repo.upsert_shot(run_id, 1, planned_duration_s=6, has_dialogue=False)
    repo.upsert_shot(run_id, 2, planned_duration_s=6, has_dialogue=True)
    quiet = repo.add_shot_attempt(run_id, _video(1, 1))
    spoken = repo.add_shot_attempt(run_id, _video(2, 1))
    assert quiet.id and spoken.id
    repo.set_final_attempt(run_id, 1, quiet.id)
    assert repo.line_verification(run_id) is None  # dialogue shot not accepted yet
    repo.set_final_attempt(run_id, 2, spoken.id)
    repo.add_speech_verification(run_id, spoken.id, _check("the line", wer=0.0))
    repo.add_speech_verification(run_id, quiet.id, _check(STRAY_SPEECH_EXPECTED, wer=1.0))
    line = repo.line_verification(run_id)
    assert line is not None and line.expected == "the line" and line.passed


def test_percentile_is_nearest_rank() -> None:
    assert percentile([0.0, 0.1, 0.2], 0.95) == 0.2  # n < 20: the max
    values = [i / 100 for i in range(1, 21)]  # 0.01 .. 0.20
    assert percentile(values, 0.95) == 0.19  # 19th of 20
    assert percentile([0.3], 0.95) == 0.3
    with pytest.raises(ValueError, match="no values"):
        percentile([], 0.95)


def test_tag_stats_pools_one_line_check_per_run(repo: RunRepository) -> None:
    run_id = _new_run(repo, tag="t")
    repo.upsert_shot(run_id, 1, planned_duration_s=6, has_dialogue=True)
    repo.upsert_shot(run_id, 2, planned_duration_s=6, has_dialogue=False)
    first = repo.add_shot_attempt(run_id, _video(1, 1))
    second = repo.add_shot_attempt(run_id, _video(1, 2))
    quiet = repo.add_shot_attempt(run_id, _video(2, 1, duration_s=6.0))
    assert first.id and second.id and quiet.id
    repo.add_speech_verification(
        run_id, first.id, _check("line", wer=0.5, passed=False, failure_kind=FailureKind.TRUNCATED)
    )
    repo.add_speech_verification(run_id, second.id, _check("line", wer=0.0))
    repo.add_speech_verification(run_id, quiet.id, _check(STRAY_SPEECH_EXPECTED, wer=1.0))
    repo.set_final_attempt(run_id, 1, second.id)
    repo.set_final_attempt(run_id, 2, quiet.id)
    repo.finish_run(run_id, RunStatus.DONE)

    stats = repo.tag_stats("t")
    assert (stats.runs, stats.wer_samples, stats.mean_wer) == (1, 1, 0.0)  # stray row excluded
    assert stats.line_verified_rate == 1.0
    assert stats.failure_kinds == {"truncated": 1}
    assert stats.mean_duration_s == pytest.approx(7.0)  # final clips of both shots
