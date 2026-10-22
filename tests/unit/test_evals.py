from pathlib import Path

import pytest

from holywater.domain.idea import IdeaInput
from holywater.evals import load_cases, render_stats, run_suite
from holywater.observability.errors import ConfigError, ScriptError
from holywater.storage.repo import TagStats

REPO_CASES = Path(__file__).resolve().parents[2] / "evals" / "cases.yaml"


def test_repo_cases_load() -> None:
    suite = load_cases(REPO_CASES)
    assert len(suite.cases) >= 8
    assert len({c.id for c in suite.cases}) == len(suite.cases)


def test_invalid_cases_file(tmp_path: Path) -> None:
    bad = tmp_path / "c.yaml"
    bad.write_text("cases: [{id: x}]")
    with pytest.raises(ConfigError, match="invalid eval cases"):
        load_cases(bad)
    with pytest.raises(ConfigError, match="not found"):
        load_cases(tmp_path / "missing.yaml")


def test_run_suite_continues_after_failure_and_keeps_run_ids() -> None:
    suite = load_cases(REPO_CASES)
    calls: list[tuple[str, str]] = []

    def run_one(idea: IdeaInput, tag: str, run_id: str) -> None:
        calls.append((tag, run_id))
        if run_id.endswith("en_numbers"):
            raise ScriptError("boom")  # carries no run id, as stage errors usually do
        if run_id.endswith("en_name"):
            raise RuntimeError("a bug")  # untyped: recorded, the suite goes on

    outcomes = run_suite(suite, tag="t", run_one=run_one, make_run_id=lambda c: f"r-{c}")
    assert len(outcomes) == len(suite.cases) == len(calls)
    assert {tag for tag, _ in calls} == {"t"}
    failed = {o.case_id: o for o in outcomes if not o.success}
    assert set(failed) == {"en_numbers", "en_name"}
    assert failed["en_numbers"].run_id == "r-en_numbers"  # allocated before the run started
    assert failed["en_name"].error == "RuntimeError: a bug"
    assert all(o.run_id == f"r-{o.case_id}" for o in outcomes)


def test_default_case_run_id_is_a_valid_run_id() -> None:
    from holywater.evals import case_run_id
    from holywater.storage import validate_run_id

    assert validate_run_id(case_run_id("uk_number")).endswith("-uk_number")


def test_case_ids_must_be_path_safe(tmp_path: Path) -> None:
    bad = tmp_path / "c.yaml"
    bad.write_text("cases: [{id: ../x, text: 'Long enough text here'}]")
    with pytest.raises(ConfigError, match="invalid eval cases"):
        load_cases(bad)


def test_render_stats_table() -> None:
    table = render_stats(
        [
            TagStats(
                "v1",
                3,
                2 / 3,
                1.5,
                0.1,
                0.2,
                1 / 3,
                9.0,
                wer_samples=3,
                failure_kinds={"wrong_language": 2},
            ),
            TagStats("v2", 0, 0.0, 0.0, None, None, 0.0, None),
            TagStats("smoke", 2, 1.0, 0.0, 0.0, 0.0, 0.0, 9.0, dry_runs=2),
        ]
    )
    assert "success_rate" in table and "67%" in table and "-" in table
    assert "line_verified_rate" in table
    lines = {line.split()[0]: line.split()[1:] for line in table.splitlines()[2:]}
    assert table.splitlines()[0].split() == ["metric", "v1", "v2", "smoke"]
    assert lines["dry_run"] == ["no", "-", "yes"]
    assert lines["wer_samples"] == ["3", "0", "0"]
    assert lines["failure_kind:wrong_language"] == ["2", "0", "0"]
