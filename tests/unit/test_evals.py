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


def test_run_suite_continues_after_failure() -> None:
    suite = load_cases(REPO_CASES)
    calls: list[str] = []

    def run_one(idea: IdeaInput, tag: str, case_id: str) -> str:
        calls.append(case_id)
        if case_id == "en_numbers":
            raise ScriptError("boom", run_id="r-fail")
        return f"r-{case_id}"

    outcomes = run_suite(suite, tag="t", run_one=run_one)
    assert len(outcomes) == len(suite.cases) == len(calls)
    failed = [o for o in outcomes if not o.success]
    assert [o.case_id for o in failed] == ["en_numbers"]
    assert failed[0].run_id == "r-fail"


def test_render_stats_table() -> None:
    table = render_stats(
        [
            TagStats("v1", 3, 2 / 3, 1.5, 0.1, 0.2, 1 / 3, 9.0),
            TagStats("v2", 0, 0.0, 0.0, None, None, 0.0, None),
        ]
    )
    assert "success_rate" in table and "67%" in table and "-" in table
    assert "line_verified_rate" in table
    assert table.splitlines()[0].split() == ["metric", "v1", "v2"]
