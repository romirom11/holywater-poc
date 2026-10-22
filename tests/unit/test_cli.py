"""`hw` commands with the flow and wiring stubbed out (no Prefect run, no ffmpeg, no network)."""

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from holywater.cli import app
from holywater.config import PipelineConfig, load_config
from holywater.domain.enums import AudioStrategy, RunStatus
from holywater.domain.idea import IdeaInput
from holywater.domain.report import RunReport
from holywater.observability.errors import ScriptError
from holywater.storage import RunRepository, init_db, make_engine

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config" / "pipeline.yaml"
IDEA = "A boy shrugs and says: \"I don't care anymore, it's fine.\""


@pytest.fixture
def cfg_path(tmp_path: Path) -> Path:
    raw = load_config(REPO_CONFIG).model_dump(mode="json")
    raw["storage"]["runs_dir"] = str(tmp_path / "runs")
    raw["storage"]["db_path"] = str(tmp_path / "runs" / "hw.db")
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


@pytest.fixture
def cfg(cfg_path: Path) -> PipelineConfig:
    return load_config(cfg_path)


@pytest.fixture
def repo(cfg: PipelineConfig) -> RunRepository:
    engine = make_engine(cfg.storage.db_path)
    init_db(engine)
    return RunRepository(engine)


FlowFn = Callable[..., Any]


def _stub(
    monkeypatch: pytest.MonkeyPatch, cfg: PipelineConfig, flow: FlowFn
) -> list[dict[str, Any]]:
    """Replace build_deps (real repo on the tmp DB) and the flow; return the recorded flow calls."""
    calls: list[dict[str, Any]] = []

    def build_deps(config: PipelineConfig, _secrets: object, *, dry_run: bool) -> Any:
        engine = make_engine(config.storage.db_path)
        init_db(engine)
        return SimpleNamespace(config=config, repo=RunRepository(engine), dry_run=dry_run)

    def generate_video_flow(idea: IdeaInput, deps: Any, *, run_id: str, tag: str | None) -> Any:
        calls.append({"idea": idea.text, "run_id": run_id, "tag": tag, "dry_run": deps.dry_run})
        return flow(idea, deps, run_id=run_id, tag=tag)

    monkeypatch.setattr("holywater.flows.wiring.build_deps", build_deps)
    monkeypatch.setattr(
        "holywater.flows.generate_video_flow.generate_video_flow", generate_video_flow
    )
    return calls


def _start_run(deps: Any, idea: IdeaInput, run_id: str, tag: str | None) -> Path:
    deps.repo.create_run(
        run_id,
        idea,
        config_hash="h",
        audio_strategy=AudioStrategy.NATIVE,
        tag=tag,
        dry_run=deps.dry_run,
    )
    run_dir: Path = deps.config.storage.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _succeeding_flow(*, line_verified: bool) -> FlowFn:
    def flow(idea: IdeaInput, deps: Any, *, run_id: str, tag: str | None) -> Any:
        run_dir = _start_run(deps, idea, run_id, tag)
        deps.repo.finish_run(run_id, RunStatus.DONE)
        now = datetime.now(UTC)
        report = RunReport(
            run_id=run_id,
            tag=tag,
            success=True,
            status="done",
            input_text=idea.text,
            config_hash="h",
            audio_strategy_requested="native",
            audio_strategy_used="native",
            fallback_reason=None,
            started_at=now,
            finished_at=now,
            latency_s=1.0,
            line_verified=line_verified,
        )
        (run_dir / "report.json").write_text(report.model_dump_json(), encoding="utf-8")
        return SimpleNamespace(
            run_id=run_id,
            final=SimpleNamespace(path=run_dir / "final.mp4", duration_s=13.1),
            report_path=run_dir / "report.json",
            strategy_used=AudioStrategy.NATIVE,
            cost_usd=0.0,
        )

    return flow


def _invoke(args: list[str], *, stdin: str | None = None) -> Any:
    return CliRunner().invoke(app, args, input=stdin)


def test_prefect_telemetry_is_off_by_default() -> None:
    assert os.environ["PREFECT_SERVER_ANALYTICS_ENABLED"] == "false"
    assert os.environ["DO_NOT_TRACK"] == "1"


@pytest.mark.parametrize("args", [["generate", "-"], ["generate"]])
def test_idea_is_read_from_stdin(
    monkeypatch: pytest.MonkeyPatch, cfg: PipelineConfig, cfg_path: Path, args: list[str]
) -> None:
    calls = _stub(monkeypatch, cfg, _succeeding_flow(line_verified=True))
    result = _invoke([*args, "--dry-run", "-c", str(cfg_path)], stdin=IDEA + "\n")
    assert result.exit_code == 0, result.output
    assert calls[0]["idea"] == IDEA  # apostrophes intact, trailing newline stripped
    assert "line_verified=yes" in result.output and "WARNING" not in result.output


def test_unverified_line_is_a_visible_warning(
    monkeypatch: pytest.MonkeyPatch, cfg: PipelineConfig, cfg_path: Path
) -> None:
    _stub(monkeypatch, cfg, _succeeding_flow(line_verified=False))
    result = _invoke(["generate", IDEA, "--dry-run", "-c", str(cfg_path)])
    assert result.exit_code == 0
    assert "line_verified=no" in result.output
    assert "WARNING: the spoken line was not verified" in result.output


@pytest.mark.parametrize("bad", ["../../escaped-run", "a/b", ".x"])
def test_run_id_is_validated_before_anything_is_created(
    monkeypatch: pytest.MonkeyPatch, cfg: PipelineConfig, cfg_path: Path, bad: str
) -> None:
    calls = _stub(monkeypatch, cfg, _succeeding_flow(line_verified=True))
    result = _invoke(["generate", IDEA, "--dry-run", "--run-id", bad, "-c", str(cfg_path)])
    assert result.exit_code == 1
    assert "invalid run id" in result.output and "Traceback" not in result.output
    assert not calls and not cfg.storage.runs_dir.exists()


def test_typed_failure_prints_the_run_id_and_how_to_resume(
    monkeypatch: pytest.MonkeyPatch, cfg: PipelineConfig, cfg_path: Path
) -> None:
    def flow(idea: IdeaInput, deps: Any, *, run_id: str, tag: str | None) -> Any:
        _start_run(deps, idea, run_id, tag)
        raise ScriptError("the LLM never quoted the line", stage="write_script")

    calls = _stub(monkeypatch, cfg, flow)
    result = _invoke(["generate", IDEA, "--dry-run", "-c", str(cfg_path)])
    rid = calls[0]["run_id"]
    assert result.exit_code == 1
    assert "error: the LLM never quoted the line" in result.output
    assert f"run {rid} failed" in result.output and f"--run-id {rid}" in result.output
    assert "Traceback" not in result.output


def test_untyped_failure_is_one_line_with_the_traceback_in_the_run_log(
    monkeypatch: pytest.MonkeyPatch, cfg: PipelineConfig, cfg_path: Path
) -> None:
    def flow(idea: IdeaInput, deps: Any, *, run_id: str, tag: str | None) -> Any:
        _start_run(deps, idea, run_id, tag)
        raise KeyError("shot_attempt_ids")

    calls = _stub(monkeypatch, cfg, flow)
    result = _invoke(["generate", IDEA, "--dry-run", "-c", str(cfg_path), "--run-id", "r-bug"])
    assert result.exit_code == 1 and calls[0]["run_id"] == "r-bug"
    assert "error: unexpected KeyError: 'shot_attempt_ids'" in result.output
    assert "run r-bug failed" in result.output
    assert isinstance(result.exception, SystemExit)  # no traceback escaped to the user
    events = [
        json.loads(line)
        for line in (cfg.storage.runs_dir / "r-bug" / "logs.jsonl").read_text().splitlines()
    ]
    crash = next(e for e in events if e["event"] == "cli.unexpected_error")
    assert crash["run_id"] == "r-bug" and "KeyError" in crash["exception"]


def test_failure_before_the_run_starts_prints_no_run_line(
    monkeypatch: pytest.MonkeyPatch, cfg: PipelineConfig, cfg_path: Path
) -> None:
    def flow(idea: IdeaInput, deps: Any, *, run_id: str, tag: str | None) -> Any:
        raise ScriptError("config problem before the run row exists")

    _stub(monkeypatch, cfg, flow)
    result = _invoke(["generate", IDEA, "--dry-run", "-c", str(cfg_path)])
    assert result.exit_code == 1 and "failed; report" not in result.output


def test_a_tag_never_mixes_dry_and_real_runs(
    monkeypatch: pytest.MonkeyPatch, cfg: PipelineConfig, cfg_path: Path, repo: RunRepository
) -> None:
    repo.create_run(
        "real-1",
        IdeaInput(text=IDEA),
        config_hash="h",
        audio_strategy=AudioStrategy.NATIVE,
        tag="v1",
        dry_run=False,
    )
    calls = _stub(monkeypatch, cfg, _succeeding_flow(line_verified=True))
    result = _invoke(["generate", IDEA, "--dry-run", "--tag", "v1", "-c", str(cfg_path)])
    assert result.exit_code == 1 and not calls
    assert "tag 'v1' already holds real runs" in result.output
    resumed = _invoke(["generate", IDEA, "--dry-run", "--run-id", "real-1", "-c", str(cfg_path)])
    assert resumed.exit_code == 1 and "is a real run" in resumed.output and not calls


def test_paid_eval_suite_asks_first(
    monkeypatch: pytest.MonkeyPatch, cfg: PipelineConfig, cfg_path: Path
) -> None:
    calls = _stub(monkeypatch, cfg, _succeeding_flow(line_verified=True))
    cases = Path(__file__).resolve().parents[2] / "evals" / "cases.yaml"
    args = ["eval", "run", "--tag", "paid", "--cases", str(cases), "-c", str(cfg_path)]
    declined = _invoke(args, stdin="n\n")
    assert declined.exit_code == 1 and not calls
    assert "8 PAID runs" in declined.output and "for video alone" in declined.output
    assert "aborted: no run started" in declined.output

    accepted = _invoke([*args, "--yes"])
    assert accepted.exit_code == 0, accepted.output
    assert len(calls) == 8 and {c["dry_run"] for c in calls} == {False}
    assert calls[0]["run_id"].endswith("-en_short") and {c["tag"] for c in calls} == {"paid"}


def test_dry_eval_suite_does_not_ask(
    monkeypatch: pytest.MonkeyPatch, cfg: PipelineConfig, cfg_path: Path
) -> None:
    calls = _stub(monkeypatch, cfg, _succeeding_flow(line_verified=True))
    cases = Path(__file__).resolve().parents[2] / "evals" / "cases.yaml"
    result = _invoke(
        ["eval", "run", "--tag", "smoke", "--cases", str(cases), "--dry-run", "-c", str(cfg_path)]
    )
    assert result.exit_code == 0, result.output
    assert "PAID" not in result.output and len(calls) == 8
    assert result.output.count("ok ") == 8 and "dry_run" in result.output


def test_eval_compare_shows_the_mode_of_each_tag(cfg_path: Path, repo: RunRepository) -> None:
    for run_id, tag, dry in (("a-1", "real", False), ("b-1", "smoke", True)):
        repo.create_run(
            run_id,
            IdeaInput(text=IDEA),
            config_hash="h",
            audio_strategy=AudioStrategy.NATIVE,
            tag=tag,
            dry_run=dry,
        )
    result = _invoke(["eval", "compare", "real", "smoke", "-c", str(cfg_path)])
    assert result.exit_code == 0
    row = next(line for line in result.output.splitlines() if line.startswith("dry_run"))
    assert row.split()[1:] == ["no", "yes"]


def test_eval_compare_with_an_unknown_tag_fails_and_shows_no_rates(
    cfg_path: Path, repo: RunRepository
) -> None:
    repo.create_run(
        "a-1",
        IdeaInput(text=IDEA),
        config_hash="h",
        audio_strategy=AudioStrategy.NATIVE,
        tag="real",
        dry_run=False,
    )
    result = _invoke(["eval", "compare", "real", "nosuchtag", "-c", str(cfg_path)])
    assert result.exit_code == 1
    assert "no runs under tag(s): nosuchtag" in result.output
    rows = {line.split()[0]: line.split()[1:] for line in result.output.splitlines()[2:] if line}
    assert rows["runs"] == ["1", "0"]
    for metric in ("success_rate", "line_verified_rate", "fallback_rate", "mean_cost_usd"):
        assert rows[metric][1] == "-", metric


def test_no_idea_on_a_terminal_is_a_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    import typer

    from holywater.cli import _read_idea

    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr("sys.stdin", Tty(""))
    with pytest.raises(typer.Exit) as exc_info:
        _read_idea(None)
    assert exc_info.value.exit_code == 2
    monkeypatch.setattr("sys.stdin", Tty(IDEA))
    assert _read_idea("-") == IDEA  # '-' reads even from a terminal (end with Ctrl-D)
    assert _read_idea("an idea given inline") == "an idea given inline"
