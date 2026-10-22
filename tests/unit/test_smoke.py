from pathlib import Path

from structlog.testing import capture_logs
from typer.testing import CliRunner

from holywater import __version__
from holywater.cli import app


def test_cli_version_prints_package_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == __version__


def test_cli_rejects_a_short_idea_without_a_traceback(tmp_path: Path) -> None:
    # Validation runs before config, DB or ffmpeg are touched, so nothing is created.
    result = CliRunner().invoke(
        app, ["generate", "too short", "--dry-run", "--config", str(tmp_path / "none.yaml")]
    )
    assert result.exit_code == 1
    assert "The idea is too short (9 characters)" in result.output
    assert "Traceback" not in result.output and result.exception is not None
    assert isinstance(result.exception, SystemExit)
    assert not any(tmp_path.iterdir())


def test_cli_warns_about_more_than_three_sentences() -> None:
    from holywater.cli import _parse_idea

    text = 'One. Two. Three. She says: "Four sentences now."'
    with capture_logs() as logs:
        idea = _parse_idea(text)
    assert idea.text == text and idea.sentence_count == 4
    warning = next(e for e in logs if e["event"] == "idea.too_many_sentences")
    assert warning["log_level"] == "warning" and warning["sentences"] == 4
    with capture_logs() as logs:
        _parse_idea('A taxi at night. She says: "Are you sure?"')
    assert not any(e["event"] == "idea.too_many_sentences" for e in logs)
