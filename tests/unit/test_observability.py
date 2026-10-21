import json
import logging
import warnings
from pathlib import Path

import pytest

from holywater.observability import logging as hwlog
from holywater.observability.errors import (
    FatalProviderError,
    MediaError,
    PipelineError,
    QualityGateError,
    RetryableProviderError,
    is_retryable,
)


@pytest.fixture(autouse=True)
def _configured_logging() -> None:
    hwlog.configure_logging("DEBUG", json_console=True)
    hwlog.clear_context()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_warning_is_captured_into_jsonl_sink(tmp_path: Path) -> None:
    log_file = tmp_path / "logs.jsonl"
    with hwlog.run_log_file(log_file):
        warnings.warn("something deprecated", DeprecationWarning, stacklevel=1)
    events = _read_jsonl(log_file)
    assert any("something deprecated" in str(e.get("event")) for e in events)
    assert any(e.get("level") == "warning" for e in events)


def test_run_context_is_attached_to_every_line(tmp_path: Path) -> None:
    log_file = tmp_path / "logs.jsonl"
    log = hwlog.get_logger("test")
    with hwlog.run_log_file(log_file), hwlog.log_context(run_id="r1", stage="write_script"):
        log.info("hello", attempt=2)
        logging.getLogger("third_party").info("stdlib line")
    events = _read_jsonl(log_file)
    ours = next(e for e in events if e["event"] == "hello")
    assert ours["run_id"] == "r1"
    assert ours["stage"] == "write_script"
    assert ours["attempt"] == 2
    foreign = next(e for e in events if e["event"] == "stdlib line")
    assert foreign["run_id"] == "r1"


def test_log_context_restores_previous_values(tmp_path: Path) -> None:
    log_file = tmp_path / "logs.jsonl"
    log = hwlog.get_logger("test")
    hwlog.bind_context(run_id="outer")
    with hwlog.run_log_file(log_file):
        with hwlog.log_context(run_id="inner"):
            log.info("in")
        log.info("out")
    events = {e["event"]: e for e in _read_jsonl(log_file)}
    assert events["in"]["run_id"] == "inner"
    assert events["out"]["run_id"] == "outer"


def test_exception_is_serialized_with_traceback(tmp_path: Path) -> None:
    log_file = tmp_path / "logs.jsonl"
    log = hwlog.get_logger("test")
    with hwlog.run_log_file(log_file):
        try:
            raise MediaError("ffmpeg failed", stage="assemble", context={"exit_code": 1})
        except MediaError:
            log.exception("stage failed")
    event = next(e for e in _read_jsonl(log_file) if e["event"] == "stage failed")
    assert event["level"] == "error"
    assert "MediaError" in str(event["exception"])


def test_error_taxonomy_retryable_flags() -> None:
    assert is_retryable(RetryableProviderError("429", provider="fal"))
    assert is_retryable(MediaError("boom"))
    assert not is_retryable(FatalProviderError("401", provider="fal"))
    assert not is_retryable(ValueError("plain"))


def test_error_to_dict_carries_context() -> None:
    err = QualityGateError(
        "wer too high", gate="speech", measured=0.4, threshold=0.1, stage="verify", run_id="r1"
    )
    data = err.to_dict()
    assert data["type"] == "QualityGateError"
    assert data["context"] == {"gate": "speech", "measured": 0.4, "threshold": 0.1}
    assert str(err) == "wer too high [stage=verify, run_id=r1]"
    assert isinstance(err, PipelineError)


def test_provider_error_keeps_provider_fields() -> None:
    err = FatalProviderError(
        "bad request", provider="openrouter", model="m", request_id="req", status_code=400
    )
    assert err.provider == "openrouter"
    assert err.to_dict()["context"]["status_code"] == 400
