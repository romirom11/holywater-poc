"""Structured logging (see docs/ARCHITECTURE.md §6).

- structlog on top of stdlib so that third-party loggers (prefect, httpx, warnings) share the
  same processors and sinks;
- one console sink on stderr: human-readable in a TTY, JSON lines otherwise (stdout is kept for
  command results such as the final video path);
- `logging.captureWarnings(True)` so no warning is ever lost;
- a per-run JSONL file sink added/removed around a run;
- run context (run_id, stage, attempt) bound via contextvars.
"""

from __future__ import annotations

import logging
import sys
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import structlog

_SHARED_PROCESSORS: list[Any] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
]

_configured = False
# Root plus `prefect`, which some Prefect versions stop from propagating to root.
_SINK_LOGGERS = ("", "prefect")


def configure_logging(level: str = "INFO", *, json_console: bool | None = None) -> None:
    """Configure the root logger once. Human-readable console in a TTY, JSON otherwise."""
    global _configured  # noqa: PLW0603 - single process-wide switch
    if json_console is None:
        json_console = not sys.stderr.isatty()

    structlog.configure(
        processors=[
            *_SHARED_PROCESSORS,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )

    console = StderrConsoleHandler()
    console.setFormatter(_formatter(json_output=json_console))

    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if getattr(h, "_hw_console", False) is False]
    console._hw_console = True  # type: ignore[attr-defined]
    root.addHandler(console)
    root.setLevel(level.upper())

    # Route every warnings.warn() through logging; never silence them. Toggling off/on
    # re-installs the hook even if something (e.g. pytest) swapped `showwarning` meanwhile.
    warnings.simplefilter("always")
    logging.captureWarnings(False)
    logging.captureWarnings(True)
    # Noisy third-party loggers stay at INFO regardless of our level.
    for name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(max(logging.WARNING, root.level))
    # If Prefect is already imported its root console handler would print every record twice.
    remove_foreign_console_handlers()

    _configured = True


class StderrConsoleHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Writes to whatever `sys.stderr` is at emit time (as stdlib's `lastResort` does), so a
    swapped stream (click's CliRunner, pytest capture) never leaves it writing to a closed file.

    The name must not start with `_`: Prefect's `setup_logging` keeps the root logger as it is
    only when it finds a "user" handler there, and it skips handler classes named `_...`.
    """

    def __init__(self) -> None:
        logging.Handler.__init__(self)

    @property
    def stream(self) -> Any:
        return sys.stderr


def _formatter(*, json_output: bool) -> logging.Formatter:
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_SHARED_PROCESSORS,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )


class _OncePerRecord(logging.Filter):
    """The sink hangs on several loggers; a record that propagates from one to the next (e.g.
    `prefect.*` -> root) reaches it more than once and is written only the first time."""

    def __init__(self) -> None:
        super().__init__()
        self._last: logging.LogRecord | None = None

    def filter(self, record: logging.LogRecord) -> bool:
        if record is self._last:
            return False
        self._last = record
        return True


def add_file_sink(path: Path, level: str = "DEBUG") -> logging.Handler:
    """Attach a JSONL sink (one JSON object per line). Caller removes it via `remove_sink`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level.upper())
    handler.setFormatter(_formatter(json_output=True))
    handler.addFilter(_OncePerRecord())
    for name in _SINK_LOGGERS:
        logging.getLogger(name).addHandler(handler)
    return handler


def remove_sink(handler: logging.Handler) -> None:
    for name in _SINK_LOGGERS:
        logging.getLogger(name).removeHandler(handler)
    handler.flush()
    handler.close()


def remove_foreign_console_handlers() -> None:
    """Drop the console handler Prefect attaches to the root logger.

    Prefect installs its own root console handler when it is imported, which would print every
    one of our records a second time; its records still reach our console via the root logger.
    Its own `prefect.*` loggers keep their handlers, and other root handlers (e.g. pytest's log
    capture) are left alone.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if type(handler).__module__.startswith("prefect.") and isinstance(
            handler, logging.StreamHandler
        ):
            root.removeHandler(handler)


@contextmanager
def run_log_file(path: Path) -> Iterator[None]:
    """Scope a per-run JSONL sink to a `with` block."""
    remove_foreign_console_handlers()
    handler = add_file_sink(path)
    try:
        yield
    finally:
        remove_sink(handler)


def bind_context(**values: Any) -> None:
    """Bind run-scoped fields (run_id, stage, attempt...) to every subsequent log line."""
    structlog.contextvars.bind_contextvars(**values)


def unbind_context(*keys: str) -> None:
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()


@contextmanager
def log_context(**values: Any) -> Iterator[None]:
    """Temporarily bind fields; restores the previous values on exit."""
    tokens = structlog.contextvars.bind_contextvars(**values)
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    if not _configured:
        configure_logging()
    return structlog.stdlib.get_logger(name)
