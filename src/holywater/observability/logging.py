"""Structured logging (see docs/ARCHITECTURE.md §6).

- structlog on top of stdlib so that third-party loggers (prefect, httpx, warnings) share the
  same processors and sinks;
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

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(_formatter(json_output=json_console))

    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if getattr(h, "_hw_console", False) is False]
    console._hw_console = True  # type: ignore[attr-defined]
    root.addHandler(console)
    root.setLevel(level.upper())

    # Route every warnings.warn() through logging; never silence them.
    warnings.simplefilter("always")
    logging.captureWarnings(True)
    # Noisy third-party loggers stay at INFO regardless of our level.
    for name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(max(logging.INFO, root.level))

    _configured = True


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


def add_file_sink(path: Path, level: str = "DEBUG") -> logging.Handler:
    """Attach a JSONL sink (one JSON object per line). Caller removes it via `remove_sink`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level.upper())
    handler.setFormatter(_formatter(json_output=True))
    logging.getLogger().addHandler(handler)
    return handler


def remove_sink(handler: logging.Handler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.flush()
    handler.close()


@contextmanager
def run_log_file(path: Path) -> Iterator[None]:
    """Scope a per-run JSONL sink to a `with` block."""
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
