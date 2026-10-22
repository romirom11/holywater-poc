"""Run directory layout. The only place that knows where artifacts live on disk."""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from holywater.observability.errors import ConfigError

# A run id is one path segment under `runs/`: no separators, no leading dot, at most 64 chars.
RUN_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
_RUN_ID_RE = re.compile(RUN_ID_PATTERN)


def new_run_id() -> str:
    """Sortable, collision-safe id: `20261021-143055-a1b2c3`."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


def validate_run_id(run_id: str) -> str:
    """Return `run_id` unchanged, or raise `ConfigError` if it could escape `runs/`."""
    if not _RUN_ID_RE.fullmatch(run_id):
        msg = (
            f"invalid run id {run_id!r}: use 1-64 letters, digits, '.', '_' or '-', "
            "starting with a letter or digit"
        )
        raise ConfigError(msg, context={"run_id": run_id, "pattern": RUN_ID_PATTERN})
    return run_id


@dataclass(frozen=True)
class RunPaths:
    runs_dir: Path
    run_id: str
    final_name: str = "final.mp4"

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)

    @property
    def root(self) -> Path:
        return self.runs_dir / self.run_id

    @property
    def script_json(self) -> Path:
        return self.root / "script.json"

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    @property
    def video_dir(self) -> Path:
        return self.root / "video"

    @property
    def work_dir(self) -> Path:
        """Intermediate files (normalized clips, concat lists)."""
        return self.root / "work"

    @property
    def logs_jsonl(self) -> Path:
        return self.root / "logs.jsonl"

    @property
    def report_json(self) -> Path:
        return self.root / "report.json"

    @property
    def final_video(self) -> Path:
        return self.root / self.final_name

    def audio_line(self, shot_id: int) -> Path:
        return self.audio_dir / f"line_{shot_id}.mp3"

    def shot_video(self, shot_id: int, attempt: int) -> Path:
        return self.video_dir / f"shot_{shot_id}_a{attempt}.mp4"

    def ensure(self) -> RunPaths:
        for d in (self.root, self.audio_dir, self.video_dir, self.work_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self
