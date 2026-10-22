"""Builds the dependency bundle for a run from config via the provider registry."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from holywater.config import PipelineConfig, Secrets
from holywater.media import FFmpeg
from holywater.observability.errors import ConfigError
from holywater.observability.logging import get_logger
from holywater.providers.base import (
    JudgeProvider,
    LipsyncProvider,
    LLMProvider,
    STTProvider,
    TTSProvider,
    VideoProvider,
)
from holywater.providers.registry import ProviderRegistry
from holywater.storage import RunRepository, init_db, make_engine

log = get_logger(__name__)


@dataclass
class PipelineDeps:
    config: PipelineConfig
    repo: RunRepository
    ffmpeg: FFmpeg
    llm: LLMProvider
    video: VideoProvider
    tts: TTSProvider
    stt: STTProvider
    judge: JudgeProvider
    lipsync: LipsyncProvider | None
    dry_run: bool
    git_sha: str | None = None

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source: type[Any], _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Prefect builds a pydantic schema for flow/task parameters; this bundle holds live
        clients behind Protocols, so it is passed through as an opaque value."""
        return core_schema.any_schema()


def build_deps(
    config: PipelineConfig,
    secrets: Secrets,
    *,
    dry_run: bool,
    db_path: Path | None = None,
) -> PipelineDeps:
    if not FFmpeg.is_available():
        raise ConfigError("ffmpeg/ffprobe not found on PATH; install ffmpeg first")
    ffmpeg = FFmpeg(timeout_s=config.retries.timeouts_s.ffmpeg)
    engine = make_engine(db_path or config.storage.db_path)
    init_db(engine)
    repo = RunRepository(engine)

    registry = ProviderRegistry(config, secrets, ffmpeg=ffmpeg, dry_run=dry_run)
    if dry_run:
        log.info("wiring.dry_run", note="using in-process fake providers; no external calls")
    else:
        log.info(
            "wiring.providers",
            llm=f"{config.llm.provider}:{config.llm.model}",
            video=f"{config.video.provider}:{config.video.model}",
            tts=f"{config.audio.tts.provider}:{config.audio.tts.model}",
            stt=f"{config.audio.stt.provider}:{config.audio.stt.model}",
            lipsync=(
                f"{config.audio.lipsync.provider}:{config.audio.lipsync.model}"
                if config.audio.lipsync.enabled
                else "disabled"
            ),
        )
    return PipelineDeps(
        config=config,
        repo=repo,
        ffmpeg=ffmpeg,
        llm=registry.llm(config.llm.provider),
        video=registry.video(config.video.provider),
        tts=registry.tts(config.audio.tts.provider),
        stt=registry.stt(config.audio.stt.provider),
        judge=registry.judge(),
        lipsync=(
            registry.lipsync(config.audio.lipsync.provider)
            if config.audio.lipsync.enabled
            else None
        ),
        dry_run=dry_run,
        git_sha=git_sha(),
    )


def git_sha() -> str | None:
    """Short HEAD sha stored on the run, with `-dirty` when the working tree has uncommitted or
    untracked changes (the code that ran is then not that commit); None (with a WARNING) outside
    a git checkout."""
    head = _git("rev-parse", "--short", "HEAD")
    if not head:
        return None
    status = _git("status", "--porcelain")
    if status is None:  # unknown: say so rather than claim a clean tree
        return f"{head}-unknown"
    return f"{head}-dirty" if status else head


def _git(*args: str) -> str | None:
    """stdout of one git command (stripped), or None with a WARNING when it cannot run."""
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning(
            "wiring.git_sha_unavailable",
            command=args[0],
            error=type(exc).__name__,
            message=str(exc),
        )
        return None
    if out.returncode != 0:
        log.warning(
            "wiring.git_sha_unavailable",
            command=args[0],
            returncode=out.returncode,
            stderr=out.stderr[:200],
        )
        return None
    return out.stdout.strip()
