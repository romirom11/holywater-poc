"""Builds the dependency bundle for a run: real providers or in-process fakes (`--dry-run`)."""

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
    NoopJudge,
    STTProvider,
    TTSProvider,
    VideoProvider,
)
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
    ffmpeg = FFmpeg(timeout_s=config.retries.timeouts_s.video)
    engine = make_engine(db_path or config.storage.db_path)
    init_db(engine)
    repo = RunRepository(engine)

    if dry_run:
        from holywater.providers import fake

        log.info("wiring.dry_run", note="using in-process fake providers; no external calls")
        return PipelineDeps(
            config=config,
            repo=repo,
            ffmpeg=ffmpeg,
            llm=fake.FakeLLM(),
            video=fake.FakeVideo(ffmpeg=ffmpeg),
            tts=fake.FakeTTS(ffmpeg=ffmpeg),
            stt=fake.FakeSTT(),
            judge=fake.FakeJudge() if config.judge.enabled else NoopJudge(),
            lipsync=fake.FakeLipsync() if config.audio.lipsync.enabled else None,
            dry_run=True,
            git_sha=git_sha(),
        )

    secrets.require("fal_key", "openrouter_api_key")
    from holywater.providers.fal_lipsync import FalLipsync
    from holywater.providers.fal_stt import FalSTT
    from holywater.providers.fal_tts import FalTTS
    from holywater.providers.fal_video import FalVideo
    from holywater.providers.openrouter_llm import OpenRouterLLM
    from holywater.providers.typesafe_judge import build_judge

    assert secrets.fal_key is not None and secrets.openrouter_api_key is not None
    fal_key = secrets.fal_key.get_secret_value()
    timeouts = config.retries.timeouts_s
    return PipelineDeps(
        config=config,
        repo=repo,
        ffmpeg=ffmpeg,
        llm=OpenRouterLLM(
            api_key=secrets.openrouter_api_key.get_secret_value(),
            model=config.llm.model,
            base_url=config.llm.base_url,
            timeout_s=timeouts.llm,
        ),
        video=FalVideo(api_key=fal_key, model=config.video.model, timeout_s=timeouts.video),
        tts=FalTTS(api_key=fal_key, model=config.audio.tts.model, timeout_s=timeouts.tts),
        stt=FalSTT(api_key=fal_key, model=config.audio.stt.model, timeout_s=timeouts.stt),
        judge=build_judge(config.judge, secrets.typesafe_api_key, timeout_s=timeouts.judge),
        lipsync=(
            FalLipsync(api_key=fal_key, model=config.audio.lipsync.model, timeout_s=timeouts.video)
            if config.audio.lipsync.enabled
            else None
        ),
        dry_run=False,
        git_sha=git_sha(),
    )


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None
