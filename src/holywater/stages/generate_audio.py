"""Stage 2a: deterministic TTS of the verbatim line (used by the `tts_overlay` strategy)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from holywater.config.schema import TTSConfig
from holywater.domain.artifacts import AudioArtifact
from holywater.domain.script import Script
from holywater.media import FFmpeg
from holywater.observability.logging import get_logger
from holywater.providers.base import TTSProvider, TTSRequest

log = get_logger(__name__)


@dataclass
class AudioDeps:
    tts: TTSProvider
    ffmpeg: FFmpeg
    config: TTSConfig


def generate_line_audio(script: Script, deps: AudioDeps, *, dest: Path) -> AudioArtifact:
    d = script.dialogue
    log.info(
        "generate_audio.start",
        provider=deps.config.provider,
        model=deps.config.model,
        chars=len(d.text),
        language=d.language,
        voice=deps.config.voice,
    )
    result = deps.tts.synthesize(
        TTSRequest(
            model=deps.config.model, text=d.text, language=d.language, voice=deps.config.voice
        ),
        dest=dest,
    )
    info = deps.ffmpeg.probe(result.path)
    log.info("generate_audio.done", duration_s=round(info.duration_s, 2), cost_usd=result.cost_usd)
    return AudioArtifact(
        shot_id=d.shot_id,
        path=result.path,
        duration_s=info.duration_s,
        model=result.model,
        voice=deps.config.voice,
        text=d.text,
        cost_usd=result.cost_usd,
        cost_is_estimate=result.cost_is_estimate,
    )
