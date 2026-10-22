"""Stage 2a: deterministic TTS of the verbatim line (used by the `tts_overlay` strategy)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from holywater.config.schema import TTSConfig
from holywater.domain.artifacts import AudioArtifact, ProviderCost
from holywater.domain.script import Script
from holywater.media import FFmpeg
from holywater.observability.errors import ProviderError, QualityGateError
from holywater.observability.logging import get_logger
from holywater.providers.base import TTSProvider, TTSRequest

log = get_logger(__name__)

# `ProviderCost.capability` of a TTS call (the flow maps it to `audio.tts.provider`).
TTS_CAPABILITY = "tts"


@dataclass
class AudioDeps:
    tts: TTSProvider
    ffmpeg: FFmpeg
    config: TTSConfig


def generate_line_audio(
    script: Script,
    deps: AudioDeps,
    *,
    dest: Path,
    voice: str | None = None,
    on_cost: Callable[[ProviderCost], None] | None = None,
) -> AudioArtifact:
    """TTS of the dialogue line. `voice` overrides `audio.tts.voice` (e.g. picked from the
    speaker); the returned `duration_s` is the measured length the flow fits into the shot.

    The charge is reported to `on_cost` as soon as it is known — after synthesis, before the
    file is probed, or the `spent_usd` of a failed call — so a probe failure and the task retry
    it triggers never lose it."""
    d = script.dialogue
    chosen_voice = voice if voice is not None else deps.config.voice
    log.info(
        "generate_audio.start",
        provider=deps.config.provider,
        model=deps.config.model,
        chars=len(d.text),
        language=d.language,
        voice=chosen_voice,
    )
    try:
        result = deps.tts.synthesize(
            TTSRequest(
                model=deps.config.model, text=d.text, language=d.language, voice=chosen_voice
            ),
            dest=dest,
        )
    except ProviderError as exc:
        if exc.spent_usd is not None and on_cost is not None:
            log.warning(
                "cost.spent_on_error",
                shot_id=d.shot_id,
                error=type(exc).__name__,
                spent_usd=exc.spent_usd,
                estimate=exc.spent_is_estimate,
            )
            on_cost(
                ProviderCost(
                    capability=TTS_CAPABILITY,
                    model=exc.model or deps.config.model,
                    units=float(len(d.text)),
                    cost_usd=exc.spent_usd,
                    is_estimate=exc.spent_is_estimate,
                    shot_id=d.shot_id,
                )
            )
        raise
    if on_cost is not None:
        on_cost(
            ProviderCost(
                capability=TTS_CAPABILITY,
                model=result.model,
                units=float(len(d.text)),
                cost_usd=result.cost_usd,
                is_estimate=result.cost_is_estimate,
                shot_id=d.shot_id,
            )
        )
    info = deps.ffmpeg.probe(result.path)
    if not info.has_audio or info.duration_s <= 0:
        msg = f"TTS returned no playable audio ({result.path})"
        raise QualityGateError(
            msg,
            gate="tts_audio",
            measured=round(info.duration_s, 3),
            threshold="audio stream, duration > 0",
            stage="generate_audio",
            context={
                "path": str(result.path),
                "model": result.model,
                "voice": chosen_voice,
                "cost_usd": result.cost_usd,  # paid even though unusable (reported above)
            },
        )
    log.info("generate_audio.done", duration_s=round(info.duration_s, 2), cost_usd=result.cost_usd)
    return AudioArtifact(
        shot_id=d.shot_id,
        path=result.path,
        duration_s=info.duration_s,
        model=result.model,
        voice=chosen_voice,
        text=d.text,
        cost_usd=result.cost_usd,
        cost_is_estimate=result.cost_is_estimate,
    )
