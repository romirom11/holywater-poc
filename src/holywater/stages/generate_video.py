"""Stage 3: one shot -> one clip. The prompt is assembled by code, not by the LLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from holywater.config.schema import VideoConfig
from holywater.domain.artifacts import VideoArtifact
from holywater.domain.enums import AudioStrategy
from holywater.domain.script import Script, Shot
from holywater.media import FFmpeg
from holywater.observability.logging import get_logger
from holywater.prompts import PromptTemplate, load_prompt
from holywater.providers.base import VideoProvider, VideoRequest

log = get_logger(__name__)

_LANGUAGE_NAMES = {
    "uk": "Ukrainian",
    "en": "English",
    "ru": "Russian",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pl": "Polish",
    "it": "Italian",
}


@dataclass
class VideoDeps:
    video: VideoProvider
    ffmpeg: FFmpeg
    config: VideoConfig
    prompt: PromptTemplate = field(default_factory=lambda: load_prompt("seedance_shot"))


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    version: str


def build_shot_prompt(
    script: Script,
    shot: Shot,
    *,
    strategy: AudioStrategy,
    prompt: PromptTemplate,
    strict: bool = False,
) -> RenderedPrompt:
    """Native strategy speaks the line in-model; TTS overlay asks for a silent performance."""
    speak = shot.has_dialogue and strategy is AudioStrategy.NATIVE
    common = {
        "continuity": shot.continuity,
        "style": script.style,
        "visual_prompt": shot.visual_prompt,
        "camera": shot.camera,
        "ambient_audio": shot.ambient_audio,
        "duration_s": shot.duration_s,
    }
    if not speak:
        text = prompt.render("no_dialogue", **common)
        return RenderedPrompt(text=text, version=f"{prompt.version_tag}/no_dialogue")
    d = script.dialogue
    strict_block = (" " + prompt.render("strict", line=d.text)) if strict else ""
    text = prompt.render(
        "dialogue",
        **common,
        start_s=_fmt(d.start_s),
        end_s=_fmt(d.end_s),
        speaker=d.speaker,
        line=d.text,
        emotion=d.emotion,
        language_name=_LANGUAGE_NAMES.get(d.language, d.language),
        strict_block=strict_block,
    )
    suffix = "/dialogue_strict" if strict else "/dialogue"
    return RenderedPrompt(text=text, version=prompt.version_tag + suffix)


def generate_shot(
    script: Script,
    shot: Shot,
    deps: VideoDeps,
    *,
    dest: Path,
    attempt: int,
    strategy: AudioStrategy,
    strict: bool = False,
    seed: int | None = None,
) -> VideoArtifact:
    rendered = build_shot_prompt(script, shot, strategy=strategy, prompt=deps.prompt, strict=strict)
    request = VideoRequest(
        model=deps.config.model,
        prompt=rendered.text,
        duration_s=shot.duration_s,
        resolution=deps.config.resolution,
        aspect_ratio=deps.config.aspect_ratio,
        generate_audio=deps.config.generate_audio,
        seed=seed,
    )
    log.info(
        "generate_video.start",
        shot_id=shot.id,
        attempt=attempt,
        provider=deps.config.provider,
        model=deps.config.model,
        strategy=strategy.value,
        strict=strict,
        duration_s=shot.duration_s,
        resolution=deps.config.resolution.value,
        prompt_version=rendered.version,
    )
    result = deps.video.generate(request, dest=dest)
    info = deps.ffmpeg.probe(result.path)
    log.info(
        "generate_video.done",
        shot_id=shot.id,
        attempt=attempt,
        duration_s=round(info.duration_s, 2),
        has_audio=info.has_audio,
        request_id=result.request_id,
        cost_usd=result.cost_usd,
    )
    return VideoArtifact(
        shot_id=shot.id,
        attempt=attempt,
        strategy=strategy,
        path=result.path,
        source_url=result.source_url,
        seed=result.seed,
        duration_s=info.duration_s,
        resolution=deps.config.resolution,
        has_native_audio=info.has_audio,
        prompt_version=rendered.version,
        prompt_text=rendered.text,
        provider_request_id=result.request_id,
        model=result.model,
        cost_usd=result.cost_usd,
    )


def _fmt(value: float) -> str:
    return f"{value:g}"
