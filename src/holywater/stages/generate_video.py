"""Stage 3: one shot -> one clip. The prompt is assembled by code, not by the LLM."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from holywater.config.schema import VideoConfig
from holywater.domain.artifacts import ProviderCost, VideoArtifact
from holywater.domain.enums import AudioStrategy
from holywater.domain.script import Script, Shot
from holywater.media import FFmpeg
from holywater.observability.errors import BilledJobError, MediaError, ProviderError
from holywater.observability.logging import get_logger
from holywater.prompts import PromptTemplate, load_prompt
from holywater.providers.base import VideoProvider, VideoRequest

log = get_logger(__name__)

# `ProviderCost.capability` of a video call (the flow maps it to `video.provider`).
VIDEO_CAPABILITY = "video"
_STRICT_SUFFIX = "/dialogue_strict"

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
    start_s, end_s = _fmt(d.start_s), _fmt(d.end_s)
    # No empty "0–0s" or "8–8s" segments: the scene leads into the line only when the line starts
    # after 0, and the silent hold is described only when the line ends before the shot does.
    lead_in = prompt.render(
        "lead_in" if d.start_s > 0 else "lead_in_at_start",
        start_s=start_s,
        visual_prompt=shot.visual_prompt,
        camera=shot.camera,
    )
    tail = (
        " " + prompt.render("tail", end_s=end_s, duration_s=shot.duration_s, speaker=d.speaker)
        if d.end_s < shot.duration_s
        else ""
    )
    text = prompt.render(
        "dialogue",
        **common,
        lead_in=lead_in + " ",
        tail=tail,
        start_s=start_s,
        end_s=end_s,
        speaker=d.speaker,
        line=d.text,
        emotion=d.emotion,
        language_name=_LANGUAGE_NAMES.get(d.language, d.language),
        strict_block=strict_block,
    )
    suffix = _STRICT_SUFFIX if strict else "/dialogue"
    return RenderedPrompt(text=text, version=prompt.version_tag + suffix)


def is_strict(prompt_version: str) -> bool:
    """Whether a stored `prompt_version` was rendered with the strict-line suffix."""
    return prompt_version.endswith(_STRICT_SUFFIX)


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
    on_cost: Callable[[ProviderCost], None] | None = None,
) -> VideoArtifact:
    """One paid clip. Every charge is reported to `on_cost` as soon as it is known — the finished
    job before the clip is probed, and the `spent_usd` of a failed job before the error leaves —
    so a later failure (and the task retry it triggers) never loses it. A paid clip that ffprobe
    cannot read raises `BilledJobError` (not retried by the task: a retry would buy a new job)."""
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
    try:
        result = deps.video.generate(request, dest=dest)
    except ProviderError as exc:
        if exc.spent_usd is not None and on_cost is not None:
            log.warning(
                "cost.spent_on_error",
                shot_id=shot.id,
                attempt=attempt,
                error=type(exc).__name__,
                request_id=exc.request_id,
                spent_usd=exc.spent_usd,
                estimate=exc.spent_is_estimate,
            )
            on_cost(
                ProviderCost(
                    capability=VIDEO_CAPABILITY,
                    model=exc.model or deps.config.model,
                    units=float(shot.duration_s),
                    cost_usd=exc.spent_usd,
                    is_estimate=exc.spent_is_estimate,
                    shot_id=shot.id,
                )
            )
        raise
    if on_cost is not None:
        on_cost(
            ProviderCost(
                capability=VIDEO_CAPABILITY,
                model=result.model,
                units=float(shot.duration_s),
                cost_usd=result.cost_usd,
                is_estimate=result.cost_is_estimate,
                shot_id=shot.id,
            )
        )
    try:
        info = deps.ffmpeg.probe(result.path)
    except MediaError as exc:
        # The job is paid and its cost reported above. A MediaError is retryable, and a task retry
        # would buy a new clip; BilledJobError is not, so the flow's attempt budget decides.
        job_id = result.request_id or "unknown"
        msg = f"video job {job_id} was paid but its clip is unreadable: {exc.message}"
        raise BilledJobError(
            msg,
            provider=deps.config.provider,
            model=result.model,
            job_id=job_id,
            spent_usd=result.cost_usd,
            spent_is_estimate=result.cost_is_estimate,
            context={"path": str(result.path)},
        ) from exc
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
        cost_is_estimate=result.cost_is_estimate,
    )


def _fmt(value: float) -> str:
    return f"{value:g}"
