"""Stage 5: normalize clips, (mix the TTS line), concat, loudness-normalize, gate the result."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from holywater.config.schema import LipsyncConfig, QualityConfig
from holywater.domain.artifacts import AudioArtifact, FinalVideo, VideoArtifact
from holywater.domain.enums import AudioStrategy, Resolution
from holywater.media import FFmpeg
from holywater.media.ffmpeg import VERTICAL_SIZES
from holywater.observability.errors import QualityGateError
from holywater.observability.logging import get_logger
from holywater.providers.base import LipsyncProvider, LipsyncRequest
from holywater.quality.gates import DurationGate, MediaIntegrityGate

log = get_logger(__name__)


@dataclass
class AssembleDeps:
    ffmpeg: FFmpeg
    quality: QualityConfig
    resolution: Resolution
    lipsync: LipsyncProvider | None = None
    lipsync_config: LipsyncConfig | None = None


@dataclass(frozen=True)
class VoiceOverlay:
    """The TTS line to mix into the dialogue shot (tts_overlay strategy)."""

    audio: AudioArtifact
    offset_s: float


def assemble(
    shots: list[VideoArtifact],
    deps: AssembleDeps,
    *,
    work_dir: Path,
    dest: Path,
    strategy_used: AudioStrategy,
    voice: VoiceOverlay | None = None,
    protect_until_s: float | None = None,
) -> FinalVideo:
    """`protect_until_s`: where the line ends on the final timeline; trimming never cuts it."""
    if not shots:
        msg = "assemble needs at least one shot"
        raise ValueError(msg)
    ordered = sorted(shots, key=lambda s: s.shot_id)
    width, height = VERTICAL_SIZES[deps.resolution.value]
    work_dir.mkdir(parents=True, exist_ok=True)

    normalized: list[Path] = []
    for shot in ordered:
        out = work_dir / f"norm_{shot.shot_id}.mp4"
        deps.ffmpeg.normalize_clip(shot.path, out, width=width, height=height)
        if voice is not None and shot.shot_id == voice.audio.shot_id:
            out = _mix_voice(out, voice, deps, work_dir)
        normalized.append(out)

    joined = work_dir / "joined.mp4"
    if len(normalized) == 1:
        joined = normalized[0]
    else:
        deps.ffmpeg.concat(normalized, joined, list_file=work_dir / "concat.txt")

    final_info = deps.ffmpeg.loudnorm(joined, dest)
    max_s = deps.quality.duration_s.max
    if final_info.duration_s > max_s + 0.5:
        if protect_until_s is not None and protect_until_s > max_s:
            msg = (
                f"final is {final_info.duration_s:.1f}s and the line ends at {protect_until_s:.1f}s"
            )
            raise QualityGateError(
                msg + f"; trimming to {max_s}s would cut it",
                gate="duration",
                measured=round(final_info.duration_s, 3),
                threshold=f"{deps.quality.duration_s.min}..{max_s}",
                stage="assemble",
                context={"protect_until_s": protect_until_s},
            )
        log.warning("assemble.trimming", duration_s=final_info.duration_s, max_s=max_s)
        trimmed = work_dir / "trimmed.mp4"
        deps.ffmpeg.trim(dest, trimmed, max_s=max_s)
        trimmed.replace(dest)
        final_info = deps.ffmpeg.probe(dest)

    DurationGate(min_s=deps.quality.duration_s.min, max_s=max_s, tolerance_s=0.5).check(
        final_info.duration_s
    ).raise_if_failed(stage="assemble")
    MediaIntegrityGate().check(final_info).raise_if_failed(stage="assemble")

    log.info(
        "assemble.done",
        path=str(dest),
        duration_s=round(final_info.duration_s, 2),
        size_bytes=final_info.size_bytes,
        strategy=strategy_used.value,
    )
    return FinalVideo(
        path=dest,
        duration_s=final_info.duration_s,
        width=final_info.width,
        height=final_info.height,
        fps=final_info.fps,
        has_audio=final_info.has_audio,
        strategy_used=strategy_used,
        shots=ordered,
    )


def _mix_voice(clip: Path, voice: VoiceOverlay, deps: AssembleDeps, work_dir: Path) -> Path:
    mixed = work_dir / f"{clip.stem}_voice.mp4"
    deps.ffmpeg.overlay_voice(clip, voice.audio.path, mixed, offset_s=voice.offset_s)
    if deps.lipsync is None or deps.lipsync_config is None:
        return mixed
    synced = work_dir / f"{clip.stem}_lipsync.mp4"
    result = deps.lipsync.sync(
        LipsyncRequest(
            model=deps.lipsync_config.model, video_path=mixed, audio_path=voice.audio.path
        ),
        dest=synced,
    )
    log.info("assemble.lipsync", model=result.model, cost_usd=result.cost_usd)
    return result.path
