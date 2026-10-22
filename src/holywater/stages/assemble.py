"""Stage 5: normalize clips, (mix the TTS line, lipsync it), concat, loudness-normalize, fit the
final into `quality.duration_s`, gate the result."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from holywater.config.schema import LipsyncConfig, QualityConfig
from holywater.domain.artifacts import AudioArtifact, FinalVideo, ProviderCost, VideoArtifact
from holywater.domain.enums import AudioStrategy, Resolution
from holywater.domain.media import MediaInfo
from holywater.media import FFmpeg
from holywater.media.ffmpeg import VERTICAL_SIZES
from holywater.observability.errors import QualityGateError
from holywater.observability.logging import get_logger
from holywater.providers.base import LipsyncProvider, LipsyncRequest
from holywater.quality.gates import DurationGate, MediaIntegrityGate

log = get_logger(__name__)

LIPSYNC_CAPABILITY = "lipsync"
# A final whose loudest sample is below this is silent (digital silence measures about -91 dB).
DEFAULT_MIN_AUDIO_PEAK_DB = -50.0


@dataclass
class AssembleDeps:
    ffmpeg: FFmpeg
    quality: QualityConfig
    resolution: Resolution
    lipsync: LipsyncProvider | None = None
    lipsync_config: LipsyncConfig | None = None
    min_audio_peak_db: float = DEFAULT_MIN_AUDIO_PEAK_DB


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
    """Build the final and gate it: `quality.duration_s` exactly, 9:16, audible audio.

    A final longer than the maximum is trimmed to it unless that would cut the line
    (`protect_until_s`, the line end on the final timeline, or the end of the mixed TTS voice);
    a shorter one fails the duration gate so the flow can extend a shot. Paid lipsync results
    are cached in `work_dir` and their cost is returned exactly once in
    `FinalVideo.provider_costs`.
    """
    if not shots:
        msg = "assemble needs at least one shot"
        raise ValueError(msg)
    ordered = sorted(shots, key=lambda s: s.shot_id)
    size = VERTICAL_SIZES[deps.resolution.value]
    work_dir.mkdir(parents=True, exist_ok=True)

    normalized: list[Path] = []
    timeline_s = 0.0
    protect = protect_until_s
    for shot in ordered:
        out = work_dir / f"norm_{shot.shot_id}.mp4"
        info = deps.ffmpeg.normalize_clip(shot.path, out, width=size[0], height=size[1])
        if voice is not None and shot.shot_id == voice.audio.shot_id:
            out = _voice_shot(shot, out, voice, deps, work_dir, size=size)
            voice_end_s = timeline_s + voice.offset_s + voice.audio.duration_s
            protect = voice_end_s if protect is None else max(protect, voice_end_s)
        normalized.append(out)
        timeline_s += info.duration_s

    joined = work_dir / "joined.mp4"
    if len(normalized) == 1:
        joined = normalized[0]
    else:
        deps.ffmpeg.concat(normalized, joined, list_file=work_dir / "concat.txt")

    final_info = deps.ffmpeg.loudnorm(joined, dest)
    final_info = _trim_to_max(final_info, deps, work_dir=work_dir, dest=dest, protect_s=protect)
    # Integrity first: a broken final is final, while a short one makes the flow pay to extend it.
    levels = deps.ffmpeg.audio_levels(dest) if final_info.has_audio else None
    MediaIntegrityGate(min_peak_db=deps.min_audio_peak_db).check(
        final_info, levels=levels
    ).raise_if_failed(stage="assemble")
    bounds = deps.quality.duration_s
    DurationGate(min_s=bounds.min, max_s=bounds.max).check(final_info.duration_s).raise_if_failed(
        stage="assemble"
    )

    final = FinalVideo(
        path=dest,
        duration_s=final_info.duration_s,
        width=final_info.width,
        height=final_info.height,
        fps=final_info.fps,
        has_audio=final_info.has_audio,
        strategy_used=strategy_used,
        shots=ordered,
        provider_costs=collect_unreported_costs(work_dir),
    )
    log.info(
        "assemble.done",
        path=str(dest),
        duration_s=round(final_info.duration_s, 2),
        size_bytes=final_info.size_bytes,
        strategy=strategy_used.value,
        peak_db=levels.max_volume_db if levels else None,
        provider_costs=len(final.provider_costs),
    )
    return final


def _trim_to_max(
    info: MediaInfo, deps: AssembleDeps, *, work_dir: Path, dest: Path, protect_s: float | None
) -> MediaInfo:
    max_s = deps.quality.duration_s.max
    if round(info.duration_s, 3) <= max_s:
        return info
    if protect_s is not None and protect_s > max_s:
        msg = f"final is {info.duration_s:.1f}s and the line ends at {protect_s:.1f}s"
        raise QualityGateError(
            msg + f"; trimming to {max_s}s would cut it",
            gate="duration",
            measured=round(info.duration_s, 3),
            threshold=f"{deps.quality.duration_s.min}..{max_s}",
            stage="assemble",
            context={"protect_until_s": round(protect_s, 3)},
        )
    log.warning("assemble.trimming", duration_s=round(info.duration_s, 3), max_s=max_s)
    trimmed = work_dir / "trimmed.mp4"
    deps.ffmpeg.trim(dest, trimmed, max_s=max_s)
    trimmed.replace(dest)
    return deps.ffmpeg.probe(dest)


# -- voice / lipsync ------------------------------------------------------------------------


def _voice_shot(
    shot: VideoArtifact,
    clip: Path,
    voice: VoiceOverlay,
    deps: AssembleDeps,
    work_dir: Path,
    *,
    size: tuple[int, int],
) -> Path:
    """Mix the TTS line into the normalized dialogue clip; lipsync the picture when enabled."""
    mixed = work_dir / f"{clip.stem}_voice.mp4"
    mixed_info = deps.ffmpeg.overlay_voice(clip, voice.audio.path, mixed, offset_s=voice.offset_s)
    if deps.lipsync is None or deps.lipsync_config is None:
        return mixed
    synced = _lipsync(
        deps.lipsync,
        deps.lipsync_config,
        deps.ffmpeg,
        shot=shot,
        mixed=mixed,
        clip_s=mixed_info.duration_s,
        voice=voice,
        work_dir=work_dir,
        size=size,
    )
    # The picture comes from lipsync, the sound stays the mix (ambient bed + voice at offset).
    out = work_dir / f"{clip.stem}_lipsync.mp4"
    info = deps.ffmpeg.normalize_clip(synced, out, width=size[0], height=size[1], audio_src=mixed)
    tolerance = deps.quality.shot_duration_tolerance_s
    DurationGate(
        min_s=mixed_info.duration_s - tolerance,
        max_s=mixed_info.duration_s + tolerance,
        name="lipsync_duration",
    ).check(info.duration_s).raise_if_failed(stage="assemble")
    return out


def _lipsync(
    provider: LipsyncProvider,
    config: LipsyncConfig,
    ffmpeg: FFmpeg,
    *,
    shot: VideoArtifact,
    mixed: Path,
    clip_s: float,
    voice: VoiceOverlay,
    work_dir: Path,
    size: tuple[int, int],
) -> Path:
    """One paid lipsync call per (clip, voice, offset, model, size): retries and resumes reuse
    the cached result. The voice is sent exactly as it is mixed: delayed by `offset_s` and padded
    with silence to the clip length, so mouth movement and the mixed line coincide."""
    key = _lipsync_key(shot.path, voice, model=config.model, size=size)
    synced = work_dir / f"lipsync_{key}.mp4"
    ledger_path = synced.with_suffix(".json")
    if synced.is_file() and _read_ledger(ledger_path) is not None:
        log.info("assemble.lipsync_cached", shot_id=shot.shot_id, path=str(synced))
        return synced
    placed = work_dir / f"voice_placed_{shot.shot_id}.wav"
    ffmpeg.place_voice(voice.audio.path, placed, offset_s=voice.offset_s, duration_s=clip_s)
    result = provider.sync(
        LipsyncRequest(model=config.model, video_path=mixed, audio_path=placed), dest=synced
    )
    if result.path.resolve() != synced.resolve():
        shutil.copyfile(result.path, synced)
    cost = ProviderCost(
        capability=LIPSYNC_CAPABILITY,
        model=result.model,
        units=round(clip_s, 3),
        cost_usd=result.cost_usd,
        is_estimate=result.cost_is_estimate,
        shot_id=shot.shot_id,
    )
    _write_ledger(ledger_path, _LipsyncLedger(cost=cost))
    log.info(
        "assemble.lipsync",
        shot_id=shot.shot_id,
        model=result.model,
        cost_usd=result.cost_usd,
        estimate=result.cost_is_estimate,
        cache=str(synced),
    )
    return synced


def _lipsync_key(clip: Path, voice: VoiceOverlay, *, model: str, size: tuple[int, int]) -> str:
    digest = hashlib.sha256()
    for path in (clip, voice.audio.path):
        with path.open("rb") as fh:
            digest.update(hashlib.file_digest(fh, "sha256").digest())
    digest.update(f"|{voice.offset_s:.3f}|{model}|{size[0]}x{size[1]}".encode())
    return digest.hexdigest()[:16]


class _LipsyncLedger(BaseModel):
    """Sidecar of a cached lipsync result: what it cost and whether a FinalVideo reported it."""

    model_config = ConfigDict(extra="forbid")

    cost: ProviderCost
    reported: bool = False


def _read_ledger(path: Path) -> _LipsyncLedger | None:
    if not path.is_file():
        return None
    try:
        return _LipsyncLedger.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        # Fallback: treat the cache entry as missing; the lipsync call is made (and recorded)
        # again rather than reusing a result whose cost is unknown.
        log.warning("assemble.lipsync_ledger_unreadable", path=str(path), error=str(exc))
        return None


def _write_ledger(path: Path, ledger: _LipsyncLedger) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(ledger.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)


def collect_unreported_costs(work_dir: Path) -> list[ProviderCost]:
    """Paid lipsync calls in `work_dir` that no final reported yet, marked reported now: each is
    returned exactly once. `assemble` calls it for the final it returns; the flow calls it when
    assemble fails, so a paid call is recorded even if no final is ever built."""
    ledgers = _unreported_ledgers(work_dir)
    for path, ledger in ledgers:
        _write_ledger(path, ledger.model_copy(update={"reported": True}))
    return [ledger.cost for _, ledger in ledgers]


def _unreported_ledgers(work_dir: Path) -> list[tuple[Path, _LipsyncLedger]]:
    """Lipsync calls paid in this or an earlier (failed) assemble that no final reported yet."""
    found: list[tuple[Path, _LipsyncLedger]] = []
    for path in sorted(work_dir.glob("lipsync_*.json")):
        ledger = _read_ledger(path)
        if ledger is not None and not ledger.reported:
            found.append((path, ledger))
    return found
