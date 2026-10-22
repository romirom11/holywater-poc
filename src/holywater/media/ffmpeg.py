"""Thin, logged ffmpeg/ffprobe wrapper. Every failure becomes a `MediaError` with stderr context."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from holywater.domain.media import AudioLevels, MediaInfo
from holywater.observability.errors import MediaError, QualityGateError
from holywater.observability.logging import get_logger

log = get_logger(__name__)

_STDERR_TAIL_CHARS = 1500
DEFAULT_FPS = 24
# Target frame sizes per resolution label, vertical 9:16, both dimensions even for yuv420p.
VERTICAL_SIZES: dict[str, tuple[int, int]] = {"480p": (480, 854), "720p": (720, 1280)}
# Every clip, mix and final carries 48 kHz stereo audio: concat stream-copies, so the audio
# parameters must be identical across clips, and loudnorm resamples internally to 192 kHz.
AUDIO_SAMPLE_RATE = 48000
_STEREO_48K = f"aformat=sample_rates={AUDIO_SAMPLE_RATE}:channel_layouts=stereo"
_AUDIO_OUT_ARGS: tuple[str, ...] = ("-ar", str(AUDIO_SAMPLE_RATE), "-ac", "2")
# A voice may end this much after the clip (encoder padding) before mixing counts as cutting it.
VOICE_FIT_TOLERANCE_S = 0.05


@dataclass(frozen=True)
class FFmpeg:
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    timeout_s: int = 600

    # -- availability ------------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

    # -- probing -----------------------------------------------------------------------------

    def probe(self, path: Path) -> MediaInfo:
        if not path.is_file():
            msg = f"media file not found: {path}"
            raise MediaError(msg, context={"path": str(path)})
        out = self._run(
            [
                self.ffprobe_bin,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            op="probe",
        )
        try:
            data: dict[str, Any] = json.loads(out)
        except json.JSONDecodeError as exc:
            msg = f"ffprobe returned invalid JSON for {path}"
            raise MediaError(msg, context={"path": str(path)}) from exc
        return _parse_probe(data, path)

    # -- transforms --------------------------------------------------------------------------

    def normalize_clip(
        self,
        src: Path,
        dst: Path,
        *,
        width: int,
        height: int,
        fps: int = DEFAULT_FPS,
        audio_src: Path | None = None,
    ) -> MediaInfo:
        """Re-encode to a uniform format so clips can be concatenated losslessly afterwards.

        Scales with letterboxing to `width`x`height`, forces `fps`, and guarantees a 48 kHz
        stereo AAC track (silence is synthesized when the source has none). `audio_src` takes
        the audio from another file instead (the picture of a lipsync result, the mix it was
        made from); the output then ends with the shorter of the two.
        """
        info = self.probe(src)
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},format=yuv420p"
        )
        args: list[str] = [self.ffmpeg_bin, "-y", "-v", "error", "-i", str(src)]
        if audio_src is not None:
            args += ["-i", str(audio_src)]
        elif not info.has_audio:
            args += [
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=channel_layout=stereo:sample_rate={AUDIO_SAMPLE_RATE}",
            ]
        own_audio = audio_src is None and info.has_audio
        args += ["-vf", vf, "-af", _STEREO_48K, "-map", "0:v:0"]
        args += ["-map", "0:a:0"] if own_audio else ["-map", "1:a:0", "-shortest"]
        args += [*_ENCODE_ARGS, *_AUDIO_OUT_ARGS, str(dst)]
        self._run(args, op="normalize_clip")
        return self.probe(dst)

    def concat(
        self, clips: Sequence[Path], dst: Path, *, list_file: Path | None = None
    ) -> MediaInfo:
        """Concatenate already-normalized clips with the concat demuxer (stream copy)."""
        if not clips:
            raise MediaError("concat needs at least one clip")
        _require_uniform_streams([(p, self.probe(p)) for p in clips])
        list_path = list_file or dst.with_suffix(".concat.txt")
        list_path.write_text("".join(f"file '{p.resolve()}'\n" for p in clips), encoding="utf-8")
        self._run(
            [
                self.ffmpeg_bin,
                "-y",
                "-v",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(dst),
            ],
            op="concat",
        )
        return self.probe(dst)

    def overlay_voice(
        self,
        video: Path,
        voice: Path,
        dst: Path,
        *,
        offset_s: float = 0.0,
        ambient_gain_db: float = -12.0,
        fit_tolerance_s: float = VOICE_FIT_TOLERANCE_S,
    ) -> MediaInfo:
        """Mix a voice track over the clip's own audio (ambient ducked), starting at `offset_s`.

        The mix keeps the clip's length, so a voice that would run past the clip end raises
        `QualityGateError(gate="voice_fit")` instead of being cut silently.
        """
        clip = self.probe(video)
        _check_voice_fit(
            self.probe(voice),
            voice_path=voice,
            offset_s=offset_s,
            clip_s=clip.duration_s,
            tolerance_s=fit_tolerance_s,
        )
        delay_ms = max(0, round(offset_s * 1000))
        filter_complex = (
            f"[1:a]{_STEREO_48K},adelay={delay_ms}|{delay_ms}[voice];"
            f"[0:a]volume={ambient_gain_db}dB,{_STEREO_48K}[amb];"
            f"[amb][voice]amix=inputs=2:duration=first:normalize=0,{_STEREO_48K}[a]"
        )
        self._run(
            [
                self.ffmpeg_bin,
                "-y",
                "-v",
                "error",
                "-i",
                str(video),
                "-i",
                str(voice),
                "-filter_complex",
                filter_complex,
                "-map",
                "0:v:0",
                "-map",
                "[a]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                *_AUDIO_OUT_ARGS,
                str(dst),
            ],
            op="overlay_voice",
        )
        return self.probe(dst)

    def place_voice(
        self,
        voice: Path,
        dst: Path,
        *,
        offset_s: float,
        duration_s: float,
        fit_tolerance_s: float = VOICE_FIT_TOLERANCE_S,
    ) -> MediaInfo:
        """The voice exactly as `overlay_voice` positions it: `offset_s` of leading silence, then
        silence up to `duration_s` (the clip length). 48 kHz stereo WAV, the lipsync input."""
        _check_voice_fit(
            self.probe(voice),
            voice_path=voice,
            offset_s=offset_s,
            clip_s=duration_s,
            tolerance_s=fit_tolerance_s,
        )
        delay_ms = max(0, round(offset_s * 1000))
        self._run(
            [
                self.ffmpeg_bin,
                "-y",
                "-v",
                "error",
                "-i",
                str(voice),
                "-af",
                f"{_STEREO_48K},adelay={delay_ms}|{delay_ms},apad",
                "-t",
                f"{duration_s:.3f}",
                "-c:a",
                "pcm_s16le",
                *_AUDIO_OUT_ARGS,
                str(dst),
            ],
            op="place_voice",
        )
        return self.probe(dst)

    def loudnorm(self, src: Path, dst: Path, *, target_lufs: float = -16.0) -> MediaInfo:
        self._run(
            [
                self.ffmpeg_bin,
                "-y",
                "-v",
                "error",
                "-i",
                str(src),
                "-af",
                f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                *_AUDIO_OUT_ARGS,  # loudnorm upsamples to 192 kHz internally
                "-movflags",
                "+faststart",
                str(dst),
            ],
            op="loudnorm",
        )
        return self.probe(dst)

    def extract_audio(
        self,
        src: Path,
        dst: Path,
        *,
        sample_rate: int = 16000,
        start_s: float | None = None,
        duration_s: float | None = None,
    ) -> MediaInfo:
        """Mono 16-bit WAV of the audio track — the input every STT vendor accepts.

        `start_s`/`duration_s` cut a window (used to verify only the line segment of the final).
        """
        window: list[str] = []
        if start_s is not None:
            window += ["-ss", f"{max(0.0, start_s):.3f}"]
        if duration_s is not None:
            window += ["-t", f"{max(0.1, duration_s):.3f}"]
        self._run(
            [
                self.ffmpeg_bin,
                "-y",
                "-v",
                "error",
                *window,
                "-i",
                str(src),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-c:a",
                "pcm_s16le",
                str(dst),
            ],
            op="extract_audio",
        )
        return self.probe(dst)

    def audio_levels(
        self, src: Path, *, start_s: float | None = None, duration_s: float | None = None
    ) -> AudioLevels:
        """Peak and mean level of the audio track (`volumedetect`), optionally over a window."""
        window: list[str] = []
        if start_s is not None:
            window += ["-ss", f"{max(0.0, start_s):.3f}"]
        if duration_s is not None:
            window += ["-t", f"{max(0.1, duration_s):.3f}"]
        proc = self._exec(
            [
                self.ffmpeg_bin,
                "-hide_banner",
                "-nostats",
                "-v",
                "info",
                *window,
                "-i",
                str(src),
                "-vn",
                "-af",
                "volumedetect",
                "-f",
                "null",
                "-",
            ],
            op="audio_levels",
        )
        return _parse_volumedetect(proc.stderr, src)

    def trim(self, src: Path, dst: Path, *, max_s: float) -> MediaInfo:
        self._run(
            [
                self.ffmpeg_bin,
                "-y",
                "-v",
                "error",
                "-i",
                str(src),
                "-t",
                f"{max_s:.3f}",
                *_ENCODE_ARGS,
                str(dst),
            ],
            op="trim",
        )
        return self.probe(dst)

    # -- synthetic media (tests, --dry-run) --------------------------------------------------

    def synth_clip(
        self,
        dst: Path,
        *,
        duration_s: float,
        width: int,
        height: int,
        with_audio: bool = True,
        fps: int = DEFAULT_FPS,
    ) -> MediaInfo:
        """Generate a test pattern clip (optionally with a tone) without any external model."""
        args: list[str] = [
            self.ffmpeg_bin,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={width}x{height}:rate={fps}",
        ]
        if with_audio:
            args += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
        vf = "format=yuv420p"
        args += ["-vf", vf, "-t", f"{duration_s:.3f}", *_ENCODE_ARGS, str(dst)]
        self._run(args, op="synth_clip")
        return self.probe(dst)

    def synth_audio(self, dst: Path, *, duration_s: float, frequency: int = 220) -> MediaInfo:
        self._run(
            [
                self.ffmpeg_bin,
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={frequency}:sample_rate=48000",
                "-t",
                f"{duration_s:.3f}",
                "-c:a",
                "libmp3lame",
                "-q:a",
                "4",
                str(dst),
            ],
            op="synth_audio",
        )
        return self.probe(dst)

    # -- internals ---------------------------------------------------------------------------

    def _run(self, args: list[str], *, op: str) -> str:
        return self._exec(args, op=op).stdout

    def _exec(self, args: list[str], *, op: str) -> subprocess.CompletedProcess[str]:
        log.debug("ffmpeg.start", op=op, args=args)
        try:
            proc = subprocess.run(
                args, capture_output=True, text=True, timeout=self.timeout_s, check=False
            )
        except FileNotFoundError as exc:
            msg = f"{args[0]} not found on PATH"
            raise MediaError(msg, context={"op": op}) from exc
        except subprocess.TimeoutExpired as exc:
            msg = f"{op} timed out after {self.timeout_s}s"
            raise MediaError(msg, context={"op": op, "args": args}) from exc
        if proc.returncode != 0:
            tail = proc.stderr[-_STDERR_TAIL_CHARS:]
            log.error("ffmpeg.failed", op=op, returncode=proc.returncode, stderr=tail)
            msg = f"{op} failed with exit code {proc.returncode}"
            raise MediaError(msg, context={"op": op, "returncode": proc.returncode, "stderr": tail})
        log.debug("ffmpeg.done", op=op)
        return proc


_ENCODE_ARGS: tuple[str, ...] = (
    "-c:v",
    "libx264",
    "-preset",
    "veryfast",
    "-crf",
    "20",
    "-pix_fmt",
    "yuv420p",
    "-c:a",
    "aac",
    "-b:a",
    "192k",
    "-ar",
    "48000",
    "-movflags",
    "+faststart",
)


def _parse_probe(data: dict[str, Any], path: Path) -> MediaInfo:
    streams: list[dict[str, Any]] = data.get("streams", [])
    fmt: dict[str, Any] = data.get("format", {})
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = float(fmt.get("duration") or (video or audio or {}).get("duration") or 0.0)
    fps = 0.0
    if video:  # avg_frame_rate is "0/0" for some streams; r_frame_rate is the declared rate
        fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate"))
    return MediaInfo(
        duration_s=duration,
        width=int(video.get("width", 0)) if video else 0,
        height=int(video.get("height", 0)) if video else 0,
        fps=fps,
        has_video=video is not None,
        has_audio=audio is not None,
        video_codec=video.get("codec_name") if video else None,
        audio_codec=audio.get("codec_name") if audio else None,
        audio_sample_rate=_positive_int(audio.get("sample_rate")) if audio else None,
        audio_channels=_positive_int(audio.get("channels")) if audio else None,
        size_bytes=int(fmt.get("size") or path.stat().st_size),
    )


def _positive_int(value: object) -> int | None:
    text = str(value) if isinstance(value, (int, str)) else ""
    return int(text) if text.isdigit() and int(text) > 0 else None


def _parse_rate(value: str | None) -> float:
    """ffprobe rate ("24/1", "30000/1001", "25"); 0.0 = unknown, which `MediaIntegrityGate`
    reports as a failure on the final."""
    if not value:
        return 0.0
    num, _, den = value.partition("/")
    try:
        numerator, denominator = float(num), float(den or 1)
    except ValueError:
        log.warning("ffmpeg.unparsable_rate", value=value, fallback=0.0)
        return 0.0
    return numerator / denominator if denominator else 0.0


_VOLUME_RE = {
    key: re.compile(rf"{key}:\s*(-?(?:inf|\d+(?:\.\d+)?))\s*dB")
    for key in ("mean_volume", "max_volume")
}
_NO_SAMPLES_RE = re.compile(r"n_samples:\s*0\b")
# Reported instead of -inf (no samples, or all-zero float audio); 16-bit silence reads -91 dB.
SILENCE_FLOOR_DB = -120.0


def _parse_volumedetect(stderr: str, path: Path) -> AudioLevels:
    """Last `mean_volume`/`max_volume` lines of a `volumedetect` run (earlier ones are probes).

    A window without a single audio sample (a gap in the track) measures as silence.
    """
    found: dict[str, float] = {}
    for key, pattern in _VOLUME_RE.items():
        matches = pattern.findall(stderr)
        if matches:
            found[key] = max(SILENCE_FLOOR_DB, float(matches[-1]))
    if len(found) == len(_VOLUME_RE):
        return AudioLevels(mean_volume_db=found["mean_volume"], max_volume_db=found["max_volume"])
    if not found and _NO_SAMPLES_RE.search(stderr):
        return AudioLevels(mean_volume_db=SILENCE_FLOOR_DB, max_volume_db=SILENCE_FLOOR_DB)
    msg = f"volumedetect output for {path} has no levels"
    tail = stderr[-_STDERR_TAIL_CHARS:]
    raise MediaError(msg, context={"op": "audio_levels", "path": str(path), "stderr": tail})


def _check_voice_fit(
    voice: MediaInfo, *, voice_path: Path, offset_s: float, clip_s: float, tolerance_s: float
) -> None:
    end_s = max(0.0, offset_s) + voice.duration_s
    if end_s > clip_s + tolerance_s:
        msg = (
            f"voice ends at {end_s:.2f}s but the clip is {clip_s:.2f}s long; the line would be cut"
        )
        raise QualityGateError(
            msg,
            gate="voice_fit",
            measured=round(end_s, 3),
            threshold=round(clip_s, 3),
            context={
                "voice": str(voice_path),
                "offset_s": offset_s,
                "voice_duration_s": round(voice.duration_s, 3),
                "tolerance_s": tolerance_s,
            },
        )


def _require_uniform_streams(probed: Sequence[tuple[Path, MediaInfo]]) -> None:
    """The concat demuxer stream-copies: differing sizes, codecs or audio layouts corrupt the
    join (e.g. a mono clip before a stereo one loses seconds of audio in loudnorm)."""
    formats = {
        str(path): (
            info.width,
            info.height,
            info.video_codec,
            info.audio_codec,
            info.audio_sample_rate,
            info.audio_channels,
        )
        for path, info in probed
    }
    if len(set(formats.values())) > 1:
        msg = "concat inputs differ in stream format (width, height, codecs, rate, channels)"
        raise MediaError(msg, context={"op": "concat", "formats": formats})
