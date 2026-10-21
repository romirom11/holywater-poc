"""Thin, logged ffmpeg/ffprobe wrapper. Every failure becomes a `MediaError` with stderr context."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from holywater.domain.media import MediaInfo
from holywater.observability.errors import MediaError
from holywater.observability.logging import get_logger

log = get_logger(__name__)

_STDERR_TAIL_CHARS = 1500
DEFAULT_FPS = 24
# Target frame sizes per resolution label, vertical 9:16, both dimensions even for yuv420p.
VERTICAL_SIZES: dict[str, tuple[int, int]] = {"480p": (480, 854), "720p": (720, 1280)}


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
    ) -> MediaInfo:
        """Re-encode to a uniform format so clips can be concatenated losslessly afterwards.

        Scales with letterboxing to `width`x`height`, forces `fps`, and guarantees an AAC audio
        track (silence is synthesized when the source has none).
        """
        info = self.probe(src)
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},format=yuv420p"
        )
        args: list[str] = [self.ffmpeg_bin, "-y", "-v", "error", "-i", str(src)]
        if not info.has_audio:
            args += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        args += ["-vf", vf, "-map", "0:v:0"]
        args += ["-map", "0:a:0"] if info.has_audio else ["-map", "1:a:0", "-shortest"]
        args += [*_ENCODE_ARGS, str(dst)]
        self._run(args, op="normalize_clip")
        return self.probe(dst)

    def concat(
        self, clips: Sequence[Path], dst: Path, *, list_file: Path | None = None
    ) -> MediaInfo:
        """Concatenate already-normalized clips with the concat demuxer (stream copy)."""
        if not clips:
            raise MediaError("concat needs at least one clip")
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
    ) -> MediaInfo:
        """Mix a voice track over the clip's own audio (ambient ducked), starting at `offset_s`."""
        delay_ms = max(0, round(offset_s * 1000))
        filter_complex = (
            f"[1:a]adelay={delay_ms}|{delay_ms},aformat=channel_layouts=stereo[voice];"
            f"[0:a]volume={ambient_gain_db}dB,aformat=channel_layouts=stereo[amb];"
            f"[amb][voice]amix=inputs=2:duration=first:normalize=0[a]"
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
                str(dst),
            ],
            op="overlay_voice",
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
                "-movflags",
                "+faststart",
                str(dst),
            ],
            op="loudnorm",
        )
        return self.probe(dst)

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
        return proc.stdout


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
    return MediaInfo(
        duration_s=duration,
        width=int(video.get("width", 0)) if video else 0,
        height=int(video.get("height", 0)) if video else 0,
        fps=_parse_rate(video.get("avg_frame_rate") or video.get("r_frame_rate")) if video else 0.0,
        has_video=video is not None,
        has_audio=audio is not None,
        video_codec=video.get("codec_name") if video else None,
        audio_codec=audio.get("codec_name") if audio else None,
        size_bytes=int(fmt.get("size") or path.stat().st_size),
    )


def _parse_rate(value: str | None) -> float:
    if not value:
        return 0.0
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            return float(num) / float(den) if float(den) else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0
