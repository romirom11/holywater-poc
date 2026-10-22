"""Facts measured on media files (ffprobe, ffmpeg filters). Consumed by gates and reports."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MediaInfo(BaseModel):
    """Result of probing a media file (ffprobe). Consumed by quality gates and reports."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    duration_s: float = Field(ge=0)
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    fps: float = Field(ge=0)
    has_video: bool
    has_audio: bool
    video_codec: str | None = None
    audio_codec: str | None = None
    audio_sample_rate: int | None = None
    audio_channels: int | None = None
    size_bytes: int = Field(ge=0)

    @property
    def is_vertical(self) -> bool:
        return self.height > self.width

    @property
    def aspect_ratio(self) -> float:
        """width / height; 0.0 when there is no picture."""
        return self.width / self.height if self.height else 0.0


class AudioLevels(BaseModel):
    """Peak and mean level of an audio track (ffmpeg `volumedetect`), in dBFS.

    Digital silence measures about -91 dB; speech and ambient sound sit far above -50 dB.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mean_volume_db: float
    max_volume_db: float
