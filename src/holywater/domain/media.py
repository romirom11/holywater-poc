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
    size_bytes: int = Field(ge=0)

    @property
    def is_vertical(self) -> bool:
        return self.height > self.width
