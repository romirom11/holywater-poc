from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator

MIN_IDEA_CHARS = 10
MAX_IDEA_CHARS = 1000


class IdeaInput(BaseModel):
    """The user's idea. `text` is the single source of truth for the verbatim line."""

    text: str = Field(min_length=MIN_IDEA_CHARS, max_length=MAX_IDEA_CHARS)
    language: str | None = Field(default=None, description="BCP-47 hint, e.g. 'uk' or 'en'.")
    received_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("text")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if len(value) < MIN_IDEA_CHARS:
            msg = f"idea text must be at least {MIN_IDEA_CHARS} characters after stripping"
            raise ValueError(msg)
        return value
