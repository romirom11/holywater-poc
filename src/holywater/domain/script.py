"""Script produced by the LLM (see docs/DATA_FLOW.md §3).

Self-contained invariants live here as pydantic validators. The cross-check against the input
text (the line must be a verbatim substring) needs text normalization and lives in
`quality.script_checks` so that `domain` keeps zero dependencies.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

MIN_SHOTS = 1
MAX_SHOTS = 2
MIN_SHOT_S = 4
MAX_SHOT_S = 30
MIN_TOTAL_S = 8
MAX_TOTAL_S = 30
# Conservative speech rate used to check that the line fits its time window.
CHARS_PER_SECOND = 15.0


class Dialogue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shot_id: int = Field(ge=1, le=MAX_SHOTS)
    speaker: str = Field(min_length=1, description="Visual description of who speaks.")
    text: str = Field(min_length=1, description="The line. Must be verbatim from the input.")
    language: str = Field(min_length=2, description="ISO 639-1 code of the spoken line.")
    emotion: str = Field(min_length=1, description="How the line is delivered.")
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)

    @model_validator(mode="after")
    def _window_is_positive(self) -> Self:
        if self.end_s <= self.start_s:
            msg = f"dialogue window must be positive, got {self.start_s}..{self.end_s}"
            raise ValueError(msg)
        return self

    @property
    def window_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def estimated_speech_s(self) -> float:
        return len(self.text) / CHARS_PER_SECOND


class Shot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(ge=1, le=MAX_SHOTS)
    duration_s: int = Field(ge=MIN_SHOT_S, le=MAX_SHOT_S)
    visual_prompt: str = Field(min_length=10, description="Scene, characters, action.")
    camera: str = Field(min_length=1)
    continuity: str = Field(
        min_length=1, description="Identity, wardrobe, location — identical across shots."
    )
    ambient_audio: str = Field(min_length=1)
    has_dialogue: bool


class Script(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    logline: str = Field(min_length=1)
    language: str = Field(min_length=2)
    style: str = Field(min_length=1)
    shots: list[Shot] = Field(min_length=MIN_SHOTS, max_length=MAX_SHOTS)
    dialogue: Dialogue

    @model_validator(mode="after")
    def _invariants(self) -> Self:
        problems = self.violations()
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def violations(self) -> list[str]:
        """Human-readable invariant violations; fed back to the LLM on retry."""
        problems: list[str] = []
        ids = [s.id for s in self.shots]
        if ids != list(range(1, len(ids) + 1)):
            problems.append(f"shot ids must be 1..{len(ids)} in order, got {ids}")

        total = sum(s.duration_s for s in self.shots)
        if not MIN_TOTAL_S <= total <= MAX_TOTAL_S:
            problems.append(f"total duration must be {MIN_TOTAL_S}..{MAX_TOTAL_S}s, got {total}s")

        with_dialogue = [s for s in self.shots if s.has_dialogue]
        if len(with_dialogue) != 1:
            problems.append(
                f"exactly one shot must have has_dialogue=true, got {len(with_dialogue)}"
            )
        elif with_dialogue[0].id != self.dialogue.shot_id:
            problems.append(
                f"dialogue.shot_id={self.dialogue.shot_id} but the dialogue shot is "
                f"{with_dialogue[0].id}"
            )
        else:
            shot = with_dialogue[0]
            if self.dialogue.end_s > shot.duration_s:
                problems.append(
                    f"dialogue ends at {self.dialogue.end_s}s but shot {shot.id} lasts "
                    f"{shot.duration_s}s"
                )
            if self.dialogue.window_s < self.dialogue.estimated_speech_s:
                problems.append(
                    f"dialogue window {self.dialogue.window_s:.1f}s is shorter than the estimated "
                    f"speech time {self.dialogue.estimated_speech_s:.1f}s for "
                    f"{len(self.dialogue.text)} characters"
                )
        return problems

    @property
    def dialogue_shot(self) -> Shot:
        return next(s for s in self.shots if s.id == self.dialogue.shot_id)

    @property
    def total_duration_s(self) -> int:
        return sum(s.duration_s for s in self.shots)

    def with_dialogue_text(self, text: str) -> Script:
        """Return a copy whose line is replaced by the verbatim original (ADR-005)."""
        return self.model_copy(update={"dialogue": self.dialogue.model_copy(update={"text": text})})

    def with_dialogue_window(self, start_s: float, end_s: float) -> Script:
        """Re-validated copy with a new line window (the `truncated` retry tactic)."""
        updated = self.model_copy(
            update={
                "dialogue": self.dialogue.model_copy(update={"start_s": start_s, "end_s": end_s})
            }
        )
        return Script.model_validate(updated.model_dump())

    def with_shot(self, shot: Shot) -> Script:
        """Re-validated copy with one shot replaced by id (prompt softening, duration change)."""
        shots = [shot if s.id == shot.id else s for s in self.shots]
        if shot.id not in {s.id for s in self.shots}:
            msg = f"script has no shot {shot.id}"
            raise ValueError(msg)
        return Script.model_validate(self.model_copy(update={"shots": shots}).model_dump())

    def with_shot_duration(self, shot_id: int, duration_s: int) -> Script:
        shot = next(s for s in self.shots if s.id == shot_id)
        return self.with_shot(shot.model_copy(update={"duration_s": duration_s}))
