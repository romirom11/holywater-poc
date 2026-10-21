"""Deterministic in-process providers for `--dry-run` and tests. No network, no money.

The fakes simulate the one behaviour that matters for the pipeline logic: what the video model
"speaks". `FakeVideo` writes the quoted line from the prompt to a sidecar file next to the clip,
and `FakeSTT` reads it back — so verification passes unless a `speech_mutator` is installed.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from holywater.domain.idea import IdeaInput
from holywater.domain.script import CHARS_PER_SECOND
from holywater.media import FFmpeg
from holywater.media.ffmpeg import VERTICAL_SIZES
from holywater.observability.errors import RetryableProviderError
from holywater.providers.base import (
    JudgeAnswer,
    LipsyncResult,
    LLMResult,
    STTResult,
    TTSRequest,
    TTSResult,
    VideoRequest,
    VideoResult,
)
from holywater.quality.text_norm import extract_quoted

T = TypeVar("T", bound=BaseModel)

SPEECH_SIDECAR_SUFFIX = ".speech.txt"
_YES_THRESHOLD = 0.5
# Permissive defaults: risky content is unlikely, the line fits its window.
_BOOL_DEFAULTS = {"safety_risk": 0.05, "line_fits_window": 0.95}
_QUOTED_IN_PROMPT = re.compile(r'says "([^"]+)"')


# -- LLM --------------------------------------------------------------------------------------


@dataclass
class FakeLLM:
    """Builds a plausible two-shot script from the idea. `line_override` simulates paraphrasing."""

    model: str = "fake/llm"
    line_override: str | None = None
    override_times: int = 0
    calls: int = 0

    def complete_structured(
        self, *, system: str, user: str, schema: type[T], temperature: float
    ) -> LLMResult[T]:
        self.calls += 1
        idea_text = _extract_idea(user)
        line = _pick_line(idea_text)
        if self.line_override is not None and self.calls <= self.override_times:
            line = self.line_override
        language = "uk" if re.search(r"[А-Яа-яЇїІіЄєҐґ]", line) else "en"
        payload = {
            "title": "Fake title",
            "logline": idea_text[:120],
            "language": language,
            "style": "cinematic, moody, 35mm",
            "shots": [
                {
                    "id": 1,
                    "duration_s": 5,
                    "visual_prompt": "Establishing shot derived from the idea.",
                    "camera": "slow push-in",
                    "continuity": "same character, same wardrobe, same location",
                    "ambient_audio": "room tone",
                    "has_dialogue": False,
                },
                {
                    "id": 2,
                    "duration_s": 8,
                    "visual_prompt": "Close-up on the speaker delivering the line.",
                    "camera": "handheld close-up",
                    "continuity": "same character, same wardrobe, same location",
                    "ambient_audio": "room tone",
                    "has_dialogue": True,
                },
            ],
            "dialogue": {
                "shot_id": 2,
                "speaker": "the main character",
                "text": line,
                "language": language,
                "emotion": "quiet, intense",
                "start_s": 1.0,
                "end_s": min(7.5, 1.0 + max(1.5, len(line) / CHARS_PER_SECOND + 0.5)),
            },
        }
        return LLMResult(data=schema.model_validate(payload), model=self.model)


def _extract_idea(user_prompt: str) -> str:
    match = re.search(r"<idea>\s*(.*?)\s*</idea>", user_prompt, flags=re.S)
    return match.group(1) if match else user_prompt


def _pick_line(idea_text: str) -> str:
    quoted = extract_quoted(idea_text)
    if quoted:
        return quoted[0]
    first_sentence = re.split(r"(?<=[.!?])\s+", idea_text.strip(), maxsplit=1)[0]
    return first_sentence.rstrip(".!?") or idea_text


# -- Video ------------------------------------------------------------------------------------


@dataclass
class FakeVideo:
    """Synthesizes a test-pattern clip with ffmpeg and records the "spoken" line in a sidecar."""

    ffmpeg: FFmpeg = field(default_factory=FFmpeg)
    model: str = "fake/video"
    fail_times: int = 0
    speech_mutator: Callable[[str, int], str] | None = None
    duration_offset_s: float = 0.0
    calls: int = 0

    def generate(self, request: VideoRequest, *, dest: Path) -> VideoResult:
        self.calls += 1
        if self.calls <= self.fail_times:
            msg = f"simulated transient failure #{self.calls}"
            raise RetryableProviderError(msg, provider="fake", model=self.model, status_code=503)
        width, height = VERTICAL_SIZES[request.resolution.value]
        spoken = _line_from_prompt(request.prompt)
        if spoken and self.speech_mutator is not None:
            spoken = self.speech_mutator(spoken, self.calls)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.ffmpeg.synth_clip(
            dest,
            duration_s=request.duration_s + self.duration_offset_s,
            width=width,
            height=height,
            with_audio=request.generate_audio,
        )
        sidecar = dest.with_name(dest.name + SPEECH_SIDECAR_SUFFIX)
        sidecar.write_text(spoken or "", encoding="utf-8")
        return VideoResult(
            path=dest,
            model=self.model,
            seed=request.seed if request.seed is not None else 1000 + self.calls,
            request_id=f"fake-video-{self.calls}",
            cost_usd=0.0,
        )


def _line_from_prompt(prompt: str) -> str | None:
    match = _QUOTED_IN_PROMPT.search(prompt)
    return match.group(1) if match else None


# -- TTS / STT / lipsync ----------------------------------------------------------------------


@dataclass
class FakeTTS:
    ffmpeg: FFmpeg = field(default_factory=FFmpeg)
    model: str = "fake/tts"
    calls: int = 0

    def synthesize(self, request: TTSRequest, *, dest: Path) -> TTSResult:
        self.calls += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        duration = max(0.5, len(request.text) / CHARS_PER_SECOND)
        self.ffmpeg.synth_audio(dest, duration_s=duration)
        sidecar = dest.with_name(dest.name + SPEECH_SIDECAR_SUFFIX)
        sidecar.write_text(request.text, encoding="utf-8")
        return TTSResult(path=dest, model=self.model)


@dataclass
class FakeSTT:
    """Returns the sidecar text written by `FakeVideo`/`FakeTTS`, or a queued transcript."""

    model: str = "fake/stt"
    queue: list[str] = field(default_factory=list)
    fail_times: int = 0
    calls: int = 0

    def transcribe(self, audio_or_video: Path, *, language: str | None = None) -> STTResult:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RetryableProviderError("simulated STT outage", provider="fake", model=self.model)
        if self.queue:
            return STTResult(text=self.queue.pop(0), model=self.model, language=language)
        sidecar = audio_or_video.with_name(audio_or_video.name + SPEECH_SIDECAR_SUFFIX)
        text = sidecar.read_text(encoding="utf-8") if sidecar.is_file() else ""
        return STTResult(text=text, model=self.model, language=language)


@dataclass
class FakeLipsync:
    model: str = "fake/lipsync"
    calls: int = 0

    def sync(self, video: Path, audio: Path, *, dest: Path) -> LipsyncResult:
        self.calls += 1
        shutil.copyfile(video, dest)
        return LipsyncResult(path=dest, model=self.model)


# -- Judge ------------------------------------------------------------------------------------


@dataclass
class FakeJudge:
    """Scripted answers keyed by a substring of the question; defaults are permissive."""

    enabled: bool = True
    bool_answers: dict[str, float] = field(default_factory=dict)
    choice_answers: dict[str, str] = field(default_factory=dict)
    score_answers: dict[str, int] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    def ask_bool(self, state: str, question: str) -> JudgeAnswer:
        self.calls.append(question)
        default = _lookup(_BOOL_DEFAULTS, question, 0.05)
        p_yes = _lookup(self.bool_answers, question, default)
        answer = "yes" if p_yes >= _YES_THRESHOLD else "no"
        return JudgeAnswer(
            question=question,
            answer=answer,
            probabilities={"yes": p_yes, "no": 1 - p_yes},
            model_version="fake",
            latency_ms=1,
        )

    def ask_choice(self, state: str, question: str, options: Sequence[str]) -> JudgeAnswer:
        self.calls.append(question)
        answer = _lookup(self.choice_answers, question, options[0])
        probs = {o: (0.9 if o == answer else 0.1 / max(1, len(options) - 1)) for o in options}
        return JudgeAnswer(
            question=question,
            answer=answer,
            probabilities=probs,
            model_version="fake",
            latency_ms=1,
        )

    def ask_score(self, state: str, question: str, *, levels: int) -> JudgeAnswer:
        self.calls.append(question)
        score = _lookup(self.score_answers, question, levels)
        probs = {
            str(i): (0.9 if i == score else 0.1 / max(1, levels - 1)) for i in range(1, levels + 1)
        }
        return JudgeAnswer(
            question=question,
            answer=str(score),
            probabilities=probs,
            model_version="fake",
            latency_ms=1,
        )


def _lookup[V](table: dict[str, V], question: str, default: V) -> V:
    for key, value in table.items():
        if key in question:
            return value
    return default


def paraphrase_first_n(n: int) -> Callable[[str, int], str]:
    """Speech mutator: attempts 1..n paraphrase the line, later attempts are verbatim."""

    def mutate(line: str, call_index: int) -> str:
        return f"{line} or something like that" if call_index <= n else line

    return mutate


def fake_idea(
    text: str = 'Rainy night. A taxi driver asks: "Are you sure this is the address?"',
) -> IdeaInput:
    return IdeaInput(text=text)
