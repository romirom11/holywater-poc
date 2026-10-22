"""Deterministic in-process providers for `--dry-run` and tests. No network, no money.

The fakes simulate the one behaviour that matters for the pipeline logic: what is spoken, and
when. `FakeVideo` writes the quoted line from the prompt — with the time span the prompt gives
it — to a sidecar file next to the clip, `FakeTTS` does the same for its voice file, and
`FakeSTT` reads them back, keeping only the words whose timing falls inside the segment that was
extracted for transcription. So verification passes unless a `speech_mutator` is installed or
the transcribed window misses the speech.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from holywater.domain.idea import IdeaInput
from holywater.domain.script import CHARS_PER_SECOND
from holywater.media import FFmpeg
from holywater.media.ffmpeg import VERTICAL_SIZES
from holywater.observability.errors import (
    BilledJobError,
    FatalProviderError,
    PromptRejectedError,
    RetryableProviderError,
    StructuredOutputError,
)
from holywater.observability.logging import get_logger
from holywater.providers.base import (
    JudgeAnswer,
    LipsyncRequest,
    LipsyncResult,
    LLMResult,
    STTRequest,
    STTResult,
    TTSRequest,
    TTSResult,
    VideoRequest,
    VideoResult,
)
from holywater.quality.text_norm import extract_quoted

log = get_logger(__name__)

SPEECH_SIDECAR_SUFFIX = ".speech.txt"
_YES_THRESHOLD = 0.5
# Permissive defaults: risky content is unlikely, the line fits its window.
_BOOL_DEFAULTS = {"safety_risk": 0.05, "line_fits_window": 0.95}
_SAYS_IN_PROMPT = re.compile(r'says "([^"]+)"')
_SPAN_IN_PROMPT = re.compile(r"(\d+(?:\.\d+)?)–(\d+(?:\.\d+)?)s:")


# -- LLM --------------------------------------------------------------------------------------


@dataclass
class FakeLLM:
    """Builds a plausible two-shot script from the idea.

    `fail_times` makes the first calls transient errors (429/5xx: `RetryableProviderError`);
    `line_override` simulates paraphrasing (first `override_times` calls); `invalid_times` makes
    the first calls return billed but unusable output (`StructuredOutputError`); every answer
    costs `cost_usd` (0 by default: dry runs are free).
    """

    line_override: str | None = None
    override_times: int = 0
    invalid_times: int = 0
    fail_times: int = 0
    cost_usd: float = 0.0
    voice_gender: str = "female"
    calls: int = 0
    users: list[str] = field(default_factory=list)

    def complete_structured[T: BaseModel](
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type[T],
        temperature: float | None = None,
    ) -> LLMResult[T]:
        self.calls += 1
        self.users.append(user)
        if self.calls <= self.fail_times:
            msg = f"simulated LLM 503 #{self.calls}"
            raise RetryableProviderError(
                msg, provider="fake", model=f"fake/{model}", status_code=503
            )
        if self.calls <= self.invalid_times:
            msg = f"simulated unusable LLM output #{self.calls}"
            raise StructuredOutputError(
                msg,
                provider="fake",
                model=f"fake/{model}",
                raw_text="{not json",
                spent_usd=self.cost_usd,
                spent_is_estimate=False,
            )
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
                "voice_gender": self.voice_gender,
            },
        }
        return LLMResult(
            data=schema.model_validate(payload),
            model=f"fake/{model}",
            cost_usd=self.cost_usd,
            cost_is_estimate=False,
        )


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
    """Synthesizes a test-pattern clip with ffmpeg and records the "spoken" line in a sidecar.

    Failure knobs (all counted in `calls`): `fail_times` transient errors first, `reject_times`
    content-policy rejections (`PromptRejectedError`) first, `reject_if_contains` rejects every
    prompt containing that text (e.g. the line), `expire_times` expired jobs (plain
    `FatalProviderError`) first, `billed_times` accepted-but-uncollectable jobs (`BilledJobError`
    charging `billed_spent_usd`, with `billed_status_code` as its HTTP status) first,
    `unreadable_calls` paid clips delivered as bytes ffprobe cannot read,
    `fail_from_call` transient errors from that call on,
    `duration_offset_s` applied to the first `duration_offset_times` clips (None = all) or, when
    `duration_offset_calls` is set, to exactly those calls,
    `mute_calls` clips delivered without an audio track, `stray_speech` words "spoken" in the
    first `stray_speech_times` clips whose prompt has no line (None = all). Each clip costs
    `usd_per_s` per requested second (0 by default).
    """

    ffmpeg: FFmpeg = field(default_factory=FFmpeg)
    fail_times: int = 0
    reject_times: int = 0
    reject_if_contains: str | None = None
    expire_times: int = 0
    billed_times: int = 0
    billed_spent_usd: float = 0.0
    billed_status_code: int | None = None
    unreadable_calls: set[int] = field(default_factory=set)
    fail_from_call: int | None = None
    speech_mutator: Callable[[str, int], str] | None = None
    duration_offset_s: float = 0.0
    duration_offset_times: int | None = None
    duration_offset_calls: set[int] = field(default_factory=set)
    mute_calls: set[int] = field(default_factory=set)
    stray_speech: str = ""
    stray_speech_times: int | None = None
    usd_per_s: float = 0.0
    calls: int = 0
    prompts: list[str] = field(default_factory=list)
    _stray_written: int = field(default=0, init=False, repr=False)

    def generate(self, request: VideoRequest, *, dest: Path) -> VideoResult:
        self.calls += 1
        self.prompts.append(request.prompt)
        if self.calls <= self.fail_times or (
            self.fail_from_call is not None and self.calls >= self.fail_from_call
        ):
            msg = f"simulated transient failure #{self.calls}"
            raise RetryableProviderError(msg, provider="fake", model=request.model, status_code=503)
        if self.calls <= self.reject_times or (
            self.reject_if_contains is not None and self.reject_if_contains in request.prompt
        ):
            msg = f"simulated content policy rejection #{self.calls}"
            raise PromptRejectedError(msg, provider="fake", model=request.model, status_code=400)
        if self.calls <= self.expire_times:
            msg = f"simulated expired job #{self.calls}"
            raise FatalProviderError(msg, provider="fake", model=request.model)
        if self.calls <= self.billed_times:
            msg = f"simulated uncollectable job #{self.calls}"
            raise BilledJobError(
                msg,
                provider="fake",
                model=request.model,
                job_id=f"fake-video-{self.calls}",
                spent_usd=self.billed_spent_usd,
                status_code=self.billed_status_code,
            )
        width, height = VERTICAL_SIZES[request.resolution.value]
        if self.calls in self.unreadable_calls:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"<html>not a video</html>")
            return VideoResult(
                path=dest,
                model=f"fake/{request.model}",
                request_id=f"fake-video-{self.calls}",
                cost_usd=round(self.usd_per_s * request.duration_s, 6),
            )
        spoken = _speech_from_prompt(request.prompt)
        if spoken is not None and self.speech_mutator is not None:
            text, start_s, end_s = spoken
            spoken = (self.speech_mutator(text, self.calls), start_s, end_s)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.duration_offset_calls:
            offset = self.duration_offset_s if self.calls in self.duration_offset_calls else 0.0
        elif self.duration_offset_times is None or self.calls <= self.duration_offset_times:
            offset = self.duration_offset_s
        else:
            offset = 0.0
        duration_s = request.duration_s + offset
        self.ffmpeg.synth_clip(
            dest,
            duration_s=duration_s,
            width=width,
            height=height,
            with_audio=request.generate_audio and self.calls not in self.mute_calls,
        )
        if spoken is not None:
            write_sidecar(dest, spoken[0], start_s=spoken[1], end_s=spoken[2])
        elif self.stray_speech and (
            self.stray_speech_times is None or self._stray_written < self.stray_speech_times
        ):
            self._stray_written += 1
            write_sidecar(dest, self.stray_speech, start_s=0.0, end_s=duration_s)
        else:
            write_sidecar(dest, "")
        return VideoResult(
            path=dest,
            model=f"fake/{request.model}",
            seed=request.seed if request.seed is not None else 1000 + self.calls,
            request_id=f"fake-video-{self.calls}",
            cost_usd=round(self.usd_per_s * request.duration_s, 6),
            cost_is_estimate=True,
        )


def _speech_from_prompt(prompt: str) -> tuple[str, float | None, float | None] | None:
    """The quoted line of a dialogue prompt and the `a–bs:` span it is spoken in (if given)."""
    says = _SAYS_IN_PROMPT.search(prompt)
    if says is None:
        return None
    spans = list(_SPAN_IN_PROMPT.finditer(prompt, 0, says.start()))
    if not spans:
        return says.group(1), None, None
    last = spans[-1]
    return says.group(1), float(last.group(1)), float(last.group(2))


# -- speech sidecars --------------------------------------------------------------------------


@dataclass(frozen=True)
class Speech:
    """Words "spoken" in a media file between `start_s` and `end_s` (None = the whole file)."""

    text: str
    start_s: float | None = None
    end_s: float | None = None

    def words_within(self, window: tuple[float, float | None] | None) -> list[str]:
        """The words whose evenly spaced time slots have their midpoint inside the window."""
        words = self.text.split()
        if window is None or self.start_s is None or self.end_s is None or not words:
            return words
        lo, length = window
        hi = float("inf") if length is None else lo + length
        step = (self.end_s - self.start_s) / len(words)
        return [w for i, w in enumerate(words) if lo <= self.start_s + (i + 0.5) * step <= hi]


def write_sidecar(
    media: Path, text: str, *, start_s: float | None = None, end_s: float | None = None
) -> Path:
    sidecar = media.with_name(media.name + SPEECH_SIDECAR_SUFFIX)
    payload = {"text": text, "start_s": start_s, "end_s": end_s}
    sidecar.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return sidecar


def read_speech(media: Path) -> Speech | None:
    """The sidecar of `media`; a plain-text sidecar means the text spans the whole file."""
    sidecar = media.with_name(media.name + SPEECH_SIDECAR_SUFFIX)
    if not sidecar.is_file():
        return None
    raw = sidecar.read_text(encoding="utf-8")
    if not raw.lstrip().startswith("{"):
        return Speech(text=raw)  # plain-text sidecar: the text spans the whole file
    try:
        data = json.loads(raw)
    except ValueError as exc:
        log.warning("fake.sidecar_unreadable", path=str(sidecar), error=str(exc))
        return Speech(text=raw)
    return Speech(
        text=str(data.get("text") or ""), start_s=data.get("start_s"), end_s=data.get("end_s")
    )


def read_sidecar(media: Path) -> str | None:
    speech = read_speech(media)
    return speech.text if speech is not None else None


def _assembled_speech(media: Path) -> list[Speech]:
    """What the assembled final "contains": the TTS line mixed in by `assemble`, placed where the
    run's `script.json` puts it (start of the dialogue shot + `dialogue.start_s`)."""
    for root in (media.parent, media.parent.parent):
        voices = sorted((root / "audio").glob(f"*{SPEECH_SIDECAR_SUFFIX}"))
        if not voices:
            continue
        script_path = root / "script.json"
        placement = _line_offset(script_path) if script_path.is_file() else None
        found: list[Speech] = []
        for sidecar in voices:
            speech = read_speech(
                sidecar.with_name(sidecar.name.removesuffix(SPEECH_SIDECAR_SUFFIX))
            )
            if speech is None:
                continue
            if placement is None or speech.end_s is None:
                found.append(Speech(text=speech.text))
            else:
                found.append(
                    Speech(
                        text=speech.text,
                        start_s=placement + (speech.start_s or 0.0),
                        end_s=placement + speech.end_s,
                    )
                )
        return found
    return []


def _line_offset(script_path: Path) -> float | None:
    try:
        script = json.loads(script_path.read_text(encoding="utf-8"))
        d = script["dialogue"]
        before = sum(s["duration_s"] for s in script["shots"] if s["id"] < d["shot_id"])
        return float(before) + float(d["start_s"])
    except (ValueError, KeyError, TypeError) as exc:
        log.warning("fake.script_unreadable", path=str(script_path), error=str(exc))
        return None  # the voice is then heard in any window


def _segment(audio_path: Path) -> tuple[float, float | None] | None:
    """The `(start_s, duration_s)` segment `verify_speech` cut the WAV from, if recorded."""
    meta = audio_path.with_name(audio_path.name + ".json")
    if not meta.is_file():
        return None
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
        duration = data.get("duration_s")
        return float(data.get("start_s") or 0.0), None if duration is None else float(duration)
    except (ValueError, TypeError, AttributeError) as exc:
        log.warning("fake.segment_unreadable", path=str(meta), error=str(exc))
        return None


# -- TTS / STT / lipsync ----------------------------------------------------------------------


@dataclass
class FakeTTS:
    """A tone as long as the line takes at `chars_per_s`; `reject_voices` raise a 422 like a
    vendor that does not know the voice."""

    ffmpeg: FFmpeg = field(default_factory=FFmpeg)
    chars_per_s: float = CHARS_PER_SECOND
    reject_voices: set[str] = field(default_factory=set)
    calls: int = 0
    voices: list[str | None] = field(default_factory=list)

    def synthesize(self, request: TTSRequest, *, dest: Path) -> TTSResult:
        self.calls += 1
        self.voices.append(request.voice)
        if request.voice in self.reject_voices:
            msg = f"simulated unknown voice {request.voice!r}"
            raise FatalProviderError(msg, provider="fake", model=request.model, status_code=422)
        dest.parent.mkdir(parents=True, exist_ok=True)
        duration = max(0.5, len(request.text) / self.chars_per_s)
        self.ffmpeg.synth_audio(dest, duration_s=duration)
        write_sidecar(dest, request.text, start_s=0.0, end_s=duration)
        return TTSResult(path=dest, model=f"fake/{request.model}")


@dataclass
class FakeSTT:
    """Returns the speech recorded by `FakeVideo`/`FakeTTS` inside the transcribed segment, or a
    queued transcript. `fail_times` transient outages first; `fatal_status` makes every call a
    `FatalProviderError` with that HTTP status (a misconfigured model id or key)."""

    queue: list[str] = field(default_factory=list)
    fail_times: int = 0
    fatal_status: int | None = None
    calls: int = 0

    def transcribe(self, request: STTRequest) -> STTResult:
        self.calls += 1
        model = f"fake/{request.model}"
        if self.fatal_status is not None:
            msg = f"simulated STT misconfiguration (HTTP {self.fatal_status})"
            raise FatalProviderError(
                msg, provider="fake", model=model, status_code=self.fatal_status
            )
        if self.calls <= self.fail_times:
            raise RetryableProviderError("simulated STT outage", provider="fake", model=model)
        if self.queue:
            return STTResult(text=self.queue.pop(0), model=model, language=request.language)
        window = _segment(request.audio_path)
        speech: list[Speech] = []
        for candidate in (request.source_path, request.audio_path):
            if candidate is not None and (found := read_speech(candidate)) is not None:
                speech = [found]
                break
        else:
            if request.source_path is not None:
                speech = _assembled_speech(request.source_path)
        words = [w for s in speech for w in s.words_within(window)]
        return STTResult(text=" ".join(words), model=model, language=request.language)


@dataclass
class FakeLipsync:
    """Muxes the picture with the audio it was given (as a real lipsync result carries it) and
    refuses audio whose length differs from the clip, like a mismatched input would."""

    ffmpeg: FFmpeg = field(default_factory=FFmpeg)
    max_length_mismatch_s: float = 0.1
    calls: int = 0
    requests: list[LipsyncRequest] = field(default_factory=list)

    def sync(self, request: LipsyncRequest, *, dest: Path) -> LipsyncResult:
        self.calls += 1
        self.requests.append(request)
        video = self.ffmpeg.probe(request.video_path)
        audio = self.ffmpeg.probe(request.audio_path)
        if abs(video.duration_s - audio.duration_s) > self.max_length_mismatch_s:
            msg = (
                f"lipsync audio is {audio.duration_s:.2f}s but the clip is {video.duration_s:.2f}s"
            )
            raise FatalProviderError(msg, provider="fake", model=request.model, status_code=422)
        self.ffmpeg.normalize_clip(
            request.video_path,
            dest,
            width=video.width,
            height=video.height,
            audio_src=request.audio_path,
        )
        return LipsyncResult(path=dest, model=f"fake/{request.model}")


# -- Judge ------------------------------------------------------------------------------------


@dataclass
class FakeJudge:
    """Scripted answers keyed by a substring of the question; defaults are permissive."""

    enabled: bool = True
    model: str = "fake"
    available: bool = True  # False simulates Jev being unreachable mid-run
    bool_answers: dict[str, float] = field(default_factory=dict)
    choice_answers: dict[str, str] = field(default_factory=dict)
    score_answers: dict[str, int | str] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    def ask_bool(self, state: str, question: str) -> JudgeAnswer:
        self.calls.append(question)
        if not self.available:
            return JudgeAnswer(question=question, answer="unknown", available=False)
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
        if not self.available:
            return JudgeAnswer(question=question, answer="unknown", available=False)
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
        """A scripted `str` answer (e.g. "unknown") simulates a score the SDK could not read."""
        self.calls.append(question)
        if not self.available:
            return JudgeAnswer(question=question, answer="unknown", available=False)
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
