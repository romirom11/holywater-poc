"""String enums shared across layers; stored in SQLite as `.value`."""

from __future__ import annotations

from enum import StrEnum


class StageName(StrEnum):
    WRITE_SCRIPT = "write_script"
    PREFLIGHT_SCRIPT = "preflight_script"
    GENERATE_AUDIO = "generate_audio"
    GENERATE_VIDEO = "generate_video"
    VERIFY_SPEECH = "verify_speech"
    ASSEMBLE = "assemble"
    REPORT = "report"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class RunStatus(StrEnum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class AudioStrategy(StrEnum):
    NATIVE = "native"
    TTS_OVERLAY = "tts_overlay"


class FailureKind(StrEnum):
    """Why a native-speech attempt failed the WER gate (classified by the judge)."""

    PARAPHRASED = "paraphrased"
    TRUNCATED = "truncated"
    EXTRA_WORDS = "extra_words"
    WRONG_LANGUAGE = "wrong_language"
    STT_NOISE = "stt_noise"
    SILENCE = "silence"
    UNKNOWN = "unknown"


class Resolution(StrEnum):
    P480 = "480p"
    P720 = "720p"


class VoiceGender(StrEnum):
    """Gender of the speaking voice; must match the speaker description (picks the TTS voice)."""

    FEMALE = "female"
    MALE = "male"


class AttemptStatus(StrEnum):
    """Lifecycle of one billed video attempt (`shot_attempts.status`)."""

    GENERATED = "generated"  # clip downloaded, not checked yet
    FAILED = "failed"  # failed a gate (duration / integrity / WER)
    PASSED = "passed"  # passed the WER gate
    UNVERIFIED = "unverified"  # STT unavailable: verbatim-ness not evidenced
    ACCEPTED = "accepted"  # the clip that goes into the final video
    ABANDONED = "abandoned"  # billed job whose result was never collected
    SUPERSEDED = "superseded"  # was accepted, then replaced by a longer clip of the same shot
