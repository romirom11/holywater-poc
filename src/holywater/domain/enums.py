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
