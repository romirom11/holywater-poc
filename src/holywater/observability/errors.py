"""Error taxonomy for the pipeline (see docs/ARCHITECTURE.md §5).

Every error carries enough context to be logged and persisted without the caller re-deriving it.
`retryable` is a class-level flag read by the Prefect `retry_condition_fn`.
"""

from __future__ import annotations

from typing import Any, ClassVar


class PipelineError(Exception):
    """Base class for all pipeline errors."""

    retryable: ClassVar[bool] = False

    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        run_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.run_id = run_id
        self.context: dict[str, Any] = dict(context or {})

    def to_dict(self) -> dict[str, Any]:
        """Serializable view used for logs and the `stages.error_json` column."""
        return {
            "type": type(self).__name__,
            "message": self.message,
            "stage": self.stage,
            "run_id": self.run_id,
            "retryable": self.retryable,
            "context": self.context,
        }

    def __str__(self) -> str:
        where = ", ".join(
            f"{k}={v}" for k, v in (("stage", self.stage), ("run_id", self.run_id)) if v
        )
        return f"{self.message} [{where}]" if where else self.message


class ConfigError(PipelineError):
    """Invalid configuration or missing secrets. Raised before any paid call."""


class ScriptError(PipelineError):
    """The LLM could not produce a valid script after all attempts."""


class ProviderError(PipelineError):
    """Base for failures of external APIs (fal, OpenRouter, TypeSafe)."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str | None = None,
        request_id: str | None = None,
        status_code: int | None = None,
        **kwargs: Any,
    ) -> None:
        context = dict(kwargs.pop("context", None) or {})
        context.update(
            provider=provider, model=model, request_id=request_id, status_code=status_code
        )
        super().__init__(message, context=context, **kwargs)
        self.provider = provider
        self.model = model
        self.request_id = request_id
        self.status_code = status_code


class RetryableProviderError(ProviderError):
    """429, 5xx, timeouts, network errors: worth retrying with backoff."""

    retryable: ClassVar[bool] = True


class FatalProviderError(ProviderError):
    """401, 400, content policy: retrying the same request will not help."""


class QualityGateError(PipelineError):
    """A generated artifact failed a quality gate. Handled by the flow, not by the task."""

    def __init__(
        self,
        message: str,
        *,
        gate: str,
        measured: float | str | None,
        threshold: float | str | None,
        **kwargs: Any,
    ) -> None:
        context = dict(kwargs.pop("context", None) or {})
        context.update(gate=gate, measured=measured, threshold=threshold)
        super().__init__(message, context=context, **kwargs)
        self.gate = gate
        self.measured = measured
        self.threshold = threshold


class MediaError(PipelineError):
    """ffmpeg/ffprobe failure or a corrupt media file. Retried once."""

    retryable: ClassVar[bool] = True


def is_retryable(exc: BaseException) -> bool:
    """Used as Prefect `retry_condition_fn`: only typed retryable errors are retried."""
    return isinstance(exc, PipelineError) and exc.retryable
