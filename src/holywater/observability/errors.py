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
    """Base for failures of external APIs (fal, OpenRouter, TypeSafe).

    `spent_usd` is money the vendor already charged (or will charge) for the failed call, so the
    flow can record it even though no artifact came back; `None` means nothing is known to be spent.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str | None = None,
        request_id: str | None = None,
        status_code: int | None = None,
        spent_usd: float | None = None,
        spent_is_estimate: bool = True,
        **kwargs: Any,
    ) -> None:
        context = dict(kwargs.pop("context", None) or {})
        context.update(
            provider=provider, model=model, request_id=request_id, status_code=status_code
        )
        if spent_usd is not None:
            context.update(spent_usd=spent_usd, spent_is_estimate=spent_is_estimate)
        super().__init__(message, context=context, **kwargs)
        self.provider = provider
        self.model = model
        self.request_id = request_id
        self.status_code = status_code
        self.spent_usd = spent_usd
        self.spent_is_estimate = spent_is_estimate


class RetryableProviderError(ProviderError):
    """429, 5xx, timeouts, network errors: worth retrying with backoff."""

    retryable: ClassVar[bool] = True


class FatalProviderError(ProviderError):
    """401/402/403/404, malformed responses, expired or cancelled jobs: retrying the same request
    will not help, and neither will rewriting the prompt."""


class PromptRejectedError(FatalProviderError):
    """The vendor refused the prompt itself: content policy / moderation wording, HTTP 400 on a
    video submit, or a video job that failed without an infrastructure cause. The only provider
    error the flow may answer by softening the prompt."""


class StructuredOutputError(FatalProviderError, ValueError):
    """The LLM answered (and was billed) but the output is unusable: truncated
    (`finish_reason=length`), empty, not JSON, or not valid against the schema.

    Also a `ValueError`, so the `LLMProvider` contract ("ValueError when the output does not
    validate") holds and callers can retry with feedback built from `raw_text`/`finish_reason`.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        raw_text: str = "",
        finish_reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = dict(kwargs.pop("context", None) or {})
        context.update(finish_reason=finish_reason, raw_text=raw_text[:500])
        super().__init__(message, provider=provider, context=context, **kwargs)
        self.raw_text = raw_text
        self.finish_reason = finish_reason


class BilledJobError(ProviderError):
    """The vendor accepted a paid job, but its result could not be collected (poll deadline
    exceeded, download kept failing, result unusable). Not retryable: a retry would pay for a new
    job, so that decision belongs to the flow. Not a `FatalProviderError`, so it is never answered
    by softening the prompt. `job_id` is the vendor job; `spent_usd` is what it costs."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        job_id: str,
        spent_usd: float,
        spent_is_estimate: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            request_id=job_id,
            spent_usd=spent_usd,
            spent_is_estimate=spent_is_estimate,
            **kwargs,
        )
        self.job_id = job_id


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
    """ffmpeg/ffprobe failure or a corrupt media file. Retried by the enclosing Prefect task per
    `retries.provider` (`attempts - 1` retries with `backoff_s` delays)."""

    retryable: ClassVar[bool] = True


def is_retryable(exc: BaseException) -> bool:
    """Used as Prefect `retry_condition_fn`: only typed retryable errors are retried."""
    return isinstance(exc, PipelineError) and exc.retryable
