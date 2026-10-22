"""OpenRouter (OpenAI-compatible) structured output via `response_format: json_schema` (ADR-004)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

import openai
from pydantic import BaseModel, ValidationError

from holywater.observability.errors import FatalProviderError, RetryableProviderError
from holywater.observability.logging import get_logger
from holywater.providers.base import LLMResult

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)
PROVIDER = "openrouter"
_HEADERS = {"HTTP-Referer": "https://github.com/holywater-poc", "X-Title": "holywater-poc"}


@dataclass
class OpenRouterLLM:
    api_key: str
    model: str
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_s: int = 60
    max_tokens: int = 4000
    _client: openai.OpenAI = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # SDK-level retries are disabled: the pipeline owns retry policy and logging.
        self._client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_s,
            max_retries=0,
            default_headers=_HEADERS,
        )

    def complete_structured(
        self, *, system: str, user: str, schema: type[T], temperature: float | None = None
    ) -> LLMResult[T]:
        json_schema = schema.model_json_schema()
        # Only send parameters we really need: with `require_parameters` every extra one narrows
        # the set of eligible endpoints (Claude 5.x, for example, has no `temperature`).
        optional: dict[str, Any] = {}
        if temperature is not None:
            optional["temperature"] = temperature
        started = time.monotonic()
        log.debug("llm.request", model=self.model, schema=schema.__name__, chars=len(user))
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=self.max_tokens,
                **optional,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "strict": True,
                        "schema": json_schema,
                    },
                },
                # Only route to endpoints that honour json_schema; never silently degrade.
                extra_body={"provider": {"require_parameters": True}},
            )
        except openai.RateLimitError as exc:
            raise RetryableProviderError(
                f"openrouter rate limited: {exc}",
                provider=PROVIDER,
                model=self.model,
                status_code=exc.status_code,
            ) from exc
        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            raise RetryableProviderError(
                f"openrouter connection error: {exc}", provider=PROVIDER, model=self.model
            ) from exc
        except openai.APIStatusError as exc:
            cls = RetryableProviderError if exc.status_code >= 500 else FatalProviderError  # noqa: PLR2004
            raise cls(
                f"openrouter HTTP {exc.status_code}: {exc.message}",
                provider=PROVIDER,
                model=self.model,
                status_code=exc.status_code,
                request_id=getattr(exc, "request_id", None),
            ) from exc

        choice = completion.choices[0] if completion.choices else None
        content = choice.message.content if choice and choice.message else None
        if not content:
            finish = getattr(choice, "finish_reason", None)
            msg = f"openrouter returned no content (finish_reason={finish})"
            raise FatalProviderError(
                msg, provider=PROVIDER, model=self.model, request_id=completion.id
            )
        usage = completion.usage
        extra: dict[str, Any] = (
            dict(usage.model_extra or {}) if usage is not None and usage.model_extra else {}
        )
        cost = float(extra.get("cost") or 0.0)
        log.info(
            "llm.response",
            model=completion.model or self.model,
            request_id=completion.id,
            latency_s=round(time.monotonic() - started, 2),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            cost_usd=cost,
        )
        try:
            data = schema.model_validate_json(content)
        except ValidationError:
            # Some providers wrap the object; try one level of unwrapping before giving up.
            data = schema.model_validate(_unwrap(json.loads(content)))
        return LLMResult(
            data=data,
            model=completion.model or self.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cost_usd=cost,
            raw_text=content,
        )


def _unwrap(payload: Any) -> Any:
    if isinstance(payload, dict) and len(payload) == 1:
        (inner,) = payload.values()
        if isinstance(inner, dict):
            return inner
    return payload
