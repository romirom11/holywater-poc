"""Jev (TypeSafe AI) typed judge (ADR-007). Never raises into the pipeline: any failure
returns an `unavailable` answer and a WARNING, so generation continues without pre-flight."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import SecretStr

from holywater.config.schema import JudgeConfig
from holywater.observability.logging import get_logger
from holywater.providers.base import JudgeAnswer, JudgeProvider, NoopJudge

log = get_logger(__name__)

_SCORE_LABELS = ("very low", "low", "medium", "high", "very high")
_YES_THRESHOLD = 0.5


def build_judge(
    config: JudgeConfig, api_key: SecretStr | None, *, timeout_s: int = 10
) -> JudgeProvider:
    if not config.enabled:
        log.info("judge.disabled", reason="config")
        return NoopJudge()
    if api_key is None:
        log.warning("judge.disabled", reason="TYPESAFE_API_KEY not set; running without Jev")
        return NoopJudge()
    try:
        import typesafe_sdk  # noqa: F401 - optional dependency (`uv sync --extra judge`)
    except ImportError:
        log.warning("judge.disabled", reason="typesafe-sdk not installed (uv sync --extra judge)")
        return NoopJudge()
    return TypeSafeJudge(
        api_key=api_key.get_secret_value(), model=config.model, timeout_s=timeout_s
    )


@dataclass
class TypeSafeJudge:
    api_key: str
    model: str = "jev-latest"
    timeout_s: int = 10
    _client: Any = field(init=False, repr=False)

    enabled = True

    def __post_init__(self) -> None:
        from typesafe_sdk import TypeSafeClient

        self._client = TypeSafeClient(
            api_key=self.api_key, model=self.model, timeout=self.timeout_s
        )

    # -- public API ----------------------------------------------------------------------------

    def ask_bool(self, state: str, question: str) -> JudgeAnswer:
        from typesafe_sdk import Noul

        answers = self._ask(state, question, {"q": Noul(instructions=question)})
        if answers is None:
            return _unavailable(question)
        p_yes = float(getattr(answers.get("q"), "noul", 0.0) or 0.0)
        return self._answer(
            question,
            "yes" if p_yes >= _YES_THRESHOLD else "no",
            {"yes": p_yes, "no": 1 - p_yes},
        )

    def ask_choice(self, state: str, question: str, options: Sequence[str]) -> JudgeAnswer:
        from typesafe_sdk import Choice

        q = Choice(instructions=question, criteria=dict.fromkeys(options))
        answers = self._ask(state, question, {"q": q})
        if answers is None:
            return _unavailable(question)
        ans = answers.get("q")
        choice = str(getattr(ans, "choice", "") or "")
        probs = _as_probs(getattr(ans, "probabilities", None), options)
        return self._answer(question, choice if choice in options else "unknown", probs)

    def ask_score(self, state: str, question: str, *, levels: int) -> JudgeAnswer:
        from typesafe_sdk import Score

        labels = [
            f"{i}: {_SCORE_LABELS[min(i - 1, len(_SCORE_LABELS) - 1)]}"
            for i in range(1, levels + 1)
        ]
        answers = self._ask(state, question, {"q": Score(instructions=question, criteria=labels)})
        if answers is None:
            return _unavailable(question)
        ans = answers.get("q")
        raw = getattr(ans, "score", None)
        probs = _as_probs(getattr(ans, "probabilities", None), labels)
        # Map label probabilities back to "1".."N" keys and the score to a 1-based integer.
        by_level = {str(i + 1): probs.get(label, 0.0) for i, label in enumerate(labels)}
        level = _score_to_level(raw, labels)
        return self._answer(question, str(level) if level else "unknown", by_level)

    # -- internals -----------------------------------------------------------------------------

    def _ask(self, state: str, question: str, questions: dict[str, Any]) -> dict[str, Any] | None:
        from typesafe_sdk import TypeSafeError

        started = time.monotonic()
        try:
            response = self._client.system_one(state={"state": state}, questions=questions)
        except TypeSafeError as exc:
            log.warning("judge.unavailable", error=type(exc).__name__, message=str(exc)[:300])
            return None
        self._last_latency_ms = round((time.monotonic() - started) * 1000)
        self._last_model = str(getattr(response, "model", "") or self.model)
        answers = getattr(response, "answers", None)
        if not isinstance(answers, dict):
            log.warning("judge.unexpected_response", question=question[:80])
            return None
        return answers

    def _answer(self, question: str, answer: str, probabilities: dict[str, float]) -> JudgeAnswer:
        return JudgeAnswer(
            question=question,
            answer=answer,
            probabilities=probabilities,
            model_version=getattr(self, "_last_model", self.model),
            latency_ms=getattr(self, "_last_latency_ms", None),
        )


def _unavailable(question: str) -> JudgeAnswer:
    return JudgeAnswer(question=question, answer="unknown", available=False)


def _as_probs(raw: Any, keys: Sequence[str]) -> dict[str, float]:
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items() if isinstance(v, int | float)}
    if isinstance(raw, list | tuple) and len(raw) == len(keys):
        return {k: float(v) for k, v in zip(keys, raw, strict=True)}
    return {}


def _score_to_level(raw: Any, labels: Sequence[str]) -> int | None:
    if isinstance(raw, int | float):
        value = float(raw)
        # Either already a 1-based level, or a 0..1 position on the scale.
        if value <= 1.0 and len(labels) > 1:
            return 1 + round(value * (len(labels) - 1))
        return max(1, min(len(labels), round(value)))
    if isinstance(raw, str):
        if raw in labels:
            return labels.index(raw) + 1
        head = raw.split(":", 1)[0].strip()
        return int(head) if head.isdigit() else None
    return None
