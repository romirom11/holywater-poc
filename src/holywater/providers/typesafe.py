"""Jev (TypeSafe AI) typed judge (ADR-007). Never raises into the pipeline: a blank key, a missing
SDK or a client that cannot be built gives `NoopJudge` with a WARNING; a failed or malformed call
returns an `unavailable` answer with a WARNING, so generation continues without pre-flight.

SDK answer semantics (typesafe-sdk 0.7): `NoulAnswer.noul` is P(yes); `ChoiceAnswer.choice` and
`.probabilities` are keyed by option name; `ScoreAnswer.score` is the probability-weighted mean of
the 0-based rubric positions and `.probabilities` is keyed by those 0-based ints. Our levels are
1-based ("1".."N"), so scores and probability keys are shifted by one.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeGuard

from pydantic import SecretStr

from holywater.config.schema import JudgeConfig
from holywater.observability.logging import get_logger
from holywater.providers.base import JudgeAnswer, JudgeProvider, NoopJudge

log = get_logger(__name__)

_SCORE_LABELS = ("very low", "low", "medium", "high", "very high")
_YES_THRESHOLD = 0.5
_CHARS_PER_TOKEN = 4  # token estimate when the response carries no usage


def build_judge(
    config: JudgeConfig, api_key: SecretStr | None, *, timeout_s: int = 10
) -> JudgeProvider:
    if not config.enabled:
        log.info("judge.disabled", reason="config")
        return NoopJudge()
    key = api_key.get_secret_value().strip() if api_key is not None else ""
    if not key:
        log.warning("judge.disabled", reason="TYPESAFE_API_KEY not set; running without Jev")
        return NoopJudge()
    try:
        import typesafe_sdk  # noqa: F401 - optional dependency (`uv sync --extra judge`)
    except ImportError:
        log.warning("judge.disabled", reason="typesafe-sdk not installed (uv sync --extra judge)")
        return NoopJudge()
    try:
        return TypeSafeJudge(
            api_key=key,
            model=config.model,
            timeout_s=timeout_s,
            usd_per_m_input_tokens=config.usd_per_m_input_tokens,
        )
    except Exception as exc:  # the SDK validates key, timeout and compression at construction
        log.warning(
            "judge.disabled",
            reason="TypeSafe client could not be created; running without Jev",
            error=type(exc).__name__,
            message=str(exc)[:300],
        )
        return NoopJudge()


@dataclass(frozen=True)
class _Reply:
    answer: Any
    model: str
    latency_ms: int
    cost_usd: float


@dataclass
class TypeSafeJudge:
    api_key: str = field(repr=False)
    model: str = "jev-latest"
    timeout_s: int = 10
    usd_per_m_input_tokens: float = 0.042
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

        reply = self._ask(state, question, Noul(instructions=question))
        if reply is None:
            return _unavailable(question)
        p_yes = getattr(reply.answer, "noul", None)
        if not _is_number(p_yes):
            return _unexpected(question, reply, "noul")
        p = float(p_yes)
        answer = "yes" if p >= _YES_THRESHOLD else "no"
        return _answer(question, reply, answer, {"yes": p, "no": 1 - p})

    def ask_choice(self, state: str, question: str, options: Sequence[str]) -> JudgeAnswer:
        from typesafe_sdk import Choice

        q = Choice(instructions=question, criteria=dict.fromkeys(options))
        reply = self._ask(state, question, q)
        if reply is None:
            return _unavailable(question)
        choice = getattr(reply.answer, "choice", None)
        if not isinstance(choice, str) or choice not in options:
            return _unexpected(question, reply, "choice")
        probs = getattr(reply.answer, "probabilities", None)
        by_option = (
            {str(k): float(v) for k, v in probs.items()} if isinstance(probs, Mapping) else {}
        )
        return _answer(question, reply, choice, by_option)

    def ask_score(self, state: str, question: str, *, levels: int) -> JudgeAnswer:
        from typesafe_sdk import Score

        labels = [
            f"{i}: {_SCORE_LABELS[min(i - 1, len(_SCORE_LABELS) - 1)]}"
            for i in range(1, levels + 1)
        ]
        reply = self._ask(state, question, Score(instructions=question, criteria=labels))
        if reply is None:
            return _unavailable(question)
        score = getattr(reply.answer, "score", None)
        if not _is_number(score):
            return _unexpected(question, reply, "score")
        # 0-based weighted mean -> nearest 1-based level.
        level = min(max(round(float(score)) + 1, 1), levels)
        probs = getattr(reply.answer, "probabilities", None)
        by_level = (
            {str(int(k) + 1): float(v) for k, v in probs.items()}
            if isinstance(probs, Mapping)
            else {}
        )
        return _answer(question, reply, str(level), by_level)

    # -- internals -----------------------------------------------------------------------------

    def _ask(self, state: str, question: str, q: Any) -> _Reply | None:
        from typesafe_sdk import TypeSafeError

        started = time.monotonic()
        try:
            response = self._client.system_one(state={"state": state}, questions={"q": q})
        except TypeSafeError as exc:
            log.warning("judge.unavailable", error=type(exc).__name__, message=str(exc)[:300])
            return None
        answers = getattr(response, "answers", None)
        answer = answers.get("q") if isinstance(answers, Mapping) else None
        if answer is None:
            log.warning("judge.unexpected_response", question=question[:80])
            return None
        return _Reply(
            answer=answer,
            model=str(getattr(response, "model", "") or self.model),
            latency_ms=round((time.monotonic() - started) * 1000),
            cost_usd=self._estimate_cost(response, state, q),
        )

    def _estimate_cost(self, response: Any, state: str, q: Any) -> float:
        """Input tokens x configured price; the reported token count when the API gives one,
        else chars/4. Always an estimate (the price is ours, not the API's)."""
        tokens = getattr(getattr(response, "usage", None), "input_tokens", None)
        if not isinstance(tokens, int):
            chars = len(state) + len(q.model_dump_json())
            tokens = math.ceil(chars / _CHARS_PER_TOKEN)
        return round(tokens * self.usd_per_m_input_tokens / 1_000_000, 8)


def _answer(
    question: str, reply: _Reply, answer: str, probabilities: dict[str, float]
) -> JudgeAnswer:
    return JudgeAnswer(
        question=question,
        answer=answer,
        probabilities=probabilities,
        model_version=reply.model,
        latency_ms=reply.latency_ms,
        cost_usd=reply.cost_usd,
        cost_is_estimate=True,
    )


def _unexpected(question: str, reply: _Reply, field_name: str) -> JudgeAnswer:
    """The call succeeded (and was billed) but the answer is not the expected shape."""
    log.warning(
        "judge.unexpected_answer",
        question=question[:80],
        field=field_name,
        answer_type=type(reply.answer).__name__,
    )
    return JudgeAnswer(
        question=question,
        answer="unknown",
        available=False,
        model_version=reply.model,
        latency_ms=reply.latency_ms,
        cost_usd=reply.cost_usd,
    )


def _unavailable(question: str) -> JudgeAnswer:
    return JudgeAnswer(question=question, answer="unknown", available=False)


def _is_number(value: object) -> TypeGuard[float]:
    return isinstance(value, int | float) and not isinstance(value, bool)
