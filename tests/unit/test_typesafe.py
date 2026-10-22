"""Jev judge against the real typesafe-sdk: requests go through `TypeSafeClient` with an offline
mock transport, so answers are decoded by the SDK's own models (ScoreAnswer has 0-based,
int-keyed probabilities; NoulAnswer.noul is P(yes); ChoiceAnswer is keyed by option name)."""

import json
import math
import sys
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import SecretStr
from structlog.testing import capture_logs

from holywater.config.schema import JudgeConfig
from holywater.providers.base import JudgeProvider, NoopJudge
from holywater.providers.typesafe import TypeSafeJudge, build_judge

pytest.importorskip("typesafe_sdk")

import httpx2
from typesafe_sdk import RetryPolicy, TypeSafeClient

PRICE = 0.042 / 1_000_000
Handler = Callable[[httpx2.Request], httpx2.Response]


def _judge(handler: Handler, **kwargs: Any) -> TypeSafeJudge:
    judge = TypeSafeJudge(api_key="ts-key", **kwargs)
    judge._client = TypeSafeClient(
        api_key="ts-key",
        model=judge.model,
        timeout=judge.timeout_s,
        retry=RetryPolicy(max_retries=0),
        transport=httpx2.MockTransport(handler),
    )
    return judge


def _api(
    answer: dict[str, Any], *, input_tokens: int | None = 120, seen: list[Any] | None = None
) -> Handler:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "model": "jev-1.14.0",
                "usage": {"input_tokens": input_tokens, "output_tokens": 3},
                "answers": {"q": answer},
            },
        )

    return handler


def _score(score: float, probabilities: dict[int, float]) -> dict[str, Any]:
    return {
        "type": "score",
        "score": score,
        "confidence": 0.9,
        "legend": {str(k): f"level {k}" for k in probabilities},
        "probabilities": {str(k): v for k, v in probabilities.items()},
    }


FIVE = {0: 0.0, 1: 0.1, 2: 0.8, 3: 0.1, 4: 0.0}


@pytest.mark.parametrize(
    ("score", "level"),
    [(0.0, "1"), (0.4, "1"), (0.9, "2"), (1.1, "2"), (2.0, "3"), (3.6, "5"), (4.0, "5")],
)
def test_score_is_zero_based_and_mapped_to_one_based_levels(score: float, level: str) -> None:
    seen: list[Any] = []
    judge = _judge(_api(_score(score, FIVE), seen=seen))
    out = judge.ask_score("state", "filmability?", levels=5)
    assert out.answer == level and out.available
    assert out.probabilities == {"1": 0.0, "2": 0.1, "3": 0.8, "4": 0.1, "5": 0.0}
    assert seen[0]["questions"]["q"] == {
        "type": "score",
        "instructions": "filmability?",
        "criteria": ["1: very low", "2: low", "3: medium", "4: high", "5: very high"],
    }


def test_score_mapping_is_monotonic() -> None:
    levels = [
        int(_judge(_api(_score(s / 10, FIVE))).ask_score("s", "q", levels=5).answer)
        for s in range(41)
    ]
    assert levels == sorted(levels) and levels[0] == 1 and levels[-1] == 5


def test_bool_answer_and_metadata() -> None:
    judge = _judge(_api({"type": "noul", "noul": 0.98}))
    out = judge.ask_bool("state", "Is it unsafe?")
    assert out.answer == "yes"
    assert out.probabilities == pytest.approx({"yes": 0.98, "no": 0.02})
    assert out.model_version == "jev-1.14.0" and isinstance(out.latency_ms, int)
    assert out.cost_usd == pytest.approx(120 * PRICE) and out.cost_is_estimate is True
    assert _judge(_api({"type": "noul", "noul": 0.2})).ask_bool("s", "q").answer == "no"


def test_choice_answer_keyed_by_option() -> None:
    probs = {"truncated": 0.7, "paraphrase": 0.2, "noise": 0.1}
    seen: list[Any] = []
    judge = _judge(
        _api(
            {"type": "choice", "choice": "truncated", "confidence": 0.7, "probabilities": probs},
            seen=seen,
        )
    )
    out = judge.ask_choice("state", "why?", ["truncated", "paraphrase", "noise"])
    assert out.answer == "truncated" and out.probabilities == probs
    assert seen[0]["questions"]["q"]["criteria"] == dict.fromkeys(probs)


def test_choice_outside_options_is_unavailable_with_warning() -> None:
    judge = _judge(
        _api(
            {"type": "choice", "choice": "other", "confidence": 0.5, "probabilities": {"other": 1}}
        )
    )
    with capture_logs() as logs:
        out = judge.ask_choice("s", "q", ["a", "b"])
    assert out.answer == "unknown" and not out.available
    assert out.cost_usd == pytest.approx(120 * PRICE)  # the call itself was billed
    assert any(e["event"] == "judge.unexpected_answer" for e in logs)


def test_wrong_answer_type_is_unavailable_with_warning() -> None:
    judge = _judge(_api({"type": "noul", "noul": 0.9}))
    with capture_logs() as logs:
        out = judge.ask_score("s", "q", levels=5)
    assert out.answer == "unknown" and not out.available
    assert any(
        e["event"] == "judge.unexpected_answer" and e["log_level"] == "warning" for e in logs
    )


def test_cost_falls_back_to_chars_over_four_without_token_usage() -> None:
    judge = _judge(
        _api({"type": "noul", "noul": 0.1}, input_tokens=None), usd_per_m_input_tokens=1.0
    )
    out = judge.ask_bool("x" * 400, "q?")
    assert out.cost_usd > 100 / 1_000_000  # >= the 400-char state alone
    from typesafe_sdk import Noul

    chars = 400 + len(Noul(instructions="q?").model_dump_json())
    assert out.cost_usd == pytest.approx(math.ceil(chars / 4) / 1_000_000)


@pytest.mark.parametrize("status", [401, 500])
def test_api_errors_give_unavailable_answers(status: int) -> None:
    judge = _judge(lambda _r: httpx2.Response(status, json={"detail": "nope"}))
    with capture_logs() as logs:
        out = judge.ask_bool("s", "q")
    assert out.answer == "unknown" and not out.available and out.cost_usd == 0.0
    assert any(e["event"] == "judge.unavailable" and e["log_level"] == "warning" for e in logs)


def test_judge_passes_model_and_timeout_to_the_client() -> None:
    with patch("typesafe_sdk.TypeSafeClient") as client_cls:
        judge = TypeSafeJudge(api_key="k", model="jev-x", timeout_s=7)
    client_cls.assert_called_once_with(api_key="k", model="jev-x", timeout=7)
    assert judge.model == "jev-x" and judge.enabled is True
    assert isinstance(judge, JudgeProvider)


# -- build_judge ----------------------------------------------------------------------------------


def test_build_judge_with_a_valid_key() -> None:
    config = JudgeConfig(model="jev-1.14.0", usd_per_m_input_tokens=0.5)
    judge = build_judge(config, SecretStr("ts-key"), timeout_s=4)
    assert isinstance(judge, TypeSafeJudge)
    assert (judge.model, judge.timeout_s, judge.usd_per_m_input_tokens) == ("jev-1.14.0", 4, 0.5)


@pytest.mark.parametrize("key", [None, "", "   "])
def test_build_judge_without_a_key_is_noop(key: str | None) -> None:
    with capture_logs() as logs:
        judge = build_judge(JudgeConfig(), SecretStr(key) if key is not None else None)
    assert isinstance(judge, NoopJudge) and judge.model == "noop"
    assert any(e["event"] == "judge.disabled" and e["log_level"] == "warning" for e in logs)


def test_build_judge_with_a_malformed_key_is_noop_not_a_crash() -> None:
    with capture_logs() as logs:
        judge = build_judge(JudgeConfig(), SecretStr("has space inside"))
    assert isinstance(judge, NoopJudge)
    event = next(e for e in logs if e["event"] == "judge.disabled")
    assert event["log_level"] == "warning" and event["error"] == "TypeSafeError"


def test_build_judge_without_the_sdk_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)  # import -> ImportError
    with capture_logs() as logs:
        judge = build_judge(JudgeConfig(), SecretStr("ts-key"))
    assert isinstance(judge, NoopJudge)
    assert any("not installed" in e.get("reason", "") for e in logs)


def test_build_judge_disabled_by_config() -> None:
    assert isinstance(build_judge(JudgeConfig(enabled=False), SecretStr("ts-key")), NoopJudge)


def test_noop_judge_names_itself_and_costs_nothing() -> None:
    noop = NoopJudge()
    answer = noop.ask_score("s", "q", levels=5)
    assert noop.model == "noop" and noop.enabled is False and isinstance(noop, JudgeProvider)
    assert answer.cost_usd == 0.0 and not answer.available
