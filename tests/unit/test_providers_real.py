"""Error mapping and payload parsing of the real adapters, with the SDKs mocked out."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest
from fal_client.client import FalClientHTTPError, FalClientTimeoutError, SyncClient

from holywater.domain.enums import Resolution
from holywater.domain.script import Script
from holywater.observability.errors import FatalProviderError, RetryableProviderError
from holywater.providers.base import NoopJudge, TTSRequest, VideoRequest
from holywater.providers.fal_base import FalBase, file_url
from holywater.providers.fal_stt import FalSTT
from holywater.providers.fal_tts import FalTTS
from holywater.providers.fal_video import FalVideo
from holywater.providers.openrouter_llm import OpenRouterLLM
from tests.unit.test_domain import make_script


def _http_error(status: int) -> FalClientHTTPError:
    return FalClientHTTPError(
        f"status {status}", status_code=status, response_headers={}, response=MagicMock()
    )


# -- fal base ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "retryable"), [(429, True), (503, True), (400, False), (401, False)]
)
def test_fal_http_errors_are_classified(status: int, retryable: bool) -> None:
    base = FalBase(api_key="k", model="m")
    expected = RetryableProviderError if retryable else FatalProviderError
    with (
        patch.object(SyncClient, "subscribe", side_effect=_http_error(status)),
        pytest.raises(expected) as exc_info,
    ):
        base.call({}, op="x")
    assert exc_info.value.status_code == status
    assert exc_info.value.provider == "fal"


def test_fal_timeout_and_transport_are_retryable() -> None:
    base = FalBase(api_key="k", model="m", timeout_s=5)
    with (
        patch.object(SyncClient, "subscribe", side_effect=FalClientTimeoutError("slow")),
        pytest.raises(RetryableProviderError, match="timed out"),
    ):
        base.call({}, op="x")
    with (
        patch.object(SyncClient, "subscribe", side_effect=httpx.ConnectError("down")),
        pytest.raises(RetryableProviderError, match="transport"),
    ):
        base.call({}, op="x")


def test_fal_non_object_result_is_fatal() -> None:
    base = FalBase(api_key="k", model="m")
    with (
        patch.object(SyncClient, "subscribe", return_value=["nope"]),
        pytest.raises(FatalProviderError, match="non-object"),
    ):
        base.call({}, op="x")


def test_file_url_extraction() -> None:
    assert file_url({"video": {"url": "https://x/v.mp4"}}, "video") == "https://x/v.mp4"
    with pytest.raises(FatalProviderError, match=r"video\.url"):
        file_url({"other": 1}, "video")


# -- fal adapters -------------------------------------------------------------------------------


def test_fal_video_builds_arguments_and_downloads(tmp_path: Path) -> None:
    video = FalVideo(api_key="k", model="bytedance/seedance-2.5/text-to-video")
    captured: dict[str, Any] = {}

    def fake_subscribe(model: str, arguments: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        captured.update(arguments)
        kwargs["on_enqueue"]("req-1")
        return {"video": {"url": "https://x/v.mp4"}, "seed": 77}

    dest = tmp_path / "shot.mp4"
    with (
        patch.object(SyncClient, "subscribe", side_effect=fake_subscribe),
        patch.object(FalBase, "download", return_value=dest) as dl,
    ):
        result = video.generate(
            VideoRequest(
                prompt="p",
                duration_s=8,
                resolution=Resolution.P480,
                aspect_ratio="9:16",
                generate_audio=True,
                seed=5,
            ),
            dest=dest,
        )
    assert captured == {
        "prompt": "p",
        "duration": "8",
        "resolution": "480p",
        "aspect_ratio": "9:16",
        "generate_audio": True,
        "seed": 5,
    }
    dl.assert_called_once()
    assert result.seed == 77 and result.request_id == "req-1"
    assert result.cost_usd == pytest.approx(0.2205 * 8, abs=1e-3)


def test_fal_tts_elevenlabs_disables_normalization(tmp_path: Path) -> None:
    tts = FalTTS(api_key="k", model="fal-ai/elevenlabs/tts/eleven-v3")
    captured: dict[str, Any] = {}

    def fake_subscribe(model: str, arguments: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        captured.update(arguments)
        return {"audio": {"url": "https://x/a.mp3"}}

    with (
        patch.object(SyncClient, "subscribe", side_effect=fake_subscribe),
        patch.object(FalBase, "download"),
    ):
        result = tts.synthesize(
            TTSRequest(text="Meet me at 7.", language="en", voice="Rachel"), dest=tmp_path / "a.mp3"
        )
    assert captured["apply_text_normalization"] == "off"
    assert captured["language_code"] == "en" and captured["voice"] == "Rachel"
    assert result.cost_usd > 0


def test_fal_stt_parses_text_and_language(tmp_path: Path) -> None:
    stt = FalSTT(api_key="k", model="fal-ai/whisper")
    with (
        patch.object(FalBase, "upload", return_value="https://x/in.mp4"),
        patch.object(
            SyncClient,
            "subscribe",
            return_value={"text": "  hello there ", "inferred_languages": ["en"]},
        ),
    ):
        result = stt.transcribe(tmp_path / "in.mp4", language="auto")
    assert result.text == "hello there" and result.language == "en"
    with (
        patch.object(FalBase, "upload", return_value="https://x/in.mp4"),
        patch.object(SyncClient, "subscribe", return_value={"chunks": []}),
        pytest.raises(FatalProviderError, match="no text"),
    ):
        stt.transcribe(tmp_path / "in.mp4")


# -- openrouter ---------------------------------------------------------------------------------


def _completion(content: str | None, *, cost: float = 0.0123) -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens, usage.completion_tokens = 100, 50
    usage.model_extra = {"cost": cost}
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message, choice.finish_reason = message, "stop"
    completion = MagicMock()
    completion.choices, completion.usage, completion.model, completion.id = (
        [choice],
        usage,
        "anthropic/claude-sonnet-5.5",
        "gen-1",
    )
    return completion


def test_openrouter_structured_output_roundtrip() -> None:
    llm = OpenRouterLLM(api_key="k", model="anthropic/claude-sonnet-5.5")
    script_json = Script.model_validate(make_script()).model_dump_json()
    with patch.object(
        llm._client.chat.completions, "create", return_value=_completion(script_json)
    ) as create:
        result = llm.complete_structured(system="s", user="u", schema=Script, temperature=0.2)
    kwargs = create.call_args.kwargs
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["strict"] is True
    assert kwargs["extra_body"] == {"provider": {"require_parameters": True}}
    assert result.data.dialogue.text == "Are you sure this is the address?"
    assert result.cost_usd == pytest.approx(0.0123)
    assert (result.input_tokens, result.output_tokens) == (100, 50)


def test_openrouter_unwraps_single_key_envelope() -> None:
    llm = OpenRouterLLM(api_key="k", model="m")
    wrapped = '{"Script": ' + Script.model_validate(make_script()).model_dump_json() + "}"
    with patch.object(llm._client.chat.completions, "create", return_value=_completion(wrapped)):
        result = llm.complete_structured(system="s", user="u", schema=Script, temperature=0.2)
    assert result.data.title == "Rain"


def test_openrouter_empty_content_is_fatal() -> None:
    llm = OpenRouterLLM(api_key="k", model="m")
    with (
        patch.object(llm._client.chat.completions, "create", return_value=_completion(None)),
        pytest.raises(FatalProviderError, match="no content"),
    ):
        llm.complete_structured(system="s", user="u", schema=Script, temperature=0.2)


def _status_error(status: int) -> openai.APIStatusError:
    response = httpx.Response(status, request=httpx.Request("POST", "https://x"))
    return openai.APIStatusError("boom", response=response, body=None)


def test_openrouter_error_mapping() -> None:
    llm = OpenRouterLLM(api_key="k", model="m")
    cases = [
        (
            openai.RateLimitError(
                "rl",
                response=httpx.Response(429, request=httpx.Request("POST", "https://x")),
                body=None,
            ),
            RetryableProviderError,
        ),
        (
            openai.APITimeoutError(request=httpx.Request("POST", "https://x")),
            RetryableProviderError,
        ),
        (_status_error(502), RetryableProviderError),
        (_status_error(400), FatalProviderError),
    ]
    for error, expected in cases:
        with (
            patch.object(llm._client.chat.completions, "create", side_effect=error),
            pytest.raises(expected),
        ):
            llm.complete_structured(system="s", user="u", schema=Script, temperature=0.2)


# -- judge ---------------------------------------------------------------------------------------


def test_build_judge_falls_back_to_noop() -> None:
    from pydantic import SecretStr

    from holywater.config.schema import JudgeConfig
    from holywater.providers.typesafe_judge import build_judge

    assert isinstance(build_judge(JudgeConfig(enabled=False), SecretStr("k")), NoopJudge)
    assert isinstance(build_judge(JudgeConfig(), None), NoopJudge)


def test_typesafe_judge_maps_answers() -> None:
    from holywater.providers.typesafe_judge import TypeSafeJudge

    judge = TypeSafeJudge(api_key="k")
    noul = MagicMock()
    noul.noul = 0.82
    choice = MagicMock()
    choice.choice, choice.probabilities = "truncated", {"truncated": 0.7, "paraphrased": 0.3}
    score = MagicMock()
    score.score, score.probabilities = "4: high", None

    def fake_system_one(**kwargs: Any) -> MagicMock:
        q = kwargs["questions"]["q"]
        resp = MagicMock()
        resp.model = "jev-1.14.0"
        resp.answers = {"q": {"noul": noul, "choice": choice, "score": score}[q.type]}
        return resp

    with patch.object(judge._client, "system_one", side_effect=fake_system_one):
        b = judge.ask_bool("s", "risky?")
        c = judge.ask_choice("s", "why?", ["paraphrased", "truncated"])
        s = judge.ask_score("s", "how good?", levels=5)
    assert b.answer == "yes" and b.probabilities["yes"] == pytest.approx(0.82)
    assert c.answer == "truncated" and c.confidence == pytest.approx(0.7)
    assert s.answer == "4" and b.model_version == "jev-1.14.0"


def test_typesafe_judge_unavailable_on_sdk_error() -> None:
    from typesafe_sdk import TypeSafeAPIConnectionError

    from holywater.providers.typesafe_judge import TypeSafeJudge

    judge = TypeSafeJudge(api_key="k")
    with patch.object(
        judge._client,
        "system_one",
        side_effect=TypeSafeAPIConnectionError("connection refused"),
    ):
        answer = judge.ask_bool("s", "q")
    assert answer.available is False and answer.answer == "unknown"
