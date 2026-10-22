from pathlib import Path

import pytest

from holywater.config import PipelineConfig, Secrets, load_config
from holywater.media import FFmpeg
from holywater.observability.errors import ConfigError
from holywater.providers.base import (
    LipsyncProvider,
    LLMProvider,
    NoopJudge,
    STTProvider,
    TTSProvider,
    VideoProvider,
)
from holywater.providers.fake import FakeJudge, FakeLLM, FakeSTT, FakeVideo
from holywater.providers.fal import FalProvider
from holywater.providers.openrouter import OpenRouterProvider
from holywater.providers.registry import ProviderRegistry

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config" / "pipeline.yaml"


def _secrets(**values: str) -> Secrets:
    return Secrets(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.fixture
def config() -> PipelineConfig:
    return load_config(REPO_CONFIG)


def test_repo_config_defaults_to_openrouter_for_everything_but_lipsync(
    config: PipelineConfig,
) -> None:
    assert config.vendors_in_use() == {"openrouter"}
    assert config.audio.lipsync.provider == "fal"


def test_dry_run_resolves_fakes(config: PipelineConfig) -> None:
    registry = ProviderRegistry(config, _secrets(), ffmpeg=FFmpeg(), dry_run=True)
    assert isinstance(registry.resolve(LLMProvider, "openrouter"), FakeLLM)
    assert isinstance(registry.resolve(VideoProvider, "fal"), FakeVideo)
    assert isinstance(registry.resolve(STTProvider, "openrouter"), FakeSTT)
    assert registry.resolve(STTProvider, "openrouter") is registry.resolve(STTProvider, "fal")
    assert isinstance(registry.judge(), FakeJudge)


def test_real_vendors_are_built_once_and_shared(config: PipelineConfig) -> None:
    registry = ProviderRegistry(
        config, _secrets(openrouter_api_key="o", fal_key="f"), ffmpeg=FFmpeg()
    )
    llm = registry.resolve(LLMProvider, "openrouter")
    video = registry.resolve(VideoProvider, "openrouter")
    assert isinstance(llm, OpenRouterProvider) and llm is video
    assert isinstance(registry.resolve(LipsyncProvider, "fal"), FalProvider)
    assert isinstance(registry.resolve(TTSProvider, "fal"), FalProvider)


def test_missing_capability_is_config_error(config: PipelineConfig) -> None:
    registry = ProviderRegistry(
        config, _secrets(fal_key="f", openrouter_api_key="o"), ffmpeg=FFmpeg()
    )
    with pytest.raises(ConfigError, match="no llm capability") as exc_info:
        registry.resolve(LLMProvider, "fal")
    assert "openrouter" in str(exc_info.value)
    with pytest.raises(ConfigError, match="no lipsync capability"):
        registry.resolve(LipsyncProvider, "openrouter")


def test_missing_secret_is_config_error(config: PipelineConfig) -> None:
    registry = ProviderRegistry(config, _secrets(openrouter_api_key="o"), ffmpeg=FFmpeg())
    with pytest.raises(ConfigError, match="FAL_KEY"):
        registry.resolve(VideoProvider, "fal")
    assert isinstance(registry.resolve(VideoProvider, "openrouter"), OpenRouterProvider)


def test_unknown_vendor_rejected_by_schema_and_registry(config: PipelineConfig) -> None:
    raw = config.model_dump(mode="json")
    raw["video"]["provider"] = "replicate"
    with pytest.raises(ValueError, match="replicate"):
        PipelineConfig.model_validate(raw)
    registry = ProviderRegistry(config, _secrets(openrouter_api_key="o"), ffmpeg=FFmpeg())
    with pytest.raises(ConfigError, match="unknown provider"):
        registry.resolve(VideoProvider, "replicate")


def test_judge_without_key_is_noop(config: PipelineConfig) -> None:
    registry = ProviderRegistry(config, _secrets(openrouter_api_key="o"), ffmpeg=FFmpeg())
    assert isinstance(registry.judge(), NoopJudge)


def _with(config: PipelineConfig, **sections: dict[str, object]) -> PipelineConfig:
    raw = config.model_dump(mode="json")
    for path, value in sections.items():
        section, key = path.split("__")
        raw[section][key] = value
    return PipelineConfig.model_validate(raw)


def test_dry_run_still_checks_capabilities_without_keys(config: PipelineConfig) -> None:
    bad = _with(config, llm__provider="fal")
    registry = ProviderRegistry(bad, _secrets(), ffmpeg=FFmpeg(), dry_run=True)
    with pytest.raises(ConfigError, match="no llm capability"):
        registry.llm(bad.llm.provider)
    with pytest.raises(ConfigError, match="no lipsync capability"):
        registry.lipsync("openrouter")
    with pytest.raises(ConfigError, match="unknown provider"):
        registry.resolve(VideoProvider, "replicate")


def test_vendors_get_per_capability_timeouts_and_knobs(config: PipelineConfig) -> None:
    raw = config.model_dump(mode="json")
    raw["retries"]["timeouts_s"] = {"llm": 11, "video": 222, "tts": 33, "stt": 44, "judge": 5}
    raw["providers"]["openrouter"].update(
        download_attempts=5, video_usd_per_s={"480p": 0.5, "720p": 1.0}
    )
    raw["providers"]["fal"]["download_attempts"] = 2
    cfg = PipelineConfig.model_validate(raw)
    registry = ProviderRegistry(cfg, _secrets(openrouter_api_key="o", fal_key="f"), ffmpeg=FFmpeg())
    openrouter = registry.resolve(VideoProvider, "openrouter")
    assert (openrouter.timeout_s, openrouter.llm_timeout_s) == (222, 11)
    assert (openrouter.tts_timeout_s, openrouter.stt_timeout_s) == (33, 44)
    assert openrouter.download_attempts == 5
    assert dict(openrouter.video_usd_per_s) == {"480p": 0.5, "720p": 1.0}
    fal = registry.resolve(VideoProvider, "fal")
    assert (fal.timeout_s, fal.tts_timeout_s, fal.stt_timeout_s) == (222, 33, 44)
    assert fal.download_attempts == 2


def test_blank_keys_are_missing_secrets(config: PipelineConfig) -> None:
    registry = ProviderRegistry(
        config, _secrets(openrouter_api_key="", fal_key="   "), ffmpeg=FFmpeg()
    )
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        registry.resolve(LLMProvider, "openrouter")
    with pytest.raises(ConfigError, match="FAL_KEY"):
        registry.resolve(VideoProvider, "fal")


def test_blank_judge_key_is_noop(config: PipelineConfig) -> None:
    registry = ProviderRegistry(
        config, _secrets(openrouter_api_key="o", typesafe_api_key=" "), ffmpeg=FFmpeg()
    )
    judge = registry.judge()
    assert isinstance(judge, NoopJudge) and judge.model == "noop"


def test_judge_gets_the_judge_timeout(config: PipelineConfig) -> None:
    pytest.importorskip("typesafe_sdk")
    from holywater.providers.typesafe import TypeSafeJudge

    registry = ProviderRegistry(
        config, _secrets(openrouter_api_key="o", typesafe_api_key="ts-key"), ffmpeg=FFmpeg()
    )
    judge = registry.judge()
    assert isinstance(judge, TypeSafeJudge)
    assert judge.timeout_s == config.retries.timeouts_s.judge
    assert judge.model == config.judge.model
