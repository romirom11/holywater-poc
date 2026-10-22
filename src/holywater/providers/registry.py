"""Resolves `provider:` names from config to vendor instances that implement a capability.

Vendors are built lazily and once per run, so switching `video.provider` from `fal` to
`openrouter` in `pipeline.yaml` is the whole change. A vendor that lacks the requested
capability (e.g. `fal` for the LLM) is a `ConfigError` raised before any paid call.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from holywater.config.schema import PipelineConfig
from holywater.config.settings import Secrets
from holywater.media import FFmpeg
from holywater.observability.errors import ConfigError
from holywater.observability.logging import get_logger
from holywater.providers.base import (
    JudgeProvider,
    LipsyncProvider,
    LLMProvider,
    STTProvider,
    TTSProvider,
    VideoProvider,
)

log = get_logger(__name__)

VENDORS: tuple[str, ...] = ("fal", "openrouter")
_CAPABILITY_NAMES: dict[type, str] = {
    LLMProvider: "llm",
    VideoProvider: "video",
    TTSProvider: "tts",
    STTProvider: "stt",
    LipsyncProvider: "lipsync",
}


class ProviderRegistry:
    def __init__(
        self,
        config: PipelineConfig,
        secrets: Secrets,
        *,
        ffmpeg: FFmpeg,
        dry_run: bool = False,
    ) -> None:
        self._config = config
        self._secrets = secrets
        self._ffmpeg = ffmpeg
        self._dry_run = dry_run
        self._vendors: dict[str, Any] = {}

    # Typed accessors, one per capability (mypy will not accept a Protocol as `type[T]`).
    def llm(self, vendor: str) -> LLMProvider:
        return self.resolve(LLMProvider, vendor)  # type: ignore[no-any-return]

    def video(self, vendor: str) -> VideoProvider:
        return self.resolve(VideoProvider, vendor)  # type: ignore[no-any-return]

    def tts(self, vendor: str) -> TTSProvider:
        return self.resolve(TTSProvider, vendor)  # type: ignore[no-any-return]

    def stt(self, vendor: str) -> STTProvider:
        return self.resolve(STTProvider, vendor)  # type: ignore[no-any-return]

    def lipsync(self, vendor: str) -> LipsyncProvider:
        return self.resolve(LipsyncProvider, vendor)  # type: ignore[no-any-return]

    def resolve(self, capability: Any, vendor: str) -> Any:
        """Return the vendor instance for `vendor`, checked to implement `capability`."""
        cap_name = _CAPABILITY_NAMES.get(capability, capability.__name__)
        if self._dry_run:
            return self._fake(capability)
        instance = self._vendor(vendor)
        if not isinstance(instance, capability):
            supported = [v for v in VENDORS if self._vendor_supports(v, capability)]
            msg = f"provider {vendor!r} has no {cap_name} capability; use one of {supported}"
            raise ConfigError(msg, context={"capability": cap_name, "provider": vendor})
        log.debug("registry.resolved", capability=cap_name, provider=vendor)
        return instance

    def judge(self) -> JudgeProvider:
        if self._dry_run:
            from holywater.providers.base import NoopJudge
            from holywater.providers.fake import FakeJudge

            return FakeJudge() if self._config.judge.enabled else NoopJudge()
        from holywater.providers.typesafe import build_judge

        return build_judge(
            self._config.judge,
            self._secrets.typesafe_api_key,
            timeout_s=self._config.retries.timeouts_s.judge,
        )

    # -- internals -----------------------------------------------------------------------------

    def _vendor(self, name: str) -> Any:
        if name in self._vendors:
            return self._vendors[name]
        if name not in VENDORS:
            msg = f"unknown provider {name!r}; known providers: {list(VENDORS)}"
            raise ConfigError(msg, context={"provider": name})
        instance = self._build(name)
        self._vendors[name] = instance
        log.info("registry.vendor_ready", provider=name)
        return instance

    def _build(self, name: str) -> Any:
        timeouts = self._config.retries.timeouts_s
        if name == "fal":
            self._secrets.require("fal_key")
            from holywater.providers.fal import FalProvider

            assert self._secrets.fal_key is not None
            return FalProvider(
                api_key=self._secrets.fal_key.get_secret_value(),
                timeout_s=timeouts.video,
                lipsync_sync_mode=self._config.providers.fal.lipsync_sync_mode,
            )
        if name == "openrouter":
            self._secrets.require("openrouter_api_key")
            from holywater.providers.openrouter import OpenRouterProvider

            assert self._secrets.openrouter_api_key is not None
            settings = self._config.providers.openrouter
            return OpenRouterProvider(
                api_key=self._secrets.openrouter_api_key.get_secret_value(),
                base_url=settings.base_url,
                timeout_s=timeouts.video,
                llm_timeout_s=timeouts.llm,
                video_poll_interval_s=settings.video_poll_interval_s,
            )
        msg = f"no builder for provider {name!r}"  # pragma: no cover - guarded by VENDORS
        raise ConfigError(msg)

    @staticmethod
    def _vendor_supports(name: str, capability: type) -> bool:
        from holywater.providers.fal import FalProvider
        from holywater.providers.openrouter import OpenRouterProvider

        cls = {"fal": FalProvider, "openrouter": OpenRouterProvider}[name]
        # Structural check against the class: Protocol members are methods, so this is enough.
        return all(callable(getattr(cls, m, None)) for m in _protocol_methods(capability))

    def _fake(self, capability: Any) -> Any:
        from holywater.providers import fake

        key = capability.__name__
        if key not in self._vendors:
            builders: dict[str, Any] = {
                "LLMProvider": fake.FakeLLM,
                "VideoProvider": partial(fake.FakeVideo, ffmpeg=self._ffmpeg),
                "TTSProvider": partial(fake.FakeTTS, ffmpeg=self._ffmpeg),
                "STTProvider": fake.FakeSTT,
                "LipsyncProvider": fake.FakeLipsync,
            }
            self._vendors[key] = builders[key]()
        return self._vendors[key]


def _protocol_methods(protocol: type) -> list[str]:
    attrs = getattr(protocol, "__protocol_attrs__", None)
    if attrs is None:  # pragma: no cover - older typing fallback
        attrs = {n for n in vars(protocol) if not n.startswith("_")}
    return sorted(attrs)
