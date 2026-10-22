"""Loading config and secrets. Fails with `ConfigError` before any I/O against paid APIs."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from holywater.config.schema import PipelineConfig
from holywater.observability.errors import ConfigError

DEFAULT_CONFIG_PATH = Path("config/pipeline.yaml")
CONFIG_PATH_ENV = "HW_CONFIG"


class Secrets(BaseSettings):
    """API keys from the environment / `.env`. Never logged.

    A blank or whitespace-only value (`KEY=` as shipped in `.env.example`) counts as missing, so
    `require()` raises the typed `ConfigError` instead of a vendor SDK failing later.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True
    )

    fal_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    typesafe_api_key: SecretStr | None = None

    @field_validator("fal_key", "openrouter_api_key", "typesafe_api_key", mode="before")
    @classmethod
    def _blank_is_missing(cls, value: object) -> object:
        raw = value.get_secret_value() if isinstance(value, SecretStr) else value
        if isinstance(raw, str):
            return raw.strip() or None
        return value

    def require(self, *names: str) -> None:
        """Raise `ConfigError` listing every missing key among `names`."""
        missing = [n for n in names if getattr(self, n) is None]
        if missing:
            env_names = ", ".join(n.upper() for n in missing)
            msg = f"missing required secrets: {env_names} (set them in .env or the environment)"
            raise ConfigError(msg, context={"missing": missing})


def resolve_config_path(path: Path | None = None) -> Path:
    if path is not None:
        return path
    return Path(os.environ.get(CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH))


def load_config(path: Path | None = None) -> PipelineConfig:
    """Parse and validate `pipeline.yaml`. Any problem becomes a `ConfigError`."""
    config_path = resolve_config_path(path)
    if not config_path.is_file():
        msg = f"config file not found: {config_path}"
        raise ConfigError(msg, context={"path": str(config_path)})
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"config is not valid YAML: {config_path}"
        raise ConfigError(msg, context={"path": str(config_path), "yaml_error": str(exc)}) from exc
    if not isinstance(raw, dict):
        msg = f"config root must be a mapping: {config_path}"
        raise ConfigError(msg, context={"path": str(config_path)})
    try:
        return PipelineConfig.model_validate(raw)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(map(str, err['loc'])) or '<root>'}: {err['msg']}" for err in exc.errors()
        )
        msg = f"config validation failed ({exc.error_count()} errors) in {config_path}: {details}"
        raise ConfigError(
            msg, context={"path": str(config_path), "errors": exc.errors(include_url=False)}
        ) from exc


def load_secrets() -> Secrets:
    try:
        return Secrets()
    except ValidationError as exc:  # pragma: no cover - only on malformed env values
        raise ConfigError("invalid secrets in environment", context={"errors": str(exc)}) from exc
