from pathlib import Path

import pytest

from holywater.config import PipelineConfig, Secrets, load_config
from holywater.config.settings import resolve_config_path
from holywater.domain.enums import AudioStrategy, Resolution
from holywater.observability.errors import ConfigError

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config" / "pipeline.yaml"


def test_repo_config_loads() -> None:
    cfg = load_config(REPO_CONFIG)
    assert cfg.video.resolution is Resolution.P480
    assert cfg.audio.strategy is AudioStrategy.NATIVE
    assert cfg.llm.model.startswith("anthropic/")
    assert len(cfg.config_hash()) == 16


def test_config_hash_changes_with_resolution() -> None:
    cfg = load_config(REPO_CONFIG)
    raw = cfg.model_dump(mode="json")
    raw["video"]["resolution"] = "720p"
    assert cfg.config_hash() != PipelineConfig.model_validate(raw).config_hash()


def test_missing_file_is_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml_is_config_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("llm: [unclosed")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path)


def test_unknown_key_is_config_error(tmp_path: Path) -> None:
    path = tmp_path / "extra.yaml"
    path.write_text(REPO_CONFIG.read_text() + "\nunknown_section: 1\n")
    with pytest.raises(ConfigError) as exc_info:
        load_config(path)
    assert exc_info.value.context["errors"][0]["loc"] == ("unknown_section",)


def test_out_of_range_value_is_config_error(tmp_path: Path) -> None:
    path = tmp_path / "range.yaml"
    path.write_text(REPO_CONFIG.read_text().replace("max_wer: 0.10", "max_wer: 1.5"))
    with pytest.raises(ConfigError, match="validation failed"):
        load_config(path)


def test_range_min_le_max(tmp_path: Path) -> None:
    path = tmp_path / "range.yaml"
    path.write_text(REPO_CONFIG.read_text().replace("{min: 8, max: 30}", "{min: 31, max: 30}"))
    with pytest.raises(ConfigError, match="validation failed"):
        load_config(path)


def test_config_path_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HW_CONFIG", str(tmp_path / "x.yaml"))
    assert resolve_config_path() == tmp_path / "x.yaml"
    assert resolve_config_path(Path("explicit.yaml")) == Path("explicit.yaml")


def test_secrets_require_lists_all_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("FAL_KEY", "OPENROUTER_API_KEY", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    secrets = Secrets(_env_file=None)  # type: ignore[call-arg]
    with pytest.raises(ConfigError) as exc_info:
        secrets.require("fal_key", "openrouter_api_key")
    assert exc_info.value.context["missing"] == ["fal_key", "openrouter_api_key"]
    assert "FAL_KEY" in str(exc_info.value)


def test_secrets_are_not_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAL_KEY", "super-secret")
    secrets = Secrets(_env_file=None)  # type: ignore[call-arg]
    assert "super-secret" not in repr(secrets)
    assert secrets.fal_key is not None
    assert secrets.fal_key.get_secret_value() == "super-secret"
