"""Secrets (blank keys count as missing) and the provider/retry/judge config knobs."""

from pathlib import Path

import pytest

from holywater.config import PipelineConfig, Secrets, load_config
from holywater.domain.enums import Resolution
from holywater.observability.errors import ConfigError

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config" / "pipeline.yaml"
KEYS = ("FAL_KEY", "OPENROUTER_API_KEY", "TYPESAFE_API_KEY")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)


def test_blank_keys_from_a_copied_env_example_are_missing(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("FAL_KEY=\nOPENROUTER_API_KEY=\nTYPESAFE_API_KEY=   \n")
    secrets = Secrets(_env_file=env)  # type: ignore[call-arg]
    assert (secrets.fal_key, secrets.openrouter_api_key, secrets.typesafe_api_key) == (
        None,
        None,
        None,
    )
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY") as exc_info:
        secrets.require("openrouter_api_key", "fal_key")
    assert exc_info.value.context["missing"] == ["openrouter_api_key", "fal_key"]


def test_blank_environment_variables_are_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("FAL_KEY", " \t ")
    secrets = Secrets(_env_file=None)  # type: ignore[call-arg]
    assert secrets.openrouter_api_key is None and secrets.fal_key is None


def test_keys_are_stripped_and_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "  sk-or-123 \n")
    secrets = Secrets(_env_file=None)  # type: ignore[call-arg]
    assert secrets.openrouter_api_key is not None
    assert secrets.openrouter_api_key.get_secret_value() == "sk-or-123"
    secrets.require("openrouter_api_key")
    assert "sk-or-123" not in repr(secrets)


def test_explicit_blank_values_are_missing() -> None:
    secrets = Secrets(_env_file=None, typesafe_api_key="")  # type: ignore[call-arg]
    assert secrets.typesafe_api_key is None


# -- config knobs owned by the providers ----------------------------------------------------------


def _raw() -> dict:
    return load_config(REPO_CONFIG).model_dump(mode="json")


def test_repo_config_provider_knobs() -> None:
    cfg = load_config(REPO_CONFIG)
    assert cfg.retries.provider.attempts == 3 and cfg.retries.provider.backoff_s == [5, 15]
    assert cfg.providers.openrouter.download_attempts == 3
    assert cfg.providers.openrouter.video_usd_per_s == {
        Resolution.P480: 0.1035,
        Resolution.P720: 0.233,
    }
    assert cfg.providers.fal.download_attempts == 3
    assert cfg.judge.usd_per_m_input_tokens == 0.042


@pytest.mark.parametrize(
    ("attempts", "backoff", "ok"),
    [(3, [5, 15], True), (1, [], True), (3, [5, 15, 45], False), (3, [5], False), (2, [-1], False)],
)
def test_backoff_has_one_delay_per_retry(attempts: int, backoff: list[int], ok: bool) -> None:
    raw = _raw()
    raw["retries"]["provider"] = {"attempts": attempts, "backoff_s": backoff}
    if ok:
        assert PipelineConfig.model_validate(raw).retries.provider.backoff_s == backoff
    else:
        with pytest.raises(ValueError, match="backoff_s"):
            PipelineConfig.model_validate(raw)


def test_yaml_with_a_stale_backoff_list_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "pipeline.yaml"
    path.write_text(REPO_CONFIG.read_text().replace("backoff_s: [5, 15]", "backoff_s: [5, 15, 45]"))
    with pytest.raises(ConfigError, match="validation failed"):
        load_config(path)


def test_config_error_names_the_field_and_the_reason(tmp_path: Path) -> None:
    """The CLI prints only the message, so the message must say which key is wrong and why."""
    text = REPO_CONFIG.read_text()
    video = "video:\n  provider: openrouter"
    assert video in text
    path = tmp_path / "pipeline.yaml"
    path.write_text(text.replace(video, "video:\n  provider: foo"))
    with pytest.raises(ConfigError, match=r"video\.provider: Input should be") as exc_info:
        load_config(path)
    assert str(path) in str(exc_info.value)


def test_git_sha_marks_a_dirty_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from holywater.flows import wiring

    outputs = {"rev-parse": "f7b5df3\n", "status": " M src/x.py\n"}

    def run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout=outputs[args[1]], stderr="")

    monkeypatch.setattr(wiring.subprocess, "run", run)
    assert wiring.git_sha() == "f7b5df3-dirty"
    outputs["status"] = ""
    assert wiring.git_sha() == "f7b5df3"


def test_git_sha_outside_a_checkout_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from holywater.flows import wiring

    def run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 128, stdout="", stderr="not a git repository")

    monkeypatch.setattr(wiring.subprocess, "run", run)
    assert wiring.git_sha() is None


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("openrouter", "download_attempts", 0),
        ("fal", "download_attempts", 0),
        ("openrouter", "video_usd_per_s", {"1080p": 0.5}),
    ],
)
def test_invalid_provider_knobs_are_rejected(section: str, key: str, value: object) -> None:
    raw = _raw()
    raw["providers"][section][key] = value
    with pytest.raises(ValueError, match=key):
        PipelineConfig.model_validate(raw)


def test_negative_judge_price_is_rejected() -> None:
    raw = _raw()
    raw["judge"]["usd_per_m_input_tokens"] = -1
    with pytest.raises(ValueError, match="usd_per_m_input_tokens"):
        PipelineConfig.model_validate(raw)
