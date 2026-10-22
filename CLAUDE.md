# CLAUDE.md — working agreement for coding agents

This repository is a PoC that turns a 1–3 sentence idea into an 8–30 s vertical video where the
quoted line is spoken verbatim. Read `docs/SPEC.md` and `docs/ARCHITECTURE.md` before changing
anything structural; `docs/DECISIONS.md` explains *why* things are the way they are and
`docs/DATA_FLOW.md` §6 lists every failure and its reaction.

## Non-negotiables

1. **The line is never generated.** The spoken line comes from the user's input text. The LLM
   only extracts it; `quality/script_checks.py` verifies it is one complete quoted fragment of the
   idea (or, without quotes, a verbatim fragment) and downstream code uses the original characters.
   Do not "fix", translate or normalize it anywhere except inside `quality/text_norm.py` for
   *comparison* purposes.
2. **No paid call without a gate behind it.** Every generation is followed by a check
   (`DurationGate` per clip and on the final, `SpeechGate`, `MediaIntegrityGate`, the `tts_audio` /
   `voice_fit` / `lipsync_duration` gates, the stray-speech check). New generation steps need a gate too.
3. **No error or warning is lost.** Every `except` either re-raises a typed `PipelineError`
   subclass or logs at WARNING+ and returns an explicit fallback that is recorded in the DB.
   Never `except Exception: pass`. Never print; use `get_logger(__name__)`.
4. **Config over code.** Models, thresholds, retries, timeouts live in `config/pipeline.yaml`
   and are typed in `config/schema.py`. Secrets only in `.env` (`config/settings.py`). Adding a
   knob = schema field + yaml default + test.
5. **Stages are pure; the flow decides.** `stages/*` take explicit deps and return artifacts;
   they know nothing about Prefect or SQL (`stages/report.py` reads through `RunRepository`).
   Retries live on tasks in `flows/`; quality-gate decisions (regenerate, soften, fall back, stop)
   live in the flow body. Idempotent vendor reads (polls, downloads) are retried inside the vendor
   class; anything that would pay for a new job is decided by the task or the flow (ADR-010).
6. **Capabilities are Protocols, vendors are classes.** Stages depend on a capability from
   `providers/base.py`; `providers/registry.py` picks the vendor named in config. A new vendor is
   one class in `providers/<vendor>.py` implementing every capability it offers, with one client
   and one error map (`RetryableProviderError` / `FatalProviderError` and its subclasses
   `PromptRejectedError`, `StructuredOutputError`; `BilledJobError`), registered in
   `registry.VENDORS`. A new capability needs a Protocol, a request type carrying `model`, a
   `Fake*` in `providers/fake.py`, and a config section with `provider:` + `model:`. The one
   exception is the optional judge: a single vendor (TypeSafe Jev) built by
   `providers/typesafe.build_judge` outside `registry.VENDORS`, with no `provider:` key. Request
   bodies are verified against the vendor's OpenAPI schema and recorded in `docs/PROVIDERS_API.md`;
   a 2xx from a live call is not evidence of conformance.
7. **Repo artifacts are in English** (docs, comments, commit messages, log events).

## Layering (a module imports only from rows below its own)

```
cli
 ↓
flows · evals
 ↓
stages
 ↓
providers (Protocols, registry, vendor classes, fakes)
 ↓
quality · media · prompts · storage
 ↓
config
 ↓
domain · observability          import nothing from holywater
```

`storage/` is the only place with SQL (queries in `repo.py`; engine pragmas and the additive
migration in `db.py`). `media/ffmpeg.py` is the only place that spawns ffmpeg.

## Commands

Run from the repository root (config, `.env`, `runs/` and `evals/cases.yaml` are relative paths).

```bash
uv sync                                  # + `--extra judge` for typesafe-sdk (mypy/pytest pass without it)
uv run ruff check . && uv run ruff format --check .
uv run mypy                              # strict; must be clean
uv run pytest                            # unit + integration (ffmpeg, Prefect test server, fakes)
uv run pytest tests/unit                 # ~3 s: no network, no ffmpeg, no Prefect server
uv run hw generate '<idea with "a line">' --dry-run   # full pipeline with fakes, no vendor calls, ~5-15 s
uv run hw eval run --tag <tag> --dry-run # evals/cases.yaml with fakes
```

Run all four checks (ruff, format, mypy, pytest) before claiming a change is done. `--dry-run` is
the smoke test after any change to stages, flow, storage, media or config wiring (it also validates
provider names and capabilities).

## Definition of done for a change

- ruff, mypy, pytest green; `hw generate --dry-run` produces `final.mp4` and `report.json`.
- New behaviour has a unit test; new failure mode has a test that asserts the typed error.
- New stage/gate/provider/tactic is reflected in `docs/DATA_FLOW.md` or `docs/ARCHITECTURE.md`.
- New prompt text → bump `version:` in the prompt file's frontmatter (it lands in `report.json`).
- A changed vendor request body → `docs/PROVIDERS_API.md` and the `==` body test updated together.
- Nothing under `runs/` is committed.

## Conventions

- Python 3.12+ syntax, full type hints, `from __future__ import annotations` in src modules.
- pydantic models: `extra="forbid"`; config models are frozen.
- Enums are `StrEnum`; stored in SQLite as `.value` strings, converted at the repo boundary.
- Log events are dotted lowercase (`stage.start`, `verify_speech.failed`) with structured
  fields, not f-strings. Bind run context with `log_context(...)`, not by passing `run_id` around.
- Tests: `tests/unit` must not touch network or ffmpeg and starts no Prefect server;
  `tests/integration` may use ffmpeg and the Prefect test harness (`tests/integration/conftest.py`)
  but never real providers.
- Costs: every billed call becomes one `costs` row through `flows._record_cost` and one
  `cost.recorded` log event. Stages report charges to a cost sink (`ProviderCost`, `on_cost`) as
  soon as they are known — including billed LLM answers that fail validation and failed vendor
  calls that carry `spent_usd` — so a later failure or Prefect retry cannot lose them; the flow
  adds Jev answers (provider `typesafe`) and lipsync charges returned by assemble. Rows carry the
  provider from config, `shot_id`/`attempt` when they belong to a cycle, and `is_estimate`;
  vendors set `cost_is_estimate=False` only when the API reported the charge. 0 is acceptable
  only for fakes — the known gap is fal lipsync (no price source yet; WARNING `cost.unknown_price`).

## Spending real money

Real runs need `OPENROUTER_API_KEY` in `.env` (and `FAL_KEY` only if a capability names `fal`).
With the defaults (OpenRouter, 480p) the one measured run — two 6 s shots, no retries — cost
**$1.26**: Seedance 2.5 ≈ $0.1035 per second ($0.62 per 6 s shot), the LLM ≈ $0.02, STT and Jev
well under a cent. Every regenerated or fallback clip adds its own cost again; 720p is ≈ $0.233/s
(≈ 2.25×). The fal prices ($0.22/s at 480p, $0.47/s at 720p) apply only with `video.provider: fal`.
Keep `video.resolution: 480p`. `hw eval run` without `--dry-run` is 8 paid runs (it asks first).
Do not start paid runs or eval suites on your own initiative — ask first.

## Where things are

| Need | Look at |
|------|---------|
| Switch vendor or model for a capability | `config/pipeline.yaml` (`provider:` + `model:`) |
| Add a vendor | `providers/<vendor>.py` + `providers/registry.py` (`VENDORS`, `_build`, `_vendor_supports`) |
| Check a request against the vendor API | `docs/PROVIDERS_API.md` (schemas from `openapi.json`, not tutorials) + `tests/unit/test_provider_*.py` |
| Change retry/fallback logic | `flows/generate_video_flow.py` (`_produce_shot`, `_native_line_attempts`, `_speechless_attempts`, `_fallback_to_tts`, `_generate_attempt`, `_stage_assemble`, `_verify_final`); tactic table in `docs/DATA_FLOW.md` §6 |
| Change resume rules | `flows/generate_video_flow.py` (`_script_input_hash`, `_resume_accepted`, `_resume_pending`, `_reusable_clip`, `_resume_mismatch`) |
| Tune a gate or tactic threshold | `config/pipeline.yaml` → `quality.*` (`max_wer`, `shot_duration_tolerance_s`, `truncated_window_widen_s`, `speech_chars_per_s`, `min_audio_peak_db`, `final_check_margin_s`, `stray_speech`) |
| Change what counts as "verbatim" | `quality/text_norm.py`, `quality/script_checks.py` + `tests/unit/test_text_norm.py`, `tests/unit/test_script_checks.py` |
| Change the video prompt | `src/holywater/prompts/seedance_shot.md` (bump version) |
| Change the script schema | `domain/script.py` (+ invariants), `quality/script_checks.py` and `prompts/scriptwriter.md` |
| Add a metric to the report | `domain/report.py`, `stages/report.py`, `storage/repo.py::tag_stats`, `evals/__init__.py::render_stats` |
| Record a new kind of cost | `flows/generate_video_flow.py` (`_cost_sink`, `_record_cost`), `domain/artifacts.py::ProviderCost` |
| Simulate a failure in tests | `providers/fake.py`: `FakeVideo` (`fail_times`, `reject_times` → `PromptRejectedError`, `expire_times`, `billed_times`, `mute_calls`, `duration_offset_s` / `duration_offset_calls`, `speech_mutator`, `stray_speech`, `reject_if_contains`), `FakeLLM` (`fail_times`, `invalid_times`, `line_override`), `FakeTTS` (`chars_per_s`, `reject_voices`), `FakeSTT` (`queue`, `fail_times`, `fatal_status`; it hears only the transcribed window), `FakeJudge` (`available`, `*_answers`) |
