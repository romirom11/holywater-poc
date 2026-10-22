# CLAUDE.md — working agreement for coding agents

This repository is a PoC that turns a 1–3 sentence idea into an 8–30 s vertical video where the
quoted line is spoken verbatim. Read `docs/SPEC.md` and `docs/ARCHITECTURE.md` before changing
anything structural; `docs/DECISIONS.md` explains *why* things are the way they are.

## Non-negotiables

1. **The line is never generated.** The spoken line comes from the user's input text. The LLM
   only extracts it; `quality/script_checks.py` verifies it is a verbatim substring and
   downstream code uses the original characters. Do not "fix", translate or normalize it anywhere
   except inside `quality/text_norm.py` for *comparison* purposes.
2. **No paid call without a gate behind it.** Every generation is followed by a check
   (`DurationGate`, `SpeechGate`, `MediaIntegrityGate`). New generation steps need a gate too.
3. **No error or warning is lost.** Every `except` either re-raises a typed `PipelineError`
   subclass or logs at WARNING+ and returns an explicit fallback that is recorded in the DB.
   Never `except Exception: pass`. Never print; use `get_logger(__name__)`.
4. **Config over code.** Models, thresholds, retries, timeouts live in `config/pipeline.yaml`
   and are typed in `config/schema.py`. Secrets only in `.env` (`config/settings.py`). Adding a
   knob = schema field + yaml default + test.
5. **Stages are pure; the flow decides.** `stages/*` take explicit deps and return artifacts;
   they know nothing about Prefect or SQL. Retries live on tasks in `flows/`; quality-gate
   decisions (regenerate, fall back, stop) live in the flow body.
6. **Providers only through Protocols.** New external API → adapter in `providers/` implementing
   a `providers/base.py` Protocol, error mapping to `RetryableProviderError` / `FatalProviderError`
   in one place, plus a `Fake*` counterpart in `providers/fake.py` so `--dry-run` still works.
7. **Repo artifacts are in English** (docs, comments, commit messages, log events).

## Layering (arrows point down only)

```
cli → flows → stages → providers (Protocols) → fal / openrouter / typesafe
                 ↓         ↓
              quality    media (ffmpeg)
                 ↓
              storage ← domain (pydantic / SQLModel)
              observability — imported by everyone, imports nothing
```

`domain/` and `observability/` must not import from other layers. `storage/repo.py` is the only
place with SQL. `media/ffmpeg.py` is the only place that spawns ffmpeg.

## Commands

```bash
uv sync                                  # + `--extra judge` for typesafe-sdk
uv run ruff check . && uv run ruff format --check .
uv run mypy                              # strict; must be clean
uv run pytest                            # unit + integration (ffmpeg, Prefect flow with fakes)
uv run hw generate '<idea with "a line"' --dry-run   # full pipeline, no network, ~10 s
uv run hw eval run --tag <tag> --dry-run # evals/cases.yaml
```

Run all four checks before claiming a change is done. `--dry-run` is the smoke test after any
change to stages, flow, storage or media.

## Definition of done for a change

- ruff, mypy, pytest green; `hw generate --dry-run` produces `final.mp4` and `report.json`.
- New behaviour has a unit test; new failure mode has a test that asserts the typed error.
- New stage/gate/provider is reflected in `docs/DATA_FLOW.md` or `docs/ARCHITECTURE.md`.
- New prompt text → bump `version:` in the prompt file's frontmatter (it lands in `report.json`).
- Nothing under `runs/` is committed.

## Conventions

- Python 3.12+ syntax, full type hints, `from __future__ import annotations` in src modules.
- pydantic models: `extra="forbid"`; config models are frozen.
- Enums are `StrEnum`; stored in SQLite as `.value` strings, converted at the repo boundary.
- Log events are dotted lowercase (`stage.start`, `verify_speech.failed`) with structured
  fields, not f-strings. Bind run context with `log_context(...)`, not by passing `run_id` around.
- Tests: `tests/unit` must not touch network or ffmpeg; `tests/integration` may use ffmpeg and
  the Prefect test harness but never real providers.
- Costs: every provider call records a `costs` row via `repo.add_cost` (estimate is fine, 0 is
  acceptable only for fakes).

## Spending real money

Real runs need `FAL_KEY` and `OPENROUTER_API_KEY` in `.env`. A 480p run is roughly $1.5–4
(Seedance 2.5 ≈ $0.22/s at 480p, $0.47/s at 720p; each retry doubles the shot cost). Keep
`video.resolution: 480p` while iterating; switch to `720p` only for final samples. Do not start
paid runs or eval suites on your own initiative — ask first.

## Where things are

| Need | Look at |
|------|---------|
| Add/replace a model | `config/pipeline.yaml`, `providers/*` |
| Change retry/fallback logic | `flows/generate_video_flow.py` (`_produce_shot`) |
| Change what counts as "verbatim" | `quality/text_norm.py` + tests in `tests/unit/test_text_norm.py` |
| Change the video prompt | `src/holywater/prompts/seedance_shot.md` (bump version) |
| Change the script schema | `domain/script.py` (+ invariants) and `prompts/scriptwriter.md` |
| Add a metric to the report | `domain/report.py`, `stages/report.py`, `storage/repo.py::tag_stats` |
| Simulate a failure in tests | `providers/fake.py` (`fail_times`, `speech_mutator`, `queue`) |
