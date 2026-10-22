# Architecture

## 1. Principles

1. **A stage is a pure function** `(inputs, config, ctx) -> artifact`. No stage knows about Prefect; the Prefect wrapper lives in `flows/`.
2. **Capabilities are Protocols, vendors are classes.** `LLMProvider`, `VideoProvider`, `TTSProvider`, `STTProvider`, `LipsyncProvider`, `JudgeProvider` are the contracts; one vendor class (`FalProvider`, `OpenRouterProvider`) implements several of them. Stages depend on a capability; `providers/registry.py` picks the vendor named in config (ADR-008).
3. **Anything that can change without changing logic lives in config.** Models, parameters, thresholds, retries.
4. **Errors are typed**; each one is either retried, falls back, or fails the run — but it is always logged with context.
5. **Artifacts are files on disk, metadata is in SQLite**; no "magic" paths in code — everything goes through `RunPaths`.

## 2. Repository layout

```
holywater-poc/
├── pyproject.toml              # uv, deps, ruff, mypy, pytest
├── config/
│   └── pipeline.yaml           # models, parameters, thresholds (see §4)
├── .env.example                # FAL_KEY, OPENROUTER_API_KEY, TYPESAFE_API_KEY
├── src/holywater/
│   ├── cli.py                  # typer: generate, eval, inspect
│   ├── config/
│   │   ├── settings.py         # pydantic-settings: env + yaml → PipelineConfig
│   │   └── schema.py           # pydantic config models (ModelSpec, AudioConfig, QualityConfig...)
│   ├── domain/                 # data contracts (pydantic / SQLModel), no I/O
│   │   ├── idea.py             # IdeaInput
│   │   ├── script.py           # Script, Shot, Dialogue (pydantic — LLM output)
│   │   ├── artifacts.py        # AudioArtifact, VideoArtifact, FinalVideo, ...
│   │   ├── enums.py            # StageName, StageStatus, AudioStrategy, FailureKind
│   │   └── report.py           # RunReport, StageMetrics, CostEstimate (exported from the DB)
│   ├── providers/
│   │   ├── base.py             # capability Protocols + request/result types
│   │   ├── registry.py         # provider name (config) -> vendor instance, capability-checked
│   │   ├── fal.py              # FalProvider: video + TTS + STT + lipsync (one client, one error map)
│   │   ├── openrouter.py       # OpenRouterProvider: LLM + video + TTS + STT (one key)
│   │   ├── typesafe.py         # TypeSafeJudge (Jev); NoopJudge when disabled
│   │   └── fake.py             # in-process fakes for --dry-run and tests
│   ├── stages/                 # pure stages (see DATA_FLOW)
│   │   ├── write_script.py
│   │   ├── preflight_script.py # Jev gate before paid generation
│   │   ├── generate_audio.py
│   │   ├── generate_video.py
│   │   ├── verify_speech.py    # STT + WER + (Jev) failure classification
│   │   └── assemble.py
│   ├── flows/
│   │   └── generate_video_flow.py  # Prefect flow + tasks, retries, fallback logic
│   ├── media/
│   │   └── ffmpeg.py           # probe, concat, mux, mix, loudnorm — subprocess wrapper
│   ├── quality/
│   │   ├── gates.py            # QualityGate protocol + DurationGate, SpeechGate, MediaIntegrityGate
│   │   └── text_norm.py        # text normalization + WER
│   ├── storage/
│   │   ├── paths.py            # RunPaths: runs/<run_id>/...
│   │   ├── db.py               # engine, session factory, create_all
│   │   ├── models.py           # SQLModel tables: Run, Stage, Shot, ShotAttempt, SpeechVerification, JudgeDecision, Cost
│   │   └── repo.py             # RunRepository: all queries in one place (no raw SQL in stages)
│   ├── observability/
│   │   ├── logging.py          # structlog config, captureWarnings, Prefect bridge
│   │   └── errors.py           # PipelineError hierarchy
│   └── prompts/                # prompts as files (.md/.j2), versioned
│       └── scriptwriter.md
├── evals/
│   ├── cases.yaml              # test set of ideas
│   └── baseline/               # report.json of reference runs
├── tests/
│   ├── unit/                   # domain, text_norm, gates, config, storage, ffmpeg (with fixtures)
│   └── integration/            # providers with VCR/mocks; e2e with --dry-run
├── runs/                       # run artifacts + holywater.db (gitignored)
└── docs/
```

## 3. Layers and dependencies

```
cli → flows → stages → providers (capability Protocols) → registry → fal / openrouter / typesafe
                 ↓          ↓
              quality     media (ffmpeg)
                 ↓
              storage ← domain (pydantic / SQLModel)
              observability — cross-cutting
```

Rule: arrows point down only. `domain` and `observability` import nothing from other layers.

## 4. Configuration

`config/pipeline.yaml` (secrets in `.env`). Every capability names its **vendor** (`provider:`) and a
vendor-specific **model** id; vendor-wide connection settings sit under `providers:`.

```yaml
version: 1

llm:
  provider: openrouter                       # fal has no LLM capability
  model: "anthropic/claude-sonnet-5.5"
  temperature: null                          # null = not sent (Claude 5.x endpoints reject it)
  max_attempts: 3                            # retries on invalid schema / line-substring violation

judge:                                       # Jev (TypeSafe AI); disabled → NoopJudge
  enabled: true
  model: "jev-latest"
  preflight: {enabled: true, safety_risk_threshold: 0.7, min_filmability_score: 3}
  failure_classifier: {enabled: true}

video:
  provider: openrouter                       # openrouter: bytedance/seedance-2.5 | fal: bytedance/seedance-2.5/text-to-video
  model: "bytedance/seedance-2.5"
  resolution: "480p"                         # 480p dev / 720p final — the only place this is set
  aspect_ratio: "9:16"
  generate_audio: true
  max_attempts: 2

audio:
  strategy: native                           # native | tts_overlay
  native_attempts: 2                         # native tries per shot before falling back to TTS
  tts: {provider: openrouter, model: "elevenlabs/eleven-v3", voice: "sarah"}
  stt: {provider: openrouter, model: "openai/whisper-large-v3", language: auto}
  lipsync: {provider: fal, enabled: false, model: "fal-ai/sync-lipsync/v2"}

quality:
  max_wer: 0.10
  duration_s: {min: 8, max: 30}
  shots: {min: 1, max: 2}

retries:
  provider: {attempts: 3, backoff_s: [5, 15, 45]}
  timeouts_s: {llm: 60, video: 600, tts: 60, stt: 120, judge: 10}

providers:
  openrouter: {base_url: "https://openrouter.ai/api/v1", video_poll_interval_s: 10}
  fal: {lipsync_sync_mode: cut_off}

storage:
  runs_dir: "runs"
  db_path: "runs/holywater.db"
  final_name: "final.mp4"
```

`.env`: `OPENROUTER_API_KEY` (required with the defaults), `FAL_KEY` (only if any capability names
`fal`), `TYPESAFE_API_KEY` (optional).

Config validation — pydantic at startup; an invalid config, an unknown vendor, a vendor without the
requested capability, or a missing key for a vendor in use = `ConfigError` before any paid call.

## 5. Error handling

Hierarchy (`observability/errors.py`):

```
PipelineError(stage, run_id, context: dict)
├── ConfigError            fatal, before start
├── ScriptError            fatal after LLM retries are exhausted (line not a substring / schema)
├── ProviderError          base for external APIs
│   ├── RetryableProviderError   429, 5xx, timeout, network  → Prefect retry with backoff
│   └── FatalProviderError       401, 400 (bad input), content policy → no retry
├── QualityGateError(gate, measured, threshold)   → regenerate the stage / switch strategy
└── MediaError             ffmpeg non-zero exit, corrupt file → 1 retry, then fatal
```

Rules:
- `retry_condition_fn` in Prefect tasks retries **only** `RetryableProviderError` and `MediaError`.
- `QualityGateError` is handled in the flow (not in the task): the decision "regenerate the shot / switch strategy / stop" is orchestration logic.
- No `except Exception: pass`. Every `except` either logs and re-raises a typed error, or logs and returns an explicit fallback result recorded in the DB.
- Each vendor maps its errors to the taxonomy in one place (`fal.py`, `openrouter.py`), using the shared `RETRYABLE_HTTP_STATUS` rule.
- On a fatal error `runs.status = failed`, `stages.error = {...}`, and `report.json` is still exported (partial metrics are valuable).
- Jev unavailable (no key, 429, timeout) is a `WARNING`, not an error: `NoopJudge` behavior, the `unknown` decision is written to the DB.

## 6. Logging

- **structlog** → JSON lines; processors: timestamp, level, `run_id`, `stage`, `attempt`, `provider`, `model`, `request_id`, `duration_ms`, `cost_usd`.
- Sinks: stdout (human-readable in a TTY, JSON otherwise) + `runs/<run_id>/logs.jsonl`.
- `logging.captureWarnings(True)` + `warnings.simplefilter("always")` → no `DeprecationWarning`/`UserWarning` is ever lost.
- The Prefect logger is attached to the same stdlib root → the Prefect UI sees our logs, our file sees Prefect's.
- Every provider call is logged **before** (payload without secrets, with size) and **after** (status, latency, request_id, result URL, cost estimate).
- Levels: `DEBUG` payloads, `INFO` stage transitions, `WARNING` retries/fallbacks/quality degradation, `ERROR` fatal with `exc_info`.

## 7. State and idempotency

Two states, two owners (ADR-006):
- **Prefect** — execution state: flow/task runs, retries, timings, logs. We do not duplicate it.
- **SQLite (`storage/`)** — domain state. Schema:

| Table | Key fields |
|-------|------------|
| `runs` | `id`, `input_text`, `config_hash`, `git_sha`, `status`, `audio_strategy_used`, `fallback_reason`, `started_at`, `finished_at`, `cost_usd_total`, `final_path` |
| `stages` | `run_id`, `name`, `status (pending\|running\|done\|failed\|skipped)`, `attempt`, `input_hash`, `started_at`, `finished_at`, `error_type`, `error_json` |
| `shots` | `run_id`, `shot_id`, `planned_duration_s`, `has_dialogue`, `final_attempt_id` |
| `shot_attempts` | `shot_id`, `attempt`, `strategy`, `prompt_version`, `prompt_text`, `seed`, `provider_request_id`, `path`, `duration_s`, `resolution`, `cost_usd`, `status` |
| `speech_verifications` | `shot_attempt_id`, `expected`, `transcript`, `wer`, `passed`, `stt_model`, `failure_kind` |
| `judge_decisions` | `run_id`, `stage`, `question`, `answer`, `probabilities_json`, `model_version`, `latency_ms` |
| `costs` | `run_id`, `stage`, `provider`, `model`, `units`, `unit_price`, `cost_usd` |

- All writes go through `RunRepository`; stages know nothing about SQL. Each stage status change is its own transaction.
- Media — on disk (`runs/<run_id>/video/...`); the DB holds only paths + size + sha256.
- `hw generate --run-id X` on an existing run: stages with `done` and a matching `input_hash` are skipped; shots with a valid `final_attempt_id` are not regenerated.
- `hw eval compare A B` is a `SELECT ... GROUP BY tag` over these tables, no file parsing.

## 8. Guardrails for agent-driven development

Agents write the code; the harness sets the checks:
- `ruff` (lint+format), `mypy --strict` on `src/`, `pytest` — mandatory in pre-commit and CI.
- `hw generate --dry-run` runs the whole flow with mock providers (no money) — the smoke test an agent runs after every change.
- Providers have `FakeXProvider` implementations in `tests/` for deterministic e2e.
- Every stage has a unit test for the happy path + for every error type it can raise.
- Prompts are files; changing a prompt = bumping the version in the name/frontmatter → it lands in the report for eval.
