# Architecture

## 1. Principles

1. **A stage is a pure function** `(inputs, deps) -> artifact`. No stage knows about Prefect or SQL; the Prefect wrapper lives in `flows/`.
2. **Capabilities are Protocols, vendors are classes.** `LLMProvider`, `VideoProvider`, `TTSProvider`, `STTProvider`, `LipsyncProvider`, `JudgeProvider` are the contracts; one vendor class (`FalProvider`, `OpenRouterProvider`) implements several of them. Stages depend on a capability; `providers/registry.py` picks the vendor named in config (ADR-008). The optional judge is the exception: one vendor (TypeSafe Jev), built by `providers/typesafe.build_judge`, no `provider:` key.
3. **Anything that can change without changing logic lives in config.** Models, parameters, thresholds, retries, timeouts.
4. **Errors are typed**; each one is either retried, answered by a tactic, or fails the run — but it is always logged with context and recorded in the DB.
5. **Artifacts are files on disk, metadata is in SQLite**; no "magic" paths in code — run files go through `RunPaths`.

## 2. Repository layout

```
holywater-poc/
├── pyproject.toml              # uv, deps (+ optional `judge` extra), ruff, mypy --strict, pytest
├── config/pipeline.yaml        # models, thresholds, retries, timeouts (see §4)
├── .env.example                # OPENROUTER_API_KEY, FAL_KEY, TYPESAFE_API_KEY
├── CLAUDE.md                   # working agreement for coding agents
├── README.md
├── docs/                       # SPEC, ARCHITECTURE, DATA_FLOW, DECISIONS, QUALITY_METRICS,
│                               # PROVIDERS_API, IMPLEMENTATION_PLAN
├── evals/cases.yaml            # eval ideas (8 cases)
├── samples/                    # committed example videos
├── src/holywater/
│   ├── cli.py                  # typer: generate, inspect, costs, eval run, eval compare, version
│   ├── config/
│   │   ├── schema.py           # PipelineConfig + frozen sections (LLMConfig, VideoConfig, QualityConfig, ...)
│   │   └── settings.py         # load_config (yaml → PipelineConfig), Secrets (environment / .env)
│   ├── domain/                 # data contracts (pydantic), no I/O
│   │   ├── idea.py             # IdeaInput (10–1000 chars), count_sentences
│   │   ├── script.py           # Script, Shot, Dialogue + structural invariants
│   │   ├── artifacts.py        # AudioArtifact, VideoArtifact, SpeechVerification, ProviderCost, FinalVideo
│   │   ├── media.py            # MediaInfo, AudioLevels
│   │   ├── enums.py            # StageName, StageStatus, RunStatus, AudioStrategy, FailureKind,
│   │   │                       # Resolution, VoiceGender, AttemptStatus
│   │   └── report.py           # RunReport, StageMetrics, ShotMetrics, CostLine, CostSummary, ...
│   ├── providers/
│   │   ├── base.py             # capability Protocols + request/result types, NoopJudge, RETRYABLE_HTTP_STATUS
│   │   ├── registry.py         # provider name (config) → vendor instance, capability-checked; fakes in --dry-run
│   │   ├── openrouter.py       # OpenRouterProvider: LLM + video + TTS + STT (one key, one error map)
│   │   ├── fal.py              # FalProvider: video + TTS + STT + lipsync (one client, one error map)
│   │   ├── typesafe.py         # TypeSafeJudge (Jev) + build_judge (NoopJudge when unavailable)
│   │   └── fake.py             # in-process fakes with failure knobs, for --dry-run and tests
│   ├── stages/                 # pure stages (see DATA_FLOW)
│   │   ├── write_script.py     # write_script, rewrite_script (pre-flight), soften_shot (rejected prompt)
│   │   ├── preflight_script.py # Jev checks before any paid video
│   │   ├── generate_audio.py   # TTS of the line
│   │   ├── generate_video.py   # Seedance prompt rendering + one paid clip
│   │   ├── verify_speech.py    # STT + WER + failure classification; stray-speech check
│   │   ├── assemble.py         # normalize, mix, lipsync (cached), concat, loudnorm, trim, final gates
│   │   └── report.py           # report.json from the DB (reads through RunRepository)
│   ├── flows/
│   │   ├── generate_video_flow.py  # Prefect flow + tasks: retries, tactics, fallback, resume, costs
│   │   └── wiring.py           # build_deps: config + secrets → PipelineDeps (repo, ffmpeg, providers)
│   ├── media/ffmpeg.py         # FFmpeg: probe, normalize, overlay, concat, loudnorm, trim, levels
│   ├── quality/
│   │   ├── gates.py            # DurationGate, SpeechGate, MediaIntegrityGate → GateResult
│   │   ├── text_norm.py        # normalization for comparison, WER, verbatim span, quoted fragments
│   │   └── script_checks.py    # line vs idea (quote rule), line window vs speech rate
│   ├── storage/
│   │   ├── paths.py            # RunPaths: runs/<run_id>/..., run id validation
│   │   ├── db.py               # SQLite engine (WAL, NullPool), create_all + additive migration
│   │   ├── models.py           # SQLModel tables (§7)
│   │   └── repo.py             # RunRepository: every query
│   ├── evals/__init__.py       # eval runner (run_suite) + compare table (render_stats)
│   ├── observability/
│   │   ├── errors.py           # PipelineError hierarchy (§5)
│   │   └── logging.py          # structlog setup, console + per-run JSONL sinks, log_context
│   └── prompts/
│       ├── __init__.py         # loader: `version:` frontmatter + `## section` blocks, $placeholders
│       ├── scriptwriter.md     # v3: system, user, soften, rewrite
│       └── seedance_shot.md    # v2: dialogue, lead_in, lead_in_at_start, tail, no_dialogue, strict
├── tests/
│   ├── conftest.py             # env defaults: quiet Prefect logs, no telemetry
│   ├── unit/                   # no network, no ffmpeg, no Prefect server
│   └── integration/            # ffmpeg + the Prefect test harness (conftest.py); fakes only
└── runs/                       # run artifacts + holywater.db (gitignored)
```

## 3. Layers and dependencies

```
cli
 ↓
flows · evals                       Prefect flow + wiring; eval runner
 ↓
stages                              pure functions
 ↓
providers                           Protocols, registry, vendor classes, fakes
 ↓
quality · media · prompts · storage
 ↓
config
 ↓
domain · observability              import nothing from holywater
```

Rule: a module imports only from rows below its own (verified with an AST import scan of
`src/holywater`; no test enforces it yet).
Details worth knowing: `stages/report.py` is the only stage that uses `storage`, and only through
`RunRepository`; `providers/registry.py` builds vendors from `config` and gives fakes the `FFmpeg`
wrapper; `providers/fake.py` uses `quality.text_norm` and `media`. `storage/` is the only place with
SQL (queries in `repo.py`, engine pragmas and the additive migration in `db.py`); `media/ffmpeg.py`
is the only place that spawns ffmpeg (the other subprocess is `git rev-parse` in `flows/wiring.py`).

## 4. Configuration

`config/pipeline.yaml` (secrets in `.env`), typed by frozen pydantic models with `extra="forbid"` in
`config/schema.py`. Every capability names its **vendor** (`provider:`) and a vendor-specific
**model** id; vendor-wide connection settings sit under `providers:`. Defaults:

| Section | Knobs (default) |
|---------|-----------------|
| `llm` | `provider: openrouter`, `model: anthropic/claude-sonnet-5.5`, `temperature: null` (not sent), `max_attempts: 3` (answers per script, and pre-flight rounds) |
| `judge` | `enabled: true`, `model: jev-latest`, `usd_per_m_input_tokens: 0.042`, `preflight {enabled, safety_risk_threshold: 0.7, min_filmability_score: 3}`, `failure_classifier {enabled}` |
| `video` | `provider: openrouter`, `model: bytedance/seedance-2.5`, `resolution: 480p` (or 720p), `aspect_ratio: 9:16`, `generate_audio: true`, `max_attempts: 2` |
| `audio` | `strategy: native` (or `tts_overlay`), `native_attempts: 2`; `tts {provider: openrouter, model: elevenlabs/eleven-v3, voice: sarah, voices {female: sarah, male: george}}`; `stt {provider: openrouter, model: openai/whisper-large-v3, language: auto}`; `lipsync {provider: fal, enabled: false, model: fal-ai/sync-lipsync/v2}` |
| `quality` | `max_wer: 0.10`, `duration_s {min: 8, max: 30}` (must lie within 8..30), `shot_duration_tolerance_s: 1.0`, `truncated_window_widen_s: 1.0`, `speech_chars_per_s: 15.0`, `min_audio_peak_db: -50.0`, `final_check_margin_s: 0.5`, `stray_speech {action: warn, max_words: 2}` |
| `retries` | `provider {attempts: 3, backoff_s: [5, 15]}` (exactly `attempts - 1` delays); `timeouts_s {llm: 60, video: 600, tts: 60, stt: 120, judge: 10, ffmpeg: 600}` |
| `providers` | `openrouter {base_url, video_poll_interval_s: 10, download_attempts: 3, video_usd_per_s {480p: 0.1035, 720p: 0.233}}`; `fal {lipsync_sync_mode: cut_off, download_attempts: 3}` |
| `hitl` | `after_script`, `on_unverified_speech` — `enabled: false` (true stops the run before any paid call) |
| `storage` | `runs_dir: runs`, `db_path: runs/holywater.db`, `final_name: final.mp4` (relative to the working directory) |

`.env`: `OPENROUTER_API_KEY` (required with the defaults), `FAL_KEY` (only if any capability names
`fal`), `TYPESAFE_API_KEY` (optional). A blank value counts as missing. `HW_CONFIG` and
`HW_LOG_LEVEL` are read from the process environment, not from `.env`.

Config validation — pydantic at startup; an invalid config, an unknown vendor, a vendor without the
requested capability (also in `--dry-run`), or a missing key for a vendor in use = `ConfigError` before
any paid call.

## 5. Error handling

Hierarchy (`observability/errors.py`):

```
PipelineError(message, stage, run_id, context)          retryable = False unless noted
├── ConfigError              invalid config, missing key, bad run id, enabled HITL point → before any paid call
├── ScriptError              no valid script after llm.max_attempts answers / pre-flight rounds
├── ProviderError            external API failure; may carry spent_usd / spent_is_estimate (money already charged)
│   ├── RetryableProviderError   429, 5xx, timeout, network, a failed job with an infrastructure cause   retryable
│   ├── FatalProviderError       401/402/403/404, malformed response, cancelled/expired job
│   │   ├── PromptRejectedError      the vendor refused the prompt (moderation wording, 400 on video submit,
│   │   │                            failed job without an infrastructure cause, LLM refusal/content filter)
│   │   └── StructuredOutputError    billed LLM answer that is unusable (truncated, empty, not JSON, schema);
│   │                                also a ValueError, so write_script repairs it with feedback
│   └── BilledJobError           paid job accepted, result not collectable (poll deadline, downloads failing)
├── QualityGateError(gate, measured, threshold)   duration, speech, media_integrity, tts_audio, voice_fit, lipsync_duration
└── MediaError               ffmpeg/ffprobe non-zero exit or unreadable file                          retryable
```

Who handles what:
- **Inside the vendor class:** idempotent reads against the same job id — OpenRouter poll errors
  (WARNING, keep polling until `timeouts_s.video`) and downloads of a finished job
  (`providers.<vendor>.download_attempts`). Nothing that would create a new paid job (ADR-010).
- **Prefect task retries:** `retry_condition_fn` retries only errors with `retryable = True`
  (`RetryableProviderError`, `MediaError`): `retries.provider.attempts - 1` retries after the
  `backoff_s` delays. Each retry is counted in `stages.retries`.
- **The flow body:** everything else. `QualityGateError` → regenerate / widen / extend / fall back /
  stop; `PromptRejectedError` → soften that shot once (the only error that triggers softening);
  `BilledJobError` (also a paid clip ffprobe cannot read) → record its cost and an `abandoned`
  attempt, regenerate within the budget, or stop when the job endpoints answered 401/403/404;
  `StructuredOutputError` → repaired inside `write_script`. Any other error fails the run.

Rules:
- No `except Exception: pass`. Every `except` either re-raises a typed error, or logs at WARNING+ and returns an explicit fallback recorded in the DB.
- Each vendor maps its errors to the taxonomy in one place (`openrouter.py`, `fal.py`), using the shared `RETRYABLE_HTTP_STATUS` rule (`docs/PROVIDERS_API.md`).
- On a fatal error `runs.status = failed`, the failing stage gets `error_type`/`error_json`, and `report.json` is still exported (partial metrics are valuable).
- Jev unavailable (no key, no SDK, an SDK error) is a `WARNING`, not an error: the run continues without it, and the decision is still a `judge_decisions` row (`decision=unavailable`, or `heuristic` when word-count heuristics chose the failure kind).
- Untyped exceptions are bugs: the CLI prints one `error: unexpected <Type>: ...` line and writes the traceback to `runs/<id>/logs.jsonl` (`cli.unexpected_error`).

## 6. Logging

- **structlog** on top of stdlib, so third-party loggers (prefect, httpx, warnings) share the same
  processors and sinks. Processors: `merge_contextvars`, `add_logger_name`, `add_log_level`,
  `TimeStamper(iso, utc)`, `StackInfoRenderer`, `format_exc_info`. Run context (`run_id`, `stage`,
  `shot_id`, `case_id`) is bound with `log_context(...)`; numbers such as `cost_usd`, `latency_s`,
  `request_id` are fields of the individual events (`cost.recorded`, `llm.response`, ...).
- Sinks: the console on **stderr** (human-readable in a TTY, JSON lines otherwise) — stdout carries
  only command results (the final path) — and `runs/<run_id>/logs.jsonl` for the duration of a run.
- `logging.captureWarnings(True)` + `warnings.simplefilter("always")` → no `DeprecationWarning`/`UserWarning` is lost.
- Prefect: its root console handler is removed (no duplicate lines); `prefect.*` records are copied
  into `logs.jsonl` once each; the CLI sets `PREFECT_LOGGING_LEVEL=WARNING` unless you set it. Our
  `holywater.*` logs do **not** reach the Prefect UI.
- Provider calls: request metadata is `DEBUG` (absent at the default INFO; `HW_LOG_LEVEL=DEBUG`
  shows it): OpenRouter logs the model, op, status and latency, not the body; fal logs its
  (redacted) arguments. The rendered video prompt is stored in `shot_attempts.prompt_text`, the
  script in `script.json`. Results are `INFO`, from the vendor (`llm.response`, `openrouter.video.submitted`,
  `fal.done`) or from the calling stage (`generate_video.done`, `generate_audio.done`,
  `verify_speech.result`); every charge is one `cost.recorded` event.
- Levels: `DEBUG` request metadata, `INFO` stage transitions and results, `WARNING` retries/fallbacks/quality degradation, `ERROR` fatal with `exc_info`.
- Wiring-time events (`wiring.providers`, `judge.disabled`) happen before the run's file sink exists,
  so they are on the console only; `run.start` in `logs.jsonl` records which judge runs.

## 7. State and idempotency

Two states, two owners (ADR-006):
- **Prefect** — execution state: flow/task runs, retries, timings (`~/.prefect`). We do not duplicate it.
- **SQLite (`runs/holywater.db`, `storage/`)** — domain state. Schema (`storage/models.py`):

| Table | Columns |
|-------|---------|
| `runs` | `id`, `tag`, `dry_run`, `input_text`, `language`, `config_hash`, `git_sha`, `status`, `audio_strategy_requested`, `audio_strategy_used`, `fallback_reason`, `started_at`, `finished_at`, `cost_usd_total`, `final_path`, `error_type`, `error_json` |
| `stages` | `run_id`, `name`, `status` (`running` \| `done` \| `failed` \| `skipped`), `attempt` (entries), `retries`, `input_hash`, `started_at`/`finished_at` (latest entry), `total_duration_s` (all entries), `error_type`, `error_json` |
| `shots` | `run_id`, `shot_id`, `planned_duration_s`, `has_dialogue`, `final_attempt_id` |
| `shot_attempts` | `shot_pk`, `run_id`, `shot_id`, `attempt`, `strategy`, `model`, `prompt_version`, `prompt_text`, `seed`, `provider_request_id`, `path`, `duration_s`, `resolution`, `cost_usd`, `cost_is_estimate`, `status` (`generated` \| `failed` \| `passed` \| `unverified` \| `accepted` \| `abandoned` \| `superseded`), `reason`, `size_bytes`, `sha256`, `created_at` |
| `speech_verifications` | `shot_attempt_id`, `run_id`, `expected` (`""` = stray-speech check), `transcript`, `wer`, `passed`, `verified`, `stt_model`, `failure_kind`, `created_at` |
| `judge_decisions` | `run_id`, `stage`, `shot_id`, `question`, `answer`, `decision` (`pass` \| `rewrite` \| `unavailable` \| `classified` \| `heuristic`), `probabilities_json`, `model_version`, `latency_ms`, `created_at` |
| `costs` | `run_id`, `stage`, `provider`, `model`, `units`, `unit_price`, `cost_usd`, `is_estimate`, `shot_id`, `attempt`, `created_at` — one row per billed call (LLM answers usable or not, video jobs including failed/abandoned ones that carry a charge, STT, TTS, each Jev call, lipsync); `repo.cost_breakdown()` aggregates by provider/model/stage/cycle |

- All writes go through `RunRepository`; stages know nothing about SQL. Each write is its own transaction. File DBs use `NullPool` (one connection per session).
- Enums are stored as their `.value` strings and converted at the repo boundary (`mark_attempt` takes `AttemptStatus`).
- Media — on disk (`runs/<run_id>/video/...`); the DB holds paths, and for shot clips `size_bytes` + `sha256` (checked by `RunRepository.attempt_file_ok` before a clip is reused).
- The schema evolves with an additive migration at start-up (`db.init_db`: `ALTER TABLE ... ADD COLUMN` with the model default). A column that cannot be added (NOT NULL without default) is a `ConfigError`: move the old DB aside. When `runs.dry_run` was added, runs whose every cost row names a `fake/` model were marked as dry runs.
- `hw generate --run-id X` on an existing run resumes it (DATA_FLOW §6 "Resume"): the script is reused when its input hash matches, accepted clips are reused only when their prompt, strategy, resolution, length and file still match (a `video.model` change alone does not regenerate them; `report.models.video` names the model of the clips used), a paid clip that was never gated is gated instead of paid again. A different idea, or a dry/real mismatch, is a `ConfigError`.
- `hw eval compare A B` aggregates per tag in `RunRepository.tag_stats` (bulk `SELECT`s, aggregation in Python), no file parsing. A tag holds either dry runs or real runs, never both.

## 8. Guardrails for agent-driven development

Agents write the code; these checks keep them honest (`CLAUDE.md` is the working agreement):
- `uv run ruff check . && uv run ruff format --check .`, `uv run mypy` (strict, `src/`), `uv run pytest`
  — run locally before a change is called done (`CLAUDE.md` "Definition of done"). There is no CI or
  pre-commit hook yet; wiring these four commands into CI is the obvious next step.
- `uv run hw generate '<idea>' --dry-run` runs the whole flow with the in-process fakes from
  `src/holywater/providers/fake.py` (no network, no cost) and writes `final.mp4` + `report.json` —
  the smoke test after any change to stages, flow, storage or media. `hw eval run --tag <t> --dry-run`
  does the same for the 8 eval cases.
- The fakes have failure knobs (rejected prompts, billed jobs, mute clips, duration offsets, stray
  speech, STT outages and fatal errors, invalid LLM output, slow TTS, ...), so the DATA_FLOW §6
  tactics are exercised by flow-level tests (`tests/integration/test_flow_failure_paths.py`,
  `test_flow_dry_run.py`) and the assemble gates by `tests/integration/test_assemble.py`.
- Vendor request bodies are pinned with `==` in `tests/unit/test_provider_*.py` against the shapes in
  `docs/PROVIDERS_API.md`, which are taken from the vendors' OpenAPI schemas.
- Every stage has unit tests, except `assemble` (tested against real ffmpeg in `tests/integration/test_assemble.py`).
- Prompts are files; changing a prompt = bumping `version:` in its frontmatter → it lands in `report.json` (`prompt_versions`).
