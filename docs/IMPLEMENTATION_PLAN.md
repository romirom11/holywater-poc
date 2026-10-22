# Implementation plan (for agent-driven development)

Every step has a **Definition of Done** — a check the agent runs itself. Order is bottom-up by layer,
so each step is testable without paid calls until the very end.

| # | Step | DoD | Status |
|---|------|-----|--------|
| 0 | Skeleton: `pyproject`, `uv`, ruff/mypy/pytest, `.env.example`, `.gitignore` (`runs/`, `.env`) | `uv run ruff check && uv run mypy && uv run pytest` green on the empty project | done |
| 1 | `observability/`: error taxonomy, structlog + captureWarnings + file sink | unit: a `warnings.warn` shows up in the JSONL; an error carries `run_id/stage` | done |
| 2 | `domain/`: all pydantic models + Script invariants | unit per invariant (valid/invalid) | done |
| 3 | `config/`: yaml + env → `PipelineConfig`, `ConfigError` on invalid | unit: sample config loads; a broken one fails before any I/O | done |
| 4 | `storage/`: `RunPaths`, SQLModel tables, `RunRepository`, `create_all` | unit on in-memory SQLite: start/finish stage, idempotency by `input_hash`, aggregation for compare | done (plus an additive column migration) |
| 5 | `quality/text_norm.py` + WER, `gates.py` | unit: table of normalization cases (punctuation, case, numbers, Ukrainian apostrophe) | done (plus currency, ordinals, Ukrainian numerals, the quote rule) |
| 6 | `media/ffmpeg.py`: probe, concat, mux, loudnorm | integration with synthetic clips (`ffmpeg -f lavfi`) | done |
| 7 | `providers/base.py` (incl. `JudgeProvider`) + Fake providers + `NoopJudge` | — | done |
| 8 | `stages/*` on Fake providers | unit happy path + every stage error | done (`assemble` is tested against real ffmpeg in `tests/integration`) |
| 9 | `flows/` Prefect: retries, native→tts_overlay fallback, shot regeneration | e2e `--dry-run`: scenarios "all ok", "WER fail → retry → fallback", "fal 500 → backoff" | done (the DATA_FLOW §6 tactics have flow-level tests: `tests/integration/test_flow_*.py`) |
| 10 | Real vendors: `providers/fal.py`, `providers/openrouter.py`, `providers/typesafe.py` + `registry.py` | error mapping covered by unit tests with mock responses; manual smoke at 480p; the pipeline works without `TYPESAFE_API_KEY` (NoopJudge) | partially: code, error maps and request bodies are unit-tested over mock transports; real 480p runs are only the ones listed under "Samples" in the README (the first at commit `aebdfc2`) |
| 11 | `cli.py`: `generate`, `eval`, `inspect` (+ `costs`) | `hw generate --dry-run` → `final.mp4` from fake clips | done |
| 12 | `report.json` + `evals/cases.yaml` + `hw eval compare` | compare prints a table for two dry-run tags | done |
| 13 | Real runs: 480p dev → 720p final 2 samples | 2 videos in `samples/`, reports saved as baseline | partially: 480p samples only (README "Samples": the first from `aebdfc2`, the second from `f0d289e` with its `report.json` and `script.json`); 720p samples were dropped to save money |
| 14 | README for a reviewer | a reviewer can install, dry-run, run and read the results without a call | done |

Steps 1–9 and 11–12 spend no money. The first paid call is step 10.
