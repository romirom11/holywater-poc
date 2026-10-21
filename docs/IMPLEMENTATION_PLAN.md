# Implementation plan (for agent-driven development)

Every step has a **Definition of Done** — a check the agent runs itself. Order is bottom-up by layer,
so each step is testable without paid calls until the very end.

| # | Step | DoD |
|---|------|-----|
| 0 | Skeleton: `pyproject`, `uv`, ruff/mypy/pytest, `.env.example`, `.gitignore` (`runs/`, `.env`) | `uv run ruff check && uv run mypy && uv run pytest` green on the empty project |
| 1 | `observability/`: error taxonomy, structlog + captureWarnings + file sink | unit: a `warnings.warn` shows up in the JSONL; an error carries `run_id/stage` |
| 2 | `domain/`: all pydantic models + Script invariants | unit per invariant (valid/invalid) |
| 3 | `config/`: yaml + env → `PipelineConfig`, `ConfigError` on invalid | unit: sample config loads; a broken one fails before any I/O |
| 4 | `storage/`: `RunPaths`, SQLModel tables, `RunRepository`, `create_all` | unit on in-memory SQLite: start/finish stage, idempotency by `input_hash`, aggregation for compare |
| 5 | `quality/text_norm.py` + WER, `gates.py` | unit: table of normalization cases (punctuation, case, numbers, Ukrainian apostrophe) |
| 6 | `media/ffmpeg.py`: probe, concat, mux, loudnorm | integration with synthetic clips (`ffmpeg -f lavfi`) |
| 7 | `providers/base.py` (incl. `JudgeProvider`) + Fake providers + `NoopJudge` | — |
| 8 | `stages/*` on Fake providers | unit happy path + every stage error |
| 9 | `flows/` Prefect: retries, native→tts_overlay fallback, shot regeneration | e2e `--dry-run`: scenarios "all ok", "WER fail → retry → fallback", "fal 500 → backoff" |
| 10 | Real providers: `fal_base`, `fal_video`, `fal_tts`, `fal_stt`, `openrouter_llm`, `typesafe_judge` | error mapping covered by unit tests with mock responses; manual smoke at 480p; the pipeline works without `TYPESAFE_API_KEY` (NoopJudge) |
| 11 | `cli.py`: `generate`, `eval`, `inspect` | `hw generate --dry-run` → `final.mp4` from fake clips |
| 12 | `report.json` + `evals/cases.yaml` + `hw eval compare` | compare prints a table for two dry-run tags |
| 13 | Real runs: 480p dev → 720p final 2 samples | 2 videos in `samples/`, reports saved as baseline |
| 14 | README + note (Parts 1–3, 5, "what I cut") — **Part 1 is written by the human** | — |

Steps 1–9 and 11–12 spend no money. The first paid call is step 10.
