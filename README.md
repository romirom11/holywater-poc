# HOLYWATER PoC — idea → vertical video with a verbatim spoken line

One command turns a 1–3 sentence idea into an 8–30 s vertical (9:16) video in which the quoted
line is spoken **word for word**. Script by an LLM, picture + speech by Seedance 2.5, verbatim
check by STT + WER, automatic fallback to TTS when the video model paraphrases.

```
idea ─► write_script (LLM) ─► preflight (Jev) ─► generate_video (Seedance, per shot)
                                                      │
                                                      ▼
                                   verify_speech (Whisper + WER gate)
                                     ✔ pass            ✘ fail → retry stricter → fallback to TTS overlay
                                                      │
                                                      ▼
                                   assemble (ffmpeg: normalize, concat, loudnorm) ─► final.mp4 + report.json
```

Design docs: [`docs/SPEC.md`](docs/SPEC.md), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
[`docs/DATA_FLOW.md`](docs/DATA_FLOW.md), [`docs/DECISIONS.md`](docs/DECISIONS.md),
[`docs/QUALITY_METRICS.md`](docs/QUALITY_METRICS.md), [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).

## Requirements

- Python 3.12/3.13 and [`uv`](https://docs.astral.sh/uv/)
- `ffmpeg` + `ffprobe` on `PATH` (`brew install ffmpeg`)
- API keys (see below); none are needed for `--dry-run`

## Setup

```bash
uv sync                      # add `--extra judge` to enable the optional TypeSafe Jev judge
cp .env.example .env         # then fill in the keys
```

| Key | Used for | Required |
|-----|----------|----------|
| `FAL_KEY` | Seedance video, ElevenLabs TTS, Whisper STT, lipsync | yes |
| `OPENROUTER_API_KEY` | script writing (`anthropic/claude-sonnet-5.5` by default) | yes |
| `TYPESAFE_API_KEY` | Jev typed judge (script pre-flight, failure classification) | no — runs without it |

## Run

```bash
# the real thing (spends money: ~$1.5–4 per run at 480p, 2–3× at 720p)
uv run hw generate 'Rainy night. A taxi driver asks: "Are you sure this is the address?"'

# full pipeline with in-process fake providers: no network, no cost, ~10 s
uv run hw generate 'Rainy night. A taxi driver asks: "Are you sure this is the address?"' --dry-run
```

Output: `runs/<run_id>/final.mp4`, plus `script.json`, every shot attempt under `video/`,
`logs.jsonl` and `report.json` (technical metrics, WER, cost, strategy used). Domain state
(runs, attempts, verifications, costs) lives in `runs/holywater.db` (SQLite).

Put the line in quotes. The LLM may only *extract* it; the code checks it is a verbatim substring
of your input and uses your original characters downstream.

Everything is configured in [`config/pipeline.yaml`](config/pipeline.yaml): models,
resolution (`480p` for iteration, `720p` for finals), audio strategy (`native` | `tts_overlay`),
WER threshold, retries, timeouts.

## Evals

```bash
uv run hw eval run --tag v0.1 [--dry-run]      # runs evals/cases.yaml under a tag
uv run hw eval compare v0.1 v0.2               # success rate, fallback rate, WER, cost per tag
uv run hw inspect <run_id>                     # prints report.json
```

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest            # unit + integration (ffmpeg, Prefect flow with fakes); no network
```

Layout: `domain/` (pydantic contracts) → `providers/` (adapters behind Protocols, incl. fakes)
→ `stages/` (pure functions) → `flows/` (Prefect orchestration: retries, fallbacks, state)
→ `storage/` (SQLite via SQLModel) · `media/` (ffmpeg) · `quality/` (gates, WER) ·
`observability/` (typed errors, structlog). Prompts are versioned files in `src/holywater/prompts/`.
