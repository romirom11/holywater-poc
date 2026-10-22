# HOLYWATER PoC — Specification

Status: **approved 2026-10-21** (before coding), updated by ADR-008 (2026-10-22: OpenRouter is the
default vendor for every capability) and kept in sync with the code since.

## 1. Goal

A minimal autonomous agent: input — an idea (1–3 sentences of text), output — a single video file,
8–30 s, vertical (9:16), with sound, in which the spoken line is heard **verbatim** as written in the input text.

One command → one file:

```bash
uv run hw generate 'Night, rain, a taxi driver looks in the mirror and says: "Are you sure this is the address?"'
# -> runs/<run_id>/final.mp4 (path printed on stdout), plus report.json next to it
```

## 2. Hard requirements (from the task)

| # | Requirement | How it is verified |
|---|-------------|--------------------|
| R1 | Input: idea, 1–3 sentences | CLI argument, or stdin (`hw generate -`, or no argument with input piped in). 10–1000 characters, else a one-line error (exit 1); more than 3 sentences is a WARNING |
| R2 | Output: video 8–30 s | `DurationGate` on the final (ffprobe): exactly `quality.duration_s` (8..30). Longer is trimmed unless that would cut the line; shorter extends one shot once (the shortest without a verified native line) |
| R3 | 1–2 shots concatenated into a final file | script invariant (1–2 shots) + assemble stage (ffmpeg concat) |
| R4 | Sound is mandatory; the line is spoken **word for word** | the line is taken from the input text (one complete quoted fragment; a verbatim fragment when nothing is quoted), never from the LLM; STT + WER gate on the clip (native) or on the final (TTS overlay) — `max_wer` 0.10 after normalization, i.e. exact below 10 words, one word may differ at 10–19 (ADR-009); `MediaIntegrityGate` rejects a final without audible audio |
| R5 | One command / one call | `hw generate` |
| R6 | Autonomy: retries, fallbacks, state, logs | Prefect task retries + quality gates and tactics in the flow + SQLite state + per-run JSONL log |

## 3. Non-functional requirements (ours)

- **Clean, modular code**: layers `domain / config / providers / stages / flows / media / quality / observability / storage`; every capability behind a Protocol; swapping a vendor or model = a config change.
- **DRY**: one client and one error map per vendor (`providers/fal.py`, `providers/openrouter.py`); one ffmpeg wrapper; capabilities resolved by `providers/registry.py` from config.
- **Configurable**: models, generation parameters, quality thresholds, retry policies and timeouts live in `config/pipeline.yaml`. Secrets only in `.env`.
- **Observable**: no error or warning is lost — everything goes to the per-run JSONL log, and errors and fallbacks to the DB (`stages.error_json`, `runs.fallback_reason`, `speech_verifications`, `judge_decisions`).
- **Idempotent**: re-running with the same `run_id` reuses the script when its input hash matches and every clip whose prompt, strategy, resolution, length and file still match; no accepted or pending (paid, not yet gated) clip is paid for again. Not covered: a job interrupted mid-poll, and the LLM/STT calls of a stage that had not finished (DATA_FLOW §6 "Resume").
- **Measurable**: every run ends with `report.json` (technical metrics, WER, strategy, cost split into vendor-reported and estimated).

## 4. PoC boundaries (what we deliberately do NOT do)

- No HITL pauses (the points are documented; enabling one stops the run before any paid call).
- No Postgres/S3 — local SQLite + media on disk.
- No Prefect Cloud / deployments — local ephemeral server.
- No complex editing (transitions, titles, music) — concat + loudness normalization.
- No cross-shot character consistency via reference images (documented as a next step).
- Shots are generated one after another, not in parallel.

## 5. Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Language | Python 3.12/3.13, `uv` | fast, AI SDK ecosystem |
| Schemas / validation | pydantic v2, pydantic-settings | LLM structured output, config, artifacts |
| Orchestration | Prefect 3.x | retries, state, logs out of the box; minimal boilerplate for a PoC |
| LLM | OpenRouter, `anthropic/claude-sonnet-5.5` (`response_format: json_schema`, strict) | one entry point to any model, easy to swap in config |
| Provider switch | `provider:` per capability (`openrouter` \| `fal`), ADR-008 | OpenRouter alone covers LLM + video + TTS + STT with one key; fal adds lipsync |
| Video | Seedance 2.5: OpenRouter `bytedance/seedance-2.5` (default); fal `bytedance/seedance-2.5/text-to-video` (switchable) | 4–30 s in a single pass, native audio + lip-sync, 9:16 |
| TTS (fallback / overlay) | ElevenLabs v3: OpenRouter `elevenlabs/eleven-v3` (default); fal ElevenLabs / MiniMax (switchable) | deterministic verbatim text |
| STT (verification) | Whisper large-v3: OpenRouter `openai/whisper-large-v3` (default); fal `fal-ai/whisper` (switchable) | R4 check |
| Lip-sync (optional, off) | fal `fal-ai/sync-lipsync/v2` (fal only) | for the TTS-overlay mode |
| Typed judge (optional) | TypeSafe Jev (`typesafe-sdk`, `uv sync --extra judge`) | cheap (~$0.042 per M input tokens) typed decisions: script pre-flight, failure classification |
| State | SQLite via SQLModel | queries for eval/compare, idempotency, pydantic-compatible models |
| Media | ffmpeg/ffprobe via a thin wrapper | normalize, mux, concat, loudnorm, probe |
| CLI | typer | one command |
| Logs | structlog → per-run JSONL + stderr console, `logging.captureWarnings` | machine-readable logs; stdout keeps only command results |
| Tests / quality | pytest, ruff, mypy (strict) | guardrails for the agents that write the code |

Alternatives, risks and rationale — in `docs/DECISIONS.md`.

## 6. Key decisions (approved 2026-10-21; vendor rows updated by ADR-008)

1. **Audio strategy** (ADR-002): `native` (Seedance generates speech) with STT verification and an automatic fallback to `tts_overlay`. Config default — `native`.
2. **The line is a first-class input** (ADR-005): the input text contains the line in quotes; the LLM only *extracts* it, the code checks that it is one complete quoted fragment (or, without quotes, a verbatim fragment of the input), and downstream uses **the original characters from the input**.
3. **Resolution**: 480p; `video.resolution` in config (480p/720p), switchable without code changes. The samples stay at 480p to keep the cost down; 720p is a one-line config change.
4. **Shot structure**: 1–2 shots; the shot with the line is generated by a single Seedance call (no cut in the middle of a sentence).
5. **State** (ADR-006): domain state in **SQLite** (SQLModel), media on disk; Prefect keeps only execution state.
6. **LLM**: `anthropic/claude-sonnet-5.5` via OpenRouter ($2/$10 per M tokens vs $3/$15 for 4.6).
7. **Jev (TypeSafe AI)** (ADR-007): typed judge for script pre-flight and WER-gate failure classification. Optional provider behind a Protocol.
