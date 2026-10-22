# Architecture Decision Records

Format: context → decision → alternatives → risks. Kept short — it's a PoC.

---

## ADR-001. Prefect as the orchestrator

**Context.** We need retries, per-stage state, logs and the ability to resume and inspect runs — within 8–12 hours of work.

**Decision.** Prefect 3.x locally (ephemeral API). `@flow` for the pipeline, `@task` per stage with
`retries`, `retry_delay_seconds=exponential_backoff`, `retry_condition_fn` (retry only `RetryableError`).

**Alternatives.**
- *LangGraph* — strong for agent↔tool loops, but our graph is linear with gates; the graph abstraction doesn't pay off.
- *Temporal* — the "grown-up" option (durable execution), but heavy to set up for a PoC.
- *Plain asyncio + hand-rolled state machine* — fewer dependencies, but we'd write retries/state/logs ourselves, i.e. exactly what we take Prefect for.

**Risks.** Prefect brings a local server + SQLite; cold start takes seconds. The Prefect logger and structlog must be deliberately merged into one stream (see ARCHITECTURE §6).

---

## ADR-002. Audio strategy: native + verification + fallback to TTS

**Context.** R4 — the line must be verbatim. Seedance 2.5 generates speech jointly with the picture (lip-sync for lines in quotes), but
it is a generative model: it may paraphrase, swallow a word, "forget" part of a long line, or speak the wrong language.
The vendor gives no verbatim guarantee and there are no dialogue benchmarks.

**Decision.** Configurable `audio.strategy`:

| Strategy | How | Verbatim guarantee | Cost | Wow factor |
|----------|-----|--------------------|------|------------|
| `native` | Seedance `generate_audio=true`, line in quotes in the prompt | **no** → STT gate required | 1× video | high (lip-sync out of the box) |
| `tts_overlay` | TTS (deterministic text) + Seedance without dialogue (`generate_audio=true`, prompt "no dialogue") + mux; optional lip-sync | **yes** (by construction) | video + TTS (~$0.0001–0.001/char) + lipsync ($0.7/min) | medium |

Pipeline: `native` → STT → WER ≤ `max_wer` ✔ done; otherwise retry (up to `native_attempts`, with a stricter
prompt) → after exhaustion an automatic fallback to `tts_overlay` (no human involved). Every transition is logged and recorded in the DB (`runs`, `shot_attempts`).

**Why this and not TTS from the start.** Native produces the better product (lip-sync, emotion, synchronized SFX), and this is
exactly what Holywater does. But the PoC must not fail R4, so the gate + fallback are mandatory. If in practice native
fails WER in more than ~50% of cases — we flip the default to `tts_overlay` with a one-line config change.

**Risks.**
- Cost: Seedance 2.5 at 720p ≈ $0.47/s; 2 shots × 8 s ≈ $7.5 per attempt, a retry doubles it. Mitigation: 480p ($0.22/s) for dev iterations, attempt limits in config, cost tracking in the report.
- Language: Ukrainian support in native speech is unconfirmed → test on the very first run; `tts_overlay` with ElevenLabs/MiniMax supports Ukrainian.
- STT itself makes mistakes → text normalization (case, punctuation, numbers) before WER, threshold `max_wer: 0.1` instead of 0, plus a manual spot-check.

---

## ADR-003. Seedance 2.5 vs 2.0 vs others

**Decision.** Seedance 2.5 (fal) as the default video model.

| | Seedance 2.5 | Seedance 2.0 | Veo 3 / Kling |
|-|--------------|--------------|---------------|
| Single-pass duration | 4–30 s | 4–15 s | ~8 s |
| Native audio / lip-sync | yes | yes | yes / partial |
| 1080p | no (480/720) | yes | yes |
| Price 720p/s | ~$0.47 | ~$0.30 (fast $0.24) | different pricing model |

Why 2.5: 8–30 s in one shot without cutting in the middle of a line; native 9:16; a single vendor (fal) for video + TTS + STT + lipsync.
Why not 2.0: the 15 s limit forces a cut for 20–30 s clips, and the line cannot be cut. But 2.0 stays in config
as the cheaper option (`video.model`), and the code must not depend on a specific model.

**Answer to "is Seedance 2.5 enough for the PoC"**: for picture + sound — yes, with margin.
For guaranteeing R4 — **not on its own**; only together with the STT gate and the fallback (ADR-002).

---

## ADR-004. OpenRouter for the LLM

**Decision.** OpenAI-compatible client on `https://openrouter.ai/api/v1`, default model `anthropic/claude-sonnet-5.5`
($2 / $10 per M input/output tokens vs $3 / $15 for Sonnet 4.6; the newer tokenizer yields ~30% more tokens for the
same text, but our prompts are hundreds of tokens — the difference is cents). `response_format={"type":"json_schema", strict: true}`,
`provider.require_parameters=true` so the router never sends the request to an endpoint without structured outputs.
The JSON schema is generated from pydantic (`Model.model_json_schema()`), the response is validated with `model_validate_json`.

**Alternatives.** Direct Anthropic/OpenAI SDK — fewer layers, but vendor lock-in. For a PoC with configurable models the router is more convenient.

**Risks.** Not all models/providers on the router hold the schema equally strictly → pydantic validation + 1 retry
with the validation error message (a "repair" loop).

---

## ADR-005. The line comes from the input, not from the LLM

**Context.** The simplest way to guarantee the line is never paraphrased is to never let the model rewrite it.

**Decision.** The LLM returns `dialogue_line` as part of the structured script, but the code:
1. normalizes and checks that `dialogue_line` is a substring of the input text;
2. on failure — retries the LLM with explicit feedback; after N attempts — `ScriptError` (fatal, no point burning money further);
3. downstream uses **the original** fragment from the input.

If the input has no quotes — the LLM picks a phrase from the text as the line (still a substring of the input).

---

## ADR-006. Domain state in SQLite, execution state in Prefect

**Context.** The question: "why Prefect if we manage state ourselves?" The answer — these are two different states:

| | Prefect (its own DB) | Our state |
|-|----------------------|-----------|
| What it stores | task/flow runs, statuses, retries, timings, logs | runs, shots, attempts, verifications (WER), strategy_used, cost, artifact paths |
| Who reads it | Prefect UI/CLI, retry mechanism | `hw eval compare`, `hw inspect`, stage idempotency, report |
| Can you ask "mean WER per model X this week" | no (not its data model) | yes |

Prefect gives us orchestration (retries with backoff, `retry_condition_fn`, parallel tasks, UI) and knows nothing about
a "shot" or "WER". Domain state is needed separately in any case; the only question is file vs DB.

**Decision.** **SQLite** via **SQLModel** (pydantic + SQLAlchemy — the same models as in `domain/`, no schema
duplication). One file `runs/holywater.db`. Media — on disk under `runs/<run_id>/...`; the DB holds paths, hashes, metadata.
`report.json` remains as an export to attach to the note, but the source of truth is the DB.

Tables: `runs`, `stages` (run_id, name, status, attempt, input_hash, error), `shots`, `shot_attempts`
(seed, path, duration, cost, provider_request_id), `speech_verifications` (expected, transcript, wer, passed),
`judge_decisions`, `costs`. No migrations — `SQLModel.metadata.create_all` (PoC).

**Why not a JSON file:** eval/compare needs aggregations over many runs; atomic stage updates out of the box;
one `SELECT` instead of parsing 50 manifests.

**Grown-up version:** Postgres, Alembic migrations, S3/GCS for media, Prefect deployments + work pools (or Temporal).

---

## ADR-007. Jev (TypeSafe AI) as a typed judge

**Context.** Jev is a "System One" model: input is state (text/JSON) + typed questions (`Noul` yes/no, `Choice`,
`Score`), output is answers with calibrated probabilities. It does not generate text, does not see video/audio, cannot
count. ~100 ms, $0.042/M input tokens (output free), Python SDK `typesafe-sdk`.
It is TypeSafe AI — not Typeframes.

**Decision.** Use it in two places where we need a *decision*, not generation, and where an LLM call would be slow/expensive:

1. **Script pre-flight** (between `write_script` and `generate_video`) — before spending $3–7 on Seedance:
   - `Noul`: "visual_prompt contains content likely to trip the video model's safety filter" → if p > 0.7 — rewrite the prompt with the LLM before generation, not after a 400 from fal.
   - `Score 1–5`: "how concrete and filmable the shot prompt is (one character, one action, one location)" → < 3 — retry the LLM with feedback.
   - `Noul`: "the line fits naturally into the `end_s - start_s` window" — in addition to the arithmetic estimate in code.
2. **WER-gate failure classification** — when the transcript ≠ the line, we must pick a retry tactic:
   `Choice{paraphrased, truncated, extra_words, wrong_language, stt_noise, silence}` → `paraphrased/extra_words` →
   retry with a stricter prompt; `truncated` → widen the line window; `wrong_language` → fall back to TTS immediately;
   `stt_noise` → re-run STT with another model, don't burn video.

**What Jev does NOT do here:** it does not check verbatim-ness (that's WER in code — deterministic and exact), does not judge the picture
(text-only), does not write prompts.

**Integration.** `providers/base.py: JudgeProvider` Protocol → `providers/typesafe_judge.py`. Optional: if
`TYPESAFE_API_KEY` is absent or `judge.enabled=false` — `NoopJudge` (pre-flight is skipped, the classifier returns
`unknown` → default tactic "retry with a stricter prompt"). All Jev decisions with probabilities are written to the DB
(`judge_decisions`) — later we can evaluate whether it actually saved money.

**Risks.** Early access; signups paused since 2026-09-22 — if there is no key, we run without it. English works best —
we pass state in English (shot prompts are in English anyway; for failure classification we pass both texts + a
transliterated diff). Vendor benchmarks without independent verification — so thresholds (0.7, 3) go to config and get calibrated on the eval set.

---

## ADR-008. Vendors are classes, capabilities are Protocols, config picks the vendor

**Context.** The first cut had one file per *(vendor, capability)* pair (`fal_video.py`, `fal_tts.py`,
`fal_stt.py`, `fal_lipsync.py`, `openrouter_llm.py`). With a single vendor that looked like
over-splitting, and adding OpenRouter's video/TTS/STT endpoints would have doubled the file count.

**Decision.**
- `providers/base.py` keeps the *capability* contracts (`LLMProvider`, `VideoProvider`, `TTSProvider`,
  `STTProvider`, `LipsyncProvider`, `JudgeProvider`). Every request carries its `model`, so a vendor
  instance is model-agnostic.
- One class per *vendor* implements all the capabilities it offers: `FalProvider` (video, TTS, STT,
  lipsync — one `SyncClient`, one submit/poll, one download, one error map) and `OpenRouterProvider`
  (LLM, video, TTS, STT — one key, one `httpx.Client`, one status→error rule).
- `providers/registry.py` resolves the `provider:` named in each config section to a vendor instance,
  built once per run, and raises `ConfigError` if the vendor lacks the capability or its key is missing.
  Switching video from fal to OpenRouter is a two-line YAML change (`provider`, `model`).
- Defaults: everything on OpenRouter (one key covers the whole pipeline); lipsync stays fal-only.

**OpenRouter media endpoints used.** `POST /videos` → poll `GET /videos/{id}` (`pending`/`in_progress`
→ `completed`/`failed`) → `GET /videos/{id}/content`; the poll response reports the real `usage.cost`.
`POST /audio/speech` (mp3 bytes). `POST /audio/transcriptions` with base64 WAV; the pipeline extracts
a mono 16 kHz WAV with ffmpeg first so fal and OpenRouter get identical STT input.

**Alternatives.** Keep per-capability adapters and compose them — more files, duplicated clients and
error mapping. A plugin/entry-point system — overkill for two vendors.

**Risks.** A vendor class grows with each capability; mitigated by keeping shared internals
(`_call`/`_download`/`_map_*`) private and capability methods thin. Cost figures differ in nature:
OpenRouter reports actual cost for video/STT/LLM, fal and TTS are table estimates — the report
records both the same way (`costs` rows), so compare tags with that in mind.
