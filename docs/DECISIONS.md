# Architecture Decision Records

Format: context → decision → alternatives → risks. Kept short — it's a PoC.

---

## ADR-001. Prefect as the orchestrator

**Context.** We need retries, per-stage state, logs and the ability to resume and inspect runs — within 8–12 hours of work.

**Decision.** Prefect 3.x locally (ephemeral API). One `@flow` for the pipeline, one `@task` per stage
call. Every task gets `retries = retries.provider.attempts - 1` (default 2),
`retry_delay_seconds = retries.provider.backoff_s` (default `[5, 15]`, one delay per retry; the
config validator rejects any other length) and a `retry_condition_fn` that retries only errors whose
class sets `retryable = True`: `RetryableProviderError` and `MediaError`. Every other error goes to
the flow body, which decides (ADR-010, DATA_FLOW §6).

**Alternatives.**
- *LangGraph* — strong for agent↔tool loops, but our graph is linear with gates; the graph abstraction doesn't pay off.
- *Temporal* — the "grown-up" option (durable execution), but heavy to set up for a PoC.
- *Plain asyncio + hand-rolled state machine* — fewer dependencies, but we'd write retries/state/logs ourselves, i.e. exactly what we take Prefect for.

**Risks.**
- Prefect brings a local server + SQLite (`~/.prefect`); `hw generate` and `hw eval run` start an
  ephemeral server on 127.0.0.1, which costs a few seconds. There is no UI unless you run
  `uv run prefect server start`.
- The Prefect logger and structlog must be deliberately merged into one stream (ARCHITECTURE §6).
- Shots are generated one after another in the flow body, not as parallel tasks. A deliberate cut:
  attempt numbering, cost attribution and the shot-length fixes (TTS fitting, final extension)
  depend on the other shot and stay simple when sequential. The price is wall time: the real run
  spent 150 s + 160 s in `generate_video`.

---

## ADR-002. Audio strategy: native + verification + fallback to TTS

**Context.** R4 — the line must be verbatim. Seedance 2.5 generates speech jointly with the picture (lip-sync for lines in quotes), but
it is a generative model: it may paraphrase, swallow a word, "forget" part of a long line, or speak the wrong language.
The vendor gives no verbatim guarantee and there are no dialogue benchmarks.

**Decision.** Configurable `audio.strategy`:

| Strategy | How | Verbatim guarantee | Cost | Wow factor |
|----------|-----|--------------------|------|------------|
| `native` | Seedance `generate_audio=true`, line in quotes in the prompt | **no** → STT gate required | 1× video | high (lip-sync out of the box) |
| `tts_overlay` | TTS (deterministic text) + Seedance clip with "no one speaks" + mux; optional lipsync | **yes** by construction, confirmed by STT on the final | video + TTS (estimated $0.00008/char on OpenRouter, under a cent per line) + lipsync if enabled (fal, not priced in code yet) | medium: without lipsync the line is a voice-over |

Pipeline: `native` → STT → WER ≤ `max_wer` ✔ done; otherwise retry (up to `native_attempts`, with a stricter
prompt or a wider line window) → after exhaustion an automatic fallback to `tts_overlay` (no human involved). Every transition is logged and recorded in the DB (`runs.fallback_reason`, `shot_attempts.strategy`).

**Why this and not TTS from the start.** Native produces the better product (lip-sync, emotion, synchronized SFX), and this is
exactly what Holywater does. But the PoC must not fail R4, so the gate + fallback are mandatory. If in practice native
fails WER in more than ~50% of cases — we flip the default to `tts_overlay` with a one-line config change.

**Risks.**
- Cost. Seedance 2.5 via OpenRouter (the default) ≈ $0.1035 per second at 480p (measured: $0.6211 for
  a 6 s clip) and ≈ $0.233/s at 720p (from the published token price, about 2.25× the 480p rate).
  The one real run (2 × 6 s shots, 480p, no retries) cost **$1.26** in total: video $1.24, LLM
  $0.017, STT $0.00015. Every regenerated shot adds its own cost again (≈ $0.62 per 6 s shot at
  480p). On fal the same model is ≈ $0.22/s (480p) and $0.47/s (720p). Mitigation: 480p by default,
  attempt limits in config, every charge recorded (`costs`, `hw costs <run_id>`).
- Language: Ukrainian native speech is still untested against the real model (only the dry-run eval
  cases cover Ukrainian); `tts_overlay` with ElevenLabs supports Ukrainian.
- STT itself makes mistakes → text normalization (case, punctuation, numbers, currency, Ukrainian
  numerals) before WER, and a WER threshold instead of exact match: see ADR-009 for what it lets through.

---

## ADR-003. Seedance 2.5 vs 2.0 vs others

**Decision.** Seedance 2.5 as the default video model. The vendor is chosen per ADR-008: OpenRouter
`bytedance/seedance-2.5` by default, fal `bytedance/seedance-2.5/text-to-video` by switching
`video.provider` and `video.model`. (Superseded in part by ADR-008: this ADR first named fal as the
default vendor.)

| | Seedance 2.5 | Seedance 2.0 | Veo 3 / Kling |
|-|--------------|--------------|---------------|
| Single-pass duration | 4–30 s | 4–15 s | ~8 s |
| Native audio / lip-sync | yes | yes | yes / partial |
| 1080p | OpenRouter: no (480/720); fal: yes | yes | yes |
| Price 720p/s | ~$0.233 (OpenRouter), ~$0.47 (fal) | ~$0.30 (fal; fast $0.24) | different pricing model |

Why 2.5: 8–30 s in one shot without cutting in the middle of a line; native 9:16; available from
both vendors, so moving between them is a config change.
Why not 2.0: the 15 s limit forces a cut for 20–30 s clips, and the line cannot be cut. But 2.0 stays
in config as the cheaper option (`video.model`), and the code must not depend on a specific model.

**Answer to "is Seedance 2.5 enough for the PoC"**: for picture + sound — yes, with margin.
For guaranteeing R4 — **not on its own**; only together with the STT gate and the fallback (ADR-002).

---

## ADR-004. OpenRouter for the LLM

**Decision.** OpenAI-compatible client on `https://openrouter.ai/api/v1`, default model `anthropic/claude-sonnet-5.5`
($2 / $10 per M input/output tokens vs $3 / $15 for Sonnet 4.6; the newer tokenizer yields ~30% more tokens for the
same text, but our prompts are hundreds of tokens — the difference is cents). `response_format={"type":"json_schema", strict: true}`,
`provider.require_parameters=true` so the router never sends the request to an endpoint without structured outputs;
`temperature` is not sent (Claude 5.x endpoints reject it).
The JSON schema is generated from pydantic (`Script.model_json_schema()`), the response is validated with `model_validate_json`.

**Alternatives.** Direct Anthropic/OpenAI SDK — fewer layers, but vendor lock-in. For a PoC with configurable models the router is more convenient.

**Risks.** Not all models/providers on the router hold the schema equally strictly → pydantic
validation and a repair loop: up to `llm.max_attempts` (3) answers, each rejected one fed back with
its reasons (schema error, truncated answer, line check, window check). Every billed answer is
recorded, usable or not. The exact request body and the handling of `finish_reason` / refusals are in
`docs/PROVIDERS_API.md`.

---

## ADR-005. The line comes from the input, not from the LLM

**Context.** The simplest way to guarantee the line is never paraphrased is to never let the model rewrite it.

**Decision.** The LLM returns `dialogue.text` as part of the structured script, but the code
(`quality/script_checks.py`):
1. when the idea contains quoted text, requires the line to equal **one complete quoted fragment**
   after normalization (a shortened quote, a quote plus narration, or narration instead of the quote
   is rejected); without quotes, it requires a verbatim fragment of the idea made of whole words;
2. on failure — retries the LLM with explicit feedback that lists the allowed fragments; after
   `llm.max_attempts` attempts — `ScriptError` (fatal, no point burning money further);
3. downstream uses **the original characters** from the input: the span runs from the first to the
   last matched word plus any `. ? ! …` right after it, never quote marks, commas outside the quote
   or narration words.

Normalization (`quality/text_norm.py`: case, punctuation, apostrophes, numbers and currency as
words, Ukrainian numerals in any case) is used only to *compare*; it never changes the line. It is
asymmetric where symmetric folding would equate different words: a Ukrainian ordinal ("сьомій",
"перший") or "процент" counts as the cardinal ("сім", "один") or "відсоток" only when the other
text wrote that number (or "%") in digits, so "О 7 ранку" matches "О сьомій ранку" but "Ти перший"
does not match "Ти один".

Quote parsing: “”, „“, «» and "" pair unambiguously (a `"` right after a digit is an inch mark);
single quotes double as apostrophes ("players'", "'n'", "'90s"), so every closing candidate of a
single quote is offered and the LLM's line must equal one of them. A quote nested inside a
double-style quote is part of that line, not a line of its own. A line the parser missed still
passes when it stands in the idea directly between an opening and a closing mark and overlaps no
parsed quote. A line that opens a quote keeps its closing mark (`Meet me at 7:30, "Room 12".`).

Consequence: "put the line in quotes" is part of the input contract, and the line must be **one
contiguous quote**. An idea that quotes only a name (a bar called "Moon") but leaves the spoken
sentence unquoted can only use "Moon" as the line; a line split by a speech tag
(`"Cancel the interviews," she says, "Roman gets the job."`) can only be one of its halves (joining
them would need a non-contiguous line). Without quotes the prompt asks for one sentence, but the
check only enforces a verbatim fragment of whole words.

---

## ADR-006. Domain state in SQLite, execution state in Prefect

**Context.** The question: "why Prefect if we manage state ourselves?" The answer — these are two different states:

| | Prefect (its own DB, `~/.prefect`) | Our state (`runs/holywater.db`) |
|-|----------------------|-----------|
| What it stores | flow/task runs, statuses, retries, timings | runs, stages, shots, attempts, verifications (WER), judge decisions, strategy used, costs, artifact paths |
| Who reads it | Prefect UI/CLI, the retry mechanism | `hw eval compare`, `hw inspect`, `hw costs`, resume, report |
| Can you ask "mean WER per model X this week" | no (not its data model) | yes |

Prefect gives us orchestration (retries with backoff, `retry_condition_fn`, task state, a UI when a
server is started) and knows nothing about a "shot" or "WER". Shots run sequentially inside one flow
run (ADR-001). Domain state is needed separately in any case; the only question is file vs DB.

**Decision.** **SQLite** via **SQLModel** (pydantic + SQLAlchemy). One file `runs/holywater.db`.
Media — on disk under `runs/<run_id>/...`; the DB holds paths and metadata, and for shot clips also
`size_bytes` + `sha256`, which resume checks before reusing a file. `report.json` remains as a
per-run export, but the source of truth is the DB.

Tables: `runs`, `stages`, `shots`, `shot_attempts`, `speech_verifications`, `judge_decisions`,
`costs` (columns in ARCHITECTURE §7). No migration tool: `SQLModel.metadata.create_all` plus a
forward-only additive migration at start-up (`ALTER TABLE ... ADD COLUMN` with the model default),
enough for a schema that only grows.

**Why not a JSON file:** eval/compare needs aggregations over many runs; atomic stage updates out of the box;
a few `SELECT`s instead of parsing 50 manifests.

**Grown-up version:** Postgres, Alembic migrations, S3/GCS for media, Prefect deployments + work pools (or Temporal).

---

## ADR-007. Jev (TypeSafe AI) as a typed judge

**Context.** Jev is a "System One" model: input is state (text/JSON) + typed questions (`Noul` yes/no, `Choice`,
`Score`), output is answers with calibrated probabilities. It does not generate text, does not see video/audio, cannot
count. ~0.2–0.4 s per call in the real run, $0.042/M input tokens (output free), Python SDK
`typesafe-sdk`.
It is TypeSafe AI — not Typeframes.

**Decision.** Use it in two places where we need a *decision*, not generation, and where an LLM call would be slow/expensive:

1. **Script pre-flight** (between `write_script` and `generate_video`) — before spending ≈ $1.2 (480p) on Seedance:
   - `Noul`: "visual_prompt contains content likely to trip the video model's safety filter" → if p > 0.7 — the LLM revises the script before generation, not after the video model refuses it.
   - `Score 1–5`: "how concrete and filmable the shot prompt is (one character, one action, one location)" → < 3 — the LLM revises with that feedback.
   - `Noul`: "the line fits naturally into the `end_s - start_s` window" — in addition to the arithmetic check in code.
2. **WER-gate failure classification** — when the transcript ≠ the line, we must pick a retry tactic:
   `Choice{paraphrased, truncated, extra_words, wrong_language, stt_noise, silence}` → `paraphrased/extra_words/silence` →
   retry with a stricter prompt; `truncated` → widen the line window; `wrong_language` → fall back to TTS immediately;
   `stt_noise` → re-run STT once on the same clip (same model; swapping models is a later step), don't burn video.

**What Jev does NOT do here:** it does not check verbatim-ness (that's WER in code — deterministic and exact), does not judge the picture
(text-only), does not write prompts.

**Integration.** `providers/base.py: JudgeProvider` Protocol → `providers/typesafe.py`, built by
`build_judge` outside the vendor registry (one vendor, no `provider:` key). Optional: with no key, no
SDK, a client the SDK refuses, or `judge.enabled=false` → `NoopJudge` with a WARNING; pre-flight is
skipped and the classifier falls back to word-count heuristics (no words → `silence`; under 0.6× the
expected words → `truncated`; over 1.5× → `extra_words`; else `unknown` → stricter prompt). So without
Jev, `paraphrased`, `wrong_language` and `stt_noise` never occur. Every answer is a `judge_decisions`
row with its probabilities and decision (pre-flight `pass/rewrite/unavailable`, classifier
`classified/unavailable/heuristic`), and every Jev call that reached the vendor gets an estimated
`costs` row — so we can later evaluate whether it actually saved money.

**Risks.** Early access; signups paused since 2026-09-22 — if there is no key, we run without it. English works best —
we pass state in English (shot prompts are in English anyway; for failure classification we pass both texts as they are).
Vendor benchmarks without independent verification — so thresholds (0.7, 3) go to config and get calibrated on the eval set.

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
  lipsync — one `SyncClient`, one download, one error map) and `OpenRouterProvider`
  (LLM, video, TTS, STT — one key, one `httpx.Client`, one status→error rule).
- `providers/registry.py` resolves the `provider:` named in each config section to a vendor instance,
  built once per run, and raises `ConfigError` if the vendor is unknown, lacks the capability or its
  key is missing (the first two also in `--dry-run`, which needs no keys).
  Switching video from fal to OpenRouter is a two-line YAML change (`provider`, `model`).
- Defaults: everything on OpenRouter (one key covers the whole pipeline); lipsync stays fal-only.
- The optional judge is the one exception: a single vendor (TypeSafe), built by `build_judge`
  outside `registry.VENDORS`, with no `provider:` key and no request type.

**OpenRouter media endpoints used.** `POST /videos` → poll the job (its `polling_url`, resolved against
the origin, or `GET /videos/{id}`) until `completed`/`failed`/`cancelled`/`expired` →
`GET /videos/{id}/content`; the poll response reports the real `usage.cost`. `POST /audio/speech`
(mp3 bytes). `POST /audio/transcriptions` with base64 WAV; the pipeline extracts a mono 16 kHz WAV
with ffmpeg first so fal and OpenRouter get identical STT input. Request bodies are checked against
the vendor OpenAPI schema in `docs/PROVIDERS_API.md`.

**Alternatives.** Keep per-capability adapters and compose them — more files, duplicated clients and
error mapping. A plugin/entry-point system — overkill for two vendors.

**Risks.** A vendor class grows with each capability; mitigated by keeping shared internals
private and capability methods thin. Cost figures differ in nature:
OpenRouter reports actual cost for video/STT/LLM; TTS, fal and Jev are table estimates (and fal
lipsync is not priced yet) — `costs.is_estimate` records which, so compare tags with that in mind.

---

## ADR-009. WER threshold 0.10, with the trade-off stated

**Context.** Whisper mishears: accents, proper names, numbers spelled differently. A gate at WER = 0
would reject correct native clips and pay for regenerations (≈ $0.62 per 6 s shot at 480p) and
fallbacks. A gate at 0.10 lets some real errors through.

**Decision.** Keep `quality.max_wer: 0.10`, computed after normalization (ADR-005). What that means in
words: a line under 10 normalized words must match **exactly** (one wrong word is ≥ 0.111); at 10–19
words one word may differ; at 20–29 words two. Measured on the eval lines: one changed number in the
9-word `en_numbers` line gives 0.111 and fails; dropping "never" from the 16-word `en_long_line`
gives 0.062 and **passes**, although it reverses the meaning. Short lines (the typical vertical-drama
line, e.g. the 8-word "Cancel the other interviews, Roman gets the job.") are effectively exact.

The WER of the decisive check is in `report.json` (`shots[].final_wer`), so a non-zero value is
visible, not hidden behind `line_verified=true`.

**Alternatives.** WER 0 (more false rejections, more spend), "no content-word errors" rules (needs
a stop-word and negation list per language). The grown-up fix is a stronger STT (a better model, or
an ensemble of two transcribers that must agree) rather than a stricter threshold on Whisper.

**Risks.** A meaning-changing one-word slip in a long line is reported as verified. Mitigations: the
line comes from the input (ADR-005), so only the actor can slip, and short lines are exact.

---

## ADR-010. Who retries what: idempotent reads in the vendor, paid work in the flow

**Context.** A Seedance job is polled for minutes. When a poll or a download failed with a transient
error, a Prefect task retry used to submit a **new** paid job, and the first one was billed but never
recorded.

**Decision.**
- Idempotent reads are retried inside the vendor class, against the same job id: OpenRouter poll
  errors (429/5xx/timeouts) are a WARNING and polling continues until `timeouts_s.video`; downloads of
  a finished job are retried `providers.<vendor>.download_attempts` times.
- Anything that would create a new paid job is decided by the Prefect task or the flow, from the
  typed error (ADR-001). A job that was accepted but whose result cannot be collected raises
  `BilledJobError` (not retryable, never softened) carrying `job_id` and `spent_usd`; the flow
  records its cost and an `abandoned` attempt row and regenerates the shot within its attempt budget.
  The same holds for a paid clip ffprobe cannot read (a `MediaError` there would be retried by the
  task, i.e. buy a new job) and for a JSON/text body on the download. When the job endpoints
  answered 401/403/404, the next job would fail the same way: the run stops instead.
- A `polling_url` from the 202 is used only on the API's own origin (scheme, host, port); if it
  answers 4xx, polling falls back once to `{base_url}/videos/{id}`, the URL the first real run polled.
  A 404 is tolerated for the first 3 polls (a just-accepted job may not be visible yet).
- Every charge is recorded when it happens: stages report each billed call (successful or not) to a
  cost sink that writes one `costs` row, so a call that is billed and then fails, or is retried by
  Prefect, still leaves its row.

**Risks.** A process killed while a job is being polled still loses that job (the job id is not
persisted before polling; it is only in `logs.jsonl` as `openrouter.video.submitted`), and a resume
pays for that shot again. fal's `subscribe` owns its polling, so a transient error after enqueue
still surfaces as retryable and resubmits (fal is not the default video vendor).

---

## ADR-011. Stray-speech check on speechless clips: warn by default

**Context.** Clips that must carry no speech (the non-dialogue shot, the dialogue clip under a TTS
voice) are generated with `generate_audio=true` for ambience. Seedance can add speech there, which
would put extra words into the final next to the verbatim line.

**Decision.** After its duration gate, every such clip gets one STT pass (`verify_speech.check_stray_speech`,
≈ $0.00015 per 6 s clip); more than `quality.stray_speech.max_words` (2) words is recorded as a
`speech_verifications` row with `expected=""` and `failure_kind=extra_words`. The default action is
`warn`: the clip is accepted with WARNING `shot.stray_speech`, because Whisper is known to
hallucinate words on music and ambience, and a false positive would pay for a new clip.
`action: regenerate` is available: it regenerates within `video.max_attempts` and then accepts the
clip with the fewest words. The dialogue clip under a TTS voice is always handled as `regenerate`:
its words would land in the line window and fail the final check.
The check is advisory, so an STT error on it (any provider or media error, after the task
retries) records the clip as unverified and accepts it; only 401/402/403 stop the run, since the
line check would fail the same way.

**Risks.** With `warn`, a clip that really talks ships; the report shows it (`shots[].stray_speech_words`).
