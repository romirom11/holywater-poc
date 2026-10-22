# Provider API conformance

Every request the pipeline sends is checked against the vendor's **machine-readable schema**, not
against tutorials or blog posts. Sources (fetched 2026-10-22):

- OpenRouter: `https://openrouter.ai/openapi.json` (paths `/chat/completions`, `/videos`,
  `/videos/{jobId}`, `/videos/{jobId}/content`, `/audio/speech`, `/audio/transcriptions`).
  Prices: `GET https://openrouter.ai/api/v1/videos/models` (`pricing_skus`) and
  `GET https://openrouter.ai/api/v1/models` (`pricing.prompt`), both public.
- fal: `https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=<app>` for each app we call.
- TypeSafe (Jev): the installed `typesafe-sdk` 0.7.5 (`typesafe_sdk/_schemas/models.py` mirrors the
  API's OpenAPI schema; `_core/response_types.py` holds the public answer types).

Re-run the extraction (see "How to re-verify") whenever a vendor module changes.

## Who retries what

- **Idempotent reads are retried inside the vendor class**, against the same job id: OpenRouter
  video polls (a 429/5xx/timeout on a poll GET is a WARNING `openrouter.video.poll_error` and
  polling continues until the deadline) and the download of a finished job (OpenRouter and fal,
  `providers.<vendor>.download_attempts` tries).
- **Anything that would create a new paid job is decided by the Prefect task / the flow**, from
  the typed error. A job that was accepted but whose result cannot be collected (poll deadline
  exceeded, poll rejected, download still failing, oversize body) raises `BilledJobError`:
  not retryable, not a `FatalProviderError`, with `job_id` and `spent_usd` (the vendor's
  `usage.cost` when reported, else an estimate; `spent_is_estimate` says which).
- Every `ProviderError` may carry `spent_usd`/`spent_is_estimate` for money already charged
  (e.g. a billed LLM answer that was truncated); the flow records it.

## What a default run sends

With the default config (everything on OpenRouter, Jev key set) a run whose checks all pass makes:
one `POST /chat/completions` (one more per answer that fails the script checks, up to
`llm.max_attempts`, and per pre-flight rewrite); Jev calls for safety and filmability per shot plus
one for "the line fits its window" (5 for two shots); per shot one `POST /videos`, a poll every `video_poll_interval_s` and
one `GET /videos/{id}/content`; one `POST /audio/transcriptions` on the dialogue clip and one on
each speechless clip (the stray-speech check). `POST /audio/speech` is called only when the
dialogue shot falls back to TTS (or with `audio.strategy: tts_overlay`).

## Error classes

| Error | Meaning | Raised for |
|-------|---------|------------|
| `RetryableProviderError` | retry the same request with backoff | HTTP in `RETRYABLE_HTTP_STATUS`, timeouts, transport errors, an empty download body, a failed video job whose error names an infrastructure cause, chat `finish_reason=error`, an `error` object with a 429/5xx code inside a 200 chat body |
| `PromptRejectedError` (`FatalProviderError`) | the vendor refused the prompt; the only error the flow may answer by softening the prompt | moderation / content-policy wording anywhere (`moderat`, `content policy`, `safety`, `flagged`, `nsfw`, `refus`…), HTTP 400 on a video submit, a `failed` video job without an infrastructure cause, chat `finish_reason=content_filter` or a `message.refusal` |
| `StructuredOutputError` (`FatalProviderError`, also `ValueError`) | the LLM answered and was billed, but the output is unusable; carries `raw_text`, `finish_reason`, `spent_usd` | `finish_reason=length`, empty content, non-JSON or schema-invalid output |
| `FatalProviderError` | will not help to retry or soften | 401/402/403/404 and other 4xx, `cancelled`/`expired` jobs, malformed responses (no job id, invalid JSON, a non-JSON 200 chat body, empty `choices`) |
| `BilledJobError` (`ProviderError`) | accepted job, result not collectable | see "Who retries what" |

## OpenRouter (`providers/openrouter.py`)

| Call | We send | Spec |
|------|---------|------|
| `POST /chat/completions` | `model`, `messages`, `max_tokens` (4000), `response_format={type:"json_schema", json_schema:{name:"Script", strict:true, schema}}` (schema = `Script.model_json_schema()`; every property is required, including the `dialogue.voice_gender` enum), `provider={require_parameters:true}`, `temperature` only when configured | `ChatFormatJsonSchemaConfig{type*, json_schema*{name*, schema, strict}}`; `ProviderPreferences.require_parameters: bool\|null`. Claude 5.x endpoints list no `temperature` in `supported_parameters`, hence it is optional. The spec marks `max_tokens` "deprecated, use max_completion_tokens"; we keep `max_tokens` because it is what the real run used with `require_parameters` and every structured-output endpoint accepts it; switching is a request-body change to be verified with one paid call. |
| chat response | `choices[0].finish_reason`, `message.content`, `message.refusal`, top-level `error`, `usage.cost` | `ChatFinishReasonEnum = [tool_calls, stop, length, content_filter, error, null]`; `ChatAssistantMessage.refusal: string\|null`; `ChatUsage.cost: number\|null`. Mapping: see "Error classes". |
| `POST /videos` | `model`, `prompt`, `duration` (int), `resolution` ("480p"/"720p"), `aspect_ratio` ("9:16"), `generate_audio` (bool), `seed` (int, only when given: the flow never sets one, so it is not sent and `shot_attempts.seed` stays null) | `model*: string`; `duration: integer ≥1`; `resolution: enum[360p,480p,720p,768p,1080p,1K,2K,4K]`; `aspect_ratio: enum[16:9,9:16,1:1,4:3,3:4,3:2,2:3,21:9,9:21]`; `generate_audio: boolean`; `seed: integer`. HTTP 400 → `PromptRejectedError`. |
| `202` response | read `id`, `polling_url`, `status` | `id*`, `polling_url*`, `status*: enum[pending,in_progress,completed,failed,cancelled,expired]`. The spec's own examples give `polling_url` **origin-relative** (`"/api/v1/videos/job-abc123"`). |
| `GET /videos/{jobId}` (via `polling_url`) | `polling_url` resolved against the origin (`httpx.URL(base_url).join(polling_url)`, so `/api/v1/videos/x` and absolute URLs both work; a URL on another origin — host, scheme or port, e.g. `http://` — is ignored with a WARNING, the key is never sent there); `{base_url}/videos/{id}` when absent. A `polling_url` other than that default which answers 4xx is replaced by the default once (WARNING `openrouter.video.polling_url_failed`). Loop every `video_poll_interval_s` until `completed`; read `usage.cost`, `generation_id`, `error` | operation "Poll video generation status"; same status enum; `usage: {cost: number\|null, is_byok}`; `error: string`. `failed` → `PromptRejectedError` (or `RetryableProviderError` when the error names an infrastructure cause), `cancelled`/`expired` → `FatalProviderError`, a 429/5xx/timeout on the GET → WARNING and keep polling, a 404 during the first 3 polls → WARNING and keep polling (the job may not be visible yet), any other poll failure or the `timeouts_s.video` deadline → `BilledJobError` (with the HTTP status; the flow stops on 401/403/404 instead of buying another job). |
| `GET /videos/{jobId}/content` | bearer header, default `index=0`; streamed to `<dest>.part` with a running 500 MB cap, renamed on success; retried `download_attempts` times (sleep `video_poll_interval_s`) on 429/5xx/timeouts/empty body | `index: integer\|null default 0`; `200 video/mp4` raw bytes; `409/410/404` errors. A JSON or `text/*` body is a failed download (an error sent with a 200); any other content type than `video/*` is only a WARNING (vendors sometimes send `application/octet-stream`; ffprobe is the real check, and a paid clip it cannot read is a `BilledJobError` in the stage). Exhausted retries, a non-retryable status, a JSON/text body or an oversize body → `BilledJobError`. |
| `POST /audio/speech` | `model`, `input` (string), `voice` (when set: `audio.tts.voices[dialogue.voice_gender]`, else `audio.tts.voice`; the flow retries a 400/422 once with `audio.tts.voice`), `response_format:"mp3"` | `model*`, `input*: string\|SpeechTurn[]`, `voice: string minLength 1`, `response_format: enum[mp3,pcm] default pcm`; `200 audio/*` bytes (streamed like the video download, not retried in the vendor: every POST is a new paid synthesis) |
| `POST /audio/transcriptions` | `model`, `input_audio:{data: base64 raw, format:"wav"}`, `language` (when not auto; with `audio.stt.language: auto` the flow passes the line's language, so it is always sent), `response_format:"json"` | `input_audio*: STTInlineInputAudio{data*, format*}` ("raw bytes, not a data URI"); `language: string`; `response_format: enum[json,verbose_json]`; response `text*`, `usage.cost`, `language` (verbose only) |

Status→error rule for every OpenRouter call (`_classify`): `RETRYABLE_HTTP_STATUS = {408, 409, 425,
429, 500, 502, 503, 504, 524, 529}` → `RetryableProviderError` (the spec documents
429/500/502/503/504/524/529 as transient); else moderation wording in the message or body
(OpenRouter answers flagged input with 403 and `error.metadata.flagged_input`), or a 400 on
`POST /videos` → `PromptRejectedError`; anything else → `FatalProviderError`.

Timeouts (from `retries.timeouts_s`, passed per request): chat `llm` (OpenAI client), video submit
and download `video`, each poll GET what is left of the `video` deadline, `/audio/speech` `tts`,
`/audio/transcriptions` `stt`.

Costs: `usage.cost` from the response is the actual charge (`cost_is_estimate=False`). Without
it (logged as WARNING `cost.unknown_price`) the vendor estimates: video from
`providers.openrouter.video_usd_per_s` × duration; STT from `STT_PRICE_PER_SECOND_USD` × WAV
length (`openai/whisper-large-v3`: the real run was charged $0.000152 for a 6.08 s clip =
$0.000025/s; the listed `pricing.prompt` of $0.0000075 has an undocumented unit, so only the ratio
to the turbo price is used); TTS always from `TTS_PRICE_PER_CHAR_USD` (`pricing.prompt` per
character). An unknown model id in a table logs `cost.unknown_price` and uses the most expensive
entry. `video_usd_per_s` for `bytedance/seedance-2.5`: `pricing_skus.video_tokens` = $0.0000107
per video token, tokens ≈ width × height × 24 fps × seconds / 1024 → 480p (480×854) ≈ $0.1028/s,
observed $0.6211 for a 6 s clip in the real run = **$0.1035/s** (used); 720p (720×1280) ≈
$0.231/s, scaled from the observed 480p rate by pixel count = **$0.233/s** (used). Change the
table together with `video.model`.

## fal (`providers/fal.py`)

| App | We send | Spec (`*Input` schema) |
|-----|---------|------------------------|
| `bytedance/seedance-2.5/text-to-video` | `prompt`, `duration` (**string**, e.g. `"8"`), `resolution`, `aspect_ratio`, `generate_audio` | `prompt*: string`; `duration: string enum["auto","4".."30"]`; `resolution: enum[480p,720p,1080p]`; `aspect_ratio: enum[auto,21:9,16:9,4:3,1:1,3:4,9:16]`; `generate_audio: boolean`. **No `seed` input** — seed is output-only, so a requested seed is logged and dropped. Output: `video*{url}`, `seed*`. |
| `fal-ai/whisper` | `audio_url` (uploaded WAV), `task:"transcribe"`, `chunk_level:"none"`, `language` (when not auto) | `audio_url*`; `task: enum[transcribe,translate]`; `chunk_level: enum[none,segment,word]`; `language: enum[... uk, en ...]\|null`. Output: `text*`, `inferred_languages*`. |
| `fal-ai/elevenlabs/tts/eleven-v3` | `text`, `voice`, `language_code`, `apply_text_normalization:"off"` | `text*`; `voice: string default "Rachel"`; `language_code: string\|null`; `apply_text_normalization: enum[auto,on,off]`. Output: `audio*{url}`. |
| `fal-ai/sync-lipsync/v2` | `video_url`, `audio_url` (both uploaded), `sync_mode` | `video_url*`, `audio_url*`, `sync_mode: enum[cut_off,loop,bounce,silence,remap]`, `model: enum[lipsync-2,lipsync-2-pro] default lipsync-2` (not sent → default). Output: `video*{url}`. |

Transport: one `fal_client.SyncClient(key=FAL_KEY)` for both `subscribe(app, arguments,
client_timeout=…)` and `upload_file(path)` — never the module-level `fal_client.*` functions,
which read `FAL_KEY` from `os.environ` and would ignore the key from `.env`. `client_timeout` per
operation: video and lipsync `timeouts_s.video`, TTS `tts`, STT `stt`; the client's per-HTTP-request
`default_timeout` (submit, status, result, upload) is `timeouts_s.stt`. Results are downloaded from
`video.url` / `audio.url` (temporary public CDN links, no auth header) with the operation's timeout,
streamed with the same size cap as OpenRouter (any content type is only a WARNING here; a clip
ffprobe cannot read becomes a `BilledJobError` in the stage), and retried
`providers.fal.download_attempts` times; then `BilledJobError`. A finished job whose result has no
URL is a `FatalProviderError` carrying the estimated `spent_usd`.

`FalClientHTTPError.status_code` is classified with the same rule: `RETRYABLE_HTTP_STATUS` →
retryable; moderation wording in the message or `error_type` (e.g. `content_policy_violation`), or a
400 on `text_to_video` → `PromptRejectedError`; else fatal. Known limitation: `subscribe` owns the
queue polling, so a transient error *after* the job was enqueued surfaces as
`RetryableProviderError` and a Prefect retry submits a new job (fal is not the default video vendor).
Costs are estimates from in-code tables (`VIDEO_PRICE_PER_SECOND_USD`, `TTS_PRICE_PER_1K_CHARS_USD`,
`STT_PRICE_PER_CALL_USD`); an unknown model logs `cost.unknown_price`. Lipsync has no price table
(billed per output minute, duration unknown here): `cost.unknown_price` WARNING and `cost_usd=0`.

## TypeSafe Jev (`providers/typesafe.py`)

One `TypeSafeClient(api_key, model=judge.model, timeout=timeouts_s.judge)`, one
`system_one(state={"state": ...}, questions={"q": <question>})` per decision.

| Question | We send | Answer (SDK model) → our `JudgeAnswer` |
|----------|---------|-----------------------------------------|
| `Noul(instructions=q)` | yes/no | `NoulAnswer.noul` = P(yes) → `"yes"` if ≥ 0.5, probabilities `{"yes", "no"}` |
| `Choice(instructions=q, criteria={option: None})` | one of N | `ChoiceAnswer.choice`, `.probabilities` keyed by option name → as is; a choice outside the options → unavailable + WARNING |
| `Score(instructions=q, criteria=["1: very low", …])` | 1..N levels | `ScoreAnswer.score` is the probability-weighted mean of the **0-based** rubric positions ("Each description's position determines its score, starting at zero"); `.probabilities` is `dict[int, float]` keyed 0..N-1 → level `clamp(round(score) + 1, 1, N)`, probabilities re-keyed `"1".."N"` |

Never raises into the pipeline: a disabled judge, a blank `TYPESAFE_API_KEY`, a missing SDK or
any error while building the client → WARNING `judge.disabled` (with the reason) and `NoopJudge`
(`model == "noop"`); a `TypeSafeError` on a call → WARNING `judge.unavailable` and an unavailable
answer. Each answer carries `cost_usd` (always an estimate): `usage.input_tokens` (or chars/4 when
absent) × `judge.usd_per_m_input_tokens` (default $0.042 per million input tokens; output tokens
are free per the API's `Usage` schema).

## How to re-verify

```bash
# OpenRouter: dump the request/response schemas for the paths we use
curl -s https://openrouter.ai/openapi.json | python3 -c '
import json,sys; d=json.load(sys.stdin)
for p in ["/videos","/videos/{jobId}","/videos/{jobId}/content","/audio/speech","/audio/transcriptions"]:
    print(p, json.dumps(d["paths"][p], indent=1)[:3000])'

# OpenRouter prices: video per token (pricing_skus), STT/TTS per unit (pricing.prompt)
curl -s https://openrouter.ai/api/v1/videos/models | python3 -c '
import json,sys; [print(m["id"], m["pricing_skus"]) for m in json.load(sys.stdin)["data"] if "seedance" in m["id"]]'
curl -s "https://openrouter.ai/api/v1/models?output_modalities=all" | python3 -c '
import json,sys; [print(m["id"], m["pricing"]["prompt"]) for m in json.load(sys.stdin)["data"] if "whisper" in m["id"] or "eleven" in m["id"]]'

# fal: one schema per app id
curl -s "https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=bytedance/seedance-2.5/text-to-video" \
  | python3 -c 'import json,sys; s=json.load(sys.stdin)["components"]["schemas"]; print(json.dumps({k:v for k,v in s.items() if "Input" in k or "Output" in k}, indent=1))'

# Jev: the answer models of the installed SDK
uv run python -c 'import inspect, typesafe_sdk._schemas.models as m; print(inspect.getsource(m.ScoreAnswer))'
```

Unit tests pin the exact request bodies above and every error mapping:
`tests/unit/test_provider_openrouter.py` (real OpenAI SDK and httpx over `httpx.MockTransport`),
`tests/unit/test_provider_fal.py` (SDK mocked, downloads over `MockTransport`),
`tests/unit/test_typesafe.py` (real `TypeSafeClient` over an offline mock transport, skipped
without the judge extra). Change them together with this document.
