# Provider API conformance

Every request the pipeline sends is checked against the vendor's **machine-readable schema**, not
against tutorials or blog posts. Sources (fetched 2026-10-22):

- OpenRouter: `https://openrouter.ai/openapi.json` (paths `/chat/completions`, `/videos`,
  `/videos/{jobId}`, `/videos/{jobId}/content`, `/audio/speech`, `/audio/transcriptions`).
- fal: `https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=<app>` for each app we call.

Re-run the extraction (see "How to re-verify") whenever a vendor module changes.

## OpenRouter (`providers/openrouter.py`)

| Call | We send | Spec |
|------|---------|------|
| `POST /chat/completions` | `model`, `messages`, `max_tokens`, `response_format={type:"json_schema", json_schema:{name, strict:true, schema}}`, `provider={require_parameters:true}`, `temperature` only when configured | `ChatFormatJsonSchemaConfig{type*, json_schema*{name*, schema, strict}}`; `ProviderPreferences.require_parameters: bool\|null`. Claude 5.x endpoints list no `temperature` in `supported_parameters`, hence it is optional. |
| `POST /videos` | `model`, `prompt`, `duration` (int), `resolution` ("480p"/"720p"), `aspect_ratio` ("9:16"), `generate_audio` (bool), `seed` (int, optional) | `model*: string`; `duration: integer ≥1`; `resolution: enum[360p,480p,720p,768p,1080p,1K,2K,4K]`; `aspect_ratio: enum[16:9,9:16,1:1,4:3,3:4,3:2,2:3,21:9,9:21]`; `generate_audio: boolean`; `seed: integer` |
| `202` response | read `id`, `polling_url`, `status` | `id*`, `polling_url*`, `status*: enum[pending,in_progress,completed,failed,cancelled,expired]` |
| `GET {polling_url}` | loop until `completed`; `failed\|cancelled\|expired` → `FatalProviderError`; read `usage.cost`, `generation_id`, `error` | same status enum; `usage: {cost: number\|null, is_byok}`; `error: string` |
| `GET /videos/{jobId}/content` | bearer header, default `index=0`, expect `video/mp4` | `index: integer\|null default 0`; `200 video/mp4` raw bytes; `409/410/404` errors |
| `POST /audio/speech` | `model`, `input` (string), `voice` (when set), `response_format:"mp3"` | `model*`, `input*: string\|SpeechTurn[]`, `voice: string minLength 1`, `response_format: enum[mp3,pcm] default pcm`; `200 audio/*` bytes |
| `POST /audio/transcriptions` | `model`, `input_audio:{data: base64 raw, format:"wav"}`, `language` (when not auto), `response_format:"json"` | `input_audio*: STTInlineInputAudio{data*, format*}` ("raw bytes, not a data URI"); `language: string`; `response_format: enum[json,verbose_json]`; response `text*`, `usage.cost`, `language` (verbose only) |

Status→error rule for every OpenRouter call: `RETRYABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 524, 529}`
(the spec documents 429/500/502/503/504/524/529 as transient); anything else 4xx is `FatalProviderError`.

## fal (`providers/fal.py`)

| App | We send | Spec (`*Input` schema) |
|-----|---------|------------------------|
| `bytedance/seedance-2.5/text-to-video` | `prompt`, `duration` (**string**, e.g. `"8"`), `resolution`, `aspect_ratio`, `generate_audio` | `prompt*: string`; `duration: string enum["auto","4".."30"]`; `resolution: enum[480p,720p,1080p]`; `aspect_ratio: enum[auto,21:9,16:9,4:3,1:1,3:4,9:16]`; `generate_audio: boolean`. **No `seed` input** — seed is output-only, so a requested seed is logged and dropped. Output: `video*{url}`, `seed*`. |
| `fal-ai/whisper` | `audio_url` (uploaded WAV), `task:"transcribe"`, `chunk_level:"none"`, `language` (when not auto) | `audio_url*`; `task: enum[transcribe,translate]`; `chunk_level: enum[none,segment,word]`; `language: enum[... uk, en ...]\|null`. Output: `text*`, `inferred_languages*`. |
| `fal-ai/elevenlabs/tts/eleven-v3` | `text`, `voice`, `language_code`, `apply_text_normalization:"off"` | `text*`; `voice: string default "Rachel"`; `language_code: string\|null`; `apply_text_normalization: enum[auto,on,off]`. Output: `audio*{url}`. |
| `fal-ai/sync-lipsync/v2` | `video_url`, `audio_url`, `sync_mode` | `video_url*`, `audio_url*`, `sync_mode: enum[cut_off,loop,bounce,silence,remap]`, `model: enum[lipsync-2,lipsync-2-pro] default lipsync-2` (not sent → default). Output: `video*{url}`. |

Transport: `fal_client.SyncClient.subscribe(app, arguments, client_timeout=…)`; results are
downloaded from `video.url` / `audio.url` (fal URLs are temporary). `FalClientHTTPError.status_code`
is classified with the same `RETRYABLE_HTTP_STATUS` rule.

## How to re-verify

```bash
# OpenRouter: dump the request/response schemas for the paths we use
curl -s https://openrouter.ai/openapi.json | python3 -c '
import json,sys; d=json.load(sys.stdin)
for p in ["/videos","/videos/{jobId}","/videos/{jobId}/content","/audio/speech","/audio/transcriptions"]:
    print(p, json.dumps(d["paths"][p], indent=1)[:3000])'

# fal: one schema per app id
curl -s "https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=bytedance/seedance-2.5/text-to-video" \
  | python3 -c 'import json,sys; s=json.load(sys.stdin)["components"]["schemas"]; print(json.dumps({k:v for k,v in s.items() if "Input" in k or "Output" in k}, indent=1))'
```

Unit tests in `tests/unit/test_provider_openrouter.py` and `tests/unit/test_provider_fal.py` pin
the exact request bodies above; change them together with this document.
