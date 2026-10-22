# Data flow: from idea to video

## 1. Stage diagram

```
 IdeaInput  (CLI argument or stdin, 10–1000 characters)
   │
   ▼
 [1]  write_script (LLM) ──► Script            ✘ schema / line / window check → feedback,
   │                                              up to llm.max_attempts answers → ScriptError
   ▼
 [1b] preflight_script (Jev: safety, filmability, line fits its window)
   │     ✘ issues → the LLM revises its previous script (rewrite) → [1b] again,
   │                up to llm.max_attempts rounds → ScriptError
   │     judge off / unavailable → skipped (recorded as `unavailable`)
   ▼
 [2a] generate_audio (TTS) — tts_overlay only (configured, or after a fallback);
   │     the measured voice resizes the line window / dialogue shot before its clip is made
   ▼
 [3]  generate_video — per shot, one shot after another (Seedance 2.5)
   │     every clip: DurationGate (requested ± tolerance)          ✘ → regenerate this shot
   │     dialogue shot, native:  [4] verify_speech (STT + WER)       ✘ → tactic by failure_kind
   │                              (strict prompt / wider window / listen again) → TTS fallback (→ 2a)
   │     speechless clip (other shot; dialogue shot under TTS): stray-speech STT check
   │                              (warn; the clip under a TTS voice is regenerated)
   ▼
 [5]  assemble: normalize (h264, 48 kHz stereo) → mix the TTS voice (+ lipsync if enabled)
   │     → concat → loudnorm → trim to max (never through the line)
   │     → MediaIntegrityGate (9:16, fps, audible audio) → DurationGate (quality.duration_s exactly)
   │     too short → regenerate one shot longer (the shortest without a verified native line;
   │                 → 3) and assemble once more
   ▼
 [4'] tts_overlay only: verify_speech on the voice window of final.mp4
   ▼
 [6]  report → report.json (also after a failure), runs.status
```

For the `native` strategy stage 2a is recorded as `skipped` (it runs for real only if the dialogue
shot falls back to TTS). For `tts_overlay` stage 4 runs once more on the assembled final, on the TTS
voice only (its measured length ± `quality.final_check_margin_s`, 0.5 s): TTS is deterministic, but
STT confirms the mix/lipsync/loudnorm did not break the audio and no speech from the clip under the
voice got into the line window. A failed check fails the run
(`QualityGateError`, stage `verify_speech`); an unavailable STT leaves `report.line_verified=false`
and the CLI prints a warning.

## 2. Artifacts per stage

Media and large JSON — in `runs/<run_id>/`; metadata, statuses, metrics — in SQLite (ARCHITECTURE §7):

| Stage | File on disk | DB rows | Key fields |
|-------|--------------|---------|------------|
| 0 input | — | `runs` | `input_text`, `config_hash`, `git_sha`, `dry_run`, `tag` |
| 1 | `script.json` (`Script`) | `stages`, `shots`, `costs` | see §3; one costs row per billed LLM answer |
| 1b | — | `judge_decisions`, `costs` | `shot_id`, `question`, `answer`, `probabilities`, `decision` (`pass` / `rewrite` / `unavailable`), `model_version`; one estimated costs row per Jev call |
| 2a | `audio/line_<shot>.mp3` | `stages`, `costs` | `path`, `duration_s` (measured), `model`, `voice`, `text` |
| 3 | `video/shot_<n>_a<attempt>.mp4` | `shot_attempts`, `costs` | `prompt_version`, `prompt_text`, `provider_request_id`, `path`, `duration_s`, `resolution`, `cost_usd`, `status`, `size_bytes`, `sha256`. `seed` is null on the default vendor: the flow requests none and OpenRouter returns none |
| 4 | `work/<clip>.stt.wav` + `work/<clip>.stt.wav.json` (which media and segment it was cut from) | `speech_verifications`, `judge_decisions`, `costs` | `expected` (`""` for a stray-speech check), `transcript`, `wer`, `passed`, `verified`, `failure_kind`; classifier `decision` (`classified` / `unavailable` / `heuristic`) |
| 5 | `final.mp4` (h264 / AAC 48 kHz stereo, 9:16); `work/` (`norm_*.mp4`, `concat.txt`, `joined.mp4`, `lipsync_<hash>.mp4` + `.json` ledger) | `runs.final_path`, `costs` (lipsync) | — |
| 6 | `report.json`, `logs.jsonl` | `runs.status`, `cost_usd_total` | `RunReport` — a DB export (QUALITY_METRICS §1) |

## 3. Script schema (`Script`)

LLM output via structured output; the fields that go to the video model are deliberately close to the format Seedance
understands (continuity / timeline / audio).

```python
class Dialogue(BaseModel):
    shot_id: int
    speaker: str                 # "woman in red coat"
    text: str                    # VERBATIM from IdeaInput.text — validated by code
    language: str                # ISO 639-1: "uk", "en"
    emotion: str                 # "tired, quiet"
    start_s: float; end_s: float # when in the shot it is spoken
    voice_gender: VoiceGender    # "female" | "male": picks the TTS voice if the line is dubbed

class Shot(BaseModel):
    id: int                      # 1..2
    duration_s: int              # 4..30; sum over shots 8..30
    visual_prompt: str           # scene, characters, action
    camera: str                  # "slow push-in, handheld"
    continuity: str              # identity, wardrobe, location — identical across shots
    ambient_audio: str           # "rain on roof, engine idle"
    has_dialogue: bool

class Script(BaseModel):
    title: str
    logline: str
    language: str
    style: str                   # "cinematic, moody, 35mm"
    shots: list[Shot]            # 1..2
    dialogue: Dialogue           # exactly one line (PoC)
```

Checks after every LLM answer; a violation → LLM retry with the reasons as feedback:
- structural (`domain/script.py`): shot ids `1..n` in order; `8 <= sum(duration_s) <= 30`; exactly
  one shot with `has_dialogue=True` and it equals `dialogue.shot_id`; `0 <= start_s < end_s <= `
  that shot's `duration_s`;
- the line (`quality/script_checks.py`): if the idea quotes something, `dialogue.text` must equal
  one complete quoted fragment after normalization; otherwise it must be a verbatim fragment of the
  idea made of whole words (ADR-005);
- the window: `end_s - start_s >= len(line) / quality.speech_chars_per_s` (15 chars/s).

After validation `Script.dialogue.text` **is replaced by the original characters from the input** (ADR-005).
`voice_gender` is required in the JSON schema sent to the LLM; a `script.json` without it loads as
`female`.

The scriptwriter prompt (`prompts/scriptwriter.md`, v3) has four sections: `system` (rules; the
speech rate is rendered from `quality.speech_chars_per_s`), `user` (the idea plus feedback), `soften`
(rewrite one rejected shot) and `rewrite` (revise the previous script with the pre-flight issues).

## 4. Seedance prompt (assembled by code, not by the LLM)

`stages/generate_video.py` renders `prompts/seedance_shot.md` (v2) from `Shot` + `Dialogue` fields.
The dialogue shot under `native`:

```
[Continuity] {continuity}. Style: {style}. Vertical 9:16, single continuous take, no cuts, no text on screen, no subtitles.
[Timeline] 0–{start_s}s: {visual_prompt}, {camera}. {start_s}–{end_s}s: {speaker} says "{line}" ({emotion}); the speaker's mouth moves only during this line, no other speech. {end_s}–{duration_s}s: the shot holds, {speaker} stays silent.
[Audio] {language name} speech, clean lip-synced dialogue, the exact words "{line}" and nothing else; {ambient_audio}; no music.
```

- When `start_s` is 0 the scene is described without a `0–0s` segment; when `end_s` equals the shot
  length there is no hold segment.
- The strict variant (after a failed speech check) appends: `The spoken words must be exactly
  "{line}" — do not add, drop, translate or rephrase any word.`
- Every other clip — the shot without dialogue, and the dialogue shot under `tts_overlay` — uses the
  `no_dialogue` variant: `[Timeline] 0–{duration_s}s: {visual_prompt}, {camera}. No one speaks.` and
  `[Audio] no dialogue, no speech, no voice; {ambient_audio}; no music.` So under `tts_overlay`
  nobody mouths the line on screen: without lipsync (off by default) the TTS line is a voice-over.

The rendered text and its version (`seedance_shot@2/dialogue`, `/dialogue_strict`, `/no_dialogue`)
are stored per attempt and end up in `report.prompt_versions` together with `scriptwriter@3`.

## 5. State and stitching stages together

- Stages are pure functions (`stages/*`): they take explicit inputs and provider deps, write their media file, and return a pydantic artifact with the path. They never touch SQL (`stages/report.py` reads the DB through `RunRepository`).
- The flow owns the state: `_stage(ctx, name)` marks `stages.start/finish` around each step (its time is added to `stages.total_duration_s`), and right after a task returns the flow writes the artifact's rows (`shot_attempts`, `speech_verifications`, `judge_decisions`, …). Each write is its own small transaction; there is no multi-table transaction, and a crash between two writes is recovered by resume (below), not by rollback.
- A Prefect task is a thin wrapper around one stage call; retries with backoff live on the task, decisions live in the flow. Tasks pass pydantic models with paths/ids between them, not binary data.
- Every charge is recorded when it happens: stages report each billed call — including LLM answers that fail validation and failed vendor calls that carry `spent_usd` — to a cost sink, which writes one `costs` row (`_record_cost`) before the stage continues or raises. A call that is billed and then fails, or is retried by Prefect, keeps its row. This holds for the LLM, video and TTS (the TTS charge is reported before its file is probed); lipsync results paid inside `assemble` are recorded by the assemble that returns, or, when assemble fails, before the error leaves the stage.
- The current `Script` lives in the flow context and on disk (`script.json`); every tactic that changes it (prompt softening, widening the line window, fitting the TTS voice, extending a shot) rewrites both and the `shots` rows, so the file always shows the version that was actually shot.
- Generated media is downloaded immediately (vendor URLs are temporary). Downloads are streamed to `<dest>.part` with a running 500 MB cap and renamed on success, so no partial file is left behind; an empty body is retried; a JSON or `text/*` body is a failed download (an error sent with a 200 is never media); any other content type that is not audio/video is only a WARNING (vendors send `application/octet-stream`; ffprobe is the real check).
- Before STT the audio track is extracted to a mono 16 kHz WAV (`work/<clip>.stt.wav`) so every STT vendor receives the same input. `audio.stt.language: auto` means "the language of the line": Whisper is told the expected language, it does not auto-detect.
- Every clip is normalized to the same format (h264, 24 fps, the 9:16 size of `video.resolution`, AAC 48 kHz stereo) before concat; concat refuses inputs that still differ (`MediaError`).

## 6. Error handling per stage

| Stage | Failure | Reaction |
|-------|---------|----------|
| any task | `RetryableProviderError` (429/5xx/timeout/network) or `MediaError` | Prefect retry: `retries.provider.attempts - 1` (2) retries after `backoff_s` (5 s, 15 s), counted in `stages.retries`; still failing → the error reaches the flow and the rows below apply (default: the run fails) |
| 1 script | invalid or truncated JSON (`StructuredOutputError`, billed), schema, line or window check violation | LLM retry with the reasons (up to `llm.max_attempts`) → `ScriptError` fatal; every billed answer has a costs row |
| 1 script | refusal / content filter (`PromptRejectedError`), 401–404 | fatal |
| 1b preflight | Jev: safety risk > `safety_risk_threshold`, filmability < `min_filmability_score`, or the line does not fit its window | the LLM revises its previous script with the issues (`rewrite` section); up to `llm.max_attempts` rounds in total (1 write + 2 rewrites by default), then `ScriptError` (stage `write_script`) before any video is paid |
| 1b preflight | judge disabled, no key / SDK, unavailable, or an unparsable score | WARNING, stage `skipped` when nothing usable came back, continue. Rows: one per question asked (`decision=unavailable`), or a single `preflight` row (`model_version` `disabled` / `noop`) when nothing was asked |
| 2a TTS | the vendor refuses the voice picked by `dialogue.voice_gender` (400/422) | WARNING `tts.voice_rejected`, one retry with `audio.tts.voice` |
| 2a TTS | the file has no audio or zero length | `QualityGateError(gate=tts_audio)`, its cost recorded, the run fails |
| 2a TTS | the file cannot be probed (`MediaError`) | Prefect retry (a new TTS call; each call has its own costs row, written before the probe) |
| 2a TTS | the voice does not fit the line window | the window becomes `start_s .. start_s + voice + 0.3 s`; past the shot end, the dialogue shot is lengthened before its clip is made (at most 30 s, total within `quality.duration_s.max`; WARNING `shot.extended_for_tts`), else the line starts earlier; cannot fit at all → `QualityGateError(gate=voice_fit)` before the dialogue clip is paid for |
| 3 video | the vendor refused the prompt (`PromptRejectedError`: moderation wording, HTTP 400 on submit, a failed job without an infrastructure cause) | one LLM rewrite of **that shot's wording only** (`soften` section: visual prompt, camera, continuity, ambience; duration and `has_dialogue` kept), then the same attempt again. Refused again: the native dialogue shot falls back to `tts_overlay` (its prompt omits the line; `fallback_reason` `video prompt rejected`); any other clip → fatal |
| 3 video | any other `FatalProviderError` (401–404, cancelled/expired job, malformed response) | fatal, no softening, no second paid video |
| 3 video | a failed job whose error names an infrastructure cause | `RetryableProviderError` → Prefect retry (a new job) |
| 3 video | a poll or download blip | retried inside the vendor against the same job id (polls until `timeouts_s.video`, downloads `providers.<vendor>.download_attempts` times); never a new job |
| 3 video | OpenRouter `polling_url` on another origin (host, scheme or port), or one that answers 4xx | the key is never sent off-origin: WARNING, `{base_url}/videos/{id}` is polled instead (once, for a 4xx); a 404 is tolerated for the first 3 polls (WARNING `openrouter.video.poll_not_found`) |
| 3 video | job accepted, result not collectable (`BilledJobError`: poll deadline, poll refused, downloads keep failing, a JSON/text body instead of the clip, body over 500 MB, or a paid clip ffprobe cannot read) | its `spent_usd` becomes a costs row (WARNING `cost.spent_on_error`, or the job's reported cost), the attempt is stored as `abandoned` (WARNING `shot.job_abandoned`) and counts as a failed attempt; the shot is regenerated while its budget lasts, then the last `BilledJobError` ends the run. Never a Prefect retry (that would buy a new job outside the budget) |
| 3 video | `BilledJobError` whose job endpoints answered 401/403/404 | the attempt is `abandoned` and the run stops (ERROR `shot.job_unreachable`): the next job would fail the same way |
| 3 video | clip length outside requested ± `quality.shot_duration_tolerance_s` (under a TTS voice also shorter than `start_s` + voice) | attempt `failed`, shot regenerated within its budget (`audio.native_attempts` for the native dialogue shot, `video.max_attempts` for every other clip, including the TTS fallback clip); only when **every** attempt missed the length → `QualityGateError` ("in all N attempt(s)") |
| 3 video | speechless clip carries words (> `quality.stray_speech.max_words`) | `action: warn` (default): accepted with WARNING `shot.stray_speech`. `regenerate`: new clips within `video.max_attempts`, then the one with the fewest words is accepted (WARNING). The dialogue clip under a TTS voice is always handled as `regenerate`: its words would be mixed into the line window and fail the final check. The check is advisory: any STT/media error on it (after the task retries) → `unverified` row + WARNING `shot.stray_check_unavailable`, clip accepted; only an STT 401/402/403 fails the run |
| 4 verify | native clip without an audio track | failed check, `failure_kind=silence`, empty transcript, no STT call → strict prompt |
| 4 verify | `wer > max_wer` | `failure_kind` from Jev, or from word-count heuristics without it (no words → `silence`; < 0.6× the expected words → `truncated`; > 1.5× → `extra_words`; else `unknown`). Tactic: `paraphrased/extra_words/silence/unknown` — regenerate **only this shot** with the strict prompt; `truncated` — widen the line window by `quality.truncated_window_widen_s` on each side and regenerate with the same prompt (strict when the window cannot grow); `wrong_language` — fall back to `tts_overlay` immediately; `stt_noise` — listen again (one STT re-run on the same clip, no new video), then as above. After `audio.native_attempts`, the shot falls back to `tts_overlay` if any attempt failed on speech. Without Jev, `paraphrased`, `wrong_language` and `stt_noise` never occur |
| 4 verify | STT stays unavailable after the retries (`RetryableProviderError`) | `unverified` row, attempt `unverified`, WARNING; the shot falls back to `tts_overlay` (the only path that does not need STT on the clip). `line_verified` is true only if the final check on the TTS final then succeeds |
| 4 verify | fatal STT error (400–404, unknown model) | the run fails at `verify_speech` (`stages.error`), before any more paid generation |
| 5 assemble | ffmpeg error | `MediaError` → Prefect retry (`retries.provider`) → fatal |
| 5 assemble | the TTS voice would run past the dialogue clip (+0.05 s) | `QualityGateError(gate=voice_fit)`, fatal (the fit in 2a normally prevents it) |
| 5 assemble | lipsync enabled | one paid call per (clip, voice, offset, model, size), cached in `work/lipsync_<hash>.mp4` with a `.json` ledger, so retries and resumes never pay twice; its cost is recorded once — by the assemble that returns, or before the error leaves a failed assemble; a result whose length is off → `QualityGateError(gate=lipsync_duration)` |
| 5 assemble | final > `quality.duration_s.max` | trimmed to exactly max, unless that would cut the line (the later of the line end on the final timeline and the end of the mixed TTS voice) → `QualityGateError(gate=duration)` |
| 5 assemble | final not 9:16 (±0.01), unknown frame rate, no audio, or audio peak < `quality.min_audio_peak_db` (-50 dB) | `QualityGateError(gate=media_integrity)`, fatal; checked before the length so a broken final never triggers a paid extension |
| 5 assemble | final < `quality.duration_s.min` | regenerate one shot once with `duration_s + ceil(deficit) + 1` (at most 30 s) and assemble again: the shortest shot that does not carry an already verified native line (regenerating that one would pay for the line and its check again); the replaced clip's attempt becomes `superseded`; still short → `QualityGateError` |
| 4' final check | `tts_overlay` final fails the WER gate | `QualityGateError(gate=speech)` naming the transcript heard in the line window (`context.transcript`), `failure_stage=verify_speech`, the run fails |
| any | an error reaches the flow | `runs.status=failed`, the failing stage keeps `error_type`/`error_json`, `report.json` is still written with `failure_stage` = the innermost failing stage of this invocation (stage rows failed by an earlier invocation of a resumed run are ignored; unknown → `null`) |
| start | a refusal in `_open_run` (an enabled HITL point, a different idea under an existing id, dry/real mismatch) | `ConfigError` before the run directory, run row or report is created (exit 1) |

`runs.fallback_reason` names the cause: `stt unavailable (<message>)`, `wrong_language`,
`exhausted N native attempts (last failure: <kind>)` or `video prompt rejected`; per attempt,
`shot_attempts.strategy` and `status` show the chain.

Partial regeneration: the regeneration key is `shot_id`; other shots are untouched. Attempt numbers keep growing per shot
across cycles and invocations (`shot_<n>_a<attempt>.mp4`), so no clip or cost row is ever overwritten.

Some tactic constants are code, not config: the heuristic ratios (0.6, 1.5), the pre-flight
"line fits" probability (0.5), the 0.3 s tail after a TTS voice, the 0.05 s voice-fit tolerance, the
TTS statuses treated as "voice rejected" (400, 422), the job statuses that stop the run instead of
buying another clip (401, 403, 404), the STT statuses that stop even the advisory stray-speech
check (401, 402, 403), the 3 polls a 404 is tolerated for, and the 500 MB download cap.

**Resume.** `hw generate --run-id <existing>` with the same idea reopens the run. A different idea
under the same id, or a dry run resumed for real (or the reverse), is a `ConfigError` before any call.
- `write_script` is skipped when it is `done` with the same input hash — the idea, the `llm` and
  `judge` config, `quality.speech_chars_per_s` and the scriptwriter prompt version — and
  `script.json` exists. Other config edits (resolution, thresholds) keep the script; the shots are
  then re-checked against it.
- The TTS line is reused when `generate_audio` is `done` with the same input hash (text, language,
  TTS provider, model, voice), its file exists and ffprobe still reads it; a damaged file is
  synthesized again (WARNING `stage.resume_mismatch`).
- An accepted clip is reused only if all of these still hold: the file is intact (same size and
  sha256 as recorded; rows from before those columns: the file exists); the current script renders
  the same prompt text and version for that shot; the dialogue shot's strategy matches the run's;
  the resolution matches config; the re-probed clip passes the current duration gate. Otherwise
  WARNING `shot.resume_mismatch` and only that shot is regenerated. A change of `video.model` (or
  vendor) alone does not regenerate a clip; `report.models.video` names the model of the clips used.
- With no accepted clip, the latest paid clip that was never gated (status `generated`, `passed` or
  `unverified`, passing the same checks) is gated instead of paid for again (`shot.resume_pending`).
- `assemble`, the final check and the report always run again; lipsync results come from the cache.
- Not covered: a video job interrupted while it was being polled is lost (paid, not recorded; its id
  is only in `logs.jsonl` as `openrouter.video.submitted`) and that shot is paid again;
  `preflight_rewrites` counts only the current invocation.

## 7. Human-in-the-loop

Deliberately **absent** in the PoC (requirement: "one command → file"). The points where it would exist "for real" are
already in the flow as hooks (`_hitl(ctx, point, ...)`) that only log `hitl.checkpoint`; the config flags
`hitl.<point>.enabled` exist and setting one to `true` makes the run fail with `ConfigError` **before any paid call**,
so a review is never silently skipped:

1. **After script** (`hitl.after_script`) — script approval (the cheapest point; everything before it costs cents).
2. **When speech is `unverified`** (`hitl.on_unverified_speech`) — a human confirms verbatim-ness when STT is unavailable.
3. **Before publish** — final review (outside the PoC).

Inside shot generation there is no HITL on purpose: automatic gates work there; humans only look at metrics.
