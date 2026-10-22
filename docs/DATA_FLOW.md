# Data flow: from idea to video

## 1. Stage diagram

```
 IdeaInput ──► [1] write_script ──► Script ──► [1b] preflight_script (Jev)
                                                  │  ✘ safety risk / not filmable → LLM rewrite → [1b]
                                                  │  ✔
                      ┌───────────────────────────┴───┐
                      ▼                               ▼
            [2a] generate_audio (TTS)        [3] generate_video (per shot)
             only if tts_overlay                  Seedance 2.5 → ShotVideo[]
                      │                               │
                      │                               ▼
                      │                     [4] verify_speech (STT + WER)
                      │                       ✔ pass ─────────────┐
                      │                       ✘ fail → Jev classifies failure_kind → shot retry tactic
                      │                       ✘ exhausted / wrong_language → switch to tts_overlay (→ 2a, 3 without dialogue)
                      ▼                                           ▼
                [5] assemble ◄────────────────────────────────────┘
                 normalize → (mux TTS → lipsync, dialogue shot only) → concat → loudnorm → final.mp4
                      │
                      ▼  tts_overlay only: [4] verify_speech on the line window of final.mp4
                [6] report → report.json, runs.status = done
```

For the `native` strategy stage 2a is recorded as `skipped` (it runs for real only if a shot falls back to TTS).
For `tts_overlay` stage 4 runs once more on the assembled final — only the line window, with 0.5 s of margin — as a
sanity check: TTS is deterministic, but STT confirms the mux/lipsync/loudnorm did not break the audio. A failed check
fails the run (`QualityGateError`, stage `verify_speech`); an unavailable STT leaves `report.line_verified=false`.

## 2. Artifacts per stage

Media and large JSON — in `runs/<run_id>/`; metadata, statuses, metrics — in SQLite (ARCHITECTURE §7):

| Stage | File on disk | DB rows | Key fields |
|-------|--------------|---------|------------|
| 0 input | — | `runs` | `input_text`, `config_hash`, `git_sha` |
| 1 | `script.json` (`Script`) | `stages`, `shots` | see §3 |
| 1b | — | `judge_decisions` | `question`, `answer`, `probabilities`, rewrite/pass decision |
| 2a | `audio/line_<shot>.mp3` | `stages`, `costs` | `path`, `duration_s`, `model`, `voice`, `text` |
| 3 | `video/shot_<n>_a<attempt>.mp4` | `shot_attempts`, `costs` | `seed`, `provider_request_id`, `path`, `duration_s`, `resolution`, `cost_usd` |
| 4 | — | `speech_verifications`, `judge_decisions` | `expected`, `transcript`, `wer`, `passed`, `failure_kind` |
| 5 | `final.mp4` (h264/aac 9:16) | `runs.final_path` | — |
| 6 | `report.json`, `logs.jsonl` | `runs.status=done`, `cost_usd_total` | `RunReport` — DB export for the note / eval |

## 3. Script schema (`Script`)

LLM output via structured output; the fields that go to the video model are deliberately close to the format Seedance
understands (continuity / timeline / audio).

```python
class Dialogue(BaseModel):
    shot_id: int
    speaker: str                 # "woman in red coat"
    text: str                    # VERBATIM from IdeaInput.text — validated by code
    language: str                # "uk", "en"
    emotion: str                 # "tired, quiet"
    start_s: float; end_s: float # when in the shot it is spoken

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

Invariants (checked by code after the LLM; a violation → LLM retry with feedback):
- `normalize(dialogue.text) in normalize(input.text)`;
- `8 <= sum(duration_s) <= 30`; `1 <= len(shots) <= 2`;
- exactly one shot with `has_dialogue=True`, and it equals `dialogue.shot_id`;
- `end_s - start_s` ≥ speech time estimate (chars / ~15 chars per s) — so the line fits.

After validation `Script.dialogue.text` **is overwritten with the original fragment from the input** (ADR-005).

## 4. Seedance prompt (assembled by code, not by the LLM)

`stages/generate_video.py` renders a template from `Shot` + `Dialogue` fields:

```
[Continuity] {continuity}. Style: {style}. Vertical 9:16.
[Timeline] 0–{start_s}s: {visual_prompt}, {camera}. {start_s}–{end_s}s: {speaker} says "{dialogue.text}" ({emotion}), mouth moves only during the line. {end_s}–{duration_s}s: ...
[Audio] clean lip-synced speech in {language}; {ambient_audio}; no music.
```

For `tts_overlay`: the same template, but `[Audio] no dialogue, no speech; {ambient_audio}`. The template is a file in `prompts/`.

## 5. State and stitching stages together

- Stages are pure functions (`stages/*`): they take explicit inputs and provider deps, write their media file, and return a pydantic artifact with the path. They never touch SQL.
- The flow owns the state: `_stage(ctx, name)` marks `stages.start/finish` around each task, and right after a task returns the flow writes the artifact's rows (`shot_attempts`, `speech_verifications`, `costs`, …). Each write is its own small transaction; there is no multi-table transaction, and a crash between two writes is recovered by resume (below), not by rollback.
- A Prefect task is a thin wrapper around one stage call; retries with backoff live on the task, decisions live in the flow. Tasks pass pydantic models with paths/ids between them, not binary data.
- The current `Script` lives in the flow context and on disk (`script.json`); every tactic that changes it (prompt softening, widening the line window, extending a shot) rewrites both and the `shots` rows, so the file always shows the version that was actually shot.
- Generated media is downloaded immediately (vendor URLs are temporary) — each vendor's `_download()` checks `content_type` and size.
- Before STT the audio track is extracted to a mono 16 kHz WAV (`work/<clip>.stt.wav`) so every STT vendor receives the same input.

## 6. Error handling per stage

| Stage | Failure | Reaction |
|-------|---------|----------|
| 1 script | invalid JSON / invariant violation | LLM retry with the error text (up to `llm.max_attempts`) → `ScriptError` fatal |
| 1 script | 429/5xx OpenRouter | `RetryableProviderError` → Prefect backoff |
| 1b preflight | Jev: safety risk > threshold or filmability < threshold | LLM rewrite of the shot prompt with Jev feedback (up to `llm.max_attempts`) → then `ScriptError` |
| 1b preflight | Jev unavailable | `WARNING`, stage `skipped`, continue without pre-flight |
| 3 video | 429/5xx/timeout from the vendor | Prefect backoff (`retries.provider`); each retry is counted in `stages.retries` |
| 3 video | prompt rejected (content policy / 400, or the job fails) | `FatalProviderError` → one LLM rewrite of **that shot only** (`scriptwriter` prompt, `soften` section), then the same attempt is retried; a second rejection is fatal. Statuses 401/402/403/404 are not softened (credentials/billing/endpoint, not the prompt) |
| 3 video | clip duration ≠ requested ± `quality.shot_duration_tolerance_s` | `DurationGate` in the flow → attempt marked `failed`, shot regenerated (counts against `video.max_attempts` / `audio.native_attempts`); exhausted → `QualityGateError` |
| 4 verify | `wer > max_wer` | Jev `Choice` → `failure_kind`; tactic: `paraphrased/extra_words/silence` — regenerate **only this shot** with a stricter prompt (vendor picks a new seed); `truncated` — widen the line window by `quality.truncated_window_widen_s` on each side and regenerate with the same prompt (falls back to the strict prompt when the window cannot grow); `wrong_language` — fall back to `tts_overlay` immediately; `stt_noise` — listen again (one STT re-run on the same clip, no video regenerated), then continue as `paraphrased`; `unknown` (Jev disabled) — default: stricter prompt. Limit `audio.native_attempts` → `tts_overlay` fallback for this shot |
| 4 verify | STT failed | Prefect retry; if STT stays unavailable — `WARNING`, the attempt is marked `unverified`, the shot falls back to `tts_overlay` (the only path that does not need STT) and `report.line_verified` stays `false`: the file exists but R4 is **not evidenced** |
| 5 assemble | ffmpeg error | `MediaError` → Prefect retry (`retries.provider`) → fatal |
| 5 assemble | final duration > max | trim to `max` from the end — unless that would cut the line (`protect_until_s` = dialogue shot offset + `dialogue.end_s`), then `QualityGateError` |
| 5 assemble | final duration < min | regenerate the **shortest** shot once with `duration_s + ceil(deficit) + 1` (capped at 30 s) and assemble again; still short → `QualityGateError` |

Partial regeneration: the regeneration key is `shot_id`; other shots are untouched. Attempt numbers keep growing per shot
across cycles (`shot_<n>_a<attempt>.mp4`), so no clip or cost row is ever overwritten.

Resume: `hw generate --run-id <existing>` with the same idea reopens the run. `write_script` is skipped when its row is
`done` with the same `input_hash` (idea + config hash) and `script.json` exists; shots whose `final_attempt_id` clip is on
disk are reused; the TTS line is reused if its file exists; `assemble`, the final check and the report always run again.
Nothing is paid twice. A different idea under the same run id is a `ConfigError`.

The audio fallback chain is recorded in the DB: `runs.audio_strategy_used: native | tts_overlay`, `runs.fallback_reason`, and per attempt — `shot_attempts.strategy`.

## 7. Human-in-the-loop

Deliberately **absent** in the PoC (requirement: "one command → file"). The points where it would exist "for real" are
already in the flow as hooks (`_hitl(ctx, point, ...)`) that only log `hitl.checkpoint`; the config flags
`hitl.<point>.enabled` exist and setting one to `true` makes the run fail with `ConfigError` **before any paid call**,
so a review is never silently skipped:

1. **After script** — script approval (the cheapest point; everything before it costs cents).
2. **After verify_speech when `unverified`** — a human confirms verbatim-ness when STT is unavailable.
3. **Before publish** — final review (outside the PoC).

Inside shot generation there is no HITL on purpose: automatic gates work there; humans only look at metrics.
