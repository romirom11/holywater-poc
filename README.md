# HOLYWATER PoC — idea → vertical video with a verbatim spoken line

One command turns a 1–3 sentence idea into an 8–30 s vertical (9:16) video with sound, in which the
quoted line is spoken **word for word**. An LLM writes a 1–2 shot script but may only *extract* the
line from your text; Seedance 2.5 renders picture and speech in one pass; Whisper transcribes the
clip and a WER gate checks the line; when the video model keeps paraphrasing, the pipeline falls
back to a TTS voice-over of the exact text. Every paid call is followed by a gate, every failure has
a typed reaction, every charge is recorded, and a failed run can be resumed without paying twice.
Orchestration is Prefect, state is SQLite, media is ffmpeg.

```
idea ─► write_script (LLM) ─► preflight (Jev, optional) ─► generate_video, per shot (Seedance 2.5)
                                                             │  duration gate on every clip
                                                             │  dialogue shot: STT + WER gate
                                                             │    ✘ → stricter prompt / wider window → TTS fallback
                                                             ▼
        report.json ◄─ final check (TTS only) ◄─ assemble (ffmpeg: normalize, mix, concat,
                                                   loudnorm, trim, 8–30 s + 9:16 + audio gates)
```

Full stage diagram and the failure → reaction table: [`docs/DATA_FLOW.md`](docs/DATA_FLOW.md).

## Quick start

Requirements: Python 3.12 or 3.13 and [`uv`](https://docs.astral.sh/uv/); `ffmpeg` + `ffprobe` on
`PATH` with libx264 and AAC (`brew install ffmpeg`). Run every command from the repository root:
the config, `.env`, `runs/` and `evals/cases.yaml` are relative paths.

```bash
uv sync                      # add `--extra judge` for the optional TypeSafe Jev judge

# 1) no keys, no cost: the full pipeline with in-process fake providers
uv run hw generate 'Rainy night. A taxi driver glances in the rearview mirror and asks: "Are you sure this is the address?"' --dry-run

# 2) the real thing (spends money, about $1.3 at 480p, see "Costs")
cp .env.example .env         # then set OPENROUTER_API_KEY
uv run hw generate 'In a glass meeting room, a stern hiring manager finishes an AI-made video and smiles for the first time today. She says: "Cancel the other interviews, Roman gets the job."'
```

A dry run takes about 5–15 s and writes a real `final.mp4` made of ffmpeg test-pattern clips, plus
`report.json` (`dry_run: true`). It makes no vendor calls and needs no keys, but it validates the config,
including provider names and capabilities, as a real run does (keys are not checked).

| Key (`.env`) | Used for | Required |
|--------------|----------|----------|
| `OPENROUTER_API_KEY` | LLM, Seedance video, ElevenLabs TTS, Whisper STT — the defaults use only this key | yes |
| `FAL_KEY` | any capability set to `provider: fal` (lipsync is fal-only and off by default) | only then |
| `TYPESAFE_API_KEY` | Jev typed judge (script pre-flight, failure classification); needs `uv sync --extra judge` | no — runs without it |

A blank `KEY=` line counts as missing: a required key left blank stops the run with
`error: missing required secrets: ...` before any call. Without a usable Jev key or SDK the run
continues with a WARNING (`judge.disabled`) and `report.json` says `models.judge: "noop"`.
`HW_CONFIG` and `HW_LOG_LEVEL` are environment variables (or `--config` / `--log-level`); they are
not read from `.env`.

Video, TTS and STT can be pointed at another vendor in [`config/pipeline.yaml`](config/pipeline.yaml)
(the LLM is OpenRouter-only, lipsync is fal-only):

```yaml
video:
  provider: fal                                  # instead of openrouter
  model: "bytedance/seedance-2.5/text-to-video"
audio:
  tts:
    provider: fal
    model: "fal-ai/elevenlabs/tts/eleven-v3"
    voice: "Rachel"
    voices: {}                                   # the default ids (sarah, george) are OpenRouter's
```

## Generating a video

- Wrap the idea in single quotes and the spoken line in double quotes. When the idea quotes
  something, the line must be exactly one complete, contiguous quoted fragment (not shortened, no
  narration): a line split by a speech tag (`"Cancel the interviews," she says, "Roman gets the
  job."`) can only be one of its halves. Without quotes the LLM is asked for one sentence of the
  idea; the check only enforces a verbatim fragment of whole words ([ADR-005](docs/DECISIONS.md)).
- An apostrophe breaks single quotes (`doesn't`, `п'ять`). Pass such ideas on stdin instead:

  ```bash
  uv run hw generate - < idea.txt
  uv run hw generate - --dry-run <<'EOF'
  A tired barista wipes the counter after midnight. He doesn't look up and says: "We're closed, but I'll make you one."
  EOF
  ```

  With no argument and input piped in, stdin is read as well.
- Limits: 10–1000 characters (outside them: a one-line error, exit 1); 1–3 sentences (more is a
  WARNING, the run continues).
- Languages: English and Ukrainian ideas are in the eval set and work end to end with the fakes;
  native Ukrainian speech from Seedance has not been tested with the real model yet.
- Options: `--dry-run`, `--run-id <id>` (resume a run with the same idea, or name a new one: letters,
  digits, `.`, `_`, `-`), `--tag <t>` (group runs for `hw eval compare`; a tag never mixes dry and real
  runs), `--config/-c`, `--log-level`.
- On success stdout carries only the final path; stderr ends with
  `run <id>: 13.1s, strategy=native, line_verified=yes, cost=$1.260, report=runs/<id>/report.json`,
  plus a yellow WARNING when the line could not be verified by STT (the exit code stays 0).
  On failure: `error: ...`, then `run <id> failed; report: ...; logs: ...; resume: same idea with
  --run-id <id>`, exit 1.

## Outputs

```
runs/<run_id>/
├── final.mp4                  # 9:16, h264 + AAC 48 kHz stereo, 8–30 s
├── report.json                # metrics, WER, strategy, costs (docs/QUALITY_METRICS.md §1)
├── script.json                # the script that was actually shot (after any rewrite)
├── logs.jsonl                 # structured log of the run
├── video/shot_<n>_a<k>.mp4    # every paid clip; attempts are never overwritten
├── audio/line_<n>.mp3         # the TTS line (tts_overlay only)
└── work/                      # normalized clips, concat list, STT WAVs, lipsync cache
```

Key `report.json` fields: `success`, `failure_stage`, `dry_run`, `dialogue_line` (the original
characters), `line_verified` (the decisive STT check passed), `audio_strategy_used` and
`fallback_reason`, `shots[].final_wer`, `shots[].stray_speech_words`, `failure_kinds`,
`cost_usd_total` and `cost_summary` (vendor-reported vs estimated, by provider/model/stage/shot
attempt), `prompt_versions`, `models`, `stages[].duration_s`, `latency_s`.

Domain state for every run lives in SQLite, `runs/holywater.db` (runs, stages, shots, shot attempts,
speech verifications, judge decisions, costs). Prefect keeps its own flow-run state in `~/.prefect`;
to browse it, run `uv run prefect server start` and open http://127.0.0.1:4200.

```bash
uv run hw inspect <run_id>                       # print report.json
uv run hw costs <run_id>                         # spend by provider, model, stage and shot attempt
uv run hw generate '<same idea>' --run-id <id>   # resume: reuses the script and every clip that still matches
```

## Samples

| File | Idea → line | Format | Result | Cost | Made by |
|------|-------------|--------|--------|------|---------|
| [`samples/smoke-480p-taxi.mp4`](samples/smoke-480p-taxi.mp4) | `Rainy night. A taxi driver glances in the rearview mirror and asks: "Are you sure this is the address?"` → `Are you sure this is the address?` | 480×854, 12.3 s, 2 × 6 s shots (AAC at 96 kHz, before the 48 kHz fix) | native speech, WER 0.0, no retries, Jev pre-flight ran | $1.26 (video $1.24, LLM $0.017, STT $0.00015); 5.4 min | an earlier revision, commit `aebdfc2` |
| [`samples/hire-roman-480p.mp4`](samples/hire-roman-480p.mp4) ([report](samples/hire-roman-480p.report.json), [script](samples/hire-roman-480p.script.json)) | `In a glass meeting room, … She says: "Cancel the other interviews, Roman gets the job."` → `Cancel the other interviews, Roman gets the job.` | 480×854, 10.1 s, 1 × 10 s shot, AAC 48 kHz stereo | native speech, WER 0.0, no retries, Jev pre-flight ran, 0 warnings | $1.04 (video $1.03, LLM $0.010, STT $0.00008, Jev $0.00006 est.); 2.8 min | commit `f0d289e` |

The second sample is the first real run of the current code (clean tree, so `report.json` names
`git_sha: f0d289e`). Before it, the code had been exercised only with fakes and with the real vendor
classes over mock transports; that replay sent the same request bodies as the earlier real run
except for the intended changes (the fixed `polling_url` handling, the `scriptwriter@3` system
prompt, the required `voice_gender` field in the script JSON schema, and an STT stray-speech check
per speechless clip). The LLM chose one 10 s shot, so this run had no speechless clip.

## How it stays reliable

- **Verbatim by construction** — the line comes from your text, the code rejects anything but one
  complete quoted fragment, and only your original characters go downstream ([ADR-005](docs/DECISIONS.md)).
- **Verbatim by verification** — STT + WER gate on the dialogue clip; a typed failure kind picks the
  tactic (stricter prompt, wider line window, listen again, or TTS fallback); with TTS the assembled
  final is checked once more ([ADR-002](docs/DECISIONS.md), [DATA_FLOW §6](docs/DATA_FLOW.md)).
- **A gate after every paid call** — clip length, final length exactly 8–30 s (trimmed, or the
  shortest shot without the verified line extended), 9:16 with audible audio, TTS voice fits its
  shot, no stray speech (warned; the clip under a TTS voice is regenerated).
- **Retries without double payment** — transient errors are retried by Prefect (2 retries, 5 s and
  15 s); polls and downloads are retried inside the vendor against the same job, never by paying for
  a new one; a billed job whose result was lost or unreadable is recorded and the shot regenerated
  within its budget, or the run stops when the job endpoints answer 401/403/404
  ([ADR-010](docs/DECISIONS.md)).
- **Nothing lost** — typed errors, every fallback recorded in the DB, every charge a `costs` row,
  `report.json` written even when the run fails.
- **Resume** — `--run-id` reuses the script and every clip whose prompt, strategy, resolution,
  length and file hash still match (a `video.model` change alone does not regenerate a clip;
  `report.models.video` names the model of the clips used); no accepted or pending clip is paid for
  again ([DATA_FLOW §6 Resume](docs/DATA_FLOW.md)).

## Evals

```bash
uv run hw eval run --tag v1 --dry-run     # the 8 ideas in evals/cases.yaml with fakes: free, ~20 s
uv run hw eval compare v1 v2              # success, line_verified and fallback rates, WER mean/p95, cost per tag
```

**Cost warning:** `hw eval run --tag <t>` without `--dry-run` is **8 paid generations** — about
$7–25 for video alone at 480p, more with regenerations. It prints the estimate and asks before
starting (`--yes` skips the question). So far the suite has run only in dry-run. Details:
[`docs/QUALITY_METRICS.md`](docs/QUALITY_METRICS.md) §3.

## Costs

Measured on the default path (OpenRouter, 480p), both runs with no retries: **$1.26 for 2 × 6 s
shots** and **$1.04 for 1 × 10 s shot**. Seedance 2.5 costs ≈ $0.103 per second ($0.62 per 6 s,
$1.03 per 10 s) and is almost the whole bill; the LLM ≈ $0.01–0.02, STT ≈ $0.0001 per check, Jev
≈ $0.00006 per run (an estimate: input tokens × $0.042/M from the price list; the API reports
tokens, not a charge). Every regenerated or fallback clip adds its video cost again; 720p costs
about 2.25× per second (`video.resolution: 720p`). The real runs took 5.4 and 2.8 minutes, mostly waiting for the video
model. `hw costs <run_id>` shows where the money went.

## How it was built with agents

The code was written by coding agents (Claude Code) under a harness the author set up first:
- **Docs first.** [`docs/SPEC.md`](docs/SPEC.md) (approved before any code), ADRs in
  [`docs/DECISIONS.md`](docs/DECISIONS.md), and [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md)
  with a Definition of Done per step, ordered bottom-up so that only steps 10 and 13 cost money.
- **[`CLAUDE.md`](CLAUDE.md)** — the working agreement: non-negotiables (the line is never generated,
  no paid call without a gate, no lost errors, config over code, pure stages, capabilities as
  Protocols), the layering rule, the Definition of Done and the cost rules.
- **Self-checks the agent must run** — `ruff`, `mypy --strict`, `pytest` (unit tests without network
  or ffmpeg; integration tests with ffmpeg and a Prefect test server), and the `--dry-run` smoke test
  of the whole flow. Fakes in `providers/fake.py` have failure knobs, so the DATA_FLOW §6 tactics
  are tested at flow level without spending money.
- **Vendor conformance from schemas** — request bodies are checked against the vendors' OpenAPI
  specs ([`docs/PROVIDERS_API.md`](docs/PROVIDERS_API.md)) and pinned in unit tests; a 2xx from a
  live call is not taken as proof.
- **Audit, then parallel fixes.** Before submission, review agents audited the repo by dimension
  (architecture, data flow, providers, quality, harness, README, task fit) and wrote a reproduction
  script for every finding; a verification pass confirmed or corrected each one. Fix packages then
  ran in parallel on disjoint files, each re-running its repros and the four checks, with an
  integration gate after each round; these docs were synced last.

## Known limitations

- **WER 0.10.** A line under 10 words must match exactly, but at 10–19 words one wrong word passes
  (dropping "never" from a 16-word line passes at 0.062). The fix is a stronger STT or an ensemble,
  not a stricter threshold on Whisper ([ADR-009](docs/DECISIONS.md)).
- **Whisper.** It can mishear names and numbers (normalization covers digits, currency, ordinals and
  Ukrainian numerals, not contractions or years spelled as words), and it can hear words in ambience,
  which is why the stray-speech check only warns (except on the clip under a TTS voice, which is
  regenerated). `stt.language: auto` tells Whisper the line's language rather than
  detecting it, so spotting `wrong_language` depends on Jev.
- **Without Jev** failure kinds come from word-count heuristics; `paraphrased`, `wrong_language` and
  `stt_noise` (and their tactics) never occur.
- **Thin real-world evidence.** The real runs are the ones under "Samples"; the eval suite has run
  only with fakes; Ukrainian native speech and fal lipsync are untested; fal lipsync is not priced
  (recorded as $0 with a WARNING).
- **TTS fallback is a voice-over.** Under `tts_overlay` the clip is generated with "no one speaks";
  without lipsync (off by default) nobody mouths the line on screen.
- **Shots run one after another**, so a 2-shot run takes about twice the video time.
- **Integrity is checked on the final only**: a landscape clip, or a mute clip of the shot without
  dialogue, is letterboxed or padded with silence before the final's 9:16 and audio gates see it.
- **A job lost mid-poll** (the process killed while waiting) is not recorded and is paid again on
  resume; its id is only in `logs.jsonl` (`openrouter.video.submitted`). On fal, a transient error
  after enqueue also resubmits.
- **Prefect** starts an ephemeral local server (127.0.0.1) for every `hw generate` and `hw eval run`,
  dry runs included (the CLI switches its telemetry off); there is no UI unless you run
  `uv run prefect server start`; flow-run state accumulates in `~/.prefect`.
- **No CI yet** — the checks run locally (see "Development").

## Docs

| Document | What is in it |
|----------|---------------|
| [`docs/SPEC.md`](docs/SPEC.md) | goal, hard requirements and how each is verified, stack |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | layout, layers, config, error taxonomy, logging, storage, guardrails |
| [`docs/DATA_FLOW.md`](docs/DATA_FLOW.md) | stages, artifacts, script schema, prompts, failure → reaction table, resume |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | ADRs: Prefect, audio strategy, Seedance, OpenRouter, verbatim line, SQLite, Jev, vendors, WER, retries, stray speech |
| [`docs/QUALITY_METRICS.md`](docs/QUALITY_METRICS.md) | report fields, content metrics, eval process |
| [`docs/PROVIDERS_API.md`](docs/PROVIDERS_API.md) | every vendor request checked against its OpenAPI schema; error mapping; prices |
| [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) | build steps with Definition of Done and status |
| [`CLAUDE.md`](CLAUDE.md) | working agreement for the coding agents |

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy                  # strict; passes with or without the judge extra
uv run pytest                # unit + integration (ffmpeg, Prefect flow with fakes); no network
uv run pytest tests/unit     # about 3 s, no Prefect server
```
