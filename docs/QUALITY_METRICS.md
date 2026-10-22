# Metrics and quality evaluation

## 1. Technical pipeline metrics (automatic, in `report.json`)

`report.json` is exported from the DB at the end of every run, including failed ones
(`domain/report.py::RunReport`, `stages/report.py`). `hw inspect <run_id>` prints it.

| Field | Source | Why |
|-------|--------|-----|
| `success`, `status`, `failure_stage`, `failure_type` | `runs`, `stages` | success rate, where it breaks. `failure_stage` is the innermost stage that failed (e.g. `verify_speech` inside `generate_video`); null on success |
| `dry_run` | `runs.dry_run` | fake providers: no vendor was called, the numbers are not real |
| `latency_s` | `runs.started_at/finished_at` | end-to-end time (on a resumed run it includes the pause between invocations) |
| `stages[]`: `entries`, `retries`, `duration_s`, `last_duration_s`, `error_type` | `stages` | where the time goes. `duration_s` is wall time summed over all entries of the stage (`stages.total_duration_s`); nested stages are inclusive (`generate_video` contains its `verify_speech` checks, `assemble` contains a shot extension). `entries` is how often the flow entered the stage; `retries` is Prefect task retries on transient errors |
| `retries_total` | sum of `stages.retries` | provider stability |
| `cost_usd_total`, `costs[]`, `cost_summary.{actual_usd, estimated_usd, by_provider, by_model, by_stage, by_cycle}` | `costs`: one row per billed call — LLM answers (usable or not), video jobs (including failed and abandoned ones that carry a charge), STT, TTS, Jev (estimate), lipsync | budget per video, per retry cycle, per vendor; `hw costs <run_id>`. OpenRouter reports the real charge for LLM/video/STT; TTS, fal and Jev are estimates; fal lipsync is not priced yet ($0 + WARNING) |
| `audio_strategy_requested`, `audio_strategy_used`, `fallback_reason` | `runs` | how often and why native fails R4 (`stt unavailable (…)`, `wrong_language`, `exhausted N native attempts (last failure: …)`, `video prompt rejected`) |
| `line_verified` | the latest line check (`expected` ≠ `""`) on the dialogue shot's accepted attempt: the native clip, or for `tts_overlay` the check on the assembled final; true when it was verified and passed | R4 evidence. False when STT was unavailable even if the run finished (the CLI then prints a warning) |
| `shots[]`: `has_dialogue`, `attempts`, `final_attempt`, `final_strategy`, `final_wer`, `verified`, `failure_kinds`, `stray_speech_words`, `cost_usd` | `shots`, `shot_attempts`, `speech_verifications` | per shot: WER of the decisive check, why attempts failed, words heard on a clip that must be speechless (null when not checked or STT was down), video cost |
| `shots_regenerated`, `failure_kinds` | `shot_attempts`, `speech_verifications` | the cost of quality; why native fails (run-level counts over every speech check, stray-speech checks included) |
| `judge_decisions`, `preflight_rewrites` | `judge_decisions`, flow | how often Jev was consulted and how many pre-flight rewrites it caused (counted per invocation) |
| `config_hash`, `git_sha`, `prompt_versions`, `models` | config, git, `shot_attempts` | to compare pipeline versions. `git_sha` is the short HEAD sha, with `-dirty` when the working tree had uncommitted or untracked changes (the code that ran is then not that commit). `prompt_versions` has `scriptwriter@N` and every `seedance_shot@N/<variant>` used; `models.video` names the model of the accepted clips (config's when none was accepted); `models.judge` names the judge that actually ran (`jev-latest`, `noop`, `disabled`; dry runs show `fake/<model>`) |
| `final.{duration_s, width, height, fps, has_audio}` | ffprobe | R2/R4 |

Not produced (later): a `preflight_saved_usd` estimate of what Jev pre-flight saved, and
per-stage latency percentiles across runs.

## 2. Content quality metrics

**Automated (PoC):**
- Verbatim line — WER after normalization (`quality/text_norm.py`: case, punctuation, apostrophes and
  dashes, numbers / currency / percent / ordinals / times as words, thousands separators, Ukrainian
  numerals in any case or gender, "OK"/"okay", "alright"/"all right", "e-mail"/"email", Ukrainian
  "7.00" as a time). A Ukrainian ordinal or "процент" counts as the cardinal or "відсоток" only
  against digits on the other side ("О 7 ранку" == "О сьомій ранку", but "перший" != "один"). Gate `quality.max_wer: 0.10`, target 0. In words: a line under 10
  normalized words must match exactly; at 10–19 words one word may differ, at 20–29 two — and a
  one-word slip can change the meaning (dropping "never" from the 16-word `en_long_line` gives 0.062
  and passes). The grown-up fix is a stronger STT or an ensemble, not a stricter threshold (ADR-009).
  Known normalization gaps: contractions ("you'd" vs "you would"), years read in pairs ("1990" vs
  "nineteen ninety"), "three and a half", "Mr." vs "Mister", English "7.30" as a time, compound
  numerals ("5-річний" vs "п'ятирічний").
- Duration / format / audio — `DurationGate` per clip and on the final, `MediaIntegrityGate` on the
  final (9:16, known frame rate, audio peak ≥ `quality.min_audio_peak_db`).
- Stray speech on clips that must be speechless — STT word count (warn by default; the clip under a
  TTS voice is always regenerated, ADR-011).
- Whether the line is spoken in the planned window (STT with timestamps vs `start_s/end_s`) — "someday".

**Automated (next step, LLM/VLM-as-judge):**
- Visual adherence to the prompt (VLM over 3–5 frames: character, location, action) — 1–5 scale.
- Character consistency across shots (VLM frame comparison).
- Lip-sync quality (VLM / dedicated model on the line segment).
- Generation artifacts (extra hands, morphing) — VLM checklist.

**Manual (stays with a human):**
- Whether the scene "works" dramatically, whether there is emotion in the voice.
- Fit with the vertical-series brand style.
- The final "shippable / not" verdict.

## 3. Eval process

- `evals/cases.yaml`: 8 ideas of varying difficulty: short/long line, uk/en, numbers and names in the
  line, two people in the scene, one idea without quotes.
- `uv run hw eval run --tag v0.2 [--cases evals/cases.yaml] [--dry-run] [--yes]` runs every case under
  the tag, prints `ok`/`FAIL` with the run id per case and the tag summary. Without `--dry-run` every
  case is a paid generation: the command prints the count and a rough video-only cost (8 cases at
  480p: about $7–25) and asks for confirmation; `--yes` skips the question, `--dry-run` never asks.
- Results stay where every run's results are: rows in `runs/holywater.db` and `runs/<run_id>/report.json`.
  There is no separate per-tag export.
- `uv run hw eval compare v0.1 v0.2` — one column per tag: `dry_run`, `runs`, `success_rate`,
  `line_verified_rate`, `fallback_rate`, `mean_wer`, `p95_wer` (nearest rank: the max for n < 20),
  `wer_samples`, `mean_cost_usd`, `mean_shot_duration_s`, `mean_cost_usd:<provider>`,
  `failure_kind:<kind>`. Computed in `RunRepository.tag_stats` from the tables, no file parsing.
- A tag pools every run ever recorded under it (repeats, resumes, failures) and is either all dry
  runs or all real runs (mixing is refused). Use a new tag per pipeline version.
- Regression policy (manual): a drop in success rate or `line_verified_rate`, or a rise in WER p95 or
  cost, blocks the prompt/model change until explained.
- Pipeline version = `config_hash + prompt_versions + git_sha` — all in the report (a `-dirty`
  sha means uncommitted code: commit before a run whose numbers you want to attribute).

Status: the suite has run only in dry-run (8/8 with fakes). No real eval suite has been paid for yet.

## 4. First week vs someday

**First week (in this PoC):**
- `report.json` with the technical metrics of §1, for every run including failed ones.
- WER gate, duration gates, media integrity gate, stray-speech check.
- `evals/cases.yaml` with 8 cases, `hw eval run` and `hw eval compare` across tags.
- Cost tracking per call, stage, shot attempt and vendor (`hw costs`).

**Someday:**
- VLM judge for visual adherence and consistency.
- A stronger STT (or two that must agree) for the verbatim gate.
- Human rating UI (simple Streamlit / Prefect artifact) with 1–5 scores → correlate with auto metrics.
- Prefect Cloud / metrics dashboard over time, alerts on success-rate drops.
- A/B of models (Seedance 2.5 vs 2.0 vs Veo) on the same set with the same budget.
