# Metrics and quality evaluation

## 1. Technical pipeline metrics (automatic, in `report.json`)

| Metric | Source | Why |
|--------|--------|-----|
| `success` (bool) + `failure_stage/type` | `runs`, `stages` | success rate, where it breaks |
| `latency_s` total and per stage | structlog timings | where the bottleneck is |
| `attempts` per stage, `retries_total` | `stages` | provider stability |
| `cost_usd` per stage and total | estimate from price list (sec × $/s, chars × $/1k) | video budget |
| `audio.strategy_used`, `fallback_reason` | `runs` | how often native fails R4 |
| `wer` (per shot and final) | SpeechGate | the main R4 gate |
| `final.duration_s`, `resolution`, `fps`, `has_audio` | ffprobe | R2/R4 |
| `shots_regenerated`, `failure_kind` distribution | flow, `speech_verifications` | the cost of quality; why native fails |
| `preflight_rewrites`, `preflight_saved_usd` (estimate) | `judge_decisions` | whether Jev pre-flight pays for itself |
| `config_hash`, `prompt_versions`, `models` | config | to compare pipeline versions |

## 2. Content quality metrics

**Automated (PoC):**
- Verbatim line — WER after normalization (case, punctuation, numbers as words). Threshold 0.10, target 0.
- Duration / format / audio track present — ffprobe.
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

- `evals/cases.yaml`: 8–10 ideas of varying difficulty: short/long line, uk/en, 1/2 shots, numbers and names in the line, a "hard" scene (two people).
- `hw eval --cases evals/cases.yaml --tag v0.2` runs the pipeline on each, stores results in the DB and exports `report.json` to `evals/results/<tag>/`.
- `hw eval compare v0.1 v0.2` — table: success rate, median latency, cost/run, WER mean/p95, fallback rate.
- Regression = a drop in success rate or a rise in WER p95 / cost beyond a threshold → the prompt/model change is not merged.
- Pipeline version = `config_hash + prompt_versions + git sha` — all in the report, comparisons by them.

## 4. First week vs someday

**First week (in this PoC):**
- `report.json` with all technical metrics from §1.
- WER gate + duration gate.
- `evals/cases.yaml` with 8–10 cases and `hw eval` with two-tag comparison.
- Cost tracking per stage.

**Someday:**
- VLM judge for visual adherence and consistency.
- Human rating UI (simple Streamlit / Prefect artifact) with 1–5 scores → correlate with auto metrics.
- Prefect Cloud / metrics dashboard over time, alerts on success-rate drops.
- A/B of models (Seedance 2.5 vs 2.0 vs Veo) on the same set with the same budget.
