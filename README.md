# auto-skill

Few-shot skill induction from user examples.

Core question:

> Given only several user examples, can we induce a reusable skill module that improves heldout tasks compared with prompt-only, examples-only, and one-shot skill drafting?

Current project scope has two tracks:

1. Data cleaning scripts: normalize WritingBench / PresentBench into reproducible train-example and heldout-task splits.
2. Auto-skills module: compile user examples into a structured skill package with feature extraction, cross-example aggregation, validation notes, and generated `SKILL.md`.

The auto-skill module must treat user examples as the only induction input. Benchmark rubrics/checklists may be used only for private data construction checks and heldout evaluation. They must never be normal skill-induction input; any future oracle or upper-bound experiment must be named and isolated separately.

## Repository Layout

```text
src/auto_skill/        Core library code.
scripts/               CLI entrypoints; see scripts/README.md for inventory and lifecycle.
tests/                 Minimal regression tests for shared logic and scripts.
notes/                 Research notes and design sketches.
docs/                  Stable repo/project docs for collaborators.
artifacts/             Small derived samples checked in while prototyping.
data/                  Local benchmark checkouts; ignored.
```

## Local Data

- WritingBench: `../WritingBench` in the current local workspace, or `data/WritingBench` if you prefer a repo-local checkout.
- PresentBench: `data/PresentBench_repo`

`data/` is ignored because PresentBench is large. Recreate it with:

```bash
git clone https://huggingface.co/datasets/PresentBench/PresentBench data/PresentBench_repo
git -C data/PresentBench_repo lfs install --local
git -C data/PresentBench_repo lfs pull
git -C data/PresentBench_repo lfs checkout
git clone --depth 1 --filter=blob:none --sparse https://github.com/PresentBench/PresentBench.git data/PresentBench_code
git -C data/PresentBench_code sparse-checkout set --skip-checks README.md judge.py judge_all.py scoring.py scoring_all.py utils
```

Current artifacts were built from a sibling WritingBench checkout:

```bash
git clone https://github.com/X-PLUG/WritingBench.git ../WritingBench
```

If you keep WritingBench under `data/`, pass `--writingbench-root data/WritingBench`
when rebuilding splits.

Repo-local alternative:

```bash
git clone https://github.com/X-PLUG/WritingBench.git data/WritingBench
```

## Setup

```bash
uv sync --extra dev
cp .env.example .env
```

Official PresentBench judging also needs the upstream judge runtime:

```bash
uv sync --extra dev --extra presentbench
```

Fill `.env` with the Bailian OpenAI-compatible endpoint, key, and model:

```dotenv
BAILIAN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
BAILIAN_API_KEY=replace-with-your-api-key
BAILIAN_MODEL=qwen3.5-plus
# Optional: BAILIAN_TEMPERATURE=0.2
# Optional: BAILIAN_NUM_THREADS=16
# Optional: BAILIAN_STREAM=true
```

The minimal test path uses only the Python standard library:

```bash
uv run python -m unittest discover -s tests
```

## Data Workflow

Inspect benchmark structure:

```bash
uv run python scripts/data/inspect_benchmarks.py \
  --writingbench-root ../WritingBench \
  --presentbench-root data/PresentBench_repo \
  --limit 2 \
  --out-dir artifacts
```

Build few-shot splits:

```bash
uv run python scripts/data/build_fewshot_splits.py \
  --writingbench-root ../WritingBench \
  --presentbench-root data/PresentBench_repo \
  --train-size 3 \
  --heldout-size 2 \
  --max-groups 4 \
  --out artifacts/splits/fewshot_splits.jsonl \
  --summary-out artifacts/splits/fewshot_split_summary.json
```

For the next non-MVP expansion, prefer a stratified medium run rather than a
full benchmark clean:

```bash
uv run python scripts/data/build_fewshot_splits.py \
  --writingbench-root ../WritingBench \
  --presentbench-root data/PresentBench_repo \
  --train-size 3 \
  --heldout-size 2 \
  --max-writing-groups 30 \
  --max-present-groups 20 \
  --out runs/expanded/fewshot_splits.30wb_20pb.jsonl \
  --summary-out runs/expanded/fewshot_split_summary.30wb_20pb.json
```

Only move to full cleaning after this expanded split passes
`validate_splits.py`, `audit_benchmark_flow.py`, and a sampled example-quality
audit with acceptable failure rate.

Current expanded-run evidence is summarized in
`notes/expanded_cleaning_strategy_2026-05-08.md`; artifact hashes and handoff
commands are in `docs/expanded_cleaning_manifest_2026-05-08.md`. The short
version: Qwen3.5-Plus completed the local 30WB/20PB clean with 150/150 latest
generation success and a passing benchmark-flow audit. MIMO is better used as an
independent audit or targeted regeneration model until its long PresentBench
request failures are resolved.

To re-check the local expanded-cleaning handoff state:

```bash
uv run python scripts/ops/report_expanded_cleaning_status.py --expect-status ready
```

The split builder excludes known source-data mismatches, currently
`education/CSAPP-Lectures_2015Fall/Lecture15`, whose instructions ask for
Chapter 10 System-Level I/O while its judge checklist scores Chapter 8
Exceptional Control Flow.
It also writes a machine-readable selection summary with the total group
denominator, eligible groups, skipped-too-small groups, and max-group truncation.

Validate split schema before pushing:

```bash
uv run python scripts/data/validate_splits.py artifacts/splits/fewshot_splits.jsonl
```

Construct clean example packs and private generation/eval artifacts:

```bash
uv run python scripts/data/build_example_packs.py \
  --splits artifacts/splits/fewshot_splits.jsonl \
  --packs-out artifacts/packs/example_packs.jsonl \
  --private-out artifacts/private/example_private_eval.jsonl \
  --jobs-out artifacts/jobs/example_generation_jobs.jsonl \
  --summary-out artifacts/reports/example_pack_summary.md
```

This writes the local construction layer:

- `artifacts/packs/example_packs.jsonl`: intermediate clean user-visible packs before desired outputs are applied; ignored by git.
- `artifacts/jobs/example_generation_jobs.jsonl`: private jobs for generating desired outputs/examples; ignored by git.
- `artifacts/private/example_private_eval.jsonl`: private rubric/judge/checklist metadata for construction and evaluation only.
- `artifacts/reports/example_pack_summary.md`: compact dataset summary; ignored by git.

Run desired-output generation jobs through Bailian:

```bash
uv run python scripts/data/run_generation_jobs.py \
  --jobs artifacts/jobs/example_generation_jobs.jsonl \
  --out artifacts/jobs/generated_desired_outputs.jsonl \
  --config-prefix MIMO \
  --limit 2 \
  --num-threads 16 \
  --timeout-seconds 600 \
  --max-retries 0
```

Use `--config-prefix MIMO` or `--config-prefix QWEN` when constructing a
parallel set of examples from another OpenAI-compatible model configured in
`.env` (`MIMO_MODEL`, `MIMO_BASE_URL`, `MIMO_API_KEY`, etc.). The default path
still uses the legacy `BAILIAN_*` / `OPENAI_*` chain.

The script prints per-job `duration_seconds` and total jobs/minute. Current
Qwen3.5-Plus PresentBench runs are latency-bound rather than RPM-bound; use the
timing plus observed 429/timeout rate to decide whether to increase toward
`--num-threads 32`.
For long sequential skill-induction runs, pass `--stream` to consume Qwen output
incrementally. `--stream-log` prints deltas to stderr and should only be used for
single-run debugging, not threaded batch generation.
The skill MVP runner writes an append-only stage ledger next to `--out`; after
an API failure or interruption, rerun with `--resume` to reuse successful stage
calls and completed skill rows. Do not globally enable Qwen thinking for skill
induction. Start with `--no-enable-thinking`; use
`--thinking-stages ... --thinking-budget N` only for targeted analysis/merge
A/B runs.

For long MVP experiments, use the thin Bash orchestrator rather than manually
retyping every command:

```bash
RUN_SKILL_MVP=0 RUN_MEMORY=1 RUN_WRITINGBENCH=1 RUN_PRESENTBENCH=1 RUN_VALIDATE=1 \
  PRESENTBENCH_NUM_THREADS=4 scripts/ops/run_mvp_pipeline.sh
```

It sequences skill induction, extraction-memory update, WritingBench eval,
PresentBench surrogate eval, validation, timestamps, and per-step logs under
`logs/mvp/`. The Python CLIs remain the behavior source; the shell script only
manages long-running tasks.
PresentBench surrogate eval uses cell-level `--num-threads`: each
pack/task/mode cell runs independently, but generation then judge remains
sequential inside a cell and checkpoint writes stay in the main process.

Extraction memory is written to ignored local output
`artifacts/memory/extraction_memory.v1.jsonl`. It is limited to public
user-example-derived lessons from successful skill rows and must not contain
private rubrics, heldout feedback, judge traces, or scores.

Current stable MVP run:

```bash
uv run python scripts/skills/run_skill_mvp.py \
  --packs artifacts/packs/example_packs.v1.jsonl \
  --out runs/skill_mvp.qwen.mvp.jsonl \
  --stream \
  --resume \
  --allow-partial \
  --no-enable-thinking \
  --no-leave-one-out \
  --temperature 0.2 \
  --json-temperature 0 \
  --timeout-seconds 900 \
  --max-retries 0 \
  --max-tokens 8192
```

Provider retries and parse retries are intentionally separate. `--max-retries`
covers SDK transport/rate-limit/server failures; add `--parse-max-attempts 3`
to `run_skill_mvp.py`, `run_writingbench_official_eval.py`, or
`run_heldout_eval.py` only when complete model responses intermittently return
malformed JSON or invalid judge scores. `audit_train_examples.py` supports the
same flag for train-example quality audits.

Use `--dry-run` before spending API calls:

```bash
uv run python scripts/data/run_generation_jobs.py --limit 1 --dry-run
```

Apply generated outputs back to clean packs:

```bash
uv run python scripts/data/apply_generated_outputs.py \
  --packs artifacts/packs/example_packs.jsonl \
  --generations artifacts/jobs/generated_desired_outputs.jsonl \
  --out artifacts/packs/example_packs.v1.jsonl
```

Audit the frozen benchmark-flow boundary before using packs for induction:

```bash
uv run python scripts/data/audit_benchmark_flow.py
```

## Baselines

The first stable experiment matrix should stay small:

1. `prompt_only`
2. `few_shot_examples_only`
3. `one_shot_skill_from_examples`
4. `ours_no_validation`
5. `ours_full`

`teacher_refine_on_heldout` is an upper bound, not a normal baseline, because it uses heldout-time feedback.

## Evaluation Status

WritingBench and PresentBench private rubrics/checklists are used only at heldout
evaluation time.

Avoid model monoculture in reported experiments. A Qwen-only run where the same
model family cleans examples, induces skills, generates heldout outputs, and
judges results is useful as an MVP smoke loop, but it is not strong evidence for
the method. For paper-facing runs, separate at least the judge from the
generation/induction model, and ideally report a small cross-model grid such as:

- data/example construction: Qwen3.5-Plus or a stronger independent model;
- induction/generation under test: Qwen, GPT, Claude, or other target models;
- evaluation judge: official benchmark evaluator when available, otherwise an
  independent judge model with judge-model sensitivity checks.

To split solver and judge in one run, set both `BAILIAN_*` (solver) and a second
prefix (e.g. `JUDGE_*`) in your `.env`, then pass `--judge-config-prefix JUDGE`
to the eval runner. Only `JUDGE_MODEL` is required; `JUDGE_BASE_URL` and
`JUDGE_API_KEY` fall back to `BAILIAN_*` when the prefixed values are absent.

```bash
# In .env (in addition to BAILIAN_*):
# JUDGE_BASE_URL=https://api.anthropic.com/v1
# JUDGE_API_KEY=...
# JUDGE_MODEL=claude-sonnet-4-6
uv run python scripts/eval/run_writingbench_official_eval.py \
  --packs artifacts/packs/example_packs.v1.jsonl \
  --skills runs/skill_mvp.qwen.mvp.jsonl \
  --private-eval artifacts/private/example_private_eval.jsonl \
  --writingbench-root ../WritingBench \
  --judge-config-prefix JUDGE \
  --out runs/writingbench_official_eval.qwen_solver.claude_judge.jsonl \
  --summary-out runs/writingbench_official_eval.qwen_solver.claude_judge.summary.json \
  --limit-heldout 1 --resume
```

`run_heldout_eval.py`, `run_pattern_similarity_eval.py`, and
`run_self_consistency_metric.py` accept the same `--judge-config-prefix`. Each
written eval row carries top-level `solver_model` and `judge_model`;
`scripts/metrics/summarize_mvp_metrics.py` aggregates these into a `model_inventory`
block, and `scripts/ops/report_experiment_readiness.py` emits a
`model_monoculture` warning when every row collapses to a single model.

For WritingBench, use the source benchmark's evaluator prompt and per-criterion
scoring shape:

```bash
uv run python scripts/eval/run_writingbench_official_eval.py \
  --packs artifacts/packs/example_packs.v1.jsonl \
  --skills runs/skill_mvp.qwen.mvp.jsonl \
  --private-eval artifacts/private/example_private_eval.jsonl \
  --writingbench-root ../WritingBench \
  --pack-id writingbench_Finance_Business_Financial_Reports_en \
  --limit-heldout 1 \
  --resume
```

This uses WritingBench's local `prompt.py` and Qwen as the judge model. It is the
right smoke path for this repo when we are intentionally evaluating with Qwen;
the official leaderboard currently recommends Claude or the WritingBench critic.
Both eval runners checkpoint rows to `--out`; `--resume` reuses successful cells
and retries non-success cells after interruptions or provider failures.

For PresentBench, the official evaluator is the upstream `judge_all.py` /
`judge.py` / `scoring.py` pipeline. It requires a local checkout of the official
code repo, benchmark materials, domain common judge prompts, judge weights, and
actual `slides.pdf` or `slides.pptx` artifacts under a result tree that mirrors
the benchmark data. It also requires the `presentbench` optional Python extra
because upstream imports Gemini, requests, and tqdm helpers. If generation
explicitly fails, upstream `judge_all.py` accepts `slides_generation_failed.txt`
and writes an all-zero score. The text-only
`run_heldout_eval.py` path is therefore only a Qwen
`qwen_llm_rubric_surrogate` smoke evaluator; it must not be reported as an
official PresentBench visual/PPT score. Check readiness before claiming an
official PresentBench run:

```bash
uv run python scripts/eval/export_presentbench_official_artifacts.py \
  --eval runs/presentbench_surrogate_eval.qwen.mvp.jsonl \
  --eval runs/presentbench_surrogate_eval.qwen.auto_skill.jsonl \
  --mode-result-root prompt_only=../PresentBench/results/prompt_only \
  --mode-result-root auto_skill=../PresentBench/results/auto_skill \
  --overwrite
```

```bash
uv run python scripts/eval/check_presentbench_official_eval_ready.py \
  --packs artifacts/packs/example_packs.v1.jsonl \
  --code-root data/PresentBench_code \
  --mode-result-root prompt_only=../PresentBench/results/prompt_only \
  --mode-result-root auto_skill=../PresentBench/results/auto_skill \
  --judge-model gemini-3-flash-preview
```

For CI smoke checks where slide artifacts or generated PresentBench examples are
intentionally absent, add `--allow-missing --allow-empty`; do not use those flags
when claiming official readiness.

Readiness statuses are intentionally separated:

- `ready_for_official_judge`: slides exist and upstream `judge_all.py` can score them.
- `ready_for_zero_score`: `slides_generation_failed.txt` exists and upstream can write zero.
- `scored`: an upstream `*_score.yaml` is already present.

Once readiness is `ready_for_official_judge` or `ready_for_zero_score`, run the
repo-local wrapper. It loads `.env`, renders the exact upstream `judge_all.py`
commands, and fails before any API call when the required Gemini key is missing.

```bash
uv run python scripts/eval/run_presentbench_official_judge.py \
  --code-root data/PresentBench_code \
  --data-root data/PresentBench_repo \
  --mode-result-root prompt_only=../PresentBench/results/prompt_only \
  --mode-result-root auto_skill=../PresentBench/results/auto_skill \
  --api-type gemini \
  --model gemini-3-flash-preview \
  --max-workers 4
```

Use `--dry-run --allow-missing-env` to verify commands without `GENAI_API_KEY`.
Do not use `--allow-missing-env` for a real run.

After upstream score YAMLs exist, summarize official PresentBench scores and paired
deltas with explicit mode roots:

```bash
uv run python scripts/eval/summarize_presentbench_official_scores.py \
  --packs artifacts/packs/example_packs.v1.jsonl \
  --judge-model gemini-3-flash-preview \
  --score-root prompt_only=../PresentBench/results/prompt_only \
  --score-root auto_skill=../PresentBench/results/auto_skill
```

Gate the combined experiment state before reporting results:

```bash
uv run python scripts/ops/report_experiment_readiness.py --profile mvp --expect-status ready
```

By default this uses `--profile mvp`, which expects the current MVP artifact
contract: `skill_mvp.qwen.mvp.jsonl`, 4-mode WritingBench eval, 4-mode
PresentBench surrogate eval, and no `auto_skill_ours_full` / official
PresentBench requirement. The command exits non-zero if the current MVP rows are
incomplete or contain non-success statuses; use `--allow-not-ready` when you only
want to write and inspect the report. Use explicit profiles when checking other
phases:

```bash
uv run python scripts/ops/report_experiment_readiness.py --profile smoke --limit-heldout 1
uv run python scripts/ops/report_experiment_readiness.py --profile full --expect-status not_ready
```

All profiles still fail on schema-invalid rows. `smoke` permits partial
old/small coverage, `mvp` requires current MVP coverage, and `full` requires
`auto_skill_ours_full`, `auto_skill`, and official PresentBench score rows.
`--skills`, `--writing-eval`, `--present-surrogate-eval`, and
`--present-official-scores` may be repeated when a phase is split across
multiple JSONL artifacts; the `full` profile default already merges the MVP
skill rows, WritingBench/PresentBench `ours_full` skill rows, and the
PresentBench surrogate `auto_skill` rows.

## MVP Metrics

The MVP metrics summary is a reporting artifact, not a readiness gate. It covers:

1. benchmark score and paired deltas against `prompt_only`;
2. negative transfer rate over successfully paired heldout cells, with missing
   pairs reported separately;
3. skill artifact metrics from generated `SKILL.md` files and cross-example
   reports;
4. heldout inference token usage for generation, judge, and combined model calls;
5. skill self-consistency as an eval-only diagnostic.

Generate the self-consistency diagnostic only from public train examples and the
generated skill. It must not use private rubrics, checklists, heldout feedback,
official scores, or benchmark judge traces. Treat output-example similarity as
debug-only, never as a main effect metric. The summary separates
`official_benchmark_scores` from `surrogate_debug_scores` and `debug_scores`;
do not report PresentBench surrogate scores as official visual/PPT benchmark
results. When baseline and ablation modes live in separate eval JSONL files,
use `cross_eval_score_summaries` in the output for paired deltas joined across
files with the same evaluator and judge model.

```bash
uv run python scripts/metrics/run_self_consistency_metric.py \
  --packs artifacts/packs/example_packs.v1.jsonl \
  --skills runs/skill_mvp.qwen.mvp.jsonl \
  --out runs/self_consistency.writingbench.qwen.mvp.jsonl \
  --modes one_shot_skill_from_examples,auto_skill_feature_driven_no_validation

uv run python scripts/metrics/summarize_mvp_metrics.py \
  --skills runs/skill_mvp.qwen.mvp.jsonl \
  --skills runs/skill_mvp.qwen.ours_full.writingbench.jsonl \
  --skills runs/skill_mvp.qwen.ours_full.presentbench.jsonl \
  --packs artifacts/packs/example_packs.v1.jsonl \
  --modes prompt_only,few_shot_examples_only,one_shot_skill_from_examples,ours_no_validation,auto_skill,examples_plus_one_shot_skill,examples_plus_feature_skill \
  --eval runs/writingbench_official_eval.qwen.five_modes.no_thinking_auto_skill.jsonl \
  --eval runs/writingbench_official_eval.qwen.examples_plus_skill.heldout2.jsonl \
  --eval runs/presentbench_surrogate_eval.qwen.mvp.jsonl \
  --eval runs/presentbench_surrogate_eval.qwen.auto_skill.jsonl \
  --eval runs/writingbench_official_eval.mimo_judge.five_modes.heldout2.jsonl \
  --self-consistency runs/self_consistency.writingbench.qwen.mvp.jsonl \
  --out runs/mvp_metrics.heldout2.current.summary.json
```

## Collaboration

Read `AGENTS.md`, `CLAUDE.md`, and `CONTRIBUTING.md` before starting work. Large features go through branch plus PR. Daily iterations should keep tests green and update repo docs when assumptions change.
