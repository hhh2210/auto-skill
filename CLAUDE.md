# auto-skill Collaboration Guide

`AGENTS.md` and `CLAUDE.md` must stay identical. When changing one, update the other in the same commit or PR.

Default discussion language is Chinese. Keep key technical terms in English when they are part of the method or code.

## Canonical Research Framing

The project is **few-shot skill induction from user examples**.

Canonical module input:

- user examples;
- optional user natural-language request or emphasis;
- optional materials that are part of the examples.

Do not treat rubric/checklist, critique trace, weak/strong diff, or heldout feedback as normal auto-skill module input. Benchmark code may use rubrics/checklists only for private construction checks and heldout evaluation. Any future oracle or upper-bound experiment that uses private metadata must be named and isolated separately.

The first proof target is:

```text
user examples -> reusable skill package -> better heldout performance
```

Gap/admission diagnosis is secondary debug analysis, not the headline.

## Active Workstreams

### 1. Data Cleaning Scripts

Goal: turn benchmark sources into stable JSONL artifacts that collaborators can inspect and validate.

Expected surfaces:

- WritingBench ingestion;
- PresentBench ingestion;
- grouping by domain / requirement pattern;
- train-example / heldout-task split generation;
- schema validation and summary reports;
- no large benchmark data committed to git.

### 2. Auto-Skills Module

Goal: compile examples into a structured skill package.

Expected surfaces:

- feature extraction prompts or adapters;
- cross-example stable feature aggregation;
- conflict/outlier handling;
- leave-one-out validation design when enough examples exist;
- generated `SKILL.md`, optional templates, tests, and metadata;
- baselines against `few_shot_examples_only` and `one_shot_skill_from_examples`.

## Repository Layout

- `src/auto_skill/` — library: schemas, prompt builders, evaluation helpers, readiness logic.
- `scripts/` — thin CLI wrappers around the library; behavior lives in `src/`.
- `tests/` — `unittest` regression coverage for parsing, validation, aggregation, prompt builders.
- `artifacts/` — small curated JSONL inputs/outputs checked into git (splits, packs, private eval, extraction memory).
- `runs/` — generated experiment outputs (skill rows, eval rows, readiness reports). Large/ignored.
- `notes/`, `docs/` — research notes and stable repo docs.
- `data/` — local benchmark checkouts (`PresentBench_repo`, optional `WritingBench`); ignored.

## Data Flow

`fewshot_splits.jsonl` → `example_packs.jsonl` + `example_private_eval.jsonl` + `example_generation_jobs.jsonl` → `generated_desired_outputs.jsonl` → `example_packs.v1.jsonl` → `skill_mvp.*.jsonl` (one_shot / feature_driven / ours_full) → WritingBench official + PresentBench official/surrogate eval rows → `experiment_readiness.json`. Private rubric/judge metadata only flows through `example_private_eval.jsonl` and the eval scripts; it must never reach the skill-induction prompts.

## Repository Workflow

- Large features use branch + PR.
- Branch names should be descriptive, for example `feature/data-cleaning-v1` or `feature/skill-induction-prompts`.
- Keep `main` usable. Do not push broken scripts or failing tests.
- Use `uv` for environment and command execution. Do not add a Makefile unless there is a concrete non-Python workflow that needs it.
- External API config lives in local `.env`; never commit real keys. `.env.example` documents the required Bailian/OpenAI-compatible variables.
- Desired-output generation supports `--config-prefix MIMO` / `--config-prefix QWEN`
  for building parallel example sets from separately configured OpenAI-compatible
  models in one `.env`.
- Do not jump from the 8-pack MVP directly to all benchmark cases. Prefer a
  stratified medium expansion first (for example `--max-writing-groups 30
  --max-present-groups 20`) and require split validation, benchmark-flow audit,
  and sampled example-quality audit before full cleaning.
- Current expanded-cleaning evidence lives in
  `notes/expanded_cleaning_strategy_2026-05-08.md`, with local artifact hashes in
  `docs/expanded_cleaning_manifest_2026-05-08.md`: Qwen3.5-Plus completed the
  30WB/20PB local pass; use MIMO as audit/targeted regeneration until its long
  PresentBench request failures are resolved.
- Generation sampling temperature must be explicit for production runs: use `--temperature` or `BAILIAN_TEMPERATURE`; when unset the provider/model default is used.
- API generation is sequential by default. For batch cleaning, use `--num-threads` or `BAILIAN_NUM_THREADS`; with the current high-RPM Bailian quota, start with 16 and use the printed timing/error rate to decide whether to increase toward 32.
- For long PresentBench jobs, prefer `--timeout-seconds 600 --max-retries 0` while estimating throughput; repeated 120s SDK retries hide the true per-job latency.
- For long sequential skill-induction runs, use `--stream` to consume Qwen responses
  incrementally. Add `--stream-log` only for debugging one sequential run; do not
  use stream logging in threaded batch cleaning because output will interleave.
- `run_skill_mvp.py` writes an append-only stage ledger next to `--out` by
  default. Use `--resume` after API failures or long-run interruptions so
  successful stage calls and completed skill rows are reused.
- `run_writingbench_official_eval.py` and `run_heldout_eval.py` checkpoint every
  evaluated row to `--out`. Use `--resume` on long benchmark runs so successful
  pack/task/mode cells are reused and non-success cells are retried.
- Provider retries and parse retries are separate. `--max-retries` handles SDK
  transport/rate-limit/server errors; use `--parse-max-attempts 3` only when a
  complete model response is occasionally malformed JSON or an invalid judge
  score. This is supported by `run_skill_mvp.py`, `run_writingbench_official_eval.py`,
  `run_heldout_eval.py`, and `audit_train_examples.py`.
- Do not globally enable Qwen thinking for skill induction. Default to
  `--no-enable-thinking`; only use `--thinking-stages ... --thinking-budget N`
  for targeted A/B runs on analysis or merge stages.
- Use `scripts/ops/run_mvp_pipeline.sh` as the thin Bash orchestrator for long MVP
  runs. It sequences skill induction, extraction-memory update, WritingBench
  eval, PresentBench surrogate eval, validation, timestamps, and logs while
  keeping the Python CLIs as the source of behavior.
- `run_writingbench_official_eval.py` and `run_heldout_eval.py` support
  cell-level `--num-threads`: each pack/task/mode cell runs independently, while
  generation -> judge remains sequential inside the cell and checkpoint writes
  stay in the main process. For judge-swap runs, use
  `--judge-enable-thinking/--no-judge-enable-thinking` and
  `--judge-thinking-budget` to control judge-side reasoning separately from the
  solver.
- `artifacts/memory/extraction_memory.v1.jsonl` is an ignored local output for
  public user-example-derived extraction lessons from successful skill rows. It
  must not include private rubrics, heldout feedback, judge traces, or benchmark
  scores.
- Use `scripts/README.md` as the source of truth for script ownership and
  lifecycle. New scripts need a documented category; one-off research utilities
  should be marked as archive candidates instead of silently becoming core.
- `scripts/metrics/summarize_mvp_metrics.py` is a reporting artifact, not a readiness
  gate. It must keep official benchmark scores separate from surrogate/debug
  scores, expose expected-cell coverage, and treat self-consistency as an
  eval-only skill encoding diagnostic rather than a main performance metric.
- Output-example pattern similarity is debug-only. Blind runs must not include
  candidate skill text; skill-aware runs must be named and treated as debug.
- Avoid model monoculture in paper-facing claims. Qwen-only data cleaning,
  induction, generation, and judging is acceptable for smoke/MVP iteration, but
  reported results should separate at least the judge model from the tested
  generation/induction model and should label Qwen-only results as same-model
  smoke evidence.
- Split judge from solver via `ChatCompletionConfig.from_env(prefix=...)` plus
  the eval runners' `--judge-config-prefix` flag (e.g. `JUDGE`). Only the
  prefixed `MODEL` env var is required; `BASE_URL`/`API_KEY` fall back to
  `BAILIAN_*`. Eval rows persist top-level `solver_model` and `judge_model`;
  `report_experiment_readiness.py` emits a `model_monoculture` warning when
  every row collapses to a single model.
- Before pushing code, run `uv run python -m unittest discover -s tests`,
  `uv run python scripts/data/validate_splits.py artifacts/splits/fewshot_splits.jsonl`,
  and the readiness gate below. A not-ready report is acceptable during
  prototyping only when it is explicit and expected, never silently green.
- Update `README.md`, `docs/`, or this file when project assumptions change.
- Keep generated large outputs under `data/`, `outputs/`, `runs/`, or `logs/`; these are ignored.

## Code Rules

- Prefer small, testable library functions under `src/auto_skill/`.
- Keep `scripts/` as thin CLI wrappers.
- Use explicit schemas and validation for JSONL records.
- Avoid hidden benchmark leakage: if a function consumes rubric/checklist/teacher traces, name it as benchmark/eval-only.
- Add or update tests for shared parsing, validation, aggregation, and prompt-building logic.

## Current Validation Commands

```bash
uv sync --extra dev
uv sync --extra dev --extra presentbench
uv run python -m unittest discover -s tests
diff -q AGENTS.md CLAUDE.md
uv run python -c "import google.genai, PIL, pptx, requests, tqdm"
uv run python scripts/data/validate_splits.py artifacts/splits/fewshot_splits.jsonl
uv run python scripts/data/audit_benchmark_flow.py
uv run python scripts/ops/validate_run_artifacts.py --generated-outputs tests/fixtures/generated_outputs.valid.jsonl --skills tests/fixtures/skill_rows.valid.jsonl --eval tests/fixtures/eval_rows.valid.jsonl
uv run python scripts/data/inspect_benchmarks.py --writingbench-root ../WritingBench --presentbench-root data/PresentBench_repo --limit 2 --out-dir artifacts
uv run python scripts/data/build_fewshot_splits.py --writingbench-root ../WritingBench --presentbench-root data/PresentBench_repo --train-size 3 --heldout-size 2 --max-groups 4 --out artifacts/splits/fewshot_splits.jsonl --summary-out artifacts/splits/fewshot_split_summary.json
uv run python scripts/data/build_example_packs.py --splits artifacts/splits/fewshot_splits.jsonl --packs-out artifacts/packs/example_packs.jsonl --private-out artifacts/private/example_private_eval.jsonl --jobs-out artifacts/jobs/example_generation_jobs.jsonl --summary-out artifacts/reports/example_pack_summary.md
uv run python scripts/data/run_generation_jobs.py --jobs artifacts/jobs/example_generation_jobs.jsonl --out artifacts/jobs/generated_desired_outputs.jsonl --limit 1 --dry-run
uv run python scripts/skills/run_skill_mvp.py --packs artifacts/packs/example_packs.v1.jsonl --out runs/skill_mvp.qwen.mvp.jsonl --stream --resume --allow-partial --no-enable-thinking --no-leave-one-out --temperature 0.2 --json-temperature 0 --timeout-seconds 900 --max-retries 0 --max-tokens 8192
RUN_SKILL_MVP=0 RUN_MEMORY=1 RUN_WRITINGBENCH=1 RUN_PRESENTBENCH=1 RUN_VALIDATE=1 PRESENTBENCH_NUM_THREADS=4 scripts/ops/run_mvp_pipeline.sh
uv run python scripts/eval/run_writingbench_official_eval.py --packs artifacts/packs/example_packs.v1.jsonl --skills runs/skill_mvp.qwen.mvp.jsonl --private-eval artifacts/private/example_private_eval.jsonl --writingbench-root ../WritingBench --limit-heldout 1 --resume --num-threads 16 --dry-run
uv run python scripts/eval/check_presentbench_official_eval_ready.py --packs artifacts/packs/example_packs.v1.jsonl --code-root data/PresentBench_code --judge-model gemini-3-flash-preview --limit-heldout 1 --allow-missing --allow-empty
uv run python scripts/eval/summarize_presentbench_official_scores.py --packs artifacts/packs/example_packs.v1.jsonl --judge-model gemini-3-flash-preview --score-root prompt_only=../PresentBench/results/prompt_only --score-root auto_skill=../PresentBench/results/auto_skill --limit-heldout 1
uv run python scripts/metrics/run_self_consistency_metric.py --packs artifacts/packs/example_packs.v1.jsonl --skills runs/skill_mvp.qwen.mvp.jsonl --out runs/self_consistency.writingbench.qwen.mvp.jsonl --modes one_shot_skill_from_examples,auto_skill_feature_driven_no_validation --dry-run
uv run python scripts/metrics/summarize_mvp_metrics.py --skills runs/skill_mvp.qwen.mvp.jsonl --packs artifacts/packs/example_packs.v1.jsonl --modes prompt_only,few_shot_examples_only,one_shot_skill_from_examples,ours_no_validation --limit-heldout 1 --eval runs/writingbench_official_eval.qwen.mvp.jsonl --eval runs/presentbench_surrogate_eval.qwen.mvp.jsonl --self-consistency runs/self_consistency.writingbench.qwen.mvp.jsonl --out runs/mvp_metrics.summary.json
uv run python scripts/ops/report_experiment_readiness.py --profile mvp --limit-heldout 1 --expect-status ready
uv run python scripts/ops/report_experiment_readiness.py --profile full --limit-heldout 1 --expect-status not_ready
uv run python scripts/ops/report_expanded_cleaning_status.py --expect-status ready
uv run ruff check .
```

`run_generation_jobs.py --dry-run` does not create generated outputs. Only run
`apply_generated_outputs.py` after a real generation file exists, and treat any
non-zero missing/rejected count as a failed data freeze.

For CI or smoke work where the current checked-in artifacts are intentionally
incomplete, use `report_experiment_readiness.py --profile full --expect-status
not_ready` to assert that the gate fails closed. Use `--profile smoke` for
partial smoke inspection. The default `--profile mvp` reads current MVP
artifacts and does not require `auto_skill_ours_full`, `auto_skill`, or official
PresentBench score rows; full experiment readiness requires those rows.
