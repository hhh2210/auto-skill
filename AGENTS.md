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
  30WB/20PB local pass and the max-available Qwen pass
  (`runs/expanded/example_packs.max_available.qwen.v1.jsonl`: 142 packs, 426
  frozen train examples, 284 heldout tasks, benchmark-flow `status=ok`). MIMO
  has an audited local subset pack
  (`runs/expanded/example_packs.30wb_20pb.mimo.sample.v1.jsonl`: 15 packs, 45
  frozen train examples, benchmark-flow `status=ok`), but it is not the
  canonical expanded dataset because full MIMO cleaning is not frozen.
- Generation sampling temperature must be explicit for production runs: use `--temperature` or `BAILIAN_TEMPERATURE`; when unset the provider/model default is used.
- API generation is sequential by default. For batch cleaning, use `--num-threads` or `BAILIAN_NUM_THREADS`; with the current high-RPM Bailian quota, start with 16 and use the printed timing/error rate to decide whether to increase toward 32.
- For long PresentBench jobs, prefer `--timeout-seconds 600 --max-retries 0` while estimating throughput; repeated 120s SDK retries hide the true per-job latency.
- PresentBench official judging has two routes: `--api-type gemini` uses upstream
  `judge.py` with `GENAI_API_KEY`; `--api-type openai` uses the in-repo
  chat-completions proxy wrapper with `GOOGLE_THIRD_API_URL` and
  `GOOGLE_THIRD_API_KEY`. Do not add LiteLLM unless multiple non-OpenAI-compatible
  judge backends become a real requirement.
- PresentBench is not a pure-text benchmark. Full-setting solver work needs
  multimodal/PDF material perception for source pages, figures, tables, charts,
  and layout evidence. Direct image generation is not required if we use a
  renderer-backed route, but the material-parsing stage cannot be text-only.
- Use `scripts/data/run_presentbench_material_digest.py` as the experimental
  bridge from selected PresentBench PDF pages to structured material digests.
  It renders pages to PNG and sends them as OpenAI-compatible `image_url`
  content parts to Qwen/Qwen-VL. Default to `--no-enable-thinking`; a smoke run
  with Qwen3.5-Plus confirmed `image_tokens=410`, while thinking mode consumed
  reasoning tokens and truncated the digest.
- For long sequential skill-induction runs, use `--stream` to consume Qwen responses
  incrementally. Add `--stream-log` only for debugging one sequential run; do not
  use stream logging in threaded batch cleaning because output will interleave.
- `run_skill_mvp.py` writes an append-only stage ledger next to `--out` by
  default. Use `--resume` after API failures or long-run interruptions so
  successful stage calls and completed skill rows are reused.
- `run_writingbench_official_eval.py` and `run_heldout_eval.py` checkpoint every
  evaluated row to `--out`. Use `--resume` on long benchmark runs so successful
  pack/task/mode cells are reused and non-success cells are retried.
- When resuming into an existing eval `--out`, keep the selected pack/task/mode
  scope at least as broad as the existing file. The WritingBench runner fails
  closed if a filtered resume would prune rows outside the selected cells; use a
  new output path for one-cell probes.
- Provider retries and parse retries are separate. `--max-retries` handles SDK
  transport/rate-limit/server errors; `--parse-max-attempts` handles complete
  model responses that are malformed JSON or invalid judge scores. The default
  parse retry budget is 3 for `run_skill_mvp.py`,
  `run_writingbench_official_eval.py`, `run_heldout_eval.py`,
  `audit_train_examples.py`, `run_grounding_eval.py`,
  `run_pattern_similarity_eval.py`, and `run_self_consistency_metric.py`.
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
  `report_experiment_readiness.py` emits `model_monoculture` when every visible
  row collapses to a single model. For paper-facing claims, inspect
  `model_inventory.scored`: it excludes non-success eval rows such as
  `missing_score_artifact`, so placeholder official rows cannot falsely break a
  Qwen-only scored run. In that case the gate emits `scored_model_monoculture`.
  Metrics summaries keep `model_inventory` focused on heldout eval rows and put
  self-consistency or other diagnostic rows in `diagnostic_model_inventory`.
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
uv run python scripts/data/audit_benchmark_flow.py --splits runs/expanded/fewshot_splits.30wb_20pb.jsonl --packs runs/expanded/example_packs.30wb_20pb.qwen.v1.jsonl --private-eval runs/expanded/example_private_eval.30wb_20pb.jsonl --jobs runs/expanded/example_generation_jobs.30wb_20pb.jsonl --generated-outputs runs/expanded/generated_desired_outputs.30wb_20pb.qwen.jsonl
uv run python scripts/data/audit_benchmark_flow.py --splits runs/expanded/fewshot_splits.max_available.jsonl --packs runs/expanded/example_packs.max_available.qwen.v1.jsonl --private-eval runs/expanded/example_private_eval.max_available.jsonl --jobs runs/expanded/example_generation_jobs.max_available.jsonl --generated-outputs runs/expanded/generated_desired_outputs.max_available.qwen.latest_success.jsonl
uv run python scripts/ops/validate_run_artifacts.py --generated-outputs tests/fixtures/generated_outputs.valid.jsonl --skills tests/fixtures/skill_rows.valid.jsonl --eval tests/fixtures/eval_rows.valid.jsonl
uv run python scripts/data/inspect_benchmarks.py --writingbench-root ../WritingBench --presentbench-root data/PresentBench_repo --limit 2 --out-dir artifacts
uv run python scripts/data/build_fewshot_splits.py --writingbench-root ../WritingBench --presentbench-root data/PresentBench_repo --train-size 3 --heldout-size 2 --max-groups 4 --out artifacts/splits/fewshot_splits.jsonl --summary-out artifacts/splits/fewshot_split_summary.json
uv run python scripts/data/build_example_packs.py --splits artifacts/splits/fewshot_splits.jsonl --packs-out artifacts/packs/example_packs.jsonl --private-out artifacts/private/example_private_eval.jsonl --jobs-out artifacts/jobs/example_generation_jobs.jsonl --summary-out artifacts/reports/example_pack_summary.md
uv run python scripts/data/run_generation_jobs.py --jobs artifacts/jobs/example_generation_jobs.jsonl --out artifacts/jobs/generated_desired_outputs.jsonl --limit 1 --dry-run
uv run python scripts/data/run_presentbench_material_digest.py --packs artifacts/packs/example_packs.v1.jsonl --out runs/presentbench_material_digest.qwen3.5plus.smoke.jsonl --pack-id presentbench_education_THU_DSA --role heldout --limit 1 --pages 1 --dpi 72 --temperature 0 --max-tokens 2048 --no-enable-thinking
uv run python scripts/skills/run_skill_mvp.py --packs artifacts/packs/example_packs.v1.jsonl --out runs/skill_mvp.qwen.mvp.jsonl --stream --resume --allow-partial --no-enable-thinking --no-leave-one-out --temperature 0.2 --json-temperature 0 --timeout-seconds 900 --max-retries 0 --parse-max-attempts 3 --max-tokens 8192
RUN_SKILL_MVP=0 RUN_MEMORY=1 RUN_WRITINGBENCH=1 RUN_PRESENTBENCH=1 RUN_VALIDATE=1 PRESENTBENCH_NUM_THREADS=4 scripts/ops/run_mvp_pipeline.sh
uv run python scripts/eval/run_writingbench_official_eval.py --packs artifacts/packs/example_packs.v1.jsonl --skills runs/skill_mvp.qwen.mvp.jsonl --private-eval artifacts/private/example_private_eval.jsonl --writingbench-root ../WritingBench --limit-heldout 1 --parse-max-attempts 3 --resume --num-threads 16 --dry-run
uv run python scripts/eval/check_presentbench_official_eval_ready.py --packs artifacts/packs/example_packs.v1.jsonl --code-root data/PresentBench_code --judge-model gemini-3-flash-preview --allow-missing --allow-empty
uv run python scripts/eval/run_presentbench_official_judge.py --packs artifacts/packs/example_packs.v1.jsonl --code-root data/PresentBench_code --data-root data/PresentBench_repo --mode-result-root prompt_only=../PresentBench/results/prompt_only --mode-result-root auto_skill=../PresentBench/results/auto_skill --limit-heldout 2 --dry-run --allow-missing-env --expect-commands 16 --commands-out runs/presentbench_official_judge_commands.json
# After upstream PresentBench score YAMLs exist:
# uv run python scripts/eval/summarize_presentbench_official_scores.py --packs artifacts/packs/example_packs.v1.jsonl --solver-model qwen3.5-plus --judge-model gemini-3-flash-preview --score-root prompt_only=../PresentBench/results/prompt_only --score-root auto_skill=../PresentBench/results/auto_skill
uv run python scripts/metrics/run_self_consistency_metric.py --packs artifacts/packs/example_packs.v1.jsonl --skills runs/skill_mvp.qwen.mvp.jsonl --out runs/self_consistency.writingbench.qwen.mvp.jsonl --modes one_shot_skill_from_examples,auto_skill_feature_driven_no_validation --dry-run
uv run python scripts/metrics/summarize_mvp_metrics.py --skills runs/skill_mvp.qwen.mvp.jsonl --skills runs/skill_mvp.qwen.ours_full.writingbench.jsonl --skills runs/skill_mvp.qwen.ours_full.presentbench.jsonl --packs artifacts/packs/example_packs.v1.jsonl --modes prompt_only,few_shot_examples_only,one_shot_skill_from_examples,ours_no_validation,auto_skill,examples_plus_one_shot_skill,examples_plus_feature_skill,slide_constrained_examples_plus_feature_skill,layout_plan_examples_plus_feature_skill --eval runs/writingbench_official_eval.qwen.five_modes.no_thinking_auto_skill.jsonl --eval runs/writingbench_official_eval.qwen.examples_plus_skill.heldout2.jsonl --eval runs/presentbench_surrogate_eval.qwen.mvp.jsonl --eval runs/presentbench_surrogate_eval.qwen.auto_skill.jsonl --eval runs/presentbench_surrogate_eval.qwen.examples_plus_skill.heldout2.jsonl --eval runs/presentbench_surrogate_eval.qwen.slide_constrained_examples_plus_feature.heldout2.jsonl --eval runs/presentbench_surrogate_eval.qwen.layout_plan_examples_plus_feature.heldout2.jsonl --eval runs/writingbench_official_eval.mimo_judge.five_modes.heldout2.jsonl --eval runs/writingbench_official_eval.mimo_judge.examples_plus_skill.heldout2.jsonl --self-consistency runs/self_consistency.writingbench.qwen.mvp.jsonl --out runs/mvp_metrics.heldout2.current.summary.json
uv run python scripts/ops/report_experiment_readiness.py --profile mvp --expect-status ready
uv run python scripts/ops/report_experiment_readiness.py --profile full --expect-status not_ready
uv run python scripts/ops/report_expanded_cleaning_status.py --expect-status ready
uv run python scripts/ops/report_expanded_cleaning_status.py --require-mimo-subset --expect-status ready
uv run python scripts/ops/report_expanded_cleaning_status.py --splits runs/expanded/fewshot_splits.max_available.jsonl --packs runs/expanded/example_packs.max_available.qwen.v1.jsonl --private-eval runs/expanded/example_private_eval.max_available.jsonl --jobs runs/expanded/example_generation_jobs.max_available.jsonl --generated-outputs runs/expanded/generated_desired_outputs.max_available.qwen.latest_success.jsonl --expect-packs 142 --expect-train-examples 426 --expect-heldout-tasks 284 --expect-generation-jobs 426 --skip-mimo-subset --expect-status ready
uv run python scripts/metrics/validate_disagreement_taxonomy.py --packets runs/expanded/judge_disagreements.qwen_vs_mimo.sample4_wb.heldout1.jsonl --taxonomy notes/judge_disagreement_taxonomy_2026-05-09.jsonl --expect-status ok
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
PresentBench score artifacts / success rows; full experiment readiness requires
those official scores.
Readiness input flags (`--skills`, `--writing-eval`, `--present-surrogate-eval`,
and `--present-official-scores`) are repeatable so split artifacts can be merged
without fabricating concatenated run files.
