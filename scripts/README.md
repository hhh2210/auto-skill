# Scripts Inventory

This directory is the compatibility layer for command-line entrypoints. Keep
reusable behavior in `src/auto_skill/`; scripts should mostly parse CLI args,
load JSONL files, call library functions, and write artifacts.

Do not add a new script for a one-off experiment unless it has a clear owner and
an expiry path. Prefer adding a subcommand or a library function when the behavior
will be reused.

## Core Pipeline

These scripts are part of the main benchmark and MVP workflow and should stay
stable.

### Data Construction

| Script | Purpose | Status |
| --- | --- | --- |
| `data/inspect_benchmarks.py` | Inspect WritingBench and PresentBench source structure. | diagnostic but documented |
| `data/build_fewshot_splits.py` | Build train-example / heldout-task splits. | core |
| `data/validate_splits.py` | Validate split JSONL. | CI/core |
| `data/build_example_packs.py` | Convert splits into user-visible packs plus private eval refs. | core |
| `data/run_generation_jobs.py` | Generate desired outputs for train examples. | core/API |
| `data/apply_generated_outputs.py` | Freeze generated desired outputs into example packs. | core |
| `data/audit_benchmark_flow.py` | Audit cleaned artifacts against `notes/benchmark_flow.md` leakage and split-boundary invariants. | CI/core |

### Skill Induction

| Script | Purpose | Status |
| --- | --- | --- |
| `skills/run_skill_mvp.py` | Run one-shot and feature-driven skill induction. | core/API |
| `skills/update_extraction_memory.py` | Extract public lessons from successful skill rows. | optional core |

### Heldout Evaluation

| Script | Purpose | Status |
| --- | --- | --- |
| `eval/run_writingbench_official_eval.py` | Generate heldout outputs and score with WritingBench prompt; supports cell-level `--num-threads` and judge-side thinking controls. | core/API |
| `eval/audit_train_examples.py` | Private quality audit for generated train-example outputs. | diagnostic/API |
| `eval/run_heldout_eval.py` | Text-only surrogate eval, mainly PresentBench smoke/debug. | surrogate/debug |
| `eval/check_presentbench_official_eval_ready.py` | Check official PresentBench artifact readiness. | core gate |
| `eval/summarize_presentbench_official_scores.py` | Summarize upstream PresentBench score YAMLs. | core once official scores exist |

### Metrics And Gates

| Script | Purpose | Status |
| --- | --- | --- |
| `metrics/summarize_mvp_metrics.py` | Summarize task, artifact, cost, self-consistency, and model inventory metrics. | core reporting |
| `ops/report_experiment_readiness.py` | Fail-closed experiment readiness report. | CI/core |
| `ops/report_expanded_cleaning_status.py` | Recompute local 30WB/20PB expanded-cleaning status from ignored run artifacts. | handoff gate |
| `ops/validate_run_artifacts.py` | Validate generated-output, skill, and eval artifacts. | CI/core |
| `metrics/run_self_consistency_metric.py` | Eval-only skill encoding diagnostic. | diagnostic metric |
| `metrics/run_pattern_similarity_eval.py` | Blind or skill-aware output-pattern similarity judge. | debug-only metric |

### Orchestration

| Script | Purpose | Status |
| --- | --- | --- |
| `ops/run_mvp_pipeline.sh` | Thin shell wrapper for long local MVP runs. | convenience only |

## Removed One-Off Scripts

The old GitHub PR-review seed-data utilities were removed because they were not
part of the current WritingBench / PresentBench auto-skill MVP path:

- `extract_pr_reviews.py`
- `analyze_pr_reviews.py`
- `analyze_team_consensus.py`

Reintroduce that line only as a dedicated, tested module with its own inventory
entry.

## Refactor Direction

The next structural cleanup should avoid breaking existing documented commands:

1. Move shared helpers currently imported from scripts into `src/auto_skill/`.
   Current offenders include `scripts.eval.run_heldout_eval` helpers reused by
   WritingBench, PresentBench, pattern-similarity, and official-score scripts.
2. Add `python -m auto_skill.cli ...` or `uv run auto-skill ...` entrypoints only
   after the library split is stable.
3. Keep top-level script directories (`data/`, `skills/`, `eval/`, `metrics/`,
   `ops/`) small. New scripts need a lifecycle category in this file.
