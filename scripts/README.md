# Scripts Inventory

This directory is the compatibility layer for command-line entrypoints. Keep
reusable behavior in `src/auto_skill/`; scripts should mostly parse CLI args,
load JSONL files, call library functions, and write artifacts.

Do not add a new script for a one-off experiment unless it has a clear owner and
an expiry path. Prefer adding a subcommand or a library function when the behavior
will be reused.

## Status Legend

- `core`: stable project workflow; needs tests or a documented smoke command.
- `CI/core` or `core gate`: expected to stay runnable before code handoff.
- `active diagnostic`: research/debug surface with current owner; review or
  archive if unused for six months.
- `debug-only` or `debug-only metric`: not paper-facing evidence unless promoted
  with tests, docs, and explicit leakage boundaries.
- `prototype`: allowed to change quickly; promote reusable behavior into
  `src/auto_skill/` on the second serious reuse or repeated fixes.
- `archive candidate`: disposable unless refreshed with owner, test, and expiry.
- `/API`: makes model/provider calls; must support dry-run, resume, or bounded
  smoke usage when practical.

## Active Author-Style Pipeline

New benchmark work should target the blog/reddit personal author-style path.
The source-cleaning entrypoint is currently
`python -m auto_skill.author_style_cleaning`; reusable behavior lives under
`src/auto_skill/cleaning/author_style/`.

Key active artifacts:

- `accepted_author_style_packs.jsonl`: public same-author train examples and heldout tasks.
- `accepted_author_style_private_eval.jsonl`: private target/reference material for evaluation only.
- `accepted_hard_negatives.jsonl`: other-author hard negatives.
- `author_style_smoke_summary.json`: cleaning and audit summary.

| Script | Purpose | Status |
| --- | --- | --- |
| `data/profile_author_style_corpus.py` | Profile raw blog/reddit author-style corpora before choosing filters; writes JSON and Markdown reports for length, author concentration, exact duplicates, and triple feasibility. | active diagnostic |

### Research Probes

These scripts build ignored `runs/probes/...` artifacts for metric calibration.
They should stay small and disposable: canonical cleaning and judge behavior must
remain in the library / metric runners above.

| Script | Purpose | Status |
| --- | --- | --- |
| `probes/build_author_style_probe_artifacts.py` | Build fresh-minimal T2 oracle and pairwise probe artifacts from `clean_posts.jsonl` without running judges. | active diagnostic |
| `probes/run_author_style_pairwise_separation.py` | Run gpt-5.5 Codex-OAuth within/cross author-style similarity scoring over private probe jobs. | active diagnostic/API |
| `probes/summarize_author_style_probe.py` | Summarize oracle, cross-check, pairwise, and optional legacy-strict controls into JSON plus a short Markdown report. | active diagnostic |

## Legacy WritingBench / PresentBench Pipeline

These scripts remain runnable regression surfaces. Keep them stable, but do not
extend them for new author-style experiments unless explicitly requested.

These scripts are part of the legacy MVP workflow and should stay stable while
the historical artifacts remain in use.

### Data Construction

| Script | Purpose | Status |
| --- | --- | --- |
| `data/inspect_benchmarks.py` | Inspect WritingBench and PresentBench source structure. | diagnostic but documented |
| `data/build_fewshot_splits.py` | Build train-example / heldout-task splits. | core |
| `data/validate_splits.py` | Validate split JSONL. | CI/core |
| `data/build_example_packs.py` | Convert splits into user-visible packs plus private eval refs. | core |
| `data/run_generation_jobs.py` | Generate desired outputs for train examples. | core/API |
| `data/export_latest_successful_generations.py` | Materialize latest-success rows from append-only generation logs for readiness/audit gates. | core |
| `data/apply_generated_outputs.py` | Freeze generated desired outputs into example packs. | core |
| `data/audit_benchmark_flow.py` | Audit cleaned artifacts against `docs/benchmark_flow.md` leakage, split-boundary, and optional generation provenance invariants. | CI/core |

### Skill Induction

| Script | Purpose | Status |
| --- | --- | --- |
| `skills/run_skill_mvp.py` | Run one-shot and feature-driven skill induction; supports JSON-stage `--parse-max-attempts`. | core/API |
| `skills/run_skill_minimal.py` | Run the minimal extraction -> MIMO supervisor -> revision auto-skill induction prototype with optional extraction-memory read-back. | prototype/API |
| `skills/update_extraction_memory.py` | Extract public lessons from successful skill rows. | optional core |

### Heldout Evaluation

| Script | Purpose | Status |
| --- | --- | --- |
| `eval/run_writingbench_official_eval.py` | Generate heldout outputs and score with WritingBench prompt; supports cell-level `--num-threads`, judge-side thinking controls, and judge parse retries. | core/API |
| `eval/audit_train_examples.py` | Private quality audit for generated train-example outputs; supports judge parse retries. | diagnostic/API |
| `eval/run_heldout_eval.py` | Text-only surrogate eval, mainly PresentBench smoke/debug; supports judge parse retries. | surrogate/debug |
| `eval/export_presentbench_official_artifacts.py` | Export PresentBench text generations as simple `slides.pdf` result-tree artifacts for official-judge readiness. | bridge/debug |
| `eval/check_presentbench_official_eval_ready.py` | Check official PresentBench artifact readiness; repeat `--mode-result-root` for multiple modes. | core gate |
| `eval/run_presentbench_official_judge.py` | Preflight and run upstream PresentBench `judge.py` for selected heldout cells; use `--all-presentbench` only for full-checkout `judge_all.py` runs. | core once official artifacts exist |
| `eval/summarize_presentbench_official_scores.py` | Summarize upstream PresentBench score YAMLs. | core once official scores exist |

### Metrics And Gates

| Script | Purpose | Status |
| --- | --- | --- |
| `metrics/summarize_mvp_metrics.py` | Summarize task, artifact, cost, self-consistency, and model inventory metrics. | core reporting |
| `ops/report_experiment_readiness.py` | Fail-closed experiment readiness report; repeat artifact flags to merge split runs. | CI/core |
| `ops/report_expanded_cleaning_status.py` | Recompute local 30WB/20PB expanded-cleaning status from ignored run artifacts. | handoff gate |
| `ops/validate_run_artifacts.py` | Validate generated-output, skill, and eval artifacts. | CI/core |
| `metrics/run_self_consistency_metric.py` | Eval-only skill encoding diagnostic; supports signature/judge parse retries. | diagnostic metric |
| `metrics/run_pattern_similarity_eval.py` | Blind or skill-aware output-pattern similarity judge; supports judge parse retries. | debug-only metric |
| `metrics/run_pairwise_likeness.py` | Anchor pairwise example-likeness win-rate over existing heldout outputs; supports swapped A/B order and judge parse retries. | diagnostic/API |
| `metrics/run_author_style_reference_retrieval.py` | Source-derived target/reference/hard-negative oracle metric for author-style packs. | active diagnostic/API |
| `metrics/run_skill_quality_eval.py` | Ctx2Skill-style five-dimension skill artifact quality judge over public train examples and `skill_md`; summarizes deterministic 0-100 averages. | diagnostic/API |
| `metrics/summarize_per_criterion_delta.py` | Post-process WritingBench eval rows into per-criterion deltas + keyword-bucket roll-up between two modes (no extra LLM calls). | diagnostic |
| `metrics/extract_judge_reasons.py` | Join paired baseline/candidate WritingBench judge scores and reasons for losing tasks (no extra LLM calls). | diagnostic |
| `metrics/aggregate_failure_modes.py` | Single-call MIMO aggregation of extracted judge rationales into operational failure-mode categories. | diagnostic/API |
| `metrics/run_grounding_eval.py` | Eval-only grounding/hallucination probe for heldout outputs against current task/material evidence; supports judge parse retries. | diagnostic metric |
| `metrics/export_judge_disagreements.py` | Export judge delta sign-disagreement packets with output stats and optional skill context for evaluator calibration review. | diagnostic |
| `metrics/validate_disagreement_taxonomy.py` | Validate per-packet calibration labels against judge-disagreement packet hashes. | diagnostic |
| `metrics/report_example_contamination.py` | Deterministic train-example phrase overlap diagnostic for candidate outputs. | diagnostic |

### Orchestration

| Script | Purpose | Status |
| --- | --- | --- |
| `ops/run_mvp_pipeline.sh` | Thin shell wrapper for long local MVP runs. | convenience only |
| `ops/run_minimal_memory_ablation.sh` | Repeatable runner for minimal auto-skill memory ablations plus optional feature-driven LOO comparison: shard induction, merge successful skill rows, write eval aliases, run MIMO WritingBench task-quality eval, run pairwise example-likeness eval, run skill-quality eval, and write the combined three-metric report. | convenience/prototype |
| `ops/prepare_three_metric_ablation.py` | Helper for `run_minimal_memory_ablation.sh`; writes eval aliases, pairwise inputs, skill-quality inputs, and the final markdown report. | helper |
| `ops/shard_jsonl.py` | Deterministically split JSONL rows into modulo shards for resumable local batch runs. | helper |
| `ops/combine_skill_shards.py` | Merge successful per-pack skill rows from shard outputs in original pack order. | helper |

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
