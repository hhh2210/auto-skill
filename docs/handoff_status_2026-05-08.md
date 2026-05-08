# Handoff Status 2026-05-08

This repo is ready to hand off as an engineering prototype and smoke dataset
workspace. It is not yet ready to claim full benchmark or paper-level results.

## Two Workstreams

### 1. Cleaned Dataset Artifacts

Current checked local artifacts:

- `artifacts/splits/fewshot_splits.jsonl`: 8 packs total, 4 WritingBench and 4 PresentBench.
- `artifacts/packs/example_packs.v1.jsonl`: canonical user-example packs.
- `artifacts/private/example_private_eval.jsonl`: private eval/rubric metadata for benchmark-only evaluation.

Generated queues, raw runs, reports, diagrams, and extraction-memory snapshots
are intentionally ignored so the repository handoff stays readable.

Current freeze status:

- WritingBench: 4/4 selected packs have 3 generated train examples each.
- PresentBench: 4/4 selected packs have 3 generated train examples each.

Do not treat private rubric/checklist data as auto-skill module input. It is
only for construction checks, heldout evaluation, and diagnostic audits.

### 2. Auto-Skills Prototype

Current runnable surfaces:

- `scripts/skills/run_skill_mvp.py`: one-shot, feature-driven, and LOO/ours-full skill induction.
- `scripts/eval/run_writingbench_official_eval.py`: WritingBench official-prompt eval with cell-level `--num-threads`, candidate reuse, judge-swap, and judge-side thinking controls.
- `scripts/eval/audit_train_examples.py`: private MIMO audit for train-example desired outputs.
- `scripts/metrics/run_pattern_similarity_eval.py`: output-pattern similarity diagnostic.
- `scripts/metrics/run_self_consistency_metric.py`: skill self-consistency diagnostic.
- `scripts/ops/report_experiment_readiness.py`: fail-closed readiness gate.

Recent WritingBench smoke results:

- Qwen judge, 4 packs x 5 modes: complete.
- MIMO judge-swap on the same Qwen candidates, 4 packs x 5 modes: complete.
- Current `auto_skill_ours_full` does not beat `few_shot_examples_only` or
  `one_shot_skill_from_examples` on this smoke slice.

Current MIMO train-example audit:

- 12 WritingBench train examples audited.
- Only one example scored below 5.
- This supports targeted regeneration, not full MIMO re-cleaning.

## Current Readiness

Engineering checks are green:

```bash
uv run python -m unittest discover -s tests
uv run ruff check .
diff -q AGENTS.md CLAUDE.md
```

MVP readiness now uses the current MVP artifact contract and paths:

```bash
uv run python scripts/ops/report_experiment_readiness.py --profile mvp --limit-heldout 1 --allow-not-ready
```

Full experiment readiness is intentionally not green yet:

```bash
uv run python scripts/ops/report_experiment_readiness.py --profile full --limit-heldout 1 --expect-status not_ready
```

Known blockers:

- Current MVP PresentBench surrogate rows include model errors on some cells.
- PresentBench official score rows are missing.
- Full-profile skills and heldout eval rows do not cover all 8 smoke packs yet.
- Smoke data is too small for a paper claim.

## Recommended Next Steps

1. Expand to 8-12 WritingBench packs and 4-8 PresentBench packs with at least 2
   heldout tasks each.
2. Run failure attribution on negative-transfer packs, especially where
   `auto_skill` loses to prompt-only or few-shot examples.
3. Improve skill induction and LOO merge before regenerating all examples.
4. Regenerate only examples that are independently audited as low quality.
5. Keep solver and judge models separated and record model inventory for every run.

## Push Guidance

Safe to push:

- source code under `src/`, `scripts/`, and `tests/`;
- repo docs, including `README.md`, `AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md`, and `docs/`;
- small, inspectable artifacts under `artifacts/` when they are needed for handoff.

Do not push:

- `.env` or real API keys;
- large local benchmark checkouts under `data/`;
- large or exploratory `runs/` outputs unless intentionally force-added with a
  clear handoff reason.
