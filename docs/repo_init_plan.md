# Repo Init Plan

## Current Meeting Outcome

The repo should support two immediate tracks:

1. **Data cleaning scripts**: make benchmark-derived examples and splits reproducible.
2. **Auto-skills module**: turn user examples into reusable skill packages and compare against direct examples and one-shot skill drafting.

The collaboration model should be branch + PR for large features, with identical
`AGENTS.md` / `CLAUDE.md`, `uv` commands, tests, and CI as the shared
communication layer.

## Phase 0: Data-First Guardrails

- Establish canonical framing: input is user examples only.
- Separate benchmark-only metadata from auto-skill module input.
- Build clean example packs before over-engineering the auto-skill framework.
- Add validation commands and CI as support, not as the research center.
- Keep `scripts/` thin and move shared logic under `src/auto_skill/`.

## Phase 1: Data Cleaning

Deliverables:

- unified JSONL schema for source tasks;
- split schema for `train_examples` and `heldout_tasks`;
- validators for split files;
- summary report per benchmark source;
- deterministic sampling with seed and reproducible config.

Near-term scripts:

- `scripts/data/inspect_benchmarks.py`
- `scripts/data/build_fewshot_splits.py`
- `scripts/data/build_example_packs.py`
- `scripts/data/validate_splits.py`

Primary artifacts:

- `artifacts/packs/example_packs.v1.jsonl`: checked-in clean packs after generated outputs are applied.
- `artifacts/splits/fewshot_splits.jsonl`: checked-in train/heldout split.
- `artifacts/private/example_private_eval.jsonl`: checked-in rubric/judge/checklist metadata for construction and evaluation.
- `artifacts/packs/example_packs.jsonl`: ignored intermediate user-visible inputs before desired outputs.
- `artifacts/jobs/example_generation_jobs.jsonl`: ignored private desired-output generation queue.
- `artifacts/jobs/generated_desired_outputs.jsonl`: ignored raw Bailian/OpenAI-compatible API generations.

API configuration stays local in `.env`:

- `BAILIAN_BASE_URL`
- `BAILIAN_API_KEY`
- `BAILIAN_MODEL`

## Phase 2: Auto-Skills Module

Deliverables:

- example feature extraction schema;
- cross-example aggregation schema;
- conflict/outlier reporting;
- one-shot skill baseline;
- feature-driven skill writer;
- leave-one-out validation path when there are enough examples;
- generated package shape: `SKILL.md`, `metadata.yaml`, optional `templates/`, optional `tests/`.

The first implementation can be prompt-builder plus structured IO. LLM execution and benchmark scoring can be added after schemas are stable.

## Phase 3: Evaluation

First matrix:

- `prompt_only`
- `few_shot_examples_only`
- `one_shot_skill_from_examples`
- `ours_no_validation`
- `ours_full`

Required metrics:

- heldout benchmark score;
- delta vs prompt-only and examples-only;
- negative transfer rate;
- token cost at heldout inference;
- skill length and rule support count;
- failure-mode notes.

## Ownership Suggestions

- Data owner: ingestion, cleaning, split validation, artifact reproducibility.
- Module owner: feature schema, skill compiler, skill package writer.
- Eval owner: baselines, scoring harness, result tables, CI checks.

These are ownership lanes, not silos. Schema changes should be reviewed by all active contributors.
