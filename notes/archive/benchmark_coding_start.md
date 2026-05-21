# Benchmark Coding Start

## Local Sources

- WritingBench: `/Users/larry_1/Opensource/WritingBench`
- PresentBench: `/Users/larry_1/Opensource/auto-skill/data/PresentBench_repo`

## What The Data Already Gives Us

WritingBench is the cleaner first ingest target:

- `benchmark_query/benchmark_all.jsonl`
- fields: `index`, `domain1`, `domain2`, `lang`, `query`, `checklist`
- judge shape: score each criterion from 1 to 10 with `score + reason`

PresentBench is a good second ingest target:

- 238 case directories
- each case has `generation_task/instructions.md`, `judge_prompt.json`, `statistics.yaml`
- each domain has `common_judge_prompt.json` and `judge_weights.yaml`
- material files are Git LFS objects, so `git lfs pull` is needed for actual PDFs
- judge shape: binary checklist items, strict `yes/no`, usually with `boxed{yes}` / `boxed{no}`

## Current Framing

Primary question:

> The user cannot write a skill directly, but can provide several examples. Can the system learn from those examples, extract a reusable skill, and deliver better results on new tasks?

So the main benchmark is **few-shot skill induction from examples**, not skill-gap admission.

`gap` / `admission` analysis is still useful as an optional diagnostic, but it should not drive the first prototype.

## Example Construction Assumption

The few-shot examples are not expected to come from WritingBench or PresentBench as gold artifacts.

They are produced by us:

1. pick related benchmark tasks from the same domain / requirement pattern;
2. generate baseline outputs;
3. optionally use benchmark rubrics/checklists to create high-quality solved examples;
4. strip benchmark-only rubrics, critiques, and score traces before feeding examples to the auto-skill module;
5. hold out separate benchmark tasks for downstream evaluation.

In this setup, WritingBench and PresentBench are substrates: they provide tasks, materials, and judges. They are not assumed to provide user-edited demonstrations.

## First Code Boundary

Start with dataset ingestion, not generation:

1. Normalize both sources into one `SeedExample` JSONL.
2. Group examples by domain / requirement pattern into few-shot train and heldout splits.
3. Use rubrics/checklists to generate or refine solved examples when a benchmark does not provide gold outputs, while preserving a clean user-visible example pack.
4. Extract `SKILL.md + templates + tests` from the few-shot examples.
5. Evaluate whether heldout task quality improves when the generated skill is loaded.

Initial scripts:

- `scripts/data/inspect_benchmarks.py`: inspect source datasets and export example seed records.
- `scripts/data/build_fewshot_splits.py`: build few-shot train/heldout splits for example-driven skill induction.
