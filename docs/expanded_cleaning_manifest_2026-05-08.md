# Expanded Cleaning Manifest 2026-05-08

This manifest records the local 30 WritingBench / 20 PresentBench expanded
cleaning pass. The artifacts live under ignored `runs/expanded/` because they
include generated benchmark-derived content and private eval metadata. Do not
force-add them to the public repo without an explicit data-release decision.

## Scope

- Packs: 50 total, 30 WritingBench and 20 PresentBench.
- Train examples: 150 total, 3 per pack.
- Heldout tasks: 100 total, 2 per pack.
- Primary cleaner: `qwen3.5-plus`.
- Independent audit model: `mimo-v2.5-pro` sample audit for WritingBench train
  examples. PresentBench audit is material-aware and currently uses Qwen as the
  stable judge path, with a MIMO partial run recorded to document long-request
  instability.

Coverage:

| Surface | Coverage |
| --- | --- |
| WritingBench domains | Academic & Engineering 9; Advertising & Marketing 6; Education 3; Finance & Business 5; Literature & Arts 3; Politics & Law 4 |
| PresentBench categories | academia 8; economics 4; education 5; talk 3 |

## Local Artifacts

| Artifact | Rows | Size bytes | SHA256 |
| --- | ---: | ---: | --- |
| `runs/expanded/fewshot_splits.30wb_20pb.jsonl` | 50 | 8784207 | `489bc5c92fab8b0357dbd7537154dbea3efd05fe3f07f65418b9d806089e0ce6` |
| `runs/expanded/fewshot_split_summary.30wb_20pb.json` | 63 | 1321 | `8a50d653e97cfb91748367120450e343b762d2285fdbfcc5fa8240466072ccde` |
| `runs/expanded/example_generation_jobs.30wb_20pb.jsonl` | 150 | 2609475 | `2e381cafc49f54965869b9fd159aa5aedabdb492bc44876a0a88cdea20d00b09` |
| `runs/expanded/example_packs.30wb_20pb.needs_generation.jsonl` | 50 | 2547811 | `c65bc970e04762f706587ffa5fd5a7adbfb21ca6d7af85193c6143056bd2ddd3` |
| `runs/expanded/example_private_eval.30wb_20pb.jsonl` | 50 | 6398933 | `1459856b0618d9df293bc4837b4308fae532c5b3f7ecd5943eafef588f48bb21` |
| `runs/expanded/generated_desired_outputs.30wb_20pb.qwen.jsonl` | 154 | 1565833 | `770cf0521ffb9f3253ec760cce42d362f7157900c7339233bab8fa8da909cd9b` |
| `runs/expanded/example_packs.30wb_20pb.qwen.v1.jsonl` | 50 | 3996521 | `52c43d3cce30def3666ecfdc9c5482b2a13c43987b6ac56b35f5309d0a4b622a` |
| `runs/expanded/train_example_quality_audit.30wb_20pb.mimo.writingbench.sample6.jsonl` | 6 | 40003 | `0fd31d1138cce9a3a83d7a6455611d203607227c1505d4c311c5b4ce61c8836a` |
| `runs/expanded/train_example_quality_audit.30wb_20pb.mimo.writingbench.sample6.summary.json` | 24 | 456 | `649e346ca97b5403f4539930c543836e1a49cd442f7301904bf7cbd00ffe6917` |
| `runs/expanded/train_example_quality_audit.30wb_20pb.mimo.presentbench.sample2.jsonl` | 2 | 11448 | `8b48b762434ee960337ddba6694d2c11ef161f7a4432e1e4395b71a976bf2981` |
| `runs/expanded/train_example_quality_audit.30wb_20pb.mimo.presentbench.sample2.summary.json` | 23 | 442 | `bac7646ebb442dd07e5d28105b79051cc218c6e74a74e7efb131dec0036c7325` |
| `runs/expanded/train_example_quality_audit.30wb_20pb.qwen.presentbench.sample2.jsonl` | 2 | 13745 | `b6e3fae7aeb9281865d74e823d837ea054095df5b3af70da3be7efec185c289b` |
| `runs/expanded/train_example_quality_audit.30wb_20pb.qwen.presentbench.sample2.summary.json` | 24 | 442 | `254236e53059069864ee7b960152e3d52bee13e61505ea55606962b3a712116e` |
| `runs/expanded/expanded_cleaning_status.json` | 93 | 2475 | `ed58ef6cd5a7afc896ee4ad93ba24cb993e70479a1961141e8413964d78f92c3` |

The generated-output file has 154 append-only rows because failed rows were
kept for traceability. The latest row per `(job_id, prompt_sha256)` is 150/150
`success`.

## Reproduction Commands

Build the expanded split:

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

Build packs/jobs/private eval:

```bash
uv run python scripts/data/build_example_packs.py \
  --splits runs/expanded/fewshot_splits.30wb_20pb.jsonl \
  --packs-out runs/expanded/example_packs.30wb_20pb.needs_generation.jsonl \
  --private-out runs/expanded/example_private_eval.30wb_20pb.jsonl \
  --jobs-out runs/expanded/example_generation_jobs.30wb_20pb.jsonl \
  --summary-out runs/expanded/example_pack_summary.30wb_20pb.md
```

Generate desired outputs with Qwen:

```bash
uv run python scripts/data/run_generation_jobs.py \
  --jobs runs/expanded/example_generation_jobs.30wb_20pb.jsonl \
  --out runs/expanded/generated_desired_outputs.30wb_20pb.qwen.jsonl \
  --num-threads 8 \
  --timeout-seconds 900 \
  --max-retries 0 \
  --max-tokens 8192 \
  --resume \
  --allow-partial
```

Retry only failed latest rows:

```bash
uv run python scripts/data/run_generation_jobs.py \
  --jobs runs/expanded/example_generation_jobs.30wb_20pb.jsonl \
  --out runs/expanded/generated_desired_outputs.30wb_20pb.qwen.jsonl \
  --retry-existing-failures-only \
  --num-threads 3 \
  --timeout-seconds 900 \
  --max-retries 3 \
  --max-tokens 16384 \
  --resume \
  --allow-partial
```

Apply successful generations:

```bash
uv run python scripts/data/apply_generated_outputs.py \
  --packs runs/expanded/example_packs.30wb_20pb.needs_generation.jsonl \
  --generations runs/expanded/generated_desired_outputs.30wb_20pb.qwen.jsonl \
  --out runs/expanded/example_packs.30wb_20pb.qwen.v1.jsonl
```

Audit benchmark-flow compliance:

```bash
uv run python scripts/data/audit_benchmark_flow.py \
  --splits runs/expanded/fewshot_splits.30wb_20pb.jsonl \
  --packs runs/expanded/example_packs.30wb_20pb.qwen.v1.jsonl \
  --private-eval runs/expanded/example_private_eval.30wb_20pb.jsonl
```

Observed result:

```json
{"schema_version":"benchmark-flow-audit/v1","status":"ok","errors":[],"warnings":[]}
```

Run the aggregate expanded-cleaning gate:

```bash
uv run python scripts/ops/report_expanded_cleaning_status.py \
  --expect-status ready \
  --out runs/expanded/expanded_cleaning_status.json
```

Observed result:

```json
{
  "status": "ready",
  "errors": [],
  "warnings": []
}
```

This gate recomputes split counts, pack freeze counts, latest generation status,
benchmark-flow audit, and train-example audit schema/status coverage from the
local ignored artifacts.

Run a small independent MIMO quality audit for WritingBench train examples:

```bash
uv run python scripts/eval/audit_train_examples.py \
  --packs runs/expanded/example_packs.30wb_20pb.qwen.v1.jsonl \
  --private-eval runs/expanded/example_private_eval.30wb_20pb.jsonl \
  --source WritingBench \
  --limit-examples 6 \
  --out runs/expanded/train_example_quality_audit.30wb_20pb.mimo.writingbench.sample6.jsonl \
  --summary-out runs/expanded/train_example_quality_audit.30wb_20pb.mimo.writingbench.sample6.summary.json \
  --judge-config-prefix MIMO \
  --judge-max-tokens 4096 \
  --timeout-seconds 600 \
  --max-retries 3 \
  --allow-partial
```

Observed result:

```json
{
  "status_counts": {"success": 6},
  "mean_train_example_quality_score": 6.433333333333334
}
```

Do not use `--judge-max-tokens 1024` for MIMO train-example audit; it produced
`judge_incomplete` rows because MIMO's judge response was truncated.

Run a small material-aware quality audit for PresentBench train examples:

```bash
uv run python scripts/eval/audit_train_examples.py \
  --packs runs/expanded/example_packs.30wb_20pb.qwen.v1.jsonl \
  --private-eval runs/expanded/example_private_eval.30wb_20pb.jsonl \
  --source PresentBench \
  --limit-examples 2 \
  --out runs/expanded/train_example_quality_audit.30wb_20pb.qwen.presentbench.sample2.jsonl \
  --summary-out runs/expanded/train_example_quality_audit.30wb_20pb.qwen.presentbench.sample2.summary.json \
  --judge-config-prefix BAILIAN \
  --judge-max-tokens 8192 \
  --judge-material-chars 8000 \
  --timeout-seconds 600 \
  --max-retries 3 \
  --allow-partial
```

Observed result:

```json
{
  "status_counts": {"success": 2},
  "mean_train_example_quality_score": 7.5
}
```

This PresentBench audit uses a generic rubric/checklist surrogate judge over
the generated text artifact and material excerpts. It is a quality diagnostic,
not the official PresentBench visual/PPT evaluator.

MIMO on the same PresentBench audit path succeeds on the current sample when
`--judge-max-tokens 8192` is used:

```json
{
  "status_counts": {"success": 2},
  "mean_train_example_quality_score": 8.4
}
```

## Failure Trace

Initial Qwen expanded generation produced 147/150 success. The three failures
were:

- one `APIConnectionError`, fixed by retry;
- one `finish_reason=length`, fixed by raising `--max-tokens` to 16384;
- one false `rejected_private_leak` on the phrase `evaluation criteria`, fixed
  by narrowing the private-leak detector and rerunning only the failed job.

Final latest status: 150/150 success.

## Remaining Gaps

- This is cleaned-data evidence, not a paper-level result.
- PresentBench train-example quality audit is implemented only as a
  material-aware generic checklist surrogate; it is not a replacement for the
  official visual/PPT evaluator and has only been sampled on 2 examples in this
  manifest.
- MIMO is used as audit/targeted-regeneration evidence, not as a complete
  alternative cleaner for long PresentBench prompts.
- The expanded artifacts are local ignored run outputs. A release decision is
  needed before committing or publishing them as a dataset snapshot.
