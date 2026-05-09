# Expanded Cleaning Manifest 2026-05-08

This manifest records the local expanded cleaning passes. The artifacts live
under ignored `runs/expanded/` because they include generated
benchmark-derived content and private eval metadata. Do not force-add them to
the public repo without an explicit data-release decision.

2026-05-09 refresh: the artifact paths and coverage counts below were rechecked
after commits `6212eff` and `aa85fe5`. `report_expanded_cleaning_status.py`
still reports `status=ready` for the 30WB/20PB Qwen snapshot and for the MIMO
subset. The max-available Qwen snapshot still passes `audit_benchmark_flow.py`
with `status=ok`, no errors, and no warnings when the job/generated-output
provenance files are supplied.

## Scope

### Max-Available Qwen Snapshot

- Packs: 142 total, 114 WritingBench and 28 PresentBench.
- Train examples: 426 total, 3 per pack.
- Heldout tasks: 284 total, 2 per pack.
- Primary cleaner: `qwen3.5-plus`.
- Status: frozen locally and benchmark-flow audited.

Coverage:

| Surface | Coverage |
| --- | --- |
| WritingBench languages | English 69; Chinese 45 |
| WritingBench domains | Academic & Engineering 18; Advertising & Marketing 15; Education 12; Finance & Business 23; Literature & Arts 20; Politics & Law 26 |
| PresentBench categories | academia 13; advertising 1; economics 6; education 5; talk 3 |

### 30 WritingBench / 20 PresentBench Snapshot

- Packs: 50 total, 30 WritingBench and 20 PresentBench.
- Train examples: 150 total, 3 per pack.
- Heldout tasks: 100 total, 2 per pack.
- Primary cleaner: `qwen3.5-plus`.
- Independent audit model: `mimo-v2.5-pro` sample audit for WritingBench train
  examples. MIMO also has a local audited subset pack with 15 packs and 45
  frozen train examples; it is not the canonical expanded dataset because full
  50-pack MIMO cleaning is not frozen.

Coverage:

| Surface | Coverage |
| --- | --- |
| WritingBench domains | Academic & Engineering 9; Advertising & Marketing 6; Education 3; Finance & Business 5; Literature & Arts 3; Politics & Law 4 |
| PresentBench categories | academia 8; economics 4; education 5; talk 3 |

## Local Artifacts

### Max-Available Qwen Snapshot

| Artifact | Rows / status |
| --- | ---: |
| `runs/expanded/fewshot_splits.max_available.jsonl` | 142 splits |
| `runs/expanded/fewshot_split_summary.max_available.json` | 114 eligible WritingBench groups; 28 eligible PresentBench groups |
| `runs/expanded/example_generation_jobs.max_available.jsonl` | 426 jobs |
| `runs/expanded/example_packs.max_available.needs_generation.jsonl` | 142 packs before generated outputs |
| `runs/expanded/example_private_eval.max_available.jsonl` | 142 private-eval rows |
| `runs/expanded/generated_desired_outputs.max_available.qwen.jsonl` | 430 append-only rows, including 4 retained failed attempts |
| `runs/expanded/generated_desired_outputs.max_available.qwen.latest_success.jsonl` | 426 latest successful rows |
| `runs/expanded/example_packs.max_available.qwen.v1.jsonl` | 142 frozen packs |
| `runs/expanded/benchmark_flow_audit.max_available.qwen.json` | `status=ok` |
| `runs/expanded/expanded_cleaning_status.max_available.qwen.json` | `status=ready` |

The max-available generated-output file is append-only. Use the
`*.latest_success.jsonl` view for benchmark-flow validation, because the raw
append-only log intentionally keeps old failed rows for traceability.

### 30 WritingBench / 20 PresentBench Snapshot

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
| `runs/expanded/example_packs.30wb_20pb.mimo.sample.v1.jsonl` | 15 | 987162 | `e28bb208ff8391fb03ed455f54a0d8c545b4f38a1f6b206470910cd86aab7cfc` |
| `runs/expanded/example_generation_jobs.30wb_20pb.mimo.sample.jsonl` | 45 | 546968 | `739bb92a2be9f42dd0c15b95bbf77e5b69b3816e4fd668a02077354642b44df9` |
| `runs/expanded/generated_desired_outputs.30wb_20pb.mimo.sample.latest_success.jsonl` | 45 | 367831 | `fc9c2b00b58addc7fe49057654bc89f4f1dc04e73771fdc12bec88a07d2a0f35` |
| `runs/expanded/example_private_eval.30wb_20pb.mimo.sample.jsonl` | 15 | 1201405 | `fa99debf418850ce7961e790caf7ad29ac8e32900328d26575db84bc82eff867` |
| `runs/expanded/benchmark_flow_audit.mimo.sample.json` | 6 | 102 | `f9b9b9d265a2299ad7393c678248e3af381c92c4f79675d93599355e07be1479` |

The generated-output file has 154 append-only rows because failed rows were
kept for traceability. The latest row per `(job_id, prompt_sha256)` is 150/150
`success`.

## Reproduction Commands

### Max-Available Qwen Snapshot

Build the max-available split:

```bash
uv run python scripts/data/build_fewshot_splits.py \
  --writingbench-root ../WritingBench \
  --presentbench-root data/PresentBench_repo \
  --train-size 3 \
  --heldout-size 2 \
  --max-writing-groups 999 \
  --max-present-groups 999 \
  --out runs/expanded/fewshot_splits.max_available.jsonl \
  --summary-out runs/expanded/fewshot_split_summary.max_available.json
```

Build packs/jobs/private eval:

```bash
uv run python scripts/data/build_example_packs.py \
  --splits runs/expanded/fewshot_splits.max_available.jsonl \
  --packs-out runs/expanded/example_packs.max_available.needs_generation.jsonl \
  --private-out runs/expanded/example_private_eval.max_available.jsonl \
  --jobs-out runs/expanded/example_generation_jobs.max_available.jsonl \
  --summary-out runs/expanded/example_pack_summary.max_available.md
```

Generate desired outputs with Qwen. The completed run seeded 102 prior
successes from the 30WB/20PB snapshot, then generated the remaining 324 jobs:

```bash
uv run python scripts/data/run_generation_jobs.py \
  --jobs runs/expanded/example_generation_jobs.max_available.jsonl \
  --out runs/expanded/generated_desired_outputs.max_available.qwen.jsonl \
  --resume \
  --num-threads 32 \
  --temperature 0.2 \
  --timeout-seconds 900 \
  --max-retries 0 \
  --max-tokens 8192 \
  --allow-partial
```

The first max-available pass finished 320/324 new jobs successfully in 1216.9s
and left 4 `rejected_incomplete_generation` rows. Inspection showed
`finish_reason=length` or a thinking runaway, not JSON parsing failure. The
successful cleanup was a targeted retry with thinking disabled and a larger
completion budget:

```bash
uv run python scripts/data/run_generation_jobs.py \
  --jobs runs/expanded/example_generation_jobs.max_available.jsonl \
  --out runs/expanded/generated_desired_outputs.max_available.qwen.jsonl \
  --retry-existing-failures-only \
  --resume \
  --num-threads 4 \
  --temperature 0.2 \
  --timeout-seconds 900 \
  --max-retries 0 \
  --max-tokens 20000 \
  --no-enable-thinking \
  --allow-partial
```

Apply successful generations:

```bash
uv run python scripts/data/export_latest_successful_generations.py \
  --generations runs/expanded/generated_desired_outputs.max_available.qwen.jsonl \
  --out runs/expanded/generated_desired_outputs.max_available.qwen.latest_success.jsonl \
  --expect-successes 426

uv run python scripts/data/apply_generated_outputs.py \
  --packs runs/expanded/example_packs.max_available.needs_generation.jsonl \
  --generations runs/expanded/generated_desired_outputs.max_available.qwen.latest_success.jsonl \
  --out runs/expanded/example_packs.max_available.qwen.v1.jsonl
```

Audit benchmark-flow compliance using the latest-success view:

```bash
uv run python scripts/data/audit_benchmark_flow.py \
  --splits runs/expanded/fewshot_splits.max_available.jsonl \
  --packs runs/expanded/example_packs.max_available.qwen.v1.jsonl \
  --private-eval runs/expanded/example_private_eval.max_available.jsonl \
  --jobs runs/expanded/example_generation_jobs.max_available.jsonl \
  --generated-outputs runs/expanded/generated_desired_outputs.max_available.qwen.latest_success.jsonl \
  --out runs/expanded/benchmark_flow_audit.max_available.qwen.json
```

Run the max-available cleaning gate:

```bash
uv run python scripts/ops/report_expanded_cleaning_status.py \
  --splits runs/expanded/fewshot_splits.max_available.jsonl \
  --packs runs/expanded/example_packs.max_available.qwen.v1.jsonl \
  --private-eval runs/expanded/example_private_eval.max_available.jsonl \
  --jobs runs/expanded/example_generation_jobs.max_available.jsonl \
  --generated-outputs runs/expanded/generated_desired_outputs.max_available.qwen.latest_success.jsonl \
  --expect-packs 142 \
  --expect-train-examples 426 \
  --expect-heldout-tasks 284 \
  --expect-generation-jobs 426 \
  --skip-mimo-subset \
  --expect-status ready \
  --out runs/expanded/expanded_cleaning_status.max_available.qwen.json
```

Observed result: `status=ready`, with no errors or warnings.

### 30 WritingBench / 20 PresentBench Snapshot

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
  --private-eval runs/expanded/example_private_eval.30wb_20pb.jsonl \
  --jobs runs/expanded/example_generation_jobs.30wb_20pb.jsonl \
  --generated-outputs runs/expanded/generated_desired_outputs.30wb_20pb.qwen.jsonl
```

Observed result:

```json
{"schema_version":"benchmark-flow-audit/v1","status":"ok","errors":[],"warnings":[]}
```

Run the aggregate expanded-cleaning gate:

```bash
uv run python scripts/ops/report_expanded_cleaning_status.py \
  --require-mimo-subset \
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
benchmark-flow audit, train-example audit schema/status coverage, and the local
MIMO subset status from the ignored artifacts.

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

MIMO targeted desired-output generation samples are also green in latest-row
view:

- `runs/expanded/generated_desired_outputs.30wb_20pb.mimo.sample20.jsonl`:
  36/36 latest unique WritingBench rows are `success`.
- `runs/expanded/generated_desired_outputs.30wb_20pb.pb_mimo.sample10.jsonl`:
  10/10 latest unique PresentBench rows are `success` after raising
  `--max-tokens` to 8192.

These rows were frozen into a local subset pack:

- `runs/expanded/example_packs.30wb_20pb.mimo.sample.v1.jsonl`: 15 packs
  (12 WritingBench, 3 PresentBench), 45 frozen train examples.
- `runs/expanded/benchmark_flow_audit.mimo.sample.json`: `status=ok`, no
  errors or warnings.

See `notes/mimo_generation_retry_2026-05-09.md` for the retry trace. Do not
describe this subset as the canonical expanded dataset; full 50-pack MIMO
cleaning has not been frozen.

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
- Full experiment readiness remains blocked on missing PresentBench official
  score YAMLs. The score summarizer now accepts `--solver-model` and
  `--judge-model` so placeholder rows can still carry complete model identity,
  but it does not replace the official judge run.
- PresentBench train-example quality audit is implemented only as a
  material-aware generic checklist surrogate; it is not a replacement for the
  official visual/PPT evaluator and has only been sampled on 2 examples in this
  manifest.
- MIMO is used as audit/targeted-regeneration evidence, not as a complete
  alternative cleaner. The current MIMO subset is frozen/audited locally, but it
  covers only 15 packs rather than the full 50-pack expanded split.
- The expanded artifacts are local ignored run outputs. A release decision is
  needed before committing or publishing them as a dataset snapshot.
