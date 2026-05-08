# Expanded Cleaning Strategy Snapshot

This note records the current evidence for expanding beyond the 8-pack MVP
without jumping directly to a full benchmark clean.

## Current Recommendation

Use Qwen3.5-Plus as the primary desired-output generation model for the next
expanded cleaning pass. Use MIMO as an independent audit / targeted regeneration
model, especially for shorter WritingBench text tasks. MIMO targeted generation
samples now have green latest-row status, including a small PresentBench sample
after raising `--max-tokens` to 8192, but this is raw generated-output evidence.
Do not call MIMO a primary cleaner until matching MIMO packs are frozen and
benchmark-flow audited.

## Expanded Split Trial

Command:

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

Observed split:

- 50 packs total: 30 WritingBench, 20 PresentBench.
- 150 train examples and 100 heldout tasks.
- WritingBench language coverage: 20 English packs, 10 Chinese packs.
- WritingBench primary domains covered: Academic & Engineering, Advertising &
  Marketing, Education, Finance & Business, Literature & Arts, Politics & Law.
- PresentBench categories covered: academia, economics, education, talk.

The generated job queue has 150 desired-output jobs. Prompt size is much larger
for PresentBench:

| Source | Jobs | Mean prompt chars | Max prompt chars |
| --- | ---: | ---: | ---: |
| WritingBench | 90 | 5,779 | 27,355 |
| PresentBench | 60 | 25,681 | 29,971 |
| Combined | 150 | 13,740 | 29,971 |

## API Sample Results

### WritingBench Sample

Initial 20-job WritingBench sample:

| Model | Latest unique success | Main failure mode | Throughput note |
| --- | ---: | --- | --- |
| Qwen3.5-Plus | 39 / 39 | one `finish_reason=length`, fixed by `--max-tokens 8192` | about 8.3 jobs/min in first pass |
| MIMO v2.5 Pro | 36 / 36 | transient `APIConnectionError` and one private-leak rejection, both fixed by targeted retry | 17.5 jobs/min first pass, slower on retries |

Takeaway: Qwen is more stable. MIMO can be useful, but retries and private-leak
checks are necessary.

### PresentBench Sample

10-job PresentBench sample:

| Model | Latest unique success | Main failure mode | Throughput |
| --- | ---: | --- | ---: |
| Qwen3.5-Plus | 10 / 10 | none in sample | 1.30 jobs/min |
| MIMO v2.5 Pro | 10 / 10 | default `--max-tokens 4096` recovered connection errors into `finish_reason=length`; `--max-tokens 8192` fixed the two long jobs | 1.04 jobs/min first pass |

Takeaway: PresentBench prompts are long and latency-bound. Qwen is currently the
canonical primary cleaner for the frozen 30WB/20PB pack. MIMO can generate a
small PresentBench sample when token budget is raised, but this is not yet a
frozen/audited MIMO pack.

## Operational Rules

- For transient API/network errors, use `--max-retries 3`.
- For `finish_reason=length`, increase `--max-tokens` and rerun with
  `--retry-existing-failures-only`.
- For `rejected_private_leak`, inspect whether it is a real leak or guard false
  positive before accepting output.
- Use `--source PresentBench` or `--source WritingBench` when sampling one
  benchmark family; otherwise `--limit` follows file order and will start with
  WritingBench.
- Use `--retry-existing-failures-only` for cleanup retries so failed rows are
  retried without expanding the sample window.

## Completed 30WB/20PB Cleaning Pass

The Qwen expanded pass has completed locally. See
`docs/expanded_cleaning_manifest_2026-05-08.md` for the artifact hashes,
commands, and audit evidence.

Latest local result:

- `runs/expanded/generated_desired_outputs.30wb_20pb.qwen.jsonl`: latest row per
  `(job_id, prompt_sha256)` is 150/150 `success`.
- `runs/expanded/example_packs.30wb_20pb.qwen.v1.jsonl`: 50 frozen packs, 150
  generated train examples, 100 heldout tasks.
- `audit_benchmark_flow.py`: `status=ok`, no errors or warnings.
- MIMO WritingBench audit sample: 6/6 success, mean score 6.43 with
  `--judge-max-tokens 4096`.
- PresentBench material-aware audit sample: Qwen judge 2/2 success, mean score
  7.5 with the generic rubric/checklist surrogate judge. MIMO on the same path
  succeeds 2/2 when `--judge-max-tokens 8192` is used.
- MIMO targeted desired-output generation samples: 36/36 latest WritingBench
  rows and 10/10 latest PresentBench rows are `success`; see
  `notes/mimo_generation_retry_2026-05-09.md`. These are raw generated-output
  samples, not canonical MIMO-cleaned packs.

## Reproduction Commands

Run the 30WB/20PB expanded clean with Qwen first:

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

Then retry failures by type:

```bash
uv run python scripts/data/run_generation_jobs.py \
  --jobs runs/expanded/example_generation_jobs.30wb_20pb.jsonl \
  --out runs/expanded/generated_desired_outputs.30wb_20pb.qwen.jsonl \
  --retry-existing-failures-only \
  --num-threads 4 \
  --timeout-seconds 900 \
  --max-retries 3 \
  --max-tokens 8192 \
  --resume \
  --allow-partial
```

After outputs are frozen into packs, run:

```bash
uv run python scripts/data/audit_benchmark_flow.py \
  --splits runs/expanded/fewshot_splits.30wb_20pb.jsonl \
  --packs runs/expanded/example_packs.30wb_20pb.qwen.v1.jsonl \
  --private-eval runs/expanded/example_private_eval.30wb_20pb.jsonl
```

Only consider full benchmark cleaning after sampled quality audits are expanded
or replaced with official-score-based inspection for generated slide examples.
