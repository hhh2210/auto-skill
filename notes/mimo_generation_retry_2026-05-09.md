# MIMO Generation Retry Notes 2026-05-09

Scope: targeted retry of ignored local MIMO desired-output generation samples
and freezing of the complete latest-success subset. This subset is
audit/regeneration evidence only; the canonical expanded cleaned dataset remains
the Qwen-generated pack file:

- `runs/expanded/example_packs.30wb_20pb.qwen.v1.jsonl`

## Latest Status

`runs/expanded/generated_desired_outputs.30wb_20pb.mimo.sample20.jsonl`

- Append-only rows: 43
- Latest unique job rows: 36
- Latest status: 36/36 `success`

`runs/expanded/generated_desired_outputs.30wb_20pb.pb_mimo.sample10.jsonl`

- Append-only rows: 16
- Latest unique job rows: 10
- Latest status: 10/10 `success`

## Failure Taxonomy

Observed failures were not JSON parsing failures.

- WritingBench MIMO: transient `APIConnectionError`; targeted
  `--retry-existing-failures-only --max-retries 3` recovered the remaining
  failure.
- PresentBench MIMO: transient `APIConnectionError` first recovered into
  `rejected_incomplete_generation` with default `--max-tokens 4096`; a targeted
  rerun with `--max-tokens 8192` recovered both long-output jobs.
- One older WritingBench row had `rejected_private_leak`, but the latest row for
  the same `(job_id, prompt_sha256)` is now `success`.

## Commands Used

WritingBench targeted retry:

```bash
uv run python scripts/data/run_generation_jobs.py \
  --jobs runs/expanded/example_generation_jobs.30wb_20pb.jsonl \
  --out runs/expanded/generated_desired_outputs.30wb_20pb.mimo.sample20.jsonl \
  --config-prefix MIMO \
  --retry-existing-failures-only \
  --resume \
  --num-threads 8 \
  --max-retries 3 \
  --timeout-seconds 900 \
  --allow-partial
```

PresentBench targeted retry after diagnosing length truncation:

```bash
uv run python scripts/data/run_generation_jobs.py \
  --jobs runs/expanded/example_generation_jobs.30wb_20pb.jsonl \
  --out runs/expanded/generated_desired_outputs.30wb_20pb.pb_mimo.sample10.jsonl \
  --config-prefix MIMO \
  --retry-existing-failures-only \
  --resume \
  --num-threads 2 \
  --max-retries 3 \
  --timeout-seconds 900 \
  --max-tokens 8192 \
  --allow-partial
```

Latest-row summary:

```bash
uv run python - <<'PY'
import json
from collections import Counter
from pathlib import Path

for p in [
    Path("runs/expanded/generated_desired_outputs.30wb_20pb.mimo.sample20.jsonl"),
    Path("runs/expanded/generated_desired_outputs.30wb_20pb.pb_mimo.sample10.jsonl"),
]:
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    latest = {}
    for row in rows:
        latest[(row.get("job_id"), row.get("prompt_sha256"))] = row
    print(p)
    print(" all_rows", len(rows), Counter(row.get("status") for row in rows))
    print(" latest", len(latest), Counter(row.get("status") for row in latest.values()))
PY
```

## Frozen Subset Audit

The complete latest-success MIMO subset was frozen into ignored local artifacts:

- `runs/expanded/example_packs.30wb_20pb.mimo.sample.v1.jsonl`
- `runs/expanded/example_generation_jobs.30wb_20pb.mimo.sample.jsonl`
- `runs/expanded/generated_desired_outputs.30wb_20pb.mimo.sample.latest_success.jsonl`
- `runs/expanded/example_private_eval.30wb_20pb.mimo.sample.jsonl`
- `runs/expanded/benchmark_flow_audit.mimo.sample.json`

Coverage:

- 15 packs total: 12 WritingBench, 3 PresentBench.
- 45 frozen train examples.
- Benchmark-flow audit: `status=ok`, no errors or warnings.

Audit command:

```bash
uv run python scripts/data/audit_benchmark_flow.py \
  --splits runs/expanded/fewshot_splits.30wb_20pb.jsonl \
  --packs runs/expanded/example_packs.30wb_20pb.mimo.sample.v1.jsonl \
  --private-eval runs/expanded/example_private_eval.30wb_20pb.mimo.sample.jsonl \
  --jobs runs/expanded/example_generation_jobs.30wb_20pb.mimo.sample.jsonl \
  --generated-outputs runs/expanded/generated_desired_outputs.30wb_20pb.mimo.sample.latest_success.jsonl \
  --out runs/expanded/benchmark_flow_audit.mimo.sample.json
```

Aggregate gate including the MIMO subset:

```bash
uv run python scripts/ops/report_expanded_cleaning_status.py \
  --require-mimo-subset \
  --expect-status ready
```

## Handoff Guidance

- Do not describe this MIMO subset as the canonical expanded cleaned dataset;
  the canonical 50-pack expanded artifact is still Qwen-generated.
- Use `--max-retries 3` for transient MIMO connection instability.
- Use `--max-tokens 8192` or higher for PresentBench-style desired outputs.
- JSON parsing retry is a separate concern handled by `run_skill_mvp.py` and
  judge/eval scripts via `--parse-max-attempts`; it does not apply to raw
  desired-output text generation unless a future prompt asks for structured
  output.
