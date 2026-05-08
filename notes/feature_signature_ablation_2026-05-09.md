# Feature Signature Ablation Smoke

Date: 2026-05-09

## Motivation

Previous MVP diagnostics showed that full `SKILL.md` artifacts reduce input
tokens but often hurt WritingBench heldout scores. This smoke tests a smaller
ablation surface: use only compact, example-derived `cross_example_report`
fields as "Abstract Feature Signatures" instead of the full generated skill.

The feature signature context is built from user-visible training examples only:

- `stable_features`
- `candidate_rules` plus support counts
- `optional_features`
- `conflicts`
- `outliers`

It does not use private rubrics, heldout feedback, official scores, or judge
traces.

## Code Surface

- `src/auto_skill/mvp.py`
  - `build_feature_signature_context(...)`
  - `build_heldout_generation_prompt(...)` now includes user examples for
    `examples_plus_feature_signatures`.
- `scripts/eval/run_writingbench_official_eval.py`
  - new modes: `feature_signatures_only`,
    `examples_plus_feature_signatures`.
- `scripts/eval/run_heldout_eval.py`
  - same new modes for surrogate heldout eval.

## Smoke Command

```bash
uv run python scripts/eval/run_writingbench_official_eval.py \
  --packs runs/expanded/example_packs.30wb_20pb.qwen.v1.jsonl \
  --skills runs/expanded/skill_mvp.qwen.sample4_wb.jsonl \
  --private-eval runs/expanded/example_private_eval.30wb_20pb.jsonl \
  --writingbench-root ../WritingBench \
  --pack-id writingbench_Academic_Engineering_Paper_Outline_en \
  --modes feature_signatures_only,examples_plus_feature_signatures \
  --limit-heldout 1 \
  --out runs/expanded/writingbench_official_eval.qwen.feature_signatures_smoke.jsonl \
  --summary-out runs/expanded/writingbench_official_eval.qwen.feature_signatures_smoke.summary.json \
  --num-threads 2 \
  --timeout-seconds 600 \
  --max-retries 3 \
  --parse-max-attempts 3 \
  --max-tokens 8192 \
  --judge-max-tokens 1024 \
  --allow-partial
```

## Result

Pack: `writingbench_Academic_Engineering_Paper_Outline_en`  
Task: `writingbench_Academic_Engineering_Paper_Outline_en::heldout::0`

| mode | Qwen WritingBench score |
|---|---:|
| `prompt_only` | 4.0 |
| `few_shot_examples_only` | 4.4 |
| `one_shot_skill_from_examples` | 2.4 |
| `ours_no_validation` | 2.8 |
| `examples_plus_one_shot_skill` | 2.8 |
| `examples_plus_feature_skill` | 2.8 |
| `feature_signatures_only` | 2.6 |
| `examples_plus_feature_signatures` | 3.0 |

The feature-signature modes ran successfully, but they did not close the gap to
prompt-only or few-shot examples on this task. `examples_plus_feature_signatures`
is slightly better than the previous skill modes on this single cell, but still
well below `few_shot_examples_only`.

## Interpretation

This is not evidence for a new winning method. It is evidence that simply
compressing `cross_example_report` into a shorter context does not recover the
task-completion information lost by skill-only compression.

The next method loop should focus on preserving current-task deliverable
semantics and distinguishing transferable structure from example-specific
content, not just shortening `SKILL.md`.
