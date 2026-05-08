# Deliverable Guard Probe 2026-05-09

This note records a targeted probe after adding current-task deliverable
priority guidance to `build_heldout_generation_prompt`.

## Prompt Change

The heldout generation prompt now explicitly says:

- produce the completed artifact requested by the heldout task;
- do not answer with a plan, outline, checklist, analysis, or rubric mapping
  unless the task asks for that artifact type;
- treat the heldout task and materials as higher priority than example or skill
  patterns;
- do not import example-specific facts, domains, entities, or section topics
  unless they appear in the heldout task.

The prompt contract is versioned as
`HELDOUT_GENERATION_PROMPT_VERSION = "deliverable-priority-v2"`. WritingBench
official eval and PresentBench surrogate eval rows now persist this version in
runtime metadata so `--resume` does not silently reuse rows generated under an
older prompt contract.

## Probe

Artifact:
`runs/expanded/writingbench_official_eval.qwen.deliverable_guard_probe.education.jsonl`

Command shape:

```bash
uv run python scripts/eval/run_writingbench_official_eval.py \
  --packs runs/expanded/example_packs.30wb_20pb.qwen.v1.jsonl \
  --skills runs/expanded/skill_mvp.qwen.sample4_wb.jsonl \
  --private-eval runs/expanded/example_private_eval.30wb_20pb.jsonl \
  --writingbench-root ../WritingBench \
  --pack-id writingbench_Education_Educational_Consulting_en \
  --limit-heldout 1 \
  --modes prompt_only,ours_no_validation,examples_plus_feature_skill \
  --out runs/expanded/writingbench_official_eval.qwen.deliverable_guard_probe.education.jsonl \
  --summary-out runs/expanded/writingbench_official_eval.qwen.deliverable_guard_probe.education.summary.json \
  --num-threads 3 \
  --timeout-seconds 900 \
  --max-retries 3 \
  --parse-max-attempts 3 \
  --judge-max-tokens 1024 \
  --allow-partial \
  --resume
```

The final regenerated rows all include
`heldout_generation_prompt_version=deliverable-priority-v2`.

Scores:

| Mode | Score | Delta vs prompt_only |
| --- | ---: | ---: |
| `prompt_only` | 7.6 | n/a |
| `examples_plus_feature_skill` | 7.4 | -0.2 |
| `ours_no_validation` | 5.4 | -2.2 |

## Interpretation

The guardrail did not fully fix the severe Education Consulting failure mode.
It substantially reduced the `examples_plus_feature_skill` negative delta in
this one-cell probe, but `ours_no_validation` still underperforms prompt-only
by a large margin and the result is too small to support a method claim.

The remaining failure is not simply "the model wrote an outline." The Qwen judge
still penalizes practical operational value, expert integration, and source
context pollution. This supports the current risk-register decision: do not
expand N for performance claims yet; next work should target candidate quality
and example/material contamination.

This probe artifact is local diagnostic evidence, not a canonical benchmark
artifact.
