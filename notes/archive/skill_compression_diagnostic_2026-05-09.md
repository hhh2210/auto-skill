# Skill Compression Diagnostic 2026-05-09

This note records the current mechanism diagnosis from
`runs/mvp_metrics.heldout2.current.summary.json`. It should guide the next
method loop before expanding benchmark size.

## Command

```bash
uv run python scripts/metrics/summarize_mvp_metrics.py \
  --skills runs/skill_mvp.qwen.mvp.jsonl \
  --skills runs/skill_mvp.qwen.ours_full.writingbench.jsonl \
  --skills runs/skill_mvp.qwen.ours_full.presentbench.jsonl \
  --packs artifacts/packs/example_packs.v1.jsonl \
  --modes prompt_only,few_shot_examples_only,one_shot_skill_from_examples,ours_no_validation,auto_skill,examples_plus_one_shot_skill,examples_plus_feature_skill,slide_constrained_examples_plus_feature_skill,layout_plan_examples_plus_feature_skill \
  --limit-heldout 2 \
  --eval runs/writingbench_official_eval.qwen.five_modes.no_thinking_auto_skill.jsonl \
  --eval runs/writingbench_official_eval.qwen.examples_plus_skill.heldout2.jsonl \
  --eval runs/presentbench_surrogate_eval.qwen.mvp.jsonl \
  --eval runs/presentbench_surrogate_eval.qwen.auto_skill.jsonl \
  --eval runs/presentbench_surrogate_eval.qwen.examples_plus_skill.heldout2.jsonl \
  --eval runs/presentbench_surrogate_eval.qwen.slide_constrained_examples_plus_feature.heldout2.jsonl \
  --eval runs/presentbench_surrogate_eval.qwen.layout_plan_examples_plus_feature.heldout2.jsonl \
  --eval runs/writingbench_official_eval.mimo_judge.five_modes.heldout2.jsonl \
  --eval runs/writingbench_official_eval.mimo_judge.examples_plus_skill.heldout2.jsonl \
  --self-consistency runs/self_consistency.writingbench.qwen.mvp.jsonl \
  --out runs/mvp_metrics.heldout2.current.summary.json
```

## Main Result

The current system achieves **context compression** but does not preserve enough
task-solving signal for skill-only inference.

- WritingBench skill-only modes use much less generation input than
  `few_shot_examples_only`:
  - `one_shot_skill_from_examples`: 21.2% of few-shot input tokens.
  - `ours_no_validation`: 21.4% of few-shot input tokens.
  - `auto_skill`: 24.0% of few-shot input tokens.
- PresentBench surrogate shows a similar token reduction:
  - `one_shot_skill_from_examples`: 28.7% of few-shot input tokens.
  - `ours_no_validation`: 27.0% of few-shot input tokens.
- But WritingBench skill-only negative transfer is high:
  - Qwen judge: `ours_no_validation` 75%, `auto_skill` 50%,
    `one_shot_skill_from_examples` 37.5%.
  - MIMO judge: `ours_no_validation` 75%, `auto_skill` 62.5%,
    `one_shot_skill_from_examples` 50%.
- WritingBench `examples_plus_feature_skill` is the strongest current Qwen-judge
  MVP mechanism and remains competitive under MIMO:
  - Qwen judge negative transfer: 12.5%.
  - MIMO judge negative transfer: 25%, tied with
    `examples_plus_one_shot_skill` on negative-transfer rate.

## Artifact Shape

Average skill artifact metrics across current successful rows:

| mode | chars | required rules | optional rules | candidate rules |
| --- | ---: | ---: | ---: | ---: |
| one_shot_skill_from_examples | 5,602 | 6.38 | 3.00 | 0.00 |
| auto_skill_feature_driven_no_validation | 5,247 | 7.00 | 4.31 | 4.38 |
| auto_skill_ours_full | 6,845 | 7.62 | 5.12 | 4.25 |

Feature-driven no-validation is slightly shorter than one-shot while adding
structured candidate rules. Ours-full is longer and more conditional, matching
the separate LOO diagnostic in `notes/p4a_loo_merge_diagnostic.md`.

## Self-Consistency

The self-consistency diagnostic is high for both current skill modes:

| mode | overall self-consistency | constraint recall | leakage penalty | unsupported specificity penalty |
| --- | ---: | ---: | ---: | ---: |
| one_shot_skill_from_examples | 8.5 | 8.5 | 0.5 | 1.0 |
| auto_skill_feature_driven_no_validation | 9.0 | 9.0 | 0.0 | 1.0 |

This weakens the hypothesis that the feature-driven skill simply fails to encode
the training examples. The more likely issue is that the skill encodes a broad
abstract signature but does not preserve enough task-specific control signal for
heldout generation when used alone.

## Interpretation

Do not frame the next paper claim as "skill-only beats examples" yet. The
current defensible mechanism claim is narrower:

```text
examples -> feature skill -> lower-cost structured control signal
```

The cost benefit is real only for skill-only modes. The best MVP WritingBench
performance signal appears when the skill augments examples, not when it
replaces them, but this is judge- and sample-sensitive. This is an
engineering/research tension:

- `skill-only`: cheap but unstable.
- `examples_plus_skill`: more accurate on WritingBench, but does not reduce
  context cost versus few-shot and is negative on PresentBench surrogate.

## Next Method Loop

Run targeted ablations before expanding N:

1. **Abstract example signatures**: replace raw train examples in
   `examples_plus_feature_skill` with compact signatures produced by the
   feature extractor. This tests whether we can keep the WritingBench gain while
   reducing context cost and avoiding PresentBench raw-example interference.
2. **Skill-only constraint pack**: add a compact table of task family, output
   structure, hard constraints, and do-not-generalize rules next to `SKILL.md`.
   This tests whether skill-only losses come from missing operational anchors.
3. **Disable LOO merge as mainline**: keep `auto_skill_ours_full` as an ablation
   until majority-vote contradiction handling is implemented.
4. **Evaluator calibration first**: do not expand WritingBench N until the
   Qwen/MIMO sign-flip taxonomy is reviewed or adjudicated.

## Current Confidence

Not 100% confident in the method strategy. Data cleaning is handoff-ready, but
the auto-skill strategy should pivot from standalone compression to
cost-aware examples-plus-signatures or constraint-pack induction. Do not treat
the current `examples_plus_feature_skill` result as a stable cross-benchmark
claim until the expanded WritingBench judge sensitivity is resolved.
