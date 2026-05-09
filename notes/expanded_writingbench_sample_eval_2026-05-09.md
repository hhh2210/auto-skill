# Expanded WritingBench Sample Eval 2026-05-09

This note records a small heldout eval on four WritingBench packs from the
expanded 30-WB / 20-PB cleaned workspace. These packs are not part of the
checked-in MVP four-pack slice.

## Artifacts

- Skills:
  `runs/expanded/skill_mvp.qwen.sample4_wb.jsonl`
- Qwen judge:
  `runs/expanded/writingbench_official_eval.qwen.sample4_wb.heldout1.jsonl`
- MIMO judge-swap, reusing the Qwen candidate outputs:
  `runs/expanded/writingbench_official_eval.mimo_judge.sample4_wb.heldout1.jsonl`
- Calibration packet:
  `runs/expanded/judge_disagreements.qwen_vs_mimo.sample4_wb.heldout1.jsonl`
  and
  `runs/expanded/judge_disagreements.qwen_vs_mimo.sample4_wb.heldout1.md`

Selected packs:

- `writingbench_Advertising_Marketing_Travel_Guide_zh`
- `writingbench_Education_Educational_Consulting_en`
- `writingbench_Academic_Engineering_Paper_Outline_en`
- `writingbench_Advertising_Marketing_Product_Description_en`

Setup:

- 4 packs x 1 heldout x 6 modes = 24 cells per judge.
- Skill induction used Qwen3.5-Plus with `--no-leave-one-out`.
- MIMO judge-swap used `--reuse-candidates-from` so Qwen and MIMO scored the
  same candidate outputs.
- This is a smoke sample, not a paper-level result.

## Mean Scores

| Mode | Qwen mean | Qwen delta | MIMO mean | MIMO delta | Sign |
| --- | ---: | ---: | ---: | ---: | --- |
| prompt_only | 6.050 | n/a | 6.200 | n/a | n/a |
| few_shot_examples_only | 6.200 | +0.150 | 7.250 | +1.050 | same positive |
| one_shot_skill_from_examples | 5.150 | -0.900 | 7.050 | +0.850 | flip |
| ours_no_validation | 4.700 | -1.350 | 6.500 | +0.300 | flip |
| examples_plus_one_shot_skill | 5.600 | -0.450 | 7.500 | +1.300 | flip |
| examples_plus_feature_skill | 5.250 | -0.800 | 7.200 | +1.000 | flip |

## Interpretation

The expanded sample increases, rather than reduces, judge sensitivity:

- Qwen judge says skill-only and examples-plus-skill are net negative on this
  sample.
- MIMO judge says the same candidate outputs are net positive for all skill
  modes except `ours_no_validation` is only mildly positive.
- `few_shot_examples_only` is positive under both judges.

This means the current strategy is not factually stable enough for a method
claim. The immediate research risk is no longer only "does the skill help?", but
"which judge behavior are we optimizing?" A paper-facing experiment needs at
least one of:

- a third independent judge;
- official benchmark evaluation where available;
- human/rubric spot checks on sign-flip cells;
- or a stricter evaluator calibration protocol before increasing sample size.

## Sign-Flip Cells

The strongest sign flips are:

- `writingbench_Education_Educational_Consulting_en`, `ours_no_validation`:
  Qwen delta -4.2, MIMO delta +1.2.
- `writingbench_Education_Educational_Consulting_en`,
  `examples_plus_feature_skill`: Qwen delta -2.4, MIMO delta +1.2.
- `writingbench_Academic_Engineering_Paper_Outline_en`,
  `examples_plus_one_shot_skill`: Qwen delta -1.2, MIMO delta +1.8.

These cells should be inspected manually before using either judge as the
primary decision signal.

## Third-Party Calibration Review

A background calibration review inspected the 10 sign-flip packets. This is not
a human label, but it is useful failure attribution evidence:

- For Academic Paper Outline and Education Consulting, the Qwen negative
  deltas were judged more credible. The candidates were often outlines/plans or
  source-context-polluted responses rather than completed user deliverables.
- MIMO appeared to over-reward surface structure, rubric keywords, headings,
  and source-term reuse in those cells.
- Product Description and Travel Guide were less decisive; several deltas were
  small enough that the sign flip should be treated as calibration noise.
- The next research step should be evaluator calibration and failure taxonomy,
  not larger-N WritingBench performance claims.

The regenerated packet includes full candidate/baseline output hashes and
length stats, private rubric context, task context, and the skill artifact
preview/hash used by each skill-related mode.

The calibration packet was generated with:

```bash
uv run python scripts/metrics/export_judge_disagreements.py \
  --left runs/expanded/writingbench_official_eval.qwen.sample4_wb.heldout1.jsonl \
  --right runs/expanded/writingbench_official_eval.mimo_judge.sample4_wb.heldout1.jsonl \
  --left-label qwen_judge \
  --right-label mimo_judge \
  --modes few_shot_examples_only,one_shot_skill_from_examples,ours_no_validation,examples_plus_one_shot_skill,examples_plus_feature_skill \
  --packs runs/expanded/example_packs.30wb_20pb.qwen.v1.jsonl \
  --private-eval runs/expanded/example_private_eval.30wb_20pb.jsonl \
  --skills runs/expanded/skill_mvp.qwen.sample4_wb.jsonl \
  --out runs/expanded/judge_disagreements.qwen_vs_mimo.sample4_wb.heldout1.jsonl \
  --summary-out runs/expanded/judge_disagreements.qwen_vs_mimo.sample4_wb.heldout1.md \
  --max-output-chars 3000
```
