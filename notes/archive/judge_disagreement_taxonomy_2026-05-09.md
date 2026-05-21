# Judge Disagreement Taxonomy 2026-05-09

This note freezes the first third-review pass over
`runs/expanded/judge_disagreements.qwen_vs_mimo.sample4_wb.heldout1.jsonl`.
It should be treated as calibration evidence, not as a final human gold label.
The machine-readable annotations live in
`notes/judge_disagreement_taxonomy_2026-05-09.jsonl` and are validated against
packet hashes by `scripts/metrics/validate_disagreement_taxonomy.py`.

The packet contains 10 Qwen-vs-MIMO sign flips on identical candidate and
baseline outputs. All rows include output hashes/stats, task context, private
rubric context, and skill artifact context.

## Per-Cell Labels

| Pack | Mode | Qwen delta | MIMO delta | Candidate chars | Provisional label | More credible signal |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `writingbench_Academic_Engineering_Paper_Outline_en` | `examples_plus_one_shot_skill` | -1.2 | +1.8 | 14195 | candidate_error | Qwen negative |
| `writingbench_Academic_Engineering_Paper_Outline_en` | `one_shot_skill_from_examples` | -1.6 | +1.0 | 12173 | candidate_error | Qwen negative |
| `writingbench_Advertising_Marketing_Product_Description_en` | `one_shot_skill_from_examples` | -0.4 | +1.2 | 4185 | ambiguous | unresolved |
| `writingbench_Advertising_Marketing_Product_Description_en` | `ours_no_validation` | -0.2 | +0.2 | 4224 | small_delta_noise | unresolved |
| `writingbench_Advertising_Marketing_Travel_Guide_zh` | `examples_plus_feature_skill` | -0.2 | +0.6 | 4984 | small_delta_noise | unresolved |
| `writingbench_Advertising_Marketing_Travel_Guide_zh` | `examples_plus_one_shot_skill` | -0.8 | +0.2 | 4847 | ambiguous | weak MIMO positive possible |
| `writingbench_Education_Educational_Consulting_en` | `examples_plus_feature_skill` | -2.4 | +1.2 | 22960 | candidate_error | Qwen negative |
| `writingbench_Education_Educational_Consulting_en` | `examples_plus_one_shot_skill` | -1.0 | +1.2 | 22600 | candidate_error | Qwen negative |
| `writingbench_Education_Educational_Consulting_en` | `one_shot_skill_from_examples` | -1.6 | +1.2 | 21297 | candidate_error | Qwen negative |
| `writingbench_Education_Educational_Consulting_en` | `ours_no_validation` | -4.2 | +1.2 | 24218 | candidate_error | Qwen negative |

## Failure Modes Seen

- **Outline instead of final deliverable:** Academic Paper Outline candidates
  tend to produce plans/outlines rather than the requested completed long-form
  paper artifact. MIMO appears to reward the presence of academic structure,
  while Qwen penalizes missing task completion.
- **Source-context pollution:** Education Consulting candidates overuse terms
  from examples/materials and drift into infrastructure, traffic/security, or
  system-integration language that does not match the requested consulting
  deliverable. Qwen's negative deltas are more credible on these rows.
- **Surface-structure over-reward:** MIMO tends to reward headings, rubric
  keywords, and long organized responses even when the response misses the
  concrete user deliverable.
- **Small-delta noise:** Product Description and Travel Guide contain several
  sign flips with deltas near zero on at least one side. These should not drive
  method decisions without human inspection.

## Consequence For Strategy

Do not use the expanded WritingBench sample as positive or negative paper-level
evidence until this taxonomy is validated by a human or an independent judge
protocol. The safest current interpretation is:

1. Qwen is more credible than MIMO on task-completion failures in Academic and
   Education cells.
2. MIMO is useful as a calibration stress test, not as ground truth.
3. The next method loop should target candidate quality failures before
   expanding N.

## Validation

```bash
uv run python scripts/metrics/validate_disagreement_taxonomy.py \
  --packets runs/expanded/judge_disagreements.qwen_vs_mimo.sample4_wb.heldout1.jsonl \
  --taxonomy notes/judge_disagreement_taxonomy_2026-05-09.jsonl \
  --expect-status ok
```
