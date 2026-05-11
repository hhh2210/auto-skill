# Example-Likeness Metrics Plan

## Current main-branch status

`main` already has partial support for example-likeness diagnostics:

- `scripts/metrics/run_pattern_similarity_eval.py`
  - Scores whether a heldout agent output follows reusable patterns visible in train examples.
  - Outputs `pattern_similarity_score`, `structural_similarity_score`, `style_similarity_score`, `constraint_transfer_score`, and unsupported-pattern diagnostics.
  - Supports a blind mode and a debug-only `--skill-aware` mode.
- `scripts/metrics/run_self_consistency_metric.py`
  - Generates abstract example signatures from `SKILL.md`, then judges those signatures against real train examples.
  - Outputs `stable_feature_recall`, `constraint_recall`, `structure_recall`, `style_signature`, `leakage_penalty`, `unsupported_specificity_penalty`, and `overall_self_consistency`.
  - This is close to reverse recovery, but it recovers abstract signatures, not full fake examples.
- `src/auto_skill/metrics.py::skill_artifact_summary`
  - Provides cheap structural stats for skill artifacts, such as length and rule counts.
  - It is not yet a semantic LLM judge for whether a skill is clear, usable, and skill-like.

So the answer is: we have an initial "output looks like examples" metric and a weaker self-consistency diagnostic, but we do not yet have a clean three-metric benchmark layer matching the research framing.

## Proposed three-metric layer

### 1. Output Example-Likeness

Question: given train examples, heldout task, and agent output, does the output look like the examples in reusable ways?

Inputs:

- train examples: task input, materials, desired output
- heldout task input and materials
- candidate agent output
- optional debug-only skill text

Judge model:

- MIMO by default via `--judge-config-prefix MIMO`
- blind mode is the reportable metric
- skill-aware mode is debug-only

Score schema:

```json
{
  "overall_example_likeness": 1-10,
  "structure_similarity": 1-10,
  "style_similarity": 1-10,
  "constraint_transfer": 1-10,
  "content_strategy_similarity": 1-10,
  "unsupported_pattern_risk": "low|medium|high",
  "copied_example_specifics": ["..."],
  "missed_reusable_patterns": ["..."],
  "rationale": "..."
}
```

Main distinction from benchmark score:

- Benchmark score asks: did it complete this heldout task?
- Example-likeness asks: did it preserve the example family pattern without copying one-off details?

Implementation note:

- Extend or rename the existing `run_pattern_similarity_eval.py` surface rather than inventing a parallel script.
- Add `judge_model` separation in standard reports so MIMO similarity is visible next to Qwen task score.

### 2. Skill-Likeness / Skill Quality

Question: is the induced artifact actually a good skill, not just a verbose summary of examples?

Inputs:

- generated `skill_md`
- optionally train examples for grounding check
- no heldout task, no private rubric

Judge model:

- MIMO by default

Score schema:

```json
{
  "overall_skill_quality": 1-10,
  "clarity": 1-10,
  "actionability": 1-10,
  "scope_control": 1-10,
  "example_grounding": 1-10,
  "negative_guidance_quality": 1-10,
  "overfit_risk": 1-10,
  "missing_operational_steps": ["..."],
  "unclear_rules": ["..."],
  "over_specific_rules": ["..."],
  "rationale": "..."
}
```

What it catches:

- skill is just a report, not an executable procedure
- rules are vague or non-operational
- overfits to entity names, dates, numbers, or one train example
- lacks do-not-generalize / outlier guidance
- lacks current-task override and hard-constraint handling

Implementation note:

- New script: `scripts/metrics/run_skill_quality_eval.py`
- New library module or extension: `src/auto_skill/skill_quality.py`
- Keep `skill_artifact_summary` as cheap structural metadata, not the main semantic score.

### 3. Reverse-Recovery Self-Consistency

Question: if we only have the induced skill, can a model recover fake examples that match the true example family?

Inputs:

- generated `skill_md`
- true train examples for judge only
- no private rubric, no heldout feedback

Two-stage protocol:

1. Generator produces `k` fake examples from `skill_md` only.
2. MIMO judge compares fake examples to true examples and scores the gap.

Fake example schema:

```json
{
  "fake_examples": [
    {
      "task_input": "...",
      "expected_output": "...",
      "intended_reusable_patterns": ["..."]
    }
  ]
}
```

Judge score schema:

```json
{
  "overall_recovery": 1-10,
  "task_family_match": 1-10,
  "output_structure_match": 1-10,
  "style_match": 1-10,
  "constraint_match": 1-10,
  "diversity_without_drift": 1-10,
  "literal_leakage_penalty": 1-10,
  "true_patterns_recovered": ["..."],
  "true_patterns_missing": ["..."],
  "fake_only_artifacts": ["..."],
  "rationale": "..."
}
```

Difference from the current `run_self_consistency_metric.py`:

- current: skill -> abstract signatures -> compare to true examples
- proposed: skill -> full fake task/output examples -> compare to true examples

This is a stronger and more intuitive self-consistency metric, but it costs more tokens and needs leakage checks.

Implementation note:

- Either add `--recovery-mode signatures|fake_examples` to `run_self_consistency_metric.py`, or create `run_skill_recovery_eval.py`.
- Prefer a new script if we want clean row schemas and paper-facing names.

## Recommended reporting table

| Metric | Unit | Judge | Main use | Reportable? |
|---|---|---|---|---|
| Benchmark task score | heldout output | WritingBench/PresentBench official or MIMO surrogate | final task success | yes |
| Output example-likeness | heldout output | MIMO | whether output follows train examples | yes |
| Skill quality | skill artifact | MIMO | whether artifact is a clear usable skill | yes |
| Reverse-recovery self-consistency | skill artifact | MIMO | whether skill can regenerate the example family | diagnostic first, reportable after validation |
| Grounding | heldout output | MIMO | hallucination/material fidelity | diagnostic |

## Suggested next implementation slice

1. Standardize MIMO as the default judge for diagnostics via `--judge-config-prefix MIMO`.
2. Reuse `run_pattern_similarity_eval.py` for Metric 1 and rename report labels in summaries.
3. Add `run_skill_quality_eval.py` for Metric 2.
4. Extend self-consistency with fake-example recovery for Metric 3.
5. Add one summarizer that joins task score, example-likeness, skill quality, and recovery by `(pack_id, mode)`.

## 2026-05-10 update: per-criterion diagnosis is the cheapest first slice

Before any new LLM-judge metric, the existing WritingBench eval rows already carry per-criterion data (`scores: dict[criterion_name, [{score, reason}]]`) that the collapsed `overall_score` throws away. The new module `auto_skill.criterion_diagnosis` and the CLI `scripts/metrics/summarize_per_criterion_delta.py` post-process those rows into per-criterion deltas and a coarse keyword bucket roll-up. No additional judge calls.

Run on `runs/expanded/writingbench_eval.auto_skill_minimal.30wb.jsonl` for `ours_no_validation` vs `few_shot_examples_only` (58 paired tasks, 32 of which are losing tasks, mean overall Δ -0.148; 0 tasks skipped due to mismatched criterion sets):

| Bucket | n criteria | mean Δ | W / L (per criterion) | **worst-pick on losing tasks** | worst-pick all (sanity) |
|---|---:|---:|---:|---:|---:|
| **depth_specificity_practical** | 70 | -0.31 | 13 / 27 | **13 (41%)** | 22 |
| required_sections_completeness | 23 | -0.26 | 4 / 6 | 5 (16%) | 7 |
| other | 65 | +0.03 | 11 / 14 | 5 | 12 |
| accuracy_professionalism | 24 | +0.17 | 5 / 6 | 3 | 3 |
| format_structure | 39 | -0.10 | 6 / 9 | 2 | 5 |
| evidence_citation | 8 | -0.50 | 0 / 1 | 1 | 1 |
| length_word_count | 7 | -0.43 | 1 / 2 | 0 | 0 |

(Percent in the worst-pick-on-losing column is over the 32 losing tasks. Buckets are heuristic substring-keyword groupings, not a taxonomy; cite `worst_pick_count_on_losing_tasks` for systemic-failure claims, not `worst_pick_count_all`.)

Headline finding: the systemic gap is `depth_specificity_practical` (content depth, case integration, practical specificity), not `evidence_citation` or `length_word_count`. This is the axis that abstract skill text replaces worst and raw examples preserve best, and it stays #1 under both single-label and multi-label bucket attribution. Full report: `runs/expanded/per_criterion_delta.ours_vs_example_only.30wb.headline.md`.

Implications for the three planned metrics above:

- Output example-likeness (Metric 1) is deprioritized: by construction it will favor `few_shot_examples_only` because raw examples sit in the prompt; the actionable per-axis signal already exists in WritingBench's own rubric.
- Skill quality (Metric 2) should not be a generic clarity/actionability judge — it must include a `depth_specificity_preservation` axis that punishes induced skills which strip example micro-patterns. Otherwise verbose `one_shot_skill` outputs will score well even though they scored worst on task.
- Reverse-recovery (Metric 3) is still useful for paper-facing self-consistency but is downstream of fixing the depth-preservation method gap; fake examples will be too abstract to reach training-example fidelity until the induction prompt keeps real micro-patterns.

Concrete next moves:

1. Extend the auto-skill induction prompt with a fixed-budget example-micropattern appendix (high-saliency snippets, signature phrases, depth markers); re-run on the same 30WB pack and re-summarize per-criterion deltas to see if `depth_specificity_practical` closes.
2. Re-run the same 240 cells with split MIMO judge (`--judge-config-prefix MIMO`) before declaring `example_only` strictly better; same-model qwen-judges-qwen is a known monoculture risk on depth/specificity.
3. After (1) or (2) ships, run anchor pairwise (anchor=`few_shot_examples_only`, MIMO judge, 60 cells × 2 swap per candidate) to confirm whether the proposed micropattern fix actually closes the gap.
