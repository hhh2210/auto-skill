# Completion Audit 2026-05-09

This audit maps the active research goal to concrete local evidence. The goal is
not complete. The repo is ready as an engineering prototype, cleaned-data
workspace, and MVP/surrogate eval handoff, but it is not ready for a full
benchmark or paper-level claim.

## Objective Breakdown

| Requirement | Current evidence | Status |
| --- | --- | --- |
| Construct and clean WritingBench / PresentBench example data | Checked-in MVP packs: `artifacts/packs/example_packs.v1.jsonl`, 8 packs, 24 generated train examples. Expanded local pass: `runs/expanded/example_packs.30wb_20pb.qwen.v1.jsonl`, 50 packs, 150 frozen train examples, 100 heldout tasks. | Done for MVP and local expanded workspace |
| Use Qwen3.5-Plus and MIMO APIs | Qwen generated the MVP/expanded canonical examples and solver outputs. MIMO was used for WritingBench train-example audit, WritingBench judge-swap, PresentBench audit, and a frozen local subset: `runs/expanded/example_packs.30wb_20pb.mimo.sample.v1.jsonl` has 15 packs and 45 frozen train examples with benchmark-flow `status=ok`. | Done for Qwen canonical data and MIMO audited subset; not a full 50-pack MIMO clean |
| Verify benchmark cleaning follows `notes/benchmark_flow.md` | `uv run python scripts/data/audit_benchmark_flow.py ...` returns `status=ok`, no errors/warnings for both checked-in MVP artifacts and expanded artifacts. | Done for data-flow/leakage audit |
| Auto-skill MVP evaluated on WritingBench and PresentBench | WritingBench Qwen judge: 40/40 success across 4 packs x 2 heldout x 5 modes. PresentBench surrogate: 40/40 success across 4 packs x 2 heldout x 5 modes, plus 32/32 success for examples-plus-skill, slide-constrained, and layout-plan surrogate ablations. | Done as MVP/surrogate eval |
| Mechanism ablation for examples plus skill | WritingBench Qwen judge: `runs/writingbench_official_eval.qwen.examples_plus_skill.heldout2.jsonl`, 16/16 success. WritingBench MIMO judge-swap: `runs/writingbench_official_eval.mimo_judge.examples_plus_skill.heldout2.jsonl`, 16/16 success after increasing `--judge-max-tokens` to 8192. | Done for first 4-pack smoke ablation |
| Expanded sample heldout eval | `runs/expanded/writingbench_official_eval.qwen.sample4_wb.heldout1.jsonl` and `runs/expanded/writingbench_official_eval.mimo_judge.sample4_wb.heldout1.jsonl` both have 24/24 success on four non-MVP WritingBench packs. See `notes/expanded_writingbench_sample_eval_2026-05-09.md`. | Done as expanded smoke, but results are judge-sensitive |
| Judge calibration packet | `scripts/metrics/export_judge_disagreements.py` exports the Qwen-vs-MIMO sign-flip cells to `runs/expanded/judge_disagreements.qwen_vs_mimo.sample4_wb.heldout1.jsonl` and `.md` for manual/third-judge review. Current packets include same-output checks, output hashes/stats, task/private rubric context, and skill artifact context. | Done for current expanded sample |
| Judge disagreement taxonomy | `notes/judge_disagreement_taxonomy_2026-05-09.jsonl` labels all 10 sign-flip packets. `scripts/metrics/validate_disagreement_taxonomy.py` verifies one label per packet and checks candidate/baseline hashes. | Done as provisional calibration evidence, not human gold |
| Candidate-quality guardrail probe | `notes/deliverable_guard_probe_2026-05-09.md` records a targeted Education Consulting probe after adding final-deliverable priority to the heldout prompt. `examples_plus_feature_skill` is near prompt-only, but skill-only remains strongly negative; deterministic n-gram contamination evidence is mixed. | Done as mixed/diagnostic evidence |
| Skill compression diagnostic | `notes/skill_compression_diagnostic_2026-05-09.md` shows skill-only modes cut heldout generation input to roughly 21-29% of few-shot input tokens, but negative transfer remains high; `examples_plus_feature_skill` is strongest on the MVP WritingBench Qwen run and competitive under MIMO, while expanded-sample judge sensitivity remains unresolved. | Done as strategy diagnostic, not proof of main claim |
| Compact feature-signature / operational-anchor ablation | `notes/feature_signature_ablation_2026-05-09.md` records a WritingBench smoke using `cross_example_report`-derived abstract signatures and sanitized feature-report operational anchors. Compact signatures stayed below prompt-only/few-shot; sanitized `task_first_operational_anchors` reached 6.0 on the hard Academic cell, then completed a four-pack Qwen + MIMO judge-swap check. A grounding probe found mean grounding score 4.5 and hallucination risk 6.75. | Done as mechanism smoke; current version is too hallucination-prone |
| PresentBench official evaluation | `check_presentbench_official_eval_ready.py` reports 16/16 `ready_for_official_judge`, but `runs/presentbench_official_scores.jsonl` has 16/16 `missing_score_artifact`. | Blocked: upstream `judge_all.py` score YAMLs not produced |
| Avoid model monoculture for research claims | `runs/mvp_metrics.heldout2.current.summary.json` sees Qwen and MIMO in eval model inventory. However readiness still warns that canonical readiness artifacts are Qwen-only and some old rows lack top-level model identity. | Partial |
| Use skeptical/background review | Background-agent review was performed in the Codex thread and identified PresentBench official scoring, heldout coverage, model monoculture, and method evidence as blockers. The transcript is not archived as a repo artifact, so this row is process evidence rather than file-backed experiment evidence. | Done for this iteration, but repeat after official scoring / ablations |
| Be factually confident in strategy | Current evidence shows auto-skill does not beat prompt-only/few-shot/one-shot reliably. | Not achieved |

## Current Gates

Green:

```bash
uv run python -m unittest discover -s tests
uv run ruff check .
diff -q AGENTS.md CLAUDE.md
uv run python scripts/data/audit_benchmark_flow.py
uv run python scripts/data/audit_benchmark_flow.py --splits runs/expanded/fewshot_splits.30wb_20pb.jsonl --packs runs/expanded/example_packs.30wb_20pb.qwen.v1.jsonl --private-eval runs/expanded/example_private_eval.30wb_20pb.jsonl --jobs runs/expanded/example_generation_jobs.30wb_20pb.jsonl --generated-outputs runs/expanded/generated_desired_outputs.30wb_20pb.qwen.jsonl
uv run python scripts/data/audit_benchmark_flow.py --splits runs/expanded/fewshot_splits.30wb_20pb.jsonl --packs runs/expanded/example_packs.30wb_20pb.mimo.sample.v1.jsonl --private-eval runs/expanded/example_private_eval.30wb_20pb.mimo.sample.jsonl --jobs runs/expanded/example_generation_jobs.30wb_20pb.mimo.sample.jsonl --generated-outputs runs/expanded/generated_desired_outputs.30wb_20pb.mimo.sample.latest_success.jsonl
uv run python scripts/ops/report_expanded_cleaning_status.py --expect-status ready
uv run python scripts/ops/report_expanded_cleaning_status.py --require-mimo-subset --expect-status ready
uv run python scripts/ops/report_experiment_readiness.py --profile mvp --expect-status ready
uv run python scripts/metrics/validate_disagreement_taxonomy.py --packets runs/expanded/judge_disagreements.qwen_vs_mimo.sample4_wb.heldout1.jsonl --taxonomy notes/judge_disagreement_taxonomy_2026-05-09.jsonl --expect-status ok
```

Fail-closed by design:

```bash
uv run python scripts/ops/report_experiment_readiness.py --profile full --expect-status not_ready
```

The full-profile blockers are exactly the 8 PresentBench tasks x 2 official
modes whose upstream score YAMLs are missing.

Canonical current readiness reports are:

- `runs/readiness.mvp.current.json`
- `runs/readiness.full.current.json`

Do not use older ad hoc files such as `runs/experiment_readiness.current.json`
as the source of truth; some were produced before the profile/path contract was
updated.

## Score Evidence

### WritingBench, Qwen Judge

Artifact: `runs/writingbench_official_eval.qwen.five_modes.no_thinking_auto_skill.jsonl`

Coverage: 40/40 success.

| Mode | Mean score | Delta vs prompt_only | Wins / losses / ties |
| --- | ---: | ---: | --- |
| prompt_only | 7.10 | n/a | n/a |
| few_shot_examples_only | 6.60 | -0.50 | 3 / 5 / 0 |
| one_shot_skill_from_examples | 6.825 | -0.275 | 5 / 3 / 0 |
| ours_no_validation | 6.275 | -0.825 | 1 / 6 / 1 |
| auto_skill | 6.50 | -0.60 | 3 / 4 / 1 |

Interpretation: current auto-skill does not support the headline claim on
WritingBench. Negative transfer is common.

### WritingBench, MIMO Judge-Swap

Artifact: `runs/writingbench_official_eval.mimo_judge.five_modes.heldout2.jsonl`

Coverage: 40/40 success. The earlier MIMO `content_filter` refusal on
`writingbench_Finance_Business_Tender_Document_zh::heldout::0`,
`few_shot_examples_only` was retried successfully after classifying provider
refusals separately from parse errors.

| Mode | Mean score | Delta vs prompt_only | Wins / losses / ties |
| --- | ---: | ---: | --- |
| prompt_only | 6.925 | n/a | n/a |
| few_shot_examples_only | 6.75 | -0.175 | 3 / 3 / 2 |
| one_shot_skill_from_examples | 6.425 | -0.50 | 2 / 4 / 2 |
| ours_no_validation | 6.15 | -0.775 | 1 / 6 / 1 |
| auto_skill | 6.30 | -0.625 | 3 / 5 / 0 |

Interpretation: switching the judge to MIMO does not rescue the auto-skill
claim. It also shows MIMO judge runs need retry support for provider refusals
and transient API errors.

### WritingBench, Examples Plus Skill Ablation

Artifact: `runs/writingbench_official_eval.qwen.examples_plus_skill.heldout2.jsonl`
and `runs/writingbench_official_eval.mimo_judge.examples_plus_skill.heldout2.jsonl`

Coverage: 16/16 success for each judge across 4 packs x 2 heldout x 2
ablation modes. The paired deltas below are computed by joining the ablation
artifact with the corresponding five-mode baseline artifact on `(pack_id,
task_id)`. The canonical `runs/mvp_metrics.heldout2.current.summary.json`
records these joins under `cross_eval_score_summaries`.

| Judge | Mode | Mean score | Delta vs prompt_only | Wins / losses / ties |
| --- | --- | ---: | ---: | --- |
| Qwen | examples_plus_one_shot_skill | 7.525 | +0.425 | 6 / 2 / 0 |
| Qwen | examples_plus_feature_skill | 7.775 | +0.675 | 7 / 1 / 0 |
| MIMO | examples_plus_one_shot_skill | 7.45 | +0.525 | 6 / 2 / 0 |
| MIMO | examples_plus_feature_skill | 7.35 | +0.425 | 5 / 2 / 1 |

Interpretation: on this smoke slice, skills are more useful as an augmentation
to user examples than as a replacement for examples. This weakens the current
“skill-only reusable package beats examples” claim, but gives a concrete next
mechanism direction: use induced skills to organize or constrain examples rather
than compressing examples away.

### WritingBench, Compact Feature Signatures Smoke

Artifact:
`runs/expanded/writingbench_official_eval.qwen.feature_signatures_smoke.jsonl`

Coverage: 2/2 success for one expanded WritingBench pack x one heldout x two
new modes. This is a mechanism smoke, not a benchmark result.

Pack: `writingbench_Academic_Engineering_Paper_Outline_en`

Task: `writingbench_Academic_Engineering_Paper_Outline_en::heldout::0`

| Mode | Qwen WritingBench score |
| --- | ---: |
| prompt_only | 4.0 |
| few_shot_examples_only | 4.4 |
| one_shot_skill_from_examples | 2.4 |
| ours_no_validation | 2.8 |
| examples_plus_one_shot_skill | 2.8 |
| examples_plus_feature_skill | 2.8 |
| feature_signatures_only | 2.6 |
| examples_plus_feature_signatures | 3.0 |
| task_first_feature_signatures | 3.0 |
| task_first_operational_anchors | 6.0 |

Interpretation: compact signatures derived from `cross_example_report` do not
recover the task-completion information lost by skill compression on this cell.
They are slightly better than the corresponding full-feature skill modes on this
single task, but still well below few-shot examples. The task-first variant
places heldout task/materials before signatures and removes raw examples, but
still has outline/template-like weaknesses and under-delivers technical depth
and task-specific evidence. In contrast, sanitized
`task_first_operational_anchors` recovers a much longer output (38,551 chars)
and improves technical coverage, evidence density, and structural completeness
on this one Qwen-judged cell.
However, the output may also be rewarded for hallucinated empirical detail and
generic references, so this is a candidate mechanism rather than a clean win.

### WritingBench, Operational Anchors Four-Pack Check

Artifacts:

- `runs/expanded/writingbench_official_eval.qwen.operational_anchors.sample4_wb.heldout1.jsonl`
- `runs/expanded/writingbench_official_eval.mimo_judge.operational_anchors.sample4_wb.heldout1.jsonl`

Coverage: 4/4 Qwen success and 4/4 MIMO judge-swap success on the existing
calibrated expanded WritingBench sample.

| Judge | Mode | Mean score |
| --- | --- | ---: |
| Qwen | prompt_only | 6.05 |
| Qwen | few_shot_examples_only | 6.20 |
| Qwen | one_shot_skill_from_examples | 5.15 |
| Qwen | ours_no_validation | 4.70 |
| Qwen | examples_plus_feature_skill | 5.25 |
| Qwen | task_first_operational_anchors | 6.05 |
| MIMO | prompt_only | 6.20 |
| MIMO | few_shot_examples_only | 7.25 |
| MIMO | one_shot_skill_from_examples | 7.05 |
| MIMO | ours_no_validation | 6.50 |
| MIMO | examples_plus_feature_skill | 7.20 |
| MIMO | task_first_operational_anchors | 7.05 |

Interpretation: operational anchors recover from the worst compression failures
and are competitive with prompt-only / one-shot on this small slice, but they do
not beat few-shot examples under either judge. The method direction is still
useful because it points to concrete task-instantiation information lost by
skill compression, but it is not yet a paper-level performance claim.

### WritingBench, Operational Anchors Grounding Probe

Artifacts:

- `runs/expanded/grounding_eval.mimo_judge.operational_anchors.sample4_wb.heldout1.jsonl`
- `runs/expanded/grounding_eval.mimo_judge.operational_anchors.sample4_wb.heldout1.summary.json`

Coverage: 4/4 success. The judge sees only heldout task input/materials as
factual evidence; user examples and induced skills are not treated as evidence
for heldout factual claims.

| Mode | Mean grounding score | Mean hallucination risk | Mean unsupported claim count |
| --- | ---: | ---: | ---: |
| task_first_operational_anchors | 4.5 | 6.75 | 8.0 |

Interpretation: the score improvements from operational anchors are not clean.
The probe flags unsupported specifics such as fabricated case-study details,
travel costs/routes, product compatibility claims, and academic citations. This
turns grounding/authenticity into a method blocker rather than a small caveat.
Because this is an LLM-judge diagnostic, exact scores may drift across reruns;
the stable signal is the repeated unsupported-detail findings.

### PresentBench Surrogate

Artifacts:

- `runs/presentbench_surrogate_eval.qwen.mvp.jsonl`
- `runs/presentbench_surrogate_eval.qwen.auto_skill.jsonl`
- `runs/presentbench_surrogate_eval.qwen.examples_plus_skill.heldout2.jsonl`
- `runs/presentbench_surrogate_eval.qwen.slide_constrained_examples_plus_feature.heldout2.jsonl`
- `runs/presentbench_surrogate_eval.qwen.layout_plan_examples_plus_feature.heldout2.jsonl`

Coverage: 72/72 success across the combined files. This is a text/rubric
surrogate, not the official visual/PPT evaluator.

| Mode | Mean score | Delta vs prompt_only |
| --- | ---: | ---: |
| prompt_only | 7.625 | n/a |
| few_shot_examples_only | 8.25 | +0.625 |
| one_shot_skill_from_examples | 8.85 | +1.225 |
| ours_no_validation | 8.625 | +1.00 |
| auto_skill | 8.375 | +0.75 |
| examples_plus_one_shot_skill | 6.625 | -1.00 |
| examples_plus_feature_skill | 7.00 | -0.625 |
| slide_constrained_examples_plus_feature_skill | 6.625 | -1.00 |
| layout_plan_examples_plus_feature_skill | 6.75 | -0.875 |

Interpretation: PresentBench surrogate looks more favorable than WritingBench,
but it cannot replace official PresentBench scoring. The examples-plus-skill
mechanism that is positive on WritingBench is negative on this PresentBench
surrogate slice, so it is not yet a cross-domain method claim. The failure
diagnostic in `notes/presentbench_examples_plus_failure_diagnostic.md` points to
cross-task slide layout interference from raw examples plus skill. A light
prompt-only slide-constraint guardrail did not fix the issue; an explicit
layout-plan stage also remained negative when raw examples stayed in the final
prompt.

## Data-Cleaning Evidence

Expanded local pass:

- 30 WritingBench packs.
- 20 PresentBench packs.
- 150/150 latest generated desired outputs are success.
- 100 heldout tasks.
- Expanded status gate is `ready`; strict local handoff with
  `--require-mimo-subset` is also `ready`.
- Required audits:
  - MIMO WritingBench sample: 6/6 success.
  - Qwen PresentBench material-aware sample: 2/2 success.
- Optional MIMO PresentBench audit: 2/2 success after rerun with
  `--judge-max-tokens 8192`; mean score 8.4.
- MIMO frozen subset: 15 packs, 45 frozen train examples, 30 heldout tasks,
  45 generation jobs, 12 WritingBench packs, 3 PresentBench packs, and
  benchmark-flow `status=ok`.

Interpretation: expanded cleaning is usable for handoff and follow-up
experiments, but it is not a released dataset snapshot and does not by itself
prove the auto-skill method.

## Remaining Blockers

1. PresentBench official scoring is missing. Need `GENAI_API_KEY` /
   `GENAI_BASE_URL` or equivalent upstream `judge_all.py` configuration, then
   run official scoring for `prompt_only` and `auto_skill`.
2. Current skill-only auto-skill design is not winning on WritingBench and is
   only surrogate-positive on PresentBench. The new examples-plus-skill ablation
   is positive on WritingBench, so treat feature-driven extraction as a
   promising augmentation mechanism rather than a proven standalone replacement
   for examples. A newer four-pack expanded WritingBench sample shows strong
   Qwen-vs-MIMO judge sign flips, so evaluator calibration is now a first-order
   blocker before increasing sample size. The first provisional taxonomy is in
   `notes/judge_disagreement_taxonomy_2026-05-09.md` and machine-validated
   against packet hashes. A targeted final-deliverable prompt guardrail improved
   `examples_plus_feature_skill` on the worst Education Consulting probe but
   did not fix skill-only negative transfer. A deterministic train-only n-gram
   diagnostic did not show higher contamination for `examples_plus_feature_skill`
   than for prompt-only on that probe, so the next method loop should address
   skill-only compression loss and task-specific constraint extraction before
   blaming raw example copying. The current compression diagnostic is in
   `notes/skill_compression_diagnostic_2026-05-09.md`. A compact
   feature-signature smoke in `notes/feature_signature_ablation_2026-05-09.md`
   also stayed below prompt-only/few-shot even after a task-first prompt-order
   variant, so shorter context and simple task-first ordering are not the
   missing mechanism. The later `task_first_operational_anchors` check is
   healthier than prior compression variants on the calibrated four-pack sample
   but still does not beat few-shot examples. The grounding probe confirms that
   it is partially rewarded for ungrounded detail.
3. MIMO judge-swap is now complete on the current 4-pack WritingBench smoke
   slice. A local MIMO cleaned subset is frozen and flow-audited, but it covers
   only 15 packs rather than the full 50-pack expanded split. Do not call MIMO
   the canonical cleaner until a full MIMO pass is frozen and audited.
4. Some old skill/eval rows lack top-level `solver_model` / `judge_model`.
   New runners record model identity, but old artifacts should not be used as
   paper-facing evidence without this caveat.

## Next Concrete Steps

1. Run upstream PresentBench official judge once `GENAI_*` is configured.
2. Calibrate evaluators before expanding WritingBench N. The expanded four-pack
   sample has Qwen/MIMO sign flips on the same candidate outputs, so inspect the
   sign-flip cells manually or with a third independent judge before treating
   either judge as the primary decision signal.
3. For PresentBench, test constrained slide composition that removes raw
   examples from the final prompt and passes only abstract example signatures
   plus current-task hard constraints. Do not claim cross-domain transfer from
   the current examples-plus-skill ablation.
4. After evaluator calibration, decide whether to expand the WritingBench
   examples-plus-skill ablation or redesign the skill induction mechanism.
5. Redesign operational anchors with a stricter evidence policy before
   expanding N. The grounding probe confirms the current mode often produces
   plausible unsupported specifics, so the next version needs to distinguish
   "detail slots to fill" from "facts allowed by current evidence".
