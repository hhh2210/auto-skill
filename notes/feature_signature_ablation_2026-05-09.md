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

Follow-up code added one more mode:

- `task_first_feature_signatures`
  - omits raw examples;
  - places the heldout task input and current materials before feature
    signatures;
  - treats signatures as secondary structure/tone/self-check guidance.

A later follow-up added:

- `task_first_operational_anchors`
  - omits raw examples;
  - places the heldout task input and current materials first;
  - uses sanitized user-example `feature_reports` keys to preserve
    operational/detail slots such as variables, methods, standards, datasets,
    case-study details, evidence density, and placeholder policy;
  - does not emit raw feature values from training examples.

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

## Task-First Follow-Up

Command:

```bash
uv run python scripts/eval/run_writingbench_official_eval.py \
  --packs runs/expanded/example_packs.30wb_20pb.qwen.v1.jsonl \
  --skills runs/expanded/skill_mvp.qwen.sample4_wb.jsonl \
  --private-eval runs/expanded/example_private_eval.30wb_20pb.jsonl \
  --writingbench-root ../WritingBench \
  --pack-id writingbench_Academic_Engineering_Paper_Outline_en \
  --modes task_first_feature_signatures \
  --limit-heldout 1 \
  --out runs/expanded/writingbench_official_eval.qwen.task_first_feature_signatures_smoke.jsonl \
  --summary-out runs/expanded/writingbench_official_eval.qwen.task_first_feature_signatures_smoke.summary.json \
  --num-threads 1 \
  --timeout-seconds 600 \
  --max-retries 3 \
  --parse-max-attempts 3 \
  --max-tokens 8192 \
  --judge-max-tokens 1024 \
  --allow-partial
```

Result:

| mode | Qwen WritingBench score | output chars |
|---|---:|---:|
| `task_first_feature_signatures` | 3.0 | 16,267 |

Criterion scores:

| criterion | score |
|---|---:|
| Academic Relevance and Rigor | 3 |
| Multi-layered Protection System Coverage | 2 |
| Edge Computing Security Analysis | 2 |
| Integration of Personal Experience and Industry Standards | 3 |
| Structural Coherence and Academic Format | 5 |

The task-first variant produces a longer answer than `feature_signatures_only`,
but it still has outline/template-like weaknesses and under-delivers technical
depth and task-specific evidence. The strongest few-shot row for the same cell
is much longer (41,209 chars) and scores 8 on `Integration of Personal
Experience and Industry Standards`; the compact signatures do not preserve
enough operational/detail anchors to reproduce that behavior.

## Operational Anchors Follow-Up

Command:

```bash
uv run python scripts/eval/run_writingbench_official_eval.py \
  --packs runs/expanded/example_packs.30wb_20pb.qwen.v1.jsonl \
  --skills runs/expanded/skill_mvp.qwen.sample4_wb.jsonl \
  --private-eval runs/expanded/example_private_eval.30wb_20pb.jsonl \
  --writingbench-root ../WritingBench \
  --pack-id writingbench_Academic_Engineering_Paper_Outline_en \
  --modes task_first_operational_anchors \
  --limit-heldout 1 \
  --out runs/expanded/writingbench_official_eval.qwen.task_first_operational_anchors_sanitized_smoke.jsonl \
  --summary-out runs/expanded/writingbench_official_eval.qwen.task_first_operational_anchors_sanitized_smoke.summary.json \
  --num-threads 1 \
  --timeout-seconds 600 \
  --max-retries 3 \
  --parse-max-attempts 3 \
  --max-tokens 8192 \
  --judge-max-tokens 1024 \
  --allow-partial
```

Result:

| mode | Qwen WritingBench score | output chars |
|---|---:|---:|
| `task_first_operational_anchors` | 6.0 | 38,551 |

Criterion scores:

| criterion | score |
|---|---:|
| Academic Relevance and Rigor | 5 |
| Multi-layered Protection System Coverage | 6 |
| Edge Computing Security Analysis | 5 |
| Integration of Personal Experience and Industry Standards | 7 |
| Structural Coherence and Academic Format | 7 |

This is a noisy but useful mechanism signal on this difficult cell:

- `task_first_operational_anchors` beats `prompt_only` (4.0),
  `few_shot_examples_only` (4.4), and all previous skill/signature modes on the
  same Qwen-judged cell.
- The score gain coincides with restored output length and more technical
  coverage.
- Caveat: the judged output may also be rewarded for hallucinated empirical
  detail and generic references. This smoke identifies a direction to test, not
  a clean task-completion win.
- An earlier unsanitized operational-anchor prompt scored 6.4, but that run is
  not used as current evidence because the prompt could emit raw
  training-example feature values.
- This is still a one-cell Qwen-judge smoke. It must not be reported as a
  benchmark result until rerun on the calibrated multi-pack sample and checked
  with an independent judge.

## Four-Pack Operational Anchors Check

Command:

```bash
uv run python scripts/eval/run_writingbench_official_eval.py \
  --packs runs/expanded/example_packs.30wb_20pb.qwen.v1.jsonl \
  --skills runs/expanded/skill_mvp.qwen.sample4_wb.jsonl \
  --private-eval runs/expanded/example_private_eval.30wb_20pb.jsonl \
  --writingbench-root ../WritingBench \
  --pack-id writingbench_Academic_Engineering_Paper_Outline_en \
  --pack-id writingbench_Advertising_Marketing_Product_Description_en \
  --pack-id writingbench_Advertising_Marketing_Travel_Guide_zh \
  --pack-id writingbench_Education_Educational_Consulting_en \
  --modes task_first_operational_anchors \
  --limit-heldout 1 \
  --out runs/expanded/writingbench_official_eval.qwen.operational_anchors.sample4_wb.heldout1.jsonl \
  --summary-out runs/expanded/writingbench_official_eval.qwen.operational_anchors.sample4_wb.heldout1.summary.json \
  --num-threads 4 \
  --timeout-seconds 600 \
  --max-retries 3 \
  --parse-max-attempts 3 \
  --max-tokens 8192 \
  --judge-max-tokens 1024 \
  --allow-partial
```

MIMO judge-swap used the same candidate outputs:

```bash
uv run python scripts/eval/run_writingbench_official_eval.py \
  --packs runs/expanded/example_packs.30wb_20pb.qwen.v1.jsonl \
  --skills runs/expanded/skill_mvp.qwen.sample4_wb.jsonl \
  --private-eval runs/expanded/example_private_eval.30wb_20pb.jsonl \
  --writingbench-root ../WritingBench \
  --reuse-candidates-from runs/expanded/writingbench_official_eval.qwen.operational_anchors.sample4_wb.heldout1.jsonl \
  --judge-config-prefix MIMO \
  --pack-id writingbench_Academic_Engineering_Paper_Outline_en \
  --pack-id writingbench_Advertising_Marketing_Product_Description_en \
  --pack-id writingbench_Advertising_Marketing_Travel_Guide_zh \
  --pack-id writingbench_Education_Educational_Consulting_en \
  --modes task_first_operational_anchors \
  --limit-heldout 1 \
  --out runs/expanded/writingbench_official_eval.mimo_judge.operational_anchors.sample4_wb.heldout1.jsonl \
  --summary-out runs/expanded/writingbench_official_eval.mimo_judge.operational_anchors.sample4_wb.heldout1.summary.json \
  --num-threads 4 \
  --timeout-seconds 600 \
  --max-retries 3 \
  --parse-max-attempts 3 \
  --judge-max-tokens 8192 \
  --no-judge-enable-thinking \
  --allow-partial
```

Coverage: 4/4 Qwen success and 4/4 MIMO judge-swap success.

| Judge | Mode | Mean score | Notes |
|---|---|---:|---|
| Qwen | `prompt_only` | 6.05 | Existing calibrated four-pack baseline |
| Qwen | `few_shot_examples_only` | 6.20 | Existing calibrated four-pack baseline |
| Qwen | `one_shot_skill_from_examples` | 5.15 | Existing calibrated four-pack baseline |
| Qwen | `ours_no_validation` | 4.70 | Existing calibrated four-pack baseline |
| Qwen | `examples_plus_feature_skill` | 5.25 | Existing calibrated four-pack baseline |
| Qwen | `task_first_operational_anchors` | 6.05 | New run |
| MIMO | `prompt_only` | 6.20 | Existing calibrated judge-swap baseline |
| MIMO | `few_shot_examples_only` | 7.25 | Existing calibrated judge-swap baseline |
| MIMO | `one_shot_skill_from_examples` | 7.05 | Existing calibrated judge-swap baseline |
| MIMO | `ours_no_validation` | 6.50 | Existing calibrated judge-swap baseline |
| MIMO | `examples_plus_feature_skill` | 7.20 | Existing calibrated judge-swap baseline |
| MIMO | `task_first_operational_anchors` | 7.05 | New judge-swap run |

Per-pack operational-anchor scores:

| Pack | Qwen | MIMO |
|---|---:|---:|
| `writingbench_Academic_Engineering_Paper_Outline_en` | 6.0 | 7.4 |
| `writingbench_Advertising_Marketing_Product_Description_en` | 5.0 | 4.4 |
| `writingbench_Advertising_Marketing_Travel_Guide_zh` | 7.2 | 8.0 |
| `writingbench_Education_Educational_Consulting_en` | 6.0 | 8.4 |

Interpretation:

- Operational anchors are much healthier than the previous skill-only and
  compact-signature variants on the hard Academic cell, but the four-pack mean
  does not beat `few_shot_examples_only`.
- MIMO gives a higher mean than Qwen, but still ranks few-shot and
  `examples_plus_feature_skill` above operational anchors on this slice.
- The result supports a narrower mechanism hypothesis: preserving
  task-instantiation anchors can reduce compression loss, but it is not yet a
  replacement for examples and still needs grounding checks for fabricated
  specifics.

## Grounding Probe

Because operational anchors explicitly encourage detail depth and task-specific
instantiation, they can raise WritingBench scores by producing plausible but
unsupported specifics. A follow-up grounding probe judges candidate outputs
against only the heldout task input and heldout materials. User examples and
induced skills are explicitly not factual evidence for heldout claims.

Command:

```bash
uv run python scripts/metrics/run_grounding_eval.py \
  --packs runs/expanded/example_packs.30wb_20pb.qwen.v1.jsonl \
  --eval runs/expanded/writingbench_official_eval.qwen.operational_anchors.sample4_wb.heldout1.jsonl \
  --judge-config-prefix MIMO \
  --modes task_first_operational_anchors \
  --out runs/expanded/grounding_eval.mimo_judge.operational_anchors.sample4_wb.heldout1.jsonl \
  --summary-out runs/expanded/grounding_eval.mimo_judge.operational_anchors.sample4_wb.heldout1.summary.json \
  --num-threads 4 \
  --max-tokens 4096 \
  --parse-max-attempts 3 \
  --max-evidence-chars 80000 \
  --max-candidate-chars 80000
```

Coverage: 4/4 success.

| Mode | Mean grounding score | Mean hallucination risk | Mean unsupported claim count |
|---|---:|---:|---:|
| `task_first_operational_anchors` | 4.5 | 6.75 | 8.0 |

Representative unsupported-claim findings:

| Pack | Grounding score | Risk | Example unsupported detail |
|---|---:|---:|---|
| Academic Engineering Paper Outline EN | 4 | 8 | TPM/TLS/RBAC details, six-month industrial-park test duration, and specific vulnerability findings are not in the heldout evidence. |
| Product Description EN | 6 | 5 | One-button operation, three-minute extraction, and broad capsule compatibility are not specified by the current evidence. |
| Travel Guide ZH | 4 | 8 | Distance, toll/fuel costs, parking prices, restaurant names, and detailed attraction sequence are not provided by the heldout task. |
| Education Consulting EN | 4 | 6 | The named student case study, parent-record workflow, agency interview questions, and other practical details are not in the heldout materials. |

Interpretation:

- This probe confirms the main risk: operational anchors can improve apparent
  task completeness by inventing concrete task facts. Because this is an
  LLM-judge diagnostic, exact scores can drift across reruns; the stable signal
  is the high hallucination risk and repeated unsupported-detail findings.
- The mechanism should not be expanded as-is. The next prompt/module design
  needs an evidence policy that separates "detail slots to fill" from "details
  allowed to fabricate".
- The grounding metric is diagnostic only. It is not an official benchmark
  score and should be used to reject or revise candidate methods before running
  larger N.

## Evidence-Policy Fix

A minimal prompt fix was added to `task_first_operational_anchors`:

- anchors are requests for detail types, not permission to invent facts;
- concrete names, numbers, citations, routes, costs, methods, datasets, tools,
  cases, outcomes, dates, standards, and compatibility claims must appear in
  the current task input or materials;
- when current evidence lacks a value, use a generic treatment or say the
  materials do not specify it;
- do not create fictional named cases, statistics, references, or capabilities.

Task-score artifacts:

- `runs/expanded/writingbench_official_eval.qwen.operational_anchors_evidence_policy.sample4_wb.heldout1.jsonl`
- `runs/expanded/writingbench_official_eval.mimo_judge.operational_anchors_evidence_policy.sample4_wb.heldout1.jsonl`

Grounding artifact:

- `runs/expanded/grounding_eval.mimo_judge.operational_anchors_evidence_policy.sample4_wb.heldout1.jsonl`

Coverage: 4/4 Qwen task-score success, 4/4 MIMO judge-swap success, and 4/4
MIMO grounding-probe success.

| Metric | Old anchors | Evidence-policy anchors |
|---|---:|---:|
| Qwen WritingBench mean | 6.05 | 6.25 |
| MIMO judge-swap mean | 7.05 | 6.85 |
| MIMO grounding mean | 4.5 | 5.75 |
| MIMO hallucination risk mean | 6.75 | 6.0 |
| MIMO unsupported claim count mean | 8.0 | 4.0 |

Per-pack evidence-policy scores:

| Pack | Qwen score | MIMO score | Grounding | Risk | Unsupported |
|---|---:|---:|---:|---:|---:|
| Academic Engineering Paper Outline EN | 6.8 | 7.8 | 6 | 8 | 7 |
| Product Description EN | 5.6 | 4.4 | 7 | 3 | 2 |
| Travel Guide ZH | 6.2 | 7.2 | 4 | 8 | 5 |
| Education Consulting EN | 6.4 | 8.0 | 6 | 5 | 2 |

Interpretation:

- The evidence policy is a real improvement on the grounding diagnostic while
  keeping task scores roughly intact.
- It is still not enough: Travel Guide and Academic remain high-risk, and the
  method only slightly beats few-shot under Qwen while remaining below few-shot
  under MIMO on the four-pack mean.
- The next method step should move beyond prompt admonitions to structured
  evidence extraction: enumerate allowable current-task facts first, then let
  anchors request only those facts or generic fallback content.

## Interpretation

This is not evidence for a new winning method. It is evidence that simply
compressing `cross_example_report` into a shorter context, even with a
task-first prompt order, does not recover the task-completion information lost
by skill-only compression.

The next method loop should focus on preserving current-task deliverable
semantics and distinguishing transferable structure from example-specific
content, not just shortening `SKILL.md`. In particular, a useful skill artifact
may need to preserve operational anchors such as expected detail depth,
evidence density, and how to instantiate task-specific empirical material. The
`task_first_operational_anchors` four-pack check is a concrete partial recovery
signal, but the grounding probe shows the current version still fabricates too
many unsupported specifics. The evidence-policy fix reduces this risk but does
not eliminate it.
