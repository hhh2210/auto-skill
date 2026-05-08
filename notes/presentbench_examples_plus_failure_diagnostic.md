# PresentBench Examples Plus Skill Failure Diagnostic

Date: 2026-05-09

## Scope

This note diagnoses why `examples_plus_one_shot_skill` and
`examples_plus_feature_skill` are positive on the current WritingBench smoke
slice but negative on the PresentBench surrogate slice.

Artifacts:

- `runs/presentbench_surrogate_eval.qwen.mvp.jsonl`
- `runs/presentbench_surrogate_eval.qwen.auto_skill.jsonl`
- `runs/presentbench_surrogate_eval.qwen.examples_plus_skill.heldout2.jsonl`
- `runs/mvp_metrics.heldout2.current.summary.json`

Coverage: 4 PresentBench packs x 2 heldout tasks x 7 modes, 56/56 success.
This is a text/rubric surrogate, not official PresentBench visual/PPT scoring.

## Aggregate Result

| Mode | Mean score | Delta vs prompt_only | Wins / losses / ties |
| --- | ---: | ---: | --- |
| few_shot_examples_only | 8.25 | +0.625 | 2 / 2 / 4 |
| one_shot_skill_from_examples | 8.85 | +1.225 | 2 / 2 / 4 |
| ours_no_validation | 8.625 | +1.0 | 3 / 2 / 3 |
| auto_skill | 8.375 | +0.75 | 3 / 2 / 3 |
| examples_plus_one_shot_skill | 6.625 | -1.0 | 2 / 5 / 1 |
| examples_plus_feature_skill | 7.0 | -0.625 | 2 / 5 / 1 |

The WritingBench-positive examples-plus mechanism does not transfer to
PresentBench surrogate tasks.

## Per-Pack Pattern

| Pack | examples+one-shot delta | examples+feature delta | Main signal |
| --- | ---: | ---: | --- |
| `presentbench_academia_USENIX` | +5.0, -1.0 | +5.0, -1.0 | Helps one weak prompt-only case, slightly hurts saturated case |
| `presentbench_education_CSAPP-Lectures_2015Fall` | +1.0, -1.0 | +3.0, -3.0 | Mixed; feature skill can help weak prompt-only but damages one heldout |
| `presentbench_education_THU_DSA` | -4.5, -3.5 | -4.5, -3.5 | Clear failure on both heldouts |
| `presentbench_talk_middle_school_presentation` | 0.0, -4.0 | 0.0, -1.0 | Mostly saturated, but examples+one-shot can disrupt exact wording/section placement |

## Failure Modes

### 1. Cross-Task Slide Layout Interference

For `presentbench_education_THU_DSA`, prompt-only and skill-only outputs often
kept dedicated figure slides and the requested slide count. Examples-plus
outputs became longer text outlines and merged required visual pages:

- grouped or omitted dedicated figure slides;
- combined visual requirements into summary slides;
- exceeded "max 6 bullet points" constraints;
- used generic "visual suggestion" placeholders.

This is not a lack of source context. It is interference from examples and skill
instructions competing with heldout-specific slide requirements.

### 2. Prompt Saturation On Already-Easy Tasks

For `presentbench_talk_middle_school_presentation`, prompt-only already scored
9-10. Examples-plus sometimes over-edited the structure:

- moved the required call-to-action out of the conclusion slide;
- changed exact required wording;
- split required definition content across multiple slides.

When the task input is already sufficiently specific, adding examples plus skill
increases instruction surface without adding useful information.

### 3. Skill-Only And Examples-Only Are Not The Same Failure

On PresentBench, `one_shot_skill_from_examples`, `ours_no_validation`, and
`auto_skill` remain positive overall in the surrogate. The negative result is
specific to concatenating raw examples with the skill. That suggests the issue
is not "skills are always bad for PresentBench"; it is the prompt composition
for examples-plus-skill.

## Current Hypothesis

Examples-plus-skill is useful for WritingBench because prose tasks benefit from
style/structure anchoring. For slide tasks, raw examples carry task-specific
layout decisions that conflict with heldout-specific visual and placement
constraints. The solver tends to imitate example deck shape and generic visual
language instead of obeying the heldout checklist.

## Recommended Fixes To Test

1. Do not use raw examples plus full skill as a default PresentBench mode.
2. Add a constrained composition mode for slide tasks:
   - examples only as abstract style signatures, not full task/output pairs;
   - heldout task constraints must dominate example patterns;
   - explicit "do not copy slide count, figure allocation, or section placement
     from examples unless requested by the current task".
3. Add a layout-plan intermediate step for PresentBench:
   - first extract current-task hard constraints: slide count, required figures,
     exact wording, per-slide limits, citation placement;
   - then apply skill only to fill style and coverage gaps.
4. Keep examples-plus-skill as a WritingBench-positive ablation, not a
   cross-domain method claim, until the constrained slide composition variant is
   tested.

