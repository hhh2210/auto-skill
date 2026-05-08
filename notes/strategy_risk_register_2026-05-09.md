# Strategy Risk Register 2026-05-09

This note records the current strategy confidence audit after the expanded
WritingBench sample and Qwen/MIMO judge-swap. The project is engineering-ready
for handoff, but not research-confident for a paper claim.

## Current Stable Evidence

- Data cleaning follows `notes/benchmark_flow.md` for both checked-in MVP data
  and the expanded 30-WB / 20-PB local workspace:
  - `audit_benchmark_flow.py`: `status=ok`, no errors/warnings.
  - Expanded workspace: 50 packs, 150 frozen train examples, 100 heldout tasks.
  - `report_expanded_cleaning_status.py --expect-status ready`: ready.
  - `report_expanded_cleaning_status.py --require-mimo-subset --expect-status ready`:
    ready; the local MIMO subset covers 15 packs, 45 frozen train examples, 30
    heldout tasks, 45 generation jobs, 12 WritingBench packs, and 3 PresentBench
    packs.
- MVP readiness is green as an engineering handoff:
  - `report_experiment_readiness.py --profile mvp --expect-status ready`.
- Full readiness is intentionally not green:
  - `report_experiment_readiness.py --profile full --expect-status not_ready`.
  - Blocker: 8 PresentBench heldout tasks x 2 official modes have
    `missing_score_artifact`.
- Expanded WritingBench sample shows judge sensitivity on the same candidate
  outputs:
  - 4 non-MVP packs x 1 heldout x 6 modes = 24 cells per judge.
  - Qwen and MIMO both completed 24/24 cells.
  - 10 Qwen-vs-MIMO sign flips after joining on identical candidate/baseline
    output hashes.

## Main Loopholes

| Loophole | Evidence | Risk | Fix before claim |
| --- | --- | --- | --- |
| Evaluator calibration is unstable | 10 sign flips in `runs/expanded/judge_disagreements.qwen_vs_mimo.sample4_wb.heldout1.jsonl` on identical outputs | Any mean score claim can be an artifact of judge preference | Human/third-judge adjudicate sign flips; build a failure taxonomy before expanding N |
| Skill-only is not consistently better than examples | MVP WritingBench Qwen and MIMO five-mode runs both show auto-skill below prompt/few-shot/one-shot on average | Current headline "examples -> reusable skill -> better heldout performance" is unsupported | Redesign induction or narrow the claim to examples-plus-skill augmentation only if it survives calibration |
| PresentBench official score is missing | Full readiness fails only on missing official score artifacts | Surrogate slide scores can overstate or misstate real visual/PPT performance | Run upstream PresentBench official judge for `prompt_only` and `auto_skill` |
| Same-model smoke evidence remains in canonical readiness | MVP readiness warns model monoculture and old rows missing top-level model identity | Reviewers can reject Qwen-only solver/judge evidence | Use split solver/judge model artifacts for any reported result; avoid old rows without model identity |
| MIMO may over-reward structure and rubric keywords | Background packet review found Qwen more credible for Academic/Education sign flips, while MIMO rewarded heading/rubric-term reuse | MIMO judge-swap can make weak outputs look improved | Treat MIMO as one calibration axis, not ground truth |
| MIMO subset is not full MIMO cleaning | The audited subset is 15 packs / 45 train examples, while the canonical expanded split is 50 packs / 150 train examples | Overclaiming could make handoff look like a full dual-model dataset | Label it as an audited subset unless a full MIMO pass is frozen and audited |
| PresentBench examples-plus-skill is negative under surrogate | PresentBench surrogate examples-plus modes underperform prompt-only in current smoke | WritingBench augmentation result may not transfer cross-domain | Test slide-specific constrained composition without raw examples in final prompt |
| Generic deliverable guardrail is insufficient | `notes/deliverable_guard_probe_2026-05-09.md` shows Education Consulting `examples_plus_feature_skill` improves to near prompt-only but `ours_no_validation` still scores 5.4 vs prompt-only 7.6 after adding final-deliverable priority; deterministic n-gram contamination evidence is mixed | The failure is deeper than "the model wrote an outline" | Target skill-only compression loss and task-specific constraint extraction before blaming example copying |
| Skill compression is cheap but lossy | `notes/skill_compression_diagnostic_2026-05-09.md` shows skill-only modes use about 21-29% of few-shot generation input tokens, but WritingBench negative transfer remains 37.5-75% depending on mode/judge. `notes/feature_signature_ablation_2026-05-09.md` shows compact-signature and task-first signature smokes still below prompt-only/few-shot, while sanitized `task_first_operational_anchors` reaches 6.0 on one hard cell. | A cost-only win is not enough for the main method claim; shortening context and simple task-first ordering are insufficient, but operational/detail anchors may preserve useful example signal | Validate operational anchors on the calibrated multi-pack sample before expanding N |

## Current Strategy Decision

Do not expand WritingBench N as paper evidence yet. Larger N will only make the
judge-dependence problem more expensive unless the evaluator is calibrated.

The first pass of this loop is recorded in
`notes/judge_disagreement_taxonomy_2026-05-09.md`, with machine-readable labels
in `notes/judge_disagreement_taxonomy_2026-05-09.jsonl`. The next defensible
loop is:

1. Validate or revise the provisional sign-flip labels:
   `judge_error`, `candidate_error`, `baseline_error`, `small_delta_noise`, or
   `ambiguous`.
2. Convert validated labels into a failure taxonomy:
   - missing task completion despite strong structure;
   - source-context pollution from examples/materials;
   - overlong outline/planning output instead of final deliverable;
   - rubric-keyword over-reward;
   - small numerical delta noise.
3. Decide which judge or rubric protocol is trusted for each task family.
4. Rerun the new `task_first_operational_anchors` mode on the same calibrated
   four-pack WritingBench sample before expanding N.
5. Prefer operational/detail anchors over another skill-only LOO loop; current
   evidence says standalone compression is cheap but unstable, while the first
   sanitized operational-anchor smoke improved score/output completeness on one
   cell; grounding and authenticity remain unresolved.

## Handoff Commands

```bash
uv run python -m unittest discover -s tests
uv run ruff check .
diff -q AGENTS.md CLAUDE.md
uv run python scripts/data/audit_benchmark_flow.py
uv run python scripts/data/audit_benchmark_flow.py \
  --splits runs/expanded/fewshot_splits.30wb_20pb.jsonl \
  --packs runs/expanded/example_packs.30wb_20pb.qwen.v1.jsonl \
  --private-eval runs/expanded/example_private_eval.30wb_20pb.jsonl \
  --jobs runs/expanded/example_generation_jobs.30wb_20pb.jsonl \
  --generated-outputs runs/expanded/generated_desired_outputs.30wb_20pb.qwen.jsonl
uv run python scripts/ops/report_expanded_cleaning_status.py --expect-status ready
uv run python scripts/ops/report_expanded_cleaning_status.py --require-mimo-subset --expect-status ready
uv run python scripts/ops/report_experiment_readiness.py --profile mvp --expect-status ready
uv run python scripts/ops/report_experiment_readiness.py --profile full --expect-status not_ready
uv run python scripts/metrics/validate_disagreement_taxonomy.py \
  --packets runs/expanded/judge_disagreements.qwen_vs_mimo.sample4_wb.heldout1.jsonl \
  --taxonomy notes/judge_disagreement_taxonomy_2026-05-09.jsonl \
  --expect-status ok
```
