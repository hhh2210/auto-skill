# Background Review 2026-05-09

This note preserves the skeptical background-agent review evidence for the
current auto-skill handoff. The full transcripts live in the Codex thread; this
file records the actionable conclusions so repo readers do not need the chat
history.

## Scope Reviewed

- Active objective: construct and clean WritingBench / PresentBench examples
  with Qwen3.5-Plus and MIMO, verify the data flow against
  `notes/benchmark_flow.md`, and keep the research strategy skeptical.
- Main artifacts inspected: benchmark-flow audit code, readiness reports,
  expanded cleaning artifacts, completion audit, WritingBench / PresentBench
  eval rows, MIMO subset, and current docs.

## Findings Incorporated

### Benchmark-Flow Contract

Earlier review found two P2 issues in `src/auto_skill/benchmark_flow.py`:

- private rows were keyed by `pack_id`, but their `split_id` / `source` header
  metadata was not required to match the visible pack and split;
- duplicate private `task_ref` entries could be hidden by last-write-wins dict
  construction.

These were subsequently fixed and re-reviewed. The follow-up reviewer reported
no remaining P1/P2 objections for the benchmark-flow contract after running:

```bash
uv run python -m unittest tests.test_benchmark_flow
uv run python scripts/data/audit_benchmark_flow.py
uv run python -m unittest discover -s tests
```

Current expanded artifacts also pass:

```bash
uv run python scripts/data/audit_benchmark_flow.py \
  --splits runs/expanded/fewshot_splits.max_available.jsonl \
  --packs runs/expanded/example_packs.max_available.qwen.v1.jsonl \
  --private-eval runs/expanded/example_private_eval.max_available.jsonl \
  --jobs runs/expanded/example_generation_jobs.max_available.jsonl \
  --generated-outputs runs/expanded/generated_desired_outputs.max_available.qwen.latest_success.jsonl
```

## Current Blockers

### P1: PresentBench Official Scores

Full readiness is still intentionally `not_ready`. The blocker is not missing
JSONL wrappers; it is missing upstream PresentBench official score artifacts
for 8 heldout tasks x 2 modes. `run_presentbench_official_judge.py --dry-run`
renders the selected 16 `judge.py` commands, but the local environment lacks
`GENAI_API_KEY`, and upstream PresentBench currently supports `gemini` /
`gemini_inline` rather than the Bailian/OpenAI-compatible Qwen endpoint.

### P1: Main Method Claim Is Not Supported Yet

Current auto-skill evidence is diagnostic, not paper-ready. WritingBench Qwen
and MIMO judge-swap both show that skill-only modes do not reliably beat
`prompt_only`, `few_shot_examples_only`, or `one_shot_skill_from_examples`.
The healthier signal is examples-plus-skill, but that supports a different
claim: induced skills can organize or constrain examples, not replace them.

### P2: MIMO Is an Audit / Subset Cleaner, Not Canonical Full Cleaner

Qwen is the canonical cleaner for the MVP and max-available snapshots. MIMO has
been used for judge-swap, audit, and a 15-pack frozen subset that passes
benchmark-flow. Do not describe MIMO as the complete expanded cleaner until a
full MIMO pass is frozen and audited.

### P2: Research Confidence Remains Conditional

The completion audit should keep saying the active goal is not complete. The
repo is ready for engineering handoff and cleaned-data work, but not for a
paper-level claim. Required next steps remain:

1. run PresentBench official scoring after configuring `GENAI_API_KEY`;
2. calibrate Qwen/MIMO judge disagreements before expanding claims;
3. redesign the skill mechanism or reframe the contribution around
   examples-plus-skill / operational planning, not skill-only compression.

## Accepted Handoff State

Reviewers agreed the current repository can be handed off as:

- a clean Python/uv engineering prototype;
- a benchmark-flow-audited cleaned-data workspace;
- a documented MVP/surrogate evaluation harness;
- a research state with explicit blockers and no paper-level overclaim.

They did not consider the active research objective complete.
