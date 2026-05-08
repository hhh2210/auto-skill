# Artifacts Inventory

This directory contains the minimal checked-in smoke dataset needed for
collaborator handoff. Rebuild queues, raw generations, reports, diagrams,
and exploratory outputs stay under ignored directories such as `runs/`,
`logs/`, `local_artifacts/`, or ignored subdirectories under `artifacts/`.

## Directories

| Directory | Contents | Commit policy |
| --- | --- | --- |
| `packs/` | User-visible example packs. `example_packs.v1.jsonl` is the current canonical cleaned smoke dataset. | Commit canonical smoke snapshots only. |
| `splits/` | Train-example and heldout-task split records plus split summaries. | Commit reproducible smoke splits. |
| `private/` | Benchmark-only rubrics, checklists, and eval references. | Commit only if needed to reproduce smoke evaluation. Never use as skill input. |

## Main Handoff Files

- `packs/example_packs.v1.jsonl`: canonical cleaned examples for the current smoke run.
- `splits/fewshot_splits.jsonl`: selected WritingBench and PresentBench train/heldout split.
- `private/example_private_eval.jsonl`: eval-only benchmark metadata.

## Ignored Generated Outputs

- `artifacts/jobs/`: desired-output generation queues and optional raw generation snapshots.
- `artifacts/reports/`: inspection, audit, and size reports.
- `artifacts/diagrams/`: Lark/Feishu flow diagrams.
- `artifacts/memory/`: extraction-memory snapshots.
- `artifacts/seed/`: legacy seed examples.

Generate those locally when needed; they should not clutter the handoff view.
