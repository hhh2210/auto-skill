# Minimal Memory Ablation Plan v3

## Decision

Use **方案 A: schema-stable deterministic distillation** for minimal-derived memory.

Do not change the `auto_skill_minimal` supervisor JSON contract. The supervisor keeps
the current `artifact_critique` plus `rule_grounding` shape. Memory entries are
derived deterministically in Python from the existing successful minimal rows.

方案 B, adding explicit `memory_updates` to the supervisor schema, is deferred to
future work. It would invalidate already-run minimal rows and shift the supervisor
distribution, so it is not appropriate before the 30WB baseline ablation.

## Deterministic Mapping

Only successful `mode == "auto_skill_minimal"` rows are eligible.

| Source field | Condition | Memory kind | Evidence examples |
| --- | --- | --- | --- |
| `rule_grounding[*].rule` | `supported_by` non-empty | `candidate_rule` | `supported_by` |
| `rule_grounding[*].rule` | `contradicted_by` non-empty | `conflict` | `contradicted_by` |
| `rule_grounding[*].rule` | `out_of_scope` non-empty | `outlier` | `out_of_scope` |
| `artifact_critique.over_specific[*]` | always | `do_not_generalize` | empty list |

`artifact_critique.too_generic` is skipped for now. It would need a new lesson kind
such as `over_generalization` before it can be represented cleanly.

`artifact_critique.missing` is skipped because it is prescriptive and does not carry
a stable evidence anchor in the current supervisor contract.

Critique-derived entries have `evidence_examples = []` by design. Reports must expose
the fraction of unanchored entries separately.

## Memory Provenance

Memory entries keep `schema_version = "skill-extraction-memory/v1"` for backward
compatibility, but every written entry must include:

```json
{"derivation": "feature_reports" | "minimal_supervisor_report"}
```

Legacy v1 rows without this field are read as `feature_reports`. New minimal ablation
runs must write a separate memory file, for example:

```text
artifacts/memory/extraction_memory.minimal.v1.jsonl
```

Use `update_extraction_memory.py --require-derivation minimal_supervisor_report` for
minimal-only ablation artifacts. Legacy feature-driven memory is implementation
compatible but excluded from paper-facing claims.

`derivation` is included in `memory_id`, so the same lesson emitted by the
feature-driven pipeline and by minimal supervisor distillation will intentionally
be two distinct entries. Reports must treat raw entry counts as provenance-aware
counts rather than unique semantic-lesson counts.

## Memory Scopes

| Scope | Behavior | Paper status |
| --- | --- | --- |
| `within_pack` | keep only entries whose `pack_id` equals the current pack | include |
| `cross_pack_holdout` | keep all entries except the current pack | include |
| `cross_pack` | keep all entries, including current pack | leakage-prone, exclude from paper |

`cross_pack_holdout` is implemented as a runtime filter so we do not need one memory
artifact per fold.

## Paper-Facing Ablation Matrix

| Variant | Memory source | Scope | Purpose |
| --- | --- | --- | --- |
| `auto_skill_minimal` | none | n/a | clean baseline |
| `auto_skill_minimal+mem_within` | minimal supervisor self-distillation | `within_pack` | second-pass refinement |
| `auto_skill_minimal+mem_xpack_holdout` | minimal supervisor leave-pack-out distillation | `cross_pack_holdout` | cross-pack transfer |
| `auto_skill_minimal+mem_legacy` | feature-driven rows | `cross_pack` | implementation-only; excluded from paper |

The current `ours_no_validation` / `auto_skill_feature_driven_no_validation` numbers
must not be used as the minimal baseline. Targeted v1 success criteria remain blocked
until this matrix has real 30WB minimal numbers.

## Token-Cost Guard

Every `auto_skill_minimal` row must record `extraction_prompt_tokens` from the
extraction model call when provider usage is available.

The ablation report must show median and p90 `extraction_prompt_tokens` for:

1. no memory
2. within-pack minimal memory
3. cross-pack-holdout minimal memory

It must also report `tokens_with_memory / tokens_no_memory` ratios for both memory
variants.

For `cross_pack_holdout`, prompt-token median must be no more than 2x the no-memory
median. If it exceeds that guard, add deterministic top-k memory truncation before
claiming cross-pack transfer. The top-k ranking should prefer entries with more
`evidence_examples`; unanchored critique entries sort after anchored entries.

## Execution Order

| Step | Action | Output | Blocks later steps |
| --- | --- | --- | --- |
| 0 | Lock this plan v3 | this note | yes |
| 1 | Implement minimal row memory distillation + unit tests | code + tests | yes |
| 2 | Implement `cross_pack_holdout` filter + unit tests | code + tests | yes |
| 3 | Run vanilla 30WB MIMO `auto_skill_minimal` no-memory baseline | `runs/expanded/skill_minimal.30wb.no_memory.jsonl` plus eval rows | yes |
| 4 | Distill minimal-only memory from step 3 | `artifacts/memory/extraction_memory.minimal.v1.jsonl` | yes |
| 5 | Run `mem_within` and `mem_xpack_holdout` 30WB MIMO evals | skill and eval JSONL for both variants | no |
| 6 | Write three-way ablation report | `runs/expanded/minimal_memory_ablation.30wb.mimo.md` | yes |
| 7 | Rebase targeted v1 plan on the best confirmed minimal baseline | plan v4 | n/a |
