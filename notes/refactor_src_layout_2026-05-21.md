# Refactor `src/auto_skill/` Layout — 2026-05-21

## Audience and execution model

This document is the executable plan for codex (or any contributor) to refactor
`src/auto_skill/` from a 38-file flat directory into a responsibility-scoped
package layout. The atomic unit is a **git commit** on
`feature/auto-skill-minimal` (or a dedicated branch). No PR is required between
commits; each commit must independently pass the verification suite in §6 so it
can be `git revert`-ed without follow-ups.

Codex should treat each commit section as a self-contained worklist. Do not
batch commits, and do not skip the verification suite between commits.

## 1. Goals and non-goals

### Goals

- **G1.** Move from flat 38-file layout to responsibility-scoped sub-packages
  (`cleaning/`, `baselines/`, `eval/`, `metrics/`, `diagnostics/`, ...). Directory
  structure itself should make `load-bearing` vs `debug-only` obvious.
- **G2.** Each baseline becomes its own module under `baselines/`, registered in
  a single `REGISTRY`. Kill the `if mode == "..."` cascades scattered across
  `mvp.py`, `scripts/skills/run_skill_mvp.py`, `scripts/eval/run_*.py`.
- **G3.** Zero behavior change. Every public import path that exists today
  must keep working (via facade `__init__.py` or facade `.py`) until at least
  the optional facade-cleanup commit. All existing tests, audits, and
  readiness gates must keep passing on every commit.
- **G4.** Each commit is atomically reversible.

### Non-goals

- No prompt text changes.
- No JSONL schema changes.
- No CLI flag or default changes.
- No deletion of any externally imported symbol until C8 (optional).
- No move of `scripts/` or `tests/` structure (only path strings inside them
  may change if a facade is dropped).

## 2. Target layout (final state after C7)

```
src/auto_skill/
├── __init__.py
├── io/
│   ├── __init__.py
│   ├── jsonl.py                   # load_jsonl / load_jsonl_lenient_final_line / write_jsonl
│   └── ledger.py                  # stage ledger / resume helpers
├── llm/                           # was llm.py — atomic swap, see §3
│   ├── __init__.py                # facade re-exports below
│   ├── client.py                  # ChatCompletionClient / ChatCompletionConfig / ConfigError / ChatCompletionResult
│   ├── env.py                     # first_set_env / parse_*_env helpers
│   └── parse.py                   # parse_json_object (moved from mvp.py)
├── schemas/                       # was schemas.py — atomic swap, see §3
│   ├── __init__.py                # facade re-exports
│   ├── records.py                 # SchemaValidationError, _require_*, schema-version constants
│   ├── generated_output.py        # validate_generated_output_row
│   ├── skill_row.py               # validate_skill_row
│   ├── eval_row.py                # validate_eval_row / validate_author_style_eval_row
│   ├── artifacts.py               # validate_artifact_rows
│   └── user_example.py            # UserExample dataclass
├── cleaning/
│   ├── __init__.py
│   ├── data_cleaning.py
│   ├── packs.py                   # was example_packs.py (after IO is extracted)
│   ├── generated_outputs.py
│   ├── benchmark_flow.py
│   └── author_style/
│       ├── __init__.py
│       ├── common.py              # was author_style_common.py
│       ├── sources.py             # was author_style_sources.py
│       ├── gpt.py                 # was author_style_gpt.py
│       ├── packs.py               # was author_style_packs.py
│       ├── pipeline.py            # was author_style_pipeline.py
│       ├── selection.py           # was author_style_selection.py
│       ├── audit.py               # was author_style_audit.py
│       ├── cluster_manifest.py    # was author_style_cluster_manifest.py
│       ├── eval_summary.py        # was author_style_eval_summary.py
│       ├── negative_transfer.py   # was author_style_negative_transfer.py
│       ├── stratified_eval.py     # was author_style_stratified_eval.py
│       └── quality/               # was author_style_quality.py (1248 LOC) — split in C4
│           ├── __init__.py        # facade
│           ├── audit.py
│           ├── metrics.py
│           └── report.py
├── baselines/
│   ├── __init__.py                # REGISTRY (populated in C7)
│   ├── _protocol.py               # Baseline protocol (C7)
│   ├── prompt_only.py
│   ├── few_shot_examples_only.py
│   ├── one_shot_skill_from_examples.py
│   ├── examples_plus_one_shot_skill.py
│   ├── feature_driven.py          # was skill_induction.py
│   ├── feature_driven_with_validation.py
│   ├── examples_plus_feature_skill.py
│   ├── slide_constrained.py
│   ├── layout_plan.py
│   ├── ours_full.py
│   └── minimal.py                 # was skill_induction_minimal.py
├── prompts/
│   ├── __init__.py
│   ├── judge.py                   # build_judge_prompt / evaluation_criteria / extract_overall_score
│   ├── evidence.py                # build_current_evidence_inventory / scaffolding_plan / validate_evidence_plan
│   └── operational_anchor.py      # build_operational_anchor_context / build_planned_operational_anchor_prompt
├── eval/
│   ├── __init__.py
│   ├── writingbench.py            # was writingbench_eval.py
│   ├── presentbench_surrogate.py  # part of was presentbench_eval.py
│   ├── presentbench_official.py   # part of was presentbench_eval.py
│   ├── heldout.py
│   ├── judge_reason_extraction.py # was judge_reason_extraction.py
│   └── modes.py                   # FULL_*_MODES / MVP_*_MODES / PRESENTBENCH_OFFICIAL_MODES / SKILL_REQUIRED_MODES
├── metrics/                       # was metrics.py — atomic swap, see §3
│   ├── __init__.py                # facade re-exports
│   ├── numeric.py
│   ├── token_usage.py
│   ├── skill_artifact.py
│   ├── self_consistency.py
│   ├── literal_leakage.py
│   ├── eval_summary.py            # was eval_summary.py
│   └── readiness.py               # was readiness.py
├── memory/
│   ├── __init__.py
│   ├── extraction.py              # was extraction_memory.py
│   └── skill.py                   # was skill_memory.py
└── diagnostics/
    ├── __init__.py
    ├── criterion.py               # was criterion_diagnosis.py
    ├── coding_style.py            # was coding_style_diagnostic.py
    ├── compression.py             # was author_style_compression_diagnostic.py
    ├── negative_transfer.py       # was author_style_negative_transfer_diagnostic.py
    ├── grounding.py               # was grounding.py
    ├── pairwise_likeness.py       # was pairwise_likeness.py
    └── pattern_similarity.py      # was pattern_similarity.py
```

The original flat-path modules are preserved as **facade modules** (either
`old_name.py` re-exporting from the new path, or `old_name/__init__.py` for the
three collision cases below) until at least one cycle after C7 lands.

## 3. Collision handling (atomic swap, not transitional names)

Three current files collide with their target sub-package name:

| Old file | Target package | Resolution |
|---|---|---|
| `auto_skill/llm.py` | `auto_skill/llm/` | C1 deletes file, creates package whose `__init__.py` re-exports old surface |
| `auto_skill/schemas.py` | `auto_skill/schemas/` | C1 deletes file, creates package whose `__init__.py` re-exports old surface |
| `auto_skill/metrics.py` | `auto_skill/metrics/` | C5 deletes file, creates package whose `__init__.py` re-exports old surface |

**Rule for codex:** when handling these three, perform the delete + package
creation in the same commit. Do **not** introduce transitional names like
`llm_core/`. The single-commit atomic swap is mandatory because the public
surfaces are small and stable; transitional names would create review
confusion and require a follow-up rename commit.

All other moves do not collide because the target path is in a different
sub-directory (e.g. `readiness.py` → `metrics/readiness.py`).

## 4. Facade contract

A facade is either:

1. A `.py` file at the old flat path containing only re-exports, e.g.:
   ```python
   """Compatibility facade — moved to auto_skill.memory.extraction."""
   from auto_skill.memory.extraction import *  # noqa: F401,F403
   ```
2. A package `__init__.py` (for the three collision cases) re-exporting the
   exact names the old module previously exported.

**Rules:**

- Facade must re-export every public name the old module exported. Run
  `python -c "import auto_skill.<oldname>; print(sorted(n for n in dir(...) if not n.startswith('_')))"` before and after to confirm parity.
- Facade must keep working until C8 (optional). Removing a facade before C8
  is a behavior change and forbidden.
- Facade files should never contain logic — only `from ... import` lines and
  an optional module docstring. Lint with: `grep -E "^(def |class )" src/auto_skill/<facade>.py` must return empty.

## 5. Commit plan

Each commit lists: title, scope, files, verification, risk.

Conventional commit prefixes: `refactor(src):`, `refactor(cleaning):`,
`refactor(baselines):`, etc.

### C0 — `refactor(src): scaffold target sub-packages`

**Scope:** create empty sub-packages, no logic moves.

**Add:**
```
src/auto_skill/io/__init__.py
src/auto_skill/llm/__init__.py             ← will overwrite llm.py in C1, NOT in this commit
... (skip the 3 collision dirs in C0; they are created in C1/C5)
src/auto_skill/schemas/__init__.py         ← skip in C0
src/auto_skill/metrics/__init__.py         ← skip in C0
src/auto_skill/cleaning/__init__.py
src/auto_skill/cleaning/author_style/__init__.py
src/auto_skill/cleaning/author_style/quality/__init__.py
src/auto_skill/baselines/__init__.py
src/auto_skill/prompts/__init__.py
src/auto_skill/eval/__init__.py
src/auto_skill/memory/__init__.py
src/auto_skill/diagnostics/__init__.py
```

Each `__init__.py` is a single docstring line; no re-exports yet.

**Verification:** the full suite in §6 plus
`uv run python -c "import auto_skill.cleaning, auto_skill.baselines, auto_skill.eval, auto_skill.memory, auto_skill.diagnostics, auto_skill.prompts"`.

**Risk:** none.

### C1 — `refactor(src): extract io/ and atomically replace llm.py, schemas.py with packages`

**Scope:** introduce `io/`, swap `llm.py` and `schemas.py` for sub-packages,
extract `parse_json_object` from `mvp.py` to `llm/parse.py`.

**Move (using `git mv` where possible, otherwise create + delete + record in commit message):**

1. **`io/jsonl.py`** — move `load_jsonl`, `load_jsonl_lenient_final_line`, `write_jsonl` from `example_packs.py`.
   - Inside `example_packs.py` add a re-export line at the top:
     `from auto_skill.io.jsonl import load_jsonl, load_jsonl_lenient_final_line, write_jsonl  # noqa: F401`
2. **`llm/` package — atomic swap:**
   - `git rm src/auto_skill/llm.py` and create:
     - `llm/client.py` — `ChatCompletionConfig`, `ChatCompletionClient`, `ChatCompletionResult`, `ConfigError`.
     - `llm/env.py` — every `first_set_env*` / `parse_*_env*` helper.
     - `llm/parse.py` — `parse_json_object` (moved from `mvp.py`).
     - `llm/__init__.py` — re-export the previous `llm.py` public surface:
       ```python
       from auto_skill.llm.client import (
           ChatCompletionClient, ChatCompletionConfig,
           ChatCompletionResult, ConfigError,
       )
       from auto_skill.llm.env import (
           first_set_env, first_set_env_name,
           parse_chain_float_env, parse_chain_int_env,
           parse_chain_positive_int_env, parse_chain_optional_float_env,
           parse_chain_optional_positive_int_env, parse_chain_optional_bool_env,
           parse_chain_bool_env, parse_optional_float_env, parse_float_env,
           parse_int_env, parse_positive_int_env, parse_optional_positive_int_env,
           parse_optional_bool_env, parse_bool_env,
       )
       __all__ = [...]  # mirror the names above
       ```
   - In `mvp.py`: replace the body of `parse_json_object` with `from auto_skill.llm.parse import parse_json_object  # noqa: F401` at module top (do NOT delete the name from mvp.py — downstream imports `from auto_skill.mvp import parse_json_object` and must keep working).
3. **`schemas/` package — atomic swap:**
   - `git rm src/auto_skill/schemas.py` and create:
     - `schemas/records.py` — schema-version constants, `SchemaValidationError`, `_require_*` helpers.
     - `schemas/generated_output.py` — `validate_generated_output_row`.
     - `schemas/skill_row.py` — `validate_skill_row`, `SKILL_INDUCTION_SCHEMA_VERSION` if used here.
     - `schemas/eval_row.py` — `validate_eval_row`, `validate_author_style_eval_row`.
     - `schemas/artifacts.py` — `validate_artifact_rows`.
     - `schemas/user_example.py` — `UserExample` dataclass.
     - `schemas/__init__.py` — re-export every public name from `schemas.py`.

**Symbol parity check (mandatory before commit):**
```bash
uv run python - <<'PY'
import importlib, sys
for name in ("auto_skill.llm", "auto_skill.schemas"):
    mod = importlib.import_module(name)
    print(name, sorted(n for n in dir(mod) if not n.startswith("_")))
PY
```
Compare against `git show HEAD~1:src/auto_skill/llm.py` / `schemas.py` exports.
Any missing public name must be re-exported before commit.

**Verification:** §6 suite + this extra check:
```bash
uv run python -c "from auto_skill.llm import ChatCompletionClient, ChatCompletionConfig, ConfigError, parse_bool_env"
uv run python -c "from auto_skill.llm.parse import parse_json_object as p1; from auto_skill.mvp import parse_json_object as p2; assert p1 is p2"
uv run python -c "from auto_skill.schemas import validate_skill_row, validate_eval_row, UserExample"
uv run python -c "from auto_skill.example_packs import load_jsonl, write_jsonl"
```

**Risk:** medium-high — `auto_skill.llm` and `auto_skill.schemas` are widely
imported (18 + 20 import sites). The symbol parity check above is mandatory.

### C2 — `refactor(src): move memory/ and diagnostics/ to sub-packages`

**Scope:** 9 files relocated, facades left at old paths.

**Move (use `git mv`):**

| Old | New | Facade |
|---|---|---|
| `extraction_memory.py` | `memory/extraction.py` | `extraction_memory.py` ← facade |
| `skill_memory.py` | `memory/skill.py` | `skill_memory.py` ← facade |
| `criterion_diagnosis.py` | `diagnostics/criterion.py` | facade |
| `coding_style_diagnostic.py` | `diagnostics/coding_style.py` | facade |
| `author_style_compression_diagnostic.py` | `diagnostics/compression.py` | facade |
| `author_style_negative_transfer_diagnostic.py` | `diagnostics/negative_transfer.py` | facade |
| `grounding.py` | `diagnostics/grounding.py` | facade |
| `pairwise_likeness.py` | `diagnostics/pairwise_likeness.py` | facade |
| `pattern_similarity.py` | `diagnostics/pattern_similarity.py` | facade |

**Verification:** §6 suite. Diagnostics are mostly consumed by
`scripts/metrics/*.py`; verify no script breaks.

**Risk:** low.

### C3 — `refactor(cleaning): consolidate cleaning modules under cleaning/`

**Scope:** move cleaning-side code under `cleaning/`. `author_style_quality.py`
stays unchanged here — only its location moves (still as one 1248-LOC file
under `cleaning/author_style/quality.py`); the split into `quality/` package
happens in C4.

**Move:**

| Old | New | Facade |
|---|---|---|
| `data_cleaning.py` | `cleaning/data_cleaning.py` | facade |
| `example_packs.py` | `cleaning/packs.py` | `example_packs.py` ← facade re-exporting both `cleaning.packs` and `io.jsonl` |
| `generated_outputs.py` | `cleaning/generated_outputs.py` | facade |
| `benchmark_flow.py` | `cleaning/benchmark_flow.py` | facade |
| `author_style_common.py` | `cleaning/author_style/common.py` | facade |
| `author_style_sources.py` | `cleaning/author_style/sources.py` | facade |
| `author_style_gpt.py` | `cleaning/author_style/gpt.py` | facade |
| `author_style_packs.py` | `cleaning/author_style/packs.py` | facade |
| `author_style_pipeline.py` | `cleaning/author_style/pipeline.py` | facade |
| `author_style_selection.py` | `cleaning/author_style/selection.py` | facade |
| `author_style_audit.py` | `cleaning/author_style/audit.py` | facade |
| `author_style_cluster_manifest.py` | `cleaning/author_style/cluster_manifest.py` | facade |
| `author_style_eval_summary.py` | `cleaning/author_style/eval_summary.py` | facade |
| `author_style_negative_transfer.py` | `cleaning/author_style/negative_transfer.py` | facade |
| `author_style_stratified_eval.py` | `cleaning/author_style/stratified_eval.py` | facade |
| `author_style_quality.py` | `cleaning/author_style/quality.py` (file, not package) | facade |
| `author_style_cleaning.py` | already a facade — update its inner imports to point at new paths |

**Verification:** §6 suite. Also grep `notes/` and `docs/` for hard references
to old module paths; update there too in the same commit.
```bash
grep -rn "from auto_skill\.example_packs\|from auto_skill\.author_style_" docs/ notes/ || true
```

**Risk:** medium — lots of files, but mechanical. Verify the `author_style_cleaning.py` facade still resolves its `from auto_skill.author_style_* import *` lines (update them to `from auto_skill.cleaning.author_style.* import *`).

### C4 — `refactor(cleaning): split author_style_quality into quality/ package`

**Scope:** the only logic-edge-changing commit in this refactor. Split the
1248 LOC `cleaning/author_style/quality.py` into:

- `cleaning/author_style/quality/__init__.py` — re-export every public name.
- `cleaning/author_style/quality/audit.py` — audit gates and threshold checks.
- `cleaning/author_style/quality/metrics.py` — numeric/aggregate metric helpers.
- `cleaning/author_style/quality/report.py` — the `summarize_*` and report builders.

**How to split:** read the file end-to-end first. Group functions by:

1. **audit**: anything mutating/filtering accepted packs (e.g. threshold
   evaluators, accept/reject flags).
2. **metrics**: pure computations (means, percentiles, summaries of numeric
   columns).
3. **report**: I/O-shaped functions that take loaded jsonls and emit summary
   dicts / write reports.

If a function does not fit cleanly, leave it in `report.py` and add a `# TODO`
with the function name; document the residual in the commit body.

**Symbol parity check:**
```bash
uv run python - <<'PY'
import auto_skill.cleaning.author_style.quality as q
print(sorted(n for n in dir(q) if not n.startswith("_")))
PY
```
Diff against the pre-split list. The facade `quality/__init__.py` must
re-export every name.

**Verification:** §6 suite. Plus:
```bash
uv run python scripts/metrics/summarize_author_style_quality.py --help
```
(invocation only — does not need real data; just confirm imports resolve.)

**Risk:** medium. Recommend running the `simplify` skill (or `code-simplifier`
agent in worktree mode) on the split result for a second opinion before commit.

### C5 — `refactor(eval+metrics): extract eval/ package and atomically replace metrics.py with package`

**Scope:** consolidate eval modules and split `metrics.py`.

**Move:**

| Old | New | Facade |
|---|---|---|
| `writingbench_eval.py` | `eval/writingbench.py` | facade |
| `presentbench_eval.py` | split into `eval/presentbench_surrogate.py` + `eval/presentbench_official.py` if a clean cut exists; otherwise `eval/presentbench.py` (record reason in commit body) | facade |
| `judge_reason_extraction.py` | `eval/judge_reason_extraction.py` | facade |
| `readiness.py` | `metrics/readiness.py` | facade |
| `eval_summary.py` | `metrics/eval_summary.py` | facade |

**New file:** `eval/modes.py` — centralize the mode constants. Move:

- `FULL_SKILL_MODES`, `MVP_SKILL_MODES`, `FULL_EVAL_MODES`, `MVP_EVAL_MODES`,
  `PRESENTBENCH_OFFICIAL_MODES` from `readiness.py`.
- `SKILL_REQUIRED_MODES` from `scripts/eval/run_writingbench_official_eval.py`
  and `scripts/eval/run_heldout_eval.py` (these two scripts must then import
  from `auto_skill.eval.modes`).
- `DEFAULT_BASELINE_MODE`, `DEFAULT_ARTIFACT_COMPARE_MODES` from `metrics.py`.

`metrics/readiness.py` must keep importing from `auto_skill.eval.modes` (no
duplication).

**`metrics/` package — atomic swap:**

- `git rm src/auto_skill/metrics.py` and create:
  - `metrics/numeric.py` — `_safe_mean`, `_percentile`, `numeric_summary`.
  - `metrics/token_usage.py` — `_usage_tokens`, `_empty_usage_tokens`, `_add_usage`, `_sum_usage_tokens`, `token_usage_summary`, `skill_induction_token_usage_summary`.
  - `metrics/skill_artifact.py` — `_normalize_heading`, `_section_key`, `_is_rule_line`, `count_skill_section_rules`, `_list_len`, `_candidate_support_count`, `skill_artifact_metric`, `skill_artifact_summary`, `score_and_negative_transfer_summary`.
  - `metrics/self_consistency.py` — `_example_signature_for_prompt`, `build_abstract_signatures_prompt`, `build_signature_consistency_judge_prompt`, `parse_abstract_signatures`, `_finite_score_0_to_10`, `parse_self_consistency_report`, `self_consistency_report_errors`, `self_consistency_summary`.
  - `metrics/literal_leakage.py` — `_json_blob`, `_literal_candidates_from_text`, `train_literal_candidates`, `literal_leakage_report`.
  - `metrics/__init__.py` — re-export every public name from the old `metrics.py`.

**Symbol parity check:**
```bash
uv run python - <<'PY'
import auto_skill.metrics
print(sorted(n for n in dir(auto_skill.metrics) if not n.startswith("_")))
PY
```
Compare against `git show HEAD~1:src/auto_skill/metrics.py`.

**Verification:** §6 suite. Plus:
```bash
uv run python -c "from auto_skill.metrics import numeric_summary, token_usage_summary, skill_artifact_summary, parse_self_consistency_report, literal_leakage_report"
uv run python -c "from auto_skill.eval.modes import FULL_EVAL_MODES, MVP_EVAL_MODES, PRESENTBENCH_OFFICIAL_MODES, DEFAULT_BASELINE_MODE"
```

**Risk:** medium-high. `metrics.py` is consumed by 3+ summary scripts; symbol
parity must be perfect.

### C6 — `refactor(baselines): split mvp.py into baselines/ and prompts/ (no protocol yet)`

**Scope:** mechanical split of `mvp.py` (916 LOC) into per-baseline files plus
shared `prompts/`. **`if mode == ...` cascades are preserved** — this commit
only moves code, not control flow.

**Map of mvp.py → new home:**

| Symbol | New location |
|---|---|
| `PromptRunResult` | `baselines/_protocol.py` (created here, protocol class itself comes in C7) |
| `SkillInductionResult` | `baselines/_protocol.py` |
| `_first_call_model` | `baselines/_protocol.py` (private) |
| `user_examples_from_pack`, `format_user_example` | `baselines/_shared.py` (private to baselines) |
| `build_one_shot_skill_prompt` | `baselines/one_shot_skill_from_examples.py` |
| `build_leave_one_out_validation_prompt` | `baselines/feature_driven_with_validation.py` |
| `build_validation_aware_skill_merge_prompt` | `baselines/feature_driven_with_validation.py` |
| `build_heldout_generation_prompt` | `baselines/_shared.py` — keeps internal `if mode == ...` dispatch, calls into per-baseline builders |
| `build_task_first_feature_signature_prompt` | `baselines/feature_driven.py` |
| `build_current_evidence_inventory`, `build_evidence_scaffolding_plan_prompt`, `validate_evidence_plan` | `prompts/evidence.py` |
| `build_planned_operational_anchor_prompt`, `build_feature_signature_context`, `build_operational_anchor_context` | `prompts/operational_anchor.py` |
| `build_presentbench_layout_plan_prompt` | `baselines/layout_plan.py` |
| `build_judge_prompt`, `evaluation_criteria`, `extract_overall_score` | `prompts/judge.py` |
| `_append_bullets`, `_string_list`, `_candidate_rule_lines`, `_described_item_lines`, `_single_line`, `_humanize_key`, `_dedupe_preserve_order` | `prompts/_format.py` (private) |

**Also move:**
- `skill_induction.py` → `baselines/feature_driven.py` (facade at old path).
- `skill_induction_minimal.py` → `baselines/minimal.py` (facade at old path).

**`mvp.py` after this commit:** a facade re-exporting every name listed above.
The 5 symbols consumed by `run_skill_mvp.py` (`PromptRunResult`,
`build_heldout_generation_prompt`, `format_user_example`, `parse_json_object`,
`user_examples_from_pack`) must remain importable as
`from auto_skill.mvp import ...`.

**Note:** there is **no** `auto_skill/mvp/` package created — only the file
`mvp.py` is kept (as a facade). No collision.

**Verification:** §6 suite. Plus:
```bash
uv run python -c "from auto_skill.mvp import PromptRunResult, build_heldout_generation_prompt, format_user_example, parse_json_object, user_examples_from_pack"
uv run python -c "from auto_skill.baselines.one_shot_skill_from_examples import build_one_shot_skill_prompt"
uv run python -c "from auto_skill.prompts.judge import build_judge_prompt, extract_overall_score"
uv run python scripts/skills/run_skill_mvp.py --help
```

**Risk:** medium — 16 builder functions across 10+ files. Mechanical, but
private helpers (`_append_bullets` etc.) must move with the builders that use
them.

### C7 — `refactor(baselines): introduce Baseline protocol and REGISTRY`

**Scope:** this is the only semantic change. Wraps each baseline in the
protocol from §7 and replaces `if mode == ...` cascades in scripts/eval
runners with `REGISTRY[mode]` dispatch.

**Add:**

- `baselines/_protocol.py` — `Baseline` protocol + `BaselineSpec` dataclass
  (see §7).
- `baselines/__init__.py` — populate `REGISTRY: dict[str, Baseline]` from each
  baseline module's exported instance.

**Modify (call sites):**

- `scripts/skills/run_skill_mvp.py` — replace mode switch with
  `REGISTRY[mode].induce(...)`. Behavior must match exactly. Diff against
  previous run on the smoke fixtures.
- `scripts/eval/run_writingbench_official_eval.py` — replace
  `SKILL_REQUIRED_MODES` membership tests and prompt-building dispatch with
  `REGISTRY[mode].spec.requires_skill` / `REGISTRY[mode].build_heldout_prompt`.
- `scripts/eval/run_heldout_eval.py` — same.
- `scripts/eval/run_presentbench_official_judge.py` — same where relevant.

**Add tests (mandatory before commit):**

- `tests/test_baselines_registry.py` — one smoke test per baseline:
  ```python
  def test_<name>_build_prompt_smoke(self):
      baseline = REGISTRY["<name>"]
      pack = load_fixture("tests/fixtures/example_pack.minimal.json")
      prompt = baseline.build_heldout_prompt(pack, pack["heldout_tasks"][0], skill_row=None)
      self.assertIsInstance(prompt, str)
      self.assertGreater(len(prompt), 0)
  ```
- Two heavier tests covering `feature_driven_with_validation` and `ours_full`
  with mocked `llm` clients to exercise `induce()`.

**Verification:** §6 suite. Plus:
```bash
uv run python -m unittest tests.test_baselines_registry
```

Run an end-to-end smoke against fixtures:
```bash
uv run python scripts/skills/run_skill_mvp.py \
  --packs tests/fixtures/example_packs.smoke.jsonl \
  --out /tmp/skill_smoke.jsonl \
  --dry-run
```
Compare structural output against the pre-C7 baseline (capture a `--dry-run`
snapshot in C6 for diff).

**Risk:** high — this is the only commit reviewers should diff line-by-line.
Recommend running `codex review` (the user's tool, not codex-the-executor) on
this commit before declaring done.

### C8 — `refactor(src): remove compatibility facades`

**Scope:** delete all facade modules/packages, rewrite all import sites to use
the new canonical paths.

**Do not run this commit without explicit user approval.** No fixed two-week
waiting period is required for this research repo. The old "defer ≥ 2 weeks"
rule only applies when downstream external consumers need a compatibility
window. For this branch, C8 may run as soon as:

- C7 has passed the full §6 suite.
- C7 has passed review.
- Internal imports can be rewritten cleanly:
  `rg "from auto_skill\\.(old module names)" src scripts tests`.
- The user explicitly approves removing compatibility facades.

C8 must not add extra WritingBench/PresentBench adaptations. Those benchmark
surfaces are deprecated for the forward workstream, and the next experiments
should target the blog/reddit author-style cleaning and metrics pipeline.
Changing or preserving old benchmark behavior is not a C8 goal; avoid touching
that code unless an import rewrite is mechanically required by facade removal.

When executed:
- Use `ast-grep` or an equivalent structured rewrite for all internal imports
  from old flat paths to canonical package paths.
- Delete every compatibility facade file/package.
- Do not add new WritingBench/PresentBench runner behavior, source-scope
  filtering, readiness gates, or compatibility shims.
- State in docs only where necessary that new experiments should use the
  blog/reddit author-style cleaning and metrics pipeline.
- Avoid changing prompt text, schemas, CLI defaults, or benchmark behavior.
- Run §6 suite.
- Run canonical import smoke:
  `uv run python -c "import auto_skill; import auto_skill.baselines; import auto_skill.cleaning; import auto_skill.metrics; print('ok')"`.
- Run a stale-import scan and require no remaining internal facade imports.

### C9 — `refactor(legacy): remove deprecated WritingBench and PresentBench paths` (optional)

**Scope:** remove or archive legacy benchmark business code after the new
author-style pipeline is verified.

Run C9 only after:

- Blog/reddit cleaning artifacts have a validation command.
- The new target/reference/hard-negative metric has a regression test or
  fixture-level smoke.
- Oracle sanity-check artifacts are generated from examples rather than
  hand-written baselines.
- Default docs and validation commands have moved from WritingBench/PresentBench
  to the author-style pipeline.

When executed:

- Delete or move legacy scripts/modules/tests into an explicit archive location.
- Remove or rewrite references in `AGENTS.md`, `CLAUDE.md`, `README.md`, and
  repo docs.
- Replace old readiness commands with author-style validation commands.
- Run the updated full validation suite.

## 6. Verification suite (mandatory after every commit)

```bash
diff -q AGENTS.md CLAUDE.md
uv run ruff check .
uv run python -m unittest discover -s tests
uv run python scripts/data/validate_splits.py artifacts/splits/fewshot_splits.jsonl
uv run python scripts/data/audit_benchmark_flow.py
uv run python scripts/ops/report_experiment_readiness.py --profile mvp --expect-status ready
uv run python scripts/ops/report_experiment_readiness.py --profile full --expect-status not_ready
uv run python scripts/ops/validate_run_artifacts.py \
    --generated-outputs tests/fixtures/generated_outputs.valid.jsonl \
    --skills tests/fixtures/skill_rows.valid.jsonl \
    --eval tests/fixtures/eval_rows.valid.jsonl
```

If any command fails, **do not commit**. Diagnose and fix in the same
working tree before the commit lands.

## 7. Baseline protocol contract (C7)

```python
# src/auto_skill/baselines/_protocol.py
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from auto_skill.schemas.user_example import UserExample

@dataclass(frozen=True)
class BaselineSpec:
    name: str                          # exact mode string used in CLI flags and eval rows
    requires_skill: bool               # True if eval needs a skill_row produced by induce()
    requires_examples: bool            # True if heldout prompt embeds train examples
    benchmark_scope: tuple[str, ...]   # subset of ("writingbench", "presentbench")
    description: str

@runtime_checkable
class Baseline(Protocol):
    spec: BaselineSpec
    def induce(
        self,
        pack: dict[str, Any],
        *,
        llm: Any,
        options: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Return a skill_row dict, or None if this baseline does not produce a skill."""
    def build_heldout_prompt(
        self,
        pack: dict[str, Any],
        task: dict[str, Any],
        *,
        skill_row: dict[str, Any] | None,
    ) -> str: ...

@dataclass(frozen=True)
class PromptRunResult:
    """Moved from mvp.py."""
    prompt: str
    response: str
    model: str
    usage: dict[str, Any] | None
    model_calls: list[dict[str, Any]]

@dataclass(frozen=True)
class SkillInductionResult:
    """Moved from mvp.py."""
    skill_row: dict[str, Any] | None
    model_calls: list[dict[str, Any]]
```

**Initial REGISTRY contents (C7):**

```python
REGISTRY: dict[str, Baseline] = {b.spec.name: b for b in [
    PromptOnly(),
    FewShotExamplesOnly(),
    OneShotSkillFromExamples(),
    ExamplesPlusOneShotSkill(),
    FeatureDriven(),
    FeatureDrivenWithValidation(),
    ExamplesPlusFeatureSkill(),
    SlideConstrained(),
    LayoutPlan(),
    OursFull(),
    Minimal(),
]}
```

Each `Baseline` instance lives in its own module under `baselines/`. The
module exports a single class with class-level `spec` attribute.

## 8. Rollback policy

- Every commit must be safely `git revert`-able. If a commit fails the
  verification suite in §6 and cannot be fixed within one working session,
  revert immediately rather than leaving a broken HEAD.
- Facades are not removed until C8. Until then, any reverter of C2–C7 must
  also revert the facades introduced by that commit (they are part of the
  same commit, so this is automatic).
- C7 is the only commit that changes call sites in `scripts/`. If C7 is
  reverted, also revert the test additions in `tests/test_baselines_registry.py`
  from the same commit.

## 9. Open items (escalate to user; do not decide unilaterally)

1. **C5 split of `presentbench_eval.py`** — if no clean cut between
   surrogate-vs-official exists inside the file, keep as
   `eval/presentbench.py` and document the reason. Do not force a split.
2. **C4 residual functions** — if any function in `author_style_quality.py`
   does not fit `audit / metrics / report`, leave in `report.py` with a
   `# TODO refactor` and list in commit body. Do not invent a fourth
   category.
3. **`io/ledger.py`** — current code has no obvious "stage ledger" module;
   the ledger logic lives inside `scripts/skills/run_skill_mvp.py`. Skip
   `io/ledger.py` in C1; revisit only if a subsequent commit moves ledger
   code into the library (out of scope for this refactor).
4. **`baselines/ours_full.py` content** — the `auto_skill_ours_full` mode is
   currently orchestrated entirely in `scripts/skills/run_skill_mvp.py`. C6
   should leave the orchestration in the script; `baselines/ours_full.py`
   only owns its prompt builders. C7 may add a thin `induce()` that calls
   into the script-level functions if they have been factored into the
   library — otherwise mark `ours_full.induce` as raising `NotImplementedError`
   and keep the script-level orchestration unchanged.

## 10. Done criteria

The refactor is complete when:

- All commits C0–C7 are merged on the working branch.
- The §6 verification suite passes on the HEAD of that branch.
- `wc -l src/auto_skill/*.py src/auto_skill/**/*.py` shows no file over
  500 LOC except possibly `cleaning/author_style/quality/report.py` (which
  may exceed if the split in C4 left residual content there).
- `grep -rn "if mode ==" src/auto_skill scripts/ tests/ | grep -v __pycache__`
  returns only matches inside `baselines/_shared.py` (the internal dispatch
  in `build_heldout_generation_prompt`, which the protocol intentionally
  retains) or in `scripts/` paths that have an explicit `# TODO migrate to
  REGISTRY` annotation.
- `auto_skill` package imports cleanly from a fresh Python:
  `uv run python -c "import auto_skill; import auto_skill.baselines; import auto_skill.cleaning; import auto_skill.metrics; print('ok')"`.

C8 is explicitly not part of the done criteria.
