# auto-skill Collaboration Guide

`AGENTS.md` and `CLAUDE.md` must stay identical. When changing one, update the other in the same commit or PR.

Default discussion language is Chinese. Keep key technical terms in English when they are part of the method or code.

## Living Guide Maintenance

This file is for durable collaboration rules, architecture boundaries, and
validation commands. Do not use it as a running experiment log. Dated run
evidence belongs in `notes/` or `docs/`; `AGENTS.md` should only point to that
evidence when it changes an active rule or a required validation command.

When project assumptions change, update this file in the same PR as the code or
artifact change. Keep entries short and operational: what future agents must do,
what they must not do, and which command verifies the invariant.

## Canonical Research Framing

The project is **few-shot skill induction from user examples**.

Canonical module input:

- user examples;
- optional user natural-language request or emphasis;
- optional materials that are part of the examples.

Do not treat rubric/checklist, critique trace, weak/strong diff, or heldout feedback as normal auto-skill module input. Benchmark code may use rubrics/checklists only for private construction checks and heldout evaluation. Any future oracle or upper-bound experiment that uses private metadata must be named and isolated separately.

The first proof target is:

```text
user examples -> reusable skill package -> better heldout performance
```

Gap/admission diagnosis is secondary debug analysis, not the headline.

## Active Workstreams

### 1. Data Cleaning Scripts

Goal: turn benchmark sources into stable JSONL artifacts that collaborators can inspect and validate.

Expected surfaces:

- BlogAuthorship ingestion;
- Mendeley Reddit Cross-Topic Authorship ingestion;
- grouping by hashed author id, where one author id is one style cluster;
- train-example / heldout-task split generation from same-author outputs;
- hard negative examples from other authors/styles;
- schema validation and summary reports;
- no large benchmark data committed to git.

WritingBench and PresentBench are runnable legacy surfaces. Keep their existing
commands working for regression checks, but do not add new adaptation work there
unless explicitly asked. New benchmark work should target blog/reddit
author-style cleaning and metrics.

### 2. Auto-Skills Module

Goal: compile examples into a structured skill package.

Expected surfaces:

- feature extraction prompts or adapters;
- cross-example stable feature aggregation;
- conflict/outlier handling;
- leave-one-out validation design when enough examples exist;
- generated `SKILL.md`, optional templates, tests, and metadata;
- baselines against `few_shot_examples_only` and `one_shot_skill_from_examples`.

## Repository Layout

- `src/auto_skill/` — library: schemas, prompt builders, evaluation helpers, readiness logic.
- `scripts/` — thin CLI wrappers around the library; behavior lives in `src/`.
- `tests/` — `unittest` regression coverage for parsing, validation, aggregation, prompt builders.
- `artifacts/` — small curated JSONL inputs/outputs checked into git (splits, packs, private eval, extraction memory).
- `runs/` — generated experiment outputs (skill rows, eval rows, readiness reports). Large/ignored.
- `notes/`, `docs/` — research notes and stable repo docs.
- `data/` — local benchmark checkouts (`PresentBench_repo`, optional `WritingBench`); ignored.

## Data Flow

Active author-style flow: `clean_posts.jsonl` → `author_style_splits.jsonl` → `author_style_packs.jsonl` + `author_style_private_eval.jsonl` + `hard_negatives.jsonl` → `accepted_author_style_packs.jsonl` + `accepted_author_style_private_eval.jsonl` + `accepted_hard_negatives.jsonl` → author-style heldout/likeness metrics and reports. Private author ids, author hashes, reference outputs, topics, dates, and hard-negative labels must never reach skill-induction prompts.

Legacy WritingBench/PresentBench flow: `fewshot_splits.jsonl` → `example_packs.jsonl` + `example_private_eval.jsonl` + `example_generation_jobs.jsonl` → `generated_desired_outputs.jsonl` → `example_packs.v1.jsonl` → `skill_mvp.*.jsonl` (one_shot / feature_driven / ours_full) → WritingBench official + PresentBench official/surrogate eval rows → `experiment_readiness.json`. Private rubric/judge metadata only flows through `example_private_eval.jsonl` and the eval scripts; it must never reach the skill-induction prompts.

## Repository Workflow

- Large features use branch + PR.
- Branch names should be descriptive, for example `feature/data-cleaning-v1` or `feature/skill-induction-prompts`.
- Keep `main` usable. Do not push broken scripts or failing tests.
- Use `uv` for environment and command execution. Do not add a Makefile unless there is a concrete non-Python workflow that needs it.
- External API config lives in local `.env`; never commit real keys. `.env.example` documents the required Bailian/OpenAI-compatible variables.
- Desired-output generation supports `--config-prefix MIMO` / `--config-prefix QWEN`
  for building parallel example sets from separately configured OpenAI-compatible
  models in one `.env`.
- For new author-style runs, treat one stable hashed author id as one style
  cluster. Do not add a separate cluster-cleaning pass before the same-author
  examples are built. Cluster/tag summaries are reporting and stratification
  aids only.
- Blog/reddit author-style ingestion is minimal by default: drop schema-broken or
  too-short normalized text rows, then exact-dedupe by normalized text. Do not
  filter by word count, lexical diversity, stopword hits, lyrics/copyright
  substrings, or uppercase heuristics unless running an explicitly named strict
  diagnostic. Length and quality signals belong in metric stratification reports.
- For target-nearest author-style hard-negative probes, prefer the H100 vLLM
  embedding endpoint over local MPS once the SSH tunnel is stable. The observed
  good local client shape is OpenAI-compatible embeddings through
  `127.0.0.1:18001`, `--batch-size 256`, `--embedding-workers 4`,
  `--embedding-max-retries 5`, `--embedding-expected-dim 1024`, and
  `--max-candidate-chars 3000`. Always set `NO_PROXY=127.0.0.1,localhost` for
  tunnel calls, include backend + served model id + embedding dim in the cache
  namespace, and fail loud on dimension mismatches instead of mixing caches from
  local sentence-transformers and remote vLLM.
- Do not jump from the 8-pack legacy MVP directly to all benchmark cases. Prefer a
  stratified medium expansion first (for example `--max-writing-groups 30
  --max-present-groups 20`) and require split validation, benchmark-flow audit,
  and sampled example-quality audit before full cleaning.
- Generation sampling temperature must be explicit for production runs: use `--temperature` or `BAILIAN_TEMPERATURE`; when unset the provider/model default is used.
- API generation is sequential by default. For batch cleaning, use `--num-threads` or `BAILIAN_NUM_THREADS`; with the current high-RPM Bailian quota, start with 16 and use the printed timing/error rate to decide whether to increase toward 32.
- For long PresentBench jobs, prefer `--timeout-seconds 600 --max-retries 0` while estimating throughput; repeated 120s SDK retries hide the true per-job latency.
- For long sequential skill-induction runs, use `--stream` to consume Qwen responses
  incrementally. Add `--stream-log` only for debugging one sequential run; do not
  use stream logging in threaded batch cleaning because output will interleave.
- `run_skill_mvp.py` writes an append-only stage ledger next to `--out` by
  default. Use `--resume` after API failures or long-run interruptions so
  successful stage calls and completed skill rows are reused.
- `run_writingbench_official_eval.py` and `run_heldout_eval.py` checkpoint every
  evaluated row to `--out`. Use `--resume` on long benchmark runs so successful
  pack/task/mode cells are reused and non-success cells are retried.
- When resuming into an existing eval `--out`, keep the selected pack/task/mode
  scope at least as broad as the existing file. The WritingBench runner fails
  closed if a filtered resume would prune rows outside the selected cells; use a
  new output path for one-cell probes.
- Provider retries and parse retries are separate. `--max-retries` handles SDK
  transport/rate-limit/server errors; `--parse-max-attempts` handles complete
  model responses that are malformed JSON or invalid judge scores. The default
  parse retry budget is 3 for `run_skill_mvp.py`,
  `run_writingbench_official_eval.py`, `run_heldout_eval.py`,
  `audit_train_examples.py`, `run_grounding_eval.py`,
  `run_pattern_similarity_eval.py`, and `run_self_consistency_metric.py`.
- Do not globally enable Qwen thinking for skill induction. Default to
  `--no-enable-thinking`; only use `--thinking-stages ... --thinking-budget N`
  for targeted A/B runs on analysis or merge stages.
- Use `scripts/ops/run_mvp_pipeline.sh` as the thin Bash orchestrator for long MVP
  runs. It sequences skill induction, extraction-memory update, WritingBench
  eval, PresentBench surrogate eval, validation, timestamps, and logs while
  keeping the Python CLIs as the source of behavior.
- `run_writingbench_official_eval.py` and `run_heldout_eval.py` support
  cell-level `--num-threads`: each pack/task/mode cell runs independently, while
  generation -> judge remains sequential inside the cell and checkpoint writes
  stay in the main process. For judge-swap runs, use
  `--judge-enable-thinking/--no-judge-enable-thinking` and
  `--judge-thinking-budget` to control judge-side reasoning separately from the
  solver.
- `artifacts/memory/extraction_memory.v1.jsonl` is an ignored local output for
  public user-example-derived extraction lessons from successful skill rows. It
  must not include private rubrics, heldout feedback, judge traces, or benchmark
  scores.
- Use `scripts/README.md` as the source of truth for script ownership and
  lifecycle. New scripts need a documented category; one-off research utilities
  should be marked as archive candidates instead of silently becoming core.
- `scripts/metrics/summarize_mvp_metrics.py` is a reporting artifact, not a readiness
  gate. It must keep official benchmark scores separate from surrogate/debug
  scores, expose expected-cell coverage, and treat self-consistency as an
  eval-only skill encoding diagnostic rather than a main performance metric.
- Output-example pattern similarity is debug-only. Blind runs must not include
  candidate skill text; skill-aware runs must be named and treated as debug.
- Avoid model monoculture in paper-facing claims. Qwen-only data cleaning,
  induction, generation, and judging is acceptable for smoke/MVP iteration, but
  reported results should separate at least the judge model from the tested
  generation/induction model and should label Qwen-only results as same-model
  smoke evidence.
- Split judge from solver via `ChatCompletionConfig.from_env(prefix=...)` plus
  the eval runners' `--judge-config-prefix` flag (e.g. `JUDGE`). Only the
  prefixed `MODEL` env var is required; `BASE_URL`/`API_KEY` fall back to
  `BAILIAN_*`. Eval rows persist top-level `solver_model` and `judge_model`;
  `report_experiment_readiness.py` emits `model_monoculture` when every visible
  row collapses to a single model. For paper-facing claims, inspect
  `model_inventory.scored`: it excludes non-success eval rows such as
  `missing_score_artifact`, so placeholder official rows cannot falsely break a
  Qwen-only scored run. In that case the gate emits `scored_model_monoculture`.
  Metrics summaries keep `model_inventory` focused on heldout eval rows and put
  self-consistency or other diagnostic rows in `diagnostic_model_inventory`.
- Before pushing code, run `uv run ruff check .`,
  `uv run python -m unittest discover -s tests`,
  `uv run python scripts/data/validate_splits.py artifacts/splits/fewshot_splits.jsonl`,
  and the readiness gate below. A not-ready report is acceptable during
  prototyping only when it is explicit and expected, never silently green.
- Update `README.md`, `docs/`, or this file when project assumptions change.
- Keep generated large outputs under `data/`, `outputs/`, `runs/`, or `logs/`; these are ignored.

## Code Rules

- Prefer small, testable library functions under `src/auto_skill/`.
- Keep `scripts/` as thin CLI wrappers.
- Use explicit schemas and validation for JSONL records.
- Avoid hidden benchmark leakage: if a function consumes rubric/checklist/teacher traces, name it as benchmark/eval-only.
- Add or update tests for shared parsing, validation, aggregation, and prompt-building logic.
- Prefer static correctness checks over style-only churn. `uv run ruff check .`
  is the required Python lint gate; treat undefined names/imports, bug-prone
  constructs, invalid upgrades, and import graph issues as real blockers.
  Formatting-only rewrites should be scoped, intentional, and not mixed into
  behavioral changes.
- Keep experimental code composable. Prefer building small artifact generators
  that reuse existing cleaning, negative-selection, prompt, parser, and report
  modules over adding all-in-one research scripts.
- Treat the second run of a temporary experiment as the promotion point. If an
  experiment is rerun after the initial probe, move the reusable behavior into
  maintained code under `src/auto_skill/` plus a documented thin script, or
  explicitly archive the experiment and explain why it should not become part of
  the project surface.
- Long-running or API-backed scripts need `--dry-run`, `--resume` when outputs
  are append-only, deterministic `--seed` for sampling, explicit output schemas,
  and public/private artifact boundaries.
- File size is tiered and enforced by `tools/check_module_size.py`:
  `src/**` and `tools/**` use a 400-line soft limit; `scripts/**` (CLI
  orchestrators that aggregate library calls) use 700. A patch that adds
  roughly 500 lines to one file is an architecture smell regardless of tier.
  Existing over-limit files live in `GRANDFATHERED` and must shrink, not grow.
  Generated, schema-only, or lookup-table files are exempt — state the
  exception in the PR.
- Function complexity is bounded by ruff's `C901` rule with
  `max-complexity = 15` (configured in `pyproject.toml`). Pre-existing
  offenders are listed in `[tool.ruff.lint.per-file-ignores]` and should be
  refactored opportunistically; new code under any non-listed path must stay
  within budget.
- Do not duplicate model transport logic. Add provider-specific options to the
  shared client or transport wrapper, then expose them through thin CLIs.
- New probe or diagnostic scripts must be listed in `scripts/README.md` with an
  owner category and lifecycle status (`active diagnostic`, `debug-only`,
  `archive candidate`, etc.).

## Architecture-First Discipline

Before writing a new file, adding more than ~20 lines to an existing file, or
starting any "implementation" task, run a 60-second discovery first. The goal
is to write code that lives in the right place and reuses existing helpers,
not to draft a 600-line self-contained script that the size hook will later
reject.

The fastest path is the helper script:

```bash
./tools/where_does_it_live.sh <concept-or-function-keyword>
```

It prints the current `src/auto_skill/` package tree, every existing module
that already mentions `<concept>`, and the modules closest to their tier
limit (400 for `src/**` and `tools/**`, 700 for `scripts/**`; see
`tools/check_module_size.py`). If you skip the helper, run the equivalent
commands by hand:

1. `find src/auto_skill -maxdepth 3 -type d | grep -v __pycache__` — confirm
   the package layout. New behavior must land in an existing package unless a
   new one is justified.
2. `rg -l "<core concept>|<function name>" src/auto_skill` — confirm whether
   the concept or its helpers already exist. If yes: import; do not
   re-implement. The `style_similarity` / `length_ratio` / `jaccard` /
   `days_apart` family in `cleaning/author_style/pack_negatives.py` is a
   recurring trap.
3. `wc -l <target file>` and sibling files — confirm you are not pushing the
   file past its tier limit (400 for `src/**` and `tools/**`, 700 for
   `scripts/**`). If you are, split before adding.
4. State the architectural decision in 1-2 sentences (in your reply, the
   commit body, or the PR description): `X belongs in
   src/auto_skill/<path> because <reason>; reused helpers: <list>` (or
   `<none — first implementation>`).

Skip this only for typo fixes, comment-only edits, or in-place rename
refactors of existing code with no new module.

Scripts in `scripts/` are CLI shells. They must import library logic from
`src/auto_skill/`; reimplementing helpers inside a script is a process
violation, not a style preference. When unsure where something belongs,
propose the location first and wait for confirmation — do not start writing.

## Validation Policy

Do not run every historical command by default. Choose the smallest closed-loop
validation that covers the files touched, then broaden only when the change
crosses module boundaries or affects public artifacts.

Always run these before handing off a code change:

```bash
uv run ruff check <changed paths>
uv run python -m unittest discover -s tests
diff -q AGENTS.md CLAUDE.md
```

Use targeted checks by changed area:

| Changed area | Required validation |
| --- | --- |
| `src/auto_skill/cleaning/author_style/*` | Run related unit tests plus one blog smoke and one reddit smoke. Use `--audit-thresholds none` unless explicitly testing strict cleaning. |
| `src/auto_skill/probes/*` or `scripts/probes/*` | Run a tiny probe artifact build against existing smoke `clean_posts.jsonl`; do not start a full research run as validation. |
| `src/auto_skill/metrics/*` or `scripts/metrics/*` | Run the relevant metric tests and one fixture/smoke invocation with `--dry-run` or a tiny JSONL sample when supported. |
| `src/auto_skill/llm/*` or provider config | Run LLM config tests and import checks; only call external models when the behavior cannot be validated with fixtures. |
| `scripts/data/*`, schemas, or split builders | Run split/artifact validation on the smallest checked-in fixture or smoke artifact that exercises the changed contract. |
| readiness/reporting code | Run the specific readiness/reporting test plus one expected-ready and one expected-not-ready profile when fixtures exist. |
| docs only | Run `diff -q AGENTS.md CLAUDE.md` when either collaboration file changed; skip expensive benchmark checks. |

Author-style smoke commands:

```bash
uv run --with datasets python -m auto_skill.author_style_cleaning --source blog --max-rows 200 --authors 2 --train-posts 2 --heldout-posts 1 --negatives-per-heldout 1 --audit-thresholds none --out-dir /tmp/auto_skill_author_style_blog_smoke
uv run python -m auto_skill.author_style_cleaning --source mendeley-reddit --mendeley-path data/mendeley_reddit_cross_topic/Reddit_Cross-Topic-AV-Corpus_1000_users.zip --authors 2 --negatives-per-heldout 1 --audit-thresholds none --out-dir /tmp/auto_skill_author_style_reddit_smoke
```

Environment setup is not a normal validation step. Run `uv sync --extra dev`
only when dependencies changed or the environment is missing packages. Run
`uv sync --extra dev --extra presentbench` only for legacy PresentBench work.

WritingBench and PresentBench are legacy regression surfaces. Keep their existing
commands working while the artifacts exist, but run those gates only when touching
legacy code, release-readiness scripts, or a change that could affect legacy
artifact contracts.

`run_generation_jobs.py --dry-run` does not create generated outputs. Only run
`apply_generated_outputs.py` after a real generation file exists, and treat any
non-zero missing/rejected count as a failed data freeze.

For CI or smoke work where checked-in artifacts are intentionally incomplete, use
`report_experiment_readiness.py --profile full --expect-status not_ready` to
assert that the gate fails closed. Use `--profile smoke` for partial smoke
inspection. Full experiment readiness requires official score artifacts and
success rows; do not fabricate concatenated run files when repeatable input flags
can merge split artifacts.
