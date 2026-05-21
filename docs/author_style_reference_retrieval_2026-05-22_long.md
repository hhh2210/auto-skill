# Author-Style Reference Retrieval Evidence Report - 2026-05-22

## Scope

This report records the current state of the blog/reddit author-style cleaning
artifacts and the new source-derived target/reference/hard-negative oracle
metric.

The purpose is narrow: verify whether there is a real same-author style signal
in the cleaned author-style benchmark packets, independent of any generated
baseline output. The metric asks an LLM judge to choose the candidate reference
whose reusable writing style is closest to a heldout target.

## Metric Contract

For each heldout task:

- `target`: `accepted_author_style_private_eval.jsonl`
  `heldout_private[].reference_output_private`.
- `reference`: `accepted_author_style_packs.jsonl`
  `train_examples[].desired_output.text`, source-provided from the same
  author/style id.
- `hard negatives`: `accepted_hard_negatives.jsonl`
  `public_negative_text`, selected from other author/style ids.
- `judge`: GPT-5.5 through the local Codex OAuth transport.
- `output rows`: no persisted target/reference/negative full text; only hashes,
  candidate ids, status fields, public parse metadata, and aggregate summaries.

This is an oracle/data-signal sanity metric. It is intentionally eval-only and
must not be used as normal auto-skill module input.

## Code Changes

Implemented files:

- `src/auto_skill/metrics/author_style_reference.py`
- `src/auto_skill/metrics/author_style_reference_report.py`
- `scripts/metrics/run_author_style_reference_retrieval.py`
- `tests/test_author_style_reference_metric.py`

Script entry:

```bash
uv run python scripts/metrics/run_author_style_reference_retrieval.py \
  --packs <accepted_author_style_packs.jsonl> \
  --private-eval <accepted_author_style_private_eval.jsonl> \
  --hard-negatives <accepted_hard_negatives.jsonl> \
  --backend codex-oauth \
  --model gpt-5.5 \
  --out <rows.jsonl> \
  --summary-out <summary.json> \
  --report-md <report.md>
```

Main commits:

| Commit | Purpose | Review result |
|---|---|---|
| `f7d364f` | Add reference retrieval metric and CLI | Review found leakage/counting issues |
| `6a75bed` | Redact rationale, fix hard-negative count, limit references to one, normalize `candidate <id>` | Review found parse-error report leak |
| `727f013` | Redact invalid `judge_report` fields on parse-error rows | Review found attempts leak |
| `13a5743` | Redact parse-error attempt selected IDs | Review found model-controlled parse_error leak |
| `3c7c3ad` | Normalize public parse_error to local whitelist codes | Review found no issues |

## Reviewer Loop

Every code step was committed before review. Review agents were explicitly told
to use `/Users/larry_1/.codex/skills/gstack__review/SKILL.md` and to review only
`HEAD^..HEAD`.

Important reviewer findings that changed the implementation:

- Do not persist free-form `judge_report.rationale`, because the judge saw
  private target/reference text and could echo it.
- Do not allow `--references-per-target > 1` while scoring only the first
  same-author candidate as correct.
- Do not count `same_author_reference` candidates as hard negatives.
- Do not persist invalid model-provided candidate ids on parse-error rows.
- Do not persist invalid model-provided selected ids inside
  `judge_parse_attempts`.
- Do not trust model-provided `parse_error`; public outputs now emit only local
  whitelist parse-error codes.

Final review of `3c7c3ad` reported no findings.

## Verification

Code checks run after the final metric hardening:

```bash
uv run ruff check .
uv run python -m unittest discover -s tests
diff -q AGENTS.md CLAUDE.md
```

Observed result:

- Ruff: all checks passed.
- Unit tests: 461 tests passed.
- `AGENTS.md` and `CLAUDE.md`: identical.

Targeted metric tests now cover:

- target comes from private source heldout reference;
- reference comes from same-author train example;
- negatives come from hard-negative rows;
- model `candidate <id>` prefix normalization;
- unknown candidate rejection;
- `references_per_target > 1` rejection;
- hard-negative counting;
- success-row rationale redaction;
- parse-error row invalid candidate redaction;
- model-controlled `parse_error` normalization.

## Cleaning Artifact Inventory

Existing accepted artifact counts:

| Run | Packs | Train examples | Heldout tasks | Private rows | Hard negatives |
|---|---:|---:|---:|---:|---:|
| `blog_v5_strict_freeze_2026-05-18` | 22 | 176 | 88 | 22 | 352 |
| `smoke_2026-05-18_v5_scale100` | 30 | 240 | 120 | 30 | 480 |
| `reddit_mendeley_gpt_full_2026-05-18` | 72 | 288 | 72 | 72 | 288 |
| `mendeley_relaxed160_resumed_2026-05-18` | 158 | 632 | 158 | 158 | 632 |

Interpretation:

- Blog strict freeze is smaller because it keeps only accepted packs with every
  heldout covered by at least four GPT-selected hard negatives.
- Blog scale100 is the broader blog run used for the main scaled readout here.
- Reddit GPT-full is the stricter GPT-audited reddit run.
- Reddit relaxed160 gives a larger reddit stress sample.

## Metric Runs

### Blog scale100

Input:

- Packs: `runs/author_style/smoke_2026-05-18_v5_scale100/accepted_author_style_packs.jsonl`
- Private eval: `runs/author_style/smoke_2026-05-18_v5_scale100/accepted_author_style_private_eval.jsonl`
- Hard negatives: `runs/author_style/smoke_2026-05-18_v5_scale100/accepted_hard_negatives.jsonl`

Command:

```bash
uv run python scripts/metrics/run_author_style_reference_retrieval.py \
  --packs runs/author_style/smoke_2026-05-18_v5_scale100/accepted_author_style_packs.jsonl \
  --private-eval runs/author_style/smoke_2026-05-18_v5_scale100/accepted_author_style_private_eval.jsonl \
  --hard-negatives runs/author_style/smoke_2026-05-18_v5_scale100/accepted_hard_negatives.jsonl \
  --backend codex-oauth \
  --model gpt-5.5 \
  --num-threads 4 \
  --max-tokens 768 \
  --parse-max-attempts 2 \
  --timeout-seconds 240 \
  --out runs/author_style/reference_retrieval_full_2026-05-22/blog_scale100.gpt55.jsonl \
  --summary-out runs/author_style/reference_retrieval_full_2026-05-22/blog_scale100.gpt55.summary.json \
  --report-md runs/author_style/reference_retrieval_full_2026-05-22/blog_scale100.gpt55.report.md
```

One transient `model_error` was retried with `--resume`; the final run has no
non-success rows.

Result:

| Metric | Value |
|---|---:|
| Expected rows | 120 |
| Success rows | 120 |
| Correct | 119 |
| Accuracy | 99.17% |
| Mean expected rank | 1.000 |
| Median expected rank | 1.000 |
| Status counts | `{"success": 120}` |

Failure:

- `blog_author_style_0017::heldout::2`
- Expected: `blog_author_style_0017::train::0`
- Selected: `negative::4`
- Expected rank: not returned in the ranked list.

### Blog strict freeze

Input:

- Packs: `runs/author_style/blog_v5_strict_freeze_2026-05-18/accepted_author_style_packs.jsonl`
- Private eval: `runs/author_style/blog_v5_strict_freeze_2026-05-18/accepted_author_style_private_eval.jsonl`
- Hard negatives: `runs/author_style/blog_v5_strict_freeze_2026-05-18/accepted_hard_negatives.jsonl`

Result:

| Metric | Value |
|---|---:|
| Expected rows | 88 |
| Success rows | 88 |
| Correct | 87 |
| Accuracy | 98.86% |
| Mean expected rank | 1.011 |
| Median expected rank | 1.000 |
| Status counts | `{"success": 88}` |

Failure:

- `blog_author_style_0030::heldout::0`
- Expected: `blog_author_style_0030::train::0`
- Selected: `negative::2`
- Expected rank: 2

### Reddit GPT-full

Input:

- Packs: `runs/author_style/reddit_mendeley_gpt_full_2026-05-18/accepted_author_style_packs.jsonl`
- Private eval: `runs/author_style/reddit_mendeley_gpt_full_2026-05-18/accepted_author_style_private_eval.jsonl`
- Hard negatives: `runs/author_style/reddit_mendeley_gpt_full_2026-05-18/accepted_hard_negatives.jsonl`

Result:

| Metric | Value |
|---|---:|
| Expected rows | 72 |
| Success rows | 72 |
| Correct | 72 |
| Accuracy | 100.00% |
| Mean expected rank | 1.000 |
| Median expected rank | 1.000 |
| Status counts | `{"success": 72}` |

Failures: none.

### Reddit relaxed160

Input:

- Packs: `runs/author_style/mendeley_relaxed160_resumed_2026-05-18/accepted_author_style_packs.jsonl`
- Private eval: `runs/author_style/mendeley_relaxed160_resumed_2026-05-18/accepted_author_style_private_eval.jsonl`
- Hard negatives: `runs/author_style/mendeley_relaxed160_resumed_2026-05-18/accepted_hard_negatives.jsonl`

Result:

| Metric | Value |
|---|---:|
| Expected rows | 158 |
| Success rows | 158 |
| Correct | 157 |
| Accuracy | 99.37% |
| Mean expected rank | 1.000 |
| Median expected rank | 1.000 |
| Status counts | `{"success": 158}` |

Failure:

- `author_style_reddit_cross_topic_0517::heldout::0`
- Expected: `author_style_reddit_cross_topic_0517::train::0`
- Selected: `negative::2`
- Expected rank: not returned in the ranked list.

## Aggregate Readout

Across the two larger runs:

| Group | Rows | Correct | Accuracy |
|---|---:|---:|---:|
| Blog scale100 | 120 | 119 | 99.17% |
| Reddit relaxed160 | 158 | 157 | 99.37% |
| Combined scaled readout | 278 | 276 | 99.28% |

Across all four runs:

| Group | Rows | Correct | Accuracy |
|---|---:|---:|---:|
| All evaluated rows | 438 | 435 | 99.32% |

The wrong cases are sparse and concentrated in three rows. There is no broad
source-level collapse, no parse instability in the final outputs, and no
evidence that the hard negatives are trivially weak: a few negatives can beat
the true same-author reference, but that is rare.

## Output Hygiene Scan

Final JSONL outputs scanned:

- `runs/author_style/reference_retrieval_full_2026-05-22/blog_scale100.gpt55.jsonl`
- `runs/author_style/reference_retrieval_full_2026-05-22/blog_v5_strict.gpt55.jsonl`
- `runs/author_style/reference_retrieval_full_2026-05-22/reddit_gpt_full.gpt55.jsonl`
- `runs/author_style/reference_retrieval_full_2026-05-22/reddit_relaxed160.gpt55.jsonl`

Scan result:

| File | Rows | Contains `rationale` | Contains `raw_text` | Contains `target_text` | Parse errors |
|---|---:|---:|---:|---:|---:|
| `blog_scale100.gpt55.jsonl` | 120 | no | no | no | 0 |
| `blog_v5_strict.gpt55.jsonl` | 88 | no | no | no | 0 |
| `reddit_gpt_full.gpt55.jsonl` | 72 | no | no | no | 0 |
| `reddit_relaxed160.gpt55.jsonl` | 158 | no | no | no | 0 |

The files still contain candidate ids, hashes, source task ids, judge model,
usage, status, selected id, expected id, and ranked candidate ids. They do not
persist the target/reference/negative full text.

## Drift Assessment

No major overnight-style drift is visible in this metric. The score is very
high on both sources:

- Blog is slightly below reddit but still above 98.8% on both blog views.
- Reddit is 100% on the stricter GPT-full run and 99.37% on the larger relaxed
  run.
- The oracle target is extracted from source/private heldout examples, so this
  is not inflated by baseline-generated target text.

The remaining failures are useful hard cases for later manual inspection:

- They are cases where a same-topic/time/style negative can beat the train
  same-author reference.
- They may indicate either a genuinely confusable author pair, a weak chosen
  train reference, or judge over-reliance on local topic/style markers.

## Current Limitations

- This is an LLM-judge oracle sanity check, not the final auto-skill heldout
  performance metric.
- The judge is GPT-5.5 for these runs. A cross-judge repeat with a separate
  model would reduce same-model concerns.
- The metric uses one same-author train reference per target. That is now
  intentionally enforced because the scoring contract is singular.
- Some accepted artifacts are older May 18 cleaning outputs, not freshly
  regenerated after every refactor commit. The code now documents and verifies
  that one hashed author id is one style cluster, and the active docs point new
  experiments to the blog/reddit author-style path.

## Recommended Next Checks

1. Rejudge the three wrong rows with an independent judge and inspect whether
   the negative is genuinely closer or the reference is weak.
2. Run the same reference-retrieval metric with a non-GPT judge on a sampled
   subset.
3. Add a compact report script if this metric will be repeated often; the
   current reports were assembled from generated summary JSON and row scans.
4. Decide whether blog should use strict freeze or scale100 as the canonical
   paper-facing artifact. The strict freeze is cleaner; scale100 gives broader
   coverage.
