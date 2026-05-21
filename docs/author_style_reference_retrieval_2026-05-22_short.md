# Author-Style Reference Retrieval - Short Readout - 2026-05-22

## Bottom Line

No major drift in the source-derived author-style signal. GPT-5.5 can almost
always recover the same-author reference from one heldout target plus four
hard negatives.

| Source/run | Packs | Heldout rows | Success rows | Correct | Accuracy |
|---|---:|---:|---:|---:|---:|
| Blog scale100 | 30 | 120 | 120 | 119 | 99.17% |
| Blog strict freeze | 22 | 88 | 88 | 87 | 98.86% |
| Reddit relaxed160 | 158 | 158 | 158 | 157 | 99.37% |
| Reddit GPT-full | 72 | 72 | 72 | 72 | 100.00% |

## Interpretation

The oracle sanity check is strong: the target text is extracted from the source
heldout example, the reference is a source-provided train example from the same
hashed author/style id, and negatives are other-author hard negatives. This is
not a baseline-generation target.

Observed failures are sparse:

- Blog scale100: `blog_author_style_0017::heldout::2` selected `negative::4`.
- Blog strict freeze: `blog_author_style_0030::heldout::0` selected `negative::2`.
- Reddit relaxed160: `author_style_reddit_cross_topic_0517::heldout::0` selected `negative::2`.
- Reddit GPT-full: no failures.

## Hygiene

The metric output files were scanned for persisted private text fields:

- `rationale`: absent
- `raw_text`: absent
- `target_text`: absent
- `judge_parse_error`: zero rows in the final full runs

The code path now redacts model-controlled `rationale`, invalid candidate IDs,
failed-attempt selected IDs, and model-controlled `parse_error` strings before
writing JSONL rows.

## Evidence Paths

Result summaries:

- `runs/author_style/reference_retrieval_full_2026-05-22/blog_scale100.gpt55.summary.json`
- `runs/author_style/reference_retrieval_full_2026-05-22/blog_v5_strict.gpt55.summary.json`
- `runs/author_style/reference_retrieval_full_2026-05-22/reddit_relaxed160.gpt55.summary.json`
- `runs/author_style/reference_retrieval_full_2026-05-22/reddit_gpt_full.gpt55.summary.json`

Code/review commits:

- `f7d364f feat(metrics): add author style reference retrieval`
- `6a75bed fix(metrics): harden author style retrieval rows`
- `727f013 fix(metrics): redact invalid retrieval reports`
- `13a5743 fix(metrics): redact retrieval parse attempts`
- `3c7c3ad fix(metrics): normalize retrieval parse errors`
