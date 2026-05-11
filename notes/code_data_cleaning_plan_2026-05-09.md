# Code Data Collection And Cleaning Plan 2026-05-09

This note defines the first code-domain data line for auto-skill. The goal is
not to build a generic code benchmark. The goal is to extract repeated,
reviewer-visible examples from mature repositories so the system can induce
reusable coding skills and test them on temporally later heldout PR/task slices.

## Framing

Use merged GitHub PRs and review threads as example material:

```text
old code + reviewer comments / suggestion blocks + accepted final diff
  -> cleaned user-visible examples
  -> reusable code skill package
  -> later heldout PR/task performance
```

Normal skill-induction input may include only information a user could provide:
the local code context, PR title/body, public reviewer discussion, suggestion
blocks, and the final accepted patch. It must not include hidden CI logs,
maintainer-only notes, issue triage metadata, model-generated critique traces,
or heldout evaluation feedback.

## Candidate Repositories

Live metadata was checked with `gh` on 2026-05-09. `pullRequests.totalCount`
below is the current open PR count returned by `gh repo view`; the four-month
merged count uses GitHub Search over `closed:>=2026-01-09`.

| Priority | Repository | Language | Stars | Open PRs | Merged PRs since 2026-01-09 | Why useful |
| --- | --- | --- | ---: | ---: | ---: | --- |
| P0 | `huggingface/transformers` | Python | 160,409 | 1,308 | 1,177 | Best first source for model-family patterns, naming nits, docs/tests conventions, architecture inheritance, and repeated reviewer norms. |
| P0 | `vllm-project/vllm` | Python | 79,468 | 2,895 | 3,650 | High-volume LLM serving code; likely rich in performance, scheduler, backend, test, and compatibility review patterns. |
| P1 | `ggml-org/llama.cpp` | C++ | 109,153 | 977 | 1,399 | Strong systems/inference source; useful for portability, backend flags, quantization, and C/C++ style skills. |
| P1 | `vercel/next.js` | JavaScript | 139,336 | 1,766 | 1,622 | Frontend/framework codebase; useful for tests, routing conventions, docs, examples, and repo-architecture skills. |
| P1 | `pytorch/pytorch` | Python/C++ | 99,780 | 2,270 | not checked in this pass | Very rich but may be too large/noisy for first pass; sample only after P0 pipeline is stable. |
| P2 | `kubernetes/kubernetes` | Go | 122,147 | 878 | not checked in this pass | Good Go/system source, but reviewer conventions are project-specific and triage-heavy. |
| P2 | `scikit-learn/scikit-learn` | Python | 66,012 | 511 | not checked in this pass | Lower-volume, high-quality scientific Python review style; useful for tests/docs/API consistency after P0. |
| P2 | `langchain-ai/langchain` | Python | 136,215 | 201 | not checked in this pass | Relevant to agent/tooling domain, but lower open PR surface and possible fast API churn. |

First pass should use only `transformers` plus a small `vllm` contrast sample.
Adding too many repositories early will blur whether the induced skill is
repo-specific, domain-specific, or genuinely transferable.

## Unit Of Data

A cleaned example is a PR-thread slice, not an entire PR.

Required fields:

- `repo`, `pr_number`, `merged_at`, `base_commit`, `merge_commit`;
- `thread_id`, `thread_kind`, `primary_paths`, `path_cluster`;
- `before_context`: minimal file hunks around the reviewed code;
- `review_signal`: public reviewer comment, code-owner request, or suggestion
  block;
- `accepted_change`: final diff hunk that resolves the signal;
- `example_input`: user-visible task/context for skill induction;
- `example_output`: accepted patch or structured edit recipe;
- `leakage_boundary`: what was stripped and why;
- `quality_flags`: deterministic checks and manual/audit labels.

Thread kinds:

- `suggestion_block`: reviewer gave a GitHub suggestion and the final patch
  matches or semantically follows it. This is the strongest example type.
- `code_owner_review`: comment from a maintainer/code owner that leads to an
  accepted edit. Good for conventions and repo norms.
- `discussion_resolution`: at least three messages before convergence. Good for
  design or API-shape skills, but noisier.
- `mechanical_followup`: naming, formatting, docs link, import order, test
  naming. Keep only when the same rule repeats across multiple PRs.

## Cleaning Rules

### Inclusion

Keep a thread when all of these hold:

1. The PR is merged.
2. The reviewed file and final patch are both recoverable from public Git data.
3. The thread maps to a small accepted change: ideally one to five hunks and
   fewer than about 200 changed lines.
4. The reviewer signal is actionable and local enough to become a skill rule.
5. The same pattern appears in at least three candidate PRs or belongs to a
   predeclared path cluster.

### Exclusion

Drop a thread when any of these hold:

1. The accepted change depends on hidden CI, private benchmark results, security
   disclosure material, or maintainer-only context.
2. The thread is mostly social process, release management, flaky CI retry, or
   merge-conflict cleanup.
3. The patch is a large feature implementation with no isolatable review rule.
4. The review asks for one-off product judgment rather than repeatable code
   behavior.
5. The final diff cannot be attributed to the review thread with reasonable
   confidence.
6. The source license is incompatible with storing normalized snippets for
   research artifacts.

### Leakage Guard

Use temporal and artifact separation:

- Train examples: older PRs only, for example merged before 2026-03-09 in the
  first four-month window.
- Heldout tasks: later PR slices, for example merged from 2026-03-09 through
  2026-05-09.
- Skill induction must not see heldout final patches, heldout reviewer
  discussions, benchmark/eval labels, or any post-merge repair commits.
- Store raw GitHub payloads under ignored `data/` or `runs/`; commit only small
  manifests, schemas, summaries, and curated fixture rows.
- Hash raw comments and diffs so evaluation can prove that train and heldout
  slices do not share exact PR/thread/source hunks.

## Path Clusters For Transformers

Start with path clusters that naturally express repeated skills:

| Path cluster | Skill hypothesis |
| --- | --- |
| `src/transformers/models/**` | Model implementation inheritance, config/model/tokenizer naming, checkpoint conversion, generation hooks, docstring/API consistency. |
| `tests/models/**` | Test naming, slow/remote markers, common model test mixins, expected shape/precision checks. |
| `docs/source/**` | Documentation style, model cards, examples, admonitions, cross-links, copied code snippets. |
| `src/transformers/integrations/**` | External library integration patterns, optional dependency guards, import error messages, version checks. |
| `src/transformers/generation/**` | Generation logic, cache handling, logits processors, backward compatibility, regression tests. |
| `examples/**` | Runnable example conventions, CLI args, README consistency, dependency notes. |

For `vllm`, use a smaller contrast set:

| Path cluster | Skill hypothesis |
| --- | --- |
| `vllm/engine/**`, `vllm/core/**` | Scheduler/engine invariants and regression tests. |
| `vllm/model_executor/**` | Model loader and backend compatibility patterns. |
| `tests/**` | Parametrized tests, GPU/CPU marks, flaky or expensive test gating. |
| `docs/**`, `examples/**` | Serving examples and user-facing API docs. |

## First-Pass Quotas

Target a single reproducible pilot:

- `transformers`: collect 100 merged PR candidates from the last four months.
- Keep about 60 to 80 cleaned thread examples after filters.
- Ensure at least four path clusters, with no cluster above 35% of examples.
- Require at least 20 `suggestion_block` examples.
- Reserve the latest 20% by `merged_at` as heldout before any skill induction.
- Add `vllm`: 30 candidate PRs, 10 to 20 cleaned examples only as a contrast
  audit, not as mixed training data yet.

The pilot success bar is not score improvement yet. It is:

1. deterministic extraction works;
2. train/heldout temporal split is auditable;
3. examples are visibly user-facing and not benchmark-private;
4. at least three coherent skill clusters emerge without manual cherry-picking.

## Suggested Pipeline

1. `collect_pr_index`: query merged PRs by repo/date/path hints; persist raw PR
   metadata and review-thread references.
2. `collect_pr_payloads`: fetch public comments, review threads, commits, and
   patch files into ignored raw JSONL.
3. `slice_threads`: map comments/suggestion blocks to final diff hunks using
   file path, line range, commit position, and text similarity.
4. `classify_threads`: assign `thread_kind`, `path_cluster`, size metrics,
   leakage flags, and confidence.
5. `build_code_example_packs`: create train examples and heldout tasks with the
   same public/private boundary as WritingBench/PresentBench.
6. `audit_code_benchmark_flow`: validate temporal split, hash uniqueness,
   leakage stripping, path-cluster balance, and minimum example counts.

## Open Risks

- GitHub review-thread line positions drift after force-pushes; raw patch and
  final merge commit must be kept for reproducibility.
- Suggestion blocks can overfit to direct patch imitation. Treat them as the
  strongest examples, but evaluate separately from discussion-derived examples.
- Maintainer comments may encode repo-specific norms. That is acceptable for
  repo-specific skill induction, but cross-repo claims need separate experiments.
- Large repositories may contain generated files or vendored code. Exclude
  generated paths using repo-specific rules before sampling.
- License and attribution need a separate check before committing any normalized
  code snippets beyond tiny fixtures.
