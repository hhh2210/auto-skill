# Auto-skill vs One-shot Skill From Examples

## Core Question

How do we show that our auto-skill module is better than simply asking a strong model:

> Here are a few examples. Please write a reusable skill.

The key is to avoid making this a prompt-engineering contest. The advantage should come from observable mechanisms:

1. better decomposition of visible examples;
2. better abstraction;
3. better validation against withheld examples;
4. better transfer;
5. lower negative transfer;
6. lower inference cost after extraction.

## Why One-shot Is A Strong But Flawed Baseline

`one_shot_skill_from_examples` is strong because GPT-5.5 can summarize patterns from examples well. We should treat it seriously, not strawman it.

But it has predictable weaknesses:

- It tends to produce generic prose rules that sound right but are not operational.
- It may overfit visible examples and miss the hidden rubric dimensions.
- It rarely separates trigger conditions, invariants, exceptions, failure modes, and validation checks.
- It usually does not know whether a pattern is stable across examples or accidental.
- It often compresses examples into style advice, not a reusable execution procedure.
- It does not naturally test whether the generated skill improves heldout tasks.
- It has no built-in admission or rollback mechanism when a skill hurts.

## Current Experimental Read

The current smoke experiments do not support treating `Feature-Driven Auto-Skill`
or `LOO validation` as the main method claim yet. They should be treated as
components under mechanism ablation.

Observed pattern so far:

| Strategy | Current signal | Working interpretation |
| --- | --- | --- |
| `few_shot_examples_only` | strongest overall | examples still contain information not captured by current skill artifacts |
| `one_shot_skill_from_examples` | strongest skill baseline | one-shot skill drafting is currently the most stable compression baseline |
| `Feature-Driven Auto-Skill` / `ours_no_validation` | weaker than one-shot | feature extraction may over-abstract and drop concrete constraints or writing moves |
| `LOO validation` / `auto_skill_ours_full` | weaker and longer | current merge may dilute useful rules rather than repair them |

This means the next experiment should not be a larger benchmark sweep. It should
first answer where the failure happens:

1. skill compression versus examples-only context;
2. feature extraction versus direct one-shot drafting;
3. old LOO merge versus no validation versus majority-vote validation;
4. skill-only versus examples-plus-skill marginal value.

## Candidate Mechanisms To Test

### 1. Visible Example Decomposition

Instead of feeding examples as raw text and asking for a skill, split each example into:

- task intent;
- output structure;
- recurring moves that appear in multiple examples;
- example-specific details that must not be generalized.
- user-emphasized constraints when present;
- materials/artifact properties when present.

This should make the module better at distinguishing "generalizable skill" from "surface detail".

### 2. Cross-example Pattern Mining

One-shot often extracts whatever is salient in a single example. Our module should explicitly mine:

- recurring patterns across examples;
- patterns supported by multiple examples;
- patterns that stay stable across different task topics;
- patterns that appear in the final outputs/artifacts rather than task-specific content;
- patterns that should become optional rather than required because support is partial.

This lets us claim the skill is induced from stable evidence, not one-shot summarization.

### 3. Feature-grounded Skill Compilation

For WritingBench and PresentBench, the skill should not only say "write better". It should compile visible cross-example features into operational checks:

- coverage checks;
- factual grounding checks;
- style/tone checks;
- structure/layout checks;
- anti-hallucination checks;
- "before final answer, verify X" checklists.

This is only a real advantage if it beats direct one-shot skill drafting under
the same inputs. Current smoke results do not show that yet. The likely failure
mode is excessive abstraction: the feature schema focuses the model, but may
discard concrete output moves, phrasing constraints, and example-specific
structure that matter to the heldout evaluator.

### 4. Skill Schema And Separation Of Concerns

Our module can force a structured artifact:

- `when_to_use`;
- `task_family`;
- `core_procedure`;
- `rubric_alignment`;
- `templates`;
- `common_failures`;
- `do_not_generalize`;
- `self_check`;
- optional `references` or `scripts`.

One-shot baseline may be asked to output the same schema for fairness, but our module should fill it using separate extraction passes and validation, not a single generation pass.

### 5. Validation Loop Before Heldout Use

Auto-skill should self-evaluate the candidate skill on train/example tasks before using it on heldout:

- replay the solved examples using the generated skill;
- check whether it reproduces key rubric wins;
- prune unsupported rules;
- flag rules that only appear once;
- compress redundant instructions;
- add missing self-checks from recurring visible patterns.

This is a hypothesis, not an established result. The current LOO merge appears
to hurt more than help, probably because it merges away useful specificity and
turns sharp rules into longer, softer guidance. Keep validation as an ablation:
`feature_no_validation` versus `feature_with_old_LOO` versus
`feature_with_majority_vote_LOO`.

### 6. Negative Transfer Control

A major claim from SkillsBench/SkillFlow/ClawTrace is that skills can hurt. Our module should include:

- explicit applicability conditions;
- "do not apply when..." conditions;
- confidence score per rule;
- fallback to examples-only or prompt-only when evidence is weak;
- heldout negative-transfer rate as a metric.

This can be a signature advantage: not just higher average score, but fewer harmful skill applications.

### 7. Compression Advantage

Few-shot examples may be long. A good skill should compress them into a reusable artifact:

- lower input tokens on heldout tasks than examples-only;
- equal or better score than examples-only;
- more stable score under smaller context budgets.

This is an important practical selling point: examples teach the skill once; the skill is cheaper to reuse.

### 8. Transfer Across Nearby Subdomains

One-shot may work within the exact cluster. Auto-skill should transfer better when:

- same domain, different prompt wording;
- same writing style, different topic;
- same PPT checklist dimension, different material;
- same structural convention, different language;
- same rubric but changed content scale.

We should evaluate at two distances:

- near transfer: same domain2/category;
- medium transfer: same domain1/category family, different subtopic.

## Mechanism Ablation Matrix

The next run should be named mechanism ablation, not a larger headline
experiment. Run it first on the existing 4 WritingBench packs, using the same
candidate outputs where possible and evaluating with Qwen plus an independent
judge swap.

| Condition | What it tests |
| --- | --- |
| `prompt_only` | raw model ability |
| `few_shot_examples_only` | direct in-context learning from examples |
| `one_shot_skill_from_examples` | strongest current skill compression baseline |
| `feature_skill_no_validation` | whether feature extraction and aggregation help over one-shot |
| `feature_skill_old_LOO` | whether the current LOO merge repairs or damages the feature skill |
| `feature_skill_majority_LOO` | whether support-count / majority-vote validation helps |
| `examples_plus_one_shot_skill` | whether one-shot skill has marginal value beyond raw examples |
| `examples_plus_feature_skill` | whether feature skill has marginal value beyond raw examples |

The key comparisons are:

```text
few_shot_examples_only vs one_shot_skill_from_examples
one_shot_skill_from_examples vs feature_skill_no_validation
feature_skill_no_validation vs feature_skill_old_LOO vs feature_skill_majority_LOO
few_shot_examples_only vs examples_plus_one_shot_skill vs examples_plus_feature_skill
```

This must remain same-input: the skill induction side receives only user
examples, optional user emphasis, and materials that are part of the examples.
Rubric, critique trace, weak/strong diff, and heldout feedback remain
benchmark-private and can only appear in diagnostics or explicit oracle upper
bounds.

## Mechanism Questions

### 1. Does skill compression inherently lose information?

Compare `few_shot_examples_only` against `one_shot_skill_from_examples`. If
few-shot stays stronger, the contribution should become "cheaper skill
compression that approaches few-shot quality" rather than "skill always beats
examples".

### 2. Does feature-driven compilation beat direct one-shot drafting?

Compare `one_shot_skill_from_examples` against
`feature_extract + aggregate + compile`. If the latter is weaker, the feature
schema is over-abstracting or missing concrete example evidence.

### 3. Is LOO validation a repair mechanism or a damage mechanism?

Compare no validation, old LOO, and a stricter majority-vote LOO. Keep LOO only
if it improves heldout score, lowers negative transfer, or improves robustness
without making the skill materially longer and weaker.

### 4. Does the skill add marginal information to examples?

Compare `few_shot_examples_only` against `few_shot_examples + skill`. If adding
the skill does not improve score, the skill is redundant. If it lowers score,
the skill is actively interfering with example-following.

## Metrics That Reveal The Advantage

Do not only report average score. Report:

- overall heldout score;
- rubric subscore deltas;
- negative transfer rate;
- score variance across clusters;
- token cost at heldout inference;
- skill length / number of rules;
- rule support count: how many examples justify each rule;
- unsupported-rule rate judged by an LLM;
- applicability accuracy: whether the system knows when to use the skill;
- examples-only vs skill-only vs examples+skill.

## Expected Win Patterns

The auto-skill module is likely to win most clearly when:

- examples are long and heterogeneous;
- rubrics contain multiple hidden dimensions;
- weak-to-strong diffs reveal non-obvious improvements;
- the task requires stable style plus factual/structural constraints;
- direct examples cause context bloat;
- one-shot skill drafts become generic or overfit.

It may not win much when:

- the domain is simple;
- examples are already short and directly reusable;
- the rubric is shallow;
- one-shot generated skill is enough;
- heldout tasks are too close to train examples.

These cases are still useful. They define the boundary where auto-skill is overkill.

## Productized Auto-skill Module Shape

The module can be framed as a compiler:

```text
example pack
  -> evidence parser
  -> pattern miner
  -> rubric mapper
  -> skill planner
  -> skill writer
  -> train replay validator
  -> rule pruner / repairer
  -> final skill package
```

This lets us describe the contribution as "skill induction compiler from examples", not "a prompt that asks GPT-5.5 to write a skill".

## Strongest Claim To Aim For

The old target claim was:

> Compared with one-shot skill drafting from the same user examples, our auto-skill compiler produces skills that are more operational, more compact, and less likely to cause negative transfer, leading to better heldout performance under the original benchmark evaluators.

Current evidence does not justify this as a result claim. The near-term
defensible claim is narrower:

> We build a controlled harness for isolating where example-to-skill induction
> succeeds or fails, separating skill compression loss, feature abstraction loss,
> validation/merge effects, negative transfer, and marginal value over raw
> examples.

Only promote a full auto-skill method claim after mechanism ablation shows a
component that consistently improves over `one_shot_skill_from_examples` or
delivers a clear cost/robustness tradeoff against `few_shot_examples_only`.
