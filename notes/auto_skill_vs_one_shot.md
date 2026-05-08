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

## Mechanisms Our Auto-skill Module Can Add

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

This is a direct advantage over one-shot skill drafting if we show better heldout benchmark gains without exposing benchmark rubrics to the induction module.

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

This creates a concrete delta against one-shot: one-shot writes; auto-skill writes, tests, and repairs.

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

## Experimental Baselines

The fair baseline stack should include:

| Condition | What it tests |
| --- | --- |
| `prompt_only` | raw model ability |
| `few_shot_examples_only` | direct in-context learning from examples |
| `one_shot_skill_from_examples` | whether a strong model can summarize examples into a skill |
| `ours_no_validation` | ablation: structured extraction without validation |
| `ours_no_feature_schema` | ablation: validation without structured feature extraction |
| `ours_full` | full auto-skill module |

The most important comparison is:

```text
ours_full > one_shot_skill_from_examples
```

This must be same-input: both conditions receive only user examples. Rubric/critique/full-evidence variants can be diagnostic upper baselines, but they are not the main fair comparison.

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

The strongest defensible claim is:

> Compared with one-shot skill drafting from the same user examples, our auto-skill compiler produces skills that are more operational, more compact, and less likely to cause negative transfer, leading to better heldout performance under the original benchmark evaluators.

This is better than claiming universal superiority over all self-evolving agents.
