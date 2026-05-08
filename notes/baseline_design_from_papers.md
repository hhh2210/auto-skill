# Baseline Design From Skill Papers

## Question

Do we need more baselines in the benchmark comparison stage?

Short answer: yes, but the first prototype should add **diagnostic baseline conditions**, not a large set of external systems.

For example-driven skill induction, the critical comparison is not "our system vs every evolving-agent system". The critical comparison is:

> Given the same user examples, does extracting a reusable skill improve heldout performance beyond directly prompting with the examples?

## Paper Signals

| Paper | What it implies for our baseline design |
| --- | --- |
| [SkillsBench](https://arxiv.org/abs/2602.12670) | Must include `no_skill` and `self_generated_skill`. It reports curated skills help, but self-generated skills can be flat or negative, so "ask the model to write a skill" is a necessary baseline. |
| [SkillLearnBench](https://arxiv.org/abs/2604.20087) | Do not evaluate only final task score. Track skill artifact quality, whether the agent actually follows the skill, and final task outcome. One-shot, self-feedback, teacher-feedback, and skill-creator style methods are useful baseline families. |
| [SkillFlow](https://arxiv.org/abs/2604.17308) | More skills is not necessarily better. Compact, repaired skills outperform fragmented skill inflation. Track skill count/length and negative transfer, not just average score. |
| [Trace2Skill](https://arxiv.org/abs/2603.25158) | A strong method baseline can use trajectory pools, error/success analysts, and patch consolidation. It is powerful but requires generated trajectories and is heavier than our first prototype. |
| [EvoSkills](https://arxiv.org/abs/2604.01687) | A verifier-optimized skill generator is a strong upper baseline, but expensive and harness-dependent. Useful after we have stable task generation and evaluation. |
| [How Well Do Agentic Skills Work in the Wild](https://arxiv.org/abs/2604.04323) | Retrieval/refinement is a realistic baseline, but it only helps when initially retrieved skills are relevant. If no relevant skill exists, refinement behaves more like amplification than creation. |
| [AutoSkill](https://arxiv.org/abs/2603.01145) | Supports our framing: user interaction/examples/preferences can be lifted into explicit skills with add/merge/discard lifecycle decisions. |
| [ClawTrace](https://arxiv.org/abs/2604.23853) | Add cost and regression metrics. Skill distillation should not only preserve successful behavior; it should also prune waste and check transfer to unrelated tasks. |
| [SkillCraft](https://arxiv.org/abs/2603.00718) | Efficiency matters when skills compress repeated tool/workflow chains. Less central for WritingBench/PresentBench, but useful for token/cost metrics. |

## Recommended First Experiment Matrix

Run these first:

| Condition | Purpose |
| --- | --- |
| `prompt_only` | Base model on heldout task with no examples and no skill. |
| `few_shot_examples_only` | Tests whether raw user examples already solve the problem through in-context learning. This is the most important baseline. |
| `one_shot_skill_from_examples` | Ask the model to directly write a skill from the examples, then use it. This matches SkillsBench/EvoSkills self-generated skill baselines. |
| `ours_extracted_skill` | Our pipeline: user-visible examples -> structured skill module -> heldout execution. |
| `teacher_refine_on_heldout` | Oracle-ish upper bound. It should be reported separately because it uses test-time evaluation/refinement, not just learned skill transfer. |

Add these after the first loop works:

| Condition | Why later |
| --- | --- |
| `examples_plus_skill` | Helps diagnose whether the skill compresses the examples or only works when examples are still present. |
| `retrieved_existing_skill` | Realistic, but requires building/searching a skill library. |
| `retrieved_plus_query_refine` | Good baseline against "skills in the wild", but only meaningful once retrieval quality is measurable. |
| `trace2skill_style_consolidation` | Strong method baseline, but needs many trajectories and analyst runs. |
| `verifier_optimized_skill` | Strong upper baseline, but expensive and easy to conflate with test-time optimization. |

## Why Not Add Many System Baselines Immediately

Most skill/evolving-agent papers rely on assumptions that do not match our first WritingBench/PresentBench prototype:

- They often use deterministic verifiers, while our writing/PPT tasks use LLM or MLLM rubrics.
- They often assume execution trajectories, hidden tests, or an interactive sandbox.
- Many target tool/code agents; our first domains are open-ended writing and slide generation.
- Their costs are high because each baseline multiplies generation, refinement, and evaluation calls.
- Comparing full systems too early can obscure our core claim: examples can induce a reusable skill that improves heldout performance.

## Metrics We Should Track

For every condition:

- benchmark score on heldout tasks;
- score delta vs `prompt_only`;
- score delta vs `few_shot_examples_only`;
- variance across task clusters;
- negative transfer rate: heldout tasks where the skill hurts;
- token/cost/time;
- skill length and number of modules;
- skill usage/adherence judged from output or trajectory when available.

For skill artifacts:

- coverage of recurring example patterns;
- specificity vs transferability;
- executable/resource structure when applicable;
- whether the skill encodes task-specific answers or general procedure.

## Current Recommendation

The first paper table should be small and defensible:

1. `prompt_only`
2. `few_shot_examples_only`
3. `one_shot_skill_from_examples`
4. `ours_extracted_skill`
5. `teacher_refine_on_heldout` as an upper bound

This is enough to prove the core claim. External systems like Trace2Skill/EvoSkills should be later-stage baselines or ablations after our own evaluation pipeline is stable.
