# P4a — LOO merge diagnostic (one_shot vs ours_full skill_md)

## TL;DR

Across all 4 WritingBench packs, `ours_full` produces **longer, more hedged** skill documents than `one_shot` (25–52% more chars), yet **fewer required-rule bullets** in most packs. The LOO merge prompt instructs the model to move rules into optional or remove them whenever *any single* validator flags them — even when that flag is driven by a training example whose content is genuinely different from the other two, not by a universal contradiction. The result is an over-demoted, over-conditionalized skill that hedges where `one_shot` is direct, making it a weaker instruction signal at heldout time.

---

## Per-pack table

| pack | one_shot chars | ours_full chars | required rules (one_shot → ours_full) | optional/conditional rules (one_shot → ours_full) | hedge tokens\* (one_shot → ours_full) |
|------|---------------|----------------|--------------------------------------|--------------------------------------------------|--------------------------------------|
| Slogans_en | 4,186 | 5,615 (+34%) | 14 → 9 (-36%) | 6 → 9 (+50%) | 0 → 2 |
| Financial_Reports_en | 5,296 | 7,395 (+40%) | 8 → 7 (-13%) | 6 → 12 (+100%) | 1 → 1 |
| Investment_Analysis_zh | 5,747 | 6,969 (+21%) | 6 → 5 (-17%) | 6 → 8 (+33%) | 0 → 0\*\* |
| Tender_Document_zh | 5,413 | 8,208 (+52%) | 6 → 8 (+33%) | 7 → 20 (+186%) | 0 → 3 |

\* hedge tokens counted: "only if", "if and only if", "if explicitly", "unless explicitly", "if the source", "*Note*" / "*Constraint*" inline annotations.
\*\* Investment_Analysis has 7 occurrences of "if the source" and "unless explicitly" in ours_full vs 0 in one_shot; raw count is 0 only on the narrow patterns shown.

---

## Pack-level findings

### writingbench_Advertising_Marketing_Slogans_en

- **Length**: ours_full 34% longer (5,615 vs 4,186 chars). Required rules shrank from 14 to 9; optional rules grew from 6 to 9.
- **Root cause**: LOO entry 0 (validation example `train::0`) flagged 4 rules as contradicted because that example was a trendy restaurant campaign (FOMO/slang) while the n-1 candidate was trained on a child-safety pet-salon pair. That single example poisoned 4 required rules.
- **LOO contradiction pattern**: entry 0 = 4 contradictions, entry 1 = 2, entry 2 = 0. No rule is contradicted by all 3 validators — i.e., no rule is a *genuine universal outlier*. The merge should require majority contradictions, not a single vote.
- **Representative weakened rule**:
  - one_shot Required: `"Adhere strictly to word count limits if specified (e.g., '10-15 words')."`
  - ours_full demoted to Optional (absorbed into a conditional): `"Rhetorical Devices: Use alliteration, rhyme, juxtaposition, or internet slang *only* if they fit the requested tone and audience."` — the word-count rule simply disappears from Required.
- **Added hedge example**: ours_full Required now reads `"Do not default to a multi-component structure (Main Slogan, Sub-slogan, Tagline) unless the user explicitly asks for it."` — the word "unless" makes this a conditional constraint rather than a direct instruction.

### writingbench_Finance_Business_Financial_Reports_en

- **Length**: ours_full 40% longer. Required rules 8 → 7; optional rules 6 → 12 (doubled).
- **LOO contradiction pattern**: entries 0/1/2 each contribute 1–2 contradictions with zero overlap. Every flagged rule is unique to one validator, meaning all flags are format/scope disagreements specific to one example type (press release vs risk assessment).
- **Key weakening**: one_shot Required: `"Source Grounding: All analysis of causes ... must be derived strictly from the provided context."` In ours_full this becomes conditionalized: `"Causal Linking: Explicitly connect operational events (causes) to financial outcomes ... If only temporal sequence is available, use correlational language ... rather than asserting unstated causality."` — a hard rule is rewritten as a nuanced procedure step.
- **New "Balanced Coverage" rule (ours_full Required)**: `"Conditional Balanced Coverage: The report must reflect the reality of the source data ... Do not force a non-existent counter-narrative."` The word "conditional" is in the rule name itself — this is a hedge that overrides what should be a strict mandate.
- **Representative weakened rule**:
  - one_shot: `"Do not soften negative findings (e.g., fraud, cash flow crises); report them objectively as presented in the data."` (under Do not generalize, clear directive)
  - ours_full: replaced by `"Conditional Balanced Coverage"` which instructs that if the source is *purely positive*, you do not need to mention risks. This relaxes the original intent — the validator's press-release example was purely positive, so the merge added the escape hatch for all future uses.

### writingbench_Finance_Business_Investment_Analysis_zh

- **Length**: ours_full 21% longer. Required rules 6 → 5; optional rules 6 → 8.
- **LOO contradiction pattern**: entry 0 = 0 contradictions, entry 1 = 2, entry 2 = 0. Only one validator (example `train::1`, a fund manager analysis with hallucinated CSI 300 recommendations) produced contradictions.
- **Scope narrowing caused by single outlier**: ours_full adds a "No Generic Advice" rule (`"Do not provide generic financial advice (e.g., 'diversify your portfolio,' 'switch to weekly investing') unless the source text specifically recommends it"`). This is good content but was triggered by a single hallucination event in one candidate, not a consistent pattern.
- **Representative weakened rule**:
  - one_shot: `"Contextual Framing: If the text includes macroeconomic context, weave this into the analysis to explain company performance, provided the link is made in the text."` (Optional, clear).
  - ours_full: `"Macro-Economic Context: Integrate macro-economic context *only* if explicitly mentioned in the source text or if the source text provides the necessary data points to make the connection."` — added "or if the source text provides the necessary data points" with an inline `*only if*` — effectively weakening a useful heuristic into a strict gating condition, which may suppress correct macro framing at heldout time.
- **Missing rule**: one_shot Required: `"Scope Adherence: Only analyze the periods and metrics explicitly covered in the provided text. If the text lacks data for a requested metric, *state that the data is unavailable* rather than estimating."` — ours_full drops the explicit "state unavailability" directive, replacing it with a softer "Strict Data Provenance" that only forbids fabrication.

### writingbench_Finance_Business_Tender_Document_zh

- **Length**: ours_full 52% longer (8,208 vs 5,413 chars). Optional rules exploded from 7 to 20 (+186%). Required rules actually grew slightly (6 → 8) — but most new required rules are procedural steps, while the optional section became a sprawling 20-item list.
- **LOO contradiction pattern**: entries 0/1/2 contribute 1/4/3 contradictions. Entry 1 produced 4 contradictions (structure ordering, section granularity, security placement, input scope assumption). These are all format-level style differences between examples, not genuine rule violations.
- **Critical observation**: The tender examples differ in section count (6-chapter vs 7-chapter vs 8-section). Each LOO candidate codified one structural style; each validator flagged it as wrong. The merge responded by writing `"Do not enforce a rigid, fixed chapter count (e.g., exactly 6 chapters). Allow the structure to expand or contract"` — a *non-instruction* that removes a concrete structural rule.
- **Representative weakened rule**:
  - one_shot Required: `"Completeness: Even if the user only mentions 'Scope' and 'Budget', you must generate standard sections for Qualifications, Timeline, Acceptance, and After-sales based on industry norms for that sector."` — a strong proactive completeness mandate.
  - ours_full Do not generalize: `"Do not enforce a rigid, fixed chapter count (e.g., exactly 6 chapters). Allow the structure to expand or contract based on the complexity of the Scope of Work."` — the merge turned a completeness guarantee into a flexibility disclaimer.
- **Optional rule bloat**: ours_full Optional contains 7 bullets including: `"Evaluation Table: ... if the user provides scoring logic or if the domain strongly implies a comprehensive scoring method"`, `"Financial Specifics: If budget data is provided ..."`, `"Security Standards: ... if the domain implies high-security needs or if specific modules ... are requested"`, `"Tech Stack Specifics: ... only if explicitly requested or clearly implied"`, `"Standard Defaults: In the absence of specific user input, it is permissible to fill in standard boilerplate values ..."` — all are conditionalized with if/only-if. one_shot had 3 optional bullets, all actionable.

---

## Cross-pack pattern

The evidence most strongly supports **(a) + (d)** acting together:

**(a) Merge prompt demotes on a single vote.**
The merge prompt says: `"Move partially supported or context-specific rules to optional rules."` and `"Remove rules contradicted by multiple validation examples."` In practice, "partially supported" is interpreted broadly: a rule touched by even one validator's `contradicted_rules` list gets demoted. Across 4 packs, no rule was flagged by all 3 validators — the maximum overlap is 0 shared-rule contradictions. Every demotion was triggered by exactly 1 validator.

Totals across 4 packs:
- Slogans: 6 unique contradicted rules, 0 contradicted by >1 validator.
- Financial Reports: 5 unique contradicted rules, 0 by >1.
- Investment Analysis: 2 unique contradicted rules, 0 by >1.
- Tender Document: 8 unique contradicted rules, 0 by >1.

All 21 total contradicted-rule events are singletons. The merge prompt's safeguard (`"Remove rules contradicted by *multiple* validation examples"`) never triggers. Instead the softer `"Move partially supported ... to optional"` fires for all of them.

**(d) n=3 is too small for LOO.**
With 3 train examples, each LOO candidate is built from 2 examples. Two examples are not enough to represent a task domain's full variation; the third example is almost always stylistically or thematically distinct in some way. This means every LOO round produces at least one contradiction just from normal within-domain variation, not from a genuine rule violation. The slogans case is the clearest: example `train::0` (trendy restaurant) vs examples `train::1–2` (pet salon + toy brand) are genuinely different use cases within the same domain. The candidate built from `train::1+2` naturally prioritized safety/warmth; `train::0` naturally contradicted it. That is not a bug in the rule — it is the expected diversity of the task.

**(b) Verbose hedging is a symptom, not the primary cause.**
ours_full is 21–52% longer in all 4 packs and consistently adds conditional language ("only if", "if and only if", "if explicitly", "unless explicitly") where one_shot uses direct imperatives. However, this is a *downstream effect* of (a): the merge adds prose to explain the nuanced conditions under which a demoted rule still applies, producing longer optional sections rather than shorter required sections.

**(c) Feature extraction inconsistency is not evidenced here.**
The LOO contradiction content is coherent and rule-specific — validators are flagging real format/scope mismatches, not schema noise. (c) is not the primary cause, though it could become a factor with larger n.

---

## Recommendation

In order of likely ROI:

1. **Change merge vote threshold: require majority (≥2/3) contradictions to demote a rule, not a single vote.**
   Current wording `"Move partially supported or context-specific rules to optional rules"` should be tightened to: `"Only demote a Required rule to Optional if it appears in contradicted_rules in at least 2 out of n validation reports. A single contradiction is insufficient evidence at n=3."` This single change would have preserved the specific rules that were demoted in all 4 packs here.

2. **Distinguish structural/format contradictions from content contradictions during validation.**
   The validation prompt asks for `contradicted_rules` without distinguishing format-level (section ordering, chapter count) from content-level (factual accuracy, tone) contradictions. The Tender Document pack shows format disagreements (6 vs 7 vs 8 chapters) triggering demotion of content rules. Add a field `contradiction_type: "format" | "content" | "scope"` to the validation report, and only allow `content` contradictions to trigger demotion.

3. **Add an explicit "do not add conditional prose to explain demotions" instruction to the merge prompt.**
   The optional-section explosion (7 → 20 in Tender, 6 → 12 in Financial Reports) comes from the merge model hedging its bets by writing explanatory conditions for every demoted rule. Add to the merge prompt: `"When moving a rule to Optional, write it as a single actionable bullet. Do not add explanatory conditions or if/unless clauses beyond what is strictly necessary. Optional rules should be shorter than Required rules, not longer."`

4. **Pre-filter LOO candidates by example similarity before validation.**
   With n=3, one clearly off-distribution example will always corrupt one LOO round. Before validation, compute embedding similarity between the held-out example and the induction set mean. If similarity is below a threshold (e.g., cosine < 0.6), mark that LOO round as `low_confidence` and exclude its contradictions from the demote vote. This addresses root cause (d) without requiring larger n.

5. **Run the merge on the intermediate `skill_compilation` output (the "pre-LOO" skill) as the base, not on the 3 LOO candidates.**
   Currently the merge receives all 3 LOO candidates and merges them from scratch. The pre-LOO `skill_compilation` step already produces a reasonable skill from all n examples; the LOO merge should be a targeted *revision* pass (`"Here is the current skill. Here are the validation reports. Only weaken rules that fail the majority vote. Keep everything else."`) rather than a fresh synthesis. This would preserve the structure and specificity of the baseline skill.
