# Benchmark Flow

Core claim:

> If users can only provide a few good examples, the system should infer a reusable skill from those examples and improve performance on new tasks judged by the original benchmark rubrics.

```mermaid
flowchart LR
    A["Source benchmarks<br/>WritingBench / PresentBench"] --> B["Select related task cluster<br/>same domain / style / requirement pattern"]

    B --> C["Train tasks<br/>used only to create few-shot examples"]
    B --> D["Heldout tasks<br/>used only for evaluation"]

    C --> E["Desired output generation<br/>visible task + materials only"]
    E --> F["Quality freeze checks<br/>schema / provenance / leakage guard"]
    F --> G["High-quality user-visible examples<br/>task + desired output/artifact"]

    G --> H["Few-shot example pack<br/>only user-visible content"]
    H --> I["Skill extraction<br/>derive reusable SKILL.md / templates / tests"]

    D --> J["Run heldout task without skill<br/>baseline output"]
    D --> K["Run heldout task with extracted skill<br/>skill-conditioned output"]

    J --> L["Original benchmark evaluator<br/>WritingBench rubric / PresentBench checklist"]
    K --> L

    L --> M["Score delta and error analysis<br/>quality lift, stability, transferability"]
    M --> N["Evidence that the solution works<br/>examples -> skill -> better heldout performance"]

    L -. offline analysis .-> O["Gap diagnosis<br/>debug aid only"]
```

## What This Proves

- The examples are constructed by us from benchmark tasks and visible materials.
- The generated few-shot examples simulate what a user can provide: task inputs, desired outputs/artifacts, materials, and optional user notes, not manually written skills.
- Rubrics, teacher critiques, and score traces are benchmark construction or evaluation machinery. They should be stripped before the auto-skill module sees the examples unless a named baseline explicitly exposes them.
- The extracted skill is evaluated on heldout tasks from the same benchmark distribution.
- The proof is not "we can recognize a reusable gap"; the proof is "examples can induce a reusable skill that improves benchmark-scored outputs."

## Minimal Experimental Unit

For each task cluster:

1. Sample `k` train tasks and `n` heldout tasks.
2. Use train tasks to generate solved few-shot examples; keep only user-visible content for induction.
3. Extract one transferable skill module from those examples.
4. Run heldout tasks with and without the extracted skill.
5. Compare rubric/checklist scores and inspect failure modes.
