# Contributing

This is a collaborative research repo. Keep the main branch usable and make assumptions explicit.

## Branch Policy

- Use branch + PR for large features, experiment harness changes, schema changes, and new baselines.
- Small doc/test fixes can be pushed directly only when they do not break the current workflow.
- Suggested branch prefixes:
  - `feature/`
  - `fix/`
  - `docs/`
  - `experiment/`

## Before Push

Run:

```bash
uv run python -m unittest discover -s tests
```

When touching split data or schemas, run:

```bash
uv run python scripts/data/validate_splits.py artifacts/splits/fewshot_splits.jsonl
```

When touching formatting-sensitive code, run:

```bash
uv run ruff check .
```

## PR Description Template

```markdown
## Summary
-

## Workstream
- Data cleaning / Auto-skill module / Evaluation / Docs / Tests

## Validation
- [ ] `uv run python -m unittest discover -s tests`
- [ ] `uv run python scripts/data/validate_splits.py artifacts/splits/fewshot_splits.jsonl` if data schemas or split artifacts changed
- [ ] `uv run ruff check .`
- [ ] Other:

## Leakage / Assumption Check
-
```

## Data Policy

- Do not commit full benchmark datasets.
- Keep large generated outputs in `data/`, `outputs/`, `runs/`, or `logs/`.
- Small JSONL samples under `artifacts/` are acceptable while prototyping, but they should remain inspectable and reproducible.
