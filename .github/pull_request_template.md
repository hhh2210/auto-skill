<!--
Fill all three sections before requesting review. See CLAUDE.md / AGENTS.md
"Architecture-First Discipline" for what counts.
-->

## Summary

<!-- 1-3 bullet points: what does this PR do and why. -->

-

## Architecture decision

<!--
Required for any PR that adds a new file or > 20 lines to an existing file.
Skip only for typo / comment / rename-only changes (state that explicitly).

Template:
  - This change adds <X> to <src/auto_skill/path or scripts/path> because <reason>.
  - Reused helpers: <list of imported modules/functions> (or "none — first implementation").
  - New files over 300 lines: <list> (or "none").
  - Patch size: <N> insertions / <M> deletions.

If a file would exceed 300 lines, explain why (generated code, schema-only,
explicit grandfather) or split before merging.
-->

- Adds:
- Reused helpers:
- New files > 300 lines:
- Patch size:

## Test plan

<!-- Bulleted checklist of how you verified the change. -->

- [ ] `uv run ruff check .`
- [ ] `uv run python -m unittest discover -s tests`
- [ ] `diff -q AGENTS.md CLAUDE.md` (if either was touched)
- [ ]
