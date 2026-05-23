#!/usr/bin/env bash
#
# 60-second architecture discovery. Run before writing new code so the result
# lives in the right place and reuses existing helpers.
#
# Usage:
#   ./tools/where_does_it_live.sh <concept-or-function-keyword>
#
# Prints:
#   1. The src/auto_skill/ package tree (where things can go).
#   2. Modules that already mention <concept> (so you do not re-implement).
#   3. Modules closest to the 300-line architecture-smell limit (so you do not
#      push them over).

set -euo pipefail

if [[ $# -lt 1 || -z "${1:-}" ]]; then
  cat >&2 <<'USAGE'
usage: tools/where_does_it_live.sh <concept-or-function-keyword>

Examples:
  tools/where_does_it_live.sh style_similarity
  tools/where_does_it_live.sh hard.negative
  tools/where_does_it_live.sh "ReferenceCandidate|reference_retrieval"
USAGE
  exit 2
fi

concept="$1"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

if [[ ! -d "src/auto_skill" ]]; then
  echo "error: must run from the repo root (no src/auto_skill found)" >&2
  exit 1
fi

echo "=== src/auto_skill/ package tree ==="
find src/auto_skill -maxdepth 3 -type d \
  | grep -v __pycache__ \
  | sort

echo
echo "=== existing implementations mentioning '${concept}' ==="
if command -v rg >/dev/null 2>&1; then
  if ! rg -l "${concept}" src/auto_skill 2>/dev/null; then
    echo "(no matches — concept appears to be new; double-check spelling)"
  fi
else
  if ! grep -rl -E "${concept}" src/auto_skill 2>/dev/null; then
    echo "(no matches — concept appears to be new; double-check spelling)"
  fi
fi

echo
echo "=== modules over 200 lines (closest to 300-line architecture-smell limit) ==="
find src/auto_skill scripts -name "*.py" -not -path "*__pycache__*" -print0 \
  | xargs -0 wc -l 2>/dev/null \
  | awk '$1 > 200 && $2 != "total"' \
  | sort -rn \
  | head -15

echo
cat <<'REMINDER'
=== next step ===
State the architectural decision in 1-2 sentences before writing code:
  "X belongs in src/auto_skill/<path> because <reason>;
   reused helpers: <list, or 'none — first implementation'>."

If unsure where it belongs: propose the location and wait for confirmation.
Do not start writing.
REMINDER
