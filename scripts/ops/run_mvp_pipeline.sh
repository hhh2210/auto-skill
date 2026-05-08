#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/../.."

PACKS="${PACKS:-artifacts/packs/example_packs.v1.jsonl}"
SKILLS="${SKILLS:-runs/skill_mvp.qwen.mvp.jsonl}"
PRIVATE_EVAL="${PRIVATE_EVAL:-artifacts/private/example_private_eval.jsonl}"
WRITINGBENCH_ROOT="${WRITINGBENCH_ROOT:-../WritingBench}"
WRITINGBENCH_OUT="${WRITINGBENCH_OUT:-runs/writingbench_official_eval.qwen.mvp.jsonl}"
WRITINGBENCH_SUMMARY="${WRITINGBENCH_SUMMARY:-runs/writingbench_official_eval.qwen.mvp.summary.json}"
PRESENTBENCH_OUT="${PRESENTBENCH_OUT:-runs/presentbench_surrogate_eval.qwen.mvp.jsonl}"
PRESENTBENCH_SUMMARY="${PRESENTBENCH_SUMMARY:-runs/presentbench_surrogate_eval.qwen.mvp.summary.json}"
EXTRACTION_MEMORY_OUT="${EXTRACTION_MEMORY_OUT:-artifacts/memory/extraction_memory.v1.jsonl}"
LOG_DIR="${LOG_DIR:-logs/mvp}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"

RUN_SKILL_MVP="${RUN_SKILL_MVP:-1}"
RUN_MEMORY="${RUN_MEMORY:-1}"
RUN_WRITINGBENCH="${RUN_WRITINGBENCH:-1}"
RUN_PRESENTBENCH="${RUN_PRESENTBENCH:-1}"
RUN_VALIDATE="${RUN_VALIDATE:-1}"

MODES="${MODES:-prompt_only,few_shot_examples_only,one_shot_skill_from_examples,ours_no_validation}"
LIMIT_HELDOUT="${LIMIT_HELDOUT:-1}"
TEMPERATURE="${TEMPERATURE:-0.2}"
JSON_TEMPERATURE="${JSON_TEMPERATURE:-0}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
MAX_RETRIES="${MAX_RETRIES:-0}"
SKILL_MAX_TOKENS="${SKILL_MAX_TOKENS:-8192}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-8192}"
PRESENTBENCH_NUM_THREADS="${PRESENTBENCH_NUM_THREADS:-4}"

mkdir -p "$LOG_DIR"

run_step() {
  local name="$1"
  shift
  local log="$LOG_DIR/$STAMP.$name.log"
  echo
  echo "== $name =="
  echo "+ $*"
  /usr/bin/time -p "$@" 2>&1 | tee "$log"
}

if [[ "$RUN_SKILL_MVP" == "1" ]]; then
  run_step skill_mvp \
    uv run python scripts/skills/run_skill_mvp.py \
      --packs "$PACKS" \
      --out "$SKILLS" \
      --stream \
      --resume \
      --allow-partial \
      --no-enable-thinking \
      --no-leave-one-out \
      --temperature "$TEMPERATURE" \
      --json-temperature "$JSON_TEMPERATURE" \
      --timeout-seconds "$TIMEOUT_SECONDS" \
      --max-retries "$MAX_RETRIES" \
      --max-tokens "$SKILL_MAX_TOKENS"
fi

if [[ "$RUN_MEMORY" == "1" ]]; then
  run_step extraction_memory \
    uv run python scripts/skills/update_extraction_memory.py \
      --skills "$SKILLS" \
      --out "$EXTRACTION_MEMORY_OUT" \
      --append
fi

if [[ "$RUN_WRITINGBENCH" == "1" ]]; then
  run_step writingbench_eval \
    uv run python scripts/eval/run_writingbench_official_eval.py \
      --packs "$PACKS" \
      --skills "$SKILLS" \
      --private-eval "$PRIVATE_EVAL" \
      --writingbench-root "$WRITINGBENCH_ROOT" \
      --limit-heldout "$LIMIT_HELDOUT" \
      --modes "$MODES" \
      --stream \
      --no-enable-thinking \
      --timeout-seconds "$TIMEOUT_SECONDS" \
      --max-retries "$MAX_RETRIES" \
      --max-tokens "$EVAL_MAX_TOKENS" \
      --resume \
      --allow-partial \
      --out "$WRITINGBENCH_OUT" \
      --summary-out "$WRITINGBENCH_SUMMARY"
fi

if [[ "$RUN_PRESENTBENCH" == "1" ]]; then
  run_step presentbench_surrogate_eval \
    uv run python scripts/eval/run_heldout_eval.py \
      --packs "$PACKS" \
      --skills "$SKILLS" \
      --private-eval "$PRIVATE_EVAL" \
      --source PresentBench \
      --limit-heldout "$LIMIT_HELDOUT" \
      --modes "$MODES" \
      --stream \
      --no-enable-thinking \
      --timeout-seconds "$TIMEOUT_SECONDS" \
      --max-retries "$MAX_RETRIES" \
      --max-tokens "$EVAL_MAX_TOKENS" \
      --num-threads "$PRESENTBENCH_NUM_THREADS" \
      --resume \
      --allow-partial \
      --out "$PRESENTBENCH_OUT" \
      --summary-out "$PRESENTBENCH_SUMMARY"
fi

if [[ "$RUN_VALIDATE" == "1" ]]; then
  if [[ "$RUN_WRITINGBENCH" == "1" ]]; then
    run_step validate_writingbench \
      uv run python scripts/ops/validate_run_artifacts.py --eval "$WRITINGBENCH_OUT"
  fi
  if [[ "$RUN_PRESENTBENCH" == "1" ]]; then
    run_step validate_presentbench \
      uv run python scripts/ops/validate_run_artifacts.py --eval "$PRESENTBENCH_OUT"
  fi
fi

echo
echo "MVP pipeline finished. Logs: $LOG_DIR/$STAMP.*.log"
