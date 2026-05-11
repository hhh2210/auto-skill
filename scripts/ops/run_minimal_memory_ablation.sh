#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/../.."

PACKS="${PACKS:-runs/expanded/example_packs.30wb.only.jsonl}"
PRIVATE_EVAL="${PRIVATE_EVAL:-runs/expanded/example_private_eval.30wb_20pb.jsonl}"
WRITINGBENCH_ROOT="${WRITINGBENCH_ROOT:-../WritingBench}"
BASELINE_SKILLS="${BASELINE_SKILLS:-runs/expanded/skill_mvp.one_shot.30wb.jsonl}"

VARIANT="${VARIANT:-no_memory}"
OUT_PREFIX="${OUT_PREFIX:-runs/expanded/skill_minimal.30wb.$VARIANT}"
SHARD_DIR="${SHARD_DIR:-runs/expanded/minimal_memory_shards}"
SHARDS="${SHARDS:-6}"
SHARD_JOBS="${SHARD_JOBS:-2}"
FEATURE_SHARD_DIR="${FEATURE_SHARD_DIR:-runs/expanded/feature_loo_shards}"
FEATURE_SHARDS="${FEATURE_SHARDS:-6}"
FEATURE_JOBS="${FEATURE_JOBS:-6}"

RUN_INDUCTION="${RUN_INDUCTION:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_FEATURE_INDUCTION="${RUN_FEATURE_INDUCTION:-0}"
RUN_FEATURE_EVAL="${RUN_FEATURE_EVAL:-0}"
RUN_PAIRWISE="${RUN_PAIRWISE:-0}"
RUN_SKILL_QUALITY="${RUN_SKILL_QUALITY:-0}"
RUN_REPORT="${RUN_REPORT:-0}"

NO_MEMORY="${NO_MEMORY:-0}"
MEMORY="${MEMORY:-}"
MEMORY_SCOPE="${MEMORY_SCOPE:-within_pack}"
MEMORY_TOP_K="${MEMORY_TOP_K:-}"

SOLVER_CONFIG_PREFIX="${SOLVER_CONFIG_PREFIX:-BAILIAN}"
SUPERVISOR_CONFIG_PREFIX="${SUPERVISOR_CONFIG_PREFIX:-MIMO}"
JUDGE_CONFIG_PREFIX="${JUDGE_CONFIG_PREFIX:-MIMO}"
TEMPERATURE="${TEMPERATURE:-0.2}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-8192}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
SDK_MAX_RETRIES="${SDK_MAX_RETRIES:-2}"
PARSE_MAX_ATTEMPTS="${PARSE_MAX_ATTEMPTS:-3}"
EVAL_NUM_THREADS="${EVAL_NUM_THREADS:-40}"

MODES="${MODES:-ours_no_validation}"
PAIRWISE_NUM_THREADS="${PAIRWISE_NUM_THREADS:-40}"
PAIRWISE_MAX_TOKENS="${PAIRWISE_MAX_TOKENS:-4096}"
PAIRWISE_ANCHOR_MODE="${PAIRWISE_ANCHOR_MODE:-example_only}"
PAIRWISE_CANDIDATE_MODES="${PAIRWISE_CANDIDATE_MODES:-prompt_only,one_shot_skill,minimal_no_memory,minimal_mem_within,minimal_mem_xpack_holdout,feature_loo_full}"
SKILL_QUALITY_NUM_THREADS="${SKILL_QUALITY_NUM_THREADS:-40}"
SKILL_QUALITY_MAX_TOKENS="${SKILL_QUALITY_MAX_TOKENS:-4096}"
SKILL_QUALITY_MODES="${SKILL_QUALITY_MODES:-one_shot_skill,minimal_no_memory,minimal_mem_within,minimal_mem_xpack_holdout,feature_loo_full}"

export BAILIAN_TIMEOUT_SECONDS="${BAILIAN_TIMEOUT_SECONDS:-$TIMEOUT_SECONDS}"
export MIMO_TIMEOUT_SECONDS="${MIMO_TIMEOUT_SECONDS:-$TIMEOUT_SECONDS}"
export BAILIAN_MAX_RETRIES="${BAILIAN_MAX_RETRIES:-$SDK_MAX_RETRIES}"
export MIMO_MAX_RETRIES="${MIMO_MAX_RETRIES:-$SDK_MAX_RETRIES}"

SKILLS_OUT="${OUT_PREFIX}.jsonl"
ALIAS_OUT="${OUT_PREFIX}.alias_for_eval.jsonl"
EVAL_OUT="${OUT_PREFIX}.mimo_judge_8192.eval.jsonl"
SUMMARY_OUT="${OUT_PREFIX}.mimo_judge_8192.summary.json"

TASK_EVAL_BASE="${TASK_EVAL_BASE:-runs/expanded/skill_minimal.30wb.no_memory.mimo_judge_8192.eval.jsonl}"
TASK_EVAL_WITHIN="${TASK_EVAL_WITHIN:-runs/expanded/skill_minimal.30wb.mem_within.mimo_judge_8192.eval.jsonl}"
TASK_EVAL_XPACK="${TASK_EVAL_XPACK:-runs/expanded/skill_minimal.30wb.mem_xpack_holdout.mimo_judge_8192.eval.jsonl}"
SKILLS_NO_MEMORY="${SKILLS_NO_MEMORY:-runs/expanded/skill_minimal.30wb.no_memory.jsonl}"
SKILLS_WITHIN="${SKILLS_WITHIN:-runs/expanded/skill_minimal.30wb.mem_within.jsonl}"
SKILLS_XPACK="${SKILLS_XPACK:-runs/expanded/skill_minimal.30wb.mem_xpack_holdout.jsonl}"

FEATURE_SKILLS="${FEATURE_SKILLS:-runs/expanded/skill_mvp.feature_loo_full.30wb.jsonl}"
FEATURE_EVAL="${FEATURE_EVAL:-runs/expanded/writingbench_eval.feature_loo_full.30wb.mimo_judge_8192.jsonl}"
FEATURE_SUMMARY="${FEATURE_SUMMARY:-runs/expanded/writingbench_eval.feature_loo_full.30wb.mimo_judge_8192.summary.json}"

PAIRWISE_INPUT="${PAIRWISE_INPUT:-runs/expanded/minimal_memory_ablation.30wb.pairwise_input.jsonl}"
PAIRWISE_OUT="${PAIRWISE_OUT:-runs/expanded/minimal_memory_ablation.30wb.pairwise_likeness.mimo_judge.jsonl}"
PAIRWISE_SUMMARY="${PAIRWISE_SUMMARY:-runs/expanded/minimal_memory_ablation.30wb.pairwise_likeness.mimo_judge.summary.json}"
PAIRWISE_REPORT="${PAIRWISE_REPORT:-runs/expanded/minimal_memory_ablation.30wb.pairwise_likeness.mimo_judge.md}"
SKILL_QUALITY_INPUT="${SKILL_QUALITY_INPUT:-runs/expanded/minimal_memory_ablation.30wb.skill_quality_input.jsonl}"
SKILL_QUALITY_OUT="${SKILL_QUALITY_OUT:-runs/expanded/minimal_memory_ablation.30wb.skill_quality.mimo_judge.jsonl}"
SKILL_QUALITY_SUMMARY="${SKILL_QUALITY_SUMMARY:-runs/expanded/minimal_memory_ablation.30wb.skill_quality.mimo_judge.summary.json}"
REPORT_OUT="${REPORT_OUT:-runs/expanded/minimal_memory_ablation.30wb.mimo_judge_8192.report.md}"

mkdir -p "$SHARD_DIR" "$FEATURE_SHARD_DIR" "$(dirname "$OUT_PREFIX")"

validate_variant_config() {
  if [[ "$NO_MEMORY" == "1" && "$VARIANT" != "no_memory" ]]; then
    echo "error: NO_MEMORY=1 requires VARIANT=no_memory, got VARIANT=$VARIANT" >&2
    return 2
  fi
  case "$VARIANT" in
    no_memory)
      if [[ "$NO_MEMORY" != "1" && -n "$MEMORY" ]]; then
        echo "error: VARIANT=no_memory cannot be combined with MEMORY=$MEMORY" >&2
        return 2
      fi
      ;;
    mem_within)
      if [[ -z "$MEMORY" || "$MEMORY_SCOPE" != "within_pack" ]]; then
        echo "error: VARIANT=mem_within requires MEMORY and MEMORY_SCOPE=within_pack" >&2
        return 2
      fi
      ;;
    mem_xpack_holdout)
      if [[ -z "$MEMORY" || "$MEMORY_SCOPE" != "cross_pack_holdout" ]]; then
        echo "error: VARIANT=mem_xpack_holdout requires MEMORY and MEMORY_SCOPE=cross_pack_holdout" >&2
        return 2
      fi
      ;;
    mem_*)
      echo "error: unknown memory variant $VARIANT" >&2
      return 2
      ;;
    *)
      echo "error: unknown VARIANT=$VARIANT; expected no_memory, mem_within, or mem_xpack_holdout" >&2
      return 2
      ;;
  esac
  if [[ -n "$MEMORY" && ! -f "$MEMORY" ]]; then
    echo "error: MEMORY file does not exist: $MEMORY" >&2
    return 2
  fi
}

validate_positive_int() {
  local name="$1" value="$2"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: $name must be a positive integer, got $value" >&2
    return 2
  fi
}

validate_variant_config
validate_positive_int SHARDS "$SHARDS"
validate_positive_int SHARD_JOBS "$SHARD_JOBS"
validate_positive_int FEATURE_SHARDS "$FEATURE_SHARDS"
validate_positive_int FEATURE_JOBS "$FEATURE_JOBS"

minimal_memory_args=()
if [[ "$NO_MEMORY" == "1" || -z "$MEMORY" ]]; then
  minimal_memory_args+=(--no-memory)
else
  minimal_memory_args+=(--memory "$MEMORY" --memory-scope "$MEMORY_SCOPE")
  [[ -z "$MEMORY_TOP_K" ]] || minimal_memory_args+=(--memory-top-k "$MEMORY_TOP_K")
fi

uvp() {
  uv run python "$@"
}

make_shards() {
  uvp scripts/ops/shard_jsonl.py --input "$PACKS" --out-dir "$SHARD_DIR" --prefix "$VARIANT" --shards "$SHARDS"
}

run_minimal_shard() {
  local index="$1"
  uvp scripts/skills/run_skill_minimal.py \
    --packs "$SHARD_DIR/$VARIANT.shard$index.packs.jsonl" \
    --out "$OUT_PREFIX.shard$index.jsonl" \
    "${minimal_memory_args[@]}" \
    --resume --allow-partial \
    --solver-config-prefix "$SOLVER_CONFIG_PREFIX" \
    --supervisor-config-prefix "$SUPERVISOR_CONFIG_PREFIX" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    --parse-max-attempts "$PARSE_MAX_ATTEMPTS" \
    --no-enable-thinking
}

combine_minimal_shards() {
  uvp scripts/ops/combine_skill_shards.py \
    --packs "$PACKS" \
    --out "$SKILLS_OUT" \
    --prefix "$OUT_PREFIX" \
    --shards "$SHARDS"
}

run_minimal_induction() {
  make_shards
  local pids=()
  local failed=0
  for index in $(seq 0 $((SHARDS - 1))); do
    run_minimal_shard "$index" &
    pids+=("$!")
    if [[ "${#pids[@]}" -ge "$SHARD_JOBS" ]]; then
      for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
      done
      pids=()
    fi
  done
  if [[ "${#pids[@]}" -gt 0 ]]; then
    for pid in "${pids[@]}"; do
      wait "$pid" || failed=1
    done
  fi
  if [[ "$failed" != "0" ]]; then
    echo "error: one or more minimal induction shards failed" >&2
    return 3
  fi
  combine_minimal_shards
}

run_feature_induction() {
  uvp scripts/ops/shard_jsonl.py \
    --input "$PACKS" \
    --out-dir "$FEATURE_SHARD_DIR" \
    --prefix feature_loo \
    --shards "$FEATURE_SHARDS"

  local pids=()
  local failed=0
  for index in $(seq 0 $((FEATURE_SHARDS - 1))); do
    run_feature_shard "$index" &
    pids+=("$!")
    if [[ "${#pids[@]}" -ge "$FEATURE_JOBS" ]]; then
      for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
      done
      pids=()
    fi
  done
  if [[ "${#pids[@]}" -gt 0 ]]; then
    for pid in "${pids[@]}"; do
      wait "$pid" || failed=1
    done
  fi
  if [[ "$failed" != "0" ]]; then
    echo "error: one or more feature LOO induction shards failed" >&2
    return 3
  fi

  uvp scripts/ops/combine_skill_shards.py \
    --packs "$PACKS" \
    --out "$FEATURE_SKILLS" \
    --prefix "$FEATURE_SKILLS" \
    --shards "$FEATURE_SHARDS"
}

run_feature_shard() {
  local index="$1"
  uvp scripts/skills/run_skill_mvp.py \
    --packs "$FEATURE_SHARD_DIR/feature_loo.shard$index.packs.jsonl" \
    --out "$FEATURE_SKILLS.shard$index.jsonl" \
    --stage-ledger "$FEATURE_SKILLS.shard$index.stages.jsonl" \
    --modes auto_skill_feature_driven \
    --leave-one-out \
    --resume --allow-partial --stream \
    --temperature "$TEMPERATURE" \
    --json-temperature 0 \
    --max-tokens "$MAX_TOKENS" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --max-retries "$SDK_MAX_RETRIES" \
    --parse-max-attempts "$PARSE_MAX_ATTEMPTS" \
    --no-enable-thinking
}

run_writingbench_eval() {
  local skills="$1" out="$2" summary="$3" modes="$4"
  uvp scripts/eval/run_writingbench_official_eval.py \
    --packs "$PACKS" \
    --skills "$BASELINE_SKILLS" \
    --skills "$skills" \
    --private-eval "$PRIVATE_EVAL" \
    --writingbench-root "$WRITINGBENCH_ROOT" \
    --out "$out" \
    --summary-out "$summary" \
    --modes "$modes" \
    --judge-config-prefix "$JUDGE_CONFIG_PREFIX" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    --judge-max-tokens "$JUDGE_MAX_TOKENS" \
    --parse-max-attempts "$PARSE_MAX_ATTEMPTS" \
    --num-threads "$EVAL_NUM_THREADS" \
    --resume --allow-partial \
    --no-enable-thinking --no-judge-enable-thinking
}

prepare_pairwise_input() {
  uvp scripts/ops/prepare_three_metric_ablation.py pairwise-input \
    --base-eval "$TASK_EVAL_BASE" \
    --within-eval "$TASK_EVAL_WITHIN" \
    --xpack-eval "$TASK_EVAL_XPACK" \
    --feature-eval "$FEATURE_EVAL" \
    --out "$PAIRWISE_INPUT"
}

prepare_skill_quality_input() {
  uvp scripts/ops/prepare_three_metric_ablation.py skill-quality-input \
    --baseline-skills "$BASELINE_SKILLS" \
    --no-memory-skills "$SKILLS_NO_MEMORY" \
    --within-skills "$SKILLS_WITHIN" \
    --xpack-skills "$SKILLS_XPACK" \
    --feature-skills "$FEATURE_SKILLS" \
    --out "$SKILL_QUALITY_INPUT"
}

write_report() {
  uvp scripts/ops/prepare_three_metric_ablation.py report \
    --base-eval "$TASK_EVAL_BASE" \
    --within-eval "$TASK_EVAL_WITHIN" \
    --xpack-eval "$TASK_EVAL_XPACK" \
    --feature-eval "$FEATURE_EVAL" \
    --no-memory-skills "$SKILLS_NO_MEMORY" \
    --within-skills "$SKILLS_WITHIN" \
    --xpack-skills "$SKILLS_XPACK" \
    --pairwise-summary "$PAIRWISE_SUMMARY" \
    --skill-quality-summary "$SKILL_QUALITY_SUMMARY" \
    --out "$REPORT_OUT"
}

[[ "$RUN_INDUCTION" == "1" ]] && run_minimal_induction
[[ "$RUN_FEATURE_INDUCTION" == "1" ]] && run_feature_induction

if [[ "$RUN_EVAL" == "1" ]]; then
  uvp scripts/ops/prepare_three_metric_ablation.py minimal-alias \
    --skills "$SKILLS_OUT" \
    --out "$ALIAS_OUT" \
    --variant "$VARIANT"
  run_writingbench_eval "$ALIAS_OUT" "$EVAL_OUT" "$SUMMARY_OUT" "$MODES"
fi

[[ "$RUN_FEATURE_EVAL" == "1" ]] && run_writingbench_eval "$FEATURE_SKILLS" "$FEATURE_EVAL" "$FEATURE_SUMMARY" auto_skill

if [[ "$RUN_PAIRWISE" == "1" ]]; then
  prepare_pairwise_input
  uvp scripts/metrics/run_pairwise_likeness.py \
    --packs "$PACKS" \
    --candidate-eval "$PAIRWISE_INPUT" \
    --out "$PAIRWISE_OUT" \
    --summary-out "$PAIRWISE_SUMMARY" \
    --report-md "$PAIRWISE_REPORT" \
    --anchor-mode "$PAIRWISE_ANCHOR_MODE" \
    --candidate-modes "$PAIRWISE_CANDIDATE_MODES" \
    --judge-config-prefix "$JUDGE_CONFIG_PREFIX" \
    --orders both \
    --max-tokens "$PAIRWISE_MAX_TOKENS" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --max-retries "$SDK_MAX_RETRIES" \
    --parse-max-attempts "$PARSE_MAX_ATTEMPTS" \
    --num-threads "$PAIRWISE_NUM_THREADS" \
    --no-enable-thinking \
    --resume
fi

if [[ "$RUN_SKILL_QUALITY" == "1" ]]; then
  prepare_skill_quality_input
  uvp scripts/metrics/run_skill_quality_eval.py \
    --packs "$PACKS" \
    --skills "$SKILL_QUALITY_INPUT" \
    --out "$SKILL_QUALITY_OUT" \
    --summary-out "$SKILL_QUALITY_SUMMARY" \
    --modes "$SKILL_QUALITY_MODES" \
    --judge-config-prefix "$JUDGE_CONFIG_PREFIX" \
    --max-tokens "$SKILL_QUALITY_MAX_TOKENS" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --max-retries "$SDK_MAX_RETRIES" \
    --parse-max-attempts "$PARSE_MAX_ATTEMPTS" \
    --num-threads "$SKILL_QUALITY_NUM_THREADS" \
    --no-enable-thinking \
    --resume
fi

[[ "$RUN_REPORT" == "1" ]] && write_report

cat <<EOF
skills=$SKILLS_OUT
alias=$ALIAS_OUT
eval=$EVAL_OUT
summary=$SUMMARY_OUT
feature_skills=$FEATURE_SKILLS
feature_eval=$FEATURE_EVAL
pairwise_input=$PAIRWISE_INPUT
pairwise=$PAIRWISE_OUT
pairwise_summary=$PAIRWISE_SUMMARY
skill_quality_input=$SKILL_QUALITY_INPUT
skill_quality=$SKILL_QUALITY_OUT
skill_quality_summary=$SKILL_QUALITY_SUMMARY
report=$REPORT_OUT
EOF
