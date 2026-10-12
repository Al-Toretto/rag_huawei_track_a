#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/a.tomat/work/LMCache}"
JUDGE_DIR="$REPO_DIR/benchmarks/rag/results_eval/outputs/qwen3_chinese"
JUDGE_MODEL_SNAPSHOT="${JUDGE_MODEL_SNAPSHOT:-/home/a.tomat/.cache/huggingface/hub/models--prometheus-eval--prometheus-7b-v2.0/snapshots/66ffb1fc20beebfb60a3964a957d9011723116c5}"
SMOKE_TEST="${SMOKE_TEST:-0}"
REPAIR_TEST="${REPAIR_TEST:-0}"
DRY_RUN="${DRY_RUN:-0}"
SBATCH_FILE="$REPO_DIR/benchmarks/rag/sbash_runs/run_local_judge_prometheus.sbatch"

if [[ "$SMOKE_TEST" == "1" && "$REPAIR_TEST" != "0" ]]; then
  echo "SMOKE_TEST and REPAIR_TEST cannot both be enabled" >&2
  exit 1
fi

case "$SMOKE_TEST" in
  0)
    INPUT_FILE="$JUDGE_DIR/judge_input_qwen3_chinese.jsonl"
    OUTPUT_FILE="$JUDGE_DIR/judge_outputs_qwen3_chinese.jsonl"
    ;;
  1)
    INPUT_FILE="$JUDGE_DIR/judge_input_qwen3_chinese_smoke.jsonl"
    OUTPUT_FILE="$JUDGE_DIR/judge_outputs_qwen3_chinese_smoke.jsonl"
    ;;
  *)
    echo "SMOKE_TEST must be 0 or 1: $SMOKE_TEST" >&2
    exit 1
    ;;
esac

case "$REPAIR_TEST" in
  0) ;;
  1)
    INPUT_FILE="$JUDGE_DIR/judge_input_qwen3_chinese_repair.jsonl"
    OUTPUT_FILE="$JUDGE_DIR/judge_outputs_qwen3_chinese_repair.jsonl"
    ;;
  2)
    INPUT_FILE="$JUDGE_DIR/judge_input_qwen3_chinese_long_retry.jsonl"
    OUTPUT_FILE="$JUDGE_DIR/judge_outputs_qwen3_chinese_long_retry.jsonl"
    ;;
  *)
    echo "REPAIR_TEST must be 0, 1, or 2: $REPAIR_TEST" >&2
    exit 1
    ;;
esac

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Missing judge input: $INPUT_FILE" >&2
  exit 1
fi
if [[ ! -f "$SBATCH_FILE" ]]; then
  echo "Missing judge sbatch file: $SBATCH_FILE" >&2
  exit 1
fi
if [[ "$DRY_RUN" != "1" && ! -f "$JUDGE_MODEL_SNAPSHOT/config.json" ]]; then
  echo "Missing pinned Prometheus snapshot: $JUDGE_MODEL_SNAPSHOT" >&2
  exit 1
fi

JUDGE_MAX_OUTPUT_TOKENS="${JUDGE_MAX_OUTPUT_TOKENS:-1024}"

command=(
  sbatch
  --parsable
  "--export=ALL,REPO_DIR=$REPO_DIR,INPUT_FILE=$INPUT_FILE,OUTPUT_FILE=$OUTPUT_FILE,MODEL_NAME=$JUDGE_MODEL_SNAPSHOT,JUDGE_MAX_OUTPUT_TOKENS=$JUDGE_MAX_OUTPUT_TOKENS"
  "$SBATCH_FILE"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'DRY_RUN'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi

job_id="$("${command[@]}")"
echo "Submitted Chinese judge job: $job_id"
echo "Input: $INPUT_FILE"
echo "Resume output: $OUTPUT_FILE"
echo "Judge model: $JUDGE_MODEL_SNAPSHOT"
echo "Max judge output tokens: $JUDGE_MAX_OUTPUT_TOKENS"
