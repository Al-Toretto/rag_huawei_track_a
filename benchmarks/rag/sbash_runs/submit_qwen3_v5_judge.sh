#!/bin/bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/a.tomat/work/LMCache}"
JUDGE_DIR="$REPO_DIR/benchmarks/rag/results_eval/outputs/qwen3_chat_nothink_v5_seed42"
INPUT_FILE="${INPUT_FILE:-$JUDGE_DIR/judge_input_qwen3_chat_nothink_v5_seed42.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-$JUDGE_DIR/judge_outputs_qwen3_chat_nothink_v5_seed42.jsonl}"
SBATCH_FILE="$REPO_DIR/benchmarks/rag/sbash_runs/run_local_judge_prometheus.sbatch"
DRY_RUN="${DRY_RUN:-0}"
LIMIT="${LIMIT:-}"

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Missing judge input: $INPUT_FILE" >&2
  exit 1
fi
if [[ ! -f "$SBATCH_FILE" ]]; then
  echo "Missing judge sbatch file: $SBATCH_FILE" >&2
  exit 1
fi

export_values="ALL,REPO_DIR=$REPO_DIR,INPUT_FILE=$INPUT_FILE,OUTPUT_FILE=$OUTPUT_FILE"
if [[ -n "$LIMIT" ]]; then
  export_values+=",LIMIT=$LIMIT"
fi
command=(
  sbatch
  --parsable
  "--export=$export_values"
  "$SBATCH_FILE"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'DRY_RUN'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi

job_id="$("${command[@]}")"
echo "Submitted Qwen3 v5 judge job: $job_id"
echo "Input: $INPUT_FILE"
echo "Resume output: $OUTPUT_FILE"
