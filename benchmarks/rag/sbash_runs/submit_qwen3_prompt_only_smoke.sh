#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/a.tomat/work/LMCache}"
RAG_DIR="$REPO_DIR/benchmarks/rag"
SMOKE_END_INDEX="${SMOKE_END_INDEX:-5}"
DRY_RUN="${DRY_RUN:-1}"
RUN_PREFIX="${RUN_PREFIX:-qwen3_prompt_only_v1}"
SBATCH_FILE="$RAG_DIR/sbash_runs/run_vllm_experiment_h200.sbatch"

DATASETS=(
    "2wikimqa_longbench"
    "musique_longbench"
    "samsum_longbench"
)

if [[ ! "$SMOKE_END_INDEX" =~ ^[1-9][0-9]*$ ]]; then
    echo "SMOKE_END_INDEX must be a positive integer: $SMOKE_END_INDEX"
    exit 1
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1: $DRY_RUN"
    exit 1
fi
if [[ ! -f "$SBATCH_FILE" ]]; then
    echo "Missing sbatch file: $SBATCH_FILE"
    exit 1
fi

for dataset in "${DATASETS[@]}"; do
    dataset_path="$RAG_DIR/inputs/${dataset}.json"
    if [[ ! -f "$dataset_path" ]]; then
        echo "Missing dataset: $dataset_path"
        exit 1
    fi
done

mkdir -p "$RAG_DIR/logs"
timestamp=$(date +%Y%m%d_%H%M%S)
submission_log="$RAG_DIR/logs/${RUN_PREFIX}_submission_${timestamp}.txt"

echo "repo_dir=$REPO_DIR" | tee "$submission_log"
echo "run_prefix=$RUN_PREFIX" | tee -a "$submission_log"
echo "smoke_end_index=$SMOKE_END_INDEX" | tee -a "$submission_log"
echo "dry_run=$DRY_RUN" | tee -a "$submission_log"

for dataset in "${DATASETS[@]}"; do
    dataset_path="$RAG_DIR/inputs/${dataset}.json"
    experiment_name="${RUN_PREFIX}_${dataset}_full_recompute"

    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY RUN\tdataset=%s\texperiment=%s\n' \
            "$dataset" "$experiment_name" | tee -a "$submission_log"
        continue
    fi

    job_id=$(sbatch --parsable \
        --export=ALL,REPO_DIR="$REPO_DIR",EXPERIMENT_NAME="$experiment_name",DATASET_PATH="$dataset_path",END_INDEX="$SMOKE_END_INDEX" \
        "$SBATCH_FILE")
    printf 'SUBMITTED\tdataset=%s\tjob_id=%s\texperiment=%s\n' \
        "$dataset" "$job_id" "$experiment_name" | tee -a "$submission_log"
done

echo "submission_log=$submission_log"
