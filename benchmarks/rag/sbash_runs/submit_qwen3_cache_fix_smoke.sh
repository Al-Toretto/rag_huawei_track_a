#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/a.tomat/work/LMCache}"
RAG_DIR="$REPO_DIR/benchmarks/rag"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-/home/a.tomat/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B}"
CONFIG_FILE="${CONFIG_FILE:-$RAG_DIR/lmcache_blend.yaml}"
SMOKE_END_INDEX="${SMOKE_END_INDEX:-2}"
DRY_RUN="${DRY_RUN:-1}"
RUN_PREFIX="${RUN_PREFIX:-qwen3_cachefix_stop_v2}"
SBATCH_FILE="$RAG_DIR/sbash_runs/run_lmcache_experiment_h200.sbatch"

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
if [[ "$MODEL_SNAPSHOT" != /* || ! -d "$MODEL_SNAPSHOT" ]]; then
    echo "MODEL_SNAPSHOT must be an existing absolute directory:"
    echo "  $MODEL_SNAPSHOT"
    exit 1
fi
if [[ ! -f "$MODEL_SNAPSHOT/config.json" ]] || \
    ! grep -Eq '"model_type"[[:space:]]*:[[:space:]]*"qwen3"' \
        "$MODEL_SNAPSHOT/config.json"; then
    echo "MODEL_SNAPSHOT is not a valid Qwen3 snapshot: $MODEL_SNAPSHOT"
    exit 1
fi
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Missing LMCache config: $CONFIG_FILE"
    exit 1
fi
if [[ ! -f "$SBATCH_FILE" ]]; then
    echo "Missing sbatch file: $SBATCH_FILE"
    exit 1
fi

mkdir -p "$RAG_DIR/logs"
timestamp=$(date +%Y%m%d_%H%M%S)
submission_log="$RAG_DIR/logs/${RUN_PREFIX}_cache_smoke_submission_${timestamp}.txt"

echo "repo_dir=$REPO_DIR" | tee "$submission_log"
echo "model_snapshot=$MODEL_SNAPSHOT" | tee -a "$submission_log"
echo "config_file=$CONFIG_FILE" | tee -a "$submission_log"
echo "smoke_end_index=$SMOKE_END_INDEX" | tee -a "$submission_log"
echo "dry_run=$DRY_RUN" | tee -a "$submission_log"

for dataset in "${DATASETS[@]}"; do
    dataset_path="$RAG_DIR/inputs/${dataset}.json"
    if [[ ! -f "$dataset_path" ]]; then
        echo "Missing dataset: $dataset_path"
        exit 1
    fi

    experiment_name="${RUN_PREFIX}_${dataset}_lmcache_blend_smoke"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY RUN\tdataset=%s\texperiment=%s\n' \
            "$dataset" "$experiment_name" | tee -a "$submission_log"
        continue
    fi

    job_id=$(sbatch --parsable \
        --export=ALL,REPO_DIR="$REPO_DIR",MODEL_SNAPSHOT="$MODEL_SNAPSHOT",MODEL_NAME="$MODEL_NAME",TOKENIZER_NAME="$MODEL_SNAPSHOT",MODEL_CONFIG="$MODEL_SNAPSHOT",EXPERIMENT_NAME="$experiment_name",CONFIG_FILE="$CONFIG_FILE",DATASET_PATH="$dataset_path",END_INDEX="$SMOKE_END_INDEX" \
        "$SBATCH_FILE")
    printf 'SUBMITTED\tdataset=%s\tjob_id=%s\texperiment=%s\n' \
        "$dataset" "$job_id" "$experiment_name" | tee -a "$submission_log"
done

echo "submission_log=$submission_log"
