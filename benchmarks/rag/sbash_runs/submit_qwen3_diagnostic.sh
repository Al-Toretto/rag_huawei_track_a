#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/a.tomat/work/LMCache}"
RAG_DIR="$REPO_DIR/benchmarks/rag"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-/home/a.tomat/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B}"
CONFIG_FILE="${CONFIG_FILE:-$RAG_DIR/lmcache_blend.yaml}"
DIAGNOSTIC_END_INDEX="${DIAGNOSTIC_END_INDEX:-10}"
DRY_RUN="${DRY_RUN:-1}"
SAMPLING_TEMPERATURE="${SAMPLING_TEMPERATURE:-0.7}"
SAMPLING_TOP_P="${SAMPLING_TOP_P:-0.8}"
SAMPLING_TOP_K="${SAMPLING_TOP_K:-20}"
SAMPLING_MIN_P="${SAMPLING_MIN_P:-0.0}"
GENERATION_SEED="${GENERATION_SEED:-42}"
CHAT_TEMPLATE_MODE="${CHAT_TEMPLATE_MODE:-qwen-non-thinking}"
RUN_PREFIX="${RUN_PREFIX:-qwen3_chat_nothink_v5_seed${GENERATION_SEED}_diag${DIAGNOSTIC_END_INDEX}}"
LMCACHE_SBATCH="$RAG_DIR/sbash_runs/run_lmcache_experiment_h200.sbatch"
VLLM_SBATCH="$RAG_DIR/sbash_runs/run_vllm_experiment_h200.sbatch"

DATASETS=(
    "2wikimqa_longbench"
    "musique_longbench"
    "samsum_longbench"
)

if [[ ! "$DIAGNOSTIC_END_INDEX" =~ ^[1-9][0-9]*$ ]]; then
    echo "DIAGNOSTIC_END_INDEX must be a positive integer: $DIAGNOSTIC_END_INDEX"
    exit 1
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1: $DRY_RUN"
    exit 1
fi
if [[ ! "$GENERATION_SEED" =~ ^[0-9]+$ ]]; then
    echo "GENERATION_SEED must be a non-negative integer: $GENERATION_SEED"
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
for required_file in "$CONFIG_FILE" "$LMCACHE_SBATCH" "$VLLM_SBATCH"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Missing required file: $required_file"
        exit 1
    fi
done

mkdir -p "$RAG_DIR/logs"
timestamp=$(date +%Y%m%d_%H%M%S)
submission_log="$RAG_DIR/logs/${RUN_PREFIX}_submission_${timestamp}.txt"

echo "repo_dir=$REPO_DIR" | tee "$submission_log"
echo "model_snapshot=$MODEL_SNAPSHOT" | tee -a "$submission_log"
echo "config_file=$CONFIG_FILE" | tee -a "$submission_log"
echo "diagnostic_end_index=$DIAGNOSTIC_END_INDEX" | tee -a "$submission_log"
echo "job_count=6" | tee -a "$submission_log"
echo "dry_run=$DRY_RUN" | tee -a "$submission_log"
echo "sampling_temperature=$SAMPLING_TEMPERATURE" | tee -a "$submission_log"
echo "sampling_top_p=$SAMPLING_TOP_P" | tee -a "$submission_log"
echo "sampling_top_k=$SAMPLING_TOP_K" | tee -a "$submission_log"
echo "sampling_min_p=$SAMPLING_MIN_P" | tee -a "$submission_log"
echo "generation_seed=$GENERATION_SEED" | tee -a "$submission_log"
echo "chat_template_mode=$CHAT_TEMPLATE_MODE" | tee -a "$submission_log"

for dataset in "${DATASETS[@]}"; do
    dataset_path="$RAG_DIR/inputs/${dataset}.json"
    if [[ ! -f "$dataset_path" ]]; then
        echo "Missing dataset: $dataset_path"
        exit 1
    fi
    case "$dataset" in
        samsum*) max_tokens=256 ;;
        *) max_tokens=32 ;;
    esac

    for mode in lmcache_blend full_recompute; do
        experiment_name="${RUN_PREFIX}_${dataset}_${mode}"
        if [[ "$DRY_RUN" == "1" ]]; then
            printf 'DRY RUN\tdataset=%s\tmode=%s\tend_index=%s\tmax_tokens=%s\tseed=%s\texperiment=%s\n' \
                "$dataset" "$mode" "$DIAGNOSTIC_END_INDEX" \
                "$max_tokens" "$GENERATION_SEED" "$experiment_name" \
                | tee -a "$submission_log"
            continue
        fi

        if [[ "$mode" == "lmcache_blend" ]]; then
            job_id=$(sbatch --parsable \
                --export=ALL,REPO_DIR="$REPO_DIR",MODEL_SNAPSHOT="$MODEL_SNAPSHOT",MODEL_NAME="$MODEL_NAME",TOKENIZER_NAME="$MODEL_SNAPSHOT",MODEL_CONFIG="$MODEL_SNAPSHOT",EXPERIMENT_NAME="$experiment_name",CONFIG_FILE="$CONFIG_FILE",DATASET_PATH="$dataset_path",END_INDEX="$DIAGNOSTIC_END_INDEX",MAX_TOKENS="$max_tokens",SAMPLING_TEMPERATURE="$SAMPLING_TEMPERATURE",SAMPLING_TOP_P="$SAMPLING_TOP_P",SAMPLING_TOP_K="$SAMPLING_TOP_K",SAMPLING_MIN_P="$SAMPLING_MIN_P",GENERATION_SEED="$GENERATION_SEED",CHAT_TEMPLATE_MODE="$CHAT_TEMPLATE_MODE" \
                "$LMCACHE_SBATCH")
        else
            job_id=$(sbatch --parsable \
                --export=ALL,REPO_DIR="$REPO_DIR",MODEL_SNAPSHOT="$MODEL_SNAPSHOT",MODEL_NAME="$MODEL_NAME",TOKENIZER_NAME="$MODEL_SNAPSHOT",EXPERIMENT_NAME="$experiment_name",DATASET_PATH="$dataset_path",END_INDEX="$DIAGNOSTIC_END_INDEX",MAX_TOKENS="$max_tokens",SAMPLING_TEMPERATURE="$SAMPLING_TEMPERATURE",SAMPLING_TOP_P="$SAMPLING_TOP_P",SAMPLING_TOP_K="$SAMPLING_TOP_K",SAMPLING_MIN_P="$SAMPLING_MIN_P",GENERATION_SEED="$GENERATION_SEED",CHAT_TEMPLATE_MODE="$CHAT_TEMPLATE_MODE" \
                "$VLLM_SBATCH")
        fi
        printf 'SUBMITTED\tdataset=%s\tmode=%s\tjob_id=%s\tmax_tokens=%s\texperiment=%s\n' \
            "$dataset" "$mode" "$job_id" "$max_tokens" \
            "$experiment_name" | tee -a "$submission_log"
    done
done

echo "submission_log=$submission_log"
