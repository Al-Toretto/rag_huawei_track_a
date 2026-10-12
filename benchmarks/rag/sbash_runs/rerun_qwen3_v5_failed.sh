#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/a.tomat/work/LMCache}"
RAG_DIR="$REPO_DIR/benchmarks/rag"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-/home/a.tomat/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B}"
DATASET_PATH="$RAG_DIR/inputs/2wikimqa_longbench.json"
RUN_PREFIX="${RUN_PREFIX:-qwen3_chat_nothink_v5_seed42_rerun3}"
DRY_RUN="${DRY_RUN:-1}"
SERIAL="${SERIAL:-1}"
SAMPLING_TEMPERATURE="${SAMPLING_TEMPERATURE:-0.7}"
SAMPLING_TOP_P="${SAMPLING_TOP_P:-0.8}"
SAMPLING_TOP_K="${SAMPLING_TOP_K:-20}"
SAMPLING_MIN_P="${SAMPLING_MIN_P:-0.0}"
GENERATION_SEED="${GENERATION_SEED:-42}"
CHAT_TEMPLATE_MODE="${CHAT_TEMPLATE_MODE:-qwen-non-thinking}"

# Original jobs 4308312--4308314 timed out together on their first precompute
# request after being colocated on the same node. Serialize only these three
# replacements so their 320 GiB LMCache CPU stores cannot overlap again.
CONFIG_FILES=(
    "$RAG_DIR/lmcache_blend.yaml"
    "$RAG_DIR/yaml_03_bound_and_mix/lmcache_blend_adaptive_weighted_k015_v040.yaml"
    "$RAG_DIR/yaml_03_bound_and_mix/lmcache_blend_adaptive_weighted.yaml"
)

if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1: $DRY_RUN"
    exit 1
fi
if [[ "$SERIAL" != "0" && "$SERIAL" != "1" ]]; then
    echo "SERIAL must be 0 or 1: $SERIAL"
    exit 1
fi
if [[ "$MODEL_SNAPSHOT" != /* || ! -d "$MODEL_SNAPSHOT" ]]; then
    echo "Qwen3 snapshot is not an existing absolute directory: $MODEL_SNAPSHOT"
    exit 1
fi
if [[ ! -f "$MODEL_SNAPSHOT/config.json" ]] ||
    ! grep -Eq '"model_type"[[:space:]]*:[[:space:]]*"qwen3"' \
        "$MODEL_SNAPSHOT/config.json"; then
    echo "Snapshot config is missing or is not Qwen3: $MODEL_SNAPSHOT/config.json"
    exit 1
fi
if find "$MODEL_SNAPSHOT" -xtype l -print -quit | grep -q .; then
    echo "The Qwen3 snapshot contains a broken symlink: $MODEL_SNAPSHOT"
    exit 1
fi
for required_file in "$DATASET_PATH" "${CONFIG_FILES[@]}"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file is missing: $required_file"
        exit 1
    fi
done

export MODEL_SNAPSHOT
export MODEL_NAME
export TOKENIZER_NAME="${TOKENIZER_NAME:-$MODEL_SNAPSHOT}"
export MODEL_CONFIG="${MODEL_CONFIG:-$MODEL_SNAPSHOT}"
export SAMPLING_TEMPERATURE
export SAMPLING_TOP_P
export SAMPLING_TOP_K
export SAMPLING_MIN_P
export GENERATION_SEED
export CHAT_TEMPLATE_MODE

mkdir -p "$RAG_DIR/logs"
timestamp=$(date +%Y%m%d_%H%M%S)
submission_log="$RAG_DIR/logs/${RUN_PREFIX}_submission_${timestamp}.txt"

echo "repo_dir=$REPO_DIR" | tee "$submission_log"
echo "model_snapshot=$MODEL_SNAPSHOT" | tee -a "$submission_log"
echo "dataset=$DATASET_PATH" | tee -a "$submission_log"
echo "rerun_job_count=${#CONFIG_FILES[@]}" | tee -a "$submission_log"
echo "serial=$SERIAL" | tee -a "$submission_log"
echo "dry_run=$DRY_RUN" | tee -a "$submission_log"
echo "sampling_temperature=$SAMPLING_TEMPERATURE" | tee -a "$submission_log"
echo "sampling_top_p=$SAMPLING_TOP_P" | tee -a "$submission_log"
echo "sampling_top_k=$SAMPLING_TOP_K" | tee -a "$submission_log"
echo "sampling_min_p=$SAMPLING_MIN_P" | tee -a "$submission_log"
echo "generation_seed=$GENERATION_SEED" | tee -a "$submission_log"
echo "chat_template_mode=$CHAT_TEMPLATE_MODE" | tee -a "$submission_log"

previous_job_id=""
for config_file in "${CONFIG_FILES[@]}"; do
    config_tag="${config_file#"$RAG_DIR"/}"
    config_tag="${config_tag%.yaml}"
    config_tag="${config_tag//\//__}"
    experiment_name="${RUN_PREFIX}_2wikimqa_longbench_${config_tag}"

    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY RUN\texperiment=%s\tconfig=%s\n' \
            "$experiment_name" "$config_file" | tee -a "$submission_log"
        continue
    fi

    sbatch_args=(--parsable)
    if [[ "$SERIAL" == "1" && -n "$previous_job_id" ]]; then
        sbatch_args+=(--dependency="afterok:$previous_job_id")
    fi
    job_id=$(sbatch "${sbatch_args[@]}" \
        --export=ALL,REPO_DIR="$REPO_DIR",EXPERIMENT_NAME="$experiment_name",CONFIG_FILE="$config_file",DATASET_PATH="$DATASET_PATH",MAX_TOKENS=32 \
        "$RAG_DIR/sbash_runs/run_lmcache_experiment_h200.sbatch")
    printf 'SUBMITTED\tjob_id=%s\texperiment=%s\tconfig=%s\n' \
        "$job_id" "$experiment_name" "$config_file" | tee -a "$submission_log"
    previous_job_id="$job_id"
done

echo "submission_log=$submission_log"
