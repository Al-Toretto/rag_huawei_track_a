#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/a.tomat/work/LMCache}"
RAG_DIR="$REPO_DIR/benchmarks/rag"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-/home/a.tomat/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B}"
DRY_RUN="${DRY_RUN:-1}"
INCLUDE_FULL_RECOMPUTE="${INCLUDE_FULL_RECOMPUTE:-1}"
SMOKE_TEST="${SMOKE_TEST:-0}"
SMOKE_END_INDEX="${SMOKE_END_INDEX:-2}"
SAMPLING_TEMPERATURE="${SAMPLING_TEMPERATURE:-0.7}"
SAMPLING_TOP_P="${SAMPLING_TOP_P:-0.8}"
SAMPLING_TOP_K="${SAMPLING_TOP_K:-20}"
SAMPLING_MIN_P="${SAMPLING_MIN_P:-0.0}"
GENERATION_SEED="${GENERATION_SEED:-42}"
CHAT_TEMPLATE_MODE="${CHAT_TEMPLATE_MODE:-qwen-non-thinking}"
RUN_PREFIX="${RUN_PREFIX:-qwen3_chat_nothink_zh_v1_seed${GENERATION_SEED}}"

if [[ "$MODEL_SNAPSHOT" != /* ]]; then
    echo "MODEL_SNAPSHOT must be an absolute path, got: $MODEL_SNAPSHOT"
    exit 1
fi
if [[ ! -d "$MODEL_SNAPSHOT" ]]; then
    echo "MODEL_SNAPSHOT does not exist: $MODEL_SNAPSHOT"
    exit 1
fi
if [[ "$MODEL_SNAPSHOT" != *"/models--Qwen--Qwen3-32B/snapshots/"* ]]; then
    echo "MODEL_SNAPSHOT is not from the Qwen/Qwen3-32B Hugging Face cache:"
    echo "  $MODEL_SNAPSHOT"
    exit 1
fi
if [[ ! -f "$MODEL_SNAPSHOT/config.json" ]]; then
    echo "Missing model config: $MODEL_SNAPSHOT/config.json"
    exit 1
fi
if ! grep -Eq '"model_type"[[:space:]]*:[[:space:]]*"qwen3"' \
    "$MODEL_SNAPSHOT/config.json"; then
    echo "The selected snapshot is not Qwen3 according to config.json:"
    grep -E '"model_type"|"architectures"' \
        "$MODEL_SNAPSHOT/config.json" || true
    exit 1
fi
if [[ ! "$GENERATION_SEED" =~ ^[0-9]+$ ]]; then
    echo "GENERATION_SEED must be a non-negative integer: $GENERATION_SEED"
    exit 1
fi

mapfile -t BROKEN_LINKS < <(find "$MODEL_SNAPSHOT" -xtype l -print)
if (( ${#BROKEN_LINKS[@]} > 0 )); then
    echo "The cached snapshot contains broken links:"
    printf '  %s\n' "${BROKEN_LINKS[@]}"
    exit 1
fi

ALL_CONFIG_FILES=(
    "$RAG_DIR/lmcache_blend.yaml"
    "$RAG_DIR"/yaml_03_bound_and_mix/*.yaml
    "$RAG_DIR"/yaml_04_around_high_div/*.yaml
    "$RAG_DIR"/yaml_05_no_hkvd_and_layers/*.yaml
    "$RAG_DIR"/06_blend16_shared_candidates/*.yaml
)
ALL_DATASET_FILES=(
    "$RAG_DIR/inputs/multifieldqa_zh_longbench.json"
    "$RAG_DIR/inputs/dureader_longbench.json"
)

EXPECTED_CONFIG_COUNT=64
EXPECTED_DATASET_COUNT=2
if (( ${#ALL_CONFIG_FILES[@]} != EXPECTED_CONFIG_COUNT )); then
    echo "Expected $EXPECTED_CONFIG_COUNT YAMLs, found ${#ALL_CONFIG_FILES[@]}."
    exit 1
fi
if (( ${#ALL_DATASET_FILES[@]} != EXPECTED_DATASET_COUNT )); then
    echo "Expected $EXPECTED_DATASET_COUNT datasets, found ${#ALL_DATASET_FILES[@]}."
    exit 1
fi
for required_file in "${ALL_CONFIG_FILES[@]}" "${ALL_DATASET_FILES[@]}"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file is missing: $required_file"
        exit 1
    fi
done

if [[ "$SMOKE_TEST" == "1" ]]; then
    if [[ ! "$SMOKE_END_INDEX" =~ ^[1-9][0-9]*$ ]]; then
        echo "SMOKE_END_INDEX must be a positive integer, got: $SMOKE_END_INDEX"
        exit 1
    fi
    CONFIG_FILES=("$RAG_DIR/lmcache_blend.yaml")
    DATASET_FILES=("${ALL_DATASET_FILES[@]}")
    INCLUDE_FULL_RECOMPUTE=1
    RUN_PREFIX="${RUN_PREFIX}_smoke"
    export END_INDEX="$SMOKE_END_INDEX"
else
    CONFIG_FILES=("${ALL_CONFIG_FILES[@]}")
    DATASET_FILES=("${ALL_DATASET_FILES[@]}")
    export END_INDEX=50
fi

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

LMCache_JOB_COUNT=$((${#CONFIG_FILES[@]} * ${#DATASET_FILES[@]}))
FULL_RECOMPUTE_JOB_COUNT=0
if [[ "$INCLUDE_FULL_RECOMPUTE" == "1" ]]; then
    FULL_RECOMPUTE_JOB_COUNT=${#DATASET_FILES[@]}
fi
TOTAL_JOB_COUNT=$((LMCache_JOB_COUNT + FULL_RECOMPUTE_JOB_COUNT))

echo "repo_dir=$REPO_DIR"
echo "model_snapshot=$MODEL_SNAPSHOT"
echo "model_name=$MODEL_NAME"
echo "yaml_count=${#CONFIG_FILES[@]}"
echo "dataset_count=${#DATASET_FILES[@]}"
echo "lmcache_job_count=$LMCache_JOB_COUNT"
echo "full_recompute_job_count=$FULL_RECOMPUTE_JOB_COUNT"
echo "total_job_count=$TOTAL_JOB_COUNT"
echo "dry_run=$DRY_RUN"
echo "smoke_test=$SMOKE_TEST"
echo "sampling_temperature=$SAMPLING_TEMPERATURE"
echo "sampling_top_p=$SAMPLING_TOP_P"
echo "sampling_top_k=$SAMPLING_TOP_K"
echo "sampling_min_p=$SAMPLING_MIN_P"
echo "generation_seed=$GENERATION_SEED"
echo "chat_template_mode=$CHAT_TEMPLATE_MODE"
if [[ "$SMOKE_TEST" == "1" ]]; then
    echo "smoke_end_index=$SMOKE_END_INDEX"
fi

submit_lmcache_job() {
    local experiment_name="$1"
    local config_file="$2"
    local dataset_file="$3"
    local max_tokens="$4"
    local job_id

    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY RUN lmcache\texperiment=%s\tconfig=%s\tdataset=%s\tmax_tokens=%s\n' \
            "$experiment_name" "$config_file" "$dataset_file" "$max_tokens"
        return
    fi

    job_id=$(sbatch --parsable \
        --export=ALL,EXPERIMENT_NAME="$experiment_name",CONFIG_FILE="$config_file",DATASET_PATH="$dataset_file",MAX_TOKENS="$max_tokens" \
        "$RAG_DIR/sbash_runs/run_lmcache_experiment_h200.sbatch")
    printf 'SUBMITTED lmcache\tjob_id=%s\texperiment=%s\n' \
        "$job_id" "$experiment_name"
}

submit_full_recompute_job() {
    local experiment_name="$1"
    local dataset_file="$2"
    local max_tokens="$3"
    local job_id

    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY RUN full_recompute\texperiment=%s\tdataset=%s\tmax_tokens=%s\n' \
            "$experiment_name" "$dataset_file" "$max_tokens"
        return
    fi

    job_id=$(sbatch --parsable \
        --export=ALL,EXPERIMENT_NAME="$experiment_name",DATASET_PATH="$dataset_file",MAX_TOKENS="$max_tokens" \
        "$RAG_DIR/sbash_runs/run_vllm_experiment_h200.sbatch")
    printf 'SUBMITTED full_recompute\tjob_id=%s\texperiment=%s\n' \
        "$job_id" "$experiment_name"
}

for dataset_file in "${DATASET_FILES[@]}"; do
    dataset_tag=$(basename "$dataset_file" .json)
    case "$dataset_tag" in
        multifieldqa_zh*) dataset_max_tokens=256 ;;
        dureader*) dataset_max_tokens=1024 ;;
    esac
    for config_file in "${CONFIG_FILES[@]}"; do
        config_tag="${config_file#"$RAG_DIR"/}"
        config_tag="${config_tag%.yaml}"
        config_tag="${config_tag//\//__}"
        experiment_name="${RUN_PREFIX}_${dataset_tag}_${config_tag}"
        submit_lmcache_job \
            "$experiment_name" "$config_file" "$dataset_file" \
            "$dataset_max_tokens"
    done

    if [[ "$INCLUDE_FULL_RECOMPUTE" == "1" ]]; then
        experiment_name="${RUN_PREFIX}_${dataset_tag}_full_recompute"
        submit_full_recompute_job \
            "$experiment_name" "$dataset_file" "$dataset_max_tokens"
    fi
done
