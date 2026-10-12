#!/usr/bin/env bash
# Submit the round-7 accuracy experiments with the established Qwen3 protocol.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
CONFIG_SET="${CONFIG_SET:-all}"
CONFIG_FILES="${CONFIG_FILES:-}"
INCLUDE_CONTROLS="${INCLUDE_CONTROLS:-0}"
DATASET_SELECTION="${DATASETS:-2wikimqa musique samsum multifieldqa_zh dureader}"
DRY_RUN="${DRY_RUN:-1}"
SMOKE_TEST="${SMOKE_TEST:-0}"
GENERATION_SEED="${GENERATION_SEED:-42}"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-/home/a.tomat/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137}"
RUN_PREFIX="${RUN_PREFIX:-07_accuracy_seed${GENERATION_SEED}}"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

case "$CONFIG_SET" in
    pilot|followup|all) ;;
    *) fail "CONFIG_SET must be pilot, followup, or all: $CONFIG_SET" ;;
esac
[[ "$INCLUDE_CONTROLS" == "0" ]] \
    || fail "Existing baselines are reused; INCLUDE_CONTROLS must be 0."
for flag in DRY_RUN SMOKE_TEST; do
    [[ "${!flag}" == "0" || "${!flag}" == "1" ]] \
        || fail "$flag must be 0 or 1: ${!flag}"
done
[[ "$GENERATION_SEED" =~ ^[0-9]+$ ]] \
    || fail "GENERATION_SEED must be a non-negative integer: $GENERATION_SEED"
[[ "$RUN_PREFIX" =~ ^07_[A-Za-z0-9_-]+$ ]] \
    || fail "RUN_PREFIX must start with 07_ and contain letters, digits, _ or -."
[[ -d "$REPO_DIR" ]] || fail "REPO_DIR does not exist: $REPO_DIR"
REPO_DIR=$(cd "$REPO_DIR" && pwd)
[[ "$REPO_DIR" != *,* && "$MODEL_SNAPSHOT" != *,* ]] \
    || fail "Repository and model paths cannot contain commas (Slurm export format)."
[[ "$MODEL_SNAPSHOT" == /* ]] \
    || fail "MODEL_SNAPSHOT must be an absolute path: $MODEL_SNAPSHOT"

if [[ "$SMOKE_TEST" == "1" ]]; then
    END_INDEX=2
    RUN_PREFIX="${RUN_PREFIX}_smoke"
else
    [[ "${END_INDEX:-50}" == "50" ]] \
        || fail "Normal round-7 runs require END_INDEX=50; use SMOKE_TEST=1 for two cases."
    END_INDEX=50
fi

RAG_DIR="$REPO_DIR/benchmarks/rag"
CONFIG_DIR="$RAG_DIR/07_accuracy_recovery"
LMCACHE_SBATCH="$RAG_DIR/sbash_runs/run_lmcache_experiment_h200.sbatch"
[[ -d "$CONFIG_DIR" ]] || fail "Missing config directory: $CONFIG_DIR"

CONFIG_PATHS=()
if [[ -n "$CONFIG_FILES" ]]; then
    [[ "$CONFIG_FILES" != ,* && "$CONFIG_FILES" != *, \
        && "$CONFIG_FILES" != *,,* ]] \
        || fail "CONFIG_FILES must be comma-separated YAML stems without empty entries."
    IFS=',' read -r -a requested_configs <<< "$CONFIG_FILES"
    declare -A seen_configs=()
    for config_stem in "${requested_configs[@]}"; do
        config_stem="${config_stem#"${config_stem%%[![:space:]]*}"}"
        config_stem="${config_stem%"${config_stem##*[![:space:]]}"}"
        config_stem="${config_stem%.yaml}"
        [[ "$config_stem" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
            || fail "Invalid CONFIG_FILES stem: $config_stem"
        [[ -z "${seen_configs[$config_stem]:-}" ]] \
            || fail "Duplicate CONFIG_FILES stem: $config_stem"
        seen_configs["$config_stem"]=1
        CONFIG_PATHS+=("$CONFIG_DIR/$config_stem.yaml")
    done
else
    shopt -s nullglob
    case "$CONFIG_SET" in
        pilot) CONFIG_PATHS=("$CONFIG_DIR"/a*.yaml) ;;
        followup) CONFIG_PATHS=("$CONFIG_DIR"/b*.yaml) ;;
        all) CONFIG_PATHS=("$CONFIG_DIR"/a*.yaml "$CONFIG_DIR"/b*.yaml) ;;
    esac
    shopt -u nullglob
fi
(( ${#CONFIG_PATHS[@]} > 0 )) || fail "No configs selected for CONFIG_SET=$CONFIG_SET"
(( ${#CONFIG_PATHS[@]} <= 24 )) || fail "Round 7 permits at most 24 YAML configs."

read -r -a requested_datasets <<< "$DATASET_SELECTION"
(( ${#requested_datasets[@]} > 0 )) || fail "DATASETS cannot be empty."
DATASET_STEMS=()
declare -A seen_datasets=()
for dataset in "${requested_datasets[@]}"; do
    case "$dataset" in
        2wikimqa|wikimqa|2wikimqa_longbench) dataset_stem=2wikimqa_longbench ;;
        musique|musique_longbench) dataset_stem=musique_longbench ;;
        samsum|samsum_longbench) dataset_stem=samsum_longbench ;;
        multifieldqa|multifieldqa_zh|multifieldqa_zh_longbench)
            dataset_stem=multifieldqa_zh_longbench ;;
        dureader|dureader_longbench) dataset_stem=dureader_longbench ;;
        *) fail "Unknown dataset: $dataset" ;;
    esac
    [[ -z "${seen_datasets[$dataset_stem]:-}" ]] \
        || fail "Duplicate dataset selection: $dataset_stem"
    seen_datasets["$dataset_stem"]=1
    DATASET_STEMS+=("$dataset_stem")
done

# Validate every required file before creating a log or submitting any job.
for required_file in "${CONFIG_PATHS[@]}" "$LMCACHE_SBATCH"; do
    [[ -f "$required_file" ]] || fail "Missing required file: $required_file"
done
for dataset_stem in "${DATASET_STEMS[@]}"; do
    [[ -f "$RAG_DIR/inputs/$dataset_stem.json" ]] \
        || fail "Missing dataset: $RAG_DIR/inputs/$dataset_stem.json"
done

if [[ "$DRY_RUN" == "0" ]]; then
    command -v sbatch >/dev/null || fail "sbatch is not available in this shell."
    [[ -f "$MODEL_SNAPSHOT/config.json" ]] \
        || fail "Missing model config: $MODEL_SNAPSHOT/config.json"
    grep -Eq '"model_type"[[:space:]]*:[[:space:]]*"qwen3"' \
        "$MODEL_SNAPSHOT/config.json" \
        || fail "The selected model snapshot is not Qwen3: $MODEL_SNAPSHOT"
    broken_link=$(find "$MODEL_SNAPSHOT" -xtype l -print -quit)
    [[ -z "$broken_link" ]] || fail "Broken model snapshot link: $broken_link"
fi

# Explicit values prevent inherited shell settings from changing the protocol.
MODEL_NAME=Qwen/Qwen3-32B
TOKENIZER_NAME="$MODEL_SNAPSHOT"
MODEL_CONFIG="$MODEL_SNAPSHOT"
QPS=0.25
SAMPLING_TEMPERATURE=0.7
SAMPLING_TOP_P=0.8
SAMPLING_TOP_K=20
SAMPLING_MIN_P=0.0
CHAT_TEMPLATE_MODE=qwen-non-thinking
MAX_MODEL_LEN=30000
MAX_NUM_BATCHED_TOKENS=16384
KV_STORAGE_SIZE=320GB
KV_CHUNK_SIZE=256
BLEND_SEPARATOR=' # # '
REQUEST_TIMEOUT=5400
WARMUP=0

# Each job must derive its own paths and ports, and infer QA versus few-shot
# prompting from its dataset. Shared overrides would break the manifest mapping.
unset OUTPUT_DIR LOG_DIR OUTPUT_FILE DEBUG_OUTPUT_FILE PORT KV_TRANSFER_PORT
unset PROMPT_BUILD_METHOD

COMMON_EXPORT="ALL,REPO_DIR=$REPO_DIR,MODEL_SNAPSHOT=$MODEL_SNAPSHOT,MODEL_NAME=$MODEL_NAME,TOKENIZER_NAME=$TOKENIZER_NAME,MODEL_CONFIG=$MODEL_CONFIG,QPS=$QPS,END_INDEX=$END_INDEX,SAMPLING_TEMPERATURE=$SAMPLING_TEMPERATURE,SAMPLING_TOP_P=$SAMPLING_TOP_P,SAMPLING_TOP_K=$SAMPLING_TOP_K,SAMPLING_MIN_P=$SAMPLING_MIN_P,GENERATION_SEED=$GENERATION_SEED,CHAT_TEMPLATE_MODE=$CHAT_TEMPLATE_MODE,MAX_MODEL_LEN=$MAX_MODEL_LEN,MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS,KV_STORAGE_SIZE=$KV_STORAGE_SIZE,KV_CHUNK_SIZE=$KV_CHUNK_SIZE,BLEND_SEPARATOR=$BLEND_SEPARATOR,REQUEST_TIMEOUT=$REQUEST_TIMEOUT,WARMUP=$WARMUP"

job_count=$((${#CONFIG_PATHS[@]} * ${#DATASET_STEMS[@]}))
printf 'repo_dir=%s\nconfig_set=%s\nyaml_count=%s\ndataset_count=%s\ninclude_controls=%s\njob_count=%s\ndry_run=%s\nrun_prefix=%s\nend_index=%s\ngeneration_seed=%s\n' \
    "$REPO_DIR" "$CONFIG_SET" "${#CONFIG_PATHS[@]}" \
    "${#DATASET_STEMS[@]}" "$INCLUDE_CONTROLS" "$job_count" \
    "$DRY_RUN" "$RUN_PREFIX" "$END_INDEX" "$GENERATION_SEED"

submission_log=""
if [[ "$DRY_RUN" == "0" ]]; then
    mkdir -p "$RAG_DIR/logs" "$CONFIG_DIR/submissions"
    timestamp=$(date -u +%Y%m%dT%H%M%SZ)
    submission_log="$CONFIG_DIR/submissions/${RUN_PREFIX}_${CONFIG_SET}_${timestamp}_$$.tsv"
    printf 'dataset\tconfig_key\texperiment_name\tjob_id\tgeneration_max_tokens\tend_index\tgeneration_seed\n' \
        > "$submission_log"
    printf 'submission_log=%s\n' "$submission_log"
fi

submit_job() {
    local dataset_stem="$1"
    local config_key="$2"
    local config_path="$3"
    local max_tokens="$4"
    local experiment_name="${RUN_PREFIX}_${dataset_stem}_${config_key}"
    local job_export="$COMMON_EXPORT,EXPERIMENT_NAME=$experiment_name,DATASET_PATH=$RAG_DIR/inputs/$dataset_stem.json,MAX_TOKENS=$max_tokens"
    local submission_result job_id
    local -a command

    job_export+=",CONFIG_FILE=$config_path"
    command=(
        sbatch --parsable
        "--job-name=r7_${dataset_stem%_longbench}"
        "--export=$job_export"
        "$LMCACHE_SBATCH"
    )
    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY RUN'
        printf ' %q' "${command[@]}"
        printf '\n'
        return
    fi

    submission_result=$("${command[@]}")
    job_id="${submission_result%%;*}"
    [[ "$job_id" =~ ^[0-9]+$ ]] \
        || fail "Unexpected sbatch output for $experiment_name: $submission_result"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$dataset_stem" "$config_key" "$experiment_name" "$job_id" \
        "$max_tokens" "$END_INDEX" "$GENERATION_SEED" >> "$submission_log"
    printf 'SUBMITTED\tdataset=%s\tconfig_key=%s\tjob_id=%s\texperiment=%s\n' \
        "$dataset_stem" "$config_key" "$job_id" "$experiment_name"
}

for dataset_stem in "${DATASET_STEMS[@]}"; do
    case "$dataset_stem" in
        2wikimqa*|musique*) max_tokens=32 ;;
        samsum*|multifieldqa_zh*) max_tokens=256 ;;
        dureader*) max_tokens=1024 ;;
    esac
    for config_path in "${CONFIG_PATHS[@]}"; do
        config_stem=$(basename "$config_path" .yaml)
        config_key="07_accuracy_recovery__${config_stem}"
        submit_job "$dataset_stem" "$config_key" "$config_path" "$max_tokens"
    done
done

if [[ "$DRY_RUN" == "0" ]]; then
    printf 'Submitted %s jobs. Submission TSV: %s\n' "$job_count" "$submission_log"
fi
