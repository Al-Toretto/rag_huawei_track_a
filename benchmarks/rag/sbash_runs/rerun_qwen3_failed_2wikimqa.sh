#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/a.tomat/work/LMCache}"
RAG_DIR="$REPO_DIR/benchmarks/rag"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-/home/a.tomat/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B}"
DATASET_PATH="$RAG_DIR/inputs/2wikimqa_longbench.json"
RUN_PREFIX="${RUN_PREFIX:-qwen3_model_only_rerun1}"
# The Rocky/H200 partition rejected a 400G request. Keep Slurm's existing
# memory policy, but serialize these jobs so our 320 GiB pinned LMCache stores
# cannot overlap with one another again.
SBATCH_MEM="${SBATCH_MEM:-}"
SERIAL="${SERIAL:-1}"
DRY_RUN="${DRY_RUN:-1}"

# These eight jobs produced no result CSV, and the remaining seven produced a
# CSV with unfinished requests. The list was audited from submission
# qwen3_full_submission_20260825_185942.txt and the pulled logs/results.
CONFIG_FILES=(
    "$RAG_DIR/06_blend16_shared_candidates/blend16_layer1_neighbor_top50_right_cap020_adaptive_k005_v020.yaml"
    "$RAG_DIR/06_blend16_shared_candidates/blend16_layer1_rank_adaptive_k005_v025_boundary1.yaml"
    "$RAG_DIR/06_blend16_shared_candidates/blend16_layer1_weighted_05_05_boundary1.yaml"
    "$RAG_DIR/yaml_03_bound_and_mix/lmcache_blend_boundary_2.yaml"
    "$RAG_DIR/yaml_03_bound_and_mix/lmcache_blend_selector_diff_kv_sum.yaml"
    "$RAG_DIR/yaml_03_bound_and_mix/lmcache_blend_weighted_10_10.yaml"
    "$RAG_DIR/yaml_04_around_high_div/neighbor_top50_both_cap032.yaml"
    "$RAG_DIR/yaml_04_around_high_div/neighbor_top50_right_cap024.yaml"
    "$RAG_DIR/06_blend16_shared_candidates/blend16_layer1_rank_adaptive_k005_v020.yaml"
    "$RAG_DIR/06_blend16_shared_candidates/blend16_layer1_rank_adaptive_k005_v020_gamma2_boundary1.yaml"
    "$RAG_DIR/yaml_03_bound_and_mix/lmcache_blend_selector_diff_kv_weighted_k30_v70.yaml"
    "$RAG_DIR/yaml_03_bound_and_mix/lmcache_blend_selector_diff_kv_weighted_k70_v30.yaml"
    "$RAG_DIR/yaml_04_around_high_div/neighbor_top50_both_cap024.yaml"
    "$RAG_DIR/yaml_04_around_high_div/plain_ratio_020.yaml"
    "$RAG_DIR/yaml_04_around_high_div/plain_ratio_024.yaml"
)

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

echo "repo_dir=$REPO_DIR"
echo "model_snapshot=$MODEL_SNAPSHOT"
echo "model_name=$MODEL_NAME"
echo "dataset=$DATASET_PATH"
echo "rerun_job_count=${#CONFIG_FILES[@]}"
echo "sbatch_mem=${SBATCH_MEM:-cluster_default}"
echo "serial=$SERIAL"
echo "dry_run=$DRY_RUN"

previous_job_id=""
for config_file in "${CONFIG_FILES[@]}"; do
    config_tag="${config_file#"$RAG_DIR"/}"
    config_tag="${config_tag%.yaml}"
    config_tag="${config_tag//\//__}"
    experiment_name="${RUN_PREFIX}_2wikimqa_longbench_${config_tag}"

    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY RUN\texperiment=%s\tconfig=%s\n' \
            "$experiment_name" "$config_file"
        continue
    fi

    sbatch_args=(--parsable)
    if [[ -n "$SBATCH_MEM" ]]; then
        sbatch_args+=(--mem="$SBATCH_MEM")
    fi
    if [[ "$SERIAL" == "1" && -n "$previous_job_id" ]]; then
        sbatch_args+=(--dependency="afterany:$previous_job_id")
    fi

    job_id=$(sbatch "${sbatch_args[@]}" \
        --export=ALL,EXPERIMENT_NAME="$experiment_name",CONFIG_FILE="$config_file",DATASET_PATH="$DATASET_PATH" \
        "$RAG_DIR/sbash_runs/run_lmcache_experiment_h200.sbatch")
    printf 'SUBMITTED\tjob_id=%s\texperiment=%s\n' \
        "$job_id" "$experiment_name"
    previous_job_id="$job_id"
done
