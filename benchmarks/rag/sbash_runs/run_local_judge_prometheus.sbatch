#!/bin/bash -l
#SBATCH --job-name=rag_judge
#SBATCH --partition=rocky
#SBATCH -C type_h
#SBATCH -G 1
#SBATCH -c 2
#SBATCH --time=20:00:00
#SBATCH --output=/home/a.tomat/work/LMCache/benchmarks/rag/logs/%x_%j.out
#SBATCH --error=/home/a.tomat/work/LMCache/benchmarks/rag/logs/%x_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/a.tomat/work/LMCache}"
RESULTS_EVAL_DIR="$REPO_DIR/benchmarks/rag/results_eval"
INPUT_FILE="${INPUT_FILE:-$RESULTS_EVAL_DIR/outputs/judge_input_no_zh.jsonl}"
DEFAULT_OUTPUT_FILE="$RESULTS_EVAL_DIR/outputs/judge_outputs_no_zh_repaired.jsonl"
OUTPUT_FILE="${OUTPUT_FILE:-$DEFAULT_OUTPUT_FILE}"
PROMETHEUS_MODEL_ID="${PROMETHEUS_MODEL_ID:-prometheus-eval/prometheus-7b-v2.0}"
PROMETHEUS_CACHE_DIR="${PROMETHEUS_CACHE_DIR:-/home/a.tomat/.cache/huggingface/hub/models--prometheus-eval--prometheus-7b-v2.0}"
MODEL_NAME="${MODEL_NAME:-}"
JUDGE_CONDA_ENV="${JUDGE_CONDA_ENV:-rag_judge}"
HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/Miniconda}"
CUDA_MODULE="${CUDA_MODULE:-CUDA/12.9.1}"
GIT_MODULE="${GIT_MODULE:-tools/git/v2.33}"
LIMIT="${LIMIT:-}"
JUDGE_MAX_OUTPUT_TOKENS="${JUDGE_MAX_OUTPUT_TOKENS:-1024}"

if [[ -z "$MODEL_NAME" && -f "$PROMETHEUS_CACHE_DIR/refs/main" ]]; then
  MODEL_REVISION="$(tr -d '\r\n' < "$PROMETHEUS_CACHE_DIR/refs/main")"
  CACHED_SNAPSHOT="$PROMETHEUS_CACHE_DIR/snapshots/$MODEL_REVISION"
  if [[ -f "$CACHED_SNAPSHOT/config.json" ]]; then
    MODEL_NAME="$CACHED_SNAPSHOT"
  fi
fi
if [[ -z "$MODEL_NAME" ]]; then
  shopt -s nullglob
  CACHED_SNAPSHOTS=("$PROMETHEUS_CACHE_DIR"/snapshots/*)
  shopt -u nullglob
  if [[ "${#CACHED_SNAPSHOTS[@]}" -eq 1 \
    && -f "${CACHED_SNAPSHOTS[0]}/config.json" ]]; then
    MODEL_NAME="${CACHED_SNAPSHOTS[0]}"
  fi
fi
MODEL_NAME="${MODEL_NAME:-$PROMETHEUS_MODEL_ID}"

if [[ "$MODEL_NAME" == "$PROMETHEUS_CACHE_DIR"/snapshots/* ]]; then
  BROKEN_MODEL_LINK="$(find "$MODEL_NAME" -xtype l -print -quit)"
  if [[ -n "$BROKEN_MODEL_LINK" ]]; then
    echo "Cached judge snapshot has a broken link: $BROKEN_MODEL_LINK" >&2
    exit 1
  fi
fi

load_optional_module() {
  local module_name="$1"
  if ! module load "$module_name"; then
    echo "Warning: failed to load module: $module_name"
    module spider "$module_name" || true
  fi
}

# Jobs submitted from an activated login-node environment can inherit an
# inconsistent Conda stack after modules are purged. Start activation from a
# clean state so Conda does not try to deactivate a missing CONDA_PREFIX.
unset CONDA_DEFAULT_ENV
unset CONDA_PREFIX
unset CONDA_PREFIX_1
unset CONDA_PREFIX_2
unset CONDA_PROMPT_MODIFIER
unset CONDA_SHLVL
unset CONDA_EXE
unset CONDA_PYTHON_EXE
unset CONDA_ROOT
unset _CE_CONDA
unset _CE_M

module purge
load_optional_module "$PYTHON_MODULE"
load_optional_module "$CUDA_MODULE"
load_optional_module "$GIT_MODULE"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$JUDGE_CONDA_ENV"

cd "$REPO_DIR"
mkdir -p "$RESULTS_EVAL_DIR/outputs" "$REPO_DIR/benchmarks/rag/logs"

echo "node=$(hostname)"
echo "repo_dir=$REPO_DIR"
echo "input_file=$INPUT_FILE"
echo "output_file=$OUTPUT_FILE"
echo "model_name=$MODEL_NAME"
if [[ "$MODEL_NAME" == "$PROMETHEUS_CACHE_DIR"/snapshots/* ]]; then
  echo "model_revision=$(basename "$MODEL_NAME")"
fi
echo "judge_conda_env=$JUDGE_CONDA_ENV"
echo "hf_local_files_only=$HF_LOCAL_FILES_ONLY"
echo "judge_max_output_tokens=$JUDGE_MAX_OUTPUT_TOKENS"
echo "python=$(which python)"

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("cuda_total_memory_gb", round(props.total_memory / 1024**3, 2))
PY

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

HF_CACHE_ARGS=()
if [[ "$HF_LOCAL_FILES_ONLY" == "0" ]]; then
  HF_CACHE_ARGS=(--no-hf-local-files-only)
fi

python benchmarks/rag/results_eval/run_local_judge.py \
  --backend transformers \
  --model "$MODEL_NAME" \
  --prompt-style prometheus \
  --max-output-tokens "$JUDGE_MAX_OUTPUT_TOKENS" \
  --require-complete-scores \
  --input "$INPUT_FILE" \
  --output "$OUTPUT_FILE" \
  --resume \
  "${HF_CACHE_ARGS[@]}" \
  "${LIMIT_ARGS[@]}"

echo "Judge run finished."
echo "Rows written:"
wc -l "$OUTPUT_FILE" || true
