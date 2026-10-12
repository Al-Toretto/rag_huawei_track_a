# Phase 2 Track A: selective KV recomputation for RAG

This repository contains the modified LMCache implementation, experiment
configurations, benchmark inputs, saved measurements and local semantic-evaluation
tools for Track A. It evaluates which independently cached document tokens to
recompute, and how to update their keys and values, when several documents are
assembled into a new request.

The experiments use Qwen3-32B on five English and Chinese LongBench tasks. Answer
quality is the current selection priority; TTFT, TPOT and offered-load throughput
are recorded alongside it. The implementation includes K-only, V-only and joint
K/V selectors, checking-layer changes, document-edge and neighbor augmentation,
fixed/adaptive fusion, and subset-only progressive filtering. The complete
LMCache source and build files are included.

## Start here

| Purpose | Location |
| --- | --- |
| Current five-dataset aggregate results | [TSV](benchmarks/rag/results/rag_qwen3_five_datasets_meeting_metrics_current_9622_chinese_judged.tsv), [XLSX](benchmarks/rag/results/rag_qwen3_five_datasets_meeting_metrics_current_9622_chinese_judged.xlsx) |
| Original request-level results | [benchmarks/rag/results/](benchmarks/rag/results/) |
| Saved judge inputs, outputs and run manifests | [benchmarks/rag/results_eval/outputs/](benchmarks/rag/results_eval/outputs/) |
| Selection and fusion implementation | [selection.py](lmcache/v1/compute/blend/selection.py), [fusion.py](lmcache/v1/compute/blend/fusion.py), [blender.py](lmcache/v1/compute/blend/blender.py) |
| Configuration fields | [BLEND_EXPERIMENT_PARAMS.md](benchmarks/rag/BLEND_EXPERIMENT_PARAMS.md) |
| vLLM compatibility patch | [tools/phase2/](tools/phase2/) |
| Prepared accuracy-recovery experiments | [07_accuracy_recovery/](benchmarks/rag/07_accuracy_recovery/) |
| General LMCache documentation | [README_LMCACHE.md](README_LMCACHE.md), [docs/source/](docs/source/) |

## Workflow

```text
LongBench JSON inputs + one LMCache YAML
                    |
                    v
vLLM server using this LMCache checkout
                    |
                    v
launch_lmcache.sh: independent document precomputation -> online RAG
launch_vllm.sh: full-context recomputation reference
                    |
                    v
request CSVs + benchmark logs
                    |
                    v
export_judge_input.py -> run_local_judge.py -> aggregate_judge_results.py
                    |
                    v
per-request semantic scores + dataset/method aggregates
```

Run commands below from the repository root on a Linux CUDA GPU node. Obtain a GPU
allocation first if a scheduler is required. The portable examples use the
existing launchers directly; the original cluster-specific Slurm scripts are
also preserved in [sbash_runs/](benchmarks/rag/sbash_runs/).

## Layout

```text
lmcache/                              modified serving/cache implementation
csrc/, rust/, requirements/           build and dependency sources
tools/phase2/                         guarded vLLM 0.10.0 patch
benchmarks/rag/
  rag.py, precompute.py, utils.py      request generation and lexical metrics
  launch_lmcache.sh, launch_vllm.sh     benchmark clients
  lmcache_blend.yaml                  inherited K-only 16% baseline
  phase2_smoke.yaml                   small-host-cache bring-up configuration
  yaml_03_bound_and_mix/              selectors, boundaries and fixed fusion
  yaml_04_around_high_div/            larger budgets, neighbors and gap closing
  yaml_05_no_hkvd_and_layers/          positional controls and checking layers
  06_blend16_shared_candidates/       adaptive fusion and progressive filtering
  07_accuracy_recovery/               prepared higher-budget follow-up sweep
  inputs/                            native dataset JSON files
  results/                           original measurements and aggregate tables
  logs/                              completion logs and submission records
  results_eval/                      local judge and aggregation tools
tests/                               upstream and Track A tests
```

Saved primary measurements are copied without changing their values. Large
duplicate prompt/token debug CSVs, verbose successful-run server traces and
regenerable inspection tables are omitted. Generated answers, references,
metrics, run settings, useful completion logs and saved judge records remain.
AI context files, reports, papers, source Git history, virtual environments,
model weights and compiled extensions are outside this package.

## Environment and installation

The saved serving logs record vLLM **0.10.0**, Qwen3-32B, BF16 inference, a 30,000
token context limit and disabled vLLM prefix caching. The reference installation
uses Python 3.12, PyTorch 2.7.1, xFormers 0.0.31 and Transformers 4.57.6. CUDA
drivers, a compatible CUDA toolkit with `nvcc`, and a C++ compiler are required
to build LMCache's native extensions. The larger model runs used H200; the
historical YAMLs reserve 320 GiB of host cache, in addition to other host-memory
needs. The separate smoke configuration below reserves 16 GiB.

Create a serving environment and install the pinned stack before building LMCache:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'setuptools>=77,<81' 'setuptools_scm>=8' ninja packaging wheel
python -m pip install -r requirements/phase2-serving.txt
python -m pip install -r benchmarks/rag/requirements.txt
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=0.4.2.dev28 \
  python -m pip install -e . --no-build-isolation
python tools/phase2/apply_vllm_patch.py --check
python tools/phase2/apply_vllm_patch.py
python tools/phase2/apply_vllm_patch.py --check
```

The explicit package version preserves this LMCache snapshot's version when
building with the new repository's Git history. `--no-build-isolation` preserves
the installed serving-stack Torch version. The patch registers the loaded vLLM
model before LMCache constructs its Blender. Its helper checks the vLLM version
and exact source checksum, backs up the original worker, and can be rerun safely.
See the [patch instructions](tools/phase2/README.md).

If Torch shared libraries are not found by the loader, use:

```bash
export PHASE2_TORCH_LIB="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="$PHASE2_TORCH_LIB:${LD_LIBRARY_PATH:-}"
```

Prepare local weights for `Qwen/Qwen3-32B`. The recorded generation snapshot is
`9216db5781bf21249d130ec9da846c4624c16137`. Set `MODEL_PATH` to that downloaded
snapshot and keep the API model name as `Qwen/Qwen3-32B`:

```bash
export MODEL_PATH=/path/to/local/Qwen3-32B/snapshot
```

Semantic evaluation uses `prometheus-eval/prometheus-7b-v2.0`. The saved Chinese
judge logs record revision `66ffb1fc20beebfb60a3964a957d9011723116c5`. The judge
can run in a separate environment with PyTorch, Transformers and Accelerate;
stop the serving process before judging if they share one GPU. The local runner
loads cached model files by default. No model weights are bundled here.
Install Accelerate in the judge environment with `python -m pip install accelerate`.

## Datasets and recorded protocol

| Input file in `benchmarks/rag/inputs/` | Prompt mode | Lexical metric | Generation cap |
| --- | --- | --- | --- |
| `musique_longbench.json` | `QA` | Qwen-tokenizer F1 | 32 |
| `2wikimqa_longbench.json` | `QA` | Qwen-tokenizer F1 | 32 |
| `samsum_longbench.json` | `FEW_SHOT` | ROUGE-L | 256 |
| `multifieldqa_zh_longbench.json` | `QA` | Qwen-tokenizer F1 | 256 |
| `dureader_longbench.json` | `QA` | Chinese ROUGE-L | 1024 |

The current matrix uses the 50 longest selected examples per task. Actual
available output counts and request IDs are recorded in the manifests and CSVs.
Generation uses Qwen's non-thinking chat framing, temperature 0.7, top-p 0.8,
top-k 20, min-p 0 and seed 42. Offered request load is 0.25 requests/s. This
throughput measurement is tied to that offered load; it is not a saturation
capacity measurement. Chinese inputs, reference answers and judge prompts
remain in Chinese.

For both Chinese tasks, set `PROMPT_BUILD_METHOD=QA` and the explicit generation
cap from the table. Dataset filenames select the Chinese task instructions and
scoring; this Track A launcher supports `QA` and `FEW_SHOT` modes.

## Run a CacheBlend smoke experiment

Start a fresh server in terminal 1:

```bash
LMCACHE_CONFIG_FILE="$PWD/benchmarks/rag/phase2_smoke.yaml" \
vllm serve "$MODEL_PATH" \
  --served-model-name Qwen/Qwen3-32B \
  --dtype bfloat16 --port 8000 --max-model-len 30000 \
  --max-num-batched-tokens 16384 --disable-log-requests \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_port":5555}'
```

After the health endpoint is ready, run the client in terminal 2 with the same
Python environment and model path:

```bash
MODEL_NAME=Qwen/Qwen3-32B TOKENIZER_NAME="$MODEL_PATH" MODEL_CONFIG="$MODEL_PATH" \
DATASET_PATH="$PWD/benchmarks/rag/inputs/musique_longbench.json" \
PROMPT_BUILD_METHOD=QA END_INDEX=2 MAX_TOKENS=32 \
OUTPUT_DIR="$PWD/benchmarks/rag/outputs/smoke" EXPERIMENT_NAME=smoke_k16 \
  bash benchmarks/rag/launch_lmcache.sh
```

The launcher checks the endpoint, precomputes each document's KV state, and runs
the assembled requests. A two-example smoke checks integration; use
`END_INDEX=50` for the selected benchmark workload.

For a recorded selector configuration, restart the server with its YAML, for
example [V-only 16%](benchmarks/rag/yaml_03_bound_and_mix/lmcache_blend_selector_diff_v.yaml).
Use the recorded host-cache capacity and the same inputs, sampling, separator
and generation caps. Keep independent output directories per dataset/configuration.
Restarting isolates configurations and clears the previous in-memory cache.

## Run the full-recomputation reference

Stop the previous server and start plain vLLM without the LMCache connector:

```bash
vllm serve "$MODEL_PATH" \
  --served-model-name Qwen/Qwen3-32B \
  --dtype bfloat16 --port 8000 --max-model-len 30000 \
  --max-num-batched-tokens 16384 --disable-log-requests \
  --no-enable-prefix-caching
```

Run the same task through the reference client:

```bash
MODEL_NAME=Qwen/Qwen3-32B TOKENIZER_NAME="$MODEL_PATH" \
DATASET_PATH="$PWD/benchmarks/rag/inputs/musique_longbench.json" \
PROMPT_BUILD_METHOD=QA END_INDEX=50 MAX_TOKENS=32 \
OUTPUT_DIR="$PWD/benchmarks/rag/outputs/reference" EXPERIMENT_NAME=full_recompute \
  bash benchmarks/rag/launch_vllm.sh
```

The reference recomputes the complete assembled input. Direct cached reuse
without recomputation is a separate positional/control configuration.

## Saved results and reproducible analysis

The current combined TSV/XLSX contains **65 configurations on each of five
datasets**. Historical generation and diagnostic artifacts are also preserved;
use the current combined table to locate the completed Qwen3 matrix.

| Artifact | Contents |
| --- | --- |
| `results/<run_name>/<run_name>.csv` | Request IDs, generated text, references, lexical scores, TTFT/TPOT, success/errors and generation settings |
| `logs/<run_name>/benchmark.log` | Completion and offered-load throughput information used by metric builders |
| `logs/*submission*.txt` and `.out`/`.err` | Original run/job metadata and rerun records |
| `results_eval/outputs/qwen3_chat_nothink_v5_seed42/manifest.csv` | Current English result paths and experiment labels |
| `results_eval/outputs/qwen3_chinese/manifest.csv` | Current Chinese result paths and experiment labels |
| Judge `*.jsonl` | Predictions/references, prompts, raw judge outputs and parsed semantic scores |
| Judge summary `*.csv` | Dataset/method and method-level aggregates |

Some older exploratory aggregates have no surviving raw run directories. Current
English and Chinese manifests resolve to the bundled request CSVs and inputs.
The original and reconciled Chinese judge outputs, retries and reconciliation
records are included so the saved semantic aggregates can be traced to their
source records. Check-only diagnostic outputs are not model-judge measurements.
Scores in the result tables are on 0–100; semantic means are graded scores on
0–1, not binary accuracy probabilities. TTFT and TPOT are in seconds.

Export current English judge inputs into a new analysis directory without
changing the saved records:

```bash
python benchmarks/rag/results_eval/export_judge_input.py \
  --manifest benchmarks/rag/results_eval/outputs/qwen3_chat_nothink_v5_seed42/manifest.csv \
  --output benchmarks/rag/outputs/analysis/english_judge_input.jsonl
```

For Chinese inputs, use `results_eval/outputs/qwen3_chinese/manifest.csv` instead.
Manifest CSV paths are repository-relative, so keep the repository root as the
working directory.

Inspect prompts without loading a model:

```bash
python benchmarks/rag/results_eval/run_local_judge.py \
  --input benchmarks/rag/outputs/analysis/english_judge_input.jsonl \
  --limit 2 --dry-run
```

Run semantic evaluation on an allocated GPU with local Prometheus weights:

```bash
python benchmarks/rag/results_eval/run_local_judge.py \
  --backend transformers --model /path/to/local/prometheus-7b-v2.0 \
  --input benchmarks/rag/outputs/analysis/english_judge_input.jsonl \
  --output benchmarks/rag/outputs/analysis/english_judge_outputs.jsonl \
  --max-output-tokens 1024 --resume --require-complete-scores
```

Aggregate those new scores:

```bash
python benchmarks/rag/results_eval/aggregate_judge_results.py \
  --input benchmarks/rag/outputs/analysis/english_judge_outputs.jsonl \
  --dataset-method-output benchmarks/rag/outputs/analysis/summary_by_dataset_method.csv \
  --method-output benchmarks/rag/outputs/analysis/summary_by_method.csv
```

Rebuild the saved English metrics into a separate directory:

```bash
python benchmarks/rag/results_eval/build_qwen3_v5_metrics.py \
  --judge-summary benchmarks/rag/results_eval/outputs/qwen3_chat_nothink_v5_seed42/judge_summary_qwen3_chat_nothink_v5_seed42_current_9622_by_dataset_method.csv \
  --workbook benchmarks/rag/outputs/analysis/rebuilt_english.xlsx \
  --output-dir benchmarks/rag/outputs/analysis/english --require-complete
```

The [evaluation guide](benchmarks/rag/results_eval/README.md) describes the Chinese
builder and complete five-task reconstruction. Its Chinese metric builder writes
derived files under `results/` even when an alternate `--output-dir` is supplied;
use a separate scratch copy when comparing a rebuild with the saved table.

## Prepared accuracy-recovery sweep

[Round 7](benchmarks/rag/07_accuracy_recovery/README.rst) contains 12 higher-budget
configurations and a submission/aggregation path. These files are preparation
for the next sweep; this snapshot contains no completed Round 7 runs. They test
larger K/V budgets, later checking and the supported boundary/fusion combination.
The existing low-budget and larger-rate results provide controls.

On a compatible Slurm cluster, preview the selected jobs first:

```bash
REPO_DIR="$PWD" MODEL_SNAPSHOT="$MODEL_PATH" CONFIG_SET=pilot DRY_RUN=1 \
  bash benchmarks/rag/sbash_runs/submit_round7.sh
```

The original Slurm execution scripts assume a site-specific module system and a
Conda environment named `lmcache`. Adjust those site settings before submitting
real jobs. The direct launch examples above do not depend on that scheduler setup.

## Validation and licensing

Run the CPU-only RAG and compatibility-patch checks:

```bash
python -m pip install pytest
python -m pytest -q --confcutdir=tests/benchmarks \
  tests/benchmarks/test_rag*.py tests/benchmarks/test_phase2_vllm_patch.py
```

The delivered current metric table was rebuilt from the saved CSVs/logs, and
exported English/Chinese judge inputs were compared with the saved JSON records.
These checks verify code/artifact consistency. GPU inference and hardware
performance must be measured in the configured serving environment.

Source revision and file checksums are in [snapshot.json](snapshot.json) and
[checksums.sha256](checksums.sha256). The source code retains the
[Apache-2.0 license](LICENSE) and upstream LMCache documentation.
