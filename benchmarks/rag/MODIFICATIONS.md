# Benchmark RAG Folder Modifications

## Purpose

This note documents what changed in `benchmarks/rag/`, why those changes were needed, and how the current benchmark relates to the CacheBlend paper.

## Short answer

The current `benchmarks/rag/` directory still follows the same benchmark idea as the paper:

- baseline run: full prefill / no cache reuse
- LMCache run: precompute per-document KV, then answer the full RAG prompt with CacheBlend-enabled KV reuse
- metrics: TTFT, throughput-related timing, and task quality

However, it is **not an exact reproduction** of the paper artifact yet. It is an updated, current-repo-compatible version of the same evaluation idea.

Main reasons it is not identical:

- the original code path assumed the older `lmcache_vllm` integration
- the current repo uses the newer LMCache + vLLM integration path
- the paper evaluated multiple models and all four datasets, while the updated local workflow was first validated on smaller runs
- the paper also reports results under its own hardware/storage setup

## What the paper evaluated

From the attached paper:

- models: `Mistral-7B`, `Yi-34B`, `Llama-70B`
- datasets: `2WikiMQA`, `Musique`, `SAMSum`, `MultiNews`
- hardware:
  - `128 GB RAM`
  - `2 Nvidia A40 GPUs`
  - `1 TB NVMe SSD`
  - measured SSD throughput: `4.8 GB/s`
- serving layout:
  - `1 GPU` for `Mistral-7B` and `Yi-34B`
  - `2 GPUs` for `Llama-70B`

The paper’s comparison is conceptually:

- `full KV recompute` = full prefill baseline
- `prefix caching`
- `full KV reuse`
- `CacheBlend` = selective KV recompute after loading cached KV

This RAG folder is only aimed at the `full prefill baseline` vs `CacheBlend` part of that story.

## Summary of file changes

Modified files:

- `benchmarks/rag/README.md`
- `benchmarks/rag/launch_lmcache.sh`
- `benchmarks/rag/launch_vllm.sh`
- `benchmarks/rag/precompute.py`
- `benchmarks/rag/rag.py`
- `benchmarks/rag/utils.py`

Added file:

- `benchmarks/rag/lmcache_blend.yaml`

## File-by-file details

### `README.md`

### What the old version did

- described the folder as a generic RAG benchmark area
- assumed an older workflow
- referenced `example_blending.yaml`
- assumed users would manually keep `END_INDEX` aligned with the result printed by `precompute.py`
- used examples tied to the older serving stack

### What changed

- rewrote the overview to describe the folder as a current LMCache/vLLM RAG benchmark directory
- documented repo-local dataset usage under `benchmarks/rag/inputs/`
- documented environment-variable overrides such as `MODEL_NAME`, `DATASET_PATH`, `END_INDEX`, `QPS`, `BASE_URL`, and `MAX_TOKENS`
- clarified what `launch_lmcache.sh` and `launch_vllm.sh` each do
- updated the LMCache server launch example to the current `vllm serve` + `--kv-transfer-config` flow
- documented dataset-type inference:
  - QA datasets -> `QA`
  - summarization datasets -> `FEW_SHOT`

### Why

The old README described a workflow from the older CacheBlend repo. The updated version explains the workflow that actually matches the current LMCache repository.

### `launch_lmcache.sh`

### What the old version did

- hard-coded:
  - model name
  - dataset path
  - prompt type
  - KV storage size
  - chunk size
  - base URL
- assumed datasets lived under `~/CacheBlend/inputs/`
- exported `LMCACHE_CONFIG_FILE="example_blending.yaml"`
- called `precompute.py`
- parsed the returned end index
- launched `rag.py` with separator `"[BLEND_SEP]"`

### Problems in the old version

- repo-external dataset path
- old config filename
- no endpoint reachability check
- no env-based configurability
- old blend separator placeholder did not match the current LMCache blend separator
- tightly coupled to an older repo layout

### What changed

- added `set -euo pipefail`
- made the script repo-relative using `SCRIPT_DIR`
- made the main settings configurable through env vars:
  - `MODEL_NAME`
  - `DATASET_PATH`
  - `KV_STORAGE_SIZE`
  - `KV_CHUNK_SIZE`
  - `QPS`
  - `BASE_URL`
  - `API_KEY`
  - `END_INDEX`
  - `MAX_TOKENS`
  - `BLEND_SEPARATOR`
- added a preflight endpoint check against `.../v1/models`
- added automatic prompt-build-method inference from dataset name
- switched to repo-local dataset defaults
- switched from placeholder separator to the actual blend separator:
  - `" # # "`
- made precompute smarter:
  - if `END_INDEX` is set, precompute that fixed range
  - otherwise, use `KV_STORAGE_SIZE` to determine how much fits

### Why

This script needed to become portable, reproducible, and compatible with the current LMCache serving path.

### `launch_vllm.sh`

### What the old version did

- hard-coded model, dataset path, prompt type, QPS, `END_INDEX`, and base URL
- called `rag.py` directly for the baseline run

### Problems in the old version

- repo-external dataset path
- hard-coded `END_INDEX`
- no endpoint check
- no easy environment override path
- prompt mode had to be manually edited

### What changed

- added `set -euo pipefail`
- made it repo-relative
- added env-based configurability
- added the same endpoint reachability check used by the LMCache launcher
- added automatic prompt-build-method inference
- if `END_INDEX` is not set, it now uses the whole dataset length

### Why

This makes the baseline launch script consistent with the LMCache launch script and easier to compare fairly.

### `precompute.py`

### What the old version did

- depended on `lmcache_vllm.blend_adapter.OnlineKVPreCompute`
- loaded tokenizer/config up front
- estimated KV sizes and precomputed document KV for the old integration path
- relied on old blending-specific behavior such as `_blend_add_special_in_precomp`

### Problems in the old version

- the key dependency came from the older dedicated CacheBlend repo style
- it did not match the current LMCache + vLLM integration path in this repository
- it was more brittle offline because tokenizer/config loading was assumed even when not strictly needed

### What changed

- removed the `lmcache_vllm.blend_adapter.OnlineKVPreCompute` dependency
- added `EndpointPrecomputer`, which precomputes by sending lightweight completion requests to the currently running serving endpoint
- kept size-limited precompute support, but only loads tokenizer/config when `kv_storage_size > 0`
- added helper logic for model-config compatibility:
  - `_get_num_key_value_heads`
  - `_get_head_dim`
  - `_build_kv_size_calculator`
  - `_estimate_case_size`
- improved failure messages for offline or partially configured environments

### What this means semantically

The old file precomputed document KV through an old in-process/adapter-style integration.

The new file precomputes document KV by exercising the real server endpoint, which is a better match for the current repository’s serving model.

### `rag.py`

### What the old version did

- ran the actual benchmark requests
- measured TTFT and generation timing
- computed QA/summarization quality
- used `AutoTokenizer` during scoring
- assumed successful responses

### Problems in the old version

- benchmark scoring unnecessarily depended on a tokenizer download
- failed requests could crash summary computation
- summarization streaming handling was incomplete
- nested/non-string answer formats were not normalized well enough

### What changed

- removed the tokenizer dependency from the evaluator path
- added:
  - `normalize_answers`
  - `success` and `error_message` fields on `Response`
- added explicit handling for:
  - `BadRequestError`
  - `APIConnectionError`
- made streamed response parsing work for both:
  - chat completions
  - legacy completions
- made summary statistics ignore failed requests cleanly
- made TPOT safe when no completion tokens are produced
- normalized dataset answers before quality scoring

### Why

This makes the evaluator robust enough to survive real benchmark runs, including:

- server not yet ready
- request too long for current model limit
- endpoint failure
- empty generations

### `utils.py`

### What the old version did

- built prompts
- normalized text
- computed F1 and Rouge-L
- parsed generations for QA scoring

### Problems in the old version

- `parse_generation()` could crash on empty output
- `compute_f1()` required a tokenizer object

### What changed

- made `parse_generation()` return `""` safely on empty output
- changed `compute_f1()` to use normalized whitespace tokenization instead of a Hugging Face tokenizer

### Why

This removes an unnecessary external dependency from the evaluation path and prevents crashes when a request fails or returns no usable text.

### `lmcache_blend.yaml`

### What was added

This new file was added to provide a repo-local LMCache blend config for the RAG benchmark:

```yaml
chunk_size: 256
local_cpu: True
max_local_cpu_size: 30

enable_blending: True
blend_special_str: " # # "
use_layerwise: True
```

### Why

The old RAG scripts referenced `example_blending.yaml`, which belonged to the older layout. The new file gives the benchmark its own explicit LMCache blend config in the current repo.

## What was effectively deleted

These behaviors were removed from the RAG benchmark flow:

- direct dependence on `lmcache_vllm.blend_adapter.OnlineKVPreCompute`
- assumption that datasets live outside the repo under `~/CacheBlend/inputs/`
- dependence on the placeholder separator string `"[BLEND_SEP]"`
- dependence on tokenizer download just to compute QA F1
- assumption that requests always succeed

## What the two experiments now mean

### `launch_vllm.sh`

This is the baseline experiment.

It represents:

- full prefill
- no LMCache
- no CacheBlend
- compute KV for the whole prompt during the request

This is the closest thing in this folder to the paper’s `full KV recompute` baseline.

### `launch_lmcache.sh`

This is the CacheBlend experiment.

It does two phases:

1. document precompute
   - sends one lightweight request per document so document KV is populated in LMCache-backed storage
2. measured benchmark run
   - sends the full prompt
   - CacheBlend can load cached KV and selectively recompute only the needed subset

This is conceptually aligned with the paper’s CacheBlend condition.

## Is the current version still the same benchmark as the paper?

## Same idea: yes

The current version still matches the paper’s core benchmarking idea:

- use RAG-style multi-document prompts
- compare a no-reuse baseline against a cache-assisted selective-recompute mode
- measure TTFT and quality
- support both QA and summarization datasets

## Exact reproduction: no

It is not yet the exact paper benchmark for several reasons:

- the paper compared more than one baseline
  - full KV recompute
  - prefix caching
  - full KV reuse
  - CacheBlend
- the current folder only gives you the most important pair:
  - full prefill baseline
  - CacheBlend
- the paper used:
  - `Mistral-7B`
  - `Yi-34B`
  - `Llama-70B`
- the current validation runs were first adapted around the currently working environment and smaller smoke-test settings
- the paper reports results under a specific hardware/storage setup using A40 GPUs and NVMe SSD
- the old dedicated CacheBlend repo likely had tighter coupling between benchmark scripts and that experimental environment

## Practical conclusion

This current folder should be treated as:

- a faithful modernization of the paper’s benchmark idea
- a good path for comparing `full prefill` vs `CacheBlend` in the current LMCache repo
- not yet a perfect line-by-line reproduction of the original paper artifact

## Notes on fairness under current restrictions

To keep the comparison fair, both runs should use the same serving restrictions:

- same model
- same `max_model_len`
- same `max_num_seqs`
- same `max_num_batched_tokens`
- same `gpu_memory_utilization`

These restrictions may change absolute performance numbers, but they still allow a fair like-for-like comparison between baseline and CacheBlend under the same constrained setup.

## Commands that worked so far

The following commands are the ones that successfully worked in practice for the adapted benchmark flow.

### Dataset and model used

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- dataset: `benchmarks/rag/inputs/musique_s.json`
- benchmark subset: first `10` requests

### Baseline experiment: full prefill / no LMCache

Start the baseline server in terminal 1:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --port 8000 \
  --disable-log-requests \
  --gpu-memory-utilization 0.55 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 512 \
  --enforce-eager
```

Optional readiness check:

```bash
curl http://localhost:8000/v1/models
```

Run the baseline benchmark in terminal 2:

```bash
END_INDEX=10 \
MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct \
DATASET_PATH=benchmarks/rag/inputs/musique_s.json \
bash benchmarks/rag/launch_vllm.sh
```

This produces:

- `musique_s_vllm_qps_3.5.csv`

### CacheBlend experiment: LMCache + selective recompute

Stop the baseline server first so the GPU is clean.

Then start the LMCache-enabled server in terminal 1:

```bash
LMCACHE_CONFIG_FILE=benchmarks/rag/lmcache_blend.yaml \
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --port 8000 \
  --disable-log-requests \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
  --gpu-memory-utilization 0.55 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 512 \
  --enforce-eager
```

Optional readiness check:

```bash
curl http://localhost:8000/v1/models
```

Run the LMCache benchmark in terminal 2:

```bash
END_INDEX=10 \
MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct \
DATASET_PATH=benchmarks/rag/inputs/musique_s.json \
bash benchmarks/rag/launch_lmcache.sh
```

This produces:

- `musique_s_lmcache_qps_3.5.csv`

### Why these commands matter

These two command sets are fair to compare because they use the same:

- model
- dataset
- request count
- `gpu_memory_utilization`
- `max_model_len`
- `max_num_seqs`
- `max_num_batched_tokens`

The only intended difference is the serving mode:

- baseline: full prefill, no LMCache
- LMCache run: precompute plus CacheBlend-enabled KV reuse
