# Benchmarking LLM Performance: RAG Use Case
## Overview

This directory contains benchmarking tools for evaluating RAG-style workloads with LMCache and vLLM. The main evaluator is `rag.py`, which measures request-level latency and quality over JSON datasets stored in `benchmarks/rag/inputs/`.

### Current Workloads

- **RAG Benchmark**: Simulates a real RAG dataset to evaluate key metrics such as token throughput, average time to first token, and average quality.

For the Phase 2 Track A settings, five-dataset results and cluster workflow,
start with the [repository guide](../../README.md). The examples below describe
the generic benchmark client; set its model name to match the serving endpoint.

## Setup

1. Install the required dependencies in the environment that will run the benchmark client:
   ```bash
   pip install -r requirements.txt
   ```
## Running the RAG Benchmark
To run the RAG benchmark, use `launch_lmcache.sh` and `launch_vllm.sh`.

After starting the serving engine, run `./launch_lmcache.sh` or `./launch_vllm.sh` from this directory.

Useful environment overrides:
```bash
MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
DATASET_PATH=benchmarks/rag/inputs/musique_s.json
QPS=1.0
END_INDEX=16
MAX_TOKENS=16
BASE_URL=http://localhost:8000/v1
API_KEY=EMPTY
```

`launch_lmcache.sh`:
- precomputes document KV by sending lightweight requests to the running LMCache endpoint
- then runs `rag.py` with the real benchmark prompts
- if `END_INDEX` is unset, it uses `precompute.py` to determine how many requests fit in `KV_STORAGE_SIZE`
- if `END_INDEX` is set, it precomputes exactly that range

`launch_vllm.sh`:
- runs the same workload against a plain vLLM endpoint
- if `END_INDEX` is unset, it uses the full dataset length
- for apples-to-apples comparison with LMCache, reuse the `END_INDEX` printed by `launch_lmcache.sh`

Use `Ctrl-C` to terminate the benchmark at any time; `rag.py` will still write the results collected so far to the output CSV.

Example baseline server:

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.2 --port 8000 --disable-log-requests
```

Example LMCache + CacheBlend server:

```bash
LMCACHE_CONFIG_FILE=benchmarks/rag/lmcache_blend.yaml \
vllm serve mistralai/Mistral-7B-Instruct-v0.2 \
  --port 8000 \
  --disable-log-requests \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

### What does precompute.py do
`precompute.py` sends one lightweight completion request per document to the running LMCache endpoint so the document KV can be reused later during the benchmark. If `--kv-storage-size` is provided, it estimates how many requests fit within that cache budget and stops there. If `--end-index` is provided, it precomputes that fixed range.
### Arguments
#### Configure the workload
- `--dataset <str>` The path to the dataset. The format is described in `Dataset format` section.  
- `--start-index <int>` Start from which request in the dataset.
- `--end-index <int>` End before which request in the dataset. If not set, or set to negative value and has precomputation, it will default to the value returned by precompute according to how many requests' KV cache can be held in the given size.  
- `--shuffle` Random shuffle the dataset.  
- `--system-prompt <str>` System prompt before the documents.
- `--query-prompt <str>` Query prompt after the documents and before the question in dataset.
- `--separator <str>` The text used to separate system prompt, documents and query prompt. If enabling blending, should match the blend_separator. If not, should be "".
- `--prompt-build-method <str>` Should be QA or FEW_SHOT, indicating different tasks.
- `--time <int>` The number of seconds as an upper bound for this benchmark. By default no limit.
- `--step-interval <float>` The time interval benchmarking script steps for sending requests.
- `--max-tokens <int>` Maximum number of output tokens for every request.
- `--qps <float>` Query per second. The rate to send requests.
#### Configuring the serving engine connection
- `--model <str>` The model name used by the endpoint.
- `--base-url <str>` The URL endpoint for the language model server.
- `--api-key <str>` API key for the language model server.
#### Configure precompute
To benchmark CacheBlend, we need to precompute the KV cache of documents.  
- `--tokenizer <str>` The tokenizer name. If not provided, by default the same as `--model`.
- `--model-config <str>` The model config name. If not provided, by default the same as `--model`.
- `--kv-storage-size <str>` The size used for KV cache. This will decide how many requests will be sent, because we only precompute KV cache within this limit. The same as max_local_cache_size in LMCache config yaml.
- `--kv-chunk-size <int>` The same as chunk_size in LMCache config yaml.
- `--kv-precision-bit <int>` KV cache precision bit. By default 16 for FP16. Should be a multiple of 8.
#### Configure output
- `--output <str>` The csv file to dump the detailed stats for each query (default = summary.csv)
- `--verbose` Enable verbose logging.

### Dataset notes
- QA datasets such as `musique_s.json` and `wikimqa_s.json` should use `PROMPT_BUILD_METHOD=QA`
- summarization datasets such as `samsum.json` and `multinews.json` should use `PROMPT_BUILD_METHOD=FEW_SHOT`
- the launch scripts infer this from the dataset filename by default

## Benchmark Metrics

- **Throughput**: Request processed per second.  
- **Average TTFT (Time to First Token)**: Average time taken for the model to generate the first token of a response.
- **Average Quality**: Average quality score of generation content.  

## Dataset format
Should be a json file, which is a list of dicts.  
Every item(dict) in the list is one request with the following content.  
```
 {
        "ctxs": [
            {
                "title": "",
                "text": "doc_1"
            },
            {
                "title": "",
                "text": "doc_2"
            },
            {
                "title": "",
                "text": "doc_3"
            }
        ],
        "question": "xxx ?",
        "answers": [
            "yyy"
        ]
    }
```
An example is [CacheBlend musique_s.json](https://github.com/YaoJiayi/CacheBlend/blob/main/inputs/musique_s.json)
