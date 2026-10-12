# Blend Experiment YAML Parameters

This document describes the parameters used by the RAG blend experiment YAML files in `benchmarks/rag/`.

It covers:

- the original LMCache blend-related parameters already supported by the repo
- the experiment parameters added for selector and fusion studies
- what each parameter means
- which values are currently supported in code
- the math behind the selection and fusion options

The examples in this folder are designed around YAML files such as:

- [lmcache_blend.yaml](lmcache_blend.yaml)
- [lmcache_blend_adaptive_weighted.yaml](yaml_03_bound_and_mix/lmcache_blend_adaptive_weighted.yaml)
- [lmcache_blend_selector_diff_kv_sum.yaml](yaml_03_bound_and_mix/lmcache_blend_selector_diff_kv_sum.yaml)

This reference details the deviation selectors and fusion rules below. The
complete implementations, including position-only selection, neighbors and
rank-adaptive fusion, are in [selection.py](../../lmcache/v1/compute/blend/selection.py)
and [fusion.py](../../lmcache/v1/compute/blend/fusion.py).
See the [repository guide](../../README.md) for the current Track A workflow.

## Full YAML Shape

Typical structure:

```yaml
chunk_size: 256
local_cpu: True
max_local_cpu_size: 320
save_unfull_chunk: True

enable_blending: True
blend_special_str: " # # "
use_layerwise: True
blend_check_layers: [1]
blend_recompute_ratios: [0.16]

extra_config:
  blend_selection_strategy: "topk_diff_k"
  blend_selection_k_weight: 1.0
  blend_selection_v_weight: 1.0
  blend_fusion_strategy: "overwrite_selected"
  blend_log_selection_stats: true
  blend_log_fusion_stats: true
  blend_force_chunk_boundary_tokens: false
  blend_chunk_boundary_token_count: 1
  blend_cached_k_weight: 0.0
  blend_cached_v_weight: 0.0
  blend_min_cached_k_weight: 0.0
  blend_max_cached_k_weight: 0.3
  blend_min_cached_v_weight: 0.0
  blend_max_cached_v_weight: 0.3
```

## Core Storage And Retrieval Parameters

### `chunk_size`

Type: `int`

Typical value:

```yaml
chunk_size: 256
```

Meaning:

- LMCache chunk size in tokens for chunk-based storage logic.
- In your experiments this is typically `256`.

Notes:

- This is not the same as the blend selector ratio.
- It affects storage layout and cache lookup granularity.

### `local_cpu`

Type: `bool`

Typical values:

- `True`
- `False`

Meaning:

- Whether LMCache stores KV on CPU memory.

### `max_local_cpu_size`

Type: `float`

Current convention for your experiments:

```yaml
max_local_cpu_size: 320
```

Meaning:

- Maximum CPU-side LMCache capacity.
- In your recent experiments, this should now always be `320`.

### `save_unfull_chunk`

Type: `bool`

Typical value:

```yaml
save_unfull_chunk: True
```

Meaning:

- Whether incomplete trailing chunks are also stored.

## Original Blend Parameters

These are the main blend-specific parameters already supported by LMCache.

### `enable_blending`

Type: `bool`

Typical value:

```yaml
enable_blending: True
```

Meaning:

- Enables the blend code path instead of plain cache retrieval.

### `blend_special_str`

Type: `str`

Typical value:

```yaml
blend_special_str: " # # "
```

Meaning:

- Separator string used to split reused segments/chunks.
- This separator must match the prompt-building logic in the benchmark.

Important:

- If this does not match the actual prompt separator, cached segments may not align correctly.

### `use_layerwise`

Type: `bool`

Typical value:

```yaml
use_layerwise: True
```

Meaning:

- Enables layerwise KV retrieval/blending.
- Blend experiments rely on this being `True`.

### `blend_check_layers`

Type: `list[int]`

Examples:

```yaml
blend_check_layers: [1]
blend_check_layers: [0]
blend_check_layers: [1, 3]
```

Meaning:

- Transformer layers where token selection is performed.
- Layer numbering is zero-based.

Behavior:

- If `blend_check_layers: [1]`, then:
  - layer `0`: all active tokens are recomputed
  - layer `1`: token importance is scored and the recompute subset is chosen
  - later layers: only the selected subset is recomputed

Important:

- In the current implementation, later check layers refine the already surviving subset.
- They do not reopen the full token set.

### `blend_recompute_ratios`

Type: `list[float]`

Typical value:

```yaml
blend_recompute_ratios: [0.16]
```

Meaning:

- Fraction of active tokens selected by the top-k selector at check layers.

Mathematically:

If there are `N` active tokens at the check layer and the ratio is `r`, then:

```text
topk_num = max(1, floor(N * r))
```

With `r = 0.16`, the selector first picks the top `16%` by score.

Notes:

- Boundary-forced tokens, if enabled, are added on top of this top-k set.

## Experiment Parameters In `extra_config`

These are the custom experiment knobs added for selector and fusion studies.

## Selector Parameters

### `blend_selection_strategy`

Type: `str`

Deviation-selector values covered here:

- `"topk_diff_k"`
- `"topk_diff_v"`
- `"topk_diff_kv_sum"`
- `"topk_diff_kv_weighted"`

Meaning:

- Chooses how token importance scores are computed at the check layer.

### `blend_selection_k_weight`

Type: `float`

Used only when:

- `blend_selection_strategy: "topk_diff_kv_weighted"`

Meaning:

- Weight for the K-difference term in the weighted K/V selector.

### `blend_selection_v_weight`

Type: `float`

Used only when:

- `blend_selection_strategy: "topk_diff_kv_weighted"`

Meaning:

- Weight for the V-difference term in the weighted K/V selector.

## Selector Math

Let:

- `K_new` be the recomputed K tensor for the currently active tokens
- `V_new` be the recomputed V tensor
- `K_cache` be the cached K tensor loaded from LMCache
- `V_cache` be the cached V tensor

For token `i`, define:

```text
dK_i = ||K_new[i] - K_cache[i]||_2^2
dV_i = ||V_new[i] - V_cache[i]||_2^2
```

The deviation selectors covered here are:

### `topk_diff_k`

Score:

```text
score_i = dK_i
```

Interpretation:

- Current original selector.
- Rank tokens by K mismatch only.

### `topk_diff_v`

Score:

```text
score_i = dV_i
```

Interpretation:

- Rank tokens by V mismatch only.

### `topk_diff_kv_sum`

Score:

```text
score_i = dK_i + dV_i
```

Interpretation:

- Equal combination of K and V mismatch.

### `topk_diff_kv_weighted`

Score:

```text
score_i = alpha * dK_i + beta * dV_i
```

where:

- `alpha = blend_selection_k_weight`
- `beta = blend_selection_v_weight`

Interpretation:

- Lets you emphasize K more, V more, or balance them.

Selection rule for the four deviation selectors above:

```text
Sk = top topk_num tokens by score
```

where `topk_num = floor(N * recompute_ratio)` up to the `max(1, ...)` guard.

## Boundary Parameters

### `blend_force_chunk_boundary_tokens`

Type: `bool`

Supported values:

- `true`
- `false`

Meaning:

- If `true`, chunk-edge tokens are forced into the selected set in addition to top-k.

Current behavior:

- The selector computes top-k first.
- Then chunk-boundary token indices are unioned with that set.

Mathematically:

```text
S_final = S_topk ∪ S_boundary
```

### `blend_chunk_boundary_token_count`

Type: `int`

Examples:

- `1`
- `2`
- `3`

Meaning:

- Number of tokens taken from each side of each detected chunk.

Interpretation:

- `1` means:
  - first token of chunk
  - last token of chunk
- `2` means:
  - first 2 tokens
  - last 2 tokens
- `3` means:
  - first 3 tokens
  - last 3 tokens

If a chunk is shorter than this width, the width is clipped to the chunk length.

## Fusion Parameters

### `blend_fusion_strategy`

Type: `str`

Fusion values covered here:

- `"overwrite_selected"`
- `"weighted_selected"`
- `"adaptive_weighted_selected"`

Meaning:

- Controls how cached and recomputed K/V are merged for the selected tokens.

### `blend_cached_k_weight`

Type: `float`

Used by:

- `"weighted_selected"`

Meaning:

- Global cached-K mixing weight.

### `blend_cached_v_weight`

Type: `float`

Used by:

- `"weighted_selected"`

Meaning:

- Global cached-V mixing weight.

## Fusion Math

### `overwrite_selected`

Meaning:

- Fully trust recomputed K/V at selected positions.

Mathematically:

```text
K_final[i] = K_new[i]
V_final[i] = V_new[i]
```

for all selected tokens `i`.

### `weighted_selected`

Meaning:

- Use one global constant cached weight for all selected tokens.

Mathematically:

```text
K_final[i] = wK * K_cache[i] + (1 - wK) * K_new[i]
V_final[i] = wV * V_cache[i] + (1 - wV) * V_new[i]
```

where:

- `wK = blend_cached_k_weight`
- `wV = blend_cached_v_weight`

Examples:

- `0.5 / 0.5` means 50% cached + 50% recomputed
- `0.1 / 0.1` means light cached contribution
- `0.0 / 0.2` means overwrite K, but lightly mix cached V

### `adaptive_weighted_selected`

Meaning:

- Compute token-wise cached weights from the mismatch magnitude.
- Larger mismatch means trust recomputed tensors more.
- Smaller mismatch means allow more cached contribution.

Additional parameters:

- `blend_min_cached_k_weight`
- `blend_max_cached_k_weight`
- `blend_min_cached_v_weight`
- `blend_max_cached_v_weight`

For each selected token `i`, define:

```text
dK_i = ||K_new[i] - K_cache[i]||_2^2
dV_i = ||V_new[i] - V_cache[i]||_2^2
```

Then normalize inside the selected set:

```text
nK_i = (dK_i - min(dK)) / (max(dK) - min(dK) + eps)
nV_i = (dV_i - min(dV)) / (max(dV) - min(dV) + eps)
```

Cached weights are then:

```text
wK_i = maxK - nK_i * (maxK - minK)
wV_i = maxV - nV_i * (maxV - minV)
```

and fusion is:

```text
K_final[i] = wK_i * K_cache[i] + (1 - wK_i) * K_new[i]
V_final[i] = wV_i * V_cache[i] + (1 - wV_i) * V_new[i]
```

Interpretation:

- token with larger mismatch gets lower cached weight
- token with smaller mismatch gets higher cached weight

## Logging Parameters

### `blend_log_selection_stats`

Type: `bool`

Meaning:

- Emit per-request selection logs, including:
  - active token count
  - top-k count
  - detected boundary positions
  - overlap
  - final selected count

### `blend_log_fusion_stats`

Type: `bool`

Meaning:

- Emit per-request fusion logs, including:
  - fusion strategy
  - selected token count
  - cached weights or adaptive-weight bounds

## Practical Defaults

### Stable baseline

```yaml
blend_selection_strategy: "topk_diff_k"
blend_fusion_strategy: "overwrite_selected"
blend_check_layers: [1]
blend_recompute_ratios: [0.16]
```

### First selector experiments

```yaml
blend_selection_strategy: "topk_diff_kv_sum"
blend_fusion_strategy: "overwrite_selected"
```

```yaml
blend_selection_strategy: "topk_diff_v"
blend_fusion_strategy: "overwrite_selected"
```

### First adaptive fusion experiment

```yaml
blend_selection_strategy: "topk_diff_k"
blend_fusion_strategy: "adaptive_weighted_selected"
blend_min_cached_k_weight: 0.0
blend_max_cached_k_weight: 0.3
blend_min_cached_v_weight: 0.0
blend_max_cached_v_weight: 0.3
```

## Around High-Divergence Token Experiments

These configs live in `benchmarks/rag/yaml_04_around_high_div/`.
Recommended run order:

1. `plain_ratio_020.yaml` — plain K-deviation top-k with 20% recompute; establishes the first higher-budget baseline.
2. `plain_ratio_024.yaml` — plain K-deviation top-k with 24% recompute; compare against capped neighbor/span variants at the same budget.
3. `plain_ratio_032.yaml` — plain K-deviation top-k with 32% recompute; checks whether accuracy mainly needs more budget.
4. `neighbor_top25_both_cap024.yaml` — start from 16%, expand left and right around only the top 25% highest-divergence selected tokens, capped at 24%.
5. `neighbor_top50_both_cap024.yaml` — same as above, but expand around the top 50% selected tokens, capped at 24%.
6. `span_close_gap1_cap024.yaml` — start from 16%, fill single-token gaps between selected tokens, capped at 24%.
7. `span_close_gap2_cap024.yaml` — start from 16%, fill gaps of up to two tokens between selected tokens, capped at 24%.
8. `neighbor_top50_left_cap024.yaml` — expand only to the previous token around the top 50% selected tokens, capped at 24%.
9. `neighbor_top50_right_cap024.yaml` — expand only to the next token around the top 50% selected tokens, capped at 24%.
10. `neighbor_top25_both_cap032.yaml` — top-25% two-sided expansion with a larger 32% cap.
11. `neighbor_top50_both_cap032.yaml` — top-50% two-sided expansion with a larger 32% cap.
12. `span_close_gap2_cap032.yaml` — gap-closing up to two tokens with a larger 32% cap.
13. `neighbor_all_both_cap048.yaml` — expand left and right around all selected tokens, capped at 48%; useful as the expensive stress test.

## Important Notes

- Old experiments remain valid:
  - if you keep `blend_fusion_strategy: "overwrite_selected"` and `blend_selection_strategy: "topk_diff_k"`, behavior matches the original logic.
- Boundary forcing is additive:
  - it adds tokens beyond top-k, it does not replace top-k.
- In your current workflow, use:
  - `max_local_cpu_size: 320`
  - for new YAMLs as well.
