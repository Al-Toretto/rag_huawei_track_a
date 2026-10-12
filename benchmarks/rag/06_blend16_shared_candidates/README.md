# Round 6: selection and fusion configurations

This folder contains 20 Track A configurations built around a nominal 16%
selection budget. They compare fixed and adaptive KV fusion, document-edge
augmentation, neighboring tokens and progressive filtering. The evaluation
covers MuSiQue, 2WikiMQA, SAMSum, DuReader and MultiFieldQA-ZH.

These configurations provide low-budget controls for the current accuracy-first
study. Budgets of 24% and above remain candidates when they improve quality;
TTFT is measured alongside quality and optimized after a configuration is
selected. See the [round-7 study](../07_accuracy_recovery/README.rst) for the
higher-budget matrix and the [repository guide](../../../README.md) for setup,
results and execution instructions.

## Baseline and selection semantics

The [baseline](../lmcache_blend.yaml) checks at layer 1, selects 16% of active
positions by squared key deviation and overwrites selected keys and values.
Layer indices are zero-based. The selection fraction applies to the active
blended prefix, not to the complete request or total GPU computation.

Before the first check, every active prefix token is recomputed. Later checking
layers refine only the surviving positions; they cannot restore dropped tokens.
Boundary augmentation adds the first and last token of each independently
cached document segment. These additions can exceed the nominal top-k budget.

Fusion weights describe the cached contribution at selected positions:

```text
K_final = wK * K_cached + (1 - wK) * K_fresh
V_final = wV * V_cached + (1 - wV) * V_fresh
```

Unselected positions retain cached states. A 10–10 setting therefore retains
10% cached K and V; it does not mean 10% recomputation.

## Fixed fusion and neighbor controls

All checks in this table are at layer 1. Neighbor variants begin with K-only
16% selection, add the next token to the highest-ranked half of the selected
positions and limit the final set to the stated cap. A cap can remain
underfilled.

| Configuration | Selection and update |
| --- | --- |
| [weighted 05–05 + boundary 1](blend16_layer1_weighted_05_05_boundary1.yaml) | K-only 16% plus document edges; retain 5% cached K and V. |
| [weighted 10–10 + boundary 1](blend16_layer1_weighted_10_10_boundary1.yaml) | K-only 16% plus document edges; retain 10% cached K and V. |
| [weighted 15–15 + boundary 1](blend16_layer1_weighted_15_15_boundary1.yaml) | K-only 16% plus document edges; retain 15% cached K and V. |
| [K+V sum + weighted 10–10](blend16_layer1_kvsum_weighted10.yaml) | Rank by the sum of squared K and V deviations; select 16%; retain 10% cached K and V. |
| [right neighbor, cap 20% + weighted 10–10](blend16_layer1_neighbor_top50_right_cap020_weighted10.yaml) | K-only selection with right-neighbor expansion and a 20% cap; retain 10% cached K and V. |
| [right neighbor, cap 24% + weighted 10–10](blend16_layer1_neighbor_top50_right_cap024_weighted10.yaml) | The same expansion rule with a 24% cap; retain 10% cached K and V. |

## Magnitude-adaptive fusion

These settings normalize K and V deviations within the selected set. A larger
deviation gives a smaller cached contribution. Each boundary setting uses
K-only 16% selection at layer 1 plus one token at each document edge.

| Configuration | Cached-weight ranges |
| --- | --- |
| [adaptive K0–5 / V0–15 + boundary 1](blend16_layer1_adaptive_k005_v015_boundary1.yaml) | K: 0–5%; V: 0–15%. |
| [adaptive K0–5 / V0–20 + boundary 1](blend16_layer1_adaptive_k005_v020_boundary1.yaml) | K: 0–5%; V: 0–20%. |
| [adaptive K0–10 / V0–20 + boundary 1](blend16_layer1_adaptive_k010_v020_boundary1.yaml) | K: 0–10%; V: 0–20%. |
| [right neighbor, cap 20% + adaptive K0–5 / V0–20](blend16_layer1_neighbor_top50_right_cap020_adaptive_k005_v020.yaml) | K: 0–5%; V: 0–20%; right-neighbor expansion without boundary augmentation. |

## Rank-adaptive fusion

These settings use the relative rank of selected-token deviations:

```text
cached_weight = min_weight + (max_weight - min_weight) * (1 - percentile) ** gamma
```

The most divergent selected tokens receive the smallest cached weights. All
four settings use K-only 16% selection at layer 1.

| Configuration | Update and boundary setting |
| --- | --- |
| [rank K0–5 / V0–20](blend16_layer1_rank_adaptive_k005_v020.yaml) | K: 0–5%; V: 0–20%; exponent 1; no boundary additions. |
| [rank K0–5 / V0–20 + boundary 1](blend16_layer1_rank_adaptive_k005_v020_boundary1.yaml) | The same weights and exponent, with document-edge augmentation. |
| [rank K0–5 / V0–25 + boundary 1](blend16_layer1_rank_adaptive_k005_v025_boundary1.yaml) | Increase the maximum cached V contribution to 25%; retain the K range and boundary setting. |
| [rank K0–5 / V0–20, exponent 2 + boundary 1](blend16_layer1_rank_adaptive_k005_v020_gamma2_boundary1.yaml) | Use exponent 2 for the cached-weight curve; retain the weight ranges and boundary setting. |

## Progressive filtering

The later ratio is applied to the surviving set. Products below describe the
nominal selected fraction before boundary additions. Unless stated otherwise,
fusion uses rank-adaptive cached K weights of 0–5%, V weights of 0–20% and
exponent 1.

| Configuration | Checking layers and nominal fractions |
| --- | --- |
| [L1→L2, 18%→16%](blend16_multilayer_l1_l2_18_to_16_rank_adaptive_k005_v020.yaml) | Layers 1 and 2; 0.18 × 0.89 = 0.1602. |
| [L1→L3, 18%→16%](blend16_multilayer_l1_l3_18_to_16_rank_adaptive_k005_v020.yaml) | Layers 1 and 3; 0.18 × 0.89 = 0.1602. |
| [L1→L3, 18%→16% + boundary 1](blend16_multilayer_l1_l3_18_to_16_rank_adaptive_k005_v020_boundary1.yaml) | The same schedule with document-edge augmentation at the checks. |
| [L1→L3, 18%→16% + weighted 10–10](blend16_multilayer_l1_l3_18_to_16_weighted10.yaml) | The same schedule with fixed 10% cached K and V contributions. |
| [L1→L3, 20%→16%](blend16_multilayer_l1_l3_20_to_16_rank_adaptive_k005_v020.yaml) | Layers 1 and 3; 0.20 × 0.80 = 0.1600. |
| [L1→L3→L5, 20%→18%→16%](blend16_multilayer_l1_l3_l5_20_to_18_to_16_rank_adaptive_k005_v020.yaml) | Layers 1, 3 and 5; 0.20 × 0.90 × 0.89 = 0.1602. |

## Run a configuration

Run from the repository root after preparing the serving environment and model
snapshot described in the repository guide. For example:

```bash
export MODEL_SNAPSHOT=/absolute/path/to/Qwen3-32B
export MODEL_NAME=Qwen/Qwen3-32B

sbatch \
  --export="ALL,EXPERIMENT_NAME=06_musique_rank_adaptive,CONFIG_FILE=$PWD/benchmarks/rag/06_blend16_shared_candidates/blend16_layer1_rank_adaptive_k005_v020.yaml,DATASET_PATH=$PWD/benchmarks/rag/inputs/musique_longbench.json" \
  benchmarks/rag/sbash_runs/run_lmcache_experiment_h200.sbatch
```

Use a distinct experiment name for each configuration/dataset run. Apply the
same task framing, decoding settings and evaluated request range to the reuse
method and its full-recomputation reference. Compare both lexical and semantic
quality across all five tasks before selecting a common configuration.
