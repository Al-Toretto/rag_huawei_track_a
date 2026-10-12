# Track A result evaluation

This directory contains the scripts used to collect lexical metrics, inspect
saved answers and run reference-guided Prometheus evaluation. The current
scope covers MuSiQue, 2WikiMQA, SAMSum, DuReader and MultiFieldQA-ZH.

Start with the [repository guide](../../../README.md) for environment setup,
the current result files, portable artifact paths and complete commands. The
[round-7 guide](../07_accuracy_recovery/README.rst) documents collection and
judging for the higher-budget experiment matrix.

## Existing outputs

- [English v5 outputs](outputs/qwen3_chat_nothink_v5_seed42/) contain the saved
  manifest, predictions, judge inputs/outputs and semantic summaries.
- [Chinese outputs](outputs/qwen3_chinese/) contain the corresponding Chinese
  records and summaries.
- [Five-dataset metrics](../results/rag_qwen3_five_datasets_meeting_metrics_current_9622_chinese_judged.xlsx)
  combine the current aggregate results. Earlier files remain separate by run
  name so that model, prompt and generation changes can be distinguished.

Saved manifests may contain machine-local paths. Regenerate or relocate them
using the repository guide before exporting from a different checkout.

## Workflow

1. Collect benchmark CSVs and their matching `benchmark.log` files. Keep each
   experiment's model, prompt, generation settings and request range together.
2. Build the lexical metrics and manifest for the selected runs. The manifest
   identifies exactly which saved predictions will be judged.
3. Export judge input, inspect the rendered prompts and run Prometheus.
   Chinese datasets use the localized QA rubric; SAMSum uses the summarization
   rubric. Use a fresh output path when predictions change.
4. Aggregate the saved judge output and join it to the lexical metrics.
   Compare each method with the full-recomputation reference from the same
   model/prompt experiment series.

| Script | Role |
| --- | --- |
| [build_qwen3_v5_metrics.py](build_qwen3_v5_metrics.py) | Rebuild the English v5 lexical workbook, answer tables and judge manifest from the submission record. |
| [build_chinese_metrics.py](build_chinese_metrics.py) | Collect the Chinese matrix, check generation metadata and write its manifest; optionally join English and semantic summaries. |
| [build_round7_metrics.py](build_round7_metrics.py) | Collect only the runs identified by round-7 submission TSVs, then join complete judge results. |
| [export_judge_input.py](export_judge_input.py) | Export the selected manifest as JSONL predictions and rendered-prompt records. |
| [run_local_judge.py](run_local_judge.py) | Run the local reference-guided evaluator or inspect prompts with `--dry-run`. |
| [aggregate_judge_results.py](aggregate_judge_results.py) | Aggregate saved judge scores by dataset and method. |
| [build_answer_tables.py](build_answer_tables.py) | Inspect the older experiment collection; this is not the default selector for the current five-dataset matrix. |

All commands run from the repository root. Pass explicit input, output and
manifest paths for the chosen experiment series. For example, after preparing
the English v5 manifest's paths:

```bash
python benchmarks/rag/results_eval/export_judge_input.py \
  --manifest benchmarks/rag/results_eval/outputs/qwen3_chat_nothink_v5_seed42/manifest.csv \
  --output benchmarks/rag/results_eval/outputs/new_evaluation/judge_input.jsonl

python benchmarks/rag/results_eval/run_local_judge.py \
  --input benchmarks/rag/results_eval/outputs/new_evaluation/judge_input.jsonl \
  --limit 3 --dry-run
```

For actual model judging, select `--backend transformers` and provide the
locally available Prometheus model. The runner defaults to local-only model
loading for that backend. See the repository guide for the pinned model and
Slurm command. `--check-only` produces diagnostics rather than semantic model
ratings.

## Interpretation and reruns

MuSiQue, 2WikiMQA and MultiFieldQA-ZH use Qwen-tokenizer F1; SAMSum and DuReader
use ROUGE-L. Lexical scores and reference-guided semantic scores measure
different properties and should remain separate outcomes.

The semantic evaluator receives the question/reference and prediction for QA,
or the reference summary and prediction for SAMSum. Method labels, lexical
scores and latency are kept outside its prompt. It does not independently
check predictions against the source passages or dialogue.

Resume matching uses the dataset, request ID and experiment label. If the saved
prediction changes under the same label, an existing judge row can be skipped;
use a fresh judge-output file for replacement predictions. Write rebuilt
artifacts to new paths and preserve the saved result series used for comparison.
