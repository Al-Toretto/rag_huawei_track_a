# Local LLM Judge Methodology For CacheBlend RAG Experiments

This note summarizes the LLM-as-a-judge evaluation pipeline we added for the
LMCache CacheBlend RAG experiments. It is intended as a partner-facing
description of what was done, why, and how to interpret the output tables.

## Why We Added An LLM Judge

The original RAG benchmark already reports word-level F1 and tokenizer-level
F1. Those metrics are useful and should stay in the results. However, they are
sensitive to answer formatting:

- `X`, `Answer: X`, and `The answer is X` can receive different F1 scores even
  though they are semantically the same.
- The existing QA parser keeps only the first generated line, so an answer that
  appears after a newline can be missed.
- Some outputs contain the correct answer but later contradict it or end with a
  different final answer.
- Some outputs hit the max-token limit and repeat prompt patterns, especially
  on SAMSum.

The goal of the LLM judge is therefore not to replace F1, but to add a
reference-guided semantic correctness metric and separate generation-pathology
diagnostics.

## Model Choice

We use:

```text
prometheus-eval/prometheus-7b-v2.0
```

This is Prometheus 2 7B, an open-source evaluator model. We chose it because:

- it is designed for model-output evaluation rather than ordinary generation;
- it supports rubric-based absolute grading;
- it can run locally without OpenAI API calls or paid judge services;
- it runs on one cluster H200 GPU; although the weights fit in 32 GB, the
  current judge PyTorch/CUDA environment is not compatible with the V100 pool;
- it gives a reproducible local judge setup once the model and scripts are
  fixed.

We ran it locally/cluster-side through Hugging Face Transformers. The judge is
blinded to the method name: it does not receive whether a prediction came from
Full Recompute, CacheBlend, a neighbor experiment, etc.

## Data Given To The Judge

The judge input is prepared by:

```bash
python3 benchmarks/rag/results_eval/export_judge_input.py
```

The no-Chinese subset used for the current run was prepared with:

```bash
python3 benchmarks/rag/results_eval/export_judge_input.py \
  --dataset 2wikimqa_longbench \
  --dataset musique_longbench \
  --dataset musique_s \
  --dataset samsum_longbench \
  --dataset wikimqa_s \
  --output benchmarks/rag/results_eval/outputs/judge_input_no_zh.jsonl
```

For QA datasets, each judged item contains only:

```json
{
  "question": "...",
  "gold_answers": ["..."],
  "prediction": "..."
}
```

For SAMSum, each judged item contains only:

```json
{
  "reference_summary": "...",
  "prediction": "..."
}
```

The judge does not receive retrieval contexts, latency, TTFT, F1 values, run
names, or method labels. Those fields are kept separately in
`analysis_metadata` for aggregation after judging.

## Prompt Style

Prometheus 2 is not primarily a JSON-form judge. It expects an absolute-grading
format with a rubric and returns:

```text
Feedback: ... [RESULT] N
```

where `N` is an integer from 1 to 5.

The local runner therefore uses Prometheus-native prompts for Prometheus models
and parses `[RESULT]` back into our normalized JSON output.

The system prompt used for Prometheus-native judging is:

```text
You are a fair judge assistant tasked with providing clear, objective feedback
based on specific criteria, ensuring each assessment reflects the absolute
standards set for performance.
```

The user prompt contains:

- an instruction describing the task;
- the model response to evaluate;
- the reference answer or reference summary;
- a task-specific rubric;
- a required output format: `Feedback: ... [RESULT] 1-5`.

## QA Rubric

The QA rubric is used for:

```text
2WikiMQA, WikiMQA, MuSiQue
```

and can also be used for DuReader / MultiFieldQA if we include those datasets.

The key rule is:

```text
A bare concise answer like "X" is fully correct.
Harmless wrappers such as "Answer: X" or "The answer is X" are allowed but are
not required and must not be preferred over a bare answer.
```

The QA scoring rubric is:

```text
Score 5:
The response gives the same answer as the reference answer(s) and contains no
contradiction. Bare answers and harmless wrappers are both fully correct.

Score 4:
The response gives the main correct answer but is slightly incomplete, mildly
ambiguous, or missing a minor required component.

Score 3:
The response captures some important part of the reference answer but misses a
central component or is too ambiguous for higher credit.

Score 2:
The response mentions the reference answer or relevant evidence, but the final
answer is wrong, contradictory, or not clearly selected.

Score 1:
The response gives the wrong answer, no answer, a refusal, an unrelated answer,
or a contradiction that makes the answer unusable.
```

Important examples:

- `English.` vs gold `England` should be score 5.
- `1986`, `Answer: 1986`, and `The answer is 1986` should all be score 5.
- If the prediction mentions the gold answer but later concludes with a
  different answer, it should not be score 5.
- A no-answer response is incorrect when the gold answer exists.

## SAMSum Rubric

SAMSum is judged separately because it is summarization, not answer extraction.

The SAMSum scoring rubric is:

```text
Score 5:
The response captures all or nearly all central information in the reference
summary with no important hallucination.

Score 4:
The response captures the main information but misses minor details, adds minor
harmless information, or is somewhat verbose or redundant without distorting
meaning.

Score 3:
The response captures some important information but misses at least one
central point, is overly vague, or contains a moderate unsupported detail.

Score 2:
The response has a small amount of relevant content but is dominated by
omissions, hallucination, repetition, prompt continuation, or degeneration.

Score 1:
The response is unrelated, empty, mostly hallucinated, mostly copied prompt
text, mostly repeated text, or contains no usable summary.
```

This is important because SAMSum outputs often repeat prompt patterns such as:

```text
Dialogue: ...
Summary: ...
# #
Dialogue: ...
Summary: ...
```

Those pathologies are also detected by deterministic rule-based diagnostics.

## Rule-Based Diagnostics

In addition to the LLM judge, we compute deterministic diagnostics in Python.
These do not depend on Prometheus and are reproducible:

- empty output;
- very short output;
- repeated n-grams;
- repeated lines;
- prompt-continuation markers such as repeated `Dialogue:` / `Summary:`;
- max-token degeneration when `generation_token_cnt` reaches the threshold.

These fields are reported separately because they are generation-quality
pathologies, not the same thing as semantic correctness.

## Main Scripts And Files

Answer-table extraction:

```bash
python3 benchmarks/rag/results_eval/build_answer_tables.py
```

Judge-input preparation:

```bash
python3 benchmarks/rag/results_eval/export_judge_input.py
```

Local Prometheus judge runner:

```bash
python3 benchmarks/rag/results_eval/run_local_judge.py
```

Shutdown/corruption repair:

```bash
python3 benchmarks/rag/results_eval/repair_judge_outputs.py
```

Aggregation:

```bash
python3 benchmarks/rag/results_eval/aggregate_judge_results.py
```

Validation against manual labels:

```bash
python3 benchmarks/rag/results_eval/validate_judge_results.py
```

Cluster Slurm script:

```text
benchmarks/rag/sbash_runs/run_local_judge_prometheus.sbatch
```

Current no-Chinese input:

```text
benchmarks/rag/results_eval/outputs/judge_input_no_zh.jsonl
```

Current no-Chinese judge output:

```text
benchmarks/rag/results_eval/outputs/judge_outputs_no_zh_repaired.jsonl
```

First two datasets summary:

```text
benchmarks/rag/results_eval/outputs/judge_summary_first2_by_dataset_method.csv
benchmarks/rag/results_eval/outputs/judge_summary_first2_by_method.csv
```

## Summary CSV Column Descriptions

The aggregation script writes two table types:

- by dataset and experiment;
- by experiment pooled across selected datasets.

The by-dataset table also has a `dataset` column. The pooled table starts at
`experiment`. The columns below match the CSV output names. In spreadsheets we
sometimes display the names with spaces, for example `mean_semantic_score` as
`mean semantic score`.

| Column | Spreadsheet label | Description |
|---|---|---|
| `dataset` | dataset | Dataset name. Present only in the by-dataset table. |
| `experiment` | experiment | Display label for the run/method, for example `Full Recompute`, `Blend (16%) layer 1`, or `neighbor top50 both cap024`. |
| `n` | n | Number of judge-output rows in this group, including rows with parse errors. This is the denominator for diagnostics computed over all predictions. |
| `n_parsed` | n parsed | Number of rows where the judge output was successfully parsed into structured fields. This is the denominator for judge-label rates. |
| `parse_error_rate` | parse error rate | Fraction of rows where the judge output could not be parsed. For Prometheus, this usually means the model produced feedback but did not include a valid `[RESULT]` score. |
| `mean_semantic_score` | mean semantic score | Mean normalized semantic score over parsed rows. This is on a 0-to-1 scale. For Prometheus, scores are mapped as 5 -> 1.00, 4 -> 0.75, 3 -> 0.50, 2 -> 0.25, 1 -> 0.00. |
| `correct_rate` | correct rate | Fraction of parsed rows with semantic label `correct`. For QA this means the answer is semantically equivalent to the gold answer. For SAMSum this means a high-quality semantic summary match. |
| `mostly_correct_rate` | mostly correct rate | Fraction of parsed rows with semantic label `mostly_correct`. The output gets the main point right but has a minor omission, ambiguity, or harmless extra information. |
| `partial_rate` | partial rate | Fraction of parsed rows with semantic label `partially_correct`. The output contains some useful correct content but misses central information. |
| `minimal_rate` | minimal rate | Fraction of parsed rows with semantic label `minimally_correct`. The output has only a small amount of relevant content or mentions useful evidence but gives an unclear, wrong, or unusable final answer. |
| `incorrect_rate` | incorrect rate | Fraction of parsed rows with label `incorrect` or `no_usable_summary`. For older JSON-style judge outputs, `no_answer` is also counted here. |
| `no_answer_rate` | no answer rate | Fraction of parsed rows labeled `no_answer` or `no_usable_summary`. For QA this is explicit no-answer/refusal behavior. For SAMSum this is an unusable or effectively missing summary. |
| `contradiction_rate` | contradiction rate | Fraction of parsed rows where the judge explicitly returned `contradicts_gold_answer = true`. This is populated for JSON-style judge outputs. For Prometheus-native outputs it can be blank because Prometheus returns feedback plus `[RESULT]`, not every diagnostic boolean. |
| `judge_repetition_rate` | judge repetition rate | Fraction of parsed rows where the LLM judge itself explicitly returned `repetition_detected = true`. This is mainly relevant for JSON-style judge outputs. In the Prometheus-native run this is usually blank, because Prometheus returns only feedback plus `[RESULT]`, not this boolean field. |
| `judge_prompt_continuation_rate` | judge prompt continuation rate | Fraction of parsed rows where the LLM judge itself explicitly returned `prompt_continuation_detected = true`. This is also mainly relevant for JSON-style judge outputs and is usually blank for Prometheus-native outputs. |
| `rule_repetition_rate` | rule repetition rate | Fraction of all rows where deterministic Python checks detected repetitive degeneration in the prediction. This includes repeated n-grams, repeated lines, or prompt-continuation patterns. It does not depend on the LLM judge and is the preferred repetition metric for Prometheus outputs. |
| `rule_prompt_continuation_rate` | rule prompt continuation rate | Fraction of all rows where deterministic Python checks detected prompt-pattern continuation in the prediction. Examples include generated text continuing with markers like `Dialogue:`, `Summary:`, `Question:`, `Answer:`, or the Blend separator pattern. |
| `max_token_rate` | max token rate | Fraction of all rows whose `generation_token_cnt` reached the configured max-token threshold. This flags likely max-token degeneration or outputs cut off by the generation limit. |
| `empty_output_rate` | empty output rate | Fraction of all rows with an empty prediction after stripping whitespace. |
| `very_short_output_rate` | very short output rate | Fraction of all non-empty rows with at most 2 whitespace-separated words in the prediction. This is diagnostic only: short outputs are not automatically bad, because many QA answers should be short. |
| `mean_f1_word` | F1 word | Mean existing word-level F1 from the benchmark result CSVs. This is a lexical overlap metric from the original benchmark, not the LLM judge. For SAMSum it is blank because SAMSum uses ROUGE-L in the `quality` column instead. |
| `mean_f1_tokenizer` | F1 tokenizer | Mean existing tokenizer-level F1 from the benchmark result CSVs. This is also an original benchmark lexical metric, not the LLM judge. For SAMSum it is blank. |

### Judge Versus Rule Diagnostics

The columns with `judge_` are diagnostics that the LLM judge itself may return.
They are useful when using the JSON-style judge prompt. In the current
Prometheus-native run, these fields are often blank because Prometheus returns
only free-text feedback plus `[RESULT]`.

The columns with `rule_` are deterministic diagnostics computed before or after
judging from the raw prediction text and metadata. They are reproducible, do not
depend on Prometheus, and are the preferred pathology indicators for the current
Prometheus results.

The exact rule logic is:

```text
very_short_output:
  prediction is non-empty and has <= 2 whitespace-separated words

repeated_ngram_detected:
  the most common normalized 3-, 4-, 5-, 6-, or 8-word n-gram appears >= 4 times

repeated_line_detected:
  the most common non-empty normalized line appears >= 3 times

prompt_continuation_detected:
  either:
    - at least 1 "Dialogue:" marker and at least 2 "Summary:" markers; or
    - at least 2 "Question:" markers and at least 2 "Answer:" markers; or
    - at least 2 Blend separator patterns matching "# #"

repetition_detected:
  repeated_ngram_detected
  OR repeated_line_detected
  OR prompt_continuation_detected

max_token_detected:
  generation_token_cnt >= 2048
```

The rule rates use all rows in the group as the denominator. For example, if
20 out of 50 predictions have `rule_diagnostics.repetition_detected = true`,
then `rule_repetition_rate = 0.40`.

The judge rates use only rows where the judge output was parsed and where that
specific boolean field exists. For example, `judge_repetition_rate` is computed
from `judge_output_parsed.repetition_detected`, not from the raw prediction
text. That is why the `judge_` fields are not the main ones for the Prometheus
run.

## Why There Are Two Score Scales

Prometheus natively returns an integer score from 1 to 5:

```text
1, 2, 3, 4, 5
```

We keep this raw score in the judge output as:

```text
native_prometheus_score
```

This is useful for auditability because it records exactly what the evaluator
model produced.

For analysis and paper tables, we convert the 1-to-5 score to a normalized
semantic score from 0 to 1:

```text
5 -> 1.00
4 -> 0.75
3 -> 0.50
2 -> 0.25
1 -> 0.00
```

This normalized field is:

```text
semantic_score
```

and group averages are reported as:

```text
mean_semantic_score
```

The reason for keeping both is simple:

- `native_prometheus_score` preserves the original model output;
- `semantic_score` makes the result easier to average, plot, and compare with
  other 0-to-1 metrics such as F1.

So the 1-to-5 score is the judge model's native rubric score, while the 0-to-1
score is our normalized analysis score.

## Recommended Reporting

Do not report only one number. For the paper or partner update, report:

- existing word F1;
- existing tokenizer F1;
- mean semantic score;
- correct / mostly-correct / incorrect rates;
- rule-based repetition rate;
- prompt-continuation rate;
- max-token rate;
- TTFT / latency metrics from the original benchmark.

This separates semantic quality from output-format and degeneration issues,
which is the main reason we added the judge in the first place.
