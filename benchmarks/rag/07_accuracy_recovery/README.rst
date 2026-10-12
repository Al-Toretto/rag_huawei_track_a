RAG Blend experiments: round 7
==============================

This round seeks one configuration with accuracy close to full recompute on
MuSiQue, 2WikiMQA, SAMSum, MultiFieldQA zh, and DuReader. Run all twelve
configurations on all five datasets: 60 jobs. The YAMLs and this explanation
are in ``benchmarks/rag/07_accuracy_recovery/``. They use existing Blend
features; no serving implementation changes are required. Existing baseline
values are reused; neither standard Blend 16% nor plain vLLM is submitted.

Evidence and scope
------------------

The design uses
``rag_qwen3_five_datasets_meeting_metrics_current_9622_chinese_judged.xlsx``
in ``benchmarks/rag/results/``, the previous YAMLs, the current Blend source,
and the checked-in submission scripts. The workbook contains
65 methods on each of five datasets. Historical logs and individual result
CSVs were not used for this design.

The table shows intrinsic score on a 0--100 scale followed by mean judge
score on a 0--1 scale. Intrinsic score is F1 for the three QA datasets and
ROUGE-L for SAMSum and DuReader. All Blend rows here use K-deviation
selection at layer 1 and overwrite fusion.

.. list-table:: Main budget evidence: intrinsic score / judge score
   :header-rows: 1
   :widths: 24 19 19 19 19

   * - Dataset
     - Full recompute
     - Blend 16%
     - Blend 24%
     - Blend 32%
   * - MuSiQue
     - 43.557 / 0.465
     - 44.133 / 0.445
     - 47.830 / 0.505
     - 44.954 / 0.470
   * - 2WikiMQA
     - 51.901 / 0.640
     - 37.019 / 0.525
     - 45.870 / 0.615
     - 51.002 / 0.610
   * - SAMSum
     - 38.511 / 0.670
     - 38.459 / 0.698
     - 38.023 / 0.693
     - 39.133 / 0.667
   * - MultiFieldQA zh
     - 61.196 / 0.810
     - 58.943 / 0.820
     - 60.745 / 0.820
     - 62.407 / 0.805
   * - DuReader
     - 22.931 / 0.730
     - 24.308 / 0.745
     - 23.999 / 0.735
     - 23.317 / 0.735

Interpretation:

* More recomputation is the strongest observed direction for 2WikiMQA.
  Moving from 16% to 32% recovers 13.983 F1 points and leaves a 0.899-point
  gap to full recompute. The judge gap shrinks from 0.115 to 0.030.
* Plain 24% has the highest five-dataset macro judge score among the
  workbook's shared methods: 0.67354 versus 0.66300 for full recompute.
  Its 2WikiMQA F1 still trails full recompute by 6.031 points, so a macro
  judge average alone would select a method with a substantial remaining gap.
* Results are not monotonic. The 24% to 32% judge change on 2WikiMQA is
  -0.005 despite higher F1. With 50 examples and quarter-point judge grades,
  0.005 can represent one grading step on one example. The workbook does
  not contain the paired scores needed for a confidence interval.
* V-deviation selection at 16% improves both English QA judge scores
  relative to K selection: MuSiQue 0.445 to 0.500 and 2WikiMQA 0.525 to
  0.575. It reduces SAMSum and DuReader judge scores, so its higher-budget
  probe must be checked on all five datasets.
* Selecting at layer 2 rather than layer 1 improves MuSiQue at 16% to
  F1 47.664 and judge 0.505, while 2WikiMQA remains at F1 41.927 and judge
  0.550. A later check combined with 32% tests whether better early
  contextualization helps at the higher budget.
* Most round-6 rank-adaptive, neighbor, and narrow cascade variants do not
  close the 2WikiMQA gap. Weighted 15/15 with one segment-edge token is the
  strongest round-6 shared candidate by worst judge deficit; its 2WikiMQA
  F1 43.825 and judge 0.570 still motivate increasing its budget.

SAMSum coverage needs special care. The historical Blend runs contain 48
examples and full recompute contains 50. The recorded comparison for full
recompute on the same first 48 examples has ROUGE-L about 38.0389 and judge
0.666667. Thus historical Blend 24% is essentially equal on ROUGE-L and
higher on judge on matched coverage. Round 7 explicitly evaluates all
50 examples for every new configuration. Existing full-recompute values
already cover the same 50 request IDs. The historical Blend-16% SAMSum
reference still has 48 examples, so comparisons to that reference have
different coverage.

Blend 100% is a pipeline check, rather than an empirical guarantee that
larger budgets converge exactly to the workbook's full-recompute outputs.
English QA F1 matches at 100%, but MultiFieldQA zh F1 differs by +3.337
points and matched SAMSum ROUGE-L differs by about -0.243 points. Compare
round 7 against the existing plain-vLLM full-recompute values.

The historical 32% TTFT is already about 1.06, 0.99, 1.08, 0.86, and 0.98
times full recompute for MuSiQue, 2WikiMQA, SAMSum, MultiFieldQA zh, and
DuReader respectively. Higher budgets may lose the latency benefit; this
round intentionally prioritizes accuracy. The measured request throughput
is constrained by the offered QPS of 0.25. It does not measure maximum
serving capacity. The unusually slow historical MuSiQue 16% TTFT should
be treated cautiously when interpreting historical latency comparisons.

Experiment matrix
-----------------

All plain variants use overwrite fusion without added segment-edge tokens.
Only the final configuration uses cached K/V weights of 0.15 and adds one
token at each segment edge. Layers are zero-based.
That configuration tests the established fusion/edge combination; it does not
attribute an effect to either change independently.

.. list-table:: Group A: budget, selector, and selection-layer comparisons
   :header-rows: 1
   :widths: 43 12 45

   * - YAML stem
     - Budget
     - Question answered
   * - ``a01_k_l1_r024``
     - 24%
     - Does the strongest historical shared judge result reproduce?
   * - ``a02_k_l1_r032``
     - 32%
     - Does near-full 2WikiMQA F1 reproduce on matched coverage?
   * - ``a03_k_l1_r040``
     - 40%
     - Does more budget close the remaining accuracy gaps?
   * - ``a04_k_l1_r048``
     - 48%
     - Is there further benefit beyond 40%?
   * - ``a05_v_l1_r032``
     - 32%
     - Does V-deviation selection beat K at the same budget?
   * - ``a06_k_l2_r032``
     - 32%
     - Does one more full-context layer before selection help?

.. list-table:: Group B: finer budgets and additional combinations
   :header-rows: 1
   :widths: 43 12 45

   * - YAML stem
     - Budget
     - Question answered
   * - ``b01_k_l1_r028``
     - 28%
     - Does 28% improve the trade-off between 24% and 32%?
   * - ``b02_k_l1_r036``
     - 36%
     - Can 36% recover accuracy while remaining below 40%?
   * - ``b03_k_l1_r060``
     - 60%
     - Does 60% close deficits left at 40% or 48%?
   * - ``b04_k_l3_r032``
     - 32%
     - Does delaying selection to layer 3 help at 32%?
   * - ``b05_v_l1_r040``
     - 40%
     - Does V selection benefit from increasing the budget to 40%?
   * - ``b06_k_l1_r032_weighted15_boundary1``
     - 32% + edges
     - Test the best round-6 fusion/edge combination at a higher budget.

The recompute ratio is a fraction of the currently active blended cached
prefix, including gaps inside that prefix. The final question or target
suffix is handled separately by vLLM. The ratio is neither a fraction of the
complete prompt nor a measurement of total GPU work. Before the first
check, every active prefix token is recomputed. Later check layers can
only refine surviving tokens; they cannot recover tokens dropped earlier.

Neighbor caps in the old YAMLs are ceilings and can remain underfilled.
They cannot be compared to an equal plain top-k rate without measured
selection counts. Weighted fusion preserves some cached KV even for
selected tokens, so it has a different accuracy limit from overwrite.

Choose one shared configuration by checking the largest per-dataset
deficits first, then macro averages, then TTFT among similarly accurate
candidates. Suggested screening targets are no more than one intrinsic
score point and 0.02 judge score below existing full recompute on any dataset;
these are practical targets, not significance thresholds. Keep F1/ROUGE-L
and judge visible as separate outcomes. Recheck the shortlist on more
examples or fresh evaluation examples before claiming it generalizes.

Copy and submit
---------------

From the local repository root, copy the new files. The remote repository
is assumed to contain the same serving code and existing scripts.

.. code-block:: bash

   rsync -avR -e "ssh -p 2222" \
     benchmarks/rag/07_accuracy_recovery/ \
     benchmarks/rag/sbash_runs/submit_round7.sh \
     benchmarks/rag/results_eval/build_round7_metrics.py \
     a.tomat@cluster.hpc.hse.ru:/home/a.tomat/work/LMCache/

   ssh -p 2222 a.tomat@cluster.hpc.hse.ru
   cd ~/work/LMCache

Preview and submit all twelve configurations on all five datasets:

.. code-block:: bash

   DRY_RUN=1 bash benchmarks/rag/sbash_runs/submit_round7.sh
   DRY_RUN=0 bash benchmarks/rag/sbash_runs/submit_round7.sh
   squeue -u a.tomat

This submits 60 jobs: twelve configurations on each of five datasets.
There are no baseline jobs. Each job evaluates request IDs 0--49.
The submitter prints and saves the exact job/run mapping in a TSV under
``benchmarks/rag/07_accuracy_recovery/submissions/``. Save that path for
collection. A dry run writes no submission TSV and submits no jobs.

The wrapper pins the existing Qwen3-32B snapshot, non-thinking chat template,
sampling temperature 0.7, top-p 0.8, top-k 20, min-p 0, generation seed 42,
QPS 0.25, model length 30000, batch token limit 16384, and 320 GiB cache.
Output caps remain 32 tokens for MuSiQue and 2WikiMQA, 256 for SAMSum and
MultiFieldQA zh, and 1024 for DuReader. These settings match the existing
baseline runs.

An optional infrastructure smoke uses only two examples per job and a
separate run prefix. It cannot establish accuracy:

.. code-block:: bash

   SMOKE_TEST=1 CONFIG_FILES=a02_k_l1_r032 DATASETS="2wikimqa samsum" \
     DRY_RUN=0 bash benchmarks/rag/sbash_runs/submit_round7.sh

If a smaller subset is needed later, select its YAML stems explicitly:

.. code-block:: bash

   CONFIG_SET=followup CONFIG_FILES=b01_k_l1_r028,b02_k_l1_r036 \
     DRY_RUN=0 bash benchmarks/rag/sbash_runs/submit_round7.sh

The default ``CONFIG_SET=all`` submits all twelve configurations (60 jobs).
``CONFIG_SET=pilot`` selects the six ``a`` files (30 jobs), and
``CONFIG_SET=followup`` selects the six ``b`` files (30 jobs). These names
remain convenient grouping options; the full round runs both groups.
Baseline submission is disabled for every group. Use the same YAML on every
dataset when confirming a shared method.

Collect and judge
-----------------

After all jobs in a submission finish, activate the benchmark's existing
Python environment on the cluster. Set ``SUBMISSION`` to the TSV printed
by the submitter. Use the same output directory to resume this exact judge
input; use a fresh directory if predictions are replaced.

.. code-block:: bash

   cd ~/work/LMCache
   conda activate lmcache
   SUBMISSION=benchmarks/rag/07_accuracy_recovery/submissions/REPLACE_WITH_PRINTED_NAME.tsv
   JUDGE_DIR="$PWD/benchmarks/rag/results_eval/outputs/round7_seed42_all"

   python benchmarks/rag/results_eval/build_round7_metrics.py \
     --submission "$SUBMISSION" --output-dir "$JUDGE_DIR" \
     --reference-metrics benchmarks/rag/07_accuracy_recovery/reference_metrics.csv

   python benchmarks/rag/results_eval/export_judge_input.py \
     --manifest "$JUDGE_DIR/manifest.csv" \
     --output "$JUDGE_DIR/judge_input.jsonl"

The collector reads only result CSVs identified by the supplied TSVs. It
requires exactly 50 successful requests with unique IDs 0--49 and the
recorded generation settings. Successful empty outputs remain in the
evaluation. It writes intrinsic metrics and a manifest accepted by the
existing judge exporter. The full round contains 3,000 judge rows.

``reference_metrics.csv`` records the existing full-recompute and standard
Blend-16% values from the workbook, including dataset, metric units, coverage,
seed, and source. The collector uses the five full-recompute references for
accuracy deltas and shared-configuration rankings. Reference rows are never
added to the judge input, so they are not judged again. TTFT ratios compare
against historical runs. The 48-case historical SAMSum Blend reference
remains explicitly labeled with its original coverage.

Submit the existing pinned Prometheus judge with explicit paths:

.. code-block:: bash

   sbatch --parsable \
     --export="ALL,REPO_DIR=$PWD,INPUT_FILE=$JUDGE_DIR/judge_input.jsonl,OUTPUT_FILE=$JUDGE_DIR/judge_outputs.jsonl,MODEL_NAME=/home/a.tomat/.cache/huggingface/hub/models--prometheus-eval--prometheus-7b-v2.0/snapshots/66ffb1fc20beebfb60a3964a957d9011723116c5,JUDGE_MAX_OUTPUT_TOKENS=1024" \
     benchmarks/rag/sbash_runs/run_local_judge_prometheus.sbatch

This keeps the existing Transformers backend and v2 rubrics. The exporter
automatically uses the Chinese QA rubric for MultiFieldQA zh and DuReader.
Resume is based on dataset, request ID, and experiment label. If predictions
change under the same label, reuse of an old judge output would skip their
new evaluation. Use a new output directory for replacement predictions.

After the judge finishes, aggregate and join its results:

.. code-block:: bash

   python benchmarks/rag/results_eval/aggregate_judge_results.py \
     --input "$JUDGE_DIR/judge_outputs.jsonl" \
     --dataset-method-output "$JUDGE_DIR/judge_summary_by_dataset_method.csv" \
     --method-output "$JUDGE_DIR/judge_summary_by_method.csv"

   python benchmarks/rag/results_eval/build_round7_metrics.py \
     --submission "$SUBMISSION" --output-dir "$JUDGE_DIR" \
     --reference-metrics benchmarks/rag/07_accuracy_recovery/reference_metrics.csv \
     --judge-summary "$JUDGE_DIR/judge_summary_by_dataset_method.csv" \
     --judge-output "$JUDGE_DIR/judge_outputs.jsonl"

The join requires complete parsed judgments for every selected run; partial
or malformed judge output must be repaired before ranking. If the round is
submitted in separate groups, pass ``--submission`` twice with their respective
TSVs and export to a fresh combined judge directory. Duplicate dataset/config
runs are rejected so a replacement must be selected explicitly.

Pull round-7 results back from the cluster, running locally:

.. code-block:: bash

   rsync -av -e "ssh -p 2222" \
     a.tomat@cluster.hpc.hse.ru:/home/a.tomat/work/LMCache/benchmarks/rag/07_accuracy_recovery/submissions/ \
     benchmarks/rag/07_accuracy_recovery/submissions/

   rsync -av -e "ssh -p 2222" --include='07_*/***' --exclude='*' \
     a.tomat@cluster.hpc.hse.ru:/home/a.tomat/work/LMCache/benchmarks/rag/results/ \
     benchmarks/rag/results/

   rsync -av -e "ssh -p 2222" \
     a.tomat@cluster.hpc.hse.ru:/home/a.tomat/work/LMCache/benchmarks/rag/results_eval/outputs/round7_seed42_all/ \
     benchmarks/rag/results_eval/outputs/round7_seed42_all/

The manifest contains machine-local absolute paths. If re-exporting or
rebuilding on the local machine after copying results, rerun the collector
locally from the copied submission TSV to regenerate those paths.
