# SPDX-License-Identifier: Apache-2.0
"""Validate explicitly submitted round-seven runs and prepare metrics/judging."""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
from typing import Any, Sequence
import argparse
import csv
import json
import math
import statistics

DATASETS = {
    "musique_longbench": ("F1 tokenizer", "f1_tokenizer", 32),
    "2wikimqa_longbench": ("F1 tokenizer", "f1_tokenizer", 32),
    "samsum_longbench": ("ROUGE-L", "quality", 256),
    "multifieldqa_zh_longbench": ("F1 tokenizer", "f1_tokenizer", 256),
    "dureader_longbench": ("ROUGE-L zh", "quality", 1024),
}
SUBMISSION_FIELDS = (
    "dataset",
    "config_key",
    "experiment_name",
    "job_id",
    "generation_max_tokens",
    "end_index",
    "generation_seed",
)
MANIFEST_FIELDS = (
    "dataset",
    "label",
    "group",
    "order",
    "run_name",
    "job_id",
    "row_count",
    "result_csv",
    "dataset_path",
    "generation_max_tokens",
    "config_key",
    "generation_seed",
)
METRIC_FIELDS = (
    "dataset",
    "config_key",
    "experiment",
    "job_id",
    "generation_seed",
    "n",
    "score_metric",
    "score",
    "precision",
    "recall",
    "ttft_s",
    "empty_output_count",
    "max_token_count",
    "judge_score",
    "comparison_source",
    "full_n",
    "full_score",
    "score_delta_vs_full",
    "full_judge_score",
    "judge_delta_vs_full",
    "ttft_speedup_vs_full",
    "baseline16_n",
    "baseline16_score",
    "score_delta_vs_baseline16",
    "baseline16_judge_score",
    "judge_delta_vs_baseline16",
)
RANKING_FIELDS = (
    "config_key",
    "dataset_count",
    "worst_score_deficit",
    "macro_score_delta",
    "worst_judge_deficit",
    "macro_judge_delta",
)


def read_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    """Read a table with unique column names and well-formed rows.

    Args:
        path (Path): CSV or TSV file to read.
        delimiter (str): Field delimiter; defaults to a comma.

    Returns:
        list[dict[str, str]]: Table records in file order.

    Raises:
        ValueError: If the header or row shape is malformed.
    """
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        fields = reader.fieldnames
        if not fields or len(set(fields)) != len(fields):
            raise ValueError(f"Missing or duplicate headers: {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed table row: {path}")
    return rows


def finite_number(value: str, name: str, lower: float = 0.0) -> float:
    """Parse a finite numeric metric with a lower bound.

    Args:
        value (str): Serialized number.
        name (str): Field name for error messages.
        lower (float): Minimum permitted value.

    Returns:
        float: Validated number.

    Raises:
        ValueError: If the value is missing, nonfinite, or below the bound.
    """
    number = float(value)
    if not math.isfinite(number) or number < lower:
        raise ValueError(f"Invalid {name}: {value!r}")
    return number


def unit_number(value: str, name: str) -> float:
    """Parse a score or rate in the closed interval from zero to one.

    Args:
        value (str): Serialized score or rate.
        name (str): Field name for error messages.

    Returns:
        float: Validated score or rate.

    Raises:
        ValueError: If the value lies outside the interval or is nonfinite.
    """
    number = finite_number(value, name)
    if number > 1.0:
        raise ValueError(f"Invalid {name}: {value!r}")
    return number


def read_submissions(paths: Sequence[Path]) -> list[dict[str, str]]:
    """Load production submissions without choosing reruns implicitly.

    Args:
        paths (Sequence[Path]): Submission TSV files to combine.

    Returns:
        list[dict[str, str]]: Unique dataset/configuration submissions.

    Raises:
        ValueError: If metadata is invalid, a pair repeats, or a smoke is supplied.
    """
    submissions: list[dict[str, str]] = []
    pairs: set[tuple[str, str]] = set()
    seeds: dict[str, int] = {}
    run_names: set[str] = set()
    labels: set[tuple[str, str]] = set()
    for path in paths:
        for row in read_rows(path, "\t"):
            if any(not row.get(field, "").strip() for field in SUBMISSION_FIELDS):
                raise ValueError(f"Missing submission metadata in {path}")
            dataset = row["dataset"]
            if dataset not in DATASETS:
                raise ValueError(f"Unsupported dataset: {dataset}")
            for field in ("config_key", "experiment_name", "job_id"):
                if Path(row[field]).name != row[field] or row[field] in {".", ".."}:
                    raise ValueError(f"Unsafe {field}: {row[field]!r}")
            if int(row["job_id"]) <= 0:
                raise ValueError("job_id must be positive")
            if int(row["end_index"]) != 50:
                raise ValueError("Production metrics require end_index=50; omit smokes")
            if int(row["generation_max_tokens"]) != DATASETS[dataset][2]:
                raise ValueError(f"Unexpected generation cap for {dataset}")
            seed = int(row["generation_seed"])
            if seed < 0:
                raise ValueError("generation_seed must be non-negative")
            key = row["config_key"]
            if key in seeds and seed != seeds[key]:
                raise ValueError(f"Mixed seeds for configuration {key}")
            seeds[key] = seed
            pair = (dataset, row["config_key"])
            if pair in pairs:
                raise ValueError(
                    f"Duplicate dataset/config_key {pair}; choose one rerun"
                )
            label_pair = (dataset, row["experiment_name"])
            run_name = f"{row['experiment_name']}_{row['job_id']}"
            if label_pair in labels or run_name in run_names:
                raise ValueError("Submission labels and run paths must be unique")
            pairs.add(pair)
            labels.add(label_pair)
            run_names.add(run_name)
            submissions.append(row)
    if not submissions:
        raise ValueError("No submitted runs")
    return submissions


def validate_run(submission: dict[str, str], rag_dir: Path) -> list[dict[str, str]]:
    """Validate one submitted run using only its explicitly named result CSV.

    Args:
        submission (dict[str, str]): Production submission metadata.
        rag_dir (Path): Benchmark directory containing results and inputs.

    Returns:
        list[dict[str, str]]: Fifty successful, metadata-matched result rows.

    Raises:
        ValueError: If coverage, sampling, metrics, or reference data is invalid.
        FileNotFoundError: If the result CSV or reference dataset is absent.

    Notes:
        Successful empty generations are retained and scored as produced.
    """
    run_name = f"{submission['experiment_name']}_{submission['job_id']}"
    csv_path = rag_dir / "results" / run_name / f"{run_name}.csv"
    rows = read_rows(csv_path)
    ids = [int(row["request_id"]) for row in rows]
    if len(rows) != 50 or sorted(ids) != list(range(50)):
        raise ValueError(f"Expected exactly request IDs 0..49: {csv_path}")
    dataset = submission["dataset"]
    score_field = DATASETS[dataset][1]
    cap = int(submission["generation_max_tokens"])
    seed = int(submission["generation_seed"])
    for row in rows:
        if row.get("success", "").lower() != "true":
            raise ValueError(f"Failed request {row['request_id']}: {csv_path}")
        if "generated_text" not in row:
            raise ValueError(f"Missing generated_text: {csv_path}")
        for field, expected in (("temperature", 0.7), ("top_p", 0.8), ("min_p", 0.0)):
            if not math.isclose(
                finite_number(row[field], field), expected, abs_tol=1e-9
            ):
                raise ValueError(f"Unexpected {field}: {csv_path}")
        if int(row["top_k"]) != 20 or int(row["generation_seed"]) != seed:
            raise ValueError(f"Unexpected top_k or generation_seed: {csv_path}")
        if row["chat_template_mode"] != "qwen-non-thinking":
            raise ValueError(f"Unexpected chat_template_mode: {csv_path}")
        unit_number(row[score_field], score_field)
        finite_number(row["ttft"], "ttft")
        if not 0 <= int(row["generation_token_cnt"]) <= cap:
            raise ValueError(f"Invalid generation token count: {csv_path}")
        for field in ("precision", "recall"):
            if row.get(field, ""):
                unit_number(row[field], field)
    dataset_path = rag_dir / "inputs" / f"{dataset}.json"
    with dataset_path.open(encoding="utf-8") as file:
        examples = json.load(file)
    if not isinstance(examples, list) or len(examples) < 50:
        raise ValueError(
            f"Reference dataset needs at least 50 examples: {dataset_path}"
        )
    return rows


def read_judge_summary(
    path: Path, selected: set[tuple[str, str]] | None = None
) -> dict[tuple[str, str], float]:
    """Read complete per-dataset/per-experiment judge aggregates.

    Args:
        path (Path): CSV produced by ``aggregate_judge_results.py``.
        selected (set[tuple[str, str]] | None): Optional requested group allowlist.

    Returns:
        dict[tuple[str, str], float]: Mean judge score by dataset and label.

    Raises:
        ValueError: If groups repeat or contain incomplete or invalid judgments.
    """
    groups: dict[tuple[str, str], float] = {}
    for row in read_rows(path):
        pair = (row["dataset"], row["experiment"])
        if selected is not None and pair not in selected:
            continue
        if pair in groups:
            raise ValueError(f"Duplicate judge group: {pair}")
        if int(row["n"]) != 50 or int(row["n_parsed"]) != 50:
            raise ValueError(f"Incomplete judge group: {pair}")
        if finite_number(row["parse_error_rate"], "parse_error_rate") != 0:
            raise ValueError(f"Judge parse errors: {pair}")
        groups[pair] = unit_number(row["mean_semantic_score"], "mean_semantic_score")
    return groups


def read_reference_metrics(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load historical full-recompute and optional 16-percent Blend baselines.

    Args:
        path (Path): Baseline CSV with dataset, config_key, score_metric, score,
            judge_score, ttft_s, n, generation_seed, and source_workbook columns.

    Returns:
        dict[tuple[str, str], dict[str, Any]]: Validated historical metrics keyed
        by dataset and configuration, including all five full-recompute rows.

    Raises:
        ValueError: If metadata, bounds, coverage, or pair uniqueness is invalid.

    Notes:
        ``source_workbook`` is a provenance label and is never opened. Scores use
        0..100 units. Historical SAMSum Blend may have 48 cases; full controls
        and all other baselines must have 50 cases.
    """
    references: dict[tuple[str, str], dict[str, Any]] = {}
    fields = (
        "dataset",
        "config_key",
        "score_metric",
        "score",
        "judge_score",
        "ttft_s",
        "n",
        "generation_seed",
        "source_workbook",
    )
    for row in read_rows(path):
        if any(not row.get(field, "").strip() for field in fields):
            raise ValueError("Missing reference metrics metadata")
        dataset = row["dataset"]
        key = row["config_key"]
        if dataset not in DATASETS or key not in {"full_recompute", "lmcache_blend"}:
            raise ValueError(
                f"Unsupported reference dataset/configuration: {dataset}/{key}"
            )
        pair = (dataset, key)
        if pair in references:
            raise ValueError(f"Duplicate reference metrics pair: {pair}")
        if row["score_metric"] != DATASETS[dataset][0]:
            raise ValueError(f"Unexpected reference score metric for {dataset}")
        score = finite_number(row["score"], "reference score")
        if score > 100:
            raise ValueError("Reference score must be within 0..100")
        judge_score = unit_number(row["judge_score"], "reference judge_score")
        ttft = finite_number(row["ttft_s"], "reference ttft_s")
        if ttft <= 0:
            raise ValueError("Reference ttft_s must be positive")
        count = int(row["n"])
        permitted_counts = (
            {48, 50} if pair == ("samsum_longbench", "lmcache_blend") else {50}
        )
        if count not in permitted_counts:
            raise ValueError(f"Unexpected reference count for {pair}: {count}")
        seed = int(row["generation_seed"])
        if seed < 0:
            raise ValueError("Reference generation_seed must be non-negative")
        references[pair] = {
            **row,
            "score": score,
            "judge_score": judge_score,
            "ttft_s": ttft,
            "n": count,
            "generation_seed": seed,
        }
    if {dataset for dataset, key in references if key == "full_recompute"} != set(
        DATASETS
    ):
        raise ValueError(
            "Reference metrics require full_recompute for all five datasets"
        )
    if len({row["generation_seed"] for row in references.values()}) != 1:
        raise ValueError("Reference metrics must use the same generation seed")
    return references


def audit_judge_output(
    path: Path,
    manifests: list[dict[str, Any]],
    run_rows: dict[tuple[str, str], list[dict[str, str]]],
    summaries: dict[tuple[str, str], float],
) -> None:
    """Verify raw judgments cover the exact submitted predictions and run IDs.

    Args:
        path (Path): Judge-output JSONL from the existing local judge runner.
        manifests (list[dict[str, Any]]): Selected runs and exporter labels.
        run_rows (dict[tuple[str, str], list[dict[str, str]]]): Original result rows.
        summaries (dict[tuple[str, str], float]): Aggregate means to cross-check.

    Returns:
        None.

    Raises:
        ValueError: If a selected group is stale, duplicated, incomplete, or invalid.

    Notes:
        Unselected experiment groups are ignored, allowing combined judge files.
    """
    selected = {(row["dataset"], row["label"]): row for row in manifests}
    predictions = {
        pair: {int(row["request_id"]): row["generated_text"] for row in rows}
        for pair, rows in run_rows.items()
    }
    scores: dict[tuple[str, str], dict[int, float]] = {pair: {} for pair in selected}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Invalid judge object on line {line_number}")
            metadata = record.get("analysis_metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError(f"Invalid judge metadata on line {line_number}")
            dataset = record.get("dataset", "")
            label = metadata.get("experiment", "")
            if not isinstance(dataset, str) or not isinstance(label, str):
                raise ValueError(f"Invalid judge group on line {line_number}")
            pair = (dataset, label)
            if pair not in selected:
                continue
            manifest = selected[pair]
            request_id = record.get("request_id")
            if type(request_id) is not int or request_id not in predictions[pair]:
                raise ValueError(f"Invalid judge request ID for {pair}")
            if request_id in scores[pair]:
                raise ValueError(f"Duplicate judge request ID for {pair}")
            if (
                metadata.get("run_name") != manifest["run_name"]
                or type(metadata.get("job_id")) is not int
                or metadata.get("job_id") != int(manifest["job_id"])
            ):
                raise ValueError(f"Stale judge run metadata for {pair}")
            input_fields = record.get("input_fields", {})
            if (
                not isinstance(input_fields, dict)
                or input_fields.get("prediction") != predictions[pair][request_id]
            ):
                raise ValueError(f"Judge prediction mismatch for {pair}")
            parsed = record.get("judge_output_parsed")
            score = parsed.get("semantic_score") if isinstance(parsed, dict) else None
            if (
                record.get("parse_error")
                or type(score) not in (int, float)
                or score not in (0, 0.25, 0.5, 0.75, 1)
            ):
                raise ValueError(f"Invalid parsed judge score for {pair}")
            scores[pair][request_id] = float(score)
    for pair, values in scores.items():
        if set(values) != set(range(50)):
            raise ValueError(f"Incomplete raw judge coverage for {pair}")
        if not math.isclose(
            statistics.fmean(values.values()), summaries[pair], abs_tol=1e-12
        ):
            raise ValueError(f"Judge summary mean mismatch for {pair}")


def write_table(path: Path, fields: Sequence[str], rows: list[dict[str, Any]]) -> None:
    """Write UTF-8 records with a fixed header.

    Args:
        path (Path): CSV destination.
        fields (Sequence[str]): Output columns in order.
        rows (list[dict[str, Any]]): Records to serialize.

    Returns:
        None.
    """
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_metrics(
    submissions: Sequence[Path],
    rag_dir: Path,
    output_dir: Path,
    judge_summary: Path | None = None,
    judge_output: Path | None = None,
    reference_metrics: Path | None = None,
) -> list[dict[str, Any]]:
    """Collect validated runs, exporter manifests, and baseline comparisons.

    Args:
        submissions (Sequence[Path]): Production submission TSV files.
        rag_dir (Path): Benchmark directory containing named runs and inputs.
        output_dir (Path): Destination for manifest, metrics, and ranking CSVs.
        judge_summary (Path | None): Optional complete judge aggregate CSV.
        judge_output (Path | None): Raw JSONL required together with the summary.
        reference_metrics (Path | None): Optional historical baseline CSV; these
            references are compared but never added to the judge manifest.

    Returns:
        list[dict[str, Any]]: Per-run metrics; intrinsic scores use 0..100 units.

    Raises:
        ValueError: If a run, judge group, or baseline is invalid or ambiguous.
        FileNotFoundError: If required input files are absent.

    Notes:
        Submitted full controls share request IDs and generation seeds. Historical
        references compare recorded aggregates with matching seeds; their source
        and sample counts are reported explicitly. Rankings include only complete
        five-dataset configurations with full-recompute baselines.
        Manifest paths are absolute; rebuild after moving results to another host.
        Generation caps are checked against the submission metadata: result CSVs
        record token counts but do not record the configured generation limit.
    """
    entries = read_submissions(submissions)
    references = read_reference_metrics(reference_metrics) if reference_metrics else {}
    submitted_pairs = {(row["dataset"], row["config_key"]) for row in entries}
    if submitted_pairs.intersection(references):
        raise ValueError(
            "Ambiguous submitted and historical baselines for the same pair"
        )
    if (judge_summary is None) != (judge_output is None):
        raise ValueError("Supply --judge-summary and --judge-output together")
    selected = {(row["dataset"], row["experiment_name"]) for row in entries}
    judges = read_judge_summary(judge_summary, selected) if judge_summary else None
    manifests: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    run_rows: dict[tuple[str, str], list[dict[str, str]]] = {}
    rag_dir = rag_dir.resolve()
    for order, entry in enumerate(entries):
        rows = validate_run(entry, rag_dir)
        dataset = entry["dataset"]
        label = entry["experiment_name"]
        run_rows[(dataset, label)] = rows
        if judges is not None and (dataset, label) not in judges:
            raise ValueError(f"Missing requested judge group: {(dataset, label)}")
        run_name = f"{label}_{entry['job_id']}"
        manifests.append(
            {
                "dataset": dataset,
                "label": label,
                "group": "round07",
                "order": order,
                "run_name": run_name,
                "job_id": entry["job_id"],
                "row_count": 50,
                "result_csv": rag_dir / "results" / run_name / f"{run_name}.csv",
                "dataset_path": rag_dir / "inputs" / f"{dataset}.json",
                "generation_max_tokens": entry["generation_max_tokens"],
                "config_key": entry["config_key"],
                "generation_seed": entry["generation_seed"],
            }
        )
        metric: dict[str, Any] = dict.fromkeys(METRIC_FIELDS, "")
        metric.update(
            dataset=dataset,
            config_key=entry["config_key"],
            experiment=label,
            job_id=entry["job_id"],
            generation_seed=int(entry["generation_seed"]),
            n=50,
            score_metric=DATASETS[dataset][0],
            score=100
            * statistics.fmean(float(row[DATASETS[dataset][1]]) for row in rows),
            ttft_s=statistics.fmean(float(row["ttft"]) for row in rows),
            empty_output_count=sum(not row["generated_text"].strip() for row in rows),
            max_token_count=sum(
                int(row["generation_token_cnt"]) == int(entry["generation_max_tokens"])
                for row in rows
            ),
            judge_score=judges[(dataset, label)] if judges is not None else "",
        )
        if DATASETS[dataset][1] == "f1_tokenizer":
            for field in ("precision", "recall"):
                if all(row.get(field, "") for row in rows):
                    metric[field] = 100 * statistics.fmean(
                        float(row[field]) for row in rows
                    )
        metrics.append(metric)
    if judges is not None and judge_output is not None:
        audit_judge_output(judge_output, manifests, run_rows, judges)
    controls = {
        row["dataset"]: row for row in metrics if row["config_key"] == "full_recompute"
    }
    controls.update(
        {
            dataset: row
            for (dataset, key), row in references.items()
            if key == "full_recompute"
        }
    )
    for row in metrics:
        control = controls.get(row["dataset"])
        if control is None:
            continue
        if row["generation_seed"] != control["generation_seed"]:
            raise ValueError(f"Control seed mismatch for {row['dataset']}")
        row["comparison_source"] = (
            control["source_workbook"]
            if "source_workbook" in control
            else f"submitted_full_recompute:{control['experiment']}"
        )
        row["full_n"] = control["n"]
        row["full_score"] = control["score"]
        row["score_delta_vs_full"] = row["score"] - control["score"]
        if row["ttft_s"] > 0:
            row["ttft_speedup_vs_full"] = control["ttft_s"] / row["ttft_s"]
        if control["judge_score"] != "":
            row["full_judge_score"] = control["judge_score"]
        if judges is not None:
            row["judge_delta_vs_full"] = row["judge_score"] - control["judge_score"]
        baseline16 = references.get((row["dataset"], "lmcache_blend"))
        if baseline16 is not None:
            if row["generation_seed"] != baseline16["generation_seed"]:
                raise ValueError(f"Baseline16 seed mismatch for {row['dataset']}")
            row["baseline16_n"] = baseline16["n"]
            row["baseline16_score"] = baseline16["score"]
            row["score_delta_vs_baseline16"] = row["score"] - baseline16["score"]
            row["baseline16_judge_score"] = baseline16["judge_score"]
            if judges is not None:
                row["judge_delta_vs_baseline16"] = (
                    row["judge_score"] - baseline16["judge_score"]
                )
    rankings: list[dict[str, Any]] = []
    for key in sorted({row["config_key"] for row in metrics} - {"full_recompute"}):
        group = [row for row in metrics if row["config_key"] == key]
        if {row["dataset"] for row in group} != set(DATASETS) or set(controls) != set(
            DATASETS
        ):
            continue
        rankings.append(
            {
                "config_key": key,
                "dataset_count": 5,
                "worst_score_deficit": max(
                    0.0, *(-row["score_delta_vs_full"] for row in group)
                ),
                "macro_score_delta": statistics.fmean(
                    row["score_delta_vs_full"] for row in group
                ),
                "worst_judge_deficit": max(
                    0.0, *(-row["judge_delta_vs_full"] for row in group)
                )
                if judges is not None
                else "",
                "macro_judge_delta": statistics.fmean(
                    row["judge_delta_vs_full"] for row in group
                )
                if judges is not None
                else "",
            }
        )
    rankings.sort(
        key=lambda row: (
            row["worst_judge_deficit"]
            if judges is not None
            else row["worst_score_deficit"],
            row["worst_score_deficit"],
            -row["macro_judge_delta"]
            if judges is not None
            else -row["macro_score_delta"],
            -row["macro_score_delta"],
            row["config_key"],
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_table(output_dir / "manifest.csv", MANIFEST_FIELDS, manifests)
    write_table(output_dir / "metrics.csv", METRIC_FIELDS, metrics)
    write_table(output_dir / "rankings.csv", RANKING_FIELDS, rankings)
    return metrics


def main(argv: Sequence[str] | None = None) -> None:
    """Collect production results from a command-line invocation.

    Args:
        argv (Sequence[str] | None): Arguments, or process arguments when omitted.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, action="append", required=True)
    parser.add_argument(
        "--rag-dir", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-summary", type=Path)
    parser.add_argument("--judge-output", type=Path)
    parser.add_argument(
        "--reference-metrics",
        type=Path,
        help="Historical baseline CSV; compared without adding judge rows.",
    )
    args = parser.parse_args(argv)
    try:
        metrics = build_metrics(
            args.submission,
            args.rag_dir,
            args.output_dir,
            args.judge_summary,
            args.judge_output,
            args.reference_metrics,
        )
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Round-seven collection failed: {exc}\n")
    print(f"Validated {len(metrics)} production runs; wrote {args.output_dir}")


if __name__ == "__main__":
    main()
