# SPDX-License-Identifier: Apache-2.0
"""Validate the Chinese Qwen3 matrix and write scores plus a judge manifest."""

# Standard
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import math
import re

# Local
from build_qwen3_artifacts import (
    DATASETS as ENGLISH_DATASETS,
    EXPERIMENTS,
    WORKBOOK_COLUMNS,
    optional_float,
    percent,
    write_metrics_tsv,
    write_xlsx,
)


DATASETS = {
    "multifieldqa_zh_longbench": (
        "MultiFieldQA zh (LongBench)",
        "F1 tokenizer",
        "f1_tokenizer",
    ),
    "dureader_longbench": ("DuReader (LongBench)", "ROUGE-L zh", "quality"),
}
EXPECTED_METADATA = (0.7, 0.8, 20, 0.0, 42, "qwen-non-thinking")
FIELDS = (
    "dataset",
    "experiment",
    "score_metric",
    "score",
    "f1_word",
    "precision",
    "recall",
    "ttft_s",
    "tpot_s",
    "throughput_req_s",
    "cap_hits",
    "empty_outputs",
    "job_id",
    "run_name",
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
)
CHINESE_WORKBOOK = "rag_qwen3_chat_nothink_zh_v1_seed42_meeting_metrics.xlsx"
COMBINED_WORKBOOK = (
    "rag_qwen3_five_datasets_meeting_metrics_current_9622_chinese_unjudged.xlsx"
)
CHINESE_JUDGED_WORKBOOK = (
    "rag_qwen3_chat_nothink_zh_v1_seed42_meeting_metrics_with_judge.xlsx"
)
COMBINED_JUDGED_WORKBOOK = (
    "rag_qwen3_five_datasets_meeting_metrics_current_9622_chinese_judged.xlsx"
)


def mean(values: list[float]) -> float:
    """Return the arithmetic mean of nonempty metric values.

    Args:
        values: Metric values.

    Returns:
        Arithmetic mean.

    Raises:
        ValueError: If there are no values.
    """
    if not values:
        raise ValueError("Metric has no values")
    return sum(values) / len(values)


def optional_mean(rows: list[dict[str, str]], field: str) -> float | None:
    """Average populated cells from a result CSV field.

    Args:
        rows: Per-request result rows.
        field: Column name.

    Returns:
        Mean, or ``None`` when the column is empty.
    """
    values = [
        float(row[field]) for row in rows if row.get(field, "") not in ("", "nan")
    ]
    return mean(values) if values else None


def meeting_rows(
    summary: list[dict[str, object]],
    manifest: list[dict[str, object]],
    judge_rows: dict[tuple[str, str], dict[str, str]] | None = None,
) -> list[dict[str, object]]:
    """Convert validated Chinese metrics to the existing meeting-table schema.

    Args:
        summary: Per-run aggregates in submission order.
        manifest: Matching run records with canonical experiment order.
        judge_rows: Complete judged aggregates, or ``None`` before judging.

    Returns:
        Dataset-grouped rows with percent-scale scores and optional judge cells.

    Raises:
        ValueError: If the two inputs have different lengths.
    """
    if len(summary) != len(manifest):
        raise ValueError("Chinese summary and manifest lengths differ")
    dataset_order = {name: index for index, name in enumerate(DATASETS)}
    pairs = sorted(
        zip(summary, manifest, strict=True),
        key=lambda pair: (dataset_order[str(pair[1]["dataset"])], pair[1]["order"]),
    )
    rows = []
    for metrics, record in pairs:
        is_qa = record["dataset"] == "multifieldqa_zh_longbench"
        judge = (
            judge_rows.get((str(record["dataset"]), str(record["label"])), {})
            if judge_rows is not None
            else {}
        )
        row: dict[str, object] = {column: None for column in WORKBOOK_COLUMNS}
        row.update(
            {
                "Dataset": metrics["dataset"],
                "Experiment": metrics["experiment"],
                "Score Metric": metrics["score_metric"],
                "Score": float(metrics["score"]) * 100,
                "Precision": (
                    float(metrics["precision"]) * 100
                    if is_qa and metrics["precision"] is not None
                    else None
                ),
                "Recall": (
                    float(metrics["recall"]) * 100
                    if is_qa and metrics["recall"] is not None
                    else None
                ),
                "Judge Score (0-1)": optional_float(judge, "mean_semantic_score"),
                "Judge Correct %": percent(optional_float(judge, "correct_rate")),
                "Judge Mostly %": percent(optional_float(judge, "mostly_correct_rate")),
                "Judge Incorrect %": percent(optional_float(judge, "incorrect_rate")),
                "TTFT(s)": metrics["ttft_s"],
                "TPOT(s)": metrics["tpot_s"],
                "Throughput(req/s)": metrics["throughput_req_s"],
            }
        )
        rows.append(row)
    return rows


def read_complete_judge_summary(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Read a fully parsed 50-example judge aggregate for each Chinese run.

    Args:
        path: Dataset-method summary CSV from ``aggregate_judge_results.py``.

    Returns:
        Rows keyed by dataset stem and experiment label.

    Raises:
        ValueError: If any group is missing, duplicated, or incompletely scored.
    """
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    expected = {
        (stem, experiment.label) for stem in DATASETS for experiment in EXPERIMENTS
    }
    by_pair: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        pair = (row.get("dataset", ""), row.get("experiment", ""))
        if pair in by_pair:
            raise ValueError(f"Duplicate Chinese judge group: {pair}")
        try:
            score = float(row["mean_semantic_score"])
            error_rate = float(row["parse_error_rate"])
            count = int(row["n"])
            parsed_count = int(row["n_parsed"])
            rates = [
                float(row[field])
                for field in ("correct_rate", "mostly_correct_rate", "incorrect_rate")
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Chinese judge summary row: {pair}") from exc
        if (
            count != 50
            or parsed_count != 50
            or error_rate != 0
            or not math.isfinite(score)
            or not 0 <= score <= 1
            or any(not math.isfinite(rate) or not 0 <= rate <= 1 for rate in rates)
        ):
            raise ValueError(f"Incomplete Chinese judge group: {pair}")
        by_pair[pair] = row
    if by_pair.keys() != expected:
        missing = sorted(expected - by_pair.keys())
        extra = sorted(by_pair.keys() - expected)
        raise ValueError(
            f"Chinese judge coverage differs: missing={missing}, extra={extra}"
        )
    return by_pair


def read_english_meeting_rows(path: Path) -> list[dict[str, object]]:
    """Read and validate the existing three-dataset meeting TSV.

    Args:
        path: TSV with the canonical 13 meeting-table columns.

    Returns:
        The 195 English rows in their existing order.

    Raises:
        ValueError: If columns, experiment coverage, or scores are incomplete.
    """
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        if tuple(reader.fieldnames or ()) != WORKBOOK_COLUMNS:
            raise ValueError(f"Unexpected English meeting columns: {path}")
        rows = list(reader)
    expected = {
        (dataset.display_name, experiment.label)
        for dataset in ENGLISH_DATASETS
        for experiment in EXPERIMENTS
    }
    actual = {(row["Dataset"], row["Experiment"]) for row in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError(f"Incomplete English meeting table: {path}")
    if any(not row["Score"] for row in rows):
        raise ValueError(f"Missing English scores: {path}")
    numeric_rows: list[dict[str, object]] = []
    for raw_row in rows:
        row: dict[str, object] = dict(raw_row)
        for column in WORKBOOK_COLUMNS[3:]:
            value = raw_row[column]
            row[column] = float(value) if value else None
        numeric_rows.append(row)
    return numeric_rows


def main() -> None:
    """Validate all 130 jobs from a submission record and write local artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument(
        "--rag-dir", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--english-tsv",
        type=Path,
        default=None,
        help="Existing 195-row English meeting TSV to include in a five-dataset table.",
    )
    parser.add_argument(
        "--judge-summary",
        type=Path,
        default=None,
        help="Complete 130-group Chinese judge summary CSV.",
    )
    parser.add_argument(
        "--judge-provenance",
        choices=("model", "adjudicated"),
        default="model",
        help="Acknowledge that the judge summary includes analyst adjudication.",
    )
    args = parser.parse_args()
    if args.judge_provenance == "adjudicated" and args.judge_summary is None:
        parser.error("--judge-provenance adjudicated requires --judge-summary")
    rag_dir = args.rag_dir.resolve()
    repo_dir = rag_dir.parents[1]
    output_dir = (
        args.output_dir or rag_dir / "results_eval" / "outputs" / "qwen3_chinese"
    )
    prefix = "qwen3_chat_nothink_zh_v1_seed42_"
    specs = {spec.key: (order, spec) for order, spec in enumerate(EXPERIMENTS)}
    seen: set[tuple[str, str]] = set()
    summary: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    lines = args.submission.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if not line.startswith("SUBMITTED "):
            continue
        fields = dict(field.split("=", 1) for field in line.split("\t")[1:])
        job_id = int(fields["job_id"])
        experiment_name = fields["experiment"]
        if not experiment_name.startswith(prefix) or "_smoke_" in experiment_name:
            raise ValueError(f"Unexpected experiment name: {experiment_name}")
        remainder = experiment_name.removeprefix(prefix)
        dataset = next(
            (stem for stem in DATASETS if remainder.startswith(f"{stem}_")), None
        )
        if dataset is None:
            raise ValueError(f"Unknown dataset: {experiment_name}")
        experiment_key = remainder.removeprefix(f"{dataset}_")
        if experiment_key not in specs:
            raise ValueError(f"Unknown experiment: {experiment_name}")
        pair = (dataset, experiment_key)
        if pair in seen:
            raise ValueError(f"Duplicate submitted pair: {pair}")
        seen.add(pair)
        order, spec = specs[experiment_key]
        run_name = f"{experiment_name}_{job_id}"
        result_csv = rag_dir / "results" / run_name / f"{run_name}.csv"
        log_path = rag_dir / "logs" / run_name / "benchmark.log"
        if not result_csv.is_file() or not log_path.is_file():
            raise FileNotFoundError(f"Missing result or benchmark log: {run_name}")
        with result_csv.open(newline="", encoding="utf-8") as result_file:
            rows = list(csv.DictReader(result_file))
        request_ids = [int(row["request_id"]) for row in rows]
        if request_ids != list(range(50)):
            raise ValueError(f"Expected request IDs 0..49: {run_name}: {request_ids}")
        if any(row["success"].lower() != "true" for row in rows):
            raise ValueError(f"Failed requests: {run_name}")
        for row in rows:
            metadata = (
                float(row["temperature"]),
                float(row["top_p"]),
                int(row["top_k"]),
                float(row["min_p"]),
                int(row["generation_seed"]),
                row["chat_template_mode"],
            )
            if metadata != EXPECTED_METADATA:
                raise ValueError(f"Wrong generation settings: {run_name}")
        display, metric, score_field = DATASETS[dataset]
        score = optional_mean(rows, score_field)
        if score is None:
            raise ValueError(f"Missing score: {run_name}")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        summary_key = (
            "average_quality" if score_field == "quality" else "average_f1_tokenizer"
        )
        match = re.search(rf"{summary_key}=([0-9.eE+-]+)", log_text)
        throughput = re.search(r"throughput=([0-9.eE+-]+)", log_text)
        if (
            "Finished benchmarking" not in log_text
            or match is None
            or throughput is None
            or not math.isclose(score, float(match.group(1)), abs_tol=1e-10)
        ):
            raise ValueError(f"Result/log mismatch: {run_name}")
        cap = 256 if dataset.startswith("multifieldqa") else 1024
        summary.append(
            {
                "dataset": display,
                "experiment": spec.label,
                "score_metric": metric,
                "score": score,
                "f1_word": optional_mean(rows, "f1_word"),
                "precision": optional_mean(rows, "precision"),
                "recall": optional_mean(rows, "recall"),
                "ttft_s": optional_mean(rows, "ttft"),
                "tpot_s": optional_mean(rows, "tpot"),
                "throughput_req_s": float(throughput.group(1)),
                "cap_hits": sum(
                    int(row["generation_token_cnt"]) >= cap for row in rows
                ),
                "empty_outputs": sum(not row["generated_text"].strip() for row in rows),
                "job_id": job_id,
                "run_name": run_name,
            }
        )
        manifest.append(
            {
                "dataset": dataset,
                "label": spec.label,
                "group": spec.group,
                "order": order,
                "run_name": run_name,
                "job_id": job_id,
                "row_count": 50,
                "result_csv": result_csv.relative_to(repo_dir),
                "dataset_path": (rag_dir / "inputs" / f"{dataset}.json").relative_to(
                    repo_dir
                ),
                "generation_max_tokens": cap,
            }
        )
    expected = {(stem, spec.key) for stem in DATASETS for spec in EXPERIMENTS}
    if seen != expected or len(summary) != 130:
        raise ValueError(
            f"Incomplete Chinese matrix: missing={sorted(expected - seen)}, "
            f"extra={sorted(seen - expected)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest)
    judge_rows = (
        read_complete_judge_summary(args.judge_summary)
        if args.judge_summary is not None
        else None
    )
    chinese_rows = meeting_rows(summary, manifest, judge_rows)
    chinese_name = (
        CHINESE_JUDGED_WORKBOOK if judge_rows is not None else CHINESE_WORKBOOK
    )
    chinese_workbook = rag_dir / "results" / chinese_name
    write_xlsx(chinese_rows, chinese_workbook)
    write_metrics_tsv(chinese_rows, chinese_workbook.with_suffix(".tsv"))
    audit_path = chinese_workbook.with_name(f"{chinese_workbook.stem}_audit.txt")
    audit_path.write_text(
        "\n".join(
            "VALID\t"
            f"dataset={record['dataset']}\texperiment={record['label']}"
            f"\tjob_id={record['job_id']}\trows={record['row_count']}"
            f"\tcap_hits={metrics['cap_hits']}"
            f"\tempty_outputs={metrics['empty_outputs']}"
            for metrics, record in sorted(
                zip(summary, manifest, strict=True),
                key=lambda pair: (
                    list(DATASETS).index(str(pair[1]["dataset"])),
                    pair[1]["order"],
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    if args.english_tsv is not None:
        english_rows = read_english_meeting_rows(args.english_tsv)
        combined_name = (
            COMBINED_JUDGED_WORKBOOK if judge_rows is not None else COMBINED_WORKBOOK
        )
        combined_workbook = rag_dir / "results" / combined_name
        combined_rows = [*english_rows, *chinese_rows]
        write_xlsx(combined_rows, combined_workbook)
        write_metrics_tsv(combined_rows, combined_workbook.with_suffix(".tsv"))
        print(
            f"Five-dataset meeting table ({len(combined_rows)} rows): "
            f"{combined_workbook}"
        )
    print(f"Validated {len(summary)} Chinese runs; metrics and manifest: {output_dir}")
    print(f"Chinese meeting table: {chinese_workbook}")
    print(f"Chinese run audit: {audit_path}")


if __name__ == "__main__":
    main()
