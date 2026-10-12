# SPDX-License-Identifier: Apache-2.0
"""Build canonical Qwen3 metrics, judge manifests, and answer inspection pages."""

# Standard
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile
import argparse
import csv
import html
import json
import math
import re


@dataclass(frozen=True)
class DatasetSpec:
    """Display and input information for one benchmark dataset."""

    stem: str
    display_name: str
    score_metric: str


@dataclass(frozen=True)
class ExperimentSpec:
    """Stable run-name suffix and human-readable experiment label."""

    key: str
    label: str
    group: str


@dataclass
class RunRecord:
    """One validated canonical benchmark run."""

    dataset: DatasetSpec
    experiment: ExperimentSpec
    run_name: str
    job_id: int
    result_csv: Path
    log_dir: Path
    rows: list[dict[str, str]]
    throughput: float


DATASETS = (
    DatasetSpec("musique_longbench", "MuSiQue (LongBench)", "F1 tokenizer"),
    DatasetSpec("2wikimqa_longbench", "2WikiMQA (LongBench)", "F1 tokenizer"),
    DatasetSpec("samsum_longbench", "SAMSum (LongBench)", "ROUGE-L"),
)

EXPERIMENTS = (
    ExperimentSpec("full_recompute", "Full Recompute", "Control"),
    ExperimentSpec("lmcache_blend", "Blend (16%) layer 1", "Baseline"),
    ExperimentSpec(
        "yaml_03_bound_and_mix__lmcache_blend_boundary_1",
        "Blend (16%) layer 1 margin tokens 1",
        "YAML 03",
    ),
    ExperimentSpec(
        "yaml_03_bound_and_mix__lmcache_blend_boundary_2",
        "Blend (16%) layer 1 margin tokens 2",
        "YAML 03",
    ),
    ExperimentSpec(
        "yaml_03_bound_and_mix__lmcache_blend_boundary_3",
        "Blend (16%) layer 1 margin tokens 3",
        "YAML 03",
    ),
    ExperimentSpec(
        "yaml_03_bound_and_mix__lmcache_blend_boundary_1_weighted_50_50",
        "Blend (16%) margin 1 + weighted 50-50",
        "YAML 03",
    ),
    ExperimentSpec(
        "yaml_03_bound_and_mix__lmcache_blend_weighted_50_50",
        "Blend (16%) weighted 50-50",
        "YAML 03",
    ),
    ExperimentSpec(
        "yaml_03_bound_and_mix__lmcache_blend_weighted_30_30",
        "Blend (16%) weighted 30-30",
        "YAML 03",
    ),
    ExperimentSpec(
        "yaml_03_bound_and_mix__lmcache_blend_weighted_20_20",
        "Blend (16%) weighted 20-20",
        "YAML 03",
    ),
    ExperimentSpec(
        "yaml_03_bound_and_mix__lmcache_blend_weighted_10_10",
        "Blend (16%) weighted 10-10",
        "YAML 03",
    ),
    ExperimentSpec(
        "yaml_03_bound_and_mix__lmcache_blend_weighted_k0_v20",
        "Blend (16%) weighted K0-V20",
        "YAML 03",
    ),
    ExperimentSpec(
        "yaml_03_bound_and_mix__lmcache_blend_adaptive_weighted",
        "Blend (16%) adaptive weights K0-20, V0-20",
        "YAML 03",
    ),
    ExperimentSpec(
        "yaml_03_bound_and_mix__lmcache_blend_adaptive_weighted_k015_v040",
        "Blend (16%) adaptive weights K0-15, V0-40",
        "YAML 03",
    ),
    ExperimentSpec(
        "yaml_03_bound_and_mix__lmcache_blend_selector_diff_kv_sum",
        "SELECTOR K+V",
        "YAML 03",
    ),
    ExperimentSpec(
        "yaml_03_bound_and_mix__lmcache_blend_selector_diff_kv_weighted_k30_v70",
        "SELECTOR weighted K30-V70",
        "YAML 03",
    ),
    ExperimentSpec(
        "yaml_03_bound_and_mix__lmcache_blend_selector_diff_kv_weighted_k70_v30",
        "SELECTOR weighted K70-V30",
        "YAML 03",
    ),
    ExperimentSpec(
        "yaml_03_bound_and_mix__lmcache_blend_selector_diff_v",
        "SELECTOR V",
        "YAML 03",
    ),
    ExperimentSpec(
        "yaml_04_around_high_div__plain_ratio_020",
        "Blend (20%) layer 1",
        "YAML 04",
    ),
    ExperimentSpec(
        "yaml_04_around_high_div__plain_ratio_024",
        "Blend (24%) layer 1",
        "YAML 04",
    ),
    ExperimentSpec(
        "yaml_04_around_high_div__plain_ratio_032",
        "Blend (32%) layer 1",
        "YAML 04",
    ),
    ExperimentSpec(
        "yaml_04_around_high_div__plain_ratio_100",
        "Blend (100%) layer 1",
        "YAML 04",
    ),
    ExperimentSpec(
        "yaml_04_around_high_div__neighbor_top25_both_cap024",
        "neighbor top25 both cap024",
        "YAML 04",
    ),
    ExperimentSpec(
        "yaml_04_around_high_div__neighbor_top50_both_cap024",
        "neighbor top50 both cap024",
        "YAML 04",
    ),
    ExperimentSpec(
        "yaml_04_around_high_div__neighbor_top50_left_cap024",
        "neighbor top50 left cap024",
        "YAML 04",
    ),
    ExperimentSpec(
        "yaml_04_around_high_div__neighbor_top50_right_cap024",
        "neighbor top50 right cap024",
        "YAML 04",
    ),
    ExperimentSpec(
        "yaml_04_around_high_div__neighbor_top25_both_cap032",
        "neighbor top25 both cap032",
        "YAML 04",
    ),
    ExperimentSpec(
        "yaml_04_around_high_div__neighbor_top50_both_cap032",
        "neighbor top50 both cap032",
        "YAML 04",
    ),
    ExperimentSpec(
        "yaml_04_around_high_div__neighbor_all_both_cap048",
        "neighbor all cap048",
        "YAML 04",
    ),
    ExperimentSpec(
        "yaml_04_around_high_div__span_close_gap1_cap024",
        "close gap cap24",
        "YAML 04",
    ),
    ExperimentSpec(
        "yaml_04_around_high_div__span_close_gap2_cap024",
        "close gap2 cap24",
        "YAML 04",
    ),
    ExperimentSpec(
        "yaml_04_around_high_div__span_close_gap2_cap032",
        "close gap2 cap32",
        "YAML 04",
    ),
    ExperimentSpec(
        "yaml_05_no_hkvd_and_layers__no_recompute",
        "05 no recompute",
        "YAML 05",
    ),
    ExperimentSpec(
        "yaml_05_no_hkvd_and_layers__separator_only",
        "05 separator only",
        "YAML 05",
    ),
    ExperimentSpec(
        "yaml_05_no_hkvd_and_layers__margin_only_2",
        "05 margin only 2",
        "YAML 05",
    ),
    ExperimentSpec(
        "yaml_05_no_hkvd_and_layers__margin_only_4",
        "05 margin only 4",
        "YAML 05",
    ),
    ExperimentSpec(
        "yaml_05_no_hkvd_and_layers__margin_only_8",
        "05 margin only 8",
        "YAML 05",
    ),
    ExperimentSpec(
        "yaml_05_no_hkvd_and_layers__margin_only_16",
        "05 margin only 16",
        "YAML 05",
    ),
    ExperimentSpec(
        "yaml_05_no_hkvd_and_layers__separator_margin_2",
        "05 separator + margin 2",
        "YAML 05",
    ),
    ExperimentSpec(
        "yaml_05_no_hkvd_and_layers__separator_margin_4",
        "05 separator + margin 4",
        "YAML 05",
    ),
    ExperimentSpec(
        "yaml_05_no_hkvd_and_layers__separator_margin_8",
        "05 separator + margin 8",
        "YAML 05",
    ),
    ExperimentSpec(
        "yaml_05_no_hkvd_and_layers__separator_margin_16",
        "05 separator + margin 16",
        "YAML 05",
    ),
    ExperimentSpec(
        "yaml_05_no_hkvd_and_layers__blend_layer0_ratio016",
        "05 Blend (16%) layer 0",
        "YAML 05",
    ),
    ExperimentSpec(
        "yaml_05_no_hkvd_and_layers__blend_layer2_ratio016",
        "05 Blend (16%) layer 2",
        "YAML 05",
    ),
    ExperimentSpec(
        "yaml_05_no_hkvd_and_layers__blend_layer3_ratio016",
        "05 Blend (16%) layer 3",
        "YAML 05",
    ),
    ExperimentSpec(
        "yaml_05_no_hkvd_and_layers__blend_layers1_5_ratio016_100",
        "05 Blend 16% layer 1 then 100% layer 5",
        "YAML 05",
    ),
    ExperimentSpec(
        "06_blend16_shared_candidates__blend16_layer1_adaptive_k005_v015_boundary1",
        "06 adaptive K0-05 V0-15 + boundary1",
        "Batch 06",
    ),
    ExperimentSpec(
        "06_blend16_shared_candidates__blend16_layer1_adaptive_k005_v020_boundary1",
        "06 adaptive K0-05 V0-20 + boundary1",
        "Batch 06",
    ),
    ExperimentSpec(
        "06_blend16_shared_candidates__blend16_layer1_adaptive_k010_v020_boundary1",
        "06 adaptive K0-10 V0-20 + boundary1",
        "Batch 06",
    ),
    ExperimentSpec(
        "06_blend16_shared_candidates__blend16_layer1_kvsum_weighted10",
        "06 K+V selector + weighted 10-10",
        "Batch 06",
    ),
    ExperimentSpec(
        "06_blend16_shared_candidates__blend16_layer1_neighbor_top50_right_cap020_adaptive_k005_v020",
        "06 neighbor right top50 cap20 + adaptive K0-05 V0-20",
        "Batch 06",
    ),
    ExperimentSpec(
        "06_blend16_shared_candidates__blend16_layer1_neighbor_top50_right_cap020_weighted10",
        "06 neighbor right top50 cap20 + weighted 10-10",
        "Batch 06",
    ),
    ExperimentSpec(
        "06_blend16_shared_candidates__blend16_layer1_neighbor_top50_right_cap024_weighted10",
        "06 neighbor right top50 cap24 + weighted 10-10",
        "Batch 06",
    ),
    ExperimentSpec(
        "06_blend16_shared_candidates__blend16_layer1_rank_adaptive_k005_v020",
        "06 rank adaptive K0-05 V0-20",
        "Batch 06",
    ),
    ExperimentSpec(
        "06_blend16_shared_candidates__blend16_layer1_rank_adaptive_k005_v020_boundary1",
        "06 rank adaptive K0-05 V0-20 + boundary1",
        "Batch 06",
    ),
    ExperimentSpec(
        "06_blend16_shared_candidates__blend16_layer1_rank_adaptive_k005_v020_gamma2_boundary1",
        "06 rank adaptive K0-05 V0-20 gamma2 + boundary1",
        "Batch 06",
    ),
    ExperimentSpec(
        "06_blend16_shared_candidates__blend16_layer1_rank_adaptive_k005_v025_boundary1",
        "06 rank adaptive K0-05 V0-25 + boundary1",
        "Batch 06",
    ),
    ExperimentSpec(
        "06_blend16_shared_candidates__blend16_layer1_weighted_05_05_boundary1",
        "06 weighted 05-05 + boundary1",
        "Batch 06",
    ),
    ExperimentSpec(
        "06_blend16_shared_candidates__blend16_layer1_weighted_10_10_boundary1",
        "06 weighted 10-10 + boundary1",
        "Batch 06",
    ),
    ExperimentSpec(
        "06_blend16_shared_candidates__blend16_layer1_weighted_15_15_boundary1",
        "06 weighted 15-15 + boundary1",
        "Batch 06",
    ),
    ExperimentSpec(
        "06_blend16_shared_candidates__blend16_multilayer_l1_l2_18_to_16_rank_adaptive_k005_v020",
        "06 multilayer L1-L2 18->16 rank adaptive K0-05 V0-20",
        "Batch 06",
    ),
    ExperimentSpec(
        "06_blend16_shared_candidates__blend16_multilayer_l1_l3_18_to_16_rank_adaptive_k005_v020",
        "06 multilayer L1-L3 18->16 rank adaptive K0-05 V0-20",
        "Batch 06",
    ),
    ExperimentSpec(
        "06_blend16_shared_candidates__blend16_multilayer_l1_l3_18_to_16_rank_adaptive_k005_v020_boundary1",
        "06 multilayer L1-L3 18->16 rank adaptive K0-05 V0-20 + boundary1",
        "Batch 06",
    ),
    ExperimentSpec(
        "06_blend16_shared_candidates__blend16_multilayer_l1_l3_18_to_16_weighted10",
        "06 multilayer L1-L3 18->16 weighted 10-10",
        "Batch 06",
    ),
    ExperimentSpec(
        "06_blend16_shared_candidates__blend16_multilayer_l1_l3_20_to_16_rank_adaptive_k005_v020",
        "06 multilayer L1-L3 20->16 rank adaptive K0-05 V0-20",
        "Batch 06",
    ),
    ExperimentSpec(
        "06_blend16_shared_candidates__blend16_multilayer_l1_l3_l5_20_to_18_to_16_rank_adaptive_k005_v020",
        "06 multilayer L1-L3-L5 20->18->16 rank adaptive K0-05 V0-20",
        "Batch 06",
    ),
)

REPLACEMENT_JOB_IDS = {
    4277295: 4285156,
    4277301: 4285157,
    4277302: 4285158,
    4277252: 4285159,
    4277254: 4285160,
    4277258: 4285161,
    4277267: 4285162,
    4277269: 4285163,
    4277300: 4285164,
    4277299: 4285165,
    4277255: 4285166,
    4277256: 4285167,
    4277266: 4285168,
    4277270: 4285169,
    4277271: 4285170,
}

WORKBOOK_COLUMNS = (
    "Dataset",
    "Experiment",
    "Score Metric",
    "Score",
    "Precision",
    "Recall",
    "Judge Score (0-1)",
    "Judge Correct %",
    "Judge Mostly %",
    "Judge Incorrect %",
    "TTFT(s)",
    "TPOT(s)",
    "Throughput(req/s)",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rag-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Answer-table directory; defaults to outputs/qwen3_model_only.",
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=None,
        help="Workbook path; defaults under benchmarks/rag/results.",
    )
    parser.add_argument(
        "--judge-summary",
        type=Path,
        default=None,
        help="Optional aggregate_judge_results.py dataset-method CSV.",
    )
    return parser.parse_args()


def read_submission_manifest(path: Path) -> list[tuple[int, str]]:
    """Read job IDs and experiment names from the full submission manifest.

    Args:
        path: Submission manifest path.

    Returns:
        ``(job_id, experiment_name)`` entries.
    """
    entries = []
    with path.open(encoding="utf-8") as manifest_file:
        for line in manifest_file:
            if not line.startswith("SUBMITTED"):
                continue
            fields = {}
            for field in line.rstrip("\n").split("\t")[1:]:
                if "=" in field:
                    key, value = field.split("=", maxsplit=1)
                    fields[key] = value
            entries.append((int(fields["job_id"]), fields["experiment"]))
    return entries


def dataset_and_key(experiment_name: str) -> tuple[DatasetSpec, str]:
    """Extract dataset and experiment key from a submitted experiment name.

    Args:
        experiment_name: Slurm experiment name.

    Returns:
        Dataset specification and stable experiment key.

    Raises:
        ValueError: If the Qwen3 experiment name is unsupported.
    """
    for dataset in DATASETS:
        prefix = f"qwen3_model_only_{dataset.stem}_"
        if experiment_name.startswith(prefix):
            return dataset, experiment_name[len(prefix) :]
    raise ValueError(f"Unsupported Qwen3 experiment name: {experiment_name}")


def find_run_dir(base_dir: Path, run_name: str, job_id: int) -> Path:
    """Resolve an original or replacement run directory.

    Args:
        base_dir: Logs or results directory.
        run_name: Original exact run name.
        job_id: Canonical job ID, possibly a replacement.

    Returns:
        Existing run directory.

    Raises:
        FileNotFoundError: If no unique matching directory exists.
    """
    exact = base_dir / run_name
    if run_name.endswith(f"_{job_id}") and exact.is_dir():
        return exact
    matches = sorted(base_dir.glob(f"qwen3_model_only_rerun1_*_{job_id}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one run directory ending in _{job_id} under {base_dir}; "
            f"found {len(matches)}"
        )
    return matches[0]


def find_metric_csv(run_dir: Path) -> Path:
    """Find the non-debug metric CSV in a result directory.

    Args:
        run_dir: Result directory.

    Returns:
        Metric CSV path.

    Raises:
        FileNotFoundError: If the directory does not contain exactly one metric CSV.
    """
    paths = sorted(
        path for path in run_dir.glob("*.csv") if not path.name.endswith(".debug.csv")
    )
    if len(paths) != 1:
        raise FileNotFoundError(
            f"Expected one metric CSV in {run_dir}; found {len(paths)}"
        )
    return paths[0]


def read_result_rows(path: Path) -> list[dict[str, str]]:
    """Read and validate one result CSV.

    Args:
        path: Metric CSV path.

    Returns:
        Result rows.

    Raises:
        ValueError: If any request failed or has an empty generation.
    """
    with path.open(newline="", encoding="utf-8") as result_file:
        rows = list(csv.DictReader(result_file))
    failed = [row for row in rows if row.get("success", "").lower() != "true"]
    empty = [row for row in rows if not row.get("generated_text", "").strip()]
    if failed or empty:
        raise ValueError(
            f"Invalid result {path}: failed={len(failed)}, empty={len(empty)}"
        )
    return rows


def parse_throughput(log_path: Path) -> float:
    """Read benchmark throughput from a completed benchmark log.

    Args:
        log_path: Benchmark log path.

    Returns:
        Requests-per-second throughput.

    Raises:
        ValueError: If no summary throughput is present.
    """
    text = log_path.read_text(errors="replace")
    match = re.search(r"throughput=([0-9.eE+-]+)", text)
    if match is None:
        raise ValueError(f"No throughput summary in {log_path}")
    return float(match.group(1))


def build_records(rag_dir: Path) -> list[RunRecord]:
    """Build the complete canonical 195-run Qwen3 set.

    Args:
        rag_dir: ``benchmarks/rag`` directory.

    Returns:
        Validated run records in dataset and experiment order.

    Raises:
        ValueError: If the manifest, experiment mapping, or row counts are invalid.
    """
    manifest = rag_dir / "logs" / "qwen3_full_submission_20260825_185942.txt"
    specs_by_key = {spec.key: spec for spec in EXPERIMENTS}
    records_by_pair: dict[tuple[str, str], RunRecord] = {}
    for old_job_id, experiment_name in read_submission_manifest(manifest):
        dataset, key = dataset_and_key(experiment_name)
        if key not in specs_by_key:
            raise ValueError(f"No display specification for experiment key {key}")
        job_id = REPLACEMENT_JOB_IDS.get(old_job_id, old_job_id)
        original_run_name = f"{experiment_name}_{old_job_id}"
        result_dir = find_run_dir(rag_dir / "results", original_run_name, job_id)
        log_dir = find_run_dir(rag_dir / "logs", original_run_name, job_id)
        result_csv = find_metric_csv(result_dir)
        rows = read_result_rows(result_csv)
        expected_counts = {48, 50} if dataset.stem == "samsum_longbench" else {50}
        if len(rows) not in expected_counts:
            raise ValueError(
                f"Unexpected row count for {result_dir.name}: {len(rows)}"
            )
        record = RunRecord(
            dataset=dataset,
            experiment=specs_by_key[key],
            run_name=result_dir.name,
            job_id=job_id,
            result_csv=result_csv,
            log_dir=log_dir,
            rows=rows,
            throughput=parse_throughput(log_dir / "benchmark.log"),
        )
        pair = (dataset.stem, key)
        if pair in records_by_pair:
            raise ValueError(f"Duplicate canonical run for {pair}")
        records_by_pair[pair] = record

    expected_pairs = {
        (dataset.stem, experiment.key)
        for dataset in DATASETS
        for experiment in EXPERIMENTS
    }
    missing = sorted(expected_pairs.difference(records_by_pair))
    extra = sorted(set(records_by_pair).difference(expected_pairs))
    if missing or extra:
        raise ValueError(f"Canonical matrix mismatch: missing={missing}, extra={extra}")
    return [
        records_by_pair[(dataset.stem, experiment.key)]
        for dataset in DATASETS
        for experiment in EXPERIMENTS
    ]


def numeric_values(rows: list[dict[str, str]], field: str) -> list[float]:
    """Extract finite numeric values from a result column.

    Args:
        rows: Result rows.
        field: Column name.

    Returns:
        Finite numeric values.
    """
    values = []
    for row in rows:
        try:
            value = float(row.get(field, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def mean(values: list[float]) -> Optional[float]:
    """Return the arithmetic mean.

    Args:
        values: Numeric values.

    Returns:
        Arithmetic mean, or ``None`` when there are no values.
    """
    return sum(values) / len(values) if values else None


def load_judge_summary(path: Optional[Path]) -> dict[tuple[str, str], dict[str, str]]:
    """Load optional judge aggregates keyed by dataset and experiment.

    Args:
        path: Dataset-method summary CSV, or ``None``.

    Returns:
        Summary rows keyed by ``(dataset, experiment)``.
    """
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as summary_file:
        rows = list(csv.DictReader(summary_file))
    return {(row["dataset"], row["experiment"]): row for row in rows}


def optional_float(row: dict[str, str], key: str) -> Optional[float]:
    """Parse one optional numeric CSV field.

    Args:
        row: Source CSV row.
        key: Field to parse.

    Returns:
        Parsed value, or ``None`` for a missing or invalid value.
    """
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def workbook_row(
    record: RunRecord,
    judge_rows: dict[tuple[str, str], dict[str, str]],
) -> dict[str, Any]:
    """Create one meeting-metrics workbook row.

    Args:
        record: Canonical run.
        judge_rows: Optional judge aggregates.

    Returns:
        Values using the historical workbook columns.
    """
    score_field = (
        "quality"
        if record.dataset.score_metric == "ROUGE-L"
        else "f1_tokenizer"
    )
    score = mean(numeric_values(record.rows, score_field))
    precision = mean(numeric_values(record.rows, "precision"))
    recall = mean(numeric_values(record.rows, "recall"))
    ttft = mean(
        [value for value in numeric_values(record.rows, "ttft") if value >= 0]
    )
    tpot = mean(
        [value for value in numeric_values(record.rows, "tpot") if value >= 0]
    )
    judge = judge_rows.get((record.dataset.stem, record.experiment.label), {})
    return {
        "Dataset": record.dataset.display_name,
        "Experiment": record.experiment.label,
        "Score Metric": record.dataset.score_metric,
        "Score": score * 100 if score is not None else None,
        "Precision": (
            precision * 100
            if precision is not None and record.dataset.score_metric != "ROUGE-L"
            else None
        ),
        "Recall": (
            recall * 100
            if recall is not None and record.dataset.score_metric != "ROUGE-L"
            else None
        ),
        "Judge Score (0-1)": optional_float(judge, "mean_semantic_score"),
        "Judge Correct %": percent(optional_float(judge, "correct_rate")),
        "Judge Mostly %": percent(optional_float(judge, "mostly_correct_rate")),
        "Judge Incorrect %": percent(optional_float(judge, "incorrect_rate")),
        "TTFT(s)": ttft,
        "TPOT(s)": tpot,
        "Throughput(req/s)": record.throughput,
    }


def percent(value: Optional[float]) -> Optional[float]:
    """Convert an optional fraction to percent.

    Args:
        value: Fractional value, or ``None``.

    Returns:
        Percentage value, or ``None``.
    """
    return value * 100 if value is not None else None


def write_manifest(records: list[RunRecord], rag_dir: Path, output_path: Path) -> None:
    """Write the canonical manifest consumed by the judge exporter.

    Args:
        records: Canonical records.
        rag_dir: RAG directory.
        output_path: Manifest CSV destination.
    """
    repo_dir = rag_dir.parents[1]
    fields = (
        "dataset",
        "label",
        "group",
        "order",
        "run_name",
        "job_id",
        "row_count",
        "result_csv",
        "dataset_path",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "dataset": record.dataset.stem,
                    "label": record.experiment.label,
                    "group": record.experiment.group,
                    "order": EXPERIMENTS.index(record.experiment),
                    "run_name": record.run_name,
                    "job_id": record.job_id,
                    "row_count": len(record.rows),
                    "result_csv": record.result_csv.relative_to(repo_dir),
                    "dataset_path": (
                        rag_dir / "inputs" / f"{record.dataset.stem}.json"
                    ).relative_to(repo_dir),
                }
            )


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load a JSON-list dataset.

    Args:
        path: Dataset JSON path.

    Returns:
        Dataset examples in request order.

    Raises:
        ValueError: If the JSON root is not a list.
    """
    with path.open(encoding="utf-8") as dataset_file:
        data = json.load(dataset_file)
    if not isinstance(data, list):
        raise ValueError(f"Expected list dataset in {path}")
    return data


def flatten_answers(value: Any) -> list[str]:
    """Flatten nested answer values into strings.

    Args:
        value: Scalar or nested answer value.

    Returns:
        Flat string answers in input order.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for nested in value for item in flatten_answers(nested)]
    return [str(value)]


def answers_by_request(record: RunRecord) -> dict[int, str]:
    """Map request IDs to raw generated answers.

    Args:
        record: Canonical run record.

    Returns:
        Generated answers keyed by request ID.
    """
    return {int(row["request_id"]): row["generated_text"] for row in record.rows}


def escaped_table_text(value: Any) -> str:
    """Escape newlines for a one-row-per-request delimited table.

    Args:
        value: Table cell value.

    Returns:
        String with line breaks represented as ``\\n``.
    """
    return str(value or "").replace("\r\n", "\\n").replace("\n", "\\n")


def answer_class(gold: str, answer: str) -> str:
    """Return a lightweight exact/contains/empty CSS hint.

    Args:
        gold: Flattened gold answer.
        answer: Generated answer.

    Returns:
        CSS class describing a simple lexical relationship.
    """
    normalized_gold = re.sub(r"\s+", " ", gold.strip().lower())
    normalized_answer = re.sub(r"\s+", " ", answer.strip().lower())
    if not normalized_answer:
        return "empty"
    if normalized_gold and normalized_answer == normalized_gold:
        return "exact"
    if normalized_gold and normalized_gold in normalized_answer:
        return "contains"
    return "answer"


def write_answer_tables(
    dataset: DatasetSpec,
    records: list[RunRecord],
    rag_dir: Path,
    output_dir: Path,
) -> None:
    """Write historical-style CSV, TSV, and HTML answer tables.

    Args:
        dataset: Dataset specification.
        records: Dataset records in display order.
        rag_dir: RAG directory.
        output_dir: Artifact destination.
    """
    examples = load_dataset(rag_dir / "inputs" / f"{dataset.stem}.json")
    answer_maps = {
        record.experiment.label: answers_by_request(record) for record in records
    }
    fieldnames = ["dataset", "request_id", "gold_answer", "question"] + [
        record.experiment.label for record in records
    ]
    delimited_rows = []
    for request_id, example in enumerate(examples):
        gold = " | ".join(flatten_answers(example.get("answers", [])))
        row = {
            "dataset": dataset.stem,
            "request_id": request_id,
            "gold_answer": escaped_table_text(gold),
            "question": escaped_table_text(example.get("question", "")),
        }
        for record in records:
            row[record.experiment.label] = escaped_table_text(
                answer_maps[record.experiment.label].get(request_id, "")
            )
        delimited_rows.append(row)

    for suffix, delimiter in (("csv", ","), ("tsv", "\t")):
        path = output_dir / f"{dataset.stem}_answers_wide.{suffix}"
        with path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=fieldnames,
                delimiter=delimiter,
            )
            writer.writeheader()
            writer.writerows(delimited_rows)

    write_answer_html(
        output_dir / f"{dataset.stem}_answers_inspect.html",
        dataset,
        examples,
        records,
        answer_maps,
    )


def write_answer_html(
    output_path: Path,
    dataset: DatasetSpec,
    examples: list[dict[str, Any]],
    records: list[RunRecord],
    answer_maps: dict[str, dict[int, str]],
) -> None:
    """Write a searchable historical-style answer inspection page.

    Args:
        output_path: HTML destination.
        dataset: Dataset specification.
        examples: Dataset examples in request order.
        records: Canonical records in display order.
        answer_maps: Answers keyed by experiment label and request ID.
    """
    css = """
:root{--bg:#f6f7f9;--panel:#fff;--text:#171a1f;--muted:#5f6875;
--line:#d9dee7;--soft:#eef1f6;--exact:#e6f5ec;--contains:#fff6db;
--empty:#f3f4f6}*{box-sizing:border-box}body{margin:0;background:var(--bg);
color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
line-height:1.45}header{position:sticky;top:0;z-index:2;border-bottom:1px solid
var(--line);background:rgba(246,247,249,.96);padding:14px 20px}header h1{
margin:0 0 10px;font-size:20px}.toolbar{display:flex;gap:10px;flex-wrap:wrap}
input{width:min(760px,100%);border:1px solid var(--line);border-radius:6px;
padding:9px 11px;font:inherit}.layout{display:grid;grid-template-columns:150px
minmax(0,1fr);gap:16px;padding:16px 20px 40px}nav{position:sticky;top:91px;
align-self:start;max-height:calc(100vh - 110px);overflow:auto;border:1px solid
var(--line);border-radius:8px;background:var(--panel);padding:8px}.nav-item{
display:inline-block;width:52px;margin:3px;padding:5px 0;border-radius:5px;
color:var(--text);text-align:center;text-decoration:none;font-size:13px;
background:var(--soft)}main{min-width:0}.card{margin-bottom:16px;border:1px
solid var(--line);border-radius:8px;background:var(--panel);overflow:hidden}
.card-header{display:flex;align-items:baseline;justify-content:space-between;
gap:18px;border-bottom:1px solid var(--line);padding:12px 14px;background:#fbfcfe}
h2{margin:0;font-size:16px}.gold{color:var(--muted);text-align:right}.gold span,
.question span{margin-right:8px;color:var(--muted);font-size:12px;font-weight:700;
text-transform:uppercase}.gold strong{color:var(--text);font-size:15px}.question{
border-bottom:1px solid var(--line);padding:10px 14px}.question p{margin:6px 0 0;
white-space:pre-wrap}table{width:100%;border-collapse:collapse}th,td{
border-bottom:1px solid var(--line);padding:9px 12px;vertical-align:top;
text-align:left}tr:last-child th,tr:last-child td{border-bottom:0}th{width:310px;
color:var(--muted);font-size:13px;background:#fbfcfe}td{white-space:pre-wrap;
overflow-wrap:anywhere}tr.exact td{background:var(--exact)}tr.contains td{
background:var(--contains)}tr.empty td{color:var(--muted);background:var(--empty)}
.hidden{display:none}@media(max-width:860px){.layout{display:block;padding:12px}
nav{position:static;max-height:none;margin-bottom:12px}.card-header{display:block}
.gold{text-align:left;margin-top:8px}th,td{display:block;width:100%}}
"""
    nav = "".join(
        f'<a class="nav-item" href="#request-{index}">#{index}</a>'
        for index in range(len(examples))
    )
    with output_path.open("w", encoding="utf-8") as output_file:
        output_file.write(
            "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{html.escape(dataset.stem)} answer inspection</title>"
            f"<style>{css}</style></head><body><header><h1>"
            f"{html.escape(dataset.stem)} answer inspection</h1><div class=\"toolbar\">"
            '<input id="filter" type="search" placeholder="Filter request, gold, '
            'experiment, or generated text"></div></header><div class="layout">'
            f'<nav aria-label="Requests">{nav}</nav><main>'
        )
        for request_id, example in enumerate(examples):
            gold = " | ".join(flatten_answers(example.get("answers", [])))
            question = str(example.get("question", ""))
            output_file.write(
                f'<section class="card" id="request-{request_id}"><div '
                f'class="card-header"><h2>Request {request_id}</h2><div class="gold">'
                f'<span>Gold</span><strong>{html.escape(gold)}</strong></div></div>'
                f'<div class="question"><span>Question</span><p>{html.escape(question)}'
                "</p></div><table><tbody>"
            )
            for record in records:
                label = record.experiment.label
                answer = answer_maps[label].get(request_id, "")
                output_file.write(
                    f'<tr class="{answer_class(gold, answer)}"><th>'
                    f"{html.escape(label)}</th><td>{html.escape(answer)}</td></tr>"
                )
            output_file.write("</tbody></table></section>")
        output_file.write(
            "</main></div><script>const f=document.getElementById('filter');"
            "const c=Array.from(document.querySelectorAll('.card'));"
            "f.addEventListener('input',()=>{const n=f.value.trim().toLowerCase();"
            "for(const x of c){x.classList.toggle('hidden',n&&!x.textContent."
            "toLowerCase().includes(n));}});</script></body></html>"
        )


def write_metrics_tsv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a machine-readable companion TSV.

    Args:
        rows: Meeting-metric rows.
        output_path: TSV destination.
    """
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=WORKBOOK_COLUMNS,
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def excel_column(index: int) -> str:
    """Convert a one-based column number to Excel letters.

    Args:
        index: One-based column number.

    Returns:
        Excel column letters.
    """
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def excel_cell(reference: str, value: Any, style: int) -> str:
    """Render one inline-string or numeric worksheet cell.

    Args:
        reference: Excel cell reference.
        value: Cell value.
        style: Workbook cell-style index.

    Returns:
        Worksheet cell XML, or an empty string for an empty value.
    """
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{reference}" s="{style}"><v>{value}</v></c>'
    text = escape(str(value))
    return (
        f'<c r="{reference}" s="{style}" t="inlineStr"><is>'
        f'<t xml:space="preserve">{text}</t></is></c>'
    )


def worksheet_xml(rows: list[dict[str, Any]]) -> str:
    """Build worksheet XML with repeated dataset headers.

    Args:
        rows: Meeting-metric rows.

    Returns:
        Complete worksheet XML document.
    """
    sheet_rows: list[tuple[bool, Optional[dict[str, Any]]]] = []
    dataset_names = dict.fromkeys(row["Dataset"] for row in rows)
    for dataset_index, dataset_name in enumerate(dataset_names):
        if dataset_index:
            sheet_rows.append((False, None))
        sheet_rows.append((True, None))
        sheet_rows.extend(
            (False, row) for row in rows if row["Dataset"] == dataset_name
        )
    xml_rows = []
    numeric_two = {
        "Score",
        "Precision",
        "Recall",
        "Judge Correct %",
        "Judge Mostly %",
        "Judge Incorrect %",
    }
    numeric_three = {
        "Judge Score (0-1)",
        "TTFT(s)",
        "TPOT(s)",
        "Throughput(req/s)",
    }
    for row_index, (is_header, row) in enumerate(sheet_rows, start=1):
        if is_header:
            cells = "".join(
                excel_cell(f"{excel_column(column)}{row_index}", name, 1)
                for column, name in enumerate(WORKBOOK_COLUMNS, start=1)
            )
            xml_rows.append(
                f'<row r="{row_index}" ht="28" customHeight="1">'
                f"{cells}</row>"
            )
            continue
        if row is None:
            xml_rows.append(f'<row r="{row_index}" ht="8" customHeight="1"/>')
            continue
        cells = []
        for column, name in enumerate(WORKBOOK_COLUMNS, start=1):
            style = 3 if name in numeric_two else 4 if name in numeric_three else 2
            cells.append(
                excel_cell(f"{excel_column(column)}{row_index}", row.get(name), style)
            )
        xml_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    max_row = len(sheet_rows)
    widths = (22, 64, 13, 11, 11, 11, 18, 18, 18, 18, 12, 12, 22)
    columns = "".join(
        f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
        for index, width in enumerate(widths, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="A1:M{max_row}"/><sheetViews><sheetView workbookViewId="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        '</sheetView></sheetViews><sheetFormatPr defaultRowHeight="15"/>'
        f'<cols>{columns}</cols><sheetData>{"".join(xml_rows)}</sheetData>'
        '<pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" '
        'header="0.3" footer="0.3"/></worksheet>'
    )


def write_xlsx(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a styled dependency-free XLSX workbook.

    Args:
        rows: Meeting-metric rows.
        output_path: Workbook destination.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    parts = {
        "[Content_Types].xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>""",
        "_rels/.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>""",
        "docProps/app.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>LMCache Qwen3 artifact builder</Application></Properties>""",
        "docProps/core.xml": f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>Qwen3 model-only RAG meeting metrics</dc:title><dc:creator>LMCache</dc:creator><dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created></cp:coreProperties>""",
        "xl/workbook.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Qwen3 model-only" sheetId="1" r:id="rId1"/></sheets><calcPr fullCalcOnLoad="1"/></workbook>""",
        "xl/_rels/workbook.xml.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>""",
        "xl/styles.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><numFmts count="2"><numFmt numFmtId="164" formatCode="0.00"/><numFmt numFmtId="165" formatCode="0.000"/></numFmts><fonts count="2"><font><sz val="11"/><name val="Calibri"/><family val="2"/></font><font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font></fonts><fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="2"><border><left/><right/><top/><bottom/><diagonal/></border><border><left/><right/><top/><bottom style="thin"><color rgb="FFD9DEE7"/></bottom><diagonal/></border></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="5"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf><xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf><xf numFmtId="164" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyBorder="1"/><xf numFmtId="165" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyBorder="1"/></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>""",
        "xl/worksheets/sheet1.xml": worksheet_xml(rows),
    }
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as workbook:
        for name, content in parts.items():
            workbook.writestr(name, content)


def main() -> None:
    """Build all Qwen3 model-only analysis artifacts."""
    args = parse_args()
    rag_dir = args.rag_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else rag_dir / "results_eval" / "outputs" / "qwen3_model_only"
    )
    workbook = (
        args.workbook.resolve()
        if args.workbook is not None
        else rag_dir / "results" / "rag_qwen3_model_only_meeting_metrics.xlsx"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    records = build_records(rag_dir)
    judge_rows = load_judge_summary(
        args.judge_summary.resolve() if args.judge_summary is not None else None
    )
    meeting_rows = [workbook_row(record, judge_rows) for record in records]
    write_metrics_tsv(meeting_rows, workbook.with_suffix(".tsv"))
    write_xlsx(meeting_rows, workbook)
    write_manifest(records, rag_dir, output_dir / "manifest.csv")
    for dataset in DATASETS:
        write_answer_tables(
            dataset,
            [record for record in records if record.dataset == dataset],
            rag_dir,
            output_dir,
        )
    print(f"Validated canonical runs: {len(records)}")
    print(f"Workbook: {workbook}")
    print(f"Metrics TSV: {workbook.with_suffix('.tsv')}")
    print(f"Manifest and answer tables: {output_dir}")


if __name__ == "__main__":
    main()
