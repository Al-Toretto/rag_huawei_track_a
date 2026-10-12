# SPDX-License-Identifier: Apache-2.0
"""Build and audit the Qwen3 chat/non-thinking v5 meeting workbook."""

# Standard
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import argparse
import csv
import math
import re

# Local
from build_qwen3_artifacts import (
    DATASETS,
    EXPERIMENTS,
    WORKBOOK_COLUMNS,
    DatasetSpec,
    ExperimentSpec,
    RunRecord,
    load_judge_summary,
    mean,
    numeric_values,
    optional_float,
    percent,
    write_answer_tables,
    write_metrics_tsv,
    write_xlsx,
)


RUN_PREFIX = "qwen3_chat_nothink_v5_seed42"
DEFAULT_MANIFEST = "qwen3_chat_nothink_v5_seed42_full_submission_20260907_114344.txt"
DEFAULT_WORKBOOK = "rag_qwen3_chat_nothink_v5_seed42_meeting_metrics.xlsx"
EXPECTED_METADATA = ("0.7", "0.8", "20", "0.0", "42", "qwen-non-thinking")


@dataclass(frozen=True)
class ManifestEntry:
    """One dataset/configuration pair expected in the v5 matrix."""

    dataset: DatasetSpec
    experiment: ExperimentSpec
    original_job_id: int
    original_run_name: str


@dataclass(frozen=True)
class ValidRun:
    """One execution-valid original or replacement v5 run."""

    entry: ManifestEntry
    job_id: int
    run_name: str
    result_csv: Path
    log_dir: Path
    rows: list[dict[str, str]]
    throughput: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rag-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Submission manifest; defaults to the v5 full-grid manifest.",
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=None,
        help="Output workbook; defaults under benchmarks/rag/results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Answer-page directory; defaults under results_eval/outputs.",
    )
    parser.add_argument(
        "--judge-summary",
        type=Path,
        default=None,
        help="Optional aggregate_judge_results.py dataset-method CSV.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail instead of writing blank metric cells for missing runs.",
    )
    return parser.parse_args()


def parse_run_identity(run_name: str) -> tuple[DatasetSpec, str, int]:
    """Extract dataset, experiment key, and job ID from a v5 run name.

    Args:
        run_name: Result/log directory name ending in a Slurm job ID.

    Returns:
        Dataset specification, experiment key, and job ID.

    Raises:
        ValueError: If the run name is not a supported v5 experiment.
    """
    job_match = re.search(r"_(\d+)$", run_name)
    if job_match is None or not run_name.startswith(f"{RUN_PREFIX}_"):
        raise ValueError(f"Unsupported v5 run name: {run_name}")
    without_job = run_name[: job_match.start()]
    for dataset in DATASETS:
        marker = f"_{dataset.stem}_"
        if marker in without_job:
            return (
                dataset,
                without_job.split(marker, maxsplit=1)[1],
                int(job_match.group(1)),
            )
    raise ValueError(f"No supported dataset in v5 run name: {run_name}")


def read_manifest(path: Path) -> list[ManifestEntry]:
    """Read and validate the expected 195-entry v5 submission manifest.

    Args:
        path: Full-grid submission text file.

    Returns:
        Expected entries in submission order.

    Raises:
        ValueError: If the manifest is malformed or incomplete.
    """
    specs = {experiment.key: experiment for experiment in EXPERIMENTS}
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("SUBMITTED"):
            continue
        fields = {}
        for field in line.split("\t")[1:]:
            if "=" in field:
                key, value = field.split("=", maxsplit=1)
                fields[key] = value
        job_id = int(fields["job_id"])
        experiment_name = fields["experiment"]
        dataset, experiment_key, parsed_job_id = parse_run_identity(
            f"{experiment_name}_{job_id}"
        )
        if parsed_job_id != job_id or experiment_key not in specs:
            raise ValueError(f"Unsupported manifest entry: {line}")
        entries.append(
            ManifestEntry(
                dataset=dataset,
                experiment=specs[experiment_key],
                original_job_id=job_id,
                original_run_name=f"{experiment_name}_{job_id}",
            )
        )
    expected_pairs = {
        (dataset.stem, experiment.key)
        for dataset in DATASETS
        for experiment in EXPERIMENTS
    }
    actual_pairs = {(entry.dataset.stem, entry.experiment.key) for entry in entries}
    if len(entries) != 195 or actual_pairs != expected_pairs:
        raise ValueError(
            "V5 manifest is not the expected 195-pair matrix: "
            f"entries={len(entries)}, missing={sorted(expected_pairs - actual_pairs)}, "
            f"extra={sorted(actual_pairs - expected_pairs)}"
        )
    return entries


def expected_row_count(entry: ManifestEntry) -> int:
    """Return the intended evaluated request count for a v5 run.

    Args:
        entry: Expected matrix entry.

    Returns:
        48 for SAMSum LMCache runs and 50 otherwise.
    """
    if (
        entry.dataset.stem == "samsum_longbench"
        and entry.experiment.key != "full_recompute"
    ):
        return 48
    return 50


def metric_csv(run_dir: Path) -> Optional[Path]:
    """Find the unique non-debug metric CSV in a result directory.

    Args:
        run_dir: Candidate result directory.

    Returns:
        Metric CSV, or ``None`` if there is not exactly one.
    """
    paths = [
        path for path in run_dir.glob("*.csv") if not path.name.endswith(".debug.csv")
    ]
    return paths[0] if len(paths) == 1 else None


def metadata_tuple(row: dict[str, str]) -> tuple[str, ...]:
    """Return normalized generation metadata from one result row.

    Args:
        row: Result CSV row.

    Returns:
        Temperature, top-p, top-k, min-p, seed, and chat mode.
    """
    return (
        str(float(row["temperature"])),
        str(float(row["top_p"])),
        str(int(row["top_k"])),
        str(float(row["min_p"])),
        str(int(row["generation_seed"])),
        row["chat_template_mode"],
    )


def parse_throughput(log_path: Path) -> float:
    """Extract benchmark throughput from a completed log.

    Args:
        log_path: Benchmark log path.

    Returns:
        Requests per second.

    Raises:
        ValueError: If the completed summary is absent.
    """
    text = log_path.read_text(errors="replace")
    if "Finished benchmarking" not in text:
        raise ValueError(f"No completion marker in {log_path}")
    match = re.search(r"throughput=([0-9.eE+-]+)", text)
    if match is None:
        raise ValueError(f"No throughput summary in {log_path}")
    return float(match.group(1))


def validate_candidate(
    entry: ManifestEntry,
    result_dir: Path,
    log_dir: Path,
) -> Optional[ValidRun]:
    """Validate a candidate original or replacement run.

    Successful zero-length model outputs are retained as measured zero-score
    predictions. They are not infrastructure failures.

    Args:
        entry: Expected dataset/configuration pair.
        result_dir: Candidate result directory.
        log_dir: Matching log directory.

    Returns:
        Valid run, or ``None`` when the candidate is incomplete or invalid.

    Raises:
        ValueError: If successful rows have inconsistent generation metadata or
            the aggregate metric disagrees with the benchmark log.
    """
    result_csv = metric_csv(result_dir)
    benchmark_log = log_dir / "benchmark.log"
    if result_csv is None or not benchmark_log.is_file():
        return None
    with result_csv.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    if len(rows) != expected_row_count(entry):
        return None
    if any(row.get("success", "").lower() != "true" for row in rows):
        return None
    bad_metadata = [row for row in rows if metadata_tuple(row) != EXPECTED_METADATA]
    if bad_metadata:
        raise ValueError(f"Unexpected generation metadata in {result_csv}")
    throughput = parse_throughput(benchmark_log)
    score_field = (
        "quality" if entry.dataset.score_metric == "ROUGE-L" else "f1_tokenizer"
    )
    score_values = numeric_values(rows, score_field)
    score = mean(score_values)
    log_text = benchmark_log.read_text(errors="replace")
    log_field = (
        "average_quality"
        if entry.dataset.score_metric == "ROUGE-L"
        else "average_f1_tokenizer"
    )
    log_match = re.search(rf"{log_field}=([0-9.eE+-]+)", log_text)
    if (
        score is None
        or log_match is None
        or not math.isclose(
            score,
            float(log_match.group(1)),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"Metric/log mismatch for {result_csv}")
    _, _, job_id = parse_run_identity(result_dir.name)
    return ValidRun(
        entry=entry,
        job_id=job_id,
        run_name=result_dir.name,
        result_csv=result_csv,
        log_dir=log_dir,
        rows=rows,
        throughput=throughput,
    )


def discover_runs(
    rag_dir: Path,
    entries: list[ManifestEntry],
) -> tuple[dict[tuple[str, str], ValidRun], list[str]]:
    """Discover valid originals and later v5 replacement runs.

    Args:
        rag_dir: RAG benchmark directory.
        entries: Expected matrix entries.

    Returns:
        Canonical run by dataset/configuration and human-readable audit notes.
    """
    entries_by_pair = {
        (entry.dataset.stem, entry.experiment.key): entry for entry in entries
    }
    valid_by_pair: dict[tuple[str, str], ValidRun] = {}
    notes = []
    for result_dir in sorted((rag_dir / "results").glob(f"{RUN_PREFIX}_*")):
        if not result_dir.is_dir() or "_diag10_" in result_dir.name:
            continue
        try:
            dataset, key, _ = parse_run_identity(result_dir.name)
        except ValueError:
            continue
        pair = (dataset.stem, key)
        entry = entries_by_pair.get(pair)
        if entry is None:
            continue
        candidate = validate_candidate(
            entry,
            result_dir,
            rag_dir / "logs" / result_dir.name,
        )
        if candidate is None:
            continue
        previous = valid_by_pair.get(pair)
        if previous is None or candidate.job_id > previous.job_id:
            valid_by_pair[pair] = candidate
    for pair, entry in entries_by_pair.items():
        run = valid_by_pair.get(pair)
        if run is None:
            notes.append(
                "MISSING\t"
                f"dataset={entry.dataset.stem}\texperiment={entry.experiment.key}"
                f"\toriginal_job_id={entry.original_job_id}"
            )
            continue
        empty_count = sum(not row.get("generated_text", "").strip() for row in run.rows)
        token_cap = 256 if entry.dataset.stem == "samsum_longbench" else 32
        cap_count = sum(
            int(row.get("generation_token_cnt", "0")) >= token_cap for row in run.rows
        )
        replacement = run.job_id != entry.original_job_id
        notes.append(
            "VALID\t"
            f"dataset={entry.dataset.stem}\texperiment={entry.experiment.key}"
            f"\tjob_id={run.job_id}\treplacement={str(replacement).lower()}"
            f"\trows={len(run.rows)}\tempty_outputs={empty_count}"
            f"\tcap_hits={cap_count}"
        )
    return valid_by_pair, notes


def workbook_row(
    entry: ManifestEntry,
    run: Optional[ValidRun],
    judge_rows: dict[tuple[str, str], dict[str, str]],
) -> dict[str, Any]:
    """Create one template-compatible workbook row.

    Args:
        entry: Expected matrix entry.
        run: Canonical valid run, or ``None`` if a rerun is required.
        judge_rows: Optional judge aggregates keyed by dataset and experiment.

    Returns:
        Meeting workbook row. Missing runs have blank numeric cells.
    """
    row: dict[str, Any] = {
        "Dataset": entry.dataset.display_name,
        "Experiment": entry.experiment.label,
        "Score Metric": entry.dataset.score_metric,
    }
    row.update({column: None for column in WORKBOOK_COLUMNS[3:]})
    if run is None:
        return row
    score_field = (
        "quality" if entry.dataset.score_metric == "ROUGE-L" else "f1_tokenizer"
    )
    score = mean(numeric_values(run.rows, score_field))
    precision = mean(numeric_values(run.rows, "precision"))
    recall = mean(numeric_values(run.rows, "recall"))
    ttft = mean([value for value in numeric_values(run.rows, "ttft") if value >= 0])
    tpot = mean([value for value in numeric_values(run.rows, "tpot") if value >= 0])
    judge = judge_rows.get((entry.dataset.stem, entry.experiment.label), {})
    row.update(
        {
            "Score": score * 100 if score is not None else None,
            "Precision": (
                precision * 100
                if precision is not None and entry.dataset.score_metric != "ROUGE-L"
                else None
            ),
            "Recall": (
                recall * 100
                if recall is not None and entry.dataset.score_metric != "ROUGE-L"
                else None
            ),
            "Judge Score (0-1)": optional_float(judge, "mean_semantic_score"),
            "Judge Correct %": percent(optional_float(judge, "correct_rate")),
            "Judge Mostly %": percent(
                optional_float(judge, "mostly_correct_rate")
            ),
            "Judge Incorrect %": percent(
                optional_float(judge, "incorrect_rate")
            ),
            "TTFT(s)": ttft,
            "TPOT(s)": tpot,
            "Throughput(req/s)": run.throughput,
        }
    )
    return row


def write_judge_manifest(
    ordered_entries: list[ManifestEntry],
    runs: dict[tuple[str, str], ValidRun],
    rag_dir: Path,
    output_path: Path,
) -> None:
    """Write valid v5 runs to the canonical judge-export manifest.

    Missing runs are omitted until a valid replacement is discovered. The
    generation cap is recorded per run so deterministic judge diagnostics use
    the actual 32-token QA or 256-token SAMSum limit.

    Args:
        ordered_entries: Expected matrix entries in publication-table order.
        runs: Valid canonical runs keyed by dataset and experiment key.
        rag_dir: RAG benchmark directory.
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
        "generation_max_tokens",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for entry in ordered_entries:
            run = runs.get((entry.dataset.stem, entry.experiment.key))
            if run is None:
                continue
            writer.writerow(
                {
                    "dataset": entry.dataset.stem,
                    "label": entry.experiment.label,
                    "group": entry.experiment.group,
                    "order": EXPERIMENTS.index(entry.experiment),
                    "run_name": run.run_name,
                    "job_id": run.job_id,
                    "row_count": len(run.rows),
                    "result_csv": run.result_csv.relative_to(repo_dir),
                    "dataset_path": (
                        rag_dir / "inputs" / f"{entry.dataset.stem}.json"
                    ).relative_to(repo_dir),
                    "generation_max_tokens": (
                        256 if entry.dataset.stem == "samsum_longbench" else 32
                    ),
                }
            )


def write_audit(notes: list[str], output_path: Path) -> None:
    """Write the run-level audit companion file.

    Args:
        notes: One tab-delimited line per expected run.
        output_path: Audit destination.
    """
    output_path.write_text("\n".join(notes) + "\n", encoding="utf-8")


def answer_record(entry: ManifestEntry, run: Optional[ValidRun]) -> RunRecord:
    """Create the record consumed by the shared answer-page writer.

    Args:
        entry: Expected dataset/configuration pair.
        run: Canonical valid run, or ``None`` for a failed experiment.

    Returns:
        A display record. Missing experiments have no answer rows, so their
        cells appear blank until a valid replacement is discovered.
    """
    if run is None:
        return RunRecord(
            dataset=entry.dataset,
            experiment=entry.experiment,
            run_name=entry.original_run_name,
            job_id=entry.original_job_id,
            result_csv=Path(),
            log_dir=Path(),
            rows=[],
            throughput=float("nan"),
        )
    return RunRecord(
        dataset=entry.dataset,
        experiment=entry.experiment,
        run_name=run.run_name,
        job_id=run.job_id,
        result_csv=run.result_csv,
        log_dir=run.log_dir,
        rows=run.rows,
        throughput=run.throughput,
    )


def main() -> None:
    """Audit v5 results and write workbook, TSV, and audit artifacts."""
    args = parse_args()
    rag_dir = args.rag_dir.resolve()
    manifest = (
        args.manifest.resolve()
        if args.manifest is not None
        else rag_dir / "logs" / DEFAULT_MANIFEST
    )
    workbook = (
        args.workbook.resolve()
        if args.workbook is not None
        else rag_dir / "results" / DEFAULT_WORKBOOK
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else rag_dir / "results_eval" / "outputs" / RUN_PREFIX
    )
    entries = read_manifest(manifest)
    runs, notes = discover_runs(rag_dir, entries)
    judge_rows = load_judge_summary(
        args.judge_summary.resolve() if args.judge_summary is not None else None
    )
    entries_by_pair = {
        (entry.dataset.stem, entry.experiment.key): entry for entry in entries
    }
    ordered_entries = [
        entries_by_pair[(dataset.stem, experiment.key)]
        for dataset in DATASETS
        for experiment in EXPERIMENTS
    ]
    rows = [
        workbook_row(
            entry,
            runs.get((entry.dataset.stem, entry.experiment.key)),
            judge_rows,
        )
        for entry in ordered_entries
    ]
    missing = len(entries) - len(runs)
    if missing and args.require_complete:
        raise ValueError(f"Refusing incomplete workbook: {missing} run(s) missing")
    workbook.parent.mkdir(parents=True, exist_ok=True)
    write_xlsx(rows, workbook)
    write_metrics_tsv(rows, workbook.with_suffix(".tsv"))
    audit_path = workbook.with_name(f"{workbook.stem}_audit.txt")
    write_audit(notes, audit_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    judge_manifest = output_dir / "manifest.csv"
    write_judge_manifest(ordered_entries, runs, rag_dir, judge_manifest)
    for dataset in DATASETS:
        dataset_entries = [
            entry for entry in ordered_entries if entry.dataset == dataset
        ]
        write_answer_tables(
            dataset,
            [
                answer_record(
                    entry,
                    runs.get((entry.dataset.stem, entry.experiment.key)),
                )
                for entry in dataset_entries
            ],
            rag_dir,
            output_dir,
        )
    empty_outputs = sum(
        not row.get("generated_text", "").strip()
        for run in runs.values()
        for row in run.rows
    )
    cap_hits = sum(
        int(row.get("generation_token_cnt", "0"))
        >= (256 if run.entry.dataset.stem == "samsum_longbench" else 32)
        for run in runs.values()
        for row in run.rows
    )
    print(f"Expected matrix rows: {len(entries)}")
    print(f"Valid runs: {len(runs)}")
    print(f"Missing runs: {missing}")
    print(f"Successful empty model outputs retained: {empty_outputs}")
    print(f"Generation-cap hits retained: {cap_hits}")
    print(f"Workbook: {workbook}")
    print(f"Metrics TSV: {workbook.with_suffix('.tsv')}")
    print(f"Audit: {audit_path}")
    print(f"Judge manifest ({len(runs)} valid runs): {judge_manifest}")
    print(f"Answer pages and wide tables: {output_dir}")


if __name__ == "__main__":
    main()
