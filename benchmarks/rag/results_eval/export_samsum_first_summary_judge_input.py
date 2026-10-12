# SPDX-License-Identifier: Apache-2.0
"""Export defensible first-summary SAMSum judge inputs.

This script builds a clean SAMSum evaluation set from the meeting Excel table.
It uses the workbook display rows as the source of truth for which methods are
reported, resolves each display row to the archived raw benchmark output, and
applies one deterministic postprocessing rule:

* keep the first generated summary;
* cut only when the model starts another few-shot example, detected by
  ``# #`` or a new ``Dialogue:`` marker;
* remove a leading ``Summary:`` wrapper.

The script never rewrites, paraphrases, or repairs summary content. It writes
the clipped judge input plus audit CSV files documenting every transformation.
"""

# Standard
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from zipfile import ZipFile
import argparse
import csv
import io
import json
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

RAG_DIR = Path(__file__).resolve().parents[1]
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

# First Party
from build_answer_tables import flatten_answers, load_dataset
from judge_diagnostics import compute_rule_diagnostics
from judge_prompts import build_prompt_metadata
from utils import compute_rl


SAMSUM_DATASET = "samsum_longbench"
SAMSUM_WORKBOOK_NAME = "SAMSum (LongBench)"
JUDGE_INPUT_SCHEMA_VERSION = "reference_guided_v2"
SUMMARIZATION_JUDGE_PROMPT_NAME = "rag_samsum_semantic_quality_v2"
POSTPROCESS_VERSION = "samsum_first_summary_clip_v1"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_WORKBOOK = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "rag_new_experiments_meeting_metrics.xlsx"
)
DEFAULT_SAMSUM_TABLE = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "rag_new_experiments_samsum_longbench_table.csv"
)
DEFAULT_MANIFEST_06 = DEFAULT_OUTPUT_DIR / "manifest_06_blend16_latest.csv"
DEFAULT_ARCHIVE = Path.home() / "lmcache_rag_archives" / (
    "lmcache_rag_raw_local_20260813_125628.tar.zst"
)

CUT_PATTERNS = (
    ("blend_separator", re.compile(r"\s#\s#\s")),
    (
        "new_dialogue_marker",
        re.compile(r"(?:\r?\n|\r|\|)\s*Dialogue\s*:", re.IGNORECASE),
    ),
)
LEADING_SUMMARY_RE = re.compile(r"^\s*Summary\s*:\s*", re.IGNORECASE)
CELL_REF_RE = re.compile(r"([A-Z]+)")
WORKBOOK_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


@dataclass(frozen=True)
class WorkbookRow:
    """One SAMSum display row from the meeting workbook."""

    workbook_row_number: int
    display_label: str
    score_metric: str
    score: str


@dataclass(frozen=True)
class MethodSource:
    """Resolved source metadata for one SAMSum method."""

    display_label: str
    technical_label: str
    source_kind: str
    workbook_row_number: int
    run_name: str
    job_id: str
    result_member: str
    dataset_path: Path
    expected_n: Optional[int]
    score_metric: str
    workbook_score: str


@dataclass(frozen=True)
class ClipResult:
    """Text and audit metadata returned by the clipping rule."""

    text: str
    cut_reason: str
    removed_leading_summary_label: bool
    raw_char_count: int
    clipped_char_count: int
    raw_word_count: int
    clipped_word_count: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Export first-summary-clipped SAMSum judge input for every "
            "evaluable SAMSum row in the meeting workbook."
        )
    )
    parser.add_argument("--rag-dir", type=Path, default=RAG_DIR)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--samsum-table", type=Path, default=DEFAULT_SAMSUM_TABLE)
    parser.add_argument("--manifest-06", type=Path, default=DEFAULT_MANIFEST_06)
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help=(
            "Optional tar archive containing raw benchmarks/rag/results files. "
            "Local files are used first when present."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_OUTPUT_DIR
            / "judge_input_samsum_first_summary_all.xlsx_rows.jsonl"
        ),
    )
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "samsum_first_summary_all_diagnostics.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "samsum_first_summary_all_summary.csv",
    )
    parser.add_argument(
        "--method-manifest-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "samsum_first_summary_all_method_manifest.csv",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write available methods instead of failing on unresolved workbook rows.",
    )
    return parser.parse_args()


def workbook_samsum_rows(workbook_path: Path) -> list[WorkbookRow]:
    """Read SAMSum rows from the meeting workbook with the standard library.

    Args:
        workbook_path: Path to the ``.xlsx`` workbook.

    Returns:
        SAMSum display rows in workbook order, excluding blank separators.
    """
    raw_rows = read_first_worksheet(workbook_path)
    samsum_rows = []
    for row_number, values in raw_rows:
        if len(values) < 2 or values[0] != SAMSUM_WORKBOOK_NAME:
            continue
        display_label = values[1].strip()
        if not display_label:
            continue
        samsum_rows.append(
            WorkbookRow(
                workbook_row_number=row_number,
                display_label=display_label,
                score_metric=values[2].strip() if len(values) > 2 else "",
                score=values[3].strip() if len(values) > 3 else "",
            )
        )
    return samsum_rows


def read_first_worksheet(workbook_path: Path) -> list[tuple[int, list[str]]]:
    """Read the first worksheet from an XLSX file.

    Args:
        workbook_path: Workbook path.

    Returns:
        ``(row_number, values)`` tuples.
    """
    with ZipFile(workbook_path) as workbook_zip:
        sheet = ET.fromstring(workbook_zip.read("xl/worksheets/sheet1.xml"))
        rows = []
        sheet_data = sheet.find("a:sheetData", WORKBOOK_NS)
        if sheet_data is None:
            return rows
        for row in sheet_data:
            values_by_column = {}
            row_number = int(row.attrib["r"])
            for cell in row.findall("a:c", WORKBOOK_NS):
                ref = cell.attrib.get("r", "A")
                values_by_column[column_index(ref)] = cell_value(cell)
            if not values_by_column:
                continue
            max_column = max(values_by_column)
            rows.append(
                (
                    row_number,
                    [
                        values_by_column.get(index, "")
                        for index in range(max_column + 1)
                    ],
                )
            )
        return rows


def column_index(cell_ref: str) -> int:
    """Convert an Excel cell reference to a zero-based column index.

    Args:
        cell_ref: Cell reference such as ``B12``.

    Returns:
        Zero-based column index.
    """
    match = CELL_REF_RE.match(cell_ref)
    if match is None:
        return 0
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def cell_value(cell: ET.Element) -> str:
    """Return a worksheet cell value as text.

    Args:
        cell: XML cell element.

    Returns:
        Cell text.
    """
    if cell.attrib.get("t") == "inlineStr":
        return "".join(
            text.text or ""
            for text in cell.iter(
                "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"
            )
        )
    value = cell.find("a:v", WORKBOOK_NS)
    return "" if value is None else str(value.text or "")


def load_samsum_table(table_path: Path) -> dict[str, dict[str, str]]:
    """Load the SAMSum CSV mirror keyed by display label.

    Args:
        table_path: CSV table path.

    Returns:
        Mapping from display label to table row.
    """
    with table_path.open(newline="", encoding="utf-8") as table_file:
        rows = list(csv.DictReader(table_file))
    return {row["row"]: row for row in rows if row.get("dataset") == SAMSUM_DATASET}


def load_manifest_06(manifest_path: Path) -> list[dict[str, str]]:
    """Load latest 06 manifest rows for SAMSum.

    Args:
        manifest_path: Manifest CSV path.

    Returns:
        SAMSum rows in manifest order.
    """
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    return [row for row in rows if row.get("dataset") == SAMSUM_DATASET]


def resolve_method_sources(
    workbook_rows: list[WorkbookRow],
    samsum_table: dict[str, dict[str, str]],
    manifest_06_rows: list[dict[str, str]],
    rag_dir: Path,
    allow_missing: bool,
) -> tuple[list[MethodSource], list[dict[str, str]]]:
    """Resolve workbook rows to raw result files.

    Args:
        workbook_rows: SAMSum display rows from the workbook.
        samsum_table: Old/yaml05 table rows keyed by display label.
        manifest_06_rows: Latest 06 SAMSum manifest rows.
        rag_dir: ``benchmarks/rag`` directory.
        allow_missing: Whether to record missing rows instead of raising.

    Returns:
        ``(sources, skipped_rows)``.

    Raises:
        ValueError: If a workbook row cannot be resolved and ``allow_missing``
            is false.
    """
    sources = []
    skipped = []
    rows_06 = [row for row in workbook_rows if row.display_label.startswith("06 ")]
    if rows_06 and len(rows_06) != len(manifest_06_rows):
        raise ValueError(
            "Workbook 06 SAMSum row count does not match manifest count: "
            f"{len(rows_06)} != {len(manifest_06_rows)}"
        )
    manifest_06_by_display = dict(
        zip((row.display_label for row in rows_06), manifest_06_rows)
    )

    for workbook_row in workbook_rows:
        if not workbook_row.score:
            skipped.append(skipped_row(workbook_row, "empty workbook score"))
            continue

        if workbook_row.display_label.startswith("06 "):
            manifest_row = manifest_06_by_display.get(workbook_row.display_label)
            if manifest_row is None:
                skipped.append(skipped_row(workbook_row, "missing 06 manifest row"))
                continue
            result_member = relative_result_member(
                manifest_row["run_name"],
                debug=True,
            )
            sources.append(
                MethodSource(
                    display_label=workbook_row.display_label,
                    technical_label=manifest_row["label"],
                    source_kind="manifest_06_workbook_order",
                    workbook_row_number=workbook_row.workbook_row_number,
                    run_name=manifest_row["run_name"],
                    job_id=manifest_row["job_id"],
                    result_member=result_member,
                    dataset_path=Path(manifest_row["dataset_path"]),
                    expected_n=parse_optional_int(manifest_row.get("row_count", "")),
                    score_metric=workbook_row.score_metric,
                    workbook_score=workbook_row.score,
                )
            )
            continue

        table_row = samsum_table.get(workbook_row.display_label)
        if table_row is None or not table_row.get("run_name"):
            reason = "missing run_name in SAMSum table"
            if not allow_missing:
                raise ValueError(f"{reason}: {workbook_row.display_label}")
            skipped.append(skipped_row(workbook_row, reason))
            continue
        result_member = relative_result_member(table_row["run_name"], debug=True)
        sources.append(
            MethodSource(
                display_label=workbook_row.display_label,
                technical_label=workbook_row.display_label,
                source_kind="samsum_table",
                workbook_row_number=workbook_row.workbook_row_number,
                run_name=table_row["run_name"],
                job_id=table_row.get("job_id", ""),
                result_member=result_member,
                dataset_path=rag_dir / "inputs" / "samsum_longbench.json",
                expected_n=parse_optional_int(table_row.get("n", "")),
                score_metric=workbook_row.score_metric,
                workbook_score=workbook_row.score,
            )
        )

    return sources, skipped


def relative_result_member(run_name: str, debug: bool) -> str:
    """Return the archive member path for a run result CSV.

    Args:
        run_name: Benchmark run name.
        debug: Whether to use the debug CSV.

    Returns:
        Archive member path relative to the repository root.
    """
    suffix = ".debug.csv" if debug else ".csv"
    return f"benchmarks/rag/results/{run_name}/{run_name}{suffix}"


def skipped_row(workbook_row: WorkbookRow, reason: str) -> dict[str, str]:
    """Build a skipped-row audit record.

    Args:
        workbook_row: Workbook row.
        reason: Skip reason.

    Returns:
        CSV-ready dictionary.
    """
    return {
        "workbook_row_number": str(workbook_row.workbook_row_number),
        "display_label": workbook_row.display_label,
        "skip_reason": reason,
    }


def read_result_rows(
    source: MethodSource,
    repo_root: Path,
    archive_path: Optional[Path],
) -> list[dict[str, str]]:
    """Read one method's raw result rows from disk or archive.

    Args:
        source: Method source metadata.
        repo_root: Repository root.
        archive_path: Optional tar archive containing raw results.

    Returns:
        Result CSV rows.
    """
    local_path = repo_root / source.result_member
    if local_path.exists():
        text = local_path.read_text(encoding="utf-8", errors="replace")
    elif archive_path is not None and archive_path.exists():
        text = read_archive_member(archive_path, source.result_member)
    else:
        raise FileNotFoundError(
            f"Missing result CSV for {source.display_label}: {source.result_member}"
        )
    return list(csv.DictReader(io.StringIO(text)))


def read_archive_member(archive_path: Path, member: str) -> str:
    """Read a member from a tar archive without extracting it.

    Args:
        archive_path: Tar archive path.
        member: Member path inside the archive.

    Returns:
        Decoded member content.

    Raises:
        FileNotFoundError: If the member cannot be read.
    """
    process = subprocess.run(
        ["tar", "-xOf", str(archive_path), member],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        raise FileNotFoundError(
            f"Failed to read {member} from {archive_path}: "
            f"{process.stderr.decode('utf-8', errors='replace')}"
        )
    return process.stdout.decode("utf-8", errors="replace")


@contextmanager
def staged_result_root(
    sources: list[MethodSource],
    repo_root: Path,
    archive_path: Optional[Path],
) -> Iterator[Path]:
    """Stage all missing result CSVs from an archive in one pass.

    Args:
        sources: Method sources whose result members may need staging.
        repo_root: Repository root used for already-present local files.
        archive_path: Optional tar archive containing missing files.

    Yields:
        Repository-like root containing all required result files.

    Raises:
        FileNotFoundError: If files are missing and no archive is available.
        RuntimeError: If tar extraction fails.
    """
    missing_members = sorted(
        {
            source.result_member
            for source in sources
            if not (repo_root / source.result_member).exists()
        }
    )
    if not missing_members:
        yield repo_root
        return
    if archive_path is None or not archive_path.exists():
        raise FileNotFoundError(
            "Some result CSVs are missing locally and no archive was found."
        )

    with tempfile.TemporaryDirectory(prefix="samsum_first_summary_") as temp_dir:
        staged_root = Path(temp_dir)
        command = [
            "tar",
            "-xf",
            str(archive_path),
            "-C",
            str(staged_root),
            *missing_members,
        ]
        process = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.returncode != 0:
            raise RuntimeError(
                "Failed to stage result CSVs from archive: "
                f"{process.stderr.decode('utf-8', errors='replace')}"
            )
        yield staged_root


def clip_first_summary(prediction: str) -> ClipResult:
    """Keep the first generated summary before few-shot continuation starts.

    Args:
        prediction: Raw model output.

    Returns:
        Clipped text plus audit metadata.
    """
    raw_text = "" if prediction is None else str(prediction)
    stripped = raw_text.strip()
    cut_at: Optional[int] = None
    cut_reason = ""
    for reason, pattern in CUT_PATTERNS:
        match = pattern.search(stripped)
        if match is None or match.start() <= 0:
            continue
        if cut_at is None or match.start() < cut_at:
            cut_at = match.start()
            cut_reason = reason
    clipped = stripped[:cut_at].strip() if cut_at is not None else stripped
    removed_label = bool(LEADING_SUMMARY_RE.search(clipped))
    clipped = LEADING_SUMMARY_RE.sub("", clipped).strip()
    return ClipResult(
        text=clipped,
        cut_reason=cut_reason,
        removed_leading_summary_label=removed_label,
        raw_char_count=len(raw_text),
        clipped_char_count=len(clipped),
        raw_word_count=len(re.findall(r"\S+", raw_text)),
        clipped_word_count=len(re.findall(r"\S+", clipped)),
    )


def make_judge_row(
    source: MethodSource,
    result_row: dict[str, str],
    reference_summary: str,
    clipped: ClipResult,
    raw_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Build one first-summary judge-input row.

    Args:
        source: Method source metadata.
        result_row: Raw result CSV row.
        reference_summary: Gold SAMSum reference summary.
        clipped: Clipped prediction and audit metadata.
        raw_diagnostics: Diagnostics computed over the raw prediction.

    Returns:
        Judge-input JSON row.
    """
    return {
        "judge_input_schema_version": JUDGE_INPUT_SCHEMA_VERSION,
        "dataset": SAMSUM_DATASET,
        "task_type": "summarization",
        "request_id": int(result_row["request_id"]),
        "prompt_name": SUMMARIZATION_JUDGE_PROMPT_NAME,
        "judge_input": {
            "reference_summary": reference_summary,
            "prediction": clipped.text,
        },
        "analysis_metadata": {
            "experiment": source.display_label,
            "technical_experiment": source.technical_label,
            "source_kind": source.source_kind,
            "workbook_row_number": source.workbook_row_number,
            "run_name": source.run_name,
            "job_id": parse_optional_int(source.job_id),
            "result_csv": source.result_member,
            "success": result_row.get("success", ""),
            "error_message": result_row.get("error_message", ""),
            "generation_token_cnt": parse_optional_int(
                result_row.get("generation_token_cnt", "")
            ),
            "postprocess": POSTPROCESS_VERSION,
            "clip_cut_reason": clipped.cut_reason,
            "raw_prediction_char_count": clipped.raw_char_count,
            "clipped_prediction_char_count": clipped.clipped_char_count,
            "raw_prediction_word_count": clipped.raw_word_count,
            "clipped_prediction_word_count": clipped.clipped_word_count,
            "raw_repetition_detected": raw_diagnostics["repetition_detected"],
            "raw_prompt_continuation_detected": raw_diagnostics[
                "prompt_continuation_detected"
            ],
            "raw_max_token_detected": raw_diagnostics["max_token_detected"],
        },
    }


def build_outputs(
    sources: list[MethodSource],
    repo_root: Path,
    archive_path: Optional[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build judge rows, per-sample diagnostics, and method summaries.

    Args:
        sources: Resolved method sources.
        repo_root: Repository root.
        archive_path: Optional archive containing raw results.

    Returns:
        ``(judge_rows, diagnostic_rows, summary_rows)``.
    """
    dataset = load_dataset(sources[0].dataset_path)
    judge_rows = []
    diagnostic_rows = []
    summary_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for source in sources:
        result_rows = read_result_rows(source, repo_root, archive_path)
        if source.expected_n is not None and len(result_rows) != source.expected_n:
            print(
                "Warning: row count differs for "
                f"{source.display_label}: {len(result_rows)} != {source.expected_n}",
                file=sys.stderr,
            )
        for result_row in result_rows:
            request_id = int(result_row["request_id"])
            example = dataset[request_id]
            reference_summary = "\n".join(flatten_answers(example.get("answers", [])))
            raw_prediction = result_row.get("generated_text", "")
            raw_metadata = {
                "generation_token_cnt": parse_optional_int(
                    result_row.get("generation_token_cnt", "")
                )
            }
            raw_diagnostics = compute_rule_diagnostics(raw_prediction, raw_metadata)
            clipped = clip_first_summary(raw_prediction)
            postclip_diagnostics = compute_rule_diagnostics(clipped.text, {})
            raw_rouge = compute_rl(raw_prediction, reference_summary)
            clipped_rouge = compute_rl(clipped.text, reference_summary)
            judge_row = make_judge_row(
                source,
                result_row,
                reference_summary,
                clipped,
                raw_diagnostics,
            )
            judge_rows.append(judge_row)
            diagnostic_row = make_diagnostic_row(
                source,
                result_row,
                raw_rouge,
                clipped_rouge,
                clipped,
                raw_diagnostics,
                postclip_diagnostics,
            )
            diagnostic_rows.append(diagnostic_row)
            summary_groups[source.display_label].append(diagnostic_row)

    summary_rows = [
        summarize_method(label, rows)
        for label, rows in sorted(
            summary_groups.items(),
            key=lambda item: int(item[1][0]["workbook_row_number"]),
        )
    ]
    return judge_rows, diagnostic_rows, summary_rows


def make_diagnostic_row(
    source: MethodSource,
    result_row: dict[str, str],
    raw_rouge: float,
    clipped_rouge: float,
    clipped: ClipResult,
    raw_diagnostics: dict[str, Any],
    postclip_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Build one per-sample audit row.

    Args:
        source: Method source metadata.
        result_row: Raw result row.
        raw_rouge: ROUGE-L on the raw prediction.
        clipped_rouge: ROUGE-L on the clipped prediction.
        clipped: Clipped result metadata.
        raw_diagnostics: Diagnostics over the raw prediction.
        postclip_diagnostics: Diagnostics over the clipped prediction.

    Returns:
        CSV-ready diagnostic row.
    """
    return {
        "dataset": SAMSUM_DATASET,
        "request_id": int(result_row["request_id"]),
        "workbook_row_number": source.workbook_row_number,
        "display_label": source.display_label,
        "technical_label": source.technical_label,
        "source_kind": source.source_kind,
        "run_name": source.run_name,
        "job_id": source.job_id,
        "result_csv": source.result_member,
        "success": result_row.get("success", ""),
        "generation_token_cnt": parse_optional_int(
            result_row.get("generation_token_cnt", "")
        ),
        "raw_rouge_l": raw_rouge,
        "first_summary_rouge_l": clipped_rouge,
        "rouge_l_delta": clipped_rouge - raw_rouge,
        "clip_cut_reason": clipped.cut_reason,
        "removed_leading_summary_label": clipped.removed_leading_summary_label,
        "raw_char_count": clipped.raw_char_count,
        "first_summary_char_count": clipped.clipped_char_count,
        "raw_word_count": clipped.raw_word_count,
        "first_summary_word_count": clipped.clipped_word_count,
        "raw_repetition_detected": raw_diagnostics["repetition_detected"],
        "raw_prompt_continuation_detected": raw_diagnostics[
            "prompt_continuation_detected"
        ],
        "raw_max_token_detected": raw_diagnostics["max_token_detected"],
        "postclip_repetition_detected": postclip_diagnostics["repetition_detected"],
        "postclip_prompt_continuation_detected": postclip_diagnostics[
            "prompt_continuation_detected"
        ],
    }


def summarize_method(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one method's raw and first-summary diagnostics.

    Args:
        label: Method display label.
        rows: Per-sample diagnostic rows.

    Returns:
        CSV-ready summary row.
    """
    first = rows[0]
    return {
        "dataset": SAMSUM_DATASET,
        "workbook_row_number": first["workbook_row_number"],
        "display_label": label,
        "technical_label": first["technical_label"],
        "source_kind": first["source_kind"],
        "run_name": first["run_name"],
        "job_id": first["job_id"],
        "n": len(rows),
        "raw_rouge_l_mean": mean(row["raw_rouge_l"] for row in rows),
        "first_summary_rouge_l_mean": mean(
            row["first_summary_rouge_l"] for row in rows
        ),
        "rouge_l_delta_mean": mean(row["rouge_l_delta"] for row in rows),
        "first_summary_cut_rate": rate(bool(row["clip_cut_reason"]) for row in rows),
        "leading_summary_label_removed_rate": rate(
            bool(row["removed_leading_summary_label"]) for row in rows
        ),
        "raw_repetition_rate": rate(row["raw_repetition_detected"] for row in rows),
        "raw_prompt_continuation_rate": rate(
            row["raw_prompt_continuation_detected"] for row in rows
        ),
        "raw_max_token_rate": rate(row["raw_max_token_detected"] for row in rows),
        "postclip_repetition_rate": rate(
            row["postclip_repetition_detected"] for row in rows
        ),
        "postclip_prompt_continuation_rate": rate(
            row["postclip_prompt_continuation_detected"] for row in rows
        ),
        "first_summary_word_count_mean": mean(
            row["first_summary_word_count"] for row in rows
        ),
    }


def mean(values: Any) -> float:
    """Return the mean of numeric values.

    Args:
        values: Iterable numeric values.

    Returns:
        Arithmetic mean.
    """
    numeric = [float(value) for value in values]
    return sum(numeric) / len(numeric) if numeric else float("nan")


def rate(values: Any) -> float:
    """Return the fraction of truthy values.

    Args:
        values: Iterable values.

    Returns:
        Truthy fraction.
    """
    values_list = list(values)
    return (
        sum(1 for value in values_list if value) / len(values_list)
        if values_list
        else float("nan")
    )


def parse_optional_int(value: Any) -> Optional[int]:
    """Parse an optional integer.

    Args:
        value: Raw value.

    Returns:
        Integer or ``None``.
    """
    try:
        if value == "" or value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows.

    Args:
        path: Output path.
        rows: JSON rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV.

    Args:
        path: Output path.
        rows: CSV rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_prompt_metadata(output_path: Path) -> None:
    """Write judge prompt metadata beside a judge input file.

    Args:
        output_path: Judge input path.
    """
    prompt_path = output_path.with_name(f"{output_path.stem}_prompts.json")
    prompt_path.write_text(
        json.dumps(build_prompt_metadata(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_method_manifest(
    path: Path,
    sources: list[MethodSource],
    skipped: list[dict[str, str]],
) -> None:
    """Write a method-level provenance manifest.

    Args:
        path: Output CSV path.
        sources: Resolved methods.
        skipped: Skipped workbook rows.
    """
    rows = [
        {
            "included": True,
            "workbook_row_number": source.workbook_row_number,
            "display_label": source.display_label,
            "technical_label": source.technical_label,
            "source_kind": source.source_kind,
            "run_name": source.run_name,
            "job_id": source.job_id,
            "result_csv": source.result_member,
            "expected_n": source.expected_n,
            "score_metric": source.score_metric,
            "workbook_score": source.workbook_score,
            "skip_reason": "",
        }
        for source in sources
    ]
    rows.extend(
        {
            "included": False,
            "workbook_row_number": row["workbook_row_number"],
            "display_label": row["display_label"],
            "technical_label": "",
            "source_kind": "",
            "run_name": "",
            "job_id": "",
            "result_csv": "",
            "expected_n": "",
            "score_metric": "",
            "workbook_score": "",
            "skip_reason": row["skip_reason"],
        }
        for row in skipped
    )
    rows.sort(key=lambda row: int(row["workbook_row_number"]))
    write_csv(path, rows)


def main() -> None:
    """Run the exporter."""
    args = parse_args()
    rag_dir = args.rag_dir.resolve()
    repo_root = rag_dir.parent.parent
    archive_path = args.archive.resolve() if args.archive else None

    workbook_rows = workbook_samsum_rows(args.workbook.resolve())
    samsum_table = load_samsum_table(args.samsum_table.resolve())
    manifest_06_rows = load_manifest_06(args.manifest_06.resolve())
    sources, skipped = resolve_method_sources(
        workbook_rows,
        samsum_table,
        manifest_06_rows,
        rag_dir,
        args.allow_missing,
    )
    with staged_result_root(sources, repo_root, archive_path) as result_root:
        judge_rows, diagnostic_rows, summary_rows = build_outputs(
            sources,
            result_root,
            None,
        )

    write_jsonl(args.output, judge_rows)
    write_prompt_metadata(args.output)
    write_csv(args.diagnostics_output, diagnostic_rows)
    write_csv(args.summary_output, summary_rows)
    write_method_manifest(args.method_manifest_output, sources, skipped)

    print(f"Workbook SAMSum rows: {len(workbook_rows)}")
    print(f"Included methods: {len(sources)}")
    print(f"Skipped rows: {len(skipped)}")
    print(f"Judge rows: {len(judge_rows)}")
    print(f"Judge input: {args.output}")
    print(f"Diagnostics: {args.diagnostics_output}")
    print(f"Summary: {args.summary_output}")
    print(f"Method manifest: {args.method_manifest_output}")


if __name__ == "__main__":
    main()
