# SPDX-License-Identifier: Apache-2.0
"""Aggregate local LLM-judge JSONL outputs into paper-table CSVs."""

# Standard
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional
import argparse
import csv
import json


DEFAULT_INPUT = Path(__file__).resolve().parent / "outputs" / "judge_outputs.jsonl"
DEFAULT_DATASET_METHOD_OUTPUT = (
    Path(__file__).resolve().parent / "outputs" / "judge_summary_by_dataset_method.csv"
)
DEFAULT_METHOD_OUTPUT = (
    Path(__file__).resolve().parent / "outputs" / "judge_summary_by_method.csv"
)

CORRECT_LABELS = {"correct"}
MOSTLY_CORRECT_LABELS = {"mostly_correct"}
PARTIAL_LABELS = {"partially_correct"}
MINIMAL_LABELS = {"minimally_correct"}
NO_ANSWER_LABELS = {"no_answer", "no_usable_summary"}
INCORRECT_LABELS = {"incorrect", "no_answer", "no_usable_summary"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Aggregate local judge outputs by dataset and experiment."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset to include. May be provided multiple times.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=None,
        help="Experiment label to include. May be provided multiple times.",
    )
    parser.add_argument(
        "--dataset-method-output",
        type=Path,
        default=DEFAULT_DATASET_METHOD_OUTPUT,
    )
    parser.add_argument("--method-output", type=Path, default=DEFAULT_METHOD_OUTPUT)
    return parser.parse_args()


def filter_rows(
    rows: list[dict[str, Any]],
    datasets: list[str] | None,
    experiments: list[str] | None,
) -> list[dict[str, Any]]:
    """Filter rows by dataset and experiment labels.

    Args:
        rows: Judge output rows.
        datasets: Optional dataset allowlist.
        experiments: Optional experiment-label allowlist.

    Returns:
        Filtered rows.
    """
    dataset_filter = set(datasets or [])
    experiment_filter = set(experiments or [])
    filtered = []
    for row in rows:
        experiment = str(row.get("analysis_metadata", {}).get("experiment", ""))
        if dataset_filter and row.get("dataset") not in dataset_filter:
            continue
        if experiment_filter and experiment not in experiment_filter:
            continue
        filtered.append(row)
    return filtered


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows.

    Args:
        path: Input JSONL path.

    Returns:
        Parsed rows.
    """
    rows = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON line {line_number}: {path}") from exc
    return rows


def group_rows(
    rows: Iterable[dict[str, Any]],
    keys: tuple[str, ...],
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    """Group rows by selected logical keys.

    Args:
        rows: Judge output rows.
        keys: Supported keys: ``dataset`` and ``experiment``.

    Returns:
        Mapping from group key to rows.
    """
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        values = []
        for key in keys:
            if key == "experiment":
                values.append(str(row.get("analysis_metadata", {}).get(key, "")))
            else:
                values.append(str(row.get(key, "")))
        grouped[tuple(values)].append(row)
    return grouped


def summarize_group(
    key_values: tuple[str, ...],
    key_names: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize one row group.

    Args:
        key_values: Group key values.
        key_names: Group key field names.
        rows: Rows in the group.

    Returns:
        CSV row with aggregate metrics.
    """
    parsed_rows = [
        row for row in rows if isinstance(row.get("judge_output_parsed"), dict)
    ]
    summary = dict(zip(key_names, key_values))
    summary.update(
        {
            "n": len(rows),
            "n_parsed": len(parsed_rows),
            "parse_error_rate": rate(row.get("parse_error") for row in rows),
            "mean_semantic_score": mean(
                parsed.get("semantic_score")
                for parsed in (row["judge_output_parsed"] for row in parsed_rows)
            ),
            "correct_rate": label_rate(parsed_rows, CORRECT_LABELS),
            "mostly_correct_rate": label_rate(parsed_rows, MOSTLY_CORRECT_LABELS),
            "partial_rate": label_rate(parsed_rows, PARTIAL_LABELS),
            "minimal_rate": label_rate(parsed_rows, MINIMAL_LABELS),
            "incorrect_rate": label_rate(parsed_rows, INCORRECT_LABELS),
            "no_answer_rate": label_rate(parsed_rows, NO_ANSWER_LABELS),
            "contradiction_rate": bool_field_rate(
                parsed_rows,
                "contradicts_gold_answer",
            ),
            "judge_repetition_rate": bool_field_rate(
                parsed_rows,
                "repetition_detected",
            ),
            "judge_prompt_continuation_rate": bool_field_rate(
                parsed_rows,
                "prompt_continuation_detected",
            ),
            "rule_repetition_rate": diagnostic_rate(rows, "repetition_detected"),
            "rule_prompt_continuation_rate": diagnostic_rate(
                rows,
                "prompt_continuation_detected",
            ),
            "max_token_rate": diagnostic_rate(rows, "max_token_detected"),
            "empty_output_rate": diagnostic_rate(rows, "empty_output"),
            "very_short_output_rate": diagnostic_rate(rows, "very_short_output"),
            "mean_quality": mean_metadata(rows, "quality"),
            "mean_f1_word": mean_metadata(rows, "f1_word"),
            "mean_f1_tokenizer": mean_metadata(rows, "f1_tokenizer"),
        }
    )
    return summary


def rate(values: Iterable[Any]) -> Optional[float]:
    """Return the non-empty/error rate for an iterable.

    Args:
        values: Values where truthy values count as positive.

    Returns:
        Positive fraction, or ``None`` for empty input.
    """
    values_list = list(values)
    if not values_list:
        return None
    return sum(1 for value in values_list if value) / len(values_list)


def mean(values: Iterable[Any]) -> Optional[float]:
    """Return the mean of numeric values.

    Args:
        values: Input values.

    Returns:
        Mean, or ``None`` if no numeric values are present.
    """
    numeric = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def label_rate(rows: list[dict[str, Any]], labels: set[str]) -> Optional[float]:
    """Return semantic-label rate.

    Args:
        rows: Parsed judge rows.
        labels: Labels counted as positive.

    Returns:
        Positive fraction, or ``None`` for no parsed rows.
    """
    if not rows:
        return None
    return (
        sum(
            1
            for row in rows
            if row["judge_output_parsed"].get("semantic_label") in labels
        )
        / len(rows)
    )


def bool_field_rate(rows: list[dict[str, Any]], field: str) -> Optional[float]:
    """Return a boolean parsed-output field rate.

    Args:
        rows: Parsed judge rows.
        field: Parsed-output field name.

    Returns:
        Positive fraction, or ``None`` for no parsed rows.
    """
    values = [
        row["judge_output_parsed"].get(field)
        for row in rows
        if field in row["judge_output_parsed"]
    ]
    if not values:
        return None
    return sum(1 for value in values if value is True) / len(values)


def diagnostic_rate(rows: list[dict[str, Any]], field: str) -> Optional[float]:
    """Return a rule-diagnostic boolean field rate.

    Args:
        rows: Judge output rows.
        field: Diagnostic field name.

    Returns:
        Positive fraction, or ``None`` for no rows.
    """
    if not rows:
        return None
    return (
        sum(1 for row in rows if row.get("rule_diagnostics", {}).get(field) is True)
        / len(rows)
    )


def mean_metadata(rows: list[dict[str, Any]], field: str) -> Optional[float]:
    """Return mean for a numeric analysis metadata field.

    Args:
        rows: Judge output rows.
        field: Metadata field name.

    Returns:
        Mean, or ``None``.
    """
    return mean(row.get("analysis_metadata", {}).get(field) for row in rows)


def write_summary(
    rows: list[dict[str, Any]],
    group_keys: tuple[str, ...],
    output_path: Path,
) -> None:
    """Write an aggregate CSV.

    Args:
        rows: Judge output rows.
        group_keys: Grouping keys.
        output_path: CSV output path.
    """
    grouped = group_rows(rows, group_keys)
    summaries = [
        summarize_group(key_values, group_keys, group_rows_)
        for key_values, group_rows_ in sorted(grouped.items())
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(group_keys) + [
        "n",
        "n_parsed",
        "parse_error_rate",
        "mean_semantic_score",
        "correct_rate",
        "mostly_correct_rate",
        "partial_rate",
        "minimal_rate",
        "incorrect_rate",
        "no_answer_rate",
        "contradiction_rate",
        "judge_repetition_rate",
        "judge_prompt_continuation_rate",
        "rule_repetition_rate",
        "rule_prompt_continuation_rate",
        "max_token_rate",
        "empty_output_rate",
        "very_short_output_rate",
        "mean_quality",
        "mean_f1_word",
        "mean_f1_tokenizer",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def main() -> None:
    """Run the aggregation CLI."""
    args = parse_args()
    rows = filter_rows(read_jsonl(args.input), args.dataset, args.experiment)
    write_summary(rows, ("dataset", "experiment"), args.dataset_method_output)
    write_summary(rows, ("experiment",), args.method_output)
    print(f"Wrote {args.dataset_method_output}")
    print(f"Wrote {args.method_output}")


if __name__ == "__main__":
    main()
