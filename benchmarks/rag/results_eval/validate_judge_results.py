# SPDX-License-Identifier: Apache-2.0
"""Validate local judge outputs against manual labels."""

# Standard
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Optional
import argparse
import csv
import json


DEFAULT_JUDGE_OUTPUT = (
    Path(__file__).resolve().parent / "outputs" / "judge_outputs.jsonl"
)
DEFAULT_REPORT = Path(__file__).resolve().parent / "outputs" / "judge_validation.json"
DEFAULT_CONFUSION = (
    Path(__file__).resolve().parent / "outputs" / "judge_validation_confusion.csv"
)
DEFAULT_BINARY_CORRECT_LABELS = {"correct", "mostly_correct"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compare local judge outputs against manual labels."
    )
    parser.add_argument("--judge-output", type=Path, default=DEFAULT_JUDGE_OUTPUT)
    parser.add_argument("--human-labels", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--confusion-output", type=Path, default=DEFAULT_CONFUSION)
    parser.add_argument(
        "--binary-correct-label",
        action="append",
        default=None,
        help=(
            "Semantic labels counted as binary-correct. Defaults to correct "
            "and mostly_correct. May be provided multiple times."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows.

    Args:
        path: JSONL path.

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


def manual_label_for_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one manual-label row.

    Supported human label shapes:

    ``{"sample_id": "...", "semantic_label": "...", "semantic_score": 1.0}``

    or:

    ``{"sample_id": "...", "judge_output_parsed": {...}}``

    Args:
        row: Manual label row.

    Returns:
        Normalized label row.

    Raises:
        ValueError: If required fields are missing.
    """
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str):
        raise ValueError(f"Manual label row is missing sample_id: {row}")
    parsed = row.get("judge_output_parsed")
    if isinstance(parsed, dict):
        label = parsed.get("semantic_label")
        score = parsed.get("semantic_score")
    else:
        label = row.get("semantic_label")
        score = row.get("semantic_score")
    if not isinstance(label, str):
        raise ValueError(f"Manual label row is missing semantic_label: {row}")
    return {
        "sample_id": sample_id,
        "semantic_label": label,
        "semantic_score": parse_optional_float(score),
    }


def parse_optional_float(value: Any) -> Optional[float]:
    """Parse an optional float.

    Args:
        value: Value to parse.

    Returns:
        Parsed float, or ``None``.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parsed_judge_label(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Extract parsed judge label from one output row.

    Args:
        row: Judge output row.

    Returns:
        Parsed label fields, or ``None`` if unavailable.
    """
    parsed = row.get("judge_output_parsed")
    if not isinstance(parsed, dict):
        return None
    label = parsed.get("semantic_label")
    if not isinstance(label, str):
        return None
    return {
        "sample_id": row.get("sample_id"),
        "semantic_label": label,
        "semantic_score": parse_optional_float(parsed.get("semantic_score")),
    }


def validate(
    judge_rows: list[dict[str, Any]],
    human_rows: list[dict[str, Any]],
    binary_correct_labels: set[str],
) -> tuple[dict[str, Any], Counter[tuple[str, str]]]:
    """Compare judge outputs to manual labels.

    Args:
        judge_rows: Judge output rows.
        human_rows: Human label rows.
        binary_correct_labels: Labels counted as binary correct.

    Returns:
        ``(summary, confusion_matrix)``.
    """
    judge_by_id = {
        row["sample_id"]: row
        for row in (parsed_judge_label(row) for row in judge_rows)
        if row is not None and isinstance(row.get("sample_id"), str)
    }
    human_by_id = {
        row["sample_id"]: row
        for row in (manual_label_for_row(row) for row in human_rows)
    }

    matched_ids = sorted(set(judge_by_id) & set(human_by_id))
    confusion: Counter[tuple[str, str]] = Counter()
    exact_matches = 0
    binary_matches = 0
    score_deltas = []
    for sample_id in matched_ids:
        judge = judge_by_id[sample_id]
        human = human_by_id[sample_id]
        judge_label = str(judge["semantic_label"])
        human_label = str(human["semantic_label"])
        confusion[(human_label, judge_label)] += 1
        if judge_label == human_label:
            exact_matches += 1
        if is_binary_correct(judge_label, binary_correct_labels) == is_binary_correct(
            human_label,
            binary_correct_labels,
        ):
            binary_matches += 1
        if judge["semantic_score"] is not None and human["semantic_score"] is not None:
            score_deltas.append(abs(judge["semantic_score"] - human["semantic_score"]))

    summary = {
        "n_human_labels": len(human_by_id),
        "n_judge_labels": len(judge_by_id),
        "n_matched": len(matched_ids),
        "n_missing_in_judge": len(set(human_by_id) - set(judge_by_id)),
        "n_missing_in_human": len(set(judge_by_id) - set(human_by_id)),
        "exact_label_agreement": safe_div(exact_matches, len(matched_ids)),
        "binary_agreement": safe_div(binary_matches, len(matched_ids)),
        "semantic_score_mae": (
            sum(score_deltas) / len(score_deltas) if score_deltas else None
        ),
        "binary_correct_labels": sorted(binary_correct_labels),
    }
    return summary, confusion


def is_binary_correct(label: str, binary_correct_labels: set[str]) -> bool:
    """Return whether a semantic label counts as binary correct.

    Args:
        label: Semantic label.
        binary_correct_labels: Labels counted as correct.

    Returns:
        Whether the label is binary-correct.
    """
    return label in binary_correct_labels


def safe_div(numerator: int, denominator: int) -> Optional[float]:
    """Safely divide two integers.

    Args:
        numerator: Numerator.
        denominator: Denominator.

    Returns:
        Quotient, or ``None`` when denominator is zero.
    """
    if denominator == 0:
        return None
    return numerator / denominator


def write_confusion(
    confusion: Counter[tuple[str, str]],
    output_path: Path,
) -> None:
    """Write confusion matrix as long-form CSV.

    Args:
        confusion: Confusion counts keyed by ``(human_label, judge_label)``.
        output_path: CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["human_label", "judge_label", "count"],
        )
        writer.writeheader()
        for (human_label, judge_label), count in sorted(confusion.items()):
            writer.writerow(
                {
                    "human_label": human_label,
                    "judge_label": judge_label,
                    "count": count,
                }
            )


def main() -> None:
    """Run the validation CLI."""
    args = parse_args()
    binary_correct_labels = (
        set(args.binary_correct_label)
        if args.binary_correct_label is not None
        else DEFAULT_BINARY_CORRECT_LABELS
    )
    summary, confusion = validate(
        read_jsonl(args.judge_output),
        read_jsonl(args.human_labels),
        binary_correct_labels,
    )
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_confusion(confusion, args.confusion_output)
    print(f"Wrote {args.report_output}")
    print(f"Wrote {args.confusion_output}")


if __name__ == "__main__":
    main()
