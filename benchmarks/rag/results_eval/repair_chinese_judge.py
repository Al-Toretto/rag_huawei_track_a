# SPDX-License-Identifier: Apache-2.0
"""Prepare a small score-only retry and merge valid Chinese judge scores."""

# Standard
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import argparse
import copy
import json

# Local
from judge_prompts import (
    QA_ZH_JUDGE_PROMPT_NAME,
    QA_ZH_PROMETHEUS_RUBRIC,
    QA_ZH_PROMETHEUS_SCORE_ONLY_TEMPLATE,
    QA_ZH_PROMETHEUS_SYSTEM_PROMPT,
    QA_ZH_REPAIR_PROMPT_NAME,
    build_prompt_metadata,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "qwen3_chinese"
INPUT_PATH = OUTPUT_DIR / "judge_input_qwen3_chinese.jsonl"
ORIGINAL_PATH = OUTPUT_DIR / "judge_outputs_qwen3_chinese.jsonl"
REPAIR_INPUT_PATH = OUTPUT_DIR / "judge_input_qwen3_chinese_repair.jsonl"
LONG_RETRY_INPUT_PATH = OUTPUT_DIR / "judge_input_qwen3_chinese_long_retry.jsonl"
REPAIR_OUTPUT_PATH = OUTPUT_DIR / "judge_outputs_qwen3_chinese_repair.jsonl"
MERGED_PATH = OUTPUT_DIR / "judge_outputs_qwen3_chinese_complete.jsonl"
REPORT_PATH = OUTPUT_DIR / "judge_repair_plan_qwen3_chinese.json"
VALID_SCORES = {0.0, 0.25, 0.5, 0.75, 1.0}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL object rows, rejecting empty or malformed lines.

    Args:
        path: JSONL file path.

    Returns:
        Parsed rows in file order.

    Raises:
        ValueError: If any line is empty, malformed, or not an object.
    """
    rows = []
    for line_number, line in enumerate(path.open(encoding="utf-8"), start=1):
        if not line.strip():
            raise ValueError(f"Empty JSONL line {line_number}: {path}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSONL line {line_number}: {path}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Non-object JSONL line {line_number}: {path}")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL object rows without changing Chinese text.

    Args:
        path: Destination file.
        rows: JSON-serializable objects.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_id(row: dict[str, Any]) -> str:
    """Return the stable ID used by the existing judge runner.

    Args:
        row: Judge input row.

    Returns:
        Dataset, request ID, and experiment label joined by colons.
    """
    experiment = row.get("analysis_metadata", {}).get("experiment", "unknown")
    return f"{row.get('dataset')}:{row.get('request_id')}:{experiment}"


def input_key(row: dict[str, Any]) -> str:
    """Identify identical blinded QA inputs across experiment methods.

    Args:
        row: Judge input row or output row with echoed input fields.

    Returns:
        A stable JSON serialization of task, dataset, and blinded fields.
    """
    fields = row.get("judge_input", row.get("input_fields"))
    return json.dumps(
        [row.get("dataset"), row.get("task_type"), fields],
        ensure_ascii=False,
        sort_keys=True,
    )


def indexed_rows(
    rows: list[dict[str, Any]], *, output: bool
) -> dict[str, dict[str, Any]]:
    """Index judge rows and reject duplicate sample IDs.

    Args:
        rows: Judge input or output rows.
        output: Whether rows carry an explicit ``sample_id``.

    Returns:
        Mapping from stable sample ID to row.

    Raises:
        ValueError: If an ID is duplicated or malformed.
    """
    indexed = {}
    for row in rows:
        key = row.get("sample_id") if output else sample_id(row)
        if not isinstance(key, str) or key in indexed:
            raise ValueError(f"Missing or duplicate sample ID: {key}")
        indexed[key] = row
    return indexed


def validate_original(
    inputs: list[dict[str, Any]], outputs: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Check exact output coverage and echoed fields before preparing retries.

    Args:
        inputs: Canonical full judge input.
        outputs: Original full judge output.

    Returns:
        Input rows indexed by sample ID.

    Raises:
        ValueError: If IDs, fields, or experiment metadata disagree.
    """
    indexed_input = indexed_rows(inputs, output=False)
    indexed_output = indexed_rows(outputs, output=True)
    if indexed_input.keys() != indexed_output.keys():
        raise ValueError("Original judge output does not exactly cover input IDs")
    for key, output_row in indexed_output.items():
        input_row = indexed_input[key]
        if (
            output_row.get("dataset") != input_row.get("dataset")
            or output_row.get("request_id") != input_row.get("request_id")
            or output_row.get("task_type") != input_row.get("task_type")
            or output_row.get("input_fields") != input_row.get("judge_input")
            or output_row.get("analysis_metadata") != input_row.get("analysis_metadata")
            or output_row.get("prompt_name") != input_row.get("prompt_name")
        ):
            raise ValueError(f"Original judge echo mismatch: {key}")
    return indexed_input


def score_if_valid(row: dict[str, Any]) -> float | None:
    """Return a parsed semantic score only for a valid judge row.

    Args:
        row: Judge output row.

    Returns:
        The parsed score or ``None`` if the row needs a retry.
    """
    parsed = row.get("judge_output_parsed")
    if row.get("parse_error") is not None or not isinstance(parsed, dict):
        return None
    score = parsed.get("semantic_score")
    return float(score) if score in VALID_SCORES else None


def prepare(
    input_path: Path = INPUT_PATH,
    original_path: Path = ORIGINAL_PATH,
    repair_input_path: Path = REPAIR_INPUT_PATH,
    report_path: Path = REPORT_PATH,
    long_retry_input_path: Path | None = None,
) -> dict[str, Any]:
    """Write one score-only retry per distinct failed judge input.

    Args:
        input_path: Complete original judge input.
        original_path: Complete original judge output.
        repair_input_path: Destination for deduplicated retry input.
        report_path: Destination for a human-readable JSON plan.
        long_retry_input_path: Original-prompt retries with more output tokens.

    Returns:
        Retry plan with failed row and distinct input counts.

    Raises:
        ValueError: If the original output is incomplete or inconsistent.
    """
    inputs = read_jsonl(input_path)
    outputs = read_jsonl(original_path)
    indexed_input = validate_original(inputs, outputs)
    failed = [row for row in outputs if score_if_valid(row) is None]
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in failed:
        grouped[input_key(row)].append(row["sample_id"])
    repair_rows = []
    long_retry_rows = []
    groups = []
    for key, ids in grouped.items():
        representative = copy.deepcopy(indexed_input[ids[0]])
        long_retry_rows.append(copy.deepcopy(representative))
        representative["prompt_name"] = QA_ZH_REPAIR_PROMPT_NAME
        repair_rows.append(representative)
        matching_valid = {
            score
            for row in outputs
            if input_key(row) == key
            if (score := score_if_valid(row)) is not None
        }
        groups.append(
            {
                "representative_sample_id": ids[0],
                "failed_sample_ids": ids,
                "valid_identical_scores": sorted(matching_valid),
            }
        )
    report = {
        "input_rows": len(inputs),
        "failed_rows": len(failed),
        "distinct_retry_inputs": len(repair_rows),
        "groups": groups,
    }
    write_jsonl(repair_input_path, repair_rows)
    if long_retry_input_path is None:
        long_retry_input_path = repair_input_path.with_name(
            f"{repair_input_path.stem.removesuffix('_repair')}_long_retry.jsonl"
        )
    write_jsonl(long_retry_input_path, long_retry_rows)
    long_retry_metadata = copy.deepcopy(build_prompt_metadata())
    long_retry_metadata["runtime_settings"]["max_output_tokens"] = 1024
    long_retry_metadata["retry_note"] = (
        "Original Chinese QA prompt and rubric; only max_new_tokens increases "
        "from 512 to 1024."
    )
    long_retry_input_path.with_name(
        f"{long_retry_input_path.stem}_prompts.json"
    ).write_text(
        json.dumps(long_retry_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    prompt_metadata_path = repair_input_path.with_name(
        f"{repair_input_path.stem}_prompts.json"
    )
    prompt_metadata = build_prompt_metadata()
    prompt_metadata["prometheus_repair"] = {
        "prompt_name": QA_ZH_REPAIR_PROMPT_NAME,
        "system_prompt": QA_ZH_PROMETHEUS_SYSTEM_PROMPT,
        "user_prompt_template": QA_ZH_PROMETHEUS_SCORE_ONLY_TEMPLATE,
        "rubric": QA_ZH_PROMETHEUS_RUBRIC,
    }
    prompt_metadata_path.write_text(
        json.dumps(prompt_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def merge(
    input_path: Path = INPUT_PATH,
    original_path: Path = ORIGINAL_PATH,
    repair_input_path: Path = REPAIR_INPUT_PATH,
    repair_output_path: Path = REPAIR_OUTPUT_PATH,
    merged_path: Path = MERGED_PATH,
) -> int:
    """Replace only failed rows with matching, valid score-only judgments.

    Args:
        input_path: Complete original judge input.
        original_path: Original judge output, retained unchanged.
        repair_input_path: Deduplicated score-only retry input.
        repair_output_path: Score-only retry output.
        merged_path: Destination for the complete repaired output.

    Returns:
        Number of failed rows replaced.

    Raises:
        ValueError: If retry coverage, echoes, scores, or duplicate controls fail.
    """
    inputs = read_jsonl(input_path)
    originals = read_jsonl(original_path)
    validate_original(inputs, originals)
    repair_inputs = indexed_rows(read_jsonl(repair_input_path), output=False)
    repair_outputs = indexed_rows(read_jsonl(repair_output_path), output=True)
    if repair_inputs.keys() != repair_outputs.keys():
        raise ValueError("Repair output does not exactly cover retry input IDs")
    model_names = {row.get("judge_model") for row in originals}
    backends = {row.get("judge_backend") for row in originals}
    prompt_versions = {row.get("judge_prompt_version") for row in originals}
    if len(model_names) != 1 or len(backends) != 1 or len(prompt_versions) != 1:
        raise ValueError("Original judge output mixes models, backends, or versions")
    repairs_by_input = {}
    for key, repair in repair_outputs.items():
        expected = repair_inputs[key]
        if (
            expected.get("prompt_name")
            not in (QA_ZH_JUDGE_PROMPT_NAME, QA_ZH_REPAIR_PROMPT_NAME)
            or repair.get("prompt_name") != expected.get("prompt_name")
            or repair.get("dataset") != expected.get("dataset")
            or repair.get("request_id") != expected.get("request_id")
            or repair.get("task_type") != expected.get("task_type")
            or repair.get("input_fields") != expected.get("judge_input")
            or repair.get("analysis_metadata") != expected.get("analysis_metadata")
            or repair.get("judge_model") not in model_names
            or repair.get("judge_backend") not in backends
            or repair.get("judge_prompt_version") not in prompt_versions
            or score_if_valid(repair) is None
        ):
            raise ValueError(f"Invalid repair judgment: {key}")
        blinded_key = input_key(expected)
        if blinded_key in repairs_by_input:
            raise ValueError(f"Duplicate repair for blinded input: {key}")
        repairs_by_input[blinded_key] = repair
    valid_scores: dict[str, set[float]] = defaultdict(set)
    for row in originals:
        score = score_if_valid(row)
        if score is not None:
            valid_scores[input_key(row)].add(score)
    merged = []
    replaced = 0
    used_repairs = set()
    for original in originals:
        if score_if_valid(original) is not None:
            merged.append(original)
            continue
        key = input_key(original)
        repair = repairs_by_input.get(key)
        if repair is None:
            raise ValueError(f"No repair for {original['sample_id']}")
        repair_score = score_if_valid(repair)
        if valid_scores[key] and valid_scores[key] != {repair_score}:
            raise ValueError(
                f"Repair score conflicts with identical valid judgments: "
                f"{original['sample_id']}"
            )
        updated = copy.deepcopy(original)
        updated.update(
            {
                "judge_model": repair["judge_model"],
                "judge_backend": repair["judge_backend"],
                "judge_prompt_version": repair["judge_prompt_version"],
                "prompt_name": repair["prompt_name"],
                "judge_output_raw": repair["judge_output_raw"],
                "judge_output_parsed": repair["judge_output_parsed"],
                "parse_error": None,
                "repair_metadata": {
                    "source_sample_id": repair["sample_id"],
                    "repair_prompt_name": repair["prompt_name"],
                    "original_prompt_name": original["prompt_name"],
                    "original_parse_error": original["parse_error"],
                    "original_raw_output": original["judge_output_raw"],
                },
            }
        )
        merged.append(updated)
        used_repairs.add(key)
        replaced += 1
    if used_repairs != repairs_by_input.keys():
        raise ValueError("Repair output contains an unused input")
    write_jsonl(merged_path, merged)
    return replaced


def main() -> None:
    """Prepare a retry file or merge completed retry judgments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "merge"))
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--original", type=Path, default=ORIGINAL_PATH)
    parser.add_argument("--repair-input", type=Path, default=REPAIR_INPUT_PATH)
    parser.add_argument("--long-retry-input", type=Path, default=LONG_RETRY_INPUT_PATH)
    parser.add_argument("--repair-output", type=Path, default=REPAIR_OUTPUT_PATH)
    parser.add_argument("--merged", type=Path, default=MERGED_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    if args.action == "prepare":
        report = prepare(
            args.input,
            args.original,
            args.repair_input,
            args.report,
            args.long_retry_input,
        )
        print(
            f"Failed rows: {report['failed_rows']}; distinct retry inputs: "
            f"{report['distinct_retry_inputs']}; retry file: {args.repair_input}"
        )
    else:
        replaced = merge(
            args.input,
            args.original,
            args.repair_input,
            args.repair_output,
            args.merged,
        )
        print(f"Replaced {replaced} rows; complete output: {args.merged}")


if __name__ == "__main__":
    main()
