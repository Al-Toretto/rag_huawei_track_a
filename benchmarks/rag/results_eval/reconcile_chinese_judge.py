# SPDX-License-Identifier: Apache-2.0
"""Reconcile the Chinese judge runs with explicit evidence for two missing scores."""

# Standard
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import argparse
import copy
import json

# Local
from judge_prompts import QA_ZH_JUDGE_PROMPT_NAME, QA_ZH_REPAIR_PROMPT_NAME
from repair_chinese_judge import (
    INPUT_PATH,
    LONG_RETRY_INPUT_PATH,
    ORIGINAL_PATH,
    OUTPUT_DIR,
    REPAIR_INPUT_PATH,
    REPAIR_OUTPUT_PATH,
    indexed_rows,
    input_key,
    read_jsonl,
    score_if_valid,
    validate_original,
    write_jsonl,
)


LONG_RETRY_OUTPUT_PATH = OUTPUT_DIR / "judge_outputs_qwen3_chinese_long_retry.jsonl"
ADJUDICATED_PATH = OUTPUT_DIR / "judge_outputs_qwen3_chinese_adjudicated.jsonl"
AUDIT_PATH = OUTPUT_DIR / "judge_adjudication_qwen3_chinese.json"


def validated_retry_rows(
    input_path: Path, output_path: Path, prompt_name: str, model_name: str
) -> dict[str, dict[str, Any]]:
    """Validate retry coverage, echoed inputs, model, and prompt.

    Args:
        input_path: Six-row retry input JSONL.
        output_path: Matching retry output JSONL.
        prompt_name: Expected Chinese prompt name.
        model_name: Model snapshot from the original run.

    Returns:
        Retry outputs indexed by blinded judge input.

    Raises:
        ValueError: If any retry is missing, duplicated, or inconsistent.
    """
    inputs = indexed_rows(read_jsonl(input_path), output=False)
    outputs = indexed_rows(read_jsonl(output_path), output=True)
    if inputs.keys() != outputs.keys() or len(inputs) != 6:
        raise ValueError(f"Expected exactly six matched retries: {output_path}")
    by_input: dict[str, dict[str, Any]] = {}
    for sample_id, expected in inputs.items():
        row = outputs[sample_id]
        if (
            expected.get("prompt_name") != prompt_name
            or row.get("prompt_name") != prompt_name
            or row.get("dataset") != expected.get("dataset")
            or row.get("request_id") != expected.get("request_id")
            or row.get("task_type") != expected.get("task_type")
            or row.get("input_fields") != expected.get("judge_input")
            or row.get("analysis_metadata") != expected.get("analysis_metadata")
            or row.get("judge_model") != model_name
            or row.get("judge_backend") != "transformers"
        ):
            raise ValueError(f"Retry input/output mismatch: {sample_id}")
        key = input_key(row)
        if key in by_input:
            raise ValueError(f"Duplicate blinded retry input: {sample_id}")
        by_input[key] = row
    return by_input


def adjudicated_score(
    original: dict[str, Any],
    long_retry: dict[str, Any],
    score_only_retry: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Resolve only the two observed missing-score cases from recorded evidence.

    Args:
        original: Failed original judgment with the canonical prompt.
        long_retry: Failed 1024-token judgment with the canonical prompt.
        score_only_retry: Earlier format-change retry.

    Returns:
        Replacement judgment fields and a provenance label.

    Raises:
        ValueError: If the observed evidence differs from the audited cases.
    """
    fields = original["input_fields"]
    if (
        original["dataset"] == "dureader_longbench"
        and original["request_id"] == 33
        and fields["question"] == "国投瑞银公司怎么样"
        and (score_only_retry.get("judge_output_raw") or "").strip() == "3"
        and score_if_valid(long_retry) is None
    ):
        replacement = {
            "judge_model": score_only_retry["judge_model"],
            "judge_backend": score_only_retry["judge_backend"],
            "judge_prompt_version": score_only_retry["judge_prompt_version"],
            "prompt_name": score_only_retry["prompt_name"],
            "judge_output_raw": score_only_retry["judge_output_raw"],
            "judge_output_parsed": {
                "semantic_label": "partially_correct",
                "semantic_score": 0.5,
                "native_prometheus_score": 3,
                "brief_rationale": "",
            },
            "parse_error": None,
        }
        return replacement, "exact_bare_digit_from_score_only_retry"
    if (
        original["dataset"] == "multifieldqa_zh_longbench"
        and original["request_id"] == 29
        and fields["question"] == "孔子后代首次获得封号是在哪一代？"
        and fields["gold_answers"] == ["8世孙，孔腾被汉高祖刘邦封为奉祀君。"]
        and fields["prediction"] == "孔子后代首次获得封号是在第42代孔端友。"
        and str(original.get("judge_output_raw", "")).startswith(
            "The model's answer is incorrect"
        )
        and str(long_retry.get("judge_output_raw", "")).startswith(
            "The model's answer is incorrect"
        )
        and score_if_valid(long_retry) is None
    ):
        replacement = {
            "judge_model": "analyst-rubric-adjudication",
            "judge_backend": "analyst_adjudication",
            "judge_prompt_version": original["judge_prompt_version"],
            "prompt_name": original["prompt_name"],
            "judge_output_raw": original["judge_output_raw"],
            "judge_output_parsed": {
                "semantic_label": "incorrect",
                "semantic_score": 0.0,
                "native_prometheus_score": 1,
                "brief_rationale": (
                    "The prediction says 42nd generation; the reference says "
                    "8th generation. Both canonical-prompt feedback attempts "
                    "call the answer incorrect."
                ),
            },
            "parse_error": None,
        }
        return replacement, "analyst_rubric_adjudication_wrong_generation"
    raise ValueError(f"Unresolved missing judge score: {original['sample_id']}")


def reconcile(
    input_path: Path = INPUT_PATH,
    original_path: Path = ORIGINAL_PATH,
    long_input_path: Path = LONG_RETRY_INPUT_PATH,
    long_output_path: Path = LONG_RETRY_OUTPUT_PATH,
    score_input_path: Path = REPAIR_INPUT_PATH,
    score_output_path: Path = REPAIR_OUTPUT_PATH,
    output_path: Path = ADJUDICATED_PATH,
    audit_path: Path = AUDIT_PATH,
) -> dict[str, Any]:
    """Write a fully scored output while retaining per-row repair provenance.

    Args:
        input_path: Canonical 6500-row judge input.
        original_path: Full original judge output.
        long_input_path: Original-prompt 1024-token retry input.
        long_output_path: Original-prompt 1024-token retry output.
        score_input_path: Score-only retry input.
        score_output_path: Score-only retry output.
        output_path: Complete adjudicated output destination.
        audit_path: JSON audit destination.

    Returns:
        Counts by score provenance.

    Raises:
        ValueError: If coverage, model, scores, or case evidence differs.
    """
    originals = read_jsonl(original_path)
    validate_original(read_jsonl(input_path), originals)
    model_names = {row.get("judge_model") for row in originals}
    if len(model_names) != 1:
        raise ValueError("Original judge output mixes models")
    model_name = next(iter(model_names))
    long_retries = validated_retry_rows(
        long_input_path, long_output_path, QA_ZH_JUDGE_PROMPT_NAME, model_name
    )
    score_only_retries = validated_retry_rows(
        score_input_path, score_output_path, QA_ZH_REPAIR_PROMPT_NAME, model_name
    )
    if long_retries.keys() != score_only_retries.keys():
        raise ValueError("The two retries cover different blinded inputs")
    valid_controls: dict[str, set[float]] = defaultdict(set)
    for row in originals:
        score = score_if_valid(row)
        if score is not None:
            valid_controls[input_key(row)].add(score)
    merged = []
    provenance: Counter[str] = Counter()
    audit_groups: dict[str, dict[str, Any]] = {}
    used_keys = set()
    for original in originals:
        if score_if_valid(original) is not None:
            merged.append(original)
            continue
        key = input_key(original)
        if key not in long_retries:
            raise ValueError(f"Missing retry for {original['sample_id']}")
        long_retry = long_retries[key]
        score_only_retry = score_only_retries[key]
        score = score_if_valid(long_retry)
        if score is not None:
            replacement = {
                field: copy.deepcopy(long_retry[field])
                for field in (
                    "judge_model",
                    "judge_backend",
                    "judge_prompt_version",
                    "prompt_name",
                    "judge_output_raw",
                    "judge_output_parsed",
                    "parse_error",
                )
            }
            source = "canonical_prompt_1024_tokens"
        else:
            replacement, source = adjudicated_score(
                original, long_retry, score_only_retry
            )
            score = replacement["judge_output_parsed"]["semantic_score"]
        if valid_controls[key] and valid_controls[key] != {score}:
            raise ValueError(
                f"Conflicts with identical controls: {original['sample_id']}"
            )
        updated = copy.deepcopy(original)
        updated.update(replacement)
        source_sample_id = (
            score_only_retry["sample_id"]
            if source == "exact_bare_digit_from_score_only_retry"
            else long_retry["sample_id"]
        )
        updated["repair_metadata"] = {
            "source": source,
            "source_sample_id": source_sample_id,
            "original_parse_error": original["parse_error"],
            "original_raw_output": original["judge_output_raw"],
        }
        merged.append(updated)
        provenance[source] += 1
        used_keys.add(key)
        group = audit_groups.setdefault(
            key,
            {
                "representative_sample_id": long_retry["sample_id"],
                "source": source,
                "semantic_score": score,
                "affected_sample_ids": [],
                "identical_valid_control_scores": sorted(valid_controls[key]),
            },
        )
        group["affected_sample_ids"].append(original["sample_id"])
        if source == "analyst_rubric_adjudication_wrong_generation":
            group["evidence"] = {
                "question": original["input_fields"]["question"],
                "reference_answers": original["input_fields"]["gold_answers"],
                "prediction": original["input_fields"]["prediction"],
                "original_feedback_opening": original["judge_output_raw"][:260],
                "long_retry_feedback_opening": long_retry["judge_output_raw"][:260],
                "score_only_retry_opening": score_only_retry["judge_output_raw"][:260],
            }
    if used_keys != long_retries.keys() or len(merged) != len(originals):
        raise ValueError("Retry coverage is incomplete")
    if any(score_if_valid(row) is None for row in merged):
        raise ValueError("Reconciled output still has missing scores")
    audit = {
        "total_rows": len(merged),
        "original_parsed_rows": len(originals) - sum(provenance.values()),
        "repaired_rows": sum(provenance.values()),
        "provenance_counts": dict(provenance),
        "method_notes": {
            "canonical_prompt_1024_tokens": (
                "Same Chinese prompt, rubric, model, and greedy decoding as the "
                "original run; only output budget increased from 512 to 1024."
            ),
            "exact_bare_digit_from_score_only_retry": (
                "The entire model response was the single digit 3. It is mapped "
                "unambiguously to rubric score 3/5, or semantic score 0.5."
            ),
            "analyst_rubric_adjudication_wrong_generation": (
                "The prediction says 42nd generation while the reference says "
                "8th. Both original-prompt judge feedback attempts explicitly "
                "say incorrect but loop before a score. The analyst applied "
                "rubric score 1/5 (semantic score 0.0); this is not a "
                "model-emitted score or an independent human review."
            ),
        },
        "groups": list(audit_groups.values()),
    }
    write_jsonl(output_path, merged)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    """Reconcile the audited local Chinese judge outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--original", type=Path, default=ORIGINAL_PATH)
    parser.add_argument("--long-input", type=Path, default=LONG_RETRY_INPUT_PATH)
    parser.add_argument("--long-output", type=Path, default=LONG_RETRY_OUTPUT_PATH)
    parser.add_argument("--score-input", type=Path, default=REPAIR_INPUT_PATH)
    parser.add_argument("--score-output", type=Path, default=REPAIR_OUTPUT_PATH)
    parser.add_argument("--output", type=Path, default=ADJUDICATED_PATH)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    args = parser.parse_args()
    audit = reconcile(
        args.input,
        args.original,
        args.long_input,
        args.long_output,
        args.score_input,
        args.score_output,
        args.output,
        args.audit,
    )
    print(
        f"Complete rows: {audit['total_rows']}; repaired rows: "
        f"{audit['repaired_rows']}; provenance: {audit['provenance_counts']}"
    )


if __name__ == "__main__":
    main()
