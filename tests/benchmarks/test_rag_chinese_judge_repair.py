# SPDX-License-Identifier: Apache-2.0
"""Check targeted Chinese judge retries through their public file functions."""

# Standard
from pathlib import Path
import copy
import json
import sys

# Third Party
import pytest


RESULTS_EVAL_DIR = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "rag" / "results_eval"
)
if str(RESULTS_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(RESULTS_EVAL_DIR))

# Local
from judge_prompts import QA_ZH_REPAIR_PROMPT_NAME, render_prometheus_prompt  # noqa: E402
from repair_chinese_judge import (  # noqa: E402
    merge,
    prepare,
    read_jsonl,
    write_jsonl,
)


def _input(experiment: str) -> dict[str, object]:
    return {
        "dataset": "dureader_longbench",
        "request_id": 7,
        "task_type": "qa",
        "prompt_name": "rag_qa_zh_semantic_correctness_v2",
        "judge_input": {
            "question": "在哪里？",
            "gold_answers": ["北京"],
            "prediction": "上海",
        },
        "analysis_metadata": {"experiment": experiment},
    }


def _output(input_row: dict[str, object], score: float | None) -> dict[str, object]:
    row = copy.deepcopy(input_row)
    row["sample_id"] = (
        f"dureader_longbench:7:{input_row['analysis_metadata']['experiment']}"
    )
    row["input_fields"] = row.pop("judge_input")
    row.update(
        {
            "judge_model": "prometheus-snapshot",
            "judge_backend": "transformers",
            "judge_prompt_version": "v2",
            "judge_output_raw": "[RESULT] 1" if score is not None else "truncated",
            "judge_output_parsed": (
                {"semantic_score": score} if score is not None else None
            ),
            "parse_error": None if score is not None else "missing [RESULT] score",
        }
    )
    return row


def test_prepare_deduplicates_and_merge_checks_control(tmp_path: Path) -> None:
    """Retry identical failed inputs once and preserve every original sample."""
    inputs = [_input("A"), _input("B"), _input("C")]
    outputs = [
        _output(inputs[0], None),
        _output(inputs[1], None),
        _output(inputs[2], 0),
    ]
    full_input = tmp_path / "input.jsonl"
    original = tmp_path / "original.jsonl"
    retry_input = tmp_path / "retry_input.jsonl"
    retry_output = tmp_path / "retry_output.jsonl"
    merged = tmp_path / "merged.jsonl"
    report_path = tmp_path / "report.json"
    write_jsonl(full_input, inputs)
    write_jsonl(original, outputs)

    report = prepare(full_input, original, retry_input, report_path)
    assert report["failed_rows"] == 2
    assert report["distinct_retry_inputs"] == 1
    assert report["groups"][0]["valid_identical_scores"] == [0.0]
    retry = read_jsonl(retry_input)[0]
    long_retry = read_jsonl(tmp_path / "retry_input_long_retry.jsonl")[0]
    assert retry["prompt_name"] == QA_ZH_REPAIR_PROMPT_NAME
    assert long_retry["prompt_name"] == inputs[0]["prompt_name"]
    assert long_retry["judge_input"] == retry["judge_input"]
    _, prompt = render_prometheus_prompt(
        "qa", retry["judge_input"], retry["prompt_name"]
    )
    assert "只输出一行：[RESULT] N" in prompt
    assert "不要写反馈、分析、解释" in prompt
    assert "评分 5" in prompt
    assert json.loads(report_path.read_text(encoding="utf-8")) == report

    repaired = _output(retry, 0)
    write_jsonl(retry_output, [repaired])
    assert merge(full_input, original, retry_input, retry_output, merged) == 2
    result = read_jsonl(merged)
    assert len(result) == 3
    assert all(row["parse_error"] is None for row in result)
    assert [row["sample_id"] for row in result] == [row["sample_id"] for row in outputs]
    assert result[0]["repair_metadata"]["original_raw_output"] == "truncated"
    assert result[2] == outputs[2]
    assert read_jsonl(original) == outputs

    repaired["judge_output_parsed"]["semantic_score"] = 0.25
    write_jsonl(retry_output, [repaired])
    with pytest.raises(ValueError, match="conflicts with identical valid"):
        merge(full_input, original, retry_input, retry_output, merged)

    long_retry_path = tmp_path / "retry_input_long_retry.jsonl"
    write_jsonl(retry_output, [_output(long_retry, 0)])
    assert merge(full_input, original, long_retry_path, retry_output, merged) == 2
    assert (
        read_jsonl(merged)[0]["repair_metadata"]["repair_prompt_name"]
        == (inputs[0]["prompt_name"])
    )
