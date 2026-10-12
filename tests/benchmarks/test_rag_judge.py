# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import csv
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
from export_judge_input import export_judge_input, make_judge_row  # noqa: E402
from judge_diagnostics import compute_rule_diagnostics  # noqa: E402
from judge_prompts import (  # noqa: E402
    QA_ZH_JUDGE_PROMPT_NAME,
    render_prometheus_prompt,
)
from run_local_judge import parse_args, validate_score_coverage  # noqa: E402


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write test rows to a CSV file.

    Args:
        path: CSV destination.
        rows: Rows with identical fields.
    """
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_export_retains_empty_predictions_and_generation_cap(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "samsum_longbench.json"
    dataset_path.write_text(
        json.dumps([{"question": "Dialogue: hi", "answers": ["A greeting."]}]),
        encoding="utf-8",
    )
    result_path = tmp_path / "result.csv"
    _write_csv(
        result_path,
        [
            {
                "request_id": "0",
                "generated_text": "",
                "generation_token_cnt": "2",
                "quality": "0.0",
                "f1_word": "0.0",
                "f1_tokenizer": "0.0",
                "success": "True",
                "error_message": "",
            }
        ],
    )
    manifest_row = {
        "dataset": "samsum_longbench",
        "label": "Test method",
        "run_name": "test_run_1",
        "job_id": "1",
        "result_csv": str(result_path),
        "dataset_path": str(dataset_path),
        "generation_max_tokens": "256",
    }
    output_path = tmp_path / "judge_input.jsonl"

    rows_written = export_judge_input(
        [manifest_row],
        output_path,
        skip_empty_predictions=False,
    )

    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert rows_written == 1
    assert exported["judge_input"]["prediction"] == ""
    assert exported["analysis_metadata"]["generation_max_tokens"] == 256
    assert exported["analysis_metadata"]["quality"] == 0.0


def test_diagnostics_prefer_per_row_generation_cap() -> None:
    diagnostics = compute_rule_diagnostics(
        "a short answer",
        {
            "generation_token_cnt": 32,
            "generation_max_tokens": 32,
        },
        max_token_threshold=2048,
    )

    assert diagnostics["max_token_detected"] is True
    assert diagnostics["max_token_threshold"] == 32


def test_diagnostics_fall_back_for_historical_manifests() -> None:
    diagnostics = compute_rule_diagnostics(
        "a short answer",
        {"generation_token_cnt": 32},
        max_token_threshold=2048,
    )

    assert diagnostics["max_token_detected"] is False
    assert diagnostics["max_token_threshold"] == 2048


def test_diagnostics_detect_chinese_prompt_continuation() -> None:
    diagnostics = compute_rule_diagnostics(
        "答案：北京\n问题：下一个？\n答案：上海\n问题：再下一个？\n答案：广州"
    )

    assert diagnostics["prompt_continuation_detected"] is True
    assert diagnostics["prompt_marker_counts"]["question_zh"] == 2


def test_chinese_judge_uses_chinese_prompt_only_for_chinese_dataset() -> None:
    result = {"request_id": "0", "generated_text": "答案：派出所", "success": "True"}
    example = {"question": "在哪里查？", "answers": ["派出所"]}
    base_manifest = {
        "label": "Control",
        "run_name": "run_1",
        "job_id": "1",
        "result_csv": "result.csv",
    }
    chinese = make_judge_row(
        {**base_manifest, "dataset": "dureader_longbench"}, result, example
    )
    english = make_judge_row(
        {**base_manifest, "dataset": "musique_longbench"}, result, example
    )
    zh_system, zh_user = render_prometheus_prompt(
        "qa", chinese["judge_input"], chinese["prompt_name"]
    )
    en_system, en_user = render_prometheus_prompt(
        "qa", english["judge_input"], english["prompt_name"]
    )

    assert chinese["prompt_name"] == QA_ZH_JUDGE_PROMPT_NAME
    assert english["prompt_name"] != QA_ZH_JUDGE_PROMPT_NAME
    assert "中文问答" in zh_system
    assert "评分 5" in zh_user
    assert "Score 5" in en_user
    assert "fair judge assistant" in en_system


def test_chinese_qa_judge_treats_multiple_references_as_alternatives() -> None:
    """Render DuReader references as independent acceptable answers."""
    _, prompt = render_prometheus_prompt(
        "qa",
        {
            "question": "控制面板在哪？",
            "gold_answers": ["从开始菜单打开。", "在我的电脑地址栏输入控制面板。"],
            "prediction": "从开始菜单打开控制面板。",
        },
        QA_ZH_JUDGE_PROMPT_NAME,
    )

    assert "可选参考答案 1：从开始菜单打开。" in prompt
    assert "可选参考答案 2：在我的电脑地址栏输入控制面板。" in prompt
    assert "与其中任意一条语义一致即可" in prompt


def test_future_judge_defaults_and_complete_score_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use a longer default and reject an unscored job despite written rows."""
    monkeypatch.setattr(sys, "argv", ["run_local_judge.py"])
    assert parse_args().max_output_tokens == 1024

    source = {
        "dataset": "dureader_longbench",
        "request_id": 0,
        "task_type": "qa",
        "prompt_name": QA_ZH_JUDGE_PROMPT_NAME,
        "judge_input": {
            "question": "第几代？",
            "gold_answers": ["第8代"],
            "prediction": "第42代",
        },
        "analysis_metadata": {"experiment": "test"},
    }
    result = {
        "sample_id": "dureader_longbench:0:test",
        "dataset": source["dataset"],
        "request_id": source["request_id"],
        "task_type": source["task_type"],
        "prompt_name": source["prompt_name"],
        "input_fields": source["judge_input"],
        "analysis_metadata": source["analysis_metadata"],
        "judge_output_parsed": {"semantic_score": 0.0},
        "parse_error": None,
    }
    validate_score_coverage([source], [result])
    with pytest.raises(ValueError, match="Missing 1 selected scores"):
        validate_score_coverage([source], [])
    result["parse_error"] = "missing [RESULT]"
    with pytest.raises(ValueError, match="Invalid or unparsed selected score"):
        validate_score_coverage([source], [result])
    result["parse_error"] = None
    result["input_fields"] = {"prediction": "wrong echo"}
    with pytest.raises(ValueError, match="Invalid or unparsed selected score"):
        validate_score_coverage([source], [result])
