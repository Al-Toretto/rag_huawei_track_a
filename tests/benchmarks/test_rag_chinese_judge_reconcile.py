# SPDX-License-Identifier: Apache-2.0
"""Check the two explicit score adjudications used for the Chinese run."""

# Standard
from pathlib import Path
import sys

# Third Party
import pytest


RESULTS_EVAL_DIR = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "rag" / "results_eval"
)
if str(RESULTS_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(RESULTS_EVAL_DIR))

# Local
from reconcile_chinese_judge import adjudicated_score  # noqa: E402


def test_bare_digit_is_recovered_without_guessing() -> None:
    """Accept only the exact bare score for the affected DuReader example."""
    original = {
        "dataset": "dureader_longbench",
        "request_id": 33,
        "sample_id": "dureader_longbench:33:test",
        "input_fields": {"question": "国投瑞银公司怎么样"},
    }
    long_retry = {"judge_output_parsed": None, "parse_error": "missing result"}
    score_only = {
        "judge_output_raw": "3",
        "judge_model": "model",
        "judge_backend": "transformers",
        "judge_prompt_version": "v2",
        "prompt_name": "score_only",
    }
    replacement, source = adjudicated_score(original, long_retry, score_only)
    assert source == "exact_bare_digit_from_score_only_retry"
    assert replacement["judge_output_parsed"]["semantic_score"] == 0.5
    score_only["judge_output_raw"] = "3 or 4"
    with pytest.raises(ValueError, match="Unresolved missing judge score"):
        adjudicated_score(original, long_retry, score_only)


def test_wrong_generation_requires_exact_evidence() -> None:
    """Adjudicate the repeated answer only with its exact reference and feedback."""
    original = {
        "dataset": "multifieldqa_zh_longbench",
        "request_id": 29,
        "sample_id": "multifieldqa_zh_longbench:29:test",
        "judge_prompt_version": "v2",
        "prompt_name": "qa_zh",
        "judge_output_raw": "The model's answer is incorrect because 42 is not 8.",
        "input_fields": {
            "question": "孔子后代首次获得封号是在哪一代？",
            "gold_answers": ["8世孙，孔腾被汉高祖刘邦封为奉祀君。"],
            "prediction": "孔子后代首次获得封号是在第42代孔端友。",
        },
    }
    long_retry = {
        "judge_output_raw": "The model's answer is incorrect because 42 is not 8.",
        "judge_output_parsed": None,
        "parse_error": "missing result",
    }
    replacement, source = adjudicated_score(original, long_retry, {})
    assert source == "analyst_rubric_adjudication_wrong_generation"
    assert replacement["judge_backend"] == "analyst_adjudication"
    assert replacement["judge_output_parsed"]["semantic_score"] == 0.0
    original["input_fields"]["prediction"] = "第8世孙孔腾。"
    with pytest.raises(ValueError, match="Unresolved missing judge score"):
        adjudicated_score(original, long_retry, {})
