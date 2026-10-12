# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import sys

# Third Party
import pytest


RAG_BENCH_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "rag"
if str(RAG_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_BENCH_DIR))

# Local
from rag import build_completion_sampling_kwargs, decode_prompt_escapes  # noqa: E402
from utils import (  # noqa: E402
    ChatTemplateMode,
    EvalMethodType,
    PromptBuildMethodType,
    build_fewshot_prompt,
    build_qa_prompt,
    build_rag_prompt,
    build_rag_prompt_token_ids,
    compute_tokenizer_f1,
    compute_qa_metrics_zh,
    parse_generation,
    resolve_eval_method,
    resolve_generation_stops,
    resolve_prompt_texts,
    score_prediction,
)


def test_build_completion_sampling_kwargs_uses_constant_qwen_seed() -> None:
    sampling = build_completion_sampling_kwargs(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        seed=42,
    )

    assert sampling == {
        "temperature": 0.7,
        "top_p": 0.8,
        "seed": 42,
        "extra_body": {"top_k": 20, "min_p": 0.0},
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"temperature": -0.1}, "temperature"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_k": 0}, "top_k"),
        ({"min_p": 1.1}, "min_p"),
    ],
)
def test_build_completion_sampling_kwargs_rejects_invalid_values(
    overrides: dict[str, float | int],
    message: str,
) -> None:
    values: dict[str, float | int] = {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "seed": 42,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        build_completion_sampling_kwargs(
            temperature=float(values["temperature"]),
            top_p=float(values["top_p"]),
            top_k=int(values["top_k"]),
            min_p=float(values["min_p"]),
            seed=int(values["seed"]),
        )


class DummyTokenizer:
    bos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [0] + [ord(ch) for ch in text]


class Qwen3LikeTokenizer:
    bos_token_id = None

    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]


class FakeEncoding:
    def __init__(self, input_ids: list[int]) -> None:
        self.input_ids = input_ids


class FakeQwenChatTokenizer:
    bos_token_id = None

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        return [ord(ch) for ch in text]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> FakeEncoding:
        return FakeEncoding(self.encode(text, add_special_tokens=add_special_tokens))

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert not tokenize
        assert add_generation_prompt
        assert not enable_thinking
        return (
            f"<system>{messages[0]['content']}</system>"
            f"<user>{messages[1]['content']}</user>"
            "<assistant><final>"
        )


@pytest.fixture
def dummy_tokenizer() -> DummyTokenizer:
    return DummyTokenizer()


def test_resolve_prompt_texts_keeps_default_qa_prompt_for_musique() -> None:
    system_prompt, query_prompt = resolve_prompt_texts(
        "benchmarks/rag/inputs/musique_longbench.json",
        PromptBuildMethodType.QA,
    )

    assert "within 5 words" in system_prompt
    assert "within 5 words" in query_prompt
    assert "only the final answer" in system_prompt
    assert "another question or answer" in query_prompt


def test_fewshot_prompt_instructs_model_before_final_dialogue() -> None:
    system_prompt, query_prompt = resolve_prompt_texts(
        "benchmarks/rag/inputs/samsum_longbench.json",
        PromptBuildMethodType.FEW_SHOT,
    )
    example = {
        "ctxs": [{"text": "Dialogue: Example\nSummary: Example summary"}],
        "question": "Dialogue: Final dialogue\nSummary:",
    }

    doc_prompts, final_prompt = build_fewshot_prompt(example, query_prompt)

    assert "1 to 3 short sentences" in system_prompt
    assert "do not repeat the dialogue" in system_prompt
    assert doc_prompts == ["Dialogue: Example\nSummary: Example summary"]
    assert final_prompt.startswith("\n\nNow summarize the final dialogue below.")
    assert final_prompt.endswith("Dialogue: Final dialogue\nSummary:")


def test_qwen_non_thinking_chat_framing_preserves_cached_document_ids() -> None:
    tokenizer = FakeQwenChatTokenizer()
    separator = " # # "
    system_prompt = "Answer from the passages."
    query_prompt = "\nQuestion:"
    example = {
        "ctxs": [
            {"title": "First", "text": "Alpha"},
            {"title": "Second", "text": "Beta"},
        ],
        "question": "Where",
    }

    rendered_prompt, _ = build_rag_prompt(
        system_prompt,
        example,
        query_prompt,
        separator,
        PromptBuildMethodType.QA,
        tokenizer=tokenizer,
        chat_template_mode=ChatTemplateMode.QWEN_NON_THINKING,
    )
    prompt_ids, doc_prompts = build_rag_prompt_token_ids(
        tokenizer,
        system_prompt,
        example,
        query_prompt,
        separator,
        PromptBuildMethodType.QA,
        chat_template_mode=ChatTemplateMode.QWEN_NON_THINKING,
    )
    expected_docs, query = build_qa_prompt(example, query_prompt)

    assert doc_prompts == expected_docs
    assert prompt_ids == tokenizer.encode(rendered_prompt, add_special_tokens=False)
    assert rendered_prompt == (
        f"<system>{system_prompt}</system><user>"
        f"{separator}{separator.join([*expected_docs, query])}"
        "</user><assistant><final>"
    )
    for doc_prompt in expected_docs:
        doc_ids = tokenizer.encode(doc_prompt, add_special_tokens=False)
        assert any(
            prompt_ids[start : start + len(doc_ids)] == doc_ids
            for start in range(len(prompt_ids) - len(doc_ids) + 1)
        )


def test_generation_stops_prevent_prompt_continuation() -> None:
    assert resolve_generation_stops(
        PromptBuildMethodType.QA,
        " # # ",
    ) == ["Question:", ". Answer:", ". 答案：", "# #"]
    assert resolve_generation_stops(
        PromptBuildMethodType.FEW_SHOT,
        " # # ",
    ) == ["# #", "Dialogue:"]


@pytest.mark.parametrize(
    ("generated_text", "expected"),
    [
        ("Answer: Paris", "Paris"),
        ("Answer:\nParis", "Paris"),
        ("Final answer:\nParis\nQuestion: What next?", "Paris"),
        ("答案：Paris", "Paris"),
        ("答案：\nParis", "Paris"),
    ],
)
def test_parse_generation_skips_leading_answer_label(
    generated_text: str,
    expected: str,
) -> None:
    assert parse_generation(generated_text) == expected


def test_tokenizer_f1_keeps_qwen3_first_token() -> None:
    tokenizer = Qwen3LikeTokenizer()

    assert compute_tokenizer_f1("Paris", "Paris", tokenizer) == pytest.approx(1.0)
    assert compute_tokenizer_f1("Y", "X", tokenizer) == pytest.approx(0.0)


def test_resolve_prompt_texts_relaxes_dureader_answer_length() -> None:
    system_prompt, query_prompt = resolve_prompt_texts(
        "benchmarks/rag/inputs/dureader_longbench.json",
        PromptBuildMethodType.QA,
    )

    assert "within 5 words" not in system_prompt
    assert "within 5 words" not in query_prompt
    assert "请根据材料直接回答" in query_prompt
    assert "完整句子" in system_prompt


def test_chinese_qa_prompt_preserves_question_and_answer_label() -> None:
    system_prompt, query_prompt = resolve_prompt_texts(
        "benchmarks/rag/inputs/multifieldqa_zh_longbench.json",
        PromptBuildMethodType.QA,
    )
    example = {
        "ctxs": [{"title": "", "text": "资料正文"}],
        "question": "有哪些常见的错误消息？",
    }
    _, final_prompt = build_qa_prompt(example, query_prompt)

    assert "请根据以下材料" in system_prompt
    assert final_prompt.endswith("问题：有哪些常见的错误消息？\n答案：")
    assert "?" not in final_prompt
    assert decode_prompt_escapes(system_prompt) == system_prompt
    assert decode_prompt_escapes(r"中文\n答案：") == "中文\n答案："
    assert "\n问题：" in resolve_generation_stops(
        PromptBuildMethodType.QA,
        " # # ",
        "benchmarks/rag/inputs/multifieldqa_zh_longbench.json",
    )


def test_resolve_eval_method_uses_longbench_chinese_mappings() -> None:
    assert (
        resolve_eval_method(
            "benchmarks/rag/inputs/multifieldqa_zh_longbench.json",
            PromptBuildMethodType.QA,
        )
        == EvalMethodType.QA_ZH
    )
    assert (
        resolve_eval_method(
            "benchmarks/rag/inputs/dureader_longbench.json",
            PromptBuildMethodType.QA,
        )
        == EvalMethodType.ROUGE_ZH
    )


def test_compute_qa_metrics_zh_produces_partial_overlap_scores() -> None:
    metrics = compute_qa_metrics_zh(
        "上议院领袖由首相指派。",
        "上议院领袖由首相指派产生。",
    )

    assert 0.0 < metrics["f1"] < 1.0
    assert metrics["precision"] != metrics["recall"]


def test_score_prediction_keeps_english_qa_behavior(
    dummy_tokenizer: DummyTokenizer,
) -> None:
    score = score_prediction(
        "benchmarks/rag/inputs/musique_longbench.json",
        PromptBuildMethodType.QA,
        "Paris",
        ["Paris"],
        dummy_tokenizer,
    )

    assert score.quality == pytest.approx(1.0)
    assert score.precision == pytest.approx(1.0)
    assert score.recall == pytest.approx(1.0)
    assert score.f1 == pytest.approx(1.0)
    assert score.f1_tokenizer == pytest.approx(1.0)


def test_score_prediction_uses_zh_qa_metrics_for_multifieldqa(
    dummy_tokenizer: DummyTokenizer,
) -> None:
    score = score_prediction(
        "benchmarks/rag/inputs/multifieldqa_zh_longbench.json",
        PromptBuildMethodType.QA,
        "上议院领袖由首相指派。",
        ["上议院领袖由首相指派产生。"],
        dummy_tokenizer,
    )

    assert 0.0 < score.quality < 1.0
    assert score.precision is not None
    assert score.recall is not None
    assert score.precision != score.recall
    assert score.f1_tokenizer is not None


def test_score_prediction_uses_rouge_zh_for_dureader(
    dummy_tokenizer: DummyTokenizer,
) -> None:
    score = score_prediction(
        "benchmarks/rag/inputs/dureader_longbench.json",
        PromptBuildMethodType.QA,
        "只要登记了，在派出所里都可以查到。私人无法查询。",
        ["只要登记了，在派出所里都可以查到。"],
        dummy_tokenizer,
    )

    assert score.quality > 0.0
    assert score.precision is None
    assert score.recall is None
    assert score.f1 is None
    assert score.f1_tokenizer is None
