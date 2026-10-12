# SPDX-License-Identifier: Apache-2.0
"""Prompt definitions for local reference-guided RAG judging."""

# Standard
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json


TASK_TYPE_QA = "qa"
TASK_TYPE_SUMMARIZATION = "summarization"

JUDGE_INPUT_SCHEMA_VERSION = "reference_guided_v2"
JUDGE_PROTOCOL = "reference_guided_llm_judge_v2"
QA_JUDGE_PROMPT_NAME = "rag_qa_semantic_correctness_v2"
QA_ZH_JUDGE_PROMPT_NAME = "rag_qa_zh_semantic_correctness_v2"
QA_ZH_REPAIR_PROMPT_NAME = "rag_qa_zh_semantic_correctness_score_only_v1"
SUMMARIZATION_JUDGE_PROMPT_NAME = "rag_samsum_semantic_quality_v2"
JUDGE_PROMPT_VERSION = "v2"

RUNTIME_SETTINGS = {
    "temperature": 0,
    "top_p": 1,
    "max_output_tokens": 1024,
}

PROMETHEUS_ABS_SYSTEM_PROMPT = (
    "You are a fair judge assistant tasked with providing clear, objective "
    "feedback based on specific criteria, ensuring each assessment reflects "
    "the absolute standards set for performance."
)

PROMETHEUS_ABSOLUTE_PROMPT_TEMPLATE = """###Task Description:
An instruction, a response to evaluate, a reference answer that gets a score of
5, and a score rubric representing evaluation criteria are given.

1. Write detailed feedback that assesses the quality of the response strictly
based on the given score rubric, not evaluating in general.
2. After writing feedback, write a score that is an integer between 1 and 5.
3. The output format must be exactly:
Feedback: (write feedback) [RESULT] (integer from 1 to 5)
4. Do not generate any other opening, closing, or explanation.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
{rubric}

###Feedback:"""

QA_PROMETHEUS_RUBRIC = """Score 5: The response gives the same answer as the
reference answer(s) and contains no contradiction. A bare concise answer like
"X" is fully correct. Harmless wrappers such as "Answer: X" or "The answer is
X" are allowed but are not required and must not be preferred over a bare
answer.
Score 4: The response gives the main correct answer but is slightly incomplete,
mildly ambiguous, or missing a minor required component.
Score 3: The response captures some important part of the reference answer but
misses a central component or is too ambiguous for higher credit.
Score 2: The response mentions the reference answer or relevant evidence, but
the final answer is wrong, contradictory, or not clearly selected.
Score 1: The response gives the wrong answer, no answer, a refusal, an
unrelated answer, or a contradiction that makes the answer unusable."""

QA_ZH_PROMETHEUS_RUBRIC = """评分 5：回答与参考答案语义一致，
包含所有关键部分，没有矛盾。简短答案和“答案：X”等无害格式均可。
评分 4：主要答案正确，但有轻微遗漏或歧义。
评分 3：部分关键内容正确，但缺少一个重要部分，或歧义较大。
评分 2：提到正确线索，但最终答案错误、矛盾或不明确。
评分 1：答案错误、空白、拒答、无关，或矛盾导致无法使用。
按含义判断中文同义表达、别名、数字和日期；不要仅凭字面重合评分。
多条编号参考答案是可选答案；与其中任意一条语义一致即可，不要求同时覆盖所有参考答案。
单条参考答案中的多个关键部分仍须适当覆盖。
重复或续写只有在影响最终答案时才降低分数。"""

QA_ZH_PROMETHEUS_SYSTEM_PROMPT = (
    "你是一名公正的中文问答评审员。请仅根据问题、模型回答和参考答案判断语义正确性，"
    "不要重新回答问题，也不要使用外部资料。"
)

QA_ZH_PROMETHEUS_TEMPLATE = """### 评审任务：
根据评分标准评估模型回答。先写简短反馈，再输出 1 到 5 的整数分数。
输出格式必须为：Feedback: 反馈内容 [RESULT] 5
将末尾的 5 替换为实际的 1 到 5 整数，不要加括号或其他文字。

### 问题：
{instruction}

### 模型回答：
{response}

### 参考答案（满分 5）：
{reference_answer}

### 评分标准：
{rubric}

### Feedback:"""

QA_ZH_PROMETHEUS_SCORE_ONLY_TEMPLATE = """### 评审任务：
根据评分标准评估模型回答。只输出一行：[RESULT] N，其中 N 是 1 到 5 的整数。
例如：[RESULT] 3。不要写反馈、分析、解释或其他文字。
不要重复问题、参考答案或模型回答。

### 问题：
{instruction}

### 模型回答：
{response}

### 参考答案（满分 5）：
{reference_answer}

### 评分标准：
{rubric}

### 评分结果（只写一行）："""

SUMMARIZATION_PROMETHEUS_RUBRIC = """Score 5: The response captures all or
nearly all central information in the reference summary with no important
hallucination.
Score 4: The response captures the main information but misses minor details,
adds minor harmless information, or is somewhat verbose or redundant without
distorting meaning.
Score 3: The response captures some important information but misses at least
one central point, is overly vague, or contains a moderate unsupported detail.
Score 2: The response has a small amount of relevant content but is dominated
by omissions, hallucination, repetition, prompt continuation, or degeneration.
Score 1: The response is unrelated, empty, mostly hallucinated, mostly copied
prompt text, mostly repeated text, or contains no usable summary."""


@dataclass(frozen=True)
class JudgePrompt:
    """Prompt package for one judge task.

    Args:
        name: Stable prompt identifier.
        task_type: Task type handled by this prompt.
        system_prompt: System instruction for the judge model.
        user_prompt_template: User prompt template with ``{{field}}`` markers.
        output_schema: Expected JSON output schema.
    """

    name: str
    task_type: str
    system_prompt: str
    user_prompt_template: str
    output_schema: dict[str, Any]


QA_JUDGE_SYSTEM_PROMPT = """You are a strict but fair evaluation judge for
question-answering outputs.

Your task is to evaluate whether a model prediction is semantically correct
with respect to the provided gold answer(s).

You must judge only the prediction, the question, and the gold answer(s). Do
not re-answer the question. Do not use external knowledge except ordinary
language understanding needed to compare meanings, aliases, paraphrases, dates,
numbers, and entity names.

The retrieved context is intentionally not provided. You are evaluating final
answer correctness, not retrieval quality or reasoning quality.

Core principles:

1. Semantic correctness is primary. A prediction is correct if it expresses the
same answer as the gold answer, even if wording, formatting, capitalization, or
harmless wrappers differ.

2. Harmless wrappers must not be penalized. These are equivalent when they
express the same answer: X, Answer: X, The answer is X, The answer: X, and
Therefore, X.

3. Verbosity is not a virtue. Do not reward long explanations, reasoning
chains, or extra detail. Do not penalize concise answers.

4. Final answer dominates. If the prediction contains the gold answer somewhere
but later gives a different final answer, mark the final answer as incorrect.
If the prediction gives multiple incompatible answers and does not clearly
select the correct one, do not mark it fully correct.

5. Contradiction matters. If the prediction both states the gold answer and
contradicts it, mark contradicts_gold_answer as true. Contradictory answers
should not receive full credit.

6. No-answer behavior is incorrect when a gold answer exists. Refusals, empty
answers, placeholders, "not enough information", and "cannot determine" are
incorrect unless the gold answer itself indicates unanswerability.

7. Multi-answer questions require appropriate coverage. If the gold answer
contains multiple required components, the prediction must include all central
components for full credit.

8. Accept equivalent numeric formats, date formats, abbreviations, and
well-known aliases when meaning is unchanged. Do not accept numerically
different values, different dates, or different entities.

9. Judge semantic equivalence, not language matching.

10. Diagnose generation pathologies separately from correctness. Repetition,
prompt continuation, copied instructions, or max-token degeneration should be
flagged, but they should not automatically reduce the semantic score if a clear
correct final answer is still present. If pathology makes the final answer
unclear, contradictory, or unusable, lower the semantic score accordingly.

Scoring rubric:

- 1.0 = Correct. The prediction gives the same answer as the gold answer(s).
It may contain harmless wrappers or extra explanation, but does not contradict
the answer.
- 0.75 = Mostly correct. The prediction gives the main correct answer but is
slightly incomplete, mildly ambiguous, or missing a minor required component.
- 0.5 = Partially correct. The prediction captures some important part of the
gold answer but misses a central component, gives an incomplete multi-part
answer, or is too ambiguous for higher credit.
- 0.25 = Minimally correct / contains useful evidence but wrong final outcome.
The prediction mentions the gold answer or relevant evidence, but the final
answer is wrong, contradictory, or not clearly selected.
- 0.0 = Incorrect. The prediction gives the wrong answer, no answer, a refusal,
an unrelated answer, or a contradiction that makes the answer unusable.

Return valid JSON only. Do not include markdown. Do not include any text
outside the JSON object."""


SUMMARIZATION_JUDGE_SYSTEM_PROMPT = """You are a strict but fair evaluation
judge for dialogue summarization outputs.

Your task is to evaluate whether a model-generated summary preserves the
important information in the reference summary.

You must judge only the dialogue summary prediction against the provided
reference summary. Do not use external knowledge. Do not infer facts that are
not supported by the reference summary.

The original dialogue is intentionally not provided. You are evaluating
semantic agreement with the reference summary, not independently
re-summarizing the dialogue.

Core principles:

1. Summary quality is semantic, not stylistic. A prediction is good if it
captures the same important events, participants, intentions, decisions,
requests, plans, or outcomes as the reference summary.

2. Wording differences are acceptable. Paraphrases, reordered information,
different sentence structure, and different but equivalent wording should not
be penalized.

3. Conciseness is not penalized. A short summary can be fully correct if it
captures the key information. A longer summary should not receive extra credit
merely for being longer.

4. Hallucination matters. Penalize invented facts, invented participants,
invented actions, wrong relationships, wrong times, wrong decisions, or
unsupported claims.

5. Missing central information matters. If the prediction omits the main
event, request, plan, decision, or outcome from the reference summary, it
cannot receive full credit.

6. Minor omissions should not be over-penalized. If the prediction captures the
main point but omits a small detail, it may be mostly correct.

7. Repetition and prompt continuation must be diagnosed separately. Repetition,
copied prompt text, repeated "Dialogue:" / "Summary:" sections, instruction
continuation, or max-token degeneration should be flagged. If a usable summary
is still clearly present, score the semantic content of the usable summary. If
degeneration dominates the output or makes the summary unusable, lower the
semantic score.

8. Do not reward generic summaries. A vague summary that could apply to many
dialogues should receive partial or low credit unless it preserves the specific
content of the reference.

9. Do not require exact reference wording. Judge meaning, not lexical overlap.

Scoring rubric:

- 1.0 = Correct / high-quality semantic match. The prediction captures all or
nearly all central information in the reference summary, with no important
hallucination.
- 0.75 = Mostly correct. The prediction captures the main information but
misses minor details, adds minor harmless information, or is somewhat verbose
or redundant without distorting meaning.
- 0.5 = Partially correct. The prediction captures some important information
but misses at least one central point, is overly vague, or contains a moderate
unsupported detail.
- 0.25 = Minimally correct. The prediction has a small amount of relevant
content but is dominated by omissions, hallucination, repetition, prompt
continuation, or degeneration.
- 0.0 = Incorrect / unusable. The prediction is unrelated, empty, mostly
hallucinated, mostly copied prompt text, mostly repeated text, or contains no
usable summary.

Return valid JSON only. Do not include markdown. Do not include any text
outside the JSON object."""


QA_USER_PROMPT_TEMPLATE = """TASK TYPE: QUESTION ANSWERING

Question:
{{question}}

Gold answer(s):
{{gold_answers}}

Model prediction:
{{prediction}}

Evaluate the model prediction according to the system rubric.

Return this exact JSON object shape:

{
  "semantic_label": "correct",
  "semantic_score": 0.0,
  "contains_gold_answer": false,
  "contradicts_gold_answer": false,
  "final_answer_correct": false,
  "multi_answer_coverage": "not_applicable",
  "answer_style": "direct",
  "repetition_detected": false,
  "prompt_continuation_detected": false,
  "usable_answer_present": false,
  "brief_rationale": ""
}"""

QA_ZH_USER_PROMPT_TEMPLATE = """任务：评估中文问答答案。
只比较问题、参考答案与模型回答的语义。
问题：{{question}}
参考答案：{{gold_answers}}
模型回答：{{prediction}}
按系统评分标准，仅输出以下结构的 JSON 对象：
{
  "semantic_label": "correct",
  "semantic_score": 1.0,
  "contains_gold_answer": true,
  "contradicts_gold_answer": false,
  "final_answer_correct": true,
  "multi_answer_coverage": "not_applicable",
  "answer_style": "direct",
  "repetition_detected": false,
  "prompt_continuation_detected": false,
  "usable_answer_present": true,
  "brief_rationale": ""
}"""

QA_ZH_JUDGE_SYSTEM_PROMPT = """你是严格且公正的中文问答评审员。
仅根据问题、参考答案和模型回答判断最终答案是否正确。
不使用外部知识，也不评价检索过程。
语义一致、同义表达、别名及等价的数字和日期格式应得满分。
“答案：”等无害前缀不扣分。若最终答案与参考答案矛盾，
或多选项中没有明确选对，不得给满分。
多部分答案须覆盖关键部分。空答案、无根据拒答和无关回答为零分。
多条参考答案是可选答案；与任意一条语义一致即可。
semantic_score 只能取 1.0、0.75、0.5、0.25、0.0，
分别表示完全正确、基本正确、部分正确、仅有线索、错误。
仅返回符合英文问答评审输出结构的 JSON。"""


SUMMARIZATION_USER_PROMPT_TEMPLATE = """TASK TYPE: DIALOGUE SUMMARIZATION

Reference summary:
{{reference_summary}}

Model prediction:
{{prediction}}

Evaluate the model prediction according to the system rubric.

Return this exact JSON object shape:

{
  "semantic_label": "correct",
  "semantic_score": 0.0,
  "reference_coverage": "high | medium | low | none",
  "factual_consistency": "fully_consistent",
  "hallucination_detected": false,
  "major_omission_detected": false,
  "repetition_detected": false,
  "prompt_continuation_detected": false,
  "usable_summary_present": false,
  "summary_style": "concise",
  "brief_rationale": ""
}"""


QA_JUDGE_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "semantic_label",
        "semantic_score",
        "contains_gold_answer",
        "contradicts_gold_answer",
        "final_answer_correct",
        "multi_answer_coverage",
        "answer_style",
        "repetition_detected",
        "prompt_continuation_detected",
        "usable_answer_present",
        "brief_rationale",
    ],
    "properties": {
        "semantic_label": {
            "type": "string",
            "enum": [
                "correct",
                "mostly_correct",
                "partially_correct",
                "minimally_correct",
                "incorrect",
                "no_answer",
            ],
        },
        "semantic_score": {"type": "number", "enum": [1.0, 0.75, 0.5, 0.25, 0.0]},
        "contains_gold_answer": {"type": "boolean"},
        "contradicts_gold_answer": {"type": "boolean"},
        "final_answer_correct": {"type": "boolean"},
        "multi_answer_coverage": {
            "type": "string",
            "enum": [
                "not_applicable",
                "complete",
                "mostly_complete",
                "partial",
                "poor",
            ],
        },
        "answer_style": {
            "type": "string",
            "enum": [
                "direct",
                "wrapped",
                "verbose",
                "reasoning_plus_answer",
                "multiple_candidates",
                "non_compliant",
                "empty",
            ],
        },
        "repetition_detected": {"type": "boolean"},
        "prompt_continuation_detected": {"type": "boolean"},
        "usable_answer_present": {"type": "boolean"},
        "brief_rationale": {"type": "string", "maxLength": 500},
    },
}


SUMMARIZATION_JUDGE_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "semantic_label",
        "semantic_score",
        "reference_coverage",
        "factual_consistency",
        "hallucination_detected",
        "major_omission_detected",
        "repetition_detected",
        "prompt_continuation_detected",
        "usable_summary_present",
        "summary_style",
        "brief_rationale",
    ],
    "properties": {
        "semantic_label": {
            "type": "string",
            "enum": [
                "correct",
                "mostly_correct",
                "partially_correct",
                "minimally_correct",
                "incorrect",
                "no_usable_summary",
            ],
        },
        "semantic_score": {"type": "number", "enum": [1.0, 0.75, 0.5, 0.25, 0.0]},
        "reference_coverage": {
            "type": "string",
            "enum": ["high", "medium", "low", "none"],
        },
        "factual_consistency": {
            "type": "string",
            "enum": [
                "fully_consistent",
                "mostly_consistent",
                "mixed",
                "mostly_inconsistent",
                "unsupported",
            ],
        },
        "hallucination_detected": {"type": "boolean"},
        "major_omission_detected": {"type": "boolean"},
        "repetition_detected": {"type": "boolean"},
        "prompt_continuation_detected": {"type": "boolean"},
        "usable_summary_present": {"type": "boolean"},
        "summary_style": {
            "type": "string",
            "enum": [
                "concise",
                "normal",
                "verbose",
                "redundant",
                "degenerate",
                "empty",
            ],
        },
        "brief_rationale": {"type": "string", "maxLength": 500},
    },
}


JUDGE_PROMPTS = {
    QA_JUDGE_PROMPT_NAME: JudgePrompt(
        name=QA_JUDGE_PROMPT_NAME,
        task_type=TASK_TYPE_QA,
        system_prompt=QA_JUDGE_SYSTEM_PROMPT,
        user_prompt_template=QA_USER_PROMPT_TEMPLATE,
        output_schema=QA_JUDGE_OUTPUT_SCHEMA,
    ),
    QA_ZH_JUDGE_PROMPT_NAME: JudgePrompt(
        name=QA_ZH_JUDGE_PROMPT_NAME,
        task_type=TASK_TYPE_QA,
        system_prompt=QA_ZH_JUDGE_SYSTEM_PROMPT,
        user_prompt_template=QA_ZH_USER_PROMPT_TEMPLATE,
        output_schema=QA_JUDGE_OUTPUT_SCHEMA,
    ),
    QA_ZH_REPAIR_PROMPT_NAME: JudgePrompt(
        name=QA_ZH_REPAIR_PROMPT_NAME,
        task_type=TASK_TYPE_QA,
        system_prompt=QA_ZH_JUDGE_SYSTEM_PROMPT,
        user_prompt_template=QA_ZH_USER_PROMPT_TEMPLATE,
        output_schema=QA_JUDGE_OUTPUT_SCHEMA,
    ),
    SUMMARIZATION_JUDGE_PROMPT_NAME: JudgePrompt(
        name=SUMMARIZATION_JUDGE_PROMPT_NAME,
        task_type=TASK_TYPE_SUMMARIZATION,
        system_prompt=SUMMARIZATION_JUDGE_SYSTEM_PROMPT,
        user_prompt_template=SUMMARIZATION_USER_PROMPT_TEMPLATE,
        output_schema=SUMMARIZATION_JUDGE_OUTPUT_SCHEMA,
    ),
}


def prompt_name_for_task(task_type: str) -> str:
    """Return the prompt name for a judge task type.

    Args:
        task_type: Judge task type.

    Returns:
        Prompt identifier used in exported rows.

    Raises:
        ValueError: If ``task_type`` is unsupported.
    """
    if task_type == TASK_TYPE_QA:
        return QA_JUDGE_PROMPT_NAME
    if task_type == TASK_TYPE_SUMMARIZATION:
        return SUMMARIZATION_JUDGE_PROMPT_NAME
    raise ValueError(f"Unsupported task type: {task_type}")


def get_prompt(prompt_name: str) -> JudgePrompt:
    """Return a prompt package by name.

    Args:
        prompt_name: Prompt identifier.

    Returns:
        Matching prompt package.

    Raises:
        ValueError: If the prompt name is unknown.
    """
    try:
        return JUDGE_PROMPTS[prompt_name]
    except KeyError as exc:
        raise ValueError(f"Unknown judge prompt: {prompt_name}") from exc


def render_user_prompt(prompt: JudgePrompt, input_fields: dict[str, Any]) -> str:
    """Render a user prompt from a prompt package and input fields.

    Args:
        prompt: Prompt package.
        input_fields: Blinded fields for the judge.

    Returns:
        Rendered user prompt.
    """
    rendered = prompt.user_prompt_template
    for key, value in input_fields.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", stringify_prompt_value(value))
    return rendered


def render_prometheus_prompt(
    task_type: str,
    input_fields: dict[str, Any],
    prompt_name: str = "",
) -> tuple[str, str]:
    """Render a Prometheus-native absolute grading prompt.

    Args:
        task_type: Judge task type.
        input_fields: Blinded judge input fields.
        prompt_name: Prompt identifier; selects Chinese criteria for Chinese QA.

    Returns:
        ``(system_prompt, user_prompt)`` for Prometheus 2.

    Raises:
        ValueError: If the task type is unsupported.
    """
    chinese_qa = task_type == TASK_TYPE_QA and prompt_name in (
        QA_ZH_JUDGE_PROMPT_NAME,
        QA_ZH_REPAIR_PROMPT_NAME,
    )
    if task_type == TASK_TYPE_QA:
        instruction = (
            "Judge whether the response answers the question. The response may "
            "be a bare answer with no wrapper phrase."
        )
        if input_fields.get("question"):
            instruction = (
                "Judge whether the response answers the question. The response "
                "may be a bare answer with no wrapper phrase.\n"
                f"Question: {input_fields['question']}"
            )
        reference_answer = format_reference_answers(
            input_fields.get("gold_answers", []), alternatives=chinese_qa
        )
        response = stringify_prompt_value(input_fields.get("prediction", ""))
        rubric = QA_ZH_PROMETHEUS_RUBRIC if chinese_qa else QA_PROMETHEUS_RUBRIC
        if chinese_qa:
            instruction = stringify_prompt_value(input_fields.get("question", ""))
    elif task_type == TASK_TYPE_SUMMARIZATION:
        instruction = "Summarize the dialogue according to the reference summary."
        reference_answer = stringify_prompt_value(
            input_fields.get("reference_summary", "")
        )
        response = stringify_prompt_value(input_fields.get("prediction", ""))
        rubric = SUMMARIZATION_PROMETHEUS_RUBRIC
    else:
        raise ValueError(f"Unsupported task type for Prometheus prompt: {task_type}")

    if prompt_name == QA_ZH_REPAIR_PROMPT_NAME:
        template = QA_ZH_PROMETHEUS_SCORE_ONLY_TEMPLATE
    elif chinese_qa:
        template = QA_ZH_PROMETHEUS_TEMPLATE
    else:
        template = PROMETHEUS_ABSOLUTE_PROMPT_TEMPLATE
    user_prompt = template.format(
        instruction=instruction,
        response=response,
        reference_answer=reference_answer,
        rubric=rubric,
    )
    system_prompt = (
        QA_ZH_PROMETHEUS_SYSTEM_PROMPT if chinese_qa else PROMETHEUS_ABS_SYSTEM_PROMPT
    )
    return system_prompt, user_prompt


def format_reference_answers(value: Any, *, alternatives: bool = False) -> str:
    """Format reference answers, optionally marking multiple answers as alternatives.

    Args:
        value: Reference answer value.
        alternatives: Label each of multiple references as an acceptable option.

    Returns:
        Plain reference answer text.
    """
    if isinstance(value, list):
        if alternatives and len(value) > 1:
            return "\n".join(
                f"可选参考答案 {index}：{answer}"
                for index, answer in enumerate(value, start=1)
            )
        return " | ".join(str(item) for item in value)
    return stringify_prompt_value(value)


def stringify_prompt_value(value: Any) -> str:
    """Convert a prompt value to stable text.

    Args:
        value: Value to render.

    Returns:
        String representation for prompt insertion.
    """
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def output_schema_for_prompt(prompt_name: str) -> dict[str, Any]:
    """Return the expected output schema for a prompt.

    Args:
        prompt_name: Prompt identifier.

    Returns:
        JSON-schema-like dictionary for the judge output.
    """
    return get_prompt(prompt_name).output_schema


def build_prompt_metadata() -> dict[str, Any]:
    """Build JSON-serializable prompt metadata.

    Returns:
        Prompt metadata written next to judge input JSONL files.
    """
    return {
        "protocol": JUDGE_PROTOCOL,
        "runtime_settings": RUNTIME_SETTINGS,
        "judge_input_note": (
            "Send only the row's judge_input object to the judge prompt. "
            "Do not send analysis_metadata fields such as experiment labels, "
            "run names, F1 scores, or generation_token_cnt."
        ),
        "prompts": [
            {
                "name": prompt.name,
                "task_type": prompt.task_type,
                "system_prompt": prompt.system_prompt,
                "user_prompt_template": prompt.user_prompt_template,
                "output_schema": prompt.output_schema,
            }
            for prompt in JUDGE_PROMPTS.values()
        ],
    }
