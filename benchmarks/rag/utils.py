# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from enum import Enum, auto
from logging import Logger
from typing import Any
import asyncio
import collections
import json
import logging
import os
import re
import string
import threading

# Third Party
from rouge_score import rouge_scorer

try:
    import jieba
except ImportError:  # pragma: no cover - exercised via fallback test path.
    jieba = None


def build_format(color):
    reset = "\x1b[0m"
    underline = "\x1b[3m"
    return (
        f"{color}[%(asctime)s] %(levelname)s:{reset} %(message)s "
        + f"{underline}(%(filename)s:%(lineno)d:%(name)s){reset}"
    )


class CustomFormatter(logging.Formatter):
    grey = "\x1b[1m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: build_format(grey),
        logging.INFO: build_format(green),
        logging.WARNING: build_format(yellow),
        logging.ERROR: build_format(red),
        logging.CRITICAL: build_format(bold_red),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def init_logger(name: str, log_level=logging.DEBUG) -> Logger:
    logger = logging.getLogger(name)

    logger.handlers.clear()
    logger.propagate = False

    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    return logger


class AsyncLoopWrapper:
    _loop: asyncio.AbstractEventLoop | None = None
    _thread: threading.Thread | None = None
    _logger = init_logger("AsyncLoopWrapper")

    @classmethod
    def WaitLoop(cls):
        assert cls._loop is not None, "Loop is not started"

        async def wait_for_tasks():
            current_task = asyncio.current_task(cls._loop)
            tasks = [
                task
                for task in asyncio.all_tasks(cls._loop)
                if not task.done() and task is not current_task
            ]
            cls._logger.info(f"Waiting for {len(tasks)} tasks to finish")
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                err_cnt = sum(1 for r in results if isinstance(r, Exception))
                if err_cnt > 0:
                    cls._logger.error(f"{err_cnt} task(s) finished with exceptions")

        # Schedule the wait_for_tasks coroutine to be executed in the loop
        future = asyncio.run_coroutine_threadsafe(wait_for_tasks(), cls._loop)
        try:
            # Wait for wait_for_tasks to complete
            future.result()
        except Exception as e:
            cls._logger.error(f"Error while waiting for tasks: {e}")

    @classmethod
    def StartLoop(cls):
        if cls._loop is not None:
            cls._logger.warning("Loop is already started")
            return

        if cls._loop is None:
            cls._loop = asyncio.new_event_loop()

        def run_loop():
            asyncio.set_event_loop(cls._loop)
            cls._logger.debug("Starting the asyncio loop")
            cls._loop.run_forever()

        cls._thread = threading.Thread(target=run_loop)
        cls._thread.start()

    @classmethod
    def StopLoop(cls):
        assert cls._loop is not None, "Loop is not started"
        assert cls._thread is not None, "Thread is not started"

        def stop_loop():
            cls._logger.debug("Stopping the loop!")
            cls._loop.stop()

        cls._logger.info("Waiting for remaining tasks to finish")
        cls.WaitLoop()

        cls._loop.call_soon_threadsafe(stop_loop)
        cls._thread.join()

    @classmethod
    def GetLoop(cls) -> asyncio.AbstractEventLoop:
        assert cls._loop is not None, "Loop is not started"
        return cls._loop

    @classmethod
    def GetOrStartLoop(cls) -> asyncio.AbstractEventLoop:
        if cls._loop is None:
            cls.StartLoop()
        assert cls._loop is not None, "Loop is not started"
        return cls._loop


class PromptBuildMethodType(Enum):
    QA = auto()
    FEW_SHOT = auto()


class ChatTemplateMode(Enum):
    """Chat framing supported by the RAG prompt builder."""

    NONE = "none"
    QWEN_THINKING = "qwen-thinking"
    QWEN_NON_THINKING = "qwen-non-thinking"


class EvalMethodType(Enum):
    QA = auto()
    QA_ZH = auto()
    ROUGE = auto()
    ROUGE_ZH = auto()


@dataclass(frozen=True)
class EvaluationResult:
    quality: float
    precision: float | None
    recall: float | None
    f1: float | None
    f1_tokenizer: float | None


DEFAULT_SYSTEM_PROMPTS = {
    PromptBuildMethodType.QA: (
        "You will be asked a question after reading several passages. "
        "Please directly answer the question based on the given passages. "
        "Output only the final answer. Do not show analysis or reasoning, "
        "do not repeat the question, and do not generate another question "
        "or answer. The answer should be within 5 words.\nPassages:\n"
    ),
    PromptBuildMethodType.FEW_SHOT: (
        "Summarize the dialogue into 1 to 3 short sentences. "
        "Output only the final summary. Do not show analysis or reasoning, "
        "do not repeat the dialogue, and do not add a title, preamble, or "
        "Markdown. "
        "The following are some examples.\n\n"
    ),
}

DEFAULT_QUERY_PROMPTS = {
    PromptBuildMethodType.QA: (
        "\n\nAnswer the question directly based on the given passages."
        " Output only the final answer. Do not show analysis or reasoning, "
        "do not repeat the question, and do not generate another question "
        "or answer. The answer should be within 5 words.\nQuestion:"
    ),
    PromptBuildMethodType.FEW_SHOT: (
        "\n\nNow summarize the final dialogue below. Output only the summary:\n"
    ),
}

DATASET_PROMPT_OVERRIDES = {
    "multifieldqa_zh": {
        "system_prompt": (
            "请根据以下材料回答问题。只输出最终答案，不要展示推理过程、复述问题或续写其他问答。"
            "答案应简洁，并保留问题要求的全部关键信息。\n材料：\n"
        ),
        "query_prompt": "\n\n请根据材料直接回答下列问题，只输出最终答案。\n问题：",
    },
    "dureader": {
        "system_prompt": (
            "请根据以下材料回答问题。只输出最终答案，不要展示推理过程、复述问题或续写其他问答。"
            "可用完整句子说明关键事实，但不要添加材料中没有的内容。\n材料：\n"
        ),
        "query_prompt": (
            "\n\n请根据材料直接回答下列问题，必要时用完整句子，只输出最终答案。\n问题："
        ),
    },
}

CHINESE_PUNCTUATION = (
    "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～"
    "｟｠｢｣､、〃》「」『』〖〗〔〕〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
)


def load_dataset(dataset_path):
    print("Loading dataset:", dataset_path)
    with open(dataset_path) as f:
        return json.load(f)


def dataset_stem(dataset_path: str) -> str:
    """Return the lowercase dataset stem without its file extension."""
    return os.path.splitext(os.path.basename(dataset_path))[0].lower()


def _dataset_matches(dataset_path: str, prefix: str) -> bool:
    return dataset_stem(dataset_path).startswith(prefix)


def resolve_prompt_texts(
    dataset_path: str,
    prompt_build_method: PromptBuildMethodType,
) -> tuple[str, str]:
    """Resolve dataset-aware system and query prompts."""
    system_prompt = DEFAULT_SYSTEM_PROMPTS[prompt_build_method]
    query_prompt = DEFAULT_QUERY_PROMPTS[prompt_build_method]
    if prompt_build_method != PromptBuildMethodType.QA:
        return system_prompt, query_prompt

    for prefix, override in DATASET_PROMPT_OVERRIDES.items():
        if _dataset_matches(dataset_path, prefix):
            return override["system_prompt"], override["query_prompt"]
    return system_prompt, query_prompt


def resolve_chat_template_mode(mode: str) -> ChatTemplateMode:
    """Resolve a command-line chat-template mode.

    Args:
        mode: One of ``none``, ``qwen-thinking``, or
            ``qwen-non-thinking``.

    Returns:
        The corresponding chat-template mode.

    Raises:
        ValueError: If ``mode`` is unsupported.
    """
    normalized = mode.strip().lower()
    for candidate in ChatTemplateMode:
        if candidate.value == normalized:
            return candidate
    supported = ", ".join(candidate.value for candidate in ChatTemplateMode)
    raise ValueError(f"Unsupported chat template mode {mode!r}; choose {supported}")


def resolve_eval_method(
    dataset_path: str,
    prompt_build_method: PromptBuildMethodType,
) -> EvalMethodType:
    """Pick the evaluation method for the given dataset."""
    if _dataset_matches(dataset_path, "multifieldqa_zh"):
        return EvalMethodType.QA_ZH
    if _dataset_matches(dataset_path, "dureader"):
        return EvalMethodType.ROUGE_ZH
    if prompt_build_method == PromptBuildMethodType.FEW_SHOT:
        return EvalMethodType.ROUGE
    return EvalMethodType.QA


def resolve_generation_stops(
    prompt_build_method: PromptBuildMethodType,
    separator: str,
    dataset_path: str = "",
) -> list[str]:
    """Return task-specific strings that terminate a generated completion.

    Args:
        prompt_build_method: Prompt format used by the benchmark request.
        separator: Delimiter placed between cached prompt segments.

    Returns:
        Stop strings that prevent the model from continuing into another
        synthetic question, dialogue, or cached segment.
    """
    stops = [separator.strip()] if separator.strip() else []
    if prompt_build_method == PromptBuildMethodType.QA:
        stops = ["Question:", ". Answer:", ". 答案：", *stops]
        if resolve_eval_method(dataset_path, prompt_build_method) in (
            EvalMethodType.QA_ZH,
            EvalMethodType.ROUGE_ZH,
        ):
            stops.extend(["\n问题：", "。答案："])
    elif prompt_build_method == PromptBuildMethodType.FEW_SHOT:
        stops.append("Dialogue:")
    return list(dict.fromkeys(stops))


def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"

    return question[0].lower() + question[1:]


def parse_generation(s: object) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)

    lines = [line.strip() for line in s.splitlines() if line.strip()]
    answer_label = r"(?:(?:the\s+)?(?:final\s+)?answer\s*:|答案\s*[：:])"
    while lines and re.fullmatch(answer_label, lines[0], re.I):
        lines.pop(0)
    if not lines:
        return ""
    s = re.sub(
        rf"^{answer_label}\s*",
        "",
        lines[0],
        count=1,
        flags=re.I,
    ).strip()
    if s.startswith("Yes") or s.startswith("yes"):
        s = "Yes"
    elif s.split() and (
        (s.split()[0]).startswith("No") or (s.split()[0]).startswith("no")
    ):
        s = "No"
    return s


def strip_reasoning_trace(text: str | None) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    if "</think>" in text:
        trailing = text.rsplit("</think>", 1)[-1].strip()
        if trailing:
            text = trailing
        else:
            text = text.replace("</think>", "")

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s: str) -> str:
    """Lower text and remove punctuation and whitespace for Chinese tasks."""

    def white_space_fix(text: str) -> str:
        return "".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation + CHINESE_PUNCTUATION)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def _tokenize_zh_text(text: str) -> list[str]:
    """Tokenize Chinese text with jieba when available, else fall back to chars."""
    if jieba is not None:
        tokens = [
            normalize_zh_answer(token) for token in jieba.cut(text, cut_all=False)
        ]
        return [token for token in tokens if token]
    return [char for char in normalize_zh_answer(text) if char]


def build_qa_prompt(example, query_prompt):
    chinese_prompt = query_prompt.rstrip().endswith("问题：")
    q = (
        example["question"]
        if chinese_prompt
        else normalize_question(example["question"])
    )
    doc_prompts = [f"{ctx['title']}\n\n{ctx['text']}\n\n" for ctx in example["ctxs"]]
    answer_label = "答案：" if chinese_prompt else "Answer:"
    q_prompt = f"{query_prompt}{q}\n{answer_label}"
    return doc_prompts, q_prompt


def build_fewshot_prompt(
    example: dict,
    query_prompt: str = "",
) -> tuple[list[str], str]:
    """Build cached demonstrations and the final few-shot target prompt.

    Args:
        example: Dataset example with ``ctxs`` demonstrations and a final
            ``question`` field.
        query_prompt: Instruction inserted immediately before the final target.

    Returns:
        The demonstration strings and final target prompt.
    """
    if query_prompt:
        q = f"{query_prompt}{example['question']}"
    else:
        q = "\n\n" + example["question"]
    doc_prompts = [f"{ctx['text']}" for ctx in example["ctxs"]]
    q_prompt = f"{q}"
    return doc_prompts, q_prompt


_CHAT_TEMPLATE_CONTENT_MARKER = "LMCACHE_RAG_CONTENT_7F3A91D2"


def build_chat_template_text_parts(
    tokenizer,
    system_prompt: str,
    chat_template_mode: ChatTemplateMode,
) -> tuple[str, str]:
    """Render chat-template text before and after cacheable user content.

    The cacheable document text is represented by a unique marker during
    template rendering. Splitting around that marker lets the caller insert
    already-tokenized document segments without retokenizing or changing them.

    Args:
        tokenizer: Hugging Face tokenizer providing ``apply_chat_template``.
        system_prompt: Instruction placed in the chat system message.
        chat_template_mode: Qwen thinking/non-thinking mode, or ``NONE``.

    Returns:
        Text occurring before and after the cacheable user content.

    Raises:
        ValueError: If the rendered template does not contain the marker once.
    """
    if chat_template_mode == ChatTemplateMode.NONE:
        return "", ""
    enable_thinking = chat_template_mode == ChatTemplateMode.QWEN_THINKING
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _CHAT_TEMPLATE_CONTENT_MARKER},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if not isinstance(rendered, str):
        raise ValueError("Chat template must render text when tokenize=False")
    if rendered.count(_CHAT_TEMPLATE_CONTENT_MARKER) != 1:
        raise ValueError("Chat template did not preserve the content marker once")
    prefix_text, suffix_text = rendered.split(_CHAT_TEMPLATE_CONTENT_MARKER, 1)
    return prefix_text, suffix_text


def compute_f1(a_pred, a_gold):
    return compute_qa_metrics(a_pred, a_gold)["f1"]


def compute_tokenizer_f1(a_pred, a_gold, tokenizer):
    a_pred = parse_generation(a_pred)
    gold_toks = _strip_bos_token(tokenizer, tokenizer.encode(normalize_answer(a_gold)))
    pred_toks = _strip_bos_token(tokenizer, tokenizer.encode(normalize_answer(a_pred)))
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def compute_tokenizer_f1_zh(a_pred, a_gold, tokenizer):
    a_pred = parse_generation(a_pred)
    gold_toks = _strip_bos_token(
        tokenizer, tokenizer.encode(normalize_zh_answer(a_gold))
    )
    pred_toks = _strip_bos_token(
        tokenizer, tokenizer.encode(normalize_zh_answer(a_pred))
    )
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def compute_qa_metrics(a_pred, a_gold):
    a_pred = parse_generation(a_pred)
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        agreed = int(gold_toks == pred_toks)
        return {
            "precision": float(agreed),
            "recall": float(agreed),
            "f1": float(agreed),
        }
    if num_same == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_qa_metrics_zh(a_pred, a_gold):
    a_pred = parse_generation(a_pred)
    gold_toks = _tokenize_zh_text(a_gold)
    pred_toks = _tokenize_zh_text(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        agreed = int(gold_toks == pred_toks)
        return {
            "precision": float(agreed),
            "recall": float(agreed),
            "f1": float(agreed),
        }
    if num_same == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_rl(pred, gold):
    if pred is None:
        pred = ""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rougeL = scorer.score(gold, pred)["rougeL"].fmeasure
    return rougeL


def _lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    prev = [0] * (len(right) + 1)
    for left_token in left:
        curr = [0]
        for idx, right_token in enumerate(right, start=1):
            if left_token == right_token:
                curr.append(prev[idx - 1] + 1)
            else:
                curr.append(max(curr[-1], prev[idx]))
        prev = curr
    return prev[-1]


def compute_rl_zh(pred, gold):
    if pred is None:
        pred = ""
    pred_tokens = _tokenize_zh_text(pred)
    gold_tokens = _tokenize_zh_text(gold)
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    lcs = _lcs_length(pred_tokens, gold_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def score_prediction(
    dataset_path: str,
    prompt_build_method: PromptBuildMethodType,
    generated_text: str,
    answers: list[str],
    tokenizer,
) -> EvaluationResult:
    """Score a generated answer using the dataset-appropriate metric."""
    eval_method = resolve_eval_method(dataset_path, prompt_build_method)
    if eval_method == EvalMethodType.QA:
        best_metrics = max(
            [compute_qa_metrics(generated_text, answer) for answer in answers],
            key=lambda metrics: metrics["f1"],
        )
        best_tokenizer_f1 = max(
            [
                compute_tokenizer_f1(generated_text, answer, tokenizer)
                for answer in answers
            ]
        )
        return EvaluationResult(
            quality=best_metrics["f1"],
            precision=best_metrics["precision"],
            recall=best_metrics["recall"],
            f1=best_metrics["f1"],
            f1_tokenizer=best_tokenizer_f1,
        )
    if eval_method == EvalMethodType.QA_ZH:
        best_metrics = max(
            [compute_qa_metrics_zh(generated_text, answer) for answer in answers],
            key=lambda metrics: metrics["f1"],
        )
        best_tokenizer_f1 = max(
            [
                compute_tokenizer_f1_zh(generated_text, answer, tokenizer)
                for answer in answers
            ]
        )
        return EvaluationResult(
            quality=best_metrics["f1"],
            precision=best_metrics["precision"],
            recall=best_metrics["recall"],
            f1=best_metrics["f1"],
            f1_tokenizer=best_tokenizer_f1,
        )
    if eval_method == EvalMethodType.ROUGE:
        return EvaluationResult(
            quality=max([compute_rl(generated_text, answer) for answer in answers]),
            precision=None,
            recall=None,
            f1=None,
            f1_tokenizer=None,
        )
    if eval_method == EvalMethodType.ROUGE_ZH:
        return EvaluationResult(
            quality=max([compute_rl_zh(generated_text, answer) for answer in answers]),
            precision=None,
            recall=None,
            f1=None,
            f1_tokenizer=None,
        )
    raise ValueError(f"Unsupported eval method: {eval_method}")


def _strip_bos_token(tokenizer, token_ids: list[int]) -> list[int]:
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if token_ids and bos_token_id is not None and token_ids[0] == bos_token_id:
        return token_ids[1:]
    return token_ids


def tokenize_segment_for_cache(
    tokenizer,
    text: str,
    add_special_tokens: bool = False,
) -> list[int]:
    token_ids = tokenizer(text, add_special_tokens=add_special_tokens).input_ids
    if add_special_tokens:
        return token_ids
    return _strip_bos_token(tokenizer, token_ids)


def build_rag_prompt(
    system_prompt: str,
    example: dict[str, Any],
    query_prompt: str,
    separator: str,
    prompt_build_method: PromptBuildMethodType,
    tokenizer: Any | None = None,
    chat_template_mode: ChatTemplateMode = ChatTemplateMode.NONE,
) -> tuple[str, list[str]]:
    """Build the readable RAG prompt with optional Qwen chat framing.

    Args:
        system_prompt: Task instruction for the model.
        example: Dataset example containing contexts, question, and answers.
        query_prompt: Instruction placed before the final question/dialogue.
        separator: Delimiter placed between cacheable prompt segments.
        prompt_build_method: QA or few-shot prompt layout.
        tokenizer: Tokenizer used to render a chat template when enabled.
        chat_template_mode: Chat framing and thinking behavior.

    Returns:
        The rendered prompt and its cacheable document strings.

    Raises:
        ValueError: If the prompt method is invalid or chat framing lacks a
            tokenizer.
    """
    doc_prompts = None
    q_prompt = None
    if prompt_build_method == PromptBuildMethodType.FEW_SHOT:
        doc_prompts, q_prompt = build_fewshot_prompt(example, query_prompt)
    elif prompt_build_method == PromptBuildMethodType.QA:
        doc_prompts, q_prompt = build_qa_prompt(example, query_prompt)
    else:
        raise ValueError(f"Invalid prompt build method {prompt_build_method}")
    if chat_template_mode == ChatTemplateMode.NONE:
        final_prompt = separator.join([system_prompt] + doc_prompts + [q_prompt])
    else:
        if tokenizer is None:
            raise ValueError("tokenizer is required when using a chat template")
        prefix_text, suffix_text = build_chat_template_text_parts(
            tokenizer,
            system_prompt,
            chat_template_mode,
        )
        user_content = separator.join(doc_prompts + [q_prompt])
        if separator:
            user_content = separator + user_content
        final_prompt = prefix_text + user_content + suffix_text
    return final_prompt, doc_prompts


def build_rag_prompt_token_ids(
    tokenizer: Any,
    system_prompt: str,
    example: dict[str, Any],
    query_prompt: str,
    separator: str,
    prompt_build_method: PromptBuildMethodType,
    chat_template_mode: ChatTemplateMode = ChatTemplateMode.NONE,
) -> tuple[list[int], list[str]]:
    """Build RAG token IDs while preserving cached document token sequences.

    Chat prefix/suffix tokens are assembled around independently tokenized
    document segments. This keeps every document identical to the tokens used
    during KV precompute while still giving Qwen valid chat boundaries.

    Args:
        tokenizer: Tokenizer used for segment and chat-template encoding.
        system_prompt: Task instruction placed in the system turn.
        example: Dataset example containing contexts, question, and answers.
        query_prompt: Instruction placed before the final question/dialogue.
        separator: Delimiter placed between cacheable prompt segments.
        prompt_build_method: QA or few-shot prompt layout.
        chat_template_mode: Chat framing and thinking behavior.

    Returns:
        Final request token IDs and the cacheable document strings.

    Raises:
        ValueError: If the prompt method or chat-template rendering is invalid.
    """
    doc_prompts = None
    q_prompt = None
    if prompt_build_method == PromptBuildMethodType.FEW_SHOT:
        doc_prompts, q_prompt = build_fewshot_prompt(example, query_prompt)
    elif prompt_build_method == PromptBuildMethodType.QA:
        doc_prompts, q_prompt = build_qa_prompt(example, query_prompt)
    else:
        raise ValueError(f"Invalid prompt build method {prompt_build_method}")

    if chat_template_mode == ChatTemplateMode.NONE:
        prefix_token_ids = tokenize_segment_for_cache(
            tokenizer,
            system_prompt,
            add_special_tokens=True,
        )
        suffix_token_ids: list[int] = []
    else:
        prefix_text, suffix_text = build_chat_template_text_parts(
            tokenizer,
            system_prompt,
            chat_template_mode,
        )
        prefix_token_ids = tokenize_segment_for_cache(
            tokenizer,
            prefix_text,
            add_special_tokens=False,
        )
        suffix_token_ids = tokenize_segment_for_cache(
            tokenizer,
            suffix_text,
            add_special_tokens=False,
        )
    sep_token_ids = []
    if separator:
        sep_token_ids = tokenize_segment_for_cache(
            tokenizer,
            separator,
            add_special_tokens=False,
        )
    doc_token_ids = [
        tokenize_segment_for_cache(
            tokenizer,
            doc_prompt,
            add_special_tokens=False,
        )
        for doc_prompt in doc_prompts
    ]
    query_token_ids = tokenize_segment_for_cache(
        tokenizer,
        q_prompt,
        add_special_tokens=False,
    )

    prompt_token_ids = list(prefix_token_ids)
    if chat_template_mode != ChatTemplateMode.NONE and sep_token_ids:
        prompt_token_ids.extend(sep_token_ids)
    for doc_idx, doc_token_id in enumerate(doc_token_ids):
        if sep_token_ids and (
            chat_template_mode == ChatTemplateMode.NONE or doc_idx > 0
        ):
            prompt_token_ids.extend(sep_token_ids)
        prompt_token_ids.extend(doc_token_id)
    if sep_token_ids:
        prompt_token_ids.extend(sep_token_ids)
    prompt_token_ids.extend(query_token_ids)
    prompt_token_ids.extend(suffix_token_ids)
    return prompt_token_ids, doc_prompts
