# SPDX-License-Identifier: Apache-2.0
"""Deterministic generation diagnostics for RAG judge outputs."""

# Standard
from __future__ import annotations

from collections import Counter
from typing import Any, Optional
import re


DEFAULT_MAX_TOKEN_THRESHOLD = 2048
VERY_SHORT_WORD_THRESHOLD = 2
PROMPT_CONTINUATION_PATTERNS = (
    r"\bDialogue\s*:",
    r"\bSummary\s*:",
    r"\bQuestion\s*:",
    r"\bAnswer\s*:",
    r"\s#\s#\s",
    r"问题\s*[：:]",
    r"答案\s*[：:]",
)


def compute_rule_diagnostics(
    prediction: str,
    analysis_metadata: Optional[dict[str, Any]] = None,
    max_token_threshold: int = DEFAULT_MAX_TOKEN_THRESHOLD,
) -> dict[str, Any]:
    """Compute deterministic diagnostics for one prediction.

    Args:
        prediction: Model prediction text.
        analysis_metadata: Optional source metadata from ``judge_input.jsonl``.
        max_token_threshold: Generation-token count that indicates max-token
            degeneration when the count is available.

    Returns:
        JSON-serializable diagnostic fields.
    """
    metadata = analysis_metadata or {}
    text = "" if prediction is None else str(prediction)
    stripped = text.strip()
    words = re.findall(r"\S+", stripped)
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    generation_token_cnt = parse_optional_int(metadata.get("generation_token_cnt"))
    metadata_token_threshold = parse_optional_int(metadata.get("generation_max_tokens"))
    effective_max_token_threshold = (
        metadata_token_threshold
        if metadata_token_threshold is not None and metadata_token_threshold > 0
        else max_token_threshold
    )

    ngram_info = repeated_ngram_info(words)
    repeated_line_count = max_repeated_line_count(lines)
    prompt_counts = prompt_continuation_counts(text)
    prompt_continuation_detected = is_prompt_continuation(prompt_counts)
    repeated_ngram_detected = ngram_info["max_repeated_ngram_count"] >= 4
    repeated_line_detected = repeated_line_count >= 3
    max_token_detected = (
        generation_token_cnt is not None
        and generation_token_cnt >= effective_max_token_threshold
    )

    return {
        "empty_output": not stripped,
        "very_short_output": bool(stripped) and len(words) <= VERY_SHORT_WORD_THRESHOLD,
        "char_count": len(text),
        "word_count": len(words),
        "line_count": len(lines),
        "repetition_detected": (
            repeated_ngram_detected
            or repeated_line_detected
            or prompt_continuation_detected
        ),
        "repeated_ngram_detected": repeated_ngram_detected,
        "repeated_ngram": ngram_info["repeated_ngram"],
        "max_repeated_ngram_count": ngram_info["max_repeated_ngram_count"],
        "repeated_line_detected": repeated_line_detected,
        "max_repeated_line_count": repeated_line_count,
        "prompt_continuation_detected": prompt_continuation_detected,
        "prompt_marker_counts": prompt_counts,
        "max_token_detected": max_token_detected,
        "generation_token_cnt": generation_token_cnt,
        "max_token_threshold": effective_max_token_threshold,
    }


def parse_optional_int(value: Any) -> Optional[int]:
    """Parse an optional integer.

    Args:
        value: Value to parse.

    Returns:
        Parsed integer, or ``None``.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def repeated_ngram_info(words: list[str]) -> dict[str, Any]:
    """Find the most repeated normalized n-gram.

    Args:
        words: Prediction words.

    Returns:
        The repeated n-gram and count.
    """
    normalized = [normalize_token(word) for word in words if normalize_token(word)]
    best_ngram = ""
    best_count = 0
    for ngram_size in (3, 4, 5, 6, 8):
        if len(normalized) < ngram_size:
            continue
        ngrams = [
            " ".join(normalized[index : index + ngram_size])
            for index in range(len(normalized) - ngram_size + 1)
        ]
        if not ngrams:
            continue
        ngram, count = Counter(ngrams).most_common(1)[0]
        if count > best_count:
            best_ngram = ngram
            best_count = count
    return {
        "repeated_ngram": best_ngram,
        "max_repeated_ngram_count": best_count,
    }


def normalize_token(token: str) -> str:
    """Normalize a token for repetition checks.

    Args:
        token: Raw token.

    Returns:
        Lowercased alphanumeric token.
    """
    return re.sub(r"(^\W+|\W+$)", "", token.lower())


def max_repeated_line_count(lines: list[str]) -> int:
    """Return the maximum count for any repeated normalized line.

    Args:
        lines: Non-empty prediction lines.

    Returns:
        Maximum repeated-line count.
    """
    normalized = [re.sub(r"\s+", " ", line.lower()).strip() for line in lines]
    normalized = [line for line in normalized if line]
    if not normalized:
        return 0
    return Counter(normalized).most_common(1)[0][1]


def prompt_continuation_counts(text: str) -> dict[str, int]:
    """Count prompt-continuation markers.

    Args:
        text: Prediction text.

    Returns:
        Counts keyed by marker name.
    """
    return {
        "dialogue": len(re.findall(PROMPT_CONTINUATION_PATTERNS[0], text)),
        "summary": len(re.findall(PROMPT_CONTINUATION_PATTERNS[1], text)),
        "question": len(re.findall(PROMPT_CONTINUATION_PATTERNS[2], text)),
        "answer": len(re.findall(PROMPT_CONTINUATION_PATTERNS[3], text)),
        "blend_separator": len(re.findall(PROMPT_CONTINUATION_PATTERNS[4], text)),
        "question_zh": len(re.findall(PROMPT_CONTINUATION_PATTERNS[5], text)),
        "answer_zh": len(re.findall(PROMPT_CONTINUATION_PATTERNS[6], text)),
    }


def is_prompt_continuation(prompt_counts: dict[str, int]) -> bool:
    """Return whether prompt-continuation markers indicate degeneration.

    Args:
        prompt_counts: Counts returned by ``prompt_continuation_counts``.

    Returns:
        Whether prompt continuation was detected.
    """
    dialogue_summary_repeats = (
        prompt_counts["dialogue"] >= 1 and prompt_counts["summary"] >= 2
    )
    qa_repeats = (
        prompt_counts["question"] + prompt_counts["question_zh"] >= 2
        and prompt_counts["answer"] + prompt_counts["answer_zh"] >= 2
    )
    return (
        dialogue_summary_repeats or qa_repeats or prompt_counts["blend_separator"] >= 2
    )
