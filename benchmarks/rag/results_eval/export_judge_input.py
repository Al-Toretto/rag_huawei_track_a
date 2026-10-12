# SPDX-License-Identifier: Apache-2.0
"""Export xformers RAG results as one-prediction-per-row judge JSONL.

The exporter consumes a canonical answer-table manifest and the original result
CSVs. It keeps generated text raw, including newlines, so a later LLM-judge
runner can evaluate the same output that the model produced.
"""

# Standard
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import argparse
import csv
import json

# First Party
from judge_prompts import QA_ZH_JUDGE_PROMPT_NAME, build_prompt_metadata


QA_DATASET_PREFIXES = (
    "2wikimqa",
    "dureader",
    "multifieldqa",
    "musique",
    "wikimqa",
)
SUMMARIZATION_DATASET_PREFIXES = ("samsum",)

JUDGE_INPUT_SCHEMA_VERSION = "reference_guided_v2"
QA_JUDGE_PROMPT_NAME = "rag_qa_semantic_correctness_v2"
SUMMARIZATION_JUDGE_PROMPT_NAME = "rag_samsum_semantic_quality_v2"


def load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    """Load a JSON-list evaluation dataset.

    Args:
        dataset_path: Dataset JSON path.

    Returns:
        Dataset examples in request order.

    Raises:
        ValueError: If the JSON root is not a list.
    """
    with dataset_path.open(encoding="utf-8") as dataset_file:
        data = json.load(dataset_file)
    if not isinstance(data, list):
        raise ValueError(f"Expected list dataset in {dataset_path}")
    return data


def flatten_answers(raw_answers: Any) -> list[str]:
    """Flatten nested reference-answer values into strings.

    Args:
        raw_answers: Scalar or nested answer values.

    Returns:
        Flat answer strings in input order.
    """
    answers: list[str] = []

    def collect(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            answers.append(value)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
            return
        answers.append(str(value))

    collect(raw_answers)
    return answers


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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Export RAG result CSVs as JSONL input for LLM judging."
    )
    parser.add_argument(
        "--rag-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to benchmarks/rag.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Manifest CSV from build_answer_tables.py. Defaults to "
            "benchmarks/rag/results_eval/outputs/manifest.csv."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL path. Defaults to "
            "benchmarks/rag/results_eval/outputs/judge_input.jsonl."
        ),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset stem to include. May be provided multiple times.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=None,
        help="Experiment display label to include. May be provided multiple times.",
    )
    parser.add_argument(
        "--skip-empty-predictions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Skip rows whose generated_text is empty. Empty successful model "
            "outputs are included by default so they receive a judge score."
        ),
    )
    parser.add_argument(
        "--write-prompt-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write judge prompt metadata next to the JSONL output.",
    )
    return parser.parse_args()


def dataset_task_type(dataset_stem: str) -> str:
    """Return the judge task type for a dataset.

    Args:
        dataset_stem: Dataset file stem, such as ``musique_longbench``.

    Returns:
        ``"qa"`` or ``"summarization"``.

    Raises:
        ValueError: If the dataset family is not supported.
    """
    lowered = dataset_stem.lower()
    if lowered.startswith(SUMMARIZATION_DATASET_PREFIXES):
        return "summarization"
    if lowered.startswith(QA_DATASET_PREFIXES):
        return "qa"
    raise ValueError(f"Unsupported dataset for judge export: {dataset_stem}")


def prompt_name_for_task(task_type: str) -> str:
    """Return the prompt name for a judge task type.

    Args:
        task_type: Judge task type.

    Returns:
        Prompt identifier used in exported rows.

    Raises:
        ValueError: If ``task_type`` is not supported.
    """
    if task_type == "qa":
        return QA_JUDGE_PROMPT_NAME
    if task_type == "summarization":
        return SUMMARIZATION_JUDGE_PROMPT_NAME
    raise ValueError(f"Unsupported task type: {task_type}")


def load_manifest(
    manifest_path: Path,
    selected_datasets: Optional[set[str]],
    selected_experiments: Optional[set[str]],
) -> list[dict[str, str]]:
    """Load and filter the answer-table manifest.

    Args:
        manifest_path: Path to the manifest CSV.
        selected_datasets: Optional dataset-stem allowlist.
        selected_experiments: Optional experiment-label allowlist.

    Returns:
        Manifest rows sorted by dataset and fixed experiment order.
    """
    with manifest_path.open(newline="") as manifest_file:
        rows = list(csv.DictReader(manifest_file))

    if selected_datasets is not None:
        rows = [row for row in rows if row["dataset"] in selected_datasets]
    if selected_experiments is not None:
        rows = [row for row in rows if row["label"] in selected_experiments]

    experiment_rank: dict[str, int] = {}
    for row in rows:
        experiment_rank.setdefault(row["label"], len(experiment_rank))
    return sorted(
        rows,
        key=lambda row: (
            row["dataset"],
            parse_optional_int(row.get("order", ""))
            if parse_optional_int(row.get("order", "")) is not None
            else experiment_rank.get(row["label"], len(experiment_rank)),
            row["label"],
        ),
    )


def dataset_examples_by_id(dataset_path: Path) -> dict[int, dict[str, Any]]:
    """Load dataset examples keyed by integer request id.

    Args:
        dataset_path: Path to a JSON dataset file.

    Returns:
        Mapping from request id to dataset example.
    """
    return {
        request_id: example
        for request_id, example in enumerate(load_dataset(dataset_path))
    }


def make_judge_row(
    manifest_row: dict[str, str],
    result_row: dict[str, str],
    example: dict[str, Any],
) -> dict[str, Any]:
    """Build one JSON-serializable judge input row.

    Args:
        manifest_row: One row from ``manifest.csv``.
        result_row: One row from an experiment result CSV.
        example: Dataset example for the same request id.

    Returns:
        JSON-serializable row for the judge input file.
    """
    dataset = manifest_row["dataset"]
    task_type = dataset_task_type(dataset)
    request_id = int(result_row["request_id"])
    prediction = result_row.get("generated_text", "")
    gold_answers = flatten_answers(example.get("answers", []))
    return {
        "judge_input_schema_version": JUDGE_INPUT_SCHEMA_VERSION,
        "dataset": dataset,
        "task_type": task_type,
        "request_id": request_id,
        "prompt_name": (
            QA_ZH_JUDGE_PROMPT_NAME
            if dataset.lower().startswith(("multifieldqa_zh", "dureader"))
            else prompt_name_for_task(task_type)
        ),
        "judge_input": make_task_judge_input(
            task_type,
            str(example.get("question", "")),
            gold_answers,
            prediction,
        ),
        "analysis_metadata": {
            "experiment": manifest_row["label"],
            "run_name": manifest_row["run_name"],
            "job_id": int(manifest_row["job_id"]),
            "result_csv": manifest_row["result_csv"],
            "success": result_row.get("success", ""),
            "error_message": result_row.get("error_message", ""),
            "generation_token_cnt": parse_optional_int(
                result_row.get("generation_token_cnt", "")
            ),
            "generation_max_tokens": parse_optional_int(
                manifest_row.get("generation_max_tokens", "")
            ),
            "quality": parse_optional_float(result_row.get("quality", "")),
            "f1_word": parse_optional_float(result_row.get("f1_word", "")),
            "f1_tokenizer": parse_optional_float(result_row.get("f1_tokenizer", "")),
        },
    }


def make_task_judge_input(
    task_type: str,
    question: str,
    gold_answers: list[str],
    prediction: str,
) -> dict[str, Any]:
    """Build the blinded payload that should be sent to the LLM judge.

    Args:
        task_type: Judge task type.
        question: Dataset question. For SAMSum this is the dialogue prompt and
            is intentionally not included in the primary judge input.
        gold_answers: Reference answers or reference summaries.
        prediction: Model-generated text to evaluate.

    Returns:
        Task-specific judge input with no experiment or run identifiers.

    Raises:
        ValueError: If ``task_type`` is not supported.
    """
    if task_type == "qa":
        return {
            "question": question,
            "gold_answers": gold_answers,
            "prediction": prediction,
        }
    if task_type == "summarization":
        return {
            "reference_summary": "\n".join(gold_answers),
            "prediction": prediction,
        }
    raise ValueError(f"Unsupported task type: {task_type}")


def parse_optional_int(value: str) -> Optional[int]:
    """Parse an optional integer field.

    Args:
        value: String value from a CSV cell.

    Returns:
        Parsed integer, or ``None`` for empty/unparseable values.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_optional_float(value: str) -> Optional[float]:
    """Parse an optional floating-point field.

    Args:
        value: String value from a CSV cell.

    Returns:
        Parsed float, or ``None`` for empty/unparseable values.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_prompt_metadata(output_path: Path) -> None:
    """Write judge prompt metadata next to the JSONL output.

    Args:
        output_path: Judge input JSONL path.
    """
    prompt_path = output_path.with_name(f"{output_path.stem}_prompts.json")
    metadata = build_prompt_metadata()
    prompt_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def export_judge_input(
    manifest_rows: list[dict[str, str]],
    output_path: Path,
    skip_empty_predictions: bool,
) -> int:
    """Export result CSV rows to judge-input JSONL.

    Args:
        manifest_rows: Filtered manifest rows to export.
        output_path: Destination JSONL path.
        skip_empty_predictions: Whether to omit empty generated text.

    Returns:
        Number of rows written.
    """
    dataset_cache: dict[Path, dict[int, dict[str, Any]]] = {}
    rows_written = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for manifest_row in manifest_rows:
            dataset_path = Path(manifest_row["dataset_path"])
            if dataset_path not in dataset_cache:
                dataset_cache[dataset_path] = dataset_examples_by_id(dataset_path)
            examples_by_id = dataset_cache[dataset_path]

            with Path(manifest_row["result_csv"]).open(newline="") as result_file:
                reader = csv.DictReader(result_file)
                for result_row in reader:
                    prediction = result_row.get("generated_text", "")
                    if skip_empty_predictions and not prediction.strip():
                        continue
                    request_id = int(result_row["request_id"])
                    example = examples_by_id.get(request_id, {})
                    judge_row = make_judge_row(manifest_row, result_row, example)
                    output_file.write(json.dumps(judge_row, ensure_ascii=False) + "\n")
                    rows_written += 1

    return rows_written


def main() -> None:
    """Run the judge-input exporter."""
    args = parse_args()
    rag_dir = args.rag_dir.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else rag_dir / "results_eval" / "outputs" / "manifest.csv"
    )
    output_path = (
        args.output.resolve()
        if args.output is not None
        else rag_dir / "results_eval" / "outputs" / "judge_input.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_datasets = set(args.dataset) if args.dataset is not None else None
    selected_experiments = set(args.experiment) if args.experiment is not None else None
    manifest_rows = load_manifest(
        manifest_path,
        selected_datasets,
        selected_experiments,
    )
    rows_written = export_judge_input(
        manifest_rows,
        output_path,
        args.skip_empty_predictions,
    )
    if args.write_prompt_files:
        write_prompt_metadata(output_path)

    print(f"Wrote {rows_written} judge rows to {output_path}")
    print(
        f"Prompt metadata: {output_path.with_name(f'{output_path.stem}_prompts.json')}"
    )


if __name__ == "__main__":
    main()
