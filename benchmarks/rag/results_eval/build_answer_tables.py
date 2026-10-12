# SPDX-License-Identifier: Apache-2.0
"""Build fixed-order answer tables for inspecting xformers RAG runs.

The script scans ``benchmarks/rag/results`` and ``benchmarks/rag/logs`` for
completed xformers runs, matches each run to the dataset recorded in its log,
and writes one wide CSV per dataset. Rows are request IDs. Columns are the gold
answer followed by experiments in the fixed order used for manual inspection
and later LLM-judge prompts.
"""

# Standard
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import argparse
import ast
import csv
import html
import json
import re


EXPERIMENT_ORDER = [
    "Full Recompute",
    "Blend (16%) layer 0",
    "Blend (16%) layer 1",
    "Blend (16%) layer 1 margin tokens 1",
    "Blend (16%) layer 1 margin tokens 2",
    "Blend (16%) layer 1 margin tokens 3",
    "Blend (16%) weighted 50-50",
    "Blend (16%) weighted 30-30",
    "Blend (16%) weighted 20-20",
    "Blend (16%) weighted 10-10",
    "Blend (16%) weighted K0-V20",
    "Blend (16%) adaptive weights K0-20, V0-20",
    "Blend (16%) adaptive weights K0-15, V0-40",
    "SELECTOR K+V",
    "SELECTOR V",
    "Blend (20%) layer 1",
    "Blend (24%) layer 1",
    "Blend (32%) layer 1",
    "Blend (100%) layer 1",
    "neighbor top25 both cap024",
    "neighbor top50 both cap024",
    "neighbor top50 left cap024",
    "neighbor top50 right cap032",
    "neighbor top25 both cap032",
    "neighbor top50 both cap032",
    "neighbor all cap048",
    "close gap cap24",
    "close gap2 cap24",
    "close gap2 cap32",
]


@dataclass(frozen=True)
class ExperimentRule:
    label: str
    include: tuple[str, ...]
    exclude: tuple[str, ...] = ()


@dataclass
class RunRecord:
    run_name: str
    result_csv: Path
    dataset_stem: str
    dataset_path: Path
    label: str
    job_id: int
    row_count: int


EXPERIMENT_RULES = [
    ExperimentRule("Full Recompute", ("full_recompute",)),
    ExperimentRule(
        "Blend (16%) layer 0",
        ("blend16_alignfix",),
        ("margin", "boundary", "plain", "yaml_"),
    ),
    ExperimentRule(
        "Blend (16%) layer 1",
        ("lmcache_blend",),
        (
            "yaml_",
            "boundary",
            "weighted",
            "adaptive",
            "selector",
            "plain",
            "neighbor",
            "span",
        ),
    ),
    ExperimentRule(
        "Blend (16%) layer 1 margin tokens 1",
        ("boundary_1",),
        ("weighted",),
    ),
    ExperimentRule("Blend (16%) layer 1 margin tokens 2", ("boundary_2",)),
    ExperimentRule("Blend (16%) layer 1 margin tokens 3", ("boundary_3",)),
    ExperimentRule("Blend (16%) weighted 50-50", ("weighted_50_50",), ("boundary",)),
    ExperimentRule("Blend (16%) weighted 30-30", ("weighted_30_30",)),
    ExperimentRule("Blend (16%) weighted 20-20", ("weighted_20_20",)),
    ExperimentRule("Blend (16%) weighted 10-10", ("weighted_10_10",)),
    ExperimentRule("Blend (16%) weighted K0-V20", ("weighted_k0_v20",)),
    ExperimentRule(
        "Blend (16%) adaptive weights K0-20, V0-20",
        ("adaptive_weighted",),
        ("k015_v040",),
    ),
    ExperimentRule(
        "Blend (16%) adaptive weights K0-15, V0-40",
        ("adaptive_weighted_k015_v040",),
    ),
    ExperimentRule("SELECTOR K+V", ("selector_diff_kv_sum",)),
    ExperimentRule("SELECTOR V", ("selector_diff_v",)),
    ExperimentRule("Blend (20%) layer 1", ("plain_ratio_020",)),
    ExperimentRule("Blend (24%) layer 1", ("plain_ratio_024",)),
    ExperimentRule("Blend (32%) layer 1", ("plain_ratio_032",)),
    ExperimentRule("Blend (100%) layer 1", ("plain_ratio_100",)),
    ExperimentRule("neighbor top25 both cap024", ("neighbor_top25_both_cap024",)),
    ExperimentRule("neighbor top50 both cap024", ("neighbor_top50_both_cap024",)),
    ExperimentRule("neighbor top50 left cap024", ("neighbor_top50_left_cap024",)),
    ExperimentRule("neighbor top50 right cap032", ("neighbor_top50_right_cap024",)),
    ExperimentRule("neighbor top25 both cap032", ("neighbor_top25_both_cap032",)),
    ExperimentRule("neighbor top50 both cap032", ("neighbor_top50_both_cap032",)),
    ExperimentRule("neighbor all cap048", ("neighbor_all_both_cap048",)),
    ExperimentRule("close gap cap24", ("span_close_gap1_cap024",)),
    ExperimentRule("close gap2 cap24", ("span_close_gap2_cap024",)),
    ExperimentRule("close gap2 cap32", ("span_close_gap2_cap032",)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create fixed-order answer tables from xformers RAG results."
    )
    parser.add_argument(
        "--rag-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to benchmarks/rag.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to benchmarks/rag/results_eval/outputs.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help=(
            "Dataset stem to include, e.g. 2wikimqa_longbench. "
            "May be provided multiple times."
        ),
    )
    parser.add_argument(
        "--include-question",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include the dataset question column in CSV, TSV, and HTML outputs. "
            "Use --no-include-question for compact answer-only files."
        ),
    )
    parser.add_argument(
        "--preserve-newlines",
        action="store_true",
        help=(
            "Keep raw newlines inside answer cells. By default newlines are escaped "
            "as literal \\n so each request stays on one physical CSV line."
        ),
    )
    parser.add_argument(
        "--prefer-gqa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer xformers_gqa runs when multiple runs match a label.",
    )
    return parser.parse_args()


def normalize_remote_dataset_path(raw_path: str, rag_dir: Path) -> Path:
    dataset_name = Path(raw_path.strip()).name
    return rag_dir / "inputs" / dataset_name


def extract_dataset_path(run_name: str, rag_dir: Path) -> Optional[Path]:
    candidate_logs = [
        rag_dir / "logs" / run_name / "benchmark.log",
        rag_dir / "logs" / f"lmcache_rag_{extract_job_id(run_name)}.out",
        rag_dir / "logs" / f"vllm_rag_{extract_job_id(run_name)}.out",
    ]
    for log_path in candidate_logs:
        if not log_path.exists():
            continue
        dataset_path = extract_dataset_path_from_log(log_path, rag_dir)
        if dataset_path is not None:
            return dataset_path
    return infer_dataset_path_from_run_name(run_name, rag_dir)


def extract_dataset_path_from_log(log_path: Path, rag_dir: Path) -> Optional[Path]:
    dataset_path: Optional[Path] = None
    with log_path.open(errors="replace") as log_file:
        for line in log_file:
            if line.startswith("dataset_path="):
                dataset_path = normalize_remote_dataset_path(
                    line.split("=", maxsplit=1)[1], rag_dir
                )
            elif "Loading dataset:" in line:
                raw_path = line.rsplit("Loading dataset:", maxsplit=1)[1].strip()
                dataset_path = normalize_remote_dataset_path(raw_path, rag_dir)
    return dataset_path


def infer_dataset_path_from_run_name(run_name: str, rag_dir: Path) -> Optional[Path]:
    inputs_dir = rag_dir / "inputs"
    for dataset_path in sorted(inputs_dir.glob("*.json")):
        stem = dataset_path.stem
        if stem in run_name:
            return dataset_path
    if "2wikimqa" in run_name:
        return inputs_dir / "2wikimqa_longbench.json"
    if "wikimqa_s" in run_name:
        return inputs_dir / "wikimqa_s.json"
    if "musique_s" in run_name:
        return inputs_dir / "musique_s.json"
    if "musique" in run_name:
        return inputs_dir / "musique_longbench.json"
    return None


def extract_job_id(run_name: str) -> int:
    match = re.search(r"_(\d+)$", run_name)
    return int(match.group(1)) if match else -1


def classify_run(run_name: str) -> Optional[str]:
    lowered = run_name.lower()
    for rule in EXPERIMENT_RULES:
        if all(item in lowered for item in rule.include) and not any(
            item in lowered for item in rule.exclude
        ):
            return rule.label
    return None


def find_result_csv(run_dir: Path) -> Optional[Path]:
    preferred = run_dir / f"{run_dir.name}.csv"
    if preferred.exists():
        return preferred
    csv_paths = sorted(
        path for path in run_dir.glob("*.csv") if not path.name.endswith(".debug.csv")
    )
    return csv_paths[0] if csv_paths else None


def load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    with dataset_path.open() as dataset_file:
        data = json.load(dataset_file)
    if not isinstance(data, list):
        raise ValueError(f"Expected list dataset in {dataset_path}")
    return data


def flatten_answers(raw_answers: Any) -> list[str]:
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


def gold_answer_for_row(example: dict[str, Any]) -> str:
    return " | ".join(flatten_answers(example.get("answers", [])))


def count_result_rows(result_csv: Path) -> int:
    with result_csv.open(newline="") as result_file:
        reader = csv.DictReader(result_file)
        return sum(1 for _ in reader)


def discover_runs(rag_dir: Path, selected_datasets: Optional[set[str]]) -> list[RunRecord]:
    records: list[RunRecord] = []
    skipped_missing_csv: list[str] = []
    skipped_unknown_label: list[str] = []
    skipped_unknown_dataset: list[str] = []

    for run_dir in sorted((rag_dir / "results").glob("xformers*")):
        if not run_dir.is_dir():
            continue
        result_csv = find_result_csv(run_dir)
        if result_csv is None:
            skipped_missing_csv.append(run_dir.name)
            continue

        label = classify_run(run_dir.name)
        if label is None:
            skipped_unknown_label.append(run_dir.name)
            continue

        dataset_path = extract_dataset_path(run_dir.name, rag_dir)
        if dataset_path is None or not dataset_path.exists():
            skipped_unknown_dataset.append(run_dir.name)
            continue

        dataset_stem = dataset_path.stem
        if selected_datasets is not None and dataset_stem not in selected_datasets:
            continue

        records.append(
            RunRecord(
                run_name=run_dir.name,
                result_csv=result_csv,
                dataset_stem=dataset_stem,
                dataset_path=dataset_path,
                label=label,
                job_id=extract_job_id(run_dir.name),
                row_count=count_result_rows(result_csv),
            )
        )

    setattr(discover_runs, "skipped_missing_csv", skipped_missing_csv)
    setattr(discover_runs, "skipped_unknown_label", skipped_unknown_label)
    setattr(discover_runs, "skipped_unknown_dataset", skipped_unknown_dataset)
    return records


def choose_record(
    records: list[RunRecord],
    prefer_gqa: bool,
) -> tuple[RunRecord, list[RunRecord]]:
    def sort_key(record: RunRecord) -> tuple[int, int]:
        gqa_score = 1 if prefer_gqa and record.run_name.startswith("xformers_gqa_") else 0
        return (gqa_score, record.job_id)

    sorted_records = sorted(records, key=sort_key, reverse=True)
    return sorted_records[0], sorted_records[1:]


def make_cell_text(value: Any, preserve_newlines: bool) -> str:
    text = "" if value is None else str(value)
    if preserve_newlines:
        return text
    return text.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")


def read_generated_answers(
    result_csv: Path,
    preserve_newlines: bool,
) -> dict[int, str]:
    answers: dict[int, str] = {}
    with result_csv.open(newline="") as result_file:
        reader = csv.DictReader(result_file)
        required = {"request_id", "generated_text"}
        missing_columns = required.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"{result_csv} is missing columns {sorted(missing_columns)}")
        for row in reader:
            request_id = int(row["request_id"])
            answers[request_id] = make_cell_text(
                row.get("generated_text", ""),
                preserve_newlines,
            )
    return answers


def write_dataset_table(
    dataset_stem: str,
    dataset_path: Path,
    records_by_label: dict[str, RunRecord],
    output_dir: Path,
    include_question: bool,
    preserve_newlines: bool,
) -> None:
    dataset = load_dataset(dataset_path)
    answers_by_label = {
        label: read_generated_answers(record.result_csv, preserve_newlines)
        for label, record in records_by_label.items()
    }

    fieldnames = ["dataset", "request_id", "gold_answer"]
    if include_question:
        fieldnames.append("question")
    fieldnames.extend(EXPERIMENT_ORDER)

    max_request_id = max(
        [len(dataset) - 1]
        + [
            max(answer_map.keys())
            for answer_map in answers_by_label.values()
            if answer_map
        ]
    )

    rows = []
    for request_id in range(max_request_id + 1):
        example = dataset[request_id] if request_id < len(dataset) else {}
        row = {
            "dataset": dataset_stem,
            "request_id": request_id,
            "gold_answer": make_cell_text(
                gold_answer_for_row(example),
                preserve_newlines,
            ),
        }
        if include_question:
            row["question"] = make_cell_text(
                example.get("question", ""),
                preserve_newlines,
            )
        for label in EXPERIMENT_ORDER:
            row[label] = answers_by_label.get(label, {}).get(request_id, "")
        rows.append(row)

    write_delimited_table(
        output_dir / f"{dataset_stem}_answers_wide.csv",
        fieldnames,
        rows,
        delimiter=",",
    )
    write_delimited_table(
        output_dir / f"{dataset_stem}_answers_wide.tsv",
        fieldnames,
        rows,
        delimiter="\t",
    )
    write_inspection_html(
        output_dir / f"{dataset_stem}_answers_inspect.html",
        dataset_stem,
        rows,
        include_question,
    )


def write_delimited_table(
    output_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
    delimiter: str,
) -> None:
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            delimiter=delimiter,
        )
        writer.writeheader()
        writer.writerows(rows)


def normalize_for_hint(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def answer_class(gold_answer: str, generated_answer: str) -> str:
    if not generated_answer:
        return "answer empty"
    normalized_gold = normalize_for_hint(gold_answer)
    normalized_generated = normalize_for_hint(generated_answer)
    if normalized_gold and normalized_gold == normalized_generated:
        return "answer exact"
    if normalized_gold and normalized_gold in normalized_generated:
        return "answer contains"
    return "answer"


def write_inspection_html(
    output_path: Path,
    dataset_stem: str,
    rows: list[dict[str, Any]],
    include_question: bool,
) -> None:
    cards = []
    nav_items = []
    for row in rows:
        request_id = str(row["request_id"])
        gold_answer = str(row["gold_answer"])
        nav_items.append(
            '<a class="nav-item" href="#request-{id}">#{id}</a>'.format(
                id=html.escape(request_id)
            )
        )
        question_html = ""
        if include_question:
            question_html = (
                '<div class="question"><span>Question</span>'
                f"<p>{html.escape(str(row.get('question', '')))}</p></div>"
            )

        answer_rows = []
        for label in EXPERIMENT_ORDER:
            answer = str(row.get(label, ""))
            answer_rows.append(
                '<tr class="{klass}"><th>{label}</th><td>{answer}</td></tr>'.format(
                    klass=answer_class(gold_answer, answer),
                    label=html.escape(label),
                    answer=html.escape(answer),
                )
            )

        cards.append(
            """
            <section class="card" id="request-{request_id}">
              <div class="card-header">
                <h2>Request {request_id}</h2>
                <div class="gold"><span>Gold</span><strong>{gold}</strong></div>
              </div>
              {question}
              <table>
                <tbody>
                  {answer_rows}
                </tbody>
              </table>
            </section>
            """.format(
                request_id=html.escape(request_id),
                gold=html.escape(gold_answer),
                question=question_html,
                answer_rows="\n".join(answer_rows),
            )
        )

    html_text = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{dataset} answer inspection</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #171a1f;
      --muted: #5f6875;
      --line: #d9dee7;
      --soft: #eef1f6;
      --exact: #e6f5ec;
      --contains: #fff6db;
      --empty: #f3f4f6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      border-bottom: 1px solid var(--line);
      background: rgba(246, 247, 249, 0.96);
      padding: 14px 20px;
      backdrop-filter: blur(8px);
    }}
    header h1 {{
      margin: 0 0 10px;
      font-size: 20px;
      font-weight: 700;
    }}
    .toolbar {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}
    input {{
      width: min(680px, 100%);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 11px;
      font: inherit;
      background: white;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 150px minmax(0, 1fr);
      gap: 16px;
      padding: 16px 20px 40px;
    }}
    nav {{
      position: sticky;
      top: 91px;
      align-self: start;
      max-height: calc(100vh - 110px);
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 8px;
    }}
    .nav-item {{
      display: inline-block;
      width: 52px;
      margin: 3px;
      padding: 5px 0;
      border-radius: 5px;
      color: var(--text);
      text-align: center;
      text-decoration: none;
      font-size: 13px;
      background: var(--soft);
    }}
    main {{
      min-width: 0;
    }}
    .card {{
      margin-bottom: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }}
    .card-header {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 18px;
      border-bottom: 1px solid var(--line);
      padding: 12px 14px;
      background: #fbfcfe;
    }}
    h2 {{
      margin: 0;
      font-size: 16px;
    }}
    .gold {{
      color: var(--muted);
      text-align: right;
    }}
    .gold span,
    .question span {{
      margin-right: 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .gold strong {{
      color: var(--text);
      font-size: 15px;
    }}
    .question {{
      border-bottom: 1px solid var(--line);
      padding: 10px 14px;
    }}
    .question p {{
      margin: 6px 0 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th,
    td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 12px;
      vertical-align: top;
      text-align: left;
    }}
    tr:last-child th,
    tr:last-child td {{
      border-bottom: 0;
    }}
    th {{
      width: 255px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
      background: #fbfcfe;
    }}
    td {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }}
    tr.exact td {{
      background: var(--exact);
    }}
    tr.contains td {{
      background: var(--contains);
    }}
    tr.empty td {{
      color: var(--muted);
      background: var(--empty);
    }}
    mark {{
      border-radius: 3px;
      background: #ffe08a;
      padding: 0 2px;
    }}
    .hidden {{
      display: none;
    }}
    @media (max-width: 860px) {{
      .layout {{
        display: block;
        padding: 12px;
      }}
      nav {{
        position: static;
        max-height: none;
        margin-bottom: 12px;
      }}
      .card-header {{
        display: block;
      }}
      .gold {{
        margin-top: 8px;
        text-align: left;
      }}
      th,
      td {{
        display: block;
        width: 100%;
      }}
      th {{
        border-bottom: 0;
        padding-bottom: 3px;
      }}
      td {{
        padding-top: 3px;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{dataset} answer inspection</h1>
    <div class="toolbar">
      <input id="filter" type="search" placeholder="Filter by request id, gold answer, experiment name, or generated text">
    </div>
  </header>
  <div class="layout">
    <nav aria-label="Requests">{nav}</nav>
    <main>{cards}</main>
  </div>
  <script>
    const filter = document.getElementById('filter');
    const cards = Array.from(document.querySelectorAll('.card'));
    filter.addEventListener('input', () => {{
      const needle = filter.value.trim().toLowerCase();
      for (const card of cards) {{
        card.classList.toggle('hidden', needle && !card.textContent.toLowerCase().includes(needle));
      }}
    }});
  </script>
</body>
</html>
    """.format(
        dataset=html.escape(dataset_stem),
        nav="\n".join(nav_items),
        cards="\n".join(cards),
    )
    output_path.write_text(html_text)


def write_manifest(
    output_dir: Path,
    chosen_records: dict[tuple[str, str], RunRecord],
    duplicate_records: dict[tuple[str, str], list[RunRecord]],
    missing_labels: dict[str, list[str]],
) -> None:
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "dataset",
                "label",
                "run_name",
                "job_id",
                "row_count",
                "result_csv",
                "dataset_path",
            ],
        )
        writer.writeheader()
        for (dataset_stem, label), record in sorted(chosen_records.items()):
            writer.writerow(
                {
                    "dataset": dataset_stem,
                    "label": label,
                    "run_name": record.run_name,
                    "job_id": record.job_id,
                    "row_count": record.row_count,
                    "result_csv": record.result_csv,
                    "dataset_path": record.dataset_path,
                }
            )

    duplicates_path = output_dir / "duplicates.csv"
    with duplicates_path.open("w", newline="") as duplicates_file:
        writer = csv.DictWriter(
            duplicates_file,
            fieldnames=["dataset", "label", "run_name", "job_id", "result_csv"],
        )
        writer.writeheader()
        for (dataset_stem, label), records in sorted(duplicate_records.items()):
            for record in records:
                writer.writerow(
                    {
                        "dataset": dataset_stem,
                        "label": label,
                        "run_name": record.run_name,
                        "job_id": record.job_id,
                        "result_csv": record.result_csv,
                    }
                )

    missing_path = output_dir / "missing_experiments.csv"
    with missing_path.open("w", newline="") as missing_file:
        writer = csv.DictWriter(missing_file, fieldnames=["dataset", "label"])
        writer.writeheader()
        for dataset_stem, labels in sorted(missing_labels.items()):
            for label in labels:
                writer.writerow({"dataset": dataset_stem, "label": label})


def write_skipped(output_dir: Path, rag_dir: Path) -> None:
    skipped_path = output_dir / "skipped_runs.csv"
    rows = []
    for reason in [
        "missing_result_csv",
        "unknown_experiment_label",
        "unknown_dataset",
    ]:
        attr_name = {
            "missing_result_csv": "skipped_missing_csv",
            "unknown_experiment_label": "skipped_unknown_label",
            "unknown_dataset": "skipped_unknown_dataset",
        }[reason]
        for run_name in getattr(discover_runs, attr_name, []):
            rows.append({"reason": reason, "run_name": run_name})

    with skipped_path.open("w", newline="") as skipped_file:
        writer = csv.DictWriter(skipped_file, fieldnames=["reason", "run_name"])
        writer.writeheader()
        writer.writerows(rows)


def parse_answer_literal(value: str) -> str:
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value
    return " | ".join(flatten_answers(parsed))


def main() -> None:
    args = parse_args()
    rag_dir = args.rag_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else rag_dir / "results_eval" / "outputs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_datasets = set(args.dataset) if args.dataset is not None else None
    records = discover_runs(rag_dir, selected_datasets)
    records_by_dataset_label: dict[tuple[str, str], list[RunRecord]] = {}
    dataset_paths: dict[str, Path] = {}
    for record in records:
        records_by_dataset_label.setdefault((record.dataset_stem, record.label), []).append(
            record
        )
        dataset_paths[record.dataset_stem] = record.dataset_path

    chosen_records: dict[tuple[str, str], RunRecord] = {}
    duplicate_records: dict[tuple[str, str], list[RunRecord]] = {}
    for key, matching_records in records_by_dataset_label.items():
        chosen, duplicates = choose_record(matching_records, args.prefer_gqa)
        chosen_records[key] = chosen
        if duplicates:
            duplicate_records[key] = duplicates

    missing_labels: dict[str, list[str]] = {}
    for dataset_stem, dataset_path in sorted(dataset_paths.items()):
        records_by_label = {
            label: chosen_records[(dataset_stem, label)]
            for label in EXPERIMENT_ORDER
            if (dataset_stem, label) in chosen_records
        }
        missing_labels[dataset_stem] = [
            label for label in EXPERIMENT_ORDER if label not in records_by_label
        ]
        write_dataset_table(
            dataset_stem,
            dataset_path,
            records_by_label,
            output_dir,
            args.include_question,
            args.preserve_newlines,
        )

    write_manifest(output_dir, chosen_records, duplicate_records, missing_labels)
    write_skipped(output_dir, rag_dir)

    print(f"Wrote answer tables to {output_dir}")
    print(f"Datasets: {', '.join(sorted(dataset_paths))}")
    print(f"Manifest: {output_dir / 'manifest.csv'}")
    print(f"Missing experiment report: {output_dir / 'missing_experiments.csv'}")


if __name__ == "__main__":
    main()
