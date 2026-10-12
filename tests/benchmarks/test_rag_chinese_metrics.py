# SPDX-License-Identifier: Apache-2.0
"""Exercise the Chinese matrix validator with completed synthetic result files."""

# Standard
from pathlib import Path
from zipfile import ZipFile
import csv
import subprocess
import sys

# First Party
from benchmarks.rag.results_eval.build_qwen3_artifacts import (
    DATASETS,
    EXPERIMENTS,
    WORKBOOK_COLUMNS,
)


REPO_DIR = Path(__file__).resolve().parents[2]
BUILDER = REPO_DIR / "benchmarks" / "rag" / "results_eval" / "build_chinese_metrics.py"
SPECS_DIR = REPO_DIR / "benchmarks" / "rag"


def test_chinese_metric_builder_requires_full_matrix_and_correct_metric(
    tmp_path: Path,
) -> None:
    """Validate score selection and reject a run with incomplete request IDs."""
    rag_dir = tmp_path / "benchmarks" / "rag"
    (rag_dir / "results").mkdir(parents=True)
    (rag_dir / "logs").mkdir(parents=True)
    config_files = [
        SPECS_DIR / "lmcache_blend.yaml",
        *sorted((SPECS_DIR / "yaml_03_bound_and_mix").glob("*.yaml")),
        *sorted((SPECS_DIR / "yaml_04_around_high_div").glob("*.yaml")),
        *sorted((SPECS_DIR / "yaml_05_no_hkvd_and_layers").glob("*.yaml")),
        *sorted((SPECS_DIR / "06_blend16_shared_candidates").glob("*.yaml")),
    ]
    keys = ["full_recompute"] + [
        str(path.relative_to(SPECS_DIR)).removesuffix(".yaml").replace("/", "__")
        for path in config_files
    ]
    submission_lines = []
    last_csv = None
    for dataset in ("multifieldqa_zh_longbench", "dureader_longbench"):
        for job_id, key in enumerate(
            keys, start=1 if dataset.startswith("multi") else 66
        ):
            experiment = f"qwen3_chat_nothink_zh_v1_seed42_{dataset}_{key}"
            run_name = f"{experiment}_{job_id}"
            run_dir = rag_dir / "results" / run_name
            log_dir = rag_dir / "logs" / run_name
            run_dir.mkdir()
            log_dir.mkdir()
            last_csv = run_dir / f"{run_name}.csv"
            with last_csv.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=(
                        "request_id",
                        "success",
                        "quality",
                        "f1_tokenizer",
                        "f1_word",
                        "precision",
                        "recall",
                        "ttft",
                        "tpot",
                        "generation_token_cnt",
                        "generated_text",
                        "temperature",
                        "top_p",
                        "top_k",
                        "min_p",
                        "generation_seed",
                        "chat_template_mode",
                    ),
                )
                writer.writeheader()
                for request_id in range(50):
                    writer.writerow(
                        {
                            "request_id": request_id,
                            "success": "True",
                            "quality": 0.75,
                            "f1_tokenizer": 0.5,
                            "f1_word": 0.25,
                            "precision": 0.4,
                            "recall": 0.6,
                            "ttft": 1.0,
                            "tpot": 0.1,
                            "generation_token_cnt": 10,
                            "generated_text": "答案",
                            "temperature": 0.7,
                            "top_p": 0.8,
                            "top_k": 20,
                            "min_p": 0.0,
                            "generation_seed": 42,
                            "chat_template_mode": "qwen-non-thinking",
                        }
                    )
            (log_dir / "benchmark.log").write_text(
                "Finished benchmarking\nthroughput=0.25\n"
                "average_quality=0.75\naverage_f1_tokenizer=0.5\n",
                encoding="utf-8",
            )
            submission_lines.append(
                f"SUBMITTED lmcache\tjob_id={job_id}\texperiment={experiment}"
            )
    submission = tmp_path / "submission.txt"
    submission.write_text("\n".join(submission_lines) + "\n", encoding="utf-8")
    command = [
        sys.executable,
        str(BUILDER),
        "--submission",
        str(submission),
        "--rag-dir",
        str(rag_dir),
    ]

    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    output_dir = rag_dir / "results_eval" / "outputs" / "qwen3_chinese"
    with (output_dir / "metrics.csv").open(newline="", encoding="utf-8") as file:
        scores = list(csv.DictReader(file))
    assert "Validated 130 Chinese runs" in completed.stdout
    assert len(scores) == 130
    assert {
        row["score"] for row in scores if row["score_metric"] == "F1 tokenizer"
    } == {"0.5"}
    assert {row["score"] for row in scores if row["score_metric"] == "ROUGE-L zh"} == {
        "0.75"
    }
    chinese_table = (
        rag_dir
        / "results"
        / ("rag_qwen3_chat_nothink_zh_v1_seed42_meeting_metrics.tsv")
    )
    with chinese_table.open(newline="", encoding="utf-8") as file:
        meeting_rows = list(csv.DictReader(file, delimiter="\t"))
    assert len(meeting_rows) == 130
    assert meeting_rows[0]["Experiment"] == "Full Recompute"
    assert meeting_rows[0]["Score"] == "50.0"
    assert meeting_rows[0]["Precision"] == "40.0"
    assert meeting_rows[65]["Score Metric"] == "ROUGE-L zh"
    assert meeting_rows[65]["Score"] == "75.0"
    assert meeting_rows[65]["Precision"] == ""
    assert all(not row["Judge Score (0-1)"] for row in meeting_rows)
    audit = chinese_table.with_name(
        "rag_qwen3_chat_nothink_zh_v1_seed42_meeting_metrics_audit.txt"
    )
    assert len(audit.read_text(encoding="utf-8").splitlines()) == 130

    english_tsv = tmp_path / "english.tsv"
    with english_tsv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=WORKBOOK_COLUMNS, delimiter="\t")
        writer.writeheader()
        for dataset in DATASETS:
            for experiment in EXPERIMENTS:
                writer.writerow(
                    {
                        "Dataset": dataset.display_name,
                        "Experiment": experiment.label,
                        "Score Metric": dataset.score_metric,
                        "Score": 42,
                        "Judge Score (0-1)": 0.5,
                    }
                )
    subprocess.run(
        [*command, "--english-tsv", str(english_tsv)],
        capture_output=True,
        text=True,
        check=True,
    )
    combined = (
        rag_dir
        / "results"
        / ("rag_qwen3_five_datasets_meeting_metrics_current_9622_chinese_unjudged.tsv")
    )
    with combined.open(newline="", encoding="utf-8") as file:
        combined_rows = list(csv.DictReader(file, delimiter="\t"))
    assert len(combined_rows) == 325
    assert combined_rows[0]["Judge Score (0-1)"] == "0.5"
    assert combined_rows[195]["Judge Score (0-1)"] == ""
    with ZipFile(combined.with_suffix(".xlsx")) as workbook:
        sheet = workbook.read("xl/worksheets/sheet1.xml").decode("utf-8")
    assert "MultiFieldQA zh (LongBench)" in sheet
    assert "DuReader (LongBench)" in sheet
    assert '<c r="D2" s="3"><v>42.0</v></c>' in sheet

    judge_summary = tmp_path / "judge_summary.csv"
    with judge_summary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "dataset",
                "experiment",
                "n",
                "n_parsed",
                "parse_error_rate",
                "mean_semantic_score",
                "correct_rate",
                "mostly_correct_rate",
                "incorrect_rate",
            ),
        )
        writer.writeheader()
        for dataset in ("multifieldqa_zh_longbench", "dureader_longbench"):
            for experiment in EXPERIMENTS:
                writer.writerow(
                    {
                        "dataset": dataset,
                        "experiment": experiment.label,
                        "n": 50,
                        "n_parsed": 50,
                        "parse_error_rate": 0,
                        "mean_semantic_score": 0.75,
                        "correct_rate": 0.5,
                        "mostly_correct_rate": 0.25,
                        "incorrect_rate": 0.1,
                    }
                )
    subprocess.run(
        [
            *command,
            "--english-tsv",
            str(english_tsv),
            "--judge-summary",
            str(judge_summary),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    judged = (
        rag_dir
        / "results"
        / "rag_qwen3_five_datasets_meeting_metrics_current_9622_chinese_judged.tsv"
    )
    with judged.open(newline="", encoding="utf-8") as file:
        judged_rows = list(csv.DictReader(file, delimiter="\t"))
    assert len(judged_rows) == 325
    assert judged_rows[195]["Judge Score (0-1)"] == "0.75"
    assert judged_rows[195]["Judge Correct %"] == "50.0"
    assert judged_rows[195]["Judge Mostly %"] == "25.0"
    assert judged_rows[195]["Judge Incorrect %"] == "10.0"

    subprocess.run(
        [
            *command,
            "--english-tsv",
            str(english_tsv),
            "--judge-summary",
            str(judge_summary),
            "--judge-provenance",
            "adjudicated",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert judged.is_file()
    assert len(judged.read_text(encoding="utf-8").splitlines()) == 326
    assert not (
        rag_dir
        / "results"
        / "rag_qwen3_five_datasets_meeting_metrics_current_9622_chinese_adjudicated.tsv"
    ).exists()

    judge_summary.write_text(
        judge_summary.read_text(encoding="utf-8").replace(",50,50,0,", ",50,49,0,", 1),
        encoding="utf-8",
    )
    incomplete = subprocess.run(
        [*command, "--judge-summary", str(judge_summary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert incomplete.returncode != 0
    assert "Incomplete Chinese judge group" in incomplete.stderr

    assert last_csv is not None
    with last_csv.open(encoding="utf-8") as file:
        lines = file.readlines()
    last_csv.write_text("".join(lines[:-1]), encoding="utf-8")
    failed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert failed.returncode != 0
    assert "Expected request IDs 0..49" in failed.stderr
