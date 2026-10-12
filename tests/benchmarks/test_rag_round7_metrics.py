# SPDX-License-Identifier: Apache-2.0
"""Validate round-seven collection through its public API and CLI."""

# Standard
from pathlib import Path
from typing import Any
import csv
import json
import subprocess
import sys

# Third Party
import pytest

# First Party
from benchmarks.rag.results_eval.build_round7_metrics import build_metrics

REPO_DIR = Path(__file__).resolve().parents[2]
BUILDER = REPO_DIR / "benchmarks/rag/results_eval/build_round7_metrics.py"
EXPORTER = REPO_DIR / "benchmarks/rag/results_eval/export_judge_input.py"
CAPS = {
    "musique_longbench": 32,
    "2wikimqa_longbench": 32,
    "samsum_longbench": 256,
    "multifieldqa_zh_longbench": 256,
    "dureader_longbench": 1024,
}


def _write_csv(path: Path, rows: list[dict[str, Any]], delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def _run_fixture(
    rag_dir: Path,
    dataset: str = "musique_longbench",
    config: str = "plain32",
    job_id: int = 1,
    score: float = 0.5,
    seed: int = 42,
) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    experiment = f"qwen3_round07_seed{seed}_{dataset}_{config}"
    run_name = f"{experiment}_{job_id}"
    result_csv = rag_dir / "results" / run_name / f"{run_name}.csv"
    rows = [
        {
            "request_id": request_id,
            "success": True,
            "quality": score / 2,
            "f1_tokenizer": score,
            "precision": 0.4,
            "recall": 0.6,
            "ttft": 1.0,
            "generated_text": "" if request_id == 0 else "答案\nsecond line",
            "generation_token_cnt": 0 if request_id == 0 else 10,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "generation_seed": seed,
            "chat_template_mode": "qwen-non-thinking",
        }
        for request_id in range(50)
    ]
    _write_csv(result_csv, rows)
    dataset_path = rag_dir / "inputs" / f"{dataset}.json"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        json.dumps([{"question": "Q?", "answers": ["答案"]}] * 50),
        encoding="utf-8",
    )
    submission = {
        "dataset": dataset,
        "config_key": config,
        "experiment_name": experiment,
        "job_id": job_id,
        "generation_max_tokens": CAPS[dataset],
        "end_index": 50,
        "generation_seed": seed,
    }
    return submission, result_csv, rows


def _write_judge_output(
    path: Path, submissions: list[dict[str, Any]], groups: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    entries = {row["experiment_name"]: row for row in submissions}
    records = []
    for group in groups:
        entry = entries[group["experiment"]]
        quarter_points = round(200 * float(group["mean_semantic_score"]))
        for request_id in range(50):
            points = min(quarter_points, 4)
            quarter_points -= points
            records.append(
                {
                    "dataset": entry["dataset"],
                    "request_id": request_id,
                    "analysis_metadata": {
                        "experiment": entry["experiment_name"],
                        "run_name": f"{entry['experiment_name']}_{entry['job_id']}",
                        "job_id": entry["job_id"],
                    },
                    "input_fields": {
                        "prediction": "" if request_id == 0 else "答案\nsecond line"
                    },
                    "judge_output_parsed": {"semantic_score": points / 4},
                    "parse_error": None,
                }
            )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    return records


def _reference_fixture() -> list[dict[str, Any]]:
    references = []
    for dataset in CAPS:
        uses_summary_metric = dataset in {"samsum_longbench", "dureader_longbench"}
        metric = (
            "ROUGE-L zh"
            if dataset == "dureader_longbench"
            else "ROUGE-L"
            if uses_summary_metric
            else "F1 tokenizer"
        )
        for config in ("full_recompute", "lmcache_blend"):
            references.append(
                {
                    "dataset": dataset,
                    "config_key": config,
                    "score_metric": metric,
                    "score": (25 if uses_summary_metric else 50)
                    if config == "full_recompute"
                    else (20 if uses_summary_metric else 40),
                    "judge_score": 0.7 if config == "full_recompute" else 0.5,
                    "ttft_s": 2 if config == "full_recompute" else 1,
                    "n": 48
                    if (dataset, config) == ("samsum_longbench", "lmcache_blend")
                    else 50,
                    "generation_seed": 42,
                    "source_workbook": "historical_results.xlsx",
                }
            )
    return references


def test_round7_cli_exports_raw_empty_outputs_and_selects_metrics(
    tmp_path: Path,
) -> None:
    """Collect five datasets and export all fifty cases, including empty outputs."""
    rag_dir = tmp_path / "rag"
    submissions = []
    for job_id, dataset in enumerate(CAPS, 1):
        submission, _, _ = _run_fixture(rag_dir, dataset, job_id=job_id)
        submissions.append(submission)
    submission_path = tmp_path / "submission.tsv"
    _write_csv(submission_path, submissions, "\t")
    output = tmp_path / "collected"
    subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--submission",
            str(submission_path),
            "--rag-dir",
            str(rag_dir),
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with (output / "metrics.csv").open() as file:
        metrics = {row["dataset"]: row for row in csv.DictReader(file)}
    for dataset in CAPS:
        expected = 25 if dataset in {"samsum_longbench", "dureader_longbench"} else 50
        assert float(metrics[dataset]["score"]) == expected
        assert metrics[dataset]["empty_output_count"] == "1"
        assert metrics[dataset]["score_delta_vs_full"] == ""
    judge_input = tmp_path / "judge_input.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(EXPORTER),
            "--manifest",
            str(output / "manifest.csv"),
            "--output",
            str(judge_input),
            "--no-write-prompt-files",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    exported = [json.loads(line) for line in judge_input.read_text().splitlines()]
    assert len(exported) == 250
    assert sum(row["judge_input"]["prediction"] == "" for row in exported) == 5
    assert exported[1]["judge_input"]["prediction"] == "答案\nsecond line"
    labels = {row["dataset"]: row["experiment_name"] for row in submissions}
    assert (
        exported[1]["analysis_metadata"]["experiment"] == labels[exported[1]["dataset"]]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_id", 1),
        ("success", False),
        ("temperature", 0),
        ("top_p", 1),
        ("top_k", 10),
        ("min_p", 0.1),
        ("generation_seed", 123),
        ("chat_template_mode", "legacy"),
        ("f1_tokenizer", "nan"),
        ("ttft", -1),
        ("generation_token_cnt", 33),
    ],
)
def test_round7_rejects_incomparable_or_invalid_requests(
    tmp_path: Path, field: str, value: Any
) -> None:
    """Reject malformed coverage and metadata instead of averaging good rows."""
    rag_dir = tmp_path / "rag"
    submission, result_csv, rows = _run_fixture(rag_dir)
    rows[0][field] = value
    _write_csv(result_csv, rows)
    submission_path = tmp_path / "submission.tsv"
    _write_csv(submission_path, [submission], "\t")
    with pytest.raises(ValueError):
        build_metrics([submission_path], rag_dir, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_round7_rejects_missing_cases_smokes_and_ambiguous_reruns(
    tmp_path: Path,
) -> None:
    """Require full fifty-case runs and an explicit single choice of each rerun."""
    rag_dir = tmp_path / "rag"
    submission, result_csv, rows = _run_fixture(rag_dir)
    first = tmp_path / "first.tsv"
    second = tmp_path / "second.tsv"
    _write_csv(first, [submission], "\t")
    _write_csv(result_csv, rows[:48])
    with pytest.raises(ValueError, match="0..49"):
        build_metrics([first], rag_dir, tmp_path / "output")
    _write_csv(result_csv, rows)
    submission["end_index"] = 10
    _write_csv(first, [submission], "\t")
    with pytest.raises(ValueError, match="omit smokes"):
        build_metrics([first], rag_dir, tmp_path / "output")
    submission["end_index"] = 50
    _write_csv(first, [submission], "\t")
    rerun, _, _ = _run_fixture(rag_dir, job_id=2)
    _write_csv(second, [rerun], "\t")
    with pytest.raises(ValueError, match="choose one rerun"):
        build_metrics([first, second], rag_dir, tmp_path / "output")


def test_round7_ranks_shared_candidates_against_new_full_controls(
    tmp_path: Path,
) -> None:
    """Keep the largest dataset deficit visible when macro improvements mask it."""
    rag_dir = tmp_path / "rag"
    submissions = []
    judge_rows = []
    job_id = 0
    for dataset in CAPS:
        for config, score in (
            ("full_recompute", 0.5),
            ("balanced", 0.49),
            ("uneven", 0.6),
        ):
            job_id += 1
            if config == "uneven" and dataset == "2wikimqa_longbench":
                score = 0.4
            submission, _, _ = _run_fixture(rag_dir, dataset, config, job_id, score)
            submissions.append(submission)
            judge_score = 0.7 if config == "full_recompute" else 0.69
            if config == "uneven":
                judge_score = 0.6 if dataset == "2wikimqa_longbench" else 0.8
            judge_rows.append(
                {
                    "dataset": dataset,
                    "experiment": submission["experiment_name"],
                    "n": 50,
                    "n_parsed": 50,
                    "parse_error_rate": 0,
                    "mean_semantic_score": judge_score,
                }
            )
    submission_path = tmp_path / "submission.tsv"
    _write_csv(submission_path, submissions, "\t")
    judge_summary = tmp_path / "judge.csv"
    _write_csv(judge_summary, judge_rows)
    judge_output = tmp_path / "judge.jsonl"
    _write_judge_output(judge_output, submissions, judge_rows)
    output = tmp_path / "output"
    metrics = build_metrics(
        [submission_path], rag_dir, output, judge_summary, judge_output
    )
    wiki = next(
        row
        for row in metrics
        if row["dataset"] == "2wikimqa_longbench" and row["config_key"] == "uneven"
    )
    assert wiki["score_delta_vs_full"] == -10
    assert wiki["judge_delta_vs_full"] == pytest.approx(-0.1)
    with (output / "rankings.csv").open() as file:
        rankings = list(csv.DictReader(file))
    assert [row["config_key"] for row in rankings] == ["balanced", "uneven"]
    assert float(rankings[1]["macro_judge_delta"]) > 0
    assert float(rankings[1]["worst_judge_deficit"]) == pytest.approx(0.1)
    judge_rows.pop()
    _write_csv(judge_summary, judge_rows)
    with pytest.raises(ValueError, match="Missing requested judge group"):
        build_metrics([submission_path], rag_dir, output, judge_summary, judge_output)


@pytest.mark.parametrize(
    "field,value",
    [
        ("n", 48),
        ("n_parsed", 49),
        ("parse_error_rate", 0.02),
        ("mean_semantic_score", "nan"),
    ],
)
def test_round7_rejects_incomplete_or_invalid_judging(
    tmp_path: Path, field: str, value: Any
) -> None:
    """Require complete, parsed judgments for every included result group."""
    rag_dir = tmp_path / "rag"
    submission, _, _ = _run_fixture(rag_dir)
    submission_path = tmp_path / "submission.tsv"
    _write_csv(submission_path, [submission], "\t")
    judge = {
        "dataset": submission["dataset"],
        "experiment": submission["experiment_name"],
        "n": 50,
        "n_parsed": 50,
        "parse_error_rate": 0,
        "mean_semantic_score": 0.7,
    }
    judge[field] = value
    judge_path = tmp_path / "judge.csv"
    _write_csv(judge_path, [judge])
    with pytest.raises(ValueError):
        build_metrics(
            [submission_path],
            rag_dir,
            tmp_path / "output",
            judge_path,
            tmp_path / "raw.jsonl",
        )


def test_round7_rejects_wrong_caps_and_unmatched_control_seeds(tmp_path: Path) -> None:
    """Refuse generation-cap changes and full controls sampled with another seed."""
    rag_dir = tmp_path / "rag"
    submission, _, _ = _run_fixture(rag_dir)
    path = tmp_path / "submission.tsv"
    submission["generation_max_tokens"] = 256
    _write_csv(path, [submission], "\t")
    with pytest.raises(ValueError, match="generation cap"):
        build_metrics([path], rag_dir, tmp_path / "output")
    submission["generation_max_tokens"] = 32
    control, _, _ = _run_fixture(rag_dir, config="full_recompute", job_id=2, seed=123)
    _write_csv(path, [submission, control], "\t")
    with pytest.raises(ValueError, match="Control seed mismatch"):
        build_metrics([path], rag_dir, tmp_path / "output")


@pytest.mark.parametrize(
    "corruption",
    [
        "duplicate",
        "missing",
        "score",
        "no_score",
        "parse_error",
        "run",
        "prediction",
        "mean",
    ],
)
def test_round7_audits_raw_judgments(tmp_path: Path, corruption: str) -> None:
    """Reject stale or incomplete judgments even when summary counts look valid."""
    rag_dir = tmp_path / "rag"
    submission, _, _ = _run_fixture(rag_dir)
    submission_path = tmp_path / "submission.tsv"
    _write_csv(submission_path, [submission], "\t")
    group = {
        "dataset": submission["dataset"],
        "experiment": submission["experiment_name"],
        "n": 50,
        "n_parsed": 50,
        "parse_error_rate": 0,
        "mean_semantic_score": 0.7,
    }
    summary = tmp_path / "summary.csv"
    _write_csv(summary, [group])
    raw = tmp_path / "raw.jsonl"
    records = _write_judge_output(raw, [submission], [group])
    if corruption == "duplicate":
        records[-1] = records[0]
    elif corruption == "missing":
        records.pop()
    elif corruption == "score":
        records[0]["judge_output_parsed"]["semantic_score"] = 0.6
    elif corruption == "no_score":
        records[0]["judge_output_parsed"] = {}
    elif corruption == "parse_error":
        records[0]["parse_error"] = "unparsed"
    elif corruption == "run":
        records[0]["analysis_metadata"]["job_id"] = 999
    elif corruption == "prediction":
        records[0]["input_fields"]["prediction"] = "old prediction"
    elif corruption == "mean":
        records[0]["judge_output_parsed"]["semantic_score"] = 0
    raw.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    with pytest.raises(ValueError):
        build_metrics([submission_path], rag_dir, tmp_path / "output", summary, raw)


def test_round7_requires_raw_judgments_and_consistent_seeds(tmp_path: Path) -> None:
    """Require raw judging evidence and the same seed for a shared configuration."""
    rag_dir = tmp_path / "rag"
    submission, _, _ = _run_fixture(rag_dir)
    path = tmp_path / "submission.tsv"
    _write_csv(path, [submission], "\t")
    with pytest.raises(ValueError, match="together"):
        build_metrics([path], rag_dir, tmp_path / "output", tmp_path / "judge.csv")
    submission["generation_seed"] = -1
    _write_csv(path, [submission], "\t")
    with pytest.raises(ValueError, match="non-negative"):
        build_metrics([path], rag_dir, tmp_path / "output")
    submission["generation_seed"] = 42
    other, _, _ = _run_fixture(rag_dir, dataset="samsum_longbench", job_id=2, seed=123)
    _write_csv(path, [submission, other], "\t")
    with pytest.raises(ValueError, match="Mixed seeds"):
        build_metrics([path], rag_dir, tmp_path / "output")


def test_round7_compares_historical_baselines_without_judging_them(
    tmp_path: Path,
) -> None:
    """Rank five new runs against historical controls and export only new answers."""
    rag_dir = tmp_path / "rag"
    submissions = []
    groups = []
    for job_id, dataset in enumerate(CAPS, 1):
        submission, _, _ = _run_fixture(rag_dir, dataset, job_id=job_id, score=0.49)
        submissions.append(submission)
        groups.append(
            {
                "dataset": dataset,
                "experiment": submission["experiment_name"],
                "n": 50,
                "n_parsed": 50,
                "parse_error_rate": 0,
                "mean_semantic_score": 0.69,
            }
        )
    submission_path = tmp_path / "submission.tsv"
    _write_csv(submission_path, submissions, "\t")
    reference_path = tmp_path / "reference.csv"
    _write_csv(reference_path, _reference_fixture())
    summary_path = tmp_path / "judge_summary.csv"
    _write_csv(summary_path, groups)
    raw_path = tmp_path / "judge_output.jsonl"
    _write_judge_output(raw_path, submissions, groups)
    output = tmp_path / "output"
    subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--submission",
            str(submission_path),
            "--rag-dir",
            str(rag_dir),
            "--output-dir",
            str(output),
            "--reference-metrics",
            str(reference_path),
            "--judge-summary",
            str(summary_path),
            "--judge-output",
            str(raw_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with (output / "metrics.csv").open() as file:
        metrics = list(csv.DictReader(file))
    assert len(metrics) == 5
    for row in metrics:
        assert row["comparison_source"] == "historical_results.xlsx"
        assert row["n"] == row["full_n"] == "50"
        expected_count = "48" if row["dataset"] == "samsum_longbench" else "50"
        assert row["baseline16_n"] == expected_count
        expected_delta = (
            -0.5 if row["dataset"] in {"samsum_longbench", "dureader_longbench"} else -1
        )
        assert float(row["score_delta_vs_full"]) == pytest.approx(expected_delta)
        assert float(row["judge_delta_vs_full"]) == pytest.approx(-0.01)
        assert float(row["judge_delta_vs_baseline16"]) == pytest.approx(0.19)
        assert float(row["ttft_speedup_vs_full"]) == 2
    with (output / "rankings.csv").open() as file:
        ranking = list(csv.DictReader(file))
    assert len(ranking) == 1
    assert ranking[0]["config_key"] == "plain32"
    assert ranking[0]["dataset_count"] == "5"
    assert float(ranking[0]["worst_score_deficit"]) == pytest.approx(1)
    assert float(ranking[0]["worst_judge_deficit"]) == pytest.approx(0.01)
    with (output / "manifest.csv").open() as file:
        manifest = list(csv.DictReader(file))
    assert len(manifest) == 5
    assert {row["config_key"] for row in manifest} == {"plain32"}
    judge_input = tmp_path / "judge_input.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(EXPORTER),
            "--manifest",
            str(output / "manifest.csv"),
            "--output",
            str(judge_input),
            "--no-write-prompt-files",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    exported = [json.loads(line) for line in judge_input.read_text().splitlines()]
    assert len(exported) == 250
    assert {row["analysis_metadata"]["experiment"] for row in exported} == {
        row["experiment_name"] for row in submissions
    }


@pytest.mark.parametrize(
    "corruption",
    [
        "full_count",
        "blend_count",
        "seed",
        "negative_seed",
        "duplicate",
        "missing_full",
        "score",
        "judge",
        "ttft",
        "metric",
        "dataset",
        "source",
    ],
)
def test_round7_rejects_invalid_historical_reference_metrics(
    tmp_path: Path, corruption: str
) -> None:
    """Reject invalid reference cohorts instead of reporting unsupported deltas."""
    rag_dir = tmp_path / "rag"
    submission, _, _ = _run_fixture(rag_dir)
    submission_path = tmp_path / "submission.tsv"
    _write_csv(submission_path, [submission], "\t")
    references = _reference_fixture()
    if corruption == "full_count":
        references[0]["n"] = 48
    elif corruption == "blend_count":
        references[1]["n"] = 48
    elif corruption == "seed":
        for row in references:
            row["generation_seed"] = 123
    elif corruption == "negative_seed":
        references[0]["generation_seed"] = -1
    elif corruption == "duplicate":
        references.append(references[0])
    elif corruption == "missing_full":
        references.pop(0)
    elif corruption == "score":
        references[0]["score"] = 101
    elif corruption == "judge":
        references[0]["judge_score"] = "nan"
    elif corruption == "ttft":
        references[0]["ttft_s"] = 0
    elif corruption == "metric":
        references[0]["score_metric"] = "ROUGE-L"
    elif corruption == "dataset":
        references[0]["dataset"] = "unsupported"
    elif corruption == "source":
        references[0]["source_workbook"] = ""
    reference_path = tmp_path / "reference.csv"
    _write_csv(reference_path, references)
    with pytest.raises(ValueError):
        build_metrics(
            [submission_path],
            rag_dir,
            tmp_path / "output",
            reference_metrics=reference_path,
        )
    assert not (tmp_path / "output").exists()


def test_round7_rejects_ambiguous_submitted_and_historical_controls(
    tmp_path: Path,
) -> None:
    """Require an explicit choice between a new control and a historical control."""
    rag_dir = tmp_path / "rag"
    submission, _, _ = _run_fixture(rag_dir, config="full_recompute")
    submission_path = tmp_path / "submission.tsv"
    _write_csv(submission_path, [submission], "\t")
    reference_path = tmp_path / "reference.csv"
    _write_csv(reference_path, _reference_fixture())
    with pytest.raises(ValueError, match="Ambiguous"):
        build_metrics(
            [submission_path],
            rag_dir,
            tmp_path / "output",
            reference_metrics=reference_path,
        )
