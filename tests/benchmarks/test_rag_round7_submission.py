# SPDX-License-Identifier: Apache-2.0
"""Check round-7 submission commands and provenance without running Slurm jobs."""

# Standard
from pathlib import Path
import csv
import json
import os
import shlex
import subprocess

# Third Party
import pytest

REPO_DIR = Path(__file__).resolve().parents[2]
SUBMITTER = REPO_DIR / "benchmarks/rag/sbash_runs/submit_round7.sh"
DATASET_CAPS = {
    "2wikimqa_longbench": "32",
    "musique_longbench": "32",
    "samsum_longbench": "256",
    "multifieldqa_zh_longbench": "256",
    "dureader_longbench": "1024",
}
PROTOCOL = {
    "END_INDEX": "50",
    "GENERATION_SEED": "42",
    "QPS": "0.25",
    "SAMPLING_TEMPERATURE": "0.7",
    "SAMPLING_TOP_P": "0.8",
    "SAMPLING_TOP_K": "20",
    "SAMPLING_MIN_P": "0.0",
    "CHAT_TEMPLATE_MODE": "qwen-non-thinking",
    "MAX_MODEL_LEN": "30000",
    "MAX_NUM_BATCHED_TOKENS": "16384",
    "KV_STORAGE_SIZE": "320GB",
    "KV_CHUNK_SIZE": "256",
    "BLEND_SEPARATOR": " # # ",
    "WARMUP": "0",
}
INHERITED_JOB_OVERRIDES = (
    "OUTPUT_DIR",
    "LOG_DIR",
    "OUTPUT_FILE",
    "DEBUG_OUTPUT_FILE",
    "PORT",
    "KV_TRANSFER_PORT",
    "PROMPT_BUILD_METHOD",
)


def _make_repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    rag = repo / "benchmarks/rag"
    config_dir = rag / "07_accuracy_recovery"
    config_dir.mkdir(parents=True)
    for prefix in ("a", "b"):
        for index in range(1, 7):
            (config_dir / f"{prefix}{index:02d}_candidate.yaml").write_text(
                "enable_blending: true\n", encoding="utf-8"
            )
    (rag / "inputs").mkdir()
    for dataset in DATASET_CAPS:
        (rag / "inputs" / f"{dataset}.json").write_text("[]\n", encoding="utf-8")
    (rag / "sbash_runs").mkdir()
    (rag / "sbash_runs/run_lmcache_experiment_h200.sbatch").write_text(
        "#!/bin/bash\n", encoding="utf-8"
    )
    return repo


def _environment(repo: Path, **overrides: str) -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "REPO_DIR",
        "CONFIG_SET",
        "CONFIG_FILES",
        "DATASETS",
        "DRY_RUN",
        "SMOKE_TEST",
        "GENERATION_SEED",
        "MODEL_SNAPSHOT",
        "RUN_PREFIX",
        "END_INDEX",
        "INCLUDE_CONTROLS",
    ):
        env.pop(key, None)
    env.update(REPO_DIR=str(repo), **overrides)
    return env


def _run_submitter(repo: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SUBMITTER)],
        env=_environment(repo, **overrides),
        capture_output=True,
        text=True,
        check=False,
    )


def _dry_run_exports(output: str) -> list[dict[str, str]]:
    exports = []
    for line in output.splitlines():
        if not line.startswith("DRY RUN "):
            continue
        args = shlex.split(line.removeprefix("DRY RUN "))
        assert args[:2] == ["sbatch", "--parsable"]
        export_arg = next(arg for arg in args if arg.startswith("--export="))
        exports.append(
            dict(
                value.split("=", 1)
                for value in export_arg.removeprefix("--export=").split(",")
                if "=" in value
            )
        )
    return exports


def test_round7_all_preserves_protocol_and_dry_run_writes_nothing(
    tmp_path: Path,
) -> None:
    """Preview 12 new configs on five datasets and preserve their output caps."""
    repo = _make_repository(tmp_path)
    rag = repo / "benchmarks/rag"
    completed = _run_submitter(repo, SAMPLING_TEMPERATURE="0", QPS="99", WARMUP="1")

    assert completed.returncode == 0, completed.stderr
    exports = _dry_run_exports(completed.stdout)
    assert len(exports) == 60
    for fields in exports:
        assert PROTOCOL.items() <= fields.items()
        dataset = Path(fields["DATASET_PATH"]).stem
        assert fields["MAX_TOKENS"] == DATASET_CAPS[dataset]
        assert fields["EXPERIMENT_NAME"].startswith("07_accuracy_seed42_")
    assert all("CONFIG_FILE" in fields for fields in exports)
    assert {Path(fields["CONFIG_FILE"]).stem for fields in exports} == {
        f"{prefix}{index:02d}_candidate"
        for prefix in ("a", "b")
        for index in range(1, 7)
    }
    assert all(
        Path(fields["CONFIG_FILE"]).parent == rag / "07_accuracy_recovery"
        for fields in exports
    )
    assert not (rag / "logs").exists()
    assert not (rag / "07_accuracy_recovery/submissions").exists()


@pytest.mark.parametrize("config_set", ("pilot", "followup"))
def test_round7_optional_config_sets_submit_no_baselines(
    tmp_path: Path, config_set: str
) -> None:
    """Allow either six-config subset without adding existing baseline jobs."""
    repo = _make_repository(tmp_path)
    completed = _run_submitter(repo, CONFIG_SET=config_set)

    assert completed.returncode == 0, completed.stderr
    exports = _dry_run_exports(completed.stdout)
    assert len(exports) == 30
    prefix = "a" if config_set == "pilot" else "b"
    assert {Path(fields["CONFIG_FILE"]).stem for fields in exports} == {
        f"{prefix}{index:02d}_candidate" for index in range(1, 7)
    }


def test_round7_selected_followup_excludes_controls(tmp_path: Path) -> None:
    """Select particular follow-up stems and normalize dataset family aliases."""
    repo = _make_repository(tmp_path)
    completed = _run_submitter(
        repo,
        CONFIG_SET="followup",
        CONFIG_FILES="b02_candidate,b05_candidate.yaml",
        DATASETS="wikimqa multifieldqa_zh dureader",
    )

    assert completed.returncode == 0, completed.stderr
    exports = _dry_run_exports(completed.stdout)
    assert len(exports) == 6
    assert {Path(fields["CONFIG_FILE"]).stem for fields in exports} == {
        "b02_candidate",
        "b05_candidate",
    }
    assert {Path(fields["DATASET_PATH"]).stem for fields in exports} == {
        "2wikimqa_longbench",
        "multifieldqa_zh_longbench",
        "dureader_longbench",
    }


def test_round7_smoke_uses_two_cases_and_a_distinct_seed_prefix(
    tmp_path: Path,
) -> None:
    """Keep smoke outputs separate from the full-sized run for the same seed."""
    repo = _make_repository(tmp_path)
    completed = _run_submitter(
        repo,
        CONFIG_FILES="a01_candidate",
        DATASETS="samsum",
        SMOKE_TEST="1",
        GENERATION_SEED="43",
    )

    assert completed.returncode == 0, completed.stderr
    exports = _dry_run_exports(completed.stdout)
    assert len(exports) == 1
    assert all(fields["END_INDEX"] == "2" for fields in exports)
    assert all(fields["GENERATION_SEED"] == "43" for fields in exports)
    assert all(
        fields["EXPERIMENT_NAME"].startswith("07_accuracy_seed43_smoke_")
        for fields in exports
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"DRY_RUN": "2"},
        {"SMOKE_TEST": "yes"},
        {"CONFIG_SET": "missing"},
        {"INCLUDE_CONTROLS": "1"},
        {"INCLUDE_CONTROLS": "2"},
        {"CONFIG_FILES": "a01_candidate,"},
        {"CONFIG_FILES": "missing"},
        {"CONFIG_FILES": "../outside"},
        {"DATASETS": "unknown"},
        {"DATASETS": "musique musique_longbench"},
        {"END_INDEX": "48"},
        {"GENERATION_SEED": "-1"},
        {"RUN_PREFIX": "previous_round"},
    ],
)
def test_round7_rejects_invalid_selection_before_submission(
    tmp_path: Path, overrides: dict[str, str]
) -> None:
    """Reject invalid selection or protocol settings without creating artifacts."""
    repo = _make_repository(tmp_path)
    completed = _run_submitter(repo, **overrides)

    assert completed.returncode != 0
    assert "DRY RUN" not in completed.stdout
    assert not (repo / "benchmarks/rag/07_accuracy_recovery/submissions").exists()


def test_round7_rejects_repeated_baselines_before_real_submission(
    tmp_path: Path,
) -> None:
    """Reject inherited control settings before checking Slurm or writing a TSV."""
    repo = _make_repository(tmp_path)
    completed = _run_submitter(repo, DRY_RUN="0", INCLUDE_CONTROLS="1")

    assert completed.returncode != 0
    assert "Existing baselines are reused" in completed.stderr
    assert "SUBMITTED" not in completed.stdout
    assert not (repo / "benchmarks/rag/logs").exists()
    assert not (repo / "benchmarks/rag/07_accuracy_recovery/submissions").exists()


def test_round7_submission_records_ids_and_clears_inherited_job_overrides(
    tmp_path: Path,
) -> None:
    """Record the TSV contract using a fake sbatch that captures its public inputs."""
    repo = _make_repository(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "qwen3"}\n', encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls_path = tmp_path / "sbatch_calls.jsonl"
    stub = bin_dir / "sbatch"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "path = pathlib.Path(os.environ['ROUND7_STUB_CALLS'])\n"
        "calls = path.read_text().splitlines() if path.exists() else []\n"
        "capture = {'arguments': sys.argv[1:], 'environment': dict(os.environ)}\n"
        "with path.open('a') as file:\n"
        "    file.write(json.dumps(capture) + '\\n')\n"
        "print(f'{101 + len(calls)};mockcluster')\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    completed = _run_submitter(
        repo,
        DRY_RUN="0",
        CONFIG_FILES="a01_candidate",
        DATASETS="musique samsum",
        MODEL_SNAPSHOT=str(model),
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        ROUND7_STUB_CALLS=str(calls_path),
        **{name: "inherited_override" for name in INHERITED_JOB_OVERRIDES},
    )

    assert completed.returncode == 0, completed.stderr
    submission_dir = repo / "benchmarks/rag/07_accuracy_recovery/submissions"
    paths = list(submission_dir.glob("07_accuracy_seed42_all_*.tsv"))
    assert len(paths) == 1
    with paths[0].open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        assert reader.fieldnames == [
            "dataset",
            "config_key",
            "experiment_name",
            "job_id",
            "generation_max_tokens",
            "end_index",
            "generation_seed",
        ]
        rows = list(reader)
    assert len(rows) == 2
    assert [row["job_id"] for row in rows] == ["101", "102"]
    assert {row["config_key"] for row in rows} == {
        "07_accuracy_recovery__a01_candidate",
    }
    assert all(row["end_index"] == "50" for row in rows)
    assert all(row["generation_seed"] == "42" for row in rows)
    assert all(
        row["generation_max_tokens"] == DATASET_CAPS[row["dataset"]] for row in rows
    )
    assert (repo / "benchmarks/rag/logs").is_dir()
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    assert len(calls) == 2
    for call in calls:
        assert not set(INHERITED_JOB_OVERRIDES).intersection(call["environment"])
