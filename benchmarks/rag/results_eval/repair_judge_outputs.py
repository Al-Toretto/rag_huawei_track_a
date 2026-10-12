# SPDX-License-Identifier: Apache-2.0
"""Repair a judge-output JSONL file after an interrupted write."""

# Standard
from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Copy the valid JSONL prefix of an interrupted judge output."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def repair_jsonl(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Copy valid JSONL rows until the first malformed line.

    Args:
        input_path: Possibly corrupted JSONL path.
        output_path: Repaired JSONL path.

    Returns:
        Repair summary.
    """
    valid_rows = 0
    bad_line = None
    bad_error = None
    last_sample_id = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("rb") as input_file:
        with output_path.open("w", encoding="utf-8") as output_file:
            for line_number, raw_line in enumerate(input_file, start=1):
                if not raw_line.strip(b"\x00\r\n\t "):
                    bad_line = line_number
                    bad_error = "blank_or_nul_line"
                    break
                try:
                    row = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    bad_line = line_number
                    bad_error = f"{type(exc).__name__}: {exc}"
                    break
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                valid_rows += 1
                last_sample_id = row.get("sample_id")
    return {
        "input": str(input_path),
        "output": str(output_path),
        "valid_rows": valid_rows,
        "bad_line": bad_line,
        "bad_error": bad_error,
        "last_sample_id": last_sample_id,
    }


def main() -> None:
    """Run the repair CLI."""
    args = parse_args()
    summary = repair_jsonl(args.input, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
