# SPDX-License-Identifier: Apache-2.0
"""Run a zero-cost local LLM judge over RAG judge-input JSONL files."""

# Standard
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
from urllib import error, request
import argparse
import json
import math
import re
import sys

# First Party
from judge_diagnostics import DEFAULT_MAX_TOKEN_THRESHOLD, compute_rule_diagnostics
from judge_prompts import (
    JUDGE_PROMPT_VERSION,
    RUNTIME_SETTINGS,
    get_prompt,
    output_schema_for_prompt,
    render_prometheus_prompt,
    render_user_prompt,
)


DEFAULT_INPUT = Path(__file__).resolve().parent / "outputs" / "judge_input.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "judge_outputs.jsonl"
DEFAULT_OLLAMA_MODEL = "prometheus2:7b"
DEFAULT_TRANSFORMERS_MODEL = "prometheus-eval/prometheus-7b-v2.0"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_LLAMA_CPP_URL = "http://localhost:8080/v1/chat/completions"


class LocalJudgeBackend:
    """Base interface for local judge backends."""

    name = "base"

    def __init__(self, model: str, timeout: float) -> None:
        """Initialize the backend.

        Args:
            model: Local model name or path.
            timeout: Request timeout in seconds.
        """
        self.model = model
        self.timeout = timeout

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate judge output.

        Args:
            system_prompt: System prompt text.
            user_prompt: User prompt text.

        Returns:
            Raw model output.

        Raises:
            NotImplementedError: Always for the base class.
        """
        raise NotImplementedError


class OllamaBackend(LocalJudgeBackend):
    """Ollama backend using the local ``/api/generate`` endpoint."""

    name = "ollama"

    def __init__(self, model: str, timeout: float, base_url: str) -> None:
        """Initialize the Ollama backend.

        Args:
            model: Ollama model name.
            timeout: Request timeout in seconds.
            base_url: Ollama base URL.
        """
        super().__init__(model, timeout)
        self.base_url = base_url.rstrip("/")
        self._assert_model_available()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate judge output with Ollama.

        Args:
            system_prompt: System prompt text.
            user_prompt: User prompt text.

        Returns:
            Raw judge output text.
        """
        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": RUNTIME_SETTINGS["temperature"],
                "top_p": RUNTIME_SETTINGS["top_p"],
                "num_predict": RUNTIME_SETTINGS["max_output_tokens"],
            },
        }
        response = post_json(
            f"{self.base_url}/api/generate",
            payload,
            self.timeout,
        )
        if "error" in response:
            raise RuntimeError(f"Ollama error: {response['error']}")
        return str(response.get("response", ""))

    def _assert_model_available(self) -> None:
        """Raise a clear error when Ollama or the model is unavailable."""
        try:
            response = get_json(f"{self.base_url}/api/tags", self.timeout)
        except RuntimeError as exc:
            raise RuntimeError(
                "Ollama is not reachable. Start Ollama locally and install a "
                f"Prometheus 2 model, then retry. URL: {self.base_url}"
            ) from exc
        names = {
            str(model.get("name") or model.get("model"))
            for model in response.get("models", [])
        }
        if self.model not in names:
            available = ", ".join(sorted(name for name in names if name))
            raise RuntimeError(
                f"Ollama model {self.model!r} is not installed. Install a "
                "local Prometheus 2 7B/quantized model, or pass --model with "
                f"the installed name. Available models: {available or '<none>'}"
            )


class LlamaCppBackend(LocalJudgeBackend):
    """llama.cpp server backend using an OpenAI-compatible chat endpoint."""

    name = "llama-cpp"

    def __init__(self, model: str, timeout: float, url: str) -> None:
        """Initialize the llama.cpp backend.

        Args:
            model: Model name exposed by the server.
            timeout: Request timeout in seconds.
            url: OpenAI-compatible chat completions URL.
        """
        super().__init__(model, timeout)
        self.url = url

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate judge output with llama.cpp server.

        Args:
            system_prompt: System prompt text.
            user_prompt: User prompt text.

        Returns:
            Raw judge output text.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": RUNTIME_SETTINGS["temperature"],
            "top_p": RUNTIME_SETTINGS["top_p"],
            "max_tokens": RUNTIME_SETTINGS["max_output_tokens"],
        }
        response = post_json(self.url, payload, self.timeout)
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError(f"llama.cpp returned no choices: {response}")
        message = choices[0].get("message", {})
        return str(message.get("content", ""))


class TransformersBackend(LocalJudgeBackend):
    """Hugging Face Transformers backend for a locally available judge model."""

    name = "transformers"

    def __init__(
        self,
        model: str,
        timeout: float,
        local_files_only: bool,
        max_output_tokens: int,
    ) -> None:
        """Initialize the Transformers backend.

        Args:
            model: Hugging Face model id or local path.
            timeout: Unused compatibility timeout.
            local_files_only: Whether to avoid downloads.
            max_output_tokens: Maximum newly generated tokens.

        Raises:
            RuntimeError: If Transformers or the model is unavailable.
        """
        super().__init__(model, timeout)
        self.local_files_only = local_files_only
        self.max_output_tokens = max_output_tokens
        try:
            # Third Party
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformers backend requires torch and transformers. Install "
                "them locally, or use --backend ollama / --backend llama-cpp."
            ) from exc

        self.torch = torch
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model,
                local_files_only=local_files_only,
            )
            self.model_obj = AutoModelForCausalLM.from_pretrained(
                model,
                device_map="auto",
                torch_dtype="auto",
                local_files_only=local_files_only,
            )
        except Exception as exc:
            mode = "local cache" if local_files_only else "Hugging Face"
            raise RuntimeError(
                f"Could not load {model!r} from {mode}. For official results, "
                "use a local Prometheus 2 model and keep the model fixed."
            ) from exc

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate judge output with Transformers.

        Args:
            system_prompt: System prompt text.
            user_prompt: User prompt text.

        Returns:
            Raw judge output text.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = f"{system_prompt}\n\n{user_prompt}\n\nJSON:"
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {key: value.to(self.model_obj.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            output_ids = self.model_obj.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_output_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        return str(self.tokenizer.decode(new_tokens, skip_special_tokens=True))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run a local open-source LLM judge over judge_input.jsonl."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--backend",
        choices=["ollama", "llama-cpp", "transformers", "auto"],
        default="ollama",
        help="Local backend. Use --dry-run to avoid loading a model.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Local judge model name/path. Defaults to Prometheus 2 names for "
            "the selected backend."
        ),
    )
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--llama-cpp-url", default=DEFAULT_LLAMA_CPP_URL)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=RUNTIME_SETTINGS["max_output_tokens"],
        help="Generation limit for the Transformers backend (default: 1024).",
    )
    parser.add_argument(
        "--require-complete-scores",
        action="store_true",
        help="Exit unsuccessfully if selected rows lack a valid parsed score.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--experiment", action="append", default=None)
    parser.add_argument("--prompt-name", action="append", default=None)
    parser.add_argument(
        "--max-token-threshold",
        type=int,
        default=DEFAULT_MAX_TOKEN_THRESHOLD,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render prompts and diagnostics for selected rows without judging.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Write audit rows with diagnostics but without calling a model.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip sample_ids already present in the output file.",
    )
    parser.add_argument(
        "--allow-non-prometheus",
        action="store_true",
        help="Allow non-Prometheus model names for exploratory runs.",
    )
    parser.add_argument(
        "--hf-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For Transformers, load only files already present locally.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["auto", "json", "prometheus"],
        default="auto",
        help=(
            "Prompt/parser style. Auto uses Prometheus-native prompts for "
            "Prometheus models and JSON prompts for other models."
        ),
    )
    args = parser.parse_args()
    if args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be positive")
    return args


def build_backend(args: argparse.Namespace) -> LocalJudgeBackend:
    """Build a local judge backend from CLI arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Local judge backend.

    Raises:
        RuntimeError: If the requested backend is unavailable.
    """
    backend_name = args.backend
    if backend_name == "auto":
        errors = []
        for candidate in ("ollama", "llama-cpp", "transformers"):
            try:
                candidate_args = argparse.Namespace(**vars(args))
                candidate_args.backend = candidate
                return build_backend(candidate_args)
            except RuntimeError as exc:
                errors.append(f"{candidate}: {exc}")
        raise RuntimeError("No local judge backend is available:\n" + "\n".join(errors))

    model = resolve_model_name(backend_name, args.model)
    assert_prometheus_model(model, args.allow_non_prometheus)
    if backend_name == "ollama":
        return OllamaBackend(model, args.timeout, args.ollama_url)
    if backend_name == "llama-cpp":
        return LlamaCppBackend(model, args.timeout, args.llama_cpp_url)
    if backend_name == "transformers":
        return TransformersBackend(
            model, args.timeout, args.hf_local_files_only, args.max_output_tokens
        )
    raise RuntimeError(f"Unsupported backend: {backend_name}")


def resolve_model_name(backend_name: str, model: Optional[str]) -> str:
    """Resolve backend-specific default model names.

    Args:
        backend_name: Backend name.
        model: Optional user-provided model name.

    Returns:
        Model name or path.
    """
    if model:
        return model
    if backend_name == "transformers":
        return DEFAULT_TRANSFORMERS_MODEL
    return DEFAULT_OLLAMA_MODEL


def assert_prometheus_model(model: str, allow_non_prometheus: bool) -> None:
    """Require a Prometheus-like model name unless explicitly overridden.

    Args:
        model: Model name/path.
        allow_non_prometheus: Whether non-Prometheus names are allowed.

    Raises:
        RuntimeError: If the model name does not look like Prometheus.
    """
    if allow_non_prometheus:
        return
    if "prometheus" not in model.lower():
        raise RuntimeError(
            f"Refusing model {model!r} for official judging because it does "
            "not look like Prometheus. Pass --allow-non-prometheus only for "
            "exploratory debugging."
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file.

    Args:
        path: JSONL path.

    Returns:
        Parsed rows.
    """
    rows = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {path}") from exc
    return rows


def selected_rows(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Filter and slice judge-input rows.

    Args:
        rows: Input rows.
        args: Parsed CLI arguments.

    Returns:
        Selected rows.
    """
    selected = []
    dataset_filter = set(args.dataset or [])
    experiment_filter = set(args.experiment or [])
    prompt_filter = set(args.prompt_name or [])
    for row in rows:
        metadata = row.get("analysis_metadata", {})
        if dataset_filter and row.get("dataset") not in dataset_filter:
            continue
        if experiment_filter and metadata.get("experiment") not in experiment_filter:
            continue
        if prompt_filter and row.get("prompt_name") not in prompt_filter:
            continue
        selected.append(row)

    if args.start:
        selected = selected[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def load_existing_sample_ids(output_path: Path) -> set[str]:
    """Load existing output sample ids for resume mode.

    Args:
        output_path: Existing output JSONL path.

    Returns:
        Existing sample ids.
    """
    if not output_path.exists():
        return set()
    sample_ids = set()
    for row in read_jsonl(output_path):
        sample_id = row.get("sample_id")
        if isinstance(sample_id, str):
            sample_ids.add(sample_id)
    return sample_ids


def make_sample_id(row: dict[str, Any]) -> str:
    """Build a stable sample id for an input row.

    Args:
        row: Judge-input row.

    Returns:
        Stable sample id.
    """
    experiment = row.get("analysis_metadata", {}).get("experiment", "unknown")
    return f"{row.get('dataset')}:{row.get('request_id')}:{experiment}"


def validate_score_coverage(
    input_rows: list[dict[str, Any]], output_rows: list[dict[str, Any]]
) -> None:
    """Require one valid scored output for every selected judge input.

    Args:
        input_rows: Selected judge-input rows.
        output_rows: Existing judge-output rows, including resumed rows.

    Raises:
        ValueError: If an ID is missing or duplicated, echoed input differs,
            or any selected row lacks a finite score in the unit interval.
    """
    expected: dict[str, dict[str, Any]] = {}
    for row in input_rows:
        sample_id = make_sample_id(row)
        if sample_id in expected:
            raise ValueError(f"Duplicate selected input ID: {sample_id}")
        expected[sample_id] = row
    actual: dict[str, dict[str, Any]] = {}
    for row in output_rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in expected:
            continue
        if sample_id in actual:
            raise ValueError(f"Duplicate output ID: {sample_id}")
        actual[sample_id] = row
    missing = expected.keys() - actual.keys()
    if missing:
        raise ValueError(
            f"Missing {len(missing)} selected scores: {sorted(missing)[:5]}"
        )
    invalid = []
    for sample_id, output in actual.items():
        source = expected[sample_id]
        parsed = output.get("judge_output_parsed")
        score = parsed.get("semantic_score") if isinstance(parsed, dict) else None
        if (
            output.get("dataset") != source.get("dataset")
            or output.get("request_id") != source.get("request_id")
            or output.get("task_type") != source.get("task_type")
            or output.get("prompt_name") != source.get("prompt_name")
            or output.get("input_fields") != source.get("judge_input")
            or output.get("analysis_metadata") != source.get("analysis_metadata")
            or output.get("parse_error") is not None
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            invalid.append(sample_id)
    if invalid:
        raise ValueError(
            f"Invalid or unparsed selected scores: {len(invalid)}; "
            f"examples={invalid[:5]}"
        )


def make_output_row(
    row: dict[str, Any],
    backend_name: str,
    model_name: str,
    raw_output: Optional[str],
    parsed_output: Optional[dict[str, Any]],
    parse_error: Optional[str],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Build one judge-output audit row.

    Args:
        row: Original judge-input row.
        backend_name: Local backend name.
        model_name: Judge model name/path.
        raw_output: Raw model output.
        parsed_output: Parsed model JSON output.
        parse_error: Parse or validation error, if any.
        diagnostics: Deterministic diagnostics.

    Returns:
        JSON-serializable audit row.
    """
    return {
        "sample_id": make_sample_id(row),
        "dataset": row.get("dataset"),
        "request_id": row.get("request_id"),
        "task_type": row.get("task_type"),
        "judge_model": model_name,
        "judge_backend": backend_name,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "prompt_name": row.get("prompt_name"),
        "input_fields": row.get("judge_input", {}),
        "analysis_metadata": row.get("analysis_metadata", {}),
        "rule_diagnostics": diagnostics,
        "judge_output_raw": raw_output,
        "judge_output_parsed": parsed_output,
        "parse_error": parse_error,
    }


def parse_judge_output(
    raw_output: str,
    prompt_name: str,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Parse and validate raw judge output.

    Args:
        raw_output: Raw model output.
        prompt_name: Prompt identifier.

    Returns:
        ``(parsed_output, parse_error)``.
    """
    try:
        parsed = json.loads(extract_json_object(raw_output))
    except (json.JSONDecodeError, ValueError) as exc:
        return None, f"json_parse_error: {exc}"
    schema_errors = validate_output_schema(
        parsed,
        output_schema_for_prompt(prompt_name),
    )
    if schema_errors:
        return parsed, "schema_error: " + "; ".join(schema_errors)
    return parsed, None


def parse_prometheus_output(
    raw_output: str,
    task_type: str,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Parse Prometheus-native feedback plus ``[RESULT]`` score.

    Args:
        raw_output: Raw Prometheus output.
        task_type: Judge task type.

    Returns:
        ``(normalized_output, parse_error)``.
    """
    score = extract_prometheus_score(raw_output)
    if score is None:
        return None, "prometheus_parse_error: missing [RESULT] score"
    semantic_score = {
        5: 1.0,
        4: 0.75,
        3: 0.5,
        2: 0.25,
        1: 0.0,
    }[score]
    label = {
        5: "correct",
        4: "mostly_correct",
        3: "partially_correct",
        2: "minimally_correct",
        1: "no_usable_summary" if task_type == "summarization" else "incorrect",
    }[score]
    return {
        "semantic_label": label,
        "semantic_score": semantic_score,
        "native_prometheus_score": score,
        "brief_rationale": strip_prometheus_feedback(raw_output),
    }, None


def extract_prometheus_score(raw_output: str) -> Optional[int]:
    """Extract a Prometheus integer score.

    Args:
        raw_output: Raw Prometheus output.

    Returns:
        Score from 1 to 5, or ``None``.
    """
    matches = re.findall(r"\[RESULT\]\s*([1-5])\b", raw_output)
    if matches:
        return int(matches[-1])
    fallback = re.findall(r"(?:^|\n)\s*(?:score|Score)\s*:\s*([1-5])\b", raw_output)
    return int(fallback[-1]) if fallback else None


def strip_prometheus_feedback(raw_output: str) -> str:
    """Return compact feedback without the Prometheus score marker.

    Args:
        raw_output: Raw Prometheus output.

    Returns:
        Feedback capped to 500 characters.
    """
    feedback = raw_output.split("[RESULT]", maxsplit=1)[0]
    feedback = feedback.replace("Feedback:", "", 1).strip()
    return feedback[:500]


def resolve_prompt_style(requested_style: str, model_name: str) -> str:
    """Resolve prompt style for the model.

    Args:
        requested_style: CLI prompt style.
        model_name: Judge model name/path.

    Returns:
        ``"json"`` or ``"prometheus"``.
    """
    if requested_style != "auto":
        return requested_style
    if "prometheus" in model_name.lower():
        return "prometheus"
    return "json"


def extract_json_object(raw_output: str) -> str:
    """Extract the first JSON object from model output.

    Args:
        raw_output: Raw model output.

    Returns:
        JSON object string.

    Raises:
        ValueError: If no balanced JSON object is found.
    """
    text = raw_output.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("unterminated JSON object")


def validate_output_schema(
    parsed: Any,
    schema: dict[str, Any],
) -> list[str]:
    """Validate parsed judge JSON against the expected schema subset.

    Args:
        parsed: Parsed judge output.
        schema: JSON-schema-like expected output schema.

    Returns:
        Human-readable schema errors.
    """
    if not isinstance(parsed, dict):
        return ["judge output is not an object"]

    errors = []
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    for field in required:
        if field not in parsed:
            errors.append(f"missing {field}")

    if schema.get("additionalProperties") is False:
        extras = sorted(set(parsed) - set(properties))
        if extras:
            errors.append(f"unexpected fields {extras}")

    for field, value in parsed.items():
        spec = properties.get(field)
        if spec is None:
            continue
        expected_type = spec.get("type")
        if expected_type and not value_matches_type(value, expected_type):
            errors.append(f"{field} has wrong type")
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"{field} has invalid value {value!r}")
        max_length = spec.get("maxLength")
        if isinstance(max_length, int) and isinstance(value, str):
            if len(value) > max_length:
                errors.append(f"{field} exceeds maxLength {max_length}")
    return errors


def value_matches_type(value: Any, expected_type: str) -> bool:
    """Return whether a JSON value matches a schema type.

    Args:
        value: JSON value.
        expected_type: JSON schema type name.

    Returns:
        Whether the value matches.
    """
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return True


def get_json(url: str, timeout: float) -> dict[str, Any]:
    """GET JSON from a local backend.

    Args:
        url: URL to request.
        timeout: Timeout in seconds.

    Returns:
        Parsed JSON response.

    Raises:
        RuntimeError: If the request fails.
    """
    try:
        with request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"GET failed for {url}: {exc}") from exc


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """POST JSON to a local backend.

    Args:
        url: URL to request.
        payload: JSON-serializable payload.
        timeout: Timeout in seconds.

    Returns:
        Parsed JSON response.

    Raises:
        RuntimeError: If the request fails.
    """
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    req = request.Request(url, data=data, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST failed for {url}: {exc.code} {body}") from exc
    except (OSError, error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"POST failed for {url}: {exc}") from exc


def run(args: argparse.Namespace) -> int:
    """Run the local judge pipeline.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Process exit code.
    """
    rows = selected_rows(read_jsonl(args.input), args)
    if not rows:
        print("No rows selected.", file=sys.stderr)
        return 1

    if args.dry_run:
        for row in rows:
            print_dry_run_row(row, args)
        return 0

    model_name = resolve_model_name(args.backend, args.model)
    backend: Optional[LocalJudgeBackend] = None
    if not args.check_only:
        backend = build_backend(args)
        model_name = backend.model

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing_sample_ids = (
        load_existing_sample_ids(args.output) if args.resume else set()
    )
    mode = "a" if args.resume and args.output.exists() else "w"
    rows_written = 0
    with args.output.open(mode, encoding="utf-8") as output_file:
        for row in rows:
            sample_id = make_sample_id(row)
            if sample_id in existing_sample_ids:
                continue
            output_row = judge_or_check_row(
                row,
                args,
                backend,
                model_name,
            )
            output_file.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            output_file.flush()
            rows_written += 1
            print(f"Wrote {sample_id}", file=sys.stderr)

    print(f"Wrote {rows_written} rows to {args.output}", file=sys.stderr)
    if args.require_complete_scores:
        try:
            validate_score_coverage(rows, read_jsonl(args.output))
        except ValueError as exc:
            print(f"Judge output incomplete: {exc}", file=sys.stderr)
            return 1
        print(f"Verified {len(rows)} complete scored rows.", file=sys.stderr)
    return 0


def judge_or_check_row(
    row: dict[str, Any],
    args: argparse.Namespace,
    backend: Optional[LocalJudgeBackend],
    model_name: str,
) -> dict[str, Any]:
    """Judge one row or write a check-only audit row.

    Args:
        row: Judge-input row.
        args: Parsed CLI arguments.
        backend: Local judge backend, or ``None`` in check-only mode.
        model_name: Model name/path for audit output.

    Returns:
        Judge-output audit row.
    """
    input_fields = row.get("judge_input", {})
    diagnostics = compute_rule_diagnostics(
        str(input_fields.get("prediction", "")),
        row.get("analysis_metadata", {}),
        args.max_token_threshold,
    )
    if args.check_only:
        return make_output_row(
            row,
            "check-only",
            model_name,
            None,
            None,
            "check_only_no_model",
            diagnostics,
        )

    assert backend is not None
    prompt = get_prompt(str(row["prompt_name"]))
    prompt_style = resolve_prompt_style(args.prompt_style, backend.model)
    if prompt_style == "prometheus":
        system_prompt, user_prompt = render_prometheus_prompt(
            str(row.get("task_type", "")),
            input_fields,
            prompt.name,
        )
    else:
        system_prompt = prompt.system_prompt
        user_prompt = render_user_prompt(prompt, input_fields)

    raw_output = backend.generate(system_prompt, user_prompt)
    if prompt_style == "prometheus":
        parsed_output, parse_error = parse_prometheus_output(
            raw_output,
            str(row.get("task_type", "")),
        )
    else:
        parsed_output, parse_error = parse_judge_output(raw_output, prompt.name)
    return make_output_row(
        row,
        backend.name,
        backend.model,
        raw_output,
        parsed_output,
        parse_error,
        diagnostics,
    )


def print_dry_run_row(row: dict[str, Any], args: argparse.Namespace) -> None:
    """Print rendered prompt and diagnostics without judging.

    Args:
        row: Judge-input row.
        args: Parsed CLI arguments.
    """
    prompt = get_prompt(str(row["prompt_name"]))
    input_fields = row.get("judge_input", {})
    diagnostics = compute_rule_diagnostics(
        str(input_fields.get("prediction", "")),
        row.get("analysis_metadata", {}),
        args.max_token_threshold,
    )
    model_name = resolve_model_name(args.backend, args.model)
    prompt_style = resolve_prompt_style(args.prompt_style, model_name)
    if prompt_style == "prometheus":
        system_prompt, user_prompt = render_prometheus_prompt(
            str(row.get("task_type", "")),
            input_fields,
            prompt.name,
        )
    else:
        system_prompt = prompt.system_prompt
        user_prompt = render_user_prompt(prompt, input_fields)
    print("=" * 80)
    print(f"sample_id: {make_sample_id(row)}")
    print(f"prompt_name: {prompt.name}")
    print(f"prompt_style: {prompt_style}")
    print("diagnostics:")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print("system_prompt:")
    print(system_prompt)
    print("user_prompt:")
    print(user_prompt)


def main() -> None:
    """Run the CLI entrypoint."""
    try:
        raise SystemExit(run(parse_args()))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
