# SPDX-License-Identifier: Apache-2.0
"""Apply the tested model-registration correction to vLLM 0.10.0."""

# Standard
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
import argparse
import hashlib
import os
import shutil
import sys
import tempfile

EXPECTED_VERSION = "0.10.0"
ORIGINAL_SHA256 = "37c953ac5801d1647327f0b13a61831893a75c97ebc65e7e122f7968817fd8d0"
PATCHED_SHA256 = "eb5ea8148852adfba5e165fbd4af8b1a10501bae7b5883022c3c180a0d15371e"
BACKUP_SUFFIX = ".phase2-vllm-0.10.0.bak"


def _replacement_bytes(original: bytes) -> bytes:
    replacements = (
        (
            b"from vllm.v1.worker.worker_base import WorkerBase\n",
            b"from vllm.v1.worker.worker_base import WorkerBase\n"
            b"\nfrom lmcache.integration.vllm.utils import ENGINE_NAME\n"
            b"from lmcache.v1.compute.models.utils import VLLMModelTracker\n",
        ),
        (
            b"            self.model_runner.load_model(eep_scale_up=eep_scale_up)\n",
            b"            self.model_runner.load_model(eep_scale_up=eep_scale_up)\n"
            b"\n        # CacheBlend needs the underlying vLLM model "
            b"registered before\n"
            b"        # the LMCache connector creates its blender state.\n"
            b"        VLLMModelTracker.register_model(ENGINE_NAME, "
            b"self.model_runner.model)\n"
            b"        ensure_kv_transfer_initialized(self.vllm_config)\n",
        ),
        (b"    ensure_kv_transfer_initialized(vllm_config)\n", b""),
    )
    result = original
    for before, after in replacements:
        if result.count(before) != 1:
            raise ValueError("The verified patch transformation no longer matches.")
        result = result.replace(before, after, 1)
    if hashlib.sha256(result).hexdigest() != PATCHED_SHA256:
        raise ValueError("The patch did not produce the verified target checksum.")
    return result


def installed_worker() -> tuple[Path, str]:
    """Locate the installed vLLM worker without importing vLLM or Torch.

    Returns:
        tuple[Path, str]: Worker source path and installed distribution version.

    Raises:
        metadata.PackageNotFoundError: If vLLM is not installed in this Python
            environment.
    """
    distribution = metadata.distribution("vllm")
    return (
        Path(distribution.locate_file("vllm/v1/worker/gpu_worker.py")),
        distribution.version,
    )


def patch_worker(worker_file: Path, vllm_version: str, *, check: bool = False) -> str:
    """Validate or apply the exact Phase 2 vLLM 0.10.0 worker correction.

    Args:
        worker_file (Path): Installed worker or an explicit offline fixture.
        vllm_version (str): Installed or declared vLLM version; must be 0.10.0.
        check (bool): If True, validate without creating or changing any file.

    Returns:
        str: "ready" for an applicable check, "applied" after patching, or
            "already-applied" when the exact patched source is present.

    Raises:
        ValueError: If the version, worker checksum, existing backup, or patch
            transformation does not match the verified files.
        OSError: If reading, backing up, or replacing the worker fails.

    Notes:
        Application creates a sibling backup ending in BACKUP_SUFFIX before
        atomically replacing the worker. A matching backup is reused; an
        unrelated backup is never overwritten. Already-patched files are left
        unchanged. Explicit offline fixtures are verified by the same hashes.
    """
    if vllm_version != EXPECTED_VERSION:
        raise ValueError(
            f"Expected vLLM {EXPECTED_VERSION}; found {vllm_version}. "
            "No file was changed."
        )
    worker = worker_file.resolve(strict=True)
    original = worker.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    if digest == PATCHED_SHA256:
        return "already-applied"
    if digest != ORIGINAL_SHA256:
        raise ValueError(
            f"Unexpected worker SHA256: {digest}. "
            "Only the exact upstream 0.10.0 or tested patched file is supported. "
            "No file was changed."
        )
    replacement = _replacement_bytes(original)
    if check:
        return "ready"

    backup = worker.with_name(worker.name + BACKUP_SUFFIX)
    try:
        with backup.open("xb") as file:
            file.write(original)
        shutil.copystat(worker, backup)
    except FileExistsError:
        if backup.read_bytes() != original:
            raise ValueError(
                f"Refusing to overwrite unrelated backup: {backup}"
            ) from None

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=worker.parent, prefix=f".{worker.name}.phase2-", delete=False
        ) as file:
            temporary = Path(file.name)
            file.write(replacement)
            file.flush()
            os.fsync(file.fileno())
        shutil.copystat(worker, temporary)
        if worker.read_bytes() != original:
            raise ValueError("The worker changed during patching; replacement aborted.")
        temporary.replace(worker)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return "applied"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the guarded patch command.

    Args:
        argv (Sequence[str] | None): Arguments; None uses sys.argv[1:].

    Returns:
        int: Zero for a successful check, application or already-applied result;
            one for an unsupported version/content or filesystem failure.

    Raises:
        SystemExit: If command-line arguments are invalid or help is requested.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Validate without writing any file."
    )
    parser.add_argument(
        "--worker-file", type=Path, help="Use an explicit offline worker source."
    )
    parser.add_argument(
        "--vllm-version",
        help="Version assertion for --worker-file; defaults to 0.10.0.",
    )
    args = parser.parse_args(argv)
    if args.worker_file is None and args.vllm_version is not None:
        parser.error("--vllm-version requires --worker-file")
    try:
        if args.worker_file is None:
            worker, version = installed_worker()
        else:
            worker = args.worker_file
            version = args.vllm_version or EXPECTED_VERSION
        status = patch_worker(worker, version, check=args.check)
    except (ValueError, OSError, metadata.PackageNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"{status}: {worker}")
    if status == "applied":
        print(f"backup: {worker.resolve().with_name(worker.name + BACKUP_SUFFIX)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
