# vLLM 0.10.0 compatibility patch

This patch preserves the model-registration order used by the Track A experiments.
The underlying model must be registered before the LMCache connector constructs its
CacheBlend state. It moves KV-transfer initialization until after model loading and
registers the loaded model with LMCache first.

The upstream file is from [vLLM v0.10.0](https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/v1/worker/gpu_worker.py).
The upstream code and the correction use the Apache-2.0 license. The adjacent unified
diff shows the complete three changes.

## Installation

Install the modified LMCache checkout and the pinned vLLM 0.10.0 serving environment
first. From this repository, using the same Python environment that will run vLLM:

```bash
python tools/phase2/apply_vllm_patch.py --check
python tools/phase2/apply_vllm_patch.py
python tools/phase2/apply_vllm_patch.py --check
```

The helper reads distribution metadata without importing vLLM or Torch and does not
require a GPU. It rejects other vLLM versions and any worker file that differs from
the two checksums below. `--check` writes nothing. A second application reports
`already-applied` and leaves the worker and backup unchanged.

Before replacement, the original file is saved alongside the worker as
`gpu_worker.py.phase2-vllm-0.10.0.bak`. Replacement is atomic and preserves file
permissions. An existing backup must contain the verified original source; the
helper refuses to overwrite a different backup. Reinstalling vLLM can remove the
correction, so run the check again after reinstalling it.

## Offline inspection

To inspect a downloaded upstream source without an installed serving environment:

```bash
python tools/phase2/apply_vllm_patch.py --check --worker-file /path/to/gpu_worker.py
```

An explicit file is still validated by the exact source checksums. An optional
`--vllm-version 0.10.0` asserts its version; any other value is rejected. Omit
`--check` only when intending to modify that explicit file.

| Source | SHA256 |
| --- | --- |
| Upstream vLLM 0.10.0 | `37c953ac5801d1647327f0b13a61831893a75c97ebc65e7e122f7968817fd8d0` |
| Tested Track A correction | `eb5ea8148852adfba5e165fbd4af8b1a10501bae7b5883022c3c180a0d15371e` |

## Tests

```bash
python -m pytest -q --confcutdir=tests/benchmarks tests/benchmarks/test_phase2_vllm_patch.py
```

These tests use an embedded, compressed copy of the upstream worker as an offline
fixture. They verify applicability, exact output, backup preservation, repeat
application and rejection of unsupported versions or changed content. They never
patch an installed worker or start a model.
