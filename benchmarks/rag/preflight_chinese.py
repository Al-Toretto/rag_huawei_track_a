# SPDX-License-Identifier: Apache-2.0
"""Check Chinese RAG prompt lengths and cache capacity before cluster runs."""

# Standard
from __future__ import annotations

from pathlib import Path
import argparse
import math

# Third Party
import jieba  # noqa: F401 - fail before submission if Chinese scoring is unavailable.
from transformers import AutoConfig, AutoTokenizer

# Local
from utils import (
    ChatTemplateMode,
    PromptBuildMethodType,
    build_qa_prompt,
    build_rag_prompt_token_ids,
    load_dataset,
    resolve_prompt_texts,
    tokenize_segment_for_cache,
)


DATASETS = (
    ("multifieldqa_zh_longbench", 256),
    ("dureader_longbench", 1024),
)


def main() -> None:
    """Print capacity estimates and fail if the intended 50 cases cannot fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=30000)
    parser.add_argument("--kv-storage-gib", type=int, default=320)
    parser.add_argument("--kv-chunk-size", type=int, default=256)
    parser.add_argument(
        "--kv-bytes-per-token",
        type=int,
        default=None,
        help="Override model-config KV size when only the tokenizer is cached locally.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_snapshot, local_files_only=True
    )
    if args.kv_bytes_per_token is None:
        config = AutoConfig.from_pretrained(args.model_snapshot, local_files_only=True)
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        kv_bytes_per_token = (
            config.num_key_value_heads * head_dim * config.num_hidden_layers * 2 * 2
        )
    else:
        kv_bytes_per_token = args.kv_bytes_per_token
    rag_dir = Path(__file__).resolve().parent
    failures: list[str] = []
    for stem, max_output_tokens in DATASETS:
        dataset_path = rag_dir / "inputs" / f"{stem}.json"
        examples = load_dataset(str(dataset_path))
        if len(examples) != 50:
            failures.append(f"{stem}: expected 50 examples, found {len(examples)}")
        system_prompt, query_prompt = resolve_prompt_texts(
            str(dataset_path), PromptBuildMethodType.QA
        )
        unique_docs: set[str] = set()
        max_prompt_tokens = 0
        max_doc_tokens = 0
        for example in examples:
            prompt_ids, _ = build_rag_prompt_token_ids(
                tokenizer,
                system_prompt,
                example,
                query_prompt,
                " # # ",
                PromptBuildMethodType.QA,
                ChatTemplateMode.QWEN_NON_THINKING,
            )
            max_prompt_tokens = max(max_prompt_tokens, len(prompt_ids))
            docs, _ = build_qa_prompt(example, "")
            unique_docs.update(docs)
        rounded_tokens = 0
        for document in unique_docs:
            token_count = len(tokenize_segment_for_cache(tokenizer, document))
            max_doc_tokens = max(max_doc_tokens, token_count)
            rounded_tokens += (
                math.ceil(token_count / args.kv_chunk_size) * args.kv_chunk_size
            )
        unique_kv_gib = rounded_tokens * kv_bytes_per_token / 1024**3
        print(
            f"{stem}: examples={len(examples)} unique_docs={len(unique_docs)} "
            f"max_prompt_tokens={max_prompt_tokens} max_doc_tokens={max_doc_tokens} "
            f"output_cap={max_output_tokens} "
            f"estimated_unique_kv_gib={unique_kv_gib:.2f}"
        )
        if max_prompt_tokens + max_output_tokens > args.max_model_len:
            failures.append(
                f"{stem}: prompt plus output cap exceeds {args.max_model_len} tokens"
            )
        if max_doc_tokens > args.max_model_len:
            failures.append(f"{stem}: a cached document exceeds model length")
        if unique_kv_gib > args.kv_storage_gib * 0.9:
            failures.append(
                f"{stem}: unique document KV exceeds 90% of the "
                f"{args.kv_storage_gib} GiB cache budget"
            )
    if failures:
        parser.exit(1, "Preflight failed:\n" + "\n".join(failures) + "\n")
    print("Preflight passed. Confirm the first smoke run's cache statistics on H200.")


if __name__ == "__main__":
    main()
