# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.v1.compute.attention.abstract import AttentionInterface


def infer_attn_backend_from_vllm(
    vllm_attn,
    enable_sparse: bool = False,
    selected_attention_backend: str = "flash_attn",
) -> AttentionInterface:
    attn_name = type(vllm_attn.impl).__name__
    if attn_name == "FlashInferImpl" and enable_sparse:
        # Local
        from .flash_infer_sparse import LMCFlashInferSparseBackend

        return LMCFlashInferSparseBackend(vllm_attn)
    elif attn_name == "FlashAttentionImpl" and not enable_sparse:
        # Local
        from .flash_attn import LMCFlashAttnBackend

        return LMCFlashAttnBackend(vllm_attn, selected_attention_backend)
    else:
        raise ValueError(f"Attention backend {attn_name} is not supported in LMCache.")
