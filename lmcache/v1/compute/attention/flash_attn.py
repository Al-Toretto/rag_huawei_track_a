# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING

# Third Party
from vllm.attention import Attention
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
from vllm.vllm_flash_attn import flash_attn_varlen_func, get_scheduler_metadata
import torch

# First Party
from lmcache.v1.compute.attention.abstract import AttentionInterface
from lmcache.v1.compute.attention.metadata import LMCFlashAttnMetadata

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.compute.attention.metadata import LMCAttnMetadata


class LMCFlashAttnBackend(AttentionInterface):
    """
    FlashAttention backend for LMCache.
    This backend uses the FlashAttention implementation
    for efficient attention computation.
    """

    def __init__(
        self,
        vllm_attn: Attention,
        selected_attention_backend: str = "flash_attn",
    ) -> None:
        self.vllm_attn = vllm_attn
        self.vllm_attn_impl: FlashAttentionImpl = vllm_attn.impl
        self.selected_attention_backend = selected_attention_backend.lower()
        supported_selected_backends = {"flash_attn", "xformer", "xformers"}
        if self.selected_attention_backend not in supported_selected_backends:
            raise ValueError(
                "Unsupported selected-token attention backend: "
                f"{selected_attention_backend}"
            )

        # TODO(Jiayi): remove this hardcode
        self.aot_schedule = False

        idx = torch.cuda.current_device()
        self.device = torch.device(f"cuda:{idx}")

    def forward_contiguous(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: "LMCAttnMetadata",
        **kwargs,
    ) -> torch.Tensor:
        assert isinstance(attn_metadata, LMCFlashAttnMetadata)

        selected_positions = attn_metadata.selected_query_positions
        if (
            selected_positions is not None
            and len(selected_positions) != key.shape[0]
        ):
            if self.selected_attention_backend in {"xformer", "xformers"}:
                return self._forward_selected_positions_xformers(
                    query,
                    key,
                    value,
                    output,
                    selected_positions,
                )
            return self._forward_selected_positions(
                query,
                key,
                value,
                output,
                selected_positions,
            )

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        cu_seqlens_k = attn_metadata.cu_seqlens_k
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len

        descale_shape = (cu_seqlens_q.shape[0] - 1, key.shape[1])

        # TODO(Jiayi): Figure out how to use aot_schedule.
        scheduler_metadata = self._schedule(
            batch_size=1,  # NOTE(Jiayi): Assuming batch size is 1,
            # since we are processing request by request.
            cu_query_lens=cu_seqlens_q,
            max_query_len=max_seqlen_q,
            seqlens=seqused_k,
            max_seq_len=max_seqlen_k,
            causal=True,  # Assuming causal attention
        )

        flash_attn_varlen_func(
            q=query,  # contiguous
            k=key,  # contiguous
            v=value,  # contiguous
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_k,
            # seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.vllm_attn_impl.scale,
            causal=True,
            alibi_slopes=self.vllm_attn_impl.alibi_slopes,
            window_size=self.vllm_attn_impl.sliding_window,
            block_table=None,
            softcap=self.vllm_attn_impl.logits_soft_cap,
            scheduler_metadata=scheduler_metadata,
            fa_version=self.vllm_attn_impl.vllm_flash_attn_version,
            q_descale=self.vllm_attn._q_scale.expand(descale_shape),
            k_descale=self.vllm_attn._k_scale.expand(descale_shape),
            v_descale=self.vllm_attn._v_scale.expand(descale_shape),
        )

        return output

    def _forward_selected_positions_xformers(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        selected_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Run sparse-position causal attention with an xFormers additive mask."""
        try:
            # Third Party
            import xformers.ops as xops
        except ImportError as exc:
            raise ImportError(
                "blend_selected_attention_backend='xformers' requires xformers."
            ) from exc

        self._validate_xformers_selected_path()

        selected_positions = selected_positions.to(
            device=query.device,
            dtype=torch.int64,
        )

        query_heads = query.shape[1]
        kv_heads = key.shape[1]
        full_len = key.shape[0]
        num_selected = query.shape[0]
        key_positions = torch.arange(
            full_len,
            device=query.device,
            dtype=torch.int64,
        )
        blocked = key_positions.unsqueeze(0) > selected_positions.unsqueeze(1)
        padded_len = ((full_len + 7) // 8) * 8
        attn_bias_storage = torch.zeros(
            (1, 1, num_selected, padded_len),
            device=query.device,
            dtype=query.dtype,
        )
        # xFormers CUTLASS requires attn_bias.stride(-2) to be 8-aligned.
        # Slicing from a padded backing tensor preserves an aligned row stride
        # while keeping the logical mask shape [B, H, selected_q, full_k].
        attn_bias_storage[:, :, :, full_len:] = float("-inf")
        attn_bias = attn_bias_storage[:, :, :, :full_len]
        attn_bias.masked_fill_(
            blocked.unsqueeze(0).unsqueeze(0),
            float("-inf"),
        )

        if query_heads == kv_heads:
            attn_output = xops.memory_efficient_attention(
                query.contiguous().unsqueeze(0),
                key.contiguous().unsqueeze(0),
                value.contiguous().unsqueeze(0),
                attn_bias=attn_bias.expand(-1, query_heads, -1, -1),
                p=0.0,
                scale=self.vllm_attn_impl.scale,
            )
            output.copy_(attn_output.squeeze(0))
            return output

        if query_heads % kv_heads != 0:
            raise ValueError(
                "xFormers selected-token attention requires query heads to be "
                "a multiple of KV heads."
            )

        group_size = query_heads // kv_heads
        grouped_attn_bias = attn_bias.unsqueeze(2).expand(
            -1,
            kv_heads,
            group_size,
            -1,
            -1,
        )
        query_5d = query.contiguous().view(
            1,
            num_selected,
            kv_heads,
            group_size,
            query.shape[2],
        )
        key_5d = key.contiguous().view(
            1,
            full_len,
            kv_heads,
            1,
            key.shape[2],
        )
        value_5d = value.contiguous().view(
            1,
            value.shape[0],
            value.shape[1],
            1,
            value.shape[2],
        )

        attn_output = xops.memory_efficient_attention(
            query_5d,
            key_5d.expand(-1, -1, -1, group_size, -1),
            value_5d.expand(-1, -1, -1, group_size, -1),
            attn_bias=grouped_attn_bias,
            p=0.0,
            scale=self.vllm_attn_impl.scale,
        )
        output.copy_(attn_output.reshape_as(output))
        return output

    def _forward_selected_positions(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        selected_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Run causal attention for arbitrary selected original query positions."""
        selected_positions = selected_positions.to(
            device=query.device,
            dtype=torch.int64,
        )
        device = query.device
        cu_seqlens_q_dtype = torch.int32

        span_start = 0
        num_selected = len(selected_positions)
        while span_start < num_selected:
            span_end = span_start + 1
            while (
                span_end < num_selected
                and selected_positions[span_end]
                == selected_positions[span_end - 1] + 1
            ):
                span_end += 1

            prefix_len = int(selected_positions[span_end - 1].item()) + 1
            query_len = span_end - span_start

            cu_seqlens_q = torch.tensor(
                [0, query_len], dtype=cu_seqlens_q_dtype, device=device
            )
            cu_seqlens_k = torch.tensor(
                [0, prefix_len], dtype=cu_seqlens_q_dtype, device=device
            )
            seq_lens = torch.tensor([prefix_len], device=device)
            descale_shape = (1, key.shape[1])

            scheduler_metadata = self._schedule(
                batch_size=1,
                cu_query_lens=cu_seqlens_q,
                max_query_len=query_len,
                seqlens=seq_lens,
                max_seq_len=prefix_len,
                causal=True,
            )

            flash_attn_varlen_func(
                q=query[span_start:span_end],
                k=key[:prefix_len],
                v=value[:prefix_len],
                out=output[span_start:span_end],
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=query_len,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_k=prefix_len,
                softmax_scale=self.vllm_attn_impl.scale,
                causal=True,
                alibi_slopes=self.vllm_attn_impl.alibi_slopes,
                window_size=self.vllm_attn_impl.sliding_window,
                block_table=None,
                softcap=self.vllm_attn_impl.logits_soft_cap,
                scheduler_metadata=scheduler_metadata,
                fa_version=self.vllm_attn_impl.vllm_flash_attn_version,
                q_descale=self.vllm_attn._q_scale.expand(descale_shape),
                k_descale=self.vllm_attn._k_scale.expand(descale_shape),
                v_descale=self.vllm_attn._v_scale.expand(descale_shape),
            )

            span_start = span_end

        return output

    def _validate_xformers_selected_path(self) -> None:
        if self.vllm_attn_impl.alibi_slopes is not None:
            raise NotImplementedError(
                "xFormers selected-token attention does not support ALiBi yet."
            )

        sliding_window = self.vllm_attn_impl.sliding_window
        if sliding_window not in (None, -1, (-1, -1)):
            raise NotImplementedError(
                "xFormers selected-token attention does not support sliding "
                "window attention yet."
            )

        logits_soft_cap = self.vllm_attn_impl.logits_soft_cap
        if logits_soft_cap not in (None, 0.0):
            raise NotImplementedError(
                "xFormers selected-token attention does not support logits "
                "soft cap yet."
            )

    def _schedule(
        self, batch_size, cu_query_lens, max_query_len, seqlens, max_seq_len, causal
    ):
        if self.aot_schedule:
            return get_scheduler_metadata(
                batch_size=batch_size,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_seq_len,
                cache_seqlens=seqlens,
                num_heads_q=self.vllm_attn_impl.num_heads_q,
                num_heads_kv=self.vllm_attn_impl.num_heads_kv,
                headdim=self.vllm_attn_impl.headdim,
                page_size=self.vllm_attn_impl.block_size,
                cu_seqlens_q=cu_query_lens,
                causal=causal,
                window_size=self.vllm_attn_impl.aot_sliding_window,
            )
        return None

    def init_attn_metadata(
        self,
        input_ids: torch.tensor,
        **kwargs,
    ) -> LMCFlashAttnMetadata:
        seq_len = input_ids.shape[0]
        device = input_ids.device
        return LMCFlashAttnMetadata(
            query_start_loc=torch.tensor(
                [0, seq_len], dtype=torch.int32, device=device
            ),
            seq_lens=torch.tensor([seq_len], device=device),
            cu_seqlens_k=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
            max_query_len=seq_len,
            max_seq_len=seq_len,
        )
