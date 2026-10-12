# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union

# Third Party
import torch
from transformers import AutoTokenizer

# First Party
from lmcache.config import blend_default_separator
from lmcache.logging import init_logger
from lmcache.v1.compute.attention.metadata import LMCAttnMetadata
from lmcache.v1.compute.blend.fusion import (
    LMCFusionInputs,
    build_fusion_strategy,
)
from lmcache.v1.compute.blend.metadata import LMCBlendCommonMetadata, LMCBlendMetadata
from lmcache.v1.compute.blend.selection import (
    LMCSelectionInputs,
    LMCSelectionResult,
    build_selection_strategy,
)
from lmcache.v1.compute.models.utils import infer_model_from_vllm
from lmcache.v1.config import LMCacheEngineConfig

logger = init_logger(__name__)


class LMCBlender:
    """
    Cache-blender backend for LMCache.
    This backend uses the Blender implementation for efficient blending computation.
    """

    def __init__(
        self,
        cache_engine,
        gpu_connector,
        vllm_model,
        config: LMCacheEngineConfig,
    ):
        self.cache_engine = cache_engine
        self.gpu_connector = gpu_connector

        enable_sparse = False
        if config.extra_config is not None:
            enable_sparse = config.extra_config.get("enable_sparse", False)
        self.config = config
        self.selected_attention_backend = str(
            config.get_extra_config_value(
                "blend_selected_attention_backend", "flash_attn"
            )
        )

        self.layerwise_model = infer_model_from_vllm(
            vllm_model,
            self,
            enable_sparse,
            self.selected_attention_backend,
        )

        # TODO: remove this hardcode
        self.num_layers = len(vllm_model.model.layers)

        # TODO(Jiayi): support threshold-based blending
        # TODO(Jiayi): support different ratios for different layers
        # TODO(Jiayi): support "skipping blending if hit too short"
        self.common_metadata = LMCBlendCommonMetadata(
            check_layers=config.blend_check_layers,
            recomp_ratios=config.blend_recompute_ratios,
            thresholds=config.blend_thresholds,
        )
        self.selection_strategy_name = config.get_extra_config_value(
            "blend_selection_strategy", "topk_diff_k"
        )
        self.selection_k_weight = float(
            config.get_extra_config_value("blend_selection_k_weight", 1.0)
        )
        self.selection_v_weight = float(
            config.get_extra_config_value("blend_selection_v_weight", 1.0)
        )
        self.final_recompute_ratio = float(
            config.get_extra_config_value("blend_final_recompute_ratio", 0.0)
        )
        self.neighbor_expand_top_ratio = float(
            config.get_extra_config_value("blend_neighbor_expand_top_ratio", 1.0)
        )
        self.neighbor_expand_radius = int(
            config.get_extra_config_value("blend_neighbor_expand_radius", 1)
        )
        self.neighbor_expand_direction = str(
            config.get_extra_config_value("blend_neighbor_expand_direction", "both")
        )
        self.span_close_max_gap = int(
            config.get_extra_config_value("blend_span_close_max_gap", 1)
        )
        self.selection_strategy = build_selection_strategy(
            self.selection_strategy_name,
            self.selection_k_weight,
            self.selection_v_weight,
            self.final_recompute_ratio,
            self.neighbor_expand_top_ratio,
            self.neighbor_expand_radius,
            self.neighbor_expand_direction,
            self.span_close_max_gap,
        )
        self.fusion_strategy_name = config.get_extra_config_value(
            "blend_fusion_strategy", "overwrite_selected"
        )
        self.cached_k_weight = float(
            config.get_extra_config_value("blend_cached_k_weight", 0.0)
        )
        self.cached_v_weight = float(
            config.get_extra_config_value("blend_cached_v_weight", 0.0)
        )
        self.min_cached_k_weight = float(
            config.get_extra_config_value("blend_min_cached_k_weight", 0.0)
        )
        self.max_cached_k_weight = float(
            config.get_extra_config_value("blend_max_cached_k_weight", 1.0)
        )
        self.min_cached_v_weight = float(
            config.get_extra_config_value("blend_min_cached_v_weight", 0.0)
        )
        self.max_cached_v_weight = float(
            config.get_extra_config_value("blend_max_cached_v_weight", 1.0)
        )
        self.adaptive_k_gamma = float(
            config.get_extra_config_value("blend_adaptive_k_gamma", 1.0)
        )
        self.adaptive_v_gamma = float(
            config.get_extra_config_value("blend_adaptive_v_gamma", 1.0)
        )
        self.fusion_strategy = build_fusion_strategy(
            self.fusion_strategy_name,
            self.cached_k_weight,
            self.cached_v_weight,
            self.min_cached_k_weight,
            self.max_cached_k_weight,
            self.min_cached_v_weight,
            self.max_cached_v_weight,
            self.adaptive_k_gamma,
            self.adaptive_v_gamma,
        )
        self.force_chunk_boundaries = bool(
            config.get_extra_config_value("blend_force_chunk_boundary_tokens", False)
        )
        self.needs_separator_positions = self.selection_strategy_name in {
            "separator_only",
            "separator_and_segment_margin",
        }
        self.needs_segment_margin_positions = self.selection_strategy_name in {
            "segment_margin_only",
            "separator_and_segment_margin",
        }
        self.log_selection_stats = bool(
            config.get_extra_config_value("blend_log_selection_stats", False)
        )
        self.log_fusion_stats = bool(
            config.get_extra_config_value("blend_log_fusion_stats", True)
        )
        self.chunk_boundary_token_count = max(
            1,
            int(config.get_extra_config_value("blend_chunk_boundary_token_count", 1)),
        )
        self.separator_token_ids = self._build_separator_token_ids()
        self.current_req_id: Optional[str] = None

        # This will be set during the blending process
        self.metadata = LMCBlendMetadata(
            imp_indices=None,
            attn_mask=None,
            positions=None,
            chunk_boundary_positions=None,
            separator_positions=None,
        )

    def process_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual: torch.Tensor,
        layer_id: int,
        attn_output: Optional[torch.Tensor],
        attn_metadata: LMCAttnMetadata,
    ):
        logger.debug(f"Blender is processing KV for layer {layer_id}")
        old_k, old_v = self.gpu_connector.get_kv(layer_id)

        if attn_output is None:
            attn_output = torch.empty(
                q.shape,
                dtype=q.dtype,
                device=q.device,
            )

        if q.shape[0] == 0:
            if self.metadata.imp_indices is not None:
                self._log_fusion_stats(layer_id, len(self.metadata.imp_indices))
                fused_k, fused_v = self.fusion_strategy.fuse(
                    LMCFusionInputs(
                        selected_indices=self.metadata.imp_indices,
                        selected_k=k,
                        selected_v=v,
                        cached_k=old_k,
                        cached_v=old_v,
                    )
                )
                return q, fused_k, fused_v, residual, attn_output, attn_metadata
            return q, k, v, residual, attn_output, attn_metadata

        # perform positional encoding
        if self.metadata.positions is None:
            self.metadata.positions = torch.arange(
                q.shape[0], device=q.device, dtype=torch.int64
            )
        layer = self.layerwise_model.vllm_model.model.layers[layer_id]
        attn_layer = layer.self_attn
        q, k = attn_layer.rotary_emb(self.metadata.positions, q, k)

        check_layers = self.common_metadata.check_layers or []
        if layer_id in check_layers:
            assert self.common_metadata.recomp_ratios is not None
            recompute_ratio = self._get_recompute_ratio_for_layer(layer_id)
            cached_k_for_selection = old_k[self.metadata.positions]
            cached_v_for_selection = old_v[self.metadata.positions]

            selection_result = self.selection_strategy.select(
                LMCSelectionInputs(
                    positions=self.metadata.positions,
                    fresh_k=k,
                    fresh_v=v,
                    cached_k=cached_k_for_selection,
                    cached_v=cached_v_for_selection,
                    recompute_ratio=recompute_ratio,
                    chunk_boundary_positions=self.metadata.chunk_boundary_positions,
                    separator_positions=self.metadata.separator_positions,
                    force_chunk_boundaries=self.force_chunk_boundaries,
                )
            )
            active_indices = selection_result.selected_indices
            selected_positions = self.metadata.positions[active_indices]

            k, v = k[active_indices], v[active_indices]
            q = q[active_indices]
            residual = residual[active_indices]

            logger.debug(f"Number of indices picked: {len(active_indices)}")
            self._log_selection_stats(layer_id, selection_result)

            self.metadata.imp_indices = selected_positions
            self.metadata.positions = selected_positions
            attn_output = attn_output[: len(active_indices)]

            attn_metadata.update_from_top_indices(selected_positions)

        if self.metadata.imp_indices is not None:
            self._log_fusion_stats(layer_id, len(self.metadata.imp_indices))
            fused_k, fused_v = self.fusion_strategy.fuse(
                LMCFusionInputs(
                    selected_indices=self.metadata.imp_indices,
                    selected_k=k,
                    selected_v=v,
                    cached_k=old_k,
                    cached_v=old_v,
                )
            )
            return q, fused_k, fused_v, residual, attn_output, attn_metadata

        if old_k.shape == k.shape and old_v.shape == v.shape:
            # Before the first check layer, all active tokens are recomputed.
            # Store those fresh K/V values so uncached separator/gap positions
            # are not left as zeros in the final paged KV cache.
            old_k.copy_(k)
            old_v.copy_(v)
            return q, old_k, old_v, residual, attn_output, attn_metadata
        else:
            return q, k, v, residual, attn_output, attn_metadata

    # NOTE(Jiayi): Exposing this `blend_layer` interface as we might
    # want to ochestrate the blending process elsewhere
    def blend_layer(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform layerwiese retrieve + blending.
        """

        # TODO(Jiayi): store is currently not included in this function

        layerwise_model_executor = self.layerwise_model.compute_layer(tokens)
        layerwise_retriever = self.cache_engine.retrieve_layer(tokens, mask, **kwargs)

        next(layerwise_retriever)
        yield

        for i in range(self.num_layers):
            next(layerwise_retriever)
            next(layerwise_model_executor)
            yield

        next(layerwise_retriever)

        self.metadata.clean()
        yield

    def blend(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform blending for the given tokens.
        """

        if isinstance(tokens, list):
            tokens = torch.tensor(tokens).cuda()
        else:
            tokens = tokens.to(device="cuda")

        if self.needs_separator_positions:
            separator_positions = self._find_separator_positions(tokens)
            self.metadata.separator_positions = separator_positions
        if self.force_chunk_boundaries or self.needs_segment_margin_positions:
            boundary_positions = self._find_chunk_boundary_positions(
                tokens,
                self.chunk_boundary_token_count,
            )
            self.metadata.chunk_boundary_positions = boundary_positions
        self.current_req_id = kwargs.get("req_id")

        layerwise_blender = self.blend_layer(tokens, mask, **kwargs)

        for i in range(self.num_layers + 2):
            next(layerwise_blender)

    def _build_separator_token_ids(self) -> torch.Tensor:
        separator = self.config.blend_special_str
        if not separator or separator == blend_default_separator:
            return torch.empty(0, dtype=torch.long)

        model_name = self.cache_engine.metadata.model_name
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not load tokenizer for blend chunk-boundary forcing from "
                "model_name=%s: %s",
                model_name,
                exc,
            )
            return torch.empty(0, dtype=torch.long)

        token_ids = tokenizer.encode(separator)
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        if token_ids and bos_token_id is not None and token_ids[0] == bos_token_id:
            token_ids = token_ids[1:]
        logger.info(
            "Blend separator tokenization ready model_name=%s separator=%r "
            "separator_token_count=%d force_chunk_boundaries=%s "
            "chunk_boundary_token_count=%d selection_strategy=%s "
            "selection_k_weight=%.3f selection_v_weight=%.3f "
            "final_recompute_ratio=%.3f neighbor_expand_top_ratio=%.3f "
            "neighbor_expand_radius=%d neighbor_expand_direction=%s "
            "span_close_max_gap=%d "
            "fusion_strategy=%s cached_k_weight=%.3f cached_v_weight=%.3f "
            "min_cached_k_weight=%.3f max_cached_k_weight=%.3f "
            "min_cached_v_weight=%.3f max_cached_v_weight=%.3f",
            model_name,
            separator,
            len(token_ids),
            self.force_chunk_boundaries,
            self.chunk_boundary_token_count,
            self.selection_strategy_name,
            self.selection_k_weight,
            self.selection_v_weight,
            self.final_recompute_ratio,
            self.neighbor_expand_top_ratio,
            self.neighbor_expand_radius,
            self.neighbor_expand_direction,
            self.span_close_max_gap,
            self.fusion_strategy_name,
            self.cached_k_weight,
            self.cached_v_weight,
            self.min_cached_k_weight,
            self.max_cached_k_weight,
            self.min_cached_v_weight,
            self.max_cached_v_weight,
        )
        return torch.tensor(token_ids, dtype=torch.long)

    def _find_chunk_boundary_positions(
        self,
        tokens: torch.Tensor,
        edge_width: int,
    ) -> Optional[torch.Tensor]:
        if self.separator_token_ids.numel() == 0:
            return None

        cpu_tokens = tokens.detach().to(device="cpu", dtype=torch.long)
        matches = self._find_separator_match_starts(cpu_tokens)

        boundaries: set[int] = set()
        start = 0
        for match_idx in matches:
            end = match_idx
            if end > start:
                self._add_chunk_edge_positions(boundaries, start, end, edge_width)
            start = match_idx + len(self.separator_token_ids)
        if start < len(cpu_tokens):
            self._add_chunk_edge_positions(
                boundaries,
                start,
                len(cpu_tokens),
                edge_width,
            )

        if not boundaries:
            return None
        return torch.tensor(sorted(boundaries), device=tokens.device, dtype=torch.int64)

    def _find_separator_positions(self, tokens: torch.Tensor) -> Optional[torch.Tensor]:
        if self.separator_token_ids.numel() == 0:
            return None

        cpu_tokens = tokens.detach().to(device="cpu", dtype=torch.long)
        matches = self._find_separator_match_starts(cpu_tokens)
        if not matches:
            return None

        separator_positions: set[int] = set()
        for match_idx in matches:
            for offset in range(len(self.separator_token_ids)):
                separator_positions.add(match_idx + offset)

        return torch.tensor(
            sorted(separator_positions),
            device=tokens.device,
            dtype=torch.int64,
        )

    def _find_separator_match_starts(self, cpu_tokens: torch.Tensor) -> list[int]:
        sep_len = len(self.separator_token_ids)
        if sep_len == 0 or len(cpu_tokens) < sep_len:
            return []

        sep_tokens = self.separator_token_ids
        windows = cpu_tokens.unfold(0, sep_len, 1)
        return (windows == sep_tokens).all(dim=1).nonzero(as_tuple=True)[0].tolist()

    def _add_chunk_edge_positions(
        self,
        boundaries: set[int],
        start: int,
        end: int,
        edge_width: int,
    ) -> None:
        chunk_len = end - start
        if chunk_len <= 0:
            return

        edge_width = min(edge_width, chunk_len)
        for offset in range(edge_width):
            boundaries.add(start + offset)
            boundaries.add(end - 1 - offset)

    def _get_recompute_ratio_for_layer(self, layer_id: int) -> float:
        check_layers = self.common_metadata.check_layers or []
        recomp_ratios = self.common_metadata.recomp_ratios or []
        if not recomp_ratios:
            raise ValueError("blend_recompute_ratios must be set for check layers")

        check_layer_index = check_layers.index(layer_id)
        ratio_index = min(check_layer_index, len(recomp_ratios) - 1)
        return recomp_ratios[ratio_index]

    def _log_selection_stats(
        self, layer_id: int, selection_result: LMCSelectionResult
    ) -> None:
        if not self.log_selection_stats:
            return

        active_tokens = (
            len(self.metadata.positions) if self.metadata.positions is not None else 0
        )
        topk_count = len(selection_result.topk_indices)
        boundary_count = len(selection_result.boundary_indices)
        selected_count = len(selection_result.selected_indices)

        overlap_count = 0
        if topk_count > 0 and boundary_count > 0:
            overlap_count = int(
                torch.isin(
                    selection_result.boundary_indices,
                    selection_result.topk_indices,
                ).sum().item()
            )

        detected_boundary_positions = (
            0
            if self.metadata.chunk_boundary_positions is None
            else len(self.metadata.chunk_boundary_positions)
        )
        detected_separator_positions = (
            0
            if self.metadata.separator_positions is None
            else len(self.metadata.separator_positions)
        )
        added_boundary_count = selected_count - topk_count

        logger.info(
            "Blend selection stats req_id=%s layer=%d active_tokens=%d "
            "topk_count=%d detected_boundary_positions=%d "
            "detected_separator_positions=%d "
            "active_boundary_count=%d overlap_with_topk=%d "
            "added_boundary_count=%d final_selected_count=%d",
            self.current_req_id or "unknown",
            layer_id,
            active_tokens,
            topk_count,
            detected_boundary_positions,
            detected_separator_positions,
            boundary_count,
            overlap_count,
            added_boundary_count,
            selected_count,
        )

    def _log_fusion_stats(self, layer_id: int, selected_count: int) -> None:
        if not self.log_fusion_stats:
            return

        logger.info(
            "Blend fusion stats req_id=%s layer=%d strategy=%s "
            "selected_count=%d cached_k_weight=%.3f cached_v_weight=%.3f "
            "min_cached_k_weight=%.3f max_cached_k_weight=%.3f "
            "min_cached_v_weight=%.3f max_cached_v_weight=%.3f",
            self.current_req_id or "unknown",
            layer_id,
            self.fusion_strategy_name,
            selected_count,
            self.cached_k_weight,
            self.cached_v_weight,
            self.min_cached_k_weight,
            self.max_cached_k_weight,
            self.min_cached_v_weight,
            self.max_cached_v_weight,
        )
