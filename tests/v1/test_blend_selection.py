# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace

# Third Party
import torch

# First Party
from lmcache.v1.compute.blend.blender import LMCBlender
from lmcache.v1.compute.blend.fusion import OverwriteSelectedFusion
from lmcache.v1.compute.blend.metadata import LMCBlendCommonMetadata, LMCBlendMetadata
from lmcache.v1.compute.blend.selection import (
    LMCSelectionInputs,
    build_selection_strategy,
)


class _FakeGPUConnector:
    def __init__(self, old_k: torch.Tensor, old_v: torch.Tensor) -> None:
        self.old_k = old_k
        self.old_v = old_v

    def get_kv(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.old_k, self.old_v


class _RecordingAttentionMetadata:
    def __init__(self) -> None:
        self.top_indices: torch.Tensor | None = None

    def update_from_top_indices(self, top_indices: torch.Tensor) -> None:
        self.top_indices = top_indices


def _identity_rotary_emb(
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return q, k


def _selection_inputs(
    separator_positions: torch.Tensor | None = None,
    chunk_boundary_positions: torch.Tensor | None = None,
) -> LMCSelectionInputs:
    token_count = 10
    hidden_size = 2
    fresh = torch.zeros((token_count, hidden_size), dtype=torch.float32)
    cached = torch.zeros_like(fresh)
    return LMCSelectionInputs(
        positions=torch.arange(token_count),
        fresh_k=fresh,
        fresh_v=fresh,
        cached_k=cached,
        cached_v=cached,
        recompute_ratio=0.0,
        chunk_boundary_positions=chunk_boundary_positions,
        separator_positions=separator_positions,
    )


def test_no_recompute_selector_selects_no_tokens() -> None:
    selector = build_selection_strategy("no_recompute", 1.0, 1.0)

    result = selector.select(_selection_inputs())

    assert result.selected_indices.tolist() == []
    assert result.topk_indices.tolist() == []


def test_separator_only_selector_selects_only_separator_tokens() -> None:
    selector = build_selection_strategy("separator_only", 1.0, 1.0)

    result = selector.select(
        _selection_inputs(separator_positions=torch.tensor([2, 3, 7]))
    )

    assert result.selected_indices.tolist() == [2, 3, 7]
    assert result.topk_indices.tolist() == []


def test_segment_margin_only_selector_selects_only_margins() -> None:
    selector = build_selection_strategy("segment_margin_only", 1.0, 1.0)

    result = selector.select(
        _selection_inputs(chunk_boundary_positions=torch.tensor([0, 1, 8, 9]))
    )

    assert result.selected_indices.tolist() == [0, 1, 8, 9]
    assert result.topk_indices.tolist() == []


def test_separator_and_segment_margin_selector_selects_union() -> None:
    selector = build_selection_strategy("separator_and_segment_margin", 1.0, 1.0)

    result = selector.select(
        _selection_inputs(
            separator_positions=torch.tensor([2, 3]),
            chunk_boundary_positions=torch.tensor([0, 1, 8, 9]),
        )
    )

    assert result.selected_indices.tolist() == [0, 1, 2, 3, 8, 9]
    assert result.topk_indices.tolist() == []


def test_topk_zero_ratio_still_selects_one_token() -> None:
    selector = build_selection_strategy("topk_diff_k", 1.0, 1.0)
    inputs = _selection_inputs()
    inputs.fresh_k[4] = 10

    result = selector.select(inputs)

    assert result.selected_indices.tolist() == [4]


def test_process_qkv_uses_matching_recompute_ratio_for_check_layer() -> None:
    token_count = 10
    active_positions = torch.tensor([1, 3, 5, 7])
    hidden_size = 2
    q = torch.arange(len(active_positions) * hidden_size, dtype=torch.float32).reshape(
        len(active_positions), hidden_size
    )
    k = q + 10
    v = q + 20
    residual = q + 30
    old_k = torch.zeros((token_count, hidden_size), dtype=torch.float32)
    old_v = torch.zeros_like(old_k)
    attn_output = torch.empty_like(q)

    blender = LMCBlender.__new__(LMCBlender)
    blender.gpu_connector = _FakeGPUConnector(old_k, old_v)
    blender.common_metadata = LMCBlendCommonMetadata(
        check_layers=[1, 5],
        recomp_ratios=[0.16, 1.0],
    )
    blender.selection_strategy = build_selection_strategy("topk_diff_k", 1.0, 1.0)
    blender.fusion_strategy = OverwriteSelectedFusion()
    blender.force_chunk_boundaries = False
    blender.log_selection_stats = False
    blender.log_fusion_stats = False
    blender.metadata = LMCBlendMetadata(positions=active_positions)
    blender.layerwise_model = SimpleNamespace(
        vllm_model=SimpleNamespace(
            model=SimpleNamespace(
                layers=[
                    SimpleNamespace(
                        self_attn=SimpleNamespace(rotary_emb=_identity_rotary_emb)
                    )
                    for _ in range(6)
                ]
            )
        )
    )

    attention_metadata = _RecordingAttentionMetadata()
    blender.process_qkv(
        q,
        k,
        v,
        residual,
        layer_id=5,
        attn_output=attn_output,
        attn_metadata=attention_metadata,
    )

    assert attention_metadata.top_indices is not None
    assert attention_metadata.top_indices.tolist() == [1, 3, 5, 7]
