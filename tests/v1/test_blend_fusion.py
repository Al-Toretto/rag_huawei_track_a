# SPDX-License-Identifier: Apache-2.0
# Third Party
import torch

# First Party
from lmcache.v1.compute.blend.fusion import (
    LMCFusionInputs,
    RankAdaptiveWeightedSelectedFusion,
    build_fusion_strategy,
)


def test_rank_adaptive_fusion_uses_deviation_rank_for_cached_weights() -> None:
    selected_indices = torch.tensor([0, 1, 2])
    cached_k = torch.tensor([[0.0], [0.0], [0.0], [7.0]])
    cached_v = torch.tensor([[0.0], [0.0], [0.0], [8.0]])
    selected_k = torch.tensor([[0.5], [1.0], [2.0]])
    selected_v = torch.tensor([[2.0], [1.0], [0.5]])
    fusion = RankAdaptiveWeightedSelectedFusion(
        min_cached_k_weight=0.0,
        max_cached_k_weight=0.2,
        min_cached_v_weight=0.0,
        max_cached_v_weight=0.2,
    )

    fused_k, fused_v = fusion.fuse(
        LMCFusionInputs(
            selected_indices=selected_indices,
            selected_k=selected_k,
            selected_v=selected_v,
            cached_k=cached_k,
            cached_v=cached_v,
        )
    )

    expected_k = torch.tensor([[0.4], [0.9], [2.0], [7.0]])
    expected_v = torch.tensor([[2.0], [0.9], [0.4], [8.0]])
    assert torch.allclose(fused_k, expected_k)
    assert torch.allclose(fused_v, expected_v)


def test_rank_adaptive_fusion_treats_single_selected_token_as_high_deviation() -> None:
    fusion = RankAdaptiveWeightedSelectedFusion(
        min_cached_k_weight=0.25,
        max_cached_k_weight=0.75,
        min_cached_v_weight=0.25,
        max_cached_v_weight=0.75,
    )

    fused_k, fused_v = fusion.fuse(
        LMCFusionInputs(
            selected_indices=torch.tensor([0]),
            selected_k=torch.tensor([[4.0]]),
            selected_v=torch.tensor([[8.0]]),
            cached_k=torch.tensor([[0.0]]),
            cached_v=torch.tensor([[0.0]]),
        )
    )

    assert torch.allclose(fused_k, torch.tensor([[3.0]]))
    assert torch.allclose(fused_v, torch.tensor([[6.0]]))


def test_build_fusion_strategy_supports_rank_adaptive_fusion() -> None:
    strategy = build_fusion_strategy(
        strategy_name="rank_adaptive_weighted_selected",
        cached_k_weight=0.0,
        cached_v_weight=0.0,
        min_cached_k_weight=0.0,
        max_cached_k_weight=0.05,
        min_cached_v_weight=0.0,
        max_cached_v_weight=0.20,
    )

    assert isinstance(strategy, RankAdaptiveWeightedSelectedFusion)
