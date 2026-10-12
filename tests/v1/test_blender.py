# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace

# Third Party
import torch

# First Party
from lmcache.v1.compute.blend.blender import LMCBlender
from lmcache.v1.compute.blend.metadata import LMCBlendCommonMetadata, LMCBlendMetadata


class _FakeGPUConnector:
    def __init__(self, old_k: torch.Tensor, old_v: torch.Tensor) -> None:
        self.old_k = old_k
        self.old_v = old_v

    def get_kv(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.old_k, self.old_v


class _FakeAttentionMetadata:
    def update_from_top_indices(self, top_indices: torch.Tensor) -> None:
        raise AssertionError("pre-check layers should not update attention metadata")


def _identity_rotary_emb(
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return q, k


def test_process_qkv_persists_full_precheck_kv() -> None:
    token_count = 4
    hidden_size = 3
    q = torch.arange(token_count * hidden_size, dtype=torch.float32).reshape(
        token_count, hidden_size
    )
    k = q + 100
    v = q + 200
    residual = q + 300
    old_k = torch.zeros_like(k)
    old_v = torch.zeros_like(v)
    attn_output = torch.empty_like(q)

    blender = LMCBlender.__new__(LMCBlender)
    blender.gpu_connector = _FakeGPUConnector(old_k, old_v)
    blender.common_metadata = LMCBlendCommonMetadata(
        check_layers=[1],
        recomp_ratios=[1.0],
    )
    blender.metadata = LMCBlendMetadata()
    blender.layerwise_model = SimpleNamespace(
        vllm_model=SimpleNamespace(
            model=SimpleNamespace(
                layers=[
                    SimpleNamespace(
                        self_attn=SimpleNamespace(rotary_emb=_identity_rotary_emb)
                    )
                ]
            )
        )
    )

    _, returned_k, returned_v, _, _, _ = blender.process_qkv(
        q,
        k,
        v,
        residual,
        layer_id=0,
        attn_output=attn_output,
        attn_metadata=_FakeAttentionMetadata(),
    )

    assert returned_k is old_k
    assert returned_v is old_v
    assert torch.equal(old_k, k)
    assert torch.equal(old_v, v)
