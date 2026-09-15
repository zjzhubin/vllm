# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""HIP skinny MXFP4 kernel for GFX1X (gfx1201).

Hybrid dispatch, mirroring RDNAHybridW4A16LinearKernel:
  M <= MAX_SKINNY_BATCH_SIZE and K*M fits LDS:
      wvSplitK_mxfp4 — HIP kernel consuming packed fp4 weights directly
      (zero dequantization, zero copies; e8m0 scales applied in-kernel).
  else:
      EmulationMxfp4LinearKernel path — per-weight-matrix dequant + BF16
      GEMM. For large M the dequant cost is amortized across the batch and
      tensor-core GEMM saturates the CUs, so this is the high-throughput arm.
"""

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.platforms import current_platform

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig
from .emulation import EmulationMxfp4LinearKernel

logger = init_logger(__name__)

MAX_SKINNY_BATCH_SIZE = 5
# 64 KiB per-workgroup LDS limit in bf16 elements (matches int4 skinny op).
LDS_CAPACITY_ELEMENTS = 64 * 1024 // 2


class HipSkinnyMxfp4LinearKernel(MxFp4LinearKernel):
    """MXFP4 skinny GEMM: HIP direct-packed path for decode, emulation for
    large-M prefill. Routed ahead of Emulation in the ROCm candidate list."""

    def __init__(self, config: MxFp4LinearLayerConfig) -> None:
        super().__init__(config)
        self._fallback = EmulationMxfp4LinearKernel(config)

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "HipSkinnyMxfp4LinearKernel only targets ROCm"
        try:
            import vllm._rocm_C  # noqa: F401
        except ImportError:
            return False, "vllm._rocm_C extension not available"
        return True, None

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        if config.activation_quant_key is not None:
            # Dynamic MX activations go through the emulation arm's QDQ, so
            # they work, but the skinny arm is weight-only optimized.
            pass
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Keep packed uint8 weight + uint8 e8m0 scale as-is: the skinny
        # kernel consumes the checkpoint layout directly (no repack, no
        # copies). The emulation arm also consumes this layout.
        layer.weight = torch.nn.Parameter(
            layer.weight.data, requires_grad=False
        )
        layer.weight_scale = torch.nn.Parameter(
            layer.weight_scale.data, requires_grad=False
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1])
        M = x_2d.shape[0]
        K = x_2d.shape[1]

        if M <= MAX_SKINNY_BATCH_SIZE and K * M <= LDS_CAPACITY_ELEMENTS:
            N = layer.weight.shape[0]
            out = torch.ops.vllm.wvSplitK_mxfp4(
                layer.weight, x_2d, layer.weight_scale, bias,
                self._cu_count(),
            )
            return out.t().reshape(x.shape[:-1] + (N,))

        # Large M: emulation arm (dequant amortized, BF16 tensor-core GEMM).
        return self._fallback.apply_weights(layer, x, bias)

    @staticmethod
    def _cu_count() -> int:
        from vllm.utils.platform_utils import num_compute_units

        return num_compute_units()
