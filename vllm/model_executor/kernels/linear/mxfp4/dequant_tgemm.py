# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 linear kernel: one-time dequant at load, then aiter tuned BF16 GEMM.

Trade-off vs EmulationMxfp4LinearKernel: weights resident at BF16 (4x memory)
but decode drops the per-token full-matrix dequant pass. Requires TP>=2 for
27B-class models (weight copy no longer fits one 31GiB card).
Falls back to emulation when aiter tgemm is unavailable.
"""

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    dequant_mxfp4,
    quant_dequant_mxfp4,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Dynamic,
)
from vllm.model_executor.layers.quantization.utils.mxfp6_utils import (
    quant_dequant_mxfp6,
)
from vllm.platforms import current_platform

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig

logger = init_logger(__name__)

_ACT_QDQ = {
    kMxfp4Dynamic: quant_dequant_mxfp4,
}


class DequantAiterTgemmMxfp4LinearKernel(MxFp4LinearKernel):
    """Load-time dequant + aiter tuned BF16 GEMM (fast path, 4x memory)."""

    def __init__(self, config: MxFp4LinearLayerConfig) -> None:
        super().__init__(config)
        if config.activation_quant_key is None:
            self.quant_dequant_func = lambda x: x
        elif config.activation_quant_key == kMxfp4Dynamic:
            self.quant_dequant_func = quant_dequant_mxfp4
        else:
            self.quant_dequant_func = quant_dequant_mxfp6

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "requires ROCm (aiter tgemm)"
        try:
            from aiter.tuned_gemm import tgemm  # noqa: F401
        except ImportError:
            return False, "aiter tuned_gemm not available"
        return True, None

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        if config.activation_quant_key not in (None, kMxfp4Dynamic):
            return False, "only supports unquantized or MXFP4-dynamic activations"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # One-time dequant: packed fp4 [N, K/2] + e8m0 scales -> BF16 [N, K].
        # This is the only dequant for the layer's lifetime.
        dq = dequant_mxfp4(layer.weight.data, layer.weight_scale.data,
                           torch.bfloat16).contiguous()
        layer.weight = Parameter(dq, requires_grad=False)
        # scales no longer needed; free the reference.
        layer.weight_scale = Parameter(
            torch.empty(0, device=dq.device, dtype=torch.uint8),
            requires_grad=False)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qdq_x = self.quant_dequant_func(x)
        from aiter.tuned_gemm import tgemm
        return tgemm.mm(qdq_x, layer.weight, bias)
