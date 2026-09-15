# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 GEMV kernel: warp-per-row direct fp4 consumption for M<=4 decode.

Custom HIP kernel reading packed e2m1 weights directly (no dequant pass,
no repack, 13.1MB-class weight streams at full bandwidth), with e8m0
per-32 scales applied in-kernel via exp2. Falls back to emulation for
large M (prefill) or unsupported shapes.
"""

import os

import torch
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
from vllm.platforms import current_platform

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig

logger = init_logger(__name__)

# The alternate fix, opt-in. When set, the M-dependent dispatch below
# runs inside a registered torch.library.custom_op so dynamo cannot inline and
# constant-fold the small-M arm into the compiled graph. Unset (default) leaves the
# stock function untouched and never imports the op module.
_DISPATCH_OP = os.environ.get("VLLM_MXFP4_DISPATCH_OP", "0") == "1"
if _DISPATCH_OP:
    from .mxfp4_dispatch_op import mxfp4_linear_dispatch


class HipGemvMxfp4LinearKernel(MxFp4LinearKernel):
    """Direct fp4 GEMV for small-M decode; emulation for large-M."""

    MAX_M = 4  # gemv economics: beyond this, dequant+GEMM amortizes better

    def __init__(self, config: MxFp4LinearLayerConfig) -> None:
        super().__init__(config)
        if config.activation_quant_key == kMxfp4Dynamic:
            self.quant_dequant_func = quant_dequant_mxfp4
        else:
            self.quant_dequant_func = lambda x: x

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "requires ROCm (gfx1x HIP gemv)"
        try:
            import vllm._custom_ops as ops  # noqa: F401
        except ImportError:
            return False, "custom ops unavailable"
        return True, None

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        if config.activation_quant_key not in (None, kMxfp4Dynamic):
            return False, "only supports unquantized or MXFP4-dynamic activations"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = Parameter(layer.weight.data, requires_grad=False)
        layer.weight_scale = Parameter(
            layer.weight_scale.data, requires_grad=False)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x2 = self.quant_dequant_func(x)
        if _DISPATCH_OP:
            # Keep the M judgement inside an opaque custom op so it is
            # re-evaluated per step instead of being frozen at trace time.
            return mxfp4_linear_dispatch(
                x2, layer.weight, layer.weight_scale, bias)
        M = x2.shape[0]
        if M <= self.MAX_M and x2.shape[-1] % 32 == 0:
            from vllm._custom_ops import mx4_gemv as op
            # op handles [1, K] x [N, K/2]; loop over M rows (tiny).
            outs = []
            for m in range(M):
                o = op(x2[m:m+1], layer.weight, layer.weight_scale)
                outs.append(o)
            y = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
            if bias is not None:
                y = y + bias
            return y
        # large-M fallback: emulation (dequant + F.linear)
        import torch.nn.functional as F
        dq_w = dequant_mxfp4(layer.weight, layer.weight_scale, x2.dtype)
        return F.linear(x2, dq_w, bias)
