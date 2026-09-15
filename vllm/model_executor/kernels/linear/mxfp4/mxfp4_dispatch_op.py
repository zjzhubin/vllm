# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Make the mxfp4 small-M dispatch opaque to dynamo.

``HipGemvMxfp4LinearKernel.apply_weights`` (``hip_gemv.py``) chooses the ``mx4_gemv``
fast path with a data-dependent Python branch on the token count ``M``
(``if M <= MAX_M and K % 32 == 0``).  vLLM compiles the model with a dynamic token
dimension, so dynamo *inlines* ``apply_weights``, sees ``M`` as a symbolic size and
resolves the branch on whichever side the trace-time shape landed -- here the
``dequant_mxfp4`` + ``F.linear`` emulation arm -- baking it in for every decode step.
Decode then materialises the full bf16 weight per layer per token: the measured
~2.97x compile penalty and the ~6% PPL offset.

This module moves that whole dispatch into a registered ``torch.library.custom_op``.
Dynamo treats the op as opaque: it emits one node whose body runs eagerly, so the
``M`` branch is evaluated at runtime again and small-M decode recovers ``mx4_gemv``.

Env gate: ``VLLM_MXFP4_DISPATCH_OP`` (default off).  Unset, ``hip_gemv.apply_weights``
is the stock function (this module is not even imported) and the compiled graph and
its cache key are unchanged.

The gate is read straight from ``os.environ`` and is deliberately NOT registered in
``vllm.envs``: every entry of ``environment_variables`` feeds
``envs.compile_factors()``, so registering would change the torch.compile cache key
even with the gate off and break the baseline-same-source anchor.

Route isolation: mxfp4 only.  No FP8 path, no ``pass_config``, no change to
``_POSSIBLE_MXFP4_KERNELS`` or to any kernel-selection ordering.
"""

import os

import torch

_ENABLED = os.environ.get("VLLM_MXFP4_DISPATCH_OP", "0") == "1"

_MAX_M: int | None = None


def dispatch_op_enabled() -> bool:
    """Whether the env gate is on (read once at import)."""
    return _ENABLED


def _max_m() -> int:
    """``HipGemvMxfp4LinearKernel.MAX_M``, resolved lazily to avoid an import cycle."""
    global _MAX_M
    if _MAX_M is None:
        from vllm.model_executor.kernels.linear.mxfp4.hip_gemv import (
            HipGemvMxfp4LinearKernel,
        )

        _MAX_M = int(HipGemvMxfp4LinearKernel.MAX_M)
    return _MAX_M


@torch.library.custom_op("vllm_mxfp4::linear_dispatch", mutates_args=())
def mxfp4_linear_dispatch(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Own the M-dependent dispatch so dynamo never sees the branch.

    Body mirrors ``hip_gemv.apply_weights`` (:72-87) exactly, but executes eagerly
    inside the op instead of being inlined into the compiled graph.
    """
    M = x.shape[0]
    if M <= _max_m() and x.shape[-1] % 32 == 0:
        from vllm._custom_ops import mx4_gemv as op

        # op handles [1, K] x [N, K/2]; loop over M rows (tiny).
        outs = [op(x[m : m + 1], weight, weight_scale) for m in range(M)]
        y = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
        if bias is not None:
            y = y + bias
        return y
    # large-M fallback: emulation (dequant + F.linear)
    import torch.nn.functional as F

    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        dequant_mxfp4,
    )

    dq_w = dequant_mxfp4(weight, weight_scale, x.dtype)
    return F.linear(x, dq_w, bias)


@mxfp4_linear_dispatch.register_fake
def _mxfp4_linear_dispatch_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[0]), device=x.device, dtype=torch.bfloat16
    )
