# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx1201 MXFP4(weight) x FP8(activation) linear kernel plugin.

Ported from the radiance-vllm-r9700 v0.10.0 blueprint module ``radiance_mxfp4.py``
(commit 054caecf9d870a10) so the hand-written fp8-WMMA GEMM in
``radiance_mxfp4_fp8.hip`` becomes selectable through vLLM's ``MxFp4LinearKernel``
plugin ABC instead of the stock emulation path.

RDNA4/gfx1201 W4A8 fold kernel.
Scope = target (2) W4A8 only; the mxfp8 e8m0/32 adaptation (target 1) is out of scope.

The sub-MIN_M case is delegated to the stock kernel instance, not routed through aiter's
dynamic-quant custom op: on this pin that op is registered in
``kernels/linear/mxfp4/aiter.py`` only inside
``if is_aiter_found_and_supported():`` -- ``is_rocm() and IS_AITER_FOUND and
get_cdna_version() > 2`` (``vllm/_aiter_ops.py``). gfx1201 is RDNA4, so
``get_cdna_version() == 0`` and the op is never registered; invoking it (the first small-M
call happens during cudagraph capture at load) raises ``AttributeError`` and the engine
never starts. See :func:`mxfp4_linear` for the delegation.

The stock gfx1201 kernel (``HipGemvMxfp4LinearKernel``) has TWO arms: ``M <= 4`` runs the
dedicated ``mx4_gemv`` (direct fp4 GEMV), ``M > 4`` runs the
``quant_dequant_mxfp4 + dequant_mxfp4 + F.linear`` emulation chain. The plugin re-inlines
neither arm: it delegates the entire sub-MIN_M case to the stock kernel instance itself, so
the B arm runs the A arm's exact code and branch at every M (re-inlining only one arm would
leave the other M range on a different path -- at ``M <= 4`` a full weight dequant where the
A arm runs the GEMV, i.e. B/A < 1 in decode). See :func:`mxfp4_linear` and
:func:`_stock_kernel_instance`.

Landing zone: ``init_mxfp4_linear_kernel()`` selects this plugin for layers whose
``activation_quant_key`` is ``kMxfp4Dynamic`` -- the ``QuarkOCP_MX`` MXFP4-weight
route. It does NOT replace the existing static-fp8 ``QuarkW4A8_MXFP4_FP8`` scheme,
which is a different activation contract and never enters this ABC.

pybind discipline (blueprint): the extension shares an ABI domain with
quark's TileLang exception translator, so any C++ exception escaping the .so is
relabelled into a bogus "libamdhip64.so not found" error. Every scratch buffer is
therefore allocated by torch and passed to the kernel as a raw pointer; the .so
never allocates and never throws.

Env gates (all opt-in; unset leaves stock vLLM untouched):
  RADIANCE_MXFP4_W4A8          =1 enables the kernel (default 0)
  RADIANCE_MXFP4_W4A8_MIN_M    folded-kernel M threshold; below it the kernel
                               delegates to the stock gfx1201 kernel
                               (HipGemvMxfp4LinearKernel) itself (default 256)
  RADIANCE_MXFP4_DECODE_MAX_M  decode-kernel band, 0 = dark (default 0)
  RADIANCE_MXFP4_WPERM         =1 store weights in WMMA fragment order
                               (requires MIN_M=0: the fallback reads checkpoint order)
  RADIANCE_MXFP4_CHECK_ALL     "N:K,..." verify these shapes against exact fp32
  RADIANCE_MXFP4_CHECK_MAX_M   M ceiling for that verification (default 128)
"""

import os
import sys

import torch

ENABLED = os.environ.get("RADIANCE_MXFP4_W4A8", "0") == "1"
# Smallest M the folded fp8-WMMA kernel serves. Below it -- and for layers layer_is_supported()
# rejects -- the op delegates to the stock gfx1201 kernel; see mxfp4_linear.
MIN_M = int(os.environ.get("RADIANCE_MXFP4_W4A8_MIN_M", "256"))
# Decode band for the small-M kernel. 0 = dark; also gates the scratch preallocation.
DECODE_MAX_M = int(os.environ.get("RADIANCE_MXFP4_DECODE_MAX_M", "0"))
# Store the weight in WMMA fragment order rather than the checkpoint's [N, K/2]. One env var
# drives both this flag and the HIP kernel's RADIANCE_MXFP4_WPERM; if they disagree the weight
# is read as garbage. This is what libr4d's r4d_gemm_mxfp4a8_nt_m64 reads.
WPERM = os.environ.get("RADIANCE_MXFP4_WPERM", "0") == "1"
# WPERM stores the weight in WMMA fragment order; the sub-MIN_M fallback dequantizes it in
# checkpoint order. With MIN_M > 0 the same layer is served by both paths, so the two are
# incompatible -- fail at import rather than read scattered bytes as weights (fluent garbage).
if WPERM and MIN_M > 0:
    raise RuntimeError(
        "[radiance.w4a8] RADIANCE_MXFP4_WPERM=1 needs RADIANCE_MXFP4_W4A8_MIN_M=0: the "
        "sub-MIN_M fallback reads the checkpoint weight layout."
    )

_ca = os.environ.get("RADIANCE_MXFP4_CHECK_ALL", "").strip()
CHECK_ALL = (
    {tuple(int(v) for v in pair.split(":")) for pair in _ca.split(",") if pair}
    if _ca
    else None
)
CHECK_MAX_M = int(os.environ.get("RADIANCE_MXFP4_CHECK_MAX_M", "128"))

_decode_scratch_ready = [False]
_decode_scratch = [None, None]  # [partials, block counter] -- kept alive for the process
# Stock gfx1201 kernel (HipGemvMxfp4LinearKernel) the sub-MIN_M fallback delegates to. Built
# once, at weight-load time, so nothing new is imported or constructed during graph capture.
_stock_kernel = None

try:
    import radiance_mxfp4_fp8 as _ext
except Exception as e:  # ext missing: stay on whatever vLLM would have picked
    _ext = None
    ENABLED = False
    sys.stderr.write(f"[radiance.w4a8] ext import failed, disabled: {e!r}\n")

if ENABLED and _ext is not None:
    # Print WHICH .so was loaded: a stale copy on sys.path shadows a newer one and a mismatched
    # kernel fails silently (fluent-looking garbage, no error anywhere).
    sys.stderr.write(
        f"[radiance.w4a8] MXFP4 x FP8 WMMA ENABLED for M>{MIN_M} "
        f"(W4A8, not the checkpoint's W4A4 -- more precise activations, not bit-identical)\n"
        f"[radiance.w4a8] kernel: {getattr(_ext, '__file__', '?')}\n"
    )


def permute_w(packed: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """[N, K/2] uint8 -> fragment order, returned with the ORIGINAL [N, K/2] shape.

    Slot l of tile (nt, ks) is W[nt][l&15][ks][4*(l>>4) .. +4]. It is a byte permutation, not a
    reshape: layer_is_supported, the N/K the op derives and any fallback all read that shape.
    """
    nt, ks = N // 16, K // 16
    return (
        packed.view(nt, 16, ks, 2, 4)  # [n-tile][row][k-step][half][4 bytes]
        .permute(0, 2, 3, 1, 4)  # [n-tile][k-step][half][row][4 bytes]
        .contiguous()
        .view(N, K // 2)
    )


def unpermute_w(packed: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """Inverse of :func:`permute_w`; only the exact-reference path needs it."""
    nt, ks = N // 16, K // 16
    return (
        packed.view(nt, ks, 2, 16, 4)  # [n-tile][k-step][half][row][4 bytes]
        .permute(0, 3, 1, 2, 4)  # [n-tile][row][k-step][half][4 bytes]
        .contiguous()
        .view(N, K // 2)
    )


_E2M1 = None


def make_row_ref(weight_scale: torch.Tensor) -> torch.Tensor:
    """Per-output-row reference exponent, computed ONCE at load.

    Folding the MX block exponent into the weight needs one reference per row; the kernel stores
    2^(ref-127) back in the epilogue. weight_scale is [K/32, N] here (post-transpose), so the max
    is over dim 0.
    """
    return weight_scale.max(dim=0).values.contiguous()


def _exact_ref(x_fp8, x_scale, weight, weight_scale, N, K, chunk=2048):
    """fp32 reference for this layer, chunked over N so it fits alongside a loaded model."""
    global _E2M1
    if _E2M1 is None:
        _E2M1 = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            device=weight.device,
        )
    if WPERM:
        weight = unpermute_w(weight, N, K)  # the reference speaks checkpoint order only
    xr = x_fp8.float() * x_scale.view(-1, 1).float()
    outs = []
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        wc = weight[a:b]
        codes = torch.stack([wc & 0x0F, (wc >> 4) & 0x0F], -1).reshape(b - a, K)
        sc = torch.pow(2.0, weight_scale[:, a:b].float() - 127.0).T.repeat_interleave(32, dim=1)
        outs.append(xr @ (_E2M1[codes.long()] * sc).T.float())
    return torch.cat(outs, dim=1)


def _stock_kernel_instance():
    """The stock gfx1201 MXFP4 kernel (HipGemvMxfp4LinearKernel), built once.

    Constructed at weight-load time (see ``process_weights_after_loading``) so nothing is
    imported or built during CUDA-graph capture. Its ``apply_weights`` (hip_gemv.py:65-87) is
    what the sub-MIN_M fallback runs: it owns both the ``M <= MAX_M(4)`` mx4_gemv arm and the
    large-M emulation arm, so the B arm stays on the A arm's path for every M.
    """
    global _stock_kernel
    if _stock_kernel is None:
        from vllm.model_executor.kernels.linear.mxfp4.base import (
            MxFp4LinearLayerConfig,
        )
        from vllm.model_executor.kernels.linear.mxfp4.hip_gemv import (
            HipGemvMxfp4LinearKernel,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kMxfp4Dynamic,
        )

        _stock_kernel = HipGemvMxfp4LinearKernel(
            MxFp4LinearLayerConfig(activation_quant_key=kMxfp4Dynamic)
        )
    return _stock_kernel


class _WeightView:
    """Layer stand-in for ``HipGemvMxfp4LinearKernel.apply_weights``.

    That method reads only ``.weight`` and ``.weight_scale`` (hip_gemv.py:78/86), so a plain
    attribute holder is enough. ``weight_scale`` here is ``layer.weight_scale`` itself -- the
    checkpoint-order ``[N, K/32]`` tensor the plugin leaves untouched -- i.e. the same object the
    stock A arm reads. Both stock consumers (``mx4_gemv`` and ``dequant_mxfp4``) want that layout.
    """

    __slots__ = ("weight", "weight_scale")

    def __init__(self, weight: torch.Tensor, weight_scale: torch.Tensor) -> None:
        self.weight = weight
        self.weight_scale = weight_scale


@torch.library.custom_op("radiance::mxfp4_linear", mutates_args=())
def mxfp4_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_ckpt: torch.Tensor,
    weight_ref: torch.Tensor,
) -> torch.Tensor:
    """Owns the ENTIRE dispatch, because the branch must not be visible to dynamo.

    A plain ``x.shape[0] > MIN_M`` in apply_weights is a data-dependent branch: vLLM compiles the
    model with a dynamic token dimension, so it would split the graph at every linear. Inside a
    registered custom op the body runs eagerly and the Python ``if`` is free.

    Two scale tensors, both built once at load: ``weight_scale`` is the [K/32, N] transpose the
    folded launcher reads, ``weight_scale_ckpt`` is the checkpoint [N, K/32] tensor the stock
    fallback reads. Keeping both means neither path pays a per-call copy or transpose.

    weight_ref encodes the route (a Python scalar read in apply_weights would be traced and baked
    into the compiled graph): numel == N selects the folded path; numel == 2 selects the
    per-block-rescale path; numel == 1 means the kernel cannot serve this layer.
    """
    nref = weight_ref.numel()
    folded = nref == weight.shape[0]
    w4a8_ok = folded or nref == 2
    if not (w4a8_ok and x.shape[0] > MIN_M):
        # Below MIN_M (and for layers the kernel cannot serve): delegate to the stock gfx1201
        # kernel itself -- the very object init_mxfp4_linear_kernel would have selected with
        # RADIANCE_MXFP4_W4A8 unset (HipGemvMxfp4LinearKernel; hip_gemv.py:30/65-87).
        #
        # Delegating, not re-inlining, is the point. Re-inlining stock's M>4 emulation arm
        # alone would leave M<=4 on a full weight dequant while the A arm runs the
        # dedicated mx4_gemv (hip_gemv.py:73-83): the decode regression. Handing the call to the
        # stock instance keeps B on A's exact code and branch at every M. (aiter's dynamic-quant
        # custom op is not an option: never registered on gfx1201, it raises AttributeError
        # during capture, a crash-at-load failure mode.)
        #
        # Hand the stock kernel the checkpoint-order scale it reads on the A arm -- the same
        # [N, K/32] tensor, built once at load, so the fallback does no transpose or copy of its
        # own (A reads layer.weight_scale; see _WeightView).
        wv = _WeightView(weight, weight_scale_ckpt)
        return _stock_kernel_instance().apply_weights(wv, x, None)
    M, K = x.shape
    N = weight.shape[0]
    from vllm import _custom_ops as ops

    x_fp8, x_scale = ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True)
    x_scale = x_scale.view(-1).float().contiguous()
    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    _ext.launch(
        x_fp8.data_ptr(),
        weight.data_ptr(),
        weight_scale.data_ptr(),
        weight_ref.data_ptr() if folded else 0,
        x_scale.data_ptr(),
        out.data_ptr(),
        M,
        N,
        K,
        torch.cuda.current_stream().cuda_stream,
    )
    if CHECK_ALL is not None and (N, K) in CHECK_ALL and M <= CHECK_MAX_M:
        ref = _exact_ref(x_fp8, x_scale, weight, weight_scale, N, K)
        rel = ((out.float() - ref).norm() / ref.norm().clamp_min(1e-9)).item()
        sys.stderr.write(
            f"[radiance.w4a8.exact] N={N} K={K} M={M} rel_vs_fp32={rel:.5f} "
            f"{'**WRONG**' if rel > 0.02 else 'ok'}\n"
        )
        sys.stderr.flush()
    return out


@mxfp4_linear.register_fake
def _(x, weight, weight_scale, weight_scale_ckpt, weight_ref):
    return torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=torch.bfloat16)


def layer_is_supported(layer, K: int) -> bool:
    """Called ONCE per layer at load time, never in the forward path.

    Only K is a hard constraint (the launcher rejects K % BK, BK=64). N is not: a partial N tile
    is masked by the kernel.
    """
    try:
        return bool(
            ENABLED
            and K % 64 == 0
            and layer.weight.shape[1] * 2 == K
            and layer.weight_scale.dim() == 2
            and layer.weight_scale.shape[1] == K // 32
        )
    except Exception:
        return False


def _on_gfx12x() -> bool:
    try:
        from vllm.platforms.rocm import on_gfx12x

        return bool(on_gfx12x())
    except Exception:
        return False


def _asm_gemm_enabled() -> bool:
    try:
        from vllm._aiter_ops import rocm_aiter_ops

        return bool(rocm_aiter_ops.is_asm_fp4_gemm_dynamic_quant_enabled())
    except Exception:
        return False


def _make_kernel_class():
    """Built lazily so importing this module never drags in vllm.model_executor.kernels."""
    from vllm.model_executor.kernels.linear.mxfp4.base import (
        MxFp4LinearKernel,
        MxFp4LinearLayerConfig,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic

    class RadianceMxfp4W4A8LinearKernel(MxFp4LinearKernel):
        """MXFP4 weights x fp8 activations on gfx1201, via the hand-written fp8-WMMA GEMM."""

        @classmethod
        def is_supported(cls, compute_capability=None):
            if not ENABLED or _ext is None:
                return False, "RADIANCE_MXFP4_W4A8 is not enabled, or the HIP extension is missing"
            if not _on_gfx12x():
                return False, "the radiance W4A8 MXFP4 kernel is compiled for gfx12x only"
            return True, None

        @classmethod
        def can_implement(cls, config: MxFp4LinearLayerConfig):
            if config.activation_quant_key != kMxfp4Dynamic:
                return False, "only supports MXFP4 dynamic activation"
            # The asm fp4 path stores weights shuffled (16,16) and the scale swizzled; this kernel
            # reads the plain packed weight and a [K/32, N] scale. Decline rather than hand
            # shuffled operands to the non-asm aiter GEMM.
            if _asm_gemm_enabled():
                return False, "aiter asm fp4 GEMM is enabled; its weight layout is incompatible"
            return True, None

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            # Preallocate the decode kernel's split-K partial slab on the FIRST layer, which is
            # weight-load time -- outside CUDA-graph capture. It cannot be done lazily inside the
            # launcher: with a warm torch.compile cache vLLM skips the eager profile run, so the
            # first GEMM call happens during capture, where hipMalloc is illegal (and the escape
            # surfaces as the misleading libamdhip64.so-not-found error).
            if not _decode_scratch_ready[0] and DECODE_MAX_M > 0:
                _decode_scratch_ready[0] = True
                try:
                    _decode_scratch[0] = torch.empty(
                        4 * max(64, DECODE_MAX_M) * 32768,
                        dtype=torch.float32,
                        device=layer.weight.device,
                    )
                    # Block counter for the fused reduction; MUST start zeroed.
                    _decode_scratch[1] = torch.zeros(
                        32768 // 128 + 8, dtype=torch.int32, device=layer.weight.device
                    )
                    _ext.set_decode_scratch(
                        _decode_scratch[0].data_ptr(),
                        _decode_scratch[0].numel() * 4,
                        _decode_scratch[1].data_ptr(),
                    )
                    sys.stderr.write(
                        f"[radiance.w4a8] decode kernel ON (M<={DECODE_MAX_M}), "
                        f"{_decode_scratch[0].numel() * 4 >> 20} MiB split-K scratch\n"
                    )
                except Exception as _e:
                    import traceback

                    sys.stderr.write(
                        f"[radiance.w4a8] decode scratch FAILED: {_e!r}\n"
                        + traceback.format_exc()
                    )
            # Build the stock fallback kernel now, at weight-load time -- outside CUDA-graph
            # capture -- so the first small-M call during capture finds it ready (no module
            # import or object construction inside the graph).
            try:
                _stock_kernel_instance()
            except Exception as _e:
                sys.stderr.write(
                    f"[radiance.w4a8] stock fallback kernel unavailable: {_e!r}\n"
                )
            # The folded fp8-WMMA launcher reads the e8m0 scale as [K/32, N]; create_weights lays
            # it out [N, K/32] (the checkpoint order stock HipGemv also reads). Build the
            # transposed buffer once here and leave layer.weight_scale untouched, so the sub-MIN_M
            # fallback can hand the stock kernel the very tensor the A arm reads -- no per-call
            # transpose or copy.
            layer.radiance_wscale_t = torch.nn.Parameter(
                layer.weight_scale.data.T.contiguous(), requires_grad=False
            )
            K = layer.weight.shape[1] * 2  # weights are 2 e2m1 codes per byte
            ok = layer_is_supported(layer, K)
            # Every layer carries the attribute so apply_weights never branches on hasattr, which
            # dynamo would have to guard. Ineligible layers get a 1-element placeholder and the op
            # passes 0 for it, taking the per-block-rescale path.
            ref = (
                make_row_ref(layer.radiance_wscale_t.data)
                if ok
                else torch.zeros(
                    1,
                    dtype=layer.radiance_wscale_t.dtype,
                    device=layer.radiance_wscale_t.device,
                )
            )
            layer.radiance_wref = torch.nn.Parameter(ref, requires_grad=False)
            # Fragment order, in place of the checkpoint layout. Only for layers this kernel
            # serves: a fallback layer must keep the layout aiter expects.
            if WPERM and ok:
                N_ = int(layer.weight.shape[0])
                if N_ % 16 or K % 16:
                    raise RuntimeError(
                        f"[radiance.w4a8] RADIANCE_MXFP4_WPERM needs N and K divisible by 16, "
                        f"got N={N_} K={K}"
                    )
                layer.weight = torch.nn.Parameter(
                    permute_w(layer.weight.data, N_, K), requires_grad=False
                )

        def apply_weights(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            y = torch.ops.radiance.mxfp4_linear(
                x,
                layer.weight,
                layer.radiance_wscale_t,
                layer.weight_scale,
                layer.radiance_wref,
            )
            if bias is not None:
                y = y + bias
            return y

    return RadianceMxfp4W4A8LinearKernel


_KERNEL_CLS = None


def kernel_class():
    """The plugin class, built once. Returns None if anything about it is unavailable."""
    global _KERNEL_CLS
    if _KERNEL_CLS is None:
        try:
            _KERNEL_CLS = _make_kernel_class()
        except Exception as e:
            sys.stderr.write(f"[radiance.w4a8] kernel class unavailable, disabled: {e!r}\n")
            _KERNEL_CLS = False
    return _KERNEL_CLS or None
