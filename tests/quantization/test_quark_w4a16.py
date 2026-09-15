# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QuarkW4A16Int4 scheme 单测：路由 + 打包格式规范化 + 数值正确性。

权重参数按 2x AMD Radeon AI PRO R9700 (gfx1201) 主机上 amd_Qwen3.8-27B-Quark-AWQ-INT4-W4A16 的实测
quantization_config 填写：
  weight: dtype=int4, qscheme=per_group, group_size=128, symmetric=true,
          is_dynamic=false, scale_type=float
  input_tensors: null（真 W4A16，无激活量化）
  export: pack_method="reorder", weight_format="real_quantized"

CPU 侧测试（路由/格式/数值 emulation）无需 GPU；GPU kernel 端到端
测试在无 CUDA 设备时 skip。

Run:
  python -m pytest tests/quantization/test_quark_w4a16.py -v
"""

import torch
import pytest

from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
from vllm.model_executor.layers.quantization.quark.utils import (
    canonicalize_quark_packed_int4,
    parse_w4a16_int4_weight_config,
    should_ignore_layer,
)

_REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def _quark_int4_config(
    *,
    pack_method: str = "reorder",
    symmetric: bool = True,
    group_size: int = 128,
    exclude: list[str] | None = None,
    with_algo_config: bool = True,
) -> dict:
    """复刻 2x AMD Radeon AI PRO R9700 (gfx1201) 主机实测 config 的路由相关字段（与 quantization_config 等价）。"""
    cfg = {
        "quant_method": "quark",
        "export": {
            "kv_cache_group": [],
            "min_kv_scale": 0.0,
            "pack_method": pack_method,
            "weight_format": "real_quantized",
            "weight_merge_groups": None,
        },
        "global_quant_config": {
            "bias": None,
            "input_tensors": None,
            "output_tensors": None,
            "target_device": None,
            "weight": {
                "dtype": "int4",
                "group_size": group_size,
                "is_dynamic": False,
                "observer_cls": "PerGroupMinMaxObserver",
                "qscheme": "per_group",
                "scale_type": "float",
                "symmetric": symmetric,
            },
        },
        "exclude": exclude or [],
    }
    if with_algo_config:
        cfg["algo_config"] = [
            {
                "model_decoder_layers": "model.language_model.layers",
                "name": "awq",
                "scaling_layers": [
                    {
                        "inp": "mlp.gate_proj",
                        "layers": ["mlp.gate_proj", "mlp.up_proj"],
                        "module2inspect": "mlp",
                        "prev_op": "post_attention_layernorm",
                    },
                ],
            }
        ]
    return cfg


def test_quark_int4_reorder_config_routes_to_w4a16_scheme():
    """实测 config（int4/per_group/g128/symmetric/reorder）命中 QuarkW4A16Int4。

    移植前此处抛 NotImplementedError("No quark compatible scheme ...")，
    即 Quark 被剔除的根因。
    """
    from vllm.model_executor.layers.quantization.quark.schemes import QuarkW4A16Int4

    quant_config = QuarkConfig.from_config(_quark_int4_config())
    scheme = quant_config._get_scheme_from_config(
        quant_config.quant_config["global_quant_config"]
    )
    assert isinstance(scheme, QuarkW4A16Int4)
    assert scheme.group_size == 128
    assert scheme.is_symmetric
    assert scheme.pack_reorder


def test_quark_int4_scheme_rejects_activation_quant():
    """input_tensors 非空必须拒绝（真 W4A16 scheme 不支持激活量化）。"""
    cfg = _quark_int4_config()
    cfg["global_quant_config"]["input_tensors"] = {
        "dtype": "int8",
        "qscheme": "per_tensor",
        "is_dynamic": False,
        "symmetric": True,
    }
    quant_config = QuarkConfig.from_config(cfg)
    with pytest.raises(NotImplementedError):
        quant_config._get_scheme_from_config(
            quant_config.quant_config["global_quant_config"]
        )


def test_parse_w4a16_int4_weight_config_valid_and_invalid():
    weight = {"dtype": "int4", "group_size": 128, "symmetric": True}
    assert parse_w4a16_int4_weight_config(weight) == (128, True)

    for missing in ("group_size", "symmetric"):
        broken = dict(weight)
        broken.pop(missing)
        with pytest.raises(ValueError, match=missing):
            parse_w4a16_int4_weight_config(broken)

    with pytest.raises(ValueError):
        parse_w4a16_int4_weight_config(
            {"dtype": "int4", "group_size": -1, "symmetric": True}
        )
    with pytest.raises(ValueError):
        parse_w4a16_int4_weight_config(
            {"dtype": "int4", "group_size": 128, "symmetric": "yes"}
        )


def test_quark_apply_mapper_preserves_dict_valued_algo_config():
    """algo_config 是 dict 列表，不得送入 apply_list（否则 AttributeError）。"""
    quant_config = QuarkConfig.from_config(_quark_int4_config())
    before = quant_config.quant_config["algo_config"]
    quant_config.apply_vllm_mapper(_IdentityMapper())
    assert quant_config.quant_config["algo_config"] == before


class _IdentityMapper:
    """透传 mapper：模拟「无重命名」的 WeightsMapper。"""

    def apply_list(self, values):
        return list(values)

    def apply_dict(self, values):
        return dict(values)

    def _map_name(self, name):
        return name


def test_quark_bare_exclude_matches_lm_head_leaf():
    """exclude 含 bare "lm_head"（无 '.'、无通配符）须命中嵌套后的 lm_head 层名。

    模型 exclude 实测 = 111 个精确 visual 层名 + bare "lm_head"（共 112 项，
    无通配符——"model.visual.*" 是逐层展开的简写）。
    W4A16 路径要求 FP16 层（此处 lm_head）出现在 exclude 中（PR #48606 前提）。
    """
    quant_config = QuarkConfig.from_config(
        _quark_int4_config(
            exclude=["model.visual.blocks.0.attn.qkv", "lm_head"]
        )
    )
    exclude_layers = quant_config.quant_config["exclude"]
    # 精确 visual 层名直接命中
    assert should_ignore_layer(
        "model.visual.blocks.0.attn.qkv", ignore=exclude_layers
    )
    # bare "lm_head" 须匹配嵌套后的 "language_model.lm_head"（叶子名匹配）
    assert should_ignore_layer("language_model.lm_head", ignore=exclude_layers)
    # 语言模型层不应被忽略
    assert not should_ignore_layer(
        "language_model.model.layers.0.mlp.gate_proj", ignore=exclude_layers
    )


# ---------------------------------------------------------------------------
# 打包格式规范化 + 数值（CPU，emulation 参考）
# ---------------------------------------------------------------------------


def _pack_int4_nibbles(nibbles: torch.Tensor, *, pack_reorder: bool) -> torch.Tensor:
    """Pack 8 int4 values into one int32, with optional nibble reorder."""
    pack_order = (
        torch.tensor(_REVERSE_AWQ_PACK_ORDER, dtype=torch.long)
        if pack_reorder
        else torch.arange(8)
    )
    shifts = pack_order * 4
    return (
        ((nibbles.to(torch.int64) & 0xF) << shifts).sum(dim=-1).to(torch.int32)
    )


def _dequant_quark_reference(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    *,
    pack_reorder: bool,
    symmetric: bool,
) -> torch.Tensor:
    """Quark checkpoint 打包格式的纯 torch 反量化参考实现。

    packed: (K, N//8) int32；scales: (K//group_size, N) float（按输入维分组）
    返回 (K, N) 反量化权重。
    """
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    nibbles = ((packed[:, :, None] >> shifts[None, None, :]) & 0xF).to(torch.int8)
    order = (
        torch.tensor(_REVERSE_AWQ_PACK_ORDER, dtype=torch.long)
        if pack_reorder
        else torch.arange(8)
    )
    nibbles = nibbles.reshape(packed.shape[0], -1, 8)[:, :, order]
    nibbles = nibbles.reshape(packed.shape[0], -1)
    if symmetric:
        # int4 对称：无符号网格 + bias 8 → 有符号 [-8, 7]
        signed = nibbles.clone()
        high = (signed & 0x8).bool()
        signed[high] |= 0xF0  # 符号扩展
        nibbles = signed
    scales_expanded = scales.repeat_interleave(group_size, dim=0)  # (K, N)
    return nibbles.to(scales.dtype) * scales_expanded


@pytest.mark.parametrize("pack_method", ["order", "reorder"])
def test_canonicalize_quark_packed_int4_roundtrip(pack_method):
    """canonicalize 后的 AWQ 无符号格式反量化须等于 Quark 参考反量化。

    对称 int4：Quark 对称 nibble 存的是 signed-int4 补码网格；
    canonicalize 做 nibble 重排 + ^0x8 翻转为 AWQ 无符号 bias-8 约定。
    """
    pack_reorder = pack_method == "reorder"
    group_size = 2  # 小 group 简化参考实现
    k, n = 4, 8
    # 对称 int4：checkpoint nibble = 有符号值 ∈ [-8, 7] 的补码
    signed_values = torch.randint(-8, 8, (k, n), dtype=torch.int32)
    scales = (
        torch.randint(1, 5, (k // group_size, n), dtype=torch.float32) * 0.25
    )

    # Quark 参考值：直接按 checkpoint 打包语义反量化
    qweight = _pack_int4_nibbles(signed_values & 0xF, pack_reorder=pack_reorder).view(
        k, n // 8
    )
    expected = _dequant_quark_reference(
        qweight, scales, group_size, pack_reorder=pack_reorder, symmetric=True
    )
    # 参考实现应无损还原有符号值（除回 group scale）
    scales_expanded = scales.repeat_interleave(group_size, dim=0)
    torch.testing.assert_close(
        expected / scales_expanded,
        signed_values.to(scales.dtype),
        rtol=0,
        atol=0,
    )

    # canonicalize → AWQ 无符号格式，再用 AWQ 语义（iweight^0x8）反量化
    canonical = canonicalize_quark_packed_int4(
        qweight, pack_reorder=pack_reorder, is_symmetric=True
    )
    awq_dequant = _dequant_awq_unsigned(canonical, scales, group_size)
    torch.testing.assert_close(awq_dequant, expected, rtol=0, atol=0)


def _dequant_awq_unsigned(
    qweight: torch.Tensor, scales: torch.Tensor, group_size: int
) -> torch.Tensor:
    """AWQ/bias-8 打包反量化：存储 nibble = 值 + 8（uint4b8 约定），
    nibble 顺序为 AWQ 打包序（_REVERSE_AWQ_PACK_ORDER）。"""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    nibbles = ((qweight[:, :, None] >> shifts[None, None, :]) & 0xF).to(
        torch.int32
    )
    order = torch.tensor(_REVERSE_AWQ_PACK_ORDER, dtype=torch.long)
    nibbles = nibbles.reshape(qweight.shape[0], -1, 8)[:, :, order]
    nibbles = nibbles.reshape(qweight.shape[0], -1)
    signed = nibbles - 8
    scales_expanded = scales.repeat_interleave(group_size, dim=0)
    return signed.to(scales.dtype) * scales_expanded


@pytest.mark.gpu
def test_w4a16_scheme_create_and_apply_gpu():
    """GPU 端到端：create_weights + 填充合成权重 + apply vs BF16 参考。

    无 GPU 环境自动 skip。
    """
    pytest.importorskip("torch", minversion="2.0")
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")

    import torch.nn.functional as F

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.model_executor.layers.quantization.quark.schemes import (
        QuarkW4A16Int4,
    )
    from vllm.utils.torch_utils import set_random_seed

    set_random_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    # 对齐 RDNAHybridW4A16LinearKernel 门控：gs=128、K%16==0、K%gs==0
    k, n, group_size = 256, 256, 128

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:0",
            local_rank=0,
        )
        ensure_model_parallel_initialized(1, 1)

    class DummyLayer(torch.nn.Module):
        pass

    scheme = QuarkW4A16Int4(
        group_size=group_size, pack_method="reorder", is_symmetric=True
    )
    layer = DummyLayer()

    scheme.create_weights(
        layer,
        output_partition_sizes=[n],
        input_size_per_partition=k,
        params_dtype=dtype,
        weight_loader=lambda *args, **kwargs: None,
        input_size=k,
        output_size=n,
    )
    assert tuple(layer.weight.shape) == (k, n // 8)
    assert tuple(layer.weight_scale.shape) == (k // group_size, n)
    assert tuple(layer.weight_zero_point.shape) == (k // group_size, n // 8)

    # 合成有符号 int4 权重，按 Quark checkpoint 语义（reorder + 补码 nibble）打包
    signed = torch.randint(-8, 8, (k, n), dtype=torch.int32, device=device)
    packed = _pack_int4_nibbles(
        (signed & 0xF).reshape(k, n // 8, 8).cpu(), pack_reorder=True
    ).to(device)
    scales = (
        torch.rand((k // group_size, n), dtype=torch.float32, device=device)
        * 0.05
        + 0.05
    ).to(dtype)

    layer.weight.data = packed
    layer.weight_scale.data = scales
    layer.weight_zero_point.data.zero_()

    scheme.process_weights_after_loading(layer)

    # 参考输出：F.linear(x, dequant(w, scales))，dequant 按 Quark 对称
    # 补码语义（nibble ≥ 8 → 负数，即 signed int4 补码网格）。
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=device)
    nibbles = ((packed[:, :, None] >> shifts) & 0xF).reshape(k, n // 8, 8)
    order = torch.tensor(_REVERSE_AWQ_PACK_ORDER, dtype=torch.long, device=device)
    nibbles = nibbles[:, :, order].reshape(k, n)
    nibbles = torch.where(nibbles >= 8, nibbles - 16, nibbles).to(dtype)
    w_dequant = nibbles * scales.repeat_interleave(group_size, dim=0)
    x_m1 = (0.25 * torch.randn((1, k), device=device, dtype=torch.float32) / 16).to(
        dtype
    )
    x_m64 = (0.25 * torch.randn((64, k), device=device, dtype=torch.float32) / 16).to(
        dtype
    )
    # 参考矩阵乘用 fp32 累加（权重仍为 BF16 反量化值）：kernel 输出经 BF16
    # 舍入，而 fp32 参考无舍入；输出幅度压到 ~0.1 量级使 BF16 ULP 舍入
    # （~1e-3）远小于 1e-2 容差，浮出的才是真实 kernel 误差。
    ref_m1 = F.linear(x_m1.float(), w_dequant.float().t())
    ref_m64 = F.linear(x_m64.float(), w_dequant.float().t())

    # M=1 走 skinny decode 路径，M=64 走 Triton prefill 路径
    out_m1 = scheme.apply_weights(layer, x_m1)
    out_m64 = scheme.apply_weights(layer, x_m64)
    assert out_m1.shape == (1, n) and out_m64.shape == (64, n)

    torch.testing.assert_close(
        out_m1.float(), ref_m1, rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(
        out_m64.float(), ref_m64, rtol=1e-2, atol=1e-2
    )
