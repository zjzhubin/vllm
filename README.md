<!-- markdownlint-disable MD001 MD041 -->
<h3 align="center">vLLM on AMD Radeon AI PRO R9700 (RDNA4 / gfx1201)</h3>

<p align="center">
A working, measured 2-card stack — a fork of <a href="https://github.com/vllm-project/vllm">vllm-project/vllm</a>
</p>

<p align="center">
<a href="#english">English</a> · <a href="#中文说明">中文</a>
</p>

---

> **This is a fork.** Branch `gfx1201-r9700` adds the RDNA4 (gfx1201) support described
> below. The companion changes on the attention/GEMM library side live in
> [zjzhubin/aiter](https://github.com/zjzhubin/aiter). Everything else in this tree is
> upstream vLLM, unmodified. Upstream components and their licences are listed in
> [`NOTICE`](NOTICE). Upstream project: https://github.com/vllm-project/vllm

<a id="english"></a>

# English

## 0. Read this first

**What this is.** A set of modifications that make vLLM run and serve on **2× AMD Radeon
AI PRO R9700** (RDNA4, `gfx1201`) with tensor-parallel size 2, in FP8, INT4/W4A16, MXFP4
and AWQ-FP8 quantisation, with speculative decoding (DFlash2) and a custom all-reduce.

**What this is not.**

- **Not a supported product.** No CI, no release cadence, no on-call. This fork does not
  track upstream.
- **Never built end-to-end as published.** Every individual piece was built and run during
  development, but the exact pin combination in the published build recipe was not
  rebuilt from scratch before publication. Treat the build as untested until you have run
  it. This is stated plainly in the recipe's own build notes.
- **Not benchmark-clean.** The performance numbers below were measured on our internal
  validation builds, not on this exact published revision, and several of them carry
  caveats that are documented rather than hidden. See §5.
- **RDNA4 only.** `gfx1201` is hard-coded and guarded; other architectures are not
  supported and will fail the build on purpose.

**What we are looking for.** Somebody with the engineering skill to take this over and
maintain it. The open items are listed in §7.

---

## 1. What this fork changes

### 1.1 On the vLLM side — 11 commits on top of upstream

| # | Commit subject | What it does |
|---|---|---|
| 01 | `build(rocm): add mx4 gemv kernel and CMake wiring` | Hand-written `mx4_gemv` HIP kernel (see `csrc/rocm/mx4_gemv.cu`) and its CMake registration. |
| 02 | `feat(mxfp4): add gfx1201 MXFP4 linear kernels` | MXFP4 linear-layer kernels and dispatch for RDNA4. |
| 03 | `feat(w4a8): add radiance W4A8 fold kernel` | W4A8 fold path — `kernels/linear/mxfp4/radiance_w4a8.py`. |
| 04 | `feat(quark): add W4A16 int4 scheme` | Quark W4A16 (INT4) quantisation scheme. |
| 05 | `feat(routing): add runtime routing and custom allreduce gating` | Runtime kernel routing table plus gating around the custom all-reduce. |
| 06 | `feat(spec-decode): dflash draft quantization and sample method derivation` | DFlash draft quantisation support and derivation of the draft sample method. |
| 07 | `feat(kv-cache): add KV cache group padding` | KV-cache group padding for hybrid (GDN + full-attention + drafter) configurations. |
| 08 | `perf(fp8): extend FP8 and aiter scaled_mm paths` | Extended FP8 and aiter `scaled_mm` paths, including RDNA4 gating. |
| 09 | `feat(attention): update ROCm aiter unified attention backend` | aiter unified-attention backend for the V1 attention path. |
| 10 | `chore(rocm): platform gates, env switches and compilation registration` | Platform gates, environment switches, compilation registration. |
| 11 | `docs: add NOTICE for upstream components` | The [`NOTICE`](NOTICE) file. |

### 1.2 On the aiter side — 5 commits, in [zjzhubin/aiter](https://github.com/zjzhubin/aiter)

| # | Commit subject |
|---|---|
| 01 | `chore(rocm): register gfx1201 arch and compilation flags` |
| 02 | `feat(attention): enable unified attention on gfx1201` |
| 03 | `feat(sampling): add hardware FP8 conversion and tensor top-k resolution` |
| 04 | `feat(comm): add runtime routing and fix PCIe all-reduce visibility` |
| 05 | `feat(tuner): key tuned GEMM configs on the runtime CU count` |

Both repositories carry a tag `gfx1201-r9700-v1.0-public` on the branch tip.

---

## 2. Getting it running

### 2.1 Build the image

The build recipe is **not** part of this tree. It is the `gfx1201-stack` bundle attached to
the release `gfx1201-r9700-v1.0-public` on this repository, and it also contains the patch
series if you would rather apply the changes to a clean upstream checkout instead of using
this branch.

The recipe is a five-stage Docker chain (`p6.01` … `p6.05`). Two notes before you start:

- **It needs outbound network** for the first build (it clones pinned upstream commits and
  may compile MLIR from source). Expect a long, quiet MLIR/LLVM build if no prebuilt MLIR
  is provided.
- **The base is ROCm 10.1.0a (TheRock nightly) + PyTorch for that ROCm build.** The recipe
  pins those; do not substitute.

### 2.2 Run it

```bash
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --shm-size 16g \
  -v /path/to/models:/models \
  -p 8000:8000 \
  <your-image-tag> \
  /opt/venv/bin/vllm serve /models/<your-model> \
      --tensor-parallel-size 2 --host 0.0.0.0 --port 8000
```

Clear `/dev/shm` before starting a server if a previous run left state behind.

### 2.3 Smoke check

There is no smoke-test script in the repository. The two checks we use are:

```bash
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health   # expect 200
```

and a fixed arithmetic prompt through the OpenAI-compatible endpoint — on a healthy
gfx1201 stack the answer comes back correct and clean, not garbled. A garbled response is
a real symptom (see §6), not a prompt problem.

---

## 3. Supported models

All of these exist on Hugging Face, are public, and require no licence acceptance. Weights
are **not** distributed here; fetch them yourself and put them under `/models`.

| # | Weights | Quantisation | Linear-layer path |
|---|---|---|---|
| 1 | `Qwen/Qwen3.8-27B-FP8` | FP8 block=128 (official) | `AiterFp8BlockScaledMMKernel` (true block-scaled GEMM) |
| 2 | `cyankiwi/Qwen3.8-27B-AWQ-INT4` | INT4 gs=32 | `RDNAHybridW4A16` |
| 3 | `philbert440/Qwen3.8-27B-W4A16-AWQ` | INT4 gs=128 (W4A16) | `RDNAHybridW4A16` |
| 4 | `amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16` | INT4 | `QuarkW4A16Int4` → `RDNAHybrid` |
| 5 | `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` | MXFP4 (e2m1 weights + e8m0/32 scale) | `HipGemvMxfp4` (stock) / `radiance_mxfp4_fp8` fold kernel |
| 6 | `cyankiwi/Qwen3.8-27B-AWQ-FP8` | FP8 per-token (W8A8) | `RowWiseTorchFP8ScaledMMLinearKernel` (unfused dequant) |

Speculative-decoding draft models, all `--speculative-config` compatible:

| Draft weights | Format | Pairs with | `num_speculative_tokens` |
|---|---|---|---|
| `tcclaviger/Qwen3.8-27B-DFlash2-FP8` | FP8 | #1 | 3 |
| `syvai/Qwen3.8-27B-DFlash2-W4A16` | W4A16 | #3 | 2 |
| `z-lab/Qwen3.8-27B-DFlash2` | BF16 | #1 | 3 — **produces garbled output on this stack, see §6** |

### 3.1 Serve arguments we use

Common to every model: `--tensor-parallel-size 2`, `--enable-chunked-prefill`,
`--kv-cache-dtype fp8`, `--enable-prefix-caching`,
`--attention-backend ROCM_AITER_UNIFIED_ATTN` (which must be paired with
`VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1`).

Per model:

| # | Serve arguments |
|---|---|
| 1 | `--served-model-name qwen36 --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.92` |
| 2 | `--served-model-name cyankiwi --max-model-len 262144 --max-num-seqs 12 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.8` |
| 3 | `--served-model-name qwen36 --max-model-len 262144 --max-num-seqs 16 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.8` |
| 4 | `--served-model-name quark --max-model-len 262144 --max-num-seqs 16 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.8` |
| 5 | `--served-model-name mxfp4 --max-model-len 262144 --max-num-seqs 16 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.8` |
| 6 | `--served-model-name awqfp8 --max-model-len 262144 --max-num-seqs 16 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.8` |

For #1 and #3 with their drafts:

```bash
--speculative-config '{"method":"dflash","model":"<draft-weights-path>","num_speculative_tokens":3}'
```

---

## 4. Environment variables

### 4.1 aiter switches (set by the recipe; these are the values we run)

| Variable | We run | Default | Effect |
|---|---|---|---|
| `VLLM_ROCM_USE_AITER` | `1` | `False` | Master switch for all aiter kernels. |
| `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION` | `1` | `False` | V1 attention via aiter triton unified attention. |
| `VLLM_ROCM_USE_AITER_TRITON_ROPE` | `1` | `False` | aiter triton RoPE. |
| `VLLM_ROCM_USE_AITER_LINEAR` | `1` | `True` | aiter linear kernels (scaled_mm per-tensor/rowwise + tuned unquantised GEMM). |
| `VLLM_ROCM_USE_AITER_LINEAR_HIPBMM` | `1` | `False` | aiter `hipb_mm`. |
| `VLLM_ROCM_USE_AITER_TRITON_GEMM` | `1` | `True` | aiter triton GEMM. |
| `VLLM_ROCM_USE_AITER_RMSNORM` | `1` | `True` | aiter rmsnorm. |
| `VLLM_ROCM_USE_AITER_MHA` | `1` | `True` | aiter MHA kernels. |
| `VLLM_ROCM_USE_AITER_CUSTOM_AR` | `1` | `True` | aiter custom all-reduce backend. |
| `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM` | `0` | — | Dead key — no consumer anywhere in the tree. |
| `AITER_ROPE_NATIVE_BACKEND` | `1` | — | aiter-side RoPE native backend selection. |
| `FLASH_ATTENTION_TRITON_AMD_ENABLE` | `TRUE` | — | Triton AMD FlashAttention. |
| `RDNA_AITER_SAMPLER` | `1` | `0` | Explicitly enables the aiter sampler. |
| `HIP_VISIBLE_DEVICES` | `0,1` | — | Required for TP=2. |

### 4.2 Switches added by this fork

| Variable | Applies to | Default | Effect |
|---|---|---|---|
| `KV_GROUP_SIZE` | all; largest gain on hybrid configs with a drafter | unset (stock behaviour) | KV-cache group padding. `auto` minimises padded layer slots, an integer forces that group size. In our runs `=8` on the #1 configuration gave **+17.9% KV capacity** (capacity, i.e. slots — not a throughput figure). |
| `RADIANCE_MXFP4_W4A8` (+ `_MIN_M`) | #5 only | `0` | Prefill through the self-built FP8-WMMA fold path. |
| `VLLM_MXFP4_DISPATCH_OP` | #5 only | `0` | Fixes dynamo folding of the MXFP4 dispatch. |
| `RADIANCE_MXFP4_DECODE_MAX_M` | #5 only | `0` | Small-M decode kernel band. |
| `RADIANCE_MXFP8_ACT_PER_TOKEN` | MXFP8 path | — | Per-token E8M0 passthrough. **Leave off** — measured 0.837× on our runs. |
| `RADIANCE_MXFP4_WPERM` / `_TN4_MIN_M` / `_CHECK_ALL` / `_CHECK_MAX_M` | #5 only | `0` / `2048` / empty / `128` | Kernel-shape knobs for the fold path. |

---

## 5. Measured performance

### 5.1 How to read these numbers

Read this before quoting any figure below.

1. **These were not measured on this exact published revision.** They come from internal
   validation builds of this stack taken between 2026-09-10 and 2026-09-13. The published
   tree is the accumulated state of that work. No single row should be read as a
   characterisation of the published pin.
2. **A single number is not a baseline.** On the same machine we have observed the *same
   nominal configuration* measured on different builds to give decode c1 of 32.4 and 47.7
   tok/s — a ~47% spread with no identified code difference between them. We do not
   attribute that spread to any change.
3. **"Window state."** We have also measured the same configuration twice inside one
   session, minutes apart, and seen a >15% swing. This is why, internally, we require any
   performance comparison to carry a same-window anchor arm. Where a row below says
   *no anchor recorded*, no such arm was taken, and the figure should be treated as a point
   observation, not a reproducible measurement.
4. **What the recipe ships is a config, not a claim.** Nothing here is a guarantee of what
   you will see on your hardware.

### 5.2 Common measurement setup

2× Radeon AI PRO R9700 (gfx1201) · TP=2 · `--kv-cache-dtype fp8` · `--enable-chunked-prefill`
· `--enable-prefix-caching` · `--attention-backend ROCM_AITER_UNIFIED_ATTN` · the aiter
environment of §4.1 · default CUDA/HIP graph mode · client **GuideLLM 0.5.4**, metric is
aggregate output tokens/s, `c` = concurrent requests.

### 5.3 Numbers

> **Two measurement windows are in play here — read the `measured` column.**
>
> The table below is our **first** window (2026-09-10 to 09-13). Those numbers stand exactly as
> recorded, including the caveats in §5.4. A **second window** was run on **2026-09-15/16**,
> re-measuring every weight format twice per concurrency level, and adding a
> **production-form** configuration (GMU 0.92 / `--max-num-seqs 8`). Its headline results are
> FP8 + DFlash2 at **344.9** (c12) / **324.1** (c16) agg tok/s, prefill at 32k of **3,285**
> tok/s, and the same stack at production settings scoring c1 **68.0** / c8 **276.5**.
>
> **The two windows are not directly comparable to each other** — absolute values drift
> between windows (see §5.1 and §5.4). We publish both rather than overwriting the older
> numbers, because the earlier rows are the ones our own caveats refer to.
>
> Full second-window report, with a config fingerprint on every row:
> `https://github.com/zjzhubin/vllm/blob/gfx1201-r9700/gfx1201/benchmarks/vllm-gfx1201-Phase6-%E4%B8%89%E6%A8%A1%E5%9E%8B%E5%8F%8C%E5%9C%BA%E6%99%AF%E6%B5%8B%E9%80%9F%E6%B1%87%E6%80%BB-2026-09-15.md`

| Weights | Draft | decode c1 | c4 | c12 / c16 | prefill tok/s | measured | evidence |
|---|---|---|---|---|---|---|---|
| #1 `Qwen/Qwen3.8-27B-FP8` | DFlash2-FP8, ns=3 | **64.4** | **176.3** | 278.7 (c12) | 3,870 | 2026-09-13 | bracketed run — **anchor arms did not pass**, see §5.4 |
| #1 `Qwen/Qwen3.8-27B-FP8` | none | 32.4 | 112.0 | 245.1 (c16) | 2,957 | — | full fingerprint, no anchor recorded |
| #2 `cyankiwi/…-AWQ-INT4` | none | 45.1 | 119.6 | 159.4 (c16) | — | 2026-09-10 | full fingerprint, no anchor recorded |
| #6 `cyankiwi/…-AWQ-FP8` | none | 25.1 | 87.7 | 218.1 (c16) | — | 2026-09-10 | full fingerprint, no anchor recorded; c1 low, not attributed |
| #3 `philbert440/…-W4A16` | DFlash2-W4A16, ns=2 | 67.83 | — | — | — | — | partial fingerprint; without the draft, c1 29.34 |
| #4 `amd/…-Quark-INT4-W4A16` | none | 29.68 | — | — | 2,419–2,448 | — | partial fingerprint |

A note on the two FP8 rows: 64.4 (with draft) against 32.4 (without) is the speculative
decoding effect, not a build difference. The 47.7 figure in §5.1 is a third measurement of
the *no-draft* configuration taken in a different window.

### 5.4 What we do not vouch for

We are listing these because they are the things you would otherwise rediscover the hard
way. All of them come from our own records:

- **The 2026-09-13 FP8 bracketed run failed its own anchor check.** Same-configuration
  anchors placed before and after the measurement window disagreed by more than our 5%
  threshold, which means the window flipped during the run. The row above is reported
  because it is the only full-suite FP8 measurement we have; it should not be treated as
  stable.
- **An earlier MXFP4 decode figure we published internally is withdrawn.** It measured at
  9.7 tok/s and turned out to be a client-side harness artifact, not a kernel result. Any
  speed-up figure computed against that 9.7 is meaningless. If you see a "4.2×" MXFP4
  claim anywhere, that is where it came from — ignore it.
- **MXFP4 W4A8 fold path.** Works, but has not been re-measured on the production
  configuration; treat its numbers as provisional.
- **Quark INT4 (#4) at c16** measured a ~9.8% regression against an earlier build. Not
  attributed, not fixed.
- **Numerical/accuracy comparisons.** Our relative-perplexity numbers were measured without
  an established noise floor, so we cannot tell signal from noise in them. We are not
  publishing any accuracy claim, including any "negligible quality loss" claim.
- **TP=1 is out of scope.** We withdrew all single-card data; this stack is TP=2 only.

---

## 6. Known issues

- **BF16 draft + FP8 target produces garbled output.** `z-lab/Qwen3.8-27B-DFlash2` (BF16)
  paired with `Qwen/Qwen3.8-27B-FP8` returns garbage. We measured a 0% acceptance rate. Use
  the FP8 draft (`tcclaviger/…-DFlash2-FP8`) instead, which works.
- **FP8 + DFlash2 at long context.** A 128k refill has been observed to hang silently —
  no error, no output. Restart the server.
- **Do not install `amdsmi` 7.0.2 from PyPI.** It does not match this ROCm tree's
  `libamd_smi.so` and importing it crashes torch. The recipe works around this; if you
  build your own image, mind it.
- **The tuned GEMM config overlay is load-bearing.** The recipe injects tuned A8W8
  block-scaled GEMM configurations into aiter's config directory at build time and fails
  the build if they are missing. Losing them silently costs 1.9%–30.4% on the affected
  shapes with no warning. This is a deliberate hard failure, not an accident.
- **No CI, no gfx1201 test suite.** Nothing here is regression-protected.

---

## 7. Status and hand-over

This fork was built by one person with heavy AI assistance. It works, and the numbers in §5
are the real ones, caveats included. It deserves a maintainer who can do the things we
cannot:

| Open item | Why it matters |
|---|---|
| Build the published pin end-to-end | Nobody has. Everything else is downstream of this. |
| Re-run §5.3 with proper same-window anchors | Every number in §5.3 that says "no anchor" is a point observation, not a measurement. |
| Attribute the FP8 window instability (32.4 vs 47.7 vs the failed 2026-09-13 bracket) | Without it, no FP8 performance claim is safe. |
| Establish a PPL noise floor, then redo the accuracy comparisons | Our accuracy conclusions cannot currently discriminate signal from noise. |
| Quark INT4 c16 regression | Open, unattributed. |
| MXFP4 fold path: re-measure on the production config | M1/`radiance_mxfp4_fp8` is the line we were most actively working on when we stopped. |
| Make the stack upstreamable | Upstream has no gfx1201 support. Much of what is here belongs in vLLM and aiter proper, not in a fork. |

Issues and PRs on this fork are welcome. There is no guarantee of a timely reply.

---

## 8. Relationship to upstream and licence

This repository is a fork of [vllm-project/vllm](https://github.com/vllm-project/vllm).
The upstream project is Apache-2.0 licensed; this fork adds no licence of its own and is
distributed under the same terms. This fork links against ROCm/aiter,
ROCm/composable_kernel, ROCm/FlyDSL, Dao-AILab/flash-attention, ROCm/triton and PyTorch;
those components, their licences, and the modifications made to them are documented in
[`NOTICE`](NOTICE).

Upstream vLLM documentation, quickstart and model list remain valid for everything not
covered by §1: https://docs.vllm.ai

If you use vLLM for research, please cite the upstream paper:

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

---
---

<a id="中文说明"></a>

# 中文说明

## 0. 先读这一节

**这是什么。** 一组让 vLLM 能在 **2× AMD Radeon AI PRO R9700**（RDNA4，`gfx1201`）、
张量并行度 2 的条件下真正跑起来并对外服务的改动，覆盖 FP8、INT4/W4A16、MXFP4、AWQ-FP8
四种量化，带推测解码（DFlash2）和自定义 all-reduce。

**这不是什么。**

- **不是一个有支持的产品。** 没有 CI、没有发版节奏、没有值班。本 fork 不跟上游同步。
- **公开发布的这个形态从未被端到端构建过。** 每一块单独都构建和运行过，但发布食谱里
  那一组确切的版本组合，在发布前没有从头重建过。在你亲手跑通之前，请当作未经验证。
  食谱自带的构建说明里也是这么写的。
- **测速数据不干净。** 下面的性能数字来自我们内部的验证构建，不是这份发布版本，而且
  其中若干条带有明确记录的保留意见，不是被藏起来的。详见 §5。
- **只支持 RDNA4。** `gfx1201` 是硬编码并带守卫的，其它架构不被支持，构建时会**故意
  报错退出**。

**我们在找什么。** 找一个有工程能力的人接手维护。待办事项列在 §7。

---

## 1. 这个 fork 改了什么

### 1.1 vLLM 侧——在上游之上新增 11 个提交

| # | 提交标题 | 做了什么 |
|---|---|---|
| 01 | `build(rocm): add mx4 gemv kernel and CMake wiring` | 手写 `mx4_gemv` HIP kernel（见 `csrc/rocm/mx4_gemv.cu`）及其 CMake 注册。 |
| 02 | `feat(mxfp4): add gfx1201 MXFP4 linear kernels` | RDNA4 的 MXFP4 线性层 kernel 与分发。 |
| 03 | `feat(w4a8): add radiance W4A8 fold kernel` | W4A8 折叠路径——`kernels/linear/mxfp4/radiance_w4a8.py`。 |
| 04 | `feat(quark): add W4A16 int4 scheme` | Quark W4A16（INT4）量化方案。 |
| 05 | `feat(routing): add runtime routing and custom allreduce gating` | 运行时 kernel 路由表，以及自定义 all-reduce 的门控。 |
| 06 | `feat(spec-decode): dflash draft quantization and sample method derivation` | DFlash 草稿模型量化支持，以及采样方式推导。 |
| 07 | `feat(kv-cache): add KV cache group padding` | 面向 hybrid 配置（GDN + 全注意力 + 草稿）的 KV cache 分组 padding。 |
| 08 | `perf(fp8): extend FP8 and aiter scaled_mm paths` | 扩展 FP8 与 aiter `scaled_mm` 路径，含 RDNA4 门控。 |
| 09 | `feat(attention): update ROCm aiter unified attention backend` | V1 注意力路径接入 aiter unified attention 后端。 |
| 10 | `chore(rocm): platform gates, env switches and compilation registration` | 平台门控、环境开关、编译注册。 |
| 11 | `docs: add NOTICE for upstream components` | [`NOTICE`](NOTICE) 文件。 |

### 1.2 aiter 侧——5 个提交，在 [zjzhubin/aiter](https://github.com/zjzhubin/aiter)

| # | 提交标题 |
|---|---|
| 01 | `chore(rocm): register gfx1201 arch and compilation flags` |
| 02 | `feat(attention): enable unified attention on gfx1201` |
| 03 | `feat(sampling): add hardware FP8 conversion and tensor top-k resolution` |
| 04 | `feat(comm): add runtime routing and fix PCIe all-reduce visibility` |
| 05 | `feat(tuner): key tuned GEMM configs on the runtime CU count` |

两个仓库都在分支顶端打了 tag `gfx1201-r9700-v1.0-public`。

---

## 2. 怎么跑起来

### 2.1 构建镜像

构建食谱**不在本仓库里**。它是本仓库 release `gfx1201-r9700-v1.0-public` 附带的
`gfx1201-stack` 包，里面同时含补丁包——如果你不想用本分支，也可以拿补丁打到干净的上游
检出上。

食谱是一条五级 Docker 链（`p6.01` … `p6.05`）。开始前有两点：

- **首次构建需要外网**（要 clone 钉死的上游提交，若没有预制 MLIR 还可能现场编译 MLIR）。
  没有预制 MLIR 时，准备好等一次漫长且安静的 MLIR/LLVM 构建。
- **底座是 ROCm 10.1.0a（TheRock nightly）+ 对应这个 ROCm 版本的 PyTorch。** 食谱已经
  钉死，不要自行替换。

### 2.2 启动

```bash
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --shm-size 16g \
  -v /path/to/models:/models \
  -p 8000:8000 \
  <your-image-tag> \
  /opt/venv/bin/vllm serve /models/<你的模型> \
      --tensor-parallel-size 2 --host 0.0.0.0 --port 8000
```

上次跑完若在 `/dev/shm` 留了状态，起服前先清掉。

### 2.3 冒烟检查

仓库里**没有**冒烟脚本。我们用的两个检查是：

```bash
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health   # 期望 200
```

以及通过 OpenAI 兼容接口发一个固定的算术题——在健康的 gfx1201 栈上，答案正确且干净，
不出现乱码。**出现乱码是真实症状（见 §6），不是提示词的问题。**

---

## 3. 支持的模型

以下权重在 Hugging Face 上全部存在、公开、无需申请授权。权重**不随本仓分发**，请自行
获取并放到 `/models` 下。

| # | 权重 | 量化 | 线性层路径 |
|---|---|---|---|
| 1 | `Qwen/Qwen3.8-27B-FP8` | FP8 block=128（官方） | `AiterFp8BlockScaledMMKernel`（真 blockscale GEMM） |
| 2 | `cyankiwi/Qwen3.8-27B-AWQ-INT4` | INT4 gs=32 | `RDNAHybridW4A16` |
| 3 | `philbert440/Qwen3.8-27B-W4A16-AWQ` | INT4 gs=128（W4A16） | `RDNAHybridW4A16` |
| 4 | `amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16` | INT4 | `QuarkW4A16Int4` → `RDNAHybrid` |
| 5 | `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` | MXFP4（e2m1 权重 + e8m0/32 scale） | `HipGemvMxfp4`（stock）/ `radiance_mxfp4_fp8` 折叠 kernel |
| 6 | `cyankiwi/Qwen3.8-27B-AWQ-FP8` | FP8 per-token（W8A8） | `RowWiseTorchFP8ScaledMMLinearKernel`（非融合反量化） |

草稿模型（均可用于 `--speculative-config`）：

| 草稿权重 | 格式 | 配套 | `num_speculative_tokens` |
|---|---|---|---|
| `tcclaviger/Qwen3.8-27B-DFlash2-FP8` | FP8 | #1 | 3 |
| `syvai/Qwen3.8-27B-DFlash2-W4A16` | W4A16 | #3 | 2 |
| `z-lab/Qwen3.8-27B-DFlash2` | BF16 | #1 | 3 —— **在本栈上输出乱码，见 §6** |

### 3.1 我们用的 serve 参数

所有模型公共：`--tensor-parallel-size 2`、`--enable-chunked-prefill`、`--kv-cache-dtype fp8`、
`--enable-prefix-caching`、`--attention-backend ROCM_AITER_UNIFIED_ATTN`（必须与
`VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1` 配对使用）。

分模型：

| # | serve 参数 |
|---|---|
| 1 | `--served-model-name qwen36 --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.92` |
| 2 | `--served-model-name cyankiwi --max-model-len 262144 --max-num-seqs 12 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.8` |
| 3 | `--served-model-name qwen36 --max-model-len 262144 --max-num-seqs 16 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.8` |
| 4 | `--served-model-name quark --max-model-len 262144 --max-num-seqs 16 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.8` |
| 5 | `--served-model-name mxfp4 --max-model-len 262144 --max-num-seqs 16 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.8` |
| 6 | `--served-model-name awqfp8 --max-model-len 262144 --max-num-seqs 16 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.8` |

#1 与 #3 配草稿时：

```bash
--speculative-config '{"method":"dflash","model":"<草稿权重路径>","num_speculative_tokens":3}'
```

---

## 4. 环境变量

### 4.1 aiter 开关（食谱已设；以下是我们实际运行的取值）

| 变量 | 我们设 | 默认 | 作用 |
|---|---|---|---|
| `VLLM_ROCM_USE_AITER` | `1` | `False` | aiter 算子总开关。 |
| `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION` | `1` | `False` | V1 注意力走 aiter triton unified attention。 |
| `VLLM_ROCM_USE_AITER_TRITON_ROPE` | `1` | `False` | aiter triton RoPE。 |
| `VLLM_ROCM_USE_AITER_LINEAR` | `1` | `True` | aiter linear 算子（scaled_mm per-tensor/rowwise + 未量化 tuned GEMM）。 |
| `VLLM_ROCM_USE_AITER_LINEAR_HIPBMM` | `1` | `False` | aiter `hipb_mm`。 |
| `VLLM_ROCM_USE_AITER_TRITON_GEMM` | `1` | `True` | aiter triton GEMM。 |
| `VLLM_ROCM_USE_AITER_RMSNORM` | `1` | `True` | aiter rmsnorm。 |
| `VLLM_ROCM_USE_AITER_MHA` | `1` | `True` | aiter MHA 算子。 |
| `VLLM_ROCM_USE_AITER_CUSTOM_AR` | `1` | `True` | aiter 自定义 all-reduce 后端。 |
| `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM` | `0` | — | 死键——全树无消费点。 |
| `AITER_ROPE_NATIVE_BACKEND` | `1` | — | aiter 侧 rope native backend 选择。 |
| `FLASH_ATTENTION_TRITON_AMD_ENABLE` | `TRUE` | — | Triton 版 AMD FlashAttention。 |
| `RDNA_AITER_SAMPLER` | `1` | `0` | 显式启用 aiter sampler。 |
| `HIP_VISIBLE_DEVICES` | `0,1` | — | TP=2 必需。 |

### 4.2 本 fork 新增的开关

| 变量 | 适用 | 默认 | 作用 |
|---|---|---|---|
| `KV_GROUP_SIZE` | 全部；带草稿的 hybrid 配置收益最大 | unset（stock 行为） | KV cache 分组 padding。`auto` 最小化 padding 后的层槽数，整数则强制该分组大小。我们的运行里，#1 配置上 `=8` 带来 **+17.9% KV 容量**（是容量，即槽位数——不是吞吐指标）。 |
| `RADIANCE_MXFP4_W4A8`（+ `_MIN_M`） | 仅 #5 | `0` | prefill 走自研 FP8-WMMA 折叠路径。 |
| `VLLM_MXFP4_DISPATCH_OP` | 仅 #5 | `0` | 修 MXFP4 分发的 dynamo 折叠。 |
| `RADIANCE_MXFP4_DECODE_MAX_M` | 仅 #5 | `0` | decode 小 M kernel 档。 |
| `RADIANCE_MXFP8_ACT_PER_TOKEN` | MXFP8 路径 | — | per-token E8M0 直通。**不要开**——我们的运行里实测 0.837×。 |
| `RADIANCE_MXFP4_WPERM` / `_TN4_MIN_M` / `_CHECK_ALL` / `_CHECK_MAX_M` | 仅 #5 | `0` / `2048` / 空 / `128` | 折叠路径的 kernel 形状旋钮。 |

---

## 5. 实测性能

### 5.1 怎么读这些数字

引用下面任何一个数字之前，请先读这一节。

1. **不是在这份发布版本上测的。** 它们来自 2026-09-10 至 2026-09-13 期间对本栈内部验证
   构建的测量。发布版是那批工作累积下来的状态。**不要把任何一行读成对发布版本的性能刻画。**
2. **单点数字不是基准线。** 在同一台机器上，我们见过**同一名义配置**在不同构建上测出
   decode c1 = 32.4 与 47.7 tok/s——相差约 47%，且两者之间找不到已识别的代码差异。
   我们**不把这个差异归因于任何改动**。
3. **「窗口态」。** 我们还见过同一次会话内、相隔几分钟测同一配置，摆动超过 15%。正因如此，
   项目内部要求任何性能比较都必须带**同窗口锚臂**。下表标注「无锚臂记录」的行，意味着没有
   取过这样的锚臂，该数字应视为一次点观测，不是可复现的测量。
4. **食谱交付的是配置，不是承诺。** 这里没有一条能保证你在自己硬件上看到相同结果。

### 5.2 公共测量口径

2× Radeon AI PRO R9700（gfx1201）· TP=2 · `--kv-cache-dtype fp8` · `--enable-chunked-prefill`
· `--enable-prefix-caching` · `--attention-backend ROCM_AITER_UNIFIED_ATTN` · §4.1 的 aiter
环境 · 默认 CUDA/HIP graph 模式 · 客户端 **GuideLLM 0.5.4**，指标为聚合输出 tok/s，
`c` = 并发请求数。

### 5.3 数字

> **这里有两套测量窗口——请看「测量日」这一列。**
>
> 下表是**第一套窗口**（2026-09-10 ~ 09-13）。这些数字**按当时记录原样保留**，包括 §5.4 的
> 保留意见。**第二套窗口**跑于 **2026-09-15/16**：每种权重格式在每个并发档重测两遍，并补了
> **生产形态**配置（GMU 0.92 / `--max-num-seqs 8`）。第二窗的代表数字是 FP8 + DFlash2 的
> **344.9**（c12）/ **324.1**（c16）agg tok/s、32k prefill **3,285** tok/s，以及同一栈在生产
> 参数下的 c1 **68.0** / c8 **276.5**。
>
> **两套窗口之间不可直接比较**——绝对值存在跨窗口漂移（见 §5.1 与 §5.4）。我们**两套都列**、
> 而不是覆盖旧数字，因为上面那些保留意见指向的正是旧的那些行。
>
> 第二窗完整报告（每一行都带配置指纹）：
> `https://github.com/zjzhubin/vllm/blob/gfx1201-r9700/gfx1201/benchmarks/vllm-gfx1201-Phase6-%E4%B8%89%E6%A8%A1%E5%9E%8B%E5%8F%8C%E5%9C%BA%E6%99%AF%E6%B5%8B%E9%80%9F%E6%B1%87%E6%80%BB-2026-09-15.md`

| 权重 | 草稿 | decode c1 | c4 | c12 / c16 | prefill tok/s | 测量日 | 证据质量 |
|---|---|---|---|---|---|---|---|
| #1 `Qwen/Qwen3.8-27B-FP8` | DFlash2-FP8, ns=3 | **64.4** | **176.3** | 278.7 (c12) | 3,870 | 2026-09-13 | 带 bracket 锚臂——**锚臂未通过**，见 §5.4 |
| #1 `Qwen/Qwen3.8-27B-FP8` | 无 | 32.4 | 112.0 | 245.1 (c16) | 2,957 | — | 指纹完整，无锚臂记录 |
| #2 `cyankiwi/…-AWQ-INT4` | 无 | 45.1 | 119.6 | 159.4 (c16) | — | 2026-09-10 | 指纹完整，无锚臂记录 |
| #6 `cyankiwi/…-AWQ-FP8` | 无 | 25.1 | 87.7 | 218.1 (c16) | — | 2026-09-10 | 指纹完整，无锚臂记录；c1 偏低，未归因 |
| #3 `philbert440/…-W4A16` | DFlash2-W4A16, ns=2 | 67.83 | — | — | — | — | 指纹不完整；不带草稿时 c1 29.34 |
| #4 `amd/…-Quark-INT4-W4A16` | 无 | 29.68 | — | — | 2,419–2,448 | — | 指纹不完整 |

关于两行 FP8：64.4（带草稿）对 32.4（不带）是推测解码的效果，不是构建差异。§5.1 里的
47.7 是**不带草稿**配置在另一个窗口的第三次测量。

### 5.4 我们不背书的

列出来，是因为这些是你否则会自己踩一遍的坑。全部来自我们自己的记录：

- **2026-09-13 那轮 FP8 带锚臂测量没通过自己的锚臂检查。** 测量窗口前后放置的同配置锚臂
  差值超过了我们 5% 的阈值，说明测量期间窗口发生了翻转。上表仍然报了这一行，因为它是
  我们唯一一份完整的 FP8 全套测量；**它不应被当作稳定值。**
- **早先一个 MXFP4 decode 数字我们已撤回。** 内部曾测得 9.7 tok/s，事后坐实是客户端
  harness 的口径假象，不是 kernel 结果。**任何基于那个 9.7 算出来的加速比都没有意义。**
  如果你在哪里看到 MXFP4「4.2×」的说法，出处就是它——请忽略。
- **MXFP4 W4A8 折叠路径。** 能用，但未在生产配置上重测；其数字视为暂定。
- **Quark INT4（#4）在 c16** 相对更早的构建测得约 9.8% 回退。未归因，未修复。
- **数值/精度比较。** 我们的相对 PPL 数据在**没有建立噪声底**的前提下测得，因而无法区分
  信号与噪声。**我们不发布任何精度结论**，包括任何「质量损失可忽略」的说法。
- **TP=1 不在范围内。** 单卡数据我们已全部撤回；本栈只做 TP=2。

---

## 6. 已知问题

- **BF16 草稿 + FP8 主模型会输出乱码。** `z-lab/Qwen3.8-27B-DFlash2`（BF16）配
  `Qwen/Qwen3.8-27B-FP8` 返回垃圾内容，我们实测接受率 0%。改用 FP8 草稿
  （`tcclaviger/…-DFlash2-FP8`），那条路径是好的。
- **FP8 + DFlash2 长上下文。** 128k refill 出现过静默挂起——无报错、无输出，重启服务。
- **不要装 PyPI 上的 `amdsmi` 7.0.2。** 它与本 ROCm 树的 `libamd_smi.so` 不匹配，导入
  会崩 torch。食谱已经绕过；你自己构建镜像时请留意。
- **调优 GEMM 配置覆盖层是承重的。** 食谱在构建期把调优过的 A8W8 blockscale GEMM 配置
  注入 aiter 的配置目录，缺失即**构建失败**。丢了它，受影响形状会静默慢 1.9%–30.4%，
  没有任何警告。这是刻意的硬失败，不是意外。
- **没有 CI，没有 gfx1201 测试套件。** 这里没有任何东西受回归保护。

---

## 7. 状态与交接

本 fork 由一个人借助大量 AI 辅助完成。它能跑，§5 里的数字是真的，保留意见也一并写着。
它值得一个能做我们做不了的事的维护者：

| 待办 | 为什么重要 |
|---|---|
| 端到端构建发布版的确切版本组合 | **至今无人做过。** 其它一切都建立在它之上。 |
| 用合格的同窗口锚臂重跑 §5.3 | §5.3 里每一行标「无锚臂」的都只是点观测，不是测量。 |
| 归因 FP8 窗口不稳定（32.4 / 47.7 / 2026-09-13 锚臂失败） | 不归因，任何 FP8 性能结论都不安全。 |
| 建立 PPL 噪声底，再重做精度比较 | 目前的精度结论无法区分信号与噪声。 |
| Quark INT4 c16 回退 | 未归因，未解决。 |
| MXFP4 折叠路径在生产配置上重测 | 我们停下时最活跃的一条线。 |
| 让这套东西能进上游 | 上游没有 gfx1201 支持。这里的很多东西该进 vLLM 和 aiter 本身，而不是留在 fork 里。 |

欢迎在本 fork 提 issue 和 PR。但**不保证及时回复**。

---

## 8. 与上游的关系与授权

本仓库是 [vllm-project/vllm](https://github.com/vllm-project/vllm) 的 fork。上游项目为
Apache-2.0 授权；本 fork 不额外附加授权条款，以相同条款分发。本 fork 链接
ROCm/aiter、ROCm/composable_kernel、ROCm/FlyDSL、Dao-AILab/flash-attention、ROCm/triton
与 PyTorch；这些组件、其授权以及对其所做的修改，记录在 [`NOTICE`](NOTICE)。

§1 未覆盖的一切，上游 vLLM 的文档、快速开始与模型列表依然适用：https://docs.vllm.ai

若你在研究中使用 vLLM，请引用上游论文（见英文段的 BibTeX）。
