# Image ↔ source anchor (gfx1201)

This file records **which source tree the published gfx1201 container image was built from**, so
that anyone can check the correspondence instead of taking it on trust.

## The chain

| Step | Artifact | Identifier |
|---|---|---|
| Production source tree | `phase6-s2ar-merge` branch | commit `bea5ed7718` |
| Internal base image the stack was solidified from | tag `phase6:07-s2ar-e277655` | `461f658d3376` — **not published** |
| Internal production image | tag `phase6:07-s2ar-latest` | `sha256:ae868ff4cb93adef74c4e7771dec2b8e86b9d87366cd945c8ef57ce2ed6c446a` |
| **Published image** | `uzbn/vllm-gfx1201:gfx1201-r9700-v1.0-thin` | `sha256:1d5c4e1f76655f8edeaf6c8f9c4bf836aa20cab0619c96790edbb6df4fcf108b` |

The production image is a **delta**: the internal base, plus four files copied in. Those four are
the ones carrying this project's substantive changes.

## The four overlayed files

Measured **directly inside the published image**; all four match the expected hashes.

| Path inside the image | sha256 | Carries |
|---|---|---|
| `/src/vllm/vllm/v1/core/kv_cache_utils.py` | `47bd09ff477161e2d2a3b085e9233075468a54ce9fadad3a824169c3f8452051` | P1-6 (v1 + v2): DFlash draft-group KV annotation + `KV_GROUP_SIZE` group padding |
| `/src/vllm/vllm/model_executor/models/qwen3_dflash.py` | `acdfb21021541f31f995a88e7c382a15bd5f72ff0e9936ba2bbfe2916ab08fac` | P1-4: block-quantised FP8 draft KV dequantisation |
| `/src/vllm/vllm/model_executor/kernels/linear/__init__.py` | `da3dcfe103447ce4a9ae7935c1e47ecc390655bd479b1e53a4a0eddbeab42e88` | S9 M1: kernel registration block |
| `/src/vllm/vllm/model_executor/kernels/linear/mxfp4/radiance_w4a8.py` | `99feb8e390f2bcb3cf74b03ad89f0848dee9ce27c02405503c78d4e5ab700e42` | S9 M1: W4A8 plugin (new file) |

Check it yourself:

```bash
docker run --rm --entrypoint sha256sum uznb/vllm-gfx1201:gfx1201-r9700-v1.0-thin \
  /src/vllm/vllm/v1/core/kv_cache_utils.py \
  /src/vllm/vllm/model_executor/models/qwen3_dflash.py \
  /src/vllm/vllm/model_executor/kernels/linear/__init__.py \
  /src/vllm/vllm/model_executor/kernels/linear/mxfp4/radiance_w4a8.py
```

## Two honest notes

- **The internal base `e277655e9c` is not published.** The image was built as a delta on top of an
  internal pinned image, and that base's full source cannot be recovered commit-by-commit from the
  published trees. The four files above are what we can anchor; the rest of the image is
  "the pinned stack, unverified in public".
- **The published source branch is a superset of this image.** One commit (S8-A) landed on
  `gfx1201-r9700` after the image was built. It is off by default, so the image does not contain it
  and runtime behaviour is unchanged — building from the branch yields the same behaviour.

---

## 中文摘要

本文件记录已发布的 gfx1201 容器镜像**具体由哪份源码构建**，供任何人自行核对，而非凭信任。

- 生产源码树 = `phase6-s2ar-merge` 分支 commit `bea5ed7718`
- 内部生产镜像 = `ae868ff4cb93…`；**发布镜像** = `sha256:1d5c4e1f7665…`
- 镜像构成 = 内部基点镜像 + **4 个 overlay 文件**（见上表）；已对**发布镜像实测**，4 个 sha 全部吻合
- **两条如实声明**：
  1. 基点镜像 `e277655e9c` **未公开**，其完整源码无法从发布树逐提交复原——故只能锚定这 4 个文件，镜像其余部分属「已钉死但公开不可复核」。
  2. **发布分支 ⊃ 该镜像**——S8-A 提交在镜像构建之后才并入分支，**默认关闭**，行为无差异。
