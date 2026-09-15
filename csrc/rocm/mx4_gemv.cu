// 增加 per-warp 行并行 (每 block 8 warp 各管一行 -> 行级 ILP) + 激活 LDS 缓存一次
// 关键: 每 block 256 线程全在 1 行上 (group 级并行) 时, 行切换串行.
// warp w 处理 row = blockIdx*8 + w (grid-stride), warp 内 lane 走 group.
// 每 warp 独立行 -> 无 block barrier, 激活经 __shfl 广播.
#include <torch/all.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

typedef short __attribute__((ext_vector_type(2))) bf16x2_t;

__device__ __forceinline__ unsigned mx4_bits(unsigned n) {
  const unsigned s = (n >> 3) & 1u, e = (n >> 1) & 3u, m = n & 1u;
  if (e == 0) return (s << 15) | (m ? 0x3F00u : 0x0000u);
  return (s << 15) | ((e + 126) << 7) | (m << 6);
}

__global__ void __launch_bounds__(256)
mx4_gemv_k(const __hip_bfloat16* __restrict__ A,
              const uint8_t* __restrict__ B,
              const uint8_t* __restrict__ S,
              __hip_bfloat16* __restrict__ C,
              const int K, const int N) {
  const int K_packed = K / 2;
  const int num_groups = K / 32;
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;   // 0..7

  const int row = blockIdx.x * 8 + wid;
  if (row >= N) return;
  const uint8_t* wrow = B + (size_t)row * K_packed;
  const uint8_t* srow = S + (size_t)row * num_groups;

  float acc = 0.f;
  // lane 处理 groups g = lane, lane+32, ... (warp 内 32 路)
  for (int g = lane; g < num_groups; g += 32) {
    const int byte0 = g * 16;
    const float pw = exp2f((float)(unsigned)srow[g] - 127.0f);
    const uint4 wv = *reinterpret_cast<const uint4*>(wrow + byte0);
    const unsigned wb[4] = {wv.x, wv.y, wv.z, wv.w};
    float p = 0.f;
#pragma unroll
    for (int vI = 0; vI < 4; vI++) {
#pragma unroll
      for (int bB = 0; bB < 4; bB++) {
        const unsigned two = (wb[vI] >> (bB * 8)) & 0xFFu;
        const int k0 = (byte0 + vI*4 + bB) * 2;
        const unsigned w2 = (mx4_bits(two & 0xFu)) | (mx4_bits((two >> 4) & 0xFu) << 16);
        const unsigned a2 = (unsigned)*(const unsigned short*)&A[k0] |
                            ((unsigned)*(const unsigned short*)&A[k0 + 1] << 16);
        p = __builtin_amdgcn_fdot2_f32_bf16(*(bf16x2_t*)&a2, *(bf16x2_t*)&w2, p, false);
      }
    }
    acc += p * pw;
  }
  // warp reduce (无 cross-warp: 每 warp 独立行)
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
  if (lane == 0) C[row] = __float2bfloat16(acc);
}

torch::Tensor mx4_gemv(const at::Tensor& in_a, const at::Tensor& in_b,
                        const at::Tensor& in_scale) {
  const int K = in_a.size(1);
  const int N = in_b.size(0);
  auto C = torch::empty({1, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(in_b.device()));
  mx4_gemv_k<<<(N + 7) / 8, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __hip_bfloat16*>(in_a.data_ptr()),
      in_b.data_ptr<uint8_t>(), in_scale.data_ptr<uint8_t>(),
      reinterpret_cast<__hip_bfloat16*>(C.data_ptr()), K, N);
  return C;
}

