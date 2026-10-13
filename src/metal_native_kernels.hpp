// Native Metal kernels that replace selected uber-shader modes on Apple GPUs.
// The driver (metal_driver.mm) substitutes them only when the host flags a
// dispatch (push operation3 == 1); the uber-shader ignores that flag, so the
// SPIR-V-derived path stays a complete fallback. Bindings and the push block
// are the uber-shader's: buffers 0..4 = A, B, C, D, E; buffer 8 = Params.
#pragma once

namespace ppocr::detail {

inline constexpr const char kMetalNativeKernels[] = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct Params {
  uint count, batches, operation0, operation1, operation2, operation3;
  uint steps, right_repeat, right_per_batch, right_batch_stride, mode;
};

// Same polynomial and operation order as erf_fp32 in vulkan_binary.comp.
static float erf_fp32(float x) {
  const float sign = x < 0.0f ? -1.0f : 1.0f;
  const float ax = fabs(x);
  const float t = 1.0f / (1.0f + 0.5f * ax);
  const float polynomial = -1.26551223f + t * (1.00002368f + t * (0.37409196f +
      t * (0.09678418f + t * (-0.18628806f + t * (0.27886807f +
      t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f))))))));
  return sign * (1.0f - t * exp(-ax * ax + polynomial));
}

// Pointwise (1x1) Conv as GEMM: out[m, p] = bias[m] + sum_k W[m, k] X[k, p]
// per batch, then the mode's epilogue (uber-shader modes 3,4,5,6,7,9,10,11,
// 12,55). 128 threads = 4 simdgroups; a threadgroup owns a 32 (m) x 32 (p)
// tile and each simdgroup a 16 x 16 quarter (2 x 2 simdgroup_float8x8).
// Accumulators start from the bias, like the scalar kernels.
constant constexpr uint TM = 32, TP = 32, TK = 8;

kernel void pointwise_gemm(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device const float* c [[buffer(2)]],
                           device float* d [[buffer(3)]],
                           device const float* e [[buffer(4)]],
                           constant Params& params [[buffer(8)]],
                           uint3 group [[threadgroup_position_in_grid]],
                           uint local [[thread_index_in_threadgroup]],
                           uint simd [[simdgroup_index_in_threadgroup]]) {
  const uint plane = params.right_repeat;
  const uint channels = params.right_per_batch;
  const uint outputs = params.right_batch_stride;
  const uint batch = group.y;
  const uint plane_tiles = (plane + TP - 1) / TP;
  const uint m0 = (group.x / plane_tiles) * TM;
  const uint p0 = (group.x % plane_tiles) * TP;
  threadgroup float ws[TM * TK];
  threadgroup float xs[TK * TP];
  threadgroup float cs[TM * TP];
  const device float* x = a + batch * channels * plane;

  for (uint i = local; i < TM * TP; i += 128) {
    const uint m = m0 + i / TP;
    cs[i] = m < outputs ? c[m] : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint sm = (simd / 2) * 16, sp = (simd % 2) * 16;
  simdgroup_float8x8 acc[2][2];
  for (uint i = 0; i < 2; ++i)
    for (uint j = 0; j < 2; ++j)
      simdgroup_load(acc[i][j], cs + (sm + i * 8) * TP + sp + j * 8, TP);

  for (uint k0 = 0; k0 < channels; k0 += TK) {
    for (uint i = local; i < TM * TK; i += 128) {
      const uint m = m0 + i / TK, k = k0 + i % TK;
      ws[i] = (m < outputs && k < channels) ? b[m * channels + k] : 0.0f;
      const uint xk = k0 + i / TP, p = p0 + i % TP;
      xs[i] = (xk < channels && p < plane) ? x[xk * plane + p] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 wf[2], xf[2];
    for (uint i = 0; i < 2; ++i) simdgroup_load(wf[i], ws + (sm + i * 8) * TK, TK);
    for (uint j = 0; j < 2; ++j) simdgroup_load(xf[j], xs + sp + j * 8, TP);
    for (uint i = 0; i < 2; ++i)
      for (uint j = 0; j < 2; ++j)
        simdgroup_multiply_accumulate(acc[i][j], wf[i], xf[j], acc[i][j]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 2; ++i)
    for (uint j = 0; j < 2; ++j)
      simdgroup_store(acc[i][j], cs + (sm + i * 8) * TP + sp + j * 8, TP);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (batch >= params.batches) return;
  const uint mode = params.mode & 0xffffu;
  for (uint i = local; i < TM * TP; i += 128) {
    const uint m = m0 + i / TP, p = p0 + i % TP;
    if (m >= outputs || p >= plane) continue;
    const uint index = batch * params.count + m * plane + p;
    float value = cs[i];
    if (mode == 5u || mode == 6u || mode == 9u) value += e[index];
    if (mode == 4u || mode == 6u) value = max(value, 0.0f);
    if (mode == 7u || mode == 9u) value *= 1.0f / (1.0f + exp(-value));
    if (mode == 10u) value = 1.0f / (1.0f + exp(-value));
    if (mode == 11u) value = clamp(as_type<float>(params.operation0) * value +
                                   as_type<float>(params.operation1), 0.0f, 1.0f);
    if (mode == 12u) value *= clamp(as_type<float>(params.operation0) * value +
                                    as_type<float>(params.operation1), 0.0f, 1.0f);
    if (mode == 55u) value = 0.5f * value * (1.0f + erf_fp32(value * 0.7071067811865475f));
    d[index] = value;
  }
}
// pointwise_gemm with double-buffered staging (default; PPOCR_METAL_DISABLE_POINTWISE_DB=1 keeps the single-buffer kernel).
kernel void pointwise_gemm_db(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device const float* c [[buffer(2)]],
                           device float* d [[buffer(3)]],
                           device const float* e [[buffer(4)]],
                           constant Params& params [[buffer(8)]],
                           uint3 group [[threadgroup_position_in_grid]],
                           uint local [[thread_index_in_threadgroup]],
                           uint simd [[simdgroup_index_in_threadgroup]]) {
  const uint plane = params.right_repeat;
  const uint channels = params.right_per_batch;
  const uint outputs = params.right_batch_stride;
  const uint batch = group.y;
  const uint plane_tiles = (plane + TP - 1) / TP;
  const uint m0 = (group.x / plane_tiles) * TM;
  const uint p0 = (group.x % plane_tiles) * TP;
  threadgroup float wd[2][TM * TK];
  threadgroup float xd[2][TK * TP];
  threadgroup float cs[TM * TP];
  const device float* x = a + batch * channels * plane;

  for (uint i = local; i < TM * TP; i += 128) {
    const uint m = m0 + i / TP;
    cs[i] = m < outputs ? c[m] : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint sm = (simd / 2) * 16, sp = (simd % 2) * 16;
  simdgroup_float8x8 acc[2][2];
  for (uint i = 0; i < 2; ++i)
    for (uint j = 0; j < 2; ++j)
      simdgroup_load(acc[i][j], cs + (sm + i * 8) * TP + sp + j * 8, TP);

  // Double-buffered staging: the next 8-deep slice is read into registers
  // while the current one multiplies, then stored to the other buffer; one
  // barrier per K step. Same products in the same order as pointwise_gemm.
  float wn[2], xn[2];
#define PPOCR_PW_FETCH(K0)                                                   \
  for (uint r = 0; r < 2; ++r) {                                             \
    const uint i = local + r * 128;                                          \
    const uint m = m0 + i / TK, k = (K0) + i % TK;                           \
    wn[r] = (m < outputs && k < channels) ? b[m * channels + k] : 0.0f;      \
    const uint xk = (K0) + i / TP, p = p0 + i % TP;                          \
    xn[r] = (xk < channels && p < plane) ? x[xk * plane + p] : 0.0f;         \
  }
  PPOCR_PW_FETCH(0u)
  for (uint r = 0; r < 2; ++r) { wd[0][local + r * 128] = wn[r]; xd[0][local + r * 128] = xn[r]; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint cur = 0;
  for (uint k0 = 0; k0 < channels; k0 += TK) {
    const bool more = k0 + TK < channels;
    if (more) { PPOCR_PW_FETCH(k0 + TK) }
    simdgroup_float8x8 wf[2], xf[2];
    for (uint i = 0; i < 2; ++i) simdgroup_load(wf[i], wd[cur] + (sm + i * 8) * TK, TK);
    for (uint j = 0; j < 2; ++j) simdgroup_load(xf[j], xd[cur] + sp + j * 8, TP);
    for (uint i = 0; i < 2; ++i)
      for (uint j = 0; j < 2; ++j)
        simdgroup_multiply_accumulate(acc[i][j], wf[i], xf[j], acc[i][j]);
    if (more) {
      for (uint r = 0; r < 2; ++r) { wd[cur ^ 1][local + r * 128] = wn[r]; xd[cur ^ 1][local + r * 128] = xn[r]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    cur ^= 1;
  }
#undef PPOCR_PW_FETCH
  for (uint i = 0; i < 2; ++i)
    for (uint j = 0; j < 2; ++j)
      simdgroup_store(acc[i][j], cs + (sm + i * 8) * TP + sp + j * 8, TP);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (batch >= params.batches) return;
  const uint mode = params.mode & 0xffffu;
  for (uint i = local; i < TM * TP; i += 128) {
    const uint m = m0 + i / TP, p = p0 + i % TP;
    if (m >= outputs || p >= plane) continue;
    const uint index = batch * params.count + m * plane + p;
    float value = cs[i];
    if (mode == 5u || mode == 6u || mode == 9u) value += e[index];
    if (mode == 4u || mode == 6u) value = max(value, 0.0f);
    if (mode == 7u || mode == 9u) value *= 1.0f / (1.0f + exp(-value));
    if (mode == 10u) value = 1.0f / (1.0f + exp(-value));
    if (mode == 11u) value = clamp(as_type<float>(params.operation0) * value +
                                   as_type<float>(params.operation1), 0.0f, 1.0f);
    if (mode == 12u) value *= clamp(as_type<float>(params.operation0) * value +
                                    as_type<float>(params.operation1), 0.0f, 1.0f);
    if (mode == 55u) value = 0.5f * value * (1.0f + erf_fp32(value * 0.7071067811865475f));
    d[index] = value;
  }
}
// Modes 50/51: 3x3 Conv (stride 1 or 2, ReLU for 51) as an implicit GEMM:
// out[oc, p] = bias[oc] + sum_k W[oc, k] X[k, p] with k = ic * 9 + ky * 3 + kx,
// the uber-shader's weight layout. Pads are in the mode's high bytes; bit 15
// swaps the input (D) and output (A) bindings. Same 32x32 tiling as above.
kernel void conv3x3_gemm(device float* a [[buffer(0)]],
                         device const float* b [[buffer(1)]],
                         device const float* c [[buffer(2)]],
                         device float* d [[buffer(3)]],
                         constant Params& params [[buffer(8)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         uint local [[thread_index_in_threadgroup]],
                         uint simd [[simdgroup_index_in_threadgroup]]) {
  const uint channels = params.operation0;
  const uint input_h = params.operation1, input_w = params.operation2;
  const uint output_h = params.operation3, output_w = params.steps;
  const uint stride = params.right_batch_stride;
  const int pad_top = int((params.mode >> 16u) & 0xffu);
  const int pad_left = int((params.mode >> 24u) & 0xffu);
  const bool swap_ad = (params.mode & 0x8000u) != 0u;
  const uint plane = output_h * output_w;
  const uint outputs = params.count / plane;
  const uint depth = channels * 9u;
  const uint batch = group.y;
  const uint plane_tiles = (plane + TP - 1) / TP;
  const uint m0 = (group.x / plane_tiles) * TM;
  const uint p0 = (group.x % plane_tiles) * TP;
  threadgroup float ws[TM * TK];
  threadgroup float xs[TK * TP];
  threadgroup float cs[TM * TP];
  const uint input_plane = input_h * input_w;
  const device float* x = (swap_ad ? d : a) + batch * channels * input_plane;

  for (uint i = local; i < TM * TP; i += 128) {
    const uint m = m0 + i / TP;
    cs[i] = m < outputs ? c[m] : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint sm = (simd / 2) * 16, sp = (simd % 2) * 16;
  simdgroup_float8x8 acc[2][2];
  for (uint i = 0; i < 2; ++i)
    for (uint j = 0; j < 2; ++j)
      simdgroup_load(acc[i][j], cs + (sm + i * 8) * TP + sp + j * 8, TP);

  for (uint k0 = 0; k0 < depth; k0 += TK) {
    for (uint i = local; i < TM * TK; i += 128) {
      const uint m = m0 + i / TK, k = k0 + i % TK;
      ws[i] = (m < outputs && k < depth) ? b[m * depth + k] : 0.0f;
      const uint xk = k0 + i / TP, p = p0 + i % TP;
      float v = 0.0f;
      if (xk < depth && p < plane) {
        const uint ic = xk / 9u, r = xk - ic * 9u, ky = r / 3u, kx = r - ky * 3u;
        const uint oy = p / output_w, ox = p - oy * output_w;
        const int iy = int(oy * stride + ky) - pad_top;
        const int ix = int(ox * stride + kx) - pad_left;
        if (iy >= 0 && iy < int(input_h) && ix >= 0 && ix < int(input_w))
          v = x[ic * input_plane + uint(iy) * input_w + uint(ix)];
      }
      xs[i] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 wf[2], xf[2];
    for (uint i = 0; i < 2; ++i) simdgroup_load(wf[i], ws + (sm + i * 8) * TK, TK);
    for (uint j = 0; j < 2; ++j) simdgroup_load(xf[j], xs + sp + j * 8, TP);
    for (uint i = 0; i < 2; ++i)
      for (uint j = 0; j < 2; ++j)
        simdgroup_multiply_accumulate(acc[i][j], wf[i], xf[j], acc[i][j]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 2; ++i)
    for (uint j = 0; j < 2; ++j)
      simdgroup_store(acc[i][j], cs + (sm + i * 8) * TP + sp + j * 8, TP);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (batch >= params.batches) return;
  const bool relu = (params.mode & 0x7fffu) == 51u;
  device float* y = swap_ad ? a : d;
  for (uint i = local; i < TM * TP; i += 128) {
    const uint m = m0 + i / TP, p = p0 + i % TP;
    if (m >= outputs || p >= plane) continue;
    const float value = cs[i];
    y[batch * params.count + m * plane + p] = relu ? max(value, 0.0f) : value;
  }
}
// Modes 24/25: out[r, n] = (bias[n] if operation3) + sum_k A[r, k] B[k, n],
// Swish for 25. A is [rows, depth] row-major, or [depth, lda] when
// right_repeat (lda) is non-zero. B is [depth, columns]. Same 32x32 tiling;
// accumulators start from the bias like the uber-shader's vec4 path.
kernel void rows_gemm(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device const float* c [[buffer(2)]],
                      device float* d [[buffer(3)]],
                      constant Params& params [[buffer(8)]],
                      uint3 group [[threadgroup_position_in_grid]],
                      uint local [[thread_index_in_threadgroup]],
                      uint simd [[simdgroup_index_in_threadgroup]]) {
  const uint rows = params.operation0, depth = params.operation1, columns = params.operation2;
  const bool has_bias = params.operation3 != 0u;
  const uint lda = params.right_repeat;
  constexpr uint TK = 8;
  const uint column_tiles = (columns + TP - 1) / TP;
  const uint m0 = (group.x / column_tiles) * TM;
  const uint n0 = (group.x % column_tiles) * TP;
  threadgroup float ws[TM * TK];
  threadgroup float xs[TK * TP];
  threadgroup float cs[TM * TP];
  for (uint i = local; i < TM * TP; i += 128) {
    const uint n = n0 + i % TP;
    cs[i] = (has_bias && n < columns) ? c[n] : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint sm = (simd / 2) * 16, sp = (simd % 2) * 16;
  simdgroup_float8x8 acc[2][2];
  for (uint i = 0; i < 2; ++i)
    for (uint j = 0; j < 2; ++j)
      simdgroup_load(acc[i][j], cs + (sm + i * 8) * TP + sp + j * 8, TP);
  for (uint k0 = 0; k0 < depth; k0 += TK) {
    for (uint i = local; i < TM * TK; i += 128) {
      const uint r = m0 + i / TK, k = k0 + i % TK;
      ws[i] = (r < rows && k < depth) ? (lda == 0u ? a[r * depth + k] : a[k * lda + r]) : 0.0f;
      const uint xk = k0 + i / TP, n = n0 + i % TP;
      xs[i] = (xk < depth && n < columns) ? b[xk * columns + n] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 wf[2], xf[2];
    for (uint i = 0; i < 2; ++i) simdgroup_load(wf[i], ws + (sm + i * 8) * TK, TK);
    for (uint j = 0; j < 2; ++j) simdgroup_load(xf[j], xs + sp + j * 8, TP);
    for (uint i = 0; i < 2; ++i)
      for (uint j = 0; j < 2; ++j)
        simdgroup_multiply_accumulate(acc[i][j], wf[i], xf[j], acc[i][j]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 2; ++i)
    for (uint j = 0; j < 2; ++j)
      simdgroup_store(acc[i][j], cs + (sm + i * 8) * TP + sp + j * 8, TP);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool swish = (params.mode & 0xffffu) == 25u;
  for (uint i = local; i < TM * TP; i += 128) {
    const uint r = m0 + i / TP, n = n0 + i % TP;
    if (r >= rows || n >= columns) continue;
    float value = cs[i];
    if (swish) value *= 1.0f / (1.0f + exp(-value));
    d[r * columns + n] = value;
  }
}
)MSL";

}  // namespace ppocr::detail
