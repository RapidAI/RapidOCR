#include "kernels.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <vector>

namespace ppocr::detail::kernels {

void Avx2WidenU8ToFloat(float* dst, const std::uint8_t* src, std::size_t n) noexcept {
  std::size_t index = 0;
  for (; index + 8 <= n; index += 8) {
    const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + index));
    const __m256i expanded = _mm256_cvtepu8_epi32(bytes);
    _mm256_storeu_ps(dst + index, _mm256_cvtepi32_ps(expanded));
  }
  for (; index < n; ++index) dst[index] = static_cast<float>(src[index]);
}

int Avx2ArgMax(const float* values, int count) noexcept {
  if (!values || count <= 0) return -1;
  if (count < 8) {
    int best = 0;
    for (int i = 1; i < count; ++i) if (values[i] > values[best]) best = i;
    return best;
  }
  __m256 maximum = _mm256_loadu_ps(values);
  __m256i maximum_index = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  int index = 8;
  bool has_nan = _mm256_movemask_ps(_mm256_cmp_ps(maximum, maximum, _CMP_UNORD_Q)) != 0;
  for (; index + 8 <= count; index += 8) {
    const __m256 current = _mm256_loadu_ps(values + index);
    has_nan |= _mm256_movemask_ps(_mm256_cmp_ps(current, current, _CMP_UNORD_Q)) != 0;
    const __m256 replace = _mm256_cmp_ps(current, maximum, _CMP_GT_OQ);
    maximum = _mm256_blendv_ps(maximum, current, replace);
    maximum_index = _mm256_blendv_epi8(
        maximum_index,
        _mm256_add_epi32(_mm256_set1_epi32(index),
                          _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)),
        _mm256_castps_si256(replace));
  }
  if (has_nan) {
    int best = 0;
    for (int i = 1; i < count; ++i) if (values[i] > values[best]) best = i;
    return best;
  }
  alignas(32) float lanes[8];
  alignas(32) int lane_indices[8];
  _mm256_store_ps(lanes, maximum);
  _mm256_store_si256(reinterpret_cast<__m256i*>(lane_indices), maximum_index);
  float maximum_scalar = lanes[0];
  int best = lane_indices[0];
  for (int lane = 1; lane < 8; ++lane) {
    if (lanes[lane] > maximum_scalar ||
        (lanes[lane] == maximum_scalar && lane_indices[lane] < best)) {
      maximum_scalar = lanes[lane];
      best = lane_indices[lane];
    }
  }
  for (; index < count; ++index) {
    if (values[index] > maximum_scalar) {
      maximum_scalar = values[index];
      best = index;
    }
  }
  return best;
}

void Avx2Binary(float* dst, const float* a, const float* b, std::size_t n,
                BinaryOp op) noexcept {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 x = _mm256_loadu_ps(a + i), y = _mm256_loadu_ps(b + i);
    __m256 z = _mm256_setzero_ps();
    switch (op) {
      case BinaryOp::add: z = _mm256_add_ps(x, y); break;
      case BinaryOp::sub: z = _mm256_sub_ps(x, y); break;
      case BinaryOp::mul: z = _mm256_mul_ps(x, y); break;
      case BinaryOp::div: z = _mm256_div_ps(x, y); break;
    }
    _mm256_storeu_ps(dst + i, z);
  }
  for (; i < n; ++i) {
    switch (op) {
      case BinaryOp::add: dst[i] = a[i] + b[i]; break;
      case BinaryOp::sub: dst[i] = a[i] - b[i]; break;
      case BinaryOp::mul: dst[i] = a[i] * b[i]; break;
      case BinaryOp::div: dst[i] = a[i] / b[i]; break;
    }
  }
}

void Avx2NearestResize2xAdd(float* dst, const float* src, const float* residual,
                            int batches, int channels, int input_height,
                            int input_width) noexcept {
  const int output_width = input_width * 2;
  const std::size_t input_plane = std::size_t(input_height) * input_width;
  const std::size_t output_plane = input_plane * 4;
  for (int plane = 0; plane < batches * channels; ++plane) {
    const float* input = src + std::size_t(plane) * input_plane;
    const float* add = residual + std::size_t(plane) * output_plane;
    float* output = dst + std::size_t(plane) * output_plane;
    for (int y = 0; y < input_height; ++y) {
      const float* input_row = input + std::size_t(y) * input_width;
      const float* add0 = add + std::size_t(y * 2) * output_width;
      const float* add1 = add0 + output_width;
      float* output0 = output + std::size_t(y * 2) * output_width;
      float* output1 = output0 + output_width;
      int x = 0;
      for (; x + 8 <= input_width; x += 8) {
        const __m256 values = _mm256_loadu_ps(input_row + x);
        const __m256 low = _mm256_unpacklo_ps(values, values);
        const __m256 high = _mm256_unpackhi_ps(values, values);
        const __m256 repeated0 = _mm256_permute2f128_ps(low, high, 0x20);
        const __m256 repeated1 = _mm256_permute2f128_ps(low, high, 0x31);
        const int offset = x * 2;
        const __m256 sum0 = _mm256_add_ps(repeated0, _mm256_loadu_ps(add0 + offset));
        const __m256 sum1 = _mm256_add_ps(repeated1, _mm256_loadu_ps(add0 + offset + 8));
        _mm256_storeu_ps(output0 + offset, sum0);
        _mm256_storeu_ps(output0 + offset + 8, sum1);
        _mm256_storeu_ps(output1 + offset, _mm256_add_ps(repeated0, _mm256_loadu_ps(add1 + offset)));
        _mm256_storeu_ps(output1 + offset + 8, _mm256_add_ps(repeated1, _mm256_loadu_ps(add1 + offset + 8)));
      }
      for (; x < input_width; ++x) {
        const float value = input_row[x]; const int offset = x * 2;
        output0[offset] = value + add0[offset]; output0[offset + 1] = value + add0[offset + 1];
        output1[offset] = value + add1[offset]; output1[offset + 1] = value + add1[offset + 1];
      }
    }
  }
}

void Avx2Relu(float* dst, const float* src, std::size_t n) noexcept {
  const __m256 zero = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) _mm256_storeu_ps(dst + i, _mm256_max_ps(_mm256_loadu_ps(src + i), zero));
  for (; i < n; ++i) dst[i] = std::max(src[i], 0.F);
}

void Avx2Gelu(float* dst, const float* src, std::size_t n) noexcept {
  const __m256 half = _mm256_set1_ps(.5F);
  const __m256 one = _mm256_set1_ps(1.F);
  const __m256 c0 = _mm256_set1_ps(0.7978845608028654F);
  const __m256 c1 = _mm256_set1_ps(0.044715F);
  const __m256 p27 = _mm256_set1_ps(27.F);
  const __m256 p9 = _mm256_set1_ps(9.F);
  const __m256 negative_one = _mm256_set1_ps(-1.F);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 x = _mm256_loadu_ps(src + i);
    const __m256 x3 = _mm256_mul_ps(x, _mm256_mul_ps(x, x));
    const __m256 z = _mm256_mul_ps(c0, _mm256_fmadd_ps(c1, x3, x));
    const __m256 z2 = _mm256_mul_ps(z, z);
    // Match the AVX-512 BatchNorm+GELU Padé approximation.  The previous
    // AVX2 generic path stored every vector to scalar tanh calls, defeating
    // the point of the opt-in vector dispatch on AVX2-only machines.
    const __m256 t = _mm256_max_ps(negative_one, _mm256_min_ps(one,
        _mm256_div_ps(_mm256_mul_ps(z, _mm256_add_ps(p27, z2)),
                      _mm256_add_ps(p27, _mm256_mul_ps(p9, z2)))));
    _mm256_storeu_ps(dst + i, _mm256_mul_ps(half, _mm256_mul_ps(x, _mm256_add_ps(one, t))));
  }
  constexpr float c0s = .7978845608028654F, c1s = .044715F;
  for (; i < n; ++i) {
    const float x = src[i];
    const float z = c0s * (x + c1s * x * x * x);
    const float z2 = z * z;
    const float t = std::clamp(z * (27.F + z2) / (27.F + 9.F * z2), -1.F, 1.F);
    dst[i] = .5F * x * (1.F + t);
  }
}

namespace {

__m256 ExpPs(__m256 x) noexcept {
  const __m256 min_x = _mm256_set1_ps(-87.F);
  const __m256 max_x = _mm256_set1_ps(87.F);
  const __m256 inv_ln2 = _mm256_set1_ps(1.4426950408889634F);
  const __m256 ln2 = _mm256_set1_ps(0.6931471805599453F);
  x = _mm256_min_ps(_mm256_max_ps(x, min_x), max_x);
  const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, inv_ln2),
                                   _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  const __m256 f = _mm256_fnmadd_ps(n, ln2, x);
  __m256 p = _mm256_fmadd_ps(f, _mm256_set1_ps(1.F / 24.F), _mm256_set1_ps(1.F / 6.F));
  p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(.5F));
  p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.F));
  p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.F));
  const __m256i two_n = _mm256_slli_epi32(
      _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(p, _mm256_castsi256_ps(two_n));
}

__m256 ErfPs(__m256 x) noexcept {
  const __m256 zero = _mm256_setzero_ps();
  const __m256 one = _mm256_set1_ps(1.F);
  const __m256 half = _mm256_set1_ps(.5F);
  const __m256 sign = _mm256_or_ps(_mm256_and_ps(x, _mm256_set1_ps(-0.F)), one);
  const __m256 ax = _mm256_andnot_ps(_mm256_set1_ps(-0.F), x);
  const __m256 t = _mm256_div_ps(one, _mm256_fmadd_ps(half, ax, one));
  __m256 poly = _mm256_fmadd_ps(t, _mm256_set1_ps(.17087277F), _mm256_set1_ps(-.82215223F));
  poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(1.48851587F));
  poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(-1.13520398F));
  poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(.27886807F));
  poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(-.18628806F));
  poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(.09678418F));
  poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(.37409196F));
  poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(1.00002368F));
  poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(-1.26551223F));
  const __m256 e = ExpPs(_mm256_fmadd_ps(_mm256_mul_ps(ax, ax),
                                         _mm256_set1_ps(-1.F), poly));
  const __m256 magnitude = _mm256_sub_ps(one, _mm256_mul_ps(t, e));
  (void)zero;
  return _mm256_mul_ps(sign, magnitude);
}

}  // namespace

void Avx2Sigmoid(float* dst, const float* src, std::size_t n) noexcept {
  const __m256 one = _mm256_set1_ps(1.F);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 e = ExpPs(_mm256_sub_ps(_mm256_setzero_ps(), _mm256_loadu_ps(src + i)));
    _mm256_storeu_ps(dst + i, _mm256_div_ps(one, _mm256_add_ps(one, e)));
  }
  for (; i < n; ++i) dst[i] = 1.F / (1.F + std::exp(-src[i]));
}

void Avx2ExactGelu(float* dst, const float* src, std::size_t n) noexcept {
  const __m256 half = _mm256_set1_ps(.5F);
  const __m256 one = _mm256_set1_ps(1.F);
  const __m256 inv_sqrt2 = _mm256_set1_ps(0.7071067811865475244F);
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m256 x0 = _mm256_loadu_ps(src + i);
    const __m256 x1 = _mm256_loadu_ps(src + i + 8);
    const __m256 y0 = _mm256_mul_ps(half, _mm256_mul_ps(x0,
        _mm256_add_ps(one, ErfPs(_mm256_mul_ps(x0, inv_sqrt2)))));
    const __m256 y1 = _mm256_mul_ps(half, _mm256_mul_ps(x1,
        _mm256_add_ps(one, ErfPs(_mm256_mul_ps(x1, inv_sqrt2)))));
    _mm256_storeu_ps(dst + i, y0);
    _mm256_storeu_ps(dst + i + 8, y1);
  }
  for (; i + 8 <= n; i += 8) {
    const __m256 x = _mm256_loadu_ps(src + i);
    const __m256 gelu = _mm256_mul_ps(half, _mm256_mul_ps(x,
        _mm256_add_ps(one, ErfPs(_mm256_mul_ps(x, inv_sqrt2)))));
    _mm256_storeu_ps(dst + i, gelu);
  }
  constexpr float inv_sqrt2s = 0.7071067811865475244F;
  for (; i < n; ++i) {
    const float x = src[i];
    dst[i] = x * .5F * (1.F + std::erf(x * inv_sqrt2s));
  }
}

void Avx2BatchNormGelu(float* dst, const float* src, std::size_t n,
                       float scale, float shift) noexcept {
  const __m256 affine_scale = _mm256_set1_ps(scale);
  const __m256 affine_shift = _mm256_set1_ps(shift);
  const __m256 half = _mm256_set1_ps(.5F);
  const __m256 one = _mm256_set1_ps(1.F);
  const __m256 c0 = _mm256_set1_ps(.7978845608028654F);
  const __m256 c1 = _mm256_set1_ps(.044715F);
  const __m256 p27 = _mm256_set1_ps(27.F);
  const __m256 p9 = _mm256_set1_ps(9.F);
  const __m256 negative_one = _mm256_set1_ps(-1.F);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 x = _mm256_fmadd_ps(_mm256_loadu_ps(src + i), affine_scale,
                                     affine_shift);
    const __m256 x3 = _mm256_mul_ps(x, _mm256_mul_ps(x, x));
    const __m256 z = _mm256_mul_ps(c0, _mm256_fmadd_ps(c1, x3, x));
    const __m256 z2 = _mm256_mul_ps(z, z);
    const __m256 t = _mm256_max_ps(negative_one, _mm256_min_ps(one,
        _mm256_div_ps(_mm256_mul_ps(z, _mm256_add_ps(p27, z2)),
                      _mm256_add_ps(p27, _mm256_mul_ps(p9, z2)))));
    _mm256_storeu_ps(dst + i, _mm256_mul_ps(half, _mm256_mul_ps(x,
        _mm256_add_ps(one, t))));
  }
  constexpr float c0s = .7978845608028654F, c1s = .044715F;
  for (; i < n; ++i) {
    const float x = src[i] * scale + shift;
    const float z = c0s * (x + c1s * x * x * x);
    const float z2 = z * z;
    const float t = std::clamp(z * (27.F + z2) / (27.F + 9.F * z2), -1.F, 1.F);
    dst[i] = .5F * x * (1.F + t);
  }
}

void Avx2HardSwish(float* dst, const float* src, std::size_t n) noexcept {
  const __m256 sixth = _mm256_set1_ps(1.F / 6.F), half = _mm256_set1_ps(.5F);
  const __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.F);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 x = _mm256_loadu_ps(src + i);
    const __m256 gate = _mm256_min_ps(one, _mm256_max_ps(zero, _mm256_fmadd_ps(x, sixth, half)));
    _mm256_storeu_ps(dst + i, _mm256_mul_ps(x, gate));
  }
  for (; i < n; ++i) { const float x = src[i]; dst[i] = x * std::clamp(x / 6.F + .5F, 0.F, 1.F); }
}

void Avx2ScaleShift(float* dst, const float* src, std::size_t n, float scale,
                    float shift) noexcept {
  const __m256 s = _mm256_set1_ps(scale), b = _mm256_set1_ps(shift);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(_mm256_loadu_ps(src+i), s, b));
  for (; i < n; ++i) dst[i] = src[i] * scale + shift;
}

void Avx2BinaryScalar(float* dst, const float* src, std::size_t n, float scalar,
                      BinaryOp op, bool scalar_left) noexcept {
  const __m256 s = _mm256_set1_ps(scalar);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 x = _mm256_loadu_ps(src + i);
    __m256 z = x;
    switch (op) {
      case BinaryOp::add: z = _mm256_add_ps(x, s); break;
      case BinaryOp::sub: z = scalar_left ? _mm256_sub_ps(s, x) : _mm256_sub_ps(x, s); break;
      case BinaryOp::mul: z = _mm256_mul_ps(x, s); break;
      case BinaryOp::div: z = scalar_left ? _mm256_div_ps(s, x) : _mm256_div_ps(x, s); break;
    }
    _mm256_storeu_ps(dst + i, z);
  }
  for (; i < n; ++i) {
    switch (op) {
      case BinaryOp::add: dst[i] = src[i] + scalar; break;
      case BinaryOp::sub: dst[i] = scalar_left ? scalar - src[i] : src[i] - scalar; break;
      case BinaryOp::mul: dst[i] = src[i] * scalar; break;
      case BinaryOp::div: dst[i] = scalar_left ? scalar / src[i] : src[i] / scalar; break;
    }
  }
}

void Avx2Axpy(float* dst, const float* src, float alpha, std::size_t n) noexcept {
  const __m256 a = _mm256_set1_ps(alpha);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(a, _mm256_loadu_ps(src + i),
                                               _mm256_loadu_ps(dst + i)));
  }
  for (; i < n; ++i) dst[i] += alpha * src[i];
}

// Evaluate four output channels together.  1x1 NCHW convolution has the
// same input plane for each output channel; sharing that load cuts input
// bandwidth by four while retaining each channel's original accumulation
// order and therefore its inference numerics.
// Register-tiled 1x1 convolution. Each output vector remains live for the
// complete input-channel reduction, avoiding the former load/FMA/store of
// every output element for every input channel. Output-channel and reduction
// order are unchanged, so it remains an exact execution-path optimization.
void Avx2PointwiseConv4(float* dst, const float* src, const float* weights,
                         const float* bias, int first_output, int last_output,
                         int input_channels, std::size_t plane,
                         std::size_t index_begin, std::size_t index_end) noexcept {
  if (index_end == 0 || index_end > plane) index_end = plane;
  if (index_begin > index_end) index_begin = index_end;
  int output = first_output;
  for (; output + 4 <= last_output; output += 4) {
    float* out0 = dst + std::size_t(output) * plane;
    float* out1 = out0 + plane;
    float* out2 = out1 + plane;
    float* out3 = out2 + plane;
    const float* w0 = weights + std::size_t(output) * input_channels;
    const float* w1 = w0 + input_channels;
    const float* w2 = w1 + input_channels;
    const float* w3 = w2 + input_channels;
    const __m256 vb0 = _mm256_set1_ps(bias ? bias[output] : 0.F);
    const __m256 vb1 = _mm256_set1_ps(bias ? bias[output + 1] : 0.F);
    const __m256 vb2 = _mm256_set1_ps(bias ? bias[output + 2] : 0.F);
    const __m256 vb3 = _mm256_set1_ps(bias ? bias[output + 3] : 0.F);
    std::size_t index = index_begin;
    for (; index < index_end && (index & 7) != 0; ++index) {
      float sum0 = bias ? bias[output] : 0.F;
      float sum1 = bias ? bias[output + 1] : 0.F;
      float sum2 = bias ? bias[output + 2] : 0.F;
      float sum3 = bias ? bias[output + 3] : 0.F;
      for (int input = 0; input < input_channels; ++input) {
        const float x = src[std::size_t(input) * plane + index];
        sum0 += w0[input] * x; sum1 += w1[input] * x;
        sum2 += w2[input] * x; sum3 += w3[input] * x;
      }
      out0[index] = sum0; out1[index] = sum1; out2[index] = sum2; out3[index] = sum3;
    }
    for (; index + 8 <= index_end; index += 8) {
      __m256 sum0 = vb0;
      __m256 sum1 = vb1;
      __m256 sum2 = vb2;
      __m256 sum3 = vb3;
      for (int input = 0; input < input_channels; ++input) {
        const __m256 x = _mm256_loadu_ps(src + std::size_t(input) * plane + index);
        sum0 = _mm256_fmadd_ps(_mm256_set1_ps(w0[input]), x, sum0);
        sum1 = _mm256_fmadd_ps(_mm256_set1_ps(w1[input]), x, sum1);
        sum2 = _mm256_fmadd_ps(_mm256_set1_ps(w2[input]), x, sum2);
        sum3 = _mm256_fmadd_ps(_mm256_set1_ps(w3[input]), x, sum3);
      }
      _mm256_storeu_ps(out0 + index, sum0);
      _mm256_storeu_ps(out1 + index, sum1);
      _mm256_storeu_ps(out2 + index, sum2);
      _mm256_storeu_ps(out3 + index, sum3);
    }
    for (; index < index_end; ++index) {
      float sum0 = bias ? bias[output] : 0.F;
      float sum1 = bias ? bias[output + 1] : 0.F;
      float sum2 = bias ? bias[output + 2] : 0.F;
      float sum3 = bias ? bias[output + 3] : 0.F;
      for (int input = 0; input < input_channels; ++input) {
        const float x = src[std::size_t(input) * plane + index];
        sum0 += w0[input] * x; sum1 += w1[input] * x;
        sum2 += w2[input] * x; sum3 += w3[input] * x;
      }
      out0[index] = sum0; out1[index] = sum1; out2[index] = sum2; out3[index] = sum3;
    }
  }
  for (; output < last_output; ++output) {
    float* out = dst + std::size_t(output) * plane;
    const float* filter = weights + std::size_t(output) * input_channels;
    const __m256 vb = _mm256_set1_ps(bias ? bias[output] : 0.F);
    std::size_t index = index_begin;
    for (; index < index_end && (index & 7) != 0; ++index) {
      float sum = bias ? bias[output] : 0.F;
      for (int input = 0; input < input_channels; ++input)
        sum += filter[input] * src[std::size_t(input) * plane + index];
      out[index] = sum;
    }
    for (; index + 8 <= index_end; index += 8) {
      __m256 sum = vb;
      for (int input = 0; input < input_channels; ++input) {
        sum = _mm256_fmadd_ps(_mm256_set1_ps(filter[input]),
                            _mm256_loadu_ps(src + std::size_t(input) * plane + index), sum);
      }
      _mm256_storeu_ps(out + index, sum);
    }
    for (; index < index_end; ++index) {
      float sum = bias ? bias[output] : 0.F;
      for (int input = 0; input < input_channels; ++input)
        sum += filter[input] * src[std::size_t(input) * plane + index];
      out[index] = sum;
    }
  }
}

void Avx2Square(float* dst, const float* src, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 value = _mm256_loadu_ps(src + i);
    _mm256_storeu_ps(dst + i, _mm256_mul_ps(value, value));
  }
  for (; i < n; ++i) dst[i] = src[i] * src[i];
}

void Avx2HardSigmoid(float* dst, const float* src, std::size_t n, float alpha,
                      float beta) noexcept {
  const __m256 a = _mm256_set1_ps(alpha), b = _mm256_set1_ps(beta);
  const __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.F);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 value = _mm256_fmadd_ps(_mm256_loadu_ps(src + i), a, b);
    _mm256_storeu_ps(dst + i, _mm256_min_ps(_mm256_max_ps(value, zero), one));
  }
  for (; i < n; ++i) dst[i] = std::clamp(alpha * src[i] + beta, 0.F, 1.F);
}

// AVX2 counterpart of the AVX-512 residual projection fusion.  The add is
// performed only after the complete convolution reduction, exactly matching
// the existing Conv -> Add graph while removing the intermediate traversal.
void Avx2PointwiseConvAdd4(float* dst, const float* src, const float* weights,
                           const float* bias, const float* residual,
                           int first_output, int last_output,
                           int input_channels, std::size_t plane) noexcept {
  int output = first_output;
  for (; output + 4 <= last_output; output += 4) {
    float* o0 = dst + std::size_t(output) * plane;
    float* o1 = o0 + plane; float* o2 = o1 + plane; float* o3 = o2 + plane;
    const float* r0 = residual + std::size_t(output) * plane;
    const float* r1 = r0 + plane; const float* r2 = r1 + plane; const float* r3 = r2 + plane;
    const float* w0 = weights + std::size_t(output) * input_channels;
    const float* w1 = w0 + input_channels; const float* w2 = w1 + input_channels;
    const float* w3 = w2 + input_channels;
    std::size_t index = 0;
    for (; index + 8 <= plane; index += 8) {
      __m256 s0 = _mm256_set1_ps(bias ? bias[output] : 0.F);
      __m256 s1 = _mm256_set1_ps(bias ? bias[output + 1] : 0.F);
      __m256 s2 = _mm256_set1_ps(bias ? bias[output + 2] : 0.F);
      __m256 s3 = _mm256_set1_ps(bias ? bias[output + 3] : 0.F);
      for (int input = 0; input < input_channels; ++input) {
        const __m256 x = _mm256_loadu_ps(src + std::size_t(input) * plane + index);
        s0 = _mm256_fmadd_ps(_mm256_set1_ps(w0[input]), x, s0);
        s1 = _mm256_fmadd_ps(_mm256_set1_ps(w1[input]), x, s1);
        s2 = _mm256_fmadd_ps(_mm256_set1_ps(w2[input]), x, s2);
        s3 = _mm256_fmadd_ps(_mm256_set1_ps(w3[input]), x, s3);
      }
      _mm256_storeu_ps(o0 + index, _mm256_add_ps(s0, _mm256_loadu_ps(r0 + index)));
      _mm256_storeu_ps(o1 + index, _mm256_add_ps(s1, _mm256_loadu_ps(r1 + index)));
      _mm256_storeu_ps(o2 + index, _mm256_add_ps(s2, _mm256_loadu_ps(r2 + index)));
      _mm256_storeu_ps(o3 + index, _mm256_add_ps(s3, _mm256_loadu_ps(r3 + index)));
    }
    for (; index < plane; ++index) {
      float s0 = bias ? bias[output] : 0.F, s1 = bias ? bias[output + 1] : 0.F;
      float s2 = bias ? bias[output + 2] : 0.F, s3 = bias ? bias[output + 3] : 0.F;
      for (int input = 0; input < input_channels; ++input) {
        const float x = src[std::size_t(input) * plane + index];
        s0 += w0[input] * x; s1 += w1[input] * x;
        s2 += w2[input] * x; s3 += w3[input] * x;
      }
      o0[index] = s0 + r0[index]; o1[index] = s1 + r1[index];
      o2[index] = s2 + r2[index]; o3[index] = s3 + r3[index];
    }
  }
  for (; output < last_output; ++output) {
    float* out = dst + std::size_t(output) * plane;
    const float* add = residual + std::size_t(output) * plane;
    const float* filter = weights + std::size_t(output) * input_channels;
    std::size_t index = 0;
    for (; index + 8 <= plane; index += 8) {
      __m256 sum = _mm256_set1_ps(bias ? bias[output] : 0.F);
      for (int input = 0; input < input_channels; ++input) {
        sum = _mm256_fmadd_ps(_mm256_set1_ps(filter[input]),
                               _mm256_loadu_ps(src + std::size_t(input) * plane + index), sum);
      }
      _mm256_storeu_ps(out + index, _mm256_add_ps(sum, _mm256_loadu_ps(add + index)));
    }
    for (; index < plane; ++index) {
      float sum = bias ? bias[output] : 0.F;
      for (int input = 0; input < input_channels; ++input)
        sum += filter[input] * src[std::size_t(input) * plane + index];
      out[index] = sum + add[index];
    }
  }
}
void Avx2PointwiseConvAddRelu4(float* dst, const float* src, const float* weights,
                               const float* bias, const float* residual,
                               int first_output, int last_output,
                               int input_channels, std::size_t plane) noexcept {
  const __m256 zero = _mm256_setzero_ps();
  int output = first_output;
  for (; output + 4 <= last_output; output += 4) {
    float* o0 = dst + std::size_t(output) * plane;
    float* o1 = o0 + plane; float* o2 = o1 + plane; float* o3 = o2 + plane;
    const float* r0 = residual + std::size_t(output) * plane;
    const float* r1 = r0 + plane; const float* r2 = r1 + plane; const float* r3 = r2 + plane;
    const float* w0 = weights + std::size_t(output) * input_channels;
    const float* w1 = w0 + input_channels; const float* w2 = w1 + input_channels;
    const float* w3 = w2 + input_channels;
    std::size_t index = 0;
    for (; index + 8 <= plane; index += 8) {
      __m256 s0 = _mm256_set1_ps(bias ? bias[output] : 0.F);
      __m256 s1 = _mm256_set1_ps(bias ? bias[output + 1] : 0.F);
      __m256 s2 = _mm256_set1_ps(bias ? bias[output + 2] : 0.F);
      __m256 s3 = _mm256_set1_ps(bias ? bias[output + 3] : 0.F);
      for (int input = 0; input < input_channels; ++input) {
        const __m256 x = _mm256_loadu_ps(src + std::size_t(input) * plane + index);
        s0 = _mm256_fmadd_ps(_mm256_set1_ps(w0[input]), x, s0);
        s1 = _mm256_fmadd_ps(_mm256_set1_ps(w1[input]), x, s1);
        s2 = _mm256_fmadd_ps(_mm256_set1_ps(w2[input]), x, s2);
        s3 = _mm256_fmadd_ps(_mm256_set1_ps(w3[input]), x, s3);
      }
      _mm256_storeu_ps(o0 + index, _mm256_max_ps(_mm256_add_ps(s0, _mm256_loadu_ps(r0 + index)), zero));
      _mm256_storeu_ps(o1 + index, _mm256_max_ps(_mm256_add_ps(s1, _mm256_loadu_ps(r1 + index)), zero));
      _mm256_storeu_ps(o2 + index, _mm256_max_ps(_mm256_add_ps(s2, _mm256_loadu_ps(r2 + index)), zero));
      _mm256_storeu_ps(o3 + index, _mm256_max_ps(_mm256_add_ps(s3, _mm256_loadu_ps(r3 + index)), zero));
    }
    for (; index < plane; ++index) {
      float s0 = bias ? bias[output] : 0.F, s1 = bias ? bias[output + 1] : 0.F;
      float s2 = bias ? bias[output + 2] : 0.F, s3 = bias ? bias[output + 3] : 0.F;
      for (int input = 0; input < input_channels; ++input) {
        const float x = src[std::size_t(input) * plane + index];
        s0 += w0[input] * x; s1 += w1[input] * x;
        s2 += w2[input] * x; s3 += w3[input] * x;
      }
      o0[index] = std::max(s0 + r0[index], 0.F); o1[index] = std::max(s1 + r1[index], 0.F);
      o2[index] = std::max(s2 + r2[index], 0.F); o3[index] = std::max(s3 + r3[index], 0.F);
    }
  }
  for (; output < last_output; ++output) {
    float* out = dst + std::size_t(output) * plane;
    const float* filter = weights + std::size_t(output) * input_channels;
    const float* add = residual + std::size_t(output) * plane;
    for (std::size_t index = 0; index < plane; ++index) {
      float sum = bias ? bias[output] : 0.F;
      for (int input = 0; input < input_channels; ++input) sum += filter[input] * src[std::size_t(input) * plane + index];
      out[index] = std::max(sum + add[index], 0.F);
    }
  }
}

void Avx2ConvTranspose2x2(float* dst, const float* src, const float* weights,
                           const float* bias, int first_output, int last_output,
                           int input_channels, int output_channels, int input_h, int input_w,
                           int first_y, int last_y) noexcept {
  const int output_w = input_w * 2;
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(input_h * 2) * output_w;
  if (last_y < 0) last_y = input_h;
  if (first_y < 0) first_y = 0;
  if (last_y > input_h) last_y = input_h;
  if (first_y >= last_y) return;
  const __m256i duplicate_low = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
  const __m256i duplicate_high = _mm256_setr_epi32(4, 4, 5, 5, 6, 6, 7, 7);
  // Page-scale FPN maps spill L2. Accumulating across input channels keeps
  // one store per output instead of rewriting the plane once per channel.
  // Small maps stay on the RMW loop. `PPOCR_DISABLE_AVX2_TRANSPOSE_ACC`
  // restores that loop at every size.
  static const bool acc_disabled =
      std::getenv("PPOCR_DISABLE_AVX2_TRANSPOSE_ACC") != nullptr;
  if (!acc_disabled && output_plane >= 65536) {
    for (int output = first_output; output < last_output; ++output) {
      float* out = dst + std::size_t(output) * output_plane;
      const float base = bias ? bias[output] : 0.F;
      const __m256 biasv = _mm256_set1_ps(base);
      for (int y = first_y; y < last_y; ++y) {
        float* out0 = out + std::size_t(2 * y) * output_w;
        float* out1 = out0 + output_w;
        int x = 0;
        for (; x + 8 <= input_w; x += 8) {
          __m256 top_lo = biasv, top_hi = biasv, bot_lo = biasv, bot_hi = biasv;
          for (int input = 0; input < input_channels; ++input) {
            const float* row = src + std::size_t(input) * input_plane +
                               std::size_t(y) * input_w + x;
            const float* weight =
                weights + (std::size_t(input) * output_channels + output) * 4;
            const __m256 top = _mm256_setr_ps(weight[0], weight[1], weight[0], weight[1],
                                               weight[0], weight[1], weight[0], weight[1]);
            const __m256 bottom = _mm256_setr_ps(weight[2], weight[3], weight[2], weight[3],
                                                  weight[2], weight[3], weight[2], weight[3]);
            const __m256 input_values = _mm256_loadu_ps(row);
            const __m256 lo = _mm256_permutevar8x32_ps(input_values, duplicate_low);
            const __m256 hi = _mm256_permutevar8x32_ps(input_values, duplicate_high);
            top_lo = _mm256_fmadd_ps(lo, top, top_lo);
            top_hi = _mm256_fmadd_ps(hi, top, top_hi);
            bot_lo = _mm256_fmadd_ps(lo, bottom, bot_lo);
            bot_hi = _mm256_fmadd_ps(hi, bottom, bot_hi);
          }
          const int xx = x * 2;
          _mm256_storeu_ps(out0 + xx, top_lo);
          _mm256_storeu_ps(out0 + xx + 8, top_hi);
          _mm256_storeu_ps(out1 + xx, bot_lo);
          _mm256_storeu_ps(out1 + xx + 8, bot_hi);
        }
        for (; x < input_w; ++x) {
          float t0 = base, t1 = base, b0 = base, b1 = base;
          for (int input = 0; input < input_channels; ++input) {
            const float value =
                src[std::size_t(input) * input_plane + std::size_t(y) * input_w + x];
            const float* weight =
                weights + (std::size_t(input) * output_channels + output) * 4;
            t0 += value * weight[0];
            t1 += value * weight[1];
            b0 += value * weight[2];
            b1 += value * weight[3];
          }
          const int xx = x * 2;
          out0[xx] = t0;
          out0[xx + 1] = t1;
          out1[xx] = b0;
          out1[xx + 1] = b1;
        }
      }
    }
    return;
  }
  for (int output = first_output; output < last_output; ++output) {
    float* out = dst + std::size_t(output) * output_plane;
    const int row_pairs = last_y - first_y;
    std::fill_n(out + std::size_t(first_y * 2) * output_w,
                std::size_t(row_pairs) * 2 * output_w, bias ? bias[output] : 0.F);
    for (int input = 0; input < input_channels; ++input) {
      const float* values = src + std::size_t(input) * input_plane;
      const float* weight = weights + (std::size_t(input) * output_channels + output) * 4;
      const __m256 top = _mm256_setr_ps(weight[0], weight[1], weight[0], weight[1],
                                         weight[0], weight[1], weight[0], weight[1]);
      const __m256 bottom = _mm256_setr_ps(weight[2], weight[3], weight[2], weight[3],
                                            weight[2], weight[3], weight[2], weight[3]);
      for (int y = first_y; y < last_y; ++y) {
        const float* row = values + std::size_t(y) * input_w;
        float* out0 = out + std::size_t(2 * y) * output_w;
        float* out1 = out0 + output_w;
        int x = 0;
        for (; x + 8 <= input_w; x += 8) {
          const __m256 input_values = _mm256_loadu_ps(row + x);
          const __m256 lo = _mm256_permutevar8x32_ps(input_values, duplicate_low);
          const __m256 hi = _mm256_permutevar8x32_ps(input_values, duplicate_high);
          const int xx = x * 2;
          _mm256_storeu_ps(out0 + xx, _mm256_fmadd_ps(lo, top, _mm256_loadu_ps(out0 + xx)));
          _mm256_storeu_ps(out0 + xx + 8, _mm256_fmadd_ps(hi, top, _mm256_loadu_ps(out0 + xx + 8)));
          _mm256_storeu_ps(out1 + xx, _mm256_fmadd_ps(lo, bottom, _mm256_loadu_ps(out1 + xx)));
          _mm256_storeu_ps(out1 + xx + 8, _mm256_fmadd_ps(hi, bottom, _mm256_loadu_ps(out1 + xx + 8)));
        }
        for (; x < input_w; ++x) {
          const float value = row[x]; const int xx = x * 2;
          out0[xx] += value * weight[0]; out0[xx + 1] += value * weight[1];
          out1[xx] += value * weight[2]; out1[xx + 1] += value * weight[3];
        }
      }
    }
  }
}
void Avx2GemmRows(float* dst, const float* a, const float* b, const float* bias,
                  int first_row, int last_row, int cols, int depth) noexcept {
  // Pair independent projection rows so one immutable weight-vector load
  // feeds two accumulators. This is the AVX2 counterpart of the AVX-512
  // batch micro-kernel and keeps every row's FMA order unchanged.
  const auto one_row = [&](int row) noexcept {
    float* out = dst + std::size_t(row) * cols;
    const float* left = a + std::size_t(row) * depth;
    int col = 0;
    for (; col + 32 <= cols; col += 32) {
      __m256 c0 = bias ? _mm256_loadu_ps(bias + col) : _mm256_setzero_ps();
      __m256 c1 = bias ? _mm256_loadu_ps(bias + col + 8) : _mm256_setzero_ps();
      __m256 c2 = bias ? _mm256_loadu_ps(bias + col + 16) : _mm256_setzero_ps();
      __m256 c3 = bias ? _mm256_loadu_ps(bias + col + 24) : _mm256_setzero_ps();
      for (int k = 0; k < depth; ++k) {
        const __m256 aa = _mm256_set1_ps(left[k]);
        const float* right = b + std::size_t(k) * cols + col;
        c0 = _mm256_fmadd_ps(aa, _mm256_loadu_ps(right), c0);
        c1 = _mm256_fmadd_ps(aa, _mm256_loadu_ps(right + 8), c1);
        c2 = _mm256_fmadd_ps(aa, _mm256_loadu_ps(right + 16), c2);
        c3 = _mm256_fmadd_ps(aa, _mm256_loadu_ps(right + 24), c3);
      }
      _mm256_storeu_ps(out + col, c0); _mm256_storeu_ps(out + col + 8, c1);
      _mm256_storeu_ps(out + col + 16, c2); _mm256_storeu_ps(out + col + 24, c3);
    }
    for (; col + 8 <= cols; col += 8) {
      __m256 value = bias ? _mm256_loadu_ps(bias + col) : _mm256_setzero_ps();
      for (int k = 0; k < depth; ++k) {
        value = _mm256_fmadd_ps(_mm256_set1_ps(left[k]),
                                _mm256_loadu_ps(b + std::size_t(k) * cols + col), value);
      }
      _mm256_storeu_ps(out + col, value);
    }
    for (; col < cols; ++col) {
      float value = bias ? bias[col] : 0.F;
      for (int k = 0; k < depth; ++k) value += left[k] * b[std::size_t(k) * cols + col];
      out[col] = value;
    }
  };
  int row = first_row;
  for (; row + 1 < last_row; row += 2) {
    float* out0 = dst + std::size_t(row) * cols;
    float* out1 = out0 + cols;
    const float* left0 = a + std::size_t(row) * depth;
    const float* left1 = left0 + depth;
    int col = 0;
    for (; col + 32 <= cols; col += 32) {
      __m256 a00 = bias ? _mm256_loadu_ps(bias + col) : _mm256_setzero_ps();
      __m256 a01 = bias ? _mm256_loadu_ps(bias + col + 8) : _mm256_setzero_ps();
      __m256 a02 = bias ? _mm256_loadu_ps(bias + col + 16) : _mm256_setzero_ps();
      __m256 a03 = bias ? _mm256_loadu_ps(bias + col + 24) : _mm256_setzero_ps();
      __m256 a10 = a00, a11 = a01, a12 = a02, a13 = a03;
      for (int k = 0; k < depth; ++k) {
        const __m256 lhs0 = _mm256_set1_ps(left0[k]);
        const __m256 lhs1 = _mm256_set1_ps(left1[k]);
        const float* right = b + std::size_t(k) * cols + col;
        const __m256 b0 = _mm256_loadu_ps(right);
        const __m256 b1 = _mm256_loadu_ps(right + 8);
        const __m256 b2 = _mm256_loadu_ps(right + 16);
        const __m256 b3 = _mm256_loadu_ps(right + 24);
        a00 = _mm256_fmadd_ps(lhs0, b0, a00); a10 = _mm256_fmadd_ps(lhs1, b0, a10);
        a01 = _mm256_fmadd_ps(lhs0, b1, a01); a11 = _mm256_fmadd_ps(lhs1, b1, a11);
        a02 = _mm256_fmadd_ps(lhs0, b2, a02); a12 = _mm256_fmadd_ps(lhs1, b2, a12);
        a03 = _mm256_fmadd_ps(lhs0, b3, a03); a13 = _mm256_fmadd_ps(lhs1, b3, a13);
      }
      _mm256_storeu_ps(out0 + col, a00); _mm256_storeu_ps(out0 + col + 8, a01);
      _mm256_storeu_ps(out0 + col + 16, a02); _mm256_storeu_ps(out0 + col + 24, a03);
      _mm256_storeu_ps(out1 + col, a10); _mm256_storeu_ps(out1 + col + 8, a11);
      _mm256_storeu_ps(out1 + col + 16, a12); _mm256_storeu_ps(out1 + col + 24, a13);
    }
    for (; col + 8 <= cols; col += 8) {
      __m256 value0 = bias ? _mm256_loadu_ps(bias + col) : _mm256_setzero_ps();
      __m256 value1 = value0;
      for (int k = 0; k < depth; ++k) {
        const __m256 right = _mm256_loadu_ps(b + std::size_t(k) * cols + col);
        value0 = _mm256_fmadd_ps(_mm256_set1_ps(left0[k]), right, value0);
        value1 = _mm256_fmadd_ps(_mm256_set1_ps(left1[k]), right, value1);
      }
      _mm256_storeu_ps(out0 + col, value0);
      _mm256_storeu_ps(out1 + col, value1);
    }
    for (; col < cols; ++col) {
      float value0 = bias ? bias[col] : 0.F;
      float value1 = value0;
      for (int k = 0; k < depth; ++k) {
        const float weight = b[std::size_t(k) * cols + col];
        value0 += left0[k] * weight;
        value1 += left1[k] * weight;
      }
      out0[col] = value0;
      out1[col] = value1;
    }
  }
  if (row < last_row) one_row(row);
}

void Avx2GemmAccumulateRows(float* dst, const float* a, const float* b,
                            int first_row, int last_row, int cols, int depth) noexcept {
  // Same two-row B reuse as Avx2GemmRows. PPOCR_DISABLE_AVX2_GEMM_ACC2
  // restores the one-row walk. This TU is compiled without AVX-512 so an
  // AVX2-only machine can execute it.
  static const bool pair_rows =
      std::getenv("PPOCR_DISABLE_AVX2_GEMM_ACC2") == nullptr;
  int row = first_row;
  if (pair_rows) {
    for (; row + 1 < last_row; row += 2) {
      float* out0 = dst + std::size_t(row) * cols;
      float* out1 = out0 + cols;
      const float* left0 = a + std::size_t(row) * depth;
      const float* left1 = left0 + depth;
      int col = 0;
      for (; col + 32 <= cols; col += 32) {
        __m256 a0 = _mm256_loadu_ps(out0 + col);
        __m256 a1 = _mm256_loadu_ps(out0 + col + 8);
        __m256 a2 = _mm256_loadu_ps(out0 + col + 16);
        __m256 a3 = _mm256_loadu_ps(out0 + col + 24);
        __m256 b0 = _mm256_loadu_ps(out1 + col);
        __m256 b1 = _mm256_loadu_ps(out1 + col + 8);
        __m256 b2 = _mm256_loadu_ps(out1 + col + 16);
        __m256 b3 = _mm256_loadu_ps(out1 + col + 24);
        for (int k = 0; k < depth; ++k) {
          const __m256 l0 = _mm256_set1_ps(left0[k]);
          const __m256 l1 = _mm256_set1_ps(left1[k]);
          const float* right = b + std::size_t(k) * cols + col;
          const __m256 r0 = _mm256_loadu_ps(right);
          const __m256 r1 = _mm256_loadu_ps(right + 8);
          const __m256 r2 = _mm256_loadu_ps(right + 16);
          const __m256 r3 = _mm256_loadu_ps(right + 24);
          a0 = _mm256_fmadd_ps(l0, r0, a0); b0 = _mm256_fmadd_ps(l1, r0, b0);
          a1 = _mm256_fmadd_ps(l0, r1, a1); b1 = _mm256_fmadd_ps(l1, r1, b1);
          a2 = _mm256_fmadd_ps(l0, r2, a2); b2 = _mm256_fmadd_ps(l1, r2, b2);
          a3 = _mm256_fmadd_ps(l0, r3, a3); b3 = _mm256_fmadd_ps(l1, r3, b3);
        }
        _mm256_storeu_ps(out0 + col, a0); _mm256_storeu_ps(out0 + col + 8, a1);
        _mm256_storeu_ps(out0 + col + 16, a2); _mm256_storeu_ps(out0 + col + 24, a3);
        _mm256_storeu_ps(out1 + col, b0); _mm256_storeu_ps(out1 + col + 8, b1);
        _mm256_storeu_ps(out1 + col + 16, b2); _mm256_storeu_ps(out1 + col + 24, b3);
      }
      for (; col + 8 <= cols; col += 8) {
        __m256 v0 = _mm256_loadu_ps(out0 + col);
        __m256 v1 = _mm256_loadu_ps(out1 + col);
        for (int k = 0; k < depth; ++k) {
          const __m256 right = _mm256_loadu_ps(b + std::size_t(k) * cols + col);
          v0 = _mm256_fmadd_ps(_mm256_set1_ps(left0[k]), right, v0);
          v1 = _mm256_fmadd_ps(_mm256_set1_ps(left1[k]), right, v1);
        }
        _mm256_storeu_ps(out0 + col, v0);
        _mm256_storeu_ps(out1 + col, v1);
      }
      for (; col < cols; ++col) {
        float v0 = out0[col];
        float v1 = out1[col];
        for (int k = 0; k < depth; ++k) {
          const float weight = b[std::size_t(k) * cols + col];
          v0 += left0[k] * weight;
          v1 += left1[k] * weight;
        }
        out0[col] = v0;
        out1[col] = v1;
      }
    }
  }
  for (; row < last_row; ++row) {
    float* out = dst + std::size_t(row) * cols;
    const float* left = a + std::size_t(row) * depth;
    int col = 0;
    for (; col + 32 <= cols; col += 32) {
      __m256 c0 = _mm256_loadu_ps(out + col);
      __m256 c1 = _mm256_loadu_ps(out + col + 8);
      __m256 c2 = _mm256_loadu_ps(out + col + 16);
      __m256 c3 = _mm256_loadu_ps(out + col + 24);
      for (int k = 0; k < depth; ++k) {
        const __m256 aa = _mm256_set1_ps(left[k]);
        const float* right = b + std::size_t(k) * cols + col;
        c0 = _mm256_fmadd_ps(aa, _mm256_loadu_ps(right), c0);
        c1 = _mm256_fmadd_ps(aa, _mm256_loadu_ps(right + 8), c1);
        c2 = _mm256_fmadd_ps(aa, _mm256_loadu_ps(right + 16), c2);
        c3 = _mm256_fmadd_ps(aa, _mm256_loadu_ps(right + 24), c3);
      }
      _mm256_storeu_ps(out + col, c0); _mm256_storeu_ps(out + col + 8, c1);
      _mm256_storeu_ps(out + col + 16, c2); _mm256_storeu_ps(out + col + 24, c3);
    }
    for (; col + 8 <= cols; col += 8) {
      __m256 value = _mm256_loadu_ps(out + col);
      for (int k = 0; k < depth; ++k) {
        value = _mm256_fmadd_ps(_mm256_set1_ps(left[k]),
                                _mm256_loadu_ps(b + std::size_t(k) * cols + col), value);
      }
      _mm256_storeu_ps(out + col, value);
    }
    for (; col < cols; ++col) {
      float value = out[col];
      for (int k = 0; k < depth; ++k) value += left[k] * b[std::size_t(k) * cols + col];
      out[col] = value;
    }
  }
}

void Avx2DepthwiseConv(float* dst, const float* src, const float* weights,
                       const float* bias, int first_channel, int last_channel,
                       int input_h, int input_w, int output_h, int output_w,
                       int kernel_h, int kernel_w, int pad_top, int pad_left) noexcept {
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(output_h) * output_w;
  const std::size_t kernel_plane = std::size_t(kernel_h) * kernel_w;
  const int first_y = std::max(0, pad_top);
  const int first_x = std::max(0, pad_left);
  const int last_y = std::min(output_h, input_h + pad_top - kernel_h + 1);
  const int last_x = std::min(output_w, input_w + pad_left - kernel_w + 1);
  for (int channel = first_channel; channel < last_channel; ++channel) {
    const float* in = src + std::size_t(channel) * input_plane;
    const float* filter = weights + std::size_t(channel) * kernel_plane;
    float* out = dst + std::size_t(channel) * output_plane;
    const float base = bias ? bias[channel] : 0.F;
    for (int y = 0; y < output_h; ++y) {
      const int iy0 = y - pad_top;
      int x = 0;
      const bool row_interior = y >= first_y && y < last_y;
      const int interior_begin = first_x;
      const int interior_end = last_x;
      for (; x < interior_begin; ++x) {
        float sum = base; const int ix0 = x - pad_left;
        for (int ky = 0; ky < kernel_h; ++ky) { const int iy = iy0 + ky; if (iy < 0 || iy >= input_h) continue;
          for (int kx = 0; kx < kernel_w; ++kx) { const int ix = ix0 + kx; if (ix >= 0 && ix < input_w) sum += in[std::size_t(iy) * input_w + ix] * filter[ky * kernel_w + kx]; }
        }
        out[std::size_t(y) * output_w + x] = sum;
      }
      if (row_interior) {
        for (; x + 8 <= interior_end; x += 8) {
          __m256 sum = _mm256_set1_ps(base);
          for (int ky = 0; ky < kernel_h; ++ky) {
            const float* row = in + std::size_t(iy0 + ky) * input_w + x - pad_left;
            const float* kernel = filter + ky * kernel_w;
            for (int kx = 0; kx < kernel_w; ++kx) sum = _mm256_fmadd_ps(_mm256_set1_ps(kernel[kx]), _mm256_loadu_ps(row + kx), sum);
          }
          _mm256_storeu_ps(out + std::size_t(y) * output_w + x, sum);
        }
      } else {
        for (; x + 8 <= interior_end; x += 8) {
          __m256 sum = _mm256_set1_ps(base);
          for (int ky = 0; ky < kernel_h; ++ky) {
            const int iy = iy0 + ky;
            if (iy < 0 || iy >= input_h) continue;
            const float* row = in + std::size_t(iy) * input_w + x - pad_left;
            const float* kernel = filter + ky * kernel_w;
            for (int kx = 0; kx < kernel_w; ++kx) sum = _mm256_fmadd_ps(_mm256_set1_ps(kernel[kx]), _mm256_loadu_ps(row + kx), sum);
          }
          _mm256_storeu_ps(out + std::size_t(y) * output_w + x, sum);
        }
      }
      for (; x < output_w; ++x) {
        float sum = base; const int ix0 = x - pad_left;
        for (int ky = 0; ky < kernel_h; ++ky) { const int iy = iy0 + ky; if (iy < 0 || iy >= input_h) continue;
          for (int kx = 0; kx < kernel_w; ++kx) { const int ix = ix0 + kx; if (ix >= 0 && ix < input_w) sum += in[std::size_t(iy) * input_w + ix] * filter[ky * kernel_w + kx]; }
        }
        out[std::size_t(y) * output_w + x] = sum;
      }
    }
  }
}

void Avx2SpatialMean(float* dst, const float* src, std::size_t planes,
                     std::size_t spatial) noexcept {
  if (spatial == 0) return;
  const __m256 scale = _mm256_set1_ps(1.F / static_cast<float>(spatial));
  for (std::size_t plane = 0; plane < planes; ++plane) {
    const float* values = src + plane * spatial;
    __m256 accum = _mm256_setzero_ps();
    std::size_t index = 0;
    for (; index + 8 <= spatial; index += 8) accum = _mm256_add_ps(accum, _mm256_loadu_ps(values + index));
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, accum);
    float sum = 0.F;
    for (float lane : lanes) sum += lane;
    for (; index < spatial; ++index) sum += values[index];
    dst[plane] = _mm256_cvtss_f32(_mm256_mul_ps(_mm256_set1_ps(sum), scale));
  }
}

void Avx2MaxPool2x2Same(float* dst, const float* src, int first_plane,
                        int last_plane, int height, int width) noexcept {
  const std::size_t plane_size = std::size_t(height) * width;
  for (int plane = first_plane; plane < last_plane; ++plane) {
    const float* in = src + std::size_t(plane) * plane_size;
    float* out = dst + std::size_t(plane) * plane_size;
    for (int y = 0; y + 1 < height; ++y) {
      const float* row0 = in + std::size_t(y) * width;
      const float* row1 = row0 + width;
      int x = 0;
      for (; x + 8 <= width - 1; x += 8) {
        const __m256 top = _mm256_max_ps(_mm256_loadu_ps(row0 + x), _mm256_loadu_ps(row0 + x + 1));
        const __m256 bottom = _mm256_max_ps(_mm256_loadu_ps(row1 + x), _mm256_loadu_ps(row1 + x + 1));
        _mm256_storeu_ps(out + std::size_t(y) * width + x, _mm256_max_ps(top, bottom));
      }
      for (; x + 1 < width; ++x) out[std::size_t(y) * width + x] =
          std::max(std::max(row0[x], row0[x + 1]), std::max(row1[x], row1[x + 1]));
      out[std::size_t(y) * width + width - 1] = std::max(row0[width - 1], row1[width - 1]);
    }
    if (height == 1) {
      for (int x = 0; x + 1 < width; ++x) out[x] = std::max(in[x], in[x + 1]);
    } else {
      const float* row = in + std::size_t(height - 1) * width;
      float* out_row = out + std::size_t(height - 1) * width;
      int x = 0;
      for (; x + 8 <= width - 1; x += 8) {
        _mm256_storeu_ps(out_row + x, _mm256_max_ps(_mm256_loadu_ps(row + x), _mm256_loadu_ps(row + x + 1)));
      }
      for (; x + 1 < width; ++x) out_row[x] = std::max(row[x], row[x + 1]);
    }
    out[plane_size - 1] = in[plane_size - 1];
  }
}

void Avx2MaxPool2x2Valid(float* dst, const float* src, int first_plane,
                         int last_plane, int height, int width) noexcept {
  const int output_height = height - 1;
  const int output_width = width - 1;
  const std::size_t input_plane = std::size_t(height) * width;
  const std::size_t output_plane = std::size_t(output_height) * output_width;
  for (int plane = first_plane; plane < last_plane; ++plane) {
    const float* in = src + std::size_t(plane) * input_plane;
    float* out = dst + std::size_t(plane) * output_plane;
    for (int y = 0; y < output_height; ++y) {
      const float* row0 = in + std::size_t(y) * width;
      const float* row1 = row0 + width;
      float* row_out = out + std::size_t(y) * output_width;
      int x = 0;
      for (; x + 8 <= output_width; x += 8) {
        const __m256 top = _mm256_max_ps(_mm256_loadu_ps(row0 + x), _mm256_loadu_ps(row0 + x + 1));
        const __m256 bottom = _mm256_max_ps(_mm256_loadu_ps(row1 + x), _mm256_loadu_ps(row1 + x + 1));
        _mm256_storeu_ps(row_out + x, _mm256_max_ps(top, bottom));
      }
      for (; x < output_width; ++x) {
        row_out[x] = std::max(std::max(row0[x], row0[x + 1]),
                              std::max(row1[x], row1[x + 1]));
      }
    }
  }
}

void Avx2Conv2d(float* dst, const float* src, const float* weights,
                const float* bias, int first_output, int last_output,
                int input_channels, int input_h, int input_w, int output_h,
                int output_w, int kernel_h, int kernel_w, int pad_top,
                int pad_left) noexcept {
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(output_h) * output_w;
  const std::size_t filter_plane = std::size_t(kernel_h) * kernel_w;
  const int first_y = std::max(0, pad_top);
  const int first_x = std::max(0, pad_left);
  const int last_y = std::min(output_h, input_h + pad_top - kernel_h + 1);
  const int last_x = std::min(output_w, input_w + pad_left - kernel_w + 1);
  for (int output = first_output; output < last_output; ++output) {
    float* out = dst + std::size_t(output) * output_plane;
    const float* filter = weights + std::size_t(output) * input_channels * filter_plane;
    const float base = bias ? bias[output] : 0.F;
    for (int y = 0; y < output_h; ++y) {
      const int iy0 = y - pad_top;
      int x = 0;
      const int interior_begin = y >= first_y && y < last_y ? first_x : 0;
      const int interior_end = y >= first_y && y < last_y ? last_x : 0;
      for (; x < interior_begin; ++x) {
        float sum = base; const int ix0 = x - pad_left;
        for (int input = 0; input < input_channels; ++input) { const float* plane = src + std::size_t(input) * input_plane; const float* kernel = filter + std::size_t(input) * filter_plane;
          for (int ky = 0; ky < kernel_h; ++ky) { const int iy = iy0 + ky; if (iy < 0 || iy >= input_h) continue;
            for (int kx = 0; kx < kernel_w; ++kx) { const int ix = ix0 + kx; if (ix >= 0 && ix < input_w) sum += plane[std::size_t(iy) * input_w + ix] * kernel[ky * kernel_w + kx]; }
          }
        }
        out[std::size_t(y) * output_w + x] = sum;
      }
      for (; x + 8 <= interior_end; x += 8) {
        __m256 sum = _mm256_set1_ps(base);
        for (int input = 0; input < input_channels; ++input) {
          const float* plane = src + std::size_t(input) * input_plane + std::size_t(iy0) * input_w + x - pad_left;
          const float* kernel = filter + std::size_t(input) * filter_plane;
          for (int ky = 0; ky < kernel_h; ++ky) { const float* row = plane + std::size_t(ky) * input_w; const float* krow = kernel + ky * kernel_w;
            for (int kx = 0; kx < kernel_w; ++kx) sum = _mm256_fmadd_ps(_mm256_set1_ps(krow[kx]), _mm256_loadu_ps(row + kx), sum);
          }
        }
        _mm256_storeu_ps(out + std::size_t(y) * output_w + x, sum);
      }
      for (; x < output_w; ++x) {
        float sum = base; const int ix0 = x - pad_left;
        for (int input = 0; input < input_channels; ++input) { const float* plane = src + std::size_t(input) * input_plane; const float* kernel = filter + std::size_t(input) * filter_plane;
          for (int ky = 0; ky < kernel_h; ++ky) { const int iy = iy0 + ky; if (iy < 0 || iy >= input_h) continue;
            for (int kx = 0; kx < kernel_w; ++kx) { const int ix = ix0 + kx; if (ix >= 0 && ix < input_w) sum += plane[std::size_t(iy) * input_w + ix] * kernel[ky * kernel_w + kx]; }
          }
        }
        out[std::size_t(y) * output_w + x] = sum;
      }
    }
  }
}

// Valid 2x2 is common in the detector reconstruction head.  It has no edge
// handling and every output vector reads four contiguous input vectors.
void Avx2Conv2x2Valid(float* dst, const float* src, const float* weights,
                      const float* bias, int first_output, int last_output,
                      int input_channels, int input_h, int input_w, bool relu) noexcept {
  const int output_h = input_h - 1;
  const int output_w = input_w - 1;
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(output_h) * output_w;
  constexpr int kLanes = 8;
  for (int output = first_output; output < last_output; ++output) {
    float* out = dst + std::size_t(output) * output_plane;
    const float* filter = weights + std::size_t(output) * input_channels * 4;
    const float base = bias ? bias[output] : 0.F;
    for (int y = 0; y < output_h; ++y) {
      int x = 0;
      for (; x + kLanes <= output_w; x += kLanes) {
        __m256 sum = _mm256_set1_ps(base);
        for (int input = 0; input < input_channels; ++input) {
          const float* row0 = src + std::size_t(input) * input_plane + std::size_t(y) * input_w + x;
          const float* row1 = row0 + input_w;
          const float* k = filter + std::size_t(input) * 4;
          sum = _mm256_fmadd_ps(_mm256_set1_ps(k[0]), _mm256_loadu_ps(row0), sum);
          sum = _mm256_fmadd_ps(_mm256_set1_ps(k[1]), _mm256_loadu_ps(row0 + 1), sum);
          sum = _mm256_fmadd_ps(_mm256_set1_ps(k[2]), _mm256_loadu_ps(row1), sum);
          sum = _mm256_fmadd_ps(_mm256_set1_ps(k[3]), _mm256_loadu_ps(row1 + 1), sum);
        }
        if (relu) sum = _mm256_max_ps(sum, _mm256_setzero_ps());
        _mm256_storeu_ps(out + std::size_t(y) * output_w + x, sum);
      }
      for (; x < output_w; ++x) {
        float sum = base;
        for (int input = 0; input < input_channels; ++input) {
          const float* row0 = src + std::size_t(input) * input_plane + std::size_t(y) * input_w + x;
          const float* k = filter + std::size_t(input) * 4;
          sum += row0[0] * k[0] + row0[1] * k[1] + row0[input_w] * k[2] + row0[input_w + 1] * k[3];
        }
        out[std::size_t(y) * output_w + x] = relu ? std::max(sum, 0.F) : sum;
      }
    }
  }
}

// SAME_UPPER 2x2 (output size == input size, pad 0). Interior pixels share
// four contiguous loads across four or eight output filters. The right column
// and bottom row drop the out-of-range tap, matching the scalar reduction
// order. `PPOCR_DISABLE_AVX2_CONV2X2_SAME8` keeps the four-output tile.
void Avx2Conv2x2SameUpper(float* dst, const float* src, const float* weights,
                          const float* bias, int first_output, int last_output,
                          int input_channels, int input_h, int input_w,
                          bool relu, int row_begin, int row_end) noexcept {
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = input_plane;
  const int y_begin = row_begin < 0 ? 0 : std::min(input_h, row_begin);
  const int y_end = row_end < 0 ? input_h : std::min(input_h, row_end);
  const __m256 zero = _mm256_setzero_ps();
  int output = first_output;
  static const bool use_same8 =
      std::getenv("PPOCR_DISABLE_AVX2_CONV2X2_SAME8") == nullptr;
  if (use_same8 && first_output + 8 <= last_output) {
    const auto store8 = [&](float* const out[8], std::size_t index, const float* s) {
      for (int q = 0; q < 8; ++q) out[q][index] = relu ? std::max(s[q], 0.F) : s[q];
    };
    for (int y = y_begin; y + 1 < input_h && y < y_end; ++y) {
      for (int oc = first_output; oc + 8 <= last_output; oc += 8) {
        float* out[8];
        const float* f[8];
        float b[8];
        for (int q = 0; q < 8; ++q) {
          out[q] = dst + std::size_t(oc + q) * output_plane;
          f[q] = weights + std::size_t(oc + q) * input_channels * 4;
          b[q] = bias ? bias[oc + q] : 0.F;
        }
        int x = 0;
        for (; x + 8 <= input_w - 1; x += 8) {
          __m256 s0 = _mm256_set1_ps(b[0]), s1 = _mm256_set1_ps(b[1]);
          __m256 s2 = _mm256_set1_ps(b[2]), s3 = _mm256_set1_ps(b[3]);
          __m256 s4 = _mm256_set1_ps(b[4]), s5 = _mm256_set1_ps(b[5]);
          __m256 s6 = _mm256_set1_ps(b[6]), s7 = _mm256_set1_ps(b[7]);
          for (int input = 0; input < input_channels; ++input) {
            const float* row0 = src + std::size_t(input) * input_plane +
                                std::size_t(y) * input_w + x;
            const float* row1 = row0 + input_w;
            const __m256 v00 = _mm256_loadu_ps(row0);
            const __m256 v01 = _mm256_loadu_ps(row0 + 1);
            const __m256 v10 = _mm256_loadu_ps(row1);
            const __m256 v11 = _mm256_loadu_ps(row1 + 1);
            const auto fma4 = [&](__m256& acc, const float* k) {
              acc = _mm256_fmadd_ps(_mm256_set1_ps(k[0]), v00, acc);
              acc = _mm256_fmadd_ps(_mm256_set1_ps(k[1]), v01, acc);
              acc = _mm256_fmadd_ps(_mm256_set1_ps(k[2]), v10, acc);
              acc = _mm256_fmadd_ps(_mm256_set1_ps(k[3]), v11, acc);
            };
            fma4(s0, f[0] + std::size_t(input) * 4);
            fma4(s1, f[1] + std::size_t(input) * 4);
            fma4(s2, f[2] + std::size_t(input) * 4);
            fma4(s3, f[3] + std::size_t(input) * 4);
            fma4(s4, f[4] + std::size_t(input) * 4);
            fma4(s5, f[5] + std::size_t(input) * 4);
            fma4(s6, f[6] + std::size_t(input) * 4);
            fma4(s7, f[7] + std::size_t(input) * 4);
          }
          if (relu) {
            s0 = _mm256_max_ps(s0, zero); s1 = _mm256_max_ps(s1, zero);
            s2 = _mm256_max_ps(s2, zero); s3 = _mm256_max_ps(s3, zero);
            s4 = _mm256_max_ps(s4, zero); s5 = _mm256_max_ps(s5, zero);
            s6 = _mm256_max_ps(s6, zero); s7 = _mm256_max_ps(s7, zero);
          }
          const std::size_t index = std::size_t(y) * input_w + x;
          _mm256_storeu_ps(out[0] + index, s0); _mm256_storeu_ps(out[1] + index, s1);
          _mm256_storeu_ps(out[2] + index, s2); _mm256_storeu_ps(out[3] + index, s3);
          _mm256_storeu_ps(out[4] + index, s4); _mm256_storeu_ps(out[5] + index, s5);
          _mm256_storeu_ps(out[6] + index, s6); _mm256_storeu_ps(out[7] + index, s7);
        }
        for (; x + 1 < input_w; ++x) {
          float s[8] = {b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]};
          for (int input = 0; input < input_channels; ++input) {
            const float* row0 = src + std::size_t(input) * input_plane +
                                std::size_t(y) * input_w + x;
            for (int q = 0; q < 8; ++q) {
              const float* k = f[q] + std::size_t(input) * 4;
              s[q] += row0[0] * k[0] + row0[1] * k[1] +
                      row0[input_w] * k[2] + row0[input_w + 1] * k[3];
            }
          }
          store8(out, std::size_t(y) * input_w + x, s);
        }
        float right[8] = {b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]};
        for (int input = 0; input < input_channels; ++input) {
          const float* row0 = src + std::size_t(input) * input_plane +
                              std::size_t(y) * input_w + input_w - 1;
          for (int q = 0; q < 8; ++q) {
            const float* k = f[q] + std::size_t(input) * 4;
            right[q] += row0[0] * k[0] + row0[input_w] * k[2];
          }
        }
        store8(out, std::size_t(y) * input_w + input_w - 1, right);
      }
    }
    if (y_end >= input_h) {
      const int y = input_h - 1;
      for (int oc = first_output; oc + 8 <= last_output; oc += 8) {
        float* out[8];
        const float* f[8];
        float b[8];
        for (int q = 0; q < 8; ++q) {
          out[q] = dst + std::size_t(oc + q) * output_plane;
          f[q] = weights + std::size_t(oc + q) * input_channels * 4;
          b[q] = bias ? bias[oc + q] : 0.F;
        }
        for (int x = 0; x + 1 < input_w; ++x) {
          float s[8] = {b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]};
          for (int input = 0; input < input_channels; ++input) {
            const float* row = src + std::size_t(input) * input_plane +
                               std::size_t(y) * input_w + x;
            for (int q = 0; q < 8; ++q) {
              const float* k = f[q] + std::size_t(input) * 4;
              s[q] += row[0] * k[0] + row[1] * k[1];
            }
          }
          store8(out, std::size_t(y) * input_w + x, s);
        }
        float corner[8] = {b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]};
        for (int input = 0; input < input_channels; ++input) {
          const float value = src[std::size_t(input) * input_plane + input_plane - 1];
          for (int q = 0; q < 8; ++q) corner[q] += value * f[q][std::size_t(input) * 4];
        }
        store8(out, input_plane - 1, corner);
      }
    }
    while (output + 8 <= last_output) output += 8;
  }
  for (int y = y_begin; y + 1 < input_h && y < y_end; ++y) {
    for (int oc = output; oc + 4 <= last_output; oc += 4) {
      float* out0 = dst + std::size_t(oc) * output_plane;
      float* out1 = out0 + output_plane;
      float* out2 = out1 + output_plane;
      float* out3 = out2 + output_plane;
      const float* f0 = weights + std::size_t(oc) * input_channels * 4;
      const float* f1 = f0 + std::size_t(input_channels) * 4;
      const float* f2 = f1 + std::size_t(input_channels) * 4;
      const float* f3 = f2 + std::size_t(input_channels) * 4;
      const float b0 = bias ? bias[oc] : 0.F;
      const float b1 = bias ? bias[oc + 1] : 0.F;
      const float b2 = bias ? bias[oc + 2] : 0.F;
      const float b3 = bias ? bias[oc + 3] : 0.F;
      int x = 0;
      for (; x + 8 <= input_w - 1; x += 8) {
        __m256 s0 = _mm256_set1_ps(b0), s1 = _mm256_set1_ps(b1);
        __m256 s2 = _mm256_set1_ps(b2), s3 = _mm256_set1_ps(b3);
        for (int input = 0; input < input_channels; ++input) {
          const float* row0 = src + std::size_t(input) * input_plane +
                              std::size_t(y) * input_w + x;
          const float* row1 = row0 + input_w;
          const __m256 v00 = _mm256_loadu_ps(row0);
          const __m256 v01 = _mm256_loadu_ps(row0 + 1);
          const __m256 v10 = _mm256_loadu_ps(row1);
          const __m256 v11 = _mm256_loadu_ps(row1 + 1);
          const float* k0 = f0 + std::size_t(input) * 4;
          const float* k1 = f1 + std::size_t(input) * 4;
          const float* k2 = f2 + std::size_t(input) * 4;
          const float* k3 = f3 + std::size_t(input) * 4;
          s0 = _mm256_fmadd_ps(_mm256_set1_ps(k0[0]), v00, s0);
          s0 = _mm256_fmadd_ps(_mm256_set1_ps(k0[1]), v01, s0);
          s0 = _mm256_fmadd_ps(_mm256_set1_ps(k0[2]), v10, s0);
          s0 = _mm256_fmadd_ps(_mm256_set1_ps(k0[3]), v11, s0);
          s1 = _mm256_fmadd_ps(_mm256_set1_ps(k1[0]), v00, s1);
          s1 = _mm256_fmadd_ps(_mm256_set1_ps(k1[1]), v01, s1);
          s1 = _mm256_fmadd_ps(_mm256_set1_ps(k1[2]), v10, s1);
          s1 = _mm256_fmadd_ps(_mm256_set1_ps(k1[3]), v11, s1);
          s2 = _mm256_fmadd_ps(_mm256_set1_ps(k2[0]), v00, s2);
          s2 = _mm256_fmadd_ps(_mm256_set1_ps(k2[1]), v01, s2);
          s2 = _mm256_fmadd_ps(_mm256_set1_ps(k2[2]), v10, s2);
          s2 = _mm256_fmadd_ps(_mm256_set1_ps(k2[3]), v11, s2);
          s3 = _mm256_fmadd_ps(_mm256_set1_ps(k3[0]), v00, s3);
          s3 = _mm256_fmadd_ps(_mm256_set1_ps(k3[1]), v01, s3);
          s3 = _mm256_fmadd_ps(_mm256_set1_ps(k3[2]), v10, s3);
          s3 = _mm256_fmadd_ps(_mm256_set1_ps(k3[3]), v11, s3);
        }
        if (relu) {
          s0 = _mm256_max_ps(s0, zero); s1 = _mm256_max_ps(s1, zero);
          s2 = _mm256_max_ps(s2, zero); s3 = _mm256_max_ps(s3, zero);
        }
        const std::size_t index = std::size_t(y) * input_w + x;
        _mm256_storeu_ps(out0 + index, s0); _mm256_storeu_ps(out1 + index, s1);
        _mm256_storeu_ps(out2 + index, s2); _mm256_storeu_ps(out3 + index, s3);
      }
      for (; x + 1 < input_w; ++x) {
        float s0 = b0, s1 = b1, s2 = b2, s3 = b3;
        for (int input = 0; input < input_channels; ++input) {
          const float* row0 = src + std::size_t(input) * input_plane +
                              std::size_t(y) * input_w + x;
          const float* k0 = f0 + std::size_t(input) * 4;
          const float* k1 = f1 + std::size_t(input) * 4;
          const float* k2 = f2 + std::size_t(input) * 4;
          const float* k3 = f3 + std::size_t(input) * 4;
          s0 += row0[0] * k0[0] + row0[1] * k0[1] + row0[input_w] * k0[2] + row0[input_w + 1] * k0[3];
          s1 += row0[0] * k1[0] + row0[1] * k1[1] + row0[input_w] * k1[2] + row0[input_w + 1] * k1[3];
          s2 += row0[0] * k2[0] + row0[1] * k2[1] + row0[input_w] * k2[2] + row0[input_w + 1] * k2[3];
          s3 += row0[0] * k3[0] + row0[1] * k3[1] + row0[input_w] * k3[2] + row0[input_w + 1] * k3[3];
        }
        const std::size_t index = std::size_t(y) * input_w + x;
        out0[index] = relu ? std::max(s0, 0.F) : s0;
        out1[index] = relu ? std::max(s1, 0.F) : s1;
        out2[index] = relu ? std::max(s2, 0.F) : s2;
        out3[index] = relu ? std::max(s3, 0.F) : s3;
      }
      float s0 = b0, s1 = b1, s2 = b2, s3 = b3;
      for (int input = 0; input < input_channels; ++input) {
        const float* row0 = src + std::size_t(input) * input_plane +
                            std::size_t(y) * input_w + input_w - 1;
        const float* k0 = f0 + std::size_t(input) * 4;
        const float* k1 = f1 + std::size_t(input) * 4;
        const float* k2 = f2 + std::size_t(input) * 4;
        const float* k3 = f3 + std::size_t(input) * 4;
        s0 += row0[0] * k0[0] + row0[input_w] * k0[2];
        s1 += row0[0] * k1[0] + row0[input_w] * k1[2];
        s2 += row0[0] * k2[0] + row0[input_w] * k2[2];
        s3 += row0[0] * k3[0] + row0[input_w] * k3[2];
      }
      const std::size_t index = std::size_t(y) * input_w + input_w - 1;
      out0[index] = relu ? std::max(s0, 0.F) : s0;
      out1[index] = relu ? std::max(s1, 0.F) : s1;
      out2[index] = relu ? std::max(s2, 0.F) : s2;
      out3[index] = relu ? std::max(s3, 0.F) : s3;
    }
  }
  if (y_end >= input_h) {
    const int y = input_h - 1;
    for (int oc = output; oc + 4 <= last_output; oc += 4) {
      float* out0 = dst + std::size_t(oc) * output_plane;
      float* out1 = out0 + output_plane;
      float* out2 = out1 + output_plane;
      float* out3 = out2 + output_plane;
      const float* f0 = weights + std::size_t(oc) * input_channels * 4;
      const float* f1 = f0 + std::size_t(input_channels) * 4;
      const float* f2 = f1 + std::size_t(input_channels) * 4;
      const float* f3 = f2 + std::size_t(input_channels) * 4;
      const float b0 = bias ? bias[oc] : 0.F;
      const float b1 = bias ? bias[oc + 1] : 0.F;
      const float b2 = bias ? bias[oc + 2] : 0.F;
      const float b3 = bias ? bias[oc + 3] : 0.F;
      for (int x = 0; x + 1 < input_w; ++x) {
        float s0 = b0, s1 = b1, s2 = b2, s3 = b3;
        for (int input = 0; input < input_channels; ++input) {
          const float* row = src + std::size_t(input) * input_plane +
                             std::size_t(y) * input_w + x;
          const float* k0 = f0 + std::size_t(input) * 4;
          const float* k1 = f1 + std::size_t(input) * 4;
          const float* k2 = f2 + std::size_t(input) * 4;
          const float* k3 = f3 + std::size_t(input) * 4;
          s0 += row[0] * k0[0] + row[1] * k0[1];
          s1 += row[0] * k1[0] + row[1] * k1[1];
          s2 += row[0] * k2[0] + row[1] * k2[1];
          s3 += row[0] * k3[0] + row[1] * k3[1];
        }
        const std::size_t index = std::size_t(y) * input_w + x;
        out0[index] = relu ? std::max(s0, 0.F) : s0;
        out1[index] = relu ? std::max(s1, 0.F) : s1;
        out2[index] = relu ? std::max(s2, 0.F) : s2;
        out3[index] = relu ? std::max(s3, 0.F) : s3;
      }
      float s0 = b0, s1 = b1, s2 = b2, s3 = b3;
      for (int input = 0; input < input_channels; ++input) {
        const float value = src[std::size_t(input) * input_plane + input_plane - 1];
        s0 += value * f0[std::size_t(input) * 4];
        s1 += value * f1[std::size_t(input) * 4];
        s2 += value * f2[std::size_t(input) * 4];
        s3 += value * f3[std::size_t(input) * 4];
      }
      out0[input_plane - 1] = relu ? std::max(s0, 0.F) : s0;
      out1[input_plane - 1] = relu ? std::max(s1, 0.F) : s1;
      out2[input_plane - 1] = relu ? std::max(s2, 0.F) : s2;
      out3[input_plane - 1] = relu ? std::max(s3, 0.F) : s3;
    }
  }
  while (output + 4 <= last_output) output += 4;
  for (; output < last_output; ++output) {
    float* out = dst + std::size_t(output) * output_plane;
    const float* filter = weights + std::size_t(output) * input_channels * 4;
    const float base = bias ? bias[output] : 0.F;
    for (int y = y_begin; y + 1 < input_h && y < y_end; ++y) {
      int x = 0;
      for (; x + 8 <= input_w - 1; x += 8) {
        __m256 sum = _mm256_set1_ps(base);
        for (int input = 0; input < input_channels; ++input) {
          const float* row0 = src + std::size_t(input) * input_plane +
                              std::size_t(y) * input_w + x;
          const float* row1 = row0 + input_w;
          const float* k = filter + std::size_t(input) * 4;
          sum = _mm256_fmadd_ps(_mm256_set1_ps(k[0]), _mm256_loadu_ps(row0), sum);
          sum = _mm256_fmadd_ps(_mm256_set1_ps(k[1]), _mm256_loadu_ps(row0 + 1), sum);
          sum = _mm256_fmadd_ps(_mm256_set1_ps(k[2]), _mm256_loadu_ps(row1), sum);
          sum = _mm256_fmadd_ps(_mm256_set1_ps(k[3]), _mm256_loadu_ps(row1 + 1), sum);
        }
        if (relu) sum = _mm256_max_ps(sum, zero);
        _mm256_storeu_ps(out + std::size_t(y) * input_w + x, sum);
      }
      for (; x + 1 < input_w; ++x) {
        float sum = base;
        for (int input = 0; input < input_channels; ++input) {
          const float* row0 = src + std::size_t(input) * input_plane +
                              std::size_t(y) * input_w + x;
          const float* k = filter + std::size_t(input) * 4;
          sum += row0[0] * k[0] + row0[1] * k[1] +
                 row0[input_w] * k[2] + row0[input_w + 1] * k[3];
        }
        out[std::size_t(y) * input_w + x] = relu ? std::max(sum, 0.F) : sum;
      }
      float sum = base;
      for (int input = 0; input < input_channels; ++input) {
        const float* row0 = src + std::size_t(input) * input_plane +
                            std::size_t(y) * input_w + input_w - 1;
        const float* k = filter + std::size_t(input) * 4;
        sum += row0[0] * k[0] + row0[input_w] * k[2];
      }
      out[std::size_t(y) * input_w + input_w - 1] = relu ? std::max(sum, 0.F) : sum;
    }
    if (y_end < input_h) continue;
    const int y = input_h - 1;
    for (int x = 0; x + 1 < input_w; ++x) {
      float sum = base;
      for (int input = 0; input < input_channels; ++input) {
        const float* row = src + std::size_t(input) * input_plane +
                           std::size_t(y) * input_w + x;
        const float* k = filter + std::size_t(input) * 4;
        sum += row[0] * k[0] + row[1] * k[1];
      }
      out[std::size_t(y) * input_w + x] = relu ? std::max(sum, 0.F) : sum;
    }
    float sum = base;
    for (int input = 0; input < input_channels; ++input) {
      const float value = src[std::size_t(input) * input_plane + input_plane - 1];
      sum += value * filter[std::size_t(input) * 4];
    }
    out[input_plane - 1] = relu ? std::max(sum, 0.F) : sum;
  }
}

// AVX2 counterpart of the AVX-512 four-output 3x3 kernel.  It keeps x86
// machines without AVX-512 on the same input-reuse algorithm rather than
// falling back to four separate input traversals.
void Avx2Conv3x3Stride1x4(float* dst, const float* src, const float* weights,
                          const float* bias, int first_output, int last_output,
                          int input_channels, int input_h, int input_w,
                          int output_h, int output_w, int pad_top,
                          int pad_left, bool relu, int row_begin, int row_end,
                          bool accumulate) noexcept {
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(output_h) * output_w;
  int y0 = row_begin < 0 ? 0 : row_begin;
  int y1 = row_end < 0 ? output_h : row_end;
  if (y0 < 0) y0 = 0;
  if (y1 > output_h) y1 = output_h;
  if (y0 >= y1) return;
  const int first_y = std::max(0, pad_top);
  const int first_x = std::max(0, pad_left);
  const int last_y = std::min(output_h, input_h + pad_top - 2);
  const int last_x = std::min(output_w, input_w + pad_left - 2);
  const auto scalar4 = [&](float* out0, float* out1, float* out2, float* out3,
                           const float* f0, const float* f1, const float* f2, const float* f3,
                           float b0, float b1, float b2, float b3, int y, int x) {
    const int iy0 = y - pad_top, ix0 = x - pad_left;
    float s0 = b0, s1 = b1, s2 = b2, s3 = b3;
    for (int input = 0; input < input_channels; ++input) {
      const float* plane = src + std::size_t(input) * input_plane;
      const float* k0 = f0 + std::size_t(input) * 9;
      const float* k1 = f1 + std::size_t(input) * 9;
      const float* k2 = f2 + std::size_t(input) * 9;
      const float* k3 = f3 + std::size_t(input) * 9;
      for (int ky = 0; ky < 3; ++ky) {
        const int iy = iy0 + ky;
        if (iy < 0 || iy >= input_h) continue;
        for (int kx = 0; kx < 3; ++kx) {
          const int ix = ix0 + kx;
          if (ix < 0 || ix >= input_w) continue;
          const float value = plane[std::size_t(iy) * input_w + ix];
          const int ki = ky * 3 + kx;
          s0 += value * k0[ki]; s1 += value * k1[ki];
          s2 += value * k2[ki]; s3 += value * k3[ki];
        }
      }
    }
    const auto index = std::size_t(y) * output_w + x;
    if (accumulate) {
      s0 += out0[index]; s1 += out1[index];
      s2 += out2[index]; s3 += out3[index];
    }
    out0[index] = relu ? std::max(s0, 0.F) : s0;
    out1[index] = relu ? std::max(s1, 0.F) : s1;
    out2[index] = relu ? std::max(s2, 0.F) : s2;
    out3[index] = relu ? std::max(s3, 0.F) : s3;
  };
  int output = first_output;
  for (; output + 4 <= last_output; output += 4) {
    float* out0 = dst + std::size_t(output) * output_plane;
    float* out1 = out0 + output_plane;
    float* out2 = out1 + output_plane;
    float* out3 = out2 + output_plane;
    const float* f0 = weights + std::size_t(output) * input_channels * 9;
    const float* f1 = f0 + std::size_t(input_channels) * 9;
    const float* f2 = f1 + std::size_t(input_channels) * 9;
    const float* f3 = f2 + std::size_t(input_channels) * 9;
    const float b0 = bias ? bias[output] : 0.F, b1 = bias ? bias[output + 1] : 0.F;
    const float b2 = bias ? bias[output + 2] : 0.F, b3 = bias ? bias[output + 3] : 0.F;
    for (int y = y0; y < y1; ++y) {
      const int iy0 = y - pad_top;
      const bool interior_y = y >= first_y && y < last_y;
      int x = 0;
      for (; x < output_w; ++x) {
        if (interior_y && x >= first_x && x + 8 <= last_x) break;
        scalar4(out0, out1, out2, out3, f0, f1, f2, f3, b0, b1, b2, b3, y, x);
      }
      // Two spatial vectors share each tap broadcast and give eight ymm
      // chains, enough to cover two FMAs per cycle. Per-pixel tap order is
      // unchanged. `PPOCR_DISABLE_AVX2_CONV3_X16` keeps the eight-wide loop.
      static const bool wide16 =
          std::getenv("PPOCR_DISABLE_AVX2_CONV3_X16") == nullptr;
      if (wide16) {
        for (; x + 16 <= last_x; x += 16) {
          const auto index = std::size_t(y) * output_w + x;
          __m256 s00 = _mm256_set1_ps(b0), s01 = s00;
          __m256 s10 = _mm256_set1_ps(b1), s11 = s10;
          __m256 s20 = _mm256_set1_ps(b2), s21 = s20;
          __m256 s30 = _mm256_set1_ps(b3), s31 = s30;
          for (int input = 0; input < input_channels; ++input) {
            const float* plane = src + std::size_t(input) * input_plane +
                                 std::size_t(iy0) * input_w + x - pad_left;
            const float* k0 = f0 + std::size_t(input) * 9;
            const float* k1 = f1 + std::size_t(input) * 9;
            const float* k2 = f2 + std::size_t(input) * 9;
            const float* k3 = f3 + std::size_t(input) * 9;
            for (int ky = 0; ky < 3; ++ky) {
              const float* row = plane + std::size_t(ky) * input_w;
              for (int kx = 0; kx < 3; ++kx) {
                const __m256 v0 = _mm256_loadu_ps(row + kx);
                const __m256 v1 = _mm256_loadu_ps(row + kx + 8);
                const int ki = ky * 3 + kx;
                const __m256 t0 = _mm256_set1_ps(k0[ki]);
                const __m256 t1 = _mm256_set1_ps(k1[ki]);
                const __m256 t2 = _mm256_set1_ps(k2[ki]);
                const __m256 t3 = _mm256_set1_ps(k3[ki]);
                s00 = _mm256_fmadd_ps(t0, v0, s00);
                s01 = _mm256_fmadd_ps(t0, v1, s01);
                s10 = _mm256_fmadd_ps(t1, v0, s10);
                s11 = _mm256_fmadd_ps(t1, v1, s11);
                s20 = _mm256_fmadd_ps(t2, v0, s20);
                s21 = _mm256_fmadd_ps(t2, v1, s21);
                s30 = _mm256_fmadd_ps(t3, v0, s30);
                s31 = _mm256_fmadd_ps(t3, v1, s31);
              }
            }
          }
          if (accumulate) {
            s00 = _mm256_add_ps(s00, _mm256_loadu_ps(out0 + index));
            s01 = _mm256_add_ps(s01, _mm256_loadu_ps(out0 + index + 8));
            s10 = _mm256_add_ps(s10, _mm256_loadu_ps(out1 + index));
            s11 = _mm256_add_ps(s11, _mm256_loadu_ps(out1 + index + 8));
            s20 = _mm256_add_ps(s20, _mm256_loadu_ps(out2 + index));
            s21 = _mm256_add_ps(s21, _mm256_loadu_ps(out2 + index + 8));
            s30 = _mm256_add_ps(s30, _mm256_loadu_ps(out3 + index));
            s31 = _mm256_add_ps(s31, _mm256_loadu_ps(out3 + index + 8));
          }
          if (relu) {
            const __m256 zero = _mm256_setzero_ps();
            s00 = _mm256_max_ps(s00, zero); s01 = _mm256_max_ps(s01, zero);
            s10 = _mm256_max_ps(s10, zero); s11 = _mm256_max_ps(s11, zero);
            s20 = _mm256_max_ps(s20, zero); s21 = _mm256_max_ps(s21, zero);
            s30 = _mm256_max_ps(s30, zero); s31 = _mm256_max_ps(s31, zero);
          }
          _mm256_storeu_ps(out0 + index, s00);
          _mm256_storeu_ps(out0 + index + 8, s01);
          _mm256_storeu_ps(out1 + index, s10);
          _mm256_storeu_ps(out1 + index + 8, s11);
          _mm256_storeu_ps(out2 + index, s20);
          _mm256_storeu_ps(out2 + index + 8, s21);
          _mm256_storeu_ps(out3 + index, s30);
          _mm256_storeu_ps(out3 + index + 8, s31);
        }
      }
      for (; x + 8 <= last_x; x += 8) {
        const auto index = std::size_t(y) * output_w + x;
        __m256 s0 = _mm256_set1_ps(b0), s1 = _mm256_set1_ps(b1);
        __m256 s2 = _mm256_set1_ps(b2), s3 = _mm256_set1_ps(b3);
        for (int input = 0; input < input_channels; ++input) {
          const float* plane = src + std::size_t(input) * input_plane + std::size_t(iy0) * input_w + x - pad_left;
          const float* k0 = f0 + std::size_t(input) * 9;
          const float* k1 = f1 + std::size_t(input) * 9;
          const float* k2 = f2 + std::size_t(input) * 9;
          const float* k3 = f3 + std::size_t(input) * 9;
          for (int ky = 0; ky < 3; ++ky) {
            const float* row = plane + std::size_t(ky) * input_w;
            for (int kx = 0; kx < 3; ++kx) {
              const __m256 values = _mm256_loadu_ps(row + kx);
              const int ki = ky * 3 + kx;
              s0 = _mm256_fmadd_ps(_mm256_set1_ps(k0[ki]), values, s0);
              s1 = _mm256_fmadd_ps(_mm256_set1_ps(k1[ki]), values, s1);
              s2 = _mm256_fmadd_ps(_mm256_set1_ps(k2[ki]), values, s2);
              s3 = _mm256_fmadd_ps(_mm256_set1_ps(k3[ki]), values, s3);
            }
          }
        }
        if (accumulate) {
          s0 = _mm256_add_ps(s0, _mm256_loadu_ps(out0 + index));
          s1 = _mm256_add_ps(s1, _mm256_loadu_ps(out1 + index));
          s2 = _mm256_add_ps(s2, _mm256_loadu_ps(out2 + index));
          s3 = _mm256_add_ps(s3, _mm256_loadu_ps(out3 + index));
        }
        if (relu) { const __m256 zero = _mm256_setzero_ps(); s0 = _mm256_max_ps(s0, zero); s1 = _mm256_max_ps(s1, zero); s2 = _mm256_max_ps(s2, zero); s3 = _mm256_max_ps(s3, zero); }
        _mm256_storeu_ps(out0 + index, s0); _mm256_storeu_ps(out1 + index, s1);
        _mm256_storeu_ps(out2 + index, s2); _mm256_storeu_ps(out3 + index, s3);
      }
      for (; x < output_w; ++x) scalar4(out0, out1, out2, out3, f0, f1, f2, f3,
                                        b0, b1, b2, b3, y, x);
    }
  }
  const bool full_rows = row_begin <= 0 && (row_end < 0 || row_end >= output_h);
  if (output < last_output && !accumulate && full_rows) {
    Avx2Conv2d(dst, src, weights, bias, output, last_output, input_channels,
               input_h, input_w, output_h, output_w, 3, 3, pad_top, pad_left);
    // Keep the incomplete four-channel Conv+ReLU tail semantically identical
    // to the vectorized body above. Avx2Conv2d is intentionally activation
    // agnostic, so it needs this exact final clamp for fused graph nodes.
    if (relu) {
      for (int channel = output; channel < last_output; ++channel) {
        float* values = dst + std::size_t(channel) * output_plane;
        for (std::size_t index = 0; index < output_plane; ++index) {
          values[index] = std::max(values[index], 0.F);
        }
      }
    }
  }
}


// Interior stride-2 3x3 outputs map to every second input element.  Handle
// the short scalar border separately, then use AVX2 gathers over eight output
// pixels.  This avoids im2col materialization for the medium detector stem.
void Avx2Conv3x3Stride2(float* dst, const float* src, const float* weights,
                         const float* bias, int first_output, int last_output,
                         int input_channels, int input_h, int input_w,
                         int output_h, int output_w, int pad_top,
                         int pad_left, bool relu) noexcept {
  constexpr int lanes = 8;
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(output_h) * output_w;
  const __m256i offsets = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
  for (int output = first_output; output < last_output; ++output) {
    float* out = dst + std::size_t(output) * output_plane;
    const float* filter = weights + std::size_t(output) * input_channels * 9;
    const float base = bias ? bias[output] : 0.F;
    for (int y = 0; y < output_h; ++y) {
      const int iy0 = y * 2 - pad_top;
      const bool interior_y = iy0 >= 0 && iy0 + 2 < input_h;
      int x = 0;
      for (; x < output_w; ++x) {
        const int ix0 = x * 2 - pad_left;
        if (interior_y && ix0 >= 0 && ix0 + (lanes - 1) * 2 + 2 < input_w) break;
        float sum = base;
        for (int input = 0; input < input_channels; ++input) {
          const float* plane = src + std::size_t(input) * input_plane;
          const float* kernel = filter + std::size_t(input) * 9;
          for (int ky = 0; ky < 3; ++ky) {
            const int iy = iy0 + ky;
            if (iy < 0 || iy >= input_h) continue;
            for (int kx = 0; kx < 3; ++kx) {
              const int ix = ix0 + kx;
              if (ix >= 0 && ix < input_w) sum += plane[std::size_t(iy) * input_w + ix] * kernel[ky * 3 + kx];
            }
          }
        }
        out[std::size_t(y) * output_w + x] = relu ? std::max(sum, 0.F) : sum;
      }
      for (; x + lanes <= output_w; x += lanes) {
        const int ix0 = x * 2 - pad_left;
        if (!(interior_y && ix0 >= 0 && ix0 + (lanes - 1) * 2 + 2 < input_w)) break;
        __m256 sum = _mm256_set1_ps(base);
        for (int input = 0; input < input_channels; ++input) {
          const float* plane = src + std::size_t(input) * input_plane +
                               std::size_t(iy0) * input_w + ix0;
          const float* kernel = filter + std::size_t(input) * 9;
          for (int ky = 0; ky < 3; ++ky) {
            const float* row = plane + std::size_t(ky) * input_w;
            for (int kx = 0; kx < 3; ++kx) {
              sum = _mm256_fmadd_ps(_mm256_set1_ps(kernel[ky * 3 + kx]),
                                    _mm256_i32gather_ps(row + kx, offsets, 4), sum);
            }
          }
        }
        if (relu) sum = _mm256_max_ps(sum, _mm256_setzero_ps());
        _mm256_storeu_ps(out + std::size_t(y) * output_w + x, sum);
      }
      for (; x < output_w; ++x) {
        const int ix0 = x * 2 - pad_left;
        float sum = base;
        for (int input = 0; input < input_channels; ++input) {
          const float* plane = src + std::size_t(input) * input_plane;
          const float* kernel = filter + std::size_t(input) * 9;
          for (int ky = 0; ky < 3; ++ky) {
            const int iy = iy0 + ky;
            if (iy < 0 || iy >= input_h) continue;
            for (int kx = 0; kx < 3; ++kx) {
              const int ix = ix0 + kx;
              if (ix >= 0 && ix < input_w) sum += plane[std::size_t(iy) * input_w + ix] * kernel[ky * 3 + kx];
            }
          }
        }
        out[std::size_t(y) * output_w + x] = relu ? std::max(sum, 0.F) : sum;
      }
    }
  }
}

void Avx2Conv3x3Stride2x4(float* dst, const float* src, const float* weights,
                           const float* bias, int first_output, int last_output,
                           int input_channels, int input_h, int input_w,
                           int output_h, int output_w, int pad_top,
                           int pad_left, bool relu, int row_begin, int row_end,
                           bool accumulate) noexcept {
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(output_h) * output_w;
  int y0 = row_begin < 0 ? 0 : row_begin;
  int y1 = row_end < 0 ? output_h : row_end;
  if (y0 < 0) y0 = 0;
  if (y1 > output_h) y1 = output_h;
  if (y0 >= y1) return;
  const int first_y = std::max(0, (pad_top + 1) / 2);
  const int first_x = std::max(0, (pad_left + 1) / 2);
  const int max_y = input_h + pad_top - 3;
  const int max_x = input_w + pad_left - 3;
  const int last_y = max_y < 0 ? 0 : std::min(output_h, max_y / 2 + 1);
  const int last_x = max_x < 0 ? 0 : std::min(output_w, max_x / 2 + 1);
  const auto load_stride2 = [](const float* row) {
    // Even lanes of 16 contiguous floats. Two ymm loads match the old four
    // xmm loads (through row+15) and the same 0,2,4,6,8,10,12,14 order.
    const __m256 lo = _mm256_loadu_ps(row);
    const __m256 hi = _mm256_loadu_ps(row + 8);
    const __m256 sh = _mm256_shuffle_ps(lo, hi, 0x88);
    return _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(sh), 0xD8));
  };
  const auto scalar4 = [&](float* o0, float* o1, float* o2, float* o3,
                           const float* f0, const float* f1, const float* f2, const float* f3,
                           float b0, float b1, float b2, float b3, int y, int x) {
    const int iy0 = y * 2 - pad_top;
    const int ix0 = x * 2 - pad_left;
    float s0 = b0, s1 = b1, s2 = b2, s3 = b3;
    for (int input = 0; input < input_channels; ++input) {
      const float* plane = src + std::size_t(input) * input_plane;
      const float* k0 = f0 + std::size_t(input) * 9;
      const float* k1 = f1 + std::size_t(input) * 9;
      const float* k2 = f2 + std::size_t(input) * 9;
      const float* k3 = f3 + std::size_t(input) * 9;
      for (int ky = 0; ky < 3; ++ky) {
        const int iy = iy0 + ky;
        if (iy < 0 || iy >= input_h) continue;
        for (int kx = 0; kx < 3; ++kx) {
          const int ix = ix0 + kx;
          if (ix < 0 || ix >= input_w) continue;
          const float value = plane[std::size_t(iy) * input_w + ix];
          const int ki = ky * 3 + kx;
          s0 += value * k0[ki];
          s1 += value * k1[ki];
          s2 += value * k2[ki];
          s3 += value * k3[ki];
        }
      }
    }
    const auto index = std::size_t(y) * output_w + x;
    if (accumulate) {
      s0 += o0[index]; s1 += o1[index];
      s2 += o2[index]; s3 += o3[index];
    }
    o0[index] = relu ? std::max(s0, 0.F) : s0;
    o1[index] = relu ? std::max(s1, 0.F) : s1;
    o2[index] = relu ? std::max(s2, 0.F) : s2;
    o3[index] = relu ? std::max(s3, 0.F) : s3;
  };
  int output = first_output;
  for (; output + 4 <= last_output; output += 4) {
    float* out0 = dst + std::size_t(output) * output_plane;
    float* out1 = out0 + output_plane;
    float* out2 = out1 + output_plane;
    float* out3 = out2 + output_plane;
    const float* f0 = weights + std::size_t(output) * input_channels * 9;
    const float* f1 = f0 + std::size_t(input_channels) * 9;
    const float* f2 = f1 + std::size_t(input_channels) * 9;
    const float* f3 = f2 + std::size_t(input_channels) * 9;
    const float b0 = bias ? bias[output] : 0.F;
    const float b1 = bias ? bias[output + 1] : 0.F;
    const float b2 = bias ? bias[output + 2] : 0.F;
    const float b3 = bias ? bias[output + 3] : 0.F;
    for (int y = y0; y < y1; ++y) {
      const int iy0 = y * 2 - pad_top;
      const bool interior_y = y >= first_y && y < last_y;
      int x = 0;
      for (; x < output_w; ++x) {
        if (interior_y && x >= first_x && x + 8 <= last_x) break;
        scalar4(out0, out1, out2, out3, f0, f1, f2, f3, b0, b1, b2, b3, y, x);
      }
      for (; x + 8 <= last_x; x += 8) {
        const int ix0 = x * 2 - pad_left;
        __m256 s0 = _mm256_set1_ps(b0), s1 = _mm256_set1_ps(b1);
        __m256 s2 = _mm256_set1_ps(b2), s3 = _mm256_set1_ps(b3);
        for (int input = 0; input < input_channels; ++input) {
          const float* plane = src + std::size_t(input) * input_plane +
                               std::size_t(iy0) * input_w + ix0;
          const float* k0 = f0 + std::size_t(input) * 9;
          const float* k1 = f1 + std::size_t(input) * 9;
          const float* k2 = f2 + std::size_t(input) * 9;
          const float* k3 = f3 + std::size_t(input) * 9;
          for (int ky = 0; ky < 3; ++ky) {
            const float* row = plane + std::size_t(ky) * input_w;
            for (int kx = 0; kx < 3; ++kx) {
              const __m256 values = load_stride2(row + kx);
              const int ki = ky * 3 + kx;
              s0 = _mm256_fmadd_ps(_mm256_set1_ps(k0[ki]), values, s0);
              s1 = _mm256_fmadd_ps(_mm256_set1_ps(k1[ki]), values, s1);
              s2 = _mm256_fmadd_ps(_mm256_set1_ps(k2[ki]), values, s2);
              s3 = _mm256_fmadd_ps(_mm256_set1_ps(k3[ki]), values, s3);
            }
          }
        }
        const auto index = std::size_t(y) * output_w + x;
        if (accumulate) {
          s0 = _mm256_add_ps(s0, _mm256_loadu_ps(out0 + index));
          s1 = _mm256_add_ps(s1, _mm256_loadu_ps(out1 + index));
          s2 = _mm256_add_ps(s2, _mm256_loadu_ps(out2 + index));
          s3 = _mm256_add_ps(s3, _mm256_loadu_ps(out3 + index));
        }
        if (relu) {
          const __m256 zero = _mm256_setzero_ps();
          s0 = _mm256_max_ps(s0, zero);
          s1 = _mm256_max_ps(s1, zero);
          s2 = _mm256_max_ps(s2, zero);
          s3 = _mm256_max_ps(s3, zero);
        }
        _mm256_storeu_ps(out0 + index, s0);
        _mm256_storeu_ps(out1 + index, s1);
        _mm256_storeu_ps(out2 + index, s2);
        _mm256_storeu_ps(out3 + index, s3);
      }
      for (; x < output_w; ++x) {
        scalar4(out0, out1, out2, out3, f0, f1, f2, f3, b0, b1, b2, b3, y, x);
      }
    }
  }
  const bool full_rows = row_begin <= 0 && (row_end < 0 || row_end >= output_h);
  if (output < last_output && !accumulate && full_rows) {
    Avx2Conv3x3Stride2(dst, src, weights, bias, output, last_output, input_channels,
                       input_h, input_w, output_h, output_w, pad_top, pad_left, relu);
  }
}

void Avx2Conv3x3Stride2x8(float* dst, const float* src, const float* weights,
                           const float* bias, int first_output, int last_output,
                           int input_channels, int input_h, int input_w,
                           int output_h, int output_w, int pad_top,
                           int pad_left, bool relu, int row_begin, int row_end,
                           bool accumulate) noexcept {
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(output_h) * output_w;
  int y0 = row_begin < 0 ? 0 : row_begin;
  int y1 = row_end < 0 ? output_h : row_end;
  if (y0 < 0) y0 = 0;
  if (y1 > output_h) y1 = output_h;
  if (y0 >= y1) return;
  const int first_y = std::max(0, (pad_top + 1) / 2);
  const int first_x = std::max(0, (pad_left + 1) / 2);
  const int max_y = input_h + pad_top - 3;
  const int max_x = input_w + pad_left - 3;
  const int last_y = max_y < 0 ? 0 : std::min(output_h, max_y / 2 + 1);
  const int last_x = max_x < 0 ? 0 : std::min(output_w, max_x / 2 + 1);
  const auto load_stride2 = [](const float* row) {
    // Even lanes of 16 contiguous floats. Two ymm loads match the old four
    // xmm loads (through row+15) and the same 0,2,4,6,8,10,12,14 order.
    const __m256 lo = _mm256_loadu_ps(row);
    const __m256 hi = _mm256_loadu_ps(row + 8);
    const __m256 sh = _mm256_shuffle_ps(lo, hi, 0x88);
    return _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(sh), 0xD8));
  };
  int output = first_output;
  for (; output + 8 <= last_output; output += 8) {
    float* out[8];
    const float* filter[8];
    float base[8];
    for (int q = 0; q < 8; ++q) {
      out[q] = dst + std::size_t(output + q) * output_plane;
      filter[q] = weights + std::size_t(output + q) * input_channels * 9;
      base[q] = bias ? bias[output + q] : 0.F;
    }
    const auto scalar8 = [&](int y, int x) {
      const int iy0 = y * 2 - pad_top;
      const int ix0 = x * 2 - pad_left;
      float sum[8] = {base[0], base[1], base[2], base[3],
                      base[4], base[5], base[6], base[7]};
      for (int input = 0; input < input_channels; ++input) {
        const float* plane = src + std::size_t(input) * input_plane;
        for (int ky = 0; ky < 3; ++ky) {
          const int iy = iy0 + ky;
          if (iy < 0 || iy >= input_h) continue;
          for (int kx = 0; kx < 3; ++kx) {
            const int ix = ix0 + kx;
            if (ix < 0 || ix >= input_w) continue;
            const float value = plane[std::size_t(iy) * input_w + ix];
            const int ki = ky * 3 + kx;
            for (int q = 0; q < 8; ++q)
              sum[q] += value * filter[q][std::size_t(input) * 9 + ki];
          }
        }
      }
      const auto index = std::size_t(y) * output_w + x;
      for (int q = 0; q < 8; ++q) {
        if (accumulate) sum[q] += out[q][index];
        out[q][index] = relu ? std::max(sum[q], 0.F) : sum[q];
      }
    };
    for (int y = y0; y < y1; ++y) {
      const int iy0 = y * 2 - pad_top;
      const bool interior_y = y >= first_y && y < last_y;
      int x = 0;
      for (; x < output_w; ++x) {
        if (interior_y && x >= first_x && x + 8 <= last_x) break;
        scalar8(y, x);
      }
      for (; x + 8 <= last_x; x += 8) {
        const int ix0 = x * 2 - pad_left;
        __m256 s0 = _mm256_set1_ps(base[0]);
        __m256 s1 = _mm256_set1_ps(base[1]);
        __m256 s2 = _mm256_set1_ps(base[2]);
        __m256 s3 = _mm256_set1_ps(base[3]);
        __m256 s4 = _mm256_set1_ps(base[4]);
        __m256 s5 = _mm256_set1_ps(base[5]);
        __m256 s6 = _mm256_set1_ps(base[6]);
        __m256 s7 = _mm256_set1_ps(base[7]);
        for (int input = 0; input < input_channels; ++input) {
          const float* plane = src + std::size_t(input) * input_plane +
                               std::size_t(iy0) * input_w + ix0;
          const float* k0 = filter[0] + std::size_t(input) * 9;
          const float* k1 = filter[1] + std::size_t(input) * 9;
          const float* k2 = filter[2] + std::size_t(input) * 9;
          const float* k3 = filter[3] + std::size_t(input) * 9;
          const float* k4 = filter[4] + std::size_t(input) * 9;
          const float* k5 = filter[5] + std::size_t(input) * 9;
          const float* k6 = filter[6] + std::size_t(input) * 9;
          const float* k7 = filter[7] + std::size_t(input) * 9;
          for (int ky = 0; ky < 3; ++ky) {
            const float* row = plane + std::size_t(ky) * input_w;
            for (int kx = 0; kx < 3; ++kx) {
              const __m256 values = load_stride2(row + kx);
              const int ki = ky * 3 + kx;
              s0 = _mm256_fmadd_ps(_mm256_set1_ps(k0[ki]), values, s0);
              s1 = _mm256_fmadd_ps(_mm256_set1_ps(k1[ki]), values, s1);
              s2 = _mm256_fmadd_ps(_mm256_set1_ps(k2[ki]), values, s2);
              s3 = _mm256_fmadd_ps(_mm256_set1_ps(k3[ki]), values, s3);
              s4 = _mm256_fmadd_ps(_mm256_set1_ps(k4[ki]), values, s4);
              s5 = _mm256_fmadd_ps(_mm256_set1_ps(k5[ki]), values, s5);
              s6 = _mm256_fmadd_ps(_mm256_set1_ps(k6[ki]), values, s6);
              s7 = _mm256_fmadd_ps(_mm256_set1_ps(k7[ki]), values, s7);
            }
          }
        }
        const auto index = std::size_t(y) * output_w + x;
        if (accumulate) {
          s0 = _mm256_add_ps(s0, _mm256_loadu_ps(out[0] + index));
          s1 = _mm256_add_ps(s1, _mm256_loadu_ps(out[1] + index));
          s2 = _mm256_add_ps(s2, _mm256_loadu_ps(out[2] + index));
          s3 = _mm256_add_ps(s3, _mm256_loadu_ps(out[3] + index));
          s4 = _mm256_add_ps(s4, _mm256_loadu_ps(out[4] + index));
          s5 = _mm256_add_ps(s5, _mm256_loadu_ps(out[5] + index));
          s6 = _mm256_add_ps(s6, _mm256_loadu_ps(out[6] + index));
          s7 = _mm256_add_ps(s7, _mm256_loadu_ps(out[7] + index));
        }
        if (relu) {
          const __m256 zero = _mm256_setzero_ps();
          s0 = _mm256_max_ps(s0, zero);
          s1 = _mm256_max_ps(s1, zero);
          s2 = _mm256_max_ps(s2, zero);
          s3 = _mm256_max_ps(s3, zero);
          s4 = _mm256_max_ps(s4, zero);
          s5 = _mm256_max_ps(s5, zero);
          s6 = _mm256_max_ps(s6, zero);
          s7 = _mm256_max_ps(s7, zero);
        }
        _mm256_storeu_ps(out[0] + index, s0);
        _mm256_storeu_ps(out[1] + index, s1);
        _mm256_storeu_ps(out[2] + index, s2);
        _mm256_storeu_ps(out[3] + index, s3);
        _mm256_storeu_ps(out[4] + index, s4);
        _mm256_storeu_ps(out[5] + index, s5);
        _mm256_storeu_ps(out[6] + index, s6);
        _mm256_storeu_ps(out[7] + index, s7);
      }
      for (; x < output_w; ++x) scalar8(y, x);
    }
  }
  const bool full_rows = row_begin <= 0 && (row_end < 0 || row_end >= output_h);
  if (output < last_output && !accumulate && full_rows) {
    Avx2Conv3x3Stride2x4(dst, src, weights, bias, output, last_output, input_channels,
                         input_h, input_w, output_h, output_w, pad_top, pad_left, relu,
                         0, -1, false);
  }
}

void Avx2WriteIdentityRgbToNchw(float* dst, const std::uint8_t* rgb, int width,
                                int height, int source_width, int left, int top,
                                const float* scale, const float* shift,
                                int row_width, int row_begin, int row_end) noexcept {
  const std::size_t plane = std::size_t(height) * row_width;
  const __m256 scale_b = _mm256_set1_ps(scale[0]);
  const __m256 scale_g = _mm256_set1_ps(scale[1]);
  const __m256 scale_r = _mm256_set1_ps(scale[2]);
  const __m256 shift_b = _mm256_set1_ps(shift[0]);
  const __m256 shift_g = _mm256_set1_ps(shift[1]);
  const __m256 shift_r = _mm256_set1_ps(shift[2]);
  const __m128i shuf_r = _mm_setr_epi8(0, 3, 6, 9, -1, -1, -1, -1,
                                       -1, -1, -1, -1, -1, -1, -1, -1);
  const __m128i shuf_g = _mm_setr_epi8(1, 4, 7, 10, -1, -1, -1, -1,
                                       -1, -1, -1, -1, -1, -1, -1, -1);
  const __m128i shuf_b = _mm_setr_epi8(2, 5, 8, 11, -1, -1, -1, -1,
                                       -1, -1, -1, -1, -1, -1, -1, -1);
  if (row_begin < 0) row_begin = 0;
  if (row_end < 0 || row_end > height) row_end = height;
  for (int y = row_begin; y < row_end; ++y) {
    const auto* src = rgb + (std::size_t(top + y) * source_width + left) * 3;
    float* blue = dst + std::size_t(y) * row_width;
    float* green = blue + plane;
    float* red = green + plane;
    int x = 0;
    // Eight pixels are 24 bytes. The second 16-byte load starts at byte 12
    // and reads 28, so require two extra source pixels rather than clearing
    // a stack buffer on every group.
    for (; x + 10 <= width; x += 8) {
      const auto* pix = src + std::size_t(x) * 3;
      const __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pix));
      const __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pix + 12));
      const __m256 vr = _mm256_set_m128(
          _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi8(hi, shuf_r))),
          _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi8(lo, shuf_r))));
      const __m256 vg = _mm256_set_m128(
          _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi8(hi, shuf_g))),
          _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi8(lo, shuf_g))));
      const __m256 vb = _mm256_set_m128(
          _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi8(hi, shuf_b))),
          _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi8(lo, shuf_b))));
      _mm256_storeu_ps(blue + x, _mm256_add_ps(_mm256_mul_ps(vb, scale_b), shift_b));
      _mm256_storeu_ps(green + x, _mm256_add_ps(_mm256_mul_ps(vg, scale_g), shift_g));
      _mm256_storeu_ps(red + x, _mm256_add_ps(_mm256_mul_ps(vr, scale_r), shift_r));
    }
    for (; x < width; ++x) {
      const auto* pixel = src + std::size_t(x) * 3;
      blue[x] = static_cast<float>(pixel[2]) * scale[0] + shift[0];
      green[x] = static_cast<float>(pixel[1]) * scale[1] + shift[1];
      red[x] = static_cast<float>(pixel[0]) * scale[2] + shift[2];
    }
  }
}

void Avx2AveragePool3x2Valid(float* dst, const float* src, int first_plane,
                             int last_plane, int input_height,
                             int input_width) noexcept {
  const int output_height = (input_height - 3) / 3 + 1;
  const int output_width = (input_width - 2) / 2 + 1;
  const std::size_t input_plane = std::size_t(input_height) * input_width;
  const std::size_t output_plane = std::size_t(output_height) * output_width;
  const __m256i even = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
  for (int plane = first_plane; plane < last_plane; ++plane) {
    const float* input = src + std::size_t(plane) * input_plane;
    float* output = dst + std::size_t(plane) * output_plane;
    for (int oy = 0; oy < output_height; ++oy) {
      const float* row0 = input + std::size_t(oy * 3) * input_width;
      const float* row1 = row0 + input_width;
      const float* row2 = row1 + input_width;
      float* row_out = output + std::size_t(oy) * output_width;
      int ox = 0;
      for (; ox + 8 <= output_width; ox += 8) {
        const float* base = row0 + ox * 2;
        const float* mid = row1 + ox * 2;
        const float* bot = row2 + ox * 2;
        __m256 sum = _mm256_i32gather_ps(base, even, 4);
        sum = _mm256_add_ps(sum, _mm256_i32gather_ps(base + 1, even, 4));
        sum = _mm256_add_ps(sum, _mm256_i32gather_ps(mid, even, 4));
        sum = _mm256_add_ps(sum, _mm256_i32gather_ps(mid + 1, even, 4));
        sum = _mm256_add_ps(sum, _mm256_i32gather_ps(bot, even, 4));
        sum = _mm256_add_ps(sum, _mm256_i32gather_ps(bot + 1, even, 4));
        // Match the scalar kernel's `sum / 6.F` exactly rather than a
        // reciprocal multiply that can change a few FP32 rounding bits.
        _mm256_storeu_ps(row_out + ox, _mm256_div_ps(sum, _mm256_set1_ps(6.F)));
      }
      for (; ox < output_width; ++ox) {
        const int x = ox * 2;
        float sum = row0[x];
        sum += row0[x + 1];
        sum += row1[x];
        sum += row1[x + 1];
        sum += row2[x];
        sum += row2[x + 1];
        row_out[ox] = sum / 6.F;
      }
    }
  }
}

void Avx2LayerNormAffine(float* dst, const float* src, const float* gamma,
                         const float* beta, std::size_t width, float mean,
                         float denom) noexcept {
  const __m256 mean_v = _mm256_set1_ps(mean);
  const __m256 denom_v = _mm256_set1_ps(denom);
  std::size_t column = 0;
  for (; column + 8 <= width; column += 8) {
    __m256 value = _mm256_div_ps(
        _mm256_sub_ps(_mm256_loadu_ps(src + column), mean_v), denom_v);
    value = _mm256_fmadd_ps(value, _mm256_loadu_ps(gamma + column),
                            _mm256_loadu_ps(beta + column));
    _mm256_storeu_ps(dst + column, value);
  }
  for (; column < width; ++column) {
    dst[column] = ((src[column] - mean) / denom) * gamma[column] + beta[column];
  }
}

void Avx2WriteBilinearRgbToNchw(float* dst, const std::uint8_t* rgb, int image_width,
                                int left, int top, int source_width, int source_height,
                                int width, int height, int row_width, const int* x0,
                                const int* x1, const float* dx, const float* scale,
                                const float* shift, int first_row,
                                int last_row) noexcept {
  const std::size_t plane = std::size_t(height) * row_width;
  const __m256 scale_b = _mm256_set1_ps(scale[0]);
  const __m256 scale_g = _mm256_set1_ps(scale[1]);
  const __m256 scale_r = _mm256_set1_ps(scale[2]);
  const __m256 shift_b = _mm256_set1_ps(shift[0]);
  const __m256 shift_g = _mm256_set1_ps(shift[1]);
  const __m256 shift_r = _mm256_set1_ps(shift[2]);
  const __m256 half = _mm256_set1_ps(.5F);
  const __m256i byte_max = _mm256_set1_epi32(255);
  const __m128i shuf_r = _mm_setr_epi8(0, 3, 6, 9, -1, -1, -1, -1,
                                       -1, -1, -1, -1, -1, -1, -1, -1);
  const __m128i shuf_g = _mm_setr_epi8(1, 4, 7, 10, -1, -1, -1, -1,
                                       -1, -1, -1, -1, -1, -1, -1, -1);
  const __m128i shuf_b = _mm_setr_epi8(2, 5, 8, 11, -1, -1, -1, -1,
                                       -1, -1, -1, -1, -1, -1, -1, -1);
  thread_local std::vector<float> widened;
  widened.resize(std::size_t(source_width) * 6);
  float* const upper_b = widened.data();
  float* const upper_g = upper_b + source_width;
  float* const upper_r = upper_g + source_width;
  float* const lower_b = upper_r + source_width;
  float* const lower_g = lower_b + source_width;
  float* const lower_r = lower_g + source_width;
  int cached_y0 = -1;
  int cached_y1 = -1;
  const auto widen_row = [&](float* blue, float* green, float* red,
                             const std::uint8_t* src) {
    int x = 0;
    for (; x + 8 <= source_width; x += 8) {
      alignas(16) std::uint8_t pack[32]{};
      std::memcpy(pack, src + std::size_t(x) * 3, 24);
      const __m128i lo = _mm_load_si128(reinterpret_cast<const __m128i*>(pack));
      const __m128i hi = _mm_load_si128(reinterpret_cast<const __m128i*>(pack + 12));
      const __m256 vr = _mm256_set_m128(
          _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi8(hi, shuf_r))),
          _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi8(lo, shuf_r))));
      const __m256 vg = _mm256_set_m128(
          _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi8(hi, shuf_g))),
          _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi8(lo, shuf_g))));
      const __m256 vb = _mm256_set_m128(
          _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi8(hi, shuf_b))),
          _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi8(lo, shuf_b))));
      _mm256_storeu_ps(red + x, vr);
      _mm256_storeu_ps(green + x, vg);
      _mm256_storeu_ps(blue + x, vb);
    }
    for (; x < source_width; ++x) {
      const auto* pixel = src + std::size_t(x) * 3;
      red[x] = static_cast<float>(pixel[0]);
      green[x] = static_cast<float>(pixel[1]);
      blue[x] = static_cast<float>(pixel[2]);
    }
  };
  for (int y = first_row; y < last_row; ++y) {
    const float fy = (float(y) + .5F) * source_height / height - .5F;
    const int y_floor = int(std::floor(fy));
    const int y0 = std::clamp(y_floor, 0, source_height - 1);
    const int y1 = std::min(y0 + 1, source_height - 1);
    const float dy = fy - y_floor;
    const float inverse_dy = 1.F - dy;
    if (y0 != cached_y0) {
      widen_row(upper_b, upper_g, upper_r,
                rgb + (std::size_t(top + y0) * image_width + left) * 3);
      cached_y0 = y0;
    }
    if (y1 != cached_y1) {
      if (y1 == cached_y0) {
        std::memcpy(lower_b, upper_b, std::size_t(source_width) * 3 * sizeof(float));
      } else {
        widen_row(lower_b, lower_g, lower_r,
                  rgb + (std::size_t(top + y1) * image_width + left) * 3);
      }
      cached_y1 = y1;
    }
    float* const blue = dst + std::size_t(y) * row_width;
    float* const green = blue + plane;
    float* const red = green + plane;
    const __m256 dy_v = _mm256_set1_ps(dy);
    const __m256 inv_dy = _mm256_set1_ps(inverse_dy);
    int x = 0;
    for (; x + 8 <= width; x += 8) {
      const __m256i ix0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x0 + x));
      const __m256i ix1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x1 + x));
      const __m256 w1 = _mm256_loadu_ps(dx + x);
      const __m256 w0 = _mm256_sub_ps(_mm256_set1_ps(1.F), w1);
      const __m256 u_r = _mm256_add_ps(
          _mm256_mul_ps(_mm256_i32gather_ps(upper_r, ix0, 4), w0),
          _mm256_mul_ps(_mm256_i32gather_ps(upper_r, ix1, 4), w1));
      const __m256 l_r = _mm256_add_ps(
          _mm256_mul_ps(_mm256_i32gather_ps(lower_r, ix0, 4), w0),
          _mm256_mul_ps(_mm256_i32gather_ps(lower_r, ix1, 4), w1));
      const __m256 u_g = _mm256_add_ps(
          _mm256_mul_ps(_mm256_i32gather_ps(upper_g, ix0, 4), w0),
          _mm256_mul_ps(_mm256_i32gather_ps(upper_g, ix1, 4), w1));
      const __m256 l_g = _mm256_add_ps(
          _mm256_mul_ps(_mm256_i32gather_ps(lower_g, ix0, 4), w0),
          _mm256_mul_ps(_mm256_i32gather_ps(lower_g, ix1, 4), w1));
      const __m256 u_b = _mm256_add_ps(
          _mm256_mul_ps(_mm256_i32gather_ps(upper_b, ix0, 4), w0),
          _mm256_mul_ps(_mm256_i32gather_ps(upper_b, ix1, 4), w1));
      const __m256 l_b = _mm256_add_ps(
          _mm256_mul_ps(_mm256_i32gather_ps(lower_b, ix0, 4), w0),
          _mm256_mul_ps(_mm256_i32gather_ps(lower_b, ix1, 4), w1));
      const __m256 v_r = _mm256_add_ps(_mm256_mul_ps(u_r, inv_dy), _mm256_mul_ps(l_r, dy_v));
      const __m256 v_g = _mm256_add_ps(_mm256_mul_ps(u_g, inv_dy), _mm256_mul_ps(l_g, dy_v));
      const __m256 v_b = _mm256_add_ps(_mm256_mul_ps(u_b, inv_dy), _mm256_mul_ps(l_b, dy_v));
      __m256i r_i = _mm256_cvttps_epi32(_mm256_add_ps(v_r, half));
      __m256i g_i = _mm256_cvttps_epi32(_mm256_add_ps(v_g, half));
      __m256i b_i = _mm256_cvttps_epi32(_mm256_add_ps(v_b, half));
      r_i = _mm256_min_epi32(_mm256_max_epi32(r_i, _mm256_setzero_si256()), byte_max);
      g_i = _mm256_min_epi32(_mm256_max_epi32(g_i, _mm256_setzero_si256()), byte_max);
      b_i = _mm256_min_epi32(_mm256_max_epi32(b_i, _mm256_setzero_si256()), byte_max);
      const __m256 sampled_r = _mm256_cvtepi32_ps(r_i);
      const __m256 sampled_g = _mm256_cvtepi32_ps(g_i);
      const __m256 sampled_b = _mm256_cvtepi32_ps(b_i);
      _mm256_storeu_ps(blue + x, _mm256_add_ps(_mm256_mul_ps(sampled_b, scale_b), shift_b));
      _mm256_storeu_ps(green + x, _mm256_add_ps(_mm256_mul_ps(sampled_g, scale_g), shift_g));
      _mm256_storeu_ps(red + x, _mm256_add_ps(_mm256_mul_ps(sampled_r, scale_r), shift_r));
    }
    const auto* const upper_row =
        rgb + (std::size_t(top + y0) * image_width + left) * 3;
    const auto* const lower_row =
        rgb + (std::size_t(top + y1) * image_width + left) * 3;
    for (; x < width; ++x) {
      const auto* const upper_left = upper_row + std::size_t(x0[x]) * 3;
      const auto* const upper_right = upper_row + std::size_t(x1[x]) * 3;
      const auto* const lower_left = lower_row + std::size_t(x0[x]) * 3;
      const auto* const lower_right = lower_row + std::size_t(x1[x]) * 3;
      const float inverse_dx = 1.F - dx[x];
      const float upper_red = float(upper_left[0]) * inverse_dx + float(upper_right[0]) * dx[x];
      const float lower_red = float(lower_left[0]) * inverse_dx + float(lower_right[0]) * dx[x];
      const float sampled_red = static_cast<float>(std::clamp(
          static_cast<int>(upper_red * inverse_dy + lower_red * dy + .5F), 0, 255));
      const float upper_green = float(upper_left[1]) * inverse_dx + float(upper_right[1]) * dx[x];
      const float lower_green = float(lower_left[1]) * inverse_dx + float(lower_right[1]) * dx[x];
      const float sampled_green = static_cast<float>(std::clamp(
          static_cast<int>(upper_green * inverse_dy + lower_green * dy + .5F), 0, 255));
      const float upper_blue = float(upper_left[2]) * inverse_dx + float(upper_right[2]) * dx[x];
      const float lower_blue = float(lower_left[2]) * inverse_dx + float(lower_right[2]) * dx[x];
      const float sampled_blue = static_cast<float>(std::clamp(
          static_cast<int>(upper_blue * inverse_dy + lower_blue * dy + .5F), 0, 255));
      blue[x] = sampled_blue * scale[0] + shift[0];
      green[x] = sampled_green * scale[1] + shift[1];
      red[x] = sampled_red * scale[2] + shift[2];
    }
  }
}

namespace {

// Two ymm spatial halves share each weight. Four outputs fill eight ymm
// accumulators (AVX2's latency product) without a ninth live vector that
// would spill. Weights stay in [row][K]; this is not an 8-wide packed
// broadcast.
[[gnu::noinline]] void Avx2ExpandProjectSpatial16(
    float* __restrict__ dst, const float* __restrict__ src,
    const float* __restrict__ expand_weights, const float* expand_bias,
    const float* __restrict__ project_weights, const float* project_bias,
    int channels, int hidden, std::size_t plane, std::size_t spatial,
    float* __restrict__ packed_act, float* __restrict__ hidden_tile) noexcept {
  for (int channel = 0; channel < channels; ++channel) {
    const float* in = src + std::size_t(channel) * plane + spatial;
    float* out = packed_act + std::size_t(channel) * 16;
    _mm256_storeu_ps(out, _mm256_loadu_ps(in));
    _mm256_storeu_ps(out + 8, _mm256_loadu_ps(in + 8));
  }
  const __m256 half = _mm256_set1_ps(.5F);
  const __m256 one = _mm256_set1_ps(1.F);
  const __m256 inv_sqrt2 = _mm256_set1_ps(0.7071067811865475244F);
  const auto gelu_pair = [&](float* slot) noexcept {
    const __m256 x0 = _mm256_loadu_ps(slot);
    const __m256 x1 = _mm256_loadu_ps(slot + 8);
    _mm256_storeu_ps(slot, _mm256_mul_ps(half, _mm256_mul_ps(
        x0, _mm256_add_ps(one, ErfPs(_mm256_mul_ps(x0, inv_sqrt2))))));
    _mm256_storeu_ps(slot + 8, _mm256_mul_ps(half, _mm256_mul_ps(
        x1, _mm256_add_ps(one, ErfPs(_mm256_mul_ps(x1, inv_sqrt2))))));
  };
  int hidden_channel = 0;
  for (; hidden_channel + 4 <= hidden; hidden_channel += 4) {
    const float* e0 = expand_weights + std::size_t(hidden_channel) * channels;
    const float* e1 = e0 + channels;
    const float* e2 = e1 + channels;
    const float* e3 = e2 + channels;
    __m256 a0 = _mm256_set1_ps(expand_bias ? expand_bias[hidden_channel] : 0.F);
    __m256 b0 = a0;
    __m256 a1 = _mm256_set1_ps(expand_bias ? expand_bias[hidden_channel + 1] : 0.F);
    __m256 b1 = a1;
    __m256 a2 = _mm256_set1_ps(expand_bias ? expand_bias[hidden_channel + 2] : 0.F);
    __m256 b2 = a2;
    __m256 a3 = _mm256_set1_ps(expand_bias ? expand_bias[hidden_channel + 3] : 0.F);
    __m256 b3 = a3;
    const float* act = packed_act;
    for (int channel = 0; channel < channels; ++channel) {
      const __m256 x0 = _mm256_loadu_ps(act);
      const __m256 x1 = _mm256_loadu_ps(act + 8);
      act += 16;
      const __m256 w0 = _mm256_set1_ps(e0[channel]);
      const __m256 w1 = _mm256_set1_ps(e1[channel]);
      const __m256 w2 = _mm256_set1_ps(e2[channel]);
      const __m256 w3 = _mm256_set1_ps(e3[channel]);
      a0 = _mm256_fmadd_ps(w0, x0, a0);
      b0 = _mm256_fmadd_ps(w0, x1, b0);
      a1 = _mm256_fmadd_ps(w1, x0, a1);
      b1 = _mm256_fmadd_ps(w1, x1, b1);
      a2 = _mm256_fmadd_ps(w2, x0, a2);
      b2 = _mm256_fmadd_ps(w2, x1, b2);
      a3 = _mm256_fmadd_ps(w3, x0, a3);
      b3 = _mm256_fmadd_ps(w3, x1, b3);
    }
    float* slot = hidden_tile + std::size_t(hidden_channel) * 16;
    _mm256_storeu_ps(slot, a0);
    _mm256_storeu_ps(slot + 8, b0);
    _mm256_storeu_ps(slot + 16, a1);
    _mm256_storeu_ps(slot + 24, b1);
    _mm256_storeu_ps(slot + 32, a2);
    _mm256_storeu_ps(slot + 40, b2);
    _mm256_storeu_ps(slot + 48, a3);
    _mm256_storeu_ps(slot + 56, b3);
    gelu_pair(slot);
    gelu_pair(slot + 16);
    gelu_pair(slot + 32);
    gelu_pair(slot + 48);
  }
  for (; hidden_channel < hidden; ++hidden_channel) {
    const float* filter = expand_weights + std::size_t(hidden_channel) * channels;
    __m256 a = _mm256_set1_ps(expand_bias ? expand_bias[hidden_channel] : 0.F);
    __m256 b = a;
    const float* act = packed_act;
    for (int channel = 0; channel < channels; ++channel) {
      const __m256 x0 = _mm256_loadu_ps(act);
      const __m256 x1 = _mm256_loadu_ps(act + 8);
      act += 16;
      const __m256 w = _mm256_set1_ps(filter[channel]);
      a = _mm256_fmadd_ps(w, x0, a);
      b = _mm256_fmadd_ps(w, x1, b);
    }
    float* slot = hidden_tile + std::size_t(hidden_channel) * 16;
    _mm256_storeu_ps(slot, a);
    _mm256_storeu_ps(slot + 8, b);
    gelu_pair(slot);
  }
  int channel = 0;
  for (; channel + 4 <= channels; channel += 4) {
    const float* p0 = project_weights + std::size_t(channel) * hidden;
    const float* p1 = p0 + hidden;
    const float* p2 = p1 + hidden;
    const float* p3 = p2 + hidden;
    const float* act0 = packed_act + std::size_t(channel) * 16;
    __m256 a0 = _mm256_add_ps(_mm256_set1_ps(project_bias ? project_bias[channel] : 0.F),
                              _mm256_loadu_ps(act0));
    __m256 b0 = _mm256_add_ps(_mm256_set1_ps(project_bias ? project_bias[channel] : 0.F),
                              _mm256_loadu_ps(act0 + 8));
    const float* act1 = act0 + 16;
    __m256 a1 = _mm256_add_ps(_mm256_set1_ps(project_bias ? project_bias[channel + 1] : 0.F),
                              _mm256_loadu_ps(act1));
    __m256 b1 = _mm256_add_ps(_mm256_set1_ps(project_bias ? project_bias[channel + 1] : 0.F),
                              _mm256_loadu_ps(act1 + 8));
    const float* act2 = act1 + 16;
    __m256 a2 = _mm256_add_ps(_mm256_set1_ps(project_bias ? project_bias[channel + 2] : 0.F),
                              _mm256_loadu_ps(act2));
    __m256 b2 = _mm256_add_ps(_mm256_set1_ps(project_bias ? project_bias[channel + 2] : 0.F),
                              _mm256_loadu_ps(act2 + 8));
    const float* act3 = act2 + 16;
    __m256 a3 = _mm256_add_ps(_mm256_set1_ps(project_bias ? project_bias[channel + 3] : 0.F),
                              _mm256_loadu_ps(act3));
    __m256 b3 = _mm256_add_ps(_mm256_set1_ps(project_bias ? project_bias[channel + 3] : 0.F),
                              _mm256_loadu_ps(act3 + 8));
    const float* gelu = hidden_tile;
    for (int h = 0; h < hidden; ++h) {
      const __m256 g0 = _mm256_loadu_ps(gelu);
      const __m256 g1 = _mm256_loadu_ps(gelu + 8);
      gelu += 16;
      const __m256 w0 = _mm256_set1_ps(p0[h]);
      const __m256 w1 = _mm256_set1_ps(p1[h]);
      const __m256 w2 = _mm256_set1_ps(p2[h]);
      const __m256 w3 = _mm256_set1_ps(p3[h]);
      a0 = _mm256_fmadd_ps(w0, g0, a0);
      b0 = _mm256_fmadd_ps(w0, g1, b0);
      a1 = _mm256_fmadd_ps(w1, g0, a1);
      b1 = _mm256_fmadd_ps(w1, g1, b1);
      a2 = _mm256_fmadd_ps(w2, g0, a2);
      b2 = _mm256_fmadd_ps(w2, g1, b2);
      a3 = _mm256_fmadd_ps(w3, g0, a3);
      b3 = _mm256_fmadd_ps(w3, g1, b3);
    }
    _mm256_storeu_ps(dst + std::size_t(channel) * plane + spatial, a0);
    _mm256_storeu_ps(dst + std::size_t(channel) * plane + spatial + 8, b0);
    _mm256_storeu_ps(dst + std::size_t(channel + 1) * plane + spatial, a1);
    _mm256_storeu_ps(dst + std::size_t(channel + 1) * plane + spatial + 8, b1);
    _mm256_storeu_ps(dst + std::size_t(channel + 2) * plane + spatial, a2);
    _mm256_storeu_ps(dst + std::size_t(channel + 2) * plane + spatial + 8, b2);
    _mm256_storeu_ps(dst + std::size_t(channel + 3) * plane + spatial, a3);
    _mm256_storeu_ps(dst + std::size_t(channel + 3) * plane + spatial + 8, b3);
  }
  for (; channel < channels; ++channel) {
    const float* filter = project_weights + std::size_t(channel) * hidden;
    const float* act = packed_act + std::size_t(channel) * 16;
    __m256 a = _mm256_add_ps(_mm256_set1_ps(project_bias ? project_bias[channel] : 0.F),
                             _mm256_loadu_ps(act));
    __m256 b = _mm256_add_ps(_mm256_set1_ps(project_bias ? project_bias[channel] : 0.F),
                             _mm256_loadu_ps(act + 8));
    const float* gelu = hidden_tile;
    for (int h = 0; h < hidden; ++h) {
      const __m256 w = _mm256_set1_ps(filter[h]);
      a = _mm256_fmadd_ps(w, _mm256_loadu_ps(gelu), a);
      b = _mm256_fmadd_ps(w, _mm256_loadu_ps(gelu + 8), b);
      gelu += 16;
    }
    _mm256_storeu_ps(dst + std::size_t(channel) * plane + spatial, a);
    _mm256_storeu_ps(dst + std::size_t(channel) * plane + spatial + 8, b);
  }
}

}  // namespace

// SVTR MLP: expand 1x1, exact-style GELU, project 1x1, residual add.
// Eight spatial lanes share each weight, matching Avx512ExpandGeluProjectAdd
// with the AVX2 ErfPs already used by Avx2ExactGelu. The partial tail keeps
// the scalar channel order and std::erf.
void Avx2ExpandGeluProjectAdd(float* dst, const float* src,
                              const float* expand_weights, const float* expand_bias,
                              const float* project_weights, const float* project_bias,
                              int channels, int hidden, std::size_t plane,
                              std::size_t spatial_begin, std::size_t spatial_end) noexcept {
  if (!dst || !src || !expand_weights || !project_weights || channels <= 0 ||
      hidden <= 0 || spatial_begin >= spatial_end || spatial_end > plane) {
    return;
  }
  const __m256 half = _mm256_set1_ps(.5F);
  const __m256 one = _mm256_set1_ps(1.F);
  const __m256 inv_sqrt2 = _mm256_set1_ps(0.7071067811865475244F);
  thread_local std::vector<float> hidden_tile;
  thread_local std::vector<float> packed_act;
  // `PPOCR_DISABLE_EXPAND_SPATIAL2` restores the 8-wide tile.
  static const bool spatial2 =
      std::getenv("PPOCR_DISABLE_EXPAND_SPATIAL2") == nullptr;
  const std::size_t pack_stride = spatial2 ? 16 : 8;
  hidden_tile.resize(std::size_t(hidden) * pack_stride);
  packed_act.resize(std::size_t(channels) * pack_stride);
  const auto gelu = [&](__m256 x) noexcept {
    return _mm256_mul_ps(half, _mm256_mul_ps(x,
        _mm256_add_ps(one, ErfPs(_mm256_mul_ps(x, inv_sqrt2)))));
  };
  constexpr float inv_sqrt2s = 0.7071067811865475244F;
  const auto scalar_at = [&](std::size_t spatial) {
    for (int hidden_channel = 0; hidden_channel < hidden; ++hidden_channel) {
      float sum = expand_bias ? expand_bias[hidden_channel] : 0.F;
      const float* filter = expand_weights + std::size_t(hidden_channel) * channels;
      for (int channel = 0; channel < channels; ++channel) {
        sum += filter[channel] * src[std::size_t(channel) * plane + spatial];
      }
      hidden_tile[static_cast<std::size_t>(hidden_channel)] =
          sum * .5F * (1.F + std::erf(sum * inv_sqrt2s));
    }
    for (int channel = 0; channel < channels; ++channel) {
      float sum = (project_bias ? project_bias[channel] : 0.F) +
                  src[std::size_t(channel) * plane + spatial];
      const float* filter = project_weights + std::size_t(channel) * hidden;
      for (int hidden_channel = 0; hidden_channel < hidden; ++hidden_channel) {
        sum += filter[hidden_channel] * hidden_tile[static_cast<std::size_t>(hidden_channel)];
      }
      dst[std::size_t(channel) * plane + spatial] = sum;
    }
  };
  std::size_t spatial = spatial_begin;
  if (spatial2) {
    for (; spatial + 16 <= spatial_end; spatial += 16) {
      Avx2ExpandProjectSpatial16(
          dst, src, expand_weights, expand_bias, project_weights, project_bias,
          channels, hidden, plane, spatial, packed_act.data(), hidden_tile.data());
    }
  }
  for (; spatial + 8 <= spatial_end; spatial += 8) {
    for (int channel = 0; channel < channels; ++channel) {
      _mm256_storeu_ps(packed_act.data() + std::size_t(channel) * 8,
                       _mm256_loadu_ps(src + std::size_t(channel) * plane + spatial));
    }
    const float* act = packed_act.data();
    int hidden_channel = 0;
    // Four ymm accumulators leave AVX2 FMA latency exposed (4 cycles, 2/cycle).
    // Eight outputs keep each channel's FMA order identical and fill the pipes.
    // `PPOCR_DISABLE_AVX2_EXPAND8` restores the four-output tile.
    static const bool unroll8 =
        std::getenv("PPOCR_DISABLE_AVX2_EXPAND8") == nullptr;
    if (unroll8) {
      for (; hidden_channel + 8 <= hidden; hidden_channel += 8) {
        const float* e0 = expand_weights + std::size_t(hidden_channel) * channels;
        const float* e1 = e0 + channels;
        const float* e2 = e1 + channels;
        const float* e3 = e2 + channels;
        const float* e4 = e3 + channels;
        const float* e5 = e4 + channels;
        const float* e6 = e5 + channels;
        const float* e7 = e6 + channels;
        __m256 a0 = _mm256_set1_ps(expand_bias ? expand_bias[hidden_channel] : 0.F);
        __m256 a1 = _mm256_set1_ps(expand_bias ? expand_bias[hidden_channel + 1] : 0.F);
        __m256 a2 = _mm256_set1_ps(expand_bias ? expand_bias[hidden_channel + 2] : 0.F);
        __m256 a3 = _mm256_set1_ps(expand_bias ? expand_bias[hidden_channel + 3] : 0.F);
        __m256 a4 = _mm256_set1_ps(expand_bias ? expand_bias[hidden_channel + 4] : 0.F);
        __m256 a5 = _mm256_set1_ps(expand_bias ? expand_bias[hidden_channel + 5] : 0.F);
        __m256 a6 = _mm256_set1_ps(expand_bias ? expand_bias[hidden_channel + 6] : 0.F);
        __m256 a7 = _mm256_set1_ps(expand_bias ? expand_bias[hidden_channel + 7] : 0.F);
        for (int channel = 0; channel < channels; ++channel) {
          const __m256 x = _mm256_loadu_ps(act + std::size_t(channel) * 8);
          a0 = _mm256_fmadd_ps(_mm256_set1_ps(e0[channel]), x, a0);
          a1 = _mm256_fmadd_ps(_mm256_set1_ps(e1[channel]), x, a1);
          a2 = _mm256_fmadd_ps(_mm256_set1_ps(e2[channel]), x, a2);
          a3 = _mm256_fmadd_ps(_mm256_set1_ps(e3[channel]), x, a3);
          a4 = _mm256_fmadd_ps(_mm256_set1_ps(e4[channel]), x, a4);
          a5 = _mm256_fmadd_ps(_mm256_set1_ps(e5[channel]), x, a5);
          a6 = _mm256_fmadd_ps(_mm256_set1_ps(e6[channel]), x, a6);
          a7 = _mm256_fmadd_ps(_mm256_set1_ps(e7[channel]), x, a7);
        }
        float* slot = hidden_tile.data() + std::size_t(hidden_channel) * 8;
        _mm256_storeu_ps(slot, a0);
        _mm256_storeu_ps(slot + 8, a1);
        _mm256_storeu_ps(slot + 16, a2);
        _mm256_storeu_ps(slot + 24, a3);
        _mm256_storeu_ps(slot + 32, a4);
        _mm256_storeu_ps(slot + 40, a5);
        _mm256_storeu_ps(slot + 48, a6);
        _mm256_storeu_ps(slot + 56, a7);
        _mm256_storeu_ps(slot, gelu(_mm256_loadu_ps(slot)));
        _mm256_storeu_ps(slot + 8, gelu(_mm256_loadu_ps(slot + 8)));
        _mm256_storeu_ps(slot + 16, gelu(_mm256_loadu_ps(slot + 16)));
        _mm256_storeu_ps(slot + 24, gelu(_mm256_loadu_ps(slot + 24)));
        _mm256_storeu_ps(slot + 32, gelu(_mm256_loadu_ps(slot + 32)));
        _mm256_storeu_ps(slot + 40, gelu(_mm256_loadu_ps(slot + 40)));
        _mm256_storeu_ps(slot + 48, gelu(_mm256_loadu_ps(slot + 48)));
        _mm256_storeu_ps(slot + 56, gelu(_mm256_loadu_ps(slot + 56)));
      }
    }
    for (; hidden_channel + 4 <= hidden; hidden_channel += 4) {
      const float* e0 = expand_weights + std::size_t(hidden_channel) * channels;
      const float* e1 = e0 + channels;
      const float* e2 = e1 + channels;
      const float* e3 = e2 + channels;
      __m256 a0 = _mm256_set1_ps(expand_bias ? expand_bias[hidden_channel] : 0.F);
      __m256 a1 = _mm256_set1_ps(expand_bias ? expand_bias[hidden_channel + 1] : 0.F);
      __m256 a2 = _mm256_set1_ps(expand_bias ? expand_bias[hidden_channel + 2] : 0.F);
      __m256 a3 = _mm256_set1_ps(expand_bias ? expand_bias[hidden_channel + 3] : 0.F);
      for (int channel = 0; channel < channels; ++channel) {
        const __m256 x = _mm256_loadu_ps(act + std::size_t(channel) * 8);
        a0 = _mm256_fmadd_ps(_mm256_set1_ps(e0[channel]), x, a0);
        a1 = _mm256_fmadd_ps(_mm256_set1_ps(e1[channel]), x, a1);
        a2 = _mm256_fmadd_ps(_mm256_set1_ps(e2[channel]), x, a2);
        a3 = _mm256_fmadd_ps(_mm256_set1_ps(e3[channel]), x, a3);
      }
      _mm256_storeu_ps(hidden_tile.data() + std::size_t(hidden_channel) * 8, gelu(a0));
      _mm256_storeu_ps(hidden_tile.data() + std::size_t(hidden_channel + 1) * 8, gelu(a1));
      _mm256_storeu_ps(hidden_tile.data() + std::size_t(hidden_channel + 2) * 8, gelu(a2));
      _mm256_storeu_ps(hidden_tile.data() + std::size_t(hidden_channel + 3) * 8, gelu(a3));
    }
    for (; hidden_channel < hidden; ++hidden_channel) {
      const float* filter = expand_weights + std::size_t(hidden_channel) * channels;
      __m256 acc = _mm256_set1_ps(expand_bias ? expand_bias[hidden_channel] : 0.F);
      for (int channel = 0; channel < channels; ++channel) {
        acc = _mm256_fmadd_ps(_mm256_set1_ps(filter[channel]),
                              _mm256_loadu_ps(act + std::size_t(channel) * 8), acc);
      }
      _mm256_storeu_ps(hidden_tile.data() + std::size_t(hidden_channel) * 8, gelu(acc));
    }
    int channel = 0;
    if (unroll8) {
      for (; channel + 8 <= channels; channel += 8) {
        const float* p0 = project_weights + std::size_t(channel) * hidden;
        const float* p1 = p0 + hidden;
        const float* p2 = p1 + hidden;
        const float* p3 = p2 + hidden;
        const float* p4 = p3 + hidden;
        const float* p5 = p4 + hidden;
        const float* p6 = p5 + hidden;
        const float* p7 = p6 + hidden;
        __m256 a0 = _mm256_add_ps(_mm256_set1_ps(project_bias ? project_bias[channel] : 0.F),
                                  _mm256_loadu_ps(act + std::size_t(channel) * 8));
        __m256 a1 = _mm256_add_ps(_mm256_set1_ps(project_bias ? project_bias[channel + 1] : 0.F),
                                  _mm256_loadu_ps(act + std::size_t(channel + 1) * 8));
        __m256 a2 = _mm256_add_ps(_mm256_set1_ps(project_bias ? project_bias[channel + 2] : 0.F),
                                  _mm256_loadu_ps(act + std::size_t(channel + 2) * 8));
        __m256 a3 = _mm256_add_ps(_mm256_set1_ps(project_bias ? project_bias[channel + 3] : 0.F),
                                  _mm256_loadu_ps(act + std::size_t(channel + 3) * 8));
        __m256 a4 = _mm256_add_ps(_mm256_set1_ps(project_bias ? project_bias[channel + 4] : 0.F),
                                  _mm256_loadu_ps(act + std::size_t(channel + 4) * 8));
        __m256 a5 = _mm256_add_ps(_mm256_set1_ps(project_bias ? project_bias[channel + 5] : 0.F),
                                  _mm256_loadu_ps(act + std::size_t(channel + 5) * 8));
        __m256 a6 = _mm256_add_ps(_mm256_set1_ps(project_bias ? project_bias[channel + 6] : 0.F),
                                  _mm256_loadu_ps(act + std::size_t(channel + 6) * 8));
        __m256 a7 = _mm256_add_ps(_mm256_set1_ps(project_bias ? project_bias[channel + 7] : 0.F),
                                  _mm256_loadu_ps(act + std::size_t(channel + 7) * 8));
        for (int h = 0; h < hidden; ++h) {
          const __m256 g = _mm256_loadu_ps(hidden_tile.data() + std::size_t(h) * 8);
          a0 = _mm256_fmadd_ps(_mm256_set1_ps(p0[h]), g, a0);
          a1 = _mm256_fmadd_ps(_mm256_set1_ps(p1[h]), g, a1);
          a2 = _mm256_fmadd_ps(_mm256_set1_ps(p2[h]), g, a2);
          a3 = _mm256_fmadd_ps(_mm256_set1_ps(p3[h]), g, a3);
          a4 = _mm256_fmadd_ps(_mm256_set1_ps(p4[h]), g, a4);
          a5 = _mm256_fmadd_ps(_mm256_set1_ps(p5[h]), g, a5);
          a6 = _mm256_fmadd_ps(_mm256_set1_ps(p6[h]), g, a6);
          a7 = _mm256_fmadd_ps(_mm256_set1_ps(p7[h]), g, a7);
        }
        _mm256_storeu_ps(dst + std::size_t(channel) * plane + spatial, a0);
        _mm256_storeu_ps(dst + std::size_t(channel + 1) * plane + spatial, a1);
        _mm256_storeu_ps(dst + std::size_t(channel + 2) * plane + spatial, a2);
        _mm256_storeu_ps(dst + std::size_t(channel + 3) * plane + spatial, a3);
        _mm256_storeu_ps(dst + std::size_t(channel + 4) * plane + spatial, a4);
        _mm256_storeu_ps(dst + std::size_t(channel + 5) * plane + spatial, a5);
        _mm256_storeu_ps(dst + std::size_t(channel + 6) * plane + spatial, a6);
        _mm256_storeu_ps(dst + std::size_t(channel + 7) * plane + spatial, a7);
      }
    }
    for (; channel + 4 <= channels; channel += 4) {
      const float* p0 = project_weights + std::size_t(channel) * hidden;
      const float* p1 = p0 + hidden;
      const float* p2 = p1 + hidden;
      const float* p3 = p2 + hidden;
      __m256 a0 = _mm256_add_ps(_mm256_set1_ps(project_bias ? project_bias[channel] : 0.F),
                                _mm256_loadu_ps(act + std::size_t(channel) * 8));
      __m256 a1 = _mm256_add_ps(_mm256_set1_ps(project_bias ? project_bias[channel + 1] : 0.F),
                                _mm256_loadu_ps(act + std::size_t(channel + 1) * 8));
      __m256 a2 = _mm256_add_ps(_mm256_set1_ps(project_bias ? project_bias[channel + 2] : 0.F),
                                _mm256_loadu_ps(act + std::size_t(channel + 2) * 8));
      __m256 a3 = _mm256_add_ps(_mm256_set1_ps(project_bias ? project_bias[channel + 3] : 0.F),
                                _mm256_loadu_ps(act + std::size_t(channel + 3) * 8));
      for (int h = 0; h < hidden; ++h) {
        const __m256 g = _mm256_loadu_ps(hidden_tile.data() + std::size_t(h) * 8);
        a0 = _mm256_fmadd_ps(_mm256_set1_ps(p0[h]), g, a0);
        a1 = _mm256_fmadd_ps(_mm256_set1_ps(p1[h]), g, a1);
        a2 = _mm256_fmadd_ps(_mm256_set1_ps(p2[h]), g, a2);
        a3 = _mm256_fmadd_ps(_mm256_set1_ps(p3[h]), g, a3);
      }
      _mm256_storeu_ps(dst + std::size_t(channel) * plane + spatial, a0);
      _mm256_storeu_ps(dst + std::size_t(channel + 1) * plane + spatial, a1);
      _mm256_storeu_ps(dst + std::size_t(channel + 2) * plane + spatial, a2);
      _mm256_storeu_ps(dst + std::size_t(channel + 3) * plane + spatial, a3);
    }
    for (; channel < channels; ++channel) {
      const float* filter = project_weights + std::size_t(channel) * hidden;
      __m256 acc = _mm256_add_ps(_mm256_set1_ps(project_bias ? project_bias[channel] : 0.F),
                                 _mm256_loadu_ps(act + std::size_t(channel) * 8));
      for (int h = 0; h < hidden; ++h) {
        acc = _mm256_fmadd_ps(_mm256_set1_ps(filter[h]),
                              _mm256_loadu_ps(hidden_tile.data() + std::size_t(h) * 8), acc);
      }
      _mm256_storeu_ps(dst + std::size_t(channel) * plane + spatial, acc);
    }
  }
  for (; spatial < spatial_end; ++spatial) scalar_at(spatial);
}

// Row-fused 5x5 depthwise plus pointwise. The depthwise row stays in a
// per-thread buffer so the 1x1 does not reread a full NCHW depthwise plane.
// Channel and tap order match DepthwiseConv then PointwiseConv.
void Avx2DepthwisePointwiseConv5x5S1(float* dst, const float* src,
                                     const float* dw_weights, const float* dw_bias,
                                     const float* pw_weights, const float* pw_bias,
                                     int channels, int output_channels, int height,
                                     int width, int activation, int y0, int y1) noexcept {
  if (!dst || !src || !dw_weights || !pw_weights || channels <= 0 ||
      output_channels <= 0 || height <= 0 || width <= 0) {
    return;
  }
  if (y0 < 0) y0 = 0;
  if (y1 < 0 || y1 > height) y1 = height;
  if (y0 >= y1) return;
  const std::size_t plane = std::size_t(height) * width;
  const int first_y = 2;
  const int first_x = 2;
  const int last_y = std::min(height, height - 2);
  const int last_x = std::min(width, width - 2);
  const __m256 zero = _mm256_setzero_ps();
  const __m256 one = _mm256_set1_ps(1.F);
  const __m256 half = _mm256_set1_ps(.5F);
  const __m256 sixth = _mm256_set1_ps(1.F / 6.F);
  const __m256 inv_sqrt2 = _mm256_set1_ps(0.7071067811865475244F);
  const auto activate = [&](__m256 x) noexcept {
    if (activation == 1) return _mm256_max_ps(x, zero);
    if (activation == 2) {
      const __m256 gate = _mm256_min_ps(one, _mm256_max_ps(zero, _mm256_fmadd_ps(x, sixth, half)));
      return _mm256_mul_ps(x, gate);
    }
    if (activation == 3) {
      return _mm256_mul_ps(half, _mm256_mul_ps(x,
          _mm256_add_ps(one, ErfPs(_mm256_mul_ps(x, inv_sqrt2)))));
    }
    return x;
  };
  const auto activate_scalar = [&](float x) noexcept {
    if (activation == 1) return std::max(x, 0.F);
    if (activation == 2) return x * std::clamp(x / 6.F + .5F, 0.F, 1.F);
    if (activation == 3) return x * .5F * (1.F + std::erf(x * 0.7071067811865475244F));
    return x;
  };
  thread_local std::vector<float> dw_row;
  dw_row.resize(std::size_t(channels) * width);
  for (int y = y0; y < y1; ++y) {
    const int iy0 = y - 2;
    const int interior_begin = y >= first_y && y < last_y ? first_x : 0;
    const int interior_end = y >= first_y && y < last_y ? last_x : 0;
    for (int channel = 0; channel < channels; ++channel) {
      const float* in = src + std::size_t(channel) * plane;
      const float* filter = dw_weights + std::size_t(channel) * 25;
      float* out = dw_row.data() + std::size_t(channel) * width;
      const float base = dw_bias ? dw_bias[channel] : 0.F;
      int x = 0;
      for (; x < interior_begin; ++x) {
        float sum = base;
        const int ix0 = x - 2;
        for (int ky = 0; ky < 5; ++ky) {
          const int iy = iy0 + ky;
          if (iy < 0 || iy >= height) continue;
          for (int kx = 0; kx < 5; ++kx) {
            const int ix = ix0 + kx;
            if (ix >= 0 && ix < width) sum += in[std::size_t(iy) * width + ix] * filter[ky * 5 + kx];
          }
        }
        out[x] = sum;
      }
      for (; x + 8 <= interior_end; x += 8) {
        __m256 sum = _mm256_set1_ps(base);
        const float* row0 = in + std::size_t(iy0) * width + x - 2;
        for (int ky = 0; ky < 5; ++ky) {
          const float* row = row0 + std::size_t(ky) * width;
          const float* k = filter + ky * 5;
          sum = _mm256_fmadd_ps(_mm256_set1_ps(k[0]), _mm256_loadu_ps(row), sum);
          sum = _mm256_fmadd_ps(_mm256_set1_ps(k[1]), _mm256_loadu_ps(row + 1), sum);
          sum = _mm256_fmadd_ps(_mm256_set1_ps(k[2]), _mm256_loadu_ps(row + 2), sum);
          sum = _mm256_fmadd_ps(_mm256_set1_ps(k[3]), _mm256_loadu_ps(row + 3), sum);
          sum = _mm256_fmadd_ps(_mm256_set1_ps(k[4]), _mm256_loadu_ps(row + 4), sum);
        }
        _mm256_storeu_ps(out + x, sum);
      }
      for (; x < width; ++x) {
        float sum = base;
        const int ix0 = x - 2;
        for (int ky = 0; ky < 5; ++ky) {
          const int iy = iy0 + ky;
          if (iy < 0 || iy >= height) continue;
          for (int kx = 0; kx < 5; ++kx) {
            const int ix = ix0 + kx;
            if (ix >= 0 && ix < width) sum += in[std::size_t(iy) * width + ix] * filter[ky * 5 + kx];
          }
        }
        out[x] = sum;
      }
    }
    const std::size_t row = std::size_t(y) * width;
    int output = 0;
    for (; output + 4 <= output_channels; output += 4) {
      const float b0 = pw_bias ? pw_bias[output] : 0.F;
      const float b1 = pw_bias ? pw_bias[output + 1] : 0.F;
      const float b2 = pw_bias ? pw_bias[output + 2] : 0.F;
      const float b3 = pw_bias ? pw_bias[output + 3] : 0.F;
      const float* w0 = pw_weights + std::size_t(output) * channels;
      const float* w1 = w0 + channels;
      const float* w2 = w1 + channels;
      const float* w3 = w2 + channels;
      int x = 0;
      for (; x + 8 <= width; x += 8) {
        __m256 s0 = _mm256_set1_ps(b0), s1 = _mm256_set1_ps(b1);
        __m256 s2 = _mm256_set1_ps(b2), s3 = _mm256_set1_ps(b3);
        for (int input = 0; input < channels; ++input) {
          const __m256 v = _mm256_loadu_ps(dw_row.data() + std::size_t(input) * width + x);
          s0 = _mm256_fmadd_ps(_mm256_set1_ps(w0[input]), v, s0);
          s1 = _mm256_fmadd_ps(_mm256_set1_ps(w1[input]), v, s1);
          s2 = _mm256_fmadd_ps(_mm256_set1_ps(w2[input]), v, s2);
          s3 = _mm256_fmadd_ps(_mm256_set1_ps(w3[input]), v, s3);
        }
        s0 = activate(s0); s1 = activate(s1); s2 = activate(s2); s3 = activate(s3);
        _mm256_storeu_ps(dst + std::size_t(output) * plane + row + x, s0);
        _mm256_storeu_ps(dst + std::size_t(output + 1) * plane + row + x, s1);
        _mm256_storeu_ps(dst + std::size_t(output + 2) * plane + row + x, s2);
        _mm256_storeu_ps(dst + std::size_t(output + 3) * plane + row + x, s3);
      }
      for (; x < width; ++x) {
        float s0 = b0, s1 = b1, s2 = b2, s3 = b3;
        for (int input = 0; input < channels; ++input) {
          const float v = dw_row[std::size_t(input) * width + x];
          s0 += w0[input] * v; s1 += w1[input] * v;
          s2 += w2[input] * v; s3 += w3[input] * v;
        }
        dst[std::size_t(output) * plane + row + x] = activate_scalar(s0);
        dst[std::size_t(output + 1) * plane + row + x] = activate_scalar(s1);
        dst[std::size_t(output + 2) * plane + row + x] = activate_scalar(s2);
        dst[std::size_t(output + 3) * plane + row + x] = activate_scalar(s3);
      }
    }
    for (; output < output_channels; ++output) {
      const float* filter = pw_weights + std::size_t(output) * channels;
      const float base = pw_bias ? pw_bias[output] : 0.F;
      float* out = dst + std::size_t(output) * plane + row;
      int x = 0;
      for (; x + 8 <= width; x += 8) {
        __m256 sum = _mm256_set1_ps(base);
        for (int input = 0; input < channels; ++input) {
          sum = _mm256_fmadd_ps(_mm256_set1_ps(filter[input]),
                                _mm256_loadu_ps(dw_row.data() + std::size_t(input) * width + x),
                                sum);
        }
        _mm256_storeu_ps(out + x, activate(sum));
      }
      for (; x < width; ++x) {
        float sum = base;
        for (int input = 0; input < channels; ++input)
          sum += filter[input] * dw_row[std::size_t(input) * width + x];
        out[x] = activate_scalar(sum);
      }
    }
  }
}

// K-contiguous 32-column panels of the CTC vocabulary matrix. Each output
// element is bias plus K in ascending order, matching Avx2GemmRows.
namespace {
float Avx2ReduceMaxPs(__m256 v) noexcept {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 m = _mm_max_ps(lo, hi);
  m = _mm_max_ps(m, _mm_movehl_ps(m, m));
  m = _mm_max_ps(m, _mm_shuffle_ps(m, m, 0x1));
  return _mm_cvtss_f32(m);
}
float Avx2ReduceAddPs(__m256 v) noexcept {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(s);
}
}  // namespace

void Avx2Depthwise3x3Stride2x1(float* dst, const float* src, const float* weights,
                               const float* bias, int first_channel, int last_channel,
                               int input_h, int input_w, int output_h,
                               int output_w) noexcept {
  if (!dst || !src || !weights || output_h <= 0 || output_w <= 0) return;
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(output_h) * output_w;
  const int first_x = 1;
  const int last_x = std::max(first_x, output_w - 1);
  for (int channel = first_channel; channel < last_channel; ++channel) {
    const float* in = src + std::size_t(channel) * input_plane;
    const float* filter = weights + std::size_t(channel) * 9;
    float* out = dst + std::size_t(channel) * output_plane;
    const float base = bias ? bias[channel] : 0.F;
    for (int oy = 0; oy < output_h; ++oy) {
      const int iy0 = oy * 2 - 1;
      int ox = 0;
      for (; ox < first_x && ox < output_w; ++ox) {
        float sum = base;
        const int ix0 = ox - 1;
        for (int ky = 0; ky < 3; ++ky) {
          const int iy = iy0 + ky;
          if (iy < 0 || iy >= input_h) continue;
          for (int kx = 0; kx < 3; ++kx) {
            const int ix = ix0 + kx;
            if (ix >= 0 && ix < input_w)
              sum += in[std::size_t(iy) * input_w + ix] * filter[ky * 3 + kx];
          }
        }
        out[std::size_t(oy) * output_w + ox] = sum;
      }
      for (; ox + 8 <= last_x; ox += 8) {
        __m256 sum = _mm256_set1_ps(base);
        for (int ky = 0; ky < 3; ++ky) {
          const int iy = iy0 + ky;
          if (iy < 0 || iy >= input_h) continue;
          const float* row = in + std::size_t(iy) * input_w + ox - 1;
          const float* kernel = filter + ky * 3;
          sum = _mm256_fmadd_ps(_mm256_set1_ps(kernel[0]), _mm256_loadu_ps(row), sum);
          sum = _mm256_fmadd_ps(_mm256_set1_ps(kernel[1]), _mm256_loadu_ps(row + 1), sum);
          sum = _mm256_fmadd_ps(_mm256_set1_ps(kernel[2]), _mm256_loadu_ps(row + 2), sum);
        }
        _mm256_storeu_ps(out + std::size_t(oy) * output_w + ox, sum);
      }
      for (; ox < output_w; ++ox) {
        float sum = base;
        const int ix0 = ox - 1;
        for (int ky = 0; ky < 3; ++ky) {
          const int iy = iy0 + ky;
          if (iy < 0 || iy >= input_h) continue;
          for (int kx = 0; kx < 3; ++kx) {
            const int ix = ix0 + kx;
            if (ix >= 0 && ix < input_w)
              sum += in[std::size_t(iy) * input_w + ix] * filter[ky * 3 + kx];
          }
        }
        out[std::size_t(oy) * output_w + ox] = sum;
      }
    }
  }
}

void Avx2GemmPacked32(float* dst, const float* a, const float* packed_b,
                      const float* bias, int rows, int cols, int depth,
                      int* ctc_arg, float* ctc_max, float* ctc_sum) noexcept {
  const bool fuse_ctc = ctc_arg && ctc_max && ctc_sum;
  if ((!dst && !fuse_ctc) || !a || !packed_b || rows <= 0 || cols <= 0 || depth <= 0) return;
  if (fuse_ctc) {
    const float ninf_s = -std::numeric_limits<float>::infinity();
    for (int row = 0; row < rows; ++row) {
      ctc_arg[row] = 0;
      ctc_max[row] = ninf_s;
      ctc_sum[row] = 0.F;
    }
    const int panels = (cols + 31) / 32;
    const auto update_row = [&](__m256 v, int n0, int n_len, int row) noexcept {
      if (n_len < 8) {
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, v);
        for (int lane = n_len; lane < 8; ++lane) tmp[lane] = ninf_s;
        v = _mm256_load_ps(tmp);
      }
      const float pmax = Avx2ReduceMaxPs(v);
      const __m256 eq = _mm256_cmp_ps(v, _mm256_set1_ps(pmax), _CMP_EQ_OQ);
      const int bits = _mm256_movemask_ps(eq);
      const int lane = bits ? __builtin_ctz(bits) : 0;
      float& mx = ctc_max[row];
      float& sm = ctc_sum[row];
      if (pmax > mx) {
        if (std::isfinite(mx)) {
          sm *= std::exp(mx - pmax);
        } else {
          sm = 0.F;
        }
        mx = pmax;
        ctc_arg[row] = n0 + lane;
      }
      __m256 e = ExpPs(_mm256_sub_ps(v, _mm256_set1_ps(mx)));
      if (n_len < 8) {
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, e);
        for (int i = n_len; i < 8; ++i) tmp[i] = 0.F;
        e = _mm256_load_ps(tmp);
      }
      sm += Avx2ReduceAddPs(e);
    };
    // Eight rows share each 16-wide half of a 32-col packed panel, so B is
    // reread four times less often than the 2-row store-logits kernel.
    for (int panel = 0; panel < panels; ++panel) {
      const float* pb = packed_b + std::size_t(panel) * depth * 32;
      for (int half = 0; half < 2; ++half) {
        const int n0 = panel * 32 + half * 16;
        if (n0 >= cols) break;
        const int n_len = std::min(16, cols - n0);
        alignas(32) float btmp[16] = {};
        if (bias) {
          std::memcpy(btmp, bias + n0, std::size_t(n_len) * sizeof(float));
        }
        const __m256 bias0 = _mm256_load_ps(btmp);
        const __m256 bias1 = _mm256_load_ps(btmp + 8);
        const int low_len = std::min(8, n_len);
        const int high_len = n_len > 8 ? n_len - 8 : 0;
        // One 8-wide half at a time keeps eight accumulators in ymm registers.
        const auto accumulate8 = [&](const __m256 bias_v, int right_offset,
                                     int col0, int len) {
          if (len <= 0) return;
          int row = 0;
          for (; row + 8 <= rows; row += 8) {
            const float* left0 = a + std::size_t(row) * depth;
            __m256 a0 = bias_v, b0 = bias_v, c0 = bias_v, d0 = bias_v;
            __m256 e0 = bias_v, f0 = bias_v, g0 = bias_v, h0 = bias_v;
            for (int k = 0; k < depth; ++k) {
              const __m256 r = _mm256_loadu_ps(
                  pb + std::size_t(k) * 32 + half * 16 + right_offset);
              a0 = _mm256_fmadd_ps(_mm256_set1_ps(left0[k]), r, a0);
              b0 = _mm256_fmadd_ps(_mm256_set1_ps(left0[depth + k]), r, b0);
              c0 = _mm256_fmadd_ps(_mm256_set1_ps(left0[2 * depth + k]), r, c0);
              d0 = _mm256_fmadd_ps(_mm256_set1_ps(left0[3 * depth + k]), r, d0);
              e0 = _mm256_fmadd_ps(_mm256_set1_ps(left0[4 * depth + k]), r, e0);
              f0 = _mm256_fmadd_ps(_mm256_set1_ps(left0[5 * depth + k]), r, f0);
              g0 = _mm256_fmadd_ps(_mm256_set1_ps(left0[6 * depth + k]), r, g0);
              h0 = _mm256_fmadd_ps(_mm256_set1_ps(left0[7 * depth + k]), r, h0);
            }
            update_row(a0, col0, len, row);
            update_row(b0, col0, len, row + 1);
            update_row(c0, col0, len, row + 2);
            update_row(d0, col0, len, row + 3);
            update_row(e0, col0, len, row + 4);
            update_row(f0, col0, len, row + 5);
            update_row(g0, col0, len, row + 6);
            update_row(h0, col0, len, row + 7);
          }
          for (; row < rows; ++row) {
            const float* left = a + std::size_t(row) * depth;
            __m256 acc = bias_v;
            for (int k = 0; k < depth; ++k) {
              const __m256 r = _mm256_loadu_ps(
                  pb + std::size_t(k) * 32 + half * 16 + right_offset);
              acc = _mm256_fmadd_ps(_mm256_set1_ps(left[k]), r, acc);
            }
            update_row(acc, col0, len, row);
          }
        };
        accumulate8(bias0, 0, n0, low_len);
        accumulate8(bias1, 8, n0 + 8, high_len);
      }
    }
    return;
  }
  if (!dst) return;
  const int panels = (cols + 31) / 32;
  const auto panel_rows = [&](int row, int row_count) noexcept {
    const float* left0 = a + std::size_t(row) * depth;
    const float* left1 = row_count == 2 ? left0 + depth : nullptr;
    float* out0 = dst + std::size_t(row) * cols;
    float* out1 = row_count == 2 ? out0 + cols : nullptr;
    for (int panel = 0; panel < panels; ++panel) {
      const int n0 = panel * 32;
      const int n_len = std::min(32, cols - n0);
      const float* pb = packed_b + std::size_t(panel) * depth * 32;
      alignas(32) float btmp[32] = {};
      if (bias) std::memcpy(btmp, bias + n0, std::size_t(n_len) * sizeof(float));
      __m256 a0 = _mm256_load_ps(btmp);
      __m256 a1 = _mm256_load_ps(btmp + 8);
      __m256 a2 = _mm256_load_ps(btmp + 16);
      __m256 a3 = _mm256_load_ps(btmp + 24);
      __m256 b0 = a0, b1 = a1, b2 = a2, b3 = a3;
      for (int k = 0; k < depth; ++k) {
        const float* right = pb + std::size_t(k) * 32;
        const __m256 r0 = _mm256_loadu_ps(right);
        const __m256 r1 = _mm256_loadu_ps(right + 8);
        const __m256 r2 = _mm256_loadu_ps(right + 16);
        const __m256 r3 = _mm256_loadu_ps(right + 24);
        const __m256 s0 = _mm256_set1_ps(left0[k]);
        a0 = _mm256_fmadd_ps(s0, r0, a0);
        a1 = _mm256_fmadd_ps(s0, r1, a1);
        a2 = _mm256_fmadd_ps(s0, r2, a2);
        a3 = _mm256_fmadd_ps(s0, r3, a3);
        if (left1) {
          const __m256 s1 = _mm256_set1_ps(left1[k]);
          b0 = _mm256_fmadd_ps(s1, r0, b0);
          b1 = _mm256_fmadd_ps(s1, r1, b1);
          b2 = _mm256_fmadd_ps(s1, r2, b2);
          b3 = _mm256_fmadd_ps(s1, r3, b3);
        }
      }
      auto store = [&](float* out, __m256 c0, __m256 c1, __m256 c2, __m256 c3) {
        if (n_len == 32) {
          _mm256_storeu_ps(out + n0, c0);
          _mm256_storeu_ps(out + n0 + 8, c1);
          _mm256_storeu_ps(out + n0 + 16, c2);
          _mm256_storeu_ps(out + n0 + 24, c3);
          return;
        }
        alignas(32) float tmp[32];
        _mm256_store_ps(tmp, c0);
        _mm256_store_ps(tmp + 8, c1);
        _mm256_store_ps(tmp + 16, c2);
        _mm256_store_ps(tmp + 24, c3);
        std::memcpy(out + n0, tmp, std::size_t(n_len) * sizeof(float));
      };
      store(out0, a0, a1, a2, a3);
      if (out1) store(out1, b0, b1, b2, b3);
    }
  };
  int row = 0;
  for (; row + 2 <= rows; row += 2) panel_rows(row, 2);
  if (row < rows) panel_rows(row, 1);
}

void Avx2ThresholdRows(std::uint8_t* mask, const float* probability, int width,
                       int first_y, int last_y, float threshold) noexcept {
  const __m256 thresh = _mm256_set1_ps(threshold);
  for (int y = first_y; y < last_y; ++y) {
    const float* row = probability + std::size_t(y) * width;
    std::uint8_t* out = mask + std::size_t(y) * width;
    int x = 0;
    for (; x + 8 <= width; x += 8) {
      const int bits = _mm256_movemask_ps(
          _mm256_cmp_ps(_mm256_loadu_ps(row + x), thresh, _CMP_GT_OQ));
      for (int lane = 0; lane < 8; ++lane)
        out[x + lane] = static_cast<std::uint8_t>((bits >> lane) & 1);
    }
    for (; x < width; ++x) out[x] = row[x] > threshold ? 1 : 0;
  }
}

void Avx2HorizontalOrLeft(std::uint8_t* dst, const std::uint8_t* src, int width,
                          int first_y, int last_y) noexcept {
  for (int y = first_y; y < last_y; ++y) {
    const std::uint8_t* row = src + std::size_t(y) * width;
    std::uint8_t* out = dst + std::size_t(y) * width;
    __m128i previous = _mm_setzero_si128();
    int x = 0;
    for (; x + 16 <= width; x += 16) {
      const __m128i values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + x));
      const __m128i shifted = _mm_alignr_epi8(values, previous, 15);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(out + x), _mm_or_si128(values, shifted));
      previous = values;
    }
    if (x == 0) out[0] = row[0];
    for (; x < width; ++x)
      out[x] = static_cast<std::uint8_t>(row[x] | (x > 0 ? row[x - 1] : 0));
  }
}

void Avx2VerticalOr(std::uint8_t* dst, const std::uint8_t* src, int width,
                    int first_y, int last_y) noexcept {
  for (int y = first_y; y < last_y; ++y) {
    const std::uint8_t* row = src + std::size_t(y) * width;
    std::uint8_t* out = dst + std::size_t(y) * width;
    if (y == 0) {
      std::memcpy(out, row, static_cast<std::size_t>(width));
      continue;
    }
    const std::uint8_t* up = src + std::size_t(y - 1) * width;
    int x = 0;
    for (; x + 16 <= width; x += 16) {
      const __m128i values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + x));
      const __m128i above = _mm_loadu_si128(reinterpret_cast<const __m128i*>(up + x));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(out + x), _mm_or_si128(values, above));
    }
    for (; x < width; ++x) out[x] = static_cast<std::uint8_t>(row[x] | up[x]);
  }
}

void Avx2DetStemTailRows(float* dst, const float* conv0, const float* conv1_w,
                         const float* conv1_b, const float* conv2_w,
                         const float* conv2_b, const float* stem_w,
                         const float* stem_b, int height, int width, int out_h,
                         int out_w, int row_begin, int row_end,
                         bool stem_relu) noexcept {
  if (!dst || !conv0 || row_begin >= row_end) return;
  constexpr int kBand = 2;
  constexpr int kSpan = 16;
  const std::size_t out_plane = std::size_t(out_h) * out_w;
  const auto load_stride2 = [](const float* row) {
    // Even lanes of 16 contiguous floats. Two ymm loads match the old four
    // xmm loads (through row+15) and the same 0,2,4,6,8,10,12,14 order.
    const __m256 lo = _mm256_loadu_ps(row);
    const __m256 hi = _mm256_loadu_ps(row + 8);
    const __m256 sh = _mm256_shuffle_ps(lo, hi, 0x88);
    return _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(sh), 0xD8));
  };
  const auto c0_row = [&](int ch, int y) {
    return conv0 + (std::size_t(ch) * height + y) * width;
  };
  struct Buf {
    std::vector<float> conv1;
    std::vector<float> conv2;
    std::vector<float> pool;
  };
  static thread_local Buf buf;
  const std::size_t span_n = std::size_t(kSpan) * width;
  if (buf.conv1.size() < 8 * span_n) buf.conv1.resize(8 * span_n);
  if (buf.conv2.size() < 16 * span_n) buf.conv2.resize(16 * span_n);
  if (buf.pool.size() < 16 * span_n) buf.pool.resize(16 * span_n);
  const __m256 zero = _mm256_setzero_ps();
  for (int band = row_begin; band < row_end; band += kBand) {
    const int y1 = std::min(row_end, band + kBand);
    const int c2_r0 = std::max(0, band * 2 - 1);
    const int c2_r1 = std::min(height, y1 * 2);
    const int c1_r0 = c2_r0;
    const int c1_r1 = std::min(height, c2_r1 + 1);
    const int c2_rows = c2_r1 - c2_r0;
    const int c1_rows = c1_r1 - c1_r0;
    if (c2_rows <= 0 || c1_rows <= 0 || c2_rows > kSpan || c1_rows > kSpan) continue;
    const auto conv2x2_8 = [&](float* strip, int strip_rows, int abs_r0, int ry0, int ry1,
                               int in_ch, const float* weights, const float* bias,
                               auto&& src_row) {
      for (int y = ry0; y < ry1; ++y) {
        float* outs[8];
        const float* filt[8];
        float base[8];
        for (int q = 0; q < 8; ++q) {
          outs[q] = strip + (std::size_t(q) * strip_rows + (y - abs_r0)) * width;
          filt[q] = weights + std::size_t(q) * in_ch * 4;
          base[q] = bias[q];
        }
        const bool last_row = y + 1 >= height;
        auto scalar_at = [&](int x) {
          float s[8] = {base[0], base[1], base[2], base[3], base[4], base[5], base[6], base[7]};
          for (int ic = 0; ic < in_ch; ++ic) {
            const float* row0 = src_row(ic, y);
            const float* row1 = last_row ? nullptr : src_row(ic, y + 1);
            for (int q = 0; q < 8; ++q) {
              const float* k = filt[q] + ic * 4;
              if (x + 1 < width) {
                s[q] += row0[x] * k[0] + row0[x + 1] * k[1];
                if (row1) s[q] += row1[x] * k[2] + row1[x + 1] * k[3];
              } else {
                s[q] += row0[x] * k[0];
                if (row1) s[q] += row1[x] * k[2];
              }
            }
          }
          for (int q = 0; q < 8; ++q) outs[q][x] = std::max(s[q], 0.F);
        };
        int x = 0;
        if (!last_row) {
          for (; x + 8 <= width - 1; x += 8) {
            __m256 s0 = _mm256_set1_ps(base[0]), s1 = _mm256_set1_ps(base[1]);
            __m256 s2 = _mm256_set1_ps(base[2]), s3 = _mm256_set1_ps(base[3]);
            __m256 s4 = _mm256_set1_ps(base[4]), s5 = _mm256_set1_ps(base[5]);
            __m256 s6 = _mm256_set1_ps(base[6]), s7 = _mm256_set1_ps(base[7]);
            for (int ic = 0; ic < in_ch; ++ic) {
              const float* row0 = src_row(ic, y) + x;
              const float* row1 = src_row(ic, y + 1) + x;
              const __m256 v00 = _mm256_loadu_ps(row0);
              const __m256 v01 = _mm256_loadu_ps(row0 + 1);
              const __m256 v10 = _mm256_loadu_ps(row1);
              const __m256 v11 = _mm256_loadu_ps(row1 + 1);
              const auto fma4 = [&](__m256& acc, const float* k) {
                acc = _mm256_fmadd_ps(_mm256_set1_ps(k[0]), v00, acc);
                acc = _mm256_fmadd_ps(_mm256_set1_ps(k[1]), v01, acc);
                acc = _mm256_fmadd_ps(_mm256_set1_ps(k[2]), v10, acc);
                acc = _mm256_fmadd_ps(_mm256_set1_ps(k[3]), v11, acc);
              };
              fma4(s0, filt[0] + ic * 4);
              fma4(s1, filt[1] + ic * 4);
              fma4(s2, filt[2] + ic * 4);
              fma4(s3, filt[3] + ic * 4);
              fma4(s4, filt[4] + ic * 4);
              fma4(s5, filt[5] + ic * 4);
              fma4(s6, filt[6] + ic * 4);
              fma4(s7, filt[7] + ic * 4);
            }
            s0 = _mm256_max_ps(s0, zero); s1 = _mm256_max_ps(s1, zero);
            s2 = _mm256_max_ps(s2, zero); s3 = _mm256_max_ps(s3, zero);
            s4 = _mm256_max_ps(s4, zero); s5 = _mm256_max_ps(s5, zero);
            s6 = _mm256_max_ps(s6, zero); s7 = _mm256_max_ps(s7, zero);
            _mm256_storeu_ps(outs[0] + x, s0); _mm256_storeu_ps(outs[1] + x, s1);
            _mm256_storeu_ps(outs[2] + x, s2); _mm256_storeu_ps(outs[3] + x, s3);
            _mm256_storeu_ps(outs[4] + x, s4); _mm256_storeu_ps(outs[5] + x, s5);
            _mm256_storeu_ps(outs[6] + x, s6); _mm256_storeu_ps(outs[7] + x, s7);
          }
        }
        for (; x < width; ++x) scalar_at(x);
      }
    };
    conv2x2_8(buf.conv1.data(), c1_rows, c1_r0, c1_r0, c1_r1, 16, conv1_w, conv1_b, c0_row);
    const auto c1_row = [&](int ch, int y) {
      return buf.conv1.data() + (std::size_t(ch) * c1_rows + (y - c1_r0)) * width;
    };
    conv2x2_8(buf.conv2.data(), c2_rows, c2_r0, c2_r0, c2_r1, 8, conv2_w, conv2_b, c1_row);
    conv2x2_8(buf.conv2.data() + 8 * std::size_t(c2_rows) * width, c2_rows, c2_r0, c2_r0,
              c2_r1, 8, conv2_w + 8 * 8 * 4, conv2_b + 8, c1_row);
    for (int ch = 0; ch < 16; ++ch) {
      for (int y = c2_r0; y < c2_r1; ++y) {
        const float* row0 = c0_row(ch, y);
        float* out = buf.pool.data() + (std::size_t(ch) * c2_rows + (y - c2_r0)) * width;
        if (y + 1 < height) {
          const float* row1 = c0_row(ch, y + 1);
          int x = 0;
          for (; x + 8 <= width - 1; x += 8) {
            const __m256 top = _mm256_max_ps(_mm256_loadu_ps(row0 + x), _mm256_loadu_ps(row0 + x + 1));
            const __m256 bottom = _mm256_max_ps(_mm256_loadu_ps(row1 + x), _mm256_loadu_ps(row1 + x + 1));
            _mm256_storeu_ps(out + x, _mm256_max_ps(top, bottom));
          }
          for (; x + 1 < width; ++x)
            out[x] = std::max(std::max(row0[x], row0[x + 1]), std::max(row1[x], row1[x + 1]));
          out[width - 1] = std::max(row0[width - 1], row1[width - 1]);
        } else {
          int x = 0;
          for (; x + 8 <= width - 1; x += 8) {
            _mm256_storeu_ps(out + x, _mm256_max_ps(_mm256_loadu_ps(row0 + x),
                                                    _mm256_loadu_ps(row0 + x + 1)));
          }
          for (; x + 1 < width; ++x) out[x] = std::max(row0[x], row0[x + 1]);
          out[width - 1] = row0[width - 1];
        }
      }
    }
    const auto sample = [&](int ic, int y, int x) -> float {
      if (y < 0 || x < 0 || y >= height || x >= width) return 0.F;
      if (ic < 16) return buf.pool[(std::size_t(ic) * c2_rows + (y - c2_r0)) * width + x];
      return buf.conv2[(std::size_t(ic - 16) * c2_rows + (y - c2_r0)) * width + x];
    };
    for (int oc0 = 0; oc0 < 16; oc0 += 4) {
      float* outs[4];
      const float* filt[4];
      float base[4];
      for (int q = 0; q < 4; ++q) {
        outs[q] = dst + std::size_t(oc0 + q) * out_plane;
        filt[q] = stem_w + std::size_t(oc0 + q) * 32 * 9;
        base[q] = stem_b ? stem_b[oc0 + q] : 0.F;
      }
      for (int y = band; y < y1; ++y) {
        const int iy0 = y * 2 - 1;
        const bool interior_y = iy0 >= 0 && iy0 + 2 < height;
        auto scalar_at = [&](int x) {
          const int ix0 = x * 2 - 1;
          float s[4] = {base[0], base[1], base[2], base[3]};
          for (int ic = 0; ic < 32; ++ic) {
            for (int ky = 0; ky < 3; ++ky) {
              const int iy = iy0 + ky;
              for (int kx = 0; kx < 3; ++kx) {
                const float value = sample(ic, iy, ix0 + kx);
                const int ki = ky * 3 + kx;
                for (int q = 0; q < 4; ++q) s[q] += value * filt[q][ic * 9 + ki];
              }
            }
          }
          const std::size_t index = std::size_t(y) * out_w + x;
          for (int q = 0; q < 4; ++q)
            outs[q][index] = stem_relu ? std::max(s[q], 0.F) : s[q];
        };
        int x = 0;
        for (; x < out_w; ++x) {
          const int ix0 = x * 2 - 1;
          if (interior_y && ix0 >= 0 && ix0 + 17 < width) break;
          scalar_at(x);
        }
        for (; x + 8 <= out_w; x += 8) {
          const int ix0 = x * 2 - 1;
          if (!(interior_y && ix0 >= 0 && ix0 + 17 < width)) break;
          __m256 s0 = _mm256_set1_ps(base[0]), s1 = _mm256_set1_ps(base[1]);
          __m256 s2 = _mm256_set1_ps(base[2]), s3 = _mm256_set1_ps(base[3]);
          for (int ic = 0; ic < 32; ++ic) {
            const float* row_base = ic < 16
                ? buf.pool.data() + std::size_t(ic) * c2_rows * width
                : buf.conv2.data() + std::size_t(ic - 16) * c2_rows * width;
            for (int ky = 0; ky < 3; ++ky) {
              const float* row = row_base + std::size_t(iy0 + ky - c2_r0) * width;
              for (int kx = 0; kx < 3; ++kx) {
                const __m256 values = load_stride2(row + ix0 + kx);
                const int ki = ky * 3 + kx;
                s0 = _mm256_fmadd_ps(_mm256_set1_ps(filt[0][ic * 9 + ki]), values, s0);
                s1 = _mm256_fmadd_ps(_mm256_set1_ps(filt[1][ic * 9 + ki]), values, s1);
                s2 = _mm256_fmadd_ps(_mm256_set1_ps(filt[2][ic * 9 + ki]), values, s2);
                s3 = _mm256_fmadd_ps(_mm256_set1_ps(filt[3][ic * 9 + ki]), values, s3);
              }
            }
          }
          if (stem_relu) {
            s0 = _mm256_max_ps(s0, zero); s1 = _mm256_max_ps(s1, zero);
            s2 = _mm256_max_ps(s2, zero); s3 = _mm256_max_ps(s3, zero);
          }
          const std::size_t index = std::size_t(y) * out_w + x;
          _mm256_storeu_ps(outs[0] + index, s0); _mm256_storeu_ps(outs[1] + index, s1);
          _mm256_storeu_ps(outs[2] + index, s2); _mm256_storeu_ps(outs[3] + index, s3);
        }
        for (; x < out_w; ++x) scalar_at(x);
      }
    }
  }
}

}  // namespace ppocr::detail::kernels
