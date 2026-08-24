#include "kernels.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <immintrin.h>
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
                         int input_channels, std::size_t plane) noexcept {
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
    std::size_t index = 0;
    for (; index + 8 <= plane; index += 8) {
      __m256 sum0 = _mm256_set1_ps(bias ? bias[output] : 0.F);
      __m256 sum1 = _mm256_set1_ps(bias ? bias[output + 1] : 0.F);
      __m256 sum2 = _mm256_set1_ps(bias ? bias[output + 2] : 0.F);
      __m256 sum3 = _mm256_set1_ps(bias ? bias[output + 3] : 0.F);
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
    for (; index < plane; ++index) {
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
    std::size_t index = 0;
    for (; index + 8 <= plane; index += 8) {
      __m256 sum = _mm256_set1_ps(bias ? bias[output] : 0.F);
      for (int input = 0; input < input_channels; ++input) {
        sum = _mm256_fmadd_ps(_mm256_set1_ps(filter[input]),
                            _mm256_loadu_ps(src + std::size_t(input) * plane + index), sum);
      }
      _mm256_storeu_ps(out + index, sum);
    }
    for (; index < plane; ++index) {
      float sum = bias ? bias[output] : 0.F;
      for (int input = 0; input < input_channels; ++input) sum += filter[input] * src[std::size_t(input) * plane + index];
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
                           int input_channels, int output_channels, int input_h, int input_w) noexcept {
  const int output_w = input_w * 2;
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(input_h * 2) * output_w;
  const __m256i duplicate_low = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
  const __m256i duplicate_high = _mm256_setr_epi32(4, 4, 5, 5, 6, 6, 7, 7);
  for (int output = first_output; output < last_output; ++output) {
    float* out = dst + std::size_t(output) * output_plane;
    std::fill_n(out, output_plane, bias ? bias[output] : 0.F);
    for (int input = 0; input < input_channels; ++input) {
      const float* values = src + std::size_t(input) * input_plane;
      const float* weight = weights + (std::size_t(input) * output_channels + output) * 4;
      const __m256 top = _mm256_setr_ps(weight[0], weight[1], weight[0], weight[1],
                                         weight[0], weight[1], weight[0], weight[1]);
      const __m256 bottom = _mm256_setr_ps(weight[2], weight[3], weight[2], weight[3],
                                            weight[2], weight[3], weight[2], weight[3]);
      for (int y = 0; y < input_h; ++y) {
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
  for (int row = first_row; row < last_row; ++row) {
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
      const int interior_begin = y >= first_y && y < last_y ? first_x : 0;
      const int interior_end = y >= first_y && y < last_y ? last_x : 0;
      for (; x < interior_begin; ++x) {
        float sum = base; const int ix0 = x - pad_left;
        for (int ky = 0; ky < kernel_h; ++ky) { const int iy = iy0 + ky; if (iy < 0 || iy >= input_h) continue;
          for (int kx = 0; kx < kernel_w; ++kx) { const int ix = ix0 + kx; if (ix >= 0 && ix < input_w) sum += in[std::size_t(iy) * input_w + ix] * filter[ky * kernel_w + kx]; }
        }
        out[std::size_t(y) * output_w + x] = sum;
      }
      for (; x + 8 <= interior_end; x += 8) {
        __m256 sum = _mm256_set1_ps(base);
        for (int ky = 0; ky < kernel_h; ++ky) {
          const float* row = in + std::size_t(iy0 + ky) * input_w + x - pad_left;
          const float* kernel = filter + ky * kernel_w;
          for (int kx = 0; kx < kernel_w; ++kx) sum = _mm256_fmadd_ps(_mm256_set1_ps(kernel[kx]), _mm256_loadu_ps(row + kx), sum);
        }
        _mm256_storeu_ps(out + std::size_t(y) * output_w + x, sum);
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

// AVX2 counterpart of the AVX-512 four-output 3x3 kernel.  It keeps x86
// machines without AVX-512 on the same input-reuse algorithm rather than
// falling back to four separate input traversals.
void Avx2Conv3x3Stride1x4(float* dst, const float* src, const float* weights,
                          const float* bias, int first_output, int last_output,
                          int input_channels, int input_h, int input_w,
                          int output_h, int output_w, int pad_top,
                          int pad_left, bool relu) noexcept {
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(output_h) * output_w;
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
    for (int y = 0; y < output_h; ++y) {
      const int iy0 = y - pad_top;
      const bool interior_y = y >= first_y && y < last_y;
      int x = 0;
      for (; x < output_w; ++x) {
        if (interior_y && x >= first_x && x + 8 <= last_x) break;
        scalar4(out0, out1, out2, out3, f0, f1, f2, f3, b0, b1, b2, b3, y, x);
      }
      for (; x + 8 <= last_x; x += 8) {
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
        const auto index = std::size_t(y) * output_w + x;
        if (relu) { const __m256 zero = _mm256_setzero_ps(); s0 = _mm256_max_ps(s0, zero); s1 = _mm256_max_ps(s1, zero); s2 = _mm256_max_ps(s2, zero); s3 = _mm256_max_ps(s3, zero); }
        _mm256_storeu_ps(out0 + index, s0); _mm256_storeu_ps(out1 + index, s1);
        _mm256_storeu_ps(out2 + index, s2); _mm256_storeu_ps(out3 + index, s3);
      }
      for (; x < output_w; ++x) scalar4(out0, out1, out2, out3, f0, f1, f2, f3,
                                          b0, b1, b2, b3, y, x);
    }
  }
  if (output < last_output) {
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
                           int pad_left, bool relu) noexcept {
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(output_h) * output_w;
  const int first_y = std::max(0, (pad_top + 1) / 2);
  const int first_x = std::max(0, (pad_left + 1) / 2);
  const int max_y = input_h + pad_top - 3;
  const int max_x = input_w + pad_left - 3;
  const int last_y = max_y < 0 ? 0 : std::min(output_h, max_y / 2 + 1);
  const int last_x = max_x < 0 ? 0 : std::min(output_w, max_x / 2 + 1);
  const auto load_stride2 = [](const float* row) {
    const __m128 a = _mm_loadu_ps(row);
    const __m128 b = _mm_loadu_ps(row + 4);
    const __m128 c = _mm_loadu_ps(row + 8);
    const __m128 d = _mm_loadu_ps(row + 12);
    return _mm256_set_m128(_mm_shuffle_ps(c, d, 0x88), _mm_shuffle_ps(a, b, 0x88));
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
    for (int y = 0; y < output_h; ++y) {
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
  if (output < last_output) {
    Avx2Conv3x3Stride2(dst, src, weights, bias, output, last_output, input_channels,
                       input_h, input_w, output_h, output_w, pad_top, pad_left, relu);
  }
}

void Avx2WriteIdentityRgbToNchw(float* dst, const std::uint8_t* rgb, int width,
                                int height, int source_width, int left, int top,
                                const float* scale, const float* shift,
                                int row_width) noexcept {
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
  for (int y = 0; y < height; ++y) {
    const auto* src = rgb + (std::size_t(top + y) * source_width + left) * 3;
    float* blue = dst + std::size_t(y) * row_width;
    float* green = blue + plane;
    float* red = green + plane;
    int x = 0;
    for (; x + 8 <= width; x += 8) {
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

}  // namespace ppocr::detail::kernels
