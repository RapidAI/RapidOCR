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

namespace {
inline __mmask16 Avx512CountMask(int n) noexcept {
  if (n >= 16) return 0xFFFF;
  if (n <= 0) return 0;
  return static_cast<__mmask16>((1u << n) - 1u);
}

inline __mmask16 Avx512BoundMask(int base_ix, int input_w, __mmask16 omask) noexcept {
  const __m512i ix = _mm512_add_epi32(
      _mm512_set1_epi32(base_ix),
      _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
  const __mmask16 ge0 =
      _mm512_mask_cmpge_epi32_mask(omask, ix, _mm512_setzero_si512());
  return _mm512_mask_cmplt_epi32_mask(ge0, ix, _mm512_set1_epi32(input_w));
}

inline bool Avx512MaskTailEnabled() noexcept {
  static const bool enabled =
      std::getenv("PPOCR_DISABLE_AVX512_MASK_TAIL") == nullptr;
  return enabled;
}
}  // namespace

void Avx512WidenU8ToFloat(float* dst, const std::uint8_t* src, std::size_t n) noexcept {
  std::size_t index = 0;
  for (; index + 16 <= n; index += 16) {
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + index));
    const __m512i expanded = _mm512_cvtepu8_epi32(bytes);
    _mm512_storeu_ps(dst + index, _mm512_cvtepi32_ps(expanded));
  }
  for (; index < n; ++index) dst[index] = static_cast<float>(src[index]);
}

int Avx512ArgMax(const float* values, int count) noexcept {
  if (!values || count <= 0) return -1;
  if (count < 16) {
    int best = 0;
    for (int i = 1; i < count; ++i) if (values[i] > values[best]) best = i;
    return best;
  }
  __m512 maximum = _mm512_loadu_ps(values);
  __m512i maximum_index = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                              8, 9, 10, 11, 12, 13, 14, 15);
  int index = 16;
  bool has_nan = _mm512_cmp_ps_mask(maximum, maximum, _CMP_UNORD_Q) != 0;
  for (; index + 16 <= count; index += 16) {
    const __m512 current = _mm512_loadu_ps(values + index);
    has_nan |= _mm512_cmp_ps_mask(current, current, _CMP_UNORD_Q) != 0;
    // Strict comparison deliberately preserves the earliest candidate within
    // each SIMD lane. The scalar reduction below resolves ties across lanes
    // by their original element index, matching `if (x > best)` exactly.
    const __mmask16 replace = _mm512_cmp_ps_mask(current, maximum, _CMP_GT_OQ);
    maximum = _mm512_mask_mov_ps(maximum, replace, current);
    maximum_index = _mm512_mask_mov_epi32(
        maximum_index, replace, _mm512_add_epi32(
            _mm512_set1_epi32(index),
            _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                               8, 9, 10, 11, 12, 13, 14, 15)));
  }
  // NaNs follow the portable `>` comparison semantics; uncommon malformed
  // logits take the scalar path rather than silently changing the winner.
  if (has_nan) {
    int best = 0;
    for (int i = 1; i < count; ++i) if (values[i] > values[best]) best = i;
    return best;
  }
  alignas(64) float lanes[16];
  alignas(64) int lane_indices[16];
  _mm512_store_ps(lanes, maximum);
  _mm512_store_si512(lane_indices, maximum_index);
  float maximum_scalar = lanes[0];
  int best = lane_indices[0];
  for (int lane = 1; lane < 16; ++lane) {
    if (lanes[lane] > maximum_scalar ||
        (lanes[lane] == maximum_scalar && lane_indices[lane] < best)) {
      maximum_scalar = lanes[lane];
      best = lane_indices[lane];
    }
  }
  // A tail follows every complete vector block. Equal values occur later and
  // must not displace the first maximum, exactly like the scalar reference.
  for (; index < count; ++index) {
    if (values[index] > maximum_scalar) {
      maximum_scalar = values[index];
      best = index;
    }
  }
  return best;
}

void Avx512Binary(float* dst, const float* a, const float* b, std::size_t n,
                  BinaryOp op) noexcept {
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m512 x = _mm512_loadu_ps(a + i), y = _mm512_loadu_ps(b + i);
    __m512 z = _mm512_setzero_ps();
    switch (op) {
      case BinaryOp::add: z = _mm512_add_ps(x, y); break;
      case BinaryOp::sub: z = _mm512_sub_ps(x, y); break;
      case BinaryOp::mul: z = _mm512_mul_ps(x, y); break;
      case BinaryOp::div: z = _mm512_div_ps(x, y); break;
    }
    _mm512_storeu_ps(dst + i, z);
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

void Avx512NearestResize2xAdd(float* dst, const float* src, const float* residual,
                              int batches, int channels, int input_height,
                              int input_width) noexcept {
  const int output_width = input_width * 2;
  const std::size_t input_plane = std::size_t(input_height) * input_width;
  const std::size_t output_plane = input_plane * 4;
  const __m512i even_indices = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3,
                                                   4, 4, 5, 5, 6, 6, 7, 7);
  const __m512i even_indices_hi = _mm512_setr_epi32(8, 8, 9, 9, 10, 10, 11, 11,
                                                      12, 12, 13, 13, 14, 14, 15, 15);
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
      for (; x + 16 <= input_width; x += 16) {
        const __m512 values = _mm512_loadu_ps(input_row + x);
        const __m512 values0 = _mm512_permutexvar_ps(even_indices, values);
        const __m512 values1 = _mm512_permutexvar_ps(even_indices_hi, values);
        const int offset = x * 2;
        _mm512_storeu_ps(output0 + offset, _mm512_add_ps(values0, _mm512_loadu_ps(add0 + offset)));
        _mm512_storeu_ps(output0 + offset + 16, _mm512_add_ps(values1, _mm512_loadu_ps(add0 + offset + 16)));
        _mm512_storeu_ps(output1 + offset, _mm512_add_ps(values0, _mm512_loadu_ps(add1 + offset)));
        _mm512_storeu_ps(output1 + offset + 16, _mm512_add_ps(values1, _mm512_loadu_ps(add1 + offset + 16)));
      }
      for (; x < input_width; ++x) {
        const float value = input_row[x]; const int offset = x * 2;
        output0[offset] = value + add0[offset]; output0[offset + 1] = value + add0[offset + 1];
        output1[offset] = value + add1[offset]; output1[offset + 1] = value + add1[offset + 1];
      }
    }
  }
}

void Avx512Relu(float* dst, const float* src, std::size_t n) noexcept {
  const __m512 zero = _mm512_setzero_ps();
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    _mm512_storeu_ps(dst + i, _mm512_max_ps(_mm512_loadu_ps(src + i), zero));
  }
  for (; i < n; ++i) dst[i] = std::max(src[i], 0.F);
}

// Vector form of the standard tanh GELU approximation. The PP-OCRv6 decoder
// uses GELU only in its recognizer blocks; this removes the scalar libm `erf`
// call from their hot path while preserving CTC decisions on the validation
// corpus.  The exported graph still denotes exact GELU; this is deliberately
// an opt-in implementation detail of the model-specific fused operator.
void Avx512Gelu(float* dst, const float* src, std::size_t n) noexcept {
  const __m512 half = _mm512_set1_ps(.5F);
  const __m512 one = _mm512_set1_ps(1.F);
  const __m512 c0 = _mm512_set1_ps(0.7978845608028654F);
  const __m512 c1 = _mm512_set1_ps(0.044715F);
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m512 x = _mm512_loadu_ps(src + i);
    const __m512 x3 = _mm512_mul_ps(x, _mm512_mul_ps(x, x));
    const __m512 z = _mm512_mul_ps(c0, _mm512_fmadd_ps(c1, x3, x));
    const __m512 z2 = _mm512_mul_ps(z, z);
    // Fast bounded Pad茅 approximation of tanh(z), sufficient for the
    // PP-OCRv6 GELU activation range and expressible with AVX-512F alone.
    const __m512 t = _mm512_max_ps(_mm512_set1_ps(-1.F), _mm512_min_ps(one,
        _mm512_div_ps(_mm512_mul_ps(z, _mm512_add_ps(_mm512_set1_ps(27.F), z2)),
                      _mm512_add_ps(_mm512_set1_ps(27.F), _mm512_mul_ps(_mm512_set1_ps(9.F), z2)))));
    _mm512_storeu_ps(dst + i, _mm512_mul_ps(half, _mm512_mul_ps(x, _mm512_add_ps(one, t))));
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

__m512 ExpPs512(__m512 x) noexcept {
  const __m512 min_x = _mm512_set1_ps(-87.F);
  const __m512 max_x = _mm512_set1_ps(87.F);
  const __m512 inv_ln2 = _mm512_set1_ps(1.4426950408889634F);
  const __m512 ln2 = _mm512_set1_ps(0.6931471805599453F);
  x = _mm512_min_ps(_mm512_max_ps(x, min_x), max_x);
  const __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, inv_ln2),
                                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  const __m512 f = _mm512_fnmadd_ps(n, ln2, x);
  __m512 p = _mm512_fmadd_ps(f, _mm512_set1_ps(1.F / 24.F), _mm512_set1_ps(1.F / 6.F));
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(.5F));
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.F));
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.F));
  const __m512i two_n = _mm512_slli_epi32(
      _mm512_add_epi32(_mm512_cvtps_epi32(n), _mm512_set1_epi32(127)), 23);
  return _mm512_mul_ps(p, _mm512_castsi512_ps(two_n));
}

__m512 ErfPs512(__m512 x) noexcept {
  const __m512 one = _mm512_set1_ps(1.F);
  const __m512 half = _mm512_set1_ps(.5F);
  const __m512 sign = _mm512_or_ps(_mm512_and_ps(x, _mm512_set1_ps(-0.F)), one);
  const __m512 ax = _mm512_andnot_ps(_mm512_set1_ps(-0.F), x);
  const __m512 t = _mm512_div_ps(one, _mm512_fmadd_ps(half, ax, one));
  __m512 poly = _mm512_fmadd_ps(t, _mm512_set1_ps(.17087277F), _mm512_set1_ps(-.82215223F));
  poly = _mm512_fmadd_ps(poly, t, _mm512_set1_ps(1.48851587F));
  poly = _mm512_fmadd_ps(poly, t, _mm512_set1_ps(-1.13520398F));
  poly = _mm512_fmadd_ps(poly, t, _mm512_set1_ps(.27886807F));
  poly = _mm512_fmadd_ps(poly, t, _mm512_set1_ps(-.18628806F));
  poly = _mm512_fmadd_ps(poly, t, _mm512_set1_ps(.09678418F));
  poly = _mm512_fmadd_ps(poly, t, _mm512_set1_ps(.37409196F));
  poly = _mm512_fmadd_ps(poly, t, _mm512_set1_ps(1.00002368F));
  poly = _mm512_fmadd_ps(poly, t, _mm512_set1_ps(-1.26551223F));
  const __m512 e = ExpPs512(_mm512_fmadd_ps(_mm512_mul_ps(ax, ax),
                                            _mm512_set1_ps(-1.F), poly));
  const __m512 magnitude = _mm512_sub_ps(one, _mm512_mul_ps(t, e));
  return _mm512_mul_ps(sign, magnitude);
}

}  // namespace

void Avx512Sigmoid(float* dst, const float* src, std::size_t n) noexcept {
  const __m512 one = _mm512_set1_ps(1.F);
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m512 e = ExpPs512(_mm512_sub_ps(_mm512_setzero_ps(), _mm512_loadu_ps(src + i)));
    _mm512_storeu_ps(dst + i, _mm512_div_ps(one, _mm512_add_ps(one, e)));
  }
  for (; i < n; ++i) dst[i] = 1.F / (1.F + std::exp(-src[i]));
}

float Avx512SoftmaxDenom(const float* logits, int vocab, float maximum) noexcept {
  if (!logits || vocab <= 0) return 0.F;
  const __m512 vmax = _mm512_set1_ps(maximum);
  __m512 vsum = _mm512_setzero_ps();
  int column = 0;
  for (; column + 16 <= vocab; column += 16) {
    const __m512 x = _mm512_sub_ps(_mm512_loadu_ps(logits + column), vmax);
    vsum = _mm512_add_ps(vsum, ExpPs512(x));
  }
  float sum = _mm512_reduce_add_ps(vsum);
  for (; column < vocab; ++column)
    sum += std::exp(logits[column] - maximum);
  return sum;
}

void Avx512ExactGelu(float* dst, const float* src, std::size_t n) noexcept {
  const __m512 half = _mm512_set1_ps(.5F);
  const __m512 one = _mm512_set1_ps(1.F);
  const __m512 inv_sqrt2 = _mm512_set1_ps(0.7071067811865475244F);
  std::size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    const __m512 x0 = _mm512_loadu_ps(src + i);
    const __m512 x1 = _mm512_loadu_ps(src + i + 16);
    const __m512 y0 = _mm512_mul_ps(half, _mm512_mul_ps(x0,
        _mm512_add_ps(one, ErfPs512(_mm512_mul_ps(x0, inv_sqrt2)))));
    const __m512 y1 = _mm512_mul_ps(half, _mm512_mul_ps(x1,
        _mm512_add_ps(one, ErfPs512(_mm512_mul_ps(x1, inv_sqrt2)))));
    _mm512_storeu_ps(dst + i, y0);
    _mm512_storeu_ps(dst + i + 16, y1);
  }
  for (; i + 16 <= n; i += 16) {
    const __m512 x = _mm512_loadu_ps(src + i);
    const __m512 gelu = _mm512_mul_ps(half, _mm512_mul_ps(x,
        _mm512_add_ps(one, ErfPs512(_mm512_mul_ps(x, inv_sqrt2)))));
    _mm512_storeu_ps(dst + i, gelu);
  }
  constexpr float inv_sqrt2s = 0.7071067811865475244F;
  for (; i < n; ++i) {
    const float x = src[i];
    dst[i] = x * .5F * (1.F + std::erf(x * inv_sqrt2s));
  }
}

void Avx512BatchNormGelu(float* dst, const float* src, std::size_t n,
                         float scale, float shift) noexcept {
  const __m512 affine_scale = _mm512_set1_ps(scale);
  const __m512 affine_shift = _mm512_set1_ps(shift);
  const __m512 half = _mm512_set1_ps(.5F);
  const __m512 one = _mm512_set1_ps(1.F);
  const __m512 c0 = _mm512_set1_ps(.7978845608028654F);
  const __m512 c1 = _mm512_set1_ps(.044715F);
  const __m512 p27 = _mm512_set1_ps(27.F);
  const __m512 p9 = _mm512_set1_ps(9.F);
  const __m512 negative_one = _mm512_set1_ps(-1.F);
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m512 x = _mm512_fmadd_ps(_mm512_loadu_ps(src + i), affine_scale,
                                     affine_shift);
    const __m512 x3 = _mm512_mul_ps(x, _mm512_mul_ps(x, x));
    const __m512 z = _mm512_mul_ps(c0, _mm512_fmadd_ps(c1, x3, x));
    const __m512 z2 = _mm512_mul_ps(z, z);
    const __m512 t = _mm512_max_ps(negative_one, _mm512_min_ps(one,
        _mm512_div_ps(_mm512_mul_ps(z, _mm512_add_ps(p27, z2)),
                      _mm512_add_ps(p27, _mm512_mul_ps(p9, z2)))));
    _mm512_storeu_ps(dst + i, _mm512_mul_ps(half, _mm512_mul_ps(x,
        _mm512_add_ps(one, t))));
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

void Avx512HardSwish(float* dst, const float* src, std::size_t n) noexcept {
  const __m512 sixth = _mm512_set1_ps(1.F / 6.F), half = _mm512_set1_ps(.5F);
  const __m512 zero = _mm512_setzero_ps(), one = _mm512_set1_ps(1.F);
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m512 x = _mm512_loadu_ps(src + i);
    const __m512 gate = _mm512_min_ps(one, _mm512_max_ps(zero, _mm512_fmadd_ps(x, sixth, half)));
    _mm512_storeu_ps(dst + i, _mm512_mul_ps(x, gate));
  }
  for (; i < n; ++i) { const float x = src[i]; dst[i] = x * std::clamp(x / 6.F + .5F, 0.F, 1.F); }
}

void Avx512ScaleShift(float* dst, const float* src, std::size_t n, float scale,
                      float shift) noexcept {
  const __m512 s = _mm512_set1_ps(scale), b = _mm512_set1_ps(shift);
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) _mm512_storeu_ps(dst + i, _mm512_fmadd_ps(_mm512_loadu_ps(src+i), s, b));
  for (; i < n; ++i) dst[i] = src[i] * scale + shift;
}

void Avx512BinaryScalar(float* dst, const float* src, std::size_t n, float scalar,
                        BinaryOp op, bool scalar_left) noexcept {
  const __m512 s = _mm512_set1_ps(scalar);
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m512 x = _mm512_loadu_ps(src + i);
    __m512 z = x;
    switch (op) {
      case BinaryOp::add: z = _mm512_add_ps(x, s); break;
      case BinaryOp::sub: z = scalar_left ? _mm512_sub_ps(s, x) : _mm512_sub_ps(x, s); break;
      case BinaryOp::mul: z = _mm512_mul_ps(x, s); break;
      case BinaryOp::div: z = scalar_left ? _mm512_div_ps(s, x) : _mm512_div_ps(x, s); break;
    }
    _mm512_storeu_ps(dst + i, z);
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

void Avx512Axpy(float* dst, const float* src, float alpha, std::size_t n) noexcept {
  const __m512 a = _mm512_set1_ps(alpha);
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    _mm512_storeu_ps(dst + i, _mm512_fmadd_ps(a, _mm512_loadu_ps(src + i),
                                               _mm512_loadu_ps(dst + i)));
  }
  for (; i < n; ++i) dst[i] += alpha * src[i];
}

// Register-tiled 1x1 convolution. Each output vector remains live for the
// complete input-channel reduction, avoiding the former load/FMA/store of
// every output element for every input channel. Output-channel and reduction
// order are unchanged, so it remains an exact execution-path optimization.
void Avx512PointwiseConv4(float* dst, const float* src, const float* weights,
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
    for (; index + 16 <= plane; index += 16) {
      __m512 sum0 = _mm512_set1_ps(bias ? bias[output] : 0.F);
      __m512 sum1 = _mm512_set1_ps(bias ? bias[output + 1] : 0.F);
      __m512 sum2 = _mm512_set1_ps(bias ? bias[output + 2] : 0.F);
      __m512 sum3 = _mm512_set1_ps(bias ? bias[output + 3] : 0.F);
      for (int input = 0; input < input_channels; ++input) {
        const __m512 x = _mm512_loadu_ps(src + std::size_t(input) * plane + index);
        sum0 = _mm512_fmadd_ps(_mm512_set1_ps(w0[input]), x, sum0);
        sum1 = _mm512_fmadd_ps(_mm512_set1_ps(w1[input]), x, sum1);
        sum2 = _mm512_fmadd_ps(_mm512_set1_ps(w2[input]), x, sum2);
        sum3 = _mm512_fmadd_ps(_mm512_set1_ps(w3[input]), x, sum3);
      }
      _mm512_storeu_ps(out0 + index, sum0);
      _mm512_storeu_ps(out1 + index, sum1);
      _mm512_storeu_ps(out2 + index, sum2);
      _mm512_storeu_ps(out3 + index, sum3);
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
    for (; index + 16 <= plane; index += 16) {
      __m512 sum = _mm512_set1_ps(bias ? bias[output] : 0.F);
      for (int input = 0; input < input_channels; ++input) {
        sum = _mm512_fmadd_ps(_mm512_set1_ps(filter[input]),
                            _mm512_loadu_ps(src + std::size_t(input) * plane + index), sum);
      }
      _mm512_storeu_ps(out + index, sum);
    }
    for (; index < plane; ++index) {
      float sum = bias ? bias[output] : 0.F;
      for (int input = 0; input < input_channels; ++input) sum += filter[input] * src[std::size_t(input) * plane + index];
      out[index] = sum;
    }
  }
}

void Avx512Square(float* dst, const float* src, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m512 value = _mm512_loadu_ps(src + i);
    _mm512_storeu_ps(dst + i, _mm512_mul_ps(value, value));
  }
  for (; i < n; ++i) dst[i] = src[i] * src[i];
}

void Avx512HardSigmoid(float* dst, const float* src, std::size_t n, float alpha,
                        float beta) noexcept {
  const __m512 a = _mm512_set1_ps(alpha), b = _mm512_set1_ps(beta);
  const __m512 zero = _mm512_setzero_ps(), one = _mm512_set1_ps(1.F);
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m512 value = _mm512_fmadd_ps(_mm512_loadu_ps(src + i), a, b);
    _mm512_storeu_ps(dst + i, _mm512_min_ps(_mm512_max_ps(value, zero), one));
  }
  for (; i < n; ++i) dst[i] = std::clamp(alpha * src[i] + beta, 0.F, 1.F);
}

// Wider 1x1 register tile for the large medium recognizer/detector
// projections.  Eight independent output accumulators still fit in AVX-512's
// architectural vector register file, while each loaded input vector now
// feeds eight channels rather than four.  Every output retains the identical
// input-channel reduction order of the four-output and scalar paths.
void Avx512PointwiseConv8(float* dst, const float* src, const float* weights,
                           const float* bias, int first_output, int last_output,
                           int input_channels, std::size_t plane) noexcept {
  int output = first_output;
  for (; output + 8 <= last_output; output += 8) {
    float* o0 = dst + std::size_t(output) * plane;
    float* o1 = o0 + plane; float* o2 = o1 + plane; float* o3 = o2 + plane;
    float* o4 = o3 + plane; float* o5 = o4 + plane; float* o6 = o5 + plane;
    float* o7 = o6 + plane;
    const float* w0 = weights + std::size_t(output) * input_channels;
    const float* w1 = w0 + input_channels; const float* w2 = w1 + input_channels;
    const float* w3 = w2 + input_channels; const float* w4 = w3 + input_channels;
    const float* w5 = w4 + input_channels; const float* w6 = w5 + input_channels;
    const float* w7 = w6 + input_channels;
    std::size_t index = 0;
    for (; index + 16 <= plane; index += 16) {
      __m512 s0 = _mm512_set1_ps(bias ? bias[output] : 0.F);
      __m512 s1 = _mm512_set1_ps(bias ? bias[output + 1] : 0.F);
      __m512 s2 = _mm512_set1_ps(bias ? bias[output + 2] : 0.F);
      __m512 s3 = _mm512_set1_ps(bias ? bias[output + 3] : 0.F);
      __m512 s4 = _mm512_set1_ps(bias ? bias[output + 4] : 0.F);
      __m512 s5 = _mm512_set1_ps(bias ? bias[output + 5] : 0.F);
      __m512 s6 = _mm512_set1_ps(bias ? bias[output + 6] : 0.F);
      __m512 s7 = _mm512_set1_ps(bias ? bias[output + 7] : 0.F);
      for (int input = 0; input < input_channels; ++input) {
        const __m512 x = _mm512_loadu_ps(src + std::size_t(input) * plane + index);
        s0 = _mm512_fmadd_ps(_mm512_set1_ps(w0[input]), x, s0);
        s1 = _mm512_fmadd_ps(_mm512_set1_ps(w1[input]), x, s1);
        s2 = _mm512_fmadd_ps(_mm512_set1_ps(w2[input]), x, s2);
        s3 = _mm512_fmadd_ps(_mm512_set1_ps(w3[input]), x, s3);
        s4 = _mm512_fmadd_ps(_mm512_set1_ps(w4[input]), x, s4);
        s5 = _mm512_fmadd_ps(_mm512_set1_ps(w5[input]), x, s5);
        s6 = _mm512_fmadd_ps(_mm512_set1_ps(w6[input]), x, s6);
        s7 = _mm512_fmadd_ps(_mm512_set1_ps(w7[input]), x, s7);
      }
      _mm512_storeu_ps(o0 + index, s0); _mm512_storeu_ps(o1 + index, s1);
      _mm512_storeu_ps(o2 + index, s2); _mm512_storeu_ps(o3 + index, s3);
      _mm512_storeu_ps(o4 + index, s4); _mm512_storeu_ps(o5 + index, s5);
      _mm512_storeu_ps(o6 + index, s6); _mm512_storeu_ps(o7 + index, s7);
    }
    for (; index < plane; ++index) {
      const float x_bias0 = bias ? bias[output] : 0.F;
      const float x_bias1 = bias ? bias[output + 1] : 0.F;
      const float x_bias2 = bias ? bias[output + 2] : 0.F;
      const float x_bias3 = bias ? bias[output + 3] : 0.F;
      const float x_bias4 = bias ? bias[output + 4] : 0.F;
      const float x_bias5 = bias ? bias[output + 5] : 0.F;
      const float x_bias6 = bias ? bias[output + 6] : 0.F;
      const float x_bias7 = bias ? bias[output + 7] : 0.F;
      float s0=x_bias0,s1=x_bias1,s2=x_bias2,s3=x_bias3,s4=x_bias4,s5=x_bias5,s6=x_bias6,s7=x_bias7;
      for (int input = 0; input < input_channels; ++input) {
        const float x = src[std::size_t(input) * plane + index];
        s0 += w0[input] * x; s1 += w1[input] * x; s2 += w2[input] * x; s3 += w3[input] * x;
        s4 += w4[input] * x; s5 += w5[input] * x; s6 += w6[input] * x; s7 += w7[input] * x;
      }
      o0[index]=s0; o1[index]=s1; o2[index]=s2; o3[index]=s3;
      o4[index]=s4; o5[index]=s5; o6[index]=s6; o7[index]=s7;
    }
  }
  if (output < last_output) {
    Avx512PointwiseConv4(dst, src, weights, bias, output, last_output,
                          input_channels, plane);
  }
}

void Avx512PointwiseConv16Range(float* dst, const float* src, const float* weights,
                                 const float* bias, int first_output, int last_output,
                                 int input_channels, std::size_t plane,
                                 std::size_t first_index, std::size_t last_index) noexcept {
  if (last_index > plane) last_index = plane;
  if (first_index >= last_index) return;
  int output = first_output;
  for (; output + 16 <= last_output; output += 16) {
    float* o[16];
    const float* w[16];
    o[0] = dst + std::size_t(output) * plane;
    w[0] = weights + std::size_t(output) * input_channels;
    for (int k = 1; k < 16; ++k) {
      o[k] = o[k - 1] + plane;
      w[k] = w[k - 1] + input_channels;
    }
    std::size_t index = first_index;
    // Named 16-wide accumulators spilled/regressed Conv.63 on this host
    // (8-run 16.60/16.72 vs array 16.17). Keep ENABLE-only.
    // `PPOCR_ENABLE_AVX512_PW16_NAMED` turns named accumulators on.
    static const bool named =
        std::getenv("PPOCR_ENABLE_AVX512_PW16_NAMED") != nullptr &&
        std::getenv("PPOCR_DISABLE_AVX512_PW16_NAMED") == nullptr;
    static const bool tap_pack =
        std::getenv("PPOCR_DISABLE_AVX512_PW16_TAP_PACK") == nullptr;
    thread_local std::vector<float> packed_oc16;
    const float* tw = nullptr;
    if (tap_pack && !named) {
      packed_oc16.resize(std::size_t(input_channels) * 16);
      for (int input = 0; input < input_channels; ++input)
        for (int k = 0; k < 16; ++k)
          packed_oc16[std::size_t(input) * 16 + k] = w[k][input];
      tw = packed_oc16.data();
    }
    for (; index + 16 <= last_index; index += 16) {
      if (named) {
        __m512 s0 = _mm512_set1_ps(bias ? bias[output] : 0.F);
        __m512 s1 = _mm512_set1_ps(bias ? bias[output + 1] : 0.F);
        __m512 s2 = _mm512_set1_ps(bias ? bias[output + 2] : 0.F);
        __m512 s3 = _mm512_set1_ps(bias ? bias[output + 3] : 0.F);
        __m512 s4 = _mm512_set1_ps(bias ? bias[output + 4] : 0.F);
        __m512 s5 = _mm512_set1_ps(bias ? bias[output + 5] : 0.F);
        __m512 s6 = _mm512_set1_ps(bias ? bias[output + 6] : 0.F);
        __m512 s7 = _mm512_set1_ps(bias ? bias[output + 7] : 0.F);
        __m512 s8 = _mm512_set1_ps(bias ? bias[output + 8] : 0.F);
        __m512 s9 = _mm512_set1_ps(bias ? bias[output + 9] : 0.F);
        __m512 sa = _mm512_set1_ps(bias ? bias[output + 10] : 0.F);
        __m512 sb = _mm512_set1_ps(bias ? bias[output + 11] : 0.F);
        __m512 sc = _mm512_set1_ps(bias ? bias[output + 12] : 0.F);
        __m512 sd = _mm512_set1_ps(bias ? bias[output + 13] : 0.F);
        __m512 se = _mm512_set1_ps(bias ? bias[output + 14] : 0.F);
        __m512 sf = _mm512_set1_ps(bias ? bias[output + 15] : 0.F);
        for (int input = 0; input < input_channels; ++input) {
          const __m512 x = _mm512_loadu_ps(src + std::size_t(input) * plane + index);
          s0 = _mm512_fmadd_ps(_mm512_set1_ps(w[0][input]), x, s0);
          s1 = _mm512_fmadd_ps(_mm512_set1_ps(w[1][input]), x, s1);
          s2 = _mm512_fmadd_ps(_mm512_set1_ps(w[2][input]), x, s2);
          s3 = _mm512_fmadd_ps(_mm512_set1_ps(w[3][input]), x, s3);
          s4 = _mm512_fmadd_ps(_mm512_set1_ps(w[4][input]), x, s4);
          s5 = _mm512_fmadd_ps(_mm512_set1_ps(w[5][input]), x, s5);
          s6 = _mm512_fmadd_ps(_mm512_set1_ps(w[6][input]), x, s6);
          s7 = _mm512_fmadd_ps(_mm512_set1_ps(w[7][input]), x, s7);
          s8 = _mm512_fmadd_ps(_mm512_set1_ps(w[8][input]), x, s8);
          s9 = _mm512_fmadd_ps(_mm512_set1_ps(w[9][input]), x, s9);
          sa = _mm512_fmadd_ps(_mm512_set1_ps(w[10][input]), x, sa);
          sb = _mm512_fmadd_ps(_mm512_set1_ps(w[11][input]), x, sb);
          sc = _mm512_fmadd_ps(_mm512_set1_ps(w[12][input]), x, sc);
          sd = _mm512_fmadd_ps(_mm512_set1_ps(w[13][input]), x, sd);
          se = _mm512_fmadd_ps(_mm512_set1_ps(w[14][input]), x, se);
          sf = _mm512_fmadd_ps(_mm512_set1_ps(w[15][input]), x, sf);
        }
        _mm512_storeu_ps(o[0] + index, s0); _mm512_storeu_ps(o[1] + index, s1);
        _mm512_storeu_ps(o[2] + index, s2); _mm512_storeu_ps(o[3] + index, s3);
        _mm512_storeu_ps(o[4] + index, s4); _mm512_storeu_ps(o[5] + index, s5);
        _mm512_storeu_ps(o[6] + index, s6); _mm512_storeu_ps(o[7] + index, s7);
        _mm512_storeu_ps(o[8] + index, s8); _mm512_storeu_ps(o[9] + index, s9);
        _mm512_storeu_ps(o[10] + index, sa); _mm512_storeu_ps(o[11] + index, sb);
        _mm512_storeu_ps(o[12] + index, sc); _mm512_storeu_ps(o[13] + index, sd);
        _mm512_storeu_ps(o[14] + index, se); _mm512_storeu_ps(o[15] + index, sf);
        continue;
      }
      __m512 acc[16];
      for (int k = 0; k < 16; ++k)
        acc[k] = _mm512_set1_ps(bias ? bias[output + k] : 0.F);
      static const bool ic2 =
          std::getenv("PPOCR_ENABLE_AVX512_PW16_IC2") != nullptr &&
          std::getenv("PPOCR_DISABLE_AVX512_PW16_IC2") == nullptr;
      int input = 0;
      if (ic2) {
        for (; input + 2 <= input_channels; input += 2) {
          const __m512 x0 = _mm512_loadu_ps(src + std::size_t(input) * plane + index);
          const __m512 x1 = _mm512_loadu_ps(src + std::size_t(input + 1) * plane + index);
          for (int k = 0; k < 16; ++k) {
            acc[k] = _mm512_fmadd_ps(_mm512_set1_ps(w[k][input]), x0, acc[k]);
            acc[k] = _mm512_fmadd_ps(_mm512_set1_ps(w[k][input + 1]), x1, acc[k]);
          }
        }
      }
      for (; input < input_channels; ++input) {
        const __m512 x = _mm512_loadu_ps(src + std::size_t(input) * plane + index);
        const float* t = tw ? tw + std::size_t(input) * 16 : nullptr;
        for (int k = 0; k < 16; ++k)
          acc[k] = _mm512_fmadd_ps(_mm512_set1_ps(t ? t[k] : w[k][input]), x, acc[k]);
      }
      for (int k = 0; k < 16; ++k) _mm512_storeu_ps(o[k] + index, acc[k]);
    }
    for (; index < last_index; ++index) {
      float acc[16];
      for (int k = 0; k < 16; ++k) acc[k] = bias ? bias[output + k] : 0.F;
      for (int input = 0; input < input_channels; ++input) {
        const float x = src[std::size_t(input) * plane + index];
        for (int k = 0; k < 16; ++k) acc[k] += w[k][input] * x;
      }
      for (int k = 0; k < 16; ++k) o[k][index] = acc[k];
    }
  }
  if (output < last_output) {
    Avx512PointwiseConv8(dst, src, weights, bias, output, last_output,
                          input_channels, plane);
  }
}

void Avx512PointwiseConv16(float* dst, const float* src, const float* weights,
                            const float* bias, int first_output, int last_output,
                            int input_channels, std::size_t plane) noexcept {
  Avx512PointwiseConv16Range(dst, src, weights, bias, first_output, last_output,
                             input_channels, plane, 0, plane);
}

// Conv -> residual Add fusion.  The residual is loaded only after each
// channel reduction is complete, preserving the unfused Conv then Add order
// while avoiding Conv's intermediate activation write/read pair.
void Avx512PointwiseConvAdd4(float* dst, const float* src, const float* weights,
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
    for (; index + 16 <= plane; index += 16) {
      __m512 s0 = _mm512_set1_ps(bias ? bias[output] : 0.F);
      __m512 s1 = _mm512_set1_ps(bias ? bias[output + 1] : 0.F);
      __m512 s2 = _mm512_set1_ps(bias ? bias[output + 2] : 0.F);
      __m512 s3 = _mm512_set1_ps(bias ? bias[output + 3] : 0.F);
      for (int input = 0; input < input_channels; ++input) {
        const __m512 x = _mm512_loadu_ps(src + std::size_t(input) * plane + index);
        s0 = _mm512_fmadd_ps(_mm512_set1_ps(w0[input]), x, s0);
        s1 = _mm512_fmadd_ps(_mm512_set1_ps(w1[input]), x, s1);
        s2 = _mm512_fmadd_ps(_mm512_set1_ps(w2[input]), x, s2);
        s3 = _mm512_fmadd_ps(_mm512_set1_ps(w3[input]), x, s3);
      }
      _mm512_storeu_ps(o0 + index, _mm512_add_ps(s0, _mm512_loadu_ps(r0 + index)));
      _mm512_storeu_ps(o1 + index, _mm512_add_ps(s1, _mm512_loadu_ps(r1 + index)));
      _mm512_storeu_ps(o2 + index, _mm512_add_ps(s2, _mm512_loadu_ps(r2 + index)));
      _mm512_storeu_ps(o3 + index, _mm512_add_ps(s3, _mm512_loadu_ps(r3 + index)));
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
    for (; index + 16 <= plane; index += 16) {
      __m512 sum = _mm512_set1_ps(bias ? bias[output] : 0.F);
      for (int input = 0; input < input_channels; ++input) {
        sum = _mm512_fmadd_ps(_mm512_set1_ps(filter[input]),
                               _mm512_loadu_ps(src + std::size_t(input) * plane + index), sum);
      }
      _mm512_storeu_ps(out + index, _mm512_add_ps(sum, _mm512_loadu_ps(add + index)));
    }
    for (; index < plane; ++index) {
      float sum = bias ? bias[output] : 0.F;
      for (int input = 0; input < input_channels; ++input)
        sum += filter[input] * src[std::size_t(input) * plane + index];
      out[index] = sum + add[index];
    }
  }
}

void Avx512PointwiseConvAdd8(float* dst, const float* src, const float* weights,
                             const float* bias, const float* residual,
                             int first_output, int last_output,
                             int input_channels, std::size_t plane) noexcept {
  int output = first_output;
  for (; output + 8 <= last_output; output += 8) {
    float* o0 = dst + std::size_t(output) * plane;
    float* o1 = o0 + plane; float* o2 = o1 + plane; float* o3 = o2 + plane;
    float* o4 = o3 + plane; float* o5 = o4 + plane; float* o6 = o5 + plane;
    float* o7 = o6 + plane;
    const float* r0 = residual + std::size_t(output) * plane;
    const float* r1 = r0 + plane; const float* r2 = r1 + plane; const float* r3 = r2 + plane;
    const float* r4 = r3 + plane; const float* r5 = r4 + plane; const float* r6 = r5 + plane;
    const float* r7 = r6 + plane;
    const float* w0 = weights + std::size_t(output) * input_channels;
    const float* w1 = w0 + input_channels; const float* w2 = w1 + input_channels;
    const float* w3 = w2 + input_channels; const float* w4 = w3 + input_channels;
    const float* w5 = w4 + input_channels; const float* w6 = w5 + input_channels;
    const float* w7 = w6 + input_channels;
    std::size_t index = 0;
    for (; index + 16 <= plane; index += 16) {
      __m512 s0 = _mm512_set1_ps(bias ? bias[output] : 0.F);
      __m512 s1 = _mm512_set1_ps(bias ? bias[output + 1] : 0.F);
      __m512 s2 = _mm512_set1_ps(bias ? bias[output + 2] : 0.F);
      __m512 s3 = _mm512_set1_ps(bias ? bias[output + 3] : 0.F);
      __m512 s4 = _mm512_set1_ps(bias ? bias[output + 4] : 0.F);
      __m512 s5 = _mm512_set1_ps(bias ? bias[output + 5] : 0.F);
      __m512 s6 = _mm512_set1_ps(bias ? bias[output + 6] : 0.F);
      __m512 s7 = _mm512_set1_ps(bias ? bias[output + 7] : 0.F);
      for (int input = 0; input < input_channels; ++input) {
        const __m512 x = _mm512_loadu_ps(src + std::size_t(input) * plane + index);
        s0 = _mm512_fmadd_ps(_mm512_set1_ps(w0[input]), x, s0);
        s1 = _mm512_fmadd_ps(_mm512_set1_ps(w1[input]), x, s1);
        s2 = _mm512_fmadd_ps(_mm512_set1_ps(w2[input]), x, s2);
        s3 = _mm512_fmadd_ps(_mm512_set1_ps(w3[input]), x, s3);
        s4 = _mm512_fmadd_ps(_mm512_set1_ps(w4[input]), x, s4);
        s5 = _mm512_fmadd_ps(_mm512_set1_ps(w5[input]), x, s5);
        s6 = _mm512_fmadd_ps(_mm512_set1_ps(w6[input]), x, s6);
        s7 = _mm512_fmadd_ps(_mm512_set1_ps(w7[input]), x, s7);
      }
      _mm512_storeu_ps(o0 + index, _mm512_add_ps(s0, _mm512_loadu_ps(r0 + index)));
      _mm512_storeu_ps(o1 + index, _mm512_add_ps(s1, _mm512_loadu_ps(r1 + index)));
      _mm512_storeu_ps(o2 + index, _mm512_add_ps(s2, _mm512_loadu_ps(r2 + index)));
      _mm512_storeu_ps(o3 + index, _mm512_add_ps(s3, _mm512_loadu_ps(r3 + index)));
      _mm512_storeu_ps(o4 + index, _mm512_add_ps(s4, _mm512_loadu_ps(r4 + index)));
      _mm512_storeu_ps(o5 + index, _mm512_add_ps(s5, _mm512_loadu_ps(r5 + index)));
      _mm512_storeu_ps(o6 + index, _mm512_add_ps(s6, _mm512_loadu_ps(r6 + index)));
      _mm512_storeu_ps(o7 + index, _mm512_add_ps(s7, _mm512_loadu_ps(r7 + index)));
    }
    for (; index < plane; ++index) {
      float s0 = bias ? bias[output] : 0.F, s1 = bias ? bias[output + 1] : 0.F;
      float s2 = bias ? bias[output + 2] : 0.F, s3 = bias ? bias[output + 3] : 0.F;
      float s4 = bias ? bias[output + 4] : 0.F, s5 = bias ? bias[output + 5] : 0.F;
      float s6 = bias ? bias[output + 6] : 0.F, s7 = bias ? bias[output + 7] : 0.F;
      for (int input = 0; input < input_channels; ++input) {
        const float x = src[std::size_t(input) * plane + index];
        s0 += w0[input] * x; s1 += w1[input] * x; s2 += w2[input] * x; s3 += w3[input] * x;
        s4 += w4[input] * x; s5 += w5[input] * x; s6 += w6[input] * x; s7 += w7[input] * x;
      }
      o0[index] = s0 + r0[index]; o1[index] = s1 + r1[index];
      o2[index] = s2 + r2[index]; o3[index] = s3 + r3[index];
      o4[index] = s4 + r4[index]; o5[index] = s5 + r5[index];
      o6[index] = s6 + r6[index]; o7[index] = s7 + r7[index];
    }
  }
  if (output < last_output) {
    Avx512PointwiseConvAdd4(dst, src, weights, bias, residual, output, last_output,
                            input_channels, plane);
  }
}

// Conv+residual-Add+ReLU fusion. This differs from applying ReLU after the
// existing Add kernel by keeping the final sum in registers for the clamp.
void Avx512PointwiseConvAddRelu4(float* dst, const float* src, const float* weights,
                                 const float* bias, const float* residual,
                                 int first_output, int last_output,
                                 int input_channels, std::size_t plane) noexcept {
  const __m512 zero = _mm512_setzero_ps();
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
    for (; index + 16 <= plane; index += 16) {
      __m512 s0 = _mm512_set1_ps(bias ? bias[output] : 0.F);
      __m512 s1 = _mm512_set1_ps(bias ? bias[output + 1] : 0.F);
      __m512 s2 = _mm512_set1_ps(bias ? bias[output + 2] : 0.F);
      __m512 s3 = _mm512_set1_ps(bias ? bias[output + 3] : 0.F);
      for (int input = 0; input < input_channels; ++input) {
        const __m512 x = _mm512_loadu_ps(src + std::size_t(input) * plane + index);
        s0 = _mm512_fmadd_ps(_mm512_set1_ps(w0[input]), x, s0);
        s1 = _mm512_fmadd_ps(_mm512_set1_ps(w1[input]), x, s1);
        s2 = _mm512_fmadd_ps(_mm512_set1_ps(w2[input]), x, s2);
        s3 = _mm512_fmadd_ps(_mm512_set1_ps(w3[input]), x, s3);
      }
      _mm512_storeu_ps(o0 + index, _mm512_max_ps(_mm512_add_ps(s0, _mm512_loadu_ps(r0 + index)), zero));
      _mm512_storeu_ps(o1 + index, _mm512_max_ps(_mm512_add_ps(s1, _mm512_loadu_ps(r1 + index)), zero));
      _mm512_storeu_ps(o2 + index, _mm512_max_ps(_mm512_add_ps(s2, _mm512_loadu_ps(r2 + index)), zero));
      _mm512_storeu_ps(o3 + index, _mm512_max_ps(_mm512_add_ps(s3, _mm512_loadu_ps(r3 + index)), zero));
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

void Avx512PointwiseConvAddRelu8(float* dst, const float* src, const float* weights,
                                 const float* bias, const float* residual,
                                 int first_output, int last_output,
                                 int input_channels, std::size_t plane) noexcept {
  // Eight live accumulators make this form register-pressure bound on both
  // AVX-512 client and server cores. Preserve the entry point for ABI and A/B
  // compatibility, but delegate to the proven four-output implementation.
  Avx512PointwiseConvAddRelu4(dst, src, weights, bias, residual, first_output,
                              last_output, input_channels, plane);
}

// Conv+ReLU fusion for 1x1 detector projections.  Clamp while the four output
// vectors are still resident in registers, removing a separate full-tensor
// read/write pass after the convolution.
void Avx512PointwiseConvRelu4(float* dst, const float* src, const float* weights,
                              const float* bias, int first_output, int last_output,
                              int input_channels, std::size_t plane) noexcept {
  const __m512 zero = _mm512_setzero_ps();
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
    for (; index + 16 <= plane; index += 16) {
      __m512 s0 = _mm512_set1_ps(bias ? bias[output] : 0.F);
      __m512 s1 = _mm512_set1_ps(bias ? bias[output + 1] : 0.F);
      __m512 s2 = _mm512_set1_ps(bias ? bias[output + 2] : 0.F);
      __m512 s3 = _mm512_set1_ps(bias ? bias[output + 3] : 0.F);
      for (int input = 0; input < input_channels; ++input) {
        const __m512 x = _mm512_loadu_ps(src + std::size_t(input) * plane + index);
        s0 = _mm512_fmadd_ps(_mm512_set1_ps(w0[input]), x, s0);
        s1 = _mm512_fmadd_ps(_mm512_set1_ps(w1[input]), x, s1);
        s2 = _mm512_fmadd_ps(_mm512_set1_ps(w2[input]), x, s2);
        s3 = _mm512_fmadd_ps(_mm512_set1_ps(w3[input]), x, s3);
      }
      _mm512_storeu_ps(out0 + index, _mm512_max_ps(s0, zero));
      _mm512_storeu_ps(out1 + index, _mm512_max_ps(s1, zero));
      _mm512_storeu_ps(out2 + index, _mm512_max_ps(s2, zero));
      _mm512_storeu_ps(out3 + index, _mm512_max_ps(s3, zero));
    }
    for (; index < plane; ++index) {
      float s0 = bias ? bias[output] : 0.F, s1 = bias ? bias[output + 1] : 0.F;
      float s2 = bias ? bias[output + 2] : 0.F, s3 = bias ? bias[output + 3] : 0.F;
      for (int input = 0; input < input_channels; ++input) {
        const float x = src[std::size_t(input) * plane + index];
        s0 += w0[input] * x; s1 += w1[input] * x;
        s2 += w2[input] * x; s3 += w3[input] * x;
      }
      out0[index] = std::max(s0, 0.F); out1[index] = std::max(s1, 0.F);
      out2[index] = std::max(s2, 0.F); out3[index] = std::max(s3, 0.F);
    }
  }
  for (; output < last_output; ++output) {
    float* out = dst + std::size_t(output) * plane;
    const float* filter = weights + std::size_t(output) * input_channels;
    std::size_t index = 0;
    for (; index + 16 <= plane; index += 16) {
      __m512 sum = _mm512_set1_ps(bias ? bias[output] : 0.F);
      for (int input = 0; input < input_channels; ++input) {
        sum = _mm512_fmadd_ps(_mm512_set1_ps(filter[input]),
                              _mm512_loadu_ps(src + std::size_t(input) * plane + index), sum);
      }
      _mm512_storeu_ps(out + index, _mm512_max_ps(sum, zero));
    }
    for (; index < plane; ++index) {
      float sum = bias ? bias[output] : 0.F;
      for (int input = 0; input < input_channels; ++input) sum += filter[input] * src[std::size_t(input) * plane + index];
      out[index] = std::max(sum, 0.F);
    }
  }
}

// Eight-output Conv+ReLU tile.  ReLU is applied before the one-and-only
// stores, preserving the existing fused activation memory traffic while
// doubling input-vector reuse for wide detector stem/projection layers.
void Avx512PointwiseConvRelu8(float* dst, const float* src, const float* weights,
                               const float* bias, int first_output, int last_output,
                               int input_channels, std::size_t plane) noexcept {
  const __m512 zero = _mm512_setzero_ps();
  int output = first_output;
  for (; output + 8 <= last_output; output += 8) {
    float* o0=dst+std::size_t(output)*plane; float* o1=o0+plane; float* o2=o1+plane; float* o3=o2+plane;
    float* o4=o3+plane; float* o5=o4+plane; float* o6=o5+plane; float* o7=o6+plane;
    const float* w0=weights+std::size_t(output)*input_channels; const float* w1=w0+input_channels;
    const float* w2=w1+input_channels; const float* w3=w2+input_channels; const float* w4=w3+input_channels;
    const float* w5=w4+input_channels; const float* w6=w5+input_channels; const float* w7=w6+input_channels;
    std::size_t index=0;
    for (; index+16<=plane; index+=16) {
      __m512 s0=_mm512_set1_ps(bias?bias[output]:0.F), s1=_mm512_set1_ps(bias?bias[output+1]:0.F);
      __m512 s2=_mm512_set1_ps(bias?bias[output+2]:0.F), s3=_mm512_set1_ps(bias?bias[output+3]:0.F);
      __m512 s4=_mm512_set1_ps(bias?bias[output+4]:0.F), s5=_mm512_set1_ps(bias?bias[output+5]:0.F);
      __m512 s6=_mm512_set1_ps(bias?bias[output+6]:0.F), s7=_mm512_set1_ps(bias?bias[output+7]:0.F);
      for (int input=0; input<input_channels; ++input) {
        const __m512 x=_mm512_loadu_ps(src+std::size_t(input)*plane+index);
        s0=_mm512_fmadd_ps(_mm512_set1_ps(w0[input]),x,s0); s1=_mm512_fmadd_ps(_mm512_set1_ps(w1[input]),x,s1);
        s2=_mm512_fmadd_ps(_mm512_set1_ps(w2[input]),x,s2); s3=_mm512_fmadd_ps(_mm512_set1_ps(w3[input]),x,s3);
        s4=_mm512_fmadd_ps(_mm512_set1_ps(w4[input]),x,s4); s5=_mm512_fmadd_ps(_mm512_set1_ps(w5[input]),x,s5);
        s6=_mm512_fmadd_ps(_mm512_set1_ps(w6[input]),x,s6); s7=_mm512_fmadd_ps(_mm512_set1_ps(w7[input]),x,s7);
      }
      _mm512_storeu_ps(o0+index,_mm512_max_ps(s0,zero)); _mm512_storeu_ps(o1+index,_mm512_max_ps(s1,zero));
      _mm512_storeu_ps(o2+index,_mm512_max_ps(s2,zero)); _mm512_storeu_ps(o3+index,_mm512_max_ps(s3,zero));
      _mm512_storeu_ps(o4+index,_mm512_max_ps(s4,zero)); _mm512_storeu_ps(o5+index,_mm512_max_ps(s5,zero));
      _mm512_storeu_ps(o6+index,_mm512_max_ps(s6,zero)); _mm512_storeu_ps(o7+index,_mm512_max_ps(s7,zero));
    }
    for (; index<plane; ++index) {
      float s0=bias?bias[output]:0.F,s1=bias?bias[output+1]:0.F,s2=bias?bias[output+2]:0.F,s3=bias?bias[output+3]:0.F;
      float s4=bias?bias[output+4]:0.F,s5=bias?bias[output+5]:0.F,s6=bias?bias[output+6]:0.F,s7=bias?bias[output+7]:0.F;
      for (int input=0; input<input_channels; ++input) { const float x=src[std::size_t(input)*plane+index];
        s0+=w0[input]*x;s1+=w1[input]*x;s2+=w2[input]*x;s3+=w3[input]*x;
        s4+=w4[input]*x;s5+=w5[input]*x;s6+=w6[input]*x;s7+=w7[input]*x; }
      o0[index]=std::max(s0,0.F);o1[index]=std::max(s1,0.F);o2[index]=std::max(s2,0.F);o3[index]=std::max(s3,0.F);
      o4[index]=std::max(s4,0.F);o5[index]=std::max(s5,0.F);o6[index]=std::max(s6,0.F);o7[index]=std::max(s7,0.F);
    }
  }
  if (output<last_output) Avx512PointwiseConvRelu4(dst,src,weights,bias,output,last_output,input_channels,plane);
}
void Avx512ConvTranspose2x2(float* dst, const float* src, const float* weights,
                              const float* bias, int first_output, int last_output,
                              int input_channels, int output_channels, int input_h, int input_w) noexcept {
  const int output_w = input_w * 2;
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(input_h * 2) * output_w;
  const __m512i duplicate = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3,
                                                4, 4, 5, 5, 6, 6, 7, 7);
  const __m512i dup_lo = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3,
                                            4, 4, 5, 5, 6, 6, 7, 7);
  const __m512i dup_hi = _mm512_setr_epi32(8, 8, 9, 9, 10, 10, 11, 11,
                                            12, 12, 13, 13, 14, 14, 15, 15);
  const __m512i top_idx = _mm512_setr_epi32(0, 1, 0, 1, 0, 1, 0, 1,
                                             0, 1, 0, 1, 0, 1, 0, 1);
  const __m512i bot_idx = _mm512_setr_epi32(2, 3, 2, 3, 2, 3, 2, 3,
                                             2, 3, 2, 3, 2, 3, 2, 3);
  // FPN ConvTranspose is 16-ch 40x176 / 80x352. Register-accumulate writes
  // each output pair once instead of RMW'ing per IC, but 8-run e2e lost to
  // the plane-hot RMW (16.24/15.76 vs 15.29/15.46): 110 KB output stays in
  // L2 while IC-inner src is 28 KB-strided. Keep as an explicit A/B.
  // `PPOCR_ENABLE_AVX512_TRANSPOSE_ACC` turns it on.
  static const bool acc =
      std::getenv("PPOCR_ENABLE_AVX512_TRANSPOSE_ACC") != nullptr &&
      std::getenv("PPOCR_DISABLE_AVX512_TRANSPOSE_ACC") == nullptr;
  if (acc) {
    for (int output = first_output; output < last_output; ++output) {
      float* out = dst + std::size_t(output) * output_plane;
      const float base = bias ? bias[output] : 0.F;
      const __m512 biasv = _mm512_set1_ps(base);
      for (int y = 0; y < input_h; ++y) {
        float* out0 = out + std::size_t(2 * y) * output_w;
        float* out1 = out0 + output_w;
        int x = 0;
        for (; x + 16 <= input_w; x += 16) {
          __m512 top0 = biasv, top1 = biasv, bot0 = biasv, bot1 = biasv;
          for (int input = 0; input < input_channels; ++input) {
            const float* row = src + std::size_t(input) * input_plane +
                               std::size_t(y) * input_w + x;
            const float* weight =
                weights + (std::size_t(input) * output_channels + output) * 4;
            const __m512 w = _mm512_broadcast_f32x4(_mm_loadu_ps(weight));
            const __m512 top = _mm512_permutexvar_ps(top_idx, w);
            const __m512 bottom = _mm512_permutexvar_ps(bot_idx, w);
            const __m512 values = _mm512_loadu_ps(row);
            const __m512 e0 = _mm512_permutexvar_ps(dup_lo, values);
            const __m512 e1 = _mm512_permutexvar_ps(dup_hi, values);
            top0 = _mm512_fmadd_ps(e0, top, top0);
            top1 = _mm512_fmadd_ps(e1, top, top1);
            bot0 = _mm512_fmadd_ps(e0, bottom, bot0);
            bot1 = _mm512_fmadd_ps(e1, bottom, bot1);
          }
          const int xx = x * 2;
          _mm512_storeu_ps(out0 + xx, top0);
          _mm512_storeu_ps(out0 + xx + 16, top1);
          _mm512_storeu_ps(out1 + xx, bot0);
          _mm512_storeu_ps(out1 + xx + 16, bot1);
        }
        for (; x + 8 <= input_w; x += 8) {
          __m512 topv = biasv, botv = biasv;
          for (int input = 0; input < input_channels; ++input) {
            const float* row = src + std::size_t(input) * input_plane +
                               std::size_t(y) * input_w + x;
            const float* weight =
                weights + (std::size_t(input) * output_channels + output) * 4;
            const __m512 w = _mm512_broadcast_f32x4(_mm_loadu_ps(weight));
            const __m512 top = _mm512_permutexvar_ps(top_idx, w);
            const __m512 bottom = _mm512_permutexvar_ps(bot_idx, w);
            const __m256 input8 = _mm256_loadu_ps(row);
            const __m512 duplicated =
                _mm512_permutexvar_ps(duplicate, _mm512_zextps256_ps512(input8));
            topv = _mm512_fmadd_ps(duplicated, top, topv);
            botv = _mm512_fmadd_ps(duplicated, bottom, botv);
          }
          const int xx = x * 2;
          _mm512_storeu_ps(out0 + xx, topv);
          _mm512_storeu_ps(out1 + xx, botv);
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
    std::fill_n(out, output_plane, bias ? bias[output] : 0.F);
    for (int input = 0; input < input_channels; ++input) {
      const float* values = src + std::size_t(input) * input_plane;
      const float* weight = weights + (std::size_t(input) * output_channels + output) * 4;
      const __m512 top = _mm512_setr_ps(weight[0], weight[1], weight[0], weight[1],
                                         weight[0], weight[1], weight[0], weight[1],
                                         weight[0], weight[1], weight[0], weight[1],
                                         weight[0], weight[1], weight[0], weight[1]);
      const __m512 bottom = _mm512_setr_ps(weight[2], weight[3], weight[2], weight[3],
                                            weight[2], weight[3], weight[2], weight[3],
                                            weight[2], weight[3], weight[2], weight[3],
                                            weight[2], weight[3], weight[2], weight[3]);
      for (int y = 0; y < input_h; ++y) {
        const float* row = values + std::size_t(y) * input_w;
        float* out0 = out + std::size_t(2 * y) * output_w;
        float* out1 = out0 + output_w;
        int x = 0;
        for (; x + 8 <= input_w; x += 8) {
          const __m256 input8 = _mm256_loadu_ps(row + x);
          const __m512 duplicated = _mm512_permutexvar_ps(duplicate, _mm512_zextps256_ps512(input8));
          const int xx = x * 2;
          _mm512_storeu_ps(out0 + xx, _mm512_fmadd_ps(duplicated, top, _mm512_loadu_ps(out0 + xx)));
          _mm512_storeu_ps(out1 + xx, _mm512_fmadd_ps(duplicated, bottom, _mm512_loadu_ps(out1 + xx)));
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

// The detector FPN's 2x2 stride-two transposed convolutions have independent
// output channels. Expand one input vector once, then FMA it into four output
// planes while their weights are hot. This preserves every output's increasing
// input-channel accumulation order and uses the existing one-output routine
// for the incomplete channel tail.
void Avx512ConvTranspose2x2x4(float* dst, const float* src, const float* weights,
                              const float* bias, int first_output, int last_output,
                              int input_channels, int output_channels, int input_h,
                              int input_w) noexcept {
  const int output_w = input_w * 2;
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(input_h * 2) * output_w;
  const __m512i duplicate = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3,
                                                4, 4, 5, 5, 6, 6, 7, 7);
  int output = first_output;
  for (; output + 4 <= last_output; output += 4) {
    float* out0 = dst + std::size_t(output) * output_plane;
    float* out1 = out0 + output_plane;
    float* out2 = out1 + output_plane;
    float* out3 = out2 + output_plane;
    std::fill_n(out0, output_plane, bias ? bias[output] : 0.F);
    std::fill_n(out1, output_plane, bias ? bias[output + 1] : 0.F);
    std::fill_n(out2, output_plane, bias ? bias[output + 2] : 0.F);
    std::fill_n(out3, output_plane, bias ? bias[output + 3] : 0.F);
    for (int input = 0; input < input_channels; ++input) {
      const float* values = src + std::size_t(input) * input_plane;
      const float* w0 = weights + (std::size_t(input) * output_channels + output) * 4;
      const float* w1 = w0 + 4;
      const float* w2 = w1 + 4;
      const float* w3 = w2 + 4;
      const __m512 top0 = _mm512_setr_ps(w0[0], w0[1], w0[0], w0[1], w0[0], w0[1], w0[0], w0[1],
                                           w0[0], w0[1], w0[0], w0[1], w0[0], w0[1], w0[0], w0[1]);
      const __m512 bottom0 = _mm512_setr_ps(w0[2], w0[3], w0[2], w0[3], w0[2], w0[3], w0[2], w0[3],
                                              w0[2], w0[3], w0[2], w0[3], w0[2], w0[3], w0[2], w0[3]);
      const __m512 top1 = _mm512_setr_ps(w1[0], w1[1], w1[0], w1[1], w1[0], w1[1], w1[0], w1[1],
                                           w1[0], w1[1], w1[0], w1[1], w1[0], w1[1], w1[0], w1[1]);
      const __m512 bottom1 = _mm512_setr_ps(w1[2], w1[3], w1[2], w1[3], w1[2], w1[3], w1[2], w1[3],
                                              w1[2], w1[3], w1[2], w1[3], w1[2], w1[3], w1[2], w1[3]);
      const __m512 top2 = _mm512_setr_ps(w2[0], w2[1], w2[0], w2[1], w2[0], w2[1], w2[0], w2[1],
                                           w2[0], w2[1], w2[0], w2[1], w2[0], w2[1], w2[0], w2[1]);
      const __m512 bottom2 = _mm512_setr_ps(w2[2], w2[3], w2[2], w2[3], w2[2], w2[3], w2[2], w2[3],
                                              w2[2], w2[3], w2[2], w2[3], w2[2], w2[3], w2[2], w2[3]);
      const __m512 top3 = _mm512_setr_ps(w3[0], w3[1], w3[0], w3[1], w3[0], w3[1], w3[0], w3[1],
                                           w3[0], w3[1], w3[0], w3[1], w3[0], w3[1], w3[0], w3[1]);
      const __m512 bottom3 = _mm512_setr_ps(w3[2], w3[3], w3[2], w3[3], w3[2], w3[3], w3[2], w3[3],
                                              w3[2], w3[3], w3[2], w3[3], w3[2], w3[3], w3[2], w3[3]);
      for (int y = 0; y < input_h; ++y) {
        const float* row = values + std::size_t(y) * input_w;
        float* o00 = out0 + std::size_t(2 * y) * output_w;
        float* o01 = o00 + output_w;
        float* o10 = out1 + std::size_t(2 * y) * output_w;
        float* o11 = o10 + output_w;
        float* o20 = out2 + std::size_t(2 * y) * output_w;
        float* o21 = o20 + output_w;
        float* o30 = out3 + std::size_t(2 * y) * output_w;
        float* o31 = o30 + output_w;
        int x = 0;
        for (; x + 8 <= input_w; x += 8) {
          const __m512 expanded = _mm512_permutexvar_ps(duplicate, _mm512_zextps256_ps512(_mm256_loadu_ps(row + x)));
          const int xx = x * 2;
          _mm512_storeu_ps(o00 + xx, _mm512_fmadd_ps(expanded, top0, _mm512_loadu_ps(o00 + xx)));
          _mm512_storeu_ps(o01 + xx, _mm512_fmadd_ps(expanded, bottom0, _mm512_loadu_ps(o01 + xx)));
          _mm512_storeu_ps(o10 + xx, _mm512_fmadd_ps(expanded, top1, _mm512_loadu_ps(o10 + xx)));
          _mm512_storeu_ps(o11 + xx, _mm512_fmadd_ps(expanded, bottom1, _mm512_loadu_ps(o11 + xx)));
          _mm512_storeu_ps(o20 + xx, _mm512_fmadd_ps(expanded, top2, _mm512_loadu_ps(o20 + xx)));
          _mm512_storeu_ps(o21 + xx, _mm512_fmadd_ps(expanded, bottom2, _mm512_loadu_ps(o21 + xx)));
          _mm512_storeu_ps(o30 + xx, _mm512_fmadd_ps(expanded, top3, _mm512_loadu_ps(o30 + xx)));
          _mm512_storeu_ps(o31 + xx, _mm512_fmadd_ps(expanded, bottom3, _mm512_loadu_ps(o31 + xx)));
        }
        for (; x < input_w; ++x) {
          const float value = row[x]; const int xx = x * 2;
          o00[xx] += value * w0[0]; o00[xx + 1] += value * w0[1]; o01[xx] += value * w0[2]; o01[xx + 1] += value * w0[3];
          o10[xx] += value * w1[0]; o10[xx + 1] += value * w1[1]; o11[xx] += value * w1[2]; o11[xx + 1] += value * w1[3];
          o20[xx] += value * w2[0]; o20[xx + 1] += value * w2[1]; o21[xx] += value * w2[2]; o21[xx + 1] += value * w2[3];
          o30[xx] += value * w3[0]; o30[xx + 1] += value * w3[1]; o31[xx] += value * w3[2]; o31[xx + 1] += value * w3[3];
        }
      }
    }
  }
  if (output < last_output) {
    Avx512ConvTranspose2x2(dst, src, weights, bias, output, last_output,
                            input_channels, output_channels, input_h, input_w);
  }
}

void Avx512ConvTranspose2x2Chain16x16x1(float* dst, const float* src, const float* w0,
                                        const float* b0, const float* w1, const float* b1,
                                        int input_h, int input_w) noexcept {
  const int output_w = input_w * 4;
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  alignas(64) float w0_pack[16][4][16];
  alignas(64) float w1_pack[4][16];
  for (int input = 0; input < 16; ++input) {
    for (int mid = 0; mid < 16; ++mid) {
      const float* w = w0 + (std::size_t(input) * 16 + mid) * 4;
      w0_pack[input][0][mid] = w[0];
      w0_pack[input][1][mid] = w[1];
      w0_pack[input][2][mid] = w[2];
      w0_pack[input][3][mid] = w[3];
    }
  }
  for (int mid = 0; mid < 16; ++mid) {
    const float* w = w1 + std::size_t(mid) * 4;
    w1_pack[0][mid] = w[0];
    w1_pack[1][mid] = w[1];
    w1_pack[2][mid] = w[2];
    w1_pack[3][mid] = w[3];
  }
  const __m512 b0v = b0 ? _mm512_loadu_ps(b0) : _mm512_setzero_ps();
  const __m512 b1v = _mm512_set1_ps(b1 ? b1[0] : 0.F);
  const __m512 one = _mm512_set1_ps(1.F);
  const __m512 zero = _mm512_setzero_ps();
  const auto body = [&](int first, int last) {
    for (int y = first; y < last; ++y) {
      for (int x = 0; x < input_w; ++x) {
        const std::size_t spatial = std::size_t(y) * input_w + x;
        __m512 t0 = b0v, t1 = b0v, t2 = b0v, t3 = b0v;
        for (int input = 0; input < 16; ++input) {
          const __m512 v = _mm512_set1_ps(src[std::size_t(input) * input_plane + spatial]);
          t0 = _mm512_fmadd_ps(v, _mm512_load_ps(w0_pack[input][0]), t0);
          t1 = _mm512_fmadd_ps(v, _mm512_load_ps(w0_pack[input][1]), t1);
          t2 = _mm512_fmadd_ps(v, _mm512_load_ps(w0_pack[input][2]), t2);
          t3 = _mm512_fmadd_ps(v, _mm512_load_ps(w0_pack[input][3]), t3);
        }
        t0 = _mm512_max_ps(t0, zero);
        t1 = _mm512_max_ps(t1, zero);
        t2 = _mm512_max_ps(t2, zero);
        t3 = _mm512_max_ps(t3, zero);
        const __m512* mid_taps[4] = {&t0, &t1, &t2, &t3};
        alignas(64) float logits[16];
        int slot = 0;
        for (int ky = 0; ky < 2; ++ky) {
          for (int kx = 0; kx < 2; ++kx) {
            const __m512 mid = *mid_taps[ky * 2 + kx];
            for (int ly = 0; ly < 2; ++ly) {
              for (int lx = 0; lx < 2; ++lx) {
                const __m512 prod = _mm512_mul_ps(mid, _mm512_load_ps(w1_pack[ly * 2 + lx]));
                logits[slot++] = _mm512_reduce_add_ps(prod);
              }
            }
          }
        }
        const __m512 sig = _mm512_div_ps(
            one, _mm512_add_ps(one, ExpPs512(_mm512_sub_ps(zero, _mm512_add_ps(
                     b1v, _mm512_load_ps(logits))))));
        alignas(64) float stored[16];
        _mm512_store_ps(stored, sig);
        const int y0 = y * 4;
        const int x0 = x * 4;
        for (int ky = 0; ky < 2; ++ky) {
          for (int kx = 0; kx < 2; ++kx) {
            for (int ly = 0; ly < 2; ++ly) {
              float* row = dst + std::size_t(y0 + ky * 2 + ly) * output_w + (x0 + kx * 2);
              const int base = ((ky * 2 + kx) * 2 + ly) * 2;
              row[0] = stored[base];
              row[1] = stored[base + 1];
            }
          }
        }
      }
    }
  };
  const auto work = std::int64_t(input_h) * input_w * 16 * 16;
  if (work >= 1000000 && input_h > 1) ParallelForRange(input_h, body);
  else body(0, input_h);
}

void Avx512GemmPanelLdb(float* dst, const float* a, const float* b, const float* bias,
                        int rows, int cols, int depth, int ldb) noexcept {
  if (!dst || !a || !b || rows <= 0 || cols <= 0 || depth <= 0 || ldb < cols) return;
  int row = 0;
  for (; row + 8 <= rows; row += 8) {
    float* out[8];
    const float* left[8];
    out[0] = dst + std::size_t(row) * cols;
    left[0] = a + std::size_t(row) * depth;
    for (int r = 1; r < 8; ++r) {
      out[r] = out[r - 1] + cols;
      left[r] = left[r - 1] + depth;
    }
    int col = 0;
    for (; col + 16 <= cols; col += 16) {
      __m512 v[8];
      const __m512 init = bias ? _mm512_loadu_ps(bias + col) : _mm512_setzero_ps();
      for (int r = 0; r < 8; ++r) v[r] = init;
      for (int k = 0; k < depth; ++k) {
        const __m512 right = _mm512_loadu_ps(b + std::size_t(k) * ldb + col);
        for (int r = 0; r < 8; ++r)
          v[r] = _mm512_fmadd_ps(_mm512_set1_ps(left[r][k]), right, v[r]);
      }
      for (int r = 0; r < 8; ++r) _mm512_storeu_ps(out[r] + col, v[r]);
    }
    for (; col < cols; ++col) {
      float s[8];
      const float init = bias ? bias[col] : 0.F;
      for (int r = 0; r < 8; ++r) s[r] = init;
      for (int k = 0; k < depth; ++k) {
        const float weight = b[std::size_t(k) * ldb + col];
        for (int r = 0; r < 8; ++r) s[r] += left[r][k] * weight;
      }
      for (int r = 0; r < 8; ++r) out[r][col] = s[r];
    }
  }
  for (; row < rows; ++row) {
    float* out = dst + std::size_t(row) * cols;
    const float* left = a + std::size_t(row) * depth;
    int col = 0;
    for (; col + 16 <= cols; col += 16) {
      __m512 v = bias ? _mm512_loadu_ps(bias + col) : _mm512_setzero_ps();
      for (int k = 0; k < depth; ++k) {
        v = _mm512_fmadd_ps(_mm512_set1_ps(left[k]),
                            _mm512_loadu_ps(b + std::size_t(k) * ldb + col), v);
      }
      _mm512_storeu_ps(out + col, v);
    }
    for (; col < cols; ++col) {
      float sum = bias ? bias[col] : 0.F;
      for (int k = 0; k < depth; ++k) sum += left[k] * b[std::size_t(k) * ldb + col];
      out[col] = sum;
    }
  }
}

void Avx512GemmPacked32(float* dst, const float* a, const float* packed_b,
                        const float* bias, int rows, int cols, int depth,
                        int* ctc_arg, float* ctc_max, float* ctc_sum) noexcept {
  // packed_b is [ceil(cols/32)][depth][32]: each vocabulary panel is K-contiguous
  // so the 8-row 32-wide micro-kernel streams 10 KB instead of striding 6906 floats.
  // Optional CTC buffers skip the [steps,vocab] store and keep a running
  // argmax + online softmax denominator. `PPOCR_DISABLE_CTC_ONLINE` keeps
  // the materialised logits path in the caller.
  const bool fuse_ctc = ctc_arg && ctc_max && ctc_sum;
  if ((!dst && !fuse_ctc) || !a || !packed_b || rows <= 0 || cols <= 0 || depth <= 0) return;
  if (fuse_ctc) {
    for (int row = 0; row < rows; ++row) {
      ctc_arg[row] = 0;
      ctc_max[row] = -std::numeric_limits<float>::infinity();
      ctc_sum[row] = 0.F;
    }
  }
  const int panels = (cols + 31) / 32;
  // Pack every 8-row A tile as [K][8] so named+K2 broadcasts consecutive
  // floats. Same-host 8-run stacked default missed versus stride-80 gathers
  // (`alloff` 16.21 vs default 16.82). Keep ENABLE-only.
  // `PPOCR_ENABLE_CTC_APACK` turns the [K][8] pack back on.
  static const bool apack =
      std::getenv("PPOCR_ENABLE_CTC_APACK") != nullptr &&
      std::getenv("PPOCR_DISABLE_CTC_APACK") == nullptr;
  static const bool prefetch =
      std::getenv("PPOCR_ENABLE_CTC_PREFETCH") != nullptr &&
      std::getenv("PPOCR_DISABLE_CTC_PREFETCH") == nullptr;
  thread_local std::vector<float> a_pack;
  const float* a_tiles = nullptr;
  const int tiles8 = rows / 8;
  if (apack && tiles8 > 0 && depth > 0 && depth <= 1024) {
    a_pack.resize(std::size_t(tiles8) * std::size_t(depth) * 8);
    for (int tile = 0; tile < tiles8; ++tile) {
      const float* left0 = a + std::size_t(tile * 8) * depth;
      float* dst8 = a_pack.data() + std::size_t(tile) * depth * 8;
      for (int k = 0; k < depth; ++k)
        for (int r = 0; r < 8; ++r) dst8[std::size_t(k) * 8 + r] = left0[r * depth + k];
    }
    a_tiles = a_pack.data();
  }
  for (int panel = 0; panel < panels; ++panel) {
    const int n0 = panel * 32;
    const int n_len = std::min(32, cols - n0);
    const float* pb = packed_b + std::size_t(panel) * depth * 32;
    const float* panel_bias = bias ? bias + n0 : nullptr;
    const __mmask16 mask0 = n_len >= 16 ? 0xFFFF : (__mmask16)((1u << n_len) - 1u);
    const __mmask16 mask1 = n_len <= 16 ? 0
                                        : (__mmask16)((1u << (n_len - 16)) - 1u);
    const __m512 b0init = panel_bias ? _mm512_maskz_loadu_ps(mask0, panel_bias)
                                     : _mm512_setzero_ps();
    const __m512 b1init = (panel_bias && n_len > 16)
        ? _mm512_maskz_loadu_ps(mask1, panel_bias + 16)
        : _mm512_setzero_ps();
    const auto store_rows = [&](float* const* out, const __m512* v0, const __m512* v1,
                                int count) noexcept {
      for (int r = 0; r < count; ++r) {
        _mm512_mask_storeu_ps(out[r] + n0, mask0, v0[r]);
        if (n_len > 16) _mm512_mask_storeu_ps(out[r] + n0 + 16, mask1, v1[r]);
      }
    };
    const __m512 ninf = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    const auto update_ctc_row = [&](__m512 v0, __m512 v1, int row) noexcept {
      v0 = _mm512_mask_mov_ps(ninf, mask0, v0);
      v1 = n_len > 16 ? _mm512_mask_mov_ps(ninf, mask1, v1) : ninf;
      const float pmax0 = _mm512_reduce_max_ps(v0);
      const float pmax1 = n_len > 16 ? _mm512_reduce_max_ps(v1) : -std::numeric_limits<float>::infinity();
      float pmax = pmax0;
      int parg = n0;
      if (pmax1 > pmax0) {
        pmax = pmax1;
        const __mmask16 eq = _mm512_cmp_ps_mask(v1, _mm512_set1_ps(pmax1), _CMP_EQ_OQ);
        parg = n0 + 16 + static_cast<int>(_tzcnt_u32(eq));
      } else {
        const __mmask16 eq = _mm512_cmp_ps_mask(v0, _mm512_set1_ps(pmax0), _CMP_EQ_OQ);
        parg = n0 + static_cast<int>(_tzcnt_u32(eq ? eq : 1u));
      }
      float& mx = ctc_max[row];
      float& sm = ctc_sum[row];
      if (pmax > mx) {
        if (std::isfinite(mx)) {
          sm *= _mm512_cvtss_f32(ExpPs512(_mm512_set1_ps(mx - pmax)));
        } else {
          sm = 0.F;
        }
        mx = pmax;
        ctc_arg[row] = parg;
      }
      const __m512 vmax = _mm512_set1_ps(mx);
      __m512 e0 = _mm512_maskz_mov_ps(mask0, ExpPs512(_mm512_sub_ps(v0, vmax)));
      float add = _mm512_reduce_add_ps(e0);
      if (n_len > 16) {
        const __m512 e1 = _mm512_maskz_mov_ps(mask1, ExpPs512(_mm512_sub_ps(v1, vmax)));
        add += _mm512_reduce_add_ps(e1);
      }
      sm += add;
    };
    const auto emit_rows = [&](float* const* out, const __m512* v0, const __m512* v1,
                               int count, int first_row) noexcept {
      if (fuse_ctc) {
        for (int r = 0; r < count; ++r) update_ctc_row(v0[r], v1[r], first_row + r);
        return;
      }
      store_rows(out, v0, v1, count);
    };
    if (prefetch && panel + 1 < panels) {
      const float* next_panel = packed_b + std::size_t(panel + 1) * depth * 32;
      for (int line = 0; line < 4 && line < depth; ++line)
        _mm_prefetch(reinterpret_cast<const char*>(next_panel + std::size_t(line) * 32),
                     _MM_HINT_T1);
    }
    int row = 0;
    for (; row + 8 <= rows; row += 8) {
      float* out[8]{};
      const float* left[8];
      left[0] = a + std::size_t(row) * depth;
      if (!fuse_ctc) out[0] = dst + std::size_t(row) * cols;
      for (int r = 1; r < 8; ++r) {
        if (!fuse_ctc) out[r] = out[r - 1] + cols;
        left[r] = left[r - 1] + depth;
      }
      __m512 v0[8], v1[8];
      for (int r = 0; r < 8; ++r) { v0[r] = b0init; v1[r] = b1init; }
      const float* ap = a_tiles ? a_tiles + std::size_t(row / 8) * depth * 8 : nullptr;
      // Named accumulators keep the 8×32 tile in the AVX-512 file. The
      // `v0[8]` form spilled on MSVC /O2. Combined with K2 this is a new
      // default versus the previous ENABLE-only named-without-K2 miss.
      // `PPOCR_DISABLE_AVX512_CTC_NAMED` restores the array loop.
      static const bool named =
          std::getenv("PPOCR_DISABLE_AVX512_CTC_NAMED") == nullptr;
      if (named) {
        __m512 a0 = b0init, a1 = b1init, b0 = b0init, b1 = b1init;
        __m512 c0 = b0init, c1 = b1init, d0 = b0init, d1 = b1init;
        __m512 e0 = b0init, e1 = b1init, f0 = b0init, f1 = b1init;
        __m512 g0 = b0init, g1 = b1init, h0 = b0init, h1 = b1init;
        static const bool k2 =
            std::getenv("PPOCR_DISABLE_CTC_K2") == nullptr;
        int k = 0;
        if (k2) {
          for (; k + 2 <= depth; k += 2) {
            const float* right = pb + std::size_t(k) * 32;
            if (prefetch && k + 4 <= depth) {
              _mm_prefetch(reinterpret_cast<const char*>(right + 64), _MM_HINT_T0);
              if (ap) _mm_prefetch(reinterpret_cast<const char*>(ap + std::size_t(k + 2) * 8),
                                   _MM_HINT_T0);
            }
            const __m512 rb00 = _mm512_loadu_ps(right);
            const __m512 rb01 = _mm512_loadu_ps(right + 16);
            const __m512 rb10 = _mm512_loadu_ps(right + 32);
            const __m512 rb11 = _mm512_loadu_ps(right + 48);
            const float* ar0 = ap ? ap + std::size_t(k) * 8 : nullptr;
            const float* ar1 = ap ? ar0 + 8 : nullptr;
            const float a00 = ar0 ? ar0[0] : left[0][k];
            const float a01 = ar0 ? ar0[1] : left[1][k];
            const float a02 = ar0 ? ar0[2] : left[2][k];
            const float a03 = ar0 ? ar0[3] : left[3][k];
            const float a04 = ar0 ? ar0[4] : left[4][k];
            const float a05 = ar0 ? ar0[5] : left[5][k];
            const float a06 = ar0 ? ar0[6] : left[6][k];
            const float a07 = ar0 ? ar0[7] : left[7][k];
            const float b00 = ar1 ? ar1[0] : left[0][k + 1];
            const float b01 = ar1 ? ar1[1] : left[1][k + 1];
            const float b02 = ar1 ? ar1[2] : left[2][k + 1];
            const float b03 = ar1 ? ar1[3] : left[3][k + 1];
            const float b04 = ar1 ? ar1[4] : left[4][k + 1];
            const float b05 = ar1 ? ar1[5] : left[5][k + 1];
            const float b06 = ar1 ? ar1[6] : left[6][k + 1];
            const float b07 = ar1 ? ar1[7] : left[7][k + 1];
            a0 = _mm512_fmadd_ps(_mm512_set1_ps(a00), rb00, a0);
            a1 = _mm512_fmadd_ps(_mm512_set1_ps(a00), rb01, a1);
            a0 = _mm512_fmadd_ps(_mm512_set1_ps(b00), rb10, a0);
            a1 = _mm512_fmadd_ps(_mm512_set1_ps(b00), rb11, a1);
            b0 = _mm512_fmadd_ps(_mm512_set1_ps(a01), rb00, b0);
            b1 = _mm512_fmadd_ps(_mm512_set1_ps(a01), rb01, b1);
            b0 = _mm512_fmadd_ps(_mm512_set1_ps(b01), rb10, b0);
            b1 = _mm512_fmadd_ps(_mm512_set1_ps(b01), rb11, b1);
            c0 = _mm512_fmadd_ps(_mm512_set1_ps(a02), rb00, c0);
            c1 = _mm512_fmadd_ps(_mm512_set1_ps(a02), rb01, c1);
            c0 = _mm512_fmadd_ps(_mm512_set1_ps(b02), rb10, c0);
            c1 = _mm512_fmadd_ps(_mm512_set1_ps(b02), rb11, c1);
            d0 = _mm512_fmadd_ps(_mm512_set1_ps(a03), rb00, d0);
            d1 = _mm512_fmadd_ps(_mm512_set1_ps(a03), rb01, d1);
            d0 = _mm512_fmadd_ps(_mm512_set1_ps(b03), rb10, d0);
            d1 = _mm512_fmadd_ps(_mm512_set1_ps(b03), rb11, d1);
            e0 = _mm512_fmadd_ps(_mm512_set1_ps(a04), rb00, e0);
            e1 = _mm512_fmadd_ps(_mm512_set1_ps(a04), rb01, e1);
            e0 = _mm512_fmadd_ps(_mm512_set1_ps(b04), rb10, e0);
            e1 = _mm512_fmadd_ps(_mm512_set1_ps(b04), rb11, e1);
            f0 = _mm512_fmadd_ps(_mm512_set1_ps(a05), rb00, f0);
            f1 = _mm512_fmadd_ps(_mm512_set1_ps(a05), rb01, f1);
            f0 = _mm512_fmadd_ps(_mm512_set1_ps(b05), rb10, f0);
            f1 = _mm512_fmadd_ps(_mm512_set1_ps(b05), rb11, f1);
            g0 = _mm512_fmadd_ps(_mm512_set1_ps(a06), rb00, g0);
            g1 = _mm512_fmadd_ps(_mm512_set1_ps(a06), rb01, g1);
            g0 = _mm512_fmadd_ps(_mm512_set1_ps(b06), rb10, g0);
            g1 = _mm512_fmadd_ps(_mm512_set1_ps(b06), rb11, g1);
            h0 = _mm512_fmadd_ps(_mm512_set1_ps(a07), rb00, h0);
            h1 = _mm512_fmadd_ps(_mm512_set1_ps(a07), rb01, h1);
            h0 = _mm512_fmadd_ps(_mm512_set1_ps(b07), rb10, h0);
            h1 = _mm512_fmadd_ps(_mm512_set1_ps(b07), rb11, h1);
          }
        }
        for (; k < depth; ++k) {
          const float* right = pb + std::size_t(k) * 32;
          const __m512 rb0 = _mm512_loadu_ps(right);
          const __m512 rb1 = _mm512_loadu_ps(right + 16);
          const float* ar = ap ? ap + std::size_t(k) * 8 : nullptr;
          const float a00 = ar ? ar[0] : left[0][k];
          const float a01 = ar ? ar[1] : left[1][k];
          const float a02 = ar ? ar[2] : left[2][k];
          const float a03 = ar ? ar[3] : left[3][k];
          const float a04 = ar ? ar[4] : left[4][k];
          const float a05 = ar ? ar[5] : left[5][k];
          const float a06 = ar ? ar[6] : left[6][k];
          const float a07 = ar ? ar[7] : left[7][k];
          a0 = _mm512_fmadd_ps(_mm512_set1_ps(a00), rb0, a0);
          a1 = _mm512_fmadd_ps(_mm512_set1_ps(a00), rb1, a1);
          b0 = _mm512_fmadd_ps(_mm512_set1_ps(a01), rb0, b0);
          b1 = _mm512_fmadd_ps(_mm512_set1_ps(a01), rb1, b1);
          c0 = _mm512_fmadd_ps(_mm512_set1_ps(a02), rb0, c0);
          c1 = _mm512_fmadd_ps(_mm512_set1_ps(a02), rb1, c1);
          d0 = _mm512_fmadd_ps(_mm512_set1_ps(a03), rb0, d0);
          d1 = _mm512_fmadd_ps(_mm512_set1_ps(a03), rb1, d1);
          e0 = _mm512_fmadd_ps(_mm512_set1_ps(a04), rb0, e0);
          e1 = _mm512_fmadd_ps(_mm512_set1_ps(a04), rb1, e1);
          f0 = _mm512_fmadd_ps(_mm512_set1_ps(a05), rb0, f0);
          f1 = _mm512_fmadd_ps(_mm512_set1_ps(a05), rb1, f1);
          g0 = _mm512_fmadd_ps(_mm512_set1_ps(a06), rb0, g0);
          g1 = _mm512_fmadd_ps(_mm512_set1_ps(a06), rb1, g1);
          h0 = _mm512_fmadd_ps(_mm512_set1_ps(a07), rb0, h0);
          h1 = _mm512_fmadd_ps(_mm512_set1_ps(a07), rb1, h1);
        }
        v0[0] = a0; v1[0] = a1; v0[1] = b0; v1[1] = b1;
        v0[2] = c0; v1[2] = c1; v0[3] = d0; v1[3] = d1;
        v0[4] = e0; v1[4] = e1; v0[5] = f0; v1[5] = f1;
        v0[6] = g0; v1[6] = g1; v0[7] = h0; v1[7] = h1;
      } else {
        // Pair K so each 32-wide B panel is in-flight across two A broadcasts.
        // Same FMAs; `PPOCR_DISABLE_CTC_K2` restores one-K.
        static const bool k2 =
            std::getenv("PPOCR_DISABLE_CTC_K2") == nullptr;
        int k = 0;
        if (k2) {
          for (; k + 2 <= depth; k += 2) {
            const float* right0 = pb + std::size_t(k) * 32;
            const float* right1 = right0 + 32;
            const __m512 rb00 = _mm512_loadu_ps(right0);
            const __m512 rb01 = _mm512_loadu_ps(right0 + 16);
            const __m512 rb10 = _mm512_loadu_ps(right1);
            const __m512 rb11 = _mm512_loadu_ps(right1 + 16);
            const float* ar0 = ap ? ap + std::size_t(k) * 8 : nullptr;
            const float* ar1 = ap ? ar0 + 8 : nullptr;
            for (int r = 0; r < 8; ++r) {
              const __m512 av0 = _mm512_set1_ps(ar0 ? ar0[r] : left[r][k]);
              const __m512 av1 = _mm512_set1_ps(ar1 ? ar1[r] : left[r][k + 1]);
              v0[r] = _mm512_fmadd_ps(av0, rb00, v0[r]);
              v1[r] = _mm512_fmadd_ps(av0, rb01, v1[r]);
              v0[r] = _mm512_fmadd_ps(av1, rb10, v0[r]);
              v1[r] = _mm512_fmadd_ps(av1, rb11, v1[r]);
            }
          }
        }
        for (; k < depth; ++k) {
          const float* right = pb + std::size_t(k) * 32;
          const __m512 rb0 = _mm512_loadu_ps(right);
          const __m512 rb1 = _mm512_loadu_ps(right + 16);
          const float* ar = ap ? ap + std::size_t(k) * 8 : nullptr;
          for (int r = 0; r < 8; ++r) {
            const __m512 av = _mm512_set1_ps(ar ? ar[r] : left[r][k]);
            v0[r] = _mm512_fmadd_ps(av, rb0, v0[r]);
            v1[r] = _mm512_fmadd_ps(av, rb1, v1[r]);
          }
        }
      }
      emit_rows(out, v0, v1, 8, row);
    }
    for (; row + 4 <= rows; row += 4) {
      float* out[4]{};
      const float* left[4];
      left[0] = a + std::size_t(row) * depth;
      if (!fuse_ctc) out[0] = dst + std::size_t(row) * cols;
      for (int r = 1; r < 4; ++r) {
        if (!fuse_ctc) out[r] = out[r - 1] + cols;
        left[r] = left[r - 1] + depth;
      }
      __m512 v0[4], v1[4];
      for (int r = 0; r < 4; ++r) { v0[r] = b0init; v1[r] = b1init; }
      for (int k = 0; k < depth; ++k) {
        const float* right = pb + std::size_t(k) * 32;
        const __m512 rb0 = _mm512_loadu_ps(right);
        const __m512 rb1 = _mm512_loadu_ps(right + 16);
        for (int r = 0; r < 4; ++r) {
          const __m512 av = _mm512_set1_ps(left[r][k]);
          v0[r] = _mm512_fmadd_ps(av, rb0, v0[r]);
          v1[r] = _mm512_fmadd_ps(av, rb1, v1[r]);
        }
      }
      emit_rows(out, v0, v1, 4, row);
    }
    for (; row < rows; ++row) {
      float* out = fuse_ctc ? nullptr : dst + std::size_t(row) * cols;
      const float* left = a + std::size_t(row) * depth;
      __m512 v0 = b0init, v1 = b1init;
      for (int k = 0; k < depth; ++k) {
        const float* right = pb + std::size_t(k) * 32;
        const __m512 av = _mm512_set1_ps(left[k]);
        v0 = _mm512_fmadd_ps(av, _mm512_loadu_ps(right), v0);
        v1 = _mm512_fmadd_ps(av, _mm512_loadu_ps(right + 16), v1);
      }
      if (fuse_ctc) {
        update_ctc_row(v0, v1, row);
      } else {
        _mm512_mask_storeu_ps(out + n0, mask0, v0);
        if (n_len > 16) _mm512_mask_storeu_ps(out + n0 + 16, mask1, v1);
      }
    }
  }
}

void Avx512GemmRows(float* dst, const float* a, const float* b, const float* bias,
                    int first_row, int last_row, int cols, int depth) noexcept {
  // Transformer projections arrive as a short batch of independent rows.
  // Pairing rows keeps the numerical order for each output unchanged while
  // every B vector is loaded once and consumed by two FMA accumulator sets.
  // This matters for recognition batches where B is a large immutable model
  // matrix and the former one-row kernel repeatedly streamed it from cache.
  const auto one_row = [&](int row) noexcept {
    float* out = dst + std::size_t(row) * cols;
    const float* left = a + std::size_t(row) * depth;
    int col = 0;
    for (; col + 64 <= cols; col += 64) {
      __m512 c0 = bias ? _mm512_loadu_ps(bias + col) : _mm512_setzero_ps();
      __m512 c1 = bias ? _mm512_loadu_ps(bias + col + 16) : _mm512_setzero_ps();
      __m512 c2 = bias ? _mm512_loadu_ps(bias + col + 32) : _mm512_setzero_ps();
      __m512 c3 = bias ? _mm512_loadu_ps(bias + col + 48) : _mm512_setzero_ps();
      for (int k = 0; k < depth; ++k) {
        const __m512 av = _mm512_set1_ps(left[k]);
        const float* right = b + std::size_t(k) * cols + col;
        c0 = _mm512_fmadd_ps(av, _mm512_loadu_ps(right), c0);
        c1 = _mm512_fmadd_ps(av, _mm512_loadu_ps(right + 16), c1);
        c2 = _mm512_fmadd_ps(av, _mm512_loadu_ps(right + 32), c2);
        c3 = _mm512_fmadd_ps(av, _mm512_loadu_ps(right + 48), c3);
      }
      _mm512_storeu_ps(out + col, c0); _mm512_storeu_ps(out + col + 16, c1);
      _mm512_storeu_ps(out + col + 32, c2); _mm512_storeu_ps(out + col + 48, c3);
    }
    for (; col + 16 <= cols; col += 16) {
      __m512 value = bias ? _mm512_loadu_ps(bias + col) : _mm512_setzero_ps();
      for (int k = 0; k < depth; ++k) {
        value = _mm512_fmadd_ps(_mm512_set1_ps(left[k]),
                                _mm512_loadu_ps(b + std::size_t(k) * cols + col), value);
      }
      _mm512_storeu_ps(out + col, value);
    }
    for (; col < cols; ++col) {
      float value = bias ? bias[col] : 0.F;
      for (int k = 0; k < depth; ++k) value += left[k] * b[std::size_t(k) * cols + col];
      out[col] = value;
    }
  };
  int row = first_row;
  // Recognition batches commonly expose four or more independent transformer
  // rows.  The two-row micro-kernel already shares B across rows; on AVX-512
  // there are enough architectural registers to extend its 64-column tile to
  // four rows without spilling its accumulators.  This halves the immutable
  // B-matrix loads again while each row preserves exactly the same ascending
  // K FMA sequence as the scalar and two-row paths.
  // Keep an A/B escape hatch because the four-row tile trades additional
  // register pressure for fewer immutable-B loads. It is entered dynamically
  // on AVX-512 only; AVX2/NEON/scalar dispatch and output semantics stay
  // unchanged.
  // The 4x64 tile is retained for deployment A/B: it can reduce immutable B
  // loads, but end-to-end OCR timing is sensitive to register pressure and
  // cache residency. The conservative two-row path is the portable default.
  // This code is reached only after runtime AVX-512 dispatch, so ARM and older
  // x86 systems keep their existing SIMD/scalar paths unchanged.
  // Tiny CTC is 4–8 rows by 6906 classes at depth 80/192. Sharing B across
  // four rows halves the vocabulary-matrix traffic versus the two-row path.
  // Attention GEMMs keep cols≈192 and stay on two-row. Disable with
  // PPOCR_DISABLE_AVX512_GEMM4ROW; PPOCR_ENABLE_AVX512_GEMM4ROW remains a
  // no-op alias for older scripts.
  // Four-row 16-wide: four accumulators share each B vector. That is enough
  // to cut tiny-CTC vocabulary traffic without the 16-accumulator 64-wide
  // tile that spilled. The wide tile stays opt-in.
  static const bool gemm4_narrow =
      std::getenv("PPOCR_DISABLE_AVX512_GEMM4NARROW") == nullptr;
  static const bool gemm4_wide =
      std::getenv("PPOCR_ENABLE_AVX512_GEMM4ROW") != nullptr;
  static const bool gemm8_narrow =
      std::getenv("PPOCR_DISABLE_AVX512_GEMM8NARROW") == nullptr;
  static const bool gemm16_narrow =
      std::getenv("PPOCR_ENABLE_AVX512_GEMM16NARROW") != nullptr &&
      std::getenv("PPOCR_DISABLE_AVX512_GEMM16NARROW") == nullptr;
  const bool wide_vocab = depth >= 64 && cols >= 2048;
  const bool use_four_row_tile = gemm4_wide && wide_vocab;
  // Sixteen rows share each 16-wide B vector, cutting tiny-CTC vocabulary
  // traffic versus the eight-row 32-wide tile. Rec Gemm stays serial: 16 rows
  // is below workers*2. `PPOCR_DISABLE_AVX512_GEMM16NARROW` restores 8-row.
  const bool use_sixteen_row_narrow =
      !use_four_row_tile && gemm16_narrow && wide_vocab;
  const bool use_eight_row_narrow =
      !use_four_row_tile && !use_sixteen_row_narrow && gemm8_narrow && wide_vocab;
  const bool use_four_row_narrow =
      !use_four_row_tile && !use_sixteen_row_narrow && !use_eight_row_narrow &&
      gemm4_narrow && wide_vocab;
  for (; use_sixteen_row_narrow && row + 15 < last_row; row += 16) {
    float* out[16];
    const float* left[16];
    out[0] = dst + std::size_t(row) * cols;
    left[0] = a + std::size_t(row) * depth;
    for (int r = 1; r < 16; ++r) {
      out[r] = out[r - 1] + cols;
      left[r] = left[r - 1] + depth;
    }
    int col = 0;
    for (; col + 16 <= cols; col += 16) {
      __m512 v[16];
      const __m512 init = bias ? _mm512_loadu_ps(bias + col) : _mm512_setzero_ps();
      for (int r = 0; r < 16; ++r) v[r] = init;
      for (int k = 0; k < depth; ++k) {
        const __m512 right = _mm512_loadu_ps(b + std::size_t(k) * cols + col);
        for (int r = 0; r < 16; ++r)
          v[r] = _mm512_fmadd_ps(_mm512_set1_ps(left[r][k]), right, v[r]);
      }
      for (int r = 0; r < 16; ++r) _mm512_storeu_ps(out[r] + col, v[r]);
    }
    for (; col < cols; ++col) {
      float s[16];
      const float init = bias ? bias[col] : 0.F;
      for (int r = 0; r < 16; ++r) s[r] = init;
      for (int k = 0; k < depth; ++k) {
        const float weight = b[std::size_t(k) * cols + col];
        for (int r = 0; r < 16; ++r) s[r] += left[r][k] * weight;
      }
      for (int r = 0; r < 16; ++r) out[r][col] = s[r];
    }
  }
  for (; use_eight_row_narrow && row + 7 < last_row; row += 8) {
    float* out[8];
    const float* left[8];
    out[0] = dst + std::size_t(row) * cols;
    left[0] = a + std::size_t(row) * depth;
    for (int r = 1; r < 8; ++r) {
      out[r] = out[r - 1] + cols;
      left[r] = left[r - 1] + depth;
    }
    int col = 0;
    for (; col + 32 <= cols; col += 32) {
      __m512 v0[8], v1[8];
      const __m512 b0init = bias ? _mm512_loadu_ps(bias + col) : _mm512_setzero_ps();
      const __m512 b1init = bias ? _mm512_loadu_ps(bias + col + 16) : _mm512_setzero_ps();
      for (int r = 0; r < 8; ++r) { v0[r] = b0init; v1[r] = b1init; }
      for (int k = 0; k < depth; ++k) {
        const float* right = b + std::size_t(k) * cols + col;
        const __m512 rb0 = _mm512_loadu_ps(right);
        const __m512 rb1 = _mm512_loadu_ps(right + 16);
        for (int r = 0; r < 8; ++r) {
          const __m512 av = _mm512_set1_ps(left[r][k]);
          v0[r] = _mm512_fmadd_ps(av, rb0, v0[r]);
          v1[r] = _mm512_fmadd_ps(av, rb1, v1[r]);
        }
      }
      for (int r = 0; r < 8; ++r) {
        _mm512_storeu_ps(out[r] + col, v0[r]);
        _mm512_storeu_ps(out[r] + col + 16, v1[r]);
      }
    }
    for (; col + 16 <= cols; col += 16) {
      __m512 v[8];
      const __m512 init = bias ? _mm512_loadu_ps(bias + col) : _mm512_setzero_ps();
      for (int r = 0; r < 8; ++r) v[r] = init;
      for (int k = 0; k < depth; ++k) {
        const __m512 right = _mm512_loadu_ps(b + std::size_t(k) * cols + col);
        for (int r = 0; r < 8; ++r)
          v[r] = _mm512_fmadd_ps(_mm512_set1_ps(left[r][k]), right, v[r]);
      }
      for (int r = 0; r < 8; ++r) _mm512_storeu_ps(out[r] + col, v[r]);
    }
    for (; col < cols; ++col) {
      float s[8];
      const float init = bias ? bias[col] : 0.F;
      for (int r = 0; r < 8; ++r) s[r] = init;
      for (int k = 0; k < depth; ++k) {
        const float weight = b[std::size_t(k) * cols + col];
        for (int r = 0; r < 8; ++r) s[r] += left[r][k] * weight;
      }
      for (int r = 0; r < 8; ++r) out[r][col] = s[r];
    }
  }
  for (; use_four_row_narrow && row + 3 < last_row; row += 4) {
    float* out0 = dst + std::size_t(row) * cols;
    float* out1 = out0 + cols;
    float* out2 = out1 + cols;
    float* out3 = out2 + cols;
    const float* left0 = a + std::size_t(row) * depth;
    const float* left1 = left0 + depth;
    const float* left2 = left1 + depth;
    const float* left3 = left2 + depth;
    int col = 0;
    for (; col + 32 <= cols; col += 32) {
      __m512 v00 = bias ? _mm512_loadu_ps(bias + col) : _mm512_setzero_ps();
      __m512 v01 = bias ? _mm512_loadu_ps(bias + col + 16) : _mm512_setzero_ps();
      __m512 v10 = v00, v11 = v01, v20 = v00, v21 = v01, v30 = v00, v31 = v01;
      for (int k = 0; k < depth; ++k) {
        const float* right = b + std::size_t(k) * cols + col;
        const __m512 b0 = _mm512_loadu_ps(right);
        const __m512 b1 = _mm512_loadu_ps(right + 16);
        const __m512 a0 = _mm512_set1_ps(left0[k]);
        const __m512 a1 = _mm512_set1_ps(left1[k]);
        const __m512 a2 = _mm512_set1_ps(left2[k]);
        const __m512 a3 = _mm512_set1_ps(left3[k]);
        v00 = _mm512_fmadd_ps(a0, b0, v00); v01 = _mm512_fmadd_ps(a0, b1, v01);
        v10 = _mm512_fmadd_ps(a1, b0, v10); v11 = _mm512_fmadd_ps(a1, b1, v11);
        v20 = _mm512_fmadd_ps(a2, b0, v20); v21 = _mm512_fmadd_ps(a2, b1, v21);
        v30 = _mm512_fmadd_ps(a3, b0, v30); v31 = _mm512_fmadd_ps(a3, b1, v31);
      }
      _mm512_storeu_ps(out0 + col, v00); _mm512_storeu_ps(out0 + col + 16, v01);
      _mm512_storeu_ps(out1 + col, v10); _mm512_storeu_ps(out1 + col + 16, v11);
      _mm512_storeu_ps(out2 + col, v20); _mm512_storeu_ps(out2 + col + 16, v21);
      _mm512_storeu_ps(out3 + col, v30); _mm512_storeu_ps(out3 + col + 16, v31);
    }
    for (; col + 16 <= cols; col += 16) {
      __m512 v0 = bias ? _mm512_loadu_ps(bias + col) : _mm512_setzero_ps();
      __m512 v1 = v0, v2 = v0, v3 = v0;
      for (int k = 0; k < depth; ++k) {
        const __m512 right = _mm512_loadu_ps(b + std::size_t(k) * cols + col);
        v0 = _mm512_fmadd_ps(_mm512_set1_ps(left0[k]), right, v0);
        v1 = _mm512_fmadd_ps(_mm512_set1_ps(left1[k]), right, v1);
        v2 = _mm512_fmadd_ps(_mm512_set1_ps(left2[k]), right, v2);
        v3 = _mm512_fmadd_ps(_mm512_set1_ps(left3[k]), right, v3);
      }
      _mm512_storeu_ps(out0 + col, v0); _mm512_storeu_ps(out1 + col, v1);
      _mm512_storeu_ps(out2 + col, v2); _mm512_storeu_ps(out3 + col, v3);
    }
    for (; col < cols; ++col) {
      float s0 = bias ? bias[col] : 0.F, s1 = s0, s2 = s0, s3 = s0;
      for (int k = 0; k < depth; ++k) {
        const float weight = b[std::size_t(k) * cols + col];
        s0 += left0[k] * weight; s1 += left1[k] * weight;
        s2 += left2[k] * weight; s3 += left3[k] * weight;
      }
      out0[col] = s0; out1[col] = s1; out2[col] = s2; out3[col] = s3;
    }
  }
  for (; use_four_row_tile && row + 3 < last_row; row += 4) {
    float* out0 = dst + std::size_t(row) * cols;
    float* out1 = out0 + cols;
    float* out2 = out1 + cols;
    float* out3 = out2 + cols;
    const float* left0 = a + std::size_t(row) * depth;
    const float* left1 = left0 + depth;
    const float* left2 = left1 + depth;
    const float* left3 = left2 + depth;
    int col = 0;
    for (; col + 64 <= cols; col += 64) {
      __m512 a00 = bias ? _mm512_loadu_ps(bias + col) : _mm512_setzero_ps();
      __m512 a01 = bias ? _mm512_loadu_ps(bias + col + 16) : _mm512_setzero_ps();
      __m512 a02 = bias ? _mm512_loadu_ps(bias + col + 32) : _mm512_setzero_ps();
      __m512 a03 = bias ? _mm512_loadu_ps(bias + col + 48) : _mm512_setzero_ps();
      __m512 a10 = a00, a11 = a01, a12 = a02, a13 = a03;
      __m512 a20 = a00, a21 = a01, a22 = a02, a23 = a03;
      __m512 a30 = a00, a31 = a01, a32 = a02, a33 = a03;
      for (int k = 0; k < depth; ++k) {
        const __m512 lhs0 = _mm512_set1_ps(left0[k]);
        const __m512 lhs1 = _mm512_set1_ps(left1[k]);
        const __m512 lhs2 = _mm512_set1_ps(left2[k]);
        const __m512 lhs3 = _mm512_set1_ps(left3[k]);
        const float* right = b + std::size_t(k) * cols + col;
        const __m512 b0 = _mm512_loadu_ps(right);
        const __m512 b1 = _mm512_loadu_ps(right + 16);
        const __m512 b2 = _mm512_loadu_ps(right + 32);
        const __m512 b3 = _mm512_loadu_ps(right + 48);
        a00 = _mm512_fmadd_ps(lhs0, b0, a00); a10 = _mm512_fmadd_ps(lhs1, b0, a10);
        a20 = _mm512_fmadd_ps(lhs2, b0, a20); a30 = _mm512_fmadd_ps(lhs3, b0, a30);
        a01 = _mm512_fmadd_ps(lhs0, b1, a01); a11 = _mm512_fmadd_ps(lhs1, b1, a11);
        a21 = _mm512_fmadd_ps(lhs2, b1, a21); a31 = _mm512_fmadd_ps(lhs3, b1, a31);
        a02 = _mm512_fmadd_ps(lhs0, b2, a02); a12 = _mm512_fmadd_ps(lhs1, b2, a12);
        a22 = _mm512_fmadd_ps(lhs2, b2, a22); a32 = _mm512_fmadd_ps(lhs3, b2, a32);
        a03 = _mm512_fmadd_ps(lhs0, b3, a03); a13 = _mm512_fmadd_ps(lhs1, b3, a13);
        a23 = _mm512_fmadd_ps(lhs2, b3, a23); a33 = _mm512_fmadd_ps(lhs3, b3, a33);
      }
      _mm512_storeu_ps(out0 + col, a00); _mm512_storeu_ps(out0 + col + 16, a01);
      _mm512_storeu_ps(out0 + col + 32, a02); _mm512_storeu_ps(out0 + col + 48, a03);
      _mm512_storeu_ps(out1 + col, a10); _mm512_storeu_ps(out1 + col + 16, a11);
      _mm512_storeu_ps(out1 + col + 32, a12); _mm512_storeu_ps(out1 + col + 48, a13);
      _mm512_storeu_ps(out2 + col, a20); _mm512_storeu_ps(out2 + col + 16, a21);
      _mm512_storeu_ps(out2 + col + 32, a22); _mm512_storeu_ps(out2 + col + 48, a23);
      _mm512_storeu_ps(out3 + col, a30); _mm512_storeu_ps(out3 + col + 16, a31);
      _mm512_storeu_ps(out3 + col + 32, a32); _mm512_storeu_ps(out3 + col + 48, a33);
    }
    for (; col + 16 <= cols; col += 16) {
      __m512 value0 = bias ? _mm512_loadu_ps(bias + col) : _mm512_setzero_ps();
      __m512 value1 = value0, value2 = value0, value3 = value0;
      for (int k = 0; k < depth; ++k) {
        const __m512 right = _mm512_loadu_ps(b + std::size_t(k) * cols + col);
        value0 = _mm512_fmadd_ps(_mm512_set1_ps(left0[k]), right, value0);
        value1 = _mm512_fmadd_ps(_mm512_set1_ps(left1[k]), right, value1);
        value2 = _mm512_fmadd_ps(_mm512_set1_ps(left2[k]), right, value2);
        value3 = _mm512_fmadd_ps(_mm512_set1_ps(left3[k]), right, value3);
      }
      _mm512_storeu_ps(out0 + col, value0); _mm512_storeu_ps(out1 + col, value1);
      _mm512_storeu_ps(out2 + col, value2); _mm512_storeu_ps(out3 + col, value3);
    }
    for (; col < cols; ++col) {
      float value0 = bias ? bias[col] : 0.F;
      float value1 = value0, value2 = value0, value3 = value0;
      for (int k = 0; k < depth; ++k) {
        const float weight = b[std::size_t(k) * cols + col];
        value0 += left0[k] * weight; value1 += left1[k] * weight;
        value2 += left2[k] * weight; value3 += left3[k] * weight;
      }
      out0[col] = value0; out1[col] = value1;
      out2[col] = value2; out3[col] = value3;
    }
  }
  for (; row + 1 < last_row; row += 2) {
    float* out0 = dst + std::size_t(row) * cols;
    float* out1 = out0 + cols;
    const float* left0 = a + std::size_t(row) * depth;
    const float* left1 = left0 + depth;
    int col = 0;
    for (; col + 64 <= cols; col += 64) {
      __m512 a00 = bias ? _mm512_loadu_ps(bias + col) : _mm512_setzero_ps();
      __m512 a01 = bias ? _mm512_loadu_ps(bias + col + 16) : _mm512_setzero_ps();
      __m512 a02 = bias ? _mm512_loadu_ps(bias + col + 32) : _mm512_setzero_ps();
      __m512 a03 = bias ? _mm512_loadu_ps(bias + col + 48) : _mm512_setzero_ps();
      __m512 a10 = a00, a11 = a01, a12 = a02, a13 = a03;
      for (int k = 0; k < depth; ++k) {
        const __m512 lhs0 = _mm512_set1_ps(left0[k]);
        const __m512 lhs1 = _mm512_set1_ps(left1[k]);
        const float* right = b + std::size_t(k) * cols + col;
        const __m512 b0 = _mm512_loadu_ps(right);
        const __m512 b1 = _mm512_loadu_ps(right + 16);
        const __m512 b2 = _mm512_loadu_ps(right + 32);
        const __m512 b3 = _mm512_loadu_ps(right + 48);
        a00 = _mm512_fmadd_ps(lhs0, b0, a00); a10 = _mm512_fmadd_ps(lhs1, b0, a10);
        a01 = _mm512_fmadd_ps(lhs0, b1, a01); a11 = _mm512_fmadd_ps(lhs1, b1, a11);
        a02 = _mm512_fmadd_ps(lhs0, b2, a02); a12 = _mm512_fmadd_ps(lhs1, b2, a12);
        a03 = _mm512_fmadd_ps(lhs0, b3, a03); a13 = _mm512_fmadd_ps(lhs1, b3, a13);
      }
      _mm512_storeu_ps(out0 + col, a00); _mm512_storeu_ps(out0 + col + 16, a01);
      _mm512_storeu_ps(out0 + col + 32, a02); _mm512_storeu_ps(out0 + col + 48, a03);
      _mm512_storeu_ps(out1 + col, a10); _mm512_storeu_ps(out1 + col + 16, a11);
      _mm512_storeu_ps(out1 + col + 32, a12); _mm512_storeu_ps(out1 + col + 48, a13);
    }
    for (; col + 16 <= cols; col += 16) {
      __m512 value0 = bias ? _mm512_loadu_ps(bias + col) : _mm512_setzero_ps();
      __m512 value1 = value0;
      for (int k = 0; k < depth; ++k) {
        const __m512 right = _mm512_loadu_ps(b + std::size_t(k) * cols + col);
        value0 = _mm512_fmadd_ps(_mm512_set1_ps(left0[k]), right, value0);
        value1 = _mm512_fmadd_ps(_mm512_set1_ps(left1[k]), right, value1);
      }
      _mm512_storeu_ps(out0 + col, value0);
      _mm512_storeu_ps(out1 + col, value1);
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

void Avx512GemmAccumulateRows(float* dst, const float* a, const float* b,
                              int first_row, int last_row, int cols, int depth) noexcept {
  for (int row = first_row; row < last_row; ++row) {
    float* out = dst + std::size_t(row) * cols;
    const float* left = a + std::size_t(row) * depth;
    int col = 0;
    for (; col + 64 <= cols; col += 64) {
      __m512 c0 = _mm512_loadu_ps(out + col);
      __m512 c1 = _mm512_loadu_ps(out + col + 16);
      __m512 c2 = _mm512_loadu_ps(out + col + 32);
      __m512 c3 = _mm512_loadu_ps(out + col + 48);
      for (int k = 0; k < depth; ++k) {
        const __m512 av = _mm512_set1_ps(left[k]);
        const float* right = b + std::size_t(k) * cols + col;
        c0 = _mm512_fmadd_ps(av, _mm512_loadu_ps(right), c0);
        c1 = _mm512_fmadd_ps(av, _mm512_loadu_ps(right + 16), c1);
        c2 = _mm512_fmadd_ps(av, _mm512_loadu_ps(right + 32), c2);
        c3 = _mm512_fmadd_ps(av, _mm512_loadu_ps(right + 48), c3);
      }
      _mm512_storeu_ps(out + col, c0); _mm512_storeu_ps(out + col + 16, c1);
      _mm512_storeu_ps(out + col + 32, c2); _mm512_storeu_ps(out + col + 48, c3);
    }
    for (; col + 16 <= cols; col += 16) {
      __m512 value = _mm512_loadu_ps(out + col);
      for (int k = 0; k < depth; ++k) {
        value = _mm512_fmadd_ps(_mm512_set1_ps(left[k]),
                                _mm512_loadu_ps(b + std::size_t(k) * cols + col), value);
      }
      _mm512_storeu_ps(out + col, value);
    }
    for (; col < cols; ++col) {
      float value = out[col];
      for (int k = 0; k < depth; ++k) value += left[k] * b[std::size_t(k) * cols + col];
      out[col] = value;
    }
  }
}

void Avx512DepthwiseConv(float* dst, const float* src, const float* weights,
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
    // Conv.78 is 64-ch 5x5 SAME on 40x176. Unroll the 25 taps so the
    // interior 16-wide body is a straight FMA chain instead of a 5x5 loop.
    // `PPOCR_DISABLE_AVX512_DW5_UNROLL` restores the generic nested loops.
    static const bool dw5_unroll =
        std::getenv("PPOCR_DISABLE_AVX512_DW5_UNROLL") == nullptr;
    if (dw5_unroll && kernel_h == 5 && kernel_w == 5) {
      for (int y = 0; y < output_h; ++y) {
        const int iy0 = y - pad_top;
        const int interior_begin = y >= first_y && y < last_y ? first_x : 0;
        const int interior_end = y >= first_y && y < last_y ? last_x : 0;
        int x = 0;
        for (; x < interior_begin; ++x) {
          float sum = base;
          const int ix0 = x - pad_left;
          for (int ky = 0; ky < 5; ++ky) {
            const int iy = iy0 + ky;
            if (iy < 0 || iy >= input_h) continue;
            for (int kx = 0; kx < 5; ++kx) {
              const int ix = ix0 + kx;
              if (ix >= 0 && ix < input_w)
                sum += in[std::size_t(iy) * input_w + ix] * filter[ky * 5 + kx];
            }
          }
          out[std::size_t(y) * output_w + x] = sum;
        }
        static const bool x32 =
            std::getenv("PPOCR_ENABLE_AVX512_DW5_X32") != nullptr &&
            std::getenv("PPOCR_DISABLE_AVX512_DW5_X32") == nullptr;
        for (; x32 && x + 32 <= interior_end; x += 32) {
          __m512 sum0 = _mm512_set1_ps(base);
          __m512 sum1 = _mm512_set1_ps(base);
          const float* row0 = in + std::size_t(iy0) * input_w + x - pad_left;
          for (int ky = 0; ky < 5; ++ky) {
            const float* row = row0 + std::size_t(ky) * input_w;
            const float* k = filter + ky * 5;
            const __m512 c0 = _mm512_set1_ps(k[0]);
            const __m512 c1 = _mm512_set1_ps(k[1]);
            const __m512 c2 = _mm512_set1_ps(k[2]);
            const __m512 c3 = _mm512_set1_ps(k[3]);
            const __m512 c4 = _mm512_set1_ps(k[4]);
            sum0 = _mm512_fmadd_ps(c0, _mm512_loadu_ps(row), sum0);
            sum1 = _mm512_fmadd_ps(c0, _mm512_loadu_ps(row + 16), sum1);
            sum0 = _mm512_fmadd_ps(c1, _mm512_loadu_ps(row + 1), sum0);
            sum1 = _mm512_fmadd_ps(c1, _mm512_loadu_ps(row + 17), sum1);
            sum0 = _mm512_fmadd_ps(c2, _mm512_loadu_ps(row + 2), sum0);
            sum1 = _mm512_fmadd_ps(c2, _mm512_loadu_ps(row + 18), sum1);
            sum0 = _mm512_fmadd_ps(c3, _mm512_loadu_ps(row + 3), sum0);
            sum1 = _mm512_fmadd_ps(c3, _mm512_loadu_ps(row + 19), sum1);
            sum0 = _mm512_fmadd_ps(c4, _mm512_loadu_ps(row + 4), sum0);
            sum1 = _mm512_fmadd_ps(c4, _mm512_loadu_ps(row + 20), sum1);
          }
          _mm512_storeu_ps(out + std::size_t(y) * output_w + x, sum0);
          _mm512_storeu_ps(out + std::size_t(y) * output_w + x + 16, sum1);
        }
        for (; x + 16 <= interior_end; x += 16) {
          __m512 sum = _mm512_set1_ps(base);
          const float* row0 = in + std::size_t(iy0) * input_w + x - pad_left;
          for (int ky = 0; ky < 5; ++ky) {
            const float* row = row0 + std::size_t(ky) * input_w;
            const float* k = filter + ky * 5;
            sum = _mm512_fmadd_ps(_mm512_set1_ps(k[0]), _mm512_loadu_ps(row), sum);
            sum = _mm512_fmadd_ps(_mm512_set1_ps(k[1]), _mm512_loadu_ps(row + 1), sum);
            sum = _mm512_fmadd_ps(_mm512_set1_ps(k[2]), _mm512_loadu_ps(row + 2), sum);
            sum = _mm512_fmadd_ps(_mm512_set1_ps(k[3]), _mm512_loadu_ps(row + 3), sum);
            sum = _mm512_fmadd_ps(_mm512_set1_ps(k[4]), _mm512_loadu_ps(row + 4), sum);
          }
          _mm512_storeu_ps(out + std::size_t(y) * output_w + x, sum);
        }
        if (Avx512MaskTailEnabled() && y >= first_y && y < last_y && x < output_w &&
            output_w - x <= 16) {
          const int remain = output_w - x;
          const __mmask16 omask = Avx512CountMask(remain);
          __m512 sum = _mm512_set1_ps(base);
          const float* row0 = in + std::size_t(iy0) * input_w + x - pad_left;
          for (int ky = 0; ky < 5; ++ky) {
            const float* row = row0 + std::size_t(ky) * input_w;
            const float* k = filter + ky * 5;
            for (int kx = 0; kx < 5; ++kx) {
              const __mmask16 kmask =
                  Avx512BoundMask(x - pad_left + kx, input_w, omask);
              sum = _mm512_fmadd_ps(_mm512_set1_ps(k[kx]),
                                    _mm512_maskz_loadu_ps(kmask, row + kx), sum);
            }
          }
          _mm512_mask_storeu_ps(out + std::size_t(y) * output_w + x, omask, sum);
          x = output_w;
        }
        for (; x < output_w; ++x) {
          float sum = base;
          const int ix0 = x - pad_left;
          for (int ky = 0; ky < 5; ++ky) {
            const int iy = iy0 + ky;
            if (iy < 0 || iy >= input_h) continue;
            for (int kx = 0; kx < 5; ++kx) {
              const int ix = ix0 + kx;
              if (ix >= 0 && ix < input_w)
                sum += in[std::size_t(iy) * input_w + ix] * filter[ky * 5 + kx];
            }
          }
          out[std::size_t(y) * output_w + x] = sum;
        }
      }
      continue;
    }
    for (int y = 0; y < output_h; ++y) {
      const int iy0 = y - pad_top;
      int x = 0;
      // Left border (and whole rows that touch vertical padding).
      const int interior_begin = y >= first_y && y < last_y ? first_x : 0;
      const int interior_end = y >= first_y && y < last_y ? last_x : 0;
      for (; x < interior_begin; ++x) {
        float sum = base;
        const int ix0 = x - pad_left;
        for (int ky = 0; ky < kernel_h; ++ky) {
          const int iy = iy0 + ky;
          if (iy < 0 || iy >= input_h) continue;
          for (int kx = 0; kx < kernel_w; ++kx) {
            const int ix = ix0 + kx;
            if (ix >= 0 && ix < input_w) sum += in[std::size_t(iy) * input_w + ix] * filter[ky * kernel_w + kx];
          }
        }
        out[std::size_t(y) * output_w + x] = sum;
      }
      for (; x + 16 <= interior_end; x += 16) {
        __m512 sum = _mm512_set1_ps(base);
        for (int ky = 0; ky < kernel_h; ++ky) {
          const float* row = in + std::size_t(iy0 + ky) * input_w + x - pad_left;
          const float* kernel = filter + ky * kernel_w;
          for (int kx = 0; kx < kernel_w; ++kx) {
            sum = _mm512_fmadd_ps(_mm512_set1_ps(kernel[kx]),
                                  _mm512_loadu_ps(row + kx), sum);
          }
        }
        _mm512_storeu_ps(out + std::size_t(y) * output_w + x, sum);
      }
      if (Avx512MaskTailEnabled() && y >= first_y && y < last_y && x < output_w &&
          output_w - x <= 16) {
        const int remain = output_w - x;
        const __mmask16 omask = Avx512CountMask(remain);
        __m512 sum = _mm512_set1_ps(base);
        for (int ky = 0; ky < kernel_h; ++ky) {
          const float* row = in + std::size_t(iy0 + ky) * input_w + x - pad_left;
          const float* kernel = filter + ky * kernel_w;
          for (int kx = 0; kx < kernel_w; ++kx) {
            const __mmask16 kmask =
                Avx512BoundMask(x - pad_left + kx, input_w, omask);
            sum = _mm512_fmadd_ps(_mm512_set1_ps(kernel[kx]),
                                  _mm512_maskz_loadu_ps(kmask, row + kx), sum);
          }
        }
        _mm512_mask_storeu_ps(out + std::size_t(y) * output_w + x, omask, sum);
        x = output_w;
      }
      for (; x < output_w; ++x) {
        float sum = base;
        const int ix0 = x - pad_left;
        for (int ky = 0; ky < kernel_h; ++ky) {
          const int iy = iy0 + ky;
          if (iy < 0 || iy >= input_h) continue;
          for (int kx = 0; kx < kernel_w; ++kx) {
            const int ix = ix0 + kx;
            if (ix >= 0 && ix < input_w) sum += in[std::size_t(iy) * input_w + ix] * filter[ky * kernel_w + kx];
          }
        }
        out[std::size_t(y) * output_w + x] = sum;
      }
    }
  }
}

void Avx512DepthwisePointwiseConv3x3S1(float* dst, const float* src,
                                       const float* dw_weights, const float* dw_bias,
                                       const float* pw_weights, const float* pw_bias,
                                       int channels, int output_channels, int height,
                                       int width, int activation,
                                       bool approx_gelu) noexcept {
  if (!dst || !src || !dw_weights || !pw_weights || channels <= 0 ||
      output_channels <= 0 || height <= 0 || width <= 0) {
    return;
  }
  const std::size_t plane = std::size_t(height) * width;
  const int first_y = 1;
  const int first_x = 1;
  const int last_y = std::min(height, height - 1);
  const int last_x = std::min(width, width - 1);
  const __m512 zero = _mm512_setzero_ps();
  const __m512 one = _mm512_set1_ps(1.F);
  const __m512 half = _mm512_set1_ps(.5F);
  const __m512 sixth = _mm512_set1_ps(1.F / 6.F);
  const __m512 inv_sqrt2 = _mm512_set1_ps(0.7071067811865475244F);
  const __m512 gelu_c0 = _mm512_set1_ps(0.7978845608028654F);
  const __m512 gelu_c1 = _mm512_set1_ps(0.044715F);
  const __m512 p27 = _mm512_set1_ps(27.F);
  const __m512 p9 = _mm512_set1_ps(9.F);
  const __m512 negative_one = _mm512_set1_ps(-1.F);
  const auto activate = [&](__m512 x) noexcept {
    if (activation == 1) return _mm512_max_ps(x, zero);
    if (activation == 2) {
      const __m512 gate = _mm512_min_ps(one, _mm512_max_ps(zero,
          _mm512_fmadd_ps(x, sixth, half)));
      return _mm512_mul_ps(x, gate);
    }
    if (activation == 3) {
      if (approx_gelu) {
        const __m512 x3 = _mm512_mul_ps(x, _mm512_mul_ps(x, x));
        const __m512 z = _mm512_mul_ps(gelu_c0, _mm512_fmadd_ps(gelu_c1, x3, x));
        const __m512 z2 = _mm512_mul_ps(z, z);
        const __m512 t = _mm512_max_ps(negative_one, _mm512_min_ps(one,
            _mm512_div_ps(_mm512_mul_ps(z, _mm512_add_ps(p27, z2)),
                          _mm512_add_ps(p27, _mm512_mul_ps(p9, z2)))));
        return _mm512_mul_ps(half, _mm512_mul_ps(x, _mm512_add_ps(one, t)));
      }
      return _mm512_mul_ps(half, _mm512_mul_ps(x,
          _mm512_add_ps(one, ErfPs512(_mm512_mul_ps(x, inv_sqrt2)))));
    }
    return x;
  };
  const auto activate_scalar = [&](float x) noexcept {
    if (activation == 1) return std::max(x, 0.F);
    if (activation == 2) return x * std::clamp(x / 6.F + .5F, 0.F, 1.F);
    if (activation == 3) {
      if (approx_gelu) {
        const float z = 0.7978845608028654F * (x + 0.044715F * x * x * x);
        const float z2 = z * z;
        const float t = std::clamp(z * (27.F + z2) / (27.F + 9.F * z2), -1.F, 1.F);
        return .5F * x * (1.F + t);
      }
      return x * .5F * (1.F + std::erf(x * 0.7071067811865475244F));
    }
    return x;
  };
  thread_local std::vector<float> dw_row;
  dw_row.resize(std::size_t(channels) * width);
  for (int y = 0; y < height; ++y) {
    const int iy0 = y - 1;
    const int interior_begin = y >= first_y && y < last_y ? first_x : 0;
    const int interior_end = y >= first_y && y < last_y ? last_x : 0;
    for (int channel = 0; channel < channels; ++channel) {
      const float* in = src + std::size_t(channel) * plane;
      const float* filter = dw_weights + std::size_t(channel) * 9;
      float* out = dw_row.data() + std::size_t(channel) * width;
      const float base = dw_bias ? dw_bias[channel] : 0.F;
      int x = 0;
      for (; x < interior_begin; ++x) {
        float sum = base;
        const int ix0 = x - 1;
        for (int ky = 0; ky < 3; ++ky) {
          const int iy = iy0 + ky;
          if (iy < 0 || iy >= height) continue;
          for (int kx = 0; kx < 3; ++kx) {
            const int ix = ix0 + kx;
            if (ix >= 0 && ix < width) {
              sum += in[std::size_t(iy) * width + ix] * filter[ky * 3 + kx];
            }
          }
        }
        out[x] = sum;
      }
      for (; x + 16 <= interior_end; x += 16) {
        __m512 sum = _mm512_set1_ps(base);
        for (int ky = 0; ky < 3; ++ky) {
          const float* row = in + std::size_t(iy0 + ky) * width + x - 1;
          const float* kernel = filter + ky * 3;
          for (int kx = 0; kx < 3; ++kx) {
            sum = _mm512_fmadd_ps(_mm512_set1_ps(kernel[kx]),
                                  _mm512_loadu_ps(row + kx), sum);
          }
        }
        _mm512_storeu_ps(out + x, sum);
      }
      for (; x < width; ++x) {
        float sum = base;
        const int ix0 = x - 1;
        for (int ky = 0; ky < 3; ++ky) {
          const int iy = iy0 + ky;
          if (iy < 0 || iy >= height) continue;
          for (int kx = 0; kx < 3; ++kx) {
            const int ix = ix0 + kx;
            if (ix >= 0 && ix < width) {
              sum += in[std::size_t(iy) * width + ix] * filter[ky * 3 + kx];
            }
          }
        }
        out[x] = sum;
      }
    }
    const std::size_t row = std::size_t(y) * width;
    int output = 0;
    for (; output + 4 <= output_channels; output += 4) {
      const float* w0 = pw_weights + std::size_t(output) * channels;
      const float* w1 = w0 + channels;
      const float* w2 = w1 + channels;
      const float* w3 = w2 + channels;
      float* o0 = dst + std::size_t(output) * plane + row;
      float* o1 = dst + std::size_t(output + 1) * plane + row;
      float* o2 = dst + std::size_t(output + 2) * plane + row;
      float* o3 = dst + std::size_t(output + 3) * plane + row;
      int x = 0;
      for (; x + 16 <= width; x += 16) {
        __m512 s0 = _mm512_set1_ps(pw_bias ? pw_bias[output] : 0.F);
        __m512 s1 = _mm512_set1_ps(pw_bias ? pw_bias[output + 1] : 0.F);
        __m512 s2 = _mm512_set1_ps(pw_bias ? pw_bias[output + 2] : 0.F);
        __m512 s3 = _mm512_set1_ps(pw_bias ? pw_bias[output + 3] : 0.F);
        for (int input = 0; input < channels; ++input) {
          const __m512 v = _mm512_loadu_ps(dw_row.data() + std::size_t(input) * width + x);
          s0 = _mm512_fmadd_ps(_mm512_set1_ps(w0[input]), v, s0);
          s1 = _mm512_fmadd_ps(_mm512_set1_ps(w1[input]), v, s1);
          s2 = _mm512_fmadd_ps(_mm512_set1_ps(w2[input]), v, s2);
          s3 = _mm512_fmadd_ps(_mm512_set1_ps(w3[input]), v, s3);
        }
        _mm512_storeu_ps(o0 + x, activate(s0));
        _mm512_storeu_ps(o1 + x, activate(s1));
        _mm512_storeu_ps(o2 + x, activate(s2));
        _mm512_storeu_ps(o3 + x, activate(s3));
      }
      for (; x < width; ++x) {
        float s0 = pw_bias ? pw_bias[output] : 0.F;
        float s1 = pw_bias ? pw_bias[output + 1] : 0.F;
        float s2 = pw_bias ? pw_bias[output + 2] : 0.F;
        float s3 = pw_bias ? pw_bias[output + 3] : 0.F;
        for (int input = 0; input < channels; ++input) {
          const float v = dw_row[std::size_t(input) * width + x];
          s0 += w0[input] * v;
          s1 += w1[input] * v;
          s2 += w2[input] * v;
          s3 += w3[input] * v;
        }
        o0[x] = activate_scalar(s0);
        o1[x] = activate_scalar(s1);
        o2[x] = activate_scalar(s2);
        o3[x] = activate_scalar(s3);
      }
    }
    for (; output < output_channels; ++output) {
      const float* filter = pw_weights + std::size_t(output) * channels;
      float* out = dst + std::size_t(output) * plane + row;
      int x = 0;
      for (; x + 16 <= width; x += 16) {
        __m512 sum = _mm512_set1_ps(pw_bias ? pw_bias[output] : 0.F);
        for (int input = 0; input < channels; ++input) {
          sum = _mm512_fmadd_ps(_mm512_set1_ps(filter[input]),
                                _mm512_loadu_ps(dw_row.data() + std::size_t(input) * width + x),
                                sum);
        }
        _mm512_storeu_ps(out + x, activate(sum));
      }
      for (; x < width; ++x) {
        float sum = pw_bias ? pw_bias[output] : 0.F;
        for (int input = 0; input < channels; ++input) {
          sum += filter[input] * dw_row[std::size_t(input) * width + x];
        }
        out[x] = activate_scalar(sum);
      }
    }
  }
}

void Avx512DepthwisePointwiseConv5x5S1(float* dst, const float* src,
                                       const float* dw_weights, const float* dw_bias,
                                       const float* pw_weights, const float* pw_bias,
                                       int channels, int output_channels, int height,
                                       int width, int activation,
                                       bool approx_gelu, int y0, int y1) noexcept {
  if (!dst || !src || !dw_weights || !pw_weights || channels <= 0 ||
      output_channels <= 0 || height <= 0 || width <= 0) {
    return;
  }
  if (y0 < 0) y0 = 0;
  // Conv.78's 16-ch 1x1 walks 64 strided OC rows per spatial vec. Packing
  // each 8-OC group as [IC][8] was an 8-run maybe, 20-run miss (15.80 vs
  // 14.91). Keep ENABLE-only. `PPOCR_ENABLE_AVX512_DWPW5_PWPACK` turns it on.
  static const bool pw_pack =
      std::getenv("PPOCR_ENABLE_AVX512_DWPW5_PWPACK") != nullptr &&
      std::getenv("PPOCR_DISABLE_AVX512_DWPW5_PWPACK") == nullptr;
  const int oc_group = 8;
  const int pack_groups = (output_channels / oc_group) * oc_group;
  thread_local std::vector<float> packed_pw;
  const float* pw8 = pw_weights;
  if (pw_pack && pack_groups >= oc_group && channels > 0) {
    packed_pw.resize(std::size_t(pack_groups) * channels);
    for (int group = 0; group < pack_groups; group += oc_group) {
      float* dst8 = packed_pw.data() + std::size_t(group) * channels;
      for (int input = 0; input < channels; ++input) {
        float* lane = dst8 + std::size_t(input) * oc_group;
        for (int oc = 0; oc < oc_group; ++oc)
          lane[oc] = pw_weights[std::size_t(group + oc) * channels + input];
      }
    }
    pw8 = packed_pw.data();
  }
  if (y1 < 0 || y1 > height) y1 = height;
  if (y0 >= y1) return;
  const std::size_t plane = std::size_t(height) * width;
  const int first_y = 2;
  const int first_x = 2;
  const int last_y = std::min(height, height - 2);
  const int last_x = std::min(width, width - 2);
  const __m512 zero = _mm512_setzero_ps();
  const __m512 one = _mm512_set1_ps(1.F);
  const __m512 half = _mm512_set1_ps(.5F);
  const __m512 sixth = _mm512_set1_ps(1.F / 6.F);
  const __m512 inv_sqrt2 = _mm512_set1_ps(0.7071067811865475244F);
  const auto activate = [&](__m512 x) noexcept {
    if (activation == 1) return _mm512_max_ps(x, zero);
    if (activation == 2) {
      const __m512 gate = _mm512_min_ps(one, _mm512_max_ps(zero,
          _mm512_fmadd_ps(x, sixth, half)));
      return _mm512_mul_ps(x, gate);
    }
    if (activation == 3) {
      return _mm512_mul_ps(half, _mm512_mul_ps(x,
          _mm512_add_ps(one, ErfPs512(_mm512_mul_ps(x, inv_sqrt2)))));
    }
    return x;
  };
  const auto activate_scalar = [&](float x) noexcept {
    if (activation == 1) return std::max(x, 0.F);
    if (activation == 2) return x * std::clamp(x / 6.F + .5F, 0.F, 1.F);
    if (activation == 3) return x * .5F * (1.F + std::erf(x * 0.7071067811865475244F));
    return x;
  };
  (void)approx_gelu;
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
            if (ix >= 0 && ix < width)
              sum += in[std::size_t(iy) * width + ix] * filter[ky * 5 + kx];
          }
        }
        out[x] = sum;
      }
      // Conv.78 is 64-ch 5x5 SAME on 40x176. A 32-wide interior body issues
      // two independent FMA chains per tap. Same-host 8-run was noise versus
      // the 16-wide walk, so this stays ENABLE-only like unfused DW X32.
      // `PPOCR_ENABLE_AVX512_DWPW5_X32` turns it on.
      static const bool dw5_x32 =
          std::getenv("PPOCR_ENABLE_AVX512_DWPW5_X32") != nullptr &&
          std::getenv("PPOCR_DISABLE_AVX512_DWPW5_X32") == nullptr;
      for (; dw5_x32 && x + 32 <= interior_end; x += 32) {
        __m512 sum0 = _mm512_set1_ps(base);
        __m512 sum1 = _mm512_set1_ps(base);
        const float* row0 = in + std::size_t(iy0) * width + x - 2;
        for (int ky = 0; ky < 5; ++ky) {
          const float* row = row0 + std::size_t(ky) * width;
          const float* k = filter + ky * 5;
          const __m512 c0 = _mm512_set1_ps(k[0]);
          const __m512 c1 = _mm512_set1_ps(k[1]);
          const __m512 c2 = _mm512_set1_ps(k[2]);
          const __m512 c3 = _mm512_set1_ps(k[3]);
          const __m512 c4 = _mm512_set1_ps(k[4]);
          sum0 = _mm512_fmadd_ps(c0, _mm512_loadu_ps(row), sum0);
          sum1 = _mm512_fmadd_ps(c0, _mm512_loadu_ps(row + 16), sum1);
          sum0 = _mm512_fmadd_ps(c1, _mm512_loadu_ps(row + 1), sum0);
          sum1 = _mm512_fmadd_ps(c1, _mm512_loadu_ps(row + 17), sum1);
          sum0 = _mm512_fmadd_ps(c2, _mm512_loadu_ps(row + 2), sum0);
          sum1 = _mm512_fmadd_ps(c2, _mm512_loadu_ps(row + 18), sum1);
          sum0 = _mm512_fmadd_ps(c3, _mm512_loadu_ps(row + 3), sum0);
          sum1 = _mm512_fmadd_ps(c3, _mm512_loadu_ps(row + 19), sum1);
          sum0 = _mm512_fmadd_ps(c4, _mm512_loadu_ps(row + 4), sum0);
          sum1 = _mm512_fmadd_ps(c4, _mm512_loadu_ps(row + 20), sum1);
        }
        _mm512_storeu_ps(out + x, sum0);
        _mm512_storeu_ps(out + x + 16, sum1);
      }
      for (; x + 16 <= interior_end; x += 16) {
        __m512 sum = _mm512_set1_ps(base);
        const float* row0 = in + std::size_t(iy0) * width + x - 2;
        for (int ky = 0; ky < 5; ++ky) {
          const float* row = row0 + std::size_t(ky) * width;
          const float* k = filter + ky * 5;
          sum = _mm512_fmadd_ps(_mm512_set1_ps(k[0]), _mm512_loadu_ps(row), sum);
          sum = _mm512_fmadd_ps(_mm512_set1_ps(k[1]), _mm512_loadu_ps(row + 1), sum);
          sum = _mm512_fmadd_ps(_mm512_set1_ps(k[2]), _mm512_loadu_ps(row + 2), sum);
          sum = _mm512_fmadd_ps(_mm512_set1_ps(k[3]), _mm512_loadu_ps(row + 3), sum);
          sum = _mm512_fmadd_ps(_mm512_set1_ps(k[4]), _mm512_loadu_ps(row + 4), sum);
        }
        _mm512_storeu_ps(out + x, sum);
      }
      for (; x < width; ++x) {
        float sum = base;
        const int ix0 = x - 2;
        for (int ky = 0; ky < 5; ++ky) {
          const int iy = iy0 + ky;
          if (iy < 0 || iy >= height) continue;
          for (int kx = 0; kx < 5; ++kx) {
            const int ix = ix0 + kx;
            if (ix >= 0 && ix < width)
              sum += in[std::size_t(iy) * width + ix] * filter[ky * 5 + kx];
          }
        }
        out[x] = sum;
      }
    }
    const std::size_t row = std::size_t(y) * width;
    int output = 0;
    // 16-OC tiles spilled on Conv.78 (M=16). Eight outputs still cut the
    // 64-ch depthwise row reread from four 4-OC walks to two.
    const int oc_tile = output_channels >= 8 && (output_channels % 8) == 0 ? 8 : 4;
    // Named 8-OC accumulators keep Conv.78's 16-ch pointwise in ZMM. The
    // `sums[16]` array spilled on MSVC /O2. Same-host 8-run was noise, so
    // this stays ENABLE-only. `PPOCR_ENABLE_AVX512_DWPW5_NAMED` turns it on.
    static const bool named8_env =
        std::getenv("PPOCR_ENABLE_AVX512_DWPW5_NAMED") != nullptr &&
        std::getenv("PPOCR_DISABLE_AVX512_DWPW5_NAMED") == nullptr;
    const bool named8 = oc_tile == 8 && named8_env;
    for (; output + oc_tile <= output_channels; output += oc_tile) {
      int x = 0;
      if (named8) {
        const float b0 = pw_bias ? pw_bias[output] : 0.F;
        const float b1 = pw_bias ? pw_bias[output + 1] : 0.F;
        const float b2 = pw_bias ? pw_bias[output + 2] : 0.F;
        const float b3 = pw_bias ? pw_bias[output + 3] : 0.F;
        const float b4 = pw_bias ? pw_bias[output + 4] : 0.F;
        const float b5 = pw_bias ? pw_bias[output + 5] : 0.F;
        const float b6 = pw_bias ? pw_bias[output + 6] : 0.F;
        const float b7 = pw_bias ? pw_bias[output + 7] : 0.F;
        for (; x + 16 <= width; x += 16) {
          __m512 s0 = _mm512_set1_ps(b0), s1 = _mm512_set1_ps(b1);
          __m512 s2 = _mm512_set1_ps(b2), s3 = _mm512_set1_ps(b3);
          __m512 s4 = _mm512_set1_ps(b4), s5 = _mm512_set1_ps(b5);
          __m512 s6 = _mm512_set1_ps(b6), s7 = _mm512_set1_ps(b7);
          for (int input = 0; input < channels; ++input) {
            const __m512 v = _mm512_loadu_ps(dw_row.data() + std::size_t(input) * width + x);
            if (pw8 != pw_weights) {
              const float* w = pw8 + std::size_t(output) * channels + std::size_t(input) * 8;
              s0 = _mm512_fmadd_ps(_mm512_set1_ps(w[0]), v, s0);
              s1 = _mm512_fmadd_ps(_mm512_set1_ps(w[1]), v, s1);
              s2 = _mm512_fmadd_ps(_mm512_set1_ps(w[2]), v, s2);
              s3 = _mm512_fmadd_ps(_mm512_set1_ps(w[3]), v, s3);
              s4 = _mm512_fmadd_ps(_mm512_set1_ps(w[4]), v, s4);
              s5 = _mm512_fmadd_ps(_mm512_set1_ps(w[5]), v, s5);
              s6 = _mm512_fmadd_ps(_mm512_set1_ps(w[6]), v, s6);
              s7 = _mm512_fmadd_ps(_mm512_set1_ps(w[7]), v, s7);
            } else {
              const float* w = pw_weights + std::size_t(output) * channels + input;
              s0 = _mm512_fmadd_ps(_mm512_set1_ps(w[0]), v, s0);
              s1 = _mm512_fmadd_ps(_mm512_set1_ps(w[std::size_t(channels)]), v, s1);
              s2 = _mm512_fmadd_ps(_mm512_set1_ps(w[std::size_t(2) * channels]), v, s2);
              s3 = _mm512_fmadd_ps(_mm512_set1_ps(w[std::size_t(3) * channels]), v, s3);
              s4 = _mm512_fmadd_ps(_mm512_set1_ps(w[std::size_t(4) * channels]), v, s4);
              s5 = _mm512_fmadd_ps(_mm512_set1_ps(w[std::size_t(5) * channels]), v, s5);
              s6 = _mm512_fmadd_ps(_mm512_set1_ps(w[std::size_t(6) * channels]), v, s6);
              s7 = _mm512_fmadd_ps(_mm512_set1_ps(w[std::size_t(7) * channels]), v, s7);
            }
          }
          _mm512_storeu_ps(dst + std::size_t(output) * plane + row + x, activate(s0));
          _mm512_storeu_ps(dst + std::size_t(output + 1) * plane + row + x, activate(s1));
          _mm512_storeu_ps(dst + std::size_t(output + 2) * plane + row + x, activate(s2));
          _mm512_storeu_ps(dst + std::size_t(output + 3) * plane + row + x, activate(s3));
          _mm512_storeu_ps(dst + std::size_t(output + 4) * plane + row + x, activate(s4));
          _mm512_storeu_ps(dst + std::size_t(output + 5) * plane + row + x, activate(s5));
          _mm512_storeu_ps(dst + std::size_t(output + 6) * plane + row + x, activate(s6));
          _mm512_storeu_ps(dst + std::size_t(output + 7) * plane + row + x, activate(s7));
        }
      } else {
        for (; x + 16 <= width; x += 16) {
          __m512 sums[16];
          for (int oc = 0; oc < oc_tile; ++oc)
            sums[oc] = _mm512_set1_ps(pw_bias ? pw_bias[output + oc] : 0.F);
          for (int input = 0; input < channels; ++input) {
            const __m512 v = _mm512_loadu_ps(dw_row.data() + std::size_t(input) * width + x);
            if (pw8 != pw_weights && oc_tile == 8) {
              const float* w = pw8 + std::size_t(output) * channels + std::size_t(input) * 8;
              for (int oc = 0; oc < oc_tile; ++oc)
                sums[oc] = _mm512_fmadd_ps(_mm512_set1_ps(w[oc]), v, sums[oc]);
            } else {
              const float* w = pw_weights + std::size_t(output) * channels + input;
              for (int oc = 0; oc < oc_tile; ++oc)
                sums[oc] = _mm512_fmadd_ps(_mm512_set1_ps(w[std::size_t(oc) * channels]), v, sums[oc]);
            }
          }
          for (int oc = 0; oc < oc_tile; ++oc)
            _mm512_storeu_ps(dst + std::size_t(output + oc) * plane + row + x, activate(sums[oc]));
        }
      }
      for (; x < width; ++x) {
        float sums[16];
        for (int oc = 0; oc < oc_tile; ++oc)
          sums[oc] = pw_bias ? pw_bias[output + oc] : 0.F;
        for (int input = 0; input < channels; ++input) {
          const float v = dw_row[std::size_t(input) * width + x];
          if (pw8 != pw_weights && oc_tile == 8) {
            const float* w = pw8 + std::size_t(output) * channels + std::size_t(input) * 8;
            for (int oc = 0; oc < oc_tile; ++oc) sums[oc] += w[oc] * v;
          } else {
            const float* w = pw_weights + std::size_t(output) * channels + input;
            for (int oc = 0; oc < oc_tile; ++oc)
              sums[oc] += w[std::size_t(oc) * channels] * v;
          }
        }
        for (int oc = 0; oc < oc_tile; ++oc)
          dst[std::size_t(output + oc) * plane + row + x] = activate_scalar(sums[oc]);
      }
    }
    for (; output < output_channels; ++output) {
      const float* filter = pw_weights + std::size_t(output) * channels;
      float* out = dst + std::size_t(output) * plane + row;
      int x = 0;
      for (; x + 16 <= width; x += 16) {
        __m512 sum = _mm512_set1_ps(pw_bias ? pw_bias[output] : 0.F);
        for (int input = 0; input < channels; ++input) {
          sum = _mm512_fmadd_ps(_mm512_set1_ps(filter[input]),
                                _mm512_loadu_ps(dw_row.data() + std::size_t(input) * width + x),
                                sum);
        }
        _mm512_storeu_ps(out + x, activate(sum));
      }
      for (; x < width; ++x) {
        float sum = pw_bias ? pw_bias[output] : 0.F;
        for (int input = 0; input < channels; ++input)
          sum += filter[input] * dw_row[std::size_t(input) * width + x];
        out[x] = activate_scalar(sum);
      }
    }
  }
}

void Avx512ExpandGeluProjectAdd(float* dst, const float* src,
                                const float* expand_weights, const float* expand_bias,
                                const float* project_weights, const float* project_bias,
                                int channels, int hidden, std::size_t plane,
                                std::size_t spatial_begin,
                                std::size_t spatial_end) noexcept;

void Avx512DepthwiseExpandGeluProjectAdd3x3S1(
    float* dst, const float* src, const float* dw_weights, const float* dw_bias,
    const float* expand_weights, const float* expand_bias,
    const float* project_weights, const float* project_bias, int channels,
    int hidden, int height, int width, int y0, int y1) noexcept {
  if (!dst || !src || !dw_weights || !expand_weights || !project_weights ||
      channels <= 0 || hidden <= 0 || height <= 0 || width <= 0) {
    return;
  }
  if (y0 < 0) y0 = 0;
  if (y1 < 0 || y1 > height) y1 = height;
  if (y0 >= y1) return;
  const std::size_t plane = std::size_t(height) * width;
  const int first_y = 1;
  const int first_x = 1;
  const int last_y = std::min(height, height - 1);
  const int last_x = std::min(width, width - 1);
  thread_local std::vector<float> dw_row;
  thread_local std::vector<float> out_row;
  dw_row.resize(std::size_t(channels) * width);
  out_row.resize(std::size_t(channels) * width);
  for (int y = y0; y < y1; ++y) {
    const int iy0 = y - 1;
    const int interior_begin = y >= first_y && y < last_y ? first_x : 0;
    const int interior_end = y >= first_y && y < last_y ? last_x : 0;
    for (int channel = 0; channel < channels; ++channel) {
      const float* in = src + std::size_t(channel) * plane;
      const float* filter = dw_weights + std::size_t(channel) * 9;
      float* out = dw_row.data() + std::size_t(channel) * width;
      const float base = dw_bias ? dw_bias[channel] : 0.F;
      int x = 0;
      for (; x < interior_begin; ++x) {
        float sum = base;
        const int ix0 = x - 1;
        for (int ky = 0; ky < 3; ++ky) {
          const int iy = iy0 + ky;
          if (iy < 0 || iy >= height) continue;
          for (int kx = 0; kx < 3; ++kx) {
            const int ix = ix0 + kx;
            if (ix >= 0 && ix < width) {
              sum += in[std::size_t(iy) * width + ix] * filter[ky * 3 + kx];
            }
          }
        }
        out[x] = sum;
      }
      for (; x + 16 <= interior_end; x += 16) {
        __m512 sum = _mm512_set1_ps(base);
        for (int ky = 0; ky < 3; ++ky) {
          const float* row = in + std::size_t(iy0 + ky) * width + x - 1;
          const float* kernel = filter + ky * 3;
          for (int kx = 0; kx < 3; ++kx) {
            sum = _mm512_fmadd_ps(_mm512_set1_ps(kernel[kx]),
                                  _mm512_loadu_ps(row + kx), sum);
          }
        }
        _mm512_storeu_ps(out + x, sum);
      }
      for (; x < width; ++x) {
        float sum = base;
        const int ix0 = x - 1;
        for (int ky = 0; ky < 3; ++ky) {
          const int iy = iy0 + ky;
          if (iy < 0 || iy >= height) continue;
          for (int kx = 0; kx < 3; ++kx) {
            const int ix = ix0 + kx;
            if (ix >= 0 && ix < width) {
              sum += in[std::size_t(iy) * width + ix] * filter[ky * 3 + kx];
            }
          }
        }
        out[x] = sum;
      }
    }
    Avx512ExpandGeluProjectAdd(out_row.data(), dw_row.data(), expand_weights,
                               expand_bias, project_weights, project_bias, channels,
                               hidden, std::size_t(width), 0, std::size_t(width));
    const std::size_t row = std::size_t(y) * width;
    for (int channel = 0; channel < channels; ++channel) {
      std::memcpy(dst + std::size_t(channel) * plane + row,
                  out_row.data() + std::size_t(channel) * width,
                  std::size_t(width) * sizeof(float));
    }
  }
}

void Avx512SpatialMean(float* dst, const float* src, std::size_t planes,
                       std::size_t spatial) noexcept {
  if (spatial == 0) return;
  const __m512 scale = _mm512_set1_ps(1.F / static_cast<float>(spatial));
  for (std::size_t plane = 0; plane < planes; ++plane) {
    const float* values = src + plane * spatial;
    __m512 accum = _mm512_setzero_ps();
    std::size_t index = 0;
    for (; index + 16 <= spatial; index += 16) accum = _mm512_add_ps(accum, _mm512_loadu_ps(values + index));
    alignas(64) float lanes[16];
    _mm512_store_ps(lanes, accum);
    float sum = 0.F;
    for (float lane : lanes) sum += lane;
    for (; index < spatial; ++index) sum += values[index];
    dst[plane] = _mm512_cvtss_f32(_mm512_mul_ps(_mm512_set1_ps(sum), scale));
  }
}

void Avx512MaxPool2x2Same(float* dst, const float* src, int first_plane,
                          int last_plane, int height, int width) noexcept {
  const std::size_t plane_size = std::size_t(height) * width;
  for (int plane = first_plane; plane < last_plane; ++plane) {
    const float* in = src + std::size_t(plane) * plane_size;
    float* out = dst + std::size_t(plane) * plane_size;
    for (int y = 0; y + 1 < height; ++y) {
      const float* row0 = in + std::size_t(y) * width;
      const float* row1 = row0 + width;
      int x = 0;
      for (; x + 16 <= width - 1; x += 16) {
        const __m512 top = _mm512_max_ps(_mm512_loadu_ps(row0 + x), _mm512_loadu_ps(row0 + x + 1));
        const __m512 bottom = _mm512_max_ps(_mm512_loadu_ps(row1 + x), _mm512_loadu_ps(row1 + x + 1));
        _mm512_storeu_ps(out + std::size_t(y) * width + x, _mm512_max_ps(top, bottom));
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
      for (; x + 16 <= width - 1; x += 16) {
        _mm512_storeu_ps(out_row + x, _mm512_max_ps(_mm512_loadu_ps(row + x), _mm512_loadu_ps(row + x + 1)));
      }
      for (; x + 1 < width; ++x) out_row[x] = std::max(row[x], row[x + 1]);
    }
    out[plane_size - 1] = in[plane_size - 1];
  }
}

void Avx512MaxPool2x2Valid(float* dst, const float* src, int first_plane,
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
      for (; x + 16 <= output_width; x += 16) {
        const __m512 top = _mm512_max_ps(_mm512_loadu_ps(row0 + x), _mm512_loadu_ps(row0 + x + 1));
        const __m512 bottom = _mm512_max_ps(_mm512_loadu_ps(row1 + x), _mm512_loadu_ps(row1 + x + 1));
        _mm512_storeu_ps(row_out + x, _mm512_max_ps(top, bottom));
      }
      for (; x < output_width; ++x) {
        row_out[x] = std::max(std::max(row0[x], row0[x + 1]),
                              std::max(row1[x], row1[x + 1]));
      }
    }
  }
}

void Avx512Conv2d(float* dst, const float* src, const float* weights,
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
        for (int input = 0; input < input_channels; ++input) {
          const float* plane = src + std::size_t(input) * input_plane;
          const float* kernel = filter + std::size_t(input) * filter_plane;
          for (int ky = 0; ky < kernel_h; ++ky) { const int iy = iy0 + ky; if (iy < 0 || iy >= input_h) continue;
            for (int kx = 0; kx < kernel_w; ++kx) { const int ix = ix0 + kx; if (ix >= 0 && ix < input_w) sum += plane[std::size_t(iy) * input_w + ix] * kernel[ky * kernel_w + kx]; }
          }
        }
        out[std::size_t(y) * output_w + x] = sum;
      }
      for (; x + 16 <= interior_end; x += 16) {
        __m512 sum = _mm512_set1_ps(base);
        for (int input = 0; input < input_channels; ++input) {
          const float* plane = src + std::size_t(input) * input_plane + std::size_t(iy0) * input_w + x - pad_left;
          const float* kernel = filter + std::size_t(input) * filter_plane;
          for (int ky = 0; ky < kernel_h; ++ky) {
            const float* row = plane + std::size_t(ky) * input_w;
            const float* krow = kernel + ky * kernel_w;
            for (int kx = 0; kx < kernel_w; ++kx) sum = _mm512_fmadd_ps(_mm512_set1_ps(krow[kx]), _mm512_loadu_ps(row + kx), sum);
          }
        }
        _mm512_storeu_ps(out + std::size_t(y) * output_w + x, sum);
      }
      for (; x < output_w; ++x) {
        float sum = base; const int ix0 = x - pad_left;
        for (int input = 0; input < input_channels; ++input) {
          const float* plane = src + std::size_t(input) * input_plane;
          const float* kernel = filter + std::size_t(input) * filter_plane;
          for (int ky = 0; ky < kernel_h; ++ky) { const int iy = iy0 + ky; if (iy < 0 || iy >= input_h) continue;
            for (int kx = 0; kx < kernel_w; ++kx) { const int ix = ix0 + kx; if (ix >= 0 && ix < input_w) sum += plane[std::size_t(iy) * input_w + ix] * kernel[ky * kernel_w + kx]; }
          }
        }
        out[std::size_t(y) * output_w + x] = sum;
      }
    }
  }
}

// The detector reconstruction head has several 2x2 valid convolutions.  Its
// output pixels map to four contiguous input pixels, making a sixteen-wide
// AVX-512 accumulation substantially cheaper than the generic border-aware
// path.  This keeps the scalar accumulation order per lane unchanged.
void Avx512Conv2x2Valid(float* dst, const float* src, const float* weights,
                        const float* bias, int first_output, int last_output,
                        int input_channels, int input_h, int input_w, bool relu) noexcept {
  const int output_h = input_h - 1;
  const int output_w = input_w - 1;
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(output_h) * output_w;
  constexpr int kLanes = 16;
  for (int output = first_output; output < last_output; ++output) {
    float* out = dst + std::size_t(output) * output_plane;
    const float* filter = weights + std::size_t(output) * input_channels * 4;
    const float base = bias ? bias[output] : 0.F;
    for (int y = 0; y < output_h; ++y) {
      int x = 0;
      for (; x + kLanes <= output_w; x += kLanes) {
        __m512 sum = _mm512_set1_ps(base);
        for (int input = 0; input < input_channels; ++input) {
          const float* row0 = src + std::size_t(input) * input_plane + std::size_t(y) * input_w + x;
          const float* row1 = row0 + input_w;
          const float* k = filter + std::size_t(input) * 4;
          sum = _mm512_fmadd_ps(_mm512_set1_ps(k[0]), _mm512_loadu_ps(row0), sum);
          sum = _mm512_fmadd_ps(_mm512_set1_ps(k[1]), _mm512_loadu_ps(row0 + 1), sum);
          sum = _mm512_fmadd_ps(_mm512_set1_ps(k[2]), _mm512_loadu_ps(row1), sum);
          sum = _mm512_fmadd_ps(_mm512_set1_ps(k[3]), _mm512_loadu_ps(row1 + 1), sum);
        }
        if (relu) sum = _mm512_max_ps(sum, _mm512_setzero_ps());
        _mm512_storeu_ps(out + std::size_t(y) * output_w + x, sum);
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

// Unit-stride 2x2 SAME_UPPER: its H-1 by W-1 interior is a valid 2x2
// convolution, with only the final column, final row and corner requiring
// ONNX's implicit zero padding. The explicit four-FMA body avoids the generic
// dynamic kernel loops on PP-OCRv6's large early detector feature maps.
void Avx512Conv2x2SameUpper(float* dst, const float* src, const float* weights,
                             const float* bias, int first_output, int last_output,
                             int input_channels, int input_h, int input_w,
                             bool relu, int row_begin, int row_end) noexcept {
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = input_plane;
  const int y_begin = row_begin < 0 ? 0 : std::min(input_h, row_begin);
  const int y_end = row_end < 0 ? input_h : std::min(input_h, row_end);
  const __m512 zero = _mm512_setzero_ps();
  int output = first_output;
  // Y-outer eight-output tiles share each pair of source rows across Conv.1
  // (8 OC) and Conv.2 (16 OC). The older OC-outer walk reread every plane
  // per tile. `PPOCR_DISABLE_AVX512_CONV2X2_SAME8` restores four-output tiles.
  static const bool use_same8 =
      std::getenv("PPOCR_DISABLE_AVX512_CONV2X2_SAME8") == nullptr;
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
        for (; x + 16 <= input_w - 1; x += 16) {
          __m512 s0 = _mm512_set1_ps(b[0]), s1 = _mm512_set1_ps(b[1]);
          __m512 s2 = _mm512_set1_ps(b[2]), s3 = _mm512_set1_ps(b[3]);
          __m512 s4 = _mm512_set1_ps(b[4]), s5 = _mm512_set1_ps(b[5]);
          __m512 s6 = _mm512_set1_ps(b[6]), s7 = _mm512_set1_ps(b[7]);
          for (int input = 0; input < input_channels; ++input) {
            const float* row0 = src + std::size_t(input) * input_plane +
                                std::size_t(y) * input_w + x;
            const float* row1 = row0 + input_w;
            const __m512 v00 = _mm512_loadu_ps(row0);
            const __m512 v01 = _mm512_loadu_ps(row0 + 1);
            const __m512 v10 = _mm512_loadu_ps(row1);
            const __m512 v11 = _mm512_loadu_ps(row1 + 1);
            const auto fma4 = [&](__m512& acc, const float* k) {
              acc = _mm512_fmadd_ps(_mm512_set1_ps(k[0]), v00, acc);
              acc = _mm512_fmadd_ps(_mm512_set1_ps(k[1]), v01, acc);
              acc = _mm512_fmadd_ps(_mm512_set1_ps(k[2]), v10, acc);
              acc = _mm512_fmadd_ps(_mm512_set1_ps(k[3]), v11, acc);
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
            s0 = _mm512_max_ps(s0, zero); s1 = _mm512_max_ps(s1, zero);
            s2 = _mm512_max_ps(s2, zero); s3 = _mm512_max_ps(s3, zero);
            s4 = _mm512_max_ps(s4, zero); s5 = _mm512_max_ps(s5, zero);
            s6 = _mm512_max_ps(s6, zero); s7 = _mm512_max_ps(s7, zero);
          }
          const std::size_t index = std::size_t(y) * input_w + x;
          _mm512_storeu_ps(out[0] + index, s0); _mm512_storeu_ps(out[1] + index, s1);
          _mm512_storeu_ps(out[2] + index, s2); _mm512_storeu_ps(out[3] + index, s3);
          _mm512_storeu_ps(out[4] + index, s4); _mm512_storeu_ps(out[5] + index, s5);
          _mm512_storeu_ps(out[6] + index, s6); _mm512_storeu_ps(out[7] + index, s7);
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
  const int four_begin = output;
  for (int y = y_begin; y + 1 < input_h && y < y_end; ++y) {
    for (int oc = four_begin; oc + 4 <= last_output; oc += 4) {
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
    const auto store4 = [&](std::size_t index, float s0, float s1, float s2, float s3) {
      out0[index] = relu ? std::max(s0, 0.F) : s0;
      out1[index] = relu ? std::max(s1, 0.F) : s1;
      out2[index] = relu ? std::max(s2, 0.F) : s2;
      out3[index] = relu ? std::max(s3, 0.F) : s3;
    };
    {
      int x = 0;
      for (; x + 16 <= input_w - 1; x += 16) {
        __m512 s0 = _mm512_set1_ps(b0), s1 = _mm512_set1_ps(b1);
        __m512 s2 = _mm512_set1_ps(b2), s3 = _mm512_set1_ps(b3);
        for (int input = 0; input < input_channels; ++input) {
          const float* row0 = src + std::size_t(input) * input_plane +
                              std::size_t(y) * input_w + x;
          const float* row1 = row0 + input_w;
          const __m512 v00 = _mm512_loadu_ps(row0);
          const __m512 v01 = _mm512_loadu_ps(row0 + 1);
          const __m512 v10 = _mm512_loadu_ps(row1);
          const __m512 v11 = _mm512_loadu_ps(row1 + 1);
          const float* k0 = f0 + std::size_t(input) * 4;
          const float* k1 = f1 + std::size_t(input) * 4;
          const float* k2 = f2 + std::size_t(input) * 4;
          const float* k3 = f3 + std::size_t(input) * 4;
          s0 = _mm512_fmadd_ps(_mm512_set1_ps(k0[0]), v00, s0);
          s0 = _mm512_fmadd_ps(_mm512_set1_ps(k0[1]), v01, s0);
          s0 = _mm512_fmadd_ps(_mm512_set1_ps(k0[2]), v10, s0);
          s0 = _mm512_fmadd_ps(_mm512_set1_ps(k0[3]), v11, s0);
          s1 = _mm512_fmadd_ps(_mm512_set1_ps(k1[0]), v00, s1);
          s1 = _mm512_fmadd_ps(_mm512_set1_ps(k1[1]), v01, s1);
          s1 = _mm512_fmadd_ps(_mm512_set1_ps(k1[2]), v10, s1);
          s1 = _mm512_fmadd_ps(_mm512_set1_ps(k1[3]), v11, s1);
          s2 = _mm512_fmadd_ps(_mm512_set1_ps(k2[0]), v00, s2);
          s2 = _mm512_fmadd_ps(_mm512_set1_ps(k2[1]), v01, s2);
          s2 = _mm512_fmadd_ps(_mm512_set1_ps(k2[2]), v10, s2);
          s2 = _mm512_fmadd_ps(_mm512_set1_ps(k2[3]), v11, s2);
          s3 = _mm512_fmadd_ps(_mm512_set1_ps(k3[0]), v00, s3);
          s3 = _mm512_fmadd_ps(_mm512_set1_ps(k3[1]), v01, s3);
          s3 = _mm512_fmadd_ps(_mm512_set1_ps(k3[2]), v10, s3);
          s3 = _mm512_fmadd_ps(_mm512_set1_ps(k3[3]), v11, s3);
        }
        if (relu) {
          s0 = _mm512_max_ps(s0, zero); s1 = _mm512_max_ps(s1, zero);
          s2 = _mm512_max_ps(s2, zero); s3 = _mm512_max_ps(s3, zero);
        }
        const std::size_t index = std::size_t(y) * input_w + x;
        _mm512_storeu_ps(out0 + index, s0); _mm512_storeu_ps(out1 + index, s1);
        _mm512_storeu_ps(out2 + index, s2); _mm512_storeu_ps(out3 + index, s3);
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
        store4(std::size_t(y) * input_w + x, s0, s1, s2, s3);
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
      store4(std::size_t(y) * input_w + input_w - 1, s0, s1, s2, s3);
    }
    }
  }
  for (int oc = four_begin; oc + 4 <= last_output; oc += 4) {
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
    const auto store4 = [&](std::size_t index, float s0, float s1, float s2, float s3) {
      out0[index] = relu ? std::max(s0, 0.F) : s0;
      out1[index] = relu ? std::max(s1, 0.F) : s1;
      out2[index] = relu ? std::max(s2, 0.F) : s2;
      out3[index] = relu ? std::max(s3, 0.F) : s3;
    };
    if (y_end < input_h) continue;
    const int y = input_h - 1;
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
      store4(std::size_t(y) * input_w + x, s0, s1, s2, s3);
    }
    float s0 = b0, s1 = b1, s2 = b2, s3 = b3;
    for (int input = 0; input < input_channels; ++input) {
      const float value = src[std::size_t(input) * input_plane + input_plane - 1];
      s0 += value * f0[std::size_t(input) * 4];
      s1 += value * f1[std::size_t(input) * 4];
      s2 += value * f2[std::size_t(input) * 4];
      s3 += value * f3[std::size_t(input) * 4];
    }
    store4(input_plane - 1, s0, s1, s2, s3);
  }
  output = four_begin;
  while (output + 4 <= last_output) output += 4;
  for (; output < last_output; ++output) {
    float* out = dst + std::size_t(output) * output_plane;
    const float* filter = weights + std::size_t(output) * input_channels * 4;
    const float base = bias ? bias[output] : 0.F;
    for (int y = y_begin; y + 1 < input_h && y < y_end; ++y) {
      int x = 0;
      for (; x + 16 <= input_w - 1; x += 16) {
        __m512 sum = _mm512_set1_ps(base);
        for (int input = 0; input < input_channels; ++input) {
          const float* row0 = src + std::size_t(input) * input_plane +
                              std::size_t(y) * input_w + x;
          const float* row1 = row0 + input_w;
          const float* k = filter + std::size_t(input) * 4;
          sum = _mm512_fmadd_ps(_mm512_set1_ps(k[0]), _mm512_loadu_ps(row0), sum);
          sum = _mm512_fmadd_ps(_mm512_set1_ps(k[1]), _mm512_loadu_ps(row0 + 1), sum);
          sum = _mm512_fmadd_ps(_mm512_set1_ps(k[2]), _mm512_loadu_ps(row1), sum);
          sum = _mm512_fmadd_ps(_mm512_set1_ps(k[3]), _mm512_loadu_ps(row1 + 1), sum);
        }
        if (relu) sum = _mm512_max_ps(sum, zero);
        _mm512_storeu_ps(out + std::size_t(y) * input_w + x, sum);
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
      // Right-most output sees only the first kernel column.
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
    // Bottom outputs see only the first kernel row.
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

// Four-output version of the valid 2x2 reconstruction convolution. It
// reuses each source vector for four output filters, which is particularly
// effective for the medium detector's 64x32 and 32x64 reconstruction steps.
// The scalar tail delegates to the established one-output kernel, preserving
// output/channel/kernel accumulation order and all odd-channel semantics.
void Avx512Conv2x2Validx4(float* dst, const float* src, const float* weights,
                          const float* bias, int first_output, int last_output,
                          int input_channels, int input_h, int input_w, bool relu) noexcept {
  const int output_h = input_h - 1;
  const int output_w = input_w - 1;
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(output_h) * output_w;
  int output = first_output;
  for (; output + 4 <= last_output; output += 4) {
    float* out0 = dst + std::size_t(output) * output_plane;
    float* out1 = out0 + output_plane;
    float* out2 = out1 + output_plane;
    float* out3 = out2 + output_plane;
    const float* f0 = weights + std::size_t(output) * input_channels * 4;
    const float* f1 = f0 + std::size_t(input_channels) * 4;
    const float* f2 = f1 + std::size_t(input_channels) * 4;
    const float* f3 = f2 + std::size_t(input_channels) * 4;
    const float b0 = bias ? bias[output] : 0.F;
    const float b1 = bias ? bias[output + 1] : 0.F;
    const float b2 = bias ? bias[output + 2] : 0.F;
    const float b3 = bias ? bias[output + 3] : 0.F;
    for (int y = 0; y < output_h; ++y) {
      int x = 0;
      for (; x + 16 <= output_w; x += 16) {
        __m512 s0 = _mm512_set1_ps(b0), s1 = _mm512_set1_ps(b1);
        __m512 s2 = _mm512_set1_ps(b2), s3 = _mm512_set1_ps(b3);
        for (int input = 0; input < input_channels; ++input) {
          const float* row0 = src + std::size_t(input) * input_plane +
                              std::size_t(y) * input_w + x;
          const float* row1 = row0 + input_w;
          const __m512 v00 = _mm512_loadu_ps(row0);
          const __m512 v01 = _mm512_loadu_ps(row0 + 1);
          const __m512 v10 = _mm512_loadu_ps(row1);
          const __m512 v11 = _mm512_loadu_ps(row1 + 1);
          const float* k0 = f0 + std::size_t(input) * 4;
          const float* k1 = f1 + std::size_t(input) * 4;
          const float* k2 = f2 + std::size_t(input) * 4;
          const float* k3 = f3 + std::size_t(input) * 4;
#define PPOCR_FMA_2X2(S, K) \
          S = _mm512_fmadd_ps(_mm512_set1_ps(K[0]), v00, S); \
          S = _mm512_fmadd_ps(_mm512_set1_ps(K[1]), v01, S); \
          S = _mm512_fmadd_ps(_mm512_set1_ps(K[2]), v10, S); \
          S = _mm512_fmadd_ps(_mm512_set1_ps(K[3]), v11, S)
          PPOCR_FMA_2X2(s0, k0); PPOCR_FMA_2X2(s1, k1);
          PPOCR_FMA_2X2(s2, k2); PPOCR_FMA_2X2(s3, k3);
#undef PPOCR_FMA_2X2
        }
        if (relu) {
          const __m512 zero = _mm512_setzero_ps();
          s0 = _mm512_max_ps(s0, zero); s1 = _mm512_max_ps(s1, zero);
          s2 = _mm512_max_ps(s2, zero); s3 = _mm512_max_ps(s3, zero);
        }
        const std::size_t index = std::size_t(y) * output_w + x;
        _mm512_storeu_ps(out0 + index, s0); _mm512_storeu_ps(out1 + index, s1);
        _mm512_storeu_ps(out2 + index, s2); _mm512_storeu_ps(out3 + index, s3);
      }
      for (; x < output_w; ++x) {
        float s0 = b0, s1 = b1, s2 = b2, s3 = b3;
        for (int input = 0; input < input_channels; ++input) {
          const float* row0 = src + std::size_t(input) * input_plane +
                              std::size_t(y) * input_w + x;
          const float* k0 = f0 + std::size_t(input) * 4;
          const float* k1 = f1 + std::size_t(input) * 4;
          const float* k2 = f2 + std::size_t(input) * 4;
          const float* k3 = f3 + std::size_t(input) * 4;
#define PPOCR_ACC_2X2(S, K) S += row0[0] * K[0] + row0[1] * K[1] + row0[input_w] * K[2] + row0[input_w + 1] * K[3]
          PPOCR_ACC_2X2(s0, k0); PPOCR_ACC_2X2(s1, k1);
          PPOCR_ACC_2X2(s2, k2); PPOCR_ACC_2X2(s3, k3);
#undef PPOCR_ACC_2X2
        }
        const std::size_t index = std::size_t(y) * output_w + x;
        out0[index] = relu ? std::max(s0, 0.F) : s0;
        out1[index] = relu ? std::max(s1, 0.F) : s1;
        out2[index] = relu ? std::max(s2, 0.F) : s2;
        out3[index] = relu ? std::max(s3, 0.F) : s3;
      }
    }
  }
  if (output < last_output) {
    Avx512Conv2x2Valid(dst, src, weights, bias, output, last_output,
                        input_channels, input_h, input_w, relu);
  }
}

// 3x3 stride-one detector layers dominate medium.  Calculate four independent
// output channels from each loaded input vector so the input hierarchy is
// traversed once per channel quartet.  Each accumulator keeps its original
// input/channel/kernel order, preserving the model's inference arithmetic.
void Avx512Conv3x3Stride1x4(float* dst, const float* src, const float* weights,
                             const float* bias, int first_output, int last_output,
                             int input_channels, int input_h, int input_w,
                             int output_h, int output_w, int pad_top,
                             int pad_left, bool relu,
                             const float* const* src_planes, int row0,
                             int row1, bool accumulate) noexcept {
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(output_h) * output_w;
  const int row_begin = row0 >= 0 ? row0 : 0;
  const int row_end = row1 >= 0 ? row1 : output_h;
  const auto source_plane = [&](int input) -> const float* {
    return src_planes ? src_planes[input] : src + std::size_t(input) * input_plane;
  };
  const int first_y = std::max(0, pad_top);
  const int first_x = std::max(0, pad_left);
  const int last_y = std::min(output_h, input_h + pad_top - 2);
  const int last_x = std::min(output_w, input_w + pad_left - 2);
  const auto scalar4 = [&](float* out0, float* out1, float* out2, float* out3,
                           const float* f0, const float* f1, const float* f2, const float* f3,
                           float b0, float b1, float b2, float b3, int y, int x) {
    const int iy0 = y - pad_top, ix0 = x - pad_left;
    const auto index = std::size_t(y) * output_w + x;
    float s0 = b0, s1 = b1, s2 = b2, s3 = b3;
    if (accumulate) {
      s0 += out0[index]; s1 += out1[index]; s2 += out2[index]; s3 += out3[index];
    }
    for (int input = 0; input < input_channels; ++input) {
      const float* plane = source_plane(input);
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
    // IC-outer RMW helps huge packed maps but Concat sources live in four
    // separate allocations; spilling four accumulators per input channel
    // there was slower than keeping them in registers.
    const bool block_channels = src_planes == nullptr && input_channels >= 32 &&
        last_x - first_x >= 32;
    for (int y = row_begin; y < row_end; ++y) {
      const int iy0 = y - pad_top;
      const bool interior_y = y >= first_y && y < last_y;
      int x = 0;
      for (; x < output_w; ++x) {
        if (interior_y && x >= first_x && x + 16 <= last_x) break;
        scalar4(out0, out1, out2, out3, f0, f1, f2, f3, b0, b1, b2, b3, y, x);
      }
      if (block_channels && interior_y) {
        const int interior_x = x;
        int interior_last = x;
        for (; interior_last + 16 <= last_x; interior_last += 16) {
        }
        for (int input = 0; input < input_channels; ++input) {
          const float* plane = source_plane(input) + std::size_t(iy0) * input_w - pad_left;
          const float* k0 = f0 + std::size_t(input) * 9;
          const float* k1 = f1 + std::size_t(input) * 9;
          const float* k2 = f2 + std::size_t(input) * 9;
          const float* k3 = f3 + std::size_t(input) * 9;
          const bool last_input = input + 1 == input_channels;
          for (int xx = interior_x; xx + 16 <= last_x; xx += 16) {
            const std::size_t index = std::size_t(y) * output_w + xx;
            __m512 s0, s1, s2, s3;
            if (input == 0) {
              s0 = _mm512_set1_ps(b0); s1 = _mm512_set1_ps(b1);
              s2 = _mm512_set1_ps(b2); s3 = _mm512_set1_ps(b3);
              if (accumulate) {
                s0 = _mm512_add_ps(s0, _mm512_loadu_ps(out0 + index));
                s1 = _mm512_add_ps(s1, _mm512_loadu_ps(out1 + index));
                s2 = _mm512_add_ps(s2, _mm512_loadu_ps(out2 + index));
                s3 = _mm512_add_ps(s3, _mm512_loadu_ps(out3 + index));
              }
            } else {
              s0 = _mm512_loadu_ps(out0 + index); s1 = _mm512_loadu_ps(out1 + index);
              s2 = _mm512_loadu_ps(out2 + index); s3 = _mm512_loadu_ps(out3 + index);
            }
            const float* row = plane + xx;
            for (int ky = 0; ky < 3; ++ky) {
              const float* src_row = row + std::size_t(ky) * input_w;
              for (int kx = 0; kx < 3; ++kx) {
                const __m512 values = _mm512_loadu_ps(src_row + kx);
                const int ki = ky * 3 + kx;
                s0 = _mm512_fmadd_ps(_mm512_set1_ps(k0[ki]), values, s0);
                s1 = _mm512_fmadd_ps(_mm512_set1_ps(k1[ki]), values, s1);
                s2 = _mm512_fmadd_ps(_mm512_set1_ps(k2[ki]), values, s2);
                s3 = _mm512_fmadd_ps(_mm512_set1_ps(k3[ki]), values, s3);
              }
            }
            if (last_input && relu) {
              const __m512 zero = _mm512_setzero_ps();
              s0 = _mm512_max_ps(s0, zero); s1 = _mm512_max_ps(s1, zero);
              s2 = _mm512_max_ps(s2, zero); s3 = _mm512_max_ps(s3, zero);
            }
            _mm512_storeu_ps(out0 + index, s0); _mm512_storeu_ps(out1 + index, s1);
            _mm512_storeu_ps(out2 + index, s2); _mm512_storeu_ps(out3 + index, s3);
          }
        }
        x = interior_last;
      }
      for (; x + 16 <= last_x; x += 16) {
        const auto index = std::size_t(y) * output_w + x;
        __m512 s0 = _mm512_set1_ps(b0), s1 = _mm512_set1_ps(b1);
        __m512 s2 = _mm512_set1_ps(b2), s3 = _mm512_set1_ps(b3);
        if (accumulate) {
          s0 = _mm512_add_ps(s0, _mm512_loadu_ps(out0 + index));
          s1 = _mm512_add_ps(s1, _mm512_loadu_ps(out1 + index));
          s2 = _mm512_add_ps(s2, _mm512_loadu_ps(out2 + index));
          s3 = _mm512_add_ps(s3, _mm512_loadu_ps(out3 + index));
        }
        for (int input = 0; input < input_channels; ++input) {
          const float* plane = source_plane(input) + std::size_t(iy0) * input_w + x - pad_left;
          const float* k0 = f0 + std::size_t(input) * 9;
          const float* k1 = f1 + std::size_t(input) * 9;
          const float* k2 = f2 + std::size_t(input) * 9;
          const float* k3 = f3 + std::size_t(input) * 9;
          for (int ky = 0; ky < 3; ++ky) {
            const float* row = plane + std::size_t(ky) * input_w;
            for (int kx = 0; kx < 3; ++kx) {
              const __m512 values = _mm512_loadu_ps(row + kx);
              const int ki = ky * 3 + kx;
              s0 = _mm512_fmadd_ps(_mm512_set1_ps(k0[ki]), values, s0);
              s1 = _mm512_fmadd_ps(_mm512_set1_ps(k1[ki]), values, s1);
              s2 = _mm512_fmadd_ps(_mm512_set1_ps(k2[ki]), values, s2);
              s3 = _mm512_fmadd_ps(_mm512_set1_ps(k3[ki]), values, s3);
            }
          }
        }
        if (relu) { const __m512 zero = _mm512_setzero_ps(); s0 = _mm512_max_ps(s0, zero); s1 = _mm512_max_ps(s1, zero); s2 = _mm512_max_ps(s2, zero); s3 = _mm512_max_ps(s3, zero); }
        _mm512_storeu_ps(out0 + index, s0); _mm512_storeu_ps(out1 + index, s1);
        _mm512_storeu_ps(out2 + index, s2); _mm512_storeu_ps(out3 + index, s3);
      }
      // SAME 3x3 on 40x176 leaves ~15 right-edge pixels on the scalar path.
      // One masked 16-wide covers them; masked-off lanes do not fault.
      // `PPOCR_DISABLE_AVX512_MASK_TAIL` restores the scalar remainder.
      if (Avx512MaskTailEnabled() && interior_y && x < output_w &&
          output_w - x <= 16) {
        const int remain = output_w - x;
        const __mmask16 omask = Avx512CountMask(remain);
        const auto index = std::size_t(y) * output_w + x;
        __m512 s0 = _mm512_set1_ps(b0), s1 = _mm512_set1_ps(b1);
        __m512 s2 = _mm512_set1_ps(b2), s3 = _mm512_set1_ps(b3);
        if (accumulate) {
          s0 = _mm512_add_ps(s0, _mm512_maskz_loadu_ps(omask, out0 + index));
          s1 = _mm512_add_ps(s1, _mm512_maskz_loadu_ps(omask, out1 + index));
          s2 = _mm512_add_ps(s2, _mm512_maskz_loadu_ps(omask, out2 + index));
          s3 = _mm512_add_ps(s3, _mm512_maskz_loadu_ps(omask, out3 + index));
        }
        for (int input = 0; input < input_channels; ++input) {
          const float* plane = source_plane(input) + std::size_t(iy0) * input_w +
                               x - pad_left;
          const float* k0 = f0 + std::size_t(input) * 9;
          const float* k1 = f1 + std::size_t(input) * 9;
          const float* k2 = f2 + std::size_t(input) * 9;
          const float* k3 = f3 + std::size_t(input) * 9;
          for (int ky = 0; ky < 3; ++ky) {
            const float* row = plane + std::size_t(ky) * input_w;
            for (int kx = 0; kx < 3; ++kx) {
              const __mmask16 kmask =
                  Avx512BoundMask(x - pad_left + kx, input_w, omask);
              const __m512 values = _mm512_maskz_loadu_ps(kmask, row + kx);
              const int ki = ky * 3 + kx;
              s0 = _mm512_fmadd_ps(_mm512_set1_ps(k0[ki]), values, s0);
              s1 = _mm512_fmadd_ps(_mm512_set1_ps(k1[ki]), values, s1);
              s2 = _mm512_fmadd_ps(_mm512_set1_ps(k2[ki]), values, s2);
              s3 = _mm512_fmadd_ps(_mm512_set1_ps(k3[ki]), values, s3);
            }
          }
        }
        if (relu) {
          const __m512 zero = _mm512_setzero_ps();
          s0 = _mm512_max_ps(s0, zero); s1 = _mm512_max_ps(s1, zero);
          s2 = _mm512_max_ps(s2, zero); s3 = _mm512_max_ps(s3, zero);
        }
        _mm512_mask_storeu_ps(out0 + index, omask, s0);
        _mm512_mask_storeu_ps(out1 + index, omask, s1);
        _mm512_mask_storeu_ps(out2 + index, omask, s2);
        _mm512_mask_storeu_ps(out3 + index, omask, s3);
        x = output_w;
      }
      for (; x < output_w; ++x) scalar4(out0, out1, out2, out3, f0, f1, f2, f3,
                                          b0, b1, b2, b3, y, x);
    }
  }
  if (output < last_output) {
    if (src_planes) {
      for (int channel = output; channel < last_output; ++channel) {
        float* out = dst + std::size_t(channel) * output_plane;
        const float* filter = weights + std::size_t(channel) * input_channels * 9;
        const float base = bias ? bias[channel] : 0.F;
        for (int y = row_begin; y < row_end; ++y) {
          for (int x = 0; x < output_w; ++x) {
            const int iy0 = y - pad_top, ix0 = x - pad_left;
            const auto index = std::size_t(y) * output_w + x;
            float sum = base;
            if (accumulate) sum += out[index];
            for (int input = 0; input < input_channels; ++input) {
              const float* plane = source_plane(input);
              const float* kernel = filter + std::size_t(input) * 9;
              for (int ky = 0; ky < 3; ++ky) {
                const int iy = iy0 + ky;
                if (iy < 0 || iy >= input_h) continue;
                for (int kx = 0; kx < 3; ++kx) {
                  const int ix = ix0 + kx;
                  if (ix >= 0 && ix < input_w)
                    sum += plane[std::size_t(iy) * input_w + ix] * kernel[ky * 3 + kx];
                }
              }
            }
            out[index] = relu ? std::max(sum, 0.F) : sum;
          }
        }
      }
    } else {
      Avx512Conv2d(dst, src, weights, bias, output, last_output, input_channels,
                   input_h, input_w, output_h, output_w, 3, 3, pad_top, pad_left);
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
}

// Concat.2 16-ch source-hot 16→16: one 16-OC walk keeps the 16 input planes
// in L2 instead of rereading them across four 4-OC tiles. Row range and
// accumulate match Avx512Conv3x3Stride1x4 so ConcatChannelConv2d can swap.
void Avx512Conv3x3Stride1x16(float* dst, const float* src, const float* weights,
                              const float* bias, int first_output, int last_output,
                              int input_channels, int input_h, int input_w,
                              int output_h, int output_w, int pad_top,
                              int pad_left, bool relu, int row0, int row1,
                              bool accumulate, const float* tap_pack) noexcept {
  if (last_output - first_output < 16) {
    Avx512Conv3x3Stride1x4(dst, src, weights, bias, first_output, last_output,
                           input_channels, input_h, input_w, output_h, output_w,
                           pad_top, pad_left, relu, nullptr, row0, row1,
                           accumulate);
    return;
  }
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(output_h) * output_w;
  const int row_begin = row0 >= 0 ? row0 : 0;
  const int row_end = row1 >= 0 ? row1 : output_h;
  const int first_y = std::max(0, pad_top);
  const int first_x = std::max(0, pad_left);
  const int last_y = std::min(output_h, input_h + pad_top - 2);
  const int last_x = std::min(output_w, input_w + pad_left - 2);
  const auto scalar16 = [&](float* const out[16], const float* const filter[16],
                            const float base[16], int y, int x) {
    const int iy0 = y - pad_top, ix0 = x - pad_left;
    const auto index = std::size_t(y) * output_w + x;
    float sum[16];
    for (int q = 0; q < 16; ++q) {
      sum[q] = base[q];
      if (accumulate) sum[q] += out[q][index];
    }
    for (int input = 0; input < input_channels; ++input) {
      const float* plane = src + std::size_t(input) * input_plane;
      for (int ky = 0; ky < 3; ++ky) {
        const int iy = iy0 + ky;
        if (iy < 0 || iy >= input_h) continue;
        for (int kx = 0; kx < 3; ++kx) {
          const int ix = ix0 + kx;
          if (ix < 0 || ix >= input_w) continue;
          const float value = plane[std::size_t(iy) * input_w + ix];
          const int ki = input * 9 + ky * 3 + kx;
          for (int q = 0; q < 16; ++q) sum[q] += value * filter[q][ki];
        }
      }
    }
    for (int q = 0; q < 16; ++q)
      out[q][index] = relu ? std::max(sum[q], 0.F) : sum[q];
  };
  int output = first_output;
  for (; output + 16 <= last_output; output += 16) {
    float* out[16];
    const float* filter[16];
    float base[16];
    for (int q = 0; q < 16; ++q) {
      out[q] = dst + std::size_t(output + q) * output_plane;
      filter[q] = weights + std::size_t(output + q) * input_channels * 9;
      base[q] = bias ? bias[output + q] : 0.F;
    }
    static const bool named =
        std::getenv("PPOCR_DISABLE_AVX512_CONCAT_NAMED") == nullptr;
    // Same-host 8-run missed (17.35 vs 16.43). 32 named accumulators spill
    // on MSVC /O2. Keep ENABLE-only.
    static const bool dual_row =
        std::getenv("PPOCR_ENABLE_AVX512_CONCAT_DUALROW") != nullptr &&
        std::getenv("PPOCR_DISABLE_AVX512_CONCAT_DUALROW") == nullptr;
    const auto apply16 = [&](__m512& s0, __m512& s1, __m512& s2, __m512& s3,
                             __m512& s4, __m512& s5, __m512& s6, __m512& s7,
                             __m512& s8, __m512& s9, __m512& sa, __m512& sb,
                             __m512& sc, __m512& sd, __m512& se, __m512& sf,
                             const __m512& value, const float* tw, int ki) {
      s0 = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[0] : filter[0][ki]), value, s0);
      s1 = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[1] : filter[1][ki]), value, s1);
      s2 = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[2] : filter[2][ki]), value, s2);
      s3 = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[3] : filter[3][ki]), value, s3);
      s4 = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[4] : filter[4][ki]), value, s4);
      s5 = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[5] : filter[5][ki]), value, s5);
      s6 = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[6] : filter[6][ki]), value, s6);
      s7 = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[7] : filter[7][ki]), value, s7);
      s8 = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[8] : filter[8][ki]), value, s8);
      s9 = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[9] : filter[9][ki]), value, s9);
      sa = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[10] : filter[10][ki]), value, sa);
      sb = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[11] : filter[11][ki]), value, sb);
      sc = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[12] : filter[12][ki]), value, sc);
      sd = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[13] : filter[13][ki]), value, sd);
      se = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[14] : filter[14][ki]), value, se);
      sf = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[15] : filter[15][ki]), value, sf);
    };
    int y = row_begin;
    if (named && dual_row && row_end == row_begin + 2 &&
        row_begin >= first_y && row_begin + 1 < last_y) {
      const int iy0 = row_begin - pad_top;
      int x = 0;
      for (; x < output_w; ++x) {
        if (x >= first_x && x + 16 <= last_x) break;
        scalar16(out, filter, base, row_begin, x);
        scalar16(out, filter, base, row_begin + 1, x);
      }
      for (; x + 16 <= last_x; x += 16) {
        const auto index0 = std::size_t(row_begin) * output_w + x;
        const auto index1 = index0 + output_w;
        __m512 a0 = _mm512_set1_ps(base[0]), a1 = _mm512_set1_ps(base[1]);
        __m512 a2 = _mm512_set1_ps(base[2]), a3 = _mm512_set1_ps(base[3]);
        __m512 a4 = _mm512_set1_ps(base[4]), a5 = _mm512_set1_ps(base[5]);
        __m512 a6 = _mm512_set1_ps(base[6]), a7 = _mm512_set1_ps(base[7]);
        __m512 a8 = _mm512_set1_ps(base[8]), a9 = _mm512_set1_ps(base[9]);
        __m512 aa = _mm512_set1_ps(base[10]), ab = _mm512_set1_ps(base[11]);
        __m512 ac = _mm512_set1_ps(base[12]), ad = _mm512_set1_ps(base[13]);
        __m512 ae = _mm512_set1_ps(base[14]), af = _mm512_set1_ps(base[15]);
        __m512 b0 = a0, b1 = a1, b2 = a2, b3 = a3, b4 = a4, b5 = a5, b6 = a6, b7 = a7;
        __m512 b8 = a8, b9 = a9, ba = aa, bb = ab, bc = ac, bd = ad, be = ae, bf = af;
        if (accumulate) {
          a0 = _mm512_add_ps(a0, _mm512_loadu_ps(out[0] + index0));
          a1 = _mm512_add_ps(a1, _mm512_loadu_ps(out[1] + index0));
          a2 = _mm512_add_ps(a2, _mm512_loadu_ps(out[2] + index0));
          a3 = _mm512_add_ps(a3, _mm512_loadu_ps(out[3] + index0));
          a4 = _mm512_add_ps(a4, _mm512_loadu_ps(out[4] + index0));
          a5 = _mm512_add_ps(a5, _mm512_loadu_ps(out[5] + index0));
          a6 = _mm512_add_ps(a6, _mm512_loadu_ps(out[6] + index0));
          a7 = _mm512_add_ps(a7, _mm512_loadu_ps(out[7] + index0));
          a8 = _mm512_add_ps(a8, _mm512_loadu_ps(out[8] + index0));
          a9 = _mm512_add_ps(a9, _mm512_loadu_ps(out[9] + index0));
          aa = _mm512_add_ps(aa, _mm512_loadu_ps(out[10] + index0));
          ab = _mm512_add_ps(ab, _mm512_loadu_ps(out[11] + index0));
          ac = _mm512_add_ps(ac, _mm512_loadu_ps(out[12] + index0));
          ad = _mm512_add_ps(ad, _mm512_loadu_ps(out[13] + index0));
          ae = _mm512_add_ps(ae, _mm512_loadu_ps(out[14] + index0));
          af = _mm512_add_ps(af, _mm512_loadu_ps(out[15] + index0));
          b0 = _mm512_add_ps(b0, _mm512_loadu_ps(out[0] + index1));
          b1 = _mm512_add_ps(b1, _mm512_loadu_ps(out[1] + index1));
          b2 = _mm512_add_ps(b2, _mm512_loadu_ps(out[2] + index1));
          b3 = _mm512_add_ps(b3, _mm512_loadu_ps(out[3] + index1));
          b4 = _mm512_add_ps(b4, _mm512_loadu_ps(out[4] + index1));
          b5 = _mm512_add_ps(b5, _mm512_loadu_ps(out[5] + index1));
          b6 = _mm512_add_ps(b6, _mm512_loadu_ps(out[6] + index1));
          b7 = _mm512_add_ps(b7, _mm512_loadu_ps(out[7] + index1));
          b8 = _mm512_add_ps(b8, _mm512_loadu_ps(out[8] + index1));
          b9 = _mm512_add_ps(b9, _mm512_loadu_ps(out[9] + index1));
          ba = _mm512_add_ps(ba, _mm512_loadu_ps(out[10] + index1));
          bb = _mm512_add_ps(bb, _mm512_loadu_ps(out[11] + index1));
          bc = _mm512_add_ps(bc, _mm512_loadu_ps(out[12] + index1));
          bd = _mm512_add_ps(bd, _mm512_loadu_ps(out[13] + index1));
          be = _mm512_add_ps(be, _mm512_loadu_ps(out[14] + index1));
          bf = _mm512_add_ps(bf, _mm512_loadu_ps(out[15] + index1));
        }
        for (int input = 0; input < input_channels; ++input) {
          const float* plane = src + std::size_t(input) * input_plane +
                               std::size_t(iy0) * input_w + x - pad_left;
          const float* r0 = plane;
          const float* r1 = plane + input_w;
          const float* r2 = r1 + input_w;
          const float* r3 = r2 + input_w;
          for (int kx = 0; kx < 3; ++kx) {
            const __m512 v0 = _mm512_loadu_ps(r0 + kx);
            const __m512 v1 = _mm512_loadu_ps(r1 + kx);
            const __m512 v2 = _mm512_loadu_ps(r2 + kx);
            const __m512 v3 = _mm512_loadu_ps(r3 + kx);
            const int k0 = input * 9 + kx;
            const int k1 = k0 + 3;
            const int k2 = k0 + 6;
            const float* tw0 = tap_pack ? tap_pack + std::size_t(k0) * 16 : nullptr;
            const float* tw1 = tap_pack ? tap_pack + std::size_t(k1) * 16 : nullptr;
            const float* tw2 = tap_pack ? tap_pack + std::size_t(k2) * 16 : nullptr;
            apply16(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af, v0, tw0, k0);
            apply16(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af, v1, tw1, k1);
            apply16(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af, v2, tw2, k2);
            apply16(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, ba, bb, bc, bd, be, bf, v1, tw0, k0);
            apply16(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, ba, bb, bc, bd, be, bf, v2, tw1, k1);
            apply16(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, ba, bb, bc, bd, be, bf, v3, tw2, k2);
          }
        }
        if (relu) {
          const __m512 zero = _mm512_setzero_ps();
          a0 = _mm512_max_ps(a0, zero); a1 = _mm512_max_ps(a1, zero);
          a2 = _mm512_max_ps(a2, zero); a3 = _mm512_max_ps(a3, zero);
          a4 = _mm512_max_ps(a4, zero); a5 = _mm512_max_ps(a5, zero);
          a6 = _mm512_max_ps(a6, zero); a7 = _mm512_max_ps(a7, zero);
          a8 = _mm512_max_ps(a8, zero); a9 = _mm512_max_ps(a9, zero);
          aa = _mm512_max_ps(aa, zero); ab = _mm512_max_ps(ab, zero);
          ac = _mm512_max_ps(ac, zero); ad = _mm512_max_ps(ad, zero);
          ae = _mm512_max_ps(ae, zero); af = _mm512_max_ps(af, zero);
          b0 = _mm512_max_ps(b0, zero); b1 = _mm512_max_ps(b1, zero);
          b2 = _mm512_max_ps(b2, zero); b3 = _mm512_max_ps(b3, zero);
          b4 = _mm512_max_ps(b4, zero); b5 = _mm512_max_ps(b5, zero);
          b6 = _mm512_max_ps(b6, zero); b7 = _mm512_max_ps(b7, zero);
          b8 = _mm512_max_ps(b8, zero); b9 = _mm512_max_ps(b9, zero);
          ba = _mm512_max_ps(ba, zero); bb = _mm512_max_ps(bb, zero);
          bc = _mm512_max_ps(bc, zero); bd = _mm512_max_ps(bd, zero);
          be = _mm512_max_ps(be, zero); bf = _mm512_max_ps(bf, zero);
        }
        _mm512_storeu_ps(out[0] + index0, a0); _mm512_storeu_ps(out[1] + index0, a1);
        _mm512_storeu_ps(out[2] + index0, a2); _mm512_storeu_ps(out[3] + index0, a3);
        _mm512_storeu_ps(out[4] + index0, a4); _mm512_storeu_ps(out[5] + index0, a5);
        _mm512_storeu_ps(out[6] + index0, a6); _mm512_storeu_ps(out[7] + index0, a7);
        _mm512_storeu_ps(out[8] + index0, a8); _mm512_storeu_ps(out[9] + index0, a9);
        _mm512_storeu_ps(out[10] + index0, aa); _mm512_storeu_ps(out[11] + index0, ab);
        _mm512_storeu_ps(out[12] + index0, ac); _mm512_storeu_ps(out[13] + index0, ad);
        _mm512_storeu_ps(out[14] + index0, ae); _mm512_storeu_ps(out[15] + index0, af);
        _mm512_storeu_ps(out[0] + index1, b0); _mm512_storeu_ps(out[1] + index1, b1);
        _mm512_storeu_ps(out[2] + index1, b2); _mm512_storeu_ps(out[3] + index1, b3);
        _mm512_storeu_ps(out[4] + index1, b4); _mm512_storeu_ps(out[5] + index1, b5);
        _mm512_storeu_ps(out[6] + index1, b6); _mm512_storeu_ps(out[7] + index1, b7);
        _mm512_storeu_ps(out[8] + index1, b8); _mm512_storeu_ps(out[9] + index1, b9);
        _mm512_storeu_ps(out[10] + index1, ba); _mm512_storeu_ps(out[11] + index1, bb);
        _mm512_storeu_ps(out[12] + index1, bc); _mm512_storeu_ps(out[13] + index1, bd);
        _mm512_storeu_ps(out[14] + index1, be); _mm512_storeu_ps(out[15] + index1, bf);
      }
      for (; x < output_w; ++x) {
        scalar16(out, filter, base, row_begin, x);
        scalar16(out, filter, base, row_begin + 1, x);
      }
      y = row_end;
    }
    for (; y < row_end; ++y) {
      const int iy0 = y - pad_top;
      const bool interior_y = y >= first_y && y < last_y;
      int x = 0;
      for (; x < output_w; ++x) {
        if (interior_y && x >= first_x && x + 16 <= last_x) break;
        scalar16(out, filter, base, y, x);
      }
      for (; x + 16 <= last_x; x += 16) {
        const auto index = std::size_t(y) * output_w + x;
        // Named 16-OC accumulators keep Concat.2 in the AVX-512 file. The
        // `acc[16]` form spilled on MSVC /O2 (stack RMW every 3x3 tap).
        // `PPOCR_DISABLE_AVX512_CONCAT_NAMED` restores the array loop.
        static const bool named =
            std::getenv("PPOCR_DISABLE_AVX512_CONCAT_NAMED") == nullptr;
        if (named) {
          __m512 s0 = _mm512_set1_ps(base[0]), s1 = _mm512_set1_ps(base[1]);
          __m512 s2 = _mm512_set1_ps(base[2]), s3 = _mm512_set1_ps(base[3]);
          __m512 s4 = _mm512_set1_ps(base[4]), s5 = _mm512_set1_ps(base[5]);
          __m512 s6 = _mm512_set1_ps(base[6]), s7 = _mm512_set1_ps(base[7]);
          __m512 s8 = _mm512_set1_ps(base[8]), s9 = _mm512_set1_ps(base[9]);
          __m512 sa = _mm512_set1_ps(base[10]), sb = _mm512_set1_ps(base[11]);
          __m512 sc = _mm512_set1_ps(base[12]), sd = _mm512_set1_ps(base[13]);
          __m512 se = _mm512_set1_ps(base[14]), sf = _mm512_set1_ps(base[15]);
          if (accumulate) {
            s0 = _mm512_add_ps(s0, _mm512_loadu_ps(out[0] + index));
            s1 = _mm512_add_ps(s1, _mm512_loadu_ps(out[1] + index));
            s2 = _mm512_add_ps(s2, _mm512_loadu_ps(out[2] + index));
            s3 = _mm512_add_ps(s3, _mm512_loadu_ps(out[3] + index));
            s4 = _mm512_add_ps(s4, _mm512_loadu_ps(out[4] + index));
            s5 = _mm512_add_ps(s5, _mm512_loadu_ps(out[5] + index));
            s6 = _mm512_add_ps(s6, _mm512_loadu_ps(out[6] + index));
            s7 = _mm512_add_ps(s7, _mm512_loadu_ps(out[7] + index));
            s8 = _mm512_add_ps(s8, _mm512_loadu_ps(out[8] + index));
            s9 = _mm512_add_ps(s9, _mm512_loadu_ps(out[9] + index));
            sa = _mm512_add_ps(sa, _mm512_loadu_ps(out[10] + index));
            sb = _mm512_add_ps(sb, _mm512_loadu_ps(out[11] + index));
            sc = _mm512_add_ps(sc, _mm512_loadu_ps(out[12] + index));
            sd = _mm512_add_ps(sd, _mm512_loadu_ps(out[13] + index));
            se = _mm512_add_ps(se, _mm512_loadu_ps(out[14] + index));
            sf = _mm512_add_ps(sf, _mm512_loadu_ps(out[15] + index));
          }
          for (int input = 0; input < input_channels; ++input) {
            const float* plane = src + std::size_t(input) * input_plane +
                                 std::size_t(iy0) * input_w + x - pad_left;
            for (int ky = 0; ky < 3; ++ky) {
              const float* row = plane + std::size_t(ky) * input_w;
              for (int kx = 0; kx < 3; ++kx) {
                const __m512 value = _mm512_loadu_ps(row + kx);
                const int ki = input * 9 + ky * 3 + kx;
                const float* tw = tap_pack
                    ? tap_pack + std::size_t(ki) * 16
                    : nullptr;
                s0 = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[0] : filter[0][ki]), value, s0);
                s1 = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[1] : filter[1][ki]), value, s1);
                s2 = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[2] : filter[2][ki]), value, s2);
                s3 = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[3] : filter[3][ki]), value, s3);
                s4 = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[4] : filter[4][ki]), value, s4);
                s5 = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[5] : filter[5][ki]), value, s5);
                s6 = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[6] : filter[6][ki]), value, s6);
                s7 = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[7] : filter[7][ki]), value, s7);
                s8 = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[8] : filter[8][ki]), value, s8);
                s9 = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[9] : filter[9][ki]), value, s9);
                sa = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[10] : filter[10][ki]), value, sa);
                sb = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[11] : filter[11][ki]), value, sb);
                sc = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[12] : filter[12][ki]), value, sc);
                sd = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[13] : filter[13][ki]), value, sd);
                se = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[14] : filter[14][ki]), value, se);
                sf = _mm512_fmadd_ps(_mm512_set1_ps(tw ? tw[15] : filter[15][ki]), value, sf);
              }
            }
          }
          if (relu) {
            const __m512 zero = _mm512_setzero_ps();
            s0 = _mm512_max_ps(s0, zero); s1 = _mm512_max_ps(s1, zero);
            s2 = _mm512_max_ps(s2, zero); s3 = _mm512_max_ps(s3, zero);
            s4 = _mm512_max_ps(s4, zero); s5 = _mm512_max_ps(s5, zero);
            s6 = _mm512_max_ps(s6, zero); s7 = _mm512_max_ps(s7, zero);
            s8 = _mm512_max_ps(s8, zero); s9 = _mm512_max_ps(s9, zero);
            sa = _mm512_max_ps(sa, zero); sb = _mm512_max_ps(sb, zero);
            sc = _mm512_max_ps(sc, zero); sd = _mm512_max_ps(sd, zero);
            se = _mm512_max_ps(se, zero); sf = _mm512_max_ps(sf, zero);
          }
          _mm512_storeu_ps(out[0] + index, s0); _mm512_storeu_ps(out[1] + index, s1);
          _mm512_storeu_ps(out[2] + index, s2); _mm512_storeu_ps(out[3] + index, s3);
          _mm512_storeu_ps(out[4] + index, s4); _mm512_storeu_ps(out[5] + index, s5);
          _mm512_storeu_ps(out[6] + index, s6); _mm512_storeu_ps(out[7] + index, s7);
          _mm512_storeu_ps(out[8] + index, s8); _mm512_storeu_ps(out[9] + index, s9);
          _mm512_storeu_ps(out[10] + index, sa); _mm512_storeu_ps(out[11] + index, sb);
          _mm512_storeu_ps(out[12] + index, sc); _mm512_storeu_ps(out[13] + index, sd);
          _mm512_storeu_ps(out[14] + index, se); _mm512_storeu_ps(out[15] + index, sf);
        } else {
          __m512 acc[16];
          for (int q = 0; q < 16; ++q) {
            acc[q] = _mm512_set1_ps(base[q]);
            if (accumulate) acc[q] = _mm512_add_ps(acc[q], _mm512_loadu_ps(out[q] + index));
          }
          for (int input = 0; input < input_channels; ++input) {
            const float* plane = src + std::size_t(input) * input_plane +
                                 std::size_t(iy0) * input_w + x - pad_left;
            for (int ky = 0; ky < 3; ++ky) {
              const float* row = plane + std::size_t(ky) * input_w;
              for (int kx = 0; kx < 3; ++kx) {
                const __m512 value = _mm512_loadu_ps(row + kx);
                const int ki = input * 9 + ky * 3 + kx;
                for (int q = 0; q < 16; ++q)
                  acc[q] = _mm512_fmadd_ps(_mm512_set1_ps(filter[q][ki]), value, acc[q]);
              }
            }
          }
          if (relu) {
            const __m512 zero = _mm512_setzero_ps();
            for (int q = 0; q < 16; ++q) acc[q] = _mm512_max_ps(acc[q], zero);
          }
          for (int q = 0; q < 16; ++q) _mm512_storeu_ps(out[q] + index, acc[q]);
        }
      }
      if (Avx512MaskTailEnabled() && interior_y && x < output_w &&
          output_w - x <= 16) {
        const int remain = output_w - x;
        const __mmask16 omask = Avx512CountMask(remain);
        const auto index = std::size_t(y) * output_w + x;
        __m512 acc[16];
        for (int q = 0; q < 16; ++q) {
          acc[q] = _mm512_set1_ps(base[q]);
          if (accumulate)
            acc[q] = _mm512_add_ps(acc[q], _mm512_maskz_loadu_ps(omask, out[q] + index));
        }
        for (int input = 0; input < input_channels; ++input) {
          const float* plane = src + std::size_t(input) * input_plane +
                               std::size_t(iy0) * input_w + x - pad_left;
          for (int ky = 0; ky < 3; ++ky) {
            const float* row = plane + std::size_t(ky) * input_w;
            for (int kx = 0; kx < 3; ++kx) {
              const __mmask16 kmask =
                  Avx512BoundMask(x - pad_left + kx, input_w, omask);
              const __m512 value = _mm512_maskz_loadu_ps(kmask, row + kx);
              const int ki = input * 9 + ky * 3 + kx;
              for (int q = 0; q < 16; ++q)
                acc[q] = _mm512_fmadd_ps(_mm512_set1_ps(filter[q][ki]), value, acc[q]);
            }
          }
        }
        if (relu) {
          const __m512 zero = _mm512_setzero_ps();
          for (int q = 0; q < 16; ++q) acc[q] = _mm512_max_ps(acc[q], zero);
        }
        for (int q = 0; q < 16; ++q)
          _mm512_mask_storeu_ps(out[q] + index, omask, acc[q]);
        x = output_w;
      }
      for (; x < output_w; ++x) scalar16(out, filter, base, y, x);
    }
  }
  if (output < last_output) {
    Avx512Conv3x3Stride1x4(dst, src, weights, bias, output, last_output,
                           input_channels, input_h, input_w, output_h, output_w,
                           pad_top, pad_left, relu, nullptr, row0, row1,
                           accumulate);
  }
}

// Concat.2 is four 16-ch sources. Walking each source as its own 16→16 3x3
// RMW'd the 16x40x176 destination four times. One output-channel tile that
// folds every source into registers writes dst once (and fuses ReLU).
void Avx512ConcatSourcesConv3x3x4(float* dst, const float* const* sources,
                                  int source_count, const float* wpacks,
                                  const float* bias, int ic, int oc, int input_h,
                                  int input_w, int output_h, int output_w,
                                  int pad_top, int pad_left, bool relu, int row0,
                                  int row1) noexcept {
  if (!dst || !sources || !wpacks || source_count <= 0 || ic <= 0 || oc <= 0)
    return;
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(output_h) * output_w;
  const std::size_t w_src = std::size_t(oc) * ic * 9;
  const int row_begin = row0 >= 0 ? row0 : 0;
  const int row_end = row1 >= 0 ? row1 : output_h;
  const int first_y = std::max(0, pad_top);
  const int first_x = std::max(0, pad_left);
  const int last_y = std::min(output_h, input_h + pad_top - 2);
  const int last_x = std::min(output_w, input_w + pad_left - 2);
  const auto scalar4 = [&](float* out0, float* out1, float* out2, float* out3,
                           int output, float b0, float b1, float b2, float b3,
                           int y, int x) {
    const int iy0 = y - pad_top, ix0 = x - pad_left;
    const auto index = std::size_t(y) * output_w + x;
    float s0 = b0, s1 = b1, s2 = b2, s3 = b3;
    for (int source = 0; source < source_count; ++source) {
      const float* src = sources[source];
      if (!src) continue;
      const float* f0 = wpacks + std::size_t(source) * w_src +
          std::size_t(output) * ic * 9;
      const float* f1 = f0 + std::size_t(ic) * 9;
      const float* f2 = f1 + std::size_t(ic) * 9;
      const float* f3 = f2 + std::size_t(ic) * 9;
      for (int input = 0; input < ic; ++input) {
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
    }
    out0[index] = relu ? std::max(s0, 0.F) : s0;
    out1[index] = relu ? std::max(s1, 0.F) : s1;
    out2[index] = relu ? std::max(s2, 0.F) : s2;
    out3[index] = relu ? std::max(s3, 0.F) : s3;
  };
  int output = 0;
  for (; output + 4 <= oc; output += 4) {
    float* out0 = dst + std::size_t(output) * output_plane;
    float* out1 = out0 + output_plane;
    float* out2 = out1 + output_plane;
    float* out3 = out2 + output_plane;
    const float b0 = bias ? bias[output] : 0.F, b1 = bias ? bias[output + 1] : 0.F;
    const float b2 = bias ? bias[output + 2] : 0.F, b3 = bias ? bias[output + 3] : 0.F;
    for (int y = row_begin; y < row_end; ++y) {
      const int iy0 = y - pad_top;
      const bool interior_y = y >= first_y && y < last_y;
      int x = 0;
      for (; x < output_w; ++x) {
        if (interior_y && x >= first_x && x + 16 <= last_x) break;
        scalar4(out0, out1, out2, out3, output, b0, b1, b2, b3, y, x);
      }
      for (; x + 16 <= last_x; x += 16) {
        const auto index = std::size_t(y) * output_w + x;
        __m512 s0 = _mm512_set1_ps(b0), s1 = _mm512_set1_ps(b1);
        __m512 s2 = _mm512_set1_ps(b2), s3 = _mm512_set1_ps(b3);
        for (int source = 0; source < source_count; ++source) {
          const float* src = sources[source];
          if (!src) continue;
          const float* f0 = wpacks + std::size_t(source) * w_src +
              std::size_t(output) * ic * 9;
          const float* f1 = f0 + std::size_t(ic) * 9;
          const float* f2 = f1 + std::size_t(ic) * 9;
          const float* f3 = f2 + std::size_t(ic) * 9;
          for (int input = 0; input < ic; ++input) {
            const float* plane =
                src + std::size_t(input) * input_plane + std::size_t(iy0) * input_w +
                x - pad_left;
            const float* k0 = f0 + std::size_t(input) * 9;
            const float* k1 = f1 + std::size_t(input) * 9;
            const float* k2 = f2 + std::size_t(input) * 9;
            const float* k3 = f3 + std::size_t(input) * 9;
            for (int ky = 0; ky < 3; ++ky) {
              const float* row = plane + std::size_t(ky) * input_w;
              for (int kx = 0; kx < 3; ++kx) {
                const __m512 values = _mm512_loadu_ps(row + kx);
                const int ki = ky * 3 + kx;
                s0 = _mm512_fmadd_ps(_mm512_set1_ps(k0[ki]), values, s0);
                s1 = _mm512_fmadd_ps(_mm512_set1_ps(k1[ki]), values, s1);
                s2 = _mm512_fmadd_ps(_mm512_set1_ps(k2[ki]), values, s2);
                s3 = _mm512_fmadd_ps(_mm512_set1_ps(k3[ki]), values, s3);
              }
            }
          }
        }
        if (relu) {
          const __m512 zero = _mm512_setzero_ps();
          s0 = _mm512_max_ps(s0, zero); s1 = _mm512_max_ps(s1, zero);
          s2 = _mm512_max_ps(s2, zero); s3 = _mm512_max_ps(s3, zero);
        }
        _mm512_storeu_ps(out0 + index, s0); _mm512_storeu_ps(out1 + index, s1);
        _mm512_storeu_ps(out2 + index, s2); _mm512_storeu_ps(out3 + index, s3);
      }
      for (; x < output_w; ++x)
        scalar4(out0, out1, out2, out3, output, b0, b1, b2, b3, y, x);
    }
  }
  for (; output < oc; ++output) {
    float* out = dst + std::size_t(output) * output_plane;
    const float base = bias ? bias[output] : 0.F;
    for (int y = row_begin; y < row_end; ++y) {
      for (int x = 0; x < output_w; ++x) {
        const int iy0 = y - pad_top, ix0 = x - pad_left;
        const auto index = std::size_t(y) * output_w + x;
        float sum = base;
        for (int source = 0; source < source_count; ++source) {
          const float* src = sources[source];
          if (!src) continue;
          const float* filter = wpacks + std::size_t(source) * w_src +
              std::size_t(output) * ic * 9;
          for (int input = 0; input < ic; ++input) {
            const float* plane = src + std::size_t(input) * input_plane;
            const float* kernel = filter + std::size_t(input) * 9;
            for (int ky = 0; ky < 3; ++ky) {
              const int iy = iy0 + ky;
              if (iy < 0 || iy >= input_h) continue;
              for (int kx = 0; kx < 3; ++kx) {
                const int ix = ix0 + kx;
                if (ix >= 0 && ix < input_w)
                  sum += plane[std::size_t(iy) * input_w + ix] * kernel[ky * 3 + kx];
              }
            }
          }
        }
        out[index] = relu ? std::max(sum, 0.F) : sum;
      }
    }
  }
}

// Eight-channel variant for the large medium-detector 3x3 maps.  It keeps
// eight independent reduction chains but shares each of the nine input-vector
// loads across all output channels. Border pixels remain scalar so the packed
// interior never reads outside the NCHW plane.
void Avx512Conv3x3Stride1x8(float* dst, const float* src, const float* weights,
                             const float* bias, int first_output, int last_output,
                             int input_channels, int input_h, int input_w,
                             int output_h, int output_w, int pad_top,
                             int pad_left, bool relu,
                             const float* const* src_planes) noexcept {
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(output_h) * output_w;
  const auto source_plane = [&](int input) -> const float* {
    return src_planes ? src_planes[input] : src + std::size_t(input) * input_plane;
  };
  const int first_y = std::max(0, pad_top);
  const int first_x = std::max(0, pad_left);
  const int last_y = std::min(output_h, input_h + pad_top - 2);
  const int last_x = std::min(output_w, input_w + pad_left - 2);
  const auto scalar8 = [&](float* const out[8], const float* const filter[8],
                           const float base[8], int y, int x) {
    const int iy0 = y - pad_top, ix0 = x - pad_left;
    float sum[8]{base[0], base[1], base[2], base[3], base[4], base[5], base[6], base[7]};
    for (int input = 0; input < input_channels; ++input) {
      const float* plane = source_plane(input);
      for (int ky = 0; ky < 3; ++ky) {
        const int iy = iy0 + ky;
        if (iy < 0 || iy >= input_h) continue;
        for (int kx = 0; kx < 3; ++kx) {
          const int ix = ix0 + kx;
          if (ix < 0 || ix >= input_w) continue;
          const float value = plane[std::size_t(iy) * input_w + ix];
          const int ki = std::size_t(input) * 9 + ky * 3 + kx;
          sum[0] += value * filter[0][ki]; sum[1] += value * filter[1][ki];
          sum[2] += value * filter[2][ki]; sum[3] += value * filter[3][ki];
          sum[4] += value * filter[4][ki]; sum[5] += value * filter[5][ki];
          sum[6] += value * filter[6][ki]; sum[7] += value * filter[7][ki];
        }
      }
    }
    const auto index = std::size_t(y) * output_w + x;
    for (int q = 0; q < 8; ++q) out[q][index] = relu ? std::max(sum[q], 0.F) : sum[q];
  };
  int output = first_output;
  for (; output + 8 <= last_output; output += 8) {
    float* out[8]; const float* filter[8]; float base[8];
    for (int q = 0; q < 8; ++q) {
      out[q] = dst + std::size_t(output + q) * output_plane;
      filter[q] = weights + std::size_t(output + q) * input_channels * 9;
      base[q] = bias ? bias[output + q] : 0.F;
    }
    for (int y = 0; y < output_h; ++y) {
      const int iy0 = y - pad_top;
      const bool interior_y = y >= first_y && y < last_y;
      int x = 0;
      for (; x < output_w; ++x) {
        if (interior_y && x >= first_x && x + 16 <= last_x) break;
        scalar8(out, filter, base, y, x);
      }
      for (; x + 16 <= last_x; x += 16) {
        __m512 s0 = _mm512_set1_ps(base[0]), s1 = _mm512_set1_ps(base[1]);
        __m512 s2 = _mm512_set1_ps(base[2]), s3 = _mm512_set1_ps(base[3]);
        __m512 s4 = _mm512_set1_ps(base[4]), s5 = _mm512_set1_ps(base[5]);
        __m512 s6 = _mm512_set1_ps(base[6]), s7 = _mm512_set1_ps(base[7]);
        for (int input = 0; input < input_channels; ++input) {
          const float* plane = source_plane(input) +
                               std::size_t(iy0) * input_w + x - pad_left;
          for (int ky = 0; ky < 3; ++ky) {
            const float* row = plane + std::size_t(ky) * input_w;
            for (int kx = 0; kx < 3; ++kx) {
              const __m512 value = _mm512_loadu_ps(row + kx);
              const int ki = std::size_t(input) * 9 + ky * 3 + kx;
              s0 = _mm512_fmadd_ps(_mm512_set1_ps(filter[0][ki]), value, s0);
              s1 = _mm512_fmadd_ps(_mm512_set1_ps(filter[1][ki]), value, s1);
              s2 = _mm512_fmadd_ps(_mm512_set1_ps(filter[2][ki]), value, s2);
              s3 = _mm512_fmadd_ps(_mm512_set1_ps(filter[3][ki]), value, s3);
              s4 = _mm512_fmadd_ps(_mm512_set1_ps(filter[4][ki]), value, s4);
              s5 = _mm512_fmadd_ps(_mm512_set1_ps(filter[5][ki]), value, s5);
              s6 = _mm512_fmadd_ps(_mm512_set1_ps(filter[6][ki]), value, s6);
              s7 = _mm512_fmadd_ps(_mm512_set1_ps(filter[7][ki]), value, s7);
            }
          }
        }
        if (relu) {
          const __m512 zero = _mm512_setzero_ps();
          s0 = _mm512_max_ps(s0, zero); s1 = _mm512_max_ps(s1, zero);
          s2 = _mm512_max_ps(s2, zero); s3 = _mm512_max_ps(s3, zero);
          s4 = _mm512_max_ps(s4, zero); s5 = _mm512_max_ps(s5, zero);
          s6 = _mm512_max_ps(s6, zero); s7 = _mm512_max_ps(s7, zero);
        }
        const auto index = std::size_t(y) * output_w + x;
        _mm512_storeu_ps(out[0] + index, s0); _mm512_storeu_ps(out[1] + index, s1);
        _mm512_storeu_ps(out[2] + index, s2); _mm512_storeu_ps(out[3] + index, s3);
        _mm512_storeu_ps(out[4] + index, s4); _mm512_storeu_ps(out[5] + index, s5);
        _mm512_storeu_ps(out[6] + index, s6); _mm512_storeu_ps(out[7] + index, s7);
      }
      for (; x < output_w; ++x) scalar8(out, filter, base, y, x);
    }
  }
  if (output < last_output) {
    Avx512Conv3x3Stride1x4(dst, src, weights, bias, output, last_output,
                            input_channels, input_h, input_w, output_h, output_w,
                            pad_top, pad_left, relu, src_planes, -1, -1, false);
  }
}

// Medium's detector context blocks are 5x5/7x7 stride-one convolutions.
// Reuse one input vector across four output channels here too; their wider
// kernels make avoiding four independent input walks particularly valuable.
void Avx512ConvOddStride1x4(float* dst, const float* src, const float* weights,
                             const float* bias, int first_output, int last_output,
                             int input_channels, int input_h, int input_w,
                             int output_h, int output_w, int kernel, int pad_top,
                             int pad_left) noexcept {
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(output_h) * output_w;
  const std::size_t filter_plane = std::size_t(kernel) * kernel;
  const int first_y = std::max(0, pad_top);
  const int first_x = std::max(0, pad_left);
  const int last_y = std::min(output_h, input_h + pad_top - kernel + 1);
  const int last_x = std::min(output_w, input_w + pad_left - kernel + 1);
  const auto scalar4 = [&](float* out0, float* out1, float* out2, float* out3,
                           const float* f0, const float* f1, const float* f2, const float* f3,
                           float b0, float b1, float b2, float b3, int y, int x) {
    const int iy0 = y - pad_top, ix0 = x - pad_left;
    float s0 = b0, s1 = b1, s2 = b2, s3 = b3;
    for (int input = 0; input < input_channels; ++input) {
      const float* plane = src + std::size_t(input) * input_plane;
      const float* k0 = f0 + std::size_t(input) * filter_plane;
      const float* k1 = f1 + std::size_t(input) * filter_plane;
      const float* k2 = f2 + std::size_t(input) * filter_plane;
      const float* k3 = f3 + std::size_t(input) * filter_plane;
      for (int ky = 0; ky < kernel; ++ky) {
        const int iy = iy0 + ky;
        if (iy < 0 || iy >= input_h) continue;
        for (int kx = 0; kx < kernel; ++kx) {
          const int ix = ix0 + kx;
          if (ix < 0 || ix >= input_w) continue;
          const float value = plane[std::size_t(iy) * input_w + ix];
          const int ki = ky * kernel + kx;
          s0 += value * k0[ki]; s1 += value * k1[ki];
          s2 += value * k2[ki]; s3 += value * k3[ki];
        }
      }
    }
    const auto index = std::size_t(y) * output_w + x;
    out0[index] = s0; out1[index] = s1; out2[index] = s2; out3[index] = s3;
  };
  int output = first_output;
  for (; output + 4 <= last_output; output += 4) {
    float* out0 = dst + std::size_t(output) * output_plane;
    float* out1 = out0 + output_plane;
    float* out2 = out1 + output_plane;
    float* out3 = out2 + output_plane;
    const float* f0 = weights + std::size_t(output) * input_channels * filter_plane;
    const float* f1 = f0 + std::size_t(input_channels) * filter_plane;
    const float* f2 = f1 + std::size_t(input_channels) * filter_plane;
    const float* f3 = f2 + std::size_t(input_channels) * filter_plane;
    const float b0 = bias ? bias[output] : 0.F, b1 = bias ? bias[output + 1] : 0.F;
    const float b2 = bias ? bias[output + 2] : 0.F, b3 = bias ? bias[output + 3] : 0.F;
    for (int y = 0; y < output_h; ++y) {
      const int iy0 = y - pad_top;
      const bool interior_y = y >= first_y && y < last_y;
      int x = 0;
      for (; x < output_w; ++x) {
        if (interior_y && x >= first_x && x + 16 <= last_x) break;
        scalar4(out0, out1, out2, out3, f0, f1, f2, f3, b0, b1, b2, b3, y, x);
      }
      for (; x + 16 <= last_x; x += 16) {
        __m512 s0 = _mm512_set1_ps(b0), s1 = _mm512_set1_ps(b1);
        __m512 s2 = _mm512_set1_ps(b2), s3 = _mm512_set1_ps(b3);
        for (int input = 0; input < input_channels; ++input) {
          const float* plane = src + std::size_t(input) * input_plane +
                               std::size_t(iy0) * input_w + x - pad_left;
          const float* k0 = f0 + std::size_t(input) * filter_plane;
          const float* k1 = f1 + std::size_t(input) * filter_plane;
          const float* k2 = f2 + std::size_t(input) * filter_plane;
          const float* k3 = f3 + std::size_t(input) * filter_plane;
          for (int ky = 0; ky < kernel; ++ky) {
            const float* row = plane + std::size_t(ky) * input_w;
            for (int kx = 0; kx < kernel; ++kx) {
              const __m512 values = _mm512_loadu_ps(row + kx);
              const int ki = ky * kernel + kx;
              s0 = _mm512_fmadd_ps(_mm512_set1_ps(k0[ki]), values, s0);
              s1 = _mm512_fmadd_ps(_mm512_set1_ps(k1[ki]), values, s1);
              s2 = _mm512_fmadd_ps(_mm512_set1_ps(k2[ki]), values, s2);
              s3 = _mm512_fmadd_ps(_mm512_set1_ps(k3[ki]), values, s3);
            }
          }
        }
        const auto index = std::size_t(y) * output_w + x;
        _mm512_storeu_ps(out0 + index, s0); _mm512_storeu_ps(out1 + index, s1);
        _mm512_storeu_ps(out2 + index, s2); _mm512_storeu_ps(out3 + index, s3);
      }
      for (; x < output_w; ++x) scalar4(out0, out1, out2, out3, f0, f1, f2, f3,
                                          b0, b1, b2, b3, y, x);
    }
  }
  if (output < last_output) {
    Avx512Conv2d(dst, src, weights, bias, output, last_output, input_channels,
                 input_h, input_w, output_h, output_w, kernel, kernel, pad_top, pad_left);
  }
}

// Eight-output counterpart for the medium detector's 5x5/7x7 context maps.
// It deliberately keeps the four-channel routine as its tail implementation:
// each output retains the same increasing input/channel/kernel accumulation
// sequence while eight broad output channels share every AVX-512 input load.
// C++ dispatch keeps this opt-in because eight live accumulators can trade
// cache traffic for frequency/register pressure on some client CPUs.
void Avx512ConvOddStride1x8(float* dst, const float* src, const float* weights,
                             const float* bias, int first_output, int last_output,
                             int input_channels, int input_h, int input_w,
                             int output_h, int output_w, int kernel, int pad_top,
                             int pad_left) noexcept {
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(output_h) * output_w;
  const std::size_t filter_plane = std::size_t(kernel) * kernel;
  const int first_y = std::max(0, pad_top);
  const int first_x = std::max(0, pad_left);
  const int last_y = std::min(output_h, input_h + pad_top - kernel + 1);
  const int last_x = std::min(output_w, input_w + pad_left - kernel + 1);
  const auto scalar8 = [&](float* const out[8], const float* const filter[8],
                           const float base[8], int y, int x) {
    const int iy0 = y - pad_top, ix0 = x - pad_left;
    float sum[8]{base[0], base[1], base[2], base[3],
                 base[4], base[5], base[6], base[7]};
    for (int input = 0; input < input_channels; ++input) {
      const float* plane = src + std::size_t(input) * input_plane;
      for (int ky = 0; ky < kernel; ++ky) {
        const int iy = iy0 + ky;
        if (iy < 0 || iy >= input_h) continue;
        for (int kx = 0; kx < kernel; ++kx) {
          const int ix = ix0 + kx;
          if (ix < 0 || ix >= input_w) continue;
          const float value = plane[std::size_t(iy) * input_w + ix];
          const auto index = std::size_t(input) * filter_plane + ky * kernel + kx;
          for (int output = 0; output < 8; ++output) sum[output] += value * filter[output][index];
        }
      }
    }
    const auto index = std::size_t(y) * output_w + x;
    for (int output = 0; output < 8; ++output) out[output][index] = sum[output];
  };
  int output = first_output;
  for (; output + 8 <= last_output; output += 8) {
    float* out[8]; const float* filter[8]; float base[8];
    for (int q = 0; q < 8; ++q) {
      out[q] = dst + std::size_t(output + q) * output_plane;
      filter[q] = weights + std::size_t(output + q) * input_channels * filter_plane;
      base[q] = bias ? bias[output + q] : 0.F;
    }
    for (int y = 0; y < output_h; ++y) {
      const int iy0 = y - pad_top;
      const bool interior_y = y >= first_y && y < last_y;
      int x = 0;
      for (; x < output_w; ++x) {
        if (interior_y && x >= first_x && x + 16 <= last_x) break;
        scalar8(out, filter, base, y, x);
      }
      for (; x + 16 <= last_x; x += 16) {
        __m512 sums[8]{_mm512_set1_ps(base[0]), _mm512_set1_ps(base[1]),
                        _mm512_set1_ps(base[2]), _mm512_set1_ps(base[3]),
                        _mm512_set1_ps(base[4]), _mm512_set1_ps(base[5]),
                        _mm512_set1_ps(base[6]), _mm512_set1_ps(base[7])};
        for (int input = 0; input < input_channels; ++input) {
          const float* plane = src + std::size_t(input) * input_plane +
                               std::size_t(iy0) * input_w + x - pad_left;
          for (int ky = 0; ky < kernel; ++ky) {
            const float* row = plane + std::size_t(ky) * input_w;
            for (int kx = 0; kx < kernel; ++kx) {
              const __m512 values = _mm512_loadu_ps(row + kx);
              const auto index = std::size_t(input) * filter_plane + ky * kernel + kx;
              for (int q = 0; q < 8; ++q) {
                sums[q] = _mm512_fmadd_ps(_mm512_set1_ps(filter[q][index]), values, sums[q]);
              }
            }
          }
        }
        const auto index = std::size_t(y) * output_w + x;
        for (int q = 0; q < 8; ++q) _mm512_storeu_ps(out[q] + index, sums[q]);
      }
      for (; x < output_w; ++x) scalar8(out, filter, base, y, x);
    }
  }
  if (output < last_output) {
    Avx512ConvOddStride1x4(dst, src, weights, bias, output, last_output,
                            input_channels, input_h, input_w, output_h, output_w,
                            kernel, pad_top, pad_left);
  }
}

// Four-output fusion for the detector's asymmetric 1xK/Kx1 context filters.
// The contiguous axis is vectorized while the orthogonal/padded border uses
// the same scalar accumulation order as the general convolution path.
void Avx512ConvAsymmetricStride1x4(float* dst, const float* src, const float* weights,
                                   const float* bias, int first_output, int last_output,
                                   int input_channels, int input_h, int input_w,
                                   int output_h, int output_w, int kernel_h, int kernel_w,
                                   int pad_top, int pad_left) noexcept {
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(output_h) * output_w;
  const std::size_t filter_plane = std::size_t(kernel_h) * kernel_w;
  const int first_y = std::max(0, pad_top);
  const int first_x = std::max(0, pad_left);
  const int last_y = std::min(output_h, input_h + pad_top - kernel_h + 1);
  const int last_x = std::min(output_w, input_w + pad_left - kernel_w + 1);
  const auto scalar4 = [&](float* o0, float* o1, float* o2, float* o3,
                           const float* f0, const float* f1, const float* f2, const float* f3,
                           float b0, float b1, float b2, float b3, int y, int x) {
    const int iy0 = y - pad_top, ix0 = x - pad_left;
    float s0 = b0, s1 = b1, s2 = b2, s3 = b3;
    for (int input = 0; input < input_channels; ++input) {
      const float* plane = src + std::size_t(input) * input_plane;
      const float* k0 = f0 + std::size_t(input) * filter_plane;
      const float* k1 = f1 + std::size_t(input) * filter_plane;
      const float* k2 = f2 + std::size_t(input) * filter_plane;
      const float* k3 = f3 + std::size_t(input) * filter_plane;
      for (int ky = 0; ky < kernel_h; ++ky) { const int iy = iy0 + ky; if (iy < 0 || iy >= input_h) continue;
        for (int kx = 0; kx < kernel_w; ++kx) { const int ix = ix0 + kx; if (ix < 0 || ix >= input_w) continue;
          const float v = plane[std::size_t(iy) * input_w + ix]; const int ki = ky * kernel_w + kx;
          s0 += v * k0[ki]; s1 += v * k1[ki]; s2 += v * k2[ki]; s3 += v * k3[ki];
        }
      }
    }
    const auto index = std::size_t(y) * output_w + x;
    o0[index] = s0; o1[index] = s1; o2[index] = s2; o3[index] = s3;
  };
  int output = first_output;
  for (; output + 4 <= last_output; output += 4) {
    float* o0 = dst + std::size_t(output) * output_plane;
    float* o1 = o0 + output_plane; float* o2 = o1 + output_plane; float* o3 = o2 + output_plane;
    const float* f0 = weights + std::size_t(output) * input_channels * filter_plane;
    const float* f1 = f0 + std::size_t(input_channels) * filter_plane;
    const float* f2 = f1 + std::size_t(input_channels) * filter_plane;
    const float* f3 = f2 + std::size_t(input_channels) * filter_plane;
    const float b0 = bias ? bias[output] : 0.F, b1 = bias ? bias[output + 1] : 0.F;
    const float b2 = bias ? bias[output + 2] : 0.F, b3 = bias ? bias[output + 3] : 0.F;
    for (int y = 0; y < output_h; ++y) {
      const int iy0 = y - pad_top;
      const bool interior_y = y >= first_y && y < last_y;
      int x = 0;
      for (; x < output_w; ++x) {
        if (interior_y && x >= first_x && x + 16 <= last_x) break;
        scalar4(o0,o1,o2,o3,f0,f1,f2,f3,b0,b1,b2,b3,y,x);
      }
      for (; x + 16 <= last_x; x += 16) {
        __m512 s0 = _mm512_set1_ps(b0), s1 = _mm512_set1_ps(b1), s2 = _mm512_set1_ps(b2), s3 = _mm512_set1_ps(b3);
        for (int input = 0; input < input_channels; ++input) {
          const float* plane = src + std::size_t(input) * input_plane + std::size_t(iy0) * input_w + x - pad_left;
          const float* k0 = f0 + std::size_t(input) * filter_plane;
          const float* k1 = f1 + std::size_t(input) * filter_plane;
          const float* k2 = f2 + std::size_t(input) * filter_plane;
          const float* k3 = f3 + std::size_t(input) * filter_plane;
          for (int ky = 0; ky < kernel_h; ++ky) for (int kx = 0; kx < kernel_w; ++kx) {
            const __m512 values = _mm512_loadu_ps(plane + std::size_t(ky) * input_w + kx);
            const int ki = ky * kernel_w + kx;
            s0 = _mm512_fmadd_ps(_mm512_set1_ps(k0[ki]), values, s0);
            s1 = _mm512_fmadd_ps(_mm512_set1_ps(k1[ki]), values, s1);
            s2 = _mm512_fmadd_ps(_mm512_set1_ps(k2[ki]), values, s2);
            s3 = _mm512_fmadd_ps(_mm512_set1_ps(k3[ki]), values, s3);
          }
        }
        const auto index = std::size_t(y) * output_w + x;
        _mm512_storeu_ps(o0 + index, s0); _mm512_storeu_ps(o1 + index, s1);
        _mm512_storeu_ps(o2 + index, s2); _mm512_storeu_ps(o3 + index, s3);
      }
      for (; x < output_w; ++x) scalar4(o0,o1,o2,o3,f0,f1,f2,f3,b0,b1,b2,b3,y,x);
    }
  }
  if (output < last_output) {
    Avx512Conv2d(dst, src, weights, bias, output, last_output, input_channels, input_h, input_w,
                 output_h, output_w, kernel_h, kernel_w, pad_top, pad_left);
  }
}

// Same model-specific stride-2 3x3 path as AVX2, widened to sixteen output
// pixels. The scalar border retains exact zero-padding behavior.
void Avx512Conv3x3Stride2(float* dst, const float* src, const float* weights,
                           const float* bias, int first_output, int last_output,
                           int input_channels, int input_h, int input_w,
                           int output_h, int output_w, int pad_top,
                           int pad_left, bool relu, bool packed_loads,
                           const float* const* src_planes = nullptr, int row0 = -1,
                           int row1 = -1) noexcept {
  constexpr int lanes = 16;
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(output_h) * output_w;
  const int row_begin = row0 >= 0 ? row0 : 0;
  const int row_end = row1 >= 0 ? row1 : output_h;
  const auto source_plane = [&](int input) -> const float* {
    return src_planes ? src_planes[input] : src + std::size_t(input) * input_plane;
  };
  const __m512i offsets = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14,
                                              16, 18, 20, 22, 24, 26, 28, 30);
  // Select even lanes from two adjacent 16-float loads. For the interior of
  // a stride-two 3x3 window this replaces nine AVX-512 gathers with eighteen
  // contiguous loads plus permutes, which is markedly friendlier to cache and
  // address-generation hardware on common desktop AVX-512 CPUs.
  const __m512i even_lanes = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14,
                                                16, 18, 20, 22, 24, 26, 28, 30);
  const auto load_stride2 = [&](const float* row) {
    if (!packed_loads) return _mm512_i32gather_ps(offsets, row, 4);
    const __m512 lo = _mm512_loadu_ps(row);
    const __m512 hi = _mm512_loadu_ps(row + 16);
    return _mm512_permutex2var_ps(lo, even_lanes, hi);
  };
  // The packed form loads one harmless odd lane beyond the final sampled
  // element, so keep that physical load inside the current source row.
  const int packed_extent = packed_loads ? 33 : 32;
  const auto scalar4 = [&](float* out0, float* out1, float* out2, float* out3,
                           const float* filter0, const float* filter1,
                           const float* filter2, const float* filter3,
                           float base0, float base1, float base2, float base3,
                           int y, int x) {
    const int iy0 = y * 2 - pad_top, ix0 = x * 2 - pad_left;
    float sum0 = base0, sum1 = base1, sum2 = base2, sum3 = base3;
    for (int input = 0; input < input_channels; ++input) {
      const float* plane = source_plane(input);
      const float* k0 = filter0 + std::size_t(input) * 9;
      const float* k1 = filter1 + std::size_t(input) * 9;
      const float* k2 = filter2 + std::size_t(input) * 9;
      const float* k3 = filter3 + std::size_t(input) * 9;
      for (int ky = 0; ky < 3; ++ky) {
        const int iy = iy0 + ky;
        if (iy < 0 || iy >= input_h) continue;
        for (int kx = 0; kx < 3; ++kx) {
          const int ix = ix0 + kx;
          if (ix < 0 || ix >= input_w) continue;
          const float value = plane[std::size_t(iy) * input_w + ix];
          const int ki = ky * 3 + kx;
          sum0 += value * k0[ki]; sum1 += value * k1[ki];
          sum2 += value * k2[ki]; sum3 += value * k3[ki];
        }
      }
    }
    const auto index = std::size_t(y) * output_w + x;
    out0[index] = relu ? std::max(sum0, 0.F) : sum0;
    out1[index] = relu ? std::max(sum1, 0.F) : sum1;
    out2[index] = relu ? std::max(sum2, 0.F) : sum2;
    out3[index] = relu ? std::max(sum3, 0.F) : sum3;
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
    for (int y = row_begin; y < row_end; ++y) {
      const int iy0 = y * 2 - pad_top;
      const bool interior_y = iy0 >= 0 && iy0 + 2 < input_h;
      int x = 0;
      for (; x < output_w; ++x) {
        const int ix0 = x * 2 - pad_left;
        if (interior_y && ix0 >= 0 && ix0 + packed_extent < input_w) break;
        scalar4(out[0], out[1], out[2], out[3], filter[0], filter[1], filter[2], filter[3],
                base[0], base[1], base[2], base[3], y, x);
        scalar4(out[4], out[5], out[6], out[7], filter[4], filter[5], filter[6], filter[7],
                base[4], base[5], base[6], base[7], y, x);
      }
      for (; x + lanes <= output_w; x += lanes) {
        const int ix0 = x * 2 - pad_left;
        if (!(interior_y && ix0 >= 0 && ix0 + packed_extent < input_w)) break;
        __m512 s0 = _mm512_set1_ps(base[0]), s1 = _mm512_set1_ps(base[1]);
        __m512 s2 = _mm512_set1_ps(base[2]), s3 = _mm512_set1_ps(base[3]);
        __m512 s4 = _mm512_set1_ps(base[4]), s5 = _mm512_set1_ps(base[5]);
        __m512 s6 = _mm512_set1_ps(base[6]), s7 = _mm512_set1_ps(base[7]);
        for (int input = 0; input < input_channels; ++input) {
          const float* plane = source_plane(input) + std::size_t(iy0) * input_w + ix0;
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
              const __m512 values = load_stride2(row + kx);
              const int ki = ky * 3 + kx;
              s0 = _mm512_fmadd_ps(_mm512_set1_ps(k0[ki]), values, s0);
              s1 = _mm512_fmadd_ps(_mm512_set1_ps(k1[ki]), values, s1);
              s2 = _mm512_fmadd_ps(_mm512_set1_ps(k2[ki]), values, s2);
              s3 = _mm512_fmadd_ps(_mm512_set1_ps(k3[ki]), values, s3);
              s4 = _mm512_fmadd_ps(_mm512_set1_ps(k4[ki]), values, s4);
              s5 = _mm512_fmadd_ps(_mm512_set1_ps(k5[ki]), values, s5);
              s6 = _mm512_fmadd_ps(_mm512_set1_ps(k6[ki]), values, s6);
              s7 = _mm512_fmadd_ps(_mm512_set1_ps(k7[ki]), values, s7);
            }
          }
        }
        const auto index = std::size_t(y) * output_w + x;
        if (relu) {
          const __m512 zero = _mm512_setzero_ps();
          s0 = _mm512_max_ps(s0, zero); s1 = _mm512_max_ps(s1, zero);
          s2 = _mm512_max_ps(s2, zero); s3 = _mm512_max_ps(s3, zero);
          s4 = _mm512_max_ps(s4, zero); s5 = _mm512_max_ps(s5, zero);
          s6 = _mm512_max_ps(s6, zero); s7 = _mm512_max_ps(s7, zero);
        }
        _mm512_storeu_ps(out[0] + index, s0); _mm512_storeu_ps(out[1] + index, s1);
        _mm512_storeu_ps(out[2] + index, s2); _mm512_storeu_ps(out[3] + index, s3);
        _mm512_storeu_ps(out[4] + index, s4); _mm512_storeu_ps(out[5] + index, s5);
        _mm512_storeu_ps(out[6] + index, s6); _mm512_storeu_ps(out[7] + index, s7);
      }
      for (; x < output_w; ++x) {
        scalar4(out[0], out[1], out[2], out[3], filter[0], filter[1], filter[2], filter[3],
                base[0], base[1], base[2], base[3], y, x);
        scalar4(out[4], out[5], out[6], out[7], filter[4], filter[5], filter[6], filter[7],
                base[4], base[5], base[6], base[7], y, x);
      }
    }
  }
  for (; output + 4 <= last_output; output += 4) {
    float* out0 = dst + std::size_t(output) * output_plane;
    float* out1 = out0 + output_plane;
    float* out2 = out1 + output_plane;
    float* out3 = out2 + output_plane;
    const float* filter0 = weights + std::size_t(output) * input_channels * 9;
    const float* filter1 = filter0 + std::size_t(input_channels) * 9;
    const float* filter2 = filter1 + std::size_t(input_channels) * 9;
    const float* filter3 = filter2 + std::size_t(input_channels) * 9;
    const float base0 = bias ? bias[output] : 0.F;
    const float base1 = bias ? bias[output + 1] : 0.F;
    const float base2 = bias ? bias[output + 2] : 0.F;
    const float base3 = bias ? bias[output + 3] : 0.F;
    for (int y = row_begin; y < row_end; ++y) {
      const int iy0 = y * 2 - pad_top;
      const bool interior_y = iy0 >= 0 && iy0 + 2 < input_h;
      int x = 0;
      for (; x < output_w; ++x) {
        const int ix0 = x * 2 - pad_left;
        if (interior_y && ix0 >= 0 && ix0 + packed_extent < input_w) break;
        scalar4(out0, out1, out2, out3, filter0, filter1, filter2, filter3,
                base0, base1, base2, base3, y, x);
      }
      for (; x + lanes <= output_w; x += lanes) {
        const int ix0 = x * 2 - pad_left;
        if (!(interior_y && ix0 >= 0 && ix0 + packed_extent < input_w)) break;
        __m512 sum0 = _mm512_set1_ps(base0), sum1 = _mm512_set1_ps(base1);
        __m512 sum2 = _mm512_set1_ps(base2), sum3 = _mm512_set1_ps(base3);
        for (int input = 0; input < input_channels; ++input) {
          const float* plane = source_plane(input) +
                               std::size_t(iy0) * input_w + ix0;
          const float* k0 = filter0 + std::size_t(input) * 9;
          const float* k1 = filter1 + std::size_t(input) * 9;
          const float* k2 = filter2 + std::size_t(input) * 9;
          const float* k3 = filter3 + std::size_t(input) * 9;
          for (int ky = 0; ky < 3; ++ky) {
            const float* row = plane + std::size_t(ky) * input_w;
            for (int kx = 0; kx < 3; ++kx) {
              const __m512 values = load_stride2(row + kx);
              const int ki = ky * 3 + kx;
              sum0 = _mm512_fmadd_ps(_mm512_set1_ps(k0[ki]), values, sum0);
              sum1 = _mm512_fmadd_ps(_mm512_set1_ps(k1[ki]), values, sum1);
              sum2 = _mm512_fmadd_ps(_mm512_set1_ps(k2[ki]), values, sum2);
              sum3 = _mm512_fmadd_ps(_mm512_set1_ps(k3[ki]), values, sum3);
            }
          }
        }
        const auto index = std::size_t(y) * output_w + x;
        if (relu) { const __m512 zero = _mm512_setzero_ps(); sum0 = _mm512_max_ps(sum0, zero); sum1 = _mm512_max_ps(sum1, zero); sum2 = _mm512_max_ps(sum2, zero); sum3 = _mm512_max_ps(sum3, zero); }
        _mm512_storeu_ps(out0 + index, sum0); _mm512_storeu_ps(out1 + index, sum1);
        _mm512_storeu_ps(out2 + index, sum2); _mm512_storeu_ps(out3 + index, sum3);
      }
      for (; x < output_w; ++x) {
        scalar4(out0, out1, out2, out3, filter0, filter1, filter2, filter3,
                base0, base1, base2, base3, y, x);
      }
    }
  }
  for (; output < last_output; ++output) {
    float* out = dst + std::size_t(output) * output_plane;
    const float* filter = weights + std::size_t(output) * input_channels * 9;
    const float base = bias ? bias[output] : 0.F;
    for (int y = row_begin; y < row_end; ++y) {
      const int iy0 = y * 2 - pad_top;
      const bool interior_y = iy0 >= 0 && iy0 + 2 < input_h;
      int x = 0;
      for (; x < output_w; ++x) {
        const int ix0 = x * 2 - pad_left;
        if (interior_y && ix0 >= 0 && ix0 + packed_extent < input_w) break;
        float sum = base;
        for (int input = 0; input < input_channels; ++input) {
          const float* plane = source_plane(input);
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
        if (!(interior_y && ix0 >= 0 && ix0 + packed_extent < input_w)) break;
        __m512 sum = _mm512_set1_ps(base);
        for (int input = 0; input < input_channels; ++input) {
          const float* plane = source_plane(input) +
                               std::size_t(iy0) * input_w + ix0;
          const float* kernel = filter + std::size_t(input) * 9;
          for (int ky = 0; ky < 3; ++ky) {
            const float* row = plane + std::size_t(ky) * input_w;
            for (int kx = 0; kx < 3; ++kx) {
              sum = _mm512_fmadd_ps(_mm512_set1_ps(kernel[ky * 3 + kx]),
                                    load_stride2(row + kx), sum);
            }
          }
        }
        if (relu) sum = _mm512_max_ps(sum, _mm512_setzero_ps());
        _mm512_storeu_ps(out + std::size_t(y) * output_w + x, sum);
      }
      for (; x < output_w; ++x) {
        const int ix0 = x * 2 - pad_left;
        float sum = base;
        for (int input = 0; input < input_channels; ++input) {
          const float* plane = source_plane(input);
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

void Avx512Conv3x3Stride2C3(float* dst, const float* src, const float* weights,
                             const float* bias, int first_output, int last_output,
                             int input_h, int input_w, int output_h, int output_w,
                             int pad_top, int pad_left, bool relu,
                             bool packed_loads, int row0, int row1) noexcept {
  constexpr int lanes = 16;
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(output_h) * output_w;
  const int row_begin = row0 < 0 ? 0 : std::min(output_h, row0);
  const int row_end = row1 < 0 ? output_h : std::min(output_h, row1);
  if (row_begin >= row_end) return;
  const __m512i even_lanes = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14,
                                                16, 18, 20, 22, 24, 26, 28, 30);
  const auto load_stride2 = [&](const float* row) {
    if (!packed_loads) {
      const __m512i offsets = even_lanes;
      return _mm512_i32gather_ps(offsets, row, 4);
    }
    const __m512 lo = _mm512_loadu_ps(row);
    const __m512 hi = _mm512_loadu_ps(row + 16);
    return _mm512_permutex2var_ps(lo, even_lanes, hi);
  };
  const int packed_extent = packed_loads ? 33 : 32;
  int output = first_output;
  for (; output + 16 <= last_output; output += 16) {
    float* out[16];
    const float* filter[16];
    float base[16];
    for (int q = 0; q < 16; ++q) {
      out[q] = dst + std::size_t(output + q) * output_plane;
      filter[q] = weights + std::size_t(output + q) * 27;
      base[q] = bias ? bias[output + q] : 0.F;
    }
    const auto scalar16 = [&](int y, int x) {
      for (int q = 0; q < 16; ++q) {
        float sum = base[q];
        for (int input = 0; input < 3; ++input) {
          const float* plane = src + std::size_t(input) * input_plane;
          const float* kernel = filter[q] + input * 9;
          for (int ky = 0; ky < 3; ++ky) {
            const int iy = y * 2 - pad_top + ky;
            if (iy < 0 || iy >= input_h) continue;
            for (int kx = 0; kx < 3; ++kx) {
              const int ix = x * 2 - pad_left + kx;
              if (ix >= 0 && ix < input_w)
                sum += plane[std::size_t(iy) * input_w + ix] * kernel[ky * 3 + kx];
            }
          }
        }
        out[q][std::size_t(y) * output_w + x] = relu ? std::max(sum, 0.F) : sum;
      }
    };
    for (int y = row_begin; y < row_end; ++y) {
      const int iy0 = y * 2 - pad_top;
      const bool interior_y = iy0 >= 0 && iy0 + 2 < input_h;
      int x = 0;
      for (; x < output_w; ++x) {
        const int ix0 = x * 2 - pad_left;
        if (interior_y && ix0 >= 0 && ix0 + packed_extent < input_w) break;
        scalar16(y, x);
      }
      for (; x + lanes <= output_w; x += lanes) {
        const int ix0 = x * 2 - pad_left;
        if (!(interior_y && ix0 >= 0 && ix0 + packed_extent < input_w)) break;
        __m512 s0 = _mm512_set1_ps(base[0]), s1 = _mm512_set1_ps(base[1]);
        __m512 s2 = _mm512_set1_ps(base[2]), s3 = _mm512_set1_ps(base[3]);
        __m512 s4 = _mm512_set1_ps(base[4]), s5 = _mm512_set1_ps(base[5]);
        __m512 s6 = _mm512_set1_ps(base[6]), s7 = _mm512_set1_ps(base[7]);
        __m512 s8 = _mm512_set1_ps(base[8]), s9 = _mm512_set1_ps(base[9]);
        __m512 s10 = _mm512_set1_ps(base[10]), s11 = _mm512_set1_ps(base[11]);
        __m512 s12 = _mm512_set1_ps(base[12]), s13 = _mm512_set1_ps(base[13]);
        __m512 s14 = _mm512_set1_ps(base[14]), s15 = _mm512_set1_ps(base[15]);
        for (int input = 0; input < 3; ++input) {
          const float* plane = src + std::size_t(input) * input_plane +
                               std::size_t(iy0) * input_w + ix0;
          const float* k[16];
          for (int q = 0; q < 16; ++q) k[q] = filter[q] + input * 9;
          for (int ky = 0; ky < 3; ++ky) {
            const float* row = plane + std::size_t(ky) * input_w;
            for (int kx = 0; kx < 3; ++kx) {
              const __m512 values = load_stride2(row + kx);
              const int ki = ky * 3 + kx;
              s0 = _mm512_fmadd_ps(_mm512_set1_ps(k[0][ki]), values, s0);
              s1 = _mm512_fmadd_ps(_mm512_set1_ps(k[1][ki]), values, s1);
              s2 = _mm512_fmadd_ps(_mm512_set1_ps(k[2][ki]), values, s2);
              s3 = _mm512_fmadd_ps(_mm512_set1_ps(k[3][ki]), values, s3);
              s4 = _mm512_fmadd_ps(_mm512_set1_ps(k[4][ki]), values, s4);
              s5 = _mm512_fmadd_ps(_mm512_set1_ps(k[5][ki]), values, s5);
              s6 = _mm512_fmadd_ps(_mm512_set1_ps(k[6][ki]), values, s6);
              s7 = _mm512_fmadd_ps(_mm512_set1_ps(k[7][ki]), values, s7);
              s8 = _mm512_fmadd_ps(_mm512_set1_ps(k[8][ki]), values, s8);
              s9 = _mm512_fmadd_ps(_mm512_set1_ps(k[9][ki]), values, s9);
              s10 = _mm512_fmadd_ps(_mm512_set1_ps(k[10][ki]), values, s10);
              s11 = _mm512_fmadd_ps(_mm512_set1_ps(k[11][ki]), values, s11);
              s12 = _mm512_fmadd_ps(_mm512_set1_ps(k[12][ki]), values, s12);
              s13 = _mm512_fmadd_ps(_mm512_set1_ps(k[13][ki]), values, s13);
              s14 = _mm512_fmadd_ps(_mm512_set1_ps(k[14][ki]), values, s14);
              s15 = _mm512_fmadd_ps(_mm512_set1_ps(k[15][ki]), values, s15);
            }
          }
        }
        const auto index = std::size_t(y) * output_w + x;
        if (relu) {
          const __m512 zero = _mm512_setzero_ps();
          s0 = _mm512_max_ps(s0, zero); s1 = _mm512_max_ps(s1, zero);
          s2 = _mm512_max_ps(s2, zero); s3 = _mm512_max_ps(s3, zero);
          s4 = _mm512_max_ps(s4, zero); s5 = _mm512_max_ps(s5, zero);
          s6 = _mm512_max_ps(s6, zero); s7 = _mm512_max_ps(s7, zero);
          s8 = _mm512_max_ps(s8, zero); s9 = _mm512_max_ps(s9, zero);
          s10 = _mm512_max_ps(s10, zero); s11 = _mm512_max_ps(s11, zero);
          s12 = _mm512_max_ps(s12, zero); s13 = _mm512_max_ps(s13, zero);
          s14 = _mm512_max_ps(s14, zero); s15 = _mm512_max_ps(s15, zero);
        }
        _mm512_storeu_ps(out[0] + index, s0); _mm512_storeu_ps(out[1] + index, s1);
        _mm512_storeu_ps(out[2] + index, s2); _mm512_storeu_ps(out[3] + index, s3);
        _mm512_storeu_ps(out[4] + index, s4); _mm512_storeu_ps(out[5] + index, s5);
        _mm512_storeu_ps(out[6] + index, s6); _mm512_storeu_ps(out[7] + index, s7);
        _mm512_storeu_ps(out[8] + index, s8); _mm512_storeu_ps(out[9] + index, s9);
        _mm512_storeu_ps(out[10] + index, s10); _mm512_storeu_ps(out[11] + index, s11);
        _mm512_storeu_ps(out[12] + index, s12); _mm512_storeu_ps(out[13] + index, s13);
        _mm512_storeu_ps(out[14] + index, s14); _mm512_storeu_ps(out[15] + index, s15);
      }
      for (; x < output_w; ++x) scalar16(y, x);
    }
  }
  for (; output + 8 <= last_output; output += 8) {
    float* out[8];
    const float* filter[8];
    float base[8];
    for (int q = 0; q < 8; ++q) {
      out[q] = dst + std::size_t(output + q) * output_plane;
      filter[q] = weights + std::size_t(output + q) * 27;
      base[q] = bias ? bias[output + q] : 0.F;
    }
    for (int y = row_begin; y < row_end; ++y) {
      const int iy0 = y * 2 - pad_top;
      const bool interior_y = iy0 >= 0 && iy0 + 2 < input_h;
      int x = 0;
      for (; x < output_w; ++x) {
        const int ix0 = x * 2 - pad_left;
        if (interior_y && ix0 >= 0 && ix0 + packed_extent < input_w) break;
        for (int q = 0; q < 8; ++q) {
          float sum = base[q];
          for (int input = 0; input < 3; ++input) {
            const float* plane = src + std::size_t(input) * input_plane;
            const float* kernel = filter[q] + input * 9;
            for (int ky = 0; ky < 3; ++ky) {
              const int iy = iy0 + ky;
              if (iy < 0 || iy >= input_h) continue;
              for (int kx = 0; kx < 3; ++kx) {
                const int ix = ix0 + kx;
                if (ix >= 0 && ix < input_w)
                  sum += plane[std::size_t(iy) * input_w + ix] * kernel[ky * 3 + kx];
              }
            }
          }
          out[q][std::size_t(y) * output_w + x] = relu ? std::max(sum, 0.F) : sum;
        }
      }
      for (; x + lanes <= output_w; x += lanes) {
        const int ix0 = x * 2 - pad_left;
        if (!(interior_y && ix0 >= 0 && ix0 + packed_extent < input_w)) break;
        __m512 s0 = _mm512_set1_ps(base[0]), s1 = _mm512_set1_ps(base[1]);
        __m512 s2 = _mm512_set1_ps(base[2]), s3 = _mm512_set1_ps(base[3]);
        __m512 s4 = _mm512_set1_ps(base[4]), s5 = _mm512_set1_ps(base[5]);
        __m512 s6 = _mm512_set1_ps(base[6]), s7 = _mm512_set1_ps(base[7]);
        for (int input = 0; input < 3; ++input) {
          const float* plane = src + std::size_t(input) * input_plane +
                               std::size_t(iy0) * input_w + ix0;
          const float* k0 = filter[0] + input * 9;
          const float* k1 = filter[1] + input * 9;
          const float* k2 = filter[2] + input * 9;
          const float* k3 = filter[3] + input * 9;
          const float* k4 = filter[4] + input * 9;
          const float* k5 = filter[5] + input * 9;
          const float* k6 = filter[6] + input * 9;
          const float* k7 = filter[7] + input * 9;
          for (int ky = 0; ky < 3; ++ky) {
            const float* row = plane + std::size_t(ky) * input_w;
            for (int kx = 0; kx < 3; ++kx) {
              const __m512 values = load_stride2(row + kx);
              const int ki = ky * 3 + kx;
              s0 = _mm512_fmadd_ps(_mm512_set1_ps(k0[ki]), values, s0);
              s1 = _mm512_fmadd_ps(_mm512_set1_ps(k1[ki]), values, s1);
              s2 = _mm512_fmadd_ps(_mm512_set1_ps(k2[ki]), values, s2);
              s3 = _mm512_fmadd_ps(_mm512_set1_ps(k3[ki]), values, s3);
              s4 = _mm512_fmadd_ps(_mm512_set1_ps(k4[ki]), values, s4);
              s5 = _mm512_fmadd_ps(_mm512_set1_ps(k5[ki]), values, s5);
              s6 = _mm512_fmadd_ps(_mm512_set1_ps(k6[ki]), values, s6);
              s7 = _mm512_fmadd_ps(_mm512_set1_ps(k7[ki]), values, s7);
            }
          }
        }
        const auto index = std::size_t(y) * output_w + x;
        if (relu) {
          const __m512 zero = _mm512_setzero_ps();
          s0 = _mm512_max_ps(s0, zero); s1 = _mm512_max_ps(s1, zero);
          s2 = _mm512_max_ps(s2, zero); s3 = _mm512_max_ps(s3, zero);
          s4 = _mm512_max_ps(s4, zero); s5 = _mm512_max_ps(s5, zero);
          s6 = _mm512_max_ps(s6, zero); s7 = _mm512_max_ps(s7, zero);
        }
        _mm512_storeu_ps(out[0] + index, s0); _mm512_storeu_ps(out[1] + index, s1);
        _mm512_storeu_ps(out[2] + index, s2); _mm512_storeu_ps(out[3] + index, s3);
        _mm512_storeu_ps(out[4] + index, s4); _mm512_storeu_ps(out[5] + index, s5);
        _mm512_storeu_ps(out[6] + index, s6); _mm512_storeu_ps(out[7] + index, s7);
      }
      for (; x < output_w; ++x) {
        const int ix0 = x * 2 - pad_left;
        for (int q = 0; q < 8; ++q) {
          float sum = base[q];
          for (int input = 0; input < 3; ++input) {
            const float* plane = src + std::size_t(input) * input_plane;
            const float* kernel = filter[q] + input * 9;
            for (int ky = 0; ky < 3; ++ky) {
              const int iy = iy0 + ky;
              if (iy < 0 || iy >= input_h) continue;
              for (int kx = 0; kx < 3; ++kx) {
                const int ix = ix0 + kx;
                if (ix >= 0 && ix < input_w)
                  sum += plane[std::size_t(iy) * input_w + ix] * kernel[ky * 3 + kx];
              }
            }
          }
          out[q][std::size_t(y) * output_w + x] = relu ? std::max(sum, 0.F) : sum;
        }
      }
    }
  }
  if (output < last_output) {
    Avx512Conv3x3Stride2(dst, src, weights, bias, output, last_output, 3,
                         input_h, input_w, output_h, output_w, pad_top, pad_left,
                         relu, packed_loads, nullptr, row_begin, row_end);
  }
}

void Avx512WriteIdentityRgbToNchw(float* dst, const std::uint8_t* rgb, int width,
                                  int height, int source_width, int left, int top,
                                  const float* scale, const float* shift,
                                  int row_width) noexcept {
  const std::size_t plane = std::size_t(height) * row_width;
  const __m512 scale_b = _mm512_set1_ps(scale[0]);
  const __m512 scale_g = _mm512_set1_ps(scale[1]);
  const __m512 scale_r = _mm512_set1_ps(scale[2]);
  const __m512 shift_b = _mm512_set1_ps(shift[0]);
  const __m512 shift_g = _mm512_set1_ps(shift[1]);
  const __m512 shift_r = _mm512_set1_ps(shift[2]);
  const __m128i shuf_r = _mm_setr_epi8(0, 3, 6, 9, -1, -1, -1, -1,
                                       -1, -1, -1, -1, -1, -1, -1, -1);
  const __m128i shuf_g = _mm_setr_epi8(1, 4, 7, 10, -1, -1, -1, -1,
                                       -1, -1, -1, -1, -1, -1, -1, -1);
  const __m128i shuf_b = _mm_setr_epi8(2, 5, 8, 11, -1, -1, -1, -1,
                                       -1, -1, -1, -1, -1, -1, -1, -1);
  const auto load4 = [&](const std::uint8_t* src, const __m128i& shuf) {
    return _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi8(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src)), shuf)));
  };
  for (int y = 0; y < height; ++y) {
    const auto* src = rgb + (std::size_t(top + y) * source_width + left) * 3;
    float* blue = dst + std::size_t(y) * row_width;
    float* green = blue + plane;
    float* red = green + plane;
    int x = 0;
    for (; x + 16 <= width; x += 16) {
      alignas(16) std::uint8_t pack[64]{};
      std::memcpy(pack, src + std::size_t(x) * 3, 48);
      const __m512 vr = _mm512_insertf32x4(
          _mm512_insertf32x4(
              _mm512_insertf32x4(_mm512_castps128_ps512(load4(pack, shuf_r)),
                                 load4(pack + 12, shuf_r), 1),
              load4(pack + 24, shuf_r), 2),
          load4(pack + 36, shuf_r), 3);
      const __m512 vg = _mm512_insertf32x4(
          _mm512_insertf32x4(
              _mm512_insertf32x4(_mm512_castps128_ps512(load4(pack, shuf_g)),
                                 load4(pack + 12, shuf_g), 1),
              load4(pack + 24, shuf_g), 2),
          load4(pack + 36, shuf_g), 3);
      const __m512 vb = _mm512_insertf32x4(
          _mm512_insertf32x4(
              _mm512_insertf32x4(_mm512_castps128_ps512(load4(pack, shuf_b)),
                                 load4(pack + 12, shuf_b), 1),
              load4(pack + 24, shuf_b), 2),
          load4(pack + 36, shuf_b), 3);
      _mm512_storeu_ps(blue + x, _mm512_add_ps(_mm512_mul_ps(vb, scale_b), shift_b));
      _mm512_storeu_ps(green + x, _mm512_add_ps(_mm512_mul_ps(vg, scale_g), shift_g));
      _mm512_storeu_ps(red + x, _mm512_add_ps(_mm512_mul_ps(vr, scale_r), shift_r));
    }
    for (; x + 8 <= width; x += 8) {
      alignas(16) std::uint8_t pack[32]{};
      std::memcpy(pack, src + std::size_t(x) * 3, 24);
      const __m256 vr = _mm256_set_m128(load4(pack + 12, shuf_r), load4(pack, shuf_r));
      const __m256 vg = _mm256_set_m128(load4(pack + 12, shuf_g), load4(pack, shuf_g));
      const __m256 vb = _mm256_set_m128(load4(pack + 12, shuf_b), load4(pack, shuf_b));
      _mm256_storeu_ps(blue + x, _mm256_add_ps(_mm256_mul_ps(vb, _mm256_set1_ps(scale[0])),
                                               _mm256_set1_ps(shift[0])));
      _mm256_storeu_ps(green + x, _mm256_add_ps(_mm256_mul_ps(vg, _mm256_set1_ps(scale[1])),
                                                _mm256_set1_ps(shift[1])));
      _mm256_storeu_ps(red + x, _mm256_add_ps(_mm256_mul_ps(vr, _mm256_set1_ps(scale[2])),
                                              _mm256_set1_ps(shift[2])));
    }
    for (; x < width; ++x) {
      const auto* pixel = src + std::size_t(x) * 3;
      blue[x] = static_cast<float>(pixel[2]) * scale[0] + shift[0];
      green[x] = static_cast<float>(pixel[1]) * scale[1] + shift[1];
      red[x] = static_cast<float>(pixel[0]) * scale[2] + shift[2];
    }
  }
}

void Avx512AveragePool3x2Valid(float* dst, const float* src, int first_plane,
                               int last_plane, int input_height,
                               int input_width) noexcept {
  const int output_height = (input_height - 3) / 3 + 1;
  const int output_width = (input_width - 2) / 2 + 1;
  const std::size_t input_plane = std::size_t(input_height) * input_width;
  const std::size_t output_plane = std::size_t(output_height) * output_width;
  const __m512i even = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14,
                                         16, 18, 20, 22, 24, 26, 28, 30);
  const __m512 six = _mm512_set1_ps(6.F);
  for (int plane = first_plane; plane < last_plane; ++plane) {
    const float* input = src + std::size_t(plane) * input_plane;
    float* output = dst + std::size_t(plane) * output_plane;
    for (int oy = 0; oy < output_height; ++oy) {
      const float* row0 = input + std::size_t(oy * 3) * input_width;
      const float* row1 = row0 + input_width;
      const float* row2 = row1 + input_width;
      float* row_out = output + std::size_t(oy) * output_width;
      int ox = 0;
      for (; ox + 16 <= output_width; ox += 16) {
        const float* base = row0 + ox * 2;
        const float* mid = row1 + ox * 2;
        const float* bot = row2 + ox * 2;
        __m512 sum = _mm512_i32gather_ps(even, base, 4);
        sum = _mm512_add_ps(sum, _mm512_i32gather_ps(even, base + 1, 4));
        sum = _mm512_add_ps(sum, _mm512_i32gather_ps(even, mid, 4));
        sum = _mm512_add_ps(sum, _mm512_i32gather_ps(even, mid + 1, 4));
        sum = _mm512_add_ps(sum, _mm512_i32gather_ps(even, bot, 4));
        sum = _mm512_add_ps(sum, _mm512_i32gather_ps(even, bot + 1, 4));
        _mm512_storeu_ps(row_out + ox, _mm512_div_ps(sum, six));
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

void Avx512ExpandGeluProjectAdd(float* dst, const float* src,
                                const float* expand_weights, const float* expand_bias,
                                const float* project_weights, const float* project_bias,
                                int channels, int hidden, std::size_t plane,
                                std::size_t spatial_begin,
                                std::size_t spatial_end) noexcept {
  if (!dst || !src || !expand_weights || !project_weights || channels <= 0 ||
      hidden <= 0 || spatial_begin >= spatial_end || spatial_end > plane) {
    return;
  }
  const __m512 half = _mm512_set1_ps(.5F);
  const __m512 one = _mm512_set1_ps(1.F);
  const __m512 inv_sqrt2 = _mm512_set1_ps(0.7071067811865475244F);
  thread_local std::vector<float> hidden_tile;
  hidden_tile.resize(std::size_t(hidden) * 16);
  const auto exact_gelu = [&](__m512 x) noexcept {
    return _mm512_mul_ps(half, _mm512_mul_ps(x,
        _mm512_add_ps(one, ErfPs512(_mm512_mul_ps(x, inv_sqrt2)))));
  };
  const auto expand_tile = [&](std::size_t spatial, __mmask16 mask) noexcept {
    int hidden_channel = 0;
    for (; hidden_channel + 4 <= hidden; hidden_channel += 4) {
      const float* e0 = expand_weights + std::size_t(hidden_channel) * channels;
      const float* e1 = e0 + channels;
      const float* e2 = e1 + channels;
      const float* e3 = e2 + channels;
      __m512 a0 = _mm512_set1_ps(expand_bias ? expand_bias[hidden_channel] : 0.F);
      __m512 a1 = _mm512_set1_ps(expand_bias ? expand_bias[hidden_channel + 1] : 0.F);
      __m512 a2 = _mm512_set1_ps(expand_bias ? expand_bias[hidden_channel + 2] : 0.F);
      __m512 a3 = _mm512_set1_ps(expand_bias ? expand_bias[hidden_channel + 3] : 0.F);
      for (int channel = 0; channel < channels; ++channel) {
        const __m512 x = _mm512_maskz_loadu_ps(
            mask, src + std::size_t(channel) * plane + spatial);
        a0 = _mm512_fmadd_ps(_mm512_set1_ps(e0[channel]), x, a0);
        a1 = _mm512_fmadd_ps(_mm512_set1_ps(e1[channel]), x, a1);
        a2 = _mm512_fmadd_ps(_mm512_set1_ps(e2[channel]), x, a2);
        a3 = _mm512_fmadd_ps(_mm512_set1_ps(e3[channel]), x, a3);
      }
      _mm512_storeu_ps(hidden_tile.data() + std::size_t(hidden_channel) * 16, exact_gelu(a0));
      _mm512_storeu_ps(hidden_tile.data() + std::size_t(hidden_channel + 1) * 16, exact_gelu(a1));
      _mm512_storeu_ps(hidden_tile.data() + std::size_t(hidden_channel + 2) * 16, exact_gelu(a2));
      _mm512_storeu_ps(hidden_tile.data() + std::size_t(hidden_channel + 3) * 16, exact_gelu(a3));
    }
    for (; hidden_channel < hidden; ++hidden_channel) {
      const float* filter = expand_weights + std::size_t(hidden_channel) * channels;
      __m512 acc = _mm512_set1_ps(expand_bias ? expand_bias[hidden_channel] : 0.F);
      for (int channel = 0; channel < channels; ++channel) {
        acc = _mm512_fmadd_ps(_mm512_set1_ps(filter[channel]),
            _mm512_maskz_loadu_ps(mask, src + std::size_t(channel) * plane + spatial), acc);
      }
      _mm512_storeu_ps(hidden_tile.data() + std::size_t(hidden_channel) * 16, exact_gelu(acc));
    }
    int channel = 0;
    for (; channel + 4 <= channels; channel += 4) {
      const float* p0 = project_weights + std::size_t(channel) * hidden;
      const float* p1 = p0 + hidden;
      const float* p2 = p1 + hidden;
      const float* p3 = p2 + hidden;
      __m512 a0 = _mm512_add_ps(
          _mm512_set1_ps(project_bias ? project_bias[channel] : 0.F),
          _mm512_maskz_loadu_ps(mask, src + std::size_t(channel) * plane + spatial));
      __m512 a1 = _mm512_add_ps(
          _mm512_set1_ps(project_bias ? project_bias[channel + 1] : 0.F),
          _mm512_maskz_loadu_ps(mask, src + std::size_t(channel + 1) * plane + spatial));
      __m512 a2 = _mm512_add_ps(
          _mm512_set1_ps(project_bias ? project_bias[channel + 2] : 0.F),
          _mm512_maskz_loadu_ps(mask, src + std::size_t(channel + 2) * plane + spatial));
      __m512 a3 = _mm512_add_ps(
          _mm512_set1_ps(project_bias ? project_bias[channel + 3] : 0.F),
          _mm512_maskz_loadu_ps(mask, src + std::size_t(channel + 3) * plane + spatial));
      for (int h = 0; h < hidden; ++h) {
        const __m512 g = _mm512_loadu_ps(hidden_tile.data() + std::size_t(h) * 16);
        a0 = _mm512_fmadd_ps(_mm512_set1_ps(p0[h]), g, a0);
        a1 = _mm512_fmadd_ps(_mm512_set1_ps(p1[h]), g, a1);
        a2 = _mm512_fmadd_ps(_mm512_set1_ps(p2[h]), g, a2);
        a3 = _mm512_fmadd_ps(_mm512_set1_ps(p3[h]), g, a3);
      }
      _mm512_mask_storeu_ps(dst + std::size_t(channel) * plane + spatial, mask, a0);
      _mm512_mask_storeu_ps(dst + std::size_t(channel + 1) * plane + spatial, mask, a1);
      _mm512_mask_storeu_ps(dst + std::size_t(channel + 2) * plane + spatial, mask, a2);
      _mm512_mask_storeu_ps(dst + std::size_t(channel + 3) * plane + spatial, mask, a3);
    }
    for (; channel < channels; ++channel) {
      const float* filter = project_weights + std::size_t(channel) * hidden;
      __m512 acc = _mm512_add_ps(
          _mm512_set1_ps(project_bias ? project_bias[channel] : 0.F),
          _mm512_maskz_loadu_ps(mask, src + std::size_t(channel) * plane + spatial));
      for (int h = 0; h < hidden; ++h) {
        acc = _mm512_fmadd_ps(_mm512_set1_ps(filter[h]),
            _mm512_loadu_ps(hidden_tile.data() + std::size_t(h) * 16), acc);
      }
      _mm512_mask_storeu_ps(dst + std::size_t(channel) * plane + spatial, mask, acc);
    }
  };
  std::size_t spatial = spatial_begin;
  for (; spatial + 16 <= spatial_end; spatial += 16) expand_tile(spatial, 0xFFFF);
  if (spatial < spatial_end) {
    const unsigned width = static_cast<unsigned>(spatial_end - spatial);
    expand_tile(spatial, static_cast<__mmask16>((1u << width) - 1u));
  }
}

void Avx512LayerNormAffine(float* dst, const float* src, const float* gamma,
                           const float* beta, std::size_t width, float mean,
                           float denom) noexcept {
  const __m512 mean_v = _mm512_set1_ps(mean);
  const __m512 denom_v = _mm512_set1_ps(denom);
  std::size_t column = 0;
  for (; column + 16 <= width; column += 16) {
    __m512 value = _mm512_div_ps(
        _mm512_sub_ps(_mm512_loadu_ps(src + column), mean_v), denom_v);
    value = _mm512_fmadd_ps(value, _mm512_loadu_ps(gamma + column),
                            _mm512_loadu_ps(beta + column));
    _mm512_storeu_ps(dst + column, value);
  }
  for (; column < width; ++column) {
    dst[column] = ((src[column] - mean) / denom) * gamma[column] + beta[column];
  }
}

}  // namespace ppocr::detail::kernels
