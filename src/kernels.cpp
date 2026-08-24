#include "kernels.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace ppocr::detail::kernels {

thread_local int g_inner_parallelism_cap = 0;

void SetInnerParallelismCap(int cap) noexcept {
  g_inner_parallelism_cap = cap < 0 ? 0 : cap;
}

#if defined(PPOCR_HAS_AVX2_KERNELS)
void Avx2Binary(float* dst, const float* a, const float* b, std::size_t n,
                BinaryOp op) noexcept;
void Avx2WidenU8ToFloat(float* dst, const std::uint8_t* src, std::size_t n) noexcept;
void Avx2NearestResize2xAdd(float*, const float*, const float*, int, int, int, int) noexcept;
int Avx2ArgMax(const float* values, int count) noexcept;
void Avx2Relu(float* dst, const float* src, std::size_t n) noexcept;
void Avx2Sigmoid(float* dst, const float* src, std::size_t n) noexcept;
void Avx2Gelu(float* dst, const float* src, std::size_t n) noexcept;
void Avx2ExactGelu(float* dst, const float* src, std::size_t n) noexcept;
void Avx2BatchNormGelu(float* dst, const float* src, std::size_t n, float scale,
                       float shift) noexcept;
void Avx2HardSwish(float* dst, const float* src, std::size_t n) noexcept;
void Avx2HardSigmoid(float* dst, const float* src, std::size_t n, float, float) noexcept;
void Avx2ScaleShift(float* dst, const float* src, std::size_t n, float scale,
                    float shift) noexcept;
void Avx2Axpy(float* dst, const float* src, float alpha, std::size_t n) noexcept;
void Avx2PointwiseConv4(float* dst, const float* src, const float* weights,
                        const float* bias, int first_output, int last_output,
                        int input_channels, std::size_t plane) noexcept;
void Avx2PointwiseConvAdd4(float* dst, const float* src, const float* weights,
                           const float* bias, const float* residual,
                           int first_output, int last_output, int input_channels,
                           std::size_t plane) noexcept;
void Avx2PointwiseConvAddRelu4(float* dst, const float* src, const float* weights,
                               const float* bias, const float* residual,
                               int first_output, int last_output, int input_channels,
                               std::size_t plane) noexcept;
void Avx2GemmRows(float* dst, const float* a, const float* b, const float* bias,
                  int first_row, int last_row, int cols, int depth) noexcept;
void Avx2GemmAccumulateRows(float* dst, const float* a, const float* b,
                            int first_row, int last_row, int cols, int depth) noexcept;
void Avx2BinaryScalar(float* dst, const float* src, std::size_t n, float scalar,
                      BinaryOp op, bool scalar_left) noexcept;
void Avx2Square(float* dst, const float* src, std::size_t n) noexcept;
void Avx2DepthwiseConv(float* dst, const float* src, const float* weights,
                       const float* bias, int first_channel, int last_channel,
                       int input_h, int input_w, int output_h, int output_w,
                       int kernel_h, int kernel_w, int pad_top, int pad_left) noexcept;
void Avx2SpatialMean(float* dst, const float* src, std::size_t planes,
                     std::size_t spatial) noexcept;
void Avx2MaxPool2x2Same(float* dst, const float* src, int first_plane,
                        int last_plane, int height, int width) noexcept;
void Avx2MaxPool2x2Valid(float* dst, const float* src, int first_plane,
                         int last_plane, int height, int width) noexcept;
void Avx2ConvTranspose2x2(float* dst, const float* src, const float* weights, const float* bias,
                           int first_output, int last_output, int input_channels,
                           int output_channels, int input_h, int input_w) noexcept;
void Avx2Conv2d(float* dst, const float* src, const float* weights,
                const float* bias, int first_output, int last_output,
                int input_channels, int input_h, int input_w, int output_h,
                int output_w, int kernel_h, int kernel_w, int pad_top,
                int pad_left) noexcept;
void Avx2Conv2x2Valid(float* dst, const float* src, const float* weights,
                      const float* bias, int first_output, int last_output,
                      int input_channels, int input_h, int input_w, bool relu) noexcept;
void Avx2Conv3x3Stride1x4(float* dst, const float* src, const float* weights,
                          const float* bias, int first_output, int last_output,
                          int input_channels, int input_h, int input_w,
                          int output_h, int output_w, int pad_top,
                          int pad_left, bool relu) noexcept;
void Avx2Conv3x3Stride2(float* dst, const float* src, const float* weights,
                         const float* bias, int first_output, int last_output,
                         int input_channels, int input_h, int input_w,
                         int output_h, int output_w, int pad_top,
                         int pad_left, bool relu) noexcept;
void Avx2Conv3x3Stride2x4(float* dst, const float* src, const float* weights,
                           const float* bias, int first_output, int last_output,
                           int input_channels, int input_h, int input_w,
                           int output_h, int output_w, int pad_top,
                           int pad_left, bool relu) noexcept;
void Avx2WriteIdentityRgbToNchw(float* dst, const std::uint8_t* rgb, int width,
                                int height, int source_width, int left, int top,
                                const float* scale, const float* shift,
                                int row_width) noexcept;
void Avx2WriteBilinearRgbToNchw(float* dst, const std::uint8_t* rgb, int image_width,
                                int left, int top, int source_width, int source_height,
                                int width, int height, int row_width, const int* x0,
                                const int* x1, const float* dx, const float* scale,
                                const float* shift, int first_row, int last_row) noexcept;
void Avx2AveragePool3x2Valid(float* dst, const float* src, int first_plane,
                             int last_plane, int input_height,
                             int input_width) noexcept;
void Avx2LayerNormAffine(float* dst, const float* src, const float* gamma,
                         const float* beta, std::size_t width, float mean,
                         float denom) noexcept;
#endif

#if defined(PPOCR_HAS_AVX512_KERNELS)
void Avx512Binary(float*, const float*, const float*, std::size_t, BinaryOp) noexcept;
void Avx512WidenU8ToFloat(float*, const std::uint8_t*, std::size_t) noexcept;
void Avx512NearestResize2xAdd(float*, const float*, const float*, int, int, int, int) noexcept;
int Avx512ArgMax(const float*, int) noexcept;
void Avx512Relu(float*, const float*, std::size_t) noexcept;
void Avx512Sigmoid(float*, const float*, std::size_t) noexcept;
void Avx512Gelu(float*, const float*, std::size_t) noexcept;
void Avx512ExactGelu(float*, const float*, std::size_t) noexcept;
void Avx512BatchNormGelu(float*, const float*, std::size_t, float, float) noexcept;
void Avx512HardSwish(float*, const float*, std::size_t) noexcept;
void Avx512HardSigmoid(float*, const float*, std::size_t, float, float) noexcept;
void Avx512ScaleShift(float*, const float*, std::size_t, float, float) noexcept;
void Avx512BinaryScalar(float*, const float*, std::size_t, float, BinaryOp, bool) noexcept;
void Avx512Square(float*, const float*, std::size_t) noexcept;
void Avx512Axpy(float*, const float*, float, std::size_t) noexcept;
void Avx512PointwiseConv4(float*, const float*, const float*, const float*, int, int, int, std::size_t) noexcept;
void Avx512PointwiseConv8(float*, const float*, const float*, const float*, int, int, int, std::size_t) noexcept;
void Avx512PointwiseConv16(float*, const float*, const float*, const float*, int, int, int, std::size_t) noexcept;
void Avx512PointwiseConv16Range(float*, const float*, const float*, const float*, int, int, int,
                                std::size_t, std::size_t, std::size_t) noexcept;
void Avx512PointwiseConvAdd4(float*, const float*, const float*, const float*, const float*, int, int, int, std::size_t) noexcept;
void Avx512PointwiseConvAdd8(float*, const float*, const float*, const float*, const float*, int, int, int, std::size_t) noexcept;
void Avx512PointwiseConvAddRelu4(float*, const float*, const float*, const float*, const float*, int, int, int, std::size_t) noexcept;
void Avx512PointwiseConvAddRelu8(float*, const float*, const float*, const float*, const float*, int, int, int, std::size_t) noexcept;
void Avx512PointwiseConvRelu4(float*, const float*, const float*, const float*, int, int, int, std::size_t) noexcept;
void Avx512PointwiseConvRelu8(float*, const float*, const float*, const float*, int, int, int, std::size_t) noexcept;
void Avx512ConvTranspose2x2(float*, const float*, const float*, const float*, int, int, int, int, int, int) noexcept;
void Avx512ConvTranspose2x2x4(float*, const float*, const float*, const float*, int, int, int, int, int, int) noexcept;
void Avx512ConvTranspose2x2Chain16x16x1(float*, const float*, const float*, const float*,
                                        const float*, const float*, int, int) noexcept;
void Avx512GemmRows(float*, const float*, const float*, const float*, int, int, int, int) noexcept;
void Avx512GemmPanelLdb(float*, const float*, const float*, const float*, int, int, int, int) noexcept;
void Avx512GemmPacked32(float*, const float*, const float*, const float*, int, int, int,
                        int* = nullptr, float* = nullptr, float* = nullptr) noexcept;
float Avx512SoftmaxDenom(const float*, int, float) noexcept;
void Avx512GemmAccumulateRows(float*, const float*, const float*, int, int, int, int) noexcept;
void Avx512DepthwiseConv(float*, const float*, const float*, const float*, int, int, int, int, int, int, int, int, int, int) noexcept;
void Avx512DepthwisePointwiseConv3x3S1(float*, const float*, const float*, const float*,
                                       const float*, const float*, int, int, int, int, int,
                                       bool) noexcept;
void Avx512DepthwisePointwiseConv5x5S1(float*, const float*, const float*, const float*,
                                       const float*, const float*, int, int, int, int, int,
                                       bool, int, int) noexcept;
void Avx512DepthwiseExpandGeluProjectAdd3x3S1(
    float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, int, int, int, int, int, int) noexcept;
void Avx512SpatialMean(float*, const float*, std::size_t, std::size_t) noexcept;
void Avx512MaxPool2x2Same(float*, const float*, int, int, int, int) noexcept;
void Avx512MaxPool2x2Valid(float*, const float*, int, int, int, int) noexcept;
void Avx512Conv2d(float*, const float*, const float*, const float*, int, int, int, int, int, int, int, int, int, int, int) noexcept;
void Avx512Conv2x2Valid(float*, const float*, const float*, const float*, int, int,
                        int, int, int, bool) noexcept;
void Avx512Conv2x2Validx4(float*, const float*, const float*, const float*, int, int,
                          int, int, int, bool) noexcept;
void Avx512Conv2x2SameUpper(float*, const float*, const float*, const float*, int, int,
                             int, int, int, bool, int = 0, int = -1) noexcept;
void Avx512Conv3x3Stride1x4(float*, const float*, const float*, const float*, int, int,
                            int, int, int, int, int, int, int, bool,
                            const float* const* = nullptr, int = -1, int = -1,
                            bool = false) noexcept;
void Avx512Conv3x3Stride1x16(float*, const float*, const float*, const float*, int, int,
                             int, int, int, int, int, int, int, bool, int = -1,
                             int = -1, bool = false, const float* = nullptr) noexcept;
void Avx512ConcatSourcesConv3x3x4(float*, const float* const*, int, const float*,
                                  const float*, int, int, int, int, int, int, int, int,
                                  bool, int, int) noexcept;
void Avx512Conv3x3Stride2C3(float*, const float*, const float*, const float*, int, int,
                            int, int, int, int, int, int, bool, bool, int = -1,
                            int = -1) noexcept;
void Avx512Conv3x3Stride1x8(float*, const float*, const float*, const float*, int, int,
                            int, int, int, int, int, int, int, bool,
                            const float* const* = nullptr) noexcept;
void Avx512ConvOddStride1x4(float*, const float*, const float*, const float*, int, int,
                             int, int, int, int, int, int, int, int) noexcept;
void Avx512ConvOddStride1x8(float*, const float*, const float*, const float*, int, int,
                             int, int, int, int, int, int, int, int) noexcept;
void Avx512ConvAsymmetricStride1x4(float*, const float*, const float*, const float*, int, int,
                                   int, int, int, int, int, int, int, int, int) noexcept;
void Avx512Conv3x3Stride2(float*, const float*, const float*, const float*, int, int,
                           int, int, int, int, int, int, int, bool, bool,
                           const float* const* = nullptr, int = -1, int = -1) noexcept;
void Avx512WriteIdentityRgbToNchw(float*, const std::uint8_t*, int, int, int, int, int,
                                  const float*, const float*, int) noexcept;
void Avx512AveragePool3x2Valid(float*, const float*, int, int, int, int) noexcept;
void Avx512LayerNormAffine(float*, const float*, const float*, const float*,
                           std::size_t, float, float) noexcept;
#endif

namespace {

bool HasAvx2() noexcept {
#if defined(PPOCR_HAS_AVX2_KERNELS)
  // ISA availability and OS XSAVE state are process-invariant. PP-OCR's
  // compact graph invokes this dispatcher hundreds of times per image, so do
  // the CPUID/XCR0 probe only once instead of repeating it for every kernel.
  static const bool available = [] {
#if defined(_MSC_VER)
  int info[4]{}; __cpuidex(info, 1, 0);
  if ((info[2] & (1 << 27)) == 0 || (info[2] & (1 << 28)) == 0) return false;
  if ((_xgetbv(0) & 0x6) != 0x6) return false;
  __cpuidex(info, 7, 0); return (info[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
  return false;
#endif
  }();
  return available;
#else
  return false;
#endif
}

bool HasAvx512() noexcept {
#if defined(PPOCR_HAS_AVX512_KERNELS)
  // Allow a deployment to opt out while preserving the same binary and the
  // AVX2/scalar fallback.  This is useful on CPUs where AVX-512 downclocks.
  static const bool available = [] {
  if (std::getenv("PPOCR_DISABLE_AVX512") != nullptr) return false;
#if defined(_MSC_VER)
  int info[4]{}; __cpuidex(info, 1, 0);
  if ((info[2] & (1 << 27)) == 0 || (info[2] & (1 << 28)) == 0) return false;
  if ((_xgetbv(0) & 0xe6) != 0xe6) return false;
  __cpuidex(info, 7, 0); return (info[1] & (1 << 16)) != 0 && (info[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
  return false;
#endif
  }();
  return available;
#else
  return false;
#endif
}

bool UseAvx512Pointwise8(int output_channels, int input_channels,
                         std::size_t plane) noexcept {
#if defined(PPOCR_HAS_AVX512_KERNELS)
  // Eight-output tiles reuse each NCHW input vector across twice as many
  // channels as the four-output kernel. Tiny SE 1x1s stay on the four-way
  // path (plane==1). Detector FPN laterals such as Conv.60 (48->64 on a
  // 40x176-class map) were previously excluded by the 192x192 cutoff and
  // reloaded every input plane 16 times.
  static const bool enabled =
      std::getenv("PPOCR_DISABLE_AVX512_POINTWISE8") == nullptr;
  if (!enabled) return false;
  if (output_channels >= 192 && input_channels >= 192 && plane >= 256) return true;
  // Conv.4 is 16→32 on 40x176. The IC>=24 floor kept it on four-output tiles
  // that reread 16 source planes eight times. Rec 12xW stays under plane 2048.
  // `PPOCR_DISABLE_AVX512_POINTWISE8_IC16` restores IC>=24.
  // Conv.4 is 16→32 on 40x176. Disable with PPOCR_DISABLE_AVX512_POINTWISE8_IC16.
  static const bool ic16 =
      std::getenv("PPOCR_DISABLE_AVX512_POINTWISE8_IC16") == nullptr;
  if (ic16 && output_channels >= 32 && input_channels >= 16 && plane >= 2048)
    return true;
  return output_channels >= 32 && input_channels >= 24 && plane >= 512;
#else
  (void)output_channels;
  (void)input_channels;
  (void)plane;
  return false;
#endif
}

bool UseAvx512Pointwise16(int output_channels, int input_channels,
                          std::size_t plane) noexcept {
#if defined(PPOCR_HAS_AVX512_KERNELS)
  static const bool enabled =
      std::getenv("PPOCR_DISABLE_AVX512_POINTWISE16") == nullptr;
  // Conv.4_relu is 16→32 on 40x176. Sixteen-output tiles reread the 16-ch
  // source twice instead of four. Same-host 20-run missed versus Relu8
  // (try30 16.69/18.22 vs try20 14.91/15.03). Keep ENABLE-only.
  // `PPOCR_ENABLE_AVX512_POINTWISE16_OC32` turns it on.
  static const bool oc32 =
      std::getenv("PPOCR_ENABLE_AVX512_POINTWISE16_OC32") != nullptr &&
      std::getenv("PPOCR_DISABLE_AVX512_POINTWISE16_OC32") == nullptr;
  if (enabled && oc32 && output_channels >= 32 && input_channels >= 16 &&
      plane >= 2048u) {
    return true;
  }
  // Sixteen output accumulators still fit in the AVX-512 file. This is the
  // Conv.60-class FPN 1x1 (48->64, thousands of spatial points).
  // Conv.60/15 are 48–64 OC on 20x88 (plane 1760). The 2048 floor left them
  // on eight-output tiles. `PPOCR_DISABLE_AVX512_POINTWISE16_PLANE1536`
  // restores 2048.
  static const bool plane1536 =
      std::getenv("PPOCR_DISABLE_AVX512_POINTWISE16_PLANE1536") == nullptr;
  if (enabled && output_channels >= 48 && input_channels >= 32 &&
      plane >= (plane1536 ? 1536u : 2048u)) {
    return true;
  }
  // Conv.79 is 64→16 on 40x176. Four 4-output tiles reread 64 source planes.
  // One 16-wide walk keeps those planes in L2. This is not POINTWISE16_TINY
  // (IC>=16), which lost 8-run on 16→16 1x1s. Rec 12xW stays under plane 2048.
  // The cross-tier mixed-page recheck improved tiny/small/medium with this
  // shape gate enabled. `PPOCR_DISABLE_AVX512_POINTWISE16_NARROW` restores
  // the four-output implementation for deployment A/B.
  static const bool narrow =
      std::getenv("PPOCR_DISABLE_AVX512_POINTWISE16_NARROW") == nullptr;
  if (enabled && narrow && output_channels >= 16 && input_channels >= 48 &&
      plane >= 2048) {
    return true;
  }
  // Det 16-ch 40x176 1x1s previously ran four 4-output tiles and reread the
  // 16 source planes four times. One 16-wide serial walk keeps those planes
  // in L1. Rec 12xW (plane 912) stays on the four-output kernel. The
  // cross-tier mixed-page recheck improved all three model sizes with this
  // detector-only gate. `PPOCR_DISABLE_AVX512_POINTWISE16_TINY` restores
  // four-wide output tiles for deployment A/B.
  static const bool tiny =
      std::getenv("PPOCR_DISABLE_AVX512_POINTWISE16_TINY") == nullptr;
  return enabled && tiny && output_channels >= 16 && input_channels >= 16 &&
      plane >= 2048;
#else
  (void)output_channels;
  (void)input_channels;
  (void)plane;
  return false;
#endif
}

int Avx512PointwiseTile(int output_channels, int input_channels,
                        std::size_t plane) noexcept {
  if (UseAvx512Pointwise16(output_channels, input_channels, plane)) return 16;
  if (UseAvx512Pointwise8(output_channels, input_channels, plane)) return 8;
  return 4;
}

bool PointwiseWorkParallel(std::int64_t work, int groups, int output_channels,
                           std::size_t plane = 0) noexcept {
  // Rec 12xW 1x1s can exceed the 3M work floor (96×48×912) and would nest
  // ParallelFor under the two-crop rec pool. Same-host 8-run default missed
  // versus allowing overflow-pool inner tiles (`nopwser` 16.19 vs default
  // 17.23). Keep the skip ENABLE-only. Det 40x176 stays eligible.
  // `PPOCR_ENABLE_AVX512_REC_PW_SERIAL` restores serial rec-sized 1x1s.
  static const bool rec_serial =
      std::getenv("PPOCR_ENABLE_AVX512_REC_PW_SERIAL") != nullptr &&
      std::getenv("PPOCR_DISABLE_AVX512_REC_PW_SERIAL") == nullptr;
  if (rec_serial && plane > 0 && plane < 2048) return false;
  if (work < std::int64_t{2000000} || groups <= 1) return false;
  return output_channels >= 32;
}

bool UseAvx512PointwiseRelu8(int output_channels, int input_channels,
                             std::size_t plane) noexcept {
#if defined(PPOCR_HAS_AVX512_KERNELS)
  // The four-filter kernel is the production default.  Although eight
  // filters can reduce input traffic, it also raises AVX-512 register/freq
  // pressure and did not produce a stable end-to-end win across all three
  // PP-OCRv6 tiers on the target client CPU. Retain it as an explicit
  // deployment experiment instead of silently regressing tiny/small.
  // Conv.4_relu is 16→32 on 40x176. Eight-output tiles reread the 16-ch
  // source four times instead of eight. Rec maps stay under plane 2048.
  // `PPOCR_DISABLE_AVX512_POINTWISE_RELU8_DET` restores four-output tiles.
  // Conv.4_relu is 16→32 on 40x176. Disable with
  // PPOCR_DISABLE_AVX512_POINTWISE_RELU8_DET.
  static const bool det16 =
      std::getenv("PPOCR_DISABLE_AVX512_POINTWISE_RELU8_DET") == nullptr;
  if (det16 && output_channels >= 32 && input_channels >= 16 && plane >= 2048)
    return true;
  static const bool enabled =
      std::getenv("PPOCR_ENABLE_AVX512_POINTWISE_RELU8") != nullptr &&
      std::getenv("PPOCR_DISABLE_AVX512_POINTWISE_RELU8") == nullptr;
  return enabled && output_channels >= 128 && input_channels >= 256 && plane >= 256;
#else
  (void)output_channels;
  (void)input_channels;
  (void)plane;
  return false;
#endif
}

bool UseAvx512Stride2PackedLoads() noexcept {
#if defined(PPOCR_HAS_AVX512_KERNELS)
  // AVX-512F can turn a pair of contiguous row loads into the sixteen
  // stride-two samples required by a 3x3 window.  Keep a deployment A/B
  // switch because gather-versus-permute throughput can differ on unusual
  // AVX-512 microarchitectures; disabling it retains the proven gather path.
  static const bool enabled = std::getenv("PPOCR_DISABLE_AVX512_STRIDE2_PACKED") == nullptr;
  return enabled;
#else
  return false;
#endif
}

bool UseAvx512Conv3x3Tile8() noexcept {
#if defined(PPOCR_HAS_AVX512_KERNELS)
  // Eight 3x3 output accumulators can share every input load, but the larger
  // tile may increase register pressure on some AVX-512 CPUs. Keep the proven
  // four-channel kernel as a runtime A/B fallback.
  // Eight live accumulators reduce input loads but cause register spills and
  // AVX-512 frequency pressure on the measured client CPU. The four-output
  // kernel is consistently faster for PP-OCRv6's 32/64-channel detector
  // maps, so make the larger tile an explicit deployment experiment rather
  // than a default. `PPOCR_ENABLE_AVX512_CONV3X3_TILE8=1` restores it.
  static const bool enabled = std::getenv("PPOCR_ENABLE_AVX512_CONV3X3_TILE8") != nullptr &&
      std::getenv("PPOCR_DISABLE_AVX512_CONV3X3_TILE8") == nullptr;
  return enabled;
#else
  return false;
#endif
}

bool UseAvx512ConvTransposeTile4(int output_channels = 0, int input_h = 0,
                                 int input_w = 0) noexcept {
#if defined(PPOCR_HAS_AVX512_KERNELS)
  // Medium mixed-image A/B favoured the one-output tile (286.985 vs 296.744).
  // Tiny det FPN is 16-ch 40x176 / 80x352: four outputs share each expanded
  // input vector. Shape-gate that case; do not blanket-enable on medium.
  // `PPOCR_DISABLE_AVX512_TRANSPOSE2X2_TILE4` restores one-output.
  // `PPOCR_ENABLE_AVX512_TRANSPOSE2X2_TILE4` forces it on every shape.
  static const bool disabled =
      std::getenv("PPOCR_DISABLE_AVX512_TRANSPOSE2X2_TILE4") != nullptr;
  static const bool forced =
      std::getenv("PPOCR_ENABLE_AVX512_TRANSPOSE2X2_TILE4") != nullptr;
  if (disabled) return false;
  if (forced) return true;
  // Tiny-shape x4 was an 8-run miss versus one-output (16.21 vs 16.56).
  (void)output_channels;
  (void)input_h;
  (void)input_w;
  return false;
#else
  (void)output_channels;
  (void)input_h;
  (void)input_w;
  return false;
#endif
}

bool UseAvx512Conv2x2Tile4(int output_channels, int input_channels,
                           int input_h, int input_w) noexcept {
#if defined(PPOCR_HAS_AVX512_KERNELS)
  // Reconstruction 2x2 valid convolutions have contiguous output windows.
  // Four accumulators reuse every input vector over the medium detector's
  // [64,32,2,2] and [32,64,2,2] reconstruction pair. A renewed 16-worker
  // AVX-512 A/B on both text and dense-page inputs made that tile a net win,
  // so make the already narrow, medium-only shape predicate the default.
  // Retain a deployment A/B escape hatch for CPUs where AVX-512 frequency
  // behaviour differs materially from the measured target.
  static const bool enabled =
      std::getenv("PPOCR_DISABLE_AVX512_CONV2X2_TILE4") == nullptr;
  return enabled && output_channels >= 4 && input_channels >= 8 && input_h >= 16 && input_w >= 16;
#else
  (void)output_channels;
  (void)input_channels;
  (void)input_h;
  (void)input_w;
  return false;
#endif
}

bool UseAvx512ConvWideTile8(int output_channels, int input_channels,
                            int input_h, int input_w, int kernel) noexcept {
#if defined(PPOCR_HAS_AVX512_KERNELS)
  // The eight-output form trades input reuse for much higher register
  // pressure. On the current PP-OCRv6 medium detector its four-output AVX512
  // sibling wins on both normal and low-contrast images. Retain the broad
  // tile as a positive opt-in for a server CPU that proves otherwise.
  static const bool enabled =
      std::getenv("PPOCR_ENABLE_AVX512_CONV_WIDE8") != nullptr &&
      std::getenv("PPOCR_DISABLE_AVX512_CONV_WIDE8") == nullptr;
  return enabled && output_channels >= 8 && input_channels >= 32 && input_h >= 32 &&
         input_w >= 32 && (kernel == 5 || kernel == 7);
#else
  (void)output_channels;
  (void)input_channels;
  (void)input_h;
  (void)input_w;
  (void)kernel;
  return false;
#endif
}

constexpr bool HasNeon() noexcept {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  return true;
#else
  return false;
#endif
}

int RequestedParallelism() noexcept {
  const unsigned hw = std::thread::hardware_concurrency();
  // A persistent worker set amortizes launch cost.  Larger detector/medium
  // convolutions can scale past eight workers on some servers, but PP-OCR's
  // small-image graph has many short channel tiles.  On the local AVX-512
  // client CPU, twelve workers already saturated the memory/cache bandwidth;
  // sixteen added scheduling pressure and regressed tiny/small/medium. Keep
  // eight as the portable baseline: this also leaves enough cores for the
  // bounded outer crop scheduler on dense pages. The explicit deployment
  // override below remains authoritative.
  // The default keeps enough workers for wide detector channels. Workloads
  // that have measured a different sweet spot can use PPOCR_BENCH_THREADS;
  // the public default must remain throughput-neutral for all three model
  // tiers rather than embedding one host-specific benchmark setting.
  int result = std::min(16, static_cast<int>(hw == 0 ? 4 : hw));
  if (const char* configured = std::getenv("PPOCR_BENCH_THREADS")) {
    char* end = nullptr;
    const long parsed = std::strtol(configured, &end, 10);
    if (end != configured && *end == '\0' && parsed > 0 && parsed <= 1024) {
      result = static_cast<int>(parsed);
    }
  }
  return result;
}

// Direct convolutions and reductions are short, numerous jobs. Creating and
// joining native threads for every one of them caused long scheduler outliers
// on Windows. Keep a process-lifetime worker set instead. The executor has a
// single active job by design: a PP-OCR graph is sequential at this level, and
// serialising rare concurrent callers prevents nested graph parallelism from
// multiplying the configured CPU budget.
thread_local bool g_parallel_worker = false;

template <class Fn>
void OverflowParallelFor(int count, Fn&& fn);

class PersistentParallelExecutor {
 public:
  explicit PersistentParallelExecutor(int workers) : workers_(std::max(1, workers)) {
    threads_.reserve(static_cast<std::size_t>(workers_ - 1));
    for (int worker = 1; worker < workers_; ++worker) {
      threads_.emplace_back([this, worker] { WorkerLoop(worker); });
    }
  }

  template <class Fn>
  void Run(int count, Fn&& fn, bool allow_overflow = true) {
    if (count <= 1 || workers_ <= 1 || g_parallel_worker) {
      fn(0, count);
      return;
    }
    // Only one caller installs work at a time on this pool. A second
    // recognizer batch used to fall back to a fully serial inner loop, which
    // made the slower crop the end-to-end wall. Overflow uses a smaller
    // persistent pool so two independent graphs can both keep SIMD tiles.
    std::unique_lock submit_lock(submit_mutex_, std::try_to_lock);
    if (!submit_lock.owns_lock()) {
      if (allow_overflow) OverflowParallelFor(count, fn);
      else fn(0, count);
      return;
    }
    int pool = workers_;
    if (g_inner_parallelism_cap > 0)
      pool = std::min(pool, std::max(1, g_inner_parallelism_cap));
    const int active_workers = std::min(count, pool);
    if (active_workers <= 1) {
      fn(0, count);
      return;
    }
    const int block = (count + active_workers - 1) / active_workers;
    {
      std::lock_guard lock(mutex_);
      count_ = count;
      block_ = block;
      participants_ = active_workers - 1;
      active_.store(participants_, std::memory_order_relaxed);
      task_ = [&fn](int first, int last) { fn(first, last); };
      generation_.fetch_add(1, std::memory_order_release);
    }
    work_ready_.notify_all();
    fn(0, std::min(count, block));
    static const bool spin =
        std::getenv("PPOCR_ENABLE_PARALLEL_SPIN") != nullptr &&
        std::getenv("PPOCR_DISABLE_PARALLEL_SPIN") == nullptr;
    if (spin) {
      for (int i = 0; i < 4096; ++i) {
        if (active_.load(std::memory_order_acquire) == 0) break;
#if defined(_MSC_VER)
        _mm_pause();
#else
        std::this_thread::yield();
#endif
      }
    }
    std::unique_lock lock(mutex_);
    complete_.wait(lock, [this] { return active_.load(std::memory_order_relaxed) == 0; });
    task_ = {};
  }

 private:
  void WorkerLoop(int worker) {
    g_parallel_worker = true;
    std::uint64_t observed_generation = 0;
    static const bool spin =
        std::getenv("PPOCR_ENABLE_PARALLEL_SPIN") != nullptr &&
        std::getenv("PPOCR_DISABLE_PARALLEL_SPIN") == nullptr;
    for (;;) {
      if (spin) {
        for (int i = 0; i < 4096; ++i) {
          if (generation_.load(std::memory_order_acquire) != observed_generation)
            break;
#if defined(_MSC_VER)
          _mm_pause();
#else
          std::this_thread::yield();
#endif
        }
      }
      std::function<void(int, int)> task;
      int first = 0;
      int last = 0;
      {
        std::unique_lock lock(mutex_);
        work_ready_.wait(lock, [&] {
          return generation_.load(std::memory_order_relaxed) != observed_generation;
        });
        observed_generation = generation_.load(std::memory_order_relaxed);
        if (worker > participants_) continue;
        first = worker * block_;
        last = std::min(count_, first + block_);
        task = task_;
      }
      if (first < last) task(first, last);
      {
        std::lock_guard lock(mutex_);
        if (active_.fetch_sub(1, std::memory_order_acq_rel) == 1)
          complete_.notify_one();
      }
    }
  }

  const int workers_;
  std::vector<std::thread> threads_;
  std::mutex submit_mutex_;
  std::mutex mutex_;
  std::condition_variable work_ready_;
  std::condition_variable complete_;
  std::function<void(int, int)> task_;
  std::atomic<std::uint64_t> generation_{0};
  int count_ = 0;
  int block_ = 0;
  int participants_ = 0;
  std::atomic<int> active_{0};
};

PersistentParallelExecutor& ParallelExecutor() {
  // Deliberately leak the executor. It is used by static-library clients and
  // has no reliable shutdown ordering relative to inference objects; Windows
  // reclaims its worker threads and memory at process exit.
  static auto* executor = new PersistentParallelExecutor(RequestedParallelism());
  return *executor;
}

PersistentParallelExecutor& OverflowExecutor() {
  int overflow = RequestedParallelism();
  if (const char* configured = std::getenv("PPOCR_OVERFLOW_THREADS")) {
    char* end = nullptr;
    const long parsed = std::strtol(configured, &end, 10);
    if (end != configured && *end == '\0' && parsed > 0 && parsed <= 1024)
      overflow = static_cast<int>(parsed);
  }
  static auto* executor = new PersistentParallelExecutor(std::max(2, overflow));
  return *executor;
}

template <class Fn>
void OverflowParallelFor(int count, Fn&& fn) {
  static const bool disabled =
      std::getenv("PPOCR_DISABLE_PARALLEL_OVERFLOW") != nullptr;
  if (disabled) {
    fn(0, count);
    return;
  }
  OverflowExecutor().Run(count, std::forward<Fn>(fn), false);
}

template <class Fn>
void ParallelFor(int count, Fn&& fn) {
  ParallelExecutor().Run(count, std::forward<Fn>(fn));
}

void ScalarOrNeonGemmRows(float* dst, const float* a, const float* b, const float* bias,
                          int first_row, int last_row, int cols, int depth) noexcept {
  for (int row = first_row; row < last_row; ++row) {
    float* out = dst + std::size_t(row) * cols;
    // Bias is a column vector shared by every row.  The previous scalar/NEON
    // fallback accidentally broadcast `bias[row]`, which is both incorrect
    // for ARM deployments and prevents this path from being a valid oracle
    // for the x86 SIMD implementations.
    if (bias) std::copy_n(bias, cols, out);
    else std::fill_n(out, cols, 0.F);
    const float* lhs = a + std::size_t(row) * depth;
    for (int k = 0; k < depth; ++k) {
      const float alpha = lhs[k]; const float* rhs = b + std::size_t(k) * cols;
      int col = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
      if (HasNeon()) { const auto scale = vdupq_n_f32(alpha); for (; col + 4 <= cols; col += 4)
        vst1q_f32(out + col, vmlaq_f32(vld1q_f32(out + col), scale, vld1q_f32(rhs + col))); }
#endif
      for (; col < cols; ++col) out[col] += alpha * rhs[col];
    }
  }
}


void ScalarOrNeonGemmAccumulateRows(float* dst, const float* a, const float* b,
                                    int first_row, int last_row, int cols, int depth) noexcept {
  for (int row = first_row; row < last_row; ++row) {
    float* out = dst + std::size_t(row) * cols; const float* lhs = a + std::size_t(row) * depth;
    for (int k = 0; k < depth; ++k) {
      const float alpha = lhs[k]; const float* rhs = b + std::size_t(k) * cols;
      int col = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
      if (HasNeon()) { const auto scale = vdupq_n_f32(alpha); for (; col + 4 <= cols; col += 4)
        vst1q_f32(out + col, vmlaq_f32(vld1q_f32(out + col), scale, vld1q_f32(rhs + col))); }
#endif
      for (; col < cols; ++col) out[col] += alpha * rhs[col];
    }
  }
}

}  // namespace

void ParallelForRange(int count, const std::function<void(int, int)>& fn) {
  ParallelFor(count, fn);
}

void WriteIdentityRgbToNchw(float* dst, const std::uint8_t* rgb, int width,
                            int height, int source_width, int left, int top,
                            const float scale[3], const float shift[3],
                            int row_width) noexcept {
  if (!dst || !rgb || !scale || !shift || width <= 0 || height <= 0 ||
      source_width <= 0 || row_width < width || left < 0 || top < 0) return;
  static const bool simd_enabled =
      std::getenv("PPOCR_DISABLE_IDENTITY_RGB_SIMD") == nullptr;
#if defined(PPOCR_HAS_AVX2_KERNELS)
  // Identity RGB is store-bound; the AVX2 eight-pixel path matches the
  // scalar affine order without requiring AVX-512 VL/DQ encodings.
  if (simd_enabled && HasAvx2()) {
    Avx2WriteIdentityRgbToNchw(dst, rgb, width, height, source_width, left, top,
                               scale, shift, row_width);
    return;
  }
#endif
  const std::size_t plane = std::size_t(height) * row_width;
  for (int y = 0; y < height; ++y) {
    const auto* src = rgb + (std::size_t(top + y) * source_width + left) * 3;
    float* blue = dst + std::size_t(y) * row_width;
    float* green = blue + plane;
    float* red = green + plane;
    for (int x = 0; x < width; ++x, src += 3) {
      blue[x] = static_cast<float>(src[2]) * scale[0] + shift[0];
      green[x] = static_cast<float>(src[1]) * scale[1] + shift[1];
      red[x] = static_cast<float>(src[0]) * scale[2] + shift[2];
    }
  }
}

namespace {

void WriteBilinearRgbRowScalar(float* dst, const std::uint8_t* rgb, int image_width,
                               int left, int top, int source_width, int source_height,
                               int width, int height, int row_width, const int* x0,
                               const int* x1, const float* dx, const float* scale,
                               const float* shift, int y) noexcept {
  (void)source_width;
  const float fy = (float(y) + .5F) * source_height / height - .5F;
  const int y_floor = int(std::floor(fy));
  const int y0 = std::clamp(y_floor, 0, source_height - 1);
  const int y1 = std::min(y0 + 1, source_height - 1);
  const float dy = fy - y_floor;
  const float inverse_dy = 1.F - dy;
  const auto* const upper_row =
      rgb + (std::size_t(top + y0) * image_width + left) * 3;
  const auto* const lower_row =
      rgb + (std::size_t(top + y1) * image_width + left) * 3;
  const std::size_t plane = std::size_t(height) * row_width;
  float* const blue = dst + std::size_t(y) * row_width;
  float* const green = blue + plane;
  float* const red = green + plane;
  for (int x = 0; x < width; ++x) {
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

void WriteBilinearRgbRows(float* dst, const std::uint8_t* rgb, int image_width,
                          int left, int top, int source_width, int source_height,
                          int width, int height, int row_width, const int* x0,
                          const int* x1, const float* dx, const float* scale,
                          const float* shift, int first_row, int last_row) noexcept {
  // AVX-512 byte-gather / insertf32x4 widen took STATUS_ILLEGAL_INSTRUCTION
  // on this host. AVX2 uses the identity-RGB shuffle plus gather_ps, which
  // AveragePool already ships. ParallelFor still fans independent dest rows.
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2()) {
    Avx2WriteBilinearRgbToNchw(dst, rgb, image_width, left, top, source_width,
                               source_height, width, height, row_width, x0, x1, dx,
                               scale, shift, first_row, last_row);
    return;
  }
#endif
  for (int y = first_row; y < last_row; ++y) {
    WriteBilinearRgbRowScalar(dst, rgb, image_width, left, top, source_width,
                              source_height, width, height, row_width, x0, x1, dx,
                              scale, shift, y);
  }
}

}  // namespace

void WriteBilinearRgbToNchw(float* dst, const std::uint8_t* rgb, int image_width,
                            int image_height, int left, int top, int source_width,
                            int source_height, int width, int height,
                            const float scale[3], const float shift[3],
                            int row_width, bool parallel) noexcept {
  (void)image_height;
  if (!dst || !rgb || !scale || !shift || image_width <= 0 || source_width <= 0 ||
      source_height <= 0 || width <= 0 || height <= 0 || row_width < width ||
      left < 0 || top < 0) {
    return;
  }
  constexpr int kStackLimit = 3200;
  int stack_x0[kStackLimit];
  int stack_x1[kStackLimit];
  float stack_dx[kStackLimit];
  std::vector<int> heap_x0;
  std::vector<int> heap_x1;
  std::vector<float> heap_dx;
  int* x0 = stack_x0;
  int* x1 = stack_x1;
  float* dx = stack_dx;
  if (width > kStackLimit) {
    heap_x0.resize(static_cast<std::size_t>(width));
    heap_x1.resize(static_cast<std::size_t>(width));
    heap_dx.resize(static_cast<std::size_t>(width));
    x0 = heap_x0.data();
    x1 = heap_x1.data();
    dx = heap_dx.data();
  }
  for (int x = 0; x < width; ++x) {
    const float fx = (float(x) + .5F) * source_width / width - .5F;
    const int x_floor = int(std::floor(fx));
    x0[x] = std::clamp(x_floor, 0, source_width - 1);
    x1[x] = std::min(x0[x] + 1, source_width - 1);
    dx[x] = fx - x_floor;
  }
  static const bool simd_enabled =
      std::getenv("PPOCR_DISABLE_BILINEAR_RGB_SIMD") == nullptr;
  static const bool pf_enabled =
      std::getenv("PPOCR_DISABLE_BILINEAR_RGB_PF") == nullptr;
  const auto body = [&](int first, int last) {
    if (simd_enabled) {
      WriteBilinearRgbRows(dst, rgb, image_width, left, top, source_width,
                           source_height, width, height, row_width, x0, x1, dx,
                           scale, shift, first, last);
    } else {
      for (int y = first; y < last; ++y) {
        WriteBilinearRgbRowScalar(dst, rgb, image_width, left, top, source_width,
                                  source_height, width, height, row_width, x0, x1,
                                  dx, scale, shift, y);
      }
    }
  };
  // Detector pages are hundreds of independent output rows. Rec crops already
  // run under an outer ParallelFor and must keep this walk serial.
  if (parallel && pf_enabled && height >= 32 &&
      std::int64_t(width) * height >= 16384) {
    ParallelFor(height, body);
  } else {
    body(0, height);
  }
}

void WidenU8ToFloat(float* dst, const std::uint8_t* src, std::size_t n) noexcept {
  if (!dst || !src || n == 0) return;
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512()) {
    Avx512WidenU8ToFloat(dst, src, n);
    return;
  }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2()) {
    Avx2WidenU8ToFloat(dst, src, n);
    return;
  }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  std::size_t index = 0;
  for (; index + 16 <= n; index += 16) {
    const uint8x16_t bytes = vld1q_u8(src + index);
    const uint16x8_t low16 = vmovl_u8(vget_low_u8(bytes));
    const uint16x8_t high16 = vmovl_u8(vget_high_u8(bytes));
    const uint32x4_t low0 = vmovl_u16(vget_low_u16(low16));
    const uint32x4_t low1 = vmovl_u16(vget_high_u16(low16));
    const uint32x4_t high0 = vmovl_u16(vget_low_u16(high16));
    const uint32x4_t high1 = vmovl_u16(vget_high_u16(high16));
    vst1q_f32(dst + index, vcvtq_f32_u32(low0));
    vst1q_f32(dst + index + 4, vcvtq_f32_u32(low1));
    vst1q_f32(dst + index + 8, vcvtq_f32_u32(high0));
    vst1q_f32(dst + index + 12, vcvtq_f32_u32(high1));
  }
  for (; index < n; ++index) dst[index] = static_cast<float>(src[index]);
  return;
#else
  for (std::size_t index = 0; index < n; ++index) dst[index] = static_cast<float>(src[index]);
#endif
}

void Binary(float* dst, const float* a, const float* b, std::size_t n,
            BinaryOp op) noexcept {
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512()) { Avx512Binary(dst, a, b, n, op); return; }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2()) { Avx2Binary(dst, a, b, n, op); return; }
#endif
  std::size_t i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  if (HasNeon() && op != BinaryOp::div) for (; i + 4 <= n; i += 4) {
    const float32x4_t x = vld1q_f32(a + i), y = vld1q_f32(b + i);
    if (op == BinaryOp::add) vst1q_f32(dst + i, vaddq_f32(x, y));
    else if (op == BinaryOp::sub) vst1q_f32(dst + i, vsubq_f32(x, y));
    else vst1q_f32(dst + i, vmulq_f32(x, y));
  }
#endif
  for (; i < n; ++i) {
    if (op == BinaryOp::add) dst[i] = a[i] + b[i]; else if (op == BinaryOp::sub) dst[i] = a[i] - b[i];
    else if (op == BinaryOp::mul) dst[i] = a[i] * b[i]; else dst[i] = a[i] / b[i];
  }
}

void BinaryInplace(float* left, const float* right, std::size_t n,
                   BinaryOp op) noexcept {
  if (!left || !right || n == 0) return;
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512()) { Avx512Binary(left, left, right, n, op); return; }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2()) { Avx2Binary(left, left, right, n, op); return; }
#endif
  for (std::size_t i = 0; i < n; ++i) {
    switch (op) {
      case BinaryOp::add: left[i] += right[i]; break;
      case BinaryOp::sub: left[i] -= right[i]; break;
      case BinaryOp::mul: left[i] *= right[i]; break;
      case BinaryOp::div: left[i] /= right[i]; break;
    }
  }
}

void BinaryBroadcast(float* dst, const float* a, const float* b, std::size_t n,
                     std::size_t a_repeat, std::size_t b_repeat,
                     BinaryOp op) noexcept {
  // Contiguous equal tensors retain the dedicated SIMD implementation.
  if (a_repeat == 1 && b_repeat == 1) { Binary(dst, a, b, n, op); return; }
  // A common PP-OCRv6 squeeze-excitation operation is NCHW [N,C,H,W]
  // multiplied by [N,C,1,1].  The generic expression below performs an
  // integer divide for every pixel, even though the broadcast value is
  // constant for an entire H*W plane.  Process each contiguous run through
  // the scalar-broadcast kernel instead; this preserves the operation order
  // while enabling AVX-512/AVX2/NEON for the full plane.
  if (a_repeat == 1 && b_repeat > 1) {
    const std::size_t runs = (n + b_repeat - 1) / b_repeat;
    const auto body = [&](int first, int last) {
      for (int run = first; run < last; ++run) {
        const auto offset = std::size_t(run) * b_repeat;
        const auto count = std::min(b_repeat, n - offset);
        BinaryScalar(dst + offset, a + offset, count, b[run], op, false);
      }
    };
    // Squeeze-excitation gates broadcast one [N,C,1,1] coefficient over an
    // independent NCHW plane.  These planes have no recurrence or shared
    // writes, so a large detector map should not leave the existing SIMD
    // broadcast loop on one core. Keep small tensors local to avoid a pool
    // barrier for short recognizer maps.
    // Plane-parallel broadcast was A/B tested on the full PP-OCR graph. Its
    // extra pool barrier regressed ordinary pages and did not yield a stable
    // large-page gain once the surrounding convolutions already saturated
    // the cache. Keep the compact SIMD-per-plane loop as the default.
    body(0, static_cast<int>(runs));
    return;
  }
  // Symmetric contiguous case: a scalar/row/plane from the left operand is
  // broadcast over a full right run.  Sub/Div retain the required left-hand
  // operand ordering through `scalar_left`.
  if (b_repeat == 1 && a_repeat > 1) {
    const std::size_t runs = (n + a_repeat - 1) / a_repeat;
    const auto body = [&](int first, int last) {
      for (int run = first; run < last; ++run) {
        const auto offset = std::size_t(run) * a_repeat;
        const auto count = std::min(a_repeat, n - offset);
        BinaryScalar(dst + offset, b + offset, count, a[run], op, true);
      }
    };
    body(0, static_cast<int>(runs));
    return;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const float x = a[i / a_repeat], y = b[i / b_repeat];
    switch (op) {
      case BinaryOp::add: dst[i] = x + y; break;
      case BinaryOp::sub: dst[i] = x - y; break;
      case BinaryOp::mul: dst[i] = x * y; break;
      case BinaryOp::div: dst[i] = x / y; break;
    }
  }
}

void BinaryBroadcastRightInplace(float* left, const float* right, std::size_t n,
                                 std::size_t right_repeat, std::size_t right_elements,
                                 BinaryOp op) noexcept {
  if (right_repeat == 1) { Binary(left, left, right, n, op); return; }
  // NCHW channel broadcasts are contiguous runs (usually H*W). Work one run
  // at a time: this removes the div/mod pair from every element and lets the
  // existing scalar/SIMD dispatch operate on a broadcast scalar. Different
  // output planes are independent; parallelize the plane loop for the large
  // detector gates while retaining the same per-element arithmetic and an
  // in-place alias-safe source/destination contract.
  if (right_elements == 0) return;
  const std::size_t runs = (n + right_repeat - 1) / right_repeat;
  const auto body = [&](int first, int last) {
    for (int run = first; run < last; ++run) {
      const auto offset = std::size_t(run) * right_repeat;
      const auto count = std::min(right_repeat, n - offset);
      BinaryScalar(left + offset, left + offset, count,
                   right[std::size_t(run) % right_elements], op, false);
    }
  };
  static const bool parallel_planes =
      std::getenv("PPOCR_DISABLE_AVX512_BROADCAST_PARALLEL") == nullptr;
  if (parallel_planes && n >= 65536 && runs >= 8)
    ParallelFor(static_cast<int>(runs), body);
  else
    body(0, static_cast<int>(runs));
}

void NearestResize2xAdd(float* dst, const float* src, const float* residual,
                        int batches, int channels, int input_height,
                        int input_width) noexcept {
  if (!dst || !src || !residual || batches <= 0 || channels <= 0 ||
      input_height <= 0 || input_width <= 0) return;
  // Preserve a scalar A/B path for deployment verification. This controls
  // only the new FPN resize-add microkernel; it does not disable AVX for the
  // rest of PP-OCR inference.
  const bool simd_enabled = std::getenv("PPOCR_DISABLE_RESIZE2X_ADD_SIMD") == nullptr;
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (simd_enabled && HasAvx512()) {
    const int planes = batches * channels;
    const std::size_t input_plane = std::size_t(input_height) * input_width;
    const std::size_t output_plane = input_plane * 4;
    // Detector FPN Resize.2 is 16 independent 20x88→40x176 planes. Rec maps
    // do not use this kernel. Generic ENABLE missed e2e; default only the
    // 16-plane 20x88-class tail. `PPOCR_DISABLE_AVX512_RESIZE2X_PF` is serial.
    // `PPOCR_ENABLE_AVX512_RESIZE2X_PF` restores the broader predicate.
    static const bool disabled =
        std::getenv("PPOCR_DISABLE_AVX512_RESIZE2X_PF") != nullptr;
    static const bool forced =
        std::getenv("PPOCR_ENABLE_AVX512_RESIZE2X_PF") != nullptr;
    const bool plane_pf = !disabled &&
        (forced ? (planes >= 8 && input_height >= 16 && input_width >= 32)
                : (planes >= 16 && input_height >= 16 && input_width >= 64));
    if (plane_pf) {
      ParallelFor(planes, [&](int first, int last) {
        for (int plane = first; plane < last; ++plane) {
          Avx512NearestResize2xAdd(
              dst + std::size_t(plane) * output_plane,
              src + std::size_t(plane) * input_plane,
              residual + std::size_t(plane) * output_plane, 1, 1, input_height,
              input_width);
        }
      });
      return;
    }
    Avx512NearestResize2xAdd(dst, src, residual, batches, channels, input_height, input_width);
    return;
  }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (simd_enabled && HasAvx2()) {
    Avx2NearestResize2xAdd(dst, src, residual, batches, channels, input_height, input_width);
    return;
  }
#endif
  const int output_width = input_width * 2;
  const std::size_t input_plane = std::size_t(input_height) * input_width;
  const std::size_t output_plane = input_plane * 4;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  if (simd_enabled) if constexpr (HasNeon()) {
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
        for (; x + 4 <= input_width; x += 4) {
          const float32x4_t values = vld1q_f32(input_row + x);
          const float32x4x2_t repeated = vzipq_f32(values, values);
          const int offset = x * 2;
          vst1q_f32(output0 + offset, vaddq_f32(repeated.val[0], vld1q_f32(add0 + offset)));
          vst1q_f32(output0 + offset + 4, vaddq_f32(repeated.val[1], vld1q_f32(add0 + offset + 4)));
          vst1q_f32(output1 + offset, vaddq_f32(repeated.val[0], vld1q_f32(add1 + offset)));
          vst1q_f32(output1 + offset + 4, vaddq_f32(repeated.val[1], vld1q_f32(add1 + offset + 4)));
        }
        for (; x < input_width; ++x) {
          const float value = input_row[x]; const int offset = x * 2;
          output0[offset] = value + add0[offset]; output0[offset + 1] = value + add0[offset + 1];
          output1[offset] = value + add1[offset]; output1[offset + 1] = value + add1[offset + 1];
        }
      }
    }
    return;
  }
#endif
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
      for (int x = 0; x < input_width; ++x) {
        const float value = input_row[x]; const int offset = x * 2;
        output0[offset] = value + add0[offset]; output0[offset + 1] = value + add0[offset + 1];
        output1[offset] = value + add1[offset]; output1[offset + 1] = value + add1[offset + 1];
      }
    }
  }
}

bool ApproximateGeluEnabled() noexcept {
  // Exact Erf is the compatibility default because the imported ONNX graph
  // expresses GELU by that operation chain. The approximate path stays an
  // explicit deployment experiment and is further size-gated in Gelu(),
  // because tiny boundary inputs are sensitive to this replacement.
  return std::getenv("PPOCR_APPROX_GELU") != nullptr;
}

bool ApproximateGeluSelected(std::size_t logical_elements) noexcept {
  constexpr std::size_t kApproximateGeluMinElements = 262144;
  if (!ApproximateGeluEnabled() || logical_elements < kApproximateGeluMinElements ||
      std::getenv("PPOCR_DISABLE_APPROX_GELU_SIMD") != nullptr) return false;
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512()) return true;
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2()) return true;
#endif
  return false;
}

void Square(float* dst, const float* src, std::size_t n) noexcept {
  if (!dst || !src || n == 0) return;
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512()) { Avx512Square(dst, src, n); return; }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2()) { Avx2Square(dst, src, n); return; }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  if (HasNeon()) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
      const float32x4_t value = vld1q_f32(src + i);
      vst1q_f32(dst + i, vmulq_f32(value, value));
    }
    for (; i < n; ++i) dst[i] = src[i] * src[i];
    return;
  }
#endif
  for (std::size_t i = 0; i < n; ++i) dst[i] = src[i] * src[i];
}

void LayerNorm(float* dst, const float* src, const float* gamma, const float* beta,
               std::size_t rows, std::size_t width, float epsilon) noexcept {
  if (!dst || !src || !gamma || !beta || rows == 0 || width == 0) return;
  // Keep reduction and division order fixed; this deliberately avoids an
  // approximate reciprocal-square-root path so the fused graph tracks ONNX.
  const auto body = [&](int first, int last) {
    for (int row = first; row < last; ++row) {
      const float* input = src + std::size_t(row) * width;
      float* output = dst + std::size_t(row) * width;
      float mean = 0.F;
      for (std::size_t column = 0; column < width; ++column) mean += input[column];
      mean /= static_cast<float>(width);
      float variance = 0.F;
      for (std::size_t column = 0; column < width; ++column) {
        const float centered = input[column] - mean;
        variance += centered * centered;
      }
      const float denominator = std::sqrt(variance / static_cast<float>(width) + epsilon);
      static const bool simd_affine =
          std::getenv("PPOCR_DISABLE_LAYERNORM_SIMD") == nullptr;
#if defined(PPOCR_HAS_AVX512_KERNELS)
      if (simd_affine && HasAvx512() && width >= 16) {
        Avx512LayerNormAffine(output, input, gamma, beta, width, mean, denominator);
        continue;
      }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
      if (simd_affine && HasAvx2() && width >= 8) {
        Avx2LayerNormAffine(output, input, gamma, beta, width, mean, denominator);
        continue;
      }
#endif
      for (std::size_t column = 0; column < width; ++column) {
        output[column] = ((input[column] - mean) / denominator) * gamma[column] + beta[column];
      }
    }
  };
  const auto work = rows * width;
  if (work >= 65536 && rows >= 2) {
    ParallelFor(static_cast<int>(rows), body);
  } else {
    body(0, static_cast<int>(rows));
  }
}

void Gelu(float* dst, const float* src, std::size_t n,
          std::size_t logical_elements) noexcept {
  // Exact Erf is the compatibility default.  In the explicitly opt-in
  // approximate deployment mode, however, generic FusedGelu nodes must use
  // the same x86 vector implementation as BatchNorm+GELU.  Leaving these
  // nodes on scalar libm meant PPOCR_APPROX_GELU accelerated only a subset of
  // recognizer blocks and left the detector's independent GELUs as a visible
  // hotspot.  Do not size-gate this dispatch: every element follows the same
  // selected formula whether it is inferred alone or in an NCHW batch.
  // Tiny/small recognizers have decision-sensitive short GELU tensors; keep
  // their existing exact execution even when PPOCR_APPROX_GELU is enabled.
  // Medium's graph has the broad 256+ channel maps that amortize this vector
  // kernel and has separately passed its ONNX decode corpus. Derive this
  // decision from one logical sample rather than total NCHW storage, so a
  // page receives the same selected formula alone and inside a batch.
  if (logical_elements == 0 || logical_elements > n || n % logical_elements != 0) {
    logical_elements = n;
  }
  if (ApproximateGeluSelected(logical_elements)) {
#if defined(PPOCR_HAS_AVX512_KERNELS)
    if (HasAvx512()) {
      Avx512Gelu(dst, src, n);
      return;
    }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
    if (HasAvx2()) {
      Avx2Gelu(dst, src, n);
      return;
    }
#endif
  }
  ExactGelu(dst, src, n);
}

void ExactGelu(float* dst, const float* src, std::size_t n) noexcept {
  if (!dst || !src || n == 0) return;
  static const bool simd_enabled =
      std::getenv("PPOCR_DISABLE_EXACT_GELU_SIMD") == nullptr;
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (simd_enabled && HasAvx512()) {
    const int blocks = static_cast<int>((n + 32767) / 32768);
    if (n >= 65536 && blocks > 1) {
      ParallelFor(blocks, [&](int first, int last) {
        for (int block = first; block < last; ++block) {
          const std::size_t begin = std::size_t(block) * 32768;
          const std::size_t count = std::min<std::size_t>(32768, n - begin);
          Avx512ExactGelu(dst + begin, src + begin, count);
        }
      });
    } else {
      Avx512ExactGelu(dst, src, n);
    }
    return;
  }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (simd_enabled && HasAvx2()) {
    const int blocks = static_cast<int>((n + 32767) / 32768);
    if (n >= 65536 && blocks > 1) {
      ParallelFor(blocks, [&](int first, int last) {
        for (int block = first; block < last; ++block) {
          const std::size_t begin = std::size_t(block) * 32768;
          const std::size_t count = std::min<std::size_t>(32768, n - begin);
          Avx2ExactGelu(dst + begin, src + begin, count);
        }
      });
    } else {
      Avx2ExactGelu(dst, src, n);
    }
    return;
  }
#endif
  constexpr float inv_sqrt2 = 0.7071067811865475244F;
  const auto exact = [&](int first, int last) {
    for (std::size_t i = std::size_t(first) * 32768;
         i < std::min(n, std::size_t(last) * 32768); ++i) {
      const float x = src[i];
      dst[i] = x * .5F * (1.F + std::erf(x * inv_sqrt2));
    }
  };
  const int blocks = static_cast<int>((n + 32767) / 32768);
  if (n >= 65536 && blocks > 1) ParallelFor(blocks, exact);
  else exact(0, blocks);
}

void BatchNormGelu(float* dst, const float* src, const float* scale,
                   const float* shift, int batches, int channels,
                   std::size_t spatial) noexcept {
  if (!dst || !src || !scale || !shift || batches <= 0 || channels <= 0 || spatial == 0) return;
  const int planes = batches * channels;
  const auto work = std::size_t(planes) * spatial;
  const auto body = [&](int first, int last) {
    constexpr float inv_sqrt2 = 0.7071067811865475244F;
    for (int plane = first; plane < last; ++plane) {
      const int channel = plane % channels;
      const float factor = scale[channel];
      const float offset = shift[channel];
      const auto base = std::size_t(plane) * spatial;
      if (ApproximateGeluEnabled() && spatial >= 65536 &&
          std::getenv("PPOCR_DISABLE_FUSED_BN_GELU") == nullptr) {
        // The opt-in approximate path used to store an affine plane and then
        // load it again for GELU. Fuse both stages so the intermediate never
        // reaches memory, and apply the same AVX-512/AVX2 Pad鑼?approximation
        // consistently to a large fused activation group. Exact ONNX GELU
        // remains below unchanged.
#if defined(PPOCR_HAS_AVX512_KERNELS)
        if (HasAvx512()) {
          Avx512BatchNormGelu(dst + base, src + base, spatial, factor, offset);
          continue;
        }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
        if (HasAvx2()) {
          Avx2BatchNormGelu(dst + base, src + base, spatial, factor, offset);
          continue;
        }
#endif
        // On targets without the x86 approximate kernels, Gelu() deliberately
        // retains exact Erf even when the deployment flag is set. Preserve
        // that cross-architecture contract here as well.
        for (std::size_t index = 0; index < spatial; ++index) {
          const float x = src[base + index] * factor + offset;
          dst[base + index] = x * .5F * (1.F + std::erf(x * inv_sqrt2));
        }
      } else {
        for (std::size_t index = 0; index < spatial; ++index) {
          const float x = src[base + index] * factor + offset;
          dst[base + index] = x * .5F * (1.F + std::erf(x * inv_sqrt2));
        }
      }
    }
  };
  if (work >= 65536 && planes >= 2) ParallelFor(planes, body);
  else body(0, planes);
}

void BatchNormAffine(float* dst, const float* src, const float* scale,
                     const float* shift, int batches, int channels,
                     std::size_t spatial) noexcept {
  if (!dst || !src || !scale || !shift || batches <= 0 || channels <= 0 || spatial == 0) return;
  const int planes = batches * channels;
  const auto body = [&](int first, int last) {
    for (int plane = first; plane < last; ++plane) {
      const int channel = plane % channels;
      const auto base = std::size_t(plane) * spatial;
      // ScaleShift is explicitly alias-safe and selects AVX-512, AVX2, NEON
      // or scalar at runtime.  Keeping each plane contiguous also avoids any
      // NCHW index arithmetic in the hot loop.
      ScaleShift(dst + base, src + base, spatial, scale[channel], shift[channel]);
    }
  };
  const auto work = std::size_t(planes) * spatial;
  if (work >= 65536 && planes >= 2) ParallelFor(planes, body);
  else body(0, planes);
}

void BatchNormSwish(float* dst, const float* src, const float* scale,
                    const float* shift, int batches, int channels,
                    std::size_t spatial) noexcept {
  if (!dst || !src || !scale || !shift || batches <= 0 || channels <= 0 || spatial == 0) return;
  const int planes = batches * channels;
  const auto body = [&](int first, int last) {
    for (int plane = first; plane < last; ++plane) {
      const int channel = plane % channels;
      const float factor = scale[channel];
      const float offset = shift[channel];
      const auto base = std::size_t(plane) * spatial;
      std::size_t index = 0;
      // Preserve the original affine -> logistic -> multiply sequence for
      // every lane. Four independent exact exponentials improve instruction
      // level parallelism on both x86 and ARM without a numerical shortcut.
      for (; index + 4 <= spatial; index += 4) {
        const float v0 = src[base + index] * factor + offset;
        const float v1 = src[base + index + 1] * factor + offset;
        const float v2 = src[base + index + 2] * factor + offset;
        const float v3 = src[base + index + 3] * factor + offset;
        const float e0 = std::exp(-v0), e1 = std::exp(-v1);
        const float e2 = std::exp(-v2), e3 = std::exp(-v3);
        dst[base + index] = v0 * (1.F / (1.F + e0));
        dst[base + index + 1] = v1 * (1.F / (1.F + e1));
        dst[base + index + 2] = v2 * (1.F / (1.F + e2));
        dst[base + index + 3] = v3 * (1.F / (1.F + e3));
      }
      for (; index < spatial; ++index) {
        // Keep the original ONNX operation sequence explicit: BN is rounded
        // to FP32 before it becomes the input to both Sigmoid and Mul.
        const float value = src[base + index] * factor + offset;
        const float gate = 1.F / (1.F + std::exp(-value));
        dst[base + index] = value * gate;
      }
    }
  };
  const auto work = std::size_t(planes) * spatial;
  if (work >= 65536 && planes >= 2) ParallelFor(planes, body);
  else body(0, planes);
}

void BatchNormSwishBinary(float* dst, const float* src, const float* scale,
                          const float* shift, const float* right, int batches,
                          int channels, std::size_t spatial,
                          BinaryOp operation) noexcept {
  if (!dst || !src || !scale || !shift || !right || batches <= 0 || channels <= 0 ||
      spatial == 0) return;
  const int planes = batches * channels;
  const auto apply = [&](float value, float other) noexcept {
    switch (operation) {
      case BinaryOp::add: return value + other;
      case BinaryOp::sub: return value - other;
      case BinaryOp::mul: return value * other;
      case BinaryOp::div: return value / other;
    }
    return value;
  };
  const auto body = [&](int first, int last) {
    for (int plane = first; plane < last; ++plane) {
      const int channel = plane % channels;
      const float factor = scale[channel];
      const float offset = shift[channel];
      const auto base = std::size_t(plane) * spatial;
      std::size_t index = 0;
      for (; index + 4 <= spatial; index += 4) {
        const float v0 = src[base + index] * factor + offset;
        const float v1 = src[base + index + 1] * factor + offset;
        const float v2 = src[base + index + 2] * factor + offset;
        const float v3 = src[base + index + 3] * factor + offset;
        const float s0 = v0 * (1.F / (1.F + std::exp(-v0)));
        const float s1 = v1 * (1.F / (1.F + std::exp(-v1)));
        const float s2 = v2 * (1.F / (1.F + std::exp(-v2)));
        const float s3 = v3 * (1.F / (1.F + std::exp(-v3)));
        dst[base + index] = apply(s0, right[base + index]);
        dst[base + index + 1] = apply(s1, right[base + index + 1]);
        dst[base + index + 2] = apply(s2, right[base + index + 2]);
        dst[base + index + 3] = apply(s3, right[base + index + 3]);
      }
      for (; index < spatial; ++index) {
        const float value = src[base + index] * factor + offset;
        const float swish = value * (1.F / (1.F + std::exp(-value)));
        dst[base + index] = apply(swish, right[base + index]);
      }
    }
  };
  const auto work = std::size_t(planes) * spatial;
  if (work >= 65536 && planes >= 2) ParallelFor(planes, body);
  else body(0, planes);
}

void HardSwish(float* dst, const float* src, std::size_t n) noexcept {
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512()) { Avx512HardSwish(dst, src, n); return; }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2()) { Avx2HardSwish(dst, src, n); return; }
#endif
  std::size_t i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  if (HasNeon()) {
    const auto sixth = vdupq_n_f32(1.F / 6.F), half = vdupq_n_f32(.5F);
    const auto zero = vdupq_n_f32(0.F), one = vdupq_n_f32(1.F);
    for (; i + 4 <= n; i += 4) {
      const auto x = vld1q_f32(src + i);
      const auto gate = vminq_f32(one, vmaxq_f32(zero, vmlaq_f32(half, x, sixth)));
      vst1q_f32(dst + i, vmulq_f32(x, gate));
    }
  }
#endif
  for (; i < n; ++i) {
    const float x = src[i];
    dst[i] = x * std::clamp(x / 6.F + .5F, 0.F, 1.F);
  }
}

void ScaleShift(float* dst, const float* src, std::size_t n, float scale, float shift) noexcept {
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512()) { Avx512ScaleShift(dst, src, n, scale, shift); return; }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2()) { Avx2ScaleShift(dst, src, n, scale, shift); return; }
#endif
  std::size_t i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  if (HasNeon()) {
    const auto s = vdupq_n_f32(scale), b = vdupq_n_f32(shift);
    for (; i + 4 <= n; i += 4) vst1q_f32(dst+i, vmlaq_f32(b, vld1q_f32(src+i), s));
  }
#endif
  for (; i < n; ++i) dst[i] = src[i] * scale + shift;
}

void BinaryScalar(float* dst, const float* src, std::size_t n, float scalar,
                  BinaryOp op, bool scalar_left) noexcept {
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512()) { Avx512BinaryScalar(dst, src, n, scalar, op, scalar_left); return; }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2()) { Avx2BinaryScalar(dst, src, n, scalar, op, scalar_left); return; }
#endif
  for (std::size_t i = 0; i < n; ++i) {
    const float x = src[i];
    switch (op) {
      case BinaryOp::add: dst[i] = x + scalar; break;
      case BinaryOp::sub: dst[i] = scalar_left ? scalar - x : x - scalar; break;
      case BinaryOp::mul: dst[i] = x * scalar; break;
      case BinaryOp::div: dst[i] = scalar_left ? scalar / x : x / scalar; break;
    }
  }
}

void AddChannelBias(float* dst, const float* bias, int batches, int channels,
                    std::size_t spatial) noexcept {
  if (!dst || !bias || batches <= 0 || channels <= 0 || spatial == 0) return;
  // Channel bias is a contiguous scalar broadcast per NCHW plane.  Reuse the
  // ISA-dispatched scalar binary kernel rather than open-coding a scalar loop:
  // this covers AVX-512, AVX2, NEON and the portable fallback without another
  // allocation or an expanded broadcast tensor.
  const int planes = batches * channels;
  const auto body = [&](int first, int last) {
    for (int plane = first; plane < last; ++plane) {
      float* const values = dst + std::size_t(plane) * spatial;
      BinaryScalar(values, values, spatial, bias[plane % channels], BinaryOp::add, false);
    }
  };
  const auto work = std::size_t(planes) * spatial;
  if (work >= 65536 && planes >= 2) ParallelFor(planes, body);
  else body(0, planes);
}

void HardSigmoid(float* dst, const float* src, std::size_t n, float alpha,
                 float beta) noexcept {
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512()) { Avx512HardSigmoid(dst, src, n, alpha, beta); return; }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2()) { Avx2HardSigmoid(dst, src, n, alpha, beta); return; }
#endif
  std::size_t i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  if (HasNeon()) {
    const float32x4_t a = vdupq_n_f32(alpha), b = vdupq_n_f32(beta);
    const float32x4_t zero = vdupq_n_f32(0.F), one = vdupq_n_f32(1.F);
    for (; i + 4 <= n; i += 4) {
      const float32x4_t value = vmlaq_f32(b, vld1q_f32(src + i), a);
      vst1q_f32(dst + i, vminq_f32(vmaxq_f32(value, zero), one));
    }
  }
#endif
  for (; i < n; ++i) dst[i] = std::clamp(alpha * src[i] + beta, 0.F, 1.F);
}

void Sigmoid(float* dst, const float* src, std::size_t n) noexcept {
  // `std::exp` itself is scalar in the baseline C++ runtime, but four
  // independent expressions give the compiler/libm room to overlap their
  // latency.  This retains the exact ONNX logistic expression -- unlike an
  // approximate vector exp it does not change the detector probability map.
  // The detector DB head is 16x160x704 (~1.8M). Rec maps stay well under the
  // floor, so this ParallelFor cannot nest inside a rec crop.
  // `PPOCR_DISABLE_SIGMOID_PF` restores one-thread libm.
  const auto body = [&](std::size_t begin, std::size_t end) {
#if defined(PPOCR_HAS_AVX512_KERNELS)
    static const bool simd512 =
        // The vector exponential is an approximation.  Detector sigmoid
        // values feed DB thresholding, so exact scalar libm remains the
        // default; deployments that have qualified its small FP32 drift can
        // opt in explicitly.
        std::getenv("PPOCR_ENABLE_AVX512_SIGMOID") != nullptr &&
        std::getenv("PPOCR_DISABLE_AVX512_SIGMOID") == nullptr;
    if (simd512 && HasAvx512() && end - begin >= 16) {
      Avx512Sigmoid(dst + begin, src + begin, end - begin);
      return;
    }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
    static const bool simd256 =
        std::getenv("PPOCR_ENABLE_AVX2_SIGMOID") != nullptr &&
        std::getenv("PPOCR_DISABLE_AVX2_SIGMOID") == nullptr;
    if (simd256 && HasAvx2() && end - begin >= 8) {
      Avx2Sigmoid(dst + begin, src + begin, end - begin);
      return;
    }
#endif
    std::size_t i = begin;
    for (; i + 4 <= end; i += 4) {
      const float x0 = src[i], x1 = src[i + 1], x2 = src[i + 2], x3 = src[i + 3];
      const float e0 = std::exp(-x0), e1 = std::exp(-x1);
      const float e2 = std::exp(-x2), e3 = std::exp(-x3);
      dst[i] = 1.F / (1.F + e0);
      dst[i + 1] = 1.F / (1.F + e1);
      dst[i + 2] = 1.F / (1.F + e2);
      dst[i + 3] = 1.F / (1.F + e3);
    }
    for (; i < end; ++i) dst[i] = 1.F / (1.F + std::exp(-src[i]));
  };
  static const bool row_pf = std::getenv("PPOCR_DISABLE_SIGMOID_PF") == nullptr;
  if (row_pf && n >= 131072) {
    constexpr std::size_t chunk = 16384;
    const int tasks = static_cast<int>((n + chunk - 1) / chunk);
    ParallelFor(tasks, [&](int first, int last) {
      for (int task = first; task < last; ++task) {
        const std::size_t begin = std::size_t(task) * chunk;
        body(begin, std::min(n, begin + chunk));
      }
    });
    return;
  }
  body(0, n);
}

void Swish(float* dst, const float* src, std::size_t n) noexcept {
  std::size_t i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  // NEON has no portable vector exponential in the baseline ISA.  Keep four
  // independent scalar exponentials together so compilers can schedule the
  // loads/stores efficiently while preserving the exact logistic expression.
  if (HasNeon()) {
    for (; i + 4 <= n; i += 4) {
      const auto x = vld1q_f32(src + i);
      alignas(16) float lanes[4];
      vst1q_f32(lanes, x);
      for (float& value : lanes) value *= 1.F / (1.F + std::exp(-value));
      vst1q_f32(dst + i, vld1q_f32(lanes));
    }
  }
#endif
  // As with Sigmoid, unroll the exact scalar-libm path rather than replacing
  // exp with an approximation. This also benefits x86, where the vector ISA
  // has no standard exact exponential instruction.
  for (; i + 4 <= n; i += 4) {
    const float x0 = src[i], x1 = src[i + 1], x2 = src[i + 2], x3 = src[i + 3];
    const float e0 = std::exp(-x0), e1 = std::exp(-x1);
    const float e2 = std::exp(-x2), e3 = std::exp(-x3);
    dst[i] = x0 * (1.F / (1.F + e0));
    dst[i + 1] = x1 * (1.F / (1.F + e1));
    dst[i + 2] = x2 * (1.F / (1.F + e2));
    dst[i + 3] = x3 * (1.F / (1.F + e3));
  }
  for (; i < n; ++i) {
    const float value = src[i];
    dst[i] = value * (1.F / (1.F + std::exp(-value)));
  }
}
void Relu(float* dst, const float* src, std::size_t n) noexcept {
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512()) { Avx512Relu(dst, src, n); return; }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2()) { Avx2Relu(dst, src, n); return; }
#endif
  std::size_t i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  if (HasNeon()) {
    const float32x4_t zero = vdupq_n_f32(0.F);
    for (; i + 4 <= n; i += 4) vst1q_f32(dst + i, vmaxq_f32(vld1q_f32(src + i), zero));
  }
#endif
  for (; i < n; ++i) dst[i] = std::max(src[i], 0.F);
}

void Axpy(float* dst, const float* src, float alpha, std::size_t n) noexcept {
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512()) { Avx512Axpy(dst, src, alpha, n); return; }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2()) { Avx2Axpy(dst, src, alpha, n); return; }
#endif
  std::size_t i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  if (HasNeon()) {
    const float32x4_t a = vdupq_n_f32(alpha);
    for (; i + 4 <= n; i += 4) {
      vst1q_f32(dst + i, vmlaq_f32(vld1q_f32(dst + i), a, vld1q_f32(src + i)));
    }
  }
#endif
  for (; i < n; ++i) dst[i] += alpha * src[i];
}

#if defined(PPOCR_HAS_AVX512_KERNELS)
void Avx512ExpandGeluProjectAdd(float*, const float*, const float*, const float*,
                                const float*, const float*, int, int, std::size_t,
                                std::size_t, std::size_t) noexcept;
#endif

void ExpandGeluProjectAdd(float* dst, const float* src, const float* expand_weights,
                          const float* expand_bias, const float* project_weights,
                          const float* project_bias, int channels, int hidden,
                          std::size_t plane) {
  if (!dst || !src || !expand_weights || !project_weights || channels <= 0 ||
      hidden <= 0 || plane == 0) return;
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512()) {
    const int tiles = static_cast<int>((plane + 15) / 16);
    const auto work = std::int64_t(channels) * hidden * static_cast<std::int64_t>(plane);
    const auto body = [&](int first, int last) {
      const std::size_t begin = std::size_t(first) * 16;
      const std::size_t end = std::min(plane, std::size_t(last) * 16);
      Avx512ExpandGeluProjectAdd(dst, src, expand_weights, expand_bias, project_weights,
                                 project_bias, channels, hidden, plane, begin, end);
    };
    if (work >= std::int64_t{3000000} && tiles > 2) ParallelFor(tiles, body);
    else body(0, tiles);
    return;
  }
#endif
  thread_local std::vector<float> hidden_act;
  hidden_act.resize(static_cast<std::size_t>(hidden));
  constexpr float inv_sqrt2 = 0.7071067811865475244F;
  for (std::size_t spatial = 0; spatial < plane; ++spatial) {
    for (int hidden_channel = 0; hidden_channel < hidden; ++hidden_channel) {
      float sum = expand_bias ? expand_bias[hidden_channel] : 0.F;
      const float* filter = expand_weights + std::size_t(hidden_channel) * channels;
      for (int channel = 0; channel < channels; ++channel) {
        sum += filter[channel] * src[std::size_t(channel) * plane + spatial];
      }
      hidden_act[static_cast<std::size_t>(hidden_channel)] =
          sum * .5F * (1.F + std::erf(sum * inv_sqrt2));
    }
    for (int channel = 0; channel < channels; ++channel) {
      float sum = (project_bias ? project_bias[channel] : 0.F) +
                  src[std::size_t(channel) * plane + spatial];
      const float* filter = project_weights + std::size_t(channel) * hidden;
      for (int hidden_channel = 0; hidden_channel < hidden; ++hidden_channel) {
        sum += filter[hidden_channel] * hidden_act[static_cast<std::size_t>(hidden_channel)];
      }
      dst[std::size_t(channel) * plane + spatial] = sum;
    }
  }
}

void PointwiseConv(float* dst, const float* src, const float* weights,
                   const float* bias, int output_channels, int input_channels,
                   std::size_t plane) {
  const auto body = [&](int first, int last) {
#if defined(PPOCR_HAS_AVX512_KERNELS)
    if (HasAvx512()) {
      if (UseAvx512Pointwise16(output_channels, input_channels, plane)) {
        Avx512PointwiseConv16(dst, src, weights, bias, first, last, input_channels, plane);
      } else if (UseAvx512Pointwise8(output_channels, input_channels, plane)) {
        Avx512PointwiseConv8(dst, src, weights, bias, first, last, input_channels, plane);
      } else {
        Avx512PointwiseConv4(dst, src, weights, bias, first, last, input_channels, plane);
      }
      return;
    }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
    if (HasAvx2()) {
      Avx2PointwiseConv4(dst, src, weights, bias, first, last, input_channels, plane);
      return;
    }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    if (HasNeon()) {
      int output = first;
      for (; output + 4 <= last; output += 4) {
        float* out0 = dst + std::size_t(output) * plane;
        float* out1 = out0 + plane;
        float* out2 = out1 + plane;
        float* out3 = out2 + plane;
        std::fill_n(out0, plane, bias ? bias[output] : 0.F);
        std::fill_n(out1, plane, bias ? bias[output + 1] : 0.F);
        std::fill_n(out2, plane, bias ? bias[output + 2] : 0.F);
        std::fill_n(out3, plane, bias ? bias[output + 3] : 0.F);
        const float* w0 = weights + std::size_t(output) * input_channels;
        const float* w1 = w0 + input_channels;
        const float* w2 = w1 + input_channels;
        const float* w3 = w2 + input_channels;
        for (int input = 0; input < input_channels; ++input) {
          const float* values = src + std::size_t(input) * plane;
          const float32x4_t a0 = vdupq_n_f32(w0[input]);
          const float32x4_t a1 = vdupq_n_f32(w1[input]);
          const float32x4_t a2 = vdupq_n_f32(w2[input]);
          const float32x4_t a3 = vdupq_n_f32(w3[input]);
          std::size_t index = 0;
          for (; index + 4 <= plane; index += 4) {
            const float32x4_t x = vld1q_f32(values + index);
            vst1q_f32(out0 + index, vmlaq_f32(vld1q_f32(out0 + index), a0, x));
            vst1q_f32(out1 + index, vmlaq_f32(vld1q_f32(out1 + index), a1, x));
            vst1q_f32(out2 + index, vmlaq_f32(vld1q_f32(out2 + index), a2, x));
            vst1q_f32(out3 + index, vmlaq_f32(vld1q_f32(out3 + index), a3, x));
          }
          for (; index < plane; ++index) {
            const float x = values[index];
            out0[index] += w0[input] * x;
            out1[index] += w1[input] * x;
            out2[index] += w2[input] * x;
            out3[index] += w3[input] * x;
          }
        }
      }
      for (; output < last; ++output) {
        float* out = dst + std::size_t(output) * plane;
        std::fill_n(out, plane, bias ? bias[output] : 0.F);
        const float* filter = weights + std::size_t(output) * input_channels;
        for (int input = 0; input < input_channels; ++input) {
          Axpy(out, src + std::size_t(input) * plane, filter[input], plane);
        }
      }
      return;
    }
#endif
    for (int output = first; output < last; ++output) {
      float* out = dst + std::size_t(output) * plane;
      std::fill_n(out, plane, bias ? bias[output] : 0.F);
      const float* filter = weights + std::size_t(output) * input_channels;
      for (int input = 0; input < input_channels; ++input) {
        Axpy(out, src + std::size_t(input) * plane, filter[input], plane);
      }
    }
  };
  // Pointwise blocks dominate detector compute. Splitting channels has no
  // synchronization or cache-line sharing.  Do not create threads for the
  // tiny recognizer blocks where startup dominates.
  const auto work = std::int64_t(output_channels) * input_channels *
                    static_cast<std::int64_t>(plane);
  // 1x1 Conv闂傚倸鍊烽懗鍫曞磻閵娾晛纾块柡灞诲劜閸?outputs are independent channels. Dispatch only work large
  // enough to amortize barrier/scheduling overhead on the persistent pool.
  // AVX-512's register tile shares every input vector across eight outputs.
  // Give each worker whole tiles so this reuse survives channel parallelism.
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512()) {
    const int tile = Avx512PointwiseTile(output_channels, input_channels, plane);
    // Conv.63 is 32→64 on 40x176. Spatial tiles keep the 32 source planes in
    // L1 across all 64 outputs. Same-host 8-run missed versus OC ParallelFor
    // (18.73 vs ~16.1). Rec 1x1s stay under plane 2048.
    // `PPOCR_ENABLE_AVX512_POINTWISE16_SPATIAL` is the explicit A/B.
    // Conv.63: four OC tiles reread 900 KB. Four spatial workers each keep
    // ~10 rows × 32 IC in L2 and write all 64 OC. 28-task 256-pixel PF was
    // an e2e miss. `PPOCR_DISABLE_AVX512_POINTWISE16_G4` restores OC PF.
    static const bool g4 =
        std::getenv("PPOCR_ENABLE_AVX512_POINTWISE16_G4") != nullptr &&
        std::getenv("PPOCR_DISABLE_AVX512_POINTWISE16_G4") == nullptr;
    if (g4 && tile == 16 && plane >= 2048 && output_channels >= 48 &&
        output_channels % 16 == 0 && input_channels >= 32) {
      const std::size_t chunk =
          ((plane + 3) / 4 + 15) & ~std::size_t(15);
      const int spatial_groups = static_cast<int>((plane + chunk - 1) / chunk);
      const auto spatial = [&](int first, int last) {
        for (int group = first; group < last; ++group) {
          const std::size_t i0 = std::size_t(group) * chunk;
          const std::size_t i1 = std::min(plane, i0 + chunk);
          Avx512PointwiseConv16Range(dst, src, weights, bias, 0, output_channels,
                                     input_channels, plane, i0, i1);
        }
      };
      if (spatial_groups > 1) ParallelFor(spatial_groups, spatial);
      else spatial(0, spatial_groups);
      return;
    }
    static const bool spatial_pf =
        std::getenv("PPOCR_ENABLE_AVX512_POINTWISE16_SPATIAL") != nullptr &&
        std::getenv("PPOCR_DISABLE_AVX512_POINTWISE16_SPATIAL") == nullptr;
    if (spatial_pf && tile == 16 && plane >= 2048 && output_channels >= 48 &&
        output_channels % 16 == 0 && input_channels >= 32) {
      const std::size_t chunk = 256;
      const int spatial_groups = static_cast<int>((plane + chunk - 1) / chunk);
      const auto spatial = [&](int first, int last) {
        for (int group = first; group < last; ++group) {
          const std::size_t i0 = std::size_t(group) * chunk;
          const std::size_t i1 = std::min(plane, i0 + chunk);
          Avx512PointwiseConv16Range(dst, src, weights, bias, 0, output_channels,
                                     input_channels, plane, i0, i1);
        }
      };
      if (spatial_groups > 1) ParallelFor(spatial_groups, spatial);
      else spatial(0, spatial_groups);
      return;
    }
    // Conv.63 32→64 on 40x176. OC ParallelFor rereads the 900 KB source four
    // times. Serial 64-pixel tiles keep 32×64 in L1 while all 64 OC
    // accumulate. Rec 1x1s stay under plane 2048. Spatial ParallelFor of
    // the same blocking was an e2e miss (18.73).
    // `PPOCR_DISABLE_AVX512_POINTWISE16_BLOCK` restores OC ParallelFor.
    static const bool blocked =
        std::getenv("PPOCR_ENABLE_AVX512_POINTWISE16_BLOCK") != nullptr &&
        std::getenv("PPOCR_DISABLE_AVX512_POINTWISE16_BLOCK") == nullptr;
    if (blocked && tile == 16 && plane >= 2048 && output_channels % 16 == 0 &&
        input_channels >= 32) {
      const std::size_t chunk = 64;
      for (std::size_t i0 = 0; i0 < plane; i0 += chunk) {
        Avx512PointwiseConv16Range(dst, src, weights, bias, 0, output_channels,
                                   input_channels, plane, i0,
                                   std::min(plane, i0 + chunk));
      }
      return;
    }
    // Conv.63 32→64 on 40x176: four OC tiles reread 900 KB. One serial walk
    // keeps the source in L2. Rec 1x1s stay under plane 2048.
    // `PPOCR_DISABLE_AVX512_POINTWISE16_SERIAL` restores OC ParallelFor.
    static const bool pw16_serial =
        std::getenv("PPOCR_ENABLE_AVX512_POINTWISE16_SERIAL") != nullptr &&
        std::getenv("PPOCR_DISABLE_AVX512_POINTWISE16_SERIAL") == nullptr;
    if (pw16_serial && tile == 16 && plane >= 2048) {
      body(0, output_channels);
      return;
    }
    const int groups = (output_channels + tile - 1) / tile;
    const auto grouped = [&](int first, int last) {
      body(first * tile, std::min(output_channels, last * tile));
    };
    if (PointwiseWorkParallel(work, groups, output_channels, plane)) ParallelFor(groups, grouped);
    else grouped(0, groups);
    return;
  }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2()) {
    const int groups = (output_channels + 3) / 4;
    const auto grouped = [&](int first, int last) {
      body(first * 4, std::min(output_channels, last * 4));
    };
    if (PointwiseWorkParallel(work, groups, output_channels)) ParallelFor(groups, grouped);
    else grouped(0, groups);
    return;
  }
#endif
  if (PointwiseWorkParallel(work, output_channels, output_channels)) ParallelFor(output_channels, body);
  else body(0, output_channels);
}

void PointwiseConvBatch(float* dst, const float* src, const float* weights,
                        const float* bias, int batches, int output_channels,
                        int input_channels, std::size_t plane) {
  if (!dst || !src || !weights || batches <= 0 || output_channels <= 0 ||
      input_channels <= 0 || plane == 0) return;
  if (batches == 1) {
    PointwiseConv(dst, src, weights, bias, output_channels, input_channels, plane);
    return;
  }
  const std::size_t input_batch = std::size_t(input_channels) * plane;
  const std::size_t output_batch = std::size_t(output_channels) * plane;
  const auto work = std::int64_t(batches) * output_channels * input_channels *
                    static_cast<std::int64_t>(plane);
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512()) {
    const int tile = Avx512PointwiseTile(output_channels, input_channels, plane);
    const int groups = (output_channels + tile - 1) / tile;
    const int tasks = batches * groups;
    const auto body = [&](int first, int last) {
      for (int task = first; task < last; ++task) {
        const int batch = task / groups;
        const int group = task - batch * groups;
        float* output = dst + std::size_t(batch) * output_batch;
        const float* input = src + std::size_t(batch) * input_batch;
        const int first_output = group * tile;
        const int last_output = std::min(output_channels, first_output + tile);
        if (tile == 16) {
          Avx512PointwiseConv16(output, input, weights, bias, first_output, last_output,
                                input_channels, plane);
        } else if (tile == 8) {
          Avx512PointwiseConv8(output, input, weights, bias, first_output, last_output,
                               input_channels, plane);
        } else {
          Avx512PointwiseConv4(output, input, weights, bias, first_output, last_output,
                               input_channels, plane);
        }
      }
    };
    if (work >= 3000000 && tasks > 1) ParallelFor(tasks, body); else body(0, tasks);
    return;
  }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2()) {
    const int groups = (output_channels + 3) / 4;
    const int tasks = batches * groups;
    const auto body = [&](int first, int last) {
      for (int task = first; task < last; ++task) {
        const int batch = task / groups;
        const int group = task - batch * groups;
        Avx2PointwiseConv4(dst + std::size_t(batch) * output_batch,
                           src + std::size_t(batch) * input_batch, weights, bias,
                           group * 4, std::min(output_channels, group * 4 + 4),
                           input_channels, plane);
      }
    };
    if (work >= 3000000 && tasks > 1) ParallelFor(tasks, body); else body(0, tasks);
    return;
  }
#endif
  for (int batch = 0; batch < batches; ++batch) {
    PointwiseConv(dst + std::size_t(batch) * output_batch,
                  src + std::size_t(batch) * input_batch, weights, bias,
                  output_channels, input_channels, plane);
  }
}

void PointwiseConvAdd(float* dst, const float* src, const float* weights,
                      const float* bias, const float* residual,
                      int output_channels, int input_channels,
                      std::size_t plane) {
  const auto body = [&](int first, int last) {
#if defined(PPOCR_HAS_AVX512_KERNELS)
    if (HasAvx512()) {
      if (UseAvx512Pointwise8(output_channels, input_channels, plane)) {
        Avx512PointwiseConvAdd8(dst, src, weights, bias, residual, first, last,
                                input_channels, plane);
      } else {
        Avx512PointwiseConvAdd4(dst, src, weights, bias, residual, first, last,
                                input_channels, plane);
      }
      return;
    }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
    if (HasAvx2()) {
      Avx2PointwiseConvAdd4(dst, src, weights, bias, residual, first, last,
                             input_channels, plane);
      return;
    }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    if (HasNeon()) {
      int output = first;
      for (; output + 4 <= last; output += 4) {
        float* out0 = dst + std::size_t(output) * plane;
        float* out1 = out0 + plane;
        float* out2 = out1 + plane;
        float* out3 = out2 + plane;
        const float* r0 = residual + std::size_t(output) * plane;
        const float* r1 = r0 + plane;
        const float* r2 = r1 + plane;
        const float* r3 = r2 + plane;
        const float* w0 = weights + std::size_t(output) * input_channels;
        const float* w1 = w0 + input_channels;
        const float* w2 = w1 + input_channels;
        const float* w3 = w2 + input_channels;
        std::size_t index = 0;
        for (; index + 4 <= plane; index += 4) {
          float32x4_t s0 = vdupq_n_f32(bias ? bias[output] : 0.F);
          float32x4_t s1 = vdupq_n_f32(bias ? bias[output + 1] : 0.F);
          float32x4_t s2 = vdupq_n_f32(bias ? bias[output + 2] : 0.F);
          float32x4_t s3 = vdupq_n_f32(bias ? bias[output + 3] : 0.F);
          for (int input = 0; input < input_channels; ++input) {
            const float32x4_t values = vld1q_f32(src + std::size_t(input) * plane + index);
            s0 = vmlaq_f32(s0, vdupq_n_f32(w0[input]), values);
            s1 = vmlaq_f32(s1, vdupq_n_f32(w1[input]), values);
            s2 = vmlaq_f32(s2, vdupq_n_f32(w2[input]), values);
            s3 = vmlaq_f32(s3, vdupq_n_f32(w3[input]), values);
          }
          vst1q_f32(out0 + index, vaddq_f32(s0, vld1q_f32(r0 + index)));
          vst1q_f32(out1 + index, vaddq_f32(s1, vld1q_f32(r1 + index)));
          vst1q_f32(out2 + index, vaddq_f32(s2, vld1q_f32(r2 + index)));
          vst1q_f32(out3 + index, vaddq_f32(s3, vld1q_f32(r3 + index)));
        }
        for (; index < plane; ++index) {
          float s0 = bias ? bias[output] : 0.F, s1 = bias ? bias[output + 1] : 0.F;
          float s2 = bias ? bias[output + 2] : 0.F, s3 = bias ? bias[output + 3] : 0.F;
          for (int input = 0; input < input_channels; ++input) {
            const float value = src[std::size_t(input) * plane + index];
            s0 += w0[input] * value; s1 += w1[input] * value;
            s2 += w2[input] * value; s3 += w3[input] * value;
          }
          out0[index] = s0 + r0[index]; out1[index] = s1 + r1[index];
          out2[index] = s2 + r2[index]; out3[index] = s3 + r3[index];
        }
      }
      for (; output < last; ++output) {
        float* out = dst + std::size_t(output) * plane;
        const float* add = residual + std::size_t(output) * plane;
        std::fill_n(out, plane, bias ? bias[output] : 0.F);
        const float* filter = weights + std::size_t(output) * input_channels;
        for (int input = 0; input < input_channels; ++input) {
          Axpy(out, src + std::size_t(input) * plane, filter[input], plane);
        }
        BinaryInplace(out, add, plane, BinaryOp::add);
      }
      return;
    }
#endif
    // Keep AVX2/NEON/scalar fallbacks exact.  `first` is an output-channel
    // offset, so this cannot delegate a sliced range to PointwiseConv without
    // also rebasing every tensor pointer.
    for (int output = first; output < last; ++output) {
      float* out = dst + std::size_t(output) * plane;
      std::fill_n(out, plane, bias ? bias[output] : 0.F);
      const float* filter = weights + std::size_t(output) * input_channels;
      for (int input = 0; input < input_channels; ++input) {
        Axpy(out, src + std::size_t(input) * plane, filter[input], plane);
      }
      const float* add = residual + std::size_t(output) * plane;
      BinaryInplace(out, add, plane, BinaryOp::add);
    }
  };
  const auto work = std::int64_t(output_channels) * input_channels *
                    static_cast<std::int64_t>(plane);
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512()) {
    const int tile = Avx512PointwiseTile(output_channels, input_channels, plane);
    const int groups = (output_channels + tile - 1) / tile;
    const auto grouped = [&](int first, int last) {
      body(first * tile, std::min(output_channels, last * tile));
    };
    if (PointwiseWorkParallel(work, groups, output_channels, plane)) ParallelFor(groups, grouped);
    else grouped(0, groups);
    return;
  }
#endif
  // AVX2 has a four-output fused kernel. Split whole four-channel tiles so
  // every worker retains the same input-vector reuse as the ISA kernel.
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2()) {
    const int groups = (output_channels + 3) / 4;
    const auto grouped = [&](int first, int last) {
      body(first * 4, std::min(output_channels, last * 4));
    };
    if (PointwiseWorkParallel(work, groups, output_channels)) ParallelFor(groups, grouped);
    else grouped(0, groups);
    return;
  }
#endif
  // Non-AVX-512 uses the regular convolution and a full equal-shape add.
  PointwiseConv(dst, src, weights, bias, output_channels, input_channels, plane);
  BinaryInplace(dst, residual, std::size_t(output_channels) * plane, BinaryOp::add);
}

void PointwiseConvAddBatch(float* dst, const float* src, const float* weights,
                           const float* bias, const float* residual, int batches,
                           int output_channels, int input_channels,
                           std::size_t plane) {
  if (!dst || !src || !weights || !residual || batches <= 0 || output_channels <= 0 ||
      input_channels <= 0 || plane == 0) return;
  if (batches == 1) {
    PointwiseConvAdd(dst, src, weights, bias, residual, output_channels, input_channels, plane);
    return;
  }
  const std::size_t input_batch = std::size_t(input_channels) * plane;
  const std::size_t output_batch = std::size_t(output_channels) * plane;
  const auto work = std::int64_t(batches) * output_channels * input_channels *
                    static_cast<std::int64_t>(plane);
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512()) {
    const int tile = Avx512PointwiseTile(output_channels, input_channels, plane);
    const int groups = (output_channels + tile - 1) / tile;
    const int tasks = batches * groups;
    const auto body = [&](int first, int last) {
      for (int task = first; task < last; ++task) {
        const int batch = task / groups;
        const int group = task - batch * groups;
        float* output = dst + std::size_t(batch) * output_batch;
        const float* input = src + std::size_t(batch) * input_batch;
        const float* add = residual + std::size_t(batch) * output_batch;
        const int first_output = group * tile;
        const int last_output = std::min(output_channels, first_output + tile);
        if (tile == 8) {
          Avx512PointwiseConvAdd8(output, input, weights, bias, add, first_output,
                                  last_output, input_channels, plane);
        } else {
          Avx512PointwiseConvAdd4(output, input, weights, bias, add, first_output,
                                  last_output, input_channels, plane);
        }
      }
    };
    if (work >= 3000000 && tasks > 1) ParallelFor(tasks, body); else body(0, tasks);
    return;
  }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2()) {
    const int groups = (output_channels + 3) / 4;
    const int tasks = batches * groups;
    const auto body = [&](int first, int last) {
      for (int task = first; task < last; ++task) {
        const int batch = task / groups;
        const int group = task - batch * groups;
        Avx2PointwiseConvAdd4(dst + std::size_t(batch) * output_batch,
                               src + std::size_t(batch) * input_batch, weights, bias,
                               residual + std::size_t(batch) * output_batch, group * 4,
                               std::min(output_channels, group * 4 + 4), input_channels, plane);
      }
    };
    if (work >= 3000000 && tasks > 1) ParallelFor(tasks, body); else body(0, tasks);
    return;
  }
#endif
  for (int batch = 0; batch < batches; ++batch) {
    PointwiseConvAdd(dst + std::size_t(batch) * output_batch,
                     src + std::size_t(batch) * input_batch, weights, bias,
                     residual + std::size_t(batch) * output_batch, output_channels,
                     input_channels, plane);
  }
}

void PointwiseConvAddRelu(float* dst, const float* src, const float* weights,
                          const float* bias, const float* residual,
                          int output_channels, int input_channels,
                          std::size_t plane) {
  const auto body = [&](int first, int last) {
#if defined(PPOCR_HAS_AVX512_KERNELS)
    if (HasAvx512()) {
      const bool tile8 = UseAvx512PointwiseRelu8(output_channels, input_channels, plane);
      if (tile8) {
        Avx512PointwiseConvAddRelu8(dst, src, weights, bias, residual, first, last,
                                    input_channels, plane);
      } else {
        Avx512PointwiseConvAddRelu4(dst, src, weights, bias, residual, first, last,
                                    input_channels, plane);
      }
      return;
    }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
    if (HasAvx2()) {
      Avx2PointwiseConvAddRelu4(dst, src, weights, bias, residual, first, last,
                                 input_channels, plane);
      return;
    }
#endif
    for (int output = first; output < last; ++output) {
      float* out = dst + std::size_t(output) * plane;
      const float* add = residual + std::size_t(output) * plane;
      std::fill_n(out, plane, bias ? bias[output] : 0.F);
      const float* filter = weights + std::size_t(output) * input_channels;
      for (int input = 0; input < input_channels; ++input) {
        Axpy(out, src + std::size_t(input) * plane, filter[input], plane);
      }
      for (std::size_t index = 0; index < plane; ++index) {
        out[index] = std::max(out[index] + add[index], 0.F);
      }
    }
  };
  const auto work = std::int64_t(output_channels) * input_channels *
                    static_cast<std::int64_t>(plane);
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512()) {
    const bool tile8 = UseAvx512PointwiseRelu8(output_channels, input_channels, plane);
    const int tile = tile8 ? 8 : 4;
    const int groups = (output_channels + tile - 1) / tile;
    const auto grouped = [&](int first, int last) {
      body(first * tile, std::min(output_channels, last * tile));
    };
    if (PointwiseWorkParallel(work, groups, output_channels, plane)) ParallelFor(groups, grouped);
    else grouped(0, groups);
    return;
  }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2()) {
    const int groups = (output_channels + 3) / 4;
    const auto grouped = [&](int first, int last) {
      body(first * 4, std::min(output_channels, last * 4));
    };
    if (PointwiseWorkParallel(work, groups, output_channels, plane)) ParallelFor(groups, grouped);
    else grouped(0, groups);
    return;
  }
#endif
  if (PointwiseWorkParallel(work, output_channels, output_channels, plane)) ParallelFor(output_channels, body);
  else body(0, output_channels);
}

void PointwiseConvAddReluBatch(float* dst, const float* src, const float* weights,
                               const float* bias, const float* residual, int batches,
                               int output_channels, int input_channels,
                               std::size_t plane) {
  if (!dst || !src || !weights || !residual || batches <= 0 || output_channels <= 0 ||
      input_channels <= 0 || plane == 0) return;
  if (batches == 1) {
    PointwiseConvAddRelu(dst, src, weights, bias, residual, output_channels, input_channels,
                         plane);
    return;
  }
  const std::size_t input_batch = std::size_t(input_channels) * plane;
  const std::size_t output_batch = std::size_t(output_channels) * plane;
  const auto work = std::int64_t(batches) * output_channels * input_channels *
                    static_cast<std::int64_t>(plane);
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512()) {
    const bool tile8 = UseAvx512PointwiseRelu8(output_channels, input_channels, plane);
    const int tile = tile8 ? 8 : 4;
    const int groups = (output_channels + tile - 1) / tile;
    const int tasks = batches * groups;
    const auto body = [&](int first, int last) {
      for (int task = first; task < last; ++task) {
        const int batch = task / groups;
        const int group = task - batch * groups;
        float* output = dst + std::size_t(batch) * output_batch;
        const float* input = src + std::size_t(batch) * input_batch;
        const float* add = residual + std::size_t(batch) * output_batch;
        const int first_output = group * tile;
        const int last_output = std::min(output_channels, first_output + tile);
        if (tile8) {
          Avx512PointwiseConvAddRelu8(output, input, weights, bias, add, first_output,
                                      last_output, input_channels, plane);
        } else {
          Avx512PointwiseConvAddRelu4(output, input, weights, bias, add, first_output,
                                      last_output, input_channels, plane);
        }
      }
    };
    if (work >= 3000000 && tasks > 1) ParallelFor(tasks, body); else body(0, tasks);
    return;
  }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2()) {
    const int groups = (output_channels + 3) / 4;
    const int tasks = batches * groups;
    const auto body = [&](int first, int last) {
      for (int task = first; task < last; ++task) {
        const int batch = task / groups;
        const int group = task - batch * groups;
        Avx2PointwiseConvAddRelu4(dst + std::size_t(batch) * output_batch,
                                   src + std::size_t(batch) * input_batch, weights, bias,
                                   residual + std::size_t(batch) * output_batch, group * 4,
                                   std::min(output_channels, group * 4 + 4), input_channels,
                                   plane);
      }
    };
    if (work >= 3000000 && tasks > 1) ParallelFor(tasks, body); else body(0, tasks);
    return;
  }
#endif
  for (int batch = 0; batch < batches; ++batch) {
    PointwiseConvAddRelu(dst + std::size_t(batch) * output_batch,
                         src + std::size_t(batch) * input_batch, weights, bias,
                         residual + std::size_t(batch) * output_batch, output_channels,
                         input_channels, plane);
  }
}

void PointwiseConvAddSwish(float* dst, const float* src, const float* weights,
                           const float* bias, const float* residual,
                           int output_channels, int input_channels,
                           std::size_t plane) {
  // Keep the convolution and shortcut fused in the architecture-specific
  // kernels, then overwrite the final allocation with exact Swish. This is
  // still one activation allocation and avoids the generic graph's separate
  // Add and Sigmoid tensors. The Vulkan path has a true one-dispatch mode 9.
  PointwiseConvAdd(dst, src, weights, bias, residual, output_channels,
                   input_channels, plane);
  Swish(dst, dst, std::size_t(output_channels) * plane);
}

void PointwiseConvAddSwishBatch(float* dst, const float* src, const float* weights,
                                 const float* bias, const float* residual, int batches,
                                 int output_channels, int input_channels,
                                 std::size_t plane) {
  if (!dst || !src || !weights || !residual || batches <= 0 || output_channels <= 0 ||
      input_channels <= 0 || plane == 0) return;
  PointwiseConvAddBatch(dst, src, weights, bias, residual, batches, output_channels,
                        input_channels, plane);
  Swish(dst, dst, std::size_t(batches) * output_channels * plane);
}

void PointwiseConvHardSwish(float* dst, const float* src, const float* weights,
                            const float* bias, int output_channels,
                            int input_channels, std::size_t plane) {
  // The canonical PP-OCR HardSwish remains mathematically identical to a
  // normal pointwise Conv followed by the existing ISA-dispatched activation,
  // but it avoids a graph-level intermediate allocation and lets the Conv
  // channel tiles feed the final tensor directly.
  PointwiseConv(dst, src, weights, bias, output_channels, input_channels, plane);
  HardSwish(dst, dst, std::size_t(output_channels) * plane);
}

void PointwiseConvHardSwishBatch(float* dst, const float* src, const float* weights,
                                 const float* bias, int batches, int output_channels,
                                 int input_channels, std::size_t plane) {
  if (!dst || !src || !weights || batches <= 0 || output_channels <= 0 ||
      input_channels <= 0 || plane == 0) return;
  PointwiseConvBatch(dst, src, weights, bias, batches, output_channels, input_channels, plane);
  HardSwish(dst, dst, std::size_t(batches) * output_channels * plane);
}

void PointwiseConvRelu(float* dst, const float* src, const float* weights,
                       const float* bias, int output_channels, int input_channels,
                       std::size_t plane) {
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512()) {
    const auto body = [&](int first, int last) {
      const bool tile8 = UseAvx512PointwiseRelu8(output_channels, input_channels, plane);
      if (tile8) {
        Avx512PointwiseConvRelu8(dst, src, weights, bias, first, last, input_channels, plane);
      } else {
        Avx512PointwiseConvRelu4(dst, src, weights, bias, first, last, input_channels, plane);
      }
    };
    const auto work = std::int64_t(output_channels) * input_channels * static_cast<std::int64_t>(plane);
    const bool tile8 = UseAvx512PointwiseRelu8(output_channels, input_channels, plane);
    const int tile = tile8 ? 8 : 4;
    const int groups = (output_channels + tile - 1) / tile;
    const auto grouped = [&](int first, int last) {
      body(first * tile, std::min(output_channels, last * tile));
    };
    if (PointwiseWorkParallel(work, groups, output_channels, plane)) ParallelFor(groups, grouped);
    else grouped(0, groups);
    return;
  }
#endif
  PointwiseConv(dst, src, weights, bias, output_channels, input_channels, plane);
  Relu(dst, dst, std::size_t(output_channels) * plane);
}

void PointwiseConvReluBatch(float* dst, const float* src, const float* weights,
                            const float* bias, int batches, int output_channels,
                            int input_channels, std::size_t plane) {
  if (!dst || !src || !weights || batches <= 0 || output_channels <= 0 ||
      input_channels <= 0 || plane == 0) return;
  if (batches == 1) {
    PointwiseConvRelu(dst, src, weights, bias, output_channels, input_channels, plane);
    return;
  }
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512()) {
    const auto work = std::int64_t(batches) * output_channels * input_channels *
                      static_cast<std::int64_t>(plane);
    const bool tile8 = UseAvx512PointwiseRelu8(output_channels, input_channels, plane);
    const int tile = tile8 ? 8 : 4;
    const int groups = (output_channels + tile - 1) / tile;
    const int tasks = batches * groups;
    const std::size_t input_batch = std::size_t(input_channels) * plane;
    const std::size_t output_batch = std::size_t(output_channels) * plane;
    const auto body = [&](int first, int last) {
      for (int task = first; task < last; ++task) {
        const int batch = task / groups;
        const int group = task - batch * groups;
        const int first_output = group * tile;
        const int last_output = std::min(output_channels, first_output + tile);
        if (tile8) {
          Avx512PointwiseConvRelu8(dst + std::size_t(batch) * output_batch,
                                   src + std::size_t(batch) * input_batch, weights, bias,
                                   first_output, last_output, input_channels, plane);
        } else {
          Avx512PointwiseConvRelu4(dst + std::size_t(batch) * output_batch,
                                   src + std::size_t(batch) * input_batch, weights, bias,
                                   first_output, last_output, input_channels, plane);
        }
      }
    };
    if (work >= 3000000 && tasks > 1) ParallelFor(tasks, body); else body(0, tasks);
    return;
  }
#endif
  const std::size_t input_batch = std::size_t(input_channels) * plane;
  const std::size_t output_batch = std::size_t(output_channels) * plane;
  for (int batch = 0; batch < batches; ++batch) {
    PointwiseConvRelu(dst + std::size_t(batch) * output_batch,
                      src + std::size_t(batch) * input_batch, weights, bias,
                      output_channels, input_channels, plane);
  }
}

void Conv2d(float* dst, const float* src, const float* weights, const float* bias,
            int input_channels, int output_channels, int input_h, int input_w,
            int output_h, int output_w, int kernel_h, int kernel_w,
            int stride_h, int stride_w, int pad_top, int pad_left, bool relu) {
  const std::size_t in_plane = std::size_t(input_h) * input_w;
  const std::size_t out_plane = std::size_t(output_h) * output_w;
  const std::size_t filter_plane = std::size_t(kernel_h) * kernel_w;
  const int first_y = std::max(0, (pad_top + stride_h - 1) / stride_h);
  const int first_x = std::max(0, (pad_left + stride_w - 1) / stride_w);
  const int max_y = input_h + pad_top - kernel_h;
  const int max_x = input_w + pad_left - kernel_w;
  const int last_y = max_y < 0 ? 0 : std::min(output_h, max_y / stride_h + 1);
  const int last_x = max_x < 0 ? 0 : std::min(output_w, max_x / stride_w + 1);
  const auto work = std::int64_t(input_channels) * output_channels * output_h * output_w * kernel_h * kernel_w;
  static const int parallel_min = [] {
    int threshold = 4000000;
    if (const char* configured = std::getenv("PPOCR_CONV_PARALLEL_MIN_WORK")) {
      char* end = nullptr;
      const long parsed = std::strtol(configured, &end, 10);
      if (end != configured && *end == '\0' && parsed > 0) threshold = static_cast<int>(parsed);
    }
    return threshold;
  }();
  const bool parallel = work >= parallel_min && output_channels >= 8;
  // PP-OCRv6 also has 2x2 valid and 7x7/asymmetric 3/5/7x1 detector
  // kernels; vectorize interior pixels while preserving scalar boundaries.
  const bool simd_kernel = (kernel_h == kernel_w &&
                            (kernel_h == 2 || kernel_h == 3 || kernel_h == 5 || kernel_h == 7)) ||
                           ((kernel_h == 1 || kernel_w == 1) &&
                            (kernel_h == 3 || kernel_h == 5 || kernel_h == 7 ||
                             kernel_w == 3 || kernel_w == 5 || kernel_w == 7));
#if defined(PPOCR_HAS_AVX512_KERNELS)
  // The medium detector's early reconstruction stem contains broad 2x2
  // SAME_UPPER convolutions. Their interior has the same four contiguous
  // loads as the valid 2x2 kernel, while only the final row/column sees zero
  // padding. Keep that hot interior explicitly unrolled instead of routing
  // it through the runtime-sized generic convolution loops.
  static const bool avx512_conv2x2_same_enabled =
      std::getenv("PPOCR_DISABLE_AVX512_CONV2X2_SAME") == nullptr;
  if (HasAvx512() && avx512_conv2x2_same_enabled &&
      stride_h == 1 && stride_w == 1 && kernel_h == 2 && kernel_w == 2 &&
      pad_top == 0 && pad_left == 0 && output_h == input_h && output_w == input_w) {
    // Conv.1 is 16→8: one serial y-outer walk reuses the 16 source planes
    // across both four-output tiles. Conv.2 is 8→16: OC-parallel reread the
    // 8 source maps per worker. Row tiles keep those planes in L1 and write
    // all 16 outputs once.
    if (output_channels <= 8) {
      // Conv.1 is 16→8 SAME 2x2 on 80x352. Serial OC keeps 16 source planes
      // hot; independent output rows let the SIMD pool share that walk.
      // Rec 2x2 maps are smaller than H=32. `PPOCR_DISABLE_AVX512_CONV1_ROW_PF`
      // restores one-thread y-outer.
      static const bool row_pf =
          std::getenv("PPOCR_DISABLE_AVX512_CONV1_ROW_PF") == nullptr;
      if (row_pf && input_h >= 32) {
        ParallelFor(input_h, [&](int first, int last) {
          Avx512Conv2x2SameUpper(dst, src, weights, bias, 0, output_channels,
                                  input_channels, input_h, input_w, relu, first, last);
        });
        return;
      }
      Avx512Conv2x2SameUpper(dst, src, weights, bias, 0, output_channels,
                              input_channels, input_h, input_w, relu);
      return;
    }
    // Conv.2 is 8→16 SAME 2x2 on 80x352: the same MAC count as serial
    // Conv.1 (16→8) but OC-parallel reread the 8 source maps (~0.9 MiB)
    // per worker and ran ~5x slower. One y-outer walk keeps those planes
    // in L2 and writes all 16 outputs. Fat maps with more IC still split.
    // `PPOCR_DISABLE_AVX512_CONV2X2_SERIAL_OC` restores the OC ParallelFor.
    static const bool serial_oc =
        std::getenv("PPOCR_DISABLE_AVX512_CONV2X2_SERIAL_OC") == nullptr;
    if (serial_oc && input_channels <= 8 && output_channels <= 32) {
      // Conv.2 8→16 on 80x352: serial OC keeps the 8 source maps in L2.
      // Independent output rows let the SIMD pool share that walk.
      // `PPOCR_DISABLE_AVX512_CONV2X2_ROW_PF` restores one-thread y-outer.
      static const bool row_pf =
          std::getenv("PPOCR_DISABLE_AVX512_CONV2X2_ROW_PF") == nullptr;
      if (row_pf && input_h >= 32) {
        ParallelFor(input_h, [&](int first, int last) {
          Avx512Conv2x2SameUpper(dst, src, weights, bias, 0, output_channels,
                                  input_channels, input_h, input_w, relu, first, last);
        });
        return;
      }
      Avx512Conv2x2SameUpper(dst, src, weights, bias, 0, output_channels,
                              input_channels, input_h, input_w, relu);
      return;
    }
    static const bool row_tiles =
        std::getenv("PPOCR_ENABLE_AVX512_CONV2X2_ROW_TILE") != nullptr &&
        std::getenv("PPOCR_DISABLE_AVX512_CONV2X2_ROW_TILE") == nullptr;
    if (row_tiles && input_h >= 16 && output_channels >= 16) {
      const int row_step = 8;
      const int tiles = (input_h + row_step - 1) / row_step;
      const auto avx = [&](int first, int last) {
        for (int tile = first; tile < last; ++tile) {
          const int y0 = tile * row_step;
          const int y1 = std::min(input_h, y0 + row_step);
          Avx512Conv2x2SameUpper(dst, src, weights, bias, 0, output_channels,
                                  input_channels, input_h, input_w, relu, y0, y1);
        }
      };
      if (parallel && tiles > 2) ParallelFor(tiles, avx);
      else avx(0, tiles);
      return;
    }
    const int tile = output_channels >= 4 ? 4 : 1;
    const auto avx = [&](int first, int last) {
      Avx512Conv2x2SameUpper(dst, src, weights, bias, first * tile,
                              std::min(output_channels, last * tile), input_channels,
                              input_h, input_w, relu);
    };
    const int groups = (output_channels + tile - 1) / tile;
    if (parallel && groups > 2) ParallelFor(groups, avx);
    else avx(0, groups);
    return;
  }
#endif
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512() && stride_h == 1 && stride_w == 1 && kernel_h == 2 && kernel_w == 2 &&
      pad_top == 0 && pad_left == 0 && output_h == input_h - 1 && output_w == input_w - 1) {
    const bool tile4 = UseAvx512Conv2x2Tile4(output_channels, input_channels, input_h, input_w);
    const int tile = tile4 ? 4 : 1;
    const auto avx = [&](int first, int last) {
      if (tile4) {
        Avx512Conv2x2Validx4(dst, src, weights, bias, first * tile,
                              std::min(output_channels, last * tile), input_channels,
                              input_h, input_w, relu);
      } else {
        Avx512Conv2x2Valid(dst, src, weights, bias, first, last, input_channels, input_h, input_w, relu);
      }
    };
    const int groups = (output_channels + tile - 1) / tile;
    if (parallel) ParallelFor(groups, avx); else avx(0, groups);
    return;
  }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2() && stride_h == 1 && stride_w == 1 && kernel_h == 2 && kernel_w == 2 &&
      pad_top == 0 && pad_left == 0 && output_h == input_h - 1 && output_w == input_w - 1) {
    const auto avx = [&](int first, int last) {
      Avx2Conv2x2Valid(dst, src, weights, bias, first, last, input_channels, input_h, input_w, relu);
    };
    if (parallel) ParallelFor(output_channels, avx); else avx(0, output_channels);
    return;
  }
#endif
#if defined(PPOCR_HAS_AVX512_KERNELS)
  // AVX-512 must win over the AVX2 four-output tile: HasAvx2() is also true
  // on every AVX-512 CPU, and a one-channel ParallelFor would skip the
  // four-filter body inside Avx512Conv3x3Stride2. Match the NCHW-batch
  // scheduler so the detector stem (N=1) keeps that tile.
  if (HasAvx512() && stride_h == 2 && stride_w == 2 && kernel_h == 3 && kernel_w == 3) {
    // C=3 RGB stems (det Conv.0, rec Conv.0) re-read the same three planes
    // if each output tile is a ParallelFor task. Serial C3 tiles keep those
    // planes in L2; `PPOCR_DISABLE_RGB_STRIDE2` restores the generic 4-wide
    // kernel. `PPOCR_ENABLE_RGB_STRIDE2` remains accepted as a no-op alias.
    const bool rgb = input_channels == 3 && output_channels >= 8 &&
        std::getenv("PPOCR_DISABLE_RGB_STRIDE2") == nullptr;
    static const bool rgb_tile16 =
        std::getenv("PPOCR_DISABLE_AVX512_RGB_TILE16") == nullptr;
    constexpr int tile4 = 4;
    const bool use_output_tile =
        output_channels >= 4 &&
        std::getenv("PPOCR_DISABLE_AVX512_STRIDE2_BATCH_TILE4") == nullptr;
    // Rec Conv.1 is 24→48 on a ~12x80 map: an eight-output tile cuts
    // source rereads without the register pressure seen on fat detector
    // stride-2 maps (Conv.3 is 180x38). Keep those on four-output tiles.
    // Det Conv.0 is 3→16: a 16-wide tile reads the three RGB planes once.
    const int tile_width = rgb ? (rgb_tile16 && output_channels >= 16 ? 16 : 8) :
        (use_output_tile
             ? ((input_channels >= 16 && output_channels >= 8 &&
                 output_h * output_w < 4096)
                    ? 8
                    : tile4)
             : 1);
    const auto avx = [&](int first, int last) {
      for (int group = first; group < last; ++group) {
        const int first_output = group * tile_width;
        const int last_output = std::min(output_channels, first_output + tile_width);
        if (rgb) {
          Avx512Conv3x3Stride2C3(dst, src, weights, bias, first_output, last_output,
                                 input_h, input_w, output_h, output_w, pad_top, pad_left,
                                 relu, UseAvx512Stride2PackedLoads());
        } else {
          Avx512Conv3x3Stride2(dst, src, weights, bias, first_output, last_output,
                               input_channels, input_h, input_w, output_h, output_w,
                               pad_top, pad_left, relu, UseAvx512Stride2PackedLoads());
        }
      }
    };
    const int groups = (output_channels + tile_width - 1) / tile_width;
    // Det Conv.0 is 3→16 on 80x352: serial 16-wide keeps RGB in L2 but uses
    // one worker. Row ParallelFor keeps that tile and skips rec 24-high crops.
    // `PPOCR_DISABLE_AVX512_RGB_ROW_PF` restores one-thread y-outer.
    static const bool rgb_row_pf =
        std::getenv("PPOCR_DISABLE_AVX512_RGB_ROW_PF") == nullptr;
    if (rgb && rgb_row_pf && output_h >= 64 && output_w >= 64) {
      ParallelFor(output_h, [&](int first, int last) {
        Avx512Conv3x3Stride2C3(dst, src, weights, bias, 0, output_channels, input_h,
                               input_w, output_h, output_w, pad_top, pad_left, relu,
                               UseAvx512Stride2PackedLoads(), first, last);
      });
      return;
    }
    // Fat 3x3 s2 maps (Conv.3, 32ch) still split by output tile. RGB C=3
    // stays serial so the three input planes are not reread by every worker.
    const bool split = (parallel || groups > 1) && (input_channels >= 8 || !rgb);
    if (split) ParallelFor(groups, avx);
    else avx(0, groups);
    return;
  }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2() && stride_h == 2 && stride_w == 2 && kernel_h == 3 && kernel_w == 3 &&
      output_channels >= 4 && std::getenv("PPOCR_DISABLE_STRIDE2_TILE4") == nullptr) {
    const auto avx = [&](int first, int last) {
      Avx2Conv3x3Stride2x4(dst, src, weights, bias, first * 4,
                            std::min(output_channels, last * 4), input_channels,
                            input_h, input_w, output_h, output_w, pad_top, pad_left, relu);
    };
    const int groups = (output_channels + 3) / 4;
    if (parallel) ParallelFor(groups, avx); else avx(0, groups);
    return;
  }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2() && stride_h == 2 && stride_w == 2 &&
      kernel_h == 3 && kernel_w == 3) {
    const auto avx = [&](int first, int last) {
      Avx2Conv3x3Stride2(dst, src, weights, bias, first, last, input_channels,
                          input_h, input_w, output_h, output_w, pad_top, pad_left, relu);
    };
    if (parallel) ParallelFor(output_channels, avx); else avx(0, output_channels);
    return;
  }
#endif
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512() && stride_h == 1 && stride_w == 1 && output_channels >= 4 &&
      ((kernel_h == 1 && (kernel_w == 3 || kernel_w == 5 || kernel_w == 7)) ||
       (kernel_w == 1 && (kernel_h == 3 || kernel_h == 5 || kernel_h == 7)))) {
    const auto avx = [&](int first, int last) {
      Avx512ConvAsymmetricStride1x4(dst, src, weights, bias, first * 4,
                                     std::min(output_channels, last * 4), input_channels,
                                     input_h, input_w, output_h, output_w, kernel_h, kernel_w,
                                     pad_top, pad_left);
    };
    const int groups = (output_channels + 3) / 4;
    if (parallel) ParallelFor(groups, avx); else avx(0, groups);
    return;
  }
#endif
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512() && stride_h == 1 && stride_w == 1 &&
      (kernel_h == 5 || kernel_h == 7) && kernel_h == kernel_w && output_channels >= 4) {
    const bool tile8 = UseAvx512ConvWideTile8(output_channels, input_channels,
                                               input_h, input_w, kernel_h);
    const auto avx = [&](int first, int last) {
      if (tile8) {
        Avx512ConvOddStride1x8(dst, src, weights, bias, first * 8,
                                std::min(output_channels, last * 8), input_channels,
                                input_h, input_w, output_h, output_w, kernel_h, pad_top, pad_left);
      } else {
        Avx512ConvOddStride1x4(dst, src, weights, bias, first * 4,
                                std::min(output_channels, last * 4), input_channels,
                                input_h, input_w, output_h, output_w, kernel_h, pad_top, pad_left);
      }
    };
    const int groups = (output_channels + (tile8 ? 7 : 3)) / (tile8 ? 8 : 4);
    if (parallel) ParallelFor(groups, avx); else avx(0, groups);
    return;
  }
#endif
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512() && stride_h == 1 && stride_w == 1 && kernel_h == 3 && kernel_w == 3 &&
      output_channels >= 4) {
    const bool tile8 = UseAvx512Conv3x3Tile8() && output_channels >= 8 &&
                       input_channels >= 32 && out_plane >= 16384;
    const int tile = tile8 ? 8 : 4;
    const auto avx = [&](int first, int last) {
      // The kernel fuses four adjacent output channels. Split parallel work
      // by those groups, not individual channels, so every worker retains its
      // input-load reuse and only the final incomplete group uses the scalar
      // tail path.
      if (tile8) {
        Avx512Conv3x3Stride1x8(dst, src, weights, bias, first * tile,
                                std::min(output_channels, last * tile), input_channels,
                                input_h, input_w, output_h, output_w, pad_top, pad_left, relu);
      } else {
        Avx512Conv3x3Stride1x4(dst, src, weights, bias, first * tile,
                                std::min(output_channels, last * tile), input_channels,
                                input_h, input_w, output_h, output_w, pad_top, pad_left, relu);
      }
    };
    const int groups = (output_channels + tile - 1) / tile;
    if (parallel) ParallelFor(groups, avx); else avx(0, groups);
    return;
  }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2() && stride_h == 1 && stride_w == 1 && kernel_h == 3 && kernel_w == 3 &&
      output_channels >= 4) {
    const auto avx = [&](int first, int last) {
      Avx2Conv3x3Stride1x4(dst, src, weights, bias, first * 4,
                            std::min(output_channels, last * 4), input_channels,
                            input_h, input_w, output_h, output_w, pad_top, pad_left, relu);
    };
    const int groups = (output_channels + 3) / 4;
    if (parallel) ParallelFor(groups, avx); else avx(0, groups);
    return;
  }
#endif
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512() && stride_h == 1 && stride_w == 1 && simd_kernel) {
    const auto avx = [&](int first, int last) {
      Avx512Conv2d(dst, src, weights, bias, first, last, input_channels,
                   input_h, input_w, output_h, output_w, kernel_h, kernel_w,
                   pad_top, pad_left);
      // The generic vector Conv primitive intentionally has no activation
      // argument. Do not silently drop a graph-level ReLU on shapes outside
      // the dedicated fused 2x2/3x3 kernels.
      if (relu) {
        for (int output = first; output < last; ++output) {
          float* values = dst + std::size_t(output) * out_plane;
          Relu(values, values, out_plane);
        }
      }
    };
    if (parallel) ParallelFor(output_channels, avx); else avx(0, output_channels);
    return;
  }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2() && stride_h == 1 && stride_w == 1 && simd_kernel) {
    const auto avx = [&](int first, int last) {
      Avx2Conv2d(dst, src, weights, bias, first, last, input_channels,
                 input_h, input_w, output_h, output_w, kernel_h, kernel_w,
                 pad_top, pad_left);
      if (relu) {
        for (int output = first; output < last; ++output) {
          float* values = dst + std::size_t(output) * out_plane;
          Relu(values, values, out_plane);
        }
      }
    };
    if (parallel) ParallelFor(output_channels, avx); else avx(0, output_channels);
    return;
  }
#endif
  const auto body = [&](int first, int last) {
    for (int output = first; output < last; ++output) {
      float* out = dst + std::size_t(output) * out_plane;
      const float* filter = weights + std::size_t(output) * input_channels * filter_plane;
      const float base = bias ? bias[output] : 0.F;
      for (int oy = 0; oy < output_h; ++oy) for (int ox = 0; ox < output_w; ++ox) {
        float sum = base;
        const int start_y = oy * stride_h - pad_top;
        const int start_x = ox * stride_w - pad_left;
        const bool interior = oy >= first_y && oy < last_y && ox >= first_x && ox < last_x;
        for (int input = 0; input < input_channels; ++input) {
          const float* plane = src + std::size_t(input) * in_plane;
          const float* kernel = filter + std::size_t(input) * filter_plane;
          if (interior) {
            const float* pixel = plane + std::size_t(start_y) * input_w + start_x;
            for (int ky = 0; ky < kernel_h; ++ky) {
              const float* row = pixel + std::size_t(ky) * input_w;
              const float* krow = kernel + std::size_t(ky) * kernel_w;
              for (int kx = 0; kx < kernel_w; ++kx) sum += row[kx] * krow[kx];
            }
          } else {
            for (int ky = 0; ky < kernel_h; ++ky) {
              const int iy = start_y + ky;
              if (iy < 0 || iy >= input_h) continue;
              for (int kx = 0; kx < kernel_w; ++kx) {
                const int ix = start_x + kx;
                if (ix >= 0 && ix < input_w) sum += plane[std::size_t(iy) * input_w + ix] * kernel[std::size_t(ky) * kernel_w + kx];
              }
            }
          }
        }
        out[std::size_t(oy) * output_w + ox] = relu ? std::max(sum, 0.F) : sum;
      }
    }
  };
  if (parallel) ParallelFor(output_channels, body);
  else body(0, output_channels);
}

struct ConcatWPack {
  const float* oc_major = nullptr;
  const float* tap_major = nullptr;
};

static ConcatWPack ConcatSourceWpacks(const float* weights, int source_count, int ic,
                                int output_channels, int total_channels,
                                std::vector<float>& scratch) {
  const std::size_t w_src = std::size_t(output_channels) * ic * 9;
  const std::size_t tap_src = std::size_t(ic) * 9 * 16;
  const auto pack_oc = [&](std::vector<float>& dst) {
    dst.resize(std::size_t(source_count) * w_src);
    for (int source = 0; source < source_count; ++source) {
      const int channel0 = source * ic;
      float* wsrc = dst.data() + std::size_t(source) * w_src;
      for (int output = 0; output < output_channels; ++output) {
        std::memcpy(wsrc + std::size_t(output) * ic * 9,
                    weights + (std::size_t(output) * total_channels + channel0) * 9,
                    std::size_t(ic) * 9 * sizeof(float));
      }
    }
  };
  const auto pack_tap = [&](const std::vector<float>& oc_major, std::vector<float>& dst) {
    if (output_channels != 16) {
      dst.clear();
      return;
    }
    dst.resize(std::size_t(source_count) * tap_src);
    for (int source = 0; source < source_count; ++source) {
      const float* wsrc = oc_major.data() + std::size_t(source) * w_src;
      float* tsrc = dst.data() + std::size_t(source) * tap_src;
      for (int input = 0; input < ic; ++input) {
        for (int ki = 0; ki < 9; ++ki) {
          float* lane = tsrc + (std::size_t(input) * 9 + ki) * 16;
          for (int output = 0; output < 16; ++output)
            lane[output] = wsrc[std::size_t(output) * ic * 9 + input * 9 + ki];
        }
      }
    }
  };
  static const bool cache_wpack =
      std::getenv("PPOCR_DISABLE_AVX512_CONCAT_WPACK_CACHE") == nullptr;
  static const bool tap_pack =
      std::getenv("PPOCR_DISABLE_AVX512_CONCAT_TAP_PACK") == nullptr;
  if (!cache_wpack) {
    pack_oc(scratch);
    static thread_local std::vector<float> tap_scratch;
    if (tap_pack) pack_tap(scratch, tap_scratch);
    else tap_scratch.clear();
    return ConcatWPack{scratch.data(), tap_scratch.empty() ? nullptr : tap_scratch.data()};
  }
  struct Cache {
    const float* weights = nullptr;
    int source_count = 0;
    int ic = 0;
    int output_channels = 0;
    int total_channels = 0;
    std::shared_ptr<std::vector<float>> packed;
    std::shared_ptr<std::vector<float>> tap_packed;
  };
  static std::mutex mu;
  static Cache cache;
  std::lock_guard lock(mu);
  if (cache.packed && cache.weights == weights && cache.source_count == source_count &&
      cache.ic == ic && cache.output_channels == output_channels &&
      cache.total_channels == total_channels) {
    return ConcatWPack{cache.packed->data(),
                       (tap_pack && cache.tap_packed && !cache.tap_packed->empty())
                           ? cache.tap_packed->data()
                           : nullptr};
  }
  auto packed = std::make_shared<std::vector<float>>();
  pack_oc(*packed);
  auto tap_packed = std::make_shared<std::vector<float>>();
  if (tap_pack) pack_tap(*packed, *tap_packed);
  cache = Cache{weights, source_count, ic, output_channels, total_channels, packed,
                tap_packed};
  return ConcatWPack{cache.packed->data(),
                     (tap_pack && !tap_packed->empty()) ? tap_packed->data() : nullptr};
}

void ConcatChannelConv2d(float* dst, const float* const* sources,
                         const int* source_channels, int source_count,
                         const float* weights, const float* bias,
                         int output_channels, int input_h, int input_w,
                         int output_h, int output_w, int kernel_h, int kernel_w,
                         int stride_h, int stride_w, int pad_top, int pad_left,
                         bool relu) {
  if (!dst || !sources || !source_channels || !weights || source_count <= 0 ||
      output_channels <= 0) return;
  int total_channels = 0;
  for (int source = 0; source < source_count; ++source) {
    if (!sources[source] || source_channels[source] <= 0) return;
    total_channels += source_channels[source];
  }
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_count =
      std::size_t(output_channels) * output_h * output_w;
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512() && kernel_h == 3 && kernel_w == 3 &&
      (stride_h == 1 || stride_h == 2) && stride_h == stride_w &&
      output_channels >= 4) {
    std::vector<const float*> planes(static_cast<std::size_t>(total_channels));
    int plane = 0;
    for (int source = 0; source < source_count; ++source) {
      for (int channel = 0; channel < source_channels[source]; ++channel) {
        planes[static_cast<std::size_t>(plane++)] =
            sources[source] + std::size_t(channel) * input_plane;
      }
    }
    // Concat.2 x8 was measured slower than four-output tiles on this host
    // (register pressure). Keep it as an explicit A/B.
    static const bool concat_tile8 =
        std::getenv("PPOCR_ENABLE_AVX512_CONCAT_TILE8") != nullptr &&
        std::getenv("PPOCR_DISABLE_AVX512_CONCAT_TILE8") == nullptr;
    // Fat Concat.2 row strips were measured as an e2e miss versus OC-parallel
    // x4 on this host. Keep the row-tile path as an explicit A/B.
    static const bool concat_row_tile =
        std::getenv("PPOCR_ENABLE_AVX512_CONCAT_ROW_TILE") != nullptr &&
        std::getenv("PPOCR_DISABLE_AVX512_CONCAT_ROW_TILE") == nullptr;
    const bool tile8 = concat_tile8 && stride_h == 1 && output_channels >= 8 &&
        output_channels <= 16 && total_channels >= 32;
    const int tile = tile8 ? 8 : 4;
    const int groups = (output_channels + tile - 1) / tile;
    const bool packed = UseAvx512Stride2PackedLoads();
    // Concat.2 (64→16 s1 on ~40x176) is source-bound; serial row strips keep
    // the 64 planes in L2. Stem MaxPool+Concat is stride-2 on an 80x352-class
    // map and is compute-bound: serializing its four OC tiles was an e2e
    // regression (~+4 ms), so leave s2 on the existing OC ParallelFor.
    const bool fat_sources = stride_h == 1 && total_channels >= 48 &&
        output_channels <= 16;
    static const bool concat_split =
        std::getenv("PPOCR_DISABLE_AVX512_CONCAT_SPLIT") == nullptr;
    bool even16 = source_count >= 2;
    for (int source = 0; source < source_count; ++source)
      if (source_channels[source] != 16) even16 = false;
    // Stem MaxPoolConcat is two 16-ch maps, 3x3 s2 on 80x352. A 32-ch gather
    // rereads both allocations per OC tile; two 16→16 s2 passes keep one
    // source hot. `PPOCR_DISABLE_AVX512_STEM_SPLIT` restores OC-parallel.
    static const bool stem_split =
        std::getenv("PPOCR_ENABLE_AVX512_STEM_SPLIT") != nullptr &&
        std::getenv("PPOCR_DISABLE_AVX512_STEM_SPLIT") == nullptr;
    if (stem_split && concat_split && even16 && stride_h == 2 &&
        source_count == 2 && output_channels == 16 && input_h >= 32 &&
        kernel_h == 3 && kernel_w == 3) {
      const std::size_t out_n = std::size_t(output_channels) * output_h * output_w;
      const int ic = 16;
      const std::size_t w_src = std::size_t(output_channels) * ic * 9;
      std::vector<float> wpacks(2 * w_src);
      for (int source = 0; source < 2; ++source) {
        const int channel0 = source * ic;
        float* wsrc = wpacks.data() + std::size_t(source) * w_src;
        for (int output = 0; output < output_channels; ++output) {
          std::memcpy(wsrc + std::size_t(output) * ic * 9,
                      weights + (std::size_t(output) * total_channels + channel0) * 9,
                      std::size_t(ic) * 9 * sizeof(float));
        }
      }
      std::vector<float> partial(out_n);
      Conv2d(dst, sources[0], wpacks.data(), bias, ic, output_channels, input_h,
             input_w, output_h, output_w, 3, 3, 2, 2, pad_top, pad_left, false);
      Conv2d(partial.data(), sources[1], wpacks.data() + w_src, nullptr, ic,
             output_channels, input_h, input_w, output_h, output_w, 3, 3, 2, 2,
             pad_top, pad_left, false);
      BinaryInplace(dst, partial.data(), out_n, BinaryOp::add);
      if (relu) Relu(dst, dst, out_n);
      return;
    }
    // Concat.2 is four 16-ch maps. Each 16-ch 40x176 plane fits in L2;
    // a 64-ch gather across four allocations does not. Four 16->16 3x3
    // passes plus Add matches the math and keeps one source hot.
    if (concat_split && fat_sources && even16 && source_count >= 2 && !tile8 &&
        output_channels == 16) {
        const std::size_t out_n = std::size_t(output_channels) * output_h * output_w;
        const int ic = 16;
        const std::size_t w_src = std::size_t(output_channels) * ic * 9;
        std::vector<float> wpack_scratch;
        const ConcatWPack packs = ConcatSourceWpacks(weights, source_count, ic,
                                                 output_channels, total_channels,
                                                 wpack_scratch);
        const float* wpacks = packs.oc_major;
        const std::size_t tap_src = std::size_t(ic) * 9 * 16;
        // Last-source fused ReLU reuses Avx512Conv3x3Stride1x4's existing last-input
        // max; the extra Relu pass over 16x40x176 was a separate store stream.
        // `PPOCR_DISABLE_AVX512_CONCAT_FUSED_RELU` restores the post-pass.
        static const bool fused_relu =
            std::getenv("PPOCR_DISABLE_AVX512_CONCAT_FUSED_RELU") == nullptr;
        // Four source ParallelFor under-subscribes a 16-worker pool. Independent
        // output rows let every worker keep one 16-ch source hot and accumulate
        // into exclusive dst rows (no 3x partial + Add).
        // `PPOCR_DISABLE_AVX512_CONCAT_ROW_PF` restores source ParallelFor.
        static const bool row_pf =
            std::getenv("PPOCR_DISABLE_AVX512_CONCAT_ROW_PF") == nullptr;
        if (row_pf && output_h >= 32 && output_w >= 64) {
          // One-pass folds every 16-ch source into the 4-OC accumulators so
          // dst is written once. Same-host 8-run missed versus four 16→16
          // accumulating passes (source-hot). Keep as an explicit A/B.
          static const bool onepass =
              std::getenv("PPOCR_ENABLE_AVX512_CONCAT_ONEPASS") != nullptr &&
              std::getenv("PPOCR_DISABLE_AVX512_CONCAT_ONEPASS") == nullptr;
          const auto run_rows = [&](int first, int last) {
            if (onepass) {
              Avx512ConcatSourcesConv3x3x4(
                  dst, sources, source_count, wpacks, bias, ic, output_channels,
                  input_h, input_w, output_h, output_w, pad_top, pad_left, relu,
                  first, last);
              return;
            }
            // 16-OC source-hot: each 16-ch map is read once per source instead
            // of four 4-OC tiles. `PPOCR_DISABLE_AVX512_CONCAT_OC16` restores x4.
            static const bool oc16 =
                std::getenv("PPOCR_DISABLE_AVX512_CONCAT_OC16") == nullptr;
            for (int source = 0; source < source_count; ++source) {
              const bool last_relu = relu && fused_relu && source + 1 == source_count;
              const float* wsrc = wpacks + std::size_t(source) * w_src;
              const float* source_bias = source == 0 ? bias : nullptr;
              if (oc16 && output_channels == 16 && ic == 16) {
                const float* tsrc = packs.tap_major
                    ? packs.tap_major + std::size_t(source) * tap_src
                    : nullptr;
                Avx512Conv3x3Stride1x16(
                    dst, sources[source], wsrc, source_bias, 0, output_channels, ic,
                    input_h, input_w, output_h, output_w, pad_top, pad_left,
                    last_relu, first, last, source > 0, tsrc);
              } else {
                Avx512Conv3x3Stride1x4(
                    dst, sources[source], wsrc, source_bias, 0, output_channels, ic,
                    input_h, input_w, output_h, output_w, pad_top, pad_left,
                    last_relu, nullptr, first, last, source > 0);
              }
            }
          };
          // 4-row tiles amortize the 16-OC source walk over a 3-row halo that
          // stays in L1. Per-row ParallelFor was 40 tiny tasks.
          // `PPOCR_DISABLE_AVX512_CONCAT_ROW4` restores one task per output row.
          static const bool row4 =
              std::getenv("PPOCR_ENABLE_AVX512_CONCAT_ROW4") != nullptr &&
              std::getenv("PPOCR_DISABLE_AVX512_CONCAT_ROW4") == nullptr;
          // Concat.2 is 40 rows. ROW4 yields 10 tasks and under-subscribes a
          // 16-worker pool; ROW2 yields 20 tasks with a 4-row input halo that
          // still fits L1. `PPOCR_DISABLE_AVX512_CONCAT_ROW2` restores one
          // task per output row (or ROW4 when that ENABLE is set).
          static const bool row2 =
              std::getenv("PPOCR_DISABLE_AVX512_CONCAT_ROW2") == nullptr;
          if (row4 || row2) {
            const int step = row4 ? 4 : 2;
            const int tiles = (output_h + step - 1) / step;
            ParallelFor(tiles, [&](int first, int last) {
              for (int tile = first; tile < last; ++tile) {
                const int y0 = tile * step;
                run_rows(y0, std::min(output_h, y0 + step));
              }
            });
          } else {
            ParallelFor(output_h, run_rows);
          }
          if (relu && !fused_relu && !onepass) Relu(dst, dst, out_n);
          return;
        }
        // Row tiles own exclusive output rows so all four 16-ch sources can
        // accumulate into dst without a 3x output partial plus Add.
        static const bool row_accum =
            std::getenv("PPOCR_ENABLE_AVX512_CONCAT_ROW_ACCUM") != nullptr &&
            std::getenv("PPOCR_DISABLE_AVX512_CONCAT_ROW_ACCUM") == nullptr;
        if (row_accum && output_h >= 16) {
          const int row_step = 8;
          const int tiles = (output_h + row_step - 1) / row_step;
          const auto run_tiles = [&](int first, int last) {
            for (int tile = first; tile < last; ++tile) {
              const int y0 = tile * row_step;
              const int y1 = std::min(output_h, y0 + row_step);
              for (int source = 0; source < source_count; ++source) {
                const bool last_relu = relu && fused_relu && source + 1 == source_count;
                Avx512Conv3x3Stride1x4(
                    dst, sources[source], wpacks + std::size_t(source) * w_src,
                    source == 0 ? bias : nullptr, 0, output_channels, ic, input_h,
                    input_w, output_h, output_w, pad_top, pad_left, last_relu, nullptr,
                    y0, y1, source > 0);
              }
            }
          };
          if (tiles > 2) ParallelFor(tiles, run_tiles);
          else run_tiles(0, tiles);
          if (relu && !fused_relu) Relu(dst, dst, out_n);
          return;
        }
        std::vector<float> partials(source_count > 1
            ? std::size_t(source_count - 1) * out_n : 0);
        const bool parallel_sources =
            source_count >= 2 &&
            std::getenv("PPOCR_DISABLE_AVX512_CONCAT_SPLIT_PARALLEL") == nullptr;
        const auto run_source = [&](int first, int last) {
          for (int source = first; source < last; ++source) {
            float* out = source == 0
                ? dst
                : partials.data() + std::size_t(source - 1) * out_n;
            const float* source_bias = source == 0 ? bias : nullptr;
            Avx512Conv3x3Stride1x4(
                out, sources[source], wpacks + std::size_t(source) * w_src,
                source_bias, 0, output_channels, ic, input_h, input_w, output_h,
                output_w, pad_top, pad_left, false, nullptr);
          }
        };
        if (parallel_sources) ParallelFor(source_count, run_source);
        else run_source(0, source_count);
        for (int source = 1; source < source_count; ++source)
          BinaryInplace(dst, partials.data() + std::size_t(source - 1) * out_n, out_n,
                        BinaryOp::add);
        if (relu) Relu(dst, dst, out_n);
        return;
    }
    // Stem MaxPoolConcat is 32→16 3x3 s2 on 80x352. Four OC tiles reread the
    // pooled 16-ch map plus Conv.2 per worker. Row ParallelFor keeps both
    // 16-ch sources in L2 and writes all 16 outputs once. The previous
    // ENABLE stacked with CTC 16-row was an e2e miss; this is isolated.
    // `PPOCR_DISABLE_AVX512_CONCAT_S2_ROW_PF` restores OC tiles.
    // `PPOCR_ENABLE_AVX512_CONCAT_S2_ROW_PF` remains a no-op alias.
    static const bool s2_row_pf =
        std::getenv("PPOCR_DISABLE_AVX512_CONCAT_S2_ROW_PF") == nullptr;
    if (s2_row_pf && stride_h == 2 && even16 && output_channels == 16 &&
        output_h >= 32 && input_h >= 32 && total_channels >= 32) {
      const auto run_rows = [&](int first, int last) {
        Avx512Conv3x3Stride2(dst, nullptr, weights, bias, 0, output_channels,
                             total_channels, input_h, input_w, output_h, output_w,
                             pad_top, pad_left, relu, packed, planes.data(), first,
                             last);
      };
      ParallelFor(output_h, run_rows);
      return;
    }
    const bool use_row_tile = concat_row_tile && fat_sources && !tile8 &&
        output_h >= 16;
    const int row_step = use_row_tile ? 8 : output_h;
    const auto body = [&](int first, int last) {
      for (int y0 = 0; y0 < output_h; y0 += row_step) {
        const int y1 = std::min(output_h, y0 + row_step);
        const int row0 = use_row_tile ? y0 : -1;
        const int row1 = use_row_tile ? y1 : -1;
        for (int group = first; group < last; ++group) {
          const int first_output = group * tile;
          const int last_output = std::min(output_channels, first_output + tile);
          if (stride_h == 1) {
            if (tile8) {
              Avx512Conv3x3Stride1x8(dst, nullptr, weights, bias, first_output, last_output,
                                     total_channels, input_h, input_w, output_h, output_w,
                                     pad_top, pad_left, relu, planes.data());
            } else {
              Avx512Conv3x3Stride1x4(dst, nullptr, weights, bias, first_output, last_output,
                                     total_channels, input_h, input_w, output_h, output_w,
                                     pad_top, pad_left, relu, planes.data(), row0, row1);
            }
          } else {
            Avx512Conv3x3Stride2(dst, nullptr, weights, bias, first_output, last_output,
                                 total_channels, input_h, input_w, output_h, output_w,
                                 pad_top, pad_left, relu, packed, planes.data(), row0, row1);
          }
        }
      }
    };
    // FPN 64→16 s1 with x8 is two tiles: keep them serial so 64 source
    // planes stay in L2. Fat x4 Concat now does the same; `PPOCR_DISABLE_
    // AVX512_CONCAT_ROW_TILE` restores the previous OC-parallel split.
    const bool split = !tile8 && !fat_sources && groups > 1 && total_channels >= 8 &&
        (stride_h == 1 ? (output_channels >= 8 && output_count >= 16384)
                       : true);
    if (split) ParallelFor(groups, body);
    else body(0, groups);
    return;
  }
#endif
  const int filter_plane = kernel_h * kernel_w;
  int channel0 = 0;
  for (int source = 0; source < source_count; ++source) {
    const int channels = source_channels[source];
    std::vector<float> packed(std::size_t(output_channels) * channels * filter_plane);
    for (int output = 0; output < output_channels; ++output) {
      std::memcpy(packed.data() + std::size_t(output) * channels * filter_plane,
                  weights + (std::size_t(output) * total_channels + channel0) * filter_plane,
                  std::size_t(channels) * filter_plane * sizeof(float));
    }
    if (source == 0) {
      Conv2d(dst, sources[source], packed.data(), bias, channels, output_channels,
             input_h, input_w, output_h, output_w, kernel_h, kernel_w,
             stride_h, stride_w, pad_top, pad_left, false);
    } else {
      std::vector<float> partial(output_count);
      Conv2d(partial.data(), sources[source], packed.data(), nullptr, channels,
             output_channels, input_h, input_w, output_h, output_w, kernel_h,
             kernel_w, stride_h, stride_w, pad_top, pad_left, false);
      BinaryInplace(dst, partial.data(), output_count, BinaryOp::add);
    }
    channel0 += channels;
  }
  if (relu) Relu(dst, dst, output_count);
}

void DetStemFromNchw(float* stem, const float* rgb, const float* conv0_w,
                     const float* conv0_b, int conv0_oc, const float* conv1_w,
                     const float* conv1_b, int conv1_oc, const float* conv2_w,
                     const float* conv2_b, int conv2_oc, const float* stem_w,
                     const float* stem_b, int stem_oc, int input_h, int input_w,
                     bool conv_relu, bool stem_relu) {
  if (!stem || !rgb || !conv0_w || !conv0_b || !conv1_w || !conv1_b || !conv2_w ||
      !conv2_b || !stem_w || !stem_b || conv0_oc <= 0 || conv1_oc <= 0 ||
      conv2_oc <= 0 || stem_oc <= 0 || input_h <= 0 || input_w <= 0) return;
  const int c0_h = (input_h + 2 - 3) / 2 + 1;
  const int c0_w = (input_w + 2 - 3) / 2 + 1;
  const int stem_height = (c0_h + 2 - 3) / 2 + 1;
  const int stem_width = (c0_w + 2 - 3) / 2 + 1;
  if (c0_h <= 0 || c0_w <= 0 || stem_height <= 0 || stem_width <= 0) return;
  const std::size_t c0_n = std::size_t(conv0_oc) * c0_h * c0_w;
  const std::size_t c1_n = std::size_t(conv1_oc) * c0_h * c0_w;
  const std::size_t c2_n = std::size_t(conv2_oc) * c0_h * c0_w;
  thread_local std::vector<float> conv0;
  thread_local std::vector<float> conv1;
  thread_local std::vector<float> conv2;
  thread_local std::vector<float> pooled;
  conv0.resize(c0_n);
  conv1.resize(c1_n);
  conv2.resize(c2_n);
  pooled.resize(c0_n);
  Conv2d(conv0.data(), rgb, conv0_w, conv0_b, 3, conv0_oc, input_h, input_w, c0_h,
         c0_w, 3, 3, 2, 2, 1, 1, conv_relu);
  // Pool while Conv.0 is still in cache. The graph does MaxPool only after
  // Conv.1/2, which rereads a cold 16x80x352 map.
  MaxPool2x2Same(pooled.data(), conv0.data(), std::size_t(conv0_oc), c0_h, c0_w);
  Conv2d(conv1.data(), conv0.data(), conv1_w, conv1_b, conv0_oc, conv1_oc, c0_h,
         c0_w, c0_h, c0_w, 2, 2, 1, 1, 0, 0, conv_relu);
  Conv2d(conv2.data(), conv1.data(), conv2_w, conv2_b, conv1_oc, conv2_oc, c0_h,
         c0_w, c0_h, c0_w, 2, 2, 1, 1, 0, 0, conv_relu);
  const float* sources[2] = {pooled.data(), conv2.data()};
  const int source_channels[2] = {conv0_oc, conv2_oc};
  ConcatChannelConv2d(stem, sources, source_channels, 2, stem_w, stem_b, stem_oc,
                      c0_h, c0_w, stem_height, stem_width, 3, 3, 2, 2, 1, 1, stem_relu);
}

void Conv2dBatch(float* dst, const float* src, const float* weights, const float* bias,
                 int batches, int input_channels, int output_channels,
                 int input_h, int input_w, int output_h, int output_w,
                 int kernel_h, int kernel_w, int stride_h, int stride_w,
                 int pad_top, int pad_left, bool relu) {
  if (!dst || !src || !weights || batches <= 0 || input_channels <= 0 ||
      output_channels <= 0 || input_h <= 0 || input_w <= 0 || output_h <= 0 ||
      output_w <= 0 || kernel_h <= 0 || kernel_w <= 0) return;
  if (batches == 1) {
    Conv2d(dst, src, weights, bias, input_channels, output_channels, input_h, input_w,
           output_h, output_w, kernel_h, kernel_w, stride_h, stride_w, pad_top, pad_left, relu);
    return;
  }
  const std::size_t input_batch = std::size_t(input_channels) * input_h * input_w;
  const std::size_t output_batch = std::size_t(output_channels) * output_h * output_w;
  const auto total_work = std::int64_t(batches) * input_channels * output_channels *
                          output_h * output_w * kernel_h * kernel_w;

  // These are the most frequent non-pointwise OCR convolutions. Preserve the
  // existing AVX output tiles and simply make crop index an outer tile axis.
  // That keeps every output's FP32 FMA sequence unchanged while avoiding a
  // worker-pool submission for each crop.
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512() && stride_h == 1 && stride_w == 1 && kernel_h == 2 && kernel_w == 2 &&
      pad_top == 0 && pad_left == 0 && output_h == input_h - 1 && output_w == input_w - 1) {
    // Keep the single-image and NCHW-batch routes on the same AVX-512
    // output-tile policy.  Previously the batch scheduler split by a single
    // output channel unconditionally, silently losing the four-filter input
    // reuse that the single-image reconstruction path uses.  A task now owns
    // one [batch, channel-tile] pair, so no worker races on an output plane
    // and each tile retains its original channel/K accumulation order.
    const bool tile4 = UseAvx512Conv2x2Tile4(output_channels, input_channels,
                                             input_h, input_w);
    const int tile = tile4 ? 4 : 1;
    const int groups = (output_channels + tile - 1) / tile;
    const int tasks = batches * groups;
    const auto body = [&](int first, int last) {
      for (int task = first; task < last; ++task) {
        const int batch = task / groups;
        const int group = task - batch * groups;
        const int first_output = group * tile;
        const int last_output = std::min(output_channels, first_output + tile);
        float* output = dst + std::size_t(batch) * output_batch;
        const float* input = src + std::size_t(batch) * input_batch;
        if (tile4) {
          Avx512Conv2x2Validx4(output, input, weights, bias, first_output,
                                last_output, input_channels, input_h, input_w, relu);
        } else {
          Avx512Conv2x2Valid(output, input, weights, bias, first_output,
                              last_output, input_channels, input_h, input_w, relu);
        }
      }
    };
  if (total_work >= 1000000 && tasks > 1) ParallelFor(tasks, body); else body(0, tasks);
    return;
  }
  // Medium detector context convolutions are 5x5/7x7 stride-one maps.  The
  // four/eight-output AVX-512 kernels already preserve the scalar reduction
  // order; expose crop index as an outer scheduler dimension just as for 3x3.
  if (!relu && HasAvx512() && stride_h == 1 && stride_w == 1 &&
      (kernel_h == 5 || kernel_h == 7) && kernel_h == kernel_w && output_channels >= 4) {
    const bool tile8 = UseAvx512ConvWideTile8(output_channels, input_channels,
                                               input_h, input_w, kernel_h);
    const int tile = tile8 ? 8 : 4;
    const int groups = (output_channels + tile - 1) / tile;
    const int tasks = batches * groups;
    const auto body = [&](int first, int last) {
      for (int task = first; task < last; ++task) {
        const int batch = task / groups;
        const int group = task - batch * groups;
        float* output = dst + std::size_t(batch) * output_batch;
        const float* input = src + std::size_t(batch) * input_batch;
        if (tile8) {
          Avx512ConvOddStride1x8(output, input, weights, bias, group * tile,
                                  std::min(output_channels, group * tile + tile), input_channels,
                                  input_h, input_w, output_h, output_w, kernel_h, pad_top, pad_left);
        } else {
          Avx512ConvOddStride1x4(output, input, weights, bias, group * tile,
                                  std::min(output_channels, group * tile + tile), input_channels,
                                  input_h, input_w, output_h, output_w, kernel_h, pad_top, pad_left);
        }
      }
    };
    if (total_work >= 1000000 && tasks > 1) ParallelFor(tasks, body); else body(0, tasks);
    return;
  }
  // The medium detector also contains 1x3/3x1 and 1x5/5x1 context filters.
  // They have no fused activation in the exported graphs; keep their existing
  // exact kernel and coalesce all crop/channel tiles into one pool submission.
  if (!relu && HasAvx512() && stride_h == 1 && stride_w == 1 && output_channels >= 4 &&
      ((kernel_h == 1 && (kernel_w == 3 || kernel_w == 5 || kernel_w == 7)) ||
       (kernel_w == 1 && (kernel_h == 3 || kernel_h == 5 || kernel_h == 7)))) {
    const int groups = (output_channels + 3) / 4;
    const int tasks = batches * groups;
    const auto body = [&](int first, int last) {
      for (int task = first; task < last; ++task) {
        const int batch = task / groups;
        const int group = task - batch * groups;
        Avx512ConvAsymmetricStride1x4(dst + std::size_t(batch) * output_batch,
                                       src + std::size_t(batch) * input_batch, weights, bias,
                                       group * 4, std::min(output_channels, group * 4 + 4),
                                       input_channels, input_h, input_w, output_h, output_w,
                                       kernel_h, kernel_w, pad_top, pad_left);
      }
    };
    if (total_work >= 1000000 && tasks > 1) ParallelFor(tasks, body); else body(0, tasks);
    return;
  }
  if (HasAvx512() && stride_h == 1 && stride_w == 1 && kernel_h == 3 && kernel_w == 3 &&
      output_channels >= 4) {
    const std::size_t output_plane = std::size_t(output_h) * output_w;
    const bool tile8 = UseAvx512Conv3x3Tile8() && output_channels >= 8 &&
                       input_channels >= 32 && output_plane >= 16384;
    const int tile = tile8 ? 8 : 4;
    const int groups = (output_channels + tile - 1) / tile;
    const int tasks = batches * groups;
    const auto body = [&](int first, int last) {
      for (int task = first; task < last; ++task) {
        const int batch = task / groups;
        const int group = task - batch * groups;
        const int first_output = group * tile;
        const int last_output = std::min(output_channels, first_output + tile);
        float* output = dst + std::size_t(batch) * output_batch;
        const float* input = src + std::size_t(batch) * input_batch;
        if (tile8) {
          Avx512Conv3x3Stride1x8(output, input, weights, bias, first_output, last_output,
                                  input_channels, input_h, input_w, output_h, output_w,
                                  pad_top, pad_left, relu);
        } else {
          Avx512Conv3x3Stride1x4(output, input, weights, bias, first_output, last_output,
                                  input_channels, input_h, input_w, output_h, output_w,
                                  pad_top, pad_left, relu);
        }
      }
    };
    if (total_work >= 1000000 && tasks > 1) ParallelFor(tasks, body); else body(0, tasks);
    return;
  }
  if (HasAvx512() && stride_h == 2 && stride_w == 2 && kernel_h == 3 && kernel_w == 3) {
    // `Avx512Conv3x3Stride2` has a four-output-channel body which shares
    // each gathered/packed input vector across four immutable filters.  The
    // original NCHW batch scheduler invoked it once per output channel, so
    // recognition/detector batches silently took only its one-channel tail
    // and reread the same source vectors four times.  Make one task own a
    // [batch, four-filter] tile, matching the single-image dispatcher and
    // keeping the per-output FP32 accumulation order unchanged.
    constexpr int tile = 4;
    // The four-filter body substantially reduces source-vector traffic for
    // the shipped batch shapes.  The guard is an explicit deployment A/B
    // switch; scalar/AVX2/NEON remain unchanged on other architectures.
    const bool use_output_tile =
        std::getenv("PPOCR_DISABLE_AVX512_STRIDE2_BATCH_TILE4") == nullptr;
    const int tile_width = use_output_tile ? tile : 1;
    const int groups = (output_channels + tile_width - 1) / tile_width;
    const int tasks = batches * groups;
    const auto body = [&](int first, int last) {
      for (int task = first; task < last; ++task) {
        const int batch = task / groups;
        const int group = task - batch * groups;
        const int first_output = group * tile_width;
        Avx512Conv3x3Stride2(dst + std::size_t(batch) * output_batch,
                              src + std::size_t(batch) * input_batch, weights, bias,
                              first_output, std::min(output_channels, first_output + tile_width),
                              input_channels, input_h, input_w,
                              output_h, output_w, pad_top, pad_left, relu,
                              UseAvx512Stride2PackedLoads());
      }
    };
    if (total_work >= 1000000 && tasks > 1) ParallelFor(tasks, body); else body(0, tasks);
    return;
  }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2() && stride_h == 1 && stride_w == 1 && kernel_h == 2 && kernel_w == 2 &&
      pad_top == 0 && pad_left == 0 && output_h == input_h - 1 && output_w == input_w - 1) {
    const int tasks = batches * output_channels;
    const auto body = [&](int first, int last) {
      for (int task = first; task < last; ++task) {
        const int batch = task / output_channels;
        const int output = task - batch * output_channels;
        Avx2Conv2x2Valid(dst + std::size_t(batch) * output_batch,
                          src + std::size_t(batch) * input_batch, weights, bias,
                          output, output + 1, input_channels, input_h, input_w, relu);
      }
    };
    if (total_work >= 1000000 && tasks > 1) ParallelFor(tasks, body); else body(0, tasks);
    return;
  }
  if (HasAvx2() && stride_h == 1 && stride_w == 1 && kernel_h == 3 && kernel_w == 3 &&
      output_channels >= 4) {
    const int groups = (output_channels + 3) / 4;
    const int tasks = batches * groups;
    const auto body = [&](int first, int last) {
      for (int task = first; task < last; ++task) {
        const int batch = task / groups;
        const int group = task - batch * groups;
        Avx2Conv3x3Stride1x4(dst + std::size_t(batch) * output_batch,
                              src + std::size_t(batch) * input_batch, weights, bias,
                              group * 4, std::min(output_channels, group * 4 + 4),
                              input_channels, input_h, input_w, output_h, output_w,
                              pad_top, pad_left, relu);
      }
    };
    if (total_work >= 1000000 && tasks > 1) ParallelFor(tasks, body); else body(0, tasks);
    return;
  }
  if (HasAvx2() && stride_h == 2 && stride_w == 2 && kernel_h == 3 && kernel_w == 3) {
    const int tasks = batches * output_channels;
    const auto body = [&](int first, int last) {
      for (int task = first; task < last; ++task) {
        const int batch = task / output_channels;
        const int output = task - batch * output_channels;
        Avx2Conv3x3Stride2(dst + std::size_t(batch) * output_batch,
                            src + std::size_t(batch) * input_batch, weights, bias,
                            output, output + 1, input_channels, input_h, input_w,
                            output_h, output_w, pad_top, pad_left, relu);
      }
    };
    if (total_work >= 1000000 && tasks > 1) ParallelFor(tasks, body); else body(0, tasks);
    return;
  }
#endif
  for (int batch = 0; batch < batches; ++batch) {
    Conv2d(dst + std::size_t(batch) * output_batch,
           src + std::size_t(batch) * input_batch, weights, bias, input_channels,
           output_channels, input_h, input_w, output_h, output_w, kernel_h, kernel_w,
           stride_h, stride_w, pad_top, pad_left, relu);
  }
}

void ApplyConvTransposeAct(float* plane, std::size_t n, int act) noexcept {
  if (act == 1) Relu(plane, plane, n);
  else if (act == 2) {
#if defined(PPOCR_HAS_AVX512_KERNELS)
    static const bool simd512 =
        std::getenv("PPOCR_ENABLE_AVX512_TRANSPOSE_SIGMOID") != nullptr &&
        std::getenv("PPOCR_DISABLE_AVX512_TRANSPOSE_SIGMOID") == nullptr;
    if (simd512 && HasAvx512() && n >= 16) {
      Avx512Sigmoid(plane, plane, n);
      return;
    }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
    static const bool simd256 =
        std::getenv("PPOCR_ENABLE_AVX2_SIGMOID") != nullptr &&
        std::getenv("PPOCR_DISABLE_AVX2_SIGMOID") == nullptr;
    if (simd256 && HasAvx2() && n >= 8) {
      Avx2Sigmoid(plane, plane, n);
      return;
    }
#endif
    Sigmoid(plane, plane, n);
  }
}

void ConvTranspose2x2Chain(float* dst, const float* src, const float* w0, const float* b0,
                           const float* w1, const float* b1, int input_channels,
                           int mid_channels, int output_channels, int input_h,
                           int input_w) {
  if (!dst || !src || !w0 || !w1 || input_channels <= 0 || mid_channels <= 0 ||
      output_channels <= 0 || input_h <= 0 || input_w <= 0) return;
#if defined(PPOCR_HAS_AVX512_KERNELS)
  static const bool avx_chain =
      std::getenv("PPOCR_DISABLE_AVX512_CT_CHAIN") == nullptr;
  if (avx_chain && HasAvx512() && input_channels == 16 && mid_channels == 16 &&
      output_channels == 1) {
    Avx512ConvTranspose2x2Chain16x16x1(dst, src, w0, b0, w1, b1, input_h, input_w);
    return;
  }
#endif
  const int output_h = input_h * 4, output_w = input_w * 4;
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(output_h) * output_w;
  const auto body = [&](int first, int last) {
    std::vector<float> patch(std::size_t(mid_channels) * 4);
    for (int y = first; y < last; ++y) {
      for (int x = 0; x < input_w; ++x) {
        for (int mid = 0; mid < mid_channels; ++mid) {
          const float bias = b0 ? b0[mid] : 0.F;
          float t0 = bias, t1 = bias, t2 = bias, t3 = bias;
          for (int input = 0; input < input_channels; ++input) {
            const float value = src[std::size_t(input) * input_plane +
                                    std::size_t(y) * input_w + x];
            const float* w = w0 + (std::size_t(input) * mid_channels + mid) * 4;
            t0 += value * w[0];
            t1 += value * w[1];
            t2 += value * w[2];
            t3 += value * w[3];
          }
          float* cell = patch.data() + std::size_t(mid) * 4;
          cell[0] = t0 > 0.F ? t0 : 0.F;
          cell[1] = t1 > 0.F ? t1 : 0.F;
          cell[2] = t2 > 0.F ? t2 : 0.F;
          cell[3] = t3 > 0.F ? t3 : 0.F;
        }
        const int y0 = y * 4;
        const int x0 = x * 4;
        for (int output = 0; output < output_channels; ++output) {
          float* out = dst + std::size_t(output) * output_plane;
          const float bias = b1 ? b1[output] : 0.F;
          for (int ky = 0; ky < 2; ++ky) {
            for (int kx = 0; kx < 2; ++kx) {
              const int mid_tap = ky * 2 + kx;
              for (int ly = 0; ly < 2; ++ly) {
                for (int lx = 0; lx < 2; ++lx) {
                  const int out_tap = ly * 2 + lx;
                  float sum = bias;
                  for (int mid = 0; mid < mid_channels; ++mid) {
                    sum += patch[std::size_t(mid) * 4 + mid_tap] *
                           w1[(std::size_t(mid) * output_channels + output) * 4 + out_tap];
                  }
                  out[std::size_t(y0 + ky * 2 + ly) * output_w + (x0 + kx * 2 + lx)] =
                      1.F / (1.F + std::exp(-sum));
                }
              }
            }
          }
        }
      }
    }
  };
  const auto work = std::int64_t(input_channels) * mid_channels * input_h * input_w * 4 +
                    std::int64_t(mid_channels) * output_channels * input_h * input_w * 16;
  if (work >= 1000000 && input_h > 1) ParallelFor(input_h, body);
  else body(0, input_h);
}

void ConvTranspose2x2(float* dst, const float* src, const float* weights, const float* bias,
                      int input_channels, int output_channels, int input_h, int input_w,
                      int act) {
  const int output_h = input_h * 2, output_w = input_w * 2;
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(output_h) * output_w;
#if defined(PPOCR_HAS_AVX512_KERNELS)
  const bool tile4 = HasAvx512() &&
      UseAvx512ConvTransposeTile4(output_channels, input_h, input_w);
#else
  const bool tile4 = false;
#endif
  const int tile = tile4 ? 4 : 1;
  const int groups = (output_channels + tile - 1) / tile;
  const auto body = [&](int first, int last) {
    for (int group = first; group < last; ++group) {
      const int first_output = group * tile;
      const int last_output = std::min(output_channels, first_output + tile);
#if defined(PPOCR_HAS_AVX512_KERNELS)
      if (HasAvx512()) {
        if (tile4) {
          Avx512ConvTranspose2x2x4(dst, src, weights, bias, first_output, last_output,
                                    input_channels, output_channels, input_h, input_w);
        } else {
          Avx512ConvTranspose2x2(dst, src, weights, bias, first_output, last_output,
                                  input_channels, output_channels, input_h, input_w);
        }
      } else
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
      if (HasAvx2()) {
        Avx2ConvTranspose2x2(dst, src, weights, bias, first_output, last_output,
                              input_channels, output_channels, input_h, input_w);
      } else
#endif
      {
      for (int output = first_output; output < last_output; ++output) {
        float* out = dst + std::size_t(output) * output_plane;
        std::fill_n(out, output_plane, bias ? bias[output] : 0.F);
        for (int input = 0; input < input_channels; ++input) {
          const float* in = src + std::size_t(input) * input_plane;
          const float* w = weights + (std::size_t(input) * output_channels + output) * 4;
          for (int y = 0; y < input_h; ++y) {
            const float* row = in + std::size_t(y) * input_w;
            float* dst0 = out + std::size_t(2 * y) * output_w;
            float* dst1 = dst0 + output_w;
            for (int x = 0; x < input_w; ++x) {
              const float value = row[x];
              const int xx = 2 * x;
              dst0[xx] += value * w[0]; dst0[xx + 1] += value * w[1];
              dst1[xx] += value * w[2]; dst1[xx + 1] += value * w[3];
            }
          }
        }
      }
      }
      if (act) {
        for (int output = first_output; output < last_output; ++output)
          ApplyConvTransposeAct(dst + std::size_t(output) * output_plane, output_plane, act);
      }
    }
  };
  const auto work = std::int64_t(input_channels) * output_channels * input_h * input_w * 4;
  if (work >= 1000000 && groups > 1) ParallelFor(groups, body);
  else body(0, groups);
}

void ConvTranspose2x2Batch(float* dst, const float* src, const float* weights,
                           const float* bias, int batches, int input_channels,
                           int output_channels, int input_h, int input_w,
                           int act) {
  if (!dst || !src || !weights || batches <= 0 || input_channels <= 0 ||
      output_channels <= 0 || input_h <= 0 || input_w <= 0) return;
  if (batches == 1) {
    ConvTranspose2x2(dst, src, weights, bias, input_channels, output_channels, input_h,
                     input_w, act);
    return;
  }
  const std::size_t input_batch = std::size_t(input_channels) * input_h * input_w;
  const std::size_t output_batch = std::size_t(output_channels) * (input_h * 2) * (input_w * 2);
  const auto work = std::int64_t(batches) * input_channels * output_channels * input_h * input_w * 4;
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512()) {
    const int tile = UseAvx512ConvTransposeTile4(output_channels, input_h, input_w) ? 4 : 1;
    const int groups = (output_channels + tile - 1) / tile;
    const int tasks = batches * groups;
    const auto body = [&](int first, int last) {
      for (int task = first; task < last; ++task) {
        const int batch = task / groups;
        const int group = task - batch * groups;
        const int first_output = group * tile;
        const int last_output = std::min(output_channels, first_output + tile);
        float* output = dst + std::size_t(batch) * output_batch;
        const float* input = src + std::size_t(batch) * input_batch;
        if (tile == 4) {
          Avx512ConvTranspose2x2x4(output, input, weights, bias, first_output, last_output,
                                   input_channels, output_channels, input_h, input_w);
        } else {
          Avx512ConvTranspose2x2(output, input, weights, bias, first_output, last_output,
                                 input_channels, output_channels, input_h, input_w);
        }
        if (act) {
          const std::size_t output_plane =
              std::size_t(input_h * 2) * std::size_t(input_w * 2);
          for (int oc = first_output; oc < last_output; ++oc)
            ApplyConvTransposeAct(output + std::size_t(oc) * output_plane, output_plane, act);
        }
      }
    };
    if (work >= 1000000 && tasks > 1) ParallelFor(tasks, body); else body(0, tasks);
    return;
  }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2()) {
    const int tasks = batches * output_channels;
    const auto body = [&](int first, int last) {
      for (int task = first; task < last; ++task) {
        const int batch = task / output_channels;
        const int output = task - batch * output_channels;
        float* plane = dst + std::size_t(batch) * output_batch;
        Avx2ConvTranspose2x2(plane, src + std::size_t(batch) * input_batch, weights, bias,
                              output, output + 1, input_channels, output_channels,
                              input_h, input_w);
        if (act) {
          const std::size_t output_plane =
              std::size_t(input_h * 2) * std::size_t(input_w * 2);
          ApplyConvTransposeAct(plane + std::size_t(output) * output_plane, output_plane, act);
        }
      }
    };
    if (work >= 1000000 && tasks > 1) ParallelFor(tasks, body); else body(0, tasks);
    return;
  }
#endif
  for (int batch = 0; batch < batches; ++batch) {
    ConvTranspose2x2(dst + std::size_t(batch) * output_batch,
                     src + std::size_t(batch) * input_batch, weights, bias,
                     input_channels, output_channels, input_h, input_w, act);
  }
}

void MaxPool2x2Same(float* dst, const float* src, std::size_t planes,
                    int height, int width) noexcept {
  if (height <= 0 || width <= 0 || planes == 0) return;
  const std::size_t plane_size = std::size_t(height) * width;
  // Tiny det MaxPoolConcat is 16x80x352 (450k). Channel ParallelFor under the
  // 1e6 floor was an 8-run miss vs serial. Keep it opt-in.
  static const bool tiny_parallel =
      std::getenv("PPOCR_ENABLE_AVX512_MAXPOOL_TINY_PARALLEL") != nullptr;
  const bool parallel = planes >= 16 &&
      std::size_t(planes) * plane_size >= (tiny_parallel ? 262144u : 1000000u);
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512()) {
    const auto body = [&](int first, int last) {
      Avx512MaxPool2x2Same(dst, src, first, last, height, width);
    };
    if (parallel) ParallelFor(static_cast<int>(planes), body);
    else body(0, static_cast<int>(planes));
    return;
  }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2()) {
    const auto body = [&](int first, int last) {
      Avx2MaxPool2x2Same(dst, src, first, last, height, width);
    };
    if (parallel) ParallelFor(static_cast<int>(planes), body);
    else body(0, static_cast<int>(planes));
    return;
  }
#endif
  const auto body = [&](int first, int last) {
    for (int plane = first; plane < last; ++plane) {
      const float* in = src + std::size_t(plane) * plane_size;
      float* out = dst + std::size_t(plane) * plane_size;
      for (int y = 0; y + 1 < height; ++y) {
        const float* row0 = in + std::size_t(y) * width;
        const float* row1 = row0 + width;
        int x = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        if (HasNeon()) {
          for (; x + 4 <= width - 1; x += 4) {
            const float32x4_t top = vmaxq_f32(vld1q_f32(row0 + x), vld1q_f32(row0 + x + 1));
            const float32x4_t bottom = vmaxq_f32(vld1q_f32(row1 + x), vld1q_f32(row1 + x + 1));
            vst1q_f32(out + std::size_t(y) * width + x, vmaxq_f32(top, bottom));
          }
        }
#endif
        for (; x + 1 < width; ++x) out[std::size_t(y) * width + x] =
            std::max(std::max(row0[x], row0[x + 1]), std::max(row1[x], row1[x + 1]));
        out[std::size_t(y) * width + width - 1] = std::max(row0[width - 1], row1[width - 1]);
      }
      if (height == 1) {
        for (int x = 0; x + 1 < width; ++x) out[x] = std::max(in[x], in[x + 1]);
      } else {
        const float* row = in + std::size_t(height - 1) * width;
        float* out_row = out + std::size_t(height - 1) * width;
        for (int x = 0; x + 1 < width; ++x) out_row[x] = std::max(row[x], row[x + 1]);
      }
      out[plane_size - 1] = in[plane_size - 1];
    }
  };
  if (parallel) ParallelFor(static_cast<int>(planes), body);
  else body(0, static_cast<int>(planes));
}

void MaxPool2x2Valid(float* dst, const float* src, std::size_t planes,
                     int height, int width) noexcept {
  if (!dst || !src || planes == 0 || height < 2 || width < 2) return;
  const int output_height = height - 1;
  const int output_width = width - 1;
  const std::size_t input_plane = std::size_t(height) * width;
  const std::size_t output_plane = std::size_t(output_height) * output_width;
  const bool parallel = planes >= 16 &&
      std::size_t(planes) * output_plane >= 1000000;
  static const bool simd_enabled =
      std::getenv("PPOCR_DISABLE_MAXPOOL_VALID_SIMD") == nullptr;
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (simd_enabled && HasAvx512()) {
    const auto body = [&](int first, int last) {
      Avx512MaxPool2x2Valid(dst, src, first, last, height, width);
    };
    if (parallel) ParallelFor(static_cast<int>(planes), body);
    else body(0, static_cast<int>(planes));
    return;
  }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (simd_enabled && HasAvx2()) {
    const auto body = [&](int first, int last) {
      Avx2MaxPool2x2Valid(dst, src, first, last, height, width);
    };
    if (parallel) ParallelFor(static_cast<int>(planes), body);
    else body(0, static_cast<int>(planes));
    return;
  }
#endif
  const auto body = [&](int first, int last) {
    for (int plane = first; plane < last; ++plane) {
      const float* in = src + std::size_t(plane) * input_plane;
      float* out = dst + std::size_t(plane) * output_plane;
      for (int y = 0; y < output_height; ++y) {
        const float* row0 = in + std::size_t(y) * width;
        const float* row1 = row0 + width;
        float* row_out = out + std::size_t(y) * output_width;
        int x = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        if (HasNeon()) {
          for (; x + 4 <= output_width; x += 4) {
            const float32x4_t top = vmaxq_f32(vld1q_f32(row0 + x), vld1q_f32(row0 + x + 1));
            const float32x4_t bottom = vmaxq_f32(vld1q_f32(row1 + x), vld1q_f32(row1 + x + 1));
            vst1q_f32(row_out + x, vmaxq_f32(top, bottom));
          }
        }
#endif
        for (; x < output_width; ++x) {
          row_out[x] = std::max(std::max(row0[x], row0[x + 1]),
                                std::max(row1[x], row1[x + 1]));
        }
      }
    }
  };
  if (parallel) ParallelFor(static_cast<int>(planes), body);
  else body(0, static_cast<int>(planes));
}

void AveragePool3x2Valid(float* dst, const float* src, std::size_t planes,
                         int input_height, int input_width) noexcept {
  if (!dst || !src || planes == 0 || input_height < 3 || input_width < 2) return;
  const int output_height = (input_height - 3) / 3 + 1;
  const int output_width = (input_width - 2) / 2 + 1;
  const std::size_t input_plane = std::size_t(input_height) * input_width;
  const std::size_t output_plane = std::size_t(output_height) * output_width;
  static const bool simd_enabled =
      std::getenv("PPOCR_DISABLE_AVGPOOL_SIMD") == nullptr;
  const auto work = planes * output_plane;
#if defined(PPOCR_HAS_AVX2_KERNELS)
  // AVX2 gather matches the scalar window add order. The AVX-512 gather
  // encoding is host-sensitive; eight-wide AVX2 already covers the
  // recognizer bridge widths including the odd tail.
  if (simd_enabled && HasAvx2() && output_width >= 8) {
    const auto avx = [&](int first, int last) {
      Avx2AveragePool3x2Valid(dst, src, first, last, input_height, input_width);
    };
    if (work >= 65536 && planes >= 2) ParallelFor(static_cast<int>(planes), avx);
    else avx(0, static_cast<int>(planes));
    return;
  }
#endif
  const auto body = [&](int first, int last) {
    for (int plane = first; plane < last; ++plane) {
      const float* input = src + std::size_t(plane) * input_plane;
      float* output = dst + std::size_t(plane) * output_plane;
      for (int oy = 0; oy < output_height; ++oy) {
        const float* row0 = input + std::size_t(oy * 3) * input_width;
        const float* row1 = row0 + input_width;
        const float* row2 = row1 + input_width;
        float* row_out = output + std::size_t(oy) * output_width;
        for (int ox = 0; ox < output_width; ++ox) {
          const int x = ox * 2;
          // Keep the exported pool's raster/KH/KW accumulation order rather
          // than using a reassociated SIMD horizontal reduction.
          float sum = row0[x];
          sum += row0[x + 1];
          sum += row1[x];
          sum += row1[x + 1];
          sum += row2[x];
          sum += row2[x + 1];
          // Match the generic ONNX path's final `sum / float(count)` exactly;
          // replacing this with a reciprocal multiply changes a few FP32
          // rounding bits and can alter a downstream CTC decision.
          row_out[ox] = sum / 6.F;
        }
      }
    }
  };
  if (work >= 65536 && planes >= 2) ParallelFor(static_cast<int>(planes), body);
  else body(0, static_cast<int>(planes));
}

void DepthwiseConv(float* dst, const float* src, const float* weights,
                   const float* bias, int channels, int input_h, int input_w,
                   int output_h, int output_w, int kernel_h, int kernel_w,
                   int stride_h, int stride_w, int pad_top, int pad_left) noexcept {
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(output_h) * output_w;
  const std::size_t kernel_plane = std::size_t(kernel_h) * kernel_w;
  const auto work = std::int64_t(channels) * output_h * output_w * kernel_h * kernel_w;
  const bool parallel = work >= 500000 && channels >= 16;
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512() && stride_h == 1 && stride_w == 1 &&
      (kernel_h == 3 || kernel_h == 5 || kernel_h == 7 || kernel_h == 9) &&
      kernel_h == kernel_w) {
    const auto avx = [&](int first, int last) {
      Avx512DepthwiseConv(dst, src, weights, bias, first, last, input_h, input_w,
                          output_h, output_w, kernel_h, kernel_w, pad_top, pad_left);
    };
    if (parallel) ParallelFor(channels, avx); else avx(0, channels);
    return;
  }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2() && stride_h == 1 && stride_w == 1 &&
      (kernel_h == 3 || kernel_h == 5 || kernel_h == 7 || kernel_h == 9) &&
      kernel_h == kernel_w) {
    const auto avx = [&](int first, int last) {
      Avx2DepthwiseConv(dst, src, weights, bias, first, last, input_h, input_w,
                        output_h, output_w, kernel_h, kernel_w, pad_top, pad_left);
    };
    if (parallel) ParallelFor(channels, avx); else avx(0, channels);
    return;
  }
#endif
  const auto body = [&](int first, int last) {
  for (int channel = first; channel < last; ++channel) {
    const float* in = src + std::size_t(channel) * input_plane;
    const float* filter = weights + std::size_t(channel) * kernel_plane;
    float* out = dst + std::size_t(channel) * output_plane;
    const float base = bias ? bias[channel] : 0.F;
    const int first_y = std::max(0, (pad_top + stride_h - 1) / stride_h);
    const int first_x = std::max(0, (pad_left + stride_w - 1) / stride_w);
    const int max_start_y = input_h + pad_top - kernel_h;
    const int max_start_x = input_w + pad_left - kernel_w;
    const int last_y = max_start_y < 0 ? 0 : std::min(output_h, max_start_y / stride_h + 1);
    const int last_x = max_start_x < 0 ? 0 : std::min(output_w, max_start_x / stride_w + 1);
    for (int oy = 0; oy < output_h; ++oy) {
      const int iy0 = oy * stride_h - pad_top;
      for (int ox = 0; ox < output_w; ++ox) {
        const int ix0 = ox * stride_w - pad_left;
        float sum = base;
        if (oy >= first_y && oy < last_y && ox >= first_x && ox < last_x) {
          const float* pixel = in + std::size_t(iy0) * input_w + ix0;
          for (int ky = 0; ky < kernel_h; ++ky) {
            const float* row = pixel + std::size_t(ky) * input_w;
            const float* kernel = filter + std::size_t(ky) * kernel_w;
            for (int kx = 0; kx < kernel_w; ++kx) sum += row[kx] * kernel[kx];
          }
          out[std::size_t(oy) * output_w + ox] = sum;
          continue;
        }
        for (int ky = 0; ky < kernel_h; ++ky) {
          const int iy = iy0 + ky;
          if (iy < 0 || iy >= input_h) continue;
          const float* row = in + std::size_t(iy) * input_w;
          const float* kernel = filter + std::size_t(ky) * kernel_w;
          for (int kx = 0; kx < kernel_w; ++kx) {
            const int ix = ix0 + kx;
            if (ix >= 0 && ix < input_w) sum += row[ix] * kernel[kx];
          }
        }
        out[std::size_t(oy) * output_w + ox] = sum;
      }
    }
  }
  };
  // Each NCHW channel is independent. Detector depthwise 3x3/5x5 layers are
  // large enough to profit from distributing channels across the persistent
  // pool, while recognizer-sized tensors stay single-threaded.
  if (parallel) ParallelFor(channels, body);
  else body(0, channels);
}

void DepthwiseConvBatch(float* dst, const float* src, const float* weights,
                        const float* bias, int batches, int channels,
                        int input_h, int input_w, int output_h, int output_w,
                        int kernel_h, int kernel_w, int stride_h, int stride_w,
                        int pad_top, int pad_left) noexcept {
  if (!dst || !src || !weights || batches <= 0 || channels <= 0 || input_h <= 0 ||
      input_w <= 0 || output_h <= 0 || output_w <= 0 || kernel_h <= 0 || kernel_w <= 0) {
    return;
  }
  if (batches == 1) {
    DepthwiseConv(dst, src, weights, bias, channels, input_h, input_w, output_h, output_w,
                  kernel_h, kernel_w, stride_h, stride_w, pad_top, pad_left);
    return;
  }
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(output_h) * output_w;
  const std::size_t kernel_plane = std::size_t(kernel_h) * kernel_w;
  const std::size_t input_batch = std::size_t(channels) * input_plane;
  const std::size_t output_batch = std::size_t(channels) * output_plane;
  const int tasks = batches * channels;
  const auto work = std::int64_t(batches) * channels * output_h * output_w *
                    kernel_h * kernel_w;
  const auto body = [&](int first, int last) {
    for (int task = first; task < last; ++task) {
      const int batch = task / channels;
      const int channel = task - batch * channels;
      // Reuse the well-tested single-channel ISA path. `channels=1` keeps it
      // inside the worker that owns this task (no nested ParallelFor), while
      // rebasing every tensor preserves its exact channel-local reduction.
      DepthwiseConv(dst + std::size_t(batch) * output_batch +
                        std::size_t(channel) * output_plane,
                    src + std::size_t(batch) * input_batch +
                        std::size_t(channel) * input_plane,
                    weights + std::size_t(channel) * kernel_plane,
                    bias ? bias + channel : nullptr, 1, input_h, input_w,
                    output_h, output_w, kernel_h, kernel_w, stride_h, stride_w,
                    pad_top, pad_left);
    }
  };
  // Small crop batches stay local, but large detector/recognizer batches have
  // enough independent channels to amortize the persistent queue once.
  if (work >= 500000 && tasks > 1) ParallelFor(tasks, body);
  else body(0, tasks);
}

bool DepthwisePointwiseConvFused(float* dst, const float* src,
                                 const float* depthwise_weights,
                                 const float* depthwise_bias,
                                 const float* pointwise_weights,
                                 const float* pointwise_bias, int batches,
                                 int channels, int output_channels, int height,
                                 int width, int kernel_h, int kernel_w,
                                 int stride_h, int stride_w, int pad_top,
                                 int pad_left, int activation) noexcept {
  if (!dst || !src || !depthwise_weights || !pointwise_weights || batches <= 0 ||
      channels <= 0 || output_channels <= 0 || height <= 0 || width <= 0) {
    return false;
  }
  // Conv.78 is 64-ch 5x5 SAME on 40x176 then a 16-ch 1x1. The two-pass DW
  // store plus four 4-OC pointwise tiles rereads that 450 KB plane. A C*W
  // row kernel keeps the depthwise row in L2 and walks OC once.
  // `PPOCR_DISABLE_AVX512_DWPW5_FUSED` restores the two-pass kernels.
  if (kernel_h == 5 && kernel_w == 5 && stride_h == 1 && stride_w == 1 &&
      pad_top == 2 && pad_left == 2) {
    static const bool dw5_disabled =
        std::getenv("PPOCR_DISABLE_AVX512_DWPW5_FUSED") != nullptr;
    if (dw5_disabled) return false;
#if defined(PPOCR_HAS_AVX512_KERNELS)
    if (HasAvx512()) {
      const bool approx_gelu = activation == 3 && ApproximateGeluSelected(
          std::size_t(output_channels) * std::size_t(height) * std::size_t(width));
      const std::size_t in_batch = std::size_t(channels) * height * width;
      const std::size_t out_batch = std::size_t(output_channels) * height * width;
      for (int batch = 0; batch < batches; ++batch) {
        float* batch_dst = dst + std::size_t(batch) * out_batch;
        const float* batch_src = src + std::size_t(batch) * in_batch;
        const auto body = [&](int first, int last) {
          Avx512DepthwisePointwiseConv5x5S1(
              batch_dst, batch_src, depthwise_weights, depthwise_bias,
              pointwise_weights, pointwise_bias, channels, output_channels, height,
              width, activation, approx_gelu, first, last);
        };
        // Serial y lost the 64-ch DW ParallelFor on Conv.78. Independent
        // output rows keep the C*W scratch and fill 16 workers.
        if (height >= 16) ParallelFor(height, body);
        else body(0, height);
      }
      return true;
    }
#endif
    return false;
  }
  if (kernel_h != 3 || kernel_w != 3 || stride_h != 1 || stride_w != 1 ||
      pad_top != 1 || pad_left != 1) {
    return false;
  }
  static const bool disabled =
      std::getenv("PPOCR_DISABLE_AVX512_DWPW_FUSED") != nullptr;
  if (disabled) return false;
  // Tiny/small 3x3 residuals fit a C*W row scratch in L1. Larger maps keep
  // the two-pass DW ParallelFor + pointwise tiles that already scale.
  const auto spatial = std::int64_t(height) * width;
  if (std::int64_t(channels) * output_channels * spatial >= 12000000) return false;
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512()) {
    const bool approx_gelu = activation == 3 && ApproximateGeluSelected(
        std::size_t(output_channels) * std::size_t(height) * std::size_t(width));
    const std::size_t in_batch = std::size_t(channels) * height * width;
    const std::size_t out_batch = std::size_t(output_channels) * height * width;
    for (int batch = 0; batch < batches; ++batch) {
      Avx512DepthwisePointwiseConv3x3S1(
          dst + std::size_t(batch) * out_batch, src + std::size_t(batch) * in_batch,
          depthwise_weights, depthwise_bias, pointwise_weights, pointwise_bias,
          channels, output_channels, height, width, activation, approx_gelu);
    }
    return true;
  }
#endif
  (void)depthwise_bias;
  (void)pointwise_bias;
  (void)activation;
  return false;
}

bool PointwiseDepthwiseConvFused(float* dst, const float* src,
                                 const float* pointwise_weights,
                                 const float* pointwise_bias,
                                 const float* depthwise_weights,
                                 const float* depthwise_bias, int batches,
                                 int input_channels, int output_channels,
                                 int height, int width, bool relu) noexcept {
  if (!dst || !src || !pointwise_weights || !depthwise_weights || batches <= 0 ||
      input_channels <= 0 || output_channels <= 0 || height <= 0 || width <= 0) {
    return false;
  }
  static const bool enabled =
      std::getenv("PPOCR_ENABLE_POINTWISE_DEPTHWISE") != nullptr &&
      std::getenv("PPOCR_DISABLE_POINTWISE_DEPTHWISE") == nullptr;
  if (!enabled) return false;
  // Rec CNN maps are H=6/12. Keep those on the two-pass kernels so this
  // does not nest ParallelFor under the two-crop recognizer pool. Det
  // Conv.4_relu+Conv.5 is 32-ch 40x176.
  if (height < 32) return false;

  const std::size_t plane = std::size_t(height) * width;
  constexpr int kOcTile = 8;
  const int groups = (output_channels + kOcTile - 1) / kOcTile;
  const std::size_t in_batch = std::size_t(input_channels) * plane;
  const std::size_t out_batch = std::size_t(output_channels) * plane;
  for (int batch = 0; batch < batches; ++batch) {
    const float* batch_src = src + std::size_t(batch) * in_batch;
    float* batch_dst = dst + std::size_t(batch) * out_batch;
    const auto body = [&](int first_group, int last_group) {
      thread_local std::vector<float> scratch;
      for (int group = first_group; group < last_group; ++group) {
        const int oc0 = group * kOcTile;
        const int oc1 = std::min(output_channels, oc0 + kOcTile);
        const int n_oc = oc1 - oc0;
        const std::size_t scratch_n = std::size_t(n_oc) * plane;
        if (scratch.size() < scratch_n) scratch.resize(scratch_n);
        const float* pw_w = pointwise_weights + std::size_t(oc0) * input_channels;
        const float* pw_b = pointwise_bias ? pointwise_bias + oc0 : nullptr;
#if defined(PPOCR_HAS_AVX512_KERNELS)
        if (HasAvx512() && relu && n_oc >= 8) {
          Avx512PointwiseConvRelu8(scratch.data(), batch_src, pw_w, pw_b, 0, n_oc,
                                   input_channels, plane);
        } else
#endif
            if (relu) {
          PointwiseConvRelu(scratch.data(), batch_src, pw_w, pw_b, n_oc,
                            input_channels, plane);
        } else {
          PointwiseConv(scratch.data(), batch_src, pw_w, pw_b, n_oc,
                        input_channels, plane);
        }
        DepthwiseConv(batch_dst + std::size_t(oc0) * plane, scratch.data(),
                      depthwise_weights + std::size_t(oc0) * 9,
                      depthwise_bias ? depthwise_bias + oc0 : nullptr, n_oc, height,
                      width, height, width, 3, 3, 1, 1, 1, 1);
      }
    };
    if (groups > 1 && output_channels >= 32) ParallelFor(groups, body);
    else body(0, groups);
  }
  return true;
}

void DepthwiseExpandGeluProjectAdd(float* dst, const float* src,
                                   const float* depthwise_weights,
                                   const float* depthwise_bias,
                                   const float* expand_weights,
                                   const float* expand_bias,
                                   const float* project_weights,
                                   const float* project_bias, int batches,
                                   int channels, int hidden, int height,
                                   int width, int kernel_h, int kernel_w,
                                   int stride_h, int stride_w, int pad_top,
                                   int pad_left) noexcept {
  if (!dst || !src || !depthwise_weights || !expand_weights || !project_weights ||
      batches <= 0 || channels <= 0 || hidden <= 0 || height <= 0 || width <= 0) {
    return;
  }
  const std::size_t sample = std::size_t(channels) * height * width;
#if defined(PPOCR_HAS_AVX512_KERNELS)
  // The C*W row kernel skips the depthwise NCHW store. Serial y was an e2e
  // miss (lost DW channel ParallelFor); row ParallelFor still lost the 8-run
  // A/B to two-pass (DW channel PF + expand spatial PF). Rec-sized H=6/12
  // serial fused also missed (21.3 vs two-pass 18.3). Keep tiled opt-in.
  static const bool tiled =
      std::getenv("PPOCR_ENABLE_AVX512_DW_EXPAND_GELU") != nullptr &&
      std::getenv("PPOCR_DISABLE_AVX512_DW_EXPAND_GELU") == nullptr;
  if (tiled && HasAvx512() && kernel_h == 3 && kernel_w == 3 &&
      stride_h == 1 && stride_w == 1 && pad_top == 1 && pad_left == 1) {
    for (int batch = 0; batch < batches; ++batch) {
      float* batch_dst = dst + std::size_t(batch) * sample;
      const float* batch_src = src + std::size_t(batch) * sample;
      const auto body = [&](int first, int last) {
        Avx512DepthwiseExpandGeluProjectAdd3x3S1(
            batch_dst, batch_src, depthwise_weights, depthwise_bias,
            expand_weights, expand_bias, project_weights, project_bias,
            channels, hidden, height, width, first, last);
      };
      if (height >= 32) ParallelFor(height, body);
      else body(0, height);
    }
    return;
  }
#endif
  thread_local std::vector<float> intermediate;
  if (intermediate.size() < sample) intermediate.resize(sample);
  for (int batch = 0; batch < batches; ++batch) {
    DepthwiseConv(intermediate.data(), src + std::size_t(batch) * sample,
                  depthwise_weights, depthwise_bias, channels, height, width,
                  height, width, kernel_h, kernel_w, stride_h, stride_w, pad_top,
                  pad_left);
    ExpandGeluProjectAdd(dst + std::size_t(batch) * sample, intermediate.data(),
                         expand_weights, expand_bias, project_weights,
                         project_bias, channels, hidden,
                         std::size_t(height) * width);
  }
}

void GemmAccumulate(float* dst, const float* a, const float* b,
                    int rows, int cols, int depth) {
  const auto work = std::int64_t(rows) * cols * depth;
  const bool parallel = work >= 10000000;
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512()) {
    if (!parallel) Avx512GemmAccumulateRows(dst, a, b, 0, rows, cols, depth);
    else ParallelFor(rows, [&](int first, int last) {
      Avx512GemmAccumulateRows(dst, a, b, first, last, cols, depth);
    });
    return;
  }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2()) {
    if (!parallel) Avx2GemmAccumulateRows(dst, a, b, 0, rows, cols, depth);
    else ParallelFor(rows, [&](int first, int last) {
      Avx2GemmAccumulateRows(dst, a, b, first, last, cols, depth);
    });
    return;
  }
#endif
  if (!parallel) ScalarOrNeonGemmAccumulateRows(dst, a, b, 0, rows, cols, depth);
  else ParallelFor(rows, [&](int first, int last) {
    ScalarOrNeonGemmAccumulateRows(dst, a, b, first, last, cols, depth);
  });
}

void Gemm(float* dst, const float* a, const float* b, const float* bias,
          int rows, int cols, int depth) {
  const auto work = std::int64_t(rows) * cols * depth;
  // The persistent executor removes launch churn, but tiny attention/GEMM
  // rows still suffer when each worker receives only one or two rows. Require
  // both enough arithmetic and at least two rows per default worker before
  // splitting; this specifically avoids recognizer attention oversubscription.
  const int available_workers = RequestedParallelism();
  const bool parallel = work >= 1000000 && rows >= available_workers * 2;
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512()) {
    if (!parallel) Avx512GemmRows(dst, a, b, bias, 0, rows, cols, depth);
    else ParallelFor(rows, [&](int first, int last) {
      Avx512GemmRows(dst, a, b, bias, first, last, cols, depth);
    });
    return;
  }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2()) {
    if (!parallel) Avx2GemmRows(dst, a, b, bias, 0, rows, cols, depth);
    else ParallelFor(rows, [&](int first, int last) {
      Avx2GemmRows(dst, a, b, bias, first, last, cols, depth);
    });
    return;
  }
#endif
  if (!parallel) ScalarOrNeonGemmRows(dst, a, b, bias, 0, rows, cols, depth);
  else ParallelFor(rows, [&](int first, int last) {
    ScalarOrNeonGemmRows(dst, a, b, bias, first, last, cols, depth);
  });
}

void SoftmaxRowsInplace(float* values, std::size_t rows, std::size_t width) noexcept {
  if (!values || rows == 0 || width == 0) return;
  const auto body = [&](int first, int last) {
    for (int row = first; row < last; ++row) {
      float* current = values + std::size_t(row) * width;
      float maximum = -std::numeric_limits<float>::infinity();
      for (std::size_t column = 0; column < width; ++column) {
        maximum = std::max(maximum, current[column]);
      }
      float sum{};
      std::size_t column{};
      // Preserve the existing four-independent-libm-call accumulation order
      // used by CTC confidence, while overwriting a row only after its max is
      // complete. This is safe for aliasing and cuts terminal softmax memory.
      for (; column + 4 <= width; column += 4) {
        const float e0 = std::exp(current[column] - maximum);
        const float e1 = std::exp(current[column + 1] - maximum);
        const float e2 = std::exp(current[column + 2] - maximum);
        const float e3 = std::exp(current[column + 3] - maximum);
        current[column] = e0; current[column + 1] = e1;
        current[column + 2] = e2; current[column + 3] = e3;
        sum += e0; sum += e1; sum += e2; sum += e3;
      }
      for (; column < width; ++column) {
        const float exponent = std::exp(current[column] - maximum);
        current[column] = exponent;
        sum += exponent;
      }
      for (column = 0; column < width; ++column) current[column] /= sum;
    }
  };
  const auto work = rows * width;
  if (work >= 65536 && rows >= 2) ParallelFor(static_cast<int>(rows), body);
  else body(0, static_cast<int>(rows));
}

void SpatialMean(float* dst, const float* src, std::size_t planes,
                 std::size_t spatial) noexcept {
  if (spatial == 0) return;
#if defined(PPOCR_HAS_AVX512_KERNELS)
  if (HasAvx512()) { Avx512SpatialMean(dst, src, planes, spatial); return; }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
  if (HasAvx2()) { Avx2SpatialMean(dst, src, planes, spatial); return; }
#endif
  const float scale = 1.F / static_cast<float>(spatial);
  for (std::size_t plane = 0; plane < planes; ++plane) {
    const float* values = src + plane * spatial;
    float sum = 0.F;
    std::size_t index = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    if (HasNeon()) {
      float32x4_t accum = vdupq_n_f32(0.F);
      for (; index + 4 <= spatial; index += 4) {
        accum = vaddq_f32(accum, vld1q_f32(values + index));
      }
      float lanes[4];
      vst1q_f32(lanes, accum);
      sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    }
#endif
    for (; index < spatial; ++index) sum += values[index];
    dst[plane] = sum * scale;
  }
}

void CtcTop1(int* indices, float* probabilities, const float* logits,
             std::size_t rows, int steps, int vocab) noexcept {
  // CTC confidence averages only emitted classes. First obtain every argmax,
  // then calculate the exact Softmax probability only for a non-blank,
  // non-repeated character. This keeps public confidence semantics while
  // skipping expensive exp() work for most blank/repeated time steps.
  if (!indices || !probabilities || !logits || steps <= 0 || vocab <= 0) return;
  const std::size_t sequences = rows / std::size_t(steps);
  const auto select_top1 = [&](std::size_t row) {
    const float* values = logits + row * vocab;
    int best = 0;
#if defined(PPOCR_HAS_AVX512_KERNELS)
    if (HasAvx512()) best = Avx512ArgMax(values, vocab);
    else
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
    if (HasAvx2()) best = Avx2ArgMax(values, vocab);
    else
#endif
    {
      float maximum_scalar = values[0];
      for (int i = 1; i < vocab; ++i) {
        if (values[i] > maximum_scalar) { maximum_scalar = values[i]; best = i; }
      }
    }
    indices[row] = best;
  };
  const auto body = [&](int first, int last) {
    for (int sequence = first; sequence < last; ++sequence) {
      int previous = -1;
      const std::size_t first_row = std::size_t(sequence) * steps;
      for (int step = 0; step < steps; ++step) {
        const std::size_t row = first_row + step;
        const float* values = logits + row * vocab;
        select_top1(row);
        const int best = indices[row];
        const float maximum = values[best];
        if (best != 0 && best != previous) {
          float sum = 0.F;
          // CTC's vocabulary is 6,906 classes. Keep the precise scalar-libm
          // exp operation but issue four independent calls before consuming
          // their results. This exposes latency-level parallelism on x86 and
          // ARM alike without an approximate vector exponential or a change
          // to the per-lane FP32 accumulation order.
          int i = 0;
          for (; i + 8 <= vocab; i += 8) {
            const float e0 = std::exp(values[i] - maximum);
            const float e1 = std::exp(values[i + 1] - maximum);
            const float e2 = std::exp(values[i + 2] - maximum);
            const float e3 = std::exp(values[i + 3] - maximum);
            const float e4 = std::exp(values[i + 4] - maximum);
            const float e5 = std::exp(values[i + 5] - maximum);
            const float e6 = std::exp(values[i + 6] - maximum);
            const float e7 = std::exp(values[i + 7] - maximum);
            sum += e0; sum += e1; sum += e2; sum += e3;
            sum += e4; sum += e5; sum += e6; sum += e7;
          }
          for (; i + 4 <= vocab; i += 4) {
            const float e0 = std::exp(values[i] - maximum);
            const float e1 = std::exp(values[i + 1] - maximum);
            const float e2 = std::exp(values[i + 2] - maximum);
            const float e3 = std::exp(values[i + 3] - maximum);
            sum += e0; sum += e1; sum += e2; sum += e3;
          }
          for (; i < vocab; ++i) sum += std::exp(values[i] - maximum);
          probabilities[row] = 1.F / sum;
        } else {
          probabilities[row] = 0.F;
        }
        previous = best;
      }
    }
  };
  // Each CTC sequence has private recurrence state. Dense-page recognition
  // commonly supplies several batched crops, so parallelize at that safe
  // boundary rather than splitting a sequence's blank/repeat history.
  if (sequences >= 2 && rows * std::size_t(vocab) >= 65536) {
    ParallelFor(static_cast<int>(sequences), body);
  } else {
    body(0, static_cast<int>(sequences));
  }
}

void SqueezeExcitationGateInplace(float* values, const float* first_weights,
                                  const float* first_bias,
                                  const float* second_weights,
                                  const float* second_bias, int batches,
                                  int channels, int reduced_channels,
                                  std::size_t spatial, float alpha,
                                  float beta, bool residual) noexcept {
  if (!values || !first_weights || !first_bias || !second_weights || !second_bias ||
      batches <= 0 || channels <= 0 || reduced_channels <= 0 || spatial == 0) return;
  // Gate storage is O(N*C + reduced-C), not O(N*C*H*W).  `gates` first
  // receives the spatial means, then each batch's mean row is overwritten by
  // its completed gate row.  All means are calculated before any source map
  // is scaled, so later batches still observe the unmodified input values.
  // This removes the previous second [N,C] means allocation and changes the
  // hidden workspace from [N,reduced-C] to one reusable [reduced-C] row.
  const std::size_t batch_channels = static_cast<std::size_t>(channels);
  std::vector<float> gates(static_cast<std::size_t>(batches) * batch_channels);
  std::vector<float> hidden(static_cast<std::size_t>(reduced_channels));
  SpatialMean(gates.data(), values, gates.size(), spatial);
  for (int batch = 0; batch < batches; ++batch) {
    const float* input = gates.data() + std::size_t(batch) * channels;
    float* middle = hidden.data();
    float* gate = gates.data() + std::size_t(batch) * channels;
    // These compact MLP rows deliberately retain the ordinary FP32 summation
    // order of the two graph Conv operators.  They are tiny compared with the
    // full map and vectorized/parallelized map scaling below is the hot work.
    for (int output = 0; output < reduced_channels; ++output) {
      float sum = first_bias[output];
      const float* weights = first_weights + std::size_t(output) * channels;
      for (int input_channel = 0; input_channel < channels; ++input_channel) {
        sum += weights[input_channel] * input[input_channel];
      }
      middle[output] = std::max(0.F, sum);
    }
    for (int output = 0; output < channels; ++output) {
      float sum = second_bias[output];
      const float* weights = second_weights + std::size_t(output) * reduced_channels;
      for (int input_channel = 0; input_channel < reduced_channels; ++input_channel) {
        sum += weights[input_channel] * middle[input_channel];
      }
      gate[output] = std::clamp(alpha * sum + beta, 0.F, 1.F);
    }
  }
  const int planes = batches * channels;
  const auto scale_planes = [&](int first, int last) {
    for (int plane = first; plane < last; ++plane) {
      const float scale = residual ? (1.F + gates[plane]) : gates[plane];
      BinaryScalar(values + std::size_t(plane) * spatial,
                   values + std::size_t(plane) * spatial, spatial,
                   scale, BinaryOp::mul, false);
    }
  };
  // The surrounding detector convolutions already use the persistent pool.
  // A second tiny plane task per SE gate caused cache/pool contention on
  // normal pages, so retain one contiguous ISA-dispatched sweep here. This
  // still removes the entire generic full-map destination allocation.
  scale_planes(0, planes);
}

static std::shared_ptr<std::vector<float>> PackedCtcB32(const float* b, int depth, int vocab) {
  struct Cache {
    const float* b = nullptr;
    int depth = 0;
    int vocab = 0;
    std::shared_ptr<std::vector<float>> packed;
  };
  static std::mutex mu;
  static Cache cache;
  std::lock_guard lock(mu);
  if (cache.packed && cache.b == b && cache.depth == depth && cache.vocab == vocab)
    return cache.packed;
  const int panels = (vocab + 31) / 32;
  auto packed = std::make_shared<std::vector<float>>(
      std::size_t(panels) * std::size_t(depth) * 32, 0.F);
  for (int panel = 0; panel < panels; ++panel) {
    const int n0 = panel * 32;
    const int n_len = std::min(32, vocab - n0);
    float* dst = packed->data() + std::size_t(panel) * depth * 32;
    for (int k = 0; k < depth; ++k) {
      std::memcpy(dst + std::size_t(k) * 32, b + std::size_t(k) * vocab + n0,
                  std::size_t(n_len) * sizeof(float));
    }
  }
  cache = Cache{b, depth, vocab, packed};
  return packed;
}

void GemmCtcTop1(int* indices, float* probabilities, const float* left,
                 const float* right, const float* bias, int rows, int depth,
                 int vocab, int steps) noexcept {
  if (!indices || !probabilities || !left || !right || rows <= 0 || depth <= 0 ||
      vocab <= 0 || steps <= 0 || rows % steps != 0) return;
  const int sequences = rows / steps;
  // The medium recognizer has a broad projection and enough independent rows
  // to amortize the persistent SIMD kernel. Keep its bounded 32-row workspace
  // by default; tiny/small retain the compact four-row tile. Deployments can
  // override the choice below for a local cache/throughput A/B.
  int rows_per_tile = depth >= 384 && vocab >= 1024 ? 32
      : (vocab >= 2048 && depth >= 64 ? 8 : 4);
  if (const char* configured = std::getenv("PPOCR_FUSED_TERMINAL_CTC_TILE_ROWS")) {
    char* end = nullptr;
    const long parsed = std::strtol(configured, &end, 10);
    if (end != configured && *end == '\0' &&
        (parsed == 1 || parsed == 2 || parsed == 4 || parsed == 8 || parsed == 16 || parsed == 32)) {
      rows_per_tile = static_cast<int>(parsed);
    }
  }
  // Tiny CTC streams the 80×6906 vocabulary matrix once per 8-row tile
  // (~5 DRAM passes for 38–53 steps). Blocking N into 256-col panels keeps
  // each 80KB B-slice in L2 for every time step. Gemm stays serial: each
  // panel is 8×256×80, well under the rec ParallelFor floor.
  // `PPOCR_DISABLE_CTC_VOCAB_BLOCK` restores the full-width 8-row tiles.
  static const bool vocab_block =
      std::getenv("PPOCR_ENABLE_CTC_VOCAB_BLOCK") != nullptr &&
      std::getenv("PPOCR_DISABLE_CTC_VOCAB_BLOCK") == nullptr;
  const bool use_vocab_block = vocab_block && vocab >= 2048 && depth >= 64 &&
      steps >= 16 && rows_per_tile >= 8;
  // Tiny CTC B is 80x6906 with a 27 KB K-stride. Packing 32-col panels into
  // K-contiguous 10 KB blocks streams B once from DRAM (vs ~5 full-width
  // 8-row tiles). Rec crops share the packed buffer; GEMM stays serial.
  // `PPOCR_DISABLE_CTC_PACKED_B` restores the strided 8-row tiles.
  static const bool packed_b =
      std::getenv("PPOCR_DISABLE_CTC_PACKED_B") == nullptr;
  const bool use_packed_b = packed_b && vocab >= 2048 && depth >= 64 && steps >= 16;
  constexpr int vocab_tile = 256;
  const auto emit_row = [&](int row, int previous, const float* row_logits) {
    int best = 0;
    bool vector_argmax{};
#if defined(PPOCR_HAS_AVX512_KERNELS)
    if (HasAvx512()) {
      best = Avx512ArgMax(row_logits, vocab);
      vector_argmax = true;
    }
#endif
#if defined(PPOCR_HAS_AVX2_KERNELS)
    if (!vector_argmax && HasAvx2()) {
      best = Avx2ArgMax(row_logits, vocab);
      vector_argmax = true;
    }
#endif
    if (!vector_argmax) {
      for (int column = 1; column < vocab; ++column) {
        if (row_logits[std::size_t(column)] > row_logits[std::size_t(best)]) best = column;
      }
    }
    const float maximum = row_logits[std::size_t(best)];
    indices[row] = best;
    if (best != 0 && best != previous) {
      float sum = 0.F;
#if defined(PPOCR_HAS_AVX512_KERNELS)
      static const bool simd_sm =
          std::getenv("PPOCR_DISABLE_AVX512_CTC_SOFTMAX") == nullptr;
      if (simd_sm && HasAvx512() && vocab >= 16) {
        sum = Avx512SoftmaxDenom(row_logits, vocab, maximum);
      } else
#endif
      {
        int column = 0;
        for (; column + 4 <= vocab; column += 4) {
          const float e0 = std::exp(row_logits[std::size_t(column)] - maximum);
          const float e1 = std::exp(row_logits[std::size_t(column + 1)] - maximum);
          const float e2 = std::exp(row_logits[std::size_t(column + 2)] - maximum);
          const float e3 = std::exp(row_logits[std::size_t(column + 3)] - maximum);
          sum += e0; sum += e1; sum += e2; sum += e3;
        }
        for (; column < vocab; ++column)
          sum += std::exp(row_logits[std::size_t(column)] - maximum);
      }
      probabilities[row] = 1.F / sum;
    } else {
      probabilities[row] = 0.F;
    }
    return best;
  };
  const auto body = [&](int first, int last) {
    if (use_packed_b) {
#if defined(PPOCR_HAS_AVX512_KERNELS)
      if (HasAvx512()) {
        const auto packed = PackedCtcB32(right, depth, vocab);
        // Full-M writes every time step for one 32-col panel so B streams once.
        // 8-row tiles keep the 221 KB logit workspace in L2. Default is full-M;
        // `PPOCR_DISABLE_CTC_PACKED_FULL_M` restores the tiled workspace.
        static const bool packed_full_m =
            std::getenv("PPOCR_DISABLE_CTC_PACKED_FULL_M") == nullptr;
        if (packed_full_m) {
          static const bool online =
              std::getenv("PPOCR_DISABLE_CTC_ONLINE") == nullptr;
          if (online) {
            std::vector<int> args(static_cast<std::size_t>(steps));
            std::vector<float> maxima(static_cast<std::size_t>(steps));
            std::vector<float> sums(static_cast<std::size_t>(steps));
            for (int sequence = first; sequence < last; ++sequence) {
              Avx512GemmPacked32(nullptr,
                                 left + std::size_t(sequence) * steps * depth,
                                 packed->data(), bias, steps, vocab, depth,
                                 args.data(), maxima.data(), sums.data());
              int previous = -1;
              for (int step = 0; step < steps; ++step) {
                const int row = sequence * steps + step;
                const int best = args[static_cast<std::size_t>(step)];
                indices[row] = best;
                const float denom = sums[static_cast<std::size_t>(step)];
                probabilities[row] = (best != 0 && best != previous && denom > 0.F)
                    ? 1.F / denom : 0.F;
                previous = best;
              }
            }
          } else {
            std::vector<float> logits(std::size_t(steps) * vocab);
            for (int sequence = first; sequence < last; ++sequence) {
              Avx512GemmPacked32(logits.data(),
                                 left + std::size_t(sequence) * steps * depth,
                                 packed->data(), bias, steps, vocab, depth);
              int previous = -1;
              for (int step = 0; step < steps; ++step) {
                previous = emit_row(sequence * steps + step, previous,
                                    logits.data() + std::size_t(step) * vocab);
              }
            }
          }
        } else {
          std::vector<float> tile_logits(std::size_t(rows_per_tile) * vocab);
          for (int sequence = first; sequence < last; ++sequence) {
            int previous = -1;
            for (int step_base = 0; step_base < steps; step_base += rows_per_tile) {
              const int tile_rows = std::min(rows_per_tile, steps - step_base);
              const int first_row = sequence * steps + step_base;
              Avx512GemmPacked32(tile_logits.data(),
                                 left + std::size_t(first_row) * depth,
                                 packed->data(), bias, tile_rows, vocab, depth);
              for (int tile_row = 0; tile_row < tile_rows; ++tile_row) {
                previous = emit_row(first_row + tile_row, previous,
                                    tile_logits.data() + std::size_t(tile_row) * vocab);
              }
            }
          }
        }
        return;
      }
#endif
    }
    if (use_vocab_block) {
      std::vector<float> tmp(std::size_t(rows_per_tile) * vocab_tile);
      std::vector<float> logits(std::size_t(steps) * vocab);
      for (int sequence = first; sequence < last; ++sequence) {
        const float* seq_left = left + std::size_t(sequence) * steps * depth;
        for (int n0 = 0; n0 < vocab; n0 += vocab_tile) {
          const int n_len = std::min(vocab_tile, vocab - n0);
          for (int step_base = 0; step_base < steps; step_base += rows_per_tile) {
            const int tile_rows = std::min(rows_per_tile, steps - step_base);
            const float* a = seq_left + std::size_t(step_base) * depth;
            const float* panel_bias = bias ? bias + n0 : nullptr;
            bool used_avx512{};
#if defined(PPOCR_HAS_AVX512_KERNELS)
            if (HasAvx512()) {
              Avx512GemmPanelLdb(tmp.data(), a, right + n0, panel_bias, tile_rows, n_len,
                                 depth, vocab);
              used_avx512 = true;
            }
#endif
            if (!used_avx512) {
              for (int r = 0; r < tile_rows; ++r) {
                const float* row_left = a + std::size_t(r) * depth;
                float* row_out = tmp.data() + std::size_t(r) * n_len;
                for (int col = 0; col < n_len; ++col) {
                  float sum = panel_bias ? panel_bias[col] : 0.F;
                  for (int k = 0; k < depth; ++k)
                    sum += row_left[k] * (right + n0)[std::size_t(k) * vocab + col];
                  row_out[col] = sum;
                }
              }
            }
            for (int r = 0; r < tile_rows; ++r) {
              std::memcpy(logits.data() + std::size_t(step_base + r) * vocab + n0,
                          tmp.data() + std::size_t(r) * n_len,
                          std::size_t(n_len) * sizeof(float));
            }
          }
        }
        int previous = -1;
        for (int step = 0; step < steps; ++step) {
          previous = emit_row(sequence * steps + step, previous,
                              logits.data() + std::size_t(step) * vocab);
        }
      }
      return;
    }
    std::vector<float> tile_logits(std::size_t(rows_per_tile) * vocab);
    for (int sequence = first; sequence < last; ++sequence) {
      int previous = -1;
      for (int step_base = 0; step_base < steps; step_base += rows_per_tile) {
        const int tile_rows = std::min(rows_per_tile, steps - step_base);
        const int first_row = sequence * steps + step_base;
        Gemm(tile_logits.data(), left + std::size_t(first_row) * depth,
             right, bias, tile_rows, vocab, depth);
        for (int tile_row = 0; tile_row < tile_rows; ++tile_row) {
          previous = emit_row(first_row + tile_row, previous,
                              tile_logits.data() + std::size_t(tile_row) * vocab);
        }
      }
    }
  };
  const auto work = std::size_t(rows) * depth * vocab;
  if (sequences >= 2 && work >= 65536) ParallelFor(sequences, body);
  else body(0, sequences);
}
void CtcTop1Scalar(int* indices, float* probabilities, const float* logits,
                   std::size_t rows, int steps, int vocab) noexcept {
  if (!indices || !probabilities || !logits || steps <= 0 || vocab <= 0) return;
  const std::size_t sequences = rows / std::size_t(steps);
  for (std::size_t sequence = 0; sequence < sequences; ++sequence) {
    int previous = -1;
    const std::size_t first_row = sequence * std::size_t(steps);
    for (int step = 0; step < steps; ++step) {
      const std::size_t row = first_row + std::size_t(step);
      const float* values = logits + row * vocab;
      int best = 0;
      for (int i = 1; i < vocab; ++i) {
        if (values[i] > values[best]) best = i;
      }
      const float maximum = values[best];
      indices[row] = best;
      if (best != 0 && best != previous) {
        float sum{};
        for (int i = 0; i < vocab; ++i) sum += std::exp(values[i] - maximum);
        probabilities[row] = 1.F / sum;
      } else {
        probabilities[row] = 0.F;
      }
      previous = best;
    }
  }
}

}  // namespace ppocr::detail::kernels
