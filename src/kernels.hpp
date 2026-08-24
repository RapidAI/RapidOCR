#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

namespace ppocr::detail::kernels {

// Persistent-pool range split. Used by independent 1x1 Conv islands so FPN
// laterals can run together without each nesting another ParallelFor.
void ParallelForRange(int count, const std::function<void(int, int)>& fn);

enum class BinaryOp { add, sub, mul, div };

// Runtime-dispatched utility for callers that explicitly need a widened image
// buffer. The strict Vulkan GPU-only ingress uploads native RGB bytes directly
// and does not call this helper; resize/normalization execute in the shader.
void WidenU8ToFloat(float* dst, const std::uint8_t* src, std::size_t n) noexcept;
// Packed RGB (row-major) to NCHW BGR with a per-channel affine normalize.
// This is the identity-resize PP-OCR front end: no bilinear sampling, one
// write into the graph input. `scale`/`shift` are B,G,R. `row_width` may be
// wider than `width` so recognizer right-padding stays at the caller's value.
// `PPOCR_DISABLE_IDENTITY_RGB_SIMD=1` forces the scalar tail on every pixel.
void WriteIdentityRgbToNchw(float* dst, const std::uint8_t* rgb, int width,
                            int height, int source_width, int left, int top,
                            const float scale[3], const float shift[3],
                            int row_width) noexcept;
// Packed RGB crop -> NCHW BGR bilinear resize with the same uint8 rounding
// and per-channel affine as WriteResizeRegionToNchw. `scale`/`shift` are B,G,R.
// `row_width` may be wider than `width` so recognizer right-padding stays at
// the caller's pre-zeroed value. `parallel` fans independent output rows over
// the SIMD pool; rec crops already run in an outer ParallelFor and must pass
// false. `PPOCR_DISABLE_BILINEAR_RGB_SIMD` forces the scalar body;
// `PPOCR_DISABLE_BILINEAR_RGB_PF` keeps SIMD but serializes the row loop.
void WriteBilinearRgbToNchw(float* dst, const std::uint8_t* rgb, int image_width,
                            int image_height, int left, int top, int source_width,
                            int source_height, int width, int height,
                            const float scale[3], const float shift[3],
                            int row_width, bool parallel) noexcept;

// SIMD-assisted inner kernels used only by the PP-OCRv6 executor.  They are
// deliberately small and dependency-free; runtime dispatch keeps the binary
// portable to non-AVX2 CPUs.
void Binary(float* dst, const float* a, const float* b, std::size_t n,
            BinaryOp op) noexcept;
// Applies a single binary operation in place. This makes the aliasing
// contract explicit for executor fusions that consume their left tensor.
void BinaryInplace(float* left, const float* right, std::size_t n,
                   BinaryOp op) noexcept;
// Tensor-style broadcasting without per-element shape/stride recomputation.
// `a_repeat` / `b_repeat` specify the number of adjacent output values mapped
// to one input element (1 for non-broadcast dimensions).
void BinaryBroadcast(float* dst, const float* a, const float* b, std::size_t n,
                     std::size_t a_repeat, std::size_t b_repeat,
                     BinaryOp op) noexcept;
// In-place counterpart for a full contiguous left tensor with a smaller
// suffix-broadcast right tensor. `right_repeat` is the adjacent output span
// sharing each right value (for NCHW channel vectors, H * W).
void BinaryBroadcastRightInplace(float* left, const float* right, std::size_t n,
                                 std::size_t right_repeat, std::size_t right_elements,
                                 BinaryOp op) noexcept;
// Exact NCHW nearest-neighbour 2x upsample plus equal-shaped residual Add.
// This is the PP-OCR detector FPN `Resize(asymmetric,floor,2x) -> Add` form.
// It writes the final feature map directly, using runtime AVX-512/AVX2/NEON
// dispatch where available and retaining a scalar fallback on every target.
void NearestResize2xAdd(float* dst, const float* src, const float* residual,
                        int batches, int channels, int input_height,
                        int input_width) noexcept;
// Exact squaring for LayerNorm variance. This avoids general-purpose powf for
// the PP-OCRv6 `Pow(x, 2)` pattern and is safe for in-place activations.
void Square(float* dst, const float* src, std::size_t n) noexcept;
// Faithful last-axis LayerNorm with affine scale/bias. PP-OCR's transformer
// exports this as ten nodes; this one-pass implementation removes its
// temporary centered/squared/normalized activations.
void LayerNorm(float* dst, const float* src, const float* gamma, const float* beta,
               std::size_t rows, std::size_t width, float epsilon) noexcept;
// Applies GELU to contiguous values. `logical_elements` is the number of
// elements belonging to one independently inferred sample (zero means n).
// It lets the optional approximate SIMD policy remain invariant when an
// identical sample is evaluated alone or packed into an NCHW batch.
void Gelu(float* dst, const float* src, std::size_t n,
          std::size_t logical_elements = 0) noexcept;
// AVX2 Abramowitz/Stegun Erf-GELU used by the default exact path. Kept in
// the public kernel surface so smoke can drive the shipped function against
// an independent std::erf reference, including odd tails.
void ExactGelu(float* dst, const float* src, std::size_t n) noexcept;
// True only when the explicitly opt-in approximate GELU deployment mode is
// active. Graph fusions use this to route affine+GELU through the same SIMD
// implementation; exact ONNX mode keeps its one-pass Erf expression.
[[nodiscard]] bool ApproximateGeluEnabled() noexcept;
// Whether Gelu() will select the x86 SIMD approximation for one independent
// sample of this logical size. GPU graph fusions use this rather than merely
// inspecting the environment, so they never replace the portable exact-Erf
// path on ARM or on small decision-sensitive tensors.
[[nodiscard]] bool ApproximateGeluSelected(std::size_t logical_elements) noexcept;
// Caps inner SIMD ParallelFor so concurrent recognizer crops share the
// PPOCR_BENCH_THREADS budget instead of stacking two full pools. Zero restores
// the process-wide worker count. Nested ParallelFor workers ignore the cap.
void SetInnerParallelismCap(int cap) noexcept;
struct InnerParallelismGuard {
  explicit InnerParallelismGuard(int cap) noexcept { SetInnerParallelismCap(cap); }
  ~InnerParallelismGuard() noexcept { SetInnerParallelismCap(0); }
  InnerParallelismGuard(const InnerParallelismGuard&) = delete;
  InnerParallelismGuard& operator=(const InnerParallelismGuard&) = delete;
};
void HardSwish(float* dst, const float* src, std::size_t n) noexcept;
// Exact ONNX HardSigmoid: clamp(alpha * x + beta, 0, 1). Keeping scale and
// clamp in one ISA dispatch avoids a temporary/two-pass gate activation.
void HardSigmoid(float* dst, const float* src, std::size_t n, float alpha,
                 float beta) noexcept;
void ScaleShift(float* dst, const float* src, std::size_t n, float scale, float shift) noexcept;
// Elementwise arithmetic with a scalar right-hand operand.
void BinaryScalar(float* dst, const float* src, std::size_t n, float scalar,
                  BinaryOp op, bool scalar_left) noexcept;
void AddChannelBias(float* dst, const float* bias, int batches, int channels,
                    std::size_t spatial) noexcept;
// Exact logistic sigmoid. Uses runtime ISA dispatch on x86 but retains the
// scalar/NEON-compatible fallback for targets without an available vector exp.
void Sigmoid(float* dst, const float* src, std::size_t n) noexcept;
// Exact Swish x * sigmoid(x) in one pass.  This is used after Conv+BN
// folding, where a generic Sigmoid followed by Mul would otherwise traverse
// and allocate an activation twice.
void Swish(float* dst, const float* src, std::size_t n) noexcept;
void Relu(float* dst, const float* src, std::size_t n) noexcept;
// Per-NCHW-channel affine transform used by inference BatchNormalization.
// `dst` may alias `src`: every contiguous channel plane is transformed in
// place through the runtime-dispatched ScaleShift implementation.
void BatchNormAffine(float* dst, const float* src, const float* scale,
                     const float* shift, int batches, int channels,
                     std::size_t spatial) noexcept;
// Per-NCHW-channel affine BatchNorm followed by GELU.  It keeps the exact
// Erf-GELU default and may split independent channels over the persistent
// executor for large recognizer activations.
void BatchNormGelu(float* dst, const float* src, const float* scale,
                   const float* shift, int batches, int channels,
                   std::size_t spatial) noexcept;
// Per-channel affine BatchNorm followed by exact Swish x * sigmoid(x).
// Used for recognition gates to avoid materializing both the BN and sigmoid
// activations before their dependent multiply.
void BatchNormSwish(float* dst, const float* src, const float* scale,
                    const float* shift, int batches, int channels,
                    std::size_t spatial) noexcept;
// Fuses a BatchNorm+Swish result with a same-shaped binary operand. This
// avoids writing and reading the intermediate Swish activation, while each
// element preserves the original affine -> logistic -> multiply -> binary
// FP32 operation sequence.
void BatchNormSwishBinary(float* dst, const float* src, const float* scale,
                          const float* shift, const float* right, int batches,
                          int channels, std::size_t spatial, BinaryOp operation) noexcept;
// dst[i] += alpha * src[i].  This is the inner loop of a NCHW 1x1 Conv.
void Axpy(float* dst, const float* src, float alpha, std::size_t n) noexcept;
// Accumulates rows of a 1x1 NCHW convolution. The pointwise path maps output
// channels to independent contiguous rows, making it safe to split by rows.
void PointwiseConv(float* dst, const float* src, const float* weights,
                   const float* bias, int output_channels, int input_channels,
                   std::size_t plane);
// Recognizer inverted residual: 1x1 expand, exact GELU, 1x1 project, residual
// add of `src`. Hidden activations stay in a spatial tile instead of a full
// NCHW expand tensor. `PPOCR_DISABLE_EXPAND_GELU_PROJECT` is the graph A/B.
void ExpandGeluProjectAdd(float* dst, const float* src, const float* expand_weights,
                          const float* expand_bias, const float* project_weights,
                          const float* project_bias, int channels, int hidden,
                          std::size_t plane);
// Runs a real NCHW batch of independent 1x1 projections in one scheduling
// pass.  It preserves the per-output-channel FP32 reduction order of
// PointwiseConv while flattening [N, output-channel-tile] work, avoiding one
// persistent-worker barrier per crop in recognizer batches.
void PointwiseConvBatch(float* dst, const float* src, const float* weights,
                        const float* bias, int batches, int output_channels,
                        int input_channels, std::size_t plane);
// Computes a 1x1 NCHW convolution plus an equal-shaped residual in one final
// store. The convolution reduction is completed before adding `residual`,
// matching the unfused Conv -> Add arithmetic order.
void PointwiseConvAdd(float* dst, const float* src, const float* weights,
                      const float* bias, const float* residual,
                      int output_channels, int input_channels,
                      std::size_t plane);
// Batched counterpart of PointwiseConvAdd.  It flattens independent
// [batch, output-channel-tile] work into one persistent-pool submission while
// retaining the exact per-channel reduction and final residual-add order.
void PointwiseConvAddBatch(float* dst, const float* src, const float* weights,
                           const float* bias, const float* residual, int batches,
                           int output_channels, int input_channels,
                           std::size_t plane);
// Computes a 1x1 NCHW convolution followed by an equal-shaped residual add
// and ReLU in one final store. Used only for graph-proven Conv->Add->Relu.
void PointwiseConvAddRelu(float* dst, const float* src, const float* weights,
                          const float* bias, const float* residual,
                          int output_channels, int input_channels,
                          std::size_t plane);
void PointwiseConvAddReluBatch(float* dst, const float* src, const float* weights,
                               const float* bias, const float* residual, int batches,
                               int output_channels, int input_channels,
                               std::size_t plane);
// Computes a 1x1 NCHW convolution, equal-shaped residual add, and exact
// Swish in one output traversal. This is graph-proven Conv->Add->Sigmoid->Mul
// and preserves that FP32 arithmetic order without a temporary activation.
void PointwiseConvAddSwish(float* dst, const float* src, const float* weights,
                           const float* bias, const float* residual,
                           int output_channels, int input_channels,
                           std::size_t plane);
void PointwiseConvAddSwishBatch(float* dst, const float* src, const float* weights,
                                const float* bias, const float* residual, int batches,
                                int output_channels, int input_channels,
                                std::size_t plane);
// NCHW 1x1 Conv followed by exact HardSwish. Detector squeeze-excitation
// blocks use the canonical x * clamp(x / 6 + .5, 0, 1) form. Fusing it keeps
// the convolution result in its final activation allocation.
void PointwiseConvHardSwish(float* dst, const float* src, const float* weights,
                            const float* bias, int output_channels,
                            int input_channels, std::size_t plane);
// Batched 1x1 Conv + exact HardSwish. The projection shares the regular
// cross-crop SIMD scheduler, then the contiguous final allocation is activated
// once, avoiding a per-crop worker-pool entry.
void PointwiseConvHardSwishBatch(float* dst, const float* src, const float* weights,
                                 const float* bias, int batches, int output_channels,
                                 int input_channels, std::size_t plane);
void PointwiseConvRelu(float* dst, const float* src, const float* weights,
                       const float* bias, int output_channels, int input_channels,
                       std::size_t plane);
void PointwiseConvReluBatch(float* dst, const float* src, const float* weights,
                            const float* bias, int batches, int output_channels,
                            int input_channels, std::size_t plane);
void Conv2d(float* dst, const float* src, const float* weights, const float* bias,
            int input_channels, int output_channels, int input_h, int input_w,
            int output_h, int output_w, int kernel_h, int kernel_w,
            int stride_h, int stride_w, int pad_top, int pad_left,
            bool relu = false);
// Channel-axis Concat of equal-ranked NCHW maps followed by one ordinary
// convolution. Used by the detector FPN head (four 16-channel maps -> 3x3).
// Weights stay in the unfused [M, sum(C), K, K] layout. `PPOCR_DISABLE_CONCAT_CONV`
// is the executor A/B; this helper is the numeric implementation.
void ConcatChannelConv2d(float* dst, const float* const* sources,
                         const int* source_channels, int source_count,
                         const float* weights, const float* bias,
                         int output_channels, int input_h, int input_w,
                         int output_h, int output_w, int kernel_h, int kernel_w,
                         int stride_h, int stride_w, int pad_top, int pad_left,
                         bool relu = false);
// Tiny-det RGB stem: Conv.0 3x3 s2, Conv.1/2 2x2 SAME, MaxPool(Conv.0)||Conv.2
// 3x3 s2. One workspace keeps Conv.0 hot for Conv.1 and the pool, and skips
// four graph-node allocations. `PPOCR_DISABLE_CPU_DET_STEM` is the A/B.
void DetStemFromNchw(float* stem, const float* rgb, const float* conv0_w,
                     const float* conv0_b, int conv0_oc, const float* conv1_w,
                     const float* conv1_b, int conv1_oc, const float* conv2_w,
                     const float* conv2_b, int conv2_oc, const float* stem_w,
                     const float* stem_b, int stem_oc, int input_h, int input_w,
                     bool conv_relu, bool stem_relu);
// Batched NCHW ordinary convolution. The AVX 3x3 paths flatten
// [batch, output-channel tile] work so equal-width OCR crops share a single
// persistent-pool submission. Unsupported geometries retain Conv2d's proven
// per-image dispatch exactly.
void Conv2dBatch(float* dst, const float* src, const float* weights, const float* bias,
                 int batches, int input_channels, int output_channels,
                 int input_h, int input_w, int output_h, int output_w,
                 int kernel_h, int kernel_w, int stride_h, int stride_w,
                 int pad_top, int pad_left, bool relu = false);
void ConvTranspose2x2(float* dst, const float* src, const float* weights, const float* bias,
                      int input_channels, int output_channels, int input_h, int input_w,
                      int act = 0);
// Runs independent NCHW 2x2 stride-two transposed-convolution images through
// one channel-tile schedule. This is the detector FPN upsample form.
// `act` is 0=none, 1=ReLU, 2=exact Sigmoid, applied per output plane while hot.
void ConvTranspose2x2Batch(float* dst, const float* src, const float* weights,
                           const float* bias, int batches, int input_channels,
                           int output_channels, int input_h, int input_w,
                           int act = 0);
// Two chained 2x2 stride-2 ConvTransposes: Relu on the mid 2x map, Sigmoid
// on the 4x output. Each input pixel owns an exclusive 4x4 patch, so the
// 2x intermediate is never materialised.
void ConvTranspose2x2Chain(float* dst, const float* src, const float* w0,
                           const float* b0, const float* w1, const float* b1,
                           int input_channels, int mid_channels,
                           int output_channels, int input_h, int input_w);
// Exact 2x2 stride-1 SAME_UPPER max pooling used by the detector DB head.
void MaxPool2x2Same(float* dst, const float* src, std::size_t planes,
                    int height, int width) noexcept;
// Valid 2x2 stride-1 max pool used by the tiny detector stem before Concat.
void MaxPool2x2Valid(float* dst, const float* src, std::size_t planes,
                     int height, int width) noexcept;
// Exact valid NCHW average pooling with a 3x2 window and matching 3x2
// stride. This is the PP-OCRv6 recognizer bridge between the convolutional
// stack and transformer. Windows do not overlap, so it uses one contiguous
// row pair per output row and preserves the scalar accumulation order.
void AveragePool3x2Valid(float* dst, const float* src, std::size_t planes,
                         int input_height, int input_width) noexcept;
// NCHW depthwise convolution. It covers the MobileNet-style PP-OCRv6
// depthwise 3x3/5x5 layers, including stride and zero padding.
void DepthwiseConv(float* dst, const float* src, const float* weights,
                   const float* bias, int channels, int input_h, int input_w,
                   int output_h, int output_w, int kernel_h, int kernel_w,
                   int stride_h, int stride_w, int pad_top, int pad_left) noexcept;
// Executes independent NCHW depthwise images through one scheduling pass.
// Flattening [batch, channel] avoids a persistent-worker barrier for every
// recognizer crop while retaining the existing ISA-dispatched single-channel
// arithmetic and exact FP32 accumulation order.
void DepthwiseConvBatch(float* dst, const float* src, const float* weights,
                        const float* bias, int batches, int channels,
                        int input_h, int input_w, int output_h, int output_w,
                        int kernel_h, int kernel_w, int stride_h, int stride_w,
                        int pad_top, int pad_left) noexcept;
// Fused depthwise 3x3 s1 pad1 or 5x5 s1 pad2 then 1x1 pointwise, optional
// ReLU/HardSwish/GELU. Keeps each output row's depthwise plane in a C*W
// scratch instead of a full NCHW intermediate. `activation` is 0 none, 1
// ReLU, 2 HardSwish, 3 GELU. Returns false when the geometry should stay on
// the two-pass kernels.
bool DepthwisePointwiseConvFused(float* dst, const float* src,
                                 const float* depthwise_weights,
                                 const float* depthwise_bias,
                                 const float* pointwise_weights,
                                 const float* pointwise_bias, int batches,
                                 int channels, int output_channels, int height,
                                 int width, int kernel_h, int kernel_w,
                                 int stride_h, int stride_w, int pad_top,
                                 int pad_left, int activation) noexcept;
// 1x1 pointwise (optional Relu) then 3x3 s1 pad1 depthwise. Eight-row tiles
// keep the pointwise strip in L2 for the depthwise walk. Rec-sized H<32
// stays on the two-pass kernels. Returns false when the geometry should
// stay unfused.
bool PointwiseDepthwiseConvFused(float* dst, const float* src,
                                 const float* pointwise_weights,
                                 const float* pointwise_bias,
                                 const float* depthwise_weights,
                                 const float* depthwise_bias, int batches,
                                 int input_channels, int output_channels,
                                 int height, int width, bool relu) noexcept;
// Depthwise 3x3 s1 pad1 then invert-residual expand-GELU-project-add. The
// depthwise row stays in a C*W scratch. Falls back to two-pass kernels when
// the geometry is not that SAME 3x3.
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
                                   int pad_left) noexcept;
// Adds A[rows, depth] * B[depth, cols] to an existing C[rows, cols].
void GemmAccumulate(float* dst, const float* a, const float* b,
                    int rows, int cols, int depth);
void Gemm(float* dst, const float* a, const float* b, const float* bias,
          int rows, int cols, int depth);
// Exact class-axis softmax for contiguous row-major logits. `src` and `dst`
// may alias, which lets terminal CTC Softmax reuse a dying logits activation
// instead of allocating/copying a second [N,T,V] tensor.
void SoftmaxRowsInplace(float* values, std::size_t rows, std::size_t width) noexcept;
// Per-NCHW-plane reduction used by PP-OCRv6 squeeze-excitation blocks.
void SpatialMean(float* dst, const float* src, std::size_t planes,
                 std::size_t spatial) noexcept;
// Detector squeeze-excitation: reduce every NCHW feature plane, apply a
// compact two-layer 1x1 gate with ReLU and HardSigmoid, then scale the source
// feature map in place. `values` is both input and output. This avoids the
// generic graph's full-size broadcast-Mul destination while retaining the
// original per-channel reduction and gate arithmetic order.
void SqueezeExcitationGateInplace(float* values, const float* first_weights,
                                  const float* first_bias,
                                  const float* second_weights,
                                  const float* second_bias, int batches,
                                  int channels, int reduced_channels,
                                  std::size_t spatial, float alpha,
                                  float beta, bool residual = false) noexcept;
void CtcTop1(int* indices, float* probabilities, const float* logits,
             std::size_t rows, int steps, int vocab) noexcept;
// Fuses the terminal recognizer row-major GEMM and CTC Top-1 selection. It
// deliberately avoids the full [batches,steps,vocab] logits activation by
// reusing a bounded four-row vocabulary tile per sequence; each tile calls
// the same runtime ISA-dispatched GEMM used by normal inference, then exact
// CTC probability is computed only for emitted (non-blank, non-repeated)
// classes in original step order.
void GemmCtcTop1(int* indices, float* probabilities, const float* left,
                 const float* right, const float* bias, int rows, int depth,
                 int vocab, int steps) noexcept;
// Scalar reference counterpart used by CTC's adaptive dispatch and smoke
// checks. It intentionally preserves the same selected-class probability
// semantics as CtcTop1.
void CtcTop1Scalar(int* indices, float* probabilities, const float* logits,
                   std::size_t rows, int steps, int vocab) noexcept;

}  // namespace ppocr::detail::kernels
