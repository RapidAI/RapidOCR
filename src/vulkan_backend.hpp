#pragma once

#include "ppocr/ppocr.hpp"
#include "kernels.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace ppocr::detail {

// Probes the loader and compute-capable physical devices without making
// Vulkan a required build dependency. `full_graph_gpu_available` remains
// false until every selected PP-OCRv6 graph operator has a verified
// device-resident implementation; callers use this to enforce GPU-only's
// no-silent-fallback contract.
[[nodiscard]] BackendInfo QueryVulkanBackendInfo();
// The latest Vulkan submission result from the process-local runtime. This
// is diagnostic state only: an error never authorizes a CPU fallback.
[[nodiscard]] int VulkanLastSubmissionResult() noexcept;
// A cheap process-local hybrid admission epoch. It includes the selected
// adapter identity and changes after Vulkan device-loss recovery; unlike a
// full backend probe it is safe to query at every hybrid operator boundary.
[[nodiscard]] std::uint64_t VulkanHybridAdmissionContext() noexcept;

// A compact device-resident tensor arena for graph-level Vulkan execution.
// Slots are allocated by element count, reused only after their last graph
// consumer, and can be addressed by the executor without copying an
// activation back to host memory.  This is deliberately separate from the
// synchronous Hybrid scratch buffers: GPU-only graph execution needs stable
// slot lifetimes over many dependent dispatches.
struct VulkanTensorSlot {
  std::uint32_t index{};
  std::size_t capacity_elements{};
  std::size_t live_elements{};
  bool resident{};
};

// Compact detector post-processing result produced entirely on the Vulkan
// device.  Coordinates are already in the source-image coordinate system;
// the host only turns these final records into the public Result objects.
struct VulkanDbBox {
  float x0{};
  float y0{};
  float x1{};
  float y1{};
  float score{};
};

// GPU-only graph executor's exact pointwise activation set. They operate on
// an arena slot in place, so activation boundaries never create a host copy.
enum class VulkanUnaryOp : std::uint32_t {
  relu,
  sigmoid,
  swish,
  hard_sigmoid,
  hard_swish,
  // These are graph operators, rather than a host fallback. GELU is evaluated
  // in the shader from a high-accuracy FP32 Erf approximation so a GPU-only
  // graph never has to materialise an activation on CPU just to cross this
  // boundary.
  gelu,
  sqrt,
  square,
};

class VulkanTensorArena {
 public:
  VulkanTensorArena();
  ~VulkanTensorArena();
  VulkanTensorArena(VulkanTensorArena&&) noexcept;
  VulkanTensorArena& operator=(VulkanTensorArena&&) noexcept;
  VulkanTensorArena(const VulkanTensorArena&) = delete;
  VulkanTensorArena& operator=(const VulkanTensorArena&) = delete;

  // Allocates or reuses a device storage slot. `tag` is diagnostic only and
  // does not participate in reuse; callers release a slot after its final
  // graph consumer.  A failed allocation leaves the arena unchanged.
  [[nodiscard]] VulkanTensorSlot Acquire(std::size_t elements,
                                         std::string_view tag = {},
                                         bool persistent = false);
  // Host-visible staging access used only at graph boundaries. Intermediate
  // graph values remain device-resident and are never copied through this API.
  [[nodiscard]] float* mapped_data(const VulkanTensorSlot& slot) noexcept;
  [[nodiscard]] const float* mapped_data(const VulkanTensorSlot& slot) const noexcept;
  // Boundary copies.  The caller uploads graph inputs/constants once and
  // downloads only requested graph outputs; internal values use the in-place
  // device operators below.
  [[nodiscard]] bool Upload(const VulkanTensorSlot& slot, const float* source,
                            std::size_t elements) noexcept;
  // Makes host writes to a mapped input slot visible to a following Vulkan
  // dispatch. It is a no-op for coherent arena memory, but remains explicit
  // so the input boundary is correct on every Vulkan memory implementation.
  [[nodiscard]] bool FlushHostWrites(const VulkanTensorSlot& slot,
                                     std::size_t elements) noexcept;
  [[nodiscard]] bool Download(float* destination, const VulkanTensorSlot& slot,
                              std::size_t elements) const noexcept;
  // Records following arena operations into one Vulkan command buffer.  This
  // is the strict graph executor's synchronization boundary: intermediate
  // values remain on device and EndGraphRecording submits/waits exactly once.
  // The methods are intentionally unavailable to Hybrid's short synchronous
  // primitives, whose admission timing must retain their full boundary.
  [[nodiscard]] bool BeginGraphRecording() noexcept;
  [[nodiscard]] bool EndGraphRecording(bool submit = true) noexcept;
  // Submit the current graph segment, wait for completion, then immediately
  // begin the next recording.  All live arena slots remain resident; unlike a
  // host graph boundary this emits no activation download or upload.
  [[nodiscard]] bool SubmitGraphSegment() noexcept;
  // Record a reusable whole-graph command buffer. Later inferences overwrite
  // only the pinned input slot and submit; intermediate activations stay on
  // the device with no per-operator H2D/D2H. `key` is the caller's shape id.
  [[nodiscard]] bool BeginPersistentGraphRecording(std::uint64_t key) noexcept;
  [[nodiscard]] bool EndPersistentGraphRecording(bool submit = true) noexcept;
  [[nodiscard]] bool ReplayPersistentGraph(std::uint64_t key) noexcept;
  // Submits several persistent graphs on the one queue with a single fence.
  // Command buffers run in `keys` order. Two recognizer widths on a page
  // avoid a host wait between crops. `PPOCR_DISABLE_GPU_BATCH_REPLAY` keeps
  // one submit per key.
  [[nodiscard]] bool ReplayPersistentGraphs(const std::uint64_t* keys, int count) noexcept;
  [[nodiscard]] bool HasPersistentGraph(std::uint64_t key) noexcept;
  [[nodiscard]] bool PersistentGraphRecording() noexcept;
  // Discards one recorded replay command buffer after its model releases the
  // pinned tensor slots. This is a memory-policy operation; it never changes
  // graph execution or permits a CPU fallback.
  void DropPersistentGraph(std::uint64_t key) noexcept;
  void PinForReplay(const VulkanTensorSlot& slot) noexcept;
  void UnpinForReplay(const VulkanTensorSlot& slot) noexcept;
  // RGB resize writes the pinned replay NCHW. The next ReplayPersistentGraph
  // can then submit that standalone CB and the replay CB with one host fence.
  // Same-host 8-run missed versus submit-wait on resize, so chaining stays
  // ENABLE-only (`PPOCR_ENABLE_GPU_CHAIN_RESIZE_REPLAY`).
  void DeferNextStandaloneSubmit() noexcept;
  [[nodiscard]] bool FlushDeferredStandalone() noexcept;
  // GPU image front ends. The source is a packed uint8 RGB byte stream
  // uploaded unchanged to device storage; both operators bilinearly resample
  // and normalize into NCHW without CPU floating-point image processing.
  [[nodiscard]] bool UploadRgb8(const VulkanTensorSlot& slot,
                                const std::uint8_t* source,
                                std::size_t bytes) noexcept;
  [[nodiscard]] bool ResizeRgbToNchw(const VulkanTensorSlot& rgb,
                                     VulkanTensorSlot& output,
                                     int source_width, int source_height,
                                     int left, int top, int region_width,
                                     int region_height, int output_width,
                                     int output_height, float blue_scale,
                                     float blue_offset, float green_scale,
                                     float green_offset, float red_scale,
                                     float red_offset) noexcept;
  // Writes one NCHW result plane at an offset within a larger live output
  // slot. This permits a recognizer batch to retain one device allocation
  // while each crop's GPU resize owns its exact output span.
  [[nodiscard]] bool ResizeRgbToNchwAt(const VulkanTensorSlot& rgb,
                                       VulkanTensorSlot& output,
                                       std::size_t output_offset_elements,
                                       int source_width, int source_height,
                                       int left, int top, int region_width,
                                       int region_height, int output_width,
                                       int output_height, float blue_scale,
                                       float blue_offset, float green_scale,
                                       float green_offset, float red_scale,
                                       float red_offset,
                                       int content_width = 0) noexcept;
  // Batched detector front end for same-size source pages. The packed RGB
  // inputs are concatenated in `rgb`, all pages share one resize geometry,
  // and each output NCHW plane is written by the dispatch's Y dimension.
  // This removes N separate command submissions before an N-batch detector
  // graph while preserving every page's resize result exactly.
  [[nodiscard]] bool ResizeRgbBatchToNchw(const VulkanTensorSlot& rgb,
                                          VulkanTensorSlot& output,
                                          int batches, int source_width,
                                          int source_height, int output_width,
                                          int output_height) noexcept;
  // Applies a detector threshold and downloads only a compact uint8 mask.
  // Component traversal and result-string assembly are integer/control work;
  // they do not execute any inference floating-point operator on CPU.
  [[nodiscard]] bool ThresholdToMask(const VulkanTensorSlot& probability,
                                     std::uint8_t* destination,
                                     std::size_t elements,
                                     float threshold) noexcept;
  // Complete DB detector post-processing on Vulkan.  The probability map is
  // thresholded, eight-connected components are collected, scored and
  // unclipped on device.  Only the compact surviving boxes cross the API
  // boundary. `probability_offset_elements` selects one N plane of a batched
  // detector result without materialising that plane on the CPU.
  [[nodiscard]] bool DbPostprocess(const VulkanTensorSlot& probability,
                                   std::size_t probability_offset_elements,
                                   int height, int width, int image_width,
                                   int image_height, float threshold,
                                   float box_threshold, float unclip_ratio,
                                   std::vector<VulkanDbBox>& boxes) noexcept;
  // CTC top-1 reduction runs on device. Indices/probabilities are compact
  // result metadata downloaded only once after all recognizer graph work.
  [[nodiscard]] bool CtcTop1(const VulkanTensorSlot& logits,
                             std::int32_t* indices, float* probabilities,
                             int batches, int steps, int classes,
                             bool input_is_logits = false) noexcept;
  // Device-only CTC top-1 into caller-owned arena slots. Used to record the
  // reduction into a persistent recognizer graph so replay does not pay a
  // second queue submit. `PPOCR_DISABLE_GPU_FUSE_CTC_GRAPH` restores the
  // separate CtcTop1 submit.
  [[nodiscard]] bool CtcTop1Into(const VulkanTensorSlot& logits,
                                 VulkanTensorSlot& indices,
                                 VulkanTensorSlot& probabilities,
                                 int batches, int steps, int classes,
                                 bool input_is_logits = false) noexcept;
  // Device-to-device contiguous copy. It is the graph allocator's alias-safe
  // primitive: a branch whose source remains live can still produce a new
  // activation without staging through mapped host memory.
  [[nodiscard]] bool Copy(const VulkanTensorSlot& input,
                          VulkanTensorSlot& output,
                          std::size_t elements) noexcept;

  // Executes `left = left op right` completely on Vulkan storage buffers.
  // Both tensors must be live arena slots of the same element count.  The
  // method inserts the required compute write-to-read barrier before return,
  // so its result may feed the next arena operation in the same graph.
  [[nodiscard]] bool BinaryInplace(VulkanTensorSlot& left,
                                   const VulkanTensorSlot& right,
                                   kernels::BinaryOp operation) noexcept;
  // `output = left op right`, where `right` is a contiguous suffix broadcast
  // (for example `[N,C,H,W] op [C,1,1]`). `right_repeat` is H*W and the
  // compact RHS is either shared by all batches or contains one C-vector per
  // batch. This is used by SE gates and graph arithmetic without expanding a
  // channel vector into a full temporary device tensor.
  [[nodiscard]] bool BinaryBroadcast(const VulkanTensorSlot& left,
                                     const VulkanTensorSlot& right,
                                     VulkanTensorSlot& output,
                                     std::size_t batches,
                                     std::size_t right_repeat,
                                     std::size_t right_per_batch,
                                     bool right_shared,
                                     kernels::BinaryOp operation) noexcept;
  // Concatenates two to four contiguous tensors on a host-planned ONNX axis.
  // `outer` and `inner` are the products before/after that axis, while
  // `axis_lengths` contains the per-input extent along it. Inputs and output
  // stay entirely in the arena; PP-OCRv6 uses at most four Concat inputs.
  [[nodiscard]] bool Concat(const std::vector<VulkanTensorSlot>& inputs,
                            VulkanTensorSlot& output, std::size_t outer,
                            std::size_t inner,
                            const std::vector<std::size_t>& axis_lengths) noexcept;
  // Generic integer-factor NCHW nearest-neighbour resize. This covers the
  // remaining unfused PP-OCRv6 FPN Resize nodes; scale factors may differ in
  // height and width and no expanded host activation is materialised.
  [[nodiscard]] bool NearestResize(const VulkanTensorSlot& input,
                                   VulkanTensorSlot& output,
                                   std::size_t batches, int channels,
                                   int input_height, int input_width,
                                   int scale_height, int scale_width) noexcept;
  // Executes a dependent elementwise chain without host activation copies.
  // `left` is updated after every operation; every RHS must have the same
  // shape.  The current portable descriptor layout synchronizes each
  // dispatch, while keeping all intermediate values device-resident.
  [[nodiscard]] bool BinaryChainInplace(VulkanTensorSlot& left,
                                        const std::vector<VulkanTensorSlot>& rights,
                                        const std::vector<kernels::BinaryOp>& operations) noexcept;
  [[nodiscard]] bool UnaryInplace(VulkanTensorSlot& value, VulkanUnaryOp operation,
                                  float alpha = .2F, float beta = .5F) noexcept;
  // Exact ONNX inference BatchNormalization expressed as an NCHW in-place
  // channel affine: `x = x * scale + bias`, where both vectors have C items.
  [[nodiscard]] bool ChannelAffineInplace(VulkanTensorSlot& value,
                                          const VulkanTensorSlot& scale,
                                          const VulkanTensorSlot& bias,
                                          std::size_t batches, int channels,
                                          std::size_t plane) noexcept;
  // Fused inference BatchNorm affine followed by exact Swish. The recognizer
  // uses this frequently; keeping it as one device dispatch avoids a second
  // read/write of the complete NCHW activation.
  [[nodiscard]] bool ChannelAffineSwishInplace(VulkanTensorSlot& value,
                                               const VulkanTensorSlot& scale,
                                               const VulkanTensorSlot& bias,
                                               std::size_t batches, int channels,
                                               std::size_t plane) noexcept;
  [[nodiscard]] bool ChannelAffineHardSwishInplace(VulkanTensorSlot& value,
                                                   const VulkanTensorSlot& scale,
                                                   const VulkanTensorSlot& bias,
                                                   std::size_t batches, int channels,
                                                   std::size_t plane) noexcept;
  // Exact last-axis transforms for recognizer attention. Each row is reduced
  // and rewritten wholly on device; gamma/beta are `[width]` for LayerNorm.
  [[nodiscard]] bool LayerNormInplace(VulkanTensorSlot& value,
                                      const VulkanTensorSlot& gamma,
                                      const VulkanTensorSlot& beta,
                                      std::size_t rows, std::size_t width,
                                      float epsilon) noexcept;
  [[nodiscard]] bool SoftmaxRowsInplace(VulkanTensorSlot& value,
                                        std::size_t rows,
                                        std::size_t width) noexcept;
  // General rank-4 contiguous transpose. Rank-3 graph tensors use a leading
  // unit dimension, so this covers PP-OCRv6's NTC<->NCT and attention layout
  // permutations without copying an activation through CPU memory.
  [[nodiscard]] bool Transpose4D(const VulkanTensorSlot& input,
                                 VulkanTensorSlot& output,
                                 const std::array<int, 4>& dimensions,
                                 const std::array<int, 4>& permutation) noexcept;
  // Per-NCHW-plane spatial mean. It implements the PP-OCR squeeze-excitation
  // ReduceMean/GlobalAveragePool form `[N,C,H,W] -> [N,C,1,1]` without a host
  // reduction or activation download.
  [[nodiscard]] bool SpatialMean(const VulkanTensorSlot& input,
                                 VulkanTensorSlot& output,
                                 std::size_t batches, int channels,
                                 std::size_t plane) noexcept;
  // Device-resident squeeze-excitation gate: spatial mean -> 1x1 Conv+ReLU
  // -> 1x1 Conv -> HardSigmoid -> channel broadcast multiply. Its compact
  // temporaries are allocated/released inside the arena and never cross CPU.
  [[nodiscard]] bool SqueezeExcitationGate(
      const VulkanTensorSlot& input, const VulkanTensorSlot& first_weights,
      const VulkanTensorSlot& first_bias, const VulkanTensorSlot& second_weights,
      const VulkanTensorSlot& second_bias, VulkanTensorSlot& output,
      std::size_t batches, int channels, int reduced_channels,
      std::size_t plane, float alpha, float beta) noexcept;
  // Splits `[N,T,3,H,D]` directly into three attention tensors `[N,H,T,D]`.
  // This mirrors the PP-OCRv6 fused QKV graph transformation and avoids the
  // former full `[3,N,H,T,D]` transient entirely.
  [[nodiscard]] bool QkvSplit(const VulkanTensorSlot& input,
                              VulkanTensorSlot& query,
                              VulkanTensorSlot& key,
                              VulkanTensorSlot& value,
                              std::size_t batches, int steps, int heads,
                              int head_width) noexcept;
  // Batched row-major GEMM for attention. A is `[B,M,K]`, B is `[B,K,N]`,
  // and output `[B,M,N]`, where B normally means batch*heads. The operation
  // keeps QKᵀ and attention×V fully resident between Softmax passes.
  [[nodiscard]] bool BatchedGemm(const VulkanTensorSlot& left,
                                 const VulkanTensorSlot& right,
                                 VulkanTensorSlot& output,
                                 std::size_t matrix_batches, int rows,
                                 int depth, int columns) noexcept;
  // Generic NCHW MaxPool/AveragePool for the PP-OCRv6 backbone. The output
  // remains arena-resident and can feed subsequent Conv without a boundary
  // copy. `average` follows ONNX's count_include_pad=false behavior.
  [[nodiscard]] bool Pool2d(const VulkanTensorSlot& input,
                            VulkanTensorSlot& output, std::size_t batches,
                            int channels, int input_height, int input_width,
                            int output_height, int output_width,
                            int kernel_height, int kernel_width,
                            int stride_height, int stride_width,
                            int pad_top, int pad_left, bool average) noexcept;
  // NCHW 1x1 convolution on live Vulkan buffers. This is the dominant
  // recognizer/device-resident graph primitive: input `[N,C,H,W]`, weights
  // `[M,C]`, bias `[M]`, output `[N,M,H,W]`. The output slot must already
  // hold exactly `batches * output_channels * plane` elements.
  [[nodiscard]] bool PointwiseConv(const VulkanTensorSlot& input,
                                   const VulkanTensorSlot& weights,
                                   const VulkanTensorSlot& bias,
                                   VulkanTensorSlot& output,
                                   std::size_t batches, int input_channels,
                                   int output_channels, std::size_t plane,
                                   bool relu = false, bool swish = false,
                                   bool gelu = false, bool hard_swish = false) noexcept;
  // Dedicated no-shared-memory Vulkan pipeline for the production medium
  // detector's terminal 1792->896 projection. It is a driver-stability
  // isolation, not a CPU fallback; tensors stay in the same Vulkan arena.
  [[nodiscard]] bool PointwiseConvTail(const VulkanTensorSlot& input,
                                       const VulkanTensorSlot& weights,
                                       const VulkanTensorSlot& bias,
                                       VulkanTensorSlot& output,
                                       std::size_t batches, int input_channels,
                                       int output_channels, std::size_t plane) noexcept;
  // Fuses a 1x1 projection and equal-shape residual Add into one device
  // dispatch. The output never materializes before the add, reducing both
  // memory traffic and the strict GPU graph's command count.
  [[nodiscard]] bool PointwiseConvAdd(const VulkanTensorSlot& input,
                                      const VulkanTensorSlot& weights,
                                      const VulkanTensorSlot& bias,
                                      const VulkanTensorSlot& residual,
                                      VulkanTensorSlot& output,
                                      std::size_t batches, int input_channels,
                                      int output_channels, std::size_t plane,
                                      bool relu = false, bool swish = false) noexcept;
  // Expand 1x1 + exact GELU + project 1x1 + residual(input) in one dispatch.
  // `packed_bias` is [expand_bias | project_bias]. Residual is `input`.
  [[nodiscard]] bool ExpandGeluProjectAdd(const VulkanTensorSlot& input,
                                          const VulkanTensorSlot& expand_weights,
                                          const VulkanTensorSlot& packed_bias,
                                          const VulkanTensorSlot& project_weights,
                                          VulkanTensorSlot& output,
                                          std::size_t batches, int channels,
                                          int hidden, std::size_t plane) noexcept;
  // Rec-sized 3x3 s1 pad1 depthwise + 1x1 pointwise in one dispatch.
  // `packed_bias` is [dw_bias | pw_bias]. Optional exact Erf GELU.
  [[nodiscard]] bool DepthwisePointwiseFused(const VulkanTensorSlot& input,
                                             const VulkanTensorSlot& dw_weights,
                                             const VulkanTensorSlot& packed_bias,
                                             const VulkanTensorSlot& pw_weights,
                                             VulkanTensorSlot& output,
                                             std::size_t batches, int channels,
                                             int output_channels, int height,
                                             int width, bool gelu) noexcept;
  // NCHW depthwise convolution on live Vulkan buffers. Weights are
  // `[C,KH,KW]`, bias `[C]`, and output is `[N,C,OH,OW]`. This covers the
  // MobileNet backbone without materializing its intermediate on the host.
  [[nodiscard]] bool DepthwiseConv(const VulkanTensorSlot& input,
                                   const VulkanTensorSlot& weights,
                                   const VulkanTensorSlot& bias,
                                   VulkanTensorSlot& output,
                                   std::size_t batches, int channels,
                                   int input_height, int input_width,
                                   int output_height, int output_width,
                                   int kernel_height, int kernel_width,
                                   int stride_height, int stride_width,
                                   int pad_top, int pad_left,
                                   bool relu = false, bool swish = false,
                                   bool hard_swish = false) noexcept;
  // Scalar one-value-per-invocation depthwise path for qualifying a
  // driver-sensitive tiny high-channel tail. It has the same FP32 semantics
  // as DepthwiseConv and never introduces a host activation fallback.
  [[nodiscard]] bool DepthwiseConvScalar(const VulkanTensorSlot& input,
                                         const VulkanTensorSlot& weights,
                                         const VulkanTensorSlot& bias,
                                         VulkanTensorSlot& output,
                                         std::size_t batches, int channels,
                                         int input_height, int input_width,
                                         int output_height, int output_width,
                                         int kernel_height, int kernel_width,
                                         int stride_height, int stride_width,
                                         int pad_top, int pad_left) noexcept;
  // General NCHW Conv on live Vulkan buffers. Weights are `[M,C,KH,KW]`,
  // bias is `[M]`, and output is `[N,M,OH,OW]`. This covers detector stem,
  // FPN lateral projections and detector heads without host activation
  // materialisation. PP-OCRv6 uses unit dilation and equal H/W strides here.
  [[nodiscard]] bool Conv2d(const VulkanTensorSlot& input,
                            const VulkanTensorSlot& weights,
                            const VulkanTensorSlot& bias,
                            VulkanTensorSlot& output,
                            std::size_t batches, int input_channels,
                            int output_channels, int input_height,
                            int input_width, int output_height,
                            int output_width, int kernel_height,
                            int kernel_width, int stride_height,
                            int stride_width, int pad_top, int pad_left,
                            bool relu = false, bool swish = false,
                            bool sigmoid = false,
                            bool hard_swish = false) noexcept;
  // Detector FPN's no-overlap ConvTranspose(2x2, stride=2). ONNX weights are
  // `[Cin,Cout,2,2]`; the expanded `[N,Cout,2H,2W]` result stays resident.
  [[nodiscard]] bool ConvTranspose2x2(const VulkanTensorSlot& input,
                                      const VulkanTensorSlot& weights,
                                      const VulkanTensorSlot& bias,
                                      VulkanTensorSlot& output,
                                      std::size_t batches, int input_channels,
                                      int output_channels, int input_height,
                                      int input_width, int activation = 0) noexcept;
  // Two chained 2x2 stride-2 ConvTransposes. `packed_w1_b1` is W1[Mid,Cout,2,2]
  // followed by b1[Cout]. Mid is ReLU, the 4x output is Sigmoid.
  [[nodiscard]] bool ConvTranspose2x2Chain(const VulkanTensorSlot& input,
                                           const VulkanTensorSlot& w0,
                                           const VulkanTensorSlot& b0,
                                           const VulkanTensorSlot& packed_w1_b1,
                                           VulkanTensorSlot& output,
                                           std::size_t batches, int input_channels,
                                           int mid_channels, int output_channels,
                                           int input_height, int input_width) noexcept;
  // Fuses detector FPN nearest-neighbour 2x resize and equal-shaped residual
  // addition on arena buffers. `source` is `[N,C,H,W]`, while `residual` and
  // `output` are `[N,C,2H,2W]`.
  [[nodiscard]] bool NearestResize2xAdd(const VulkanTensorSlot& source,
                                        const VulkanTensorSlot& residual,
                                        VulkanTensorSlot& output,
                                        std::size_t batches, int channels,
                                        int input_height, int input_width) noexcept;
  // Row-major FP32 GEMM on arena buffers: `[rows,depth] x [depth,columns]`
  // plus optional `[columns]` bias. This retains transformer projections and
  // terminal CTC logits on Vulkan; Swish can be folded into the final store.
  [[nodiscard]] bool Gemm(const VulkanTensorSlot& left,
                          const VulkanTensorSlot& right,
                          const VulkanTensorSlot* bias,
                          VulkanTensorSlot& output, int rows, int depth,
                          int columns, bool swish = false, int a_lda = 0) noexcept;
  // Terminal CTC projection: GEMM + greedy Top-1 / winner Softmax without a
  // materialised [T,V] logits tensor. Same-host 8-run missed versus tiled
  // GEMM then CtcTop1Into; ENABLE-only (`PPOCR_ENABLE_GPU_FUSE_GEMM_CTC`).
  [[nodiscard]] bool GemmCtcTop1(const VulkanTensorSlot& left,
                                 const VulkanTensorSlot& right,
                                 const VulkanTensorSlot* bias,
                                 VulkanTensorSlot& indices,
                                 VulkanTensorSlot& probabilities, int rows,
                                 int depth, int vocab) noexcept;
  void Release(VulkanTensorSlot& slot) noexcept;
  void Reset() noexcept;
  // Immediately releases every non-live arena allocation, including idle
  // persistent constant buffers. This is for a model hand-off after that
  // model has released its constants; live graph values are never touched.
  [[nodiscard]] bool ReclaimFreeStorage() noexcept;
  // Drops reusable activation storage while retaining persistent model
  // constants. This avoids a weight re-upload at a safe small-model hand-off.
  [[nodiscard]] bool ReclaimFreeTransientStorage() noexcept;
  [[nodiscard]] bool available() const noexcept;
  // Monotonically changes whenever Vulkan has discarded its arena storage
  // (for example after a device-loss recovery).  Model-owned constant caches
  // use this to avoid retaining slot indices from a destroyed logical device.
  [[nodiscard]] std::uint64_t generation() const noexcept;
  [[nodiscard]] std::size_t capacity_bytes() const noexcept;
  // Includes both device-local storage and host staging allocations. This is
  // the meaningful residency metric for the transfer-backed arena.
  [[nodiscard]] std::size_t allocated_bytes() const noexcept;
  [[nodiscard]] std::size_t live_bytes() const noexcept;

 private:
  struct Impl;
  Impl* impl_{};
};

// Runs a numerically exact FP32 binary elementwise kernel through Vulkan
// compute. It is intentionally a small building block for future graph
// segments; false means Vulkan is unavailable or setup/execution failed.
[[nodiscard]] bool VulkanBinary(float* output, const float* left,
                                 const float* right, std::size_t count,
                                 kernels::BinaryOp operation,
                                 bool immutable_right = false) noexcept;

// Executes one to four dependent binary operations in a single Vulkan
// dispatch. The right operand is reused at every stage, so no intermediate
// host/device transfer or temporary tensor is created for an elementwise
// graph segment.
[[nodiscard]] bool VulkanBinaryChain(float* output, const float* left,
                                      const float* right, std::size_t count,
                                      const std::vector<kernels::BinaryOp>& operations,
                                      bool immutable_right = false) noexcept;

// Batched counterpart for independent, equal-sized binary graph segments.
// Each batch item uses its own left/right/output span while one persistent
// Vulkan command submission dispatches the whole 2-D workload. This is the
// useful shape for recognition crop batches and avoids one queue wait plus
// command-buffer setup per crop.
[[nodiscard]] bool VulkanBinaryChainBatch(
    float* output, const float* left, const float* right, std::size_t count,
    std::size_t batches, const std::vector<kernels::BinaryOp>& operations,
    bool immutable_right = false) noexcept;

// NCHW suffix-broadcast counterpart. `right_repeat` is the contiguous span
// represented by one RHS value (for [N,C,H,W] op [N,C,1,1], it is H*W).
// `right_elements` may contain either one compact RHS shared by all batches
// (the normal immutable [1,C,1,1] model-constant case), or one compact RHS
// per batch. Unlike a materialized broadcast, only those compact values cross
// the host/device boundary. The batch dimension still shares one dispatch.
[[nodiscard]] bool VulkanBinaryBroadcastRightChainBatch(
    float* output, const float* left, const float* right, std::size_t count,
    std::size_t batches, std::size_t right_repeat, std::size_t right_elements,
    const std::vector<kernels::BinaryOp>& operations,
    bool immutable_right = false) noexcept;

// Fused NCHW channel affine: `left * scale + bias`.  Both RHS tensors are
// compact [C,1,1] / [N,C,1,1] suffix broadcasts and can independently be
// shared across every batch.  This is the common inference BatchNorm form;
// keeping the two immutable coefficient vectors compact avoids materializing
// either over H*W while one dispatch writes the final activation in place.
[[nodiscard]] bool VulkanChannelAffineBatch(
    float* output, const float* left, const float* scale, const float* bias,
    std::size_t count, std::size_t batches, std::size_t channel_repeat,
    std::size_t coefficient_elements, bool immutable_coefficients = false) noexcept;

// Fused NCHW channel affine followed by exact FP32 Swish.  This is the
// recognizer's BatchNorm+Swish pattern: keeping both operations in one shader
// removes the intermediate host activation pass for a whole crop batch.
[[nodiscard]] bool VulkanChannelAffineSwishBatch(
    float* output, const float* left, const float* scale, const float* bias,
    std::size_t count, std::size_t batches, std::size_t channel_repeat,
    std::size_t coefficient_elements, bool immutable_coefficients = false) noexcept;

// Fused NCHW channel affine followed by exact Swish, then an equal-shaped
// elementwise operation with a second activation. This represents the common
// `BatchNorm -> Swish -> Add/Mul` graph segment without materializing or
// reading back the intermediate Swish activation. `right` is one value per
// output element and may be an immutable model tensor.
[[nodiscard]] bool VulkanChannelAffineSwishBinaryBatch(
    float* output, const float* left, const float* scale, const float* bias,
    const float* right, std::size_t count, std::size_t batches,
    std::size_t channel_repeat, std::size_t coefficient_elements,
    kernels::BinaryOp operation, bool immutable_coefficients = false,
    bool immutable_right = false) noexcept;

// Batched NCHW 1x1 convolution. Input planes and learned [M,C] parameters
// stay compact and the shader maps Y to N, so same-width OCR crops share one
// submission. This covers only unit-stride/unpadded/ungrouped pointwise Conv.
// The optional HardSigmoid is parameterized to retain ONNX alpha/beta exactly.
[[nodiscard]] bool VulkanPointwiseConvBatch(
    float* output, const float* input, const float* weights, const float* bias,
    std::size_t batches, int input_channels, int output_channels,
    std::size_t plane, bool immutable_parameters = false,
    bool relu = false, bool swish = false, bool sigmoid = false,
    bool hard_sigmoid = false, float hard_sigmoid_alpha = .2F,
    float hard_sigmoid_beta = .5F, bool hard_swish = false) noexcept;

// Batched NCHW 1x1 convolution with an equal-shaped residual tensor. The
// residual has its own device buffer, so the projection, add, and optional
// ReLU or exact Swish complete in one dispatch rather than making an
// intermediate tensor. `relu` and `swish` are mutually exclusive.
[[nodiscard]] bool VulkanPointwiseConvAddBatch(
    float* output, const float* input, const float* weights, const float* bias,
    const float* residual, std::size_t batches, int input_channels,
    int output_channels, std::size_t plane, bool immutable_parameters = false,
    bool relu = false, bool swish = false) noexcept;

// Batched NCHW depthwise convolution.  This is the MobileNet backbone's
// other dominant convolution form: one independent [KH,KW] filter per
// channel.  It deliberately uses the same persistent parameter buffers and
// one dispatch across same-width recognition crops.  The hybrid executor
// admits it only after its complete transfer path has matched CPU SIMD.
[[nodiscard]] bool VulkanDepthwiseConvBatch(
    float* output, const float* input, const float* weights, const float* bias,
    std::size_t batches, int channels, int input_height, int input_width,
    int output_height, int output_width, int kernel_height, int kernel_width,
    int stride_height, int stride_width, int pad_top, int pad_left,
    bool immutable_parameters = false, bool relu = false,
    bool swish = false, bool hard_swish = false) noexcept;

// Executes a MobileNet depthwise Conv immediately followed by an unpadded 1x1
// pointwise Conv in one command submission. The optional approximate GELU is
// the same explicit x86 deployment policy selected by kernels::Gelu(); exact
// ONNX GELU stays on CPU. The intermediate stays in a Vulkan storage buffer;
// only the chain boundary crosses host RAM.
[[nodiscard]] bool VulkanDepthwisePointwiseConvBatch(
    float* output, const float* input,
    const float* depthwise_weights, const float* depthwise_bias,
    const float* pointwise_weights, const float* pointwise_bias,
    std::size_t batches, int channels, int output_channels,
    int input_height, int input_width, int output_height, int output_width,
    int kernel_height, int kernel_width, int stride_height, int stride_width,
    int pad_top, int pad_left, bool immutable_parameters = false,
    bool approximate_gelu = false) noexcept;


// Batched ordinary NCHW convolution for the detector's 3x3 / 2x2 layers.
// This intentionally covers only the ungrouped, non-dilated form used by the
// PP-OCRv6 graph.  The hybrid executor still compares the complete host to
// device, dispatch and readback path with the CPU SIMD kernel per shape.
[[nodiscard]] bool VulkanConv2dBatch(
    float* output, const float* input, const float* weights, const float* bias,
    std::size_t batches, int input_channels, int output_channels,
    int input_height, int input_width, int output_height, int output_width,
    int kernel_height, int kernel_width, int stride_height, int stride_width,
    int pad_top, int pad_left, bool immutable_parameters = false,
    bool relu = false, bool swish = false, bool sigmoid = false,
    bool hard_swish = false) noexcept;

// One-submit hybrid fusion for detector Concat+Conv and MaxPool+Concat+Conv.
// Sources stay in the arena: optional 2x2 SAME_UPPER MaxPool on source 0,
// channel Concat, then ordinary 3x3 Conv (+ReLU). Admission includes every
// host upload, the recorded dispatches, and the output download.
[[nodiscard]] bool VulkanConcatConvBatch(
    float* output, const float* const* sources, const int* source_channels,
    int source_count, const float* weights, const float* bias,
    std::size_t batches, int output_channels, int input_height, int input_width,
    int output_height, int output_width, int kernel_height, int kernel_width,
    int stride_height, int stride_width, int pad_top, int pad_left, bool relu,
    bool pool_first, bool immutable_parameters = false) noexcept;

// Packed RGB bilinear resize + PP-OCR detector/recognizer normalization.
// Includes host upload of native RGB bytes, one dispatch, and NCHW readback.
[[nodiscard]] bool VulkanResizeRgbToNchw(
    float* output, const std::uint8_t* rgb, std::size_t rgb_bytes,
    int source_width, int source_height, int output_width, int output_height,
    bool recognition = false) noexcept;

// One hybrid boundary: packed RGB upload, detector bilinear+ImageNet
// normalize, then a 3x3 stride-2 stem Conv(+ReLU). NCHW never returns to
// the host. Admission includes the complete H2D / one-submit / D2H path.
[[nodiscard]] bool VulkanResizeRgbAndConv2d(
    float* output, const std::uint8_t* rgb, std::size_t rgb_bytes,
    int source_width, int source_height, int nchw_width, int nchw_height,
    const float* weights, const float* bias, int output_channels,
    int kernel, int stride, int pad, bool relu) noexcept;

[[nodiscard]] bool VulkanResizeRgbAndConv2dNoSlowerThanCpu(
    int source_width, int source_height, int nchw_width, int nchw_height,
    int output_channels, int kernel, int stride, int pad, bool relu,
    double* gpu_ms = nullptr, double* cpu_ms = nullptr) noexcept;

// Recognizer crop analog of RGB+Conv.0: one H2D of the page RGB, mode-47
// bilinear crop+[-1,1] normalize, then 3x3 stride-2 first conv. NCHW never
// returns to the host. Admission includes the complete transfer path.
[[nodiscard]] bool VulkanResizeRgbCropAndConv2d(
    float* output, const std::uint8_t* rgb, std::size_t rgb_bytes,
    int source_width, int source_height, int crop_x, int crop_y, int crop_w,
    int crop_h, int nchw_width, int nchw_height, const float* weights,
    const float* bias, int output_channels, int kernel, int stride, int pad,
    bool relu) noexcept;

[[nodiscard]] bool VulkanResizeRgbCropAndConv2dNoSlowerThanCpu(
    int source_width, int source_height, int crop_x, int crop_y, int crop_w,
    int crop_h, int nchw_width, int nchw_height, int output_channels,
    int kernel, int stride, int pad, bool relu, double* gpu_ms = nullptr,
    double* cpu_ms = nullptr) noexcept;

// Transfer-amortized detector stem: RGB upload, resize, Conv.0/1/2 and
// MaxPool(Conv.0)||Conv.2 3x3 s2 stay in mapped scratch. Only the 16x40x176
// stem activation is read back. Layer descriptors are NCHW convs with
// optional ReLU; Conv.1/2 are 2x2 SAME_UPPER (pad_top=pad_left=0).
struct VulkanStemLayer {
  const float* weights = nullptr;
  const float* bias = nullptr;
  int input_channels = 0;
  int output_channels = 0;
  int kernel = 0;
  int stride = 0;
  int pad = 0;
  bool relu = false;
};

[[nodiscard]] bool VulkanResizeRgbAndStem(
    float* output, const std::uint8_t* rgb, std::size_t rgb_bytes,
    int source_width, int source_height, int nchw_width, int nchw_height,
    const VulkanStemLayer& conv0, const VulkanStemLayer& conv1,
    const VulkanStemLayer& conv2, const VulkanStemLayer& stem_conv) noexcept;

// RGB + Conv.0/1/2 without the losing pool+concat-conv tail. Conv.0 stays
// in mapped E and Conv.2 in A; both are read back so CPU MaxPoolConcat can
// continue. Admission includes both downloads.
[[nodiscard]] bool VulkanResizeRgbAndConv012(
    float* conv0, float* conv2, const std::uint8_t* rgb, std::size_t rgb_bytes,
    int source_width, int source_height, int nchw_width, int nchw_height,
    const VulkanStemLayer& layer0, const VulkanStemLayer& layer1,
    const VulkanStemLayer& layer2) noexcept;

[[nodiscard]] bool VulkanResizeRgbAndConv012NoSlowerThanCpu(
    int source_width, int source_height, int nchw_width, int nchw_height,
    const VulkanStemLayer& layer0, const VulkanStemLayer& layer1,
    const VulkanStemLayer& layer2, double* gpu_ms = nullptr,
    double* cpu_ms = nullptr) noexcept;

[[nodiscard]] bool VulkanResizeRgbAndStemNoSlowerThanCpu(
    int source_width, int source_height, int nchw_width, int nchw_height,
    const VulkanStemLayer& conv0, const VulkanStemLayer& conv1,
    const VulkanStemLayer& conv2, const VulkanStemLayer& stem_conv,
    double* gpu_ms = nullptr, double* cpu_ms = nullptr) noexcept;

[[nodiscard]] bool VulkanConcatConvBatchNoSlowerThanCpu(
    const int* source_channels, int source_count, int output_channels,
    int input_height, int input_width, int output_height, int output_width,
    int kernel_height, int kernel_width, int stride_height, int stride_width,
    int pad_top, int pad_left, bool relu, bool pool_first,
    double* gpu_ms = nullptr, double* cpu_ms = nullptr,
    bool immutable_parameters = false) noexcept;

// Batched 2x2 stride-two NCHW transposed convolution used by the detector
// FPN. The no-overlap geometry maps each output value directly to one source
// pixel, which makes it a compact GPU upsample/projection primitive.
[[nodiscard]] bool VulkanConvTranspose2x2Batch(
    float* output, const float* input, const float* weights, const float* bias,
    std::size_t batches, int input_channels, int output_channels,
    int input_height, int input_width, bool immutable_parameters = false) noexcept;

// Detector-FPN ConvTranspose(2x2,stride=2) followed by an equal-shaped
// residual Add.  The transposed-convolution result stays in binding D until
// the second dispatch consumes it, so the hybrid boundary transfers only the
// source, residual, immutable parameters, and final sum.
[[nodiscard]] bool VulkanConvTranspose2x2AddBatch(
    float* output, const float* input, const float* weights, const float* bias,
    const float* residual, std::size_t batches, int input_channels,
    int output_channels, int input_height, int input_width,
    bool immutable_parameters = false) noexcept;

// Batched detector-FPN nearest-neighbour 2x upsample followed by an
// equal-shaped residual Add.  The source and residual are distinct NCHW
// inputs; the enlarged temporary never leaves the device.
[[nodiscard]] bool VulkanNearestResize2xAddBatch(
    float* output, const float* source, const float* residual,
    std::size_t batches, int channels, int input_height, int input_width) noexcept;

// Row-major FP32 GEMM for recognizer projections: [rows,depth] x
// [depth,columns] plus an optional column bias.  The Vulkan path preserves
// the ONNX row-major result layout, so CTC can consume logits without a
// transpose or an intermediate CPU tensor.
[[nodiscard]] bool VulkanGemm(
    float* output, const float* left, const float* right, const float* bias,
    int rows, int depth, int columns, bool immutable_parameters = false) noexcept;

// Transfer-amortized recognizer terminal: row-major GEMM plus CTC Top-1.
// Logits stay in mapped scratch; only [rows] class indices and probabilities
// are read back. Admission includes the complete H2D / GEMM / CTC / D2H path.
[[nodiscard]] bool VulkanGemmCtcTop1(
    int* indices, float* probabilities, const float* left, const float* right,
    const float* bias, int rows, int depth, int vocab, int steps,
    bool immutable_parameters = false) noexcept;

[[nodiscard]] bool VulkanGemmCtcTop1NoSlowerThanCpu(
    int rows, int depth, int vocab, double* gpu_ms = nullptr, double* cpu_ms = nullptr,
    bool immutable_parameters = false) noexcept;

// Same row-major GEMM with the recognizer MLP's exact Swish fused into the
// GPU result writeback. It avoids a full CPU-side activation traversal after
// the device readback and accepts all flattened rows of an NCHW crop batch.
[[nodiscard]] bool VulkanGemmSwish(
    float* output, const float* left, const float* right, const float* bias,
    int rows, int depth, int columns, bool immutable_parameters = false) noexcept;


// Same hybrid admission rule for a fused chain: includes transfers and result
// validation, and selects GPU only when the full GPU segment is no slower than
// executing the same dependent operations with CPU SIMD.
[[nodiscard]] bool VulkanBinaryChainNoSlowerThanCpu(
    std::size_t count, const std::vector<kernels::BinaryOp>& operations,
    double* gpu_ms = nullptr, double* cpu_ms = nullptr,
    bool immutable_right = false) noexcept;

// Applies the same end-to-end admission policy to a batch. Transfers,
// dispatch, readback, and every batch result are included. A true result
// therefore means hybrid may select GPU to reduce CPU work without increasing
// the segment latency.
[[nodiscard]] bool VulkanBinaryChainBatchNoSlowerThanCpu(
    std::size_t count, std::size_t batches,
    const std::vector<kernels::BinaryOp>& operations,
    double* gpu_ms = nullptr, double* cpu_ms = nullptr,
    bool immutable_right = false) noexcept;

// End-to-end admission measurement for the suffix-broadcast batch kernel.
// It includes the compact RHS transfer, dispatch, readback and validation.
[[nodiscard]] bool VulkanBinaryBroadcastRightChainBatchNoSlowerThanCpu(
    std::size_t count, std::size_t batches, std::size_t right_repeat,
    std::size_t right_elements, const std::vector<kernels::BinaryOp>& operations,
    double* gpu_ms = nullptr, double* cpu_ms = nullptr,
    bool immutable_right = false) noexcept;

// End-to-end admission measurement for the fused channel-affine batch path.
// It includes both compact coefficient uploads, execution, readback and
// validation against the CPU SIMD Scale+Shift equivalent.
[[nodiscard]] bool VulkanChannelAffineBatchNoSlowerThanCpu(
    std::size_t count, std::size_t batches, std::size_t channel_repeat,
    std::size_t coefficient_elements, double* gpu_ms = nullptr,
    double* cpu_ms = nullptr, bool immutable_coefficients = false) noexcept;

// End-to-end hybrid admission for fused BatchNorm+Swish crop batches.  The
// CPU reference includes both affine and Swish, as does the Vulkan shader.
[[nodiscard]] bool VulkanChannelAffineSwishBatchNoSlowerThanCpu(
    std::size_t count, std::size_t batches, std::size_t channel_repeat,
    std::size_t coefficient_elements, double* gpu_ms = nullptr,
    double* cpu_ms = nullptr, bool immutable_coefficients = false) noexcept;

// End-to-end admission for the batched pointwise convolution. It includes
// transfers, validates against the CPU SIMD kernel, and returns true only
// when this whole GPU segment is no slower than CPU.
[[nodiscard]] bool VulkanPointwiseConvBatchNoSlowerThanCpu(
    std::size_t batches, int input_channels, int output_channels,
    std::size_t plane, double* gpu_ms = nullptr, double* cpu_ms = nullptr,
    bool immutable_parameters = false, bool relu = false, bool swish = false,
    bool sigmoid = false, bool hard_sigmoid = false,
    float hard_sigmoid_alpha = .2F, float hard_sigmoid_beta = .5F,
    bool hard_swish = false) noexcept;

// End-to-end admission for the residual pointwise block. Input/residual
// upload, parameter upload, dispatch, readback and exact CPU validation are
// all included, so hybrid may select Vulkan only when this returns true.
// `relu` and `swish` are mutually exclusive.
[[nodiscard]] bool VulkanPointwiseConvAddBatchNoSlowerThanCpu(
    std::size_t batches, int input_channels, int output_channels,
    std::size_t plane, double* gpu_ms = nullptr, double* cpu_ms = nullptr,
    bool immutable_parameters = false, bool relu = false,
    bool swish = false) noexcept;

// End-to-end admission for MobileNet-style depthwise convolution. Includes
// input/parameter upload, shader dispatch, readback, and an exact CPU-kernel
// validation before returning true.
[[nodiscard]] bool VulkanDepthwiseConvBatchNoSlowerThanCpu(
    std::size_t batches, int channels, int input_height, int input_width,
    int output_height, int output_width, int kernel_height, int kernel_width,
    int stride_height, int stride_width, int pad_top, int pad_left,
    double* gpu_ms = nullptr, double* cpu_ms = nullptr,
    bool immutable_parameters = false, bool relu = false,
    bool swish = false, bool hard_swish = false) noexcept;

// Full-boundary admission probe for VulkanDepthwisePointwiseConvBatch. It
// compares the same CPU batch kernels used by the executor and validates all
// resulting values before reporting that Vulkan is no slower.
[[nodiscard]] bool VulkanDepthwisePointwiseConvBatchNoSlowerThanCpu(
    std::size_t batches, int channels, int output_channels,
    int input_height, int input_width, int output_height, int output_width,
    int kernel_height, int kernel_width, int stride_height, int stride_width,
    int pad_top, int pad_left, double* gpu_ms = nullptr,
    double* cpu_ms = nullptr, bool immutable_parameters = false,
    bool approximate_gelu = false) noexcept;


// End-to-end admission for ordinary batched convolution.  A true result
// includes numerical validation and guarantees that the synchronous Vulkan
// segment is no slower than the matching CPU batched kernel.
[[nodiscard]] bool VulkanConv2dBatchNoSlowerThanCpu(
    std::size_t batches, int input_channels, int output_channels,
    int input_height, int input_width, int output_height, int output_width,
    int kernel_height, int kernel_width, int stride_height, int stride_width,
    int pad_top, int pad_left, double* gpu_ms = nullptr,
    double* cpu_ms = nullptr, bool immutable_parameters = false,
    bool relu = false, bool swish = false, bool sigmoid = false,
    bool hard_swish = false) noexcept;

// End-to-end admission for the FPN 2x2 stride-two transposed convolution.
// Includes parameter/input upload, dispatch, readback and CPU validation.
[[nodiscard]] bool VulkanConvTranspose2x2BatchNoSlowerThanCpu(
    std::size_t batches, int input_channels, int output_channels,
    int input_height, int input_width, double* gpu_ms = nullptr,
    double* cpu_ms = nullptr, bool immutable_parameters = false) noexcept;

// Full-boundary admission probe for the two-dispatch ConvTranspose+Add FPN
// segment. It validates the GPU result against the CPU SIMD ConvTranspose and
// Add sequence, then admits Vulkan only when transfer plus execution is no
// slower than that complete CPU reference.
[[nodiscard]] bool VulkanConvTranspose2x2AddBatchNoSlowerThanCpu(
    std::size_t batches, int input_channels, int output_channels,
    int input_height, int input_width, double* gpu_ms = nullptr,
    double* cpu_ms = nullptr, bool immutable_parameters = false) noexcept;

// End-to-end admission for the fused FPN resize/add batch primitive.  A true
// result includes input/residual uploads, one shader dispatch, readback and
// validation against the CPU fused kernel.
[[nodiscard]] bool VulkanNearestResize2xAddBatchNoSlowerThanCpu(
    std::size_t batches, int channels, int input_height, int input_width,
    double* gpu_ms = nullptr, double* cpu_ms = nullptr) noexcept;

// Full-boundary admission probe for the row-major recognizer GEMM.  It
// validates all values against the existing CPU SIMD GEMM and returns true
// only if upload + dispatch + fence + readback is no slower than CPU.
[[nodiscard]] bool VulkanGemmNoSlowerThanCpu(
    int rows, int depth, int columns, double* gpu_ms = nullptr,
    double* cpu_ms = nullptr, bool immutable_parameters = false) noexcept;

// End-to-end admission probe for fused GEMM+Swish. It compares one GPU
// upload/dispatch/readback against the existing SIMD GEMM followed by Swish.
[[nodiscard]] bool VulkanGemmSwishNoSlowerThanCpu(
    int rows, int depth, int columns, double* gpu_ms = nullptr,
    double* cpu_ms = nullptr, bool immutable_parameters = false) noexcept;

// Measures the complete host-to-device/device-to-host cost of the Vulkan
// binary building block against the CPU SIMD counterpart. A true value means
// the Vulkan result is correct and no slower for this tensor size; it is the
// selection rule future hybrid graph segments must use.
[[nodiscard]] bool VulkanBinaryNoSlowerThanCpu(std::size_t count,
                                                kernels::BinaryOp operation,
                                                double* gpu_ms = nullptr,
                                                double* cpu_ms = nullptr,
                                                bool immutable_right = false) noexcept;

}  // namespace ppocr::detail
