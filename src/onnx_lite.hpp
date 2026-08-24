#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ppocr/ppocr.hpp"
#include "vulkan_backend.hpp"

namespace ppocr::detail {

struct Tensor {
  std::vector<std::int64_t> shape;
  std::vector<float> data;
  Tensor() = default;
  Tensor(std::vector<std::int64_t> tensor_shape, std::vector<float> tensor_data)
      : shape(std::move(tensor_shape)), data(std::move(tensor_data)) {}
  Tensor(const Tensor&) = default;
  Tensor& operator=(const Tensor&) = default;
  Tensor(Tensor&&) noexcept = default;
  Tensor& operator=(Tensor&&) noexcept = default;
  ~Tensor();
  [[nodiscard]] std::size_t size() const;
};

// Intentionally model-specific ONNX executor.  It supports only the
// inference operators used by the PP-OCRv6 detector and recognizer.
class OnnxLite {
 public:
  // Opaque-but-movable device tensor used by the public OCR GPU-only
  // orchestrator.  The slot owns a Vulkan arena allocation until it is handed
  // to another graph or explicitly released through gpu_arena().  Shape
  // metadata is intentionally host-side; it is scheduling metadata only.
  struct GpuTensor {
    std::vector<std::int64_t> shape;
    VulkanTensorSlot slot;
  };
  // `backend` is an execution policy, not an ONNX feature. Only verified
  // elementwise Vulkan segments may be selected by hybrid; every other node
  // continues through the portable CPU executor.
  explicit OnnxLite(const std::string& file, Backend backend = Backend::cpu_only);
  // Inputs are passed by value so call sites can transfer their activation
  // storage into the executor.  This avoids copying large NCHW batches before
  // the first graph operator; lvalue callers remain supported when needed.
  [[nodiscard]] std::unordered_map<std::string, Tensor> Run(
      std::unordered_map<std::string, Tensor> inputs) const;
  // Model-specific readiness for the GPU-only executor. This is separate
  // from merely discovering a Vulkan compute adapter: unsupported graphs are
  // rejected, never silently routed through CPU execution.
  [[nodiscard]] bool SupportsGpuOnly() const noexcept;
  // Executes the strict graph directly from live Vulkan slots.  No floating
  // point activation crosses host memory between the supplied input and the
  // returned output.  Ownership of input slots is consumed; ownership of the
  // returned output slots transfers to the caller.
  [[nodiscard]] bool RunGpuOnlyDevice(
      std::unordered_map<std::string, GpuTensor> inputs,
      std::unordered_map<std::string, GpuTensor>& outputs) const;
  [[nodiscard]] VulkanTensorArena& gpu_arena() const noexcept;
  // Immutable model payload after PP-OCR-specific graph fusion. The OCR
  // scheduler uses it to apply a bounded small-model residency policy.
  [[nodiscard]] std::size_t GpuImmutableBytes() const noexcept;
  // Releases this model's immutable Vulkan slots once no output from its
  // current graph remains live. It is useful when detector and recognizer
  // models share a memory-constrained integrated GPU arena: only the active
  // model's parameters need stay resident between OCR stages.
  void ReleaseGpuConstants() const noexcept;
  struct CtcTop1Output {
    int batches{};
    // `steps` is the common sequence length for a dense result. A value of
    // zero denotes ragged batch-major output; callers then use `row_offsets`
    // and `sequence_steps` to decode each crop's original, unpadded prefix.
    int steps{};
    std::vector<int> indices;
    std::vector<float> probabilities;
    std::vector<std::size_t> row_offsets;
    std::vector<int> sequence_steps;
  };
  // Terminal recognizer graph execution followed by device-side CTC Top-1.
  // Only the compact [N,T] indices/probabilities cross back for text
  // assembly; the [N,T,V] Softmax activation is never downloaded.
  [[nodiscard]] bool RunGpuOnlyCtcTop1(
      std::unordered_map<std::string, GpuTensor> inputs,
      CtcTop1Output& output) const;
  // Replay several already-recorded recognizer graphs with one queue submit
  // and download each fused CTC Top-1. Returns false when any key is missing
  // so the caller can fall back to per-width RunGpuOnlyCtcTop1.
  // `PPOCR_DISABLE_GPU_BATCH_REPLAY` always returns false.
  [[nodiscard]] bool ReplayGpuCtcGraphs(const std::uint64_t* keys, int count,
                                        CtcTop1Output* outputs) const;
  // Pinned NCHW slot a later ReplayPersistentGraph will consume. RGB resize
  // writes it directly so replay skips a device Copy. Empty when the shape
  // has not been recorded yet. `PPOCR_DISABLE_GPU_RESIZE_INTO_REPLAY` always
  // returns an empty slot.
  [[nodiscard]] VulkanTensorSlot PeekGpuReplayInput(
      const std::vector<std::int64_t>& shape, bool fuse_terminal_ctc) const;
  // Pinned packed-RGB slot a detector persistent graph will resize inside the
  // recorded command buffer. Empty until that graph has been recorded with
  // this source size. Rec crops keep a standalone resize: their crop box is
  // not constant per key. Same-host 8-run missed versus standalone resize plus
  // replay, so this stays ENABLE-only (`PPOCR_ENABLE_GPU_RGB_IN_GRAPH`).
  [[nodiscard]] VulkanTensorSlot PeekGpuReplayRgb(
      const std::vector<std::int64_t>& nchw_shape, int source_width,
      int source_height) const;
  // Detector-only: packed RGB is the graph boundary. The persistent CB's
  // first dispatch bilinear-resizes it into pinned NCHW so replay is one
  // submit. Same-host 8-run missed versus standalone resize; ENABLE-only.
  struct GpuRgbFrontEnd {
    VulkanTensorSlot rgb{};
    int source_width = 0;
    int source_height = 0;
    int left = 0;
    int top = 0;
    int region_width = 0;
    int region_height = 0;
    int output_width = 0;
    int output_height = 0;
  };
  [[nodiscard]] bool RunGpuOnlyDeviceFromRgb(
      const GpuRgbFrontEnd& front,
      std::unordered_map<std::string, GpuTensor>& outputs) const;
  // `probabilities` retains the exact terminal-Softmax probability for every
  // emitted CTC character; other rows are intentionally zero.
  [[nodiscard]] CtcTop1Output RunCtcTop1(
      std::unordered_map<std::string, Tensor> inputs) const;
  [[nodiscard]] const std::vector<std::string>& outputs() const noexcept;
  // Detector RGB stem (Conv.0): 3x3 stride-2 C=3 with optional fused ReLU.
  // Hybrid uses this to replace DetInput+first Conv with one GPU segment.
  struct HybridRgbStem {
    std::string output_name;
    const float* weights = nullptr;
    const float* bias = nullptr;
    int output_channels = 0;
    int kernel = 0;
    int stride = 0;
    int pad = 0;
    bool relu = false;
  };
  [[nodiscard]] bool DetectorRgbStem(HybridRgbStem& stem) const noexcept;
  struct HybridRgbStemChain {
    HybridRgbStem conv0;
    HybridRgbStem conv1;
    HybridRgbStem conv2;
    HybridRgbStem stem;
    int conv1_input_channels = 0;
    int conv2_input_channels = 0;
    int stem_input_channels = 0;
  };
  [[nodiscard]] bool DetectorRgbStemChain(HybridRgbStemChain& chain) const noexcept;
  // Recognizer RGB stem (first 3x3 stride-2 C=3). Hybrid replaces RecInput
  // plus that conv with one GPU crop-resize+conv segment.
  [[nodiscard]] bool RecognizerRgbStem(HybridRgbStem& stem) const noexcept;
  [[nodiscard]] std::uint64_t GpuReplayKey(const std::vector<std::int64_t>& shape,
                                           bool fuse_terminal_ctc,
                                           int rgb_source_width = 0,
                                           int rgb_source_height = 0) const noexcept;

 private:
  [[nodiscard]] bool RunGpuOnlyInternal(
      std::unordered_map<std::string, Tensor>* host_inputs,
      std::unordered_map<std::string, GpuTensor> device_values,
      std::unordered_map<std::string, Tensor>* host_outputs,
      std::unordered_map<std::string, GpuTensor>* device_outputs,
      bool fuse_terminal_ctc_softmax = false,
      const GpuRgbFrontEnd* rgb_front = nullptr) const;
  [[nodiscard]] bool RunGpuOnly(std::unordered_map<std::string, Tensor> inputs,
                                std::unordered_map<std::string, Tensor>& outputs) const;
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace ppocr::detail
