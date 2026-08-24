#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ppocr {

// Dependency-free 8-bit RGB image.  Pixels are packed RGB, row-major.
struct Image {
  int width{};
  int height{};
  std::vector<std::uint8_t> rgb;

  [[nodiscard]] bool empty() const noexcept {
    return width <= 0 || height <= 0 ||
           rgb.size() != static_cast<std::size_t>(width) * height * 3;
  }
};

struct Point { float x{}, y{}; };

struct Result {
  std::string text;
  float confidence{};
  std::array<int, 4> bbox{};                 // x, y, width, height
  std::array<std::array<int, 2>, 4> box{};   // TL, TR, BR, BL
};

// Execution policy for PP-OCRv6. CPU is always available. GPU-only never
// silently falls back: preprocessing, neural graphs, DB detection postprocess
// and CTC reduction execute on Vulkan; the host receives only final OCR box
// records/UTF-8 assembly metadata. Hybrid may use only Vulkan segments whose
// end-to-end (including transfer) benchmark is no slower than CPU, otherwise
// it retains the CPU path.
enum class Backend {
  cpu_only,
  gpu_only,
  hybrid,
};

struct BackendInfo {
  bool vulkan_loader_available{};
  bool vulkan_compute_available{};
  bool full_graph_gpu_available{};
  std::string device_name;
  // Changes whenever Vulkan rebuilds its logical device/arena after a failed
  // submission. Hybrid admission decisions are qualified for this exact
  // runtime generation and are invalidated when it changes.
  std::uint64_t vulkan_runtime_generation{};
};

[[nodiscard]] BackendInfo QueryBackendInfo();

struct Options {
  Backend backend = Backend::hybrid;
  float det_threshold = 0.20F;
  float det_box_threshold = 0.45F;
  float det_unclip_ratio = 1.40F;
  int det_limit_side_len = 960;
  int rec_height = 48;
  int rec_max_width = 3200;
  // Recognition crops with exactly the same resized width are evaluated in
  // bounded dynamic batches. Keeping this finite controls activation peak
  // memory on dense pages while removing per-crop executor overhead.
  // Tiny/small CNN+CTC recognizers are width-padding-safe: right-zero padding
  // equals convolution's implicit boundary, so N=4 dense-page batching is the
  // measured default. Transformer exports may still change CTC results for
  // N>1; deployments can set 1 to restore strict per-crop attention.
  int rec_batch_size = 4;
  // Crops whose natural widths land in the same small bucket are packed into
  // one NCHW batch. Each crop keeps its own aspect-ratio resize and is padded
  // on the right, so bucketing avoids resampling text while increasing useful
  // batch density on heterogeneous pages. Set to 1 to retain exact-width
  // grouping (the lower-memory A/B policy) for measurements. The bounded
  // default of 256 packs nearby crop widths without multiplying activation
  // capacity by a large crop count.
  // Wider buckets are available for deployment-specific GPU/batch tuning.
  int rec_batch_width_bucket = 256;
  // Independent recognition batches are safe to run concurrently. This is
  // deliberately capped by default so dense pages do not multiply activation
  // peak memory by all logical CPU cores.
  // Dense-page sweeps on the target 16-logical-core host selected 12 outer
  // batches: it improves tiny/small throughput over eight while avoiding the
  // medium model's measured 16-way bandwidth/cache regression. The recognizer
  // itself still uses channel/row-parallel SIMD kernels.
  int rec_parallelism = 12;
  // A hybrid OCR call uses one persistent Vulkan queue/context. Concurrent
  // graph batches would serialize at that queue while retaining multiple CPU
  // activation workspaces, so hybrid defaults to one graph submitter. CPU
  // execution keeps `rec_parallelism` independent batches in flight. Set a
  // positive value here to override the hybrid cap after measuring a driver.
  int hybrid_graph_parallelism = 1;
  // Same-shape detector pages can share a real NCHW batch as well. Pages are
  // grouped only after their normal PP-OCR resize dimensions have been
  // calculated, so this never pads unlike-sized documents or changes the
  // detector's resize semantics. Keep the batch bounded because detector
  // feature maps are substantially larger than recognition crops.
  int det_batch_size = 4;
  // Resize/normalization is independent for pages that already share a
  // detector batch shape.  Build those NCHW planes concurrently before the
  // single batched graph submission, so a GPU batch does not leave CPU cores
  // idle preparing its input.  This is a bound rather than a per-page thread
  // count; use 1 for strict serial preprocessing or tune it to the host.
  int batch_preprocess_parallelism = 4;
  // Independent page images may also be submitted as one bounded batch. A
  // value of one keeps detector work serial; higher values bound detector
  // concurrency.  Regardless of this value, RecognizeBatch coalesces
  // same-width text crops across pages into real recognizer NCHW batches, so
  // CPU SIMD and Vulkan's batch dimension both see useful work without
  // padding detector pages to a common size.
  int image_batch_parallelism = 1;
};

// PP-OCRv6 detector + recognizer with a deliberately narrow built-in ONNX
// interpreter.  It does not link against ONNX Runtime, Paddle Inference, or
// any other inference engine.
class OCR {
 public:
  OCR(const std::string& det_model, const std::string& rec_model,
      std::string dictionary_path, Options options = {});
  ~OCR();
  OCR(OCR&&) noexcept;
  OCR& operator=(OCR&&) noexcept;
  OCR(const OCR&) = delete;
  OCR& operator=(const OCR&) = delete;

  [[nodiscard]] std::vector<Result> Recognize(const Image& image) const;
  // Runs a batch of independent page images. Results preserve input order;
  // each nested vector is identical to calling Recognize() for that image.
  // This is intentionally a page-level scheduler rather than padding unlike
  // sized detector images into one tensor, which would waste compute and can
  // change PP-OCR's dynamic detector resize semantics.
  [[nodiscard]] std::vector<std::vector<Result>> RecognizeBatch(
      const std::vector<Image>& images) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Small no-dependency input helper for the portable P6 PPM format.  The OCR
// core accepts Image directly, so applications may use their own PNG/JPEG
// decoder without coupling it to this library.
[[nodiscard]] Image LoadPPM(const std::string& path);

}  // namespace ppocr
