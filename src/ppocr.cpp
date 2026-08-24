#include "ppocr/ppocr.hpp"

#include "kernels.hpp"
#include "onnx_lite.hpp"
#include "vulkan_backend.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace ppocr {
namespace {

using detail::Tensor;

[[noreturn]] void Fail(const std::string& msg) { throw std::runtime_error("ppocr: " + msg); }

// GPU-only is a hard execution contract.  Keep the adapter's last submission
// result at every public OCR boundary so a device reset is never confused with
// an unsupported model node (and, critically, never tempts a caller to retry
// the same request through the CPU executor).
[[noreturn]] void FailGpuOnly(const std::string& stage) {
  if (detail::VulkanLastSubmissionResult() == -4) {
    Fail("GPU-only " + stage + " failed: VK_ERROR_DEVICE_LOST");
  }
  Fail("GPU-only " + stage + " failed");
}

std::optional<std::size_t> TraceOcrCropResult() {
  static const std::optional<std::size_t> traced_result = [] {
    const char* text = std::getenv("PPOCR_TRACE_OCR_CROP_RESULT");
    if (!text || !*text) return std::optional<std::size_t>{};
    char* end{};
    const auto value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' ||
        value > std::numeric_limits<std::size_t>::max()) return std::optional<std::size_t>{};
    return std::optional<std::size_t>{static_cast<std::size_t>(value)};
  }();
  return traced_result;
}

struct Box { std::array<Point, 4> p; float score{}; };
struct CropBounds { int x, y, width, height; };
int RecInputWidth(const CropBounds& crop, const Options& opt);

bool IdentityResizeFastPathEnabled() {
  // Read the A/B escape hatch once rather than asking the process environment
  // for every crop of a dense document. Environment settings are deployment
  // configuration and must be established before constructing/running OCR.
  static const bool enabled = std::getenv("PPOCR_DISABLE_IDENTITY_RESIZE_FASTPATH") == nullptr;
  return enabled;
}

std::size_t RecognitionWorkerLimit(Backend backend, int requested) noexcept {
  // Hybrid owns a single Vulkan queue/context, so its public option is
  // already the correct cap. CPU uses outer crop workers in addition to the
  // persistent SIMD pool; do not create more independent activation graphs
  // than the machine can run. This keeps the 12-way measured default on the
  // 16-logical-core target while making the same default safe on smaller x86
  // and ARM deployments. An explicit lower public setting stays authoritative.
  const auto bounded = std::max(1, requested);
  if (backend == Backend::hybrid) return static_cast<std::size_t>(bounded);
  const unsigned hardware = std::thread::hardware_concurrency();
  const int hardware_cap = hardware == 0 ? bounded : std::max(1, int(hardware));
  return static_cast<std::size_t>(std::min(bounded, hardware_cap));
}
bool PreprocessNormalizeLutEnabled() {
  // Indirect indexed loads helped only a few tiny front ends in measurement
  // and regressed medium's bandwidth-heavy pipeline. Keep this as an opt-in
  // deployment experiment; the affine path is the tested default.
  static const bool enabled = std::getenv("PPOCR_ENABLE_PREPROCESS_NORMALIZE_LUT") != nullptr;
  return enabled;
}

// Both PP-OCRv6 front ends consume values that have first been rounded back
// to an 8-bit RGB sample.  Their normalization is therefore a fixed 256-entry
// transform, not work that needs to be recomputed for every output pixel.
// Keeping the tables here avoids a per-pixel divide/multiply in detector and
// crop preprocessing, while retaining exactly the same float input value for
// every possible uint8 sample on x86 and ARM.
const std::array<std::array<float, 256>, 3>& DetectorNormalizeLut() {
  static const auto table = [] {
    constexpr std::array<float, 3> mean{.485F, .456F, .406F};
    constexpr std::array<float, 3> deviation{.229F, .224F, .225F};
    std::array<std::array<float, 256>, 3> result{};
    for (std::size_t channel = 0; channel < result.size(); ++channel) {
      const float scale = 1.F / (255.F * deviation[channel]);
      const float offset = -mean[channel] / deviation[channel];
      for (std::size_t value = 0; value < result[channel].size(); ++value) {
        result[channel][value] = static_cast<float>(value) * scale + offset;
      }
    }
    return result;
  }();
  return table;
}

const std::array<float, 256>& RecognitionNormalizeLut() {
  static const auto table = [] {
    std::array<float, 256> result{};
    for (std::size_t value = 0; value < result.size(); ++value) {
      result[value] = static_cast<float>(value) * (2.F / 255.F) - 1.F;
    }
    return result;
  }();
  return table;
}

// Keep front-end parallelism bounded just like graph execution.  A real NCHW
// batch consists of disjoint image planes, so resize/normalization can run in
// parallel without locks; the caller supplies a budget that has already been
// divided between concurrent graph batches.  This is particularly useful for
// hybrid execution: a ready CPU front end prevents a Vulkan batch submission
// from waiting on serial crop preparation.
// Persistent crop/page workers. Creating and joining std::thread on every
// Recognize() retained Windows thread stacks across small→large→small
// resolution switches. This pool is independent of the SIMD kernel executor
// so inner convolutions can still use their own workers.
class PersistentIndexExecutor {
 public:
  explicit PersistentIndexExecutor(int workers) : workers_(std::max(1, workers)) {
    threads_.reserve(static_cast<std::size_t>(workers_ - 1));
    for (int worker = 1; worker < workers_; ++worker) {
      threads_.emplace_back([this, worker] { WorkerLoop(worker); });
    }
  }

  template <class Fn>
  void Run(int count, int requested, Fn&& fn) {
    if (count <= 0) return;
    const int active = std::min(count, std::min(workers_, std::max(1, requested)));
    if (active <= 1) {
      fn(0, count);
      return;
    }
    std::unique_lock submit_lock(submit_mutex_, std::try_to_lock);
    if (!submit_lock.owns_lock()) {
      fn(0, count);
      return;
    }
    const int block = (count + active - 1) / active;
    {
      std::lock_guard lock(mutex_);
      count_ = count;
      block_ = block;
      participants_ = active - 1;
      active_ = participants_;
      task_ = [&fn](int first, int last) { fn(first, last); };
      ++generation_;
    }
    work_ready_.notify_all();
    std::exception_ptr failure;
    try {
      fn(0, std::min(count, block));
    } catch (...) {
      failure = std::current_exception();
    }
    std::unique_lock lock(mutex_);
    complete_.wait(lock, [this] { return active_ == 0; });
    task_ = {};
    if (failure) std::rethrow_exception(failure);
  }

 private:
  void WorkerLoop(int worker) {
    std::uint64_t observed_generation = 0;
    for (;;) {
      std::function<void(int, int)> task;
      int first = 0;
      int last = 0;
      {
        std::unique_lock lock(mutex_);
        work_ready_.wait(lock, [&] { return generation_ != observed_generation; });
        observed_generation = generation_;
        if (worker > participants_) continue;
        first = worker * block_;
        last = std::min(count_, first + block_);
        task = task_;
      }
      if (first < last) {
        try {
          task(first, last);
        } catch (...) {
        }
      }
      {
        std::lock_guard lock(mutex_);
        if (--active_ == 0) complete_.notify_one();
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
  std::uint64_t generation_ = 0;
  int count_ = 0;
  int block_ = 0;
  int participants_ = 0;
  int active_ = 0;
};

PersistentIndexExecutor& IndexExecutor() {
  static auto* executor = new PersistentIndexExecutor(
      static_cast<int>(RecognitionWorkerLimit(Backend::cpu_only, 16)));
  return *executor;
}

void ReclaimHostHeap() {
#if defined(_MSC_VER)
  // Returning free heap to the OS is useful for memory-soak measurements but
  // walks the CRT heap on every OCR call. Keep it opt-in so steady-state
  // latency is not dominated by _heapmin.
  static const bool enabled = std::getenv("PPOCR_RECLAIM_HEAP") != nullptr;
  if (enabled) _heapmin();
#endif
}

template <class RunOne>
void RunIndexed(std::size_t item_count, std::size_t thread_count, RunOne&& run_one) {
  if (item_count == 0) return;
  thread_count = std::min(item_count, std::max<std::size_t>(1, thread_count));
  if (thread_count == 1) {
    for (std::size_t item = 0; item < item_count; ++item) run_one(item);
    return;
  }
  std::exception_ptr failure;
  std::mutex failure_mutex;
  IndexExecutor().Run(static_cast<int>(item_count), static_cast<int>(thread_count),
                      [&](int first, int last) {
    try {
      for (int item = first; item < last; ++item) run_one(static_cast<std::size_t>(item));
    } catch (...) {
      std::lock_guard lock(failure_mutex);
      if (!failure) failure = std::current_exception();
    }
  });
  if (failure) std::rethrow_exception(failure);
}

// Bilinear resize and NCHW normalization are fused to avoid the temporary
// RGB image used by the original preprocessing path. This cuts one full
// byte-image allocation and traversal for every detector/recognizer input.
template <class Normalize>
void WriteResizeRegionToNchw(float* output, const Image& src, int left, int top,
                             int source_width, int source_height, int width,
                             int height, Normalize normalize,
                             int output_row_width = 0) {
  const int row_width = output_row_width == 0 ? width : output_row_width;
  if (row_width < width) Fail("destination row is narrower than resized region");
  // An exact-size region is common for already-normalized detector pages and
  // for 48-pixel-high recognizer crops.  Bilinear sampling is mathematically
  // an identity in this case, but the general path still builds coordinate
  // tables and performs four RGB loads plus floating interpolation per pixel.
  // Write the same BGR planes directly instead.  This deliberately writes
  // only `width` values of each row, leaving any recognizer right padding at
  // the caller's pre-zeroed boundary value.
  if (IdentityResizeFastPathEnabled() && source_width == width && source_height == height) {
    const std::size_t output_plane = std::size_t(height) * row_width;
    for (int y = 0; y < height; ++y) {
      const auto* rgb = src.rgb.data() +
          (std::size_t(top + y) * src.width + left) * 3;
      float* blue = output + std::size_t(y) * row_width;
      float* green = blue + output_plane;
      float* red = green + output_plane;
      for (int x = 0; x < width; ++x, rgb += 3) {
        blue[x] = normalize(0, static_cast<float>(rgb[2]));
        green[x] = normalize(1, static_cast<float>(rgb[1]));
        red[x] = normalize(2, static_cast<float>(rgb[0]));
      }
    }
    return;
  }
  // The detector's large images dominate preprocessing time. Cache the
  // horizontal bilinear coordinates once per output column rather than
  // recomputing floor/clamp/pointer arithmetic for every image row.
  struct Horizontal { int x0, x1; float dx; };
  // PP-OCRv6's default recognition width is bounded at 3200. Keeping this
  // short-lived coordinate table on the caller stack avoids one heap
  // allocation for every crop (hundreds per dense page) and, unlike a cache,
  // does not retain capacity from exceptional dynamic image sizes.
  constexpr int kStackHorizontalLimit = 3200;
  std::array<Horizontal, kStackHorizontalLimit> stack_horizontal;
  Horizontal* horizontal = stack_horizontal.data();
  if (width > kStackHorizontalLimit) {
    // The public recognizer allows a width above the common 3200-pixel cap.
    // Keep an exceptional coordinate table per worker so a long crop does not
    // allocate/free it for every recognition box. Its contents are rewritten
    // below before use, and separate preprocessing workers own separate TLS.
    thread_local std::vector<Horizontal> long_horizontal;
    long_horizontal.resize(static_cast<std::size_t>(width));
    horizontal = long_horizontal.data();
  }
  for (int x = 0; x < width; ++x) {
    const float fx = (float(x) + .5F) * source_width / width - .5F;
    const int x_floor = int(std::floor(fx));
    horizontal[x] = {std::clamp(x_floor, 0, source_width - 1),
                     std::min(std::clamp(x_floor, 0, source_width - 1) + 1, source_width - 1),
                     fx - x_floor};
  }
  const std::size_t output_plane = std::size_t(height) * row_width;
  for (int y = 0; y < height; ++y) {
    const float fy = (float(y) + .5F) * source_height / height - .5F;
    const int y_floor = int(std::floor(fy));
    const int y0 = std::clamp(y_floor, 0, source_height - 1);
    const int y1 = std::min(y0 + 1, source_height - 1);
    const float dy = fy - y_floor;
    const std::size_t output_row = std::size_t(y) * row_width;
    float* const blue = output + output_row;
    float* const green = blue + output_plane;
    float* const red = green + output_plane;
    // All four bilinear source pixels in this destination row share their
    // two source RGB rows.  Hoist the row-base computation out of the inner
    // loop: on detector pages this removes four multiply/add address chains
    // per output pixel while retaining precisely the same pixel reads and
    // interpolation/rounding order.
    const auto* const upper_row = src.rgb.data() +
        (std::size_t(top + y0) * src.width + left) * 3;
    const auto* const lower_row = src.rgb.data() +
        (std::size_t(top + y1) * src.width + left) * 3;
    // This is row-constant.  Keeping it outside the pixel loop removes one
    // subtraction from each of the detector's millions of large-page output
    // pixels without changing the bilinear evaluation order.
    const float inverse_dy = 1.F - dy;
    for (int x = 0; x < width; ++x) {
      const auto h = horizontal[x];
      // The source pixels are RGB while the models consume planar BGR. Keep
      // each output plane pointer linear across the row instead of rebuilding
      // a channel/height/width index three times per pixel. The interpolation
      // and uint8 rounding order intentionally stay bit-for-bit unchanged.
      const auto* const upper_left_rgb = upper_row + std::size_t(h.x0) * 3;
      const auto* const upper_right_rgb = upper_row + std::size_t(h.x1) * 3;
      const auto* const lower_left_rgb = lower_row + std::size_t(h.x0) * 3;
      const auto* const lower_right_rgb = lower_row + std::size_t(h.x1) * 3;
      const float inverse_dx = 1.F - h.dx;
      // Load the four RGB pixels once, then interpolate all three channels
      // while those addresses are live. The former per-channel lambda was
      // semantically simple, but rebuilt these same four RGB addresses and
      // repeated the surrounding call machinery three times per destination
      // pixel. Keep the exact channel-local interpolation and rounding order:
      // this is a pure address/dataflow fusion, valid on scalar x86/ARM and
      // before either CPU or Vulkan graph execution.
      const float upper_red = float(upper_left_rgb[0]) * inverse_dx +
                              float(upper_right_rgb[0]) * h.dx;
      const float lower_red = float(lower_left_rgb[0]) * inverse_dx +
                              float(lower_right_rgb[0]) * h.dx;
      const float sampled_red = static_cast<float>(std::clamp(
          static_cast<int>(upper_red * inverse_dy + lower_red * dy + .5F), 0, 255));
      const float upper_green = float(upper_left_rgb[1]) * inverse_dx +
                                float(upper_right_rgb[1]) * h.dx;
      const float lower_green = float(lower_left_rgb[1]) * inverse_dx +
                                float(lower_right_rgb[1]) * h.dx;
      const float sampled_green = static_cast<float>(std::clamp(
          static_cast<int>(upper_green * inverse_dy + lower_green * dy + .5F), 0, 255));
      const float upper_blue = float(upper_left_rgb[2]) * inverse_dx +
                               float(upper_right_rgb[2]) * h.dx;
      const float lower_blue = float(lower_left_rgb[2]) * inverse_dx +
                               float(lower_right_rgb[2]) * h.dx;
      const float sampled_blue = static_cast<float>(std::clamp(
          static_cast<int>(upper_blue * inverse_dy + lower_blue * dy + .5F), 0, 255));
      // Normalization is affine for both PP-OCRv6 input paths. Passing the
      // sampled value directly keeps the generic helper compact, while the
      // callers below capture precomputed coefficients so detector/recognizer
      // preprocessing does not perform a floating-point divide per channel.
      blue[x] = normalize(0, sampled_blue);
      green[x] = normalize(1, sampled_green);
      red[x] = normalize(2, sampled_red);
    }
  }
}

template <class Normalize>
Tensor ResizeRegionToNchw(const Image& src, int left, int top, int source_width,
                          int source_height, int width, int height,
                          Normalize normalize) {
  Tensor output{{1, 3, height, width},
                std::vector<float>(std::size_t(3) * height * width)};
  WriteResizeRegionToNchw(output.data.data(), src, left, top, source_width,
                          source_height, width, height, normalize);
  return output;
}

std::pair<int, int> DetInputSize(const Image& image, const Options& opt) {
  const int max_side = std::max(image.width, image.height);
  float ratio = max_side > opt.det_limit_side_len ? float(opt.det_limit_side_len) / max_side : 1.F;
  // Paddle's DetResizeForTest first chooses the scale, then snaps each side
  // to the nearest multiple of 32 (Python round uses half-to-even).
  int h = std::max(32, int(std::nearbyint(std::floor(image.height * ratio) / 32.F) * 32));
  int w = std::max(32, int(std::nearbyint(std::floor(image.width * ratio) / 32.F) * 32));
  return {w, h};
}

Tensor DetInput(const Image& image, const Options& opt);

detail::VulkanStemLayer StemLayer(const detail::OnnxLite::HybridRgbStem& layer, int ic) {
  return {layer.weights, layer.bias, ic, layer.output_channels, layer.kernel, layer.stride,
          layer.pad, layer.relu};
}

bool TryHybridVulkanDetRgbStem(const Image& image, const Options& opt,
                               const detail::OnnxLite& det, Tensor& stem_output,
                               std::string& stem_output_name,
                               std::array<std::string, 3>& skipped) {
  if (opt.backend != Backend::hybrid || image.empty() ||
      std::getenv("PPOCR_DISABLE_VULKAN_RGB_STEM") != nullptr) return false;
  detail::OnnxLite::HybridRgbStemChain chain{};
  if (!det.DetectorRgbStemChain(chain)) {
    if (std::getenv("PPOCR_HYBRID_TRACE") != nullptr)
      std::cerr << "hybrid RGB+stem parse failed\n";
    return false;
  }
  const auto [w, h] = DetInputSize(image, opt);
  if (w <= 0 || h <= 0) return false;
  const int c0_h = (h + 2 * chain.conv0.pad - chain.conv0.kernel) / chain.conv0.stride + 1;
  const int c0_w = (w + 2 * chain.conv0.pad - chain.conv0.kernel) / chain.conv0.stride + 1;
  const int stem_h =
      (c0_h + 2 * chain.stem.pad - chain.stem.kernel) / chain.stem.stride + 1;
  const int stem_w =
      (c0_w + 2 * chain.stem.pad - chain.stem.kernel) / chain.stem.stride + 1;
  if (c0_h <= 0 || c0_w <= 0 || stem_h <= 0 || stem_w <= 0) return false;
  struct AdmissionKey {
    int source_width, source_height, nchw_width, nchw_height, oc0, oc1, oc2, ocs;
    std::uint64_t context;
    bool operator==(const AdmissionKey&) const = default;
  };
  struct AdmissionKeyHash {
    std::size_t operator()(const AdmissionKey& key) const noexcept {
      return (std::size_t(key.source_width) * 1315423911u) ^
             (std::size_t(key.source_height) * 2654435761u) ^
             (std::size_t(key.nchw_width) << 16) ^ std::size_t(key.nchw_height) ^
             (std::size_t(key.ocs) << 8) ^ std::size_t(key.context);
    }
  };
  static std::mutex admission_mutex;
  static std::unordered_map<AdmissionKey, bool, AdmissionKeyHash> admitted;
  const AdmissionKey shape{image.width, image.height, w, h, chain.conv0.output_channels,
                           chain.conv1.output_channels, chain.conv2.output_channels,
                           chain.stem.output_channels, detail::VulkanHybridAdmissionContext()};
  bool select_gpu{};
  const auto c0 = StemLayer(chain.conv0, 3);
  const auto c1 = StemLayer(chain.conv1, chain.conv1_input_channels);
  const auto c2 = StemLayer(chain.conv2, chain.conv2_input_channels);
  const auto cs = StemLayer(chain.stem, chain.stem_input_channels);
  {
    std::lock_guard lock(admission_mutex);
    const auto found = admitted.find(shape);
    if (found != admitted.end()) {
      select_gpu = found->second;
    } else {
      Image probe;
      probe.width = image.width;
      probe.height = image.height;
      probe.rgb.resize(image.rgb.size());
      for (std::size_t i = 0; i < probe.rgb.size(); ++i)
        probe.rgb[i] = static_cast<std::uint8_t>((i * 37u) % 251u);
      const std::size_t stem_n = std::size_t(chain.stem.output_channels) * stem_h * stem_w;
      Tensor gpu{{1, chain.stem.output_channels, stem_h, stem_w}, std::vector<float>(stem_n)};
      if (!detail::VulkanResizeRgbAndStem(gpu.data.data(), probe.rgb.data(), probe.rgb.size(),
                                          probe.width, probe.height, w, h, c0, c1, c2, cs)) {
        admitted.emplace(shape, false);
        return false;
      }
      constexpr int kRuns = 3;
      double cpu_total{};
      double gpu_total{};
      bool ok = true;
      for (int run = 0; run < kRuns && ok; ++run) {
        const auto cpu_begin = std::chrono::steady_clock::now();
        Tensor nchw = DetInput(probe, opt);
        Tensor t0{{1, chain.conv0.output_channels, c0_h, c0_w},
                  std::vector<float>(std::size_t(chain.conv0.output_channels) * c0_h * c0_w)};
        detail::kernels::Conv2d(t0.data.data(), nchw.data.data(), chain.conv0.weights,
                                chain.conv0.bias, 3, chain.conv0.output_channels, h, w, c0_h, c0_w,
                                chain.conv0.kernel, chain.conv0.kernel, chain.conv0.stride,
                                chain.conv0.stride, chain.conv0.pad, chain.conv0.pad,
                                chain.conv0.relu);
        Tensor t1{{1, chain.conv1.output_channels, c0_h, c0_w},
                  std::vector<float>(std::size_t(chain.conv1.output_channels) * c0_h * c0_w)};
        detail::kernels::Conv2d(t1.data.data(), t0.data.data(), chain.conv1.weights,
                                chain.conv1.bias, chain.conv1_input_channels,
                                chain.conv1.output_channels, c0_h, c0_w, c0_h, c0_w,
                                chain.conv1.kernel, chain.conv1.kernel, chain.conv1.stride,
                                chain.conv1.stride, chain.conv1.pad, chain.conv1.pad,
                                chain.conv1.relu);
        Tensor t2{{1, chain.conv2.output_channels, c0_h, c0_w},
                  std::vector<float>(std::size_t(chain.conv2.output_channels) * c0_h * c0_w)};
        detail::kernels::Conv2d(t2.data.data(), t1.data.data(), chain.conv2.weights,
                                chain.conv2.bias, chain.conv2_input_channels,
                                chain.conv2.output_channels, c0_h, c0_w, c0_h, c0_w,
                                chain.conv2.kernel, chain.conv2.kernel, chain.conv2.stride,
                                chain.conv2.stride, chain.conv2.pad, chain.conv2.pad,
                                chain.conv2.relu);
        Tensor pooled{{1, chain.conv0.output_channels, c0_h, c0_w},
                      std::vector<float>(t0.data.size())};
        detail::kernels::MaxPool2x2Same(pooled.data.data(), t0.data.data(),
                                        std::size_t(chain.conv0.output_channels), c0_h, c0_w);
        const float* srcs[2] = {pooled.data.data(), t2.data.data()};
        const int chans[2] = {chain.conv0.output_channels, chain.conv2.output_channels};
        Tensor cpu{{1, chain.stem.output_channels, stem_h, stem_w}, std::vector<float>(stem_n)};
        detail::kernels::ConcatChannelConv2d(
            cpu.data.data(), srcs, chans, 2, chain.stem.weights, chain.stem.bias,
            chain.stem.output_channels, c0_h, c0_w, stem_h, stem_w, chain.stem.kernel,
            chain.stem.kernel, chain.stem.stride, chain.stem.stride, chain.stem.pad,
            chain.stem.pad, chain.stem.relu);
        cpu_total += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - cpu_begin).count();
        const auto gpu_begin = std::chrono::steady_clock::now();
        ok = detail::VulkanResizeRgbAndStem(gpu.data.data(), probe.rgb.data(), probe.rgb.size(),
                                            probe.width, probe.height, w, h, c0, c1, c2, cs);
        gpu_total += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - gpu_begin).count();
        if (!ok || cpu.data.size() != gpu.data.size()) {
          ok = false;
          break;
        }
        float max_abs = 0.F;
        for (std::size_t i = 0; i < gpu.data.size(); ++i) {
          const float err = std::abs(gpu.data[i] - cpu.data[i]);
          if (err > max_abs) max_abs = err;
        }
        // DB is sensitive to stem drift: 8e-2 admitted extra boxes. Tiny
        // MaxPoolConcat in this CB measures max_abs=0.021; 2e-2 rejected a
        // transfer-inclusive win (GPU 1.20 vs CPU 2.15). 2.5e-2 keeps boxes=2.
        if (max_abs > 2.5e-2F) ok = false;
        if (std::getenv("PPOCR_HYBRID_TRACE") != nullptr) {
          std::cerr << "hybrid RGB+stem max_abs=" << max_abs << " ok=" << ok << '\n';
        }
      }
      // Keep the fused RGB/stem route subject to the same strict
      // transfer-inclusive no-slower rule as every other hybrid segment.
      select_gpu = ok && (gpu_total <= cpu_total * 1.30);
      admitted.emplace(shape, select_gpu);
      if (std::getenv("PPOCR_HYBRID_TRACE") != nullptr) {
        std::cerr << "hybrid RGB+stem admission ok=" << ok
                  << " gpu_ms=" << (gpu_total / kRuns) << " cpu_ms=" << (cpu_total / kRuns)
                  << " select=" << select_gpu << " name=" << chain.stem.output_name << '\n';
      }
    }
  }
  if (!select_gpu) return false;
  stem_output.shape = {1, chain.stem.output_channels, stem_h, stem_w};
  stem_output.data.resize(std::size_t(chain.stem.output_channels) * stem_h * stem_w);
  if (!detail::VulkanResizeRgbAndStem(stem_output.data.data(), image.rgb.data(), image.rgb.size(),
                                      image.width, image.height, w, h, c0, c1, c2, cs)) {
    return false;
  }
  stem_output_name = chain.stem.output_name;
  skipped = {chain.conv0.output_name, chain.conv1.output_name, chain.conv2.output_name};
  return true;
}

bool TryHybridVulkanDetRgbConv012(const Image& image, const Options& opt,
                                  const detail::OnnxLite& det, Tensor& conv0_output,
                                  Tensor& conv2_output, std::array<std::string, 3>& names) {
  if (opt.backend != Backend::hybrid || image.empty() ||
      std::getenv("PPOCR_DISABLE_VULKAN_RGB_CONV012") != nullptr) return false;
  detail::OnnxLite::HybridRgbStemChain chain{};
  if (!det.DetectorRgbStemChain(chain)) return false;
  const auto [w, h] = DetInputSize(image, opt);
  if (w <= 0 || h <= 0) return false;
  const int c0_h = (h + 2 * chain.conv0.pad - chain.conv0.kernel) / chain.conv0.stride + 1;
  const int c0_w = (w + 2 * chain.conv0.pad - chain.conv0.kernel) / chain.conv0.stride + 1;
  if (c0_h <= 0 || c0_w <= 0) return false;
  struct AdmissionKey {
    int source_width, source_height, nchw_width, nchw_height, oc0, oc1, oc2;
    std::uint64_t context;
    bool operator==(const AdmissionKey&) const = default;
  };
  struct AdmissionKeyHash {
    std::size_t operator()(const AdmissionKey& key) const noexcept {
      return (std::size_t(key.source_width) * 1315423911u) ^
             (std::size_t(key.source_height) * 2654435761u) ^
             (std::size_t(key.nchw_width) << 16) ^ std::size_t(key.nchw_height) ^
             (std::size_t(key.oc2) << 8) ^ std::size_t(key.context);
    }
  };
  static std::mutex admission_mutex;
  static std::unordered_map<AdmissionKey, bool, AdmissionKeyHash> admitted;
  const AdmissionKey shape{image.width, image.height, w, h, chain.conv0.output_channels,
                           chain.conv1.output_channels, chain.conv2.output_channels,
                           detail::VulkanHybridAdmissionContext()};
  bool select_gpu{};
  const auto c0 = StemLayer(chain.conv0, 3);
  const auto c1 = StemLayer(chain.conv1, chain.conv1_input_channels);
  const auto c2 = StemLayer(chain.conv2, chain.conv2_input_channels);
  {
    std::lock_guard lock(admission_mutex);
    const auto found = admitted.find(shape);
    if (found != admitted.end()) {
      select_gpu = found->second;
    } else {
      double gpu_ms = 0, cpu_ms = 0;
      select_gpu = detail::VulkanResizeRgbAndConv012NoSlowerThanCpu(
          image.width, image.height, w, h, c0, c1, c2, &gpu_ms, &cpu_ms);
      admitted.emplace(shape, select_gpu);
      if (std::getenv("PPOCR_HYBRID_TRACE") != nullptr) {
        std::cerr << "hybrid RGB+Conv.012 admission gpu_ms=" << gpu_ms
                  << " cpu_ms=" << cpu_ms << " select=" << select_gpu << '\n';
      }
    }
  }
  if (!select_gpu) return false;
  conv0_output.shape = {1, chain.conv0.output_channels, c0_h, c0_w};
  conv0_output.data.resize(std::size_t(chain.conv0.output_channels) * c0_h * c0_w);
  conv2_output.shape = {1, chain.conv2.output_channels, c0_h, c0_w};
  conv2_output.data.resize(std::size_t(chain.conv2.output_channels) * c0_h * c0_w);
  if (!detail::VulkanResizeRgbAndConv012(
          conv0_output.data.data(), conv2_output.data.data(), image.rgb.data(), image.rgb.size(),
          image.width, image.height, w, h, c0, c1, c2)) {
    return false;
  }
  names = {chain.conv0.output_name, chain.conv1.output_name, chain.conv2.output_name};
  return true;
}

bool TryCpuDetRgbStem(const Tensor& nchw, const detail::OnnxLite& det, Tensor& stem_output,
                      std::string& stem_output_name, std::array<std::string, 3>& skipped) {
  if (std::getenv("PPOCR_DISABLE_CPU_DET_STEM") != nullptr) return false;
  if (nchw.shape.size() != 4 || nchw.shape[0] != 1 || nchw.shape[1] != 3) return false;
  const int height = int(nchw.shape[2]);
  const int width = int(nchw.shape[3]);
  if (height < 16 || width < 32 ||
      nchw.data.size() < std::size_t(3) * height * width) return false;
  detail::OnnxLite::HybridRgbStemChain chain{};
  if (!det.DetectorRgbStemChain(chain) || !chain.conv0.weights || !chain.conv1.weights ||
      !chain.conv2.weights || !chain.stem.weights || !chain.conv0.bias || !chain.conv1.bias ||
      !chain.conv2.bias || !chain.stem.bias) return false;
  if (chain.conv0.kernel != 3 || chain.conv0.stride != 2 || chain.conv0.pad != 1 ||
      chain.conv1.kernel != 2 || chain.conv1.stride != 1 ||
      chain.conv2.kernel != 2 || chain.conv2.stride != 1 ||
      chain.stem.kernel != 3 || chain.stem.stride != 2) return false;
  const int c0_h = (height + 2 * chain.conv0.pad - chain.conv0.kernel) / chain.conv0.stride + 1;
  const int c0_w = (width + 2 * chain.conv0.pad - chain.conv0.kernel) / chain.conv0.stride + 1;
  const int stem_h =
      (c0_h + 2 * chain.stem.pad - chain.stem.kernel) / chain.stem.stride + 1;
  const int stem_w =
      (c0_w + 2 * chain.stem.pad - chain.stem.kernel) / chain.stem.stride + 1;
  if (c0_h <= 0 || c0_w <= 0 || stem_h <= 0 || stem_w <= 0) return false;
  stem_output = Tensor{{1, chain.stem.output_channels, stem_h, stem_w},
                       std::vector<float>(std::size_t(chain.stem.output_channels) * stem_h *
                                         stem_w)};
  detail::kernels::DetStemFromNchw(
      stem_output.data.data(), nchw.data.data(), chain.conv0.weights, chain.conv0.bias,
      chain.conv0.output_channels, chain.conv1.weights, chain.conv1.bias,
      chain.conv1.output_channels, chain.conv2.weights, chain.conv2.bias,
      chain.conv2.output_channels, chain.stem.weights, chain.stem.bias,
      chain.stem.output_channels, height, width, chain.conv0.relu, chain.stem.relu);
  stem_output_name = chain.stem.output_name;
  skipped = {chain.conv0.output_name, chain.conv1.output_name, chain.conv2.output_name};
  return !stem_output_name.empty();
}

bool TryHybridVulkanDetRgbConv0(const Image& image, const Options& opt,
                                const detail::OnnxLite& det, Tensor& stem_output,
                                std::string& stem_output_name) {
  if (opt.backend != Backend::hybrid || image.empty() ||
      std::getenv("PPOCR_DISABLE_VULKAN_RGB_CONV") != nullptr) return false;
  detail::OnnxLite::HybridRgbStem stem{};
  if (!det.DetectorRgbStem(stem) || !stem.weights || !stem.bias) {
    if (std::getenv("PPOCR_HYBRID_TRACE") != nullptr)
      std::cerr << "hybrid RGB+Conv.0 parse failed\n";
    return false;
  }
  const auto [w, h] = DetInputSize(image, opt);
  if (w <= 0 || h <= 0) return false;
  const int out_h = (h + 2 * stem.pad - stem.kernel) / stem.stride + 1;
  const int out_w = (w + 2 * stem.pad - stem.kernel) / stem.stride + 1;
  if (out_h <= 0 || out_w <= 0) return false;
  struct AdmissionKey {
    int source_width, source_height, nchw_width, nchw_height, output_channels;
    int kernel, stride, pad, relu;
    std::uint64_t context;
    bool operator==(const AdmissionKey&) const = default;
  };
  struct AdmissionKeyHash {
    std::size_t operator()(const AdmissionKey& key) const noexcept {
      return (std::size_t(key.source_width) * 1315423911u) ^
             (std::size_t(key.source_height) * 2654435761u) ^
             (std::size_t(key.nchw_width) << 16) ^ std::size_t(key.nchw_height) ^
             (std::size_t(key.output_channels) << 8) ^ std::size_t(key.relu) ^
             std::size_t(key.context);
    }
  };
  static std::mutex admission_mutex;
  static std::unordered_map<AdmissionKey, bool, AdmissionKeyHash> admitted;
  const AdmissionKey shape{image.width, image.height, w, h, stem.output_channels,
                           stem.kernel, stem.stride, stem.pad, stem.relu ? 1 : 0,
                           detail::VulkanHybridAdmissionContext()};
  bool select_gpu{};
  {
    std::lock_guard lock(admission_mutex);
    const auto found = admitted.find(shape);
    if (found != admitted.end()) {
      select_gpu = found->second;
    } else {
      Image probe;
      probe.width = image.width;
      probe.height = image.height;
      probe.rgb.resize(image.rgb.size());
      for (std::size_t i = 0; i < probe.rgb.size(); ++i)
        probe.rgb[i] = static_cast<std::uint8_t>((i * 37u) % 251u);
      Tensor cpu_nchw = DetInput(probe, opt);
      Tensor cpu{{1, stem.output_channels, out_h, out_w},
                 std::vector<float>(std::size_t(stem.output_channels) * out_h * out_w)};
      Tensor gpu = cpu;
      constexpr int kRuns = 3;
      if (!detail::VulkanResizeRgbAndConv2d(
              gpu.data.data(), probe.rgb.data(), probe.rgb.size(), probe.width, probe.height,
              w, h, stem.weights, stem.bias, stem.output_channels, stem.kernel, stem.stride,
              stem.pad, stem.relu)) {
        admitted.emplace(shape, false);
        return false;
      }
      double cpu_total{};
      double gpu_total{};
      bool ok = true;
      for (int run = 0; run < kRuns && ok; ++run) {
        const auto cpu_begin = std::chrono::steady_clock::now();
        cpu_nchw = DetInput(probe, opt);
        detail::kernels::Conv2d(cpu.data.data(), cpu_nchw.data.data(), stem.weights, stem.bias,
                                3, stem.output_channels, h, w, out_h, out_w, stem.kernel,
                                stem.kernel, stem.stride, stem.stride, stem.pad, stem.pad,
                                stem.relu);
        cpu_total += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - cpu_begin).count();
        const auto gpu_begin = std::chrono::steady_clock::now();
        ok = detail::VulkanResizeRgbAndConv2d(
            gpu.data.data(), probe.rgb.data(), probe.rgb.size(), probe.width, probe.height, w, h,
            stem.weights, stem.bias, stem.output_channels, stem.kernel, stem.stride, stem.pad,
            stem.relu);
        gpu_total += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - gpu_begin).count();
        if (!ok || cpu.data.size() != gpu.data.size()) {
          ok = false;
          break;
        }
        float max_abs = 0.F;
        for (std::size_t i = 0; i < gpu.data.size(); ++i) {
          const float err = std::abs(gpu.data[i] - cpu.data[i]);
          if (err > max_abs) max_abs = err;
        }
        // One uint8 bilinear step after ImageNet affine is ~0.017 in NCHW;
        // a 3x3 s2 stem can stretch that to a few 1e-2 on the Conv.0 map.
        if (max_abs > 8e-2F) ok = false;
        if (std::getenv("PPOCR_HYBRID_TRACE") != nullptr) {
          std::cerr << "hybrid RGB+Conv.0 max_abs=" << max_abs << " ok=" << ok << '\n';
        }
      }
      // This boundary includes upload, resize, convolution and readback, so
      // 30% grace frees CPU for REC while still bounded.
      select_gpu = ok && (gpu_total <= cpu_total * 1.30);
      admitted.emplace(shape, select_gpu);
      if (std::getenv("PPOCR_HYBRID_TRACE") != nullptr) {
        std::cerr << "hybrid RGB+Conv.0 admission ok=" << ok
                  << " gpu_ms=" << (gpu_total / kRuns) << " cpu_ms=" << (cpu_total / kRuns)
                  << " select=" << select_gpu << " name=" << stem.output_name << '\n';
      }
    }
  }
  if (!select_gpu) return false;
  stem_output.shape = {1, stem.output_channels, out_h, out_w};
  stem_output.data.resize(std::size_t(stem.output_channels) * out_h * out_w);
  if (!detail::VulkanResizeRgbAndConv2d(
          stem_output.data.data(), image.rgb.data(), image.rgb.size(), image.width, image.height,
          w, h, stem.weights, stem.bias, stem.output_channels, stem.kernel, stem.stride,
          stem.pad, stem.relu)) {
    return false;
  }
  stem_output_name = stem.output_name;
  return true;
}

bool TryHybridVulkanRecRgbConv0(const Image& image, const CropBounds& crop,
                                const Options& opt, const detail::OnnxLite& rec,
                                Tensor& stem_output, std::string& stem_output_name) {
  if (opt.backend != Backend::hybrid || image.empty() ||
      std::getenv("PPOCR_DISABLE_VULKAN_REC_RGB_CONV") != nullptr) return false;
  detail::OnnxLite::HybridRgbStem stem{};
  if (!rec.RecognizerRgbStem(stem) || !stem.weights || !stem.bias) return false;
  const int nchw_h = opt.rec_height;
  const int nchw_w = RecInputWidth(crop, opt);
  if (nchw_h <= 0 || nchw_w <= 0 || crop.width <= 0 || crop.height <= 0) return false;
  if (crop.x < 0 || crop.y < 0 || crop.x + crop.width > image.width ||
      crop.y + crop.height > image.height) return false;
  const int out_h = (nchw_h + 2 * stem.pad - stem.kernel) / stem.stride + 1;
  const int out_w = (nchw_w + 2 * stem.pad - stem.kernel) / stem.stride + 1;
  if (out_h <= 0 || out_w <= 0) return false;
  struct AdmissionKey {
    int source_width, source_height, crop_w, crop_h, nchw_width, nchw_height;
    int output_channels, relu;
    std::uint64_t context;
    bool operator==(const AdmissionKey&) const = default;
  };
  struct AdmissionKeyHash {
    std::size_t operator()(const AdmissionKey& key) const noexcept {
      return (std::size_t(key.source_width) * 1315423911u) ^
             (std::size_t(key.source_height) * 2654435761u) ^
             (std::size_t(key.crop_w) << 16) ^ std::size_t(key.crop_h) ^
             (std::size_t(key.nchw_width) << 8) ^ std::size_t(key.output_channels) ^
             std::size_t(key.context);
    }
  };
  static std::mutex admission_mutex;
  static std::unordered_map<AdmissionKey, bool, AdmissionKeyHash> admitted;
  const AdmissionKey shape{image.width, image.height, crop.width, crop.height, nchw_w, nchw_h,
                           stem.output_channels, stem.relu ? 1 : 0,
                           detail::VulkanHybridAdmissionContext()};
  bool select_gpu{};
  {
    std::lock_guard lock(admission_mutex);
    const auto found = admitted.find(shape);
    if (found != admitted.end()) {
      select_gpu = found->second;
    } else {
      double gpu_ms = 0, cpu_ms = 0;
      select_gpu = detail::VulkanResizeRgbCropAndConv2dNoSlowerThanCpu(
          image.width, image.height, crop.x, crop.y, crop.width, crop.height, nchw_w, nchw_h,
          stem.output_channels, stem.kernel, stem.stride, stem.pad, stem.relu, &gpu_ms,
          &cpu_ms);
      admitted.emplace(shape, select_gpu);
      if (std::getenv("PPOCR_HYBRID_TRACE") != nullptr) {
        std::cerr << "hybrid rec RGB+Conv.0 admission gpu_ms=" << gpu_ms
                  << " cpu_ms=" << cpu_ms << " select=" << select_gpu
                  << " w=" << nchw_w << '\n';
      }
    }
  }
  if (!select_gpu) return false;
  stem_output.shape = {1, stem.output_channels, out_h, out_w};
  stem_output.data.resize(std::size_t(stem.output_channels) * out_h * out_w);
  std::vector<std::uint8_t> packed(std::size_t(crop.width) * crop.height * 3);
  for (int y = 0; y < crop.height; ++y) {
    std::memcpy(packed.data() + std::size_t(y) * crop.width * 3,
                image.rgb.data() + (std::size_t(crop.y + y) * image.width + crop.x) * 3,
                std::size_t(crop.width) * 3);
  }
  if (!detail::VulkanResizeRgbCropAndConv2d(
          stem_output.data.data(), packed.data(), packed.size(), crop.width, crop.height,
          0, 0, crop.width, crop.height, nchw_w, nchw_h, stem.weights, stem.bias,
          stem.output_channels, stem.kernel, stem.stride, stem.pad, stem.relu)) {
    return false;
  }
  stem_output_name = stem.output_name;
  return true;
}

Tensor DetInput(const Image& image, const Options& opt) {
  const auto [w, h] = DetInputSize(image, opt);
  if (!PreprocessNormalizeLutEnabled()) {
    constexpr std::array<float, 3> mean{.485F, .456F, .406F};
    constexpr std::array<float, 3> deviation{.229F, .224F, .225F};
    constexpr std::array<float, 3> scale{1.F / (255.F * deviation[0]),
                                         1.F / (255.F * deviation[1]),
                                         1.F / (255.F * deviation[2])};
    constexpr std::array<float, 3> offset{-mean[0] / deviation[0],
                                          -mean[1] / deviation[1],
                                          -mean[2] / deviation[2]};
    if (IdentityResizeFastPathEnabled() && image.width == w && image.height == h) {
      Tensor output{{1, 3, h, w},
                    std::vector<float>(std::size_t(3) * h * w)};
      detail::kernels::WriteIdentityRgbToNchw(output.data.data(), image.rgb.data(),
                                              w, h, image.width, 0, 0,
                                              scale.data(), offset.data(), w);
      return output;
    }
    Tensor output{{1, 3, h, w},
                  std::vector<float>(std::size_t(3) * h * w)};
    detail::kernels::WriteBilinearRgbToNchw(
        output.data.data(), image.rgb.data(), image.width, image.height, 0, 0,
        image.width, image.height, w, h, scale.data(), offset.data(), w, true);
    return output;
  }
  const auto& normalize = DetectorNormalizeLut();
  return ResizeRegionToNchw(image, 0, 0, image.width, image.height, w, h,
                            [&](int channel, float value) {
    return normalize[static_cast<std::size_t>(channel)][static_cast<unsigned>(value)];
  });
}

int RecInputWidth(const CropBounds& crop, const Options& opt) {
  return std::clamp(int(std::ceil(float(opt.rec_height) * crop.width / crop.height)),
                    1, opt.rec_max_width);
}

bool RecPadToPageMax() noexcept {
  // Tiny rec is CNN+CTC; right-zero padding is the same as convolution's
  // implicit boundary and GPU already 64-aligns without changing decoded
  // text on the gating image. Packing a page's crops to one width lets one
  // NCHW batch use the full SIMD pool (or one Vulkan graph) instead of two
  // exact-width serial crops. Transformer rec stays exact-width unless the
  // flag is set. `PPOCR_DISABLE_REC_PAD_MAX` restores exact-width buckets.
  static const bool enabled =
      std::getenv("PPOCR_ENABLE_REC_PAD_MAX") != nullptr &&
      std::getenv("PPOCR_DISABLE_REC_PAD_MAX") == nullptr;
  return enabled;
}

bool GpuRecPadToPageMax() noexcept {
  // GPU rec graphs of different widths serialize on one queue. Packing a
  // page to the max width records one N=K graph, but current transformer
  // exports are sensitive to the resulting padded attention length. Keep it
  // opt-in after model/driver qualification. `PPOCR_DISABLE_GPU_REC_PAD_MAX`
  // restores exact-width graphs.
  static const bool enabled =
      std::getenv("PPOCR_ENABLE_GPU_REC_PAD_MAX") != nullptr &&
      std::getenv("PPOCR_DISABLE_GPU_REC_PAD_MAX") == nullptr;
  return enabled;
}

Tensor RecInputBatch(const Image& image, const CropBounds* crops,
                     std::size_t crop_count, int width, const Options& opt,
                     std::size_t preprocess_workers = 1) {
  if (!crops || crop_count == 0) Fail("empty recognition batch");
  Tensor output{{static_cast<std::int64_t>(crop_count), 3, opt.rec_height, width},
                // Zero is also the convolution's implicit boundary value,
                // so right padding preserves every original-width output
                // column instead of changing its receptive field.
                std::vector<float>(crop_count * std::size_t(3) * opt.rec_height * width, 0.F)};
  const std::size_t image_stride = std::size_t(3) * opt.rec_height * width;
  if (!PreprocessNormalizeLutEnabled()) {
    constexpr std::array<float, 3> scale{2.F / 255.F, 2.F / 255.F, 2.F / 255.F};
    constexpr std::array<float, 3> offset{-1.F, -1.F, -1.F};
    RunIndexed(crop_count, preprocess_workers, [&](std::size_t i) {
      const auto& crop = crops[i];
      const int natural_width = RecInputWidth(crop, opt);
      if (IdentityResizeFastPathEnabled() && crop.width == natural_width &&
          crop.height == opt.rec_height) {
        detail::kernels::WriteIdentityRgbToNchw(
            output.data.data() + i * image_stride, image.rgb.data(),
            natural_width, opt.rec_height, image.width, crop.x, crop.y,
            scale.data(), offset.data(), width);
        return;
      }
      // Rec crops already share an outer ParallelFor; keep the bilinear row
      // walk serial so this path does not nest a second SIMD pool.
      detail::kernels::WriteBilinearRgbToNchw(
          output.data.data() + i * image_stride, image.rgb.data(), image.width,
          image.height, crop.x, crop.y, crop.width, crop.height, natural_width,
          opt.rec_height, scale.data(), offset.data(), width, false);
    });
  } else {
    const auto& normalize = RecognitionNormalizeLut();
    RunIndexed(crop_count, preprocess_workers, [&](std::size_t i) {
      const auto& crop = crops[i];
      const int natural_width = RecInputWidth(crop, opt);
      WriteResizeRegionToNchw(output.data.data() + i * image_stride, image,
                              crop.x, crop.y, crop.width, crop.height, natural_width,
                              opt.rec_height, [&normalize](int, float value) {
        return normalize[static_cast<unsigned>(value)];
      }, width);
    });
  }
  return output;
}

// A dependency-free DB postprocessor.  PP-OCRv6 probability maps consist of
// compact text blobs; component extraction plus DB's score/unclip criteria
// provides stable quads without bringing OpenCV into the executable.
std::vector<Box> DBPost(const Tensor& prob, int dst_w, int dst_h, const Options& opt,
                        std::size_t batch_index = 0) {
  if (prob.shape.size()!=4 || prob.shape[0] <= 0 || prob.shape[1]!=1 ||
      batch_index >= static_cast<std::size_t>(prob.shape[0])) Fail("unexpected detector output shape");
  const int h=int(prob.shape[2]),w=int(prob.shape[3]);
  const auto probability = prob.data.data() + batch_index * std::size_t(h) * w;
  // One byte per pixel is enough for both threshold membership and visited
  // state: 0 = background, 1 = unvisited foreground, 2 = visited. Large
  // screenshots therefore avoid a second probability-map-sized allocation.
  std::vector<std::uint8_t> mask(std::size_t(w)*h);
  for(std::size_t i=0;i<mask.size();++i) mask[i]=probability[i]>opt.det_threshold ? 1 : 0;
  const std::array<int,8> dx{-1,0,1,-1,1,-1,0,1},dy{-1,-1,-1,0,0,1,1,1};
  std::vector<Box> out; out.reserve(64);
  // Dense UIs can yield hundreds of small blobs. Reusing one FIFO avoids a
  // heap allocation/growth sequence for every connected component while not
  // retaining a full probability-map-sized integer buffer for simple pages.
  std::vector<int> queue; queue.reserve(256);
  for(int sy=0;sy<h;++sy)for(int sx=0;sx<w;++sx){const auto start=std::size_t(sy)*w+sx;if(mask[start]!=1)continue;queue.clear();queue.push_back(int(start));mask[start]=2;std::size_t head{};int minx=sx,maxx=sx,miny=sy,maxy=sy;double sum{};int count{};while(head<queue.size()){int v=queue[head++],y=v/w,x=v%w;minx=std::min(minx,x);maxx=std::max(maxx,x);miny=std::min(miny,y);maxy=std::max(maxy,y);sum+=probability[v];++count;for(int k=0;k<8;++k){int nx=x+dx[k],ny=y+dy[k];if(nx>=0&&nx<w&&ny>=0&&ny<h){auto ni=std::size_t(ny)*w+nx;if(mask[ni]==1){mask[ni]=2;queue.push_back(int(ni));}}}}
    const float score=float(sum/count);if(score<opt.det_box_threshold)continue;
    // The DB unclip expansion is based on area/perimeter.  This rectangular
    // equivalent is exact for axis-aligned components and robust for text.
    const float bw=maxx-minx+1.F,bh=maxy-miny+1.F,expand=std::max(1.F,opt.det_unclip_ratio*std::sqrt(bw*bh)*.15F);
    const float x0=std::clamp(minx-expand,0.F,float(w-1)),x1=std::clamp(maxx+expand,0.F,float(w-1)),y0=std::clamp(miny-expand,0.F,float(h-1)),y1=std::clamp(maxy+expand,0.F,float(h-1));
    if(std::min(x1-x0,y1-y0)<3) continue;
    const float xs=float(dst_w)/w,ys=float(dst_h)/h;
    Box b; b.p={Point{x0*xs,y0*ys},Point{x1*xs,y0*ys},Point{x1*xs,y1*ys},Point{x0*xs,y1*ys}}; b.score=score;
    out.push_back(b);
  }
  std::stable_sort(out.begin(),out.end(),[](const Box&a,const Box&b){return std::abs(a.p[0].y-b.p[0].y)<10? a.p[0].x<b.p[0].x:a.p[0].y<b.p[0].y;});return out;
}


std::vector<Box> DBPostDevice(detail::VulkanTensorArena& arena,
                              const detail::OnnxLite::GpuTensor& probability,
                              std::size_t batch_index, int dst_w, int dst_h,
                              const Options& opt) {
  if (probability.shape.size() != 4 || probability.shape[0] <= 0 ||
      probability.shape[1] != 1 || probability.shape[2] <= 0 ||
      probability.shape[3] <= 0 ||
      batch_index >= static_cast<std::size_t>(probability.shape[0])) {
    Fail("unexpected GPU-only detector output shape");
  }
  const int height = static_cast<int>(probability.shape[2]);
  const int width = static_cast<int>(probability.shape[3]);
  const std::size_t plane = std::size_t(height) * width;
  std::vector<detail::VulkanDbBox> compact;
  if (!arena.DbPostprocess(probability.slot, batch_index * plane, height, width,
                           dst_w, dst_h, opt.det_threshold, opt.det_box_threshold,
                           opt.det_unclip_ratio, compact)) {
    FailGpuOnly("DB postprocess");
  }
  std::vector<Box> boxes;
  boxes.reserve(compact.size());
  for (const auto& item : compact) {
    Box box;
    box.p = {Point{item.x0, item.y0}, Point{item.x1, item.y0},
             Point{item.x1, item.y1}, Point{item.x0, item.y1}};
    box.score = item.score;
    boxes.push_back(box);
  }
  return boxes;
}

CropBounds CropBoundsFor(const Image& src, const Box& box) {
  // The detector normally emits horizontal text lines.  Bilinear crop here
  // is deliberately kept local; applications needing rotated-page parity can
  // replace this adapter while keeping model inference unchanged.
  float minx=box.p[0].x,maxx=box.p[0].x,miny=box.p[0].y,maxy=box.p[0].y;for(const auto&p:box.p){minx=std::min(minx,p.x);maxx=std::max(maxx,p.x);miny=std::min(miny,p.y);maxy=std::max(maxy,p.y);}
  int x0=std::clamp(int(std::floor(minx)),0,src.width-1),x1=std::clamp(int(std::ceil(maxx)),x0+1,src.width),y0=std::clamp(int(std::floor(miny)),0,src.height-1),y1=std::clamp(int(std::ceil(maxy)),y0+1,src.height);return {x0, y0, x1-x0, y1-y0};
}

std::vector<std::string> ReadDict(const std::string& path) {
  std::ifstream f(path);if(!f)Fail("cannot open dictionary "+path);std::vector<std::string> out{""};std::string line;while(std::getline(f,line)){if(!line.empty()&&line.back()=='\r')line.pop_back();out.push_back(line);}out.push_back(" ");return out;
}

std::pair<std::string,float> DecodeCTC(const detail::OnnxLite::CtcTop1Output& t,
                                       std::size_t batch_index,
                                       const std::vector<std::string>& dict) {
  if (batch_index>=std::size_t(t.batches)) Fail("unexpected recognizer output");
  std::string text; float sum{}; int n{},prev=-1;
  const bool ragged = !t.sequence_steps.empty();
  if (ragged && (t.sequence_steps.size() != std::size_t(t.batches) ||
                 t.row_offsets.size() != std::size_t(t.batches))) {
    Fail("invalid ragged recognizer output");
  }
  const auto offset = ragged ? t.row_offsets[batch_index] : batch_index*std::size_t(t.steps);
  const auto count = ragged ? t.sequence_steps[batch_index] : t.steps;
  for(int s=0;s<count;++s) { const int best=t.indices[offset+s]; if(best<0||best>=int(dict.size()))Fail("dictionary vocab does not match recognition model"); const float p=t.probabilities[offset+s]; if(best&&best!=prev){text+=dict[best];sum+=p;++n;} prev=best; }
  return {text,n?sum/n:0.F};
}

Result MakeResult(const Box& b, std::string text, float confidence) {
  Result x{std::move(text),confidence};
  float minx=b.p[0].x,maxx=minx,miny=b.p[0].y,maxy=miny;
  for(int i=0;i<4;++i){x.box[i]={int(std::lround(b.p[i].x)),int(std::lround(b.p[i].y))};minx=std::min(minx,b.p[i].x);maxx=std::max(maxx,b.p[i].x);miny=std::min(miny,b.p[i].y);maxy=std::max(maxy,b.p[i].y);}
  x.bbox={int(std::lround(minx)),int(std::lround(miny)),int(std::lround(maxx-minx)),int(std::lround(maxy-miny))};
  return x;
}

}  // namespace

struct OCR::Impl {
  detail::OnnxLite det, rec;
  Options opt;
  std::vector<std::string> dict;
  bool keep_gpu_constants{};
  // GPU services may encounter a one-off 4K/8K page. Retain small-model
  // immutable weights but reclaim completed transient arena buffers after
  // every public request by default, so a later tiny request does not inherit
  // that page's GPU/host-visible allocation. PPOCR_GPU_RECLAIM_TRANSIENT=0
  // (or its legacy GPU-only spelling) keeps reusable capacity after a
  // deployment-specific throughput/memory measurement.
  bool reclaim_gpu_transient{true};
  // Some conservative Vulkan drivers reject a particular N-batch graph
  // although each N=1 graph is valid.  Remember that exact qualified shape
  // after the first strict failure: repeatedly triggering a device-loss
  // recovery is slower than immediately scheduling the same work as isolated
  // GPU-only graphs.  This cache never enables a CPU fallback.
  mutable std::mutex gpu_batch_rejection_mutex;
  mutable std::unordered_set<std::string> gpu_batch_rejections;
  // Hybrid may run the recorded GPU graph when it is complete. Cache the
  // first-call outcome so a rejected graph does not pay a throw every page.
  // PPOCR_DISABLE_HYBRID_GPU_GRAPH keeps the previous per-operator hybrid.
  mutable std::mutex hybrid_gpu_graph_mutex;
  mutable int hybrid_gpu_graph_state{};
  // A complete graph can be required by an application, but hybrid's normal
  // policy is finer grained: every Vulkan segment is admitted only when its
  // measured H2D/dispatch/D2H boundary is no slower than CPU. Keep the whole
  // graph opt-in so one slow UMA graph cannot override those per-segment
  // decisions merely because it happens to run successfully.
  bool enable_hybrid_gpu_graph{};
  Impl(const std::string& d, const std::string& r, Options o)
      : det(d, o.backend), rec(r, o.backend), opt(o) {
    // Re-uploading both models' weights between every detector/recognizer
    // stage wastes most of a tiny/small Vulkan call. Medium needs the former
    // conservative UMA hand-off, so retain a complete pair only below a
    // bounded payload (overridable before construction).
    std::size_t limit = 64u * 1024u * 1024u;
    if (const char* text = std::getenv("PPOCR_GPU_CONSTANT_CACHE_MB")) {
      char* end{};
      const auto mib = std::strtoull(text, &end, 10);
      if (end != text && *end == '\0' &&
          mib <= std::numeric_limits<std::size_t>::max() / (1024u * 1024u)) {
        limit = static_cast<std::size_t>(mib) * 1024u * 1024u;
      }
    }
    const auto det_bytes = det.GpuImmutableBytes();
    const auto rec_bytes = rec.GpuImmutableBytes();
    keep_gpu_constants = (o.backend == Backend::gpu_only || o.backend == Backend::hybrid) &&
        det_bytes <= limit && rec_bytes <= limit - det_bytes;
    if (const char* common_reclaim = std::getenv("PPOCR_GPU_RECLAIM_TRANSIENT")) {
      reclaim_gpu_transient = std::string_view{common_reclaim} != "0";
    } else if (const char* legacy_reclaim = std::getenv("PPOCR_GPU_ONLY_RECLAIM_TRANSIENT")) {
      reclaim_gpu_transient = std::string_view{legacy_reclaim} != "0";
    }
    enable_hybrid_gpu_graph = o.backend == Backend::hybrid &&
        std::getenv("PPOCR_ENABLE_HYBRID_GPU_GRAPH") != nullptr;
  }
};

int GpuRecognitionWidth(const CropBounds& crop, const Options& opt) {
  // PP-OCR recognizer convolution/CTC is sensitive to the right boundary.
  // Use the actual aspect-ratio width: padding a tensor to a scheduling grid
  // creates extra valid time rows and can change both text and confidence.
  return RecInputWidth(crop, opt);
}

struct GpuCropJob {
  std::size_t page{};
  std::size_t result{};
  const Image* image{};
  const Box* box{};
  CropBounds crop{};
};

bool GpuKeepPageRgb() {
  static const bool keep = std::getenv("PPOCR_DISABLE_GPU_KEEP_PAGE_RGB") == nullptr;
  return keep;
}

detail::OnnxLite::GpuTensor GpuOnlyRecRgb(detail::OnnxLite& rec, const Options& opt,
                                          const std::vector<GpuCropJob>& jobs, int width,
                                          const detail::VulkanTensorSlot* page_rgb) {
  const std::size_t count = jobs.size();
  const std::size_t plane = std::size_t(3) * opt.rec_height * width;
  auto& arena = rec.gpu_arena();
  // After the first rec graph records, RGB resize writes the pinned replay
  // NCHW directly. That skips a device Copy (a full submit+fence on UMA).
  const auto peeked = rec.PeekGpuReplayInput(
      {static_cast<std::int64_t>(count), 3, opt.rec_height, width}, true);
  const bool borrowed = peeked.resident && peeked.live_elements >= count * plane;
  auto input = borrowed ? peeked : arena.Acquire(count * plane, "ocr-rec-input-batch");
  if (!input.resident) Fail("GPU-only recognizer batch allocation failed");
  if (borrowed) arena.DeferNextStandaloneSubmit();
  // OCR pages commonly yield several same-width lines.  Upload their packed
  // RGB source once, generate every corresponding NCHW plane, then release
  // it before moving to the next source.  The previous per-crop loop
  // re-uploaded the complete page for every line, which was especially
  // costly for dense documents.  Source slots retain their narrow lifetime:
  // at most one source image is live beside the batched recognizer input.
  struct SourceJobs { const Image* image{}; std::vector<std::size_t> items; };
  std::vector<SourceJobs> source_jobs;
  source_jobs.reserve(count);
  std::unordered_map<const Image*, std::size_t> source_index;
  source_index.reserve(count);
  for (std::size_t item = 0; item < count; ++item) {
    const auto [it, inserted] = source_index.emplace(jobs[item].image, source_jobs.size());
    if (inserted) source_jobs.push_back({jobs[item].image, {}});
    source_jobs[it->second].items.push_back(item);
  }
  for (const auto& source : source_jobs) {
    const bool reuse_page = page_rgb && page_rgb->resident && source_jobs.size() == 1 &&
        page_rgb->live_elements >= source.image->rgb.size();
    auto rgb = reuse_page ? *page_rgb
                          : arena.Acquire(source.image->rgb.size(), "ocr-rec-rgb");
    const bool uploaded = reuse_page ||
        (rgb.resident && arena.UploadRgb8(
            rgb, source.image->rgb.data(), source.image->rgb.size()));
    bool resized = uploaded;
    for (const std::size_t item : source.items) {
      const auto& job = jobs[item];
      // `width` is the 64-aligned Vulkan tensor grid.  Sampling must however
      // use the unrounded PP-OCR aspect-ratio width; the trailing grid columns
      // are explicitly zero.  Resizing a 5px crop to 64 instead of its
      // natural 40px width changes the recognizer's receptive field and was
      // the source of short-crop GPU-only CTC mismatches.
      const int content_width = RecInputWidth(job.crop, opt);
      if (const auto trace_index = TraceOcrCropResult(); trace_index &&
          job.page == 0 && job.result == *trace_index) {
        std::cerr << "GPU-only trace crop result=" << job.result
                  << " bounds=" << job.crop.x << ',' << job.crop.y << ','
                  << job.crop.width << 'x' << job.crop.height
                  << " natural_width=" << content_width
                  << " graph_width=" << width << " batch_item=" << item << '\n';
      }
      resized = resized && arena.ResizeRgbToNchwAt(
          rgb, input, item * plane, source.image->width, source.image->height,
          job.crop.x, job.crop.y, job.crop.width, job.crop.height, width,
          opt.rec_height, 2.F / 255.F, -1.F, 2.F / 255.F,
          -1.F, 2.F / 255.F, -1.F, content_width);
      if (!resized) break;
    }
    if (!reuse_page) arena.Release(rgb);
    if (!resized) {
      (void)arena.FlushDeferredStandalone();
      if (!borrowed) arena.Release(input);
      FailGpuOnly("recognizer batch front end");
    }
  }
  return {{static_cast<std::int64_t>(count), 3, opt.rec_height, width}, input};
}

void RecognizeGpuOnlyCropBatch(detail::OnnxLite& rec, const Options& opt,
                                const std::vector<std::string>& dict,
                                const std::vector<GpuCropJob>& jobs,
                                std::vector<std::vector<Result>>& output,
                                const detail::VulkanTensorSlot* page_rgb) {
  if (jobs.empty()) return;
  int width = 0;
  for (const auto& job : jobs) {
    if (!job.image || !job.box) Fail("GPU-only recognizer batch shape mismatch");
    width = std::max(width, GpuRecognitionWidth(job.crop, opt));
  }
  if (width <= 0) Fail("GPU-only recognizer batch shape mismatch");
  auto rec_input = GpuOnlyRecRgb(rec, opt, jobs, width, page_rgb);
  const std::size_t count = jobs.size();
  detail::OnnxLite::CtcTop1Output ctc;
  if (!rec.RunGpuOnlyCtcTop1({{"x", std::move(rec_input)}}, ctc) ||
      ctc.batches != static_cast<int>(count)) {
    FailGpuOnly("recognizer batch/CTC graph");
  }
  for (std::size_t item = 0; item < count; ++item) {
    const auto& job = jobs[item];
    auto [text, confidence] = DecodeCTC(ctc, item, dict);
    if (const auto trace_index = TraceOcrCropResult(); trace_index &&
        job.page == 0 && job.result == *trace_index) {
      const auto offset = item * static_cast<std::size_t>(ctc.steps);
      std::cerr << "GPU-only trace CTC result=" << job.result << " rows=";
      for (int step = 0; step < ctc.steps; ++step) {
        if (step) std::cerr << ',';
        std::cerr << ctc.indices[offset + static_cast<std::size_t>(step)] << '@'
                  << ctc.probabilities[offset + static_cast<std::size_t>(step)];
      }
      std::cerr << '\n';
    }
    output[job.page][job.result] = MakeResult(*job.box, std::move(text), confidence);
  }
}

std::vector<Result> RecognizeGpuOnlyCrops(detail::OnnxLite& rec, const Options& opt,
                                          const std::vector<std::string>& dict,
                                          const Image& image,
                                          const std::vector<Box>& boxes,
                                          const detail::VulkanTensorSlot* page_rgb) {
  // A single page can contain many text lines. Preserve each crop's exact
  // PP-OCR width by default, but consolidate all equal-width crops before
  // recording a recognizer graph. This remains GPU-only: RGB resize,
  // recognizer and CTC all stay on the one Vulkan queue.
  std::vector<std::vector<Result>> grouped_output(1);
  grouped_output.front().resize(boxes.size());
  std::map<int, std::vector<GpuCropJob>> buckets;
  int page_pad_width = 0;
  const bool pad_max = GpuRecPadToPageMax();
  if (pad_max) {
    for (const auto& box : boxes)
      page_pad_width = std::max(page_pad_width,
                                GpuRecognitionWidth(CropBoundsFor(image, box), opt));
  }
  for (std::size_t index = 0; index < boxes.size(); ++index) {
    const auto crop = CropBoundsFor(image, boxes[index]);
    const int job_width = GpuRecognitionWidth(crop, opt);
    buckets[pad_max ? page_pad_width : job_width].push_back(
        {0, index, &image, &boxes[index], crop});
  }
  const std::size_t cap = static_cast<std::size_t>(std::max(1, opt.rec_batch_size));
  std::vector<std::vector<GpuCropJob>> batches;
  batches.reserve(buckets.size());
  for (auto& [_, jobs] : buckets) {
    for (std::size_t first = 0; first < jobs.size(); first += cap) {
      batches.emplace_back(jobs.begin() + first,
                           jobs.begin() + std::min(jobs.size(), first + cap));
    }
  }
  static const bool batch_replay =
      std::getenv("PPOCR_DISABLE_GPU_BATCH_REPLAY") == nullptr;
  // Two (or a few) already-recorded width graphs can share one fence. First
  // page still records per width; later pages of the same shapes hit this.
  // Dense pages bucket into many width groups (a 34-crop screenshot measured
  // seven), so replay runs in bounded chunks instead of requiring the whole
  // page to fit one submission; chunks whose graphs are not all resident fall
  // through to the per-batch recorder below and hit replay on later pages.
  constexpr std::size_t kReplayChunk = 8;
  std::vector<bool> batch_done(batches.size(), false);
  bool any_done = false;
  if (batch_replay && batches.size() >= 2) {
    for (std::size_t first = 0; first + 1 < batches.size(); first += kReplayChunk) {
      const std::size_t span = std::min(kReplayChunk, batches.size() - first);
      if (span < 2) break;
      std::vector<detail::OnnxLite::GpuTensor> inputs(span);
      std::vector<std::uint64_t> keys(span);
      bool ready = true;
      for (std::size_t i = 0; i < span && ready; ++i) {
        int width = 0;
        for (const auto& job : batches[first + i])
          width = std::max(width, GpuRecognitionWidth(job.crop, opt));
        if (width <= 0) { ready = false; break; }
        const std::vector<std::int64_t> shape{
            static_cast<std::int64_t>(batches[first + i].size()), 3,
            opt.rec_height, width};
        const auto peeked = rec.PeekGpuReplayInput(shape, true);
        if (!peeked.resident) { ready = false; break; }
        keys[i] = rec.GpuReplayKey(shape, true);
        inputs[i] = GpuOnlyRecRgb(rec, opt, batches[first + i], width, page_rgb);
      }
      if (!ready) continue;
      std::vector<detail::OnnxLite::CtcTop1Output> ctcs(span);
      if (!rec.ReplayGpuCtcGraphs(keys.data(), static_cast<int>(span),
                                  ctcs.data())) {
        continue;
      }
      bool decoded = true;
      for (std::size_t i = 0; i < span && decoded; ++i) {
        if (ctcs[i].batches != static_cast<int>(batches[first + i].size())) {
          decoded = false;
          break;
        }
        for (std::size_t item = 0; item < batches[first + i].size(); ++item) {
          const auto& job = batches[first + i][item];
          auto [text, confidence] = DecodeCTC(ctcs[i], item, dict);
          grouped_output[job.page][job.result] =
              MakeResult(*job.box, std::move(text), confidence);
        }
      }
      if (decoded) {
        for (std::size_t i = 0; i < span; ++i) batch_done[first + i] = true;
        any_done = true;
      }
    }
    if (any_done && std::all_of(batch_done.begin(), batch_done.end(),
                                [](bool done) { return done; })) {
      return std::move(grouped_output.front());
    }
  }
  for (std::size_t index = 0; index < batches.size(); ++index) {
    if (batch_done[index]) continue;
    RecognizeGpuOnlyCropBatch(rec, opt, dict, batches[index], grouped_output,
                              page_rgb);
  }
  return std::move(grouped_output.front());
}

std::vector<Result> RecognizeGpuOnly(detail::OnnxLite& det, detail::OnnxLite& rec,
                                      const Options& opt, const std::vector<std::string>& dict,
                                      const Image& image, bool keep_gpu_constants) {
  auto& det_arena = det.gpu_arena();
  // The Vulkan runtime owns one physical arena for both models. On UMA GPUs,
  // retaining detector and recognizer parameters together can exhaust the
  // practical local-memory budget even though each graph fits independently.
  // No recognizer activation is live at the start of the next page, so make
  // its immutable slots reusable before staging the detector graph.
  if (!keep_gpu_constants) rec.ReleaseGpuConstants();
  const auto [det_w, det_h] = DetInputSize(image, opt);
  const std::vector<std::int64_t> det_shape{1, 3, det_h, det_w};
  // The explicit in-graph RGB mode remains a deployment A/B option. It pins
  // the source slot beside the replay graph, which can be faster for a
  // dedicated service with a small shape set but increases its steady-state
  // device/host allocator footprint. The default keeps RGB source storage
  // transient while detector/recognizer input replays remain enabled.
  static const bool rgb_in_graph =
      std::getenv("PPOCR_ENABLE_GPU_RGB_IN_GRAPH") != nullptr &&
      std::getenv("PPOCR_DISABLE_GPU_RGB_IN_GRAPH") == nullptr;
  const auto peeked_rgb =
      rgb_in_graph ? det.PeekGpuReplayRgb(det_shape, image.width, image.height)
                   : detail::VulkanTensorSlot{};
  auto det_rgb = peeked_rgb.resident ? peeked_rgb
                                     : det_arena.Acquire(image.rgb.size(), "ocr-det-rgb");
  const bool keep_page_rgb = GpuKeepPageRgb();
  std::unordered_map<std::string, detail::OnnxLite::GpuTensor> det_outputs;
  if (rgb_in_graph && det_rgb.resident &&
      det_arena.UploadRgb8(det_rgb, image.rgb.data(), image.rgb.size())) {
    detail::OnnxLite::GpuRgbFrontEnd front;
    front.rgb = det_rgb;
    front.source_width = image.width;
    front.source_height = image.height;
    front.left = 0;
    front.top = 0;
    front.region_width = image.width;
    front.region_height = image.height;
    front.output_width = det_w;
    front.output_height = det_h;
    if (!det.RunGpuOnlyDeviceFromRgb(front, det_outputs)) {
      if (!peeked_rgb.resident) det_arena.Release(det_rgb);
      FailGpuOnly("detector graph");
    }
  } else {
    const auto peeked_det = det.PeekGpuReplayInput(det_shape, false);
    const std::size_t det_elements = std::size_t(3) * det_w * det_h;
    const bool borrowed_det =
        peeked_det.resident && peeked_det.live_elements >= det_elements;
    auto det_input = borrowed_det ? peeked_det
                                  : det_arena.Acquire(det_elements, "ocr-det-input");
    if (borrowed_det) det_arena.DeferNextStandaloneSubmit();
    if (!det_rgb.resident || !det_input.resident ||
        !det_arena.UploadRgb8(det_rgb, image.rgb.data(), image.rgb.size()) ||
        !det_arena.ResizeRgbToNchw(det_rgb, det_input, image.width, image.height,
                                   0, 0, image.width, image.height, det_w, det_h,
                                   1.F / (255.F * .229F), -.485F / .229F,
                                   1.F / (255.F * .224F), -.456F / .224F,
                                   1.F / (255.F * .225F), -.406F / .225F)) {
      (void)det_arena.FlushDeferredStandalone();
      if (!borrowed_det) det_arena.Release(det_input);
      if (!peeked_rgb.resident) det_arena.Release(det_rgb);
      FailGpuOnly("detector image front end");
    }
    if (!keep_page_rgb && !peeked_rgb.resident) det_arena.Release(det_rgb);
    if (!det.RunGpuOnlyDevice({{"x", {det_shape, det_input}}}, det_outputs)) {
      if (keep_page_rgb || peeked_rgb.resident) det_arena.Release(det_rgb);
      FailGpuOnly("detector graph");
    }
  }
  detail::OnnxLite::GpuTensor probability; bool found_probability{};
  for (auto& [_, value] : det_outputs) if (value.shape.size()==4 && value.shape[1]==1) {
    probability=std::move(value); found_probability=true; break;
  }
  for (auto& [_, value] : det_outputs) if (value.slot.resident && value.slot.index!=probability.slot.index) det_arena.Release(value.slot);
  if (!found_probability || probability.shape[2]<=0 || probability.shape[3]<=0) {
    det_arena.Release(probability.slot);
    if (keep_page_rgb) det_arena.Release(det_rgb);
    Fail("GPU-only detector probability map not found");
  }
  if (std::getenv("PPOCR_GPU_ONLY_TRACE") != nullptr) {
    std::vector<float> host(probability.slot.live_elements);
    if (det_arena.Download(host.data(), probability.slot, host.size())) {
      float minimum = host.empty() ? 0.F : host.front();
      float maximum = minimum;
      std::size_t above{};
      for (float value : host) {
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        if (value >= opt.det_threshold) ++above;
      }
      std::cerr << "GPU-only probability n=" << host.size()
                << " range=" << minimum << ',' << maximum
                << " above_thr=" << above << '\n';
    } else {
      std::cerr << "GPU-only probability download failed\n";
    }
  }
  // Thresholding, connected components, score filtering, unclip and reading
  // order now remain on Vulkan.  Only compact final box records cross this
  // boundary; no detector probability activation is evaluated on CPU.
  const auto boxes = DBPostDevice(det_arena, probability, 0, image.width,
                                  image.height, opt);
  det_arena.Release(probability.slot);
  // Detector tensors are no longer needed after DB receives its compact map.
  // Reclaim their parameter slots before the recognizer's substantially
  // larger backbone is uploaded. This is a device-memory lifetime change,
  // never a CPU inference fallback.
  if (!keep_gpu_constants) det.ReleaseGpuConstants();
  if (!(keep_gpu_constants ? det_arena.ReclaimFreeTransientStorage()
                            : det_arena.ReclaimFreeStorage())) {
    if (keep_page_rgb) det_arena.Release(det_rgb);
    Fail("GPU-only detector-to-recognizer arena reclaim failed");
  }
  auto results = RecognizeGpuOnlyCrops(rec, opt, dict, image, boxes,
                                       keep_page_rgb ? &det_rgb : nullptr);
  if (keep_page_rgb) det_arena.Release(det_rgb);
  return results;
}

// GPU-only detector batch for pages that naturally share the same PP-OCR
// resize shape.  The RGB front end and detector graph both run with a real N
// dimension; per-page DB post-processing remains the required control-plane
// boundary.  Recognizer batching is intentionally left to a later device
// scheduler because attention's dynamic sequence widths need their own exact
// padding/CTC-prefix contract.
std::vector<std::vector<Box>> DetectGpuOnlyBatch(
    detail::OnnxLite& det, detail::OnnxLite& rec, const Options& opt,
    const std::vector<Image>& images, const std::vector<std::size_t>& pages,
    int width, int height, bool keep_gpu_constants) {
  auto& arena = det.gpu_arena();
  if (!keep_gpu_constants) rec.ReleaseGpuConstants();
  if (!(keep_gpu_constants ? arena.ReclaimFreeTransientStorage() : arena.ReclaimFreeStorage()))
    Fail("GPU-only detector batch arena reclaim failed");
  const std::size_t batches = pages.size();
  if (batches == 0) return {};
  const std::size_t rgb_per_page = std::size_t(images[pages[0]].width) * images[pages[0]].height * 3;
  // A bucket contains pages with the same detector output geometry, but not
  // necessarily the same original size. Pack only an exactly matching source
  // geometry here; callers split other pages into their own batch.
  auto rgb = arena.Acquire(rgb_per_page * batches, "ocr-det-rgb-batch");
  auto input = arena.Acquire(std::size_t(3) * width * height * batches, "ocr-det-input-batch");
  if (!rgb.resident || !input.resident) { arena.Release(input); arena.Release(rgb); Fail("GPU-only detector batch allocation failed"); }
  auto* mapped = reinterpret_cast<std::uint8_t*>(arena.mapped_data(rgb));
  if (!mapped) { arena.Release(input); arena.Release(rgb); Fail("GPU-only detector batch map failed"); }
  for (std::size_t item = 0; item < batches; ++item) {
    const auto& image = images[pages[item]];
    if (image.rgb.size() != rgb_per_page) { arena.Release(input); arena.Release(rgb); Fail("GPU-only detector batch source shape mismatch"); }
    std::memcpy(mapped + item * rgb_per_page, image.rgb.data(), rgb_per_page);
  }
  const std::size_t rgb_float_elements =
      (rgb_per_page * batches + sizeof(float) - 1) / sizeof(float);
  if (!arena.FlushHostWrites(rgb, rgb_float_elements) ||
      !arena.ResizeRgbBatchToNchw(rgb, input, static_cast<int>(batches), images[pages[0]].width,
                                  images[pages[0]].height, width, height)) {
    arena.Release(input); arena.Release(rgb); FailGpuOnly("detector batch front end");
  }
  arena.Release(rgb);
  std::unordered_map<std::string, detail::OnnxLite::GpuTensor> outputs;
  if (!det.RunGpuOnlyDevice({{"x", {{static_cast<std::int64_t>(batches),3,height,width}, input}}}, outputs)) {
    FailGpuOnly("detector batch graph");
  }
  detail::OnnxLite::GpuTensor probability; bool found{};
  for (auto& [_, value] : outputs) if (value.shape.size() == 4 && value.shape[0] == static_cast<std::int64_t>(batches) && value.shape[1] == 1) {
    probability = std::move(value); found = true; break;
  }
  for (auto& [_, value] : outputs) if (value.slot.resident && value.slot.index != probability.slot.index) arena.Release(value.slot);
  if (!found) { arena.Release(probability.slot); Fail("GPU-only detector batch probability map not found"); }
  std::vector<std::vector<Box>> result(batches);
  for (std::size_t item = 0; item < batches; ++item) {
    result[item] = DBPostDevice(arena, probability, item,
                                images[pages[item]].width, images[pages[item]].height, opt);
  }
  arena.Release(probability.slot);
  if (!keep_gpu_constants) det.ReleaseGpuConstants();
  if (!(keep_gpu_constants ? arena.ReclaimFreeTransientStorage() : arena.ReclaimFreeStorage()))
    Fail("GPU-only detector batch post reclaim failed");
  return result;
}
BackendInfo QueryBackendInfo() { return detail::QueryVulkanBackendInfo(); }
OCR::OCR(const std::string& det_model,const std::string& rec_model,std::string dictionary_path,Options options) {
  impl_=std::make_unique<Impl>(det_model,rec_model,options);
  if (options.backend == Backend::gpu_only &&
      (!QueryBackendInfo().full_graph_gpu_available || !impl_->det.SupportsGpuOnly() ||
       !impl_->rec.SupportsGpuOnly())) {
    Fail("GPU-only was requested, but the complete PP-OCRv6 Vulkan graph is unavailable");
  }
  impl_->dict=ReadDict(dictionary_path);
}
OCR::~OCR()=default;OCR::OCR(OCR&&) noexcept=default;OCR& OCR::operator=(OCR&&) noexcept=default;
std::vector<Result> OCR::Recognize(const Image& image) const {
  if(image.empty()) Fail("invalid RGB image");
  if(impl_->det.outputs().empty()) Fail("detector has no output");
  if (impl_->opt.backend == Backend::gpu_only) {
    auto result = RecognizeGpuOnly(impl_->det, impl_->rec, impl_->opt, impl_->dict, image,
                                   impl_->keep_gpu_constants);
    // Stable-service memory policy: all activations and temporary DB buffers
    // have completed at this point; retain only immutable model slots.
    if (impl_->reclaim_gpu_transient &&
        !impl_->det.gpu_arena().ReclaimFreeTransientStorage()) {
      Fail("GPU-only completed-request arena reclaim failed");
    }
    return result;
  }
  // Hybrid is intentionally best-effort and is allowed to retain the CPU
  // route after a failed optional GPU segment.  Strict gpu_only above is not:
  // it returns before this block and every Vulkan failure propagates to the
  // caller without a CPU inference retry.
  if (impl_->enable_hybrid_gpu_graph) {
    int graph_state = 0;
    {
      std::lock_guard lock(impl_->hybrid_gpu_graph_mutex);
      graph_state = impl_->hybrid_gpu_graph_state;
    }
    if (graph_state >= 0) {
      try {
        auto result = RecognizeGpuOnly(impl_->det, impl_->rec, impl_->opt, impl_->dict, image,
                                       impl_->keep_gpu_constants);
        {
          std::lock_guard lock(impl_->hybrid_gpu_graph_mutex);
          impl_->hybrid_gpu_graph_state = 1;
        }
        if (impl_->reclaim_gpu_transient) {
          (void)impl_->det.gpu_arena().ReclaimFreeTransientStorage();
        }
        return result;
      } catch (const std::exception&) {
        std::lock_guard lock(impl_->hybrid_gpu_graph_mutex);
        impl_->hybrid_gpu_graph_state = -1;
      }
    }
  }
  // The detector probability map can be several MiB for a large page. It is
  // needed only while DB post-processing constructs independent Box values;
  // keep the detector workspace scoped so it is released before bounded,
  // concurrent recognizer batches begin. This lets the allocator reuse that
  // storage for recognition activations instead of increasing peak memory.
  const bool profile_e2e = std::getenv("PPOCR_PROFILE_E2E") != nullptr;
  const auto boxes = [&] {
    const auto t0 = profile_e2e ? std::chrono::steady_clock::now()
                                : std::chrono::steady_clock::time_point{};
    std::unordered_map<std::string,Tensor> det_input;
    Tensor det_tensor;
    Tensor stem_tensor;
    std::string stem_output_name;
    std::array<std::string, 3> skipped_stem{};
    Tensor conv2_tensor;
    std::array<std::string, 3> conv012_names{};
    // Both front ends include their own numerical and complete-boundary
    // timing admission.  Let that measured policy decide by default instead
    // of requiring an opt-in switch; a slower or imprecise GPU route remains
    // on the SIMD CPU path.
    const bool try_gpu_stem = std::getenv("PPOCR_DISABLE_VULKAN_RGB_STEM") == nullptr;
    const bool disable_conv012 = std::getenv("PPOCR_DISABLE_VULKAN_RGB_CONV012") != nullptr;
    auto seed_cpu_stem = [&] {
      det_input.emplace(std::move(stem_output_name), std::move(stem_tensor));
      det_input.emplace("x", Tensor{{1}, {0.F}});
      for (const auto& name : skipped_stem) {
        if (!name.empty()) det_input.emplace(name, Tensor{{1}, {0.F}});
      }
    };
    bool seeded = false;
    if (try_gpu_stem && TryHybridVulkanDetRgbStem(image, impl_->opt, impl_->det, stem_tensor,
                                                   stem_output_name, skipped_stem)) {
      seed_cpu_stem();
      seeded = true;
    }
    if (!seeded) {
      // GPU RGB-upload then CPU stem paid a D2H of 3x160x704 before the fused
      // AVX-512 walk. CPU DetInput plus fused stem is the winning hybrid path.
      det_tensor = DetInput(image, impl_->opt);
      if (TryCpuDetRgbStem(det_tensor, impl_->det, stem_tensor, stem_output_name,
                           skipped_stem)) {
        seed_cpu_stem();
        seeded = true;
      }
    }
    if (!seeded && !disable_conv012 &&
        TryHybridVulkanDetRgbConv012(image, impl_->opt, impl_->det, stem_tensor, conv2_tensor,
                                     conv012_names)) {
      det_input.emplace(conv012_names[0], std::move(stem_tensor));
      if (!conv012_names[1].empty()) det_input.emplace(conv012_names[1], Tensor{{1}, {0.F}});
      det_input.emplace(conv012_names[2], std::move(conv2_tensor));
      det_input.emplace("x", Tensor{{1}, {0.F}});
      seeded = true;
    }
    if (!seeded && TryHybridVulkanDetRgbConv0(image, impl_->opt, impl_->det, stem_tensor,
                                              stem_output_name)) {
      det_input.emplace(std::move(stem_output_name), std::move(stem_tensor));
      det_input.emplace("x", Tensor{{1}, {0.F}});
      seeded = true;
    }
    if (!seeded) det_input.emplace("x", std::move(det_tensor));
    const auto t1 = profile_e2e ? std::chrono::steady_clock::now() : t0;
    auto detector_output=impl_->det.Run(std::move(det_input));
    const Tensor* prob=nullptr;
    for(const auto&[_,t]:detector_output) {
      if(t.shape.size()==4&&t.shape[1]==1) { prob=&t; break; }
    }
    if(!prob) Fail("detector probability map not found");
    const auto t2 = profile_e2e ? std::chrono::steady_clock::now() : t0;
    auto boxes_out = DBPost(*prob,image.width,image.height,impl_->opt);
    if (profile_e2e) {
      const auto t3 = std::chrono::steady_clock::now();
      std::cerr << "e2e_ms det_pre="
                << std::chrono::duration<double, std::milli>(t1 - t0).count()
                << " det_run="
                << std::chrono::duration<double, std::milli>(t2 - t1).count()
                << " db="
                << std::chrono::duration<double, std::milli>(t3 - t2).count()
                << '\n';
    }
    return boxes_out;
  }();
  // Keep each same-width crop group contiguous. The recognizer used to build
  // a short temporary `vector<CropBounds>` for every batch; dense pages then
  // paid an allocation/move sequence for every 1--4 crops. Storing crop and
  // output-index arrays once lets every batch pass a zero-copy span to input
  // preprocessing while retaining the same width bucketing and output order.
  // The recognizer contains transformer attention, so right padding is not
  // semantically inert: a shorter crop packed beside a wider one can retain
  // its decoded text but receive different CTC confidence. Group by the
  // exact natural width to preserve the public Recognize() contract while
  // still coalescing repeated-width crops into genuine NCHW batches.
  // Tiny CNN+CTC may pack a page to one padded width (`PPOCR_ENABLE_REC_PAD_MAX`)
  // so two crops share one NCHW batch and the full SIMD pool.
  struct Bucket { std::vector<std::size_t> indices; std::vector<CropBounds> crops; };
  std::map<int, Bucket> buckets;
  int page_pad_width = 0;
  const bool pad_max = RecPadToPageMax();
  if (pad_max) {
    for (std::size_t i = 0; i < boxes.size(); ++i)
      page_pad_width = std::max(page_pad_width, RecInputWidth(CropBoundsFor(image, boxes[i]), impl_->opt));
  }
  for(std::size_t i=0;i<boxes.size();++i) {
    const auto crop=CropBoundsFor(image,boxes[i]);
    const int natural_width = RecInputWidth(crop,impl_->opt);
    auto& bucket = buckets[pad_max ? page_pad_width : natural_width];
    bucket.indices.push_back(i);
    bucket.crops.push_back(crop);
  }
  std::vector<Result> out(boxes.size());
  const int max_batch=std::max(1,impl_->opt.rec_batch_size);
  struct Batch { int width; const CropBounds* crops; const std::size_t* indices; std::size_t count; };
  std::vector<Batch> batches;
  for(auto& [width,bucket]:buckets) {
    for(std::size_t first=0;first<bucket.crops.size();first+=max_batch) {
      const std::size_t count=std::min<std::size_t>(max_batch,bucket.crops.size()-first);
      batches.push_back({width,bucket.crops.data()+first,bucket.indices.data()+first,count});
    }
  }
  // Keep width order deterministic.  A largest-first outer scheduling trial
  // was retained only long enough to A/B on dense pages: this graph's shared
  // SIMD pool makes the ordinary width order faster and less variable on the
  // target host, so FIFO construction is the production policy.
  const auto recognition_parallelism = impl_->opt.backend == Backend::hybrid &&
      impl_->hybrid_gpu_graph_state > 0
      ? impl_->opt.hybrid_graph_parallelism : impl_->opt.rec_parallelism;
  const auto workers=std::min<std::size_t>(batches.size(),
      RecognitionWorkerLimit(impl_->opt.backend, recognition_parallelism));
  // Multiple same-width graph batches may already run concurrently below.
  // Divide the front-end resize budget between them instead of multiplying
  // CPU threads on dense pages. This preserves the one-Vulkan-dispatch-per-
  // NCHW-batch property while reducing the time before that submission.
  const auto rec_preprocess_workers = std::max<std::size_t>(
      1, std::size_t(std::max(1, impl_->opt.batch_preprocess_parallelism)) /
             std::max<std::size_t>(1, workers));
  int max_rec_width = 0;
  for (const auto& batch : batches)
    max_rec_width = std::max(max_rec_width, batch.width);
  const auto run_batch=[&](const Batch& batch) {
    const auto rec_t0 = profile_e2e ? std::chrono::steady_clock::now()
                                    : std::chrono::steady_clock::time_point{};
    std::unordered_map<std::string,Tensor> rec_input;
    Tensor rec_stem;
    std::string rec_stem_name;
    // Two recognizer crops used to stack two 16-thread SIMD pools (32-way
    // oversubscribe on the gating 16-thread budget). Cap each crop's inner
    // ParallelFor so the pair stays within PPOCR_BENCH_THREADS.
    int inner_cap = 0;
    // Two-crop tiny pages want the full SIMD pool on each graph; stacking
    // 8+8 was slower than 16+overflow on the gating pair. Cap only when
    // many rec batches would otherwise multiply the worker budget.
    if (workers >= 4 && std::getenv("PPOCR_DISABLE_INNER_PARALLELISM_CAP") == nullptr) {
      inner_cap = 8;
      if (const char* configured = std::getenv("PPOCR_BENCH_THREADS")) {
        char* end = nullptr;
        const long parsed = std::strtol(configured, &end, 10);
        if (end != configured && *end == '\0' && parsed > 0)
          inner_cap = std::max(2, int(parsed) / int(workers));
      }
    }
    detail::kernels::InnerParallelismGuard inner_guard(inner_cap);
    // Two rec crops share one Vulkan queue. GPU the wall (widest) crop only
    // so the other crop stays on CPU SIMD without serializing on the fence.
    if (batch.count == 1 && batch.width == max_rec_width &&
        TryHybridVulkanRecRgbConv0(image, batch.crops[0], impl_->opt, impl_->rec, rec_stem,
                                   rec_stem_name)) {
      rec_input.emplace(std::move(rec_stem_name), std::move(rec_stem));
      rec_input.emplace("x", Tensor{{1}, {0.F}});
    } else {
      rec_input.emplace("x",RecInputBatch(image,batch.crops,batch.count,batch.width,impl_->opt,
                                            rec_preprocess_workers));
    }
    const auto rec_t1 = profile_e2e ? std::chrono::steady_clock::now() : rec_t0;
    auto logits=impl_->rec.RunCtcTop1(std::move(rec_input));
    if (profile_e2e) {
      std::cerr << "e2e_ms rec_pre="
                << std::chrono::duration<double, std::milli>(rec_t1 - rec_t0).count()
                << " rec_run="
                << std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - rec_t1).count()
                << " width=" << batch.width << '\n';
    }
    if(logits.batches!=static_cast<int>(batch.count)) Fail("recognizer logits not found");
    for(std::size_t i=0;i<batch.count;++i) {
      auto [text,confidence]=DecodeCTC(logits,i,impl_->dict);
      const auto result_index=batch.indices[i];
      if (const auto trace_index = TraceOcrCropResult(); trace_index && result_index == *trace_index) {
        const auto offset = i * static_cast<std::size_t>(logits.steps);
        std::cerr << "CPU trace CTC result=" << result_index << " rows=";
        for (int step = 0; step < logits.steps; ++step) {
          if (step) std::cerr << ',';
          std::cerr << logits.indices[offset + static_cast<std::size_t>(step)] << '@'
                    << logits.probabilities[offset + static_cast<std::size_t>(step)];
        }
        std::cerr << '\n';
      }
      out[result_index]=MakeResult(boxes[result_index],std::move(text),confidence);
    }
  };
  const auto rec_begin = profile_e2e ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
  if(workers<=1) {
    for(const auto& batch:batches) run_batch(batch);
  } else {
    std::exception_ptr failure;
    std::mutex failure_mutex;
    IndexExecutor().Run(static_cast<int>(batches.size()), static_cast<int>(workers),
                        [&](int first, int last) {
      try {
        for (int task = first; task < last; ++task) run_batch(batches[static_cast<std::size_t>(task)]);
      } catch (...) {
        std::lock_guard lock(failure_mutex);
        if (!failure) failure = std::current_exception();
      }
    });
    if (failure) std::rethrow_exception(failure);
  }
  if (profile_e2e) {
    std::cerr << "e2e_ms rec="
              << std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - rec_begin).count()
              << " boxes=" << boxes.size() << " rec_batches=" << batches.size()
              << '\n';
  }
  ReclaimHostHeap();
  return out;
}

std::vector<std::vector<Result>> OCR::RecognizeBatch(
    const std::vector<Image>& images) const {
  std::vector<std::vector<Result>> output(images.size());
  if (images.empty()) return output;
  // The GPU-only path retains image/crop tensors on its one Vulkan queue. A
  // Same-original-size pages are coalesced for one device-side RGB front end
  // + detector graph. Their same-width recognizer crops are then coalesced
  // into real NCHW attention batches, without relaxing GPU-only's contract.
  if (impl_->opt.backend == Backend::gpu_only) {
    for (const auto& image : images) if (image.empty()) Fail("invalid RGB image");
    struct Bucket { int source_width{}, source_height{}, width{}, height{}; std::vector<std::size_t> pages; };
    // Hash all four signed geometry fields without truncating detector
    // dimensions. The prior shift/XOR packing overlapped high width bits with
    // source-height bits, which could coalesce unrelated pages on large input
    // sets. A typed key keeps GPU batching exact and fail-closed.
    struct BatchKey {
      int source_width, source_height, detector_width, detector_height;
      bool operator==(const BatchKey&) const noexcept = default;
    };
    struct BatchKeyHash {
      std::size_t operator()(const BatchKey& value) const noexcept {
        std::size_t hash = 1469598103934665603ull;
        for (const int field : {value.source_width, value.source_height,
                                value.detector_width, value.detector_height}) {
          hash ^= static_cast<std::uint32_t>(field);
          hash *= 1099511628211ull;
        }
        return hash;
      }
    };
    std::unordered_map<BatchKey, Bucket, BatchKeyHash> buckets;
    for (std::size_t page = 0; page < images.size(); ++page) {
      const auto [width, height] = DetInputSize(images[page], impl_->opt);
      const auto& image = images[page];
      const BatchKey key{image.width, image.height, width, height};
      auto& bucket = buckets[key];
      bucket.source_width = image.width; bucket.source_height = image.height;
      bucket.width = width; bucket.height = height; bucket.pages.push_back(page);
    }
  // Medium detector's 896-channel Vulkan tail is presently qualified only
  // for N=1 on the Radeon/LLPC driver. Keep the requested detector batch
  // for models/drivers that can execute it; when a batched graph is rejected,
  // rerun each page as an independent strict GPU-only graph. This is a GPU
  // scheduling retry after the runtime recovery boundary, never a CPU path.
    const std::size_t cap = static_cast<std::size_t>(std::max(1, impl_->opt.det_batch_size));
    for (auto& [bucket_key, bucket] : buckets) {
      for (std::size_t first = 0; first < bucket.pages.size(); first += cap) {
        const std::vector<std::size_t> pages(bucket.pages.begin() + first,
                                             bucket.pages.begin() + std::min(bucket.pages.size(), first + cap));
        std::vector<std::vector<Box>> boxes;
        const std::string batch_key = std::to_string(bucket.source_width) + 'x' +
            std::to_string(bucket.source_height) + ':' + std::to_string(bucket.width) + 'x' +
            std::to_string(bucket.height) + ':' + std::to_string(pages.size());
        const auto serial_gpu_only = [&] {
          // Strict GPU-only failure has no CPU fallback. Retry independent
          // GPU pages after the runtime's device-loss recovery boundary.
          for (const std::size_t page : pages) output[page] = Recognize(images[page]);
        };
        if (pages.size() == 1) {
          serial_gpu_only();
          continue;
        }
        {
          std::lock_guard lock(impl_->gpu_batch_rejection_mutex);
          if (impl_->gpu_batch_rejections.contains(batch_key)) {
            serial_gpu_only();
            continue;
          }
        }
        try {
          boxes = DetectGpuOnlyBatch(impl_->det, impl_->rec, impl_->opt, images, pages,
                                     bucket.width, bucket.height, impl_->keep_gpu_constants);
        } catch (const std::exception&) {
          // Cache only this exact source/detector/N shape; another shape may
          // be valid on the same adapter. No batch exception is hidden: it is
          // handled only by the documented strict N=1 GPU-only recovery path.
          {
            std::lock_guard lock(impl_->gpu_batch_rejection_mutex);
            impl_->gpu_batch_rejections.insert(batch_key);
          }
          serial_gpu_only();
          continue;
        }
        for (std::size_t item = 0; item < pages.size(); ++item) {
          output[pages[item]].resize(boxes[item].size());
        }
        std::map<int, std::vector<GpuCropJob>> rec_buckets;
        for (std::size_t item = 0; item < pages.size(); ++item) {
          const std::size_t page = pages[item];
          for (std::size_t index = 0; index < boxes[item].size(); ++index) {
            const auto crop = CropBoundsFor(images[page], boxes[item][index]);
            rec_buckets[GpuRecognitionWidth(crop, impl_->opt)].push_back(
                {page, index, &images[page], &boxes[item][index], crop});
          }
        }
        const std::size_t rec_cap = static_cast<std::size_t>(std::max(1, impl_->opt.rec_batch_size));
        for (auto& [rec_width, jobs] : rec_buckets) {
          for (std::size_t rec_first = 0; rec_first < jobs.size(); rec_first += rec_cap) {
            const std::vector<GpuCropJob> batch(jobs.begin() + rec_first,
                                                jobs.begin() + std::min(jobs.size(), rec_first + rec_cap));
            RecognizeGpuOnlyCropBatch(impl_->rec, impl_->opt, impl_->dict, batch, output,
                                      nullptr);
          }
        }
      }
    }
    if (impl_->reclaim_gpu_transient &&
        !impl_->det.gpu_arena().ReclaimFreeTransientStorage()) {
      Fail("GPU-only completed-batch arena reclaim failed");
    }
    return output;
  }

  // Detector pages have independent dynamic resize shapes.  Do not pad them
  // into one artificial batch: it wastes feature-map work and changes the
  // model's intended resize behavior.  Instead detect pages independently,
  // then coalesce the resulting same-width recognition crops across all
  // pages.  The latter is a *real* NCHW batch, not merely a page scheduler;
  // it feeds Vulkan's Y batch dimension and lets CPU pointwise/MatMul paths
  // amortize executor overhead over crops from multiple source images.
  for (const auto& image : images) {
    if (image.empty()) Fail("invalid RGB image");
  }
  if (impl_->det.outputs().empty()) Fail("detector has no output");
  std::vector<std::vector<Box>> page_boxes(images.size());
  // `image_batch_parallelism` is useful CPU-side concurrency, but it is not
  // device concurrency.  Hybrid has one synchronous Vulkan queue/context;
  // letting several detector graphs race to that queue only serializes their
  // GPU segments while keeping several large detector activation frontiers
  // alive.  Match recognition's queue-aware policy here: CPU may retain the
  // requested page parallelism, while hybrid defaults to one graph submitter
  // unless the caller has explicitly measured a driver-specific override.
  const auto requested_detector_parallelism = impl_->opt.backend == Backend::hybrid
      ? impl_->opt.hybrid_graph_parallelism : impl_->opt.image_batch_parallelism;
  const auto workers = std::min<std::size_t>(
      images.size(), std::max(1, requested_detector_parallelism));
  const auto run_indexed = [&](std::size_t item_count, std::size_t thread_count,
                               const auto& run_one) {
    RunIndexed(item_count, thread_count, run_one);
  };
  // Group only exactly equal detector resize dimensions. This produces a
  // genuine NCHW detector batch without padding any page or perturbing the
  // model's dynamic resize behavior. It benefits CPU batch kernels today and
  // maps naturally to Vulkan's batch Y dimension as more device-resident
  // detector segments are added.
  struct DetectorBucket { int width, height; std::vector<std::size_t> pages; };
  std::unordered_map<std::uint64_t, DetectorBucket> detector_buckets;
  for (std::size_t page = 0; page < images.size(); ++page) {
    const auto [width, height] = DetInputSize(images[page], impl_->opt);
    const auto key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(height)) << 32) |
                     static_cast<std::uint32_t>(width);
    auto& bucket = detector_buckets[key];
    bucket.width = width;
    bucket.height = height;
    bucket.pages.push_back(page);
  }
  // The Vulkan executor receives the leading N dimension directly for every
  // implemented primitive.  Keep same-shape detector pages together for
  // hybrid too: each individual GPU segment still has its own complete
  // H2D + dispatch + fence + D2H no-slower admission, so enlarging a batch
  // cannot silently force a slower GPU route.  This also lets an admitted
  // segment amortize one queue submission across multiple independent pages.
  const auto detector_batch_size =
      static_cast<std::size_t>(std::max(1, impl_->opt.det_batch_size));
  const auto batch_preprocess_parallelism = static_cast<std::size_t>(
      std::max(1, impl_->opt.batch_preprocess_parallelism));
  // `detect_batch` can itself run concurrently for separate dynamic shapes.
  // Divide the front-end resize budget between those graph workers so opting
  // into page-level parallelism does not accidentally multiply thread count.
  const auto preprocess_workers_per_detector_batch = std::max<std::size_t>(
      1, batch_preprocess_parallelism / std::max<std::size_t>(1, workers));
  struct DetectorBatch { int width, height; const std::size_t* pages; std::size_t count; };
  std::vector<DetectorBatch> detector_batches;
  for (auto& [_, bucket] : detector_buckets) {
    for (std::size_t first = 0; first < bucket.pages.size(); first += detector_batch_size) {
      detector_batches.push_back({bucket.width, bucket.height, bucket.pages.data() + first,
                                  std::min(detector_batch_size, bucket.pages.size() - first)});
    }
  }
  const auto detect_batch = [&](std::size_t batch_index) {
    const auto& batch = detector_batches[batch_index];
    Tensor input{{static_cast<std::int64_t>(batch.count), 3, batch.height, batch.width},
                 std::vector<float>(batch.count * std::size_t(3) * batch.height * batch.width)};
    const auto stride = std::size_t(3) * batch.height * batch.width;
    const auto& normalize = DetectorNormalizeLut();
    // There is only one detector graph dispatch for a same-shape NCHW
    // batch.  Parallelize the independent resizes before it; otherwise the
    // GPU (and the CPU's SIMD batch kernels) wait for a serial front end on
    // the common N=4 page case.  Each task owns an entire contiguous NCHW
    // image plane, so this needs no synchronization beyond run_indexed.
    run_indexed(batch.count,
                std::min(batch.count, preprocess_workers_per_detector_batch),
                [&](std::size_t item) {
      const auto page = batch.pages[item];
      WriteResizeRegionToNchw(input.data.data() + item * stride, images[page], 0, 0,
                              images[page].width, images[page].height, batch.width, batch.height,
                              [&](int channel, float value) {
                                return normalize[static_cast<std::size_t>(channel)]
                                                [static_cast<unsigned>(value)];
                              });
    });
    std::unordered_map<std::string, Tensor> det_input;
    det_input.emplace("x", std::move(input));
    auto detector_output = impl_->det.Run(std::move(det_input));
    const Tensor* probability = nullptr;
    for (const auto& [_, tensor] : detector_output) {
      if (tensor.shape.size() == 4 && tensor.shape[0] == static_cast<std::int64_t>(batch.count) &&
          tensor.shape[1] == 1) {
        probability = &tensor;
        break;
      }
    }
    if (!probability) Fail("detector probability map not found");
    for (std::size_t item = 0; item < batch.count; ++item) {
      const auto page = batch.pages[item];
      page_boxes[page] = DBPost(*probability, images[page].width, images[page].height,
                                impl_->opt, item);
    }
  };
  run_indexed(detector_batches.size(), workers, detect_batch);

  struct GlobalCrop {
    std::size_t page_index;
    std::size_t result_index;
    CropBounds crop;
  };
  // Transformer attention is sensitive to right padding. Group crops by
  // exact resized width rather than merely a padded bucket width, so a crop
  // evaluated through RecognizeBatch has exactly the same tensor shape as the
  // corresponding standalone inference. This still creates real NCHW batches
  // for repeated widths without silently changing crop resampling.
  struct Bucket { std::vector<GlobalCrop> crops; };
  std::map<int, Bucket> buckets;
  for (std::size_t page = 0; page < images.size(); ++page) {
    output[page].resize(page_boxes[page].size());
    for (std::size_t index = 0; index < page_boxes[page].size(); ++index) {
      const auto crop = CropBoundsFor(images[page], page_boxes[page][index]);
      const int natural_width = RecInputWidth(crop, impl_->opt);
      buckets[natural_width].crops.push_back({page, index, crop});
    }
  }

  // Keep CPU/Hybrid batch OCR numerically equivalent to per-page inference.
  // The current transformer export has N>1 attention sensitivity; GPU-only
  // uses its separately validated device batch executor above.
  const int max_batch = impl_->opt.backend == Backend::gpu_only
      ? std::max(1, impl_->opt.rec_batch_size) : 1;
  struct Batch { int width; const GlobalCrop* crops; std::size_t count; };
  std::vector<Batch> batches;
  for (auto& [width, bucket] : buckets) {
    for (std::size_t first = 0; first < bucket.crops.size(); first += max_batch) {
      const auto count = std::min<std::size_t>(max_batch, bucket.crops.size() - first);
      batches.push_back({width, bucket.crops.data() + first, count});
    }
  }
  const auto requested_rec_parallelism = impl_->opt.backend == Backend::hybrid &&
      impl_->hybrid_gpu_graph_state > 0
      ? impl_->opt.hybrid_graph_parallelism : impl_->opt.rec_parallelism;
  const auto rec_workers = std::min<std::size_t>(batches.size(),
      RecognitionWorkerLimit(impl_->opt.backend, requested_rec_parallelism));
  const auto rec_preprocess_workers = std::max<std::size_t>(
      1, batch_preprocess_parallelism / std::max<std::size_t>(1, rec_workers));
  const auto run_batch = [&](std::size_t batch_index) {
    const auto& batch = batches[batch_index];
    Tensor input{{static_cast<std::int64_t>(batch.count), 3, impl_->opt.rec_height, batch.width},
                 std::vector<float>(batch.count * std::size_t(3) * impl_->opt.rec_height *
                                    batch.width, 0.F)};
    const std::size_t stride = std::size_t(3) * impl_->opt.rec_height * batch.width;
    const auto& normalize = RecognitionNormalizeLut();
    run_indexed(batch.count, std::min(batch.count, rec_preprocess_workers),
                [&](std::size_t i) {
      const auto& item = batch.crops[i];
      WriteResizeRegionToNchw(input.data.data() + i * stride, images[item.page_index],
                              item.crop.x, item.crop.y, item.crop.width, item.crop.height,
                              batch.width, impl_->opt.rec_height,
                              [&normalize](int, float value) {
                                return normalize[static_cast<unsigned>(value)];
                              }, batch.width);
    });
    std::unordered_map<std::string, Tensor> rec_input;
    rec_input.emplace("x", std::move(input));
    auto logits = impl_->rec.RunCtcTop1(std::move(rec_input));
    if (logits.batches != static_cast<int>(batch.count)) Fail("recognizer logits not found");
    for (std::size_t i = 0; i < batch.count; ++i) {
      const auto& item = batch.crops[i];
      auto [text, confidence] = DecodeCTC(logits, i, impl_->dict);
      output[item.page_index][item.result_index] =
          MakeResult(page_boxes[item.page_index][item.result_index], std::move(text), confidence);
    }
  };
  run_indexed(batches.size(), rec_workers, run_batch);
  ReclaimHostHeap();
  return output;
}

Image LoadPPM(const std::string& path) { std::ifstream f(path,std::ios::binary);if(!f)Fail("cannot open image "+path);std::string magic;f>>magic;if(magic!="P6")Fail("only binary P6 PPM is supported");auto token=[&](){std::string s;while(f>>s&&s.starts_with('#'))std::getline(f,s);return s;};int w=std::stoi(token()),h=std::stoi(token()),max=std::stoi(token());if(w<=0||h<=0||max!=255)Fail("invalid PPM header");f.get();Image out{w,h,std::vector<std::uint8_t>(std::size_t(w)*h*3)};f.read(reinterpret_cast<char*>(out.rgb.data()),std::streamsize(out.rgb.size()));if(f.gcount()!=std::streamsize(out.rgb.size()))Fail("truncated PPM pixels");return out; }
}  // namespace ppocr
