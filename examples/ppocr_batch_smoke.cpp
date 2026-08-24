#include "ppocr/ppocr.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int Positive(const char* text, const char* name) {
  try {
    const int value = std::stoi(text);
    if (value > 0) return value;
  } catch (...) {
  }
  throw std::runtime_error(std::string(name) + " must be positive");
}

void OverridePositiveFromEnv(const char* name, int& value) {
  if (const char* text = std::getenv(name)) value = Positive(text, name);
}

double Mean(const std::vector<double>& samples) {
  double sum{};
  for (const double value : samples) sum += value;
  return sum / static_cast<double>(samples.size());
}

double Median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[(samples.size() - 1) / 2];
}

void ApplyBackendFromEnv(ppocr::Backend& backend) {
  const char* text = std::getenv("PPOCR_BACKEND");
  if (!text || std::string{text} == "cpu") {
    backend = ppocr::Backend::cpu_only;
    return;
  }
  if (std::string{text} == "hybrid") {
    backend = ppocr::Backend::hybrid;
    return;
  }
  if (std::string{text} == "gpu") {
    backend = ppocr::Backend::gpu_only;
    return;
  }
  throw std::runtime_error("PPOCR_BACKEND must be cpu, hybrid, or gpu");
}

bool SameResults(const std::vector<ppocr::Result>& left,
                 const std::vector<ppocr::Result>& right) {
  if (left.size() != right.size()) return false;
  for (std::size_t i = 0; i < left.size(); ++i) {
    if (left[i].text != right[i].text || left[i].bbox != right[i].bbox ||
        // Batched detector convolutions legitimately change FP32 reduction
        // order versus serial N=1.  Text and geometry remain exact; accept a
        // small confidence-only drift while still rejecting semantic changes.
        // This matches the separately qualified FP32 GPU-only confidence
        // envelope; CPU runs normally stay well inside it.
        left[i].box != right[i].box || std::abs(left[i].confidence - right[i].confidence) > 5e-2F) {
      return false;
    }
  }
  return true;
}

std::string DescribeDifference(const std::vector<ppocr::Result>& serial,
                               const std::vector<ppocr::Result>& batched) {
  if (serial.size() != batched.size()) {
    return "result-count serial=" + std::to_string(serial.size()) +
           " batch=" + std::to_string(batched.size());
  }
  for (std::size_t index = 0; index < serial.size(); ++index) {
    const auto& lhs = serial[index];
    const auto& rhs = batched[index];
    if (lhs.text != rhs.text || lhs.bbox != rhs.bbox || lhs.box != rhs.box ||
        std::abs(lhs.confidence - rhs.confidence) > 5e-2F) {
      std::ostringstream out;
      out << "result=" << index << " serial{text='" << lhs.text << "', confidence="
          << std::setprecision(9) << lhs.confidence << ", bbox=" << lhs.bbox[0] << ','
          << lhs.bbox[1] << ',' << lhs.bbox[2] << ',' << lhs.bbox[3]
          << "} batch{text='" << rhs.text << "', confidence=" << rhs.confidence
          << ", bbox=" << rhs.bbox[0] << ',' << rhs.bbox[1] << ',' << rhs.bbox[2] << ','
          << rhs.bbox[3] << "}";
      return out.str();
    }
  }
  return {};
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) {
    std::cerr << "usage: ppocr_batch_smoke DET.onnx REC.onnx dict.txt PARALLELISM image1.ppm [image2.ppm ...]\n";
    return 2;
  }
  try {
    ppocr::Options options;
    ApplyBackendFromEnv(options.backend);
    options.image_batch_parallelism = Positive(argv[4], "PARALLELISM");
    if (const char* text = std::getenv("PPOCR_REC_BATCH_SIZE")) {
      options.rec_batch_size = Positive(text, "PPOCR_REC_BATCH_SIZE");
    }
    if (const char* text = std::getenv("PPOCR_REC_PARALLELISM")) {
      options.rec_parallelism = Positive(text, "PPOCR_REC_PARALLELISM");
    }
    if (const char* text = std::getenv("PPOCR_HYBRID_GRAPH_PARALLELISM")) {
      options.hybrid_graph_parallelism =
          Positive(text, "PPOCR_HYBRID_GRAPH_PARALLELISM");
    }
    if (const char* text = std::getenv("PPOCR_DET_BATCH_SIZE")) {
      options.det_batch_size = Positive(text, "PPOCR_DET_BATCH_SIZE");
    }
    if (const char* text = std::getenv("PPOCR_BATCH_PREPROCESS_PARALLELISM")) {
      options.batch_preprocess_parallelism =
          Positive(text, "PPOCR_BATCH_PREPROCESS_PARALLELISM");
    }
    if (const char* text = std::getenv("PPOCR_REC_WIDTH_BUCKET")) {
      options.rec_batch_width_bucket = Positive(text, "PPOCR_REC_WIDTH_BUCKET");
    }
    // Keep this executable a correctness smoke by default, while allowing it
    // to be a stable batch-throughput probe without a second benchmark
    // program.  The timed loops are deliberately after one serial and one
    // batched warm-up so Vulkan's shape admission/validation and persistent
    // CPU worker creation are not charged to the reported steady state.
    int warmup = 1;
    int runs = 1;
    OverridePositiveFromEnv("PPOCR_BATCH_WARMUP", warmup);
    OverridePositiveFromEnv("PPOCR_BATCH_RUNS", runs);
    std::vector<ppocr::Image> images;
    images.reserve(static_cast<std::size_t>(argc - 5));
    for (int i = 5; i < argc; ++i) images.push_back(ppocr::LoadPPM(argv[i]));

    ppocr::OCR ocr(argv[1], argv[2], argv[3], options);
    std::vector<std::vector<ppocr::Result>> serial;
    serial.reserve(images.size());
    // Warm the model and persistent CPU worker set before measuring either
    // scheduler. This keeps the reported comparison about page batching,
    // rather than model parsing or one-time thread startup.
    for (const auto& image : images) (void)ocr.Recognize(image);
    // Warm the actual cross-page NCHW shapes as well. Hybrid admission is
    // deliberately shape-specific and includes validation, so charging that
    // one-time safety probe to the first timed batch would measure model
    // adaptation rather than steady-state batch throughput.
    (void)ocr.RecognizeBatch(images);
    for (int iteration = 0; iteration < warmup; ++iteration) {
      for (const auto& image : images) (void)ocr.Recognize(image);
      (void)ocr.RecognizeBatch(images);
    }
    std::vector<double> serial_samples, batch_samples;
    serial_samples.reserve(runs);
    batch_samples.reserve(runs);
    std::vector<std::vector<ppocr::Result>> batched;
    for (int iteration = 0; iteration < runs; ++iteration) {
      serial.clear();
      const auto serial_begin = std::chrono::steady_clock::now();
      for (const auto& image : images) serial.push_back(ocr.Recognize(image));
      serial_samples.push_back(std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - serial_begin).count());

      const auto batch_begin = std::chrono::steady_clock::now();
      batched = ocr.RecognizeBatch(images);
      batch_samples.push_back(std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - batch_begin).count());
    }
    if (batched.size() != serial.size()) throw std::runtime_error("batch output size mismatch");
    std::size_t result_count{};
    for (std::size_t i = 0; i < serial.size(); ++i) {
      if (!SameResults(serial[i], batched[i])) {
        throw std::runtime_error("batch result differs from serial inference at image " +
                                 std::to_string(i) + ": " +
                                 DescribeDifference(serial[i], batched[i]));
      }
      result_count += batched[i].size();
    }
    std::cout << std::fixed << std::setprecision(3)
              << "batch_smoke images=" << images.size()
              << " results=" << result_count
              << " backend=" << (options.backend == ppocr::Backend::hybrid ? "hybrid" :
                                    options.backend == ppocr::Backend::gpu_only ? "gpu" : "cpu")
              << " parallelism=" << options.image_batch_parallelism
              << " preprocess_parallelism=" << options.batch_preprocess_parallelism
              << " det_batch=" << options.det_batch_size
              << " rec_batch=" << options.rec_batch_size
              << " rec_width_bucket=" << options.rec_batch_width_bucket
              << " hybrid_graph_parallelism=" << options.hybrid_graph_parallelism
              << " warmup=" << warmup
              << " runs=" << runs
              << " serial_mean_ms=" << Mean(serial_samples)
              << " serial_median_ms=" << Median(serial_samples)
              << " batch_mean_ms=" << Mean(batch_samples)
              << " batch_median_ms=" << Median(batch_samples)
              << " serial_samples_ms=";
    for (std::size_t index = 0; index < serial_samples.size(); ++index) {
      if (index) std::cout << ',';
      std::cout << serial_samples[index];
    }
    std::cout << " batch_samples_ms=";
    for (std::size_t index = 0; index < batch_samples.size(); ++index) {
      if (index) std::cout << ',';
      std::cout << batch_samples[index];
    }
    std::cout
              << " speedup=" << Mean(serial_samples) / Mean(batch_samples) << '\n';
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
