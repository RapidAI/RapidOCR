#include "ppocr/ppocr.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int ParsePositive(const char* text, const char* name) {
  try {
    const int value = std::stoi(text);
    if (value > 0) return value;
  } catch (...) {
  }
  throw std::runtime_error(std::string(name) + " must be a positive integer");
}

void OverridePositiveFromEnv(const char* name, int& value) {
  if (const char* text = std::getenv(name)) value = ParsePositive(text, name);
}

void OverrideBackendFromEnv(ppocr::Backend& backend) {
  const char* text = std::getenv("PPOCR_BACKEND");
  if (!text) return;
  const std::string value{text};
  if (value == "cpu") { backend = ppocr::Backend::cpu_only; return; }
  if (value == "hybrid") { backend = ppocr::Backend::hybrid; return; }
  if (value == "gpu") { backend = ppocr::Backend::gpu_only; return; }
  throw std::runtime_error("PPOCR_BACKEND must be cpu, hybrid, or gpu");
}

const char* BackendName(ppocr::Backend backend) {
  switch (backend) {
    case ppocr::Backend::cpu_only: return "cpu";
    case ppocr::Backend::gpu_only: return "gpu";
    case ppocr::Backend::hybrid: return "hybrid";
  }
  return "unknown";
}

double Percentile95(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const auto index = std::min(samples.size() - 1,
                              static_cast<std::size_t>(std::ceil(samples.size() * .95)) - 1);
  return samples[index];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage: ppocr_bench DET.onnx REC.onnx dict.txt image.ppm WARMUP RUNS\n";
    return 2;
  }
  try {
    const int warmup = ParsePositive(argv[5], "WARMUP");
    const int runs = ParsePositive(argv[6], "RUNS");
    if (const char* affinity = std::getenv("PPOCR_CPU_AFFINITY")) {
      std::cerr << "PPOCR_CPU_AFFINITY is requested but unsupported by this portable benchmark: " << affinity << '\n';
    }
    const auto image = ppocr::LoadPPM(argv[4]);
    const auto load_begin = std::chrono::steady_clock::now();
    ppocr::Options options;
    // The public library defaults are conservative. These optional benchmark
    // switches let deployment owners tune dense-page recognition for their
    // actual CPU without changing model accuracy or the library ABI.
    OverridePositiveFromEnv("PPOCR_REC_BATCH_SIZE", options.rec_batch_size);
    OverridePositiveFromEnv("PPOCR_REC_WIDTH_BUCKET", options.rec_batch_width_bucket);
    OverridePositiveFromEnv("PPOCR_REC_PARALLELISM", options.rec_parallelism);
    OverridePositiveFromEnv("PPOCR_HYBRID_GRAPH_PARALLELISM", options.hybrid_graph_parallelism);
    OverridePositiveFromEnv("PPOCR_DET_BATCH_SIZE", options.det_batch_size);
    OverridePositiveFromEnv("PPOCR_BATCH_PREPROCESS_PARALLELISM",
                            options.batch_preprocess_parallelism);
    OverridePositiveFromEnv("PPOCR_IMAGE_BATCH_PARALLELISM", options.image_batch_parallelism);
    OverrideBackendFromEnv(options.backend);
    ppocr::OCR ocr(argv[1], argv[2], argv[3], options);
    const double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - load_begin).count();
    for (int i = 0; i < warmup; ++i) (void)ocr.Recognize(image);

    std::vector<double> samples;
    samples.reserve(runs);
    std::size_t boxes{};
    for (int i = 0; i < runs; ++i) {
      const auto begin = std::chrono::steady_clock::now();
      boxes = ocr.Recognize(image).size();
      samples.push_back(std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - begin).count());
    }
    const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    const double mean = sum / samples.size();
    auto ordered = samples;
    std::sort(ordered.begin(), ordered.end());
    const double median = ordered[(ordered.size() - 1) / 2];
    std::cout << std::fixed << std::setprecision(3)
              << "{\n"
              << "  \"engine\": \"ppocr_cpp\",\n"
              << "  \"backend\": \"" << BackendName(options.backend) << "\",\n"
              << "  \"load_ms\": " << load_ms << ",\n"
              << "  \"warmup\": " << warmup << ",\n"
              << "  \"runs\": " << runs << ",\n"
              << "  \"boxes\": " << boxes << ",\n"
              << "  \"mean_ms\": " << mean << ",\n"
              << "  \"median_ms\": " << median << ",\n"
              << "  \"p95_ms\": " << Percentile95(samples) << ",\n"
              << "  \"min_ms\": " << ordered.front() << ",\n"
              << "  \"max_ms\": " << ordered.back() << ",\n"
              << "  \"throughput_fps\": " << 1000.0 / mean << "\n"
              << "}\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
