#include "ppocr/ppocr.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <fstream>
#include <unistd.h>
#endif

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

void ApplyBackendFromEnv(ppocr::Backend& backend) {
  const char* text = std::getenv("PPOCR_BACKEND");
  if (!text || std::string{text} == "cpu") { backend = ppocr::Backend::cpu_only; return; }
  if (std::string{text} == "hybrid") { backend = ppocr::Backend::hybrid; return; }
  if (std::string{text} == "gpu") { backend = ppocr::Backend::gpu_only; return; }
  throw std::runtime_error("PPOCR_BACKEND must be cpu, hybrid, or gpu");
}

struct ProcessMemory {
  std::uint64_t private_bytes{};
  std::uint64_t working_set_bytes{};
};

ProcessMemory ProcessMemoryUsage() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS_EX counters{};
  counters.cb = sizeof(counters);
  if (!GetProcessMemoryInfo(GetCurrentProcess(),
                            reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                            sizeof(counters))) {
    throw std::runtime_error("GetProcessMemoryInfo failed");
  }
  return {static_cast<std::uint64_t>(counters.PrivateUsage),
          static_cast<std::uint64_t>(counters.WorkingSetSize)};
#elif defined(__linux__)
  std::ifstream status("/proc/self/status");
  std::string key;
  while (status >> key) {
    if (key == "VmRSS:") {
      std::uint64_t kib{};
      status >> kib;
      return {kib * 1024, kib * 1024};
    }
    status.ignore(4096, '\n');
  }
  return {};
#else
  return {};
#endif
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 7) {
    std::cerr << "usage: ppocr_mem_soak DET.onnx REC.onnx dict.txt CYCLES image1.ppm [image2.ppm ...]\n"
                 "  Images are processed one at a time; JSON samples memory after each image.\n";
    return 2;
  }
  try {
    const int cycles = ParsePositive(argv[4], "CYCLES");
    // Keep only paths here.  Retaining every decoded RGB image would make a
    // large-page ladder look like an activation leak even after that page has
    // completed.  Each image therefore has a deliberately narrow lifetime:
    // load -> Recognize -> destroy -> sample process memory.
    std::vector<std::string> image_paths;
    image_paths.reserve(argc - 5);
    for (int i = 5; i < argc; ++i) image_paths.emplace_back(argv[i]);
    ppocr::Options options;
    ApplyBackendFromEnv(options.backend);
    OverridePositiveFromEnv("PPOCR_REC_BATCH_SIZE", options.rec_batch_size);
    OverridePositiveFromEnv("PPOCR_REC_WIDTH_BUCKET", options.rec_batch_width_bucket);
    OverridePositiveFromEnv("PPOCR_REC_PARALLELISM", options.rec_parallelism);
    OverridePositiveFromEnv("PPOCR_HYBRID_GRAPH_PARALLELISM", options.hybrid_graph_parallelism);
    OverridePositiveFromEnv("PPOCR_DET_BATCH_SIZE", options.det_batch_size);
    OverridePositiveFromEnv("PPOCR_BATCH_PREPROCESS_PARALLELISM",
                            options.batch_preprocess_parallelism);
    OverridePositiveFromEnv("PPOCR_IMAGE_BATCH_PARALLELISM", options.image_batch_parallelism);
    ppocr::OCR ocr(argv[1], argv[2], argv[3], options);

    // Establish model, worker, and every size bucket before the baseline.
    // Warm-up samples are intentionally retained as diagnostic detail, but
    // their order runs low-to-high and therefore cannot be compared with a
    // final baseline that is sampled only after the whole warm-up.
    std::vector<std::uint64_t> warmup_samples;
    warmup_samples.reserve(image_paths.size());
    for (const auto& path : image_paths) {
      {
        auto image = ppocr::LoadPPM(path);
        (void)ocr.Recognize(image);
      }
      warmup_samples.push_back(ProcessMemoryUsage().private_bytes);
    }
    const auto baseline = ProcessMemoryUsage();
    std::cout << "{\"backend\":\""
              << (options.backend == ppocr::Backend::hybrid ? "hybrid" :
                  options.backend == ppocr::Backend::gpu_only ? "gpu" : "cpu")
              << "\",\"baseline_private_bytes\":" << baseline.private_bytes
              << ",\"baseline_working_set_bytes\":" << baseline.working_set_bytes
              << ",\"warmup_samples\":[";
    for (std::size_t index = 0; index < image_paths.size(); ++index) {
      if (index) std::cout << ',';
      std::cout << "{\"index\":" << index
                << ",\"path\":\"" << std::filesystem::path(image_paths[index]).filename().string()
                << "\",\"private_bytes\":" << warmup_samples[index]
                << ",\"growth_from_baseline\":"
                << static_cast<std::int64_t>(warmup_samples[index]) -
                       static_cast<std::int64_t>(baseline.private_bytes)
                << '}';
    }
    std::cout << "],\"cycles\":[";
    for (int cycle = 0; cycle < cycles; ++cycle) {
      if (cycle) std::cout << ',';
      std::cout << "{\"cycle\":" << (cycle + 1) << ",\"samples\":[";
      for (std::size_t index = 0; index < image_paths.size(); ++index) {
        {
          auto image = ppocr::LoadPPM(image_paths[index]);
          (void)ocr.Recognize(image);
        }
        const auto memory = ProcessMemoryUsage();
        if (index) std::cout << ',';
        std::cout << "{\"index\":" << index
                  << ",\"path\":\"" << std::filesystem::path(image_paths[index]).filename().string()
                  << "\",\"private_bytes\":" << memory.private_bytes
                  << ",\"working_set_bytes\":" << memory.working_set_bytes
                  << ",\"growth_from_baseline\":"
                  << static_cast<std::int64_t>(memory.private_bytes) -
                         static_cast<std::int64_t>(baseline.private_bytes)
                  << '}';
      }
      std::cout << "]}";
    }
    std::cout << "]}\n";
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
