#include "onnx_lite.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

namespace {

int ReadPositive(const char* text, const char* name) {
  try {
    const int value = std::stoi(text);
    if (value > 0) return value;
  } catch (...) {}
  throw std::runtime_error(std::string("invalid ") + name);
}

}  // namespace

// Strict graph regression tool.  It intentionally constructs OnnxLite with
// gpu_only rather than going through OCR: the public OCR switch remains off
// until image preprocessing, DB postprocess and CTC decode are device-only as
// well.  This verifies that every neural-network node executes on Vulkan and
// compares its final device download against the portable FP32 graph.
int main(int argc, char** argv) {
  if (argc != 6 && argc != 7) {
    std::cerr << "usage: ppocr_gpu_graph_smoke MODEL.onnx N C H W [RUNS]\n";
    return 2;
  }
  try {
    const int n = ReadPositive(argv[2], "N");
    const int c = ReadPositive(argv[3], "C");
    const int h = ReadPositive(argv[4], "H");
    const int w = ReadPositive(argv[5], "W");
    const int runs = argc == 7 ? ReadPositive(argv[6], "RUNS") : 1;
    float input_scale = 1.F;
    if (const char* text = std::getenv("PPOCR_GPU_GRAPH_INPUT_SCALE")) {
      input_scale = std::strtof(text, nullptr);
      if (!std::isfinite(input_scale) || input_scale <= 0.F) throw std::runtime_error("invalid PPOCR_GPU_GRAPH_INPUT_SCALE");
    }
    ppocr::detail::Tensor input{{n, c, h, w},
        std::vector<float>(static_cast<std::size_t>(n) * c * h * w)};
    if (const char* bin = std::getenv("PPOCR_GPU_GRAPH_INPUT_BIN")) {
      std::ifstream in(bin, std::ios::binary);
      if (!in) throw std::runtime_error("cannot open PPOCR_GPU_GRAPH_INPUT_BIN");
      in.read(reinterpret_cast<char*>(input.data.data()),
              static_cast<std::streamsize>(input.data.size() * sizeof(float)));
      if (static_cast<std::size_t>(in.gcount()) != input.data.size() * sizeof(float))
        throw std::runtime_error("PPOCR_GPU_GRAPH_INPUT_BIN size mismatch");
    } else {
      for (std::size_t index = 0; index < input.data.size(); ++index) {
        // Deterministic nontrivial values cover sign changes and gate regions.
        input.data[index] = (float((index * 37u) % 251u) / 125.F - 1.F) * input_scale;
      }
    }
    ppocr::detail::OnnxLite cpu(argv[1], ppocr::Backend::cpu_only);
    ppocr::detail::OnnxLite gpu(argv[1], ppocr::Backend::gpu_only);
    const auto input_name = "x";  // PP-OCRv6 exported graph input.
    // The first pass verifies graph parity.  Optional additional passes make
    // this the reproducible whole-neural-network CPU/Vulkan boundary
    // benchmark: model parsing is outside the measurement, while each GPU
    // pass deliberately includes its current graph upload/dispatch/download
    // boundary.
    const auto cpu_start = std::chrono::steady_clock::now();
    auto expected = cpu.Run({{input_name, input}});
    double cpu_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - cpu_start).count();
    // The first strict-device call creates and uploads immutable model slots.
    // It is checked for parity but excluded from the steady-state timing so
    // the benchmark reports actual inference cost, just as a deployed model
    // instance would be warmed before serving requests.
    auto actual = gpu.Run({{input_name, input}});
    const bool stopped_at_checkpoint = std::getenv("PPOCR_GPU_ONLY_STOP_AT_CHECKPOINT") != nullptr;
    if (stopped_at_checkpoint) {
      if (!actual.empty()) throw std::runtime_error("checkpoint run unexpectedly returned graph outputs");
      std::cout << "strict Vulkan graph checkpoint completed model=" << argv[1]
                << " input_scale=" << input_scale << '\n';
      return 0;
    }
    if (expected.size() != actual.size()) throw std::runtime_error("output count mismatch");
    float worst_error{};
    std::string worst_output;
    for (const auto& [name, expected_tensor] : expected) {
      const auto found = actual.find(name);
      if (found == actual.end() || found->second.shape != expected_tensor.shape ||
          found->second.data.size() != expected_tensor.data.size()) {
        throw std::runtime_error("output shape mismatch for " + name);
      }
      for (std::size_t index = 0; index < expected_tensor.data.size(); ++index) {
        const float error = std::abs(found->second.data[index] - expected_tensor.data[index]);
        if (error > worst_error) { worst_error = error; worst_output = name; }
        // The device Sigmoid/GELU paths use bounded FP32 approximations;
        // use an absolute floor for saturated values where a relative-only
        // comparison would be meaningless, while keeping a tight relative
        // tolerance for ordinary activations.
        const float tolerance = std::max(1e-3F, 3e-4F * std::abs(expected_tensor.data[index]));
        if (error > tolerance) {
          std::cerr << "GPU graph mismatch output=" << name << " index=" << index
                    << " cpu=" << expected_tensor.data[index]
                    << " gpu=" << found->second.data[index] << " error=" << error << '\n';
          return 1;
        }
      }
    }
    double gpu_ms{};
    for (int run = 0; run < runs; ++run) {
      const auto gpu_repeat_start = std::chrono::steady_clock::now();
      (void)gpu.Run({{input_name, input}});
      gpu_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - gpu_repeat_start).count();
    }
    for (int run = 1; run < runs; ++run) {
      const auto cpu_repeat_start = std::chrono::steady_clock::now();
      (void)cpu.Run({{input_name, input}});
      cpu_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - cpu_repeat_start).count();
    }
    float cpu_min = expected.begin()->second.data.empty()
        ? 0.F : expected.begin()->second.data.front();
    float cpu_max = cpu_min, gpu_min = cpu_min, gpu_max = cpu_min;
    const auto& cpu_out = expected.begin()->second;
    const auto& gpu_out = actual.at(expected.begin()->first);
    for (float value : cpu_out.data) {
      cpu_min = std::min(cpu_min, value);
      cpu_max = std::max(cpu_max, value);
    }
    for (float value : gpu_out.data) {
      gpu_min = std::min(gpu_min, value);
      gpu_max = std::max(gpu_max, value);
    }
    std::cout << "strict Vulkan graph smoke passed model=" << argv[1]
              << " output=" << worst_output << " max_abs_error=" << worst_error
              << " cpu_range=" << cpu_min << ',' << cpu_max
              << " gpu_range=" << gpu_min << ',' << gpu_max
              << " cpu_mean_ms=" << cpu_ms / runs << " gpu_mean_ms=" << gpu_ms / runs
              << " runs=" << runs << " input_scale=" << input_scale << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "GPU graph smoke failed: " << error.what() << '\n';
    return 1;
  }
}
