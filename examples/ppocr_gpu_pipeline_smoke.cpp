#include "onnx_lite.hpp"
#include "ppocr/ppocr.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

ppocr::detail::Tensor CpuDetectorInput(const ppocr::Image& image, int width, int height) {
  ppocr::detail::Tensor output{{1, 3, height, width},
      std::vector<float>(std::size_t(3) * width * height)};
  constexpr float scale[3]{1.F / (255.F * .229F), 1.F / (255.F * .224F), 1.F / (255.F * .225F)};
  constexpr float offset[3]{-.485F / .229F, -.456F / .224F, -.406F / .225F};
  for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
    const float fx = (float(x) + .5F) * image.width / width - .5F;
    const float fy = (float(y) + .5F) * image.height / height - .5F;
    const int xf = int(std::floor(fx)), yf = int(std::floor(fy));
    const int x0 = std::clamp(xf, 0, image.width - 1), x1 = std::min(x0 + 1, image.width - 1);
    const int y0 = std::clamp(yf, 0, image.height - 1), y1 = std::min(y0 + 1, image.height - 1);
    const float dx = fx - xf, dy = fy - yf;
    for (int channel = 0; channel < 3; ++channel) {
      const int component = 2 - channel;
      const auto pixel = [&](int py, int px) {
        return float(image.rgb[(std::size_t(py) * image.width + px) * 3 + component]);
      };
      const float upper = pixel(y0, x0) * (1.F - dx) + pixel(y0, x1) * dx;
      const float lower = pixel(y1, x0) * (1.F - dx) + pixel(y1, x1) * dx;
      const float sample = float(std::clamp(std::lround(upper * (1.F - dy) + lower * dy), 0L, 255L));
      output.data[(std::size_t(channel) * height + y) * width + x] = sample * scale[channel] + offset[channel];
    }
  }
  return output;
}

void CheckFinite(const ppocr::detail::Tensor& tensor, const char* source) {
  for (std::size_t index = 0; index < tensor.data.size(); ++index) {
    if (!std::isfinite(tensor.data[index])) {
      throw std::runtime_error(std::string(source) + " produced non-finite value at " + std::to_string(index));
    }
  }
}

float MaximumAbsoluteError(const ppocr::detail::Tensor& actual,
                           const ppocr::detail::Tensor& expected,
                           const char* source) {
  if (actual.shape != expected.shape || actual.data.size() != expected.data.size()) {
    throw std::runtime_error(std::string(source) + " output shape mismatch");
  }
  float maximum{};
  for (std::size_t index = 0; index < actual.data.size(); ++index) {
    maximum = std::max(maximum, std::abs(actual.data[index] - expected.data[index]));
  }
  return maximum;
}

}  // namespace

// Regression for the actual GPU-only detector boundary. It proves that a
// Vulkan RGB resize/normalization output can feed the strict graph directly,
// then compares the device result with the portable graph's exact image input.
int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: ppocr_gpu_pipeline_smoke DETECTOR.onnx IMAGE.ppm\n";
    return 2;
  }
  try {
    const auto image = ppocr::LoadPPM(argv[2]);
    if (image.empty()) throw std::runtime_error("invalid image");
    const int width = std::max(32, (image.width / 32) * 32);
    const int height = std::max(32, (image.height / 32) * 32);

    ppocr::detail::OnnxLite cpu(argv[1], ppocr::Backend::cpu_only);
    ppocr::detail::OnnxLite gpu_host(argv[1], ppocr::Backend::gpu_only);
    ppocr::detail::OnnxLite gpu_slot(argv[1], ppocr::Backend::gpu_only);
    const auto cpu_input = CpuDetectorInput(image, width, height);
    const auto expected = cpu.Run({{"x", cpu_input}});
    if (expected.size() != 1) throw std::runtime_error("unexpected detector output count");
    ppocr::detail::OnnxLite gpu_cpu_input(argv[1], ppocr::Backend::gpu_only);
    const auto cpu_input_gpu_output = gpu_cpu_input.Run({{"x", cpu_input}});
    if (cpu_input_gpu_output.size() != 1) throw std::runtime_error("CPU-input Vulkan graph output count mismatch");
    CheckFinite(cpu_input_gpu_output.begin()->second, "Vulkan graph with CPU preprocessing");
    const float cpu_input_graph_error = MaximumAbsoluteError(cpu_input_gpu_output.begin()->second,
        expected.begin()->second, "Vulkan graph with CPU preprocessing");

    auto& arena = gpu_slot.gpu_arena();
    auto rgb = arena.Acquire(image.rgb.size(), "pipeline-rgb");
    auto input = arena.Acquire(std::size_t(3) * height * width, "pipeline-input");
    if (!rgb.resident || !input.resident || !arena.UploadRgb8(rgb, image.rgb.data(), image.rgb.size()) ||
        !arena.ResizeRgbToNchw(rgb, input, image.width, image.height, 0, 0, image.width, image.height,
                               width, height, 0, 0, 0, 0, 0, 0)) {
      throw std::runtime_error("Vulkan detector input failed");
    }
    ppocr::detail::Tensor preprocessed{{1, 3, height, width},
        std::vector<float>(input.live_elements)};
    if (!arena.Download(preprocessed.data.data(), input, preprocessed.data.size())) {
      arena.Release(input); arena.Release(rgb);
      throw std::runtime_error("Vulkan detector input download failed");
    }
    CheckFinite(preprocessed, "Vulkan preprocessing");
    float preprocess_error{};
    std::size_t preprocess_mismatch_count{};
    std::size_t first_preprocess_mismatch = preprocessed.data.size();
    std::size_t maximum_preprocess_error_index{};
    for (std::size_t index = 0; index < preprocessed.data.size(); ++index) {
      const float error = std::abs(preprocessed.data[index] - cpu_input.data[index]);
      if (error > preprocess_error) {
        preprocess_error = error;
        maximum_preprocess_error_index = index;
      }
      if (error != 0.F) {
        ++preprocess_mismatch_count;
        first_preprocess_mismatch = std::min(first_preprocess_mismatch, index);
      }
    }
    const auto [input_minimum, input_maximum] = std::minmax_element(
        preprocessed.data.begin(), preprocessed.data.end());
    std::cout << "GPU preprocessing range=" << *input_minimum << ',' << *input_maximum
              << " max_abs_error=" << preprocess_error << '\n';
    if (first_preprocess_mismatch != preprocessed.data.size()) {
      const std::size_t plane = std::size_t(width) * height;
      const std::size_t channel = first_preprocess_mismatch / plane;
      const std::size_t spatial = first_preprocess_mismatch % plane;
      std::cout << "GPU preprocessing mismatches=" << preprocess_mismatch_count
                << " first=(c=" << channel << ",y=" << spatial / width
                << ",x=" << spatial % width << ") cpu="
                << cpu_input.data[first_preprocess_mismatch] << " gpu="
                << preprocessed.data[first_preprocess_mismatch] << '\n';
      const std::size_t maximum_plane = std::size_t(width) * height;
      const std::size_t maximum_channel = maximum_preprocess_error_index / maximum_plane;
      const std::size_t maximum_spatial = maximum_preprocess_error_index % maximum_plane;
      std::cout << "GPU preprocessing maximum=(c=" << maximum_channel << ",y="
                << maximum_spatial / width << ",x=" << maximum_spatial % width
                << ") cpu=" << cpu_input.data[maximum_preprocess_error_index]
                << " gpu=" << preprocessed.data[maximum_preprocess_error_index] << '\n';
    }
    // This isolates numerical preprocessing from slot handoff. If this strict
    // host-boundary run fails, the image values themselves are invalid; if it
    // passes while the live-slot run below fails, the device-slot path is the
    // only remaining suspect.
    const auto host_boundary = gpu_host.Run({{"x", preprocessed}});
    if (host_boundary.size() != 1) throw std::runtime_error("host-boundary graph output count mismatch");
    CheckFinite(host_boundary.begin()->second, "Vulkan graph after input download/upload");
    std::unordered_map<std::string, ppocr::detail::OnnxLite::GpuTensor> output_slots;
    if (!gpu_slot.RunGpuOnlyDevice({{"x", {{1, 3, height, width}, input}}}, output_slots) ||
        output_slots.size() != 1) {
      arena.Release(rgb);
      throw std::runtime_error("strict Vulkan detector graph failed");
    }
    auto output = std::move(output_slots.begin()->second);
    ppocr::detail::Tensor actual{output.shape, std::vector<float>(output.slot.live_elements)};
    const bool downloaded = arena.Download(actual.data.data(), output.slot, actual.data.size());
    arena.Release(output.slot);
    arena.Release(rgb);
    if (!downloaded) throw std::runtime_error("Vulkan detector output download failed");
    CheckFinite(actual, "Vulkan live-slot pipeline");

    const auto& expected_tensor = expected.begin()->second;
    if (actual.shape != expected_tensor.shape || actual.data.size() != expected_tensor.data.size()) {
      throw std::runtime_error("detector output shape mismatch");
    }
    float maximum_error{};
    for (std::size_t index = 0; index < actual.data.size(); ++index) {
      maximum_error = std::max(maximum_error, std::abs(actual.data[index] - expected_tensor.data[index]));
    }
    std::cout << "GPU front-end to detector graph smoke passed shape=1x3x" << height << 'x' << width
              << " output_elements=" << actual.data.size()
              << " cpu_input_graph_max_abs_error=" << cpu_input_graph_error
              << " frontend_graph_max_abs_error=" << MaximumAbsoluteError(host_boundary.begin()->second,
                  expected_tensor, "Vulkan graph after input download/upload")
              << " live_slot_max_abs_error=" << maximum_error << '\n';
  } catch (const std::exception& error) {
    std::cerr << "GPU pipeline smoke failed: " << error.what() << '\n';
    return 1;
  }
}
