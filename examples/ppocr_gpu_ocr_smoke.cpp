#include "ppocr/ppocr.h"
#include "ppocr/ppocr.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

float BoxIou(const ppocr::Result& left, const ppocr::Result& right) {
  const float ax0 = static_cast<float>(left.bbox[0]);
  const float ay0 = static_cast<float>(left.bbox[1]);
  const float ax1 = ax0 + static_cast<float>(left.bbox[2]);
  const float ay1 = ay0 + static_cast<float>(left.bbox[3]);
  const float bx0 = static_cast<float>(right.bbox[0]);
  const float by0 = static_cast<float>(right.bbox[1]);
  const float bx1 = bx0 + static_cast<float>(right.bbox[2]);
  const float by1 = by0 + static_cast<float>(right.bbox[3]);
  const float ix0 = std::max(ax0, bx0);
  const float iy0 = std::max(ay0, by0);
  const float ix1 = std::min(ax1, bx1);
  const float iy1 = std::min(ay1, by1);
  const float iw = std::max(0.F, ix1 - ix0);
  const float ih = std::max(0.F, iy1 - iy0);
  const float inter = iw * ih;
  const float area = static_cast<float>(left.bbox[2]) * left.bbox[3] +
                     static_cast<float>(right.bbox[2]) * right.bbox[3] - inter;
  return area > 0.F ? inter / area : 0.F;
}

void CheckPage(const std::vector<ppocr::Result>& cpu,
               const std::vector<ppocr::Result>& gpu,
               const std::string& image_name, float& worst_confidence_error,
               float& worst_iou) {
  if (cpu.size() != gpu.size()) {
    throw std::runtime_error(image_name + ": result count CPU=" + std::to_string(cpu.size()) +
                             " GPU=" + std::to_string(gpu.size()));
  }
  int mismatches = 0;
  for (std::size_t index = 0; index < cpu.size(); ++index) {
    const auto& expected = cpu[index];
    const auto& actual = gpu[index];
    const float iou = BoxIou(expected, actual);
    worst_iou = std::min(worst_iou, iou);
    const auto describe = [](const ppocr::Result& value) {
      std::ostringstream out;
      out << "text='" << value.text << "' bbox=" << value.bbox[0] << ','
          << value.bbox[1] << ',' << value.bbox[2] << ',' << value.bbox[3]
          << " confidence=" << value.confidence;
      return out.str();
    };
    const float error = std::abs(expected.confidence - actual.confidence);
    worst_confidence_error = std::max(worst_confidence_error, error);
    // Detector quads can move by a pixel when the DB mask is reduced on the
    // device. Text has to match, and the axis-aligned boxes have to cover
    // the same line.
    if (expected.text != actual.text || iou < 0.95F || error > 5e-2F) {
      ++mismatches;
      std::cerr << image_name << " result " << index << " iou=" << iou
                << " conf_abs=" << error
                << " CPU{" << describe(expected) << "} GPU{" << describe(actual) << "}\n";
    }
  }
  if (mismatches) {
    throw std::runtime_error(image_name + ": " + std::to_string(mismatches) + " mismatched lines");
  }
}

}  // namespace

// Public GPU-only end-to-end regression.  This intentionally exercises OCR,
// not OnnxLite: RGB preprocessing, detector graph, DB boxes, recognizer graph
// and CTC decode are compared against the portable CPU result on real pages.
int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: ppocr_gpu_ocr_smoke DET.onnx REC.onnx dict.txt image.ppm [image.ppm ...]\n";
    return 2;
  }
  try {
    ppocr::Options cpu_options;
    cpu_options.backend = ppocr::Backend::cpu_only;
    ppocr::Options gpu_options = cpu_options;
    gpu_options.backend = ppocr::Backend::gpu_only;
    if (const char* value = std::getenv("PPOCR_VULKAN_DEVICE_INDEX")) {
      const int index = std::atoi(value);
      if (index >= 0) gpu_options.vulkan_device_index = index;
    }
    std::size_t result_count{};
    float worst_confidence_error{};
    float worst_iou = 1.F;
    const int repeats = [] {
      if (const char* value = std::getenv("PPOCR_GPU_OCR_SMOKE_REPEATS")) {
        const int parsed = std::atoi(value);
        if (parsed > 0) return parsed;
      }
      return 1;
    }();
    for (int index = 4; index < argc; ++index) {
      ppocr::OCR cpu(argv[1], argv[2], argv[3], cpu_options);
      ppocr::OCR gpu(argv[1], argv[2], argv[3], gpu_options);
      const auto image = ppocr::LoadPPM(argv[index]);
      const auto cpu_begin = std::chrono::steady_clock::now();
      const auto expected = cpu.Recognize(image);
      const double cpu_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - cpu_begin).count();
      double gpu_ms = 0;
      for (int repeat = 0; repeat < repeats; ++repeat) {
        const auto gpu_begin = std::chrono::steady_clock::now();
        const auto actual = gpu.Recognize(image);
        gpu_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - gpu_begin).count();
        CheckPage(expected, actual, argv[index], worst_confidence_error, worst_iou);
        if (repeat == 0) result_count += actual.size();
      }
      std::cout << std::fixed << std::setprecision(3)
                << "timing " << argv[index]
                << " cpu_ms=" << cpu_ms
                << " vulkan_ms=" << (gpu_ms / repeats)
                << " lines=" << expected.size() << '\n';
    }
    const int validation_errors = ppocr_vulkan_validation_error_count();
    std::cout << std::fixed << std::setprecision(7)
              << "GPU-only OCR smoke passed images=" << argc - 4
              << " repeats=" << repeats
              << " results=" << result_count
              << " max_confidence_abs_error=" << worst_confidence_error
              << " min_box_iou=" << worst_iou
              << " validation_errors=" << validation_errors << '\n';
    if (validation_errors != 0) {
      throw std::runtime_error("Vulkan validation reported " +
                               std::to_string(validation_errors) + " errors");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "GPU-only OCR smoke failed: " << error.what() << '\n';
    return 1;
  }
}
