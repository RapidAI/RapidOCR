#include "ppocr/ppocr.hpp"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void CheckPage(const std::vector<ppocr::Result>& cpu,
               const std::vector<ppocr::Result>& gpu,
               const std::string& image_name, float& worst_confidence_error) {
  if (cpu.size() != gpu.size()) {
    throw std::runtime_error(image_name + ": result count CPU=" + std::to_string(cpu.size()) +
                             " GPU=" + std::to_string(gpu.size()));
  }
  for (std::size_t index = 0; index < cpu.size(); ++index) {
    const auto& expected = cpu[index];
    const auto& actual = gpu[index];
    if (expected.text != actual.text || expected.bbox != actual.bbox || expected.box != actual.box) {
      const auto describe = [](const ppocr::Result& value) {
        std::ostringstream out;
        out << "text='" << value.text << "' bbox=" << value.bbox[0] << ','
            << value.bbox[1] << ',' << value.bbox[2] << ',' << value.bbox[3]
            << " box=";
        for (std::size_t point = 0; point < value.box.size(); ++point) {
          if (point) out << ';';
          out << value.box[point][0] << ',' << value.box[point][1];
        }
        out << " confidence=" << value.confidence;
        return out.str();
      };
      throw std::runtime_error(image_name + ": result " + std::to_string(index) +
                               " differs; CPU{" + describe(expected) + "} GPU{" +
                               describe(actual) + '}');
    }
    const float error = std::abs(expected.confidence - actual.confidence);
    worst_confidence_error = std::max(worst_confidence_error, error);
    // Neural FP32 implementations can differ slightly in reductions/erf,
    // while greedy CTC text and detector geometry must remain exact.
    // GPU-only retains the crop's exact aspect-ratio width. Minor FP32
    // reduction differences are tolerated, while decoded text and geometry
    // must remain exact.
    if (error > 5e-2F) {
      throw std::runtime_error(image_name + ": result " + std::to_string(index) +
                               " confidence error=" + std::to_string(error));
    }
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
    std::size_t result_count{};
    float worst_confidence_error{};
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
      const auto expected = cpu.Recognize(image);
      for (int repeat = 0; repeat < repeats; ++repeat) {
        const auto actual = gpu.Recognize(image);
        CheckPage(expected, actual, argv[index], worst_confidence_error);
        if (repeat == 0) result_count += actual.size();
      }
    }
    std::cout << std::fixed << std::setprecision(7)
              << "GPU-only OCR smoke passed images=" << argc - 4
              << " repeats=" << repeats
              << " results=" << result_count
              << " max_confidence_abs_error=" << worst_confidence_error << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "GPU-only OCR smoke failed: " << error.what() << '\n';
    return 1;
  }
}
