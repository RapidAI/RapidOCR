#include "ppocr/ppocr.hpp"

#include <iostream>
#include <string>

namespace {

ppocr::Backend ParseBackend(const std::string& value) {
  if (value == "cpu") return ppocr::Backend::cpu_only;
  if (value == "gpu") return ppocr::Backend::gpu_only;
  if (value == "hybrid") return ppocr::Backend::hybrid;
  throw std::runtime_error("backend must be cpu, gpu, or hybrid");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--backend-info") {
    const auto info = ppocr::QueryBackendInfo();
    std::cout << "vulkan_loader=" << info.vulkan_loader_available
              << " vulkan_compute=" << info.vulkan_compute_available
              << " full_graph_gpu=" << info.full_graph_gpu_available
              << " device=" << info.device_name << '\n';
    return 0;
  }
  if (argc != 5 && argc != 7) {
    std::cerr << "usage: ppocr_demo DET.onnx REC.onnx dict.txt image.ppm [--backend cpu|gpu|hybrid]\n";
    return 2;
  }
  try {
    ppocr::Options options;
    if (argc == 7) {
      if (std::string(argv[5]) != "--backend") throw std::runtime_error("expected --backend");
      options.backend = ParseBackend(argv[6]);
    }
    ppocr::OCR ocr(argv[1], argv[2], argv[3], options);
    for (const auto& r : ocr.Recognize(ppocr::LoadPPM(argv[4]))) {
      std::cout << r.text << "\t" << r.confidence << "\t"
                << r.bbox[0] << ',' << r.bbox[1] << ','
                << r.bbox[2] << ',' << r.bbox[3] << '\n';
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
