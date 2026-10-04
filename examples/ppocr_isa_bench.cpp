#include "kernels.hpp"
#include "ppocr/ppocr.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

template <class Fn>
double TimeMs(Fn&& fn, int warmup, int runs) {
  for (int i = 0; i < warmup; ++i) fn();
  const auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < runs; ++i) fn();
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - begin)
             .count() /
         runs;
}

}  // namespace

int main(int argc, char** argv) {
  const auto cpu = ppocr::QueryCpuInfo();
  std::cout << "{\n"
            << "  \"active_isa\": \"" << cpu.active_isa << "\",\n"
            << "  \"avx2\": " << (cpu.avx2 ? "true" : "false") << ",\n"
            << "  \"avx512\": " << (cpu.avx512 ? "true" : "false") << ",\n"
            << "  \"neon\": " << (cpu.neon ? "true" : "false") << ",\n"
            << "  \"threads\": " << cpu.threads << ",\n";

  using ppocr::detail::kernels::Gemm;
  using ppocr::detail::kernels::GemmAccumulate;
  using ppocr::detail::kernels::WriteIdentityRgbToNchw;

  constexpr int rows = 32;
  constexpr int cols = 256;
  constexpr int depth = 256;
  std::vector<float> a(static_cast<std::size_t>(rows) * depth);
  std::vector<float> b(static_cast<std::size_t>(depth) * cols);
  std::vector<float> bias(cols, 0.1F);
  std::vector<float> c(static_cast<std::size_t>(rows) * cols, 0.F);
  for (std::size_t i = 0; i < a.size(); ++i) a[i] = float(int(i % 13) - 6) * 0.05F;
  for (std::size_t i = 0; i < b.size(); ++i) b[i] = float(int(i % 11) - 5) * 0.04F;
  const double gemm_ms = TimeMs([&] {
    Gemm(c.data(), a.data(), b.data(), bias.data(), rows, cols, depth);
  }, 2, 8);
  const double acc_ms = TimeMs([&] {
    GemmAccumulate(c.data(), a.data(), b.data(), rows, cols, depth);
  }, 2, 8);

  constexpr int width = 960;
  constexpr int height = 544;
  const float scale[3] = {1.F / (255.F * .229F), 1.F / (255.F * .224F), 1.F / (255.F * .225F)};
  const float shift[3] = {-.485F / .229F, -.456F / .224F, -.406F / .225F};
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * height * 3, 128);
  std::vector<float> nchw(static_cast<std::size_t>(3) * height * width);
  const double rgb_ms = TimeMs([&] {
    WriteIdentityRgbToNchw(nchw.data(), rgb.data(), width, height, width, 0, 0,
                           scale, shift, width);
  }, 1, 4);

  std::cout << "  \"gemm_ms\": " << gemm_ms << ",\n"
            << "  \"gemm_accumulate_ms\": " << acc_ms << ",\n"
            << "  \"identity_rgb_ms\": " << rgb_ms;
  if (argc == 6) {
    try {
      ppocr::Options options;
      options.backend = ppocr::Backend::cpu_only;
      const auto image = ppocr::LoadPPM(argv[4]);
      ppocr::OCR ocr(argv[1], argv[2], argv[3], options);
      const int runs = std::stoi(argv[5]);
      const double ocr_ms = TimeMs([&] { (void)ocr.Recognize(image); }, 1, runs);
      const auto boxes = ocr.Recognize(image).size();
      std::cout << ",\n  \"ocr_ms\": " << ocr_ms << ",\n  \"boxes\": " << boxes;
    } catch (const std::exception& ex) {
      std::cerr << "error: " << ex.what() << '\n';
      return 1;
    }
  }
  std::cout << "\n}\n";
  return 0;
}
