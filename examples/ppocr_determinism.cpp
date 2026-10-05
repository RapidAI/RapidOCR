#include "kernels.hpp"
#include "ppocr/ppocr.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string Fingerprint(const std::vector<ppocr::Result>& results) {
  std::ostringstream out;
  for (const auto& result : results) {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(result.confidence));
    std::memcpy(&bits, &result.confidence, sizeof(bits));
    out << result.text << '\t' << std::hex << bits << std::dec << '\t'
        << result.bbox[0] << ',' << result.bbox[1] << ',' << result.bbox[2] << ','
        << result.bbox[3] << '\t';
    for (const auto& corner : result.box) out << corner[0] << ',' << corner[1] << ';';
    out << '\n';
  }
  return out.str();
}

int PoolCheck(int repeats) {
  constexpr int kPlanes = 48;
  constexpr int kHeight = 24;
  constexpr int kWidth = 400;
  const int output_height = (kHeight - 3) / 3 + 1;
  const int output_width = (kWidth - 2) / 2 + 1;
  const std::size_t packed =
      std::size_t(kPlanes) * output_height * output_width;
  if (packed < 65536) {
    std::cerr << "pool fixture is below the parallel threshold\n";
    return 1;
  }
  std::vector<float> input(std::size_t(kPlanes) * kHeight * kWidth);
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<float>(static_cast<int>(i % 97) - 40) * 0.015625F;
  const std::size_t output_plane = std::size_t(output_height) * output_width;
  const std::size_t input_plane = std::size_t(kHeight) * kWidth;
  // Chunks stay under the parallel threshold, so this is the serial kernel.
  std::vector<float> expected(packed);
  for (int plane = 0; plane < kPlanes;) {
    int chunk = 1;
    while (plane + chunk < kPlanes &&
           std::size_t(chunk + 1) * output_plane < 65536) {
      ++chunk;
    }
    ppocr::detail::kernels::AveragePool3x2Valid(
        expected.data() + std::size_t(plane) * output_plane,
        input.data() + std::size_t(plane) * input_plane, static_cast<std::size_t>(chunk),
        kHeight, kWidth);
    plane += chunk;
  }
  for (int repeat = 0; repeat < repeats; ++repeat) {
    auto working = input;
    ppocr::detail::kernels::AveragePool3x2Valid(
        working.data(), working.data(), kPlanes, kHeight, kWidth);
    if (working.size() < expected.size() ||
        std::memcmp(working.data(), expected.data(), expected.size() * sizeof(float)) != 0) {
      std::cerr << "in-place average pool diverged on repeat " << repeat << '\n';
      return 1;
    }
  }
  std::cerr << "pool determinism passed repeats=" << repeats
            << " planes=" << kPlanes << " packed=" << packed << '\n';
  return 0;
}

int OcrCheck(int argc, char** argv, bool fingerprint_only) {
  if (argc < 6) {
    std::cerr << "usage: ppocr_determinism ocr|fingerprint DET REC DICT IMAGE REPEATS\n";
    return 2;
  }
  const int repeats = std::atoi(argv[5]);
  if (repeats <= 0) return 2;
  ppocr::Options options;
  options.backend = ppocr::Backend::cpu_only;
  ppocr::OCR ocr(argv[1], argv[2], argv[3], options);
  const auto image = ppocr::LoadPPM(argv[4]);
  const auto info = ppocr::QueryCpuInfo();
  std::string first;
  double timed_ms = 0;
  int timed = 0;
  for (int repeat = 0; repeat < repeats; ++repeat) {
    const auto begin = std::chrono::steady_clock::now();
    const auto results = ocr.Recognize(image);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    if (repeat > 0) {
      timed_ms += ms;
      ++timed;
    }
    const auto finger = Fingerprint(results);
    if (repeat == 0) first = finger;
    else if (finger != first) {
      std::cerr << "ocr diverged on repeat " << repeat << " isa=" << info.active_isa
                << " threads=" << info.threads << '\n';
      return 1;
    }
  }
  if (fingerprint_only) {
    std::cout << first;
    return 0;
  }
  const double avg = timed > 0 ? timed_ms / timed : 0;
  std::cerr << "ocr determinism passed repeats=" << repeats
            << " isa=" << info.active_isa << " threads=" << info.threads
            << " lines=" << std::count(first.begin(), first.end(), '\n')
            << " avg_ms=" << avg << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: ppocr_determinism pool [REPEATS]\n"
                 "       ppocr_determinism ocr|fingerprint DET REC DICT IMAGE REPEATS\n";
    return 2;
  }
  try {
    const std::string mode = argv[1];
    if (mode == "pool") {
      const int repeats = argc >= 3 ? std::atoi(argv[2]) : 50;
      return PoolCheck(repeats > 0 ? repeats : 50);
    }
    if (mode == "ocr" || mode == "fingerprint")
      return OcrCheck(argc - 1, argv + 1, mode == "fingerprint");
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
  std::cerr << "unknown mode\n";
  return 2;
}
