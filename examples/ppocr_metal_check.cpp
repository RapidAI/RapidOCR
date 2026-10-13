// Metal correctness check: CPU versus Metal text and boxes on each image,
// repeated multithreaded Metal runs that must give identical results, and
// zero driver errors. Without Metal (not built, or no device) it checks that
// backend=metal falls back to CPU output and exits 0.
//
//   ppocr_metal_check DET REC DICT IMAGE.ppm [IMAGE.ppm ...]
//
// PPOCR_METAL_CHECK_THREADS (default 4) and PPOCR_METAL_CHECK_REPEATS
// (default 3) size the determinism run.
#include "ppocr/ppocr.hpp"
#include "vulkan_backend.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

std::string Fingerprint(const std::vector<ppocr::Result>& results) {
  std::ostringstream out;
  for (const auto& result : results) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &result.confidence, sizeof(bits));
    out << result.text << '\t' << std::hex << bits << std::dec;
    for (const auto& corner : result.box) out << '\t' << corner[0] << ',' << corner[1];
    out << '\n';
  }
  return out.str();
}

double Iou(const ppocr::Result& a, const ppocr::Result& b) {
  const auto bounds = [](const ppocr::Result& r) {
    int x0 = r.box[0][0], y0 = r.box[0][1], x1 = x0, y1 = y0;
    for (const auto& p : r.box) {
      x0 = std::min(x0, p[0]); y0 = std::min(y0, p[1]);
      x1 = std::max(x1, p[0]); y1 = std::max(y1, p[1]);
    }
    return std::array<int, 4>{x0, y0, x1, y1};
  };
  const auto p = bounds(a), q = bounds(b);
  const double iw = std::max(0, std::min(p[2], q[2]) - std::max(p[0], q[0]));
  const double ih = std::max(0, std::min(p[3], q[3]) - std::max(p[1], q[1]));
  const double inter = iw * ih;
  const double area = double(p[2] - p[0]) * (p[3] - p[1]) + double(q[2] - q[0]) * (q[3] - q[1]) - inter;
  return area > 0 ? inter / area : 1.0;
}

int EnvInt(const char* name, int fallback) {
  const char* text = std::getenv(name);
  return text && std::atoi(text) > 0 ? std::atoi(text) : fallback;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: ppocr_metal_check DET REC DICT IMAGE.ppm [IMAGE.ppm ...]\n";
    return 2;
  }
  const bool metal_available = ppocr::MetalAvailable();
  ppocr::Options cpu_options;
  cpu_options.backend = ppocr::Backend::cpu_only;
  ppocr::Options metal_options;
  metal_options.backend = ppocr::Backend::metal;
  ppocr::OCR cpu(argv[1], argv[2], argv[3], cpu_options);
  ppocr::OCR metal(argv[1], argv[2], argv[3], metal_options);
  const auto info = ppocr::QueryBackendInfo();
  std::cout << "metal_available=" << metal_available << " device=" << info.device_name << '\n';
  const int threads = EnvInt("PPOCR_METAL_CHECK_THREADS", 4);
  const int repeats = EnvInt("PPOCR_METAL_CHECK_REPEATS", 3);
  int failures = 0;
  for (int arg = 4; arg < argc; ++arg) {
    const auto image = ppocr::LoadPPM(argv[arg]);
    const auto cpu_results = cpu.Recognize(image);
    const auto metal_results = metal.Recognize(image);
    double min_iou = 1.0;
    int text_diffs = 0;
    if (cpu_results.size() != metal_results.size()) {
      std::cout << argv[arg] << ": box count cpu=" << cpu_results.size()
                << " metal=" << metal_results.size() << '\n';
      ++failures;
      continue;
    }
    for (std::size_t i = 0; i < cpu_results.size(); ++i) {
      if (cpu_results[i].text != metal_results[i].text) {
        ++text_diffs;
        std::cout << "  line " << i << " cpu='" << cpu_results[i].text << "' metal='"
                  << metal_results[i].text << "'\n";
      }
      min_iou = std::min(min_iou, Iou(cpu_results[i], metal_results[i]));
    }
    // Determinism: several threads share one engine and must reproduce the
    // first Metal result bit for bit.
    const std::string reference = Fingerprint(metal_results);
    std::vector<int> diverged(static_cast<std::size_t>(threads));
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) {
      workers.emplace_back([&, t] {
        for (int r = 0; r < repeats; ++r) {
          if (Fingerprint(metal.Recognize(image)) != reference) ++diverged[std::size_t(t)];
        }
      });
    }
    for (auto& worker : workers) worker.join();
    int total_diverged = 0;
    for (int d : diverged) total_diverged += d;
    const bool ok = text_diffs == 0 && min_iou >= 0.99 && total_diverged == 0;
    std::cout << argv[arg] << ": lines=" << cpu_results.size() << " text_diffs=" << text_diffs
              << " min_iou=" << min_iou << " threads=" << threads << " repeats=" << repeats
              << " diverged=" << total_diverged << (ok ? " OK" : " FAIL") << '\n';
    if (!ok) ++failures;
  }
  const int errors = ppocr::detail::MetalErrorCount();
  std::cout << "metal_errors=" << errors << '\n';
  if (errors != 0) ++failures;
  std::cout << (failures == 0 ? "metal check passed" : "metal check FAILED") << '\n';
  return failures == 0 ? 0 : 1;
}
