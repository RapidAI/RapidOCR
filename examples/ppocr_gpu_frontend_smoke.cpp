#include "ppocr/ppocr.hpp"
#include "vulkan_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> CpuRgbRegionInput(const ppocr::Image& image, int left, int top,
                                     int region_width, int region_height,
                                     int width, int height, bool recognition) {
  std::vector<float> out(std::size_t(3) * width * height);
  constexpr float detector_scale[3]{1.F / (255.F * .229F), 1.F / (255.F * .224F),
                                    1.F / (255.F * .225F)};
  constexpr float detector_offset[3]{-.485F / .229F, -.456F / .224F, -.406F / .225F};
  constexpr float recognition_scale[3]{2.F / 255.F, 2.F / 255.F, 2.F / 255.F};
  constexpr float recognition_offset[3]{-1.F, -1.F, -1.F};
  const float* scale = recognition ? recognition_scale : detector_scale;
  const float* offset = recognition ? recognition_offset : detector_offset;
  for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
    // Keep the scalar reference in the same evaluation order as the
    // production CPU preprocessor.  This narrow crop mode is intentionally
    // usable for a single OCR result when diagnosing a GPU-only mismatch.
    const float fx = (float(x) + .5F) * region_width / width - .5F;
    const float fy = (float(y) + .5F) * region_height / height - .5F;
    const int xf = int(std::floor(fx)), yf = int(std::floor(fy));
    const int x0 = std::clamp(xf, 0, region_width - 1), x1 = std::min(x0 + 1, region_width - 1);
    const int y0 = std::clamp(yf, 0, region_height - 1), y1 = std::min(y0 + 1, region_height - 1);
    const float dx = fx - xf, dy = fy - yf;
    for (int channel = 0; channel < 3; ++channel) {
      const int component = 2 - channel;
      const auto pixel = [&](int py, int px) {
        return float(image.rgb[(std::size_t(top + py) * image.width + left + px) * 3 + component]);
      };
      const float upper = pixel(y0, x0) * (1.F - dx) + pixel(y0, x1) * dx;
      const float lower = pixel(y1, x0) * (1.F - dx) + pixel(y1, x1) * dx;
      const float value = float(std::clamp(std::lround(upper * (1.F - dy) + lower * dy), 0L, 255L));
      out[(std::size_t(channel) * height + y) * width + x] = value * scale[channel] + offset[channel];
    }
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 && argc != 9 && argc != 10 && argc != 11) {
    std::cerr << "usage: ppocr_gpu_frontend_smoke IMAGE.ppm [LEFT TOP REGION_W REGION_H OUTPUT_W OUTPUT_H recognition_0_or_1 [CONTENT_W [OFFSET_ELEMENTS]]]\n";
    return 2;
  }
  try {
    const auto image = ppocr::LoadPPM(argv[1]);
    if (image.empty()) throw std::runtime_error("invalid image");
    const bool crop_mode = argc >= 9;
    const int left = crop_mode ? std::stoi(argv[2]) : 0;
    const int top = crop_mode ? std::stoi(argv[3]) : 0;
    const int region_width = crop_mode ? std::stoi(argv[4]) : image.width;
    const int region_height = crop_mode ? std::stoi(argv[5]) : image.height;
    const int width = crop_mode ? std::stoi(argv[6]) : std::max(32, (image.width / 32) * 32);
    const int height = crop_mode ? std::stoi(argv[7]) : std::max(32, (image.height / 32) * 32);
    const bool recognition = crop_mode && std::stoi(argv[8]) != 0;
    const int content_width = argc >= 10 ? std::stoi(argv[9]) : width;
    const std::size_t offset_elements = argc == 11 ? std::stoull(argv[10]) : 0;
    if (left < 0 || top < 0 || region_width <= 0 || region_height <= 0 || width <= 0 || height <= 0 ||
        content_width <= 0 || content_width > width || left + region_width > image.width ||
        top + region_height > image.height) {
      throw std::runtime_error("invalid RGB resize region");
    }
    ppocr::detail::VulkanTensorArena arena;
    auto rgb = arena.Acquire(image.rgb.size(), "rgb");
    const std::size_t output_elements = std::size_t(3) * width * height;
    auto output = arena.Acquire(offset_elements + output_elements, "output");
    if (!rgb.resident || !output.resident || !arena.UploadRgb8(rgb, image.rgb.data(), image.rgb.size()) ||
        !arena.ResizeRgbToNchwAt(rgb, output, offset_elements, image.width, image.height, left, top, region_width, region_height,
                                 width, height, 0, 0, 0, 0, 0, 0, content_width)) {
      throw std::runtime_error("GPU front end failed");
    }
    std::vector<float> downloaded(output.live_elements);
    if (!arena.Download(downloaded.data(), output, downloaded.size())) throw std::runtime_error("GPU front end download failed");
    std::vector<float> actual(downloaded.begin() + offset_elements,
                              downloaded.begin() + offset_elements + output_elements);
    if (const char* path = std::getenv("PPOCR_GPU_FRONTEND_DUMP_FILE")) {
      std::ofstream dump(path, std::ios::binary);
      if (!dump) throw std::runtime_error("cannot open PPOCR_GPU_FRONTEND_DUMP_FILE");
      dump.write(reinterpret_cast<const char*>(actual.data()),
                 static_cast<std::streamsize>(actual.size() * sizeof(float)));
      if (!dump) throw std::runtime_error("cannot write PPOCR_GPU_FRONTEND_DUMP_FILE");
    }
    const auto expected_content = CpuRgbRegionInput(image, left, top, region_width, region_height,
                                                     content_width, height, recognition);
    std::vector<float> expected(std::size_t(3) * width * height, 0.F);
    for (int channel = 0; channel < 3; ++channel) for (int y = 0; y < height; ++y) {
      const auto source_offset = (std::size_t(channel) * height + y) * content_width;
      const auto destination_offset = (std::size_t(channel) * height + y) * width;
      std::copy_n(expected_content.data() + source_offset, content_width,
                  expected.data() + destination_offset);
    }
    if (const char* path = std::getenv("PPOCR_CPU_FRONTEND_DUMP_FILE")) {
      std::ofstream dump(path, std::ios::binary);
      if (!dump) throw std::runtime_error("cannot open PPOCR_CPU_FRONTEND_DUMP_FILE");
      dump.write(reinterpret_cast<const char*>(expected.data()),
                 static_cast<std::streamsize>(expected.size() * sizeof(float)));
      if (!dump) throw std::runtime_error("cannot write PPOCR_CPU_FRONTEND_DUMP_FILE");
    }
    float max_error{}; std::size_t worst_index{};
    for (std::size_t i = 0; i < actual.size(); ++i) {
      const float error = std::abs(actual[i] - expected[i]);
      if (error > max_error) { max_error = error; worst_index = i; }
    }
    // Exercise the public device threshold/readback boundary independently;
    // DB component extraction consumes this compact mask rather than the
    // detector's FP32 probability activation.
    constexpr float source_probability[]{.01F, .20F, .20001F, .99F};
    auto probability = arena.Acquire(4, "probability");
    std::uint8_t mask[4]{};
    if (!probability.resident || !arena.Upload(probability, source_probability, 4) ||
        !arena.ThresholdToMask(probability, mask, 4, .20F) ||
        mask[0] != 0 || mask[1] != 0 || mask[2] != 1 || mask[3] != 1) {
      throw std::runtime_error("GPU threshold mask mismatch");
    }
    arena.Release(probability);
    arena.Release(output); arena.Release(rgb);
    // GPU arithmetic may fuse/interleave the same bilinear operations
    // differently from scalar C++; this is below one normalized uint8 step
    // (about 0.0175 for the detector's narrowest channel) and is validated
    // by end-to-end decoded-text parity as well.
    const float allowed_error = 0.018F;
    if (max_error > allowed_error) {
      throw std::runtime_error("front-end mismatch " + std::to_string(max_error));
    }
    std::cout << "GPU RGB resize/normalize smoke passed shape=1x3x" << height << 'x' << width
              << " region=" << left << ',' << top << ',' << region_width << 'x' << region_height
              << " recognition=" << recognition << " content_width=" << content_width
              << " offset_elements=" << offset_elements
              << " max_abs_error=" << max_error << " worst_index=" << worst_index
              << " cpu=" << expected[worst_index] << " gpu=" << actual[worst_index] << '\n';
    const std::size_t plane = std::size_t(width) * height;
    const int channel = static_cast<int>(worst_index / plane);
    const int spatial = static_cast<int>(worst_index % plane);
    const int y = spatial / width, x = spatial % width;
    if (x >= content_width) {
      std::cout << "worst_pixel falls in the explicit zero-padded tail\n";
      return 0;
    }
    const float fx = (float(x) + .5F) * region_width / content_width - .5F;
    const float fy = (float(y) + .5F) * region_height / height - .5F;
    const int xf = int(std::floor(fx)), yf = int(std::floor(fy));
    const int x0 = std::clamp(xf, 0, region_width - 1), x1 = std::min(x0 + 1, region_width - 1);
    const int y0 = std::clamp(yf, 0, region_height - 1), y1 = std::min(y0 + 1, region_height - 1);
    const float dx = fx - xf, dy = fy - yf; const int component = 2 - channel;
    const auto pixel = [&](int py, int px) {
      return float(image.rgb[(std::size_t(top + py) * image.width + left + px) * 3 + component]);
    };
    const float upper = pixel(y0, x0) * (1.F - dx) + pixel(y0, x1) * dx;
    const float lower = pixel(y1, x0) * (1.F - dx) + pixel(y1, x1) * dx;
    const float sample = upper * (1.F - dy) + lower * dy;
    std::cout << "worst_pixel channel=" << channel << " xy=" << x << ',' << y
              << " source_xy=" << x0 << ',' << y0 << '-' << x1 << ',' << y1
              << " dxdy=" << dx << ',' << dy << " samples=" << pixel(y0,x0) << ','
              << pixel(y0,x1) << ',' << pixel(y1,x0) << ',' << pixel(y1,x1)
              << " interpolated=" << sample << " rounded=" << std::lround(sample) << '\n';
  } catch (const std::exception& e) { std::cerr << "GPU front-end smoke failed: " << e.what() << '\n'; return 1; }
}
