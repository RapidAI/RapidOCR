#include "kernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

bool VerifyGemm(int rows, int cols, int depth, bool with_bias) {
  std::vector<float> a(std::size_t(rows) * depth);
  std::vector<float> b(std::size_t(depth) * cols);
  std::vector<float> bias(static_cast<std::size_t>(cols), 0.F);
  std::vector<float> output(std::size_t(rows) * cols);
  std::vector<float> expected(std::size_t(rows) * cols);
  for (std::size_t i = 0; i < a.size(); ++i) a[i] = float(int(i % 19) - 9) * .0625F;
  for (std::size_t i = 0; i < b.size(); ++i) b[i] = float(int(i % 23) - 11) * .03125F;
  for (std::size_t i = 0; i < bias.size(); ++i) bias[i] = float(int(i % 7) - 3) * .125F;

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      float value = with_bias ? bias[std::size_t(col)] : 0.F;
      for (int k = 0; k < depth; ++k) {
        value += a[std::size_t(row) * depth + k] * b[std::size_t(k) * cols + col];
      }
      expected[std::size_t(row) * cols + col] = value;
    }
  }
  ppocr::detail::kernels::Gemm(output.data(), a.data(), b.data(),
                                with_bias ? bias.data() : nullptr, rows, cols, depth);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 2e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "GEMM mismatch rows=" << rows << " cols=" << cols
                << " depth=" << depth << " bias=" << with_bias
                << " index=" << i << " got=" << output[i]
                << " expected=" << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyIdentityRgbToNchw() {
  // Packed RGB -> NCHW BGR affine, including an odd tail so AVX-512/AVX2
  // vector bodies and the scalar remainder share one independent reference.
  constexpr int width = 37, height = 5, row_width = 40, left = 2, top = 1;
  constexpr int source_width = 64, source_height = 8;
  const float scale[3] = {1.F / (255.F * .229F), 1.F / (255.F * .224F),
                          1.F / (255.F * .225F)};
  const float shift[3] = {-.485F / .229F, -.456F / .224F, -.406F / .225F};
  std::vector<std::uint8_t> rgb(std::size_t(source_width) * source_height * 3);
  for (std::size_t i = 0; i < rgb.size(); ++i) {
    rgb[i] = static_cast<std::uint8_t>((i * 41 + 17) & 255u);
  }
  std::vector<float> output(std::size_t(3) * height * row_width, -99.F);
  std::vector<float> expected(output.size(), -99.F);
  const std::size_t plane = std::size_t(height) * row_width;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const auto* pixel = rgb.data() +
          (std::size_t(top + y) * source_width + left + x) * 3;
      expected[std::size_t(y) * row_width + x] =
          static_cast<float>(pixel[2]) * scale[0] + shift[0];
      expected[plane + std::size_t(y) * row_width + x] =
          static_cast<float>(pixel[1]) * scale[1] + shift[1];
      expected[2 * plane + std::size_t(y) * row_width + x] =
          static_cast<float>(pixel[0]) * scale[2] + shift[2];
    }
  }
  ppocr::detail::kernels::WriteIdentityRgbToNchw(
      output.data(), rgb.data(), width, height, source_width, left, top,
      scale, shift, row_width);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 3e-6F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "identity RGB->NCHW mismatch index=" << i
                << " got=" << output[i] << " expected=" << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyBilinearRgbToNchw(int image_width, int image_height, int left, int top,
                             int source_width, int source_height, int width,
                             int height, int row_width, bool parallel) {
  // Independent scalar bilinear + uint8 rounding + BGR affine. Covers the
  // gating detector 720x152->704x160 map, a cropped odd tail, and a 1-pixel
  // source so gather-safe SIMD tails never hide a scalar edge.
  const float scale[3] = {1.F / (255.F * .229F), 1.F / (255.F * .224F),
                          1.F / (255.F * .225F)};
  const float shift[3] = {-.485F / .229F, -.456F / .224F, -.406F / .225F};
  std::vector<std::uint8_t> rgb(std::size_t(image_width) * image_height * 3);
  for (std::size_t i = 0; i < rgb.size(); ++i) {
    rgb[i] = static_cast<std::uint8_t>((i * 41 + 17) & 255u);
  }
  std::vector<float> output(std::size_t(3) * height * row_width, -99.F);
  std::vector<float> expected(output.size(), -99.F);
  const std::size_t plane = std::size_t(height) * row_width;
  std::vector<int> x0(static_cast<std::size_t>(width));
  std::vector<int> x1(static_cast<std::size_t>(width));
  std::vector<float> dx(static_cast<std::size_t>(width));
  for (int x = 0; x < width; ++x) {
    const float fx = (float(x) + .5F) * source_width / width - .5F;
    const int x_floor = int(std::floor(fx));
    x0[static_cast<std::size_t>(x)] = std::clamp(x_floor, 0, source_width - 1);
    x1[static_cast<std::size_t>(x)] =
        std::min(x0[static_cast<std::size_t>(x)] + 1, source_width - 1);
    dx[static_cast<std::size_t>(x)] = fx - x_floor;
  }
  for (int y = 0; y < height; ++y) {
    const float fy = (float(y) + .5F) * source_height / height - .5F;
    const int y_floor = int(std::floor(fy));
    const int y0 = std::clamp(y_floor, 0, source_height - 1);
    const int y1 = std::min(y0 + 1, source_height - 1);
    const float dy = fy - y_floor;
    const float inverse_dy = 1.F - dy;
    const auto* const upper_row =
        rgb.data() + (std::size_t(top + y0) * image_width + left) * 3;
    const auto* const lower_row =
        rgb.data() + (std::size_t(top + y1) * image_width + left) * 3;
    for (int x = 0; x < width; ++x) {
      const auto* const upper_left = upper_row + std::size_t(x0[static_cast<std::size_t>(x)]) * 3;
      const auto* const upper_right = upper_row + std::size_t(x1[static_cast<std::size_t>(x)]) * 3;
      const auto* const lower_left = lower_row + std::size_t(x0[static_cast<std::size_t>(x)]) * 3;
      const auto* const lower_right = lower_row + std::size_t(x1[static_cast<std::size_t>(x)]) * 3;
      const float inverse_dx = 1.F - dx[static_cast<std::size_t>(x)];
      const float upper_red =
          float(upper_left[0]) * inverse_dx + float(upper_right[0]) * dx[static_cast<std::size_t>(x)];
      const float lower_red =
          float(lower_left[0]) * inverse_dx + float(lower_right[0]) * dx[static_cast<std::size_t>(x)];
      const float sampled_red = static_cast<float>(std::clamp(
          static_cast<int>(upper_red * inverse_dy + lower_red * dy + .5F), 0, 255));
      const float upper_green =
          float(upper_left[1]) * inverse_dx + float(upper_right[1]) * dx[static_cast<std::size_t>(x)];
      const float lower_green =
          float(lower_left[1]) * inverse_dx + float(lower_right[1]) * dx[static_cast<std::size_t>(x)];
      const float sampled_green = static_cast<float>(std::clamp(
          static_cast<int>(upper_green * inverse_dy + lower_green * dy + .5F), 0, 255));
      const float upper_blue =
          float(upper_left[2]) * inverse_dx + float(upper_right[2]) * dx[static_cast<std::size_t>(x)];
      const float lower_blue =
          float(lower_left[2]) * inverse_dx + float(lower_right[2]) * dx[static_cast<std::size_t>(x)];
      const float sampled_blue = static_cast<float>(std::clamp(
          static_cast<int>(upper_blue * inverse_dy + lower_blue * dy + .5F), 0, 255));
      expected[std::size_t(y) * row_width + x] = sampled_blue * scale[0] + shift[0];
      expected[plane + std::size_t(y) * row_width + x] = sampled_green * scale[1] + shift[1];
      expected[2 * plane + std::size_t(y) * row_width + x] = sampled_red * scale[2] + shift[2];
    }
  }
  ppocr::detail::kernels::WriteBilinearRgbToNchw(
      output.data(), rgb.data(), image_width, image_height, left, top, source_width,
      source_height, width, height, scale, shift, row_width, parallel);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 3e-6F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "bilinear RGB->NCHW mismatch " << image_width << "x" << image_height
                << "->" << width << "x" << height << " parallel=" << parallel
                << " index=" << i << " got=" << output[i]
                << " expected=" << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyLayerNorm(std::size_t rows, std::size_t width) {
  std::vector<float> input(rows * width), gamma(width), beta(width);
  std::vector<float> output(input.size()), expected(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = float(int(i % 47) - 23) * .0625F;
  }
  for (std::size_t column = 0; column < width; ++column) {
    gamma[column] = float((column % 11) + 1) * .09375F;
    beta[column] = float((column % 7) - 3) * .125F;
  }
  constexpr float epsilon = 1e-5F;
  for (std::size_t row = 0; row < rows; ++row) {
    const float* src = input.data() + row * width;
    float* dst = expected.data() + row * width;
    float mean = 0.F;
    for (std::size_t column = 0; column < width; ++column) mean += src[column];
    mean /= static_cast<float>(width);
    float variance = 0.F;
    for (std::size_t column = 0; column < width; ++column) {
      const float centered = src[column] - mean;
      variance += centered * centered;
    }
    const float denom = std::sqrt(variance / static_cast<float>(width) + epsilon);
    for (std::size_t column = 0; column < width; ++column) {
      dst[column] = ((src[column] - mean) / denom) * gamma[column] + beta[column];
    }
  }
  ppocr::detail::kernels::LayerNorm(output.data(), input.data(), gamma.data(),
                                    beta.data(), rows, width, epsilon);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 3e-6F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "LayerNorm mismatch rows=" << rows << " width=" << width
                << " index=" << i << " got=" << output[i]
                << " expected=" << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

bool VerifySigmoid(std::size_t n) {
  std::vector<float> input(n), output(n), expected(n);
  for (std::size_t i = 0; i < n; ++i) input[i] = float(int(i % 63) - 31) * .0625F;
  for (std::size_t i = 0; i < n; ++i) expected[i] = 1.F / (1.F + std::exp(-input[i]));
  ppocr::detail::kernels::Sigmoid(output.data(), input.data(), n);
  for (std::size_t i = 0; i < n; ++i) {
    const float tolerance = 3e-6F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "Sigmoid mismatch n=" << n << " index=" << i << " got=" << output[i]
                << " expected=" << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyMaxPool2x2Same(int planes, int height, int width) {
  std::vector<float> input(std::size_t(planes) * height * width);
  std::vector<float> output(input.size());
  std::vector<float> expected(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = float(int(i % 29) - 14) * .125F;
  }
  const std::size_t plane = std::size_t(height) * width;
  for (int p = 0; p < planes; ++p) {
    const float* in = input.data() + std::size_t(p) * plane;
    float* out = expected.data() + std::size_t(p) * plane;
    for (int y = 0; y + 1 < height; ++y) {
      const float* row0 = in + std::size_t(y) * width;
      const float* row1 = row0 + width;
      for (int x = 0; x + 1 < width; ++x) {
        out[std::size_t(y) * width + x] = std::max(
            std::max(row0[x], row0[x + 1]), std::max(row1[x], row1[x + 1]));
      }
      out[std::size_t(y) * width + width - 1] =
          std::max(row0[width - 1], row1[width - 1]);
    }
    if (height == 1) {
      for (int x = 0; x + 1 < width; ++x) out[x] = std::max(in[x], in[x + 1]);
    } else {
      const float* row = in + std::size_t(height - 1) * width;
      float* out_row = out + std::size_t(height - 1) * width;
      for (int x = 0; x + 1 < width; ++x) out_row[x] = std::max(row[x], row[x + 1]);
    }
    out[plane - 1] = in[plane - 1];
  }
  ppocr::detail::kernels::MaxPool2x2Same(output.data(), input.data(), planes, height, width);
  for (std::size_t i = 0; i < output.size(); ++i) {
    if (output[i] != expected[i]) {
      std::cerr << "MaxPool2x2Same mismatch at " << i << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyMaxPool2x2Valid(int planes, int height, int width) {
  const int out_h = height - 1, out_w = width - 1;
  std::vector<float> input(std::size_t(planes) * height * width);
  std::vector<float> output(std::size_t(planes) * out_h * out_w);
  std::vector<float> expected(output.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = float(int(i % 29) - 14) * .125F;
  }
  for (int plane = 0; plane < planes; ++plane) {
    const float* in = input.data() + std::size_t(plane) * height * width;
    float* out = expected.data() + std::size_t(plane) * out_h * out_w;
    for (int y = 0; y < out_h; ++y) for (int x = 0; x < out_w; ++x) {
      out[std::size_t(y) * out_w + x] = std::max(
          std::max(in[std::size_t(y) * width + x], in[std::size_t(y) * width + x + 1]),
          std::max(in[std::size_t(y + 1) * width + x], in[std::size_t(y + 1) * width + x + 1]));
    }
  }
  ppocr::detail::kernels::MaxPool2x2Valid(output.data(), input.data(), planes, height, width);
  for (std::size_t i = 0; i < output.size(); ++i) {
    if (output[i] != expected[i]) {
      std::cerr << "MaxPool2x2Valid mismatch at " << i << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyExactGelu() {
  std::vector<float> input(4096 + 13), output(input.size()), expected(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = float(int(i % 127) - 63) * .0625F;
  }
  constexpr float inv_sqrt2 = 0.7071067811865475244F;
  for (std::size_t i = 0; i < input.size(); ++i) {
    expected[i] = input[i] * .5F * (1.F + std::erf(input[i] * inv_sqrt2));
  }
  ppocr::detail::kernels::ExactGelu(output.data(), input.data(), input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    const float tolerance = 2e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "exact GELU mismatch index=" << i << " got=" << output[i]
                << " expected=" << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyRgbByteWiden() {
  // Covers vector blocks and an odd scalar tail. The helper is shared by the
  // x86 AVX-512/AVX2 and ARM NEON GPU-upload boundary paths.
  std::vector<std::uint8_t> input(16 * 1024 + 13);
  std::vector<float> output(input.size());
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<std::uint8_t>((index * 37 + 19) & 255u);
  }
  ppocr::detail::kernels::WidenU8ToFloat(output.data(), input.data(), input.size());
  for (std::size_t index = 0; index < input.size(); ++index) {
    if (output[index] != static_cast<float>(input[index])) {
      std::cerr << "RGB byte widen mismatch index=" << index << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyConv3x3Tile(int input_channels, int output_channels, int height,
                       int width, bool relu) {
  const std::size_t input_plane = std::size_t(height) * width;
  const std::size_t output_plane = input_plane;
  std::vector<float> input(std::size_t(input_channels) * input_plane);
  std::vector<float> weights(std::size_t(output_channels) * input_channels * 9);
  std::vector<float> bias(static_cast<std::size_t>(output_channels));
  std::vector<float> output(std::size_t(output_channels) * output_plane);
  std::vector<float> expected(output.size());
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 31) - 15) * .03125F;
  for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = float(int(i % 29) - 14) * .015625F;
  for (std::size_t i = 0; i < bias.size(); ++i) bias[i] = float(int(i % 13) - 6) * .0625F;
  for (int output_channel = 0; output_channel < output_channels; ++output_channel) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        float value = bias[std::size_t(output_channel)];
        for (int input_channel = 0; input_channel < input_channels; ++input_channel) {
          const float* plane = input.data() + std::size_t(input_channel) * input_plane;
          const float* filter = weights.data() +
              (std::size_t(output_channel) * input_channels + input_channel) * 9;
          for (int ky = 0; ky < 3; ++ky) for (int kx = 0; kx < 3; ++kx) {
            const int iy = y + ky - 1, ix = x + kx - 1;
            if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
              value += plane[std::size_t(iy) * width + ix] * filter[ky * 3 + kx];
            }
          }
        }
        expected[(std::size_t(output_channel) * height + y) * width + x] =
            relu ? std::max(value, 0.F) : value;
      }
    }
  }
  ppocr::detail::kernels::Conv2d(output.data(), input.data(), weights.data(), bias.data(),
                                  input_channels, output_channels, height, width,
                                  height, width, 3, 3, 1, 1, 1, 1, relu);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 3e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "3x3 convolution mismatch C=" << input_channels << " M="
                << output_channels << " H=" << height << " W=" << width
                << " relu=" << relu << " index=" << i << " got=" << output[i]
                << " expected=" << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyConv3x3Stride2(int input_channels, int output_channels, int height,
                          int width, bool relu) {
  const int pad = 1;
  const int output_height = (height + 2 * pad - 3) / 2 + 1;
  const int output_width = (width + 2 * pad - 3) / 2 + 1;
  const std::size_t input_plane = std::size_t(height) * width;
  const std::size_t output_plane = std::size_t(output_height) * output_width;
  std::vector<float> input(std::size_t(input_channels) * input_plane);
  std::vector<float> weights(std::size_t(output_channels) * input_channels * 9);
  std::vector<float> bias(static_cast<std::size_t>(output_channels));
  std::vector<float> output(std::size_t(output_channels) * output_plane);
  std::vector<float> expected(output.size());
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 31) - 15) * .03125F;
  for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = float(int(i % 29) - 14) * .015625F;
  for (std::size_t i = 0; i < bias.size(); ++i) bias[i] = float(int(i % 13) - 6) * .0625F;
  for (int output_channel = 0; output_channel < output_channels; ++output_channel) {
    for (int y = 0; y < output_height; ++y) {
      for (int x = 0; x < output_width; ++x) {
        float value = bias[std::size_t(output_channel)];
        for (int input_channel = 0; input_channel < input_channels; ++input_channel) {
          const float* plane = input.data() + std::size_t(input_channel) * input_plane;
          const float* filter = weights.data() +
              (std::size_t(output_channel) * input_channels + input_channel) * 9;
          for (int ky = 0; ky < 3; ++ky) for (int kx = 0; kx < 3; ++kx) {
            const int iy = y * 2 + ky - pad, ix = x * 2 + kx - pad;
            if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
              value += plane[std::size_t(iy) * width + ix] * filter[ky * 3 + kx];
            }
          }
        }
        expected[(std::size_t(output_channel) * output_height + y) * output_width + x] =
            relu ? std::max(value, 0.F) : value;
      }
    }
  }
  ppocr::detail::kernels::Conv2d(output.data(), input.data(), weights.data(), bias.data(),
                                  input_channels, output_channels, height, width,
                                  output_height, output_width, 3, 3, 2, 2, pad, pad, relu);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 3e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "3x3 stride-2 mismatch C=" << input_channels << " M="
                << output_channels << " H=" << height << " W=" << width
                << " relu=" << relu << " index=" << i << " got=" << output[i]
                << " expected=" << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyConcatChannelConv(int source_count, int channels_each, int output_channels,
                             int height, int width, bool relu,
                             int stride_h = 1, int stride_w = 1) {
  const int total_channels = source_count * channels_each;
  const int pad = 1;
  const int output_height = (height + 2 * pad - 3) / stride_h + 1;
  const int output_width = (width + 2 * pad - 3) / stride_w + 1;
  const std::size_t plane = std::size_t(height) * width;
  const std::size_t output_plane = std::size_t(output_height) * output_width;
  std::vector<std::vector<float>> sources(static_cast<std::size_t>(source_count));
  std::vector<const float*> source_ptrs(static_cast<std::size_t>(source_count));
  std::vector<int> source_channels(static_cast<std::size_t>(source_count), channels_each);
  std::vector<float> concat(std::size_t(total_channels) * plane);
  for (int source = 0; source < source_count; ++source) {
    sources[static_cast<std::size_t>(source)].resize(std::size_t(channels_each) * plane);
    for (std::size_t i = 0; i < sources[static_cast<std::size_t>(source)].size(); ++i) {
      sources[static_cast<std::size_t>(source)][i] =
          float(int((i + std::size_t(source) * 17) % 31) - 15) * .03125F;
    }
    source_ptrs[static_cast<std::size_t>(source)] = sources[static_cast<std::size_t>(source)].data();
    std::memcpy(concat.data() + std::size_t(source) * channels_each * plane,
                sources[static_cast<std::size_t>(source)].data(),
                sources[static_cast<std::size_t>(source)].size() * sizeof(float));
  }
  std::vector<float> weights(std::size_t(output_channels) * total_channels * 9);
  std::vector<float> bias(static_cast<std::size_t>(output_channels));
  std::vector<float> expected(std::size_t(output_channels) * output_plane);
  std::vector<float> output(expected.size());
  for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = float(int(i % 29) - 14) * .015625F;
  for (std::size_t i = 0; i < bias.size(); ++i) bias[i] = float(int(i % 13) - 6) * .0625F;
  for (int output_channel = 0; output_channel < output_channels; ++output_channel) {
    for (int y = 0; y < output_height; ++y) for (int x = 0; x < output_width; ++x) {
      float value = bias[static_cast<std::size_t>(output_channel)];
      for (int input_channel = 0; input_channel < total_channels; ++input_channel) {
        const float* src = concat.data() + std::size_t(input_channel) * plane;
        const float* filter = weights.data() +
            (std::size_t(output_channel) * total_channels + input_channel) * 9;
        for (int ky = 0; ky < 3; ++ky) for (int kx = 0; kx < 3; ++kx) {
          const int iy = y * stride_h + ky - pad, ix = x * stride_w + kx - pad;
          if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
            value += src[std::size_t(iy) * width + ix] * filter[ky * 3 + kx];
          }
        }
      }
      expected[(std::size_t(output_channel) * output_height + y) * output_width + x] =
          relu ? std::max(value, 0.F) : value;
    }
  }
  ppocr::detail::kernels::ConcatChannelConv2d(
      output.data(), source_ptrs.data(), source_channels.data(), source_count,
      weights.data(), bias.data(), output_channels, height, width, output_height,
      output_width, 3, 3, stride_h, stride_w, pad, pad, relu);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 3e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "concat-channel conv mismatch sources=" << source_count
                << " C=" << channels_each << " M=" << output_channels
                << " stride=" << stride_h << "x" << stride_w
                << " relu=" << relu << " index=" << i << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyConcatChannelConvMixed(const int* channels, int source_count,
                                  int output_channels, int height, int width,
                                  bool relu, int stride_h, int stride_w) {
  const int pad = 1;
  const int output_height = (height + 2 * pad - 3) / stride_h + 1;
  const int output_width = (width + 2 * pad - 3) / stride_w + 1;
  const std::size_t plane = std::size_t(height) * width;
  const std::size_t output_plane = std::size_t(output_height) * output_width;
  int total_channels = 0;
  for (int source = 0; source < source_count; ++source) total_channels += channels[source];
  std::vector<std::vector<float>> sources(static_cast<std::size_t>(source_count));
  std::vector<const float*> source_ptrs(static_cast<std::size_t>(source_count));
  std::vector<int> source_channels(static_cast<std::size_t>(source_count));
  std::vector<float> concat(std::size_t(total_channels) * plane);
  int channel0 = 0;
  for (int source = 0; source < source_count; ++source) {
    source_channels[static_cast<std::size_t>(source)] = channels[source];
    sources[static_cast<std::size_t>(source)].resize(std::size_t(channels[source]) * plane);
    for (std::size_t i = 0; i < sources[static_cast<std::size_t>(source)].size(); ++i) {
      sources[static_cast<std::size_t>(source)][i] =
          float(int((i + std::size_t(source) * 17) % 31) - 15) * .03125F;
    }
    source_ptrs[static_cast<std::size_t>(source)] = sources[static_cast<std::size_t>(source)].data();
    std::memcpy(concat.data() + std::size_t(channel0) * plane,
                sources[static_cast<std::size_t>(source)].data(),
                sources[static_cast<std::size_t>(source)].size() * sizeof(float));
    channel0 += channels[source];
  }
  std::vector<float> weights(std::size_t(output_channels) * total_channels * 9);
  std::vector<float> bias(static_cast<std::size_t>(output_channels));
  std::vector<float> expected(std::size_t(output_channels) * output_plane);
  std::vector<float> output(expected.size());
  for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = float(int(i % 29) - 14) * .015625F;
  for (std::size_t i = 0; i < bias.size(); ++i) bias[i] = float(int(i % 13) - 6) * .0625F;
  for (int output_channel = 0; output_channel < output_channels; ++output_channel) {
    for (int y = 0; y < output_height; ++y) for (int x = 0; x < output_width; ++x) {
      float value = bias[static_cast<std::size_t>(output_channel)];
      for (int input_channel = 0; input_channel < total_channels; ++input_channel) {
        const float* src = concat.data() + std::size_t(input_channel) * plane;
        const float* filter = weights.data() +
            (std::size_t(output_channel) * total_channels + input_channel) * 9;
        for (int ky = 0; ky < 3; ++ky) for (int kx = 0; kx < 3; ++kx) {
          const int iy = y * stride_h + ky - pad, ix = x * stride_w + kx - pad;
          if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
            value += src[std::size_t(iy) * width + ix] * filter[ky * 3 + kx];
          }
        }
      }
      expected[(std::size_t(output_channel) * output_height + y) * output_width + x] =
          relu ? std::max(value, 0.F) : value;
    }
  }
  ppocr::detail::kernels::ConcatChannelConv2d(
      output.data(), source_ptrs.data(), source_channels.data(), source_count,
      weights.data(), bias.data(), output_channels, height, width, output_height,
      output_width, 3, 3, stride_h, stride_w, pad, pad, relu);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 3e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "mixed concat-channel conv mismatch index=" << i << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyConcatConvDualTranspose(int source_count, int channels_each, int conv_oc,
                                   int final_oc, int height, int width) {
  const int pad = 1;
  const int conv_h = height;
  const int conv_width = width;
  const int mid_h = conv_h * 2, mid_w = conv_width * 2;
  const int out_h = conv_h * 4, out_w = conv_width * 4;
  const std::size_t plane = std::size_t(height) * width;
  const int total_channels = source_count * channels_each;
  std::vector<std::vector<float>> sources(static_cast<std::size_t>(source_count));
  std::vector<const float*> source_ptrs(static_cast<std::size_t>(source_count));
  std::vector<int> source_channels(static_cast<std::size_t>(source_count), channels_each);
  for (int source = 0; source < source_count; ++source) {
    sources[static_cast<std::size_t>(source)].resize(std::size_t(channels_each) * plane);
    for (std::size_t i = 0; i < sources[static_cast<std::size_t>(source)].size(); ++i) {
      sources[static_cast<std::size_t>(source)][i] =
          float(int((i + std::size_t(source) * 13) % 29) - 14) * .03125F;
    }
    source_ptrs[static_cast<std::size_t>(source)] = sources[static_cast<std::size_t>(source)].data();
  }
  std::vector<float> conv_weights(std::size_t(conv_oc) * total_channels * 9);
  std::vector<float> conv_b(static_cast<std::size_t>(conv_oc));
  std::vector<float> ct0_w(std::size_t(conv_oc) * conv_oc * 4);
  std::vector<float> ct0_b(static_cast<std::size_t>(conv_oc));
  std::vector<float> ct1_w(std::size_t(conv_oc) * final_oc * 4);
  std::vector<float> ct1_b(static_cast<std::size_t>(final_oc));
  for (std::size_t i = 0; i < conv_weights.size(); ++i)
    conv_weights[i] = float(int(i % 29) - 14) * .015625F;
  for (std::size_t i = 0; i < conv_b.size(); ++i) conv_b[i] = float(int(i % 13) - 6) * .0625F;
  for (std::size_t i = 0; i < ct0_w.size(); ++i) ct0_w[i] = float(int(i % 23) - 11) * .03125F;
  for (std::size_t i = 0; i < ct0_b.size(); ++i) ct0_b[i] = float(int(i % 7) - 3) * .0625F;
  for (std::size_t i = 0; i < ct1_w.size(); ++i) ct1_w[i] = float(int(i % 19) - 9) * .03125F;
  for (std::size_t i = 0; i < ct1_b.size(); ++i) ct1_b[i] = float(int(i % 5) - 2) * .0625F;
  std::vector<float> concat(std::size_t(total_channels) * plane);
  for (int source = 0; source < source_count; ++source) {
    std::memcpy(concat.data() + std::size_t(source * channels_each) * plane,
                sources[static_cast<std::size_t>(source)].data(),
                sources[static_cast<std::size_t>(source)].size() * sizeof(float));
  }
  std::vector<float> conv_ref(std::size_t(conv_oc) * conv_h * conv_width);
  for (int oc = 0; oc < conv_oc; ++oc) {
    for (int y = 0; y < conv_h; ++y) for (int x = 0; x < conv_width; ++x) {
      float value = conv_b[static_cast<std::size_t>(oc)];
      for (int ic = 0; ic < total_channels; ++ic) {
        const float* src = concat.data() + std::size_t(ic) * plane;
        const float* filter =
            conv_weights.data() + (std::size_t(oc) * total_channels + ic) * 9;
        for (int ky = 0; ky < 3; ++ky) for (int kx = 0; kx < 3; ++kx) {
          const int iy = y + ky - pad, ix = x + kx - pad;
          if (iy >= 0 && iy < height && ix >= 0 && ix < width)
            value += src[std::size_t(iy) * width + ix] * filter[ky * 3 + kx];
        }
      }
      conv_ref[(std::size_t(oc) * conv_h + y) * conv_width + x] = std::max(value, 0.F);
    }
  }
  std::vector<float> mid_ref(std::size_t(conv_oc) * mid_h * mid_w, 0.F);
  const std::size_t conv_plane = std::size_t(conv_h) * conv_width;
  const std::size_t mid_plane = std::size_t(mid_h) * mid_w;
  for (int oc = 0; oc < conv_oc; ++oc) {
    float* out = mid_ref.data() + std::size_t(oc) * mid_plane;
    std::fill_n(out, mid_plane, ct0_b[static_cast<std::size_t>(oc)]);
    for (int ic = 0; ic < conv_oc; ++ic) {
      const float* src = conv_ref.data() + std::size_t(ic) * conv_plane;
      const float* w = ct0_w.data() + (std::size_t(ic) * conv_oc + oc) * 4;
      for (int y = 0; y < conv_h; ++y) {
        const float* row = src + std::size_t(y) * conv_width;
        float* dst0 = out + std::size_t(2 * y) * mid_w;
        float* dst1 = dst0 + mid_w;
        for (int x = 0; x < conv_width; ++x) {
          const float value = row[x];
          dst0[2 * x] += value * w[0]; dst0[2 * x + 1] += value * w[1];
          dst1[2 * x] += value * w[2]; dst1[2 * x + 1] += value * w[3];
        }
      }
    }
    for (std::size_t i = 0; i < mid_plane; ++i) out[i] = std::max(out[i], 0.F);
  }
  std::vector<float> expected(std::size_t(final_oc) * out_h * out_w, 0.F);
  const std::size_t out_plane = std::size_t(out_h) * out_w;
  for (int oc = 0; oc < final_oc; ++oc) {
    float* out = expected.data() + std::size_t(oc) * out_plane;
    std::fill_n(out, out_plane, ct1_b[static_cast<std::size_t>(oc)]);
    for (int ic = 0; ic < conv_oc; ++ic) {
      const float* src = mid_ref.data() + std::size_t(ic) * mid_plane;
      const float* w = ct1_w.data() + (std::size_t(ic) * final_oc + oc) * 4;
      for (int y = 0; y < mid_h; ++y) {
        const float* row = src + std::size_t(y) * mid_w;
        float* dst0 = out + std::size_t(2 * y) * out_w;
        float* dst1 = dst0 + out_w;
        for (int x = 0; x < mid_w; ++x) {
          const float value = row[x];
          dst0[2 * x] += value * w[0]; dst0[2 * x + 1] += value * w[1];
          dst1[2 * x] += value * w[2]; dst1[2 * x + 1] += value * w[3];
        }
      }
    }
    for (std::size_t i = 0; i < out_plane; ++i) out[i] = 1.F / (1.F + std::exp(-out[i]));
  }
  std::vector<float> conv(std::size_t(conv_oc) * conv_h * conv_width);
  std::vector<float> mid(std::size_t(conv_oc) * mid_h * mid_w);
  std::vector<float> output(expected.size());
  ppocr::detail::kernels::ConcatChannelConv2d(
      conv.data(), source_ptrs.data(), source_channels.data(), source_count, conv_weights.data(),
      conv_b.data(), conv_oc, height, width, conv_h, conv_width, 3, 3, 1, 1, pad, pad, true);
  ppocr::detail::kernels::ConvTranspose2x2(mid.data(), conv.data(), ct0_w.data(), ct0_b.data(),
                                           conv_oc, conv_oc, conv_h, conv_width, 1);
  ppocr::detail::kernels::ConvTranspose2x2(output.data(), mid.data(), ct1_w.data(),
                                           ct1_b.data(), conv_oc, final_oc, mid_h, mid_w, 2);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 4e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "concat-conv dual transpose mismatch H=" << height << " W=" << width
                << " index=" << i << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyDetStemFromNchw(int height, int width) {
  const int c0_h = (height + 2 - 3) / 2 + 1;
  const int c0_w = (width + 2 - 3) / 2 + 1;
  const int stem_h = (c0_h + 2 - 3) / 2 + 1;
  const int stem_w = (c0_w + 2 - 3) / 2 + 1;
  const int oc0 = 16, oc1 = 8, oc2 = 16, ocs = 16;
  std::vector<float> rgb(std::size_t(3) * height * width);
  std::vector<float> w0(std::size_t(oc0) * 3 * 9), b0(oc0);
  std::vector<float> w1(std::size_t(oc1) * oc0 * 4), b1(oc1);
  std::vector<float> w2(std::size_t(oc2) * oc1 * 4), b2(oc2);
  std::vector<float> ws(std::size_t(ocs) * (oc0 + oc2) * 9), bs(ocs);
  for (std::size_t i = 0; i < rgb.size(); ++i) rgb[i] = float(int(i % 31) - 15) * .03125F;
  for (std::size_t i = 0; i < w0.size(); ++i) w0[i] = float(int(i % 29) - 14) * .015625F;
  for (std::size_t i = 0; i < b0.size(); ++i) b0[i] = float(int(i % 13) - 6) * .0625F;
  for (std::size_t i = 0; i < w1.size(); ++i) w1[i] = float(int(i % 23) - 11) * .015625F;
  for (std::size_t i = 0; i < b1.size(); ++i) b1[i] = float(int(i % 11) - 5) * .0625F;
  for (std::size_t i = 0; i < w2.size(); ++i) w2[i] = float(int(i % 19) - 9) * .015625F;
  for (std::size_t i = 0; i < b2.size(); ++i) b2[i] = float(int(i % 7) - 3) * .0625F;
  for (std::size_t i = 0; i < ws.size(); ++i) ws[i] = float(int(i % 17) - 8) * .015625F;
  for (std::size_t i = 0; i < bs.size(); ++i) bs[i] = float(int(i % 5) - 2) * .0625F;
  std::vector<float> c0(std::size_t(oc0) * c0_h * c0_w);
  std::vector<float> c1(std::size_t(oc1) * c0_h * c0_w);
  std::vector<float> c2(std::size_t(oc2) * c0_h * c0_w);
  std::vector<float> pooled(c0.size());
  std::vector<float> expected(std::size_t(ocs) * stem_h * stem_w);
  std::vector<float> output(expected.size(), -99.F);
  ppocr::detail::kernels::Conv2d(c0.data(), rgb.data(), w0.data(), b0.data(), 3, oc0, height,
                                 width, c0_h, c0_w, 3, 3, 2, 2, 1, 1, true);
  ppocr::detail::kernels::Conv2d(c1.data(), c0.data(), w1.data(), b1.data(), oc0, oc1, c0_h,
                                 c0_w, c0_h, c0_w, 2, 2, 1, 1, 0, 0, true);
  ppocr::detail::kernels::Conv2d(c2.data(), c1.data(), w2.data(), b2.data(), oc1, oc2, c0_h,
                                 c0_w, c0_h, c0_w, 2, 2, 1, 1, 0, 0, true);
  ppocr::detail::kernels::MaxPool2x2Same(pooled.data(), c0.data(), std::size_t(oc0), c0_h, c0_w);
  const float* sources[2] = {pooled.data(), c2.data()};
  const int chans[2] = {oc0, oc2};
  ppocr::detail::kernels::ConcatChannelConv2d(expected.data(), sources, chans, 2, ws.data(),
                                              bs.data(), ocs, c0_h, c0_w, stem_h, stem_w, 3, 3,
                                              2, 2, 1, 1, true);
  ppocr::detail::kernels::DetStemFromNchw(output.data(), rgb.data(), w0.data(), b0.data(), oc0,
                                          w1.data(), b1.data(), oc1, w2.data(), b2.data(), oc2,
                                          ws.data(), bs.data(), ocs, height, width, true, true);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 3e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "det stem mismatch H=" << height << " W=" << width << " index=" << i
                << " got=" << output[i] << " expected=" << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyConv2x2SameUpper(int input_channels, int output_channels, int height,
                            int width, bool relu) {
  const std::size_t plane = std::size_t(height) * width;
  std::vector<float> input(std::size_t(input_channels) * plane);
  std::vector<float> weights(std::size_t(output_channels) * input_channels * 4);
  std::vector<float> bias(static_cast<std::size_t>(output_channels));
  std::vector<float> output(std::size_t(output_channels) * plane);
  std::vector<float> expected(output.size());
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 41) - 20) * .03125F;
  for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = float(int(i % 29) - 14) * .015625F;
  for (std::size_t i = 0; i < bias.size(); ++i) bias[i] = float(int(i % 13) - 6) * .0625F;
  for (int output_channel = 0; output_channel < output_channels; ++output_channel) {
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
      float value = bias[std::size_t(output_channel)];
      for (int input_channel = 0; input_channel < input_channels; ++input_channel) {
        const float* source = input.data() + std::size_t(input_channel) * plane;
        const float* kernel = weights.data() +
            (std::size_t(output_channel) * input_channels + input_channel) * 4;
        for (int ky = 0; ky < 2; ++ky) for (int kx = 0; kx < 2; ++kx) {
          const int iy = y + ky, ix = x + kx;
          if (iy < height && ix < width) {
            value += source[std::size_t(iy) * width + ix] * kernel[ky * 2 + kx];
          }
        }
      }
      expected[(std::size_t(output_channel) * height + y) * width + x] =
          relu ? std::max(value, 0.F) : value;
    }
  }
  ppocr::detail::kernels::Conv2d(output.data(), input.data(), weights.data(), bias.data(),
                                  input_channels, output_channels, height, width,
                                  height, width, 2, 2, 1, 1, 0, 0, relu);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 3e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "2x2 SAME_UPPER mismatch C=" << input_channels << " M="
                << output_channels << " H=" << height << " W=" << width
                << " relu=" << relu << " index=" << i << " got=" << output[i]
                << " expected=" << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyConvWideTile(int input_channels, int output_channels, int height,
                        int width, int kernel) {
  const int pad = kernel / 2;
  const std::size_t input_plane = std::size_t(height) * width;
  std::vector<float> input(std::size_t(input_channels) * input_plane);
  std::vector<float> weights(std::size_t(output_channels) * input_channels * kernel * kernel);
  std::vector<float> bias(static_cast<std::size_t>(output_channels));
  std::vector<float> output(std::size_t(output_channels) * input_plane);
  std::vector<float> expected(output.size());
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 31) - 15) * .03125F;
  for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = float(int(i % 29) - 14) * .015625F;
  for (std::size_t i = 0; i < bias.size(); ++i) bias[i] = float(int(i % 13) - 6) * .0625F;
  for (int output_channel = 0; output_channel < output_channels; ++output_channel) {
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
      float value = bias[std::size_t(output_channel)];
      for (int input_channel = 0; input_channel < input_channels; ++input_channel) {
        const float* plane = input.data() + std::size_t(input_channel) * input_plane;
        const float* filter = weights.data() +
            (std::size_t(output_channel) * input_channels + input_channel) * kernel * kernel;
        for (int ky = 0; ky < kernel; ++ky) for (int kx = 0; kx < kernel; ++kx) {
          const int iy = y + ky - pad, ix = x + kx - pad;
          if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
            value += plane[std::size_t(iy) * width + ix] * filter[ky * kernel + kx];
          }
        }
      }
      expected[(std::size_t(output_channel) * height + y) * width + x] = value;
    }
  }
  ppocr::detail::kernels::Conv2d(output.data(), input.data(), weights.data(), bias.data(),
                                  input_channels, output_channels, height, width,
                                  height, width, kernel, kernel, 1, 1, pad, pad, false);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 3e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "wide convolution mismatch K=" << kernel << " C=" << input_channels
                << " M=" << output_channels << " H=" << height << " W=" << width
                << " index=" << i << " got=" << output[i] << " expected=" << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyPointwiseConv(int input_channels, int output_channels, std::size_t plane) {
  std::vector<float> input(std::size_t(input_channels) * plane);
  std::vector<float> weights(std::size_t(output_channels) * input_channels);
  std::vector<float> bias(static_cast<std::size_t>(output_channels));
  std::vector<float> expected(std::size_t(output_channels) * plane);
  std::vector<float> output(expected.size());
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 37) - 18) * .03125F;
  for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = float(int(i % 31) - 15) * .015625F;
  for (std::size_t i = 0; i < bias.size(); ++i) bias[i] = float(int(i % 17) - 8) * .0625F;
  for (int output_channel = 0; output_channel < output_channels; ++output_channel) {
    float* dst = expected.data() + std::size_t(output_channel) * plane;
    const float* filter = weights.data() + std::size_t(output_channel) * input_channels;
    const float base = bias[static_cast<std::size_t>(output_channel)];
    for (std::size_t index = 0; index < plane; ++index) {
      float value = base;
      for (int input_channel = 0; input_channel < input_channels; ++input_channel) {
        value += filter[input_channel] *
            input[std::size_t(input_channel) * plane + index];
      }
      dst[index] = value;
    }
  }
  ppocr::detail::kernels::PointwiseConv(output.data(), input.data(), weights.data(),
                                         bias.data(), output_channels, input_channels, plane);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 3e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "pointwise conv mismatch C=" << input_channels << " M="
                << output_channels << " plane=" << plane << " index=" << i
                << " got=" << output[i] << " expected=" << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyPointwiseConvRelu(int input_channels, int output_channels, std::size_t plane) {
  std::vector<float> input(std::size_t(input_channels) * plane);
  std::vector<float> weights(std::size_t(output_channels) * input_channels);
  std::vector<float> bias(static_cast<std::size_t>(output_channels));
  std::vector<float> expected(std::size_t(output_channels) * plane);
  std::vector<float> output(expected.size());
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 37) - 18) * .03125F;
  for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = float(int(i % 31) - 15) * .015625F;
  for (std::size_t i = 0; i < bias.size(); ++i) bias[i] = float(int(i % 17) - 8) * .0625F;
  ppocr::detail::kernels::PointwiseConv(expected.data(), input.data(), weights.data(),
                                         bias.data(), output_channels, input_channels, plane);
  ppocr::detail::kernels::Relu(expected.data(), expected.data(), expected.size());
  ppocr::detail::kernels::PointwiseConvRelu(output.data(), input.data(), weights.data(),
                                            bias.data(), output_channels, input_channels, plane);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 3e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "pointwise conv relu mismatch C=" << input_channels << " M="
                << output_channels << " plane=" << plane << " index=" << i
                << " got=" << output[i] << " expected=" << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyExpandGeluProjectAdd(int channels, int hidden, std::size_t plane) {
  std::vector<float> input(std::size_t(channels) * plane);
  std::vector<float> expand_weights(std::size_t(hidden) * channels);
  std::vector<float> expand_bias(static_cast<std::size_t>(hidden));
  std::vector<float> project_weights(std::size_t(channels) * hidden);
  std::vector<float> project_bias(static_cast<std::size_t>(channels));
  std::vector<float> expected(input.size());
  std::vector<float> output(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 37) - 18) * .03125F;
  for (std::size_t i = 0; i < expand_weights.size(); ++i)
    expand_weights[i] = float(int(i % 31) - 15) * .015625F;
  for (std::size_t i = 0; i < expand_bias.size(); ++i)
    expand_bias[i] = float(int(i % 17) - 8) * .0625F;
  for (std::size_t i = 0; i < project_weights.size(); ++i)
    project_weights[i] = float(int(i % 29) - 14) * .015625F;
  for (std::size_t i = 0; i < project_bias.size(); ++i)
    project_bias[i] = float(int(i % 13) - 6) * .0625F;
  constexpr float inv_sqrt2 = 0.7071067811865475244F;
  std::vector<float> hidden_act(static_cast<std::size_t>(hidden));
  for (std::size_t spatial = 0; spatial < plane; ++spatial) {
    for (int hidden_channel = 0; hidden_channel < hidden; ++hidden_channel) {
      float sum = expand_bias[static_cast<std::size_t>(hidden_channel)];
      const float* filter = expand_weights.data() + std::size_t(hidden_channel) * channels;
      for (int channel = 0; channel < channels; ++channel) {
        sum += filter[channel] * input[std::size_t(channel) * plane + spatial];
      }
      hidden_act[static_cast<std::size_t>(hidden_channel)] =
          sum * .5F * (1.F + std::erf(sum * inv_sqrt2));
    }
    for (int channel = 0; channel < channels; ++channel) {
      float sum = project_bias[static_cast<std::size_t>(channel)] +
                  input[std::size_t(channel) * plane + spatial];
      const float* filter = project_weights.data() + std::size_t(channel) * hidden;
      for (int hidden_channel = 0; hidden_channel < hidden; ++hidden_channel) {
        sum += filter[hidden_channel] * hidden_act[static_cast<std::size_t>(hidden_channel)];
      }
      expected[std::size_t(channel) * plane + spatial] = sum;
    }
  }
  ppocr::detail::kernels::ExpandGeluProjectAdd(
      output.data(), input.data(), expand_weights.data(), expand_bias.data(),
      project_weights.data(), project_bias.data(), channels, hidden, plane);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 2e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "expand-GELU-project-add mismatch C=" << channels << " H="
                << hidden << " plane=" << plane << " index=" << i
                << " got=" << output[i] << " expected=" << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyPointwiseResidualBatch(int batches, int input_channels,
                                  int output_channels, std::size_t plane,
                                  int activation) {
  const std::size_t input_batch = std::size_t(input_channels) * plane;
  const std::size_t output_batch = std::size_t(output_channels) * plane;
  std::vector<float> input(std::size_t(batches) * input_batch);
  std::vector<float> weights(std::size_t(output_channels) * input_channels);
  std::vector<float> bias(static_cast<std::size_t>(output_channels));
  std::vector<float> residual(std::size_t(batches) * output_batch);
  std::vector<float> expected(residual.size());
  std::vector<float> output(residual.size());
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 37) - 18) * .03125F;
  for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = float(int(i % 31) - 15) * .015625F;
  for (std::size_t i = 0; i < bias.size(); ++i) bias[i] = float(int(i % 17) - 8) * .0625F;
  for (std::size_t i = 0; i < residual.size(); ++i) residual[i] = float(int(i % 29) - 14) * .03125F;

  for (int batch = 0; batch < batches; ++batch) {
    float* dst = expected.data() + std::size_t(batch) * output_batch;
    const float* src = input.data() + std::size_t(batch) * input_batch;
    const float* add = residual.data() + std::size_t(batch) * output_batch;
    if (activation == 0) {
      ppocr::detail::kernels::PointwiseConvAdd(dst, src, weights.data(), bias.data(), add,
                                                output_channels, input_channels, plane);
    } else if (activation == 1) {
      ppocr::detail::kernels::PointwiseConvAddRelu(dst, src, weights.data(), bias.data(), add,
                                                    output_channels, input_channels, plane);
    } else {
      ppocr::detail::kernels::PointwiseConvAddSwish(dst, src, weights.data(), bias.data(), add,
                                                     output_channels, input_channels, plane);
    }
  }
  if (activation == 0) {
    ppocr::detail::kernels::PointwiseConvAddBatch(output.data(), input.data(), weights.data(),
                                                   bias.data(), residual.data(), batches,
                                                   output_channels, input_channels, plane);
  } else if (activation == 1) {
    ppocr::detail::kernels::PointwiseConvAddReluBatch(output.data(), input.data(), weights.data(),
                                                       bias.data(), residual.data(), batches,
                                                       output_channels, input_channels, plane);
  } else {
    ppocr::detail::kernels::PointwiseConvAddSwishBatch(output.data(), input.data(), weights.data(),
                                                        bias.data(), residual.data(), batches,
                                                        output_channels, input_channels, plane);
  }
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 3e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "batched residual pointwise mismatch N=" << batches << " C="
                << input_channels << " M=" << output_channels << " plane=" << plane
                << " activation=" << activation << " index=" << i << " got=" << output[i]
                << " expected=" << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyDepthwisePointwiseFused(int channels, int output_channels, int height,
                                   int width, int activation) {
  const std::size_t in_n = std::size_t(channels) * height * width;
  const std::size_t out_n = std::size_t(output_channels) * height * width;
  std::vector<float> input(in_n), dw_w(std::size_t(channels) * 9);
  std::vector<float> dw_b(static_cast<std::size_t>(channels));
  std::vector<float> pw_w(std::size_t(output_channels) * channels);
  std::vector<float> pw_b(static_cast<std::size_t>(output_channels));
  std::vector<float> intermediate(in_n), expected(out_n), output(out_n, -99.F);
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 41) - 20) * .03125F;
  for (std::size_t i = 0; i < dw_w.size(); ++i) dw_w[i] = float(int(i % 23) - 11) * .03125F;
  for (std::size_t i = 0; i < dw_b.size(); ++i) dw_b[i] = float(int(i % 13) - 6) * .0625F;
  for (std::size_t i = 0; i < pw_w.size(); ++i) pw_w[i] = float(int(i % 31) - 15) * .015625F;
  for (std::size_t i = 0; i < pw_b.size(); ++i) pw_b[i] = float(int(i % 17) - 8) * .0625F;
  ppocr::detail::kernels::DepthwiseConv(
      intermediate.data(), input.data(), dw_w.data(), dw_b.data(), channels, height,
      width, height, width, 3, 3, 1, 1, 1, 1);
  if (activation == 1) {
    ppocr::detail::kernels::PointwiseConvRelu(expected.data(), intermediate.data(),
                                              pw_w.data(), pw_b.data(), output_channels,
                                              channels, std::size_t(height) * width);
  } else if (activation == 2) {
    ppocr::detail::kernels::PointwiseConvHardSwish(expected.data(), intermediate.data(),
                                                   pw_w.data(), pw_b.data(), output_channels,
                                                   channels, std::size_t(height) * width);
  } else {
    ppocr::detail::kernels::PointwiseConv(expected.data(), intermediate.data(), pw_w.data(),
                                          pw_b.data(), output_channels, channels,
                                          std::size_t(height) * width);
    if (activation == 3) {
      ppocr::detail::kernels::Gelu(expected.data(), expected.data(), expected.size(),
                                   expected.size());
    }
  }
  const bool fused = ppocr::detail::kernels::DepthwisePointwiseConvFused(
      output.data(), input.data(), dw_w.data(), dw_b.data(), pw_w.data(), pw_b.data(),
      1, channels, output_channels, height, width, 3, 3, 1, 1, 1, 1, activation);
  if (!fused) {
    // Scalar/AVX2 hosts skip the fused AVX-512 path; the two-pass kernels
    // remain the production fallback and are already covered elsewhere.
    return true;
  }
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 5e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "fused DW+PW mismatch C=" << channels << " M=" << output_channels
                << " H=" << height << " W=" << width << " act=" << activation
                << " index=" << i << " got=" << output[i]
                << " expected=" << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyDepthwisePointwiseFused5x5(int channels, int output_channels, int height,
                                      int width, int activation) {
  const std::size_t in_n = std::size_t(channels) * height * width;
  const std::size_t out_n = std::size_t(output_channels) * height * width;
  std::vector<float> input(in_n), dw_w(std::size_t(channels) * 25);
  std::vector<float> dw_b(static_cast<std::size_t>(channels));
  std::vector<float> pw_w(std::size_t(output_channels) * channels);
  std::vector<float> pw_b(static_cast<std::size_t>(output_channels));
  std::vector<float> intermediate(in_n), expected(out_n), output(out_n, -99.F);
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 41) - 20) * .03125F;
  for (std::size_t i = 0; i < dw_w.size(); ++i) dw_w[i] = float(int(i % 23) - 11) * .03125F;
  for (std::size_t i = 0; i < dw_b.size(); ++i) dw_b[i] = float(int(i % 13) - 6) * .0625F;
  for (std::size_t i = 0; i < pw_w.size(); ++i) pw_w[i] = float(int(i % 31) - 15) * .015625F;
  for (std::size_t i = 0; i < pw_b.size(); ++i) pw_b[i] = float(int(i % 17) - 8) * .0625F;
  ppocr::detail::kernels::DepthwiseConv(
      intermediate.data(), input.data(), dw_w.data(), dw_b.data(), channels, height,
      width, height, width, 5, 5, 1, 1, 2, 2);
  if (activation == 1) {
    ppocr::detail::kernels::PointwiseConvRelu(expected.data(), intermediate.data(),
                                              pw_w.data(), pw_b.data(), output_channels,
                                              channels, std::size_t(height) * width);
  } else if (activation == 2) {
    ppocr::detail::kernels::PointwiseConvHardSwish(expected.data(), intermediate.data(),
                                                   pw_w.data(), pw_b.data(), output_channels,
                                                   channels, std::size_t(height) * width);
  } else {
    ppocr::detail::kernels::PointwiseConv(expected.data(), intermediate.data(), pw_w.data(),
                                          pw_b.data(), output_channels, channels,
                                          std::size_t(height) * width);
    if (activation == 3) {
      ppocr::detail::kernels::Gelu(expected.data(), expected.data(), expected.size(),
                                   expected.size());
    }
  }
  const bool fused = ppocr::detail::kernels::DepthwisePointwiseConvFused(
      output.data(), input.data(), dw_w.data(), dw_b.data(), pw_w.data(), pw_b.data(),
      1, channels, output_channels, height, width, 5, 5, 1, 1, 2, 2, activation);
  if (!fused) {
    return true;
  }
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 5e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "fused 5x5 DW+PW mismatch C=" << channels << " M=" << output_channels
                << " H=" << height << " W=" << width << " act=" << activation
                << " index=" << i << " got=" << output[i]
                << " expected=" << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyPointwiseDepthwise(int input_channels, int output_channels, int height,
                              int width, bool relu) {
  // Tiled 8-OC kernel is ENABLE-only after the 20-run miss versus two-node.
#if defined(_WIN32)
  _putenv("PPOCR_ENABLE_POINTWISE_DEPTHWISE=1");
#else
  setenv("PPOCR_ENABLE_POINTWISE_DEPTHWISE", "1", 1);
#endif
  const std::size_t in_n = std::size_t(input_channels) * height * width;
  const std::size_t out_n = std::size_t(output_channels) * height * width;
  std::vector<float> input(in_n), pw_w(std::size_t(output_channels) * input_channels);
  std::vector<float> pw_b(static_cast<std::size_t>(output_channels));
  std::vector<float> dw_w(std::size_t(output_channels) * 9);
  std::vector<float> dw_b(static_cast<std::size_t>(output_channels));
  std::vector<float> intermediate(out_n), expected(out_n), output(out_n, -99.F);
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 41) - 20) * .03125F;
  for (std::size_t i = 0; i < pw_w.size(); ++i) pw_w[i] = float(int(i % 31) - 15) * .015625F;
  for (std::size_t i = 0; i < pw_b.size(); ++i) pw_b[i] = float(int(i % 17) - 8) * .0625F;
  for (std::size_t i = 0; i < dw_w.size(); ++i) dw_w[i] = float(int(i % 23) - 11) * .03125F;
  for (std::size_t i = 0; i < dw_b.size(); ++i) dw_b[i] = float(int(i % 13) - 6) * .0625F;
  if (relu) {
    ppocr::detail::kernels::PointwiseConvRelu(
        intermediate.data(), input.data(), pw_w.data(), pw_b.data(), output_channels,
        input_channels, std::size_t(height) * width);
  } else {
    ppocr::detail::kernels::PointwiseConv(
        intermediate.data(), input.data(), pw_w.data(), pw_b.data(), output_channels,
        input_channels, std::size_t(height) * width);
  }
  ppocr::detail::kernels::DepthwiseConv(
      expected.data(), intermediate.data(), dw_w.data(), dw_b.data(), output_channels,
      height, width, height, width, 3, 3, 1, 1, 1, 1);
  const bool fused = ppocr::detail::kernels::PointwiseDepthwiseConvFused(
      output.data(), input.data(), pw_w.data(), pw_b.data(), dw_w.data(), dw_b.data(),
      1, input_channels, output_channels, height, width, relu);
  if (!fused) {
    if (height >= 32 && std::getenv("PPOCR_DISABLE_POINTWISE_DEPTHWISE") == nullptr) {
      std::cerr << "fused PW+DW did not run IC=" << input_channels
                << " OC=" << output_channels << " H=" << height << " W=" << width
                << " relu=" << relu << '\n';
      return false;
    }
    return true;
  }
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 5e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "fused PW+DW mismatch IC=" << input_channels
                << " OC=" << output_channels << " H=" << height << " W=" << width
                << " relu=" << relu << " index=" << i << " got=" << output[i]
                << " expected=" << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyDepthwiseExpandGelu(int channels, int hidden, int height, int width) {
  // Opt-in C*W tiled kernel is ENABLE-only; force it so smoke drives the
  // shipped AVX-512 function rather than the two-pass fallback.
#if defined(_WIN32)
  _putenv("PPOCR_ENABLE_AVX512_DW_EXPAND_GELU=1");
#else
  setenv("PPOCR_ENABLE_AVX512_DW_EXPAND_GELU", "1", 1);
#endif
  const std::size_t n = std::size_t(channels) * height * width;
  std::vector<float> input(n), dw_w(std::size_t(channels) * 9);
  std::vector<float> dw_b(static_cast<std::size_t>(channels));
  std::vector<float> expand_w(std::size_t(hidden) * channels);
  std::vector<float> expand_b(static_cast<std::size_t>(hidden));
  std::vector<float> project_w(std::size_t(channels) * hidden);
  std::vector<float> project_b(static_cast<std::size_t>(channels));
  std::vector<float> intermediate(n), expected(n), output(n, -99.F);
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 41) - 20) * .03125F;
  for (std::size_t i = 0; i < dw_w.size(); ++i) dw_w[i] = float(int(i % 23) - 11) * .03125F;
  for (std::size_t i = 0; i < dw_b.size(); ++i) dw_b[i] = float(int(i % 13) - 6) * .0625F;
  for (std::size_t i = 0; i < expand_w.size(); ++i) expand_w[i] = float(int(i % 31) - 15) * .015625F;
  for (std::size_t i = 0; i < expand_b.size(); ++i) expand_b[i] = float(int(i % 17) - 8) * .0625F;
  for (std::size_t i = 0; i < project_w.size(); ++i) project_w[i] = float(int(i % 29) - 14) * .015625F;
  for (std::size_t i = 0; i < project_b.size(); ++i) project_b[i] = float(int(i % 11) - 5) * .0625F;
  ppocr::detail::kernels::DepthwiseConv(
      intermediate.data(), input.data(), dw_w.data(), dw_b.data(), channels, height,
      width, height, width, 3, 3, 1, 1, 1, 1);
  ppocr::detail::kernels::ExpandGeluProjectAdd(
      expected.data(), intermediate.data(), expand_w.data(), expand_b.data(),
      project_w.data(), project_b.data(), channels, hidden,
      std::size_t(height) * width);
  ppocr::detail::kernels::DepthwiseExpandGeluProjectAdd(
      output.data(), input.data(), dw_w.data(), dw_b.data(), expand_w.data(),
      expand_b.data(), project_w.data(), project_b.data(), 1, channels, hidden,
      height, width, 3, 3, 1, 1, 1, 1);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 8e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "fused DW+ExpandGelu mismatch C=" << channels << " hidden=" << hidden
                << " H=" << height << " W=" << width << " index=" << i
                << " got=" << output[i] << " expected=" << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyDepthwiseBatch(int batches, int channels, int input_height,
                          int input_width, int kernel, int stride, int pad) {
  const int output_height = (input_height + 2 * pad - kernel) / stride + 1;
  const int output_width = (input_width + 2 * pad - kernel) / stride + 1;
  const std::size_t input_batch = std::size_t(channels) * input_height * input_width;
  const std::size_t output_batch = std::size_t(channels) * output_height * output_width;
  std::vector<float> input(std::size_t(batches) * input_batch);
  std::vector<float> weights(std::size_t(channels) * kernel * kernel);
  std::vector<float> bias(static_cast<std::size_t>(channels));
  std::vector<float> expected(std::size_t(batches) * output_batch);
  std::vector<float> output(expected.size());
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 41) - 20) * .03125F;
  for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = float(int(i % 23) - 11) * .03125F;
  for (std::size_t i = 0; i < bias.size(); ++i) bias[i] = float(int(i % 13) - 6) * .0625F;
  for (int batch = 0; batch < batches; ++batch) {
    ppocr::detail::kernels::DepthwiseConv(
        expected.data() + std::size_t(batch) * output_batch,
        input.data() + std::size_t(batch) * input_batch, weights.data(), bias.data(), channels,
        input_height, input_width, output_height, output_width, kernel, kernel, stride, stride,
        pad, pad);
  }
  ppocr::detail::kernels::DepthwiseConvBatch(
      output.data(), input.data(), weights.data(), bias.data(), batches, channels,
      input_height, input_width, output_height, output_width, kernel, kernel, stride, stride,
      pad, pad);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 3e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "batched depthwise mismatch N=" << batches << " C=" << channels
                << " H=" << input_height << " W=" << input_width << " K=" << kernel
                << " stride=" << stride << " index=" << i << " got=" << output[i]
                << " expected=" << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyFoldedDepthwiseHardSwish(int channels, int height, int width, int kernel_h,
                                    int kernel_w, int pad_h, int pad_w) {
  const int out_h = height + 2 * pad_h - kernel_h + 1;
  const int out_w = width + 2 * pad_w - kernel_w + 1;
  if (out_h <= 0 || out_w <= 0) return false;
  const std::size_t in_n = std::size_t(channels) * height * width;
  const std::size_t out_n = std::size_t(channels) * out_h * out_w;
  std::vector<float> input(in_n), weights(std::size_t(channels) * kernel_h * kernel_w);
  std::vector<float> bias(static_cast<std::size_t>(channels));
  std::vector<float> factor(static_cast<std::size_t>(channels));
  std::vector<float> offset(static_cast<std::size_t>(channels));
  std::vector<float> expected(out_n), fused(out_n), folded_w(weights.size()), folded_b(bias.size());
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 41) - 20) * .03125F;
  for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = float(int(i % 23) - 11) * .03125F;
  for (int i = 0; i < channels; ++i) {
    bias[static_cast<std::size_t>(i)] = float(int(i % 13) - 6) * .0625F;
    factor[static_cast<std::size_t>(i)] = 0.75F + float(i % 5) * .05F;
    offset[static_cast<std::size_t>(i)] = float(int(i % 7) - 3) * .03125F;
  }
  ppocr::detail::kernels::DepthwiseConv(expected.data(), input.data(), weights.data(),
                                        bias.data(), channels, height, width, out_h, out_w,
                                        kernel_h, kernel_w, 1, 1, pad_h, pad_w);
  ppocr::detail::kernels::BatchNormAffine(expected.data(), expected.data(), factor.data(),
                                          offset.data(), 1, channels, std::size_t(out_h) * out_w);
  ppocr::detail::kernels::HardSwish(expected.data(), expected.data(), expected.size());
  const int per = kernel_h * kernel_w;
  for (int c = 0; c < channels; ++c) {
    for (int k = 0; k < per; ++k)
      folded_w[std::size_t(c) * per + k] = weights[std::size_t(c) * per + k] * factor[c];
    folded_b[static_cast<std::size_t>(c)] =
        bias[static_cast<std::size_t>(c)] * factor[c] + offset[c];
  }
  ppocr::detail::kernels::DepthwiseConv(fused.data(), input.data(), folded_w.data(),
                                        folded_b.data(), channels, height, width, out_h, out_w,
                                        kernel_h, kernel_w, 1, 1, pad_h, pad_w);
  ppocr::detail::kernels::HardSwish(fused.data(), fused.data(), fused.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const float tolerance = 4e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(fused[i] - expected[i]) > tolerance) {
      std::cerr << "folded dw+hswish mismatch C=" << channels << " " << height << "x" << width
                << " k=" << kernel_h << "x" << kernel_w << " index=" << i << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyDepthwiseScalar(int channels, int height, int width, int kernel, int pad) {
  const int output_height = height;
  const int output_width = width;
  const std::size_t plane = std::size_t(height) * width;
  std::vector<float> input(std::size_t(channels) * plane);
  std::vector<float> weights(std::size_t(channels) * kernel * kernel);
  std::vector<float> bias(static_cast<std::size_t>(channels));
  std::vector<float> expected(input.size()), output(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 41) - 20) * .03125F;
  for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = float(int(i % 23) - 11) * .03125F;
  for (int i = 0; i < channels; ++i) bias[static_cast<std::size_t>(i)] = float(int(i % 13) - 6) * .0625F;
  for (int channel = 0; channel < channels; ++channel) {
    const float* in = input.data() + std::size_t(channel) * plane;
    const float* filter = weights.data() + std::size_t(channel) * kernel * kernel;
    float* out = expected.data() + std::size_t(channel) * plane;
    const float base = bias[static_cast<std::size_t>(channel)];
    for (int y = 0; y < output_height; ++y) {
      for (int x = 0; x < output_width; ++x) {
        float sum = base;
        for (int ky = 0; ky < kernel; ++ky) {
          const int iy = y + ky - pad;
          if (iy < 0 || iy >= height) continue;
          for (int kx = 0; kx < kernel; ++kx) {
            const int ix = x + kx - pad;
            if (ix >= 0 && ix < width)
              sum += in[std::size_t(iy) * width + ix] * filter[ky * kernel + kx];
          }
        }
        out[std::size_t(y) * output_width + x] = sum;
      }
    }
  }
  ppocr::detail::kernels::DepthwiseConv(output.data(), input.data(), weights.data(),
                                        bias.data(), channels, height, width, output_height,
                                        output_width, kernel, kernel, 1, 1, pad, pad);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 3e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "depthwise scalar mismatch C=" << channels << " H=" << height
                << " W=" << width << " K=" << kernel << " index=" << i << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyConvBatch(int batches, int input_channels, int output_channels,
                     int input_height, int input_width, int kernel_height,
                     int kernel_width, int stride_height, int stride_width,
                     int pad_top, int pad_left, bool relu) {
  const int output_height = (input_height + 2 * pad_top - kernel_height) / stride_height + 1;
  const int output_width = (input_width + 2 * pad_left - kernel_width) / stride_width + 1;
  const std::size_t input_batch = std::size_t(input_channels) * input_height * input_width;
  const std::size_t output_batch = std::size_t(output_channels) * output_height * output_width;
  std::vector<float> input(std::size_t(batches) * input_batch);
  std::vector<float> weights(std::size_t(output_channels) * input_channels * kernel_height * kernel_width);
  std::vector<float> bias(static_cast<std::size_t>(output_channels));
  std::vector<float> expected(std::size_t(batches) * output_batch);
  std::vector<float> output(expected.size());
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 43) - 21) * .03125F;
  for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = float(int(i % 37) - 18) * .015625F;
  for (std::size_t i = 0; i < bias.size(); ++i) bias[i] = float(int(i % 17) - 8) * .0625F;
  for (int batch = 0; batch < batches; ++batch) {
    ppocr::detail::kernels::Conv2d(
        expected.data() + std::size_t(batch) * output_batch,
        input.data() + std::size_t(batch) * input_batch, weights.data(), bias.data(),
        input_channels, output_channels, input_height, input_width, output_height,
        output_width, kernel_height, kernel_width, stride_height, stride_width,
        pad_top, pad_left, relu);
  }
  ppocr::detail::kernels::Conv2dBatch(
      output.data(), input.data(), weights.data(), bias.data(), batches, input_channels,
      output_channels, input_height, input_width, output_height, output_width,
      kernel_height, kernel_width, stride_height, stride_width, pad_top, pad_left, relu);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 3e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "batched 3x3 mismatch N=" << batches << " C=" << input_channels
                << " M=" << output_channels << " H=" << input_height << " W="
                << input_width << " K=" << kernel_height << 'x' << kernel_width
                << " stride=" << stride_height << 'x' << stride_width << " relu=" << relu
                << " index=" << i << " got=" << output[i] << " expected=" << expected[i]
                << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyPointwiseHardSwishBatch(int batches, int input_channels,
                                   int output_channels, std::size_t plane) {
  const std::size_t input_batch = std::size_t(input_channels) * plane;
  const std::size_t output_batch = std::size_t(output_channels) * plane;
  std::vector<float> input(std::size_t(batches) * input_batch);
  std::vector<float> weights(std::size_t(output_channels) * input_channels);
  std::vector<float> bias(static_cast<std::size_t>(output_channels));
  std::vector<float> expected(std::size_t(batches) * output_batch);
  std::vector<float> output(expected.size());
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 47) - 23) * .03125F;
  for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = float(int(i % 29) - 14) * .03125F;
  for (std::size_t i = 0; i < bias.size(); ++i) bias[i] = float(int(i % 11) - 5) * .125F;
  for (int batch = 0; batch < batches; ++batch) {
    ppocr::detail::kernels::PointwiseConvHardSwish(
        expected.data() + std::size_t(batch) * output_batch,
        input.data() + std::size_t(batch) * input_batch, weights.data(), bias.data(),
        output_channels, input_channels, plane);
  }
  ppocr::detail::kernels::PointwiseConvHardSwishBatch(
      output.data(), input.data(), weights.data(), bias.data(), batches,
      output_channels, input_channels, plane);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 3e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "batched pointwise hardswish mismatch N=" << batches << " C="
                << input_channels << " M=" << output_channels << " plane=" << plane
                << " index=" << i << " got=" << output[i] << " expected=" << expected[i]
                << '\n';
      return false;
    }
  }
  return true;
}

bool VerifySqueezeExcitation(int batches, int channels, int reduced, int height,
                             int width, bool residual) {
  const std::size_t spatial = std::size_t(height) * width;
  const std::size_t count = std::size_t(batches) * channels * spatial;
  std::vector<float> values(count), first_w(std::size_t(reduced) * channels);
  std::vector<float> first_b(static_cast<std::size_t>(reduced));
  std::vector<float> second_w(std::size_t(channels) * reduced);
  std::vector<float> second_b(static_cast<std::size_t>(channels));
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = float(int(i % 41) - 20) * .03125F;
  for (std::size_t i = 0; i < first_w.size(); ++i) first_w[i] = float(int(i % 23) - 11) * .015625F;
  for (std::size_t i = 0; i < first_b.size(); ++i) first_b[i] = float(int(i % 13) - 6) * .0625F;
  for (std::size_t i = 0; i < second_w.size(); ++i) second_w[i] = float(int(i % 29) - 14) * .015625F;
  for (std::size_t i = 0; i < second_b.size(); ++i) second_b[i] = float(int(i % 11) - 5) * .0625F;
  std::vector<float> expected = values;
  constexpr float alpha = .2F, beta = .5F;
  for (int batch = 0; batch < batches; ++batch) {
    std::vector<float> means(static_cast<std::size_t>(channels), 0.F);
    for (int channel = 0; channel < channels; ++channel) {
      const float* plane = expected.data() +
          (std::size_t(batch) * channels + channel) * spatial;
      float sum = 0.F;
      for (std::size_t i = 0; i < spatial; ++i) sum += plane[i];
      means[static_cast<std::size_t>(channel)] = sum / float(spatial);
    }
    std::vector<float> hidden(static_cast<std::size_t>(reduced));
    for (int output = 0; output < reduced; ++output) {
      float sum = first_b[static_cast<std::size_t>(output)];
      const float* weights = first_w.data() + std::size_t(output) * channels;
      for (int input_channel = 0; input_channel < channels; ++input_channel)
        sum += weights[input_channel] * means[static_cast<std::size_t>(input_channel)];
      hidden[static_cast<std::size_t>(output)] = std::max(0.F, sum);
    }
    for (int output = 0; output < channels; ++output) {
      float sum = second_b[static_cast<std::size_t>(output)];
      const float* weights = second_w.data() + std::size_t(output) * reduced;
      for (int hidden_channel = 0; hidden_channel < reduced; ++hidden_channel)
        sum += weights[hidden_channel] * hidden[static_cast<std::size_t>(hidden_channel)];
      const float gate = std::clamp(alpha * sum + beta, 0.F, 1.F);
      const float scale = residual ? (1.F + gate) : gate;
      float* plane = expected.data() +
          (std::size_t(batch) * channels + output) * spatial;
      for (std::size_t i = 0; i < spatial; ++i) plane[i] *= scale;
    }
  }
  auto output = values;
  ppocr::detail::kernels::SqueezeExcitationGateInplace(
      output.data(), first_w.data(), first_b.data(), second_w.data(), second_b.data(),
      batches, channels, reduced, spatial, alpha, beta, residual);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 5e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "SE gate mismatch residual=" << residual << " C=" << channels
                << " R=" << reduced << " index=" << i << " got=" << output[i]
                << " expected=" << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyConvTransposeScalar(int input_channels, int output_channels, int input_height,
                               int input_width) {
  const int output_height = input_height * 2;
  const int output_width = input_width * 2;
  const std::size_t input_plane = std::size_t(input_channels) * input_height * input_width;
  const std::size_t output_plane = std::size_t(output_channels) * output_height * output_width;
  std::vector<float> input(input_plane), weights(std::size_t(input_channels) * output_channels * 4);
  std::vector<float> bias(static_cast<std::size_t>(output_channels));
  std::vector<float> expected(output_plane), output(output_plane);
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 41) - 20) * .03125F;
  for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = float(int(i % 23) - 11) * .03125F;
  for (int i = 0; i < output_channels; ++i) bias[static_cast<std::size_t>(i)] = float(int(i % 13) - 6) * .0625F;
  const std::size_t in_hw = std::size_t(input_height) * input_width;
  const std::size_t out_hw = std::size_t(output_height) * output_width;
  for (int output_channel = 0; output_channel < output_channels; ++output_channel) {
    float* out = expected.data() + std::size_t(output_channel) * out_hw;
    std::fill_n(out, out_hw, bias[static_cast<std::size_t>(output_channel)]);
    for (int input_channel = 0; input_channel < input_channels; ++input_channel) {
      const float* src = input.data() + std::size_t(input_channel) * in_hw;
      const float* w = weights.data() + (std::size_t(input_channel) * output_channels +
                                         output_channel) * 4;
      for (int y = 0; y < input_height; ++y) {
        const float* row = src + std::size_t(y) * input_width;
        float* dst0 = out + std::size_t(2 * y) * output_width;
        float* dst1 = dst0 + output_width;
        for (int x = 0; x < input_width; ++x) {
          const float value = row[x];
          const int xx = 2 * x;
          dst0[xx] += value * w[0]; dst0[xx + 1] += value * w[1];
          dst1[xx] += value * w[2]; dst1[xx + 1] += value * w[3];
        }
      }
    }
  }
  ppocr::detail::kernels::ConvTranspose2x2(output.data(), input.data(), weights.data(),
                                           bias.data(), input_channels, output_channels,
                                           input_height, input_width);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 3e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "transpose2x2 scalar mismatch C=" << input_channels << " M=" << output_channels
                << " H=" << input_height << " W=" << input_width << " index=" << i << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyConvTransposeAct(int input_channels, int output_channels, int input_height,
                            int input_width, int act) {
  const int output_height = input_height * 2;
  const int output_width = input_width * 2;
  const std::size_t input_plane = std::size_t(input_channels) * input_height * input_width;
  const std::size_t output_count =
      std::size_t(output_channels) * output_height * output_width;
  std::vector<float> input(input_plane), weights(std::size_t(input_channels) * output_channels * 4);
  std::vector<float> bias(static_cast<std::size_t>(output_channels));
  std::vector<float> expected(output_count), output(output_count);
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 41) - 20) * .03125F;
  for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = float(int(i % 23) - 11) * .03125F;
  for (int i = 0; i < output_channels; ++i)
    bias[static_cast<std::size_t>(i)] = float(int(i % 13) - 6) * .0625F;
  ppocr::detail::kernels::ConvTranspose2x2(expected.data(), input.data(), weights.data(),
                                           bias.data(), input_channels, output_channels,
                                           input_height, input_width, 0);
  if (act == 1) {
    for (float& value : expected) value = std::max(value, 0.F);
  } else if (act == 2) {
    for (float& value : expected) value = 1.F / (1.F + std::exp(-value));
  }
  ppocr::detail::kernels::ConvTranspose2x2(output.data(), input.data(), weights.data(),
                                           bias.data(), input_channels, output_channels,
                                           input_height, input_width, act);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = (act == 2 ? 3e-5F : 3e-5F) * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "transpose2x2 act=" << act << " mismatch C=" << input_channels
                << " M=" << output_channels << " H=" << input_height << " W=" << input_width
                << " index=" << i << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyConvTransposeChain(int input_channels, int mid_channels, int output_channels,
                              int input_height, int input_width) {
  const int mid_h = input_height * 2, mid_w = input_width * 2;
  const int out_h = input_height * 4, out_w = input_width * 4;
  const std::size_t in_n = std::size_t(input_channels) * input_height * input_width;
  const std::size_t mid_n = std::size_t(mid_channels) * mid_h * mid_w;
  const std::size_t out_n = std::size_t(output_channels) * out_h * out_w;
  std::vector<float> input(in_n), w0(std::size_t(input_channels) * mid_channels * 4);
  std::vector<float> b0(static_cast<std::size_t>(mid_channels));
  std::vector<float> w1(std::size_t(mid_channels) * output_channels * 4);
  std::vector<float> b1(static_cast<std::size_t>(output_channels));
  std::vector<float> mid(mid_n), expected(out_n), output(out_n);
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 41) - 20) * .03125F;
  for (std::size_t i = 0; i < w0.size(); ++i) w0[i] = float(int(i % 23) - 11) * .03125F;
  for (int i = 0; i < mid_channels; ++i) b0[static_cast<std::size_t>(i)] = float(int(i % 13) - 6) * .0625F;
  for (std::size_t i = 0; i < w1.size(); ++i) w1[i] = float(int(i % 19) - 9) * .03125F;
  for (int i = 0; i < output_channels; ++i) b1[static_cast<std::size_t>(i)] = float(int(i % 11) - 5) * .0625F;
  ppocr::detail::kernels::ConvTranspose2x2(mid.data(), input.data(), w0.data(), b0.data(),
                                           input_channels, mid_channels, input_height,
                                           input_width, 1);
  ppocr::detail::kernels::ConvTranspose2x2(expected.data(), mid.data(), w1.data(), b1.data(),
                                           mid_channels, output_channels, mid_h, mid_w, 2);
  ppocr::detail::kernels::ConvTranspose2x2Chain(output.data(), input.data(), w0.data(), b0.data(),
                                                w1.data(), b1.data(), input_channels, mid_channels,
                                                output_channels, input_height, input_width);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 4e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "transpose2x2 chain mismatch C=" << input_channels << " M=" << mid_channels
                << " O=" << output_channels << " H=" << input_height << " W=" << input_width
                << " index=" << i << " got=" << output[i] << " expected=" << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyConvTransposeBatch(int batches, int input_channels, int output_channels,
                              int input_height, int input_width) {
  const std::size_t input_batch = std::size_t(input_channels) * input_height * input_width;
  const std::size_t output_batch = std::size_t(output_channels) * (input_height * 2) * (input_width * 2);
  std::vector<float> input(std::size_t(batches) * input_batch);
  std::vector<float> weights(std::size_t(input_channels) * output_channels * 4);
  std::vector<float> bias(static_cast<std::size_t>(output_channels));
  std::vector<float> expected(std::size_t(batches) * output_batch);
  std::vector<float> output(expected.size());
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 41) - 20) * .03125F;
  for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = float(int(i % 23) - 11) * .03125F;
  for (std::size_t i = 0; i < bias.size(); ++i) bias[i] = float(int(i % 13) - 6) * .0625F;
  for (int batch = 0; batch < batches; ++batch) {
    ppocr::detail::kernels::ConvTranspose2x2(
        expected.data() + std::size_t(batch) * output_batch,
        input.data() + std::size_t(batch) * input_batch, weights.data(), bias.data(),
        input_channels, output_channels, input_height, input_width);
  }
  ppocr::detail::kernels::ConvTranspose2x2Batch(
      output.data(), input.data(), weights.data(), bias.data(), batches, input_channels,
      output_channels, input_height, input_width);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 3e-5F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "batched transpose2x2 mismatch N=" << batches << " C=" << input_channels
                << " M=" << output_channels << " H=" << input_height << " W=" << input_width
                << " index=" << i << " got=" << output[i] << " expected=" << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyBatchNormAffine(int batches, int channels, std::size_t spatial) {
  const std::size_t count = std::size_t(batches) * channels * spatial;
  std::vector<float> input(count), scale(channels), shift(channels), expected(count), output(count);
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 53) - 26) * .03125F;
  for (int channel = 0; channel < channels; ++channel) {
    scale[channel] = float((channel % 11) + 1) * .09375F;
    shift[channel] = float((channel % 7) - 3) * .125F;
  }
  for (int batch = 0; batch < batches; ++batch) for (int channel = 0; channel < channels; ++channel) {
    const auto offset = (std::size_t(batch) * channels + channel) * spatial;
    for (std::size_t index = 0; index < spatial; ++index) {
      expected[offset + index] = input[offset + index] * scale[channel] + shift[channel];
    }
  }
  ppocr::detail::kernels::BatchNormAffine(output.data(), input.data(), scale.data(), shift.data(),
                                           batches, channels, spatial);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 3e-6F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "BatchNormAffine output mismatch at " << i << '\n';
      return false;
    }
  }
  auto inplace = input;
  ppocr::detail::kernels::BatchNormAffine(inplace.data(), inplace.data(), scale.data(), shift.data(),
                                           batches, channels, spatial);
  for (std::size_t i = 0; i < inplace.size(); ++i) {
    if (inplace[i] != expected[i]) {
      std::cerr << "BatchNormAffine inplace mismatch at " << i << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyNearestResize2xAdd(int batches, int channels, int height, int width) {
  const int output_width = width * 2;
  const std::size_t input_count = std::size_t(batches) * channels * height * width;
  const std::size_t output_count = input_count * 4;
  std::vector<float> input(input_count), residual(output_count), expected(output_count), output(output_count);
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 59) - 29) * .03125F;
  for (std::size_t i = 0; i < residual.size(); ++i) residual[i] = float(int(i % 43) - 21) * .015625F;
  for (int plane = 0; plane < batches * channels; ++plane) {
    const float* src = input.data() + std::size_t(plane) * height * width;
    const float* add = residual.data() + std::size_t(plane) * height * width * 4;
    float* dst = expected.data() + std::size_t(plane) * height * width * 4;
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
      const float value = src[std::size_t(y) * width + x];
      const auto offset = std::size_t(y * 2) * output_width + std::size_t(x) * 2;
      dst[offset] = value + add[offset]; dst[offset + 1] = value + add[offset + 1];
      dst[offset + output_width] = value + add[offset + output_width];
      dst[offset + output_width + 1] = value + add[offset + output_width + 1];
    }
  }
  ppocr::detail::kernels::NearestResize2xAdd(output.data(), input.data(), residual.data(),
                                               batches, channels, height, width);
  for (std::size_t i = 0; i < output.size(); ++i) {
    if (output[i] != expected[i]) {
      std::cerr << "nearest resize-add mismatch N=" << batches << " C=" << channels
                << " H=" << height << " W=" << width << " index=" << i << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyAveragePool3x2Valid(int planes, int height, int width) {
  const int output_height = (height - 3) / 3 + 1;
  const int output_width = (width - 2) / 2 + 1;
  const std::size_t input_plane = std::size_t(height) * width;
  const std::size_t output_plane = std::size_t(output_height) * output_width;
  std::vector<float> input(std::size_t(planes) * input_plane);
  std::vector<float> expected(std::size_t(planes) * output_plane), output(expected.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = float(int(i % 37) - 18) * .03125F;
  }
  for (int plane = 0; plane < planes; ++plane) {
    const float* source = input.data() + std::size_t(plane) * input_plane;
    float* destination = expected.data() + std::size_t(plane) * output_plane;
    for (int y = 0; y < output_height; ++y) for (int x = 0; x < output_width; ++x) {
      float sum{};
      for (int ky = 0; ky < 3; ++ky) for (int kx = 0; kx < 2; ++kx) {
        sum += source[std::size_t(y * 3 + ky) * width + x * 2 + kx];
      }
      destination[std::size_t(y) * output_width + x] = sum / 6.F;
    }
  }
  ppocr::detail::kernels::AveragePool3x2Valid(output.data(), input.data(), planes, height, width);
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float tolerance = 3e-6F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance) {
      std::cerr << "average pool 3x2 valid mismatch at " << i << " got=" << output[i]
                << " expected=" << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyCtcTop1(int batches, int steps, int vocab) {
  const std::size_t rows = std::size_t(batches) * steps;
  std::vector<float> logits(rows * vocab);
  std::vector<int> expected_indices(rows), output_indices(rows);
  std::vector<float> expected_probabilities(rows), output_probabilities(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    for (int column = 0; column < vocab; ++column) {
      logits[row * vocab + column] = float(int((row * 31 + std::size_t(column) * 17) % 127) - 63) * .03125F;
    }
    // Cover blank and repeated top-1 decisions, including vector tails.
    logits[row * vocab + (row % 5 == 0 ? 0 : int((row + 7) % vocab))] = 3.F + float(row % 3);
  }
  ppocr::detail::kernels::CtcTop1Scalar(expected_indices.data(), expected_probabilities.data(),
                                         logits.data(), rows, steps, vocab);
  ppocr::detail::kernels::CtcTop1(output_indices.data(), output_probabilities.data(),
                                   logits.data(), rows, steps, vocab);
  for (std::size_t row = 0; row < rows; ++row) {
    if (output_indices[row] != expected_indices[row]) {
      std::cerr << "CTC Top1 index mismatch row=" << row << '\n';
      return false;
    }
    const float tolerance = 3e-6F * std::max(1.F, std::abs(expected_probabilities[row]));
    if (std::abs(output_probabilities[row] - expected_probabilities[row]) > tolerance) {
      std::cerr << "CTC Top1 probability mismatch row=" << row << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyCtcTop1FirstMaximum() {
  // SIMD ArgMax must retain the scalar executor's first-maximum rule across
  // vector lanes and tails. Use a 6,906-way row so this covers the production
  // Latin vocabulary geometry as well as a small vector-tail row.
  for (const int vocab : {31, 6906}) {
    const int batches = 1, steps = 4;
    std::vector<float> logits(std::size_t(batches) * steps * vocab, -2.F);
    for (int step = 0; step < steps; ++step) {
      float* row = logits.data() + std::size_t(step) * vocab;
      row[0] = 1.F;
      row[std::min(vocab - 1, 3)] = 5.F;
      row[std::min(vocab - 1, 16)] = 5.F;  // tie: first maximum must win.
      row[vocab - 1] = step == 1 ? 6.F : 5.F;
    }
    std::vector<int> expected_indices(steps), output_indices(steps);
    std::vector<float> expected_probabilities(steps), output_probabilities(steps);
    ppocr::detail::kernels::CtcTop1Scalar(expected_indices.data(), expected_probabilities.data(),
                                           logits.data(), steps, steps, vocab);
    ppocr::detail::kernels::CtcTop1(output_indices.data(), output_probabilities.data(),
                                     logits.data(), steps, steps, vocab);
    for (int row = 0; row < steps; ++row) {
      if (output_indices[row] != expected_indices[row] ||
          std::abs(output_probabilities[row] - expected_probabilities[row]) > 1e-5F) {
        std::cerr << "CTC Top1 first-maximum mismatch vocab=" << vocab << " row=" << row
                  << " expected_index=" << expected_indices[row]
                  << " output_index=" << output_indices[row]
                  << " expected_probability=" << expected_probabilities[row]
                  << " output_probability=" << output_probabilities[row] << '\n';
        return false;
      }
    }
  }
  return true;
}

bool VerifyGemmCtcTop1(int batches, int steps, int vocab, int depth) {
  const std::size_t rows = std::size_t(batches) * steps;
  std::vector<float> left(rows * depth), weights(std::size_t(depth) * vocab), bias(vocab), logits(rows * vocab);
  std::vector<int> expected_indices(rows), output_indices(rows);
  std::vector<float> expected_probabilities(rows), output_probabilities(rows);
  for (std::size_t i = 0; i < left.size(); ++i) left[i] = float(int(i % 37) - 18) * .0625F;
  for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = float(int(i % 41) - 20) * .03125F;
  for (int i = 0; i < vocab; ++i) bias[std::size_t(i)] = float(i % 11) * .0625F - .25F;
  ppocr::detail::kernels::Gemm(logits.data(), left.data(), weights.data(), bias.data(),
                                int(rows), vocab, depth);
  ppocr::detail::kernels::CtcTop1Scalar(expected_indices.data(), expected_probabilities.data(),
                                         logits.data(), rows, steps, vocab);
  ppocr::detail::kernels::GemmCtcTop1(output_indices.data(), output_probabilities.data(),
                                       left.data(), weights.data(), bias.data(), int(rows), depth,
                                       vocab, steps);
  for (std::size_t row = 0; row < rows; ++row) {
    if (output_indices[row] != expected_indices[row]) {
      std::cerr << "GEMM CTC Top1 index mismatch row=" << row << '\n';
      return false;
    }
    const float tolerance = 3e-6F * std::max(1.F, std::abs(expected_probabilities[row]));
    if (std::abs(output_probabilities[row] - expected_probabilities[row]) > tolerance) {
      std::cerr << "GEMM CTC Top1 probability mismatch row=" << row << '\n';
      return false;
    }
  }
  return true;
}

bool VerifyApproximateGelu() {
  // The optional x86 path is intentionally a Padé-tanh GELU approximation.
  // Validate both in-place and separate buffers against its scalar formula so
  // AVX-512, AVX2 and scalar tails share identical deployment semantics.
  // Cross the model-specific approximate-GELU threshold and retain a vector
  // tail, covering the runtime dispatch used by broad medium feature maps.
  std::vector<float> input(262144 + 13), output(input.size()), inplace(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = float(int(i % 127) - 63) * .0625F;
    inplace[i] = input[i];
  }
  constexpr float c0 = .7978845608028654F, c1 = .044715F;
  std::vector<float> expected(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    const float x = input[i];
    const float z = c0 * (x + c1 * x * x * x);
    const float z2 = z * z;
    const float t = std::clamp(z * (27.F + z2) / (27.F + 9.F * z2), -1.F, 1.F);
    expected[i] = .5F * x * (1.F + t);
  }
  ppocr::detail::kernels::Gelu(output.data(), input.data(), input.size(), input.size());
  ppocr::detail::kernels::Gelu(inplace.data(), inplace.data(), inplace.size(), inplace.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    const float tolerance = 3e-6F * std::max(1.F, std::abs(expected[i]));
    if (std::abs(output[i] - expected[i]) > tolerance ||
        std::abs(inplace[i] - expected[i]) > tolerance) {
      std::cerr << "approximate GELU mismatch index=" << i << '\n';
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  // Drive ENABLE-only MaxPoolConcat/CTC kernels from the shipped functions.
#if defined(_WIN32)
  _putenv("PPOCR_ENABLE_AVX512_MAXPOOL_TINY_PARALLEL=1");
  _putenv("PPOCR_ENABLE_AVX512_CONCAT_S2_ROW_PF=1");
  _putenv("PPOCR_ENABLE_AVX512_GEMM16NARROW=1");
  _putenv("PPOCR_ENABLE_AVX512_POINTWISE16_TINY=1");
  _putenv("PPOCR_ENABLE_CTC_VOCAB_BLOCK=1");
  _putenv("PPOCR_ENABLE_AVX512_CONCAT_ONEPASS=1");
  _putenv("PPOCR_ENABLE_AVX512_TRANSPOSE2X2_TILE4=1");
  _putenv("PPOCR_ENABLE_AVX512_RESIZE2X_PF=1");
  _putenv("PPOCR_ENABLE_POINTWISE_DEPTHWISE=1");
  _putenv("PPOCR_FUSED_TERMINAL_CTC_TILE_ROWS=16");
#else
  setenv("PPOCR_ENABLE_AVX512_MAXPOOL_TINY_PARALLEL", "1", 1);
  setenv("PPOCR_ENABLE_AVX512_CONCAT_S2_ROW_PF", "1", 1);
  setenv("PPOCR_ENABLE_AVX512_GEMM16NARROW", "1", 1);
  setenv("PPOCR_ENABLE_AVX512_POINTWISE16_TINY", "1", 1);
  setenv("PPOCR_ENABLE_CTC_VOCAB_BLOCK", "1", 1);
  setenv("PPOCR_ENABLE_AVX512_CONCAT_ONEPASS", "1", 1);
  setenv("PPOCR_ENABLE_AVX512_TRANSPOSE2X2_TILE4", "1", 1);
  setenv("PPOCR_ENABLE_AVX512_RESIZE2X_PF", "1", 1);
  setenv("PPOCR_ENABLE_POINTWISE_DEPTHWISE", "1", 1);
  setenv("PPOCR_FUSED_TERMINAL_CTC_TILE_ROWS", "16", 1);
#endif
  // Odd row counts exercise the paired-row SIMD loop plus its scalar/SIMD
  // tail; odd/vector-boundary column counts cover both vector and scalar
  // write paths. The last case is recognizer-projection sized.
  if (!VerifyRgbByteWiden() || !VerifyIdentityRgbToNchw() ||
      !VerifyBilinearRgbToNchw(720, 152, 0, 0, 720, 152, 704, 160, 704, true) ||
      !VerifyBilinearRgbToNchw(64, 32, 2, 1, 37, 19, 41, 17, 48, false) ||
      !VerifyBilinearRgbToNchw(8, 8, 0, 0, 1, 1, 5, 5, 8, false) ||
      !VerifyBilinearRgbToNchw(32, 16, 0, 0, 32, 16, 32, 16, 32, true) ||
      !VerifyExactGelu() ||
      !VerifySigmoid(17) || !VerifySigmoid(262144 + 13) ||
      !VerifyMaxPool2x2Valid(16, 41, 39) || !VerifyMaxPool2x2Valid(3, 17, 19) ||
      !VerifyMaxPool2x2Valid(16, 64, 63) || !VerifyMaxPool2x2Valid(8, 128, 127) ||
      !VerifyMaxPool2x2Same(16, 41, 39) || !VerifyMaxPool2x2Same(8, 64, 63) ||
      !VerifyMaxPool2x2Same(3, 17, 19) || !VerifyMaxPool2x2Same(16, 80, 352)) return 1;
  if (!VerifyConv3x3Stride2(3, 16, 41, 39, true) ||
      !VerifyConv3x3Stride2(3, 16, 65, 63, true) ||
      !VerifyConv3x3Stride2(3, 16, 160, 128, true) ||
      !VerifyConv3x3Stride2(3, 5, 37, 35, true) ||
      !VerifyConv3x3Stride2(11, 9, 39, 41, false) ||
      !VerifyConv3x3Stride2(24, 48, 41, 39, true) ||
      !VerifyConv3x3Stride2(32, 16, 41, 39, true)) return 1;
  if (!VerifyConcatChannelConv(4, 16, 16, 41, 39, true) ||
      !VerifyConcatChannelConv(4, 16, 16, 40, 176, true) ||
      !VerifyConcatChannelConv(4, 16, 16, 40, 176, false) ||
      !VerifyConcatChannelConv(3, 5, 8, 17, 19, false) ||
      !VerifyConcatChannelConv(2, 7, 9, 33, 31, true) ||
      !VerifyConcatChannelConv(3, 8, 16, 41, 39, true, 2, 2) ||
      !VerifyConcatChannelConv(2, 16, 16, 80, 352, true, 2, 2) ||
      !VerifyConcatChannelConv(2, 7, 9, 33, 31, false, 2, 2)) return 1;
  {
    const int stem_channels[] = {16, 8, 16};
    if (!VerifyConcatChannelConvMixed(stem_channels, 3, 16, 41, 39, true, 2, 2) ||
        !VerifyConcatChannelConvMixed(stem_channels, 3, 16, 37, 35, false, 2, 2)) {
      return 1;
    }
  }
  if (!VerifyConcatConvDualTranspose(4, 16, 16, 1, 9, 11) ||
      !VerifyConcatConvDualTranspose(4, 16, 16, 1, 40, 176)) return 1;
  if (!VerifyLayerNorm(3, 17) || !VerifyLayerNorm(5, 192) ||
      !VerifyLayerNorm(4, 193)) return 1;
  for (const auto [rows, cols, depth] : {
           std::array<int, 3>{1, 1, 1},
           std::array<int, 3>{2, 7, 5},
           std::array<int, 3>{3, 17, 13},
           std::array<int, 3>{5, 33, 31},
           std::array<int, 3>{7, 65, 47},
           std::array<int, 3>{8, 6913, 192},
           // Terminal CTC projections for the shipped recognizers.  These
           // wide, four-row cases exercise the AVX-512 four-row tile and its
           // lower-precision-free vector/tail handling without requiring an
           // OCR model load in this low-level smoke binary.
           std::array<int, 3>{4, 6906, 80},
           std::array<int, 3>{16, 6906, 80},
           std::array<int, 3>{4, 18710, 120},
           std::array<int, 3>{4, 18710, 192},
       }) {
    if (!VerifyGemm(rows, cols, depth, false) || !VerifyGemm(rows, cols, depth, true)) return 1;
  }
  // A large square map takes the AVX-512 eight-output Conv tile, while the
  // odd output count exercises its four-channel fallback/tail. ReLU covers
  // the detector's fused writeback path.
  if (!VerifyConv3x3Tile(32, 8, 128, 128, false) ||
      !VerifyConv3x3Tile(32, 9, 128, 128, true) ||
      !VerifyConv3x3Tile(64, 16, 65, 63, true)) return 1;
  if (!VerifyDetStemFromNchw(41, 39) || !VerifyDetStemFromNchw(160, 128) ||
      !VerifyDetStemFromNchw(64, 64)) return 1;
  if (!VerifyConv2x2SameUpper(32, 8, 65, 63, false) ||
      !VerifyConv2x2SameUpper(32, 9, 64, 64, true) ||
      !VerifyConv2x2SameUpper(16, 8, 41, 39, true) ||
      !VerifyConv2x2SameUpper(16, 8, 80, 352, true) ||
      !VerifyConv2x2SameUpper(8, 16, 41, 39, true) ||
      !VerifyConv2x2SameUpper(8, 16, 80, 352, true) ||
      !VerifyConv2x2SameUpper(8, 12, 17, 15, false)) return 1;
  // PP-OCRv6 medium uses 9x9 depthwise MobileNet context filters before a
  // pointwise projection.  Exercise a non-square plane and a tail channel so
  // AVX-512, AVX2, NEON, and scalar dispatches all cover the same geometry.
  if (!VerifyDepthwiseBatch(2, 257, 37, 41, 9, 1, 4)) return 1;
  // The opt-in wide tile has an eight-channel SIMD body and a four-channel
  // tail. These padded 5x5/7x7 cases cover the medium-detector context path.
  if (!VerifyConvWideTile(32, 8, 64, 63, 5) ||
      !VerifyConvWideTile(32, 9, 64, 63, 7)) return 1;
  // Exercise the NCHW cross-crop schedule, including non-SIMD tails and a
  // work size that crosses the persistent-pool threshold.
  if (!VerifyPointwiseConv(48, 96, 1284) || !VerifyPointwiseConv(96, 48, 1284) ||
      !VerifyPointwiseConv(48, 97, 1000) || !VerifyPointwiseConv(11, 9, 37) ||
      !VerifyPointwiseConv(16, 16, 7040) || !VerifyPointwiseConv(16, 24, 2112) ||
      !VerifyPointwiseConv(32, 64, 7040) || !VerifyPointwiseConv(32, 64, 272) ||
      !VerifyPointwiseConv(64, 16, 7040) || !VerifyPointwiseConv(64, 16, 272) ||
      !VerifyPointwiseConv(16, 32, 7040) || !VerifyPointwiseConv(16, 32, 80) ||
      !VerifyPointwiseConvRelu(16, 32, 7040) || !VerifyPointwiseConvRelu(16, 32, 80) ||
      !VerifyPointwiseConvRelu(32, 64, 7040) || !VerifyPointwiseConvRelu(7, 9, 37)) {
    return 1;
  }
  if (!VerifyExpandGeluProjectAdd(48, 96, 1284) ||
      !VerifyExpandGeluProjectAdd(48, 96, 1000) ||
      !VerifyExpandGeluProjectAdd(11, 9, 37) ||
      !VerifyExpandGeluProjectAdd(8, 16, 16) ||
      !VerifyExpandGeluProjectAdd(4, 5, 17)) {
    return 1;
  }
  for (int activation = 0; activation < 3; ++activation) {
    if (!VerifyPointwiseResidualBatch(3, 7, 9, 37, activation)) return 1;
  }
  if (!VerifyPointwiseResidualBatch(3, 64, 64, 1024, 0)) return 1;
  if (!VerifyDepthwiseBatch(3, 5, 17, 19, 3, 1, 1) ||
      !VerifyDepthwiseBatch(3, 19, 31, 33, 5, 2, 2) ||
      !VerifyDepthwiseBatch(1, 64, 40, 176, 5, 1, 2) ||
      !VerifyDepthwiseBatch(1, 64, 17, 19, 5, 1, 2) ||
      !VerifyDepthwiseScalar(64, 40, 176, 5, 2) ||
      !VerifyDepthwiseScalar(7, 17, 19, 5, 2) ||
      !VerifyFoldedDepthwiseHardSwish(16, 6, 76, 1, 5, 0, 2) ||
      !VerifyFoldedDepthwiseHardSwish(7, 9, 11, 3, 3, 1, 1)) return 1;
  for (int activation = 0; activation < 4; ++activation) {
    if (!VerifyDepthwisePointwiseFused(16, 16, 40, 176, activation) ||
        !VerifyDepthwisePointwiseFused(8, 6, 17, 19, activation) ||
        !VerifyDepthwisePointwiseFused(5, 7, 9, 11, activation)) {
      return 1;
    }
  }
  for (int activation = 0; activation < 4; ++activation) {
    if (!VerifyDepthwisePointwiseFused5x5(64, 16, 40, 176, activation) ||
        !VerifyDepthwisePointwiseFused5x5(7, 5, 9, 11, activation) ||
        !VerifyDepthwisePointwiseFused5x5(16, 8, 17, 19, activation)) {
      return 1;
    }
  }
  if (!VerifyPointwiseDepthwise(16, 32, 40, 176, true) ||
      !VerifyPointwiseDepthwise(16, 32, 40, 176, false) ||
      !VerifyPointwiseDepthwise(16, 32, 33, 17, true) ||
      !VerifyPointwiseDepthwise(8, 8, 32, 16, false) ||
      !VerifyPointwiseDepthwise(7, 9, 9, 11, true)) {
    return 1;
  }
  if (!VerifyDepthwiseExpandGelu(16, 32, 40, 176) ||
      !VerifyDepthwiseExpandGelu(16, 32, 33, 65) ||
      !VerifyDepthwiseExpandGelu(8, 16, 17, 19) ||
      !VerifyDepthwiseExpandGelu(5, 7, 9, 11) ||
      !VerifyDepthwiseExpandGelu(48, 96, 12, 76) ||
      !VerifyDepthwiseExpandGelu(160, 320, 6, 76)) {
    return 1;
  }
  if (!VerifyConvBatch(3, 3, 16, 41, 39, 3, 3, 2, 2, 1, 1, true) ||
      !VerifyConvBatch(3, 7, 9, 37, 35, 3, 3, 1, 1, 1, 1, false) ||
      !VerifyConvBatch(3, 32, 8, 128, 127, 3, 3, 1, 1, 1, 1, true) ||
      !VerifyConvBatch(3, 11, 13, 39, 41, 3, 3, 2, 2, 1, 1, true) ||
      !VerifyConvBatch(3, 11, 9, 39, 41, 2, 2, 1, 1, 0, 0, true) ||
      !VerifyConvBatch(3, 32, 8, 65, 63, 5, 5, 1, 1, 2, 2, false) ||
      !VerifyConvBatch(3, 32, 9, 65, 63, 7, 7, 1, 1, 3, 3, false) ||
      !VerifyConvBatch(3, 13, 9, 39, 41, 1, 5, 1, 1, 0, 2, false) ||
      !VerifyConvBatch(3, 13, 9, 39, 41, 5, 1, 1, 1, 2, 0, false)) return 1;
  if (!VerifyPointwiseHardSwishBatch(3, 7, 9, 37) ||
      !VerifyPointwiseHardSwishBatch(3, 64, 64, 1024)) return 1;
  if (!VerifySqueezeExcitation(1, 16, 4, 40, 176, false) ||
      !VerifySqueezeExcitation(1, 16, 4, 40, 176, true) ||
      !VerifySqueezeExcitation(1, 8, 2, 17, 19, true) ||
      !VerifySqueezeExcitation(2, 24, 6, 12, 37, false)) return 1;
  if (!VerifyConvTransposeBatch(3, 7, 9, 17, 19) ||
      !VerifyConvTransposeBatch(3, 32, 32, 64, 63) ||
      !VerifyConvTransposeScalar(16, 16, 40, 176) ||
      !VerifyConvTransposeScalar(16, 16, 80, 352) ||
      !VerifyConvTransposeAct(16, 16, 40, 176, 1) ||
      !VerifyConvTransposeAct(16, 16, 80, 352, 2) ||
      !VerifyConvTransposeAct(7, 9, 17, 19, 1) ||
      !VerifyConvTransposeChain(16, 16, 1, 9, 11) ||
      !VerifyConvTransposeChain(16, 16, 1, 40, 176) ||
      !VerifyConvTransposeChain(7, 9, 2, 5, 6)) return 1;
  if (!VerifyBatchNormAffine(3, 7, 37) || !VerifyBatchNormAffine(3, 64, 1024)) return 1;
  if (!VerifyNearestResize2xAdd(1, 3, 17, 19) ||
      !VerifyNearestResize2xAdd(2, 64, 64, 63) ||
      !VerifyNearestResize2xAdd(1, 16, 20, 88)) return 1;
  if (!VerifyAveragePool3x2Valid(3, 17, 19) ||
      !VerifyAveragePool3x2Valid(64, 48, 127)) return 1;
  if (!VerifyCtcTop1(3, 17, 31) || !VerifyCtcTop1(4, 41, 6906) ||
      !VerifyCtcTop1FirstMaximum()) return 1;
  if (!VerifyGemmCtcTop1(1, 17, 31, 29) || !VerifyGemmCtcTop1(4, 41, 6906, 192) ||
      !VerifyGemmCtcTop1(1, 4, 18710, 192) || !VerifyGemmCtcTop1(1, 16, 6906, 80)) return 1;
  if (std::getenv("PPOCR_APPROX_GELU") != nullptr && !VerifyApproximateGelu()) return 1;
  std::cout << "kernel RGB-upload/identity-rgb/LayerNorm/GEMM/2x2/3x3/residual-pointwise/depthwise/conv/hardswish/transpose/resize-add/BN/GELU/exact-GELU/CTC batch tile/tail smoke passed\n";
}
