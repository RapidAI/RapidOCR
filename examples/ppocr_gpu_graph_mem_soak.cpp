#include "onnx_lite.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

int Positive(const char* text, const char* name) {
  try {
    const int value = std::stoi(text);
    if (value > 0) return value;
  } catch (...) {}
  throw std::runtime_error(std::string("invalid ") + name);
}

struct Shape { int n, c, h, w; };

ppocr::detail::Tensor Input(const Shape& shape, std::uint32_t seed) {
  const std::size_t elements = std::size_t(shape.n) * shape.c * shape.h * shape.w;
  ppocr::detail::Tensor input{{shape.n, shape.c, shape.h, shape.w}, std::vector<float>(elements)};
  for (std::size_t index = 0; index < elements; ++index) {
    // Reproducible sign-changing inputs keep every activation/gate path live
    // without retaining an additional image buffer between measurements.
    input.data[index] = float((index * 37u + seed * 101u) % 251u) / 125.F - 1.F;
  }
  return input;
}

void PrintSample(const char* phase, const Shape& shape, std::size_t capacity, std::size_t allocated,
                 std::size_t live, std::size_t baseline, bool recovered) {
  std::cout << "{\"phase\":\"" << phase << "\",\"shape\":\""
            << shape.n << 'x' << shape.c << 'x' << shape.h << 'x' << shape.w
            << "\",\"capacity_bytes\":" << capacity
            << ",\"allocated_bytes\":" << allocated
            << ",\"live_bytes\":" << live
            << ",\"constants_baseline_bytes\":" << baseline
            << ",\"activation_bytes\":" << (live >= baseline ? live - baseline : 0)
            << ",\"recovered\":" << (recovered ? "true" : "false") << "}\n";
}

// Device-input execution is deliberately separate from the ordinary
// host-input ladder.  It records a persistent Vulkan command buffer, which
// pins all recorded intermediate activations.  Releasing the model cache must
// return that *complete* ownership set, rather than only the public graph
// input/output slots.
bool RunFromDevice(ppocr::detail::OnnxLite& graph, const Shape& shape,
                   std::uint32_t seed) {
  auto input = Input(shape, seed);
  auto& arena = graph.gpu_arena();
  auto slot = arena.Acquire(input.data.size(), "replay-lifetime-input");
  if (!slot.resident || !arena.Upload(slot, input.data.data(), input.data.size())) {
    arena.Release(slot);
    return false;
  }
  std::unordered_map<std::string, ppocr::detail::OnnxLite::GpuTensor> outputs;
  std::unordered_map<std::string, ppocr::detail::OnnxLite::GpuTensor> inputs;
  inputs.emplace("x", ppocr::detail::OnnxLite::GpuTensor{std::move(input.shape), slot});
  if (!graph.RunGpuOnlyDevice(std::move(inputs), outputs)) return false;
  // A replayed public output is cache-owned and release is intentionally a
  // no-op while pinned.  The post-run ReleaseGpuConstants() below is the
  // lifetime boundary being tested.
  for (auto& [_, output] : outputs) arena.Release(output.slot);
  return true;
}

}  // namespace

// Usage:
//   ppocr_gpu_graph_mem_soak MODEL.onnx N C H W [N C H W ...]
// Each supplied shape is run twice. Inputs should be listed from small to
// large, followed by a small shape again to demonstrate post-large recovery.
// Add --device-input-replay after MODEL.onnx to exercise persistent replay
// graph eviction/release.  That mode releases the model after each shape and
// requires both logical live bytes and all allocated arena storage to reach 0.
int main(int argc, char** argv) {
  const bool device_input_replay = argc >= 3 && std::string_view(argv[2]) == "--device-input-replay";
  const int first_shape = device_input_replay ? 3 : 2;
  if (argc < first_shape + 4 || ((argc - first_shape) % 4) != 0) {
    std::cerr << "usage: ppocr_gpu_graph_mem_soak MODEL.onnx [--device-input-replay] "
                 "N C H W [N C H W ...]\n";
    return 2;
  }
  try {
    std::vector<Shape> shapes;
    for (int index = first_shape; index < argc; index += 4) {
      shapes.push_back({Positive(argv[index], "N"), Positive(argv[index + 1], "C"),
                        Positive(argv[index + 2], "H"), Positive(argv[index + 3], "W")});
    }
    ppocr::detail::OnnxLite graph(argv[1], ppocr::Backend::gpu_only);
    if (!graph.SupportsGpuOnly()) throw std::runtime_error("model graph is not GPU-only compatible");
    // Upload immutable weights and create every persistent descriptor/cache
    // before defining the constants-only live baseline.
    if (device_input_replay) {
      if (!RunFromDevice(graph, shapes.front(), 1))
        throw std::runtime_error("initial persistent replay graph failed");
    } else {
      (void)graph.Run({{"x", Input(shapes.front(), 1)}});
    }
    auto& arena = graph.gpu_arena();
    const std::size_t constants_baseline = arena.live_bytes();
    if (constants_baseline == 0) throw std::runtime_error("GPU constant baseline is unexpectedly empty");
    std::size_t previous_capacity = arena.capacity_bytes();
    PrintSample("warmup", shapes.front(), previous_capacity, arena.allocated_bytes(), constants_baseline,
                constants_baseline, true);
    bool all_recovered = true;
    for (std::size_t item = 0; item < shapes.size(); ++item) {
      for (int repeat = 0; repeat < 2; ++repeat) {
        if (device_input_replay) {
          if (!RunFromDevice(graph, shapes[item], std::uint32_t(item * 2 + repeat + 2)))
            throw std::runtime_error("persistent replay graph failed");
        } else {
          (void)graph.Run({{"x", Input(shapes[item], std::uint32_t(item * 2 + repeat + 2))}});
        }
        const auto capacity = arena.capacity_bytes();
        const auto live = arena.live_bytes();
        const bool recovered = !device_input_replay && live == constants_baseline;
        all_recovered = all_recovered && (device_input_replay || recovered);
        // Ordinary ladders retain a reusable high-water capacity.  The
        // explicit replay-lifetime mode deliberately destroys free arena
        // buffers after each graph/model release, so its next shape starts
        // with fresh capacity and is allowed to be smaller.
        if (!device_input_replay && capacity < previous_capacity)
          throw std::runtime_error("arena capacity unexpectedly shrank");
        previous_capacity = capacity;
        PrintSample(repeat == 0 ? "shape" : "repeat", shapes[item], capacity, arena.allocated_bytes(), live,
                    constants_baseline, recovered);
      }
      if (device_input_replay) {
        // This exercises DropPersistentGraph() for every recorded activation,
        // then destroys free Vulkan buffers.  A non-zero result here means a
        // replay-only activation remained reachable after its model lifetime.
        graph.ReleaseGpuConstants();
        const auto live = arena.live_bytes();
        const bool reclaimed = live == 0 && arena.ReclaimFreeStorage() &&
                               arena.live_bytes() == 0 && arena.allocated_bytes() == 0;
        all_recovered = all_recovered && reclaimed;
        PrintSample("replay-release", shapes[item], arena.capacity_bytes(), arena.allocated_bytes(),
                    arena.live_bytes(), 0, reclaimed);
        previous_capacity = arena.capacity_bytes();
      }
    }
    if (!all_recovered) throw std::runtime_error("dynamic GPU activations did not recover to constants baseline");
    std::cout << "GPU graph memory soak passed samples=" << shapes.size()
              << " constants_baseline_bytes=" << constants_baseline
              << " final_capacity_bytes=" << previous_capacity << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "GPU graph memory soak failed: " << error.what() << '\n';
    return 1;
  }
}
