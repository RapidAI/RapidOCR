#include "onnx_lite.hpp"
#include "kernels.hpp"
#include "vulkan_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <bit>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <queue>
#include <memory>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

namespace ppocr::detail {
namespace {

[[noreturn]] void Fail(const std::string& message) { throw std::runtime_error("ppocr ONNX: " + message); }

void DumpTensorFile(const Tensor& tensor) {
  const char* path = std::getenv("PPOCR_DUMP_TENSOR_FILE");
  if (!path || tensor.data.empty()) return;
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(tensor.data.data()),
            static_cast<std::streamsize>(tensor.data.size() * sizeof(float)));
}

thread_local std::vector<std::vector<float>> g_activation_pool;

std::size_t ActivationPoolLimitFloats() noexcept {
  // This pool is thread-local. A seemingly modest eight retained 2M-float
  // vectors becomes a large resident footprint when dense-page recognition
  // uses several worker threads. Keep the default bounded to 8 MiB per
  // executor thread; deployments that measure a steady-state throughput win
  // may raise it explicitly without changing tensor semantics.
  constexpr std::size_t kDefault = 2u * 1024u * 1024u;
  if (const char* text = std::getenv("PPOCR_ACTIVATION_POOL_MAX_MB")) {
    char* end{};
    const auto mib = std::strtoull(text, &end, 10);
    constexpr std::size_t kMiBInFloats = 1024u * 1024u / sizeof(float);
    if (end != text && *end == '\0' &&
        mib <= std::numeric_limits<std::size_t>::max() / kMiBInFloats) {
      return static_cast<std::size_t>(mib) * kMiBInFloats;
    }
  }
  return kDefault;
}

std::size_t ActivationPoolFloats() noexcept {
  std::size_t total{};
  for (const auto& values : g_activation_pool) total += values.capacity();
  return total;
}

void ResizeUninitialized(std::vector<float>& values, std::size_t count) {
  if (count <= values.size()) {
    values.resize(count);
    return;
  }
  if (count > values.capacity()) {
    std::vector<float> grown;
    grown.reserve(count);
    if (!values.empty()) {
      grown.insert(grown.end(), values.begin(), values.end());
    }
    values.swap(grown);
  }
  if (count <= values.size()) return;
  // Default vector::resize value-initializes floats to zero. Conv/Pool/Concat
  // overwrite every element, so skip that memset on a recycled buffer.
  struct Raw {
    float value;
    Raw() noexcept {}
  };
  static_assert(sizeof(Raw) == sizeof(float));
  static_assert(alignof(Raw) == alignof(float));
  reinterpret_cast<std::vector<Raw>&>(values).resize(count);
}

std::vector<float> PooledActivation(std::size_t count) {
  static const bool disabled = std::getenv("PPOCR_DISABLE_ACTIVATION_POOL") != nullptr;
  std::vector<float> values;
  if (!disabled && !g_activation_pool.empty()) {
    std::size_t best = g_activation_pool.size();
    std::size_t best_capacity = std::numeric_limits<std::size_t>::max();
    for (std::size_t i = 0; i < g_activation_pool.size(); ++i) {
      if (g_activation_pool[i].size() == count) {
        best = i;
        break;
      }
      // Best-fit reuse avoids consuming a detector-sized buffer for a small
      // recognizer tensor merely because it appears earlier in the TLS pool.
      // That keeps the larger slot available and lowers both allocator churn
      // and the capacity retained by mixed-resolution OCR services.
      if (g_activation_pool[i].capacity() >= count &&
          g_activation_pool[i].capacity() < best_capacity) {
        best = i;
        best_capacity = g_activation_pool[i].capacity();
      }
    }
    if (best < g_activation_pool.size()) {
      values = std::move(g_activation_pool[best]);
      g_activation_pool[best] = std::move(g_activation_pool.back());
      g_activation_pool.pop_back();
      if (values.size() == count) return values;
    }
  }
  ResizeUninitialized(values, count);
  return values;
}

void RecycleActivation(std::vector<float>& values) {
  static const bool disabled = std::getenv("PPOCR_DISABLE_ACTIVATION_POOL") != nullptr;
  if (disabled) return;
  // Keep a small set of mid-size activations. Huge detector maps in a
  // 24-slot pool evict useful cache and raise RSS without helping reuse.
  constexpr std::size_t kMinFloats = 4096;
  constexpr std::size_t kMaxFloats = 2 * 1024 * 1024;
  const std::size_t limit = ActivationPoolLimitFloats();
  if (values.capacity() >= kMinFloats && values.capacity() <= kMaxFloats &&
      values.capacity() <= limit) {
    // Evict the largest stale slots first. This preserves small/medium reuse
    // after a one-off large page rather than pinning its temporary maps in
    // every recognition worker's thread-local cache.
    while ((!g_activation_pool.empty() &&
            (g_activation_pool.size() >= 8 ||
             ActivationPoolFloats() > limit - values.capacity()))) {
      const auto victim = std::max_element(
          g_activation_pool.begin(), g_activation_pool.end(),
          [](const auto& left, const auto& right) {
            return left.capacity() < right.capacity();
          });
      g_activation_pool.erase(victim);
    }
    g_activation_pool.push_back(std::move(values));
  }
}

// Hybrid's selection promise is contextual: a full H2D + Vulkan + D2H probe
// is valid only for the same Vulkan device/runtime generation and CPU worker
// budget. A static shape-only cache could otherwise retain an answer made on
// a different adapter (or before device-loss recovery) and incorrectly skip
// the required full-boundary qualification. This compact tag is mixed into
// each operator's existing shape key below; each call keeps its own map, so
// no cross-operator admission can occur.
std::uint64_t HybridAdmissionContext() noexcept {
  std::uint64_t hash = 1469598103934665603ull;
  const auto mix = [&hash](std::uint64_t value) {
    hash ^= value;
    hash *= 1099511628211ull;
  };
  mix(VulkanHybridAdmissionContext());
  // Mirrors kernels.cpp's explicit deployment tuning knob without exposing
  // internal thread-pool APIs in the public header.
  int workers = 16;
  if (const char* configured = std::getenv("PPOCR_BENCH_THREADS")) {
    char* end = nullptr;
    const long parsed = std::strtol(configured, &end, 10);
    if (end != configured && *end == '\0' && parsed > 0 && parsed <= 1024)
      workers = static_cast<int>(parsed);
  }
  mix(static_cast<std::uint64_t>(workers));
  mix(std::getenv("PPOCR_VULKAN_PREFER_DISCRETE") ? 1u : 0u);
  return hash;
}

std::uint64_t WithHybridAdmissionContext(std::uint64_t shape_key) noexcept {
  const std::uint64_t context = HybridAdmissionContext();
  return shape_key ^ (context + 0x9e3779b97f4a7c15ull + (shape_key << 6) + (shape_key >> 2));
}

std::size_t Elements(const std::vector<std::int64_t>& shape) {
  std::size_t n = 1;
  for (auto d : shape) {
    if (d < 0 || (d && n > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(d))) {
      Fail("invalid tensor shape");
    }
    n *= static_cast<std::size_t>(d);
  }
  return n;
}

struct Reader {
  const std::uint8_t* bytes{};
  std::size_t size{};
  std::size_t pos{};
  explicit Reader(const std::vector<std::uint8_t>& input) noexcept
      : bytes(input.data()), size(input.size()) {}
  Reader(const std::uint8_t* input, std::size_t input_size) noexcept
      : bytes(input), size(input_size) {}
  bool eof() const { return pos == size; }
  std::uint64_t Varint() {
    std::uint64_t out{};
    for (int shift = 0; shift < 64; shift += 7) {
      if (pos == size) Fail("truncated protobuf");
      const auto b = bytes[pos++]; out |= std::uint64_t(b & 0x7f) << shift;
      if (!(b & 0x80)) return out;
    }
    Fail("protobuf varint overflow");
  }
  std::uint32_t Fixed32() {
    if (pos + 4 > size) Fail("truncated fixed32");
    std::uint32_t v{}; std::memcpy(&v, bytes + pos, 4); pos += 4; return v;
  }
  std::pair<const std::uint8_t*, std::size_t> Span() {
    const auto n = Varint();
    if (n > size - pos) Fail("truncated byte field");
    const auto* out = bytes + pos;
    pos += static_cast<std::size_t>(n); return {out, static_cast<std::size_t>(n)};
  }
  std::vector<std::uint8_t> Bytes() {
    const auto [data, size] = Span();
    return {data, data + size};
  }
  void Skip(int wire) {
    switch (wire) {
      case 0: (void)Varint(); break;
      case 1: if (pos + 8 > size) Fail("truncated fixed64"); pos += 8; break;
      case 2: (void)Span(); break;
      case 5: (void)Fixed32(); break;
      default: Fail("unsupported protobuf wire type");
    }
  }
  std::pair<int, int> Tag() { const auto t = Varint(); return {int(t >> 3), int(t & 7)}; }
};

std::vector<std::int64_t> PackedInts(const std::vector<std::uint8_t>& b) {
  Reader r{b}; std::vector<std::int64_t> out;
  while (!r.eof()) out.push_back(static_cast<std::int64_t>(r.Varint()));
  return out;
}
std::vector<float> PackedFloats(const std::vector<std::uint8_t>& b) {
  if (b.size() % 4) Fail("malformed packed floats");
  std::vector<float> out(b.size() / 4); std::memcpy(out.data(), b.data(), b.size()); return out;
}

struct Attribute {
  std::int64_t i{};
  float f{};
  std::string s;
  std::vector<std::int64_t> ints;
  std::vector<float> floats;
};
struct Node {
  std::string op, name;
  std::vector<std::string> in, out;
  std::unordered_map<std::string, Attribute> attr;
};

Attribute ParseAttribute(const std::vector<std::uint8_t>& b, std::string& name) {
  Reader r{b}; Attribute a;
  while (!r.eof()) {
    const auto [field, wire] = r.Tag();
    if (field == 1) { auto x = r.Bytes(); name.assign(x.begin(), x.end()); }
    else if (field == 2) { if (wire != 5) Fail("bad float attribute"); const auto x = r.Fixed32(); std::memcpy(&a.f, &x, 4); }
    else if (field == 3) a.i = static_cast<std::int64_t>(r.Varint());
    else if (field == 4) { auto x = r.Bytes(); a.s.assign(x.begin(), x.end()); }
    else if (field == 7) {
      if (wire == 2) { auto x = r.Bytes(); auto v = PackedFloats(x); a.floats.insert(a.floats.end(), v.begin(), v.end()); }
      else if (wire == 5) { const auto x = r.Fixed32(); float v{}; std::memcpy(&v, &x, 4); a.floats.push_back(v); }
      else Fail("bad floats attribute");
    } else if (field == 8) {
      if (wire == 2) { auto x = r.Bytes(); auto v = PackedInts(x); a.ints.insert(a.ints.end(), v.begin(), v.end()); }
      else if (wire == 0) a.ints.push_back(static_cast<std::int64_t>(r.Varint()));
      else Fail("bad ints attribute");
    }
    else r.Skip(wire);
  }
  return a;
}

Tensor ParseTensor(const std::vector<std::uint8_t>& b, std::string& name) {
  Reader r{b}; std::vector<std::int64_t> shape; int type{}; std::vector<std::uint8_t> raw; std::vector<float> floats;
  while (!r.eof()) {
    const auto [field, wire] = r.Tag();
    if (field == 1) {
      if (wire == 2) { auto x = r.Bytes(); auto v = PackedInts(x); shape.insert(shape.end(), v.begin(), v.end()); }
      else if (wire == 0) shape.push_back(static_cast<std::int64_t>(r.Varint()));
      else Fail("bad tensor dims");
    }
    else if (field == 2) type = int(r.Varint());
    else if (field == 4) {
      if (wire == 2) { auto x = r.Bytes(); auto v = PackedFloats(x); floats.insert(floats.end(), v.begin(), v.end()); }
      else if (wire == 5) { const auto x = r.Fixed32(); float v{}; std::memcpy(&v, &x, 4); floats.push_back(v); }
      else Fail("bad tensor float data");
    }
    else if (field == 8) { auto x = r.Bytes(); name.assign(x.begin(), x.end()); }
    else if (field == 9) raw = r.Bytes();
    else r.Skip(wire);
  }
  Tensor t{shape, {}};
  if (type == 1) {
    if (!raw.empty()) { if (raw.size() % 4) Fail("bad raw float data"); t.data.resize(raw.size() / 4); std::memcpy(t.data.data(), raw.data(), raw.size()); }
    else t.data = std::move(floats);
  } else if (type == 6 || type == 7) {  // INT32 / INT64 shape constants
    const std::size_t unit = type == 6 ? 4 : 8;
    if (raw.size() % unit) Fail("bad raw integer data");
    t.data.resize(raw.size() / unit);
    for (std::size_t i = 0; i < t.data.size(); ++i) {
      std::int64_t v{};
      if (type == 6) { std::int32_t x{}; std::memcpy(&x, raw.data() + i * unit, unit); v = x; }
      else std::memcpy(&v, raw.data() + i * unit, unit);
      t.data[i] = static_cast<float>(v);
    }
  } else {
    Fail("unsupported initializer data type " + std::to_string(type));
  }
  if (Elements(t.shape) != t.data.size()) Fail("initializer size mismatch");
  return t;
}

Node ParseNode(const std::vector<std::uint8_t>& b) {
  Reader r{b}; Node n;
  while (!r.eof()) {
    const auto [field, wire] = r.Tag();
    if (field == 1) { auto x = r.Bytes(); n.in.emplace_back(x.begin(), x.end()); }
    else if (field == 2) { auto x = r.Bytes(); n.out.emplace_back(x.begin(), x.end()); }
    else if (field == 3) { auto x = r.Bytes(); n.name.assign(x.begin(), x.end()); }
    else if (field == 4) { auto x = r.Bytes(); n.op.assign(x.begin(), x.end()); }
    else if (field == 5) { auto x = r.Bytes(); std::string key; auto a = ParseAttribute(x, key); n.attr.emplace(std::move(key), std::move(a)); }
    else r.Skip(wire);
  }
  return n;
}

struct GraphData { std::vector<Node> nodes; std::unordered_map<std::string, Tensor> initializers; std::vector<std::string> inputs, outputs; int opset{}; };

void FuseActivationChains(GraphData& graph) {
  std::unordered_map<std::string, std::size_t> consumers;
  std::unordered_set<std::string> outputs(graph.outputs.begin(), graph.outputs.end());
  for (const auto& node : graph.nodes) for (const auto& name : node.in) {
    if (!name.empty()) ++consumers[name];
  }
  std::vector<Node> fused;
  fused.reserve(graph.nodes.size());
  // Fusing the full SE block eliminates several graph dispatches and a
  // full-map broadcast result, but tiny's compact channel dimensions do not
  // amortize that bookkeeping.  Large medium-detector blocks do.  Keep an
  // explicit all-shapes switch for deployment experiments, while making the
  // proven large-channel subset the portable default.
  const bool force_all_se_gate_fusion =
      std::getenv("PPOCR_ENABLE_SE_GATE_FUSION") != nullptr;
  const bool disable_se_gate_fusion =
      std::getenv("PPOCR_DISABLE_SE_GATE_FUSION") != nullptr;
  int default_se_gate_minimum_channels = 32;
  if (const char* configured = std::getenv("PPOCR_SE_GATE_FUSION_MIN_CHANNELS")) {
    char* end = nullptr;
    const long parsed = std::strtol(configured, &end, 10);
    if (end != configured && *end == '\0' && parsed > 0 && parsed <= 65536) {
      default_se_gate_minimum_channels = static_cast<int>(parsed);
    }
  }
  for (std::size_t i = 0; i < graph.nodes.size();) {
    const auto& a = graph.nodes[i];
    // PP-OCRv6 detector squeeze-excitation is an especially costly chain on
    // large pages:
    //   ReduceMean(X) -> 1x1 -> Relu -> 1x1 -> HardSigmoid -> Mul(X, gate).
    // The final channel gate is only [N,C,1,1], whereas `Mul` used to allocate
    // a second full [N,C,H,W] feature map.  Collapse the graph-proven chain
    // so its executor can retain X's dying storage, form the compact gate,
    // and apply it in place.  This is deliberately restricted to the exact
    // NCHW/global-mean/1x1 form emitted by all three PP-OCRv6 detector sizes;
    // any general ONNX broadcast or different reduction keeps its nodes.
    if (!disable_se_gate_fusion && i + 5 < graph.nodes.size() && a.op == "ReduceMean" &&
        a.in.size() == 1 && a.out.size() == 1 && consumers[a.out[0]] == 1 &&
        !outputs.contains(a.out[0]) && graph.nodes[i + 1].op == "Conv" &&
        graph.nodes[i + 2].op == "Relu" && graph.nodes[i + 3].op == "Conv" &&
        graph.nodes[i + 4].op == "HardSigmoid" && graph.nodes[i + 5].op == "Mul") {
      const auto& reduce = a;
      const auto& first = graph.nodes[i + 1];
      const auto& relu = graph.nodes[i + 2];
      const auto& second = graph.nodes[i + 3];
      const auto& hard_sigmoid = graph.nodes[i + 4];
      const auto& multiply = graph.nodes[i + 5];
      const auto axes = reduce.attr.contains("axes") ? reduce.attr.at("axes").ints
                                                       : std::vector<std::int64_t>{};
      const auto keepdims = reduce.attr.contains("keepdims") &&
          reduce.attr.at("keepdims").i != 0 ? reduce.attr.at("keepdims").i : 1;
      const auto pointwise = [&](const Node& node) {
        const auto weights = node.in.size() >= 2 ? graph.initializers.find(node.in[1])
                                                  : graph.initializers.end();
        const auto strides = node.attr.contains("strides") ? node.attr.at("strides").ints
                                                              : std::vector<std::int64_t>{1, 1};
        const auto pads = node.attr.contains("pads") ? node.attr.at("pads").ints
                                                        : std::vector<std::int64_t>{0, 0, 0, 0};
        const auto dilations = node.attr.contains("dilations") ? node.attr.at("dilations").ints
                                                                  : std::vector<std::int64_t>{1, 1};
        const auto group = node.attr.contains("group") ? node.attr.at("group").i : 1;
        return node.in.size() == 3 && weights != graph.initializers.end() &&
            graph.initializers.contains(node.in[2]) && weights->second.shape.size() == 4 &&
            weights->second.shape[2] == 1 && weights->second.shape[3] == 1 &&
            strides == std::vector<std::int64_t>{1, 1} &&
            pads == std::vector<std::int64_t>{0, 0, 0, 0} &&
            dilations == std::vector<std::int64_t>{1, 1} && group == 1;
      };
      const bool ordered = first.in.size() == 3 && first.in[0] == reduce.out[0] &&
          first.out.size() == 1 && consumers[first.out[0]] == 1 &&
          relu.in.size() == 1 && relu.in[0] == first.out[0] && relu.out.size() == 1 &&
          consumers[relu.out[0]] == 1 && second.in.size() == 3 &&
          second.in[0] == relu.out[0] && second.out.size() == 1 &&
          consumers[second.out[0]] == 1 && hard_sigmoid.in.size() == 1 &&
          hard_sigmoid.in[0] == second.out[0] && hard_sigmoid.out.size() == 1 &&
          consumers[hard_sigmoid.out[0]] == 1 && multiply.in.size() == 2 &&
          multiply.out.size() == 1 && !outputs.contains(multiply.out[0]) &&
          ((multiply.in[0] == reduce.in[0] && multiply.in[1] == hard_sigmoid.out[0]) ||
           (multiply.in[1] == reduce.in[0] && multiply.in[0] == hard_sigmoid.out[0]));
      const auto first_weights = first.in.size() >= 2 ? graph.initializers.find(first.in[1])
                                                       : graph.initializers.end();
      const auto second_weights = second.in.size() >= 2 ? graph.initializers.find(second.in[1])
                                                         : graph.initializers.end();
      const auto first_bias = first.in.size() >= 3 ? graph.initializers.find(first.in[2])
                                                    : graph.initializers.end();
      const auto second_bias = second.in.size() >= 3 ? graph.initializers.find(second.in[2])
                                                      : graph.initializers.end();
      if (ordered && first_weights != graph.initializers.end() &&
          second_weights != graph.initializers.end() &&
          axes == std::vector<std::int64_t>{2, 3} && keepdims == 1 &&
          consumers[reduce.in[0]] == 2 && pointwise(first) && pointwise(second) &&
          first_weights->second.shape[1] > 0 &&
          first_weights->second.shape[1] == second_weights->second.shape[0] &&
          first_weights->second.shape[0] == second_weights->second.shape[1] &&
          first_bias != graph.initializers.end() && second_bias != graph.initializers.end() &&
          first_bias->second.data.size() == std::size_t(first_weights->second.shape[0]) &&
          second_bias->second.data.size() == std::size_t(second_weights->second.shape[0]) &&
          (force_all_se_gate_fusion ||
           first_weights->second.shape[1] >= default_se_gate_minimum_channels)) {
        Node node;
        node.op = "FusedSEGateMul";
        node.name = reduce.name.empty() ? "fused_se_gate_mul" : reduce.name + "_se_gate_mul";
        node.in = {reduce.in[0], first.in[1], first.in[2], second.in[1], second.in[2]};
        node.out = multiply.out;
        node.attr["alpha"].f = hard_sigmoid.attr.contains("alpha")
            ? hard_sigmoid.attr.at("alpha").f : .2F;
        node.attr["beta"].f = hard_sigmoid.attr.contains("beta")
            ? hard_sigmoid.attr.at("beta").f : .5F;
        fused.push_back(std::move(node));
        i += 6;
        continue;
      }
    }
    // In inference BatchNorm is affine and its parameters are constants. Fold
    // `Conv -> BatchNorm` into the convolution weights/bias at model-load
    // time. This removes a complete activation traversal and allows the
    // existing Conv+ReLU fusion to fire after the graph is rebuilt.
    if (i + 1 < graph.nodes.size() && a.op == "Conv" && a.out.size() == 1 &&
        consumers[a.out[0]] == 1 && !outputs.contains(a.out[0]) &&
        graph.nodes[i + 1].op == "BatchNormalization" &&
        graph.nodes[i + 1].in.size() == 5 && graph.nodes[i + 1].in[0] == a.out[0] &&
        graph.nodes[i + 1].out.size() == 1 &&
        consumers[graph.nodes[i + 1].out[0]] == 1 && a.in.size() >= 2) {
      const auto& bn = graph.nodes[i + 1];
      const auto weights = graph.initializers.find(a.in[1]);
      const auto scale = graph.initializers.find(bn.in[1]);
      const auto bias = graph.initializers.find(bn.in[2]);
      const auto mean = graph.initializers.find(bn.in[3]);
      const auto variance = graph.initializers.find(bn.in[4]);
      if (weights != graph.initializers.end() && scale != graph.initializers.end() &&
          bias != graph.initializers.end() && mean != graph.initializers.end() &&
          variance != graph.initializers.end() && weights->second.shape.size() == 4) {
        const auto channels = std::size_t(weights->second.shape[0]);
        if (scale->second.data.size() == channels && bias->second.data.size() == channels &&
            mean->second.data.size() == channels && variance->second.data.size() == channels) {
          const auto epsilon = bn.attr.contains("epsilon") ? bn.attr.at("epsilon").f : 1e-5F;
          const std::string folded_weight = "__ppocr_folded_weight_" + std::to_string(i);
          const std::string folded_bias = "__ppocr_folded_bias_" + std::to_string(i);
          Tensor new_weights = weights->second;
          Tensor new_bias{{static_cast<std::int64_t>(channels)}, std::vector<float>(channels)};
          const auto per_channel = new_weights.data.size() / channels;
          const float* old_bias = nullptr;
          if (a.in.size() > 2) {
            const auto convolution_bias = graph.initializers.find(a.in[2]);
            if (convolution_bias == graph.initializers.end() ||
                convolution_bias->second.data.size() != channels) continue;
            old_bias = convolution_bias->second.data.data();
          }
          for (std::size_t channel = 0; channel < channels; ++channel) {
            const float factor = scale->second.data[channel] /
                std::sqrt(variance->second.data[channel] + epsilon);
            const float offset = bias->second.data[channel] - mean->second.data[channel] * factor;
            float* data = new_weights.data.data() + channel * per_channel;
            for (std::size_t k = 0; k < per_channel; ++k) data[k] *= factor;
            new_bias.data[channel] = (old_bias ? old_bias[channel] : 0.F) * factor + offset;
          }
          graph.initializers.emplace(folded_weight, std::move(new_weights));
          graph.initializers.emplace(folded_bias, std::move(new_bias));
          Node node = a;
          node.name = "fused_conv_batchnorm";
          node.in.resize(2);
          node.in[1] = folded_weight;
          node.in.push_back(folded_bias);
          node.out = bn.out;
          fused.push_back(std::move(node));
          i += 2;
          continue;
        }
      }
    }

    // MobileNet detector/recognizer stages repeatedly use an unactivated
    // depthwise Conv directly followed by an unpadded pointwise Conv.  The
    // Vulkan backend already has a one-submission implementation that retains
    // this intermediate on device; CPU keeps the established pair of SIMD
    // kernels.  A depthwise layer may legally omit its bias in ONNX, though;
    // supply an immutable zero tensor at graph-load time so the common exact
    // fused route is not discarded merely for that optional affine input.
    if (std::getenv("PPOCR_DISABLE_DEPTHWISE_POINTWISE_FUSION") == nullptr &&
        i + 1 < graph.nodes.size() && a.op == "Conv" &&
        (a.in.size() == 2 || a.in.size() == 3) &&
        a.out.size() == 1 && consumers[a.out[0]] == 1 && !outputs.contains(a.out[0]) &&
        graph.nodes[i + 1].op == "Conv" &&
        (graph.nodes[i + 1].in.size() == 2 || graph.nodes[i + 1].in.size() == 3) &&
        graph.nodes[i + 1].out.size() == 1 && !outputs.contains(graph.nodes[i + 1].out[0])) {
      const auto& pointwise = graph.nodes[i + 1];
      const auto depthwise_weights = graph.initializers.find(a.in[1]);
      const auto depthwise_bias = a.in.size() == 3
          ? graph.initializers.find(a.in[2]) : graph.initializers.end();
      const auto pointwise_weights = graph.initializers.find(pointwise.in[1]);
      const auto pointwise_bias = pointwise.in.size() == 3
          ? graph.initializers.find(pointwise.in[2]) : graph.initializers.end();
      const auto attrs = [](const Node& node, const char* name,
                            std::vector<std::int64_t> fallback) {
        const auto it = node.attr.find(name);
        return it == node.attr.end() ? fallback : it->second.ints;
      };
      const auto depthwise_strides = attrs(a, "strides", {1, 1});
      const auto depthwise_pads = attrs(a, "pads", {0, 0, 0, 0});
      const auto depthwise_dilations = attrs(a, "dilations", {1, 1});
      const auto pointwise_strides = attrs(pointwise, "strides", {1, 1});
      const auto pointwise_pads = attrs(pointwise, "pads", {0, 0, 0, 0});
      const auto pointwise_dilations = attrs(pointwise, "dilations", {1, 1});
      const auto group = a.attr.contains("group") ? a.attr.at("group").i : std::int64_t{1};
      const auto pointwise_group = pointwise.attr.contains("group")
          ? pointwise.attr.at("group").i : std::int64_t{1};
      const bool depthwise = depthwise_weights != graph.initializers.end() &&
          depthwise_weights->second.shape.size() == 4 &&
          group > 0 && depthwise_weights->second.shape[0] == group &&
          depthwise_weights->second.shape[1] == 1 &&
          depthwise_strides.size() == 2 && depthwise_pads.size() == 4 &&
          depthwise_dilations == std::vector<std::int64_t>{1, 1} &&
          (a.attr.find("auto_pad") == a.attr.end() || a.attr.at("auto_pad").s == "NOTSET");
      // Tiny FPN laterals (Conv.78/79) export 5x5 DW then a bias-free 1x1.
      // Fusing them removes the 16x160x704 intermediate. Disable with
      // PPOCR_DISABLE_DEPTHWISE_POINTWISE_NOBIAS.
      static const bool allow_nobias =
          std::getenv("PPOCR_DISABLE_DEPTHWISE_POINTWISE_NOBIAS") == nullptr;
      const bool pw_bias_ok = pointwise.in.size() == 3
          ? (pointwise_bias != graph.initializers.end() &&
             pointwise_bias->second.data.size() ==
                 std::size_t(pointwise_weights != graph.initializers.end()
                                 ? pointwise_weights->second.shape[0]
                                 : 0))
          : allow_nobias;
      const bool pw = pointwise.in[0] == a.out[0] &&
          pointwise_weights != graph.initializers.end() &&
          pointwise_weights->second.shape.size() == 4 &&
          pointwise_weights->second.shape[2] == 1 && pointwise_weights->second.shape[3] == 1 &&
          pointwise_group == 1 && pw_bias_ok &&
          pointwise_strides == std::vector<std::int64_t>{1, 1} &&
          pointwise_pads == std::vector<std::int64_t>{0, 0, 0, 0} &&
          pointwise_dilations == std::vector<std::int64_t>{1, 1} &&
          (pointwise.attr.find("auto_pad") == pointwise.attr.end() ||
           pointwise.attr.at("auto_pad").s == "NOTSET") &&
          pointwise_weights->second.shape[1] == group;
      if (depthwise && pw) {
        std::string depthwise_bias_name;
        if (depthwise_bias != graph.initializers.end()) {
          if (depthwise_bias->second.data.size() != std::size_t(group)) continue;
          depthwise_bias_name = a.in[2];
        } else {
          depthwise_bias_name = "__ppocr_zero_depthwise_bias_" + std::to_string(i);
          graph.initializers.emplace(depthwise_bias_name,
              Tensor{{group}, std::vector<float>(static_cast<std::size_t>(group), 0.F)});
        }
        std::string pointwise_bias_name;
        if (pointwise.in.size() == 3) {
          pointwise_bias_name = pointwise.in[2];
        } else {
          // Tiny FPN laterals export 5x5 DW then a bias-free 1x1 (Conv.78/79).
          pointwise_bias_name = "__ppocr_zero_pointwise_bias_" + std::to_string(i);
          graph.initializers.emplace(
              pointwise_bias_name,
              Tensor{{pointwise_weights->second.shape[0]},
                     std::vector<float>(static_cast<std::size_t>(
                                            pointwise_weights->second.shape[0]),
                                        0.F)});
        }
        Node node = a;
        node.op = "FusedDepthwisePointwiseConv";
        node.name = a.name.empty() ? "fused_depthwise_pointwise" :
            a.name + "_pointwise";
        node.in = {a.in[0], a.in[1], depthwise_bias_name, pointwise.in[1], pointwise_bias_name};
        node.out = pointwise.out;
        fused.push_back(std::move(node));
        i += 2;
        continue;
      }
    }

    // Transformer MLP gates export MatMul + bias + Sigmoid + Mul. Fold the
    // bias into GEMM initialization and overwrite that result with Swish;
    // the unfused form otherwise retains the GEMM output plus a full sigmoid
    // activation before the final multiply. All edges are strict single-use
    // except the MatMul output's two internal gate consumers.
    if (i + 3 < graph.nodes.size() && a.op == "MatMul" && a.out.size() == 1 &&
        consumers[a.out[0]] == 1 && !outputs.contains(a.out[0]) &&
        graph.nodes[i + 1].op == "Add" && graph.nodes[i + 1].in.size() == 2 &&
        graph.nodes[i + 1].out.size() == 1 &&
        (graph.nodes[i + 1].in[0] == a.out[0] || graph.nodes[i + 1].in[1] == a.out[0]) &&
        consumers[graph.nodes[i + 1].out[0]] == 2 &&
        graph.nodes[i + 2].op == "Sigmoid" && graph.nodes[i + 2].in.size() == 1 &&
        graph.nodes[i + 2].in[0] == graph.nodes[i + 1].out[0] &&
        graph.nodes[i + 2].out.size() == 1 && consumers[graph.nodes[i + 2].out[0]] == 1 &&
        graph.nodes[i + 3].op == "Mul" && graph.nodes[i + 3].in.size() == 2 &&
        ((graph.nodes[i + 3].in[0] == graph.nodes[i + 1].out[0] &&
          graph.nodes[i + 3].in[1] == graph.nodes[i + 2].out[0]) ||
         (graph.nodes[i + 3].in[1] == graph.nodes[i + 1].out[0] &&
          graph.nodes[i + 3].in[0] == graph.nodes[i + 2].out[0]))) {
      const auto bias_name = graph.nodes[i + 1].in[0] == a.out[0]
          ? graph.nodes[i + 1].in[1] : graph.nodes[i + 1].in[0];
      const auto bias = graph.initializers.find(bias_name);
      const auto weight = graph.initializers.find(a.in[1]);
      if (bias != graph.initializers.end() && weight != graph.initializers.end() &&
          weight->second.shape.size() == 2 && bias->second.data.size() ==
              std::size_t(weight->second.shape[1])) {
        Node node = a;
        node.op = "FusedMatMulBiasSwish";
        node.name = "fused_matmul_bias_swish";
        node.in.push_back(bias_name);
        node.out = graph.nodes[i + 3].out;
        fused.push_back(std::move(node));
        i += 4;
        continue;
      }
    }
    // Recognition head: MatMul logits followed by a vocabulary bias. Fold the
    // bias into GEMM initialization so the 6,906-wide output is written once.
    if (i + 1 < graph.nodes.size() && a.op == "MatMul" && a.out.size() == 1 &&
        consumers[a.out[0]] == 1 && graph.nodes[i+1].op == "Add" &&
        graph.nodes[i+1].in.size() == 2 &&
        (graph.nodes[i+1].in[0] == a.out[0] || graph.nodes[i+1].in[1] == a.out[0])) {
      const auto bias_name = graph.nodes[i+1].in[0] == a.out[0]
          ? graph.nodes[i+1].in[1] : graph.nodes[i+1].in[0];
      const auto bias = graph.initializers.find(bias_name);
      const auto weight = graph.initializers.find(a.in[1]);
      if (bias != graph.initializers.end() && weight != graph.initializers.end() &&
          weight->second.shape.size() == 2 && bias->second.data.size() ==
              std::size_t(weight->second.shape[1])) {
        Node node = a;
        node.op = "FusedMatMulBias";
        node.name = "fused_matmul_bias";
        node.in.push_back(bias_name);
        node.out = graph.nodes[i+1].out;
        fused.push_back(std::move(node));
        i += 2;
        continue;
      }
    }
    // Small/medium recognizers export every transformer LayerNorm as:
    // ReduceMean -> Sub -> Pow(2) -> ReduceMean -> Add(eps) -> Sqrt -> Div
    // -> Mul(gamma) -> Add(beta) -> Identity. All parameters are immutable
    // and each internal value is single-use. Collapse this exact semantic
    // sequence at load time so inference makes one output activation rather
    // than materializing five transformer-sized intermediates.
    if (i + 8 < graph.nodes.size() && a.op == "ReduceMean" && a.out.size() == 1 &&
        consumers[a.out[0]] == 1 && !outputs.contains(a.out[0])) {
      const auto& sub = graph.nodes[i + 1];
      const auto& power = graph.nodes[i + 2];
      const auto& variance = graph.nodes[i + 3];
      const auto& epsilon_add = graph.nodes[i + 4];
      const auto& sqrt = graph.nodes[i + 5];
      const auto& divide = graph.nodes[i + 6];
      const auto& scale = graph.nodes[i + 7];
      const auto& shift = graph.nodes[i + 8];
      const auto other = [](const Node& node, const std::string& value) -> std::string {
        return node.in.size() == 2 && node.in[0] == value ? node.in[1] :
               (node.in.size() == 2 && node.in[1] == value ? node.in[0] : std::string{});
      };
      const auto has_axis_minus_one = [](const Node& node) {
        const auto axes_it = node.attr.find("axes");
        const auto keepdims_it = node.attr.find("keepdims");
        const auto& axes = axes_it == node.attr.end() ? std::vector<std::int64_t>{}
                                                       : axes_it->second.ints;
        // In this compact protobuf attribute reader, an absent integer field
        // and ONNX's explicit default 1 both arrive as zero. ReduceMean's
        // exported nodes omit keepdims, so treat zero as its ONNX default.
        const auto keepdims = keepdims_it == node.attr.end() || keepdims_it->second.i == 0
            ? std::int64_t{1} : keepdims_it->second.i;
        return axes.size() == 1 && axes[0] == -1 && keepdims != 0;
      };
      const auto scalar_initializer = [&](const std::string& name, float& value) {
        const auto constant = graph.initializers.find(name);
        if (constant == graph.initializers.end() || constant->second.data.size() != 1) return false;
        value = constant->second.data[0];
        return true;
      };
      const std::string x = a.in.size() == 1 ? a.in[0] : std::string{};
      const std::string centered = sub.out.size() == 1 ? sub.out[0] : std::string{};
      const std::string squared = power.out.size() == 1 ? power.out[0] : std::string{};
      const std::string variance_value = variance.out.size() == 1 ? variance.out[0] : std::string{};
      const std::string variance_epsilon = epsilon_add.out.size() == 1 ? epsilon_add.out[0] : std::string{};
      const std::string standard_deviation = sqrt.out.size() == 1 ? sqrt.out[0] : std::string{};
      const std::string normalized = divide.out.size() == 1 ? divide.out[0] : std::string{};
      const std::string scaled = scale.out.size() == 1 ? scale.out[0] : std::string{};
      float exponent{}, epsilon{};
      const auto gamma_name = other(scale, normalized);
      const auto beta_name = other(shift, scaled);
      const auto exponent_name = other(power, centered);
      const auto epsilon_name = other(epsilon_add, variance_value);
      const auto gamma = graph.initializers.find(gamma_name);
      const auto beta = graph.initializers.find(beta_name);
      if (!x.empty() && has_axis_minus_one(a) && sub.op == "Sub" && sub.in.size() == 2 &&
          ((sub.in[0] == x && sub.in[1] == a.out[0])) && power.op == "Pow" &&
          power.in.size() == 2 && !exponent_name.empty() && scalar_initializer(exponent_name, exponent) &&
          exponent == 2.F && power.in[0] == centered && variance.op == "ReduceMean" &&
          variance.in.size() == 1 && variance.in[0] == squared && has_axis_minus_one(variance) &&
          epsilon_add.op == "Add" && !epsilon_name.empty() &&
          scalar_initializer(epsilon_name, epsilon) && sqrt.op == "Sqrt" && sqrt.in.size() == 1 &&
          sqrt.in[0] == variance_epsilon && divide.op == "Div" && divide.in.size() == 2 &&
          divide.in[0] == centered && divide.in[1] == standard_deviation && scale.op == "Mul" &&
          !gamma_name.empty() && shift.op == "Add" && !beta_name.empty() &&
          gamma != graph.initializers.end() && beta != graph.initializers.end() &&
          gamma->second.data.size() == beta->second.data.size() && !gamma->second.data.empty() &&
          consumers[centered] == 2 && consumers[squared] == 1 &&
          consumers[variance_value] == 1 && consumers[variance_epsilon] == 1 &&
          consumers[standard_deviation] == 1 && consumers[normalized] == 1 && consumers[scaled] == 1) {
        Node node;
        node.op = "FusedLayerNorm";
        node.name = "fused_layernorm";
        node.in = {x, gamma_name, beta_name};
        node.out = shift.out;
        node.attr["epsilon"].f = epsilon;
        fused.push_back(std::move(node));
        i += 9;
        continue;
      }
    }
    // A frequent recognizer inverted-residual suffix is a pointwise
    // projection plus shortcut followed by the canonical exact Swish gate:
    // Conv(x) -> Add(shortcut) -> Sigmoid(sum) -> Mul(sum, gate). Keeping the
    // full graph-proven chain together gives Vulkan one batch submission and
    // removes the otherwise materialized sigmoid activation on the CPU path.
    if (std::getenv("PPOCR_DISABLE_POINTWISE_ADD_SWISH_FUSION") == nullptr &&
        i + 3 < graph.nodes.size() && a.op == "Conv" && a.in.size() >= 2 &&
        a.out.size() == 1 && consumers[a.out[0]] == 1 &&
        graph.nodes[i + 1].op == "Add" && graph.nodes[i + 1].in.size() == 2 &&
        graph.nodes[i + 1].out.size() == 1 && consumers[graph.nodes[i + 1].out[0]] == 2 &&
        graph.nodes[i + 2].op == "Sigmoid" && graph.nodes[i + 2].in.size() == 1 &&
        graph.nodes[i + 2].in[0] == graph.nodes[i + 1].out[0] &&
        graph.nodes[i + 2].out.size() == 1 && consumers[graph.nodes[i + 2].out[0]] == 1 &&
        graph.nodes[i + 3].op == "Mul" && graph.nodes[i + 3].in.size() == 2 &&
        ((graph.nodes[i + 3].in[0] == graph.nodes[i + 1].out[0] &&
          graph.nodes[i + 3].in[1] == graph.nodes[i + 2].out[0]) ||
         (graph.nodes[i + 3].in[1] == graph.nodes[i + 1].out[0] &&
          graph.nodes[i + 3].in[0] == graph.nodes[i + 2].out[0]))) {
      const auto& add = graph.nodes[i + 1];
      const auto residual = add.in[0] == a.out[0] ? add.in[1] : add.in[0];
      const auto weights = graph.initializers.find(a.in[1]);
      const auto attr_ints = [&](const char* key, std::vector<std::int64_t> fallback) {
        const auto it = a.attr.find(key);
        return it == a.attr.end() ? fallback : it->second.ints;
      };
      const auto strides = attr_ints("strides", {1, 1});
      const auto pads = attr_ints("pads", {0, 0, 0, 0});
      const auto dilations = attr_ints("dilations", {1, 1});
      const auto group_it = a.attr.find("group");
      const auto auto_pad_it = a.attr.find("auto_pad");
      const bool pointwise = weights != graph.initializers.end() &&
          weights->second.shape.size() == 4 && weights->second.shape[2] == 1 &&
          weights->second.shape[3] == 1 && (group_it == a.attr.end() || group_it->second.i == 1) &&
          (auto_pad_it == a.attr.end() || auto_pad_it->second.s == "NOTSET") && strides.size() == 2 &&
          pads.size() == 4 && dilations.size() == 2 && strides[0] == 1 && strides[1] == 1 &&
          dilations[0] == 1 && dilations[1] == 1 && pads[0] == 0 && pads[1] == 0 &&
          pads[2] == 0 && pads[3] == 0;
      if (pointwise && !residual.empty() && !graph.initializers.contains(residual)) {
        Node node = a;
        node.op = "FusedPointwiseConvAddSwish";
        node.name = a.name.empty() ? "fused_pointwise_conv_add_swish" : a.name + "_add_swish";
        node.in = {a.in[0], a.in[1], a.in.size() > 2 ? a.in[2] : "", residual};
        node.out = graph.nodes[i + 3].out;
        fused.push_back(std::move(node));
        i += 4;
        continue;
      }
    }
    // Residual pointwise projections are common in the recognizer.  Fuse
    // only a strict dynamic equal-shape candidate here; the executor repeats
    // the shape check before selecting the kernel, so broadcast Add and any
    // unexpected export layout retain ordinary ONNX semantics.
    if (i + 2 < graph.nodes.size() && a.op == "Conv" && a.in.size() >= 2 &&
        a.out.size() == 1 && consumers[a.out[0]] == 1 &&
        graph.nodes[i + 1].op == "Add" && graph.nodes[i + 1].in.size() == 2 &&
        graph.nodes[i + 1].out.size() == 1 && consumers[graph.nodes[i + 1].out[0]] == 1 &&
        !outputs.contains(graph.nodes[i + 1].out[0]) && graph.nodes[i + 2].op == "Relu" &&
        graph.nodes[i + 2].in.size() == 1 && graph.nodes[i + 2].in[0] == graph.nodes[i + 1].out[0] &&
        (graph.nodes[i + 1].in[0] == a.out[0] || graph.nodes[i + 1].in[1] == a.out[0])) {
      const auto& add = graph.nodes[i + 1];
      const auto residual = add.in[0] == a.out[0] ? add.in[1] : add.in[0];
      const auto weights = graph.initializers.find(a.in[1]);
      const auto attr_ints = [&](const char* key, std::vector<std::int64_t> fallback) {
        const auto it = a.attr.find(key);
        return it == a.attr.end() ? fallback : it->second.ints;
      };
      const auto strides = attr_ints("strides", {1, 1});
      const auto pads = attr_ints("pads", {0, 0, 0, 0});
      const auto dilations = attr_ints("dilations", {1, 1});
      const auto group_it = a.attr.find("group");
      const auto auto_pad_it = a.attr.find("auto_pad");
      const bool pointwise = weights != graph.initializers.end() &&
          weights->second.shape.size() == 4 && weights->second.shape[2] == 1 &&
          weights->second.shape[3] == 1 && (group_it == a.attr.end() || group_it->second.i == 1) &&
          (auto_pad_it == a.attr.end() || auto_pad_it->second.s == "NOTSET") && strides.size() == 2 &&
          pads.size() == 4 && dilations.size() == 2 && strides[0] == 1 && strides[1] == 1 &&
          dilations[0] == 1 && dilations[1] == 1 && pads[0] == 0 && pads[1] == 0 &&
          pads[2] == 0 && pads[3] == 0;
      if (pointwise && !residual.empty() && !graph.initializers.contains(residual)) {
        Node node = a;
        node.op = "FusedPointwiseConvAddRelu";
        node.name = a.name.empty() ? "fused_pointwise_conv_add_relu" : a.name + "_add_relu";
        node.in = {a.in[0], a.in[1], a.in.size() > 2 ? a.in[2] : "", residual};
        node.out = graph.nodes[i + 2].out;
        fused.push_back(std::move(node));
        i += 3;
        continue;
      }
    }
    // Keep the non-ReLU residual projection in the same one-write form.  The
    // following node need not be an activation: recognizer shortcut branches
    // often feed directly into a later affine/attention stage.  As above,
    // only a dynamic residual is accepted here; an initializer Add can carry
    // broadcast bias semantics and must remain on the ordinary path.
    if (i + 1 < graph.nodes.size() && a.op == "Conv" && a.in.size() >= 2 &&
        a.out.size() == 1 && consumers[a.out[0]] == 1 &&
        graph.nodes[i + 1].op == "Add" && graph.nodes[i + 1].in.size() == 2 &&
        graph.nodes[i + 1].out.size() == 1 &&
        (graph.nodes[i + 1].in[0] == a.out[0] || graph.nodes[i + 1].in[1] == a.out[0])) {
      const auto& add = graph.nodes[i + 1];
      const auto residual = add.in[0] == a.out[0] ? add.in[1] : add.in[0];
      const auto weights = graph.initializers.find(a.in[1]);
      const auto attr_ints = [&](const char* key, std::vector<std::int64_t> fallback) {
        const auto it = a.attr.find(key);
        return it == a.attr.end() ? fallback : it->second.ints;
      };
      const auto strides = attr_ints("strides", {1, 1});
      const auto pads = attr_ints("pads", {0, 0, 0, 0});
      const auto dilations = attr_ints("dilations", {1, 1});
      const auto group_it = a.attr.find("group");
      const auto auto_pad_it = a.attr.find("auto_pad");
      const bool pointwise = weights != graph.initializers.end() &&
          weights->second.shape.size() == 4 && weights->second.shape[2] == 1 &&
          weights->second.shape[3] == 1 && (group_it == a.attr.end() || group_it->second.i == 1) &&
          (auto_pad_it == a.attr.end() || auto_pad_it->second.s == "NOTSET") && strides.size() == 2 &&
          pads.size() == 4 && dilations.size() == 2 && strides[0] == 1 && strides[1] == 1 &&
          dilations[0] == 1 && dilations[1] == 1 && pads[0] == 0 && pads[1] == 0 &&
          pads[2] == 0 && pads[3] == 0;
      if (pointwise && !residual.empty() && !graph.initializers.contains(residual)) {
        Node node = a;
        node.op = "FusedPointwiseConvAdd";
        node.name = a.name.empty() ? "fused_pointwise_conv_add" : a.name + "_add";
        node.in = {a.in[0], a.in[1], a.in.size() > 2 ? a.in[2] : "", residual};
        node.out = add.out;
        fused.push_back(std::move(node));
        i += 2;
        continue;
      }
    }
    // Most recognizer convolutions export their channel bias as a separate
    // `Add([1,C,1,1])`, including the stem layers. Fold that immutable bias
    // even when there is no exporter Identity after Add. This eliminates the
    // full broadcast pass, and, for the frequent following ReLU, lets the
    // existing fused Conv+ReLU node keep the result in one allocation. The
    // shape checks deliberately reject scalar/general broadcasts and groups
    // whose output-channel bias cannot be represented by Conv.
    if (i + 1 < graph.nodes.size() && a.op == "Conv" && a.in.size() >= 2 &&
        a.out.size() == 1 && consumers[a.out[0]] == 1 && !outputs.contains(a.out[0]) &&
        graph.nodes[i + 1].op == "Add" && graph.nodes[i + 1].in.size() == 2 &&
        graph.nodes[i + 1].out.size() == 1 &&
        (graph.nodes[i + 1].in[0] == a.out[0] || graph.nodes[i + 1].in[1] == a.out[0])) {
      const auto& add = graph.nodes[i + 1];
      const auto bias_name = add.in[0] == a.out[0] ? add.in[1] : add.in[0];
      const auto channel_bias = graph.initializers.find(bias_name);
      const auto weights = graph.initializers.find(a.in[1]);
      if (channel_bias != graph.initializers.end() && weights != graph.initializers.end() &&
          weights->second.shape.size() == 4 && channel_bias->second.shape.size() == 4 &&
          channel_bias->second.shape[0] == 1 &&
          channel_bias->second.shape[1] == weights->second.shape[0] &&
          channel_bias->second.shape[2] == 1 && channel_bias->second.shape[3] == 1) {
        const auto channels = std::size_t(weights->second.shape[0]);
        Tensor folded_bias{{static_cast<std::int64_t>(channels)}, std::vector<float>(channels)};
        if (a.in.size() > 2) {
          const auto existing_bias = graph.initializers.find(a.in[2]);
          if (existing_bias == graph.initializers.end() ||
              existing_bias->second.data.size() != channels) {
            fused.push_back(a);
            ++i;
            continue;
          }
          for (std::size_t channel = 0; channel < channels; ++channel) {
            folded_bias.data[channel] = existing_bias->second.data[channel] +
                                        channel_bias->second.data[channel];
          }
        } else {
          folded_bias.data = channel_bias->second.data;
        }
        const std::string folded_name = "__ppocr_folded_channel_bias_direct_" +
                                        std::to_string(i);
        graph.initializers.emplace(folded_name, std::move(folded_bias));
        Node node = a;
        node.name = a.name.empty() ? "fused_conv_channel_bias" : a.name + "_bias";
        node.in.resize(2);
        node.in.push_back(folded_name);
        const bool relu = i + 2 < graph.nodes.size() && consumers[add.out[0]] == 1 &&
            !outputs.contains(add.out[0]) && graph.nodes[i + 2].op == "Relu" &&
            graph.nodes[i + 2].in.size() == 1 && graph.nodes[i + 2].in[0] == add.out[0];
        if (relu) {
          node.op = "FusedConvRelu";
          node.name += "_relu";
          node.out = graph.nodes[i + 2].out;
          fused.push_back(std::move(node));
          i += 3;
        } else {
          node.out = add.out;
          fused.push_back(std::move(node));
          i += 2;
        }
        continue;
      }
    }
    // Conv followed by ReLU is common in the detector stem/head.  The normal
    // executor would first materialize Conv's output then copy it to form the
    // ReLU output. Preserve the Conv attributes/inputs and apply ReLU in
    // place, eliminating one activation-sized allocation and memory sweep.
    if (std::getenv("PPOCR_DISABLE_MAXPOOL_CONCAT_FUSION") == nullptr &&
        i + 1 < graph.nodes.size() && a.op == "MaxPool" && a.out.size() == 1 &&
        consumers[a.out[0]] == 1 && !outputs.contains(a.out[0]) &&
        graph.nodes[i + 1].op == "Concat" && graph.nodes[i + 1].in.size() >= 2 &&
        graph.nodes[i + 1].in[0] == a.out[0] && graph.nodes[i + 1].out.size() == 1) {
      const auto& concat = graph.nodes[i + 1];
      // Stem is MaxPool(Conv.0) || Conv.1 || Conv.2 followed by Conv.3 3x3 s2.
      // Fold the dying 32-channel Concat into a plane-pointer convolution so
      // Conv.3 never reads a materialized concat activation.
      if (std::getenv("PPOCR_DISABLE_MAXPOOL_CONCAT_CONV") == nullptr &&
          i + 2 < graph.nodes.size() && concat.out.size() == 1 &&
          consumers[concat.out[0]] == 1 && !outputs.contains(concat.out[0])) {
        const auto kernel_it = a.attr.find("kernel_shape");
        const auto strides_it = a.attr.find("strides");
        const auto pad_it = a.attr.find("auto_pad");
        const bool same_upper_pool =
            kernel_it != a.attr.end() && kernel_it->second.ints.size() == 2 &&
            kernel_it->second.ints[0] == 2 && kernel_it->second.ints[1] == 2 &&
            (strides_it == a.attr.end() ||
             (strides_it->second.ints.size() >= 2 && strides_it->second.ints[0] == 1 &&
              strides_it->second.ints[1] == 1)) &&
            pad_it != a.attr.end() && pad_it->second.s == "SAME_UPPER";
        const auto& conv = graph.nodes[i + 2];
        const bool conv_ok =
            (conv.op == "Conv" || conv.op == "FusedConvRelu") &&
            conv.in.size() >= 2 && conv.in[0] == concat.out[0] &&
            conv.out.size() == 1;
        if (same_upper_pool && conv_ok) {
          const auto weights = graph.initializers.find(conv.in[1]);
          bool relu = conv.op == "FusedConvRelu";
          int skip = 3;
          if (!relu && i + 3 < graph.nodes.size() &&
              consumers[conv.out[0]] == 1 && !outputs.contains(conv.out[0]) &&
              graph.nodes[i + 3].op == "Relu" && graph.nodes[i + 3].in.size() == 1 &&
              graph.nodes[i + 3].in[0] == conv.out[0]) {
            relu = true;
            skip = 4;
          }
          if (weights != graph.initializers.end() &&
              weights->second.shape.size() == 4 &&
              weights->second.shape[2] == 3 && weights->second.shape[3] == 3) {
            Node node = conv;
            node.op = relu ? "FusedMaxPoolConcatConvRelu" : "FusedMaxPoolConcatConv";
            node.name = a.name.empty() ? "fused_maxpool_concat_conv"
                                       : a.name + "_concat_conv";
            node.in = concat.in;
            node.in[0] = a.in[0];
            node.in.push_back(conv.in[1]);
            if (conv.in.size() > 2) node.in.push_back(conv.in[2]);
            if (relu && conv.op == "Conv") node.out = graph.nodes[i + 3].out;
            else node.out = conv.out;
            if (a.attr.contains("auto_pad")) {
              node.attr["__ppocr_pool_auto_pad"] = a.attr.at("auto_pad");
            }
            if (concat.attr.contains("axis")) {
              node.attr["axis"] = concat.attr.at("axis");
            }
            fused.push_back(std::move(node));
            i += skip;
            continue;
          }
        }
      }
      // Stem is MaxPool(Conv.0) || Conv.2. Fold the dying 2x2 SAME_UPPER
      // Conv+ReLU into the concat destination so Conv.2 never materializes.
      if (std::getenv("PPOCR_DISABLE_CONV_MAXPOOL_CONCAT") == nullptr &&
          !fused.empty() && fused.back().op == "FusedConvRelu" &&
          concat.in.size() == 2 && concat.in[1] == fused.back().out[0] &&
          consumers[fused.back().out[0]] == 1) {
        Node conv = std::move(fused.back());
        fused.pop_back();
        Node node = a;
        node.op = "FusedConvMaxPoolConcat";
        node.name = conv.name.empty() ? "fused_conv_maxpool_concat" :
            conv.name + "_maxpool_concat";
        node.in = {a.in[0], conv.in[0], conv.in[1]};
        if (conv.in.size() > 2) node.in.push_back(conv.in[2]);
        node.out = concat.out;
        node.attr = conv.attr;
        if (a.attr.contains("auto_pad")) node.attr["auto_pad"] = a.attr.at("auto_pad");
        if (concat.attr.contains("axis")) node.attr["axis"] = concat.attr.at("axis");
        fused.push_back(std::move(node));
        i += 2;
        continue;
      }
      Node node = a;
      node.op = "FusedMaxPoolConcat";
      node.name = a.name.empty() ? "fused_maxpool_concat" : a.name + "_concat";
      node.in = concat.in;
      node.in[0] = a.in[0];
      node.out = concat.out;
      node.attr = a.attr;
      if (concat.attr.contains("axis")) {
        node.attr["axis"] = concat.attr.at("axis");
      }
      fused.push_back(std::move(node));
      i += 2;
      continue;
    }
    // Detector FPN head: Concat(axis=1) of equal HxW maps into Conv 3x3 (+ReLU).
    // The fused kernel keeps each peer in its own allocation and never builds
    // the 64-channel Concat activation.
    if (std::getenv("PPOCR_DISABLE_CONCAT_CONV") == nullptr &&
        a.op == "Concat" && a.out.size() == 1 && a.in.size() >= 2 &&
        consumers[a.out[0]] == 1 && !outputs.contains(a.out[0]) &&
        i + 1 < graph.nodes.size() &&
        (graph.nodes[i + 1].op == "Conv" || graph.nodes[i + 1].op == "FusedConvRelu") &&
        graph.nodes[i + 1].in.size() >= 2 && graph.nodes[i + 1].in[0] == a.out[0] &&
        graph.nodes[i + 1].out.size() == 1) {
      const auto& conv = graph.nodes[i + 1];
      const auto weights = graph.initializers.find(conv.in[1]);
      const bool relu = conv.op == "FusedConvRelu" ||
          (i + 2 < graph.nodes.size() && conv.op == "Conv" &&
           consumers[conv.out[0]] == 1 && !outputs.contains(conv.out[0]) &&
           graph.nodes[i + 2].op == "Relu" && graph.nodes[i + 2].in.size() == 1 &&
           graph.nodes[i + 2].in[0] == conv.out[0]);
      if (weights != graph.initializers.end() && weights->second.shape.size() == 4) {
        Node node = conv;
        node.op = relu ? "FusedConcatConvRelu" : "FusedConcatConv";
        node.name = a.name.empty() ? "fused_concat_conv" : a.name + "_conv";
        node.in = a.in;
        node.in.push_back(conv.in[1]);
        if (conv.in.size() > 2) node.in.push_back(conv.in[2]);
        if (relu && conv.op == "Conv") node.out = graph.nodes[i + 2].out;
        else node.out = conv.out;
        fused.push_back(std::move(node));
        i += (relu && conv.op == "Conv") ? 3 : 2;
        continue;
      }
    }
    if (i + 1 < graph.nodes.size() && a.op == "Conv" && a.out.size() == 1 &&
        consumers[a.out[0]] == 1 && !outputs.contains(a.out[0]) &&
        graph.nodes[i + 1].op == "FusedGelu" && graph.nodes[i + 1].in.size() == 1 &&
        graph.nodes[i + 1].in[0] == a.out[0] &&
        std::getenv("PPOCR_DISABLE_CONV_GELU_FUSION") == nullptr) {
      Node node = a;
      node.op = "FusedConvGelu";
      node.name = a.name.empty() ? "fused_conv_gelu" : a.name + "_gelu";
      node.out = graph.nodes[i + 1].out;
      fused.push_back(std::move(node));
      i += 2;
      continue;
    }
    if (i + 1 < graph.nodes.size() && a.op == "Conv" && a.out.size() == 1 &&
        consumers[a.out[0]] == 1 && !outputs.contains(a.out[0]) &&
        graph.nodes[i+1].op == "Relu" && graph.nodes[i+1].in.size() == 1 &&
        graph.nodes[i+1].in[0] == a.out[0]) {
      Node node = a;
      node.op = "FusedConvRelu";
      node.name = a.name.empty() ? "fused_conv_relu" : a.name + "_relu";
      node.out = graph.nodes[i+1].out;
      fused.push_back(std::move(node));
      i += 2;
      continue;
    }
    // Detector DB head: ConvTranspose2x2 then Relu/Sigmoid. Fold the
    // activation onto the transpose destination so the extra full-map
    // Unary allocation disappears. `PPOCR_DISABLE_CONVTRANSPOSE_ACT_FUSION`
    // restores the two-node walk.
    if (std::getenv("PPOCR_DISABLE_CONVTRANSPOSE_ACT_FUSION") == nullptr &&
        i + 1 < graph.nodes.size() && a.op == "ConvTranspose" && a.out.size() == 1 &&
        consumers[a.out[0]] == 1 && !outputs.contains(a.out[0]) &&
        graph.nodes[i + 1].in.size() == 1 && graph.nodes[i + 1].in[0] == a.out[0] &&
        (graph.nodes[i + 1].op == "Relu" || graph.nodes[i + 1].op == "Sigmoid")) {
      Node node = a;
      node.op = graph.nodes[i + 1].op == "Relu" ? "FusedConvTransposeRelu"
                                               : "FusedConvTransposeSigmoid";
      node.name = a.name.empty() ? "fused_convtranspose_act" : a.name + "_act";
      node.out = graph.nodes[i + 1].out;
      fused.push_back(std::move(node));
      i += 2;
      continue;
    }
    // The squeeze-excitation gates and detector head contain Conv ->
    // Sigmoid/HardSigmoid pairs after affine folding. Produce the convolution
    // result directly into the activation destination and transform it there,
    // avoiding Unary's otherwise unavoidable full tensor allocation/copy.
    if (i + 1 < graph.nodes.size() && a.op == "Conv" && a.out.size() == 1 &&
        consumers[a.out[0]] == 1 && graph.nodes[i + 1].in.size() == 1 &&
        graph.nodes[i + 1].in[0] == a.out[0] &&
        (graph.nodes[i + 1].op == "Sigmoid" || graph.nodes[i + 1].op == "HardSigmoid")) {
      Node node = a;
      node.op = graph.nodes[i + 1].op == "Sigmoid" ? "FusedConvSigmoid"
                                                       : "FusedConvHardSigmoid";
      node.name = a.name.empty() ? "fused_conv_gate" : a.name + "_gate";
      node.out = graph.nodes[i + 1].out;
      if (graph.nodes[i + 1].op == "HardSigmoid") {
        const auto& gate = graph.nodes[i + 1];
        node.attr["__ppocr_fused_alpha"].f = gate.attr.contains("alpha")
            ? gate.attr.at("alpha").f : .2F;
        node.attr["__ppocr_fused_beta"].f = gate.attr.contains("beta")
            ? gate.attr.at("beta").f : .5F;
      }
      fused.push_back(std::move(node));
      i += 2;
      continue;
    }
    // After Conv+BatchNorm folding, MobileNet gates become a Conv fan-out:
    // `Conv(x) -> HardSigmoid(Conv(x)) -> Mul(Conv(x), gate)`.  Folding only
    // the latter two nodes still materializes the gate tensor because the
    // Conv result must remain alive for both Mul operands. Recognize the whole
    // canonical HardSwish sequence and overwrite the Conv destination in
    // place, eliminating that activation-sized temporary. Restrict this to
    // the standard PP-OCR alpha/beta constants; other HardSigmoid variants
    // retain their generic ONNX execution path.
    // Fold the static per-channel Add into Conv's accumulator. This removes
    // the subsequent full NCHW affine traversal. Deployment can disable it
    // for A/B investigation without changing model files.
    if (std::getenv("PPOCR_DISABLE_FOLDED_CHANNEL_BIAS") == nullptr &&
        i + 2 < graph.nodes.size() && a.op == "Conv" && a.out.size() == 1 &&
        consumers[a.out[0]] == 2 && !outputs.contains(a.out[0]) &&
        graph.nodes[i + 1].op == "HardSigmoid" && graph.nodes[i + 1].in.size() == 1 &&
        graph.nodes[i + 1].in[0] == a.out[0] && graph.nodes[i + 1].out.size() == 1 &&
        consumers[graph.nodes[i + 1].out[0]] == 1 &&
        graph.nodes[i + 2].op == "Mul" && graph.nodes[i + 2].in.size() == 2 &&
        (graph.nodes[i + 2].in[0] == a.out[0] || graph.nodes[i + 2].in[1] == a.out[0]) &&
        (graph.nodes[i + 2].in[0] == graph.nodes[i + 1].out[0] ||
         graph.nodes[i + 2].in[1] == graph.nodes[i + 1].out[0])) {
      const auto& gate = graph.nodes[i + 1];
      const float alpha = gate.attr.contains("alpha") ? gate.attr.at("alpha").f : .2F;
      const float beta = gate.attr.contains("beta") ? gate.attr.at("beta").f : .5F;
      if (std::abs(alpha - 1.F / 6.F) < 1e-5F && std::abs(beta - .5F) < 1e-5F) {
        Node node = a;
        node.op = "FusedConvHardSwish";
        node.name = a.name.empty() ? "fused_conv_hardswish" : a.name + "_hardswish";
        node.out = graph.nodes[i + 2].out;
        fused.push_back(std::move(node));
        i += 3;
        continue;
      }
    }
    // Conv->BatchNorm folding exposes recognizer Swish gates as
    // Conv(x) -> Sigmoid(Conv(x)) -> Mul(Conv(x), sigmoid). Fold this exact
    // fan-out/fan-in sequence after the affine fold so Conv's output is
    // overwritten by Swish rather than allocating both branch activations.
    if (i + 2 < graph.nodes.size() && a.op == "Conv" && a.out.size() == 1 &&
        consumers[a.out[0]] == 2 && !outputs.contains(a.out[0]) &&
        graph.nodes[i + 1].op == "Sigmoid" && graph.nodes[i + 1].in.size() == 1 &&
        graph.nodes[i + 1].in[0] == a.out[0] && graph.nodes[i + 1].out.size() == 1 &&
        consumers[graph.nodes[i + 1].out[0]] == 1 &&
        graph.nodes[i + 2].op == "Mul" && graph.nodes[i + 2].in.size() == 2 &&
        (graph.nodes[i + 2].in[0] == a.out[0] || graph.nodes[i + 2].in[1] == a.out[0]) &&
        (graph.nodes[i + 2].in[0] == graph.nodes[i + 1].out[0] ||
         graph.nodes[i + 2].in[1] == graph.nodes[i + 1].out[0])) {
      Node node = a;
      node.op = "FusedConvSwish";
      node.name = a.name.empty() ? "fused_conv_swish" : a.name + "_swish";
      node.out = graph.nodes[i + 2].out;
      fused.push_back(std::move(node));
      i += 3;
      continue;
    }
    // The small/medium recognizer's attention gates export
    // BatchNorm(x) -> Sigmoid(BatchNorm(x)) -> Mul(BatchNorm(x), sigmoid).
    // Their shared BN value has exactly two consumers, both internal to this
    // chain, so emit Swish directly and eliminate two activation tensors.
    if (i + 2 < graph.nodes.size() && a.op == "BatchNormalization" &&
        a.in.size() == 5 && a.out.size() == 1 && consumers[a.out[0]] == 2 &&
        !outputs.contains(a.out[0]) && graph.nodes[i + 1].op == "Sigmoid" &&
        graph.nodes[i + 1].in.size() == 1 && graph.nodes[i + 1].in[0] == a.out[0] &&
        graph.nodes[i + 1].out.size() == 1 && consumers[graph.nodes[i + 1].out[0]] == 1 &&
        graph.nodes[i + 2].op == "Mul" && graph.nodes[i + 2].in.size() == 2 &&
        (graph.nodes[i + 2].in[0] == a.out[0] || graph.nodes[i + 2].in[1] == a.out[0]) &&
        (graph.nodes[i + 2].in[0] == graph.nodes[i + 1].out[0] ||
         graph.nodes[i + 2].in[1] == graph.nodes[i + 1].out[0])) {
      const auto scale = graph.initializers.find(a.in[1]);
      const auto bias = graph.initializers.find(a.in[2]);
      const auto mean = graph.initializers.find(a.in[3]);
      const auto variance = graph.initializers.find(a.in[4]);
      const auto channels = scale == graph.initializers.end() ? 0 : scale->second.data.size();
      if (channels && bias != graph.initializers.end() && mean != graph.initializers.end() &&
          variance != graph.initializers.end() && bias->second.data.size() == channels &&
          mean->second.data.size() == channels && variance->second.data.size() == channels) {
        const float epsilon = a.attr.contains("epsilon") ? a.attr.at("epsilon").f : 1e-5F;
        Tensor factor{{static_cast<std::int64_t>(channels)}, std::vector<float>(channels)};
        Tensor offset{{static_cast<std::int64_t>(channels)}, std::vector<float>(channels)};
        for (std::size_t channel = 0; channel < channels; ++channel) {
          factor.data[channel] = scale->second.data[channel] /
              std::sqrt(variance->second.data[channel] + epsilon);
          offset.data[channel] = bias->second.data[channel] - mean->second.data[channel] * factor.data[channel];
        }
        const std::string factor_name = "__ppocr_bn_factor_swish_" + std::to_string(i);
        const std::string offset_name = "__ppocr_bn_offset_swish_" + std::to_string(i);
        graph.initializers.emplace(factor_name, std::move(factor));
        graph.initializers.emplace(offset_name, std::move(offset));
        Node node = a;
        node.op = "FusedBatchNormSwish";
        node.name = a.name.empty() ? "fused_batchnorm_swish" : a.name + "_swish";
        node.in = {a.in[0], factor_name, offset_name};
        node.out = graph.nodes[i + 2].out;
        fused.push_back(std::move(node));
        i += 3;
        continue;
      }
    }
    // Tiny rec's SVTR conv-BN-hswish tail: BatchNorm(x) -> HardSigmoid(BN)
    // -> Mul(BN, hardsigmoid). Same two-consumer BN as the Swish fold.
    if (i + 2 < graph.nodes.size() && a.op == "BatchNormalization" &&
        a.in.size() == 5 && a.out.size() == 1 && consumers[a.out[0]] == 2 &&
        !outputs.contains(a.out[0]) && graph.nodes[i + 1].op == "HardSigmoid" &&
        graph.nodes[i + 1].in.size() == 1 && graph.nodes[i + 1].in[0] == a.out[0] &&
        graph.nodes[i + 1].out.size() == 1 && consumers[graph.nodes[i + 1].out[0]] == 1 &&
        graph.nodes[i + 2].op == "Mul" && graph.nodes[i + 2].in.size() == 2 &&
        (graph.nodes[i + 2].in[0] == a.out[0] || graph.nodes[i + 2].in[1] == a.out[0]) &&
        (graph.nodes[i + 2].in[0] == graph.nodes[i + 1].out[0] ||
         graph.nodes[i + 2].in[1] == graph.nodes[i + 1].out[0])) {
      const float alpha = graph.nodes[i + 1].attr.contains("alpha")
          ? graph.nodes[i + 1].attr.at("alpha").f : .2F;
      const float beta = graph.nodes[i + 1].attr.contains("beta")
          ? graph.nodes[i + 1].attr.at("beta").f : .5F;
      const auto scale = graph.initializers.find(a.in[1]);
      const auto bias = graph.initializers.find(a.in[2]);
      const auto mean = graph.initializers.find(a.in[3]);
      const auto variance = graph.initializers.find(a.in[4]);
      const auto channels = scale == graph.initializers.end() ? 0 : scale->second.data.size();
      if (std::abs(alpha - 1.F/6.F) < 1e-5F && std::abs(beta - .5F) < 1e-5F &&
          channels && bias != graph.initializers.end() && mean != graph.initializers.end() &&
          variance != graph.initializers.end() && bias->second.data.size() == channels &&
          mean->second.data.size() == channels && variance->second.data.size() == channels) {
        const float epsilon = a.attr.contains("epsilon") ? a.attr.at("epsilon").f : 1e-5F;
        Tensor factor{{static_cast<std::int64_t>(channels)}, std::vector<float>(channels)};
        Tensor offset{{static_cast<std::int64_t>(channels)}, std::vector<float>(channels)};
        for (std::size_t channel = 0; channel < channels; ++channel) {
          factor.data[channel] = scale->second.data[channel] /
              std::sqrt(variance->second.data[channel] + epsilon);
          offset.data[channel] = bias->second.data[channel] - mean->second.data[channel] * factor.data[channel];
        }
        const std::string factor_name = "__ppocr_bn_factor_hswish_" + std::to_string(i);
        const std::string offset_name = "__ppocr_bn_offset_hswish_" + std::to_string(i);
        graph.initializers.emplace(factor_name, std::move(factor));
        graph.initializers.emplace(offset_name, std::move(offset));
        Node node = a;
        node.op = "FusedBatchNormHardSwish";
        node.name = a.name.empty() ? "fused_batchnorm_hardswish" : a.name + "_hardswish";
        node.in = {a.in[0], factor_name, offset_name};
        node.out = graph.nodes[i + 2].out;
        fused.push_back(std::move(node));
        i += 3;
        continue;
      }
    }
    // The detector reconstruction head ends its stride-2 2x2 transposed
    // convolutions with Add([1,C,1,1]). Fold the channel bias into the
    // transposed-convolution bias at model-load time, eliminating an entire
    // output-map traversal before the final DB activation.
    if (i + 1 < graph.nodes.size() && a.op == "ConvTranspose" && a.in.size() >= 2 &&
        a.out.size() == 1 && consumers[a.out[0]] == 1 &&
        graph.nodes[i + 1].op == "Add" && graph.nodes[i + 1].in.size() == 2 &&
        (graph.nodes[i + 1].in[0] == a.out[0] || graph.nodes[i + 1].in[1] == a.out[0])) {
      const auto& add = graph.nodes[i + 1];
      const auto bias_name = add.in[0] == a.out[0] ? add.in[1] : add.in[0];
      const auto channel_bias = graph.initializers.find(bias_name);
      const auto weights = graph.initializers.find(a.in[1]);
      const auto group_it = a.attr.find("group");
      const auto group = group_it == a.attr.end() ? std::int64_t{1} : group_it->second.i;
      if (channel_bias != graph.initializers.end() && weights != graph.initializers.end() &&
          weights->second.shape.size() == 4 && group > 0 &&
          channel_bias->second.shape.size() == 4 && channel_bias->second.shape[0] == 1 &&
          channel_bias->second.shape[2] == 1 && channel_bias->second.shape[3] == 1) {
        const auto channels = std::size_t(weights->second.shape[1] * group);
        if (channel_bias->second.data.size() == channels) {
          Tensor folded_bias{{static_cast<std::int64_t>(channels)}, std::vector<float>(channels)};
          if (a.in.size() > 2) {
            const auto existing_bias = graph.initializers.find(a.in[2]);
            if (existing_bias == graph.initializers.end() || existing_bias->second.data.size() != channels) {
              fused.push_back(a);
              ++i;
              continue;
            }
            for (std::size_t channel = 0; channel < channels; ++channel)
              folded_bias.data[channel] = existing_bias->second.data[channel] + channel_bias->second.data[channel];
          } else {
            folded_bias.data = channel_bias->second.data;
          }
          const std::string folded_name = "__ppocr_folded_transpose_bias_" + std::to_string(i);
          graph.initializers.emplace(folded_name, std::move(folded_bias));
          Node node = a;
          node.name = a.name.empty() ? "fused_convtranspose_channel_bias" : a.name + "_bias";
          node.in.resize(2);
          node.in.push_back(folded_name);
          node.out = add.out;
          fused.push_back(std::move(node));
          i += 2;
          continue;
        }
      }
    }
    // The detector FPN repeatedly performs an exact nearest-neighbour resize
    // followed by an equal-shaped residual Add.  The resized tensor has one
    // consumer and is otherwise only a bandwidth-heavy temporary. Preserve
    // the Resize attributes and replace the pair with one output write:
    // `out = nearest(source) + residual`. Runtime validation retains the
    // generic executor for any unexpected dynamic shape or resize mode.
    if (i + 1 < graph.nodes.size() && a.op == "Resize" && a.out.size() == 1 &&
        consumers[a.out[0]] == 1 && !outputs.contains(a.out[0]) &&
        graph.nodes[i + 1].op == "Add" && graph.nodes[i + 1].in.size() == 2 &&
        graph.nodes[i + 1].out.size() == 1 && !outputs.contains(graph.nodes[i + 1].out[0]) &&
        (graph.nodes[i + 1].in[0] == a.out[0] || graph.nodes[i + 1].in[1] == a.out[0])) {
      const auto& add = graph.nodes[i + 1];
      const auto residual = add.in[0] == a.out[0] ? add.in[1] : add.in[0];
      if (!residual.empty() && !graph.initializers.contains(residual)) {
        Node node = a;
        node.op = "FusedNearestResizeAdd";
        node.name = a.name.empty() ? "fused_nearest_resize_add" : a.name + "_add";
        node.in.push_back(residual);
        node.out = add.out;
        fused.push_back(std::move(node));
        i += 2;
        continue;
      }
    }
    // The recognizer repeatedly emits BatchNorm -> GELU.  Both operations are
    // pointwise and the BatchNorm result has one consumer, so producing its
    // values directly into the GELU destination removes a full activation
    // allocation/copy without changing the graph's numerical operation order.
    if (i + 1 < graph.nodes.size() && a.op == "BatchNormalization" &&
        a.out.size() == 1 && consumers[a.out[0]] == 1 &&
        !outputs.contains(a.out[0]) && graph.nodes[i + 1].op == "FusedGelu" &&
        graph.nodes[i + 1].in.size() == 1 && graph.nodes[i + 1].in[0] == a.out[0]) {
      const auto scale = a.in.size() == 5 ? graph.initializers.find(a.in[1]) : graph.initializers.end();
      const auto bias = a.in.size() == 5 ? graph.initializers.find(a.in[2]) : graph.initializers.end();
      const auto mean = a.in.size() == 5 ? graph.initializers.find(a.in[3]) : graph.initializers.end();
      const auto variance = a.in.size() == 5 ? graph.initializers.find(a.in[4]) : graph.initializers.end();
      const auto channels = scale == graph.initializers.end() ? 0 : scale->second.data.size();
      if (channels && bias != graph.initializers.end() && mean != graph.initializers.end() &&
          variance != graph.initializers.end() && bias->second.data.size() == channels &&
          mean->second.data.size() == channels && variance->second.data.size() == channels) {
        const float epsilon = a.attr.contains("epsilon") ? a.attr.at("epsilon").f : 1e-5F;
        Tensor factor{{static_cast<std::int64_t>(channels)}, std::vector<float>(channels)};
        Tensor offset{{static_cast<std::int64_t>(channels)}, std::vector<float>(channels)};
        for (std::size_t channel = 0; channel < channels; ++channel) {
          factor.data[channel] = scale->second.data[channel] /
              std::sqrt(variance->second.data[channel] + epsilon);
          offset.data[channel] = bias->second.data[channel] - mean->second.data[channel] * factor.data[channel];
        }
        const std::string factor_name = "__ppocr_bn_factor_gelu_" + std::to_string(i);
        const std::string offset_name = "__ppocr_bn_offset_gelu_" + std::to_string(i);
        graph.initializers.emplace(factor_name, std::move(factor));
        graph.initializers.emplace(offset_name, std::move(offset));
        Node node = a;
        node.op = "FusedBatchNormGelu";
        node.name = "fused_batchnorm_gelu";
        node.in = {a.in[0], factor_name, offset_name};
        node.out = graph.nodes[i + 1].out;
        fused.push_back(std::move(node));
        i += 2;
        continue;
      }
    }
    // Small/medium recognizers end their inverted residual blocks with
    // `BatchNorm -> Swish -> Add(shortcut) -> Identity`. The Swish is already
    // folded above, so recognize its immediate equal-shape dynamic Add here.
    // This removes a second full feature-map traversal and lets Vulkan keep
    // affine, Swish and residual addition inside one batch dispatch when its
    // measured complete round trip wins.
    if (std::getenv("PPOCR_DISABLE_BN_SWISH_ADD_FUSION") == nullptr &&
        i + 1 < graph.nodes.size() && a.op == "FusedBatchNormSwish" &&
        a.in.size() == 3 && a.out.size() == 1 && consumers[a.out[0]] == 1 &&
        !outputs.contains(a.out[0]) && graph.nodes[i + 1].op == "Add" &&
        graph.nodes[i + 1].in.size() == 2 && graph.nodes[i + 1].out.size() == 1 &&
        !outputs.contains(graph.nodes[i + 1].out[0]) &&
        (graph.nodes[i + 1].in[0] == a.out[0] || graph.nodes[i + 1].in[1] == a.out[0])) {
      const auto& add = graph.nodes[i + 1];
      const auto residual = add.in[0] == a.out[0] ? add.in[1] : add.in[0];
      if (!residual.empty() && !graph.initializers.contains(residual)) {
        Node node = a;
        node.op = "FusedBatchNormSwishAdd";
        node.name = a.name.empty() ? "fused_batchnorm_swish_add" : a.name + "_add";
        node.in.push_back(residual);
        node.out = graph.nodes[i + 1].out;
        fused.push_back(std::move(node));
        i += 2;
        continue;
      }
    }
    // Recognition exports several `Conv -> Add(channel_bias) -> Identity`
    // sequences. Fold the broadcast channel bias into the convolution bias
    // during model loading. This removes both the activation traversal and
    // the generic broadcast allocation. Starting every output accumulator
    // from the folded bias is mathematically the same inference affine term;
    // it also lets subsequent Conv+activation fusions apply normally.
    if (i + 2 < graph.nodes.size() && a.op == "Conv" && a.out.size() == 1 &&
        consumers[a.out[0]] == 1 && graph.nodes[i+1].op == "Add" &&
        graph.nodes[i+2].op == "Identity" && graph.nodes[i+1].out.size() == 1 &&
        graph.nodes[i+2].in.size() == 1 && graph.nodes[i+2].in[0] == graph.nodes[i+1].out[0] &&
        (graph.nodes[i+1].in[0] == a.out[0] || graph.nodes[i+1].in[1] == a.out[0])) {
      const auto& add = graph.nodes[i+1];
      const auto bias_name = add.in[0] == a.out[0] ? add.in[1] : add.in[0];
      const auto bias = graph.initializers.find(bias_name);
      const auto weights = graph.initializers.find(a.in[1]);
      if (bias != graph.initializers.end() && weights != graph.initializers.end() &&
          weights->second.shape.size() == 4 && bias->second.shape.size() == 4 &&
          bias->second.shape[0] == 1 && bias->second.shape[1] == weights->second.shape[0] &&
          bias->second.shape[2] == 1 && bias->second.shape[3] == 1) {
          const auto channels = std::size_t(weights->second.shape[0]);
          Tensor folded_bias{{static_cast<std::int64_t>(channels)},
                             std::vector<float>(channels)};
          if (a.in.size() > 2) {
            const auto existing_bias = graph.initializers.find(a.in[2]);
            if (existing_bias == graph.initializers.end() ||
                existing_bias->second.data.size() != channels) {
              fused.push_back(a);
              ++i;
              continue;
            }
            for (std::size_t channel = 0; channel < channels; ++channel) {
              folded_bias.data[channel] = existing_bias->second.data[channel] +
                                           bias->second.data[channel];
            }
          } else {
            for (std::size_t channel = 0; channel < channels; ++channel) {
              folded_bias.data[channel] = bias->second.data[channel];
            }
          }
          const std::string folded_name = "__ppocr_folded_channel_bias_" +
                                          std::to_string(i);
          graph.initializers.emplace(folded_name, std::move(folded_bias));
          Node node = a;
          node.name = "fused_conv_channel_bias";
          node.in.resize(2);
          node.in.push_back(folded_name);
          node.out = graph.nodes[i+2].out;
          fused.push_back(std::move(node));
          i += 3;
          continue;
      }
    }
    // A Transpose immediately followed by its exact inverse is a no-op. The
    // recognizer's final attention head has this exported pair; retaining it
    // needlessly copies a large [N,C,T] activation twice.
    if (i + 1 < graph.nodes.size() && a.op == "Transpose" && a.out.size() == 1 &&
        consumers[a.out[0]] == 1 && graph.nodes[i+1].op == "Transpose" &&
        graph.nodes[i+1].in.size() == 1 && graph.nodes[i+1].in[0] == a.out[0]) {
      const auto first_it = a.attr.find("perm");
      const auto second_it = graph.nodes[i+1].attr.find("perm");
      const auto first = first_it == a.attr.end() ? std::vector<std::int64_t>{} : first_it->second.ints;
      const auto second = second_it == graph.nodes[i+1].attr.end() ? std::vector<std::int64_t>{} : second_it->second.ints;
      if (!first.empty() && first.size() == second.size()) {
        bool inverse = true;
        for (std::size_t axis = 0; axis < first.size(); ++axis) {
          if (first[std::size_t(second[axis])] != std::int64_t(axis)) { inverse = false; break; }
        }
        if (inverse) {
          Node node;
          node.op = "Identity";
          node.name = "fused_transpose_inverse";
          node.in = a.in;
          node.out = graph.nodes[i+1].out;
          fused.push_back(std::move(node));
          i += 2;
          continue;
        }
      }
    }
    // Exported GELU: Div(x, sqrt(2)) -> Erf -> Add(1) -> Mul -> Mul(0.5).
    // This recognizer pattern costs four full activation passes otherwise.
    if (i + 4 < graph.nodes.size() && a.op == "Div" && a.out.size() == 1 &&
        consumers[a.out[0]] == 1 && graph.nodes[i+1].op == "Erf" &&
        graph.nodes[i+1].in.size() == 1 && graph.nodes[i+1].in[0] == a.out[0] &&
        graph.nodes[i+2].op == "Add" && graph.nodes[i+2].in.size() == 2 &&
        graph.nodes[i+3].op == "Mul" && graph.nodes[i+3].in.size() == 2 &&
        graph.nodes[i+4].op == "Mul" && graph.nodes[i+4].in.size() == 2 &&
        consumers[graph.nodes[i+1].out[0]] == 1 && consumers[graph.nodes[i+2].out[0]] == 1 &&
        consumers[graph.nodes[i+3].out[0]] == 1 && !outputs.contains(graph.nodes[i+4].out[0])) {
      const auto& erf = graph.nodes[i+1];
      const auto& add = graph.nodes[i+2];
      const auto& mul1 = graph.nodes[i+3];
      const auto& mul2 = graph.nodes[i+4];
      const auto is_init = [&](const std::string& name, float expected) {
        const auto it = graph.initializers.find(name);
        return it != graph.initializers.end() && it->second.data.size() == 1 &&
               std::abs(it->second.data[0] - expected) < 1e-5F;
      };
      const auto other = [](const Node& n, const std::string& value) -> std::string {
        return n.in[0] == value ? n.in[1] : n.in[0];
      };
      const auto x = a.in[0];
      const auto div_scale = a.in[1];
      if (is_init(div_scale, std::sqrt(2.F)) && is_init(other(add, erf.out[0]), 1.F) &&
          ((mul1.in[0] == x && mul1.in[1] == add.out[0]) || (mul1.in[1] == x && mul1.in[0] == add.out[0])) &&
          is_init(other(mul2, mul1.out[0]), .5F)) {
        Node node;
        node.op = "FusedGelu";
        node.name = "fused_gelu";
        node.in = {x};
        node.out = mul2.out;
        fused.push_back(std::move(node));
        i += 5;
        continue;
      }
    }
    // MobileNet h-swish: HardSigmoid(x) followed by Mul(x, result).
    if (i + 1 < graph.nodes.size() && a.op == "HardSigmoid" && a.out.size() == 1 &&
        consumers[a.out[0]] == 1 && graph.nodes[i+1].op == "Mul" &&
        graph.nodes[i+1].in.size() == 2 &&
        (graph.nodes[i+1].in[0] == a.out[0] || graph.nodes[i+1].in[1] == a.out[0])) {
      const auto& mul = graph.nodes[i+1];
      const auto other = mul.in[0] == a.out[0] ? mul.in[1] : mul.in[0];
      const float alpha = a.attr.contains("alpha") ? a.attr.at("alpha").f : .2F;
      const float beta = a.attr.contains("beta") ? a.attr.at("beta").f : .5F;
      if (other == a.in[0] && std::abs(alpha - 1.F/6.F) < 1e-5F && std::abs(beta - .5F) < 1e-5F) {
        Node node;
        node.op = "FusedHardSwish";
        node.name = "fused_hardswish";
        node.in = {other};
        node.out = mul.out;
        fused.push_back(std::move(node));
        i += 2;
        continue;
      }
    }
    // Constant affine pairs are emitted repeatedly in PP-OCR blocks. Fold
    // Mul(x, scale) + Add(..., shift) into one traversal.
    if (i + 1 < graph.nodes.size() && a.op == "Mul" && a.out.size() == 1 &&
        consumers[a.out[0]] == 1 && graph.nodes[i+1].op == "Add" &&
        graph.nodes[i+1].in.size() == 2 &&
        (graph.nodes[i+1].in[0] == a.out[0] || graph.nodes[i+1].in[1] == a.out[0])) {
      const auto scalar = [&](const std::string& name, float& value) {
        const auto it = graph.initializers.find(name);
        if (it == graph.initializers.end() || it->second.data.size() != 1) return false;
        value = it->second.data[0]; return true;
      };
      const auto shift_name = graph.nodes[i+1].in[0] == a.out[0] ? graph.nodes[i+1].in[1] : graph.nodes[i+1].in[0];
      float scale{}, shift{};
      std::string x;
      if (scalar(a.in[0], scale)) x = a.in[1];
      else if (scalar(a.in[1], scale)) x = a.in[0];
      if (!x.empty() && scalar(shift_name, shift)) {
        Node node;
        node.op = "FusedScaleShift";
        node.name = "fused_scale_shift";
        node.in = {x};
        node.out = graph.nodes[i+1].out;
        node.attr["scale"].f = scale;
        node.attr["shift"].f = shift;
        fused.push_back(std::move(node));
        i += 2;
        continue;
      }
    }
    fused.push_back(a);
    ++i;
  }
  graph.nodes = std::move(fused);
}

// ONNX exporters commonly retain Identity nodes as naming boundaries.  They
// have no runtime meaning, but executing them used to copy their complete
// activation buffer (including the recognizer's residual outputs).  Rewrite
// later inputs to the original value while loading the model instead.  Keep an
// Identity that is itself a public graph output so OnnxLite::outputs() remains
// byte-for-byte compatible with the model's declared output names.
// Graph fusions can replace Conv/BatchNorm weights and bias tensors with
// folded constants. Keeping the now-unreachable source weights doubles the
// resident model footprint (especially for the small and medium models) even
// though the executor never reads them. After every rewrite pass, retain only
// initializers still named by a graph input, a graph output, or a node input.
// This is load-time-only bookkeeping: it cannot affect graph scheduling or
// floating-point execution order.
void PruneUnusedInitializers(GraphData& graph) {
  std::unordered_set<std::string> live;
  live.reserve(graph.initializers.size());
  for (const auto& name : graph.inputs) live.insert(name);
  for (const auto& name : graph.outputs) live.insert(name);
  for (const auto& node : graph.nodes) {
    for (const auto& name : node.in) {
      if (!name.empty()) live.insert(name);
    }
  }
  for (auto it = graph.initializers.begin(); it != graph.initializers.end();) {
    if (!live.contains(it->first)) it = graph.initializers.erase(it);
    else ++it;
  }
}

void EliminateInternalIdentity(GraphData& graph) {
  const std::unordered_set<std::string> public_outputs(
      graph.outputs.begin(), graph.outputs.end());
  std::unordered_map<std::string, std::string> aliases;
  aliases.reserve(graph.nodes.size());
  const auto resolve = [&aliases](std::string name) {
    std::vector<std::string> path;
    for (;;) {
      const auto it = aliases.find(name);
      if (it == aliases.end()) break;
      path.push_back(name);
      name = it->second;
    }
    for (const auto& intermediate : path) aliases[intermediate] = name;
    return name;
  };

  std::vector<Node> compact;
  compact.reserve(graph.nodes.size());
  for (auto node : graph.nodes) {
    for (auto& input : node.in) {
      if (!input.empty()) input = resolve(std::move(input));
    }
    if (node.op == "Identity" && node.in.size() == 1 && node.out.size() == 1 &&
        !node.in[0].empty() && !public_outputs.contains(node.out[0])) {
      aliases[node.out[0]] = node.in[0];
      continue;
    }
    compact.push_back(std::move(node));
  }
  graph.nodes = std::move(compact);
}

// A transpose that is immediately undone is an allocation-sized no-op.  Run
// this after Identity elimination because ONNX exporters often put naming
// Identities between the pair.  The recognizer has a [N,T,C] -> [N,C,T] ->
// [N,T,C] pair before its 1-D head; removing it avoids two batch-dependent
// copies on every recognition request.
void EliminateAdjacentInverseTranspose(GraphData& graph) {
  std::unordered_map<std::string, std::size_t> consumers;
  std::unordered_set<std::string> outputs(graph.outputs.begin(), graph.outputs.end());
  for (const auto& node : graph.nodes) for (const auto& input : node.in) {
    if (!input.empty()) ++consumers[input];
  }
  std::vector<Node> compact;
  compact.reserve(graph.nodes.size());
  for (std::size_t i = 0; i < graph.nodes.size();) {
    const auto& first = graph.nodes[i];
    if (i + 1 < graph.nodes.size() && first.op == "Transpose" &&
        first.in.size() == 1 && first.out.size() == 1 &&
        consumers[first.out[0]] == 1 && !outputs.contains(first.out[0])) {
      const auto& second = graph.nodes[i + 1];
      const auto first_it = first.attr.find("perm");
      const auto second_it = second.attr.find("perm");
      if (second.op == "Transpose" && second.in.size() == 1 &&
          second.out.size() == 1 && second.in[0] == first.out[0] &&
          first_it != first.attr.end() && second_it != second.attr.end() &&
          first_it->second.ints.size() == second_it->second.ints.size()) {
        const auto& forward = first_it->second.ints;
        const auto& reverse = second_it->second.ints;
        bool inverse = true;
        for (std::size_t axis = 0; axis < forward.size(); ++axis) {
          if (reverse[axis] < 0 || std::size_t(reverse[axis]) >= forward.size() ||
              forward[std::size_t(reverse[axis])] != std::int64_t(axis)) {
            inverse = false;
            break;
          }
        }
        if (inverse) {
          Node identity;
          identity.op = "Identity";
          identity.name = "fused_transpose_inverse";
          identity.in = first.in;
          identity.out = second.out;
          compact.push_back(std::move(identity));
          i += 2;
          continue;
        }
      }
    }
    compact.push_back(first);
    ++i;
  }
  graph.nodes = std::move(compact);
}

// The transformer recognizers export their QKV projection as [N,T,3,H,D],
// transpose the complete tensor to [3,N,H,T,D], then immediately split the
// leading three entries with three Slice+Squeeze branches.  Keeping the full
// transposed tensor alive only to copy each third out again costs one complete
// activation allocation/write per attention block.  Redirect each proved
// three-way Slice to the pre-transpose layout instead.  The executor's
// FusedQkvSlice copies just one [N,H,T,D] branch into its final layout; the
// existing following Squeeze remains a metadata-only storage transfer.
//
// This is deliberately restricted to the exact PP-OCRv6 QKV layout and
// three unit axis-0 slices. Any dynamic/general ONNX Slice or additional
// transpose consumer keeps the ordinary graph untouched.
void FuseQkvTransposeSlices(GraphData& graph) {
  // The direct split is the default after parity validation.  Keep a narrow
  // deployment A/B switch for an unfamiliar recognizer export.
  if (std::getenv("PPOCR_DISABLE_QKV_SLICE_FUSION") != nullptr) return;
  std::unordered_map<std::string, std::vector<std::size_t>> consumers;
  consumers.reserve(graph.nodes.size());
  for (std::size_t index = 0; index < graph.nodes.size(); ++index) {
    for (const auto& input : graph.nodes[index].in) {
      if (!input.empty()) consumers[input].push_back(index);
    }
  }
  std::unordered_set<std::string> outputs(graph.outputs.begin(), graph.outputs.end());
  std::unordered_set<std::size_t> discard;
  struct Replacement { std::string source; std::vector<std::string> outputs; };
  std::unordered_map<std::size_t, Replacement> qkv_projection;
  for (std::size_t index = 0; index < graph.nodes.size(); ++index) {
    const auto& transpose = graph.nodes[index];
    const auto perm = transpose.attr.contains("perm") ? transpose.attr.at("perm").ints :
                                                        std::vector<std::int64_t>{};
    if (transpose.op != "Transpose" || transpose.in.size() != 1 ||
        transpose.out.size() != 1 || outputs.contains(transpose.out[0]) ||
        consumers[transpose.in[0]].size() != 1 ||
        perm != std::vector<std::int64_t>{2, 0, 3, 1, 4}) {
      continue;
    }
    const auto found = consumers.find(transpose.out[0]);
    if (found == consumers.end() || found->second.size() != 3) continue;
    std::array<int, 3> seen{-1, -1, -1};
    bool valid = true;
    for (const auto slice_index : found->second) {
      const auto& slice = graph.nodes[slice_index];
      if (slice.op != "Slice" || slice.in.size() < 3 || slice.out.size() != 1 ||
          slice.in[0] != transpose.out[0] || outputs.contains(slice.out[0])) {
        valid = false;
        break;
      }
      const auto starts = graph.initializers.find(slice.in[1]);
      const auto ends = graph.initializers.find(slice.in[2]);
      const auto axes = slice.in.size() > 3 && !slice.in[3].empty()
          ? graph.initializers.find(slice.in[3]) : graph.initializers.end();
      const auto steps = slice.in.size() > 4 && !slice.in[4].empty()
          ? graph.initializers.find(slice.in[4]) : graph.initializers.end();
      if (starts == graph.initializers.end() || ends == graph.initializers.end() ||
          starts->second.data.size() != 1 || ends->second.data.size() != 1 ||
          (axes != graph.initializers.end() && axes->second.data.size() != 1) ||
          (steps != graph.initializers.end() && steps->second.data.size() != 1)) {
        valid = false;
        break;
      }
      const int axis = axes == graph.initializers.end() ? 0 : int(axes->second.data[0]);
      const int step = steps == graph.initializers.end() ? 1 : int(steps->second.data[0]);
      const int begin = int(starts->second.data[0]);
      const int end = int(ends->second.data[0]);
      if (axis != 0 || step != 1 || begin < 0 || begin >= 3 || end != begin + 1 ||
          seen[begin] != -1 || slice_index + 1 >= graph.nodes.size()) {
        valid = false;
        break;
      }
      const auto& squeeze = graph.nodes[slice_index + 1];
      const auto squeeze_axes = squeeze.attr.contains("axes") ? squeeze.attr.at("axes").ints :
                                                                 std::vector<std::int64_t>{};
      if (squeeze.op != "Squeeze" || squeeze.in.size() != 1 || squeeze.out.size() != 1 ||
          squeeze.in[0] != slice.out[0] || squeeze_axes != std::vector<std::int64_t>{0}) {
        valid = false;
        break;
      }
      seen[begin] = int(slice_index);
    }
    if (!valid || seen[0] < 0 || seen[1] < 0 || seen[2] < 0) continue;
    const auto split_index = std::size_t(*std::min_element(seen.begin(), seen.end()));
    discard.insert(index);
    std::vector<std::string> branch_outputs(3);
    for (int projection = 0; projection < 3; ++projection) {
      const auto& squeeze = graph.nodes[std::size_t(seen[projection]) + 1];
      branch_outputs[std::size_t(projection)] = squeeze.out[0];
      if (std::size_t(seen[projection]) != split_index) discard.insert(std::size_t(seen[projection]));
      discard.insert(std::size_t(seen[projection]) + 1);
    }
    qkv_projection.emplace(split_index, Replacement{transpose.in[0], std::move(branch_outputs)});
  }
  if (discard.empty()) return;
  std::vector<Node> compact;
  compact.reserve(graph.nodes.size() - discard.size());
  for (std::size_t index = 0; index < graph.nodes.size(); ++index) {
    if (discard.contains(index)) continue;
    Node node = graph.nodes[index];
    const auto replacement = qkv_projection.find(index);
    if (replacement != qkv_projection.end()) {
      node.op = "FusedQkvSplit";
      node.name = node.name.empty() ? "fused_qkv_split" : node.name + "_qkv_split";
      node.in[0] = replacement->second.source;
      node.in.resize(1);
      node.out = replacement->second.outputs;
    }
    compact.push_back(std::move(node));
  }
  graph.nodes = std::move(compact);
}

// The detector MobileNet backbone commonly follows an already-fused
// depthwise閳姫ointwise pair with the exact ONNX GELU sequence.  Once both
// predecessor fusions have run, collapse this final single-use activation as
// well.  The output buffer is unchanged; this only removes a graph dispatch
// and makes the liveness boundary explicit for CPU and hybrid execution.
void FuseDepthwisePointwiseGelu(GraphData& graph) {
  const bool disable_gelu =
      std::getenv("PPOCR_DISABLE_DEPTHWISE_POINTWISE_GELU_FUSION") != nullptr;
  const bool disable_act =
      std::getenv("PPOCR_DISABLE_DEPTHWISE_POINTWISE_ACT_FUSION") != nullptr;
  if (disable_gelu && disable_act) return;
  std::unordered_map<std::string, std::size_t> consumers;
  std::unordered_set<std::string> outputs(graph.outputs.begin(), graph.outputs.end());
  for (const auto& node : graph.nodes) for (const auto& input : node.in) {
    if (!input.empty()) ++consumers[input];
  }
  std::vector<Node> compact;
  compact.reserve(graph.nodes.size());
  for (std::size_t i = 0; i < graph.nodes.size();) {
    const auto& pair = graph.nodes[i];
    if (i + 1 < graph.nodes.size() && pair.op == "FusedDepthwisePointwiseConv" &&
        pair.out.size() == 1 && consumers[pair.out[0]] == 1 &&
        !outputs.contains(pair.out[0])) {
      const auto& next = graph.nodes[i + 1];
      const bool single_use = next.in.size() == 1 && next.out.size() == 1 &&
          next.in[0] == pair.out[0] && !outputs.contains(next.out[0]);
      // Fold the common MobileNet suffix into the already-fused pair so the
      // pointwise store is the final activation write. HardSwish/ReLU are
      // fused at every channel width; GELU remains available for the same
      // shapes after its predecessor fusion has run.
      if (single_use && !disable_act && next.op == "FusedHardSwish") {
        Node node = pair;
        node.op = "FusedDepthwisePointwiseConvHardSwish";
        node.name = pair.name + "_hardswish";
        node.out = next.out;
        compact.push_back(std::move(node));
        i += 2;
        continue;
      }
      if (single_use && !disable_act && next.op == "Relu") {
        Node node = pair;
        node.op = "FusedDepthwisePointwiseConvRelu";
        node.name = pair.name + "_relu";
        node.out = next.out;
        compact.push_back(std::move(node));
        i += 2;
        continue;
      }
      if (single_use && !disable_gelu && next.op == "FusedGelu") {
        Node node = pair;
        node.op = "FusedDepthwisePointwiseConvGelu";
        node.name = pair.name + "_gelu";
        node.out = next.out;
        compact.push_back(std::move(node));
        i += 2;
        continue;
      }
    }
    compact.push_back(pair);
    ++i;
  }
  graph.nodes = std::move(compact);
}

// Detector FPN head: ConcatConvRelu (64→16 3x3) then 2x2 stride-2
// ConvTranspose Relu then Sigmoid. Same-host 8-run missed versus the
// three-node walk, so this stays ENABLE-only
// (`PPOCR_ENABLE_CONCAT_DUAL_TRANSPOSE`).
void FuseConcatConvDualTranspose(GraphData& graph) {
  if (std::getenv("PPOCR_ENABLE_CONCAT_DUAL_TRANSPOSE") == nullptr ||
      std::getenv("PPOCR_DISABLE_CONCAT_DUAL_TRANSPOSE") != nullptr) return;
  std::unordered_map<std::string, std::size_t> consumers;
  std::unordered_set<std::string> outputs(graph.outputs.begin(), graph.outputs.end());
  for (const auto& node : graph.nodes) for (const auto& input : node.in) {
    if (!input.empty()) ++consumers[input];
  }
  std::vector<Node> compact;
  compact.reserve(graph.nodes.size());
  for (std::size_t i = 0; i < graph.nodes.size();) {
    const auto& concat = graph.nodes[i];
    if (i + 2 < graph.nodes.size() && concat.op == "FusedConcatConvRelu" &&
        concat.out.size() == 1 && consumers[concat.out[0]] == 1 &&
        !outputs.contains(concat.out[0])) {
      const auto& first = graph.nodes[i + 1];
      const auto& second = graph.nodes[i + 2];
      const bool chain = first.op == "FusedConvTransposeRelu" && first.in.size() >= 3 &&
          first.in[0] == concat.out[0] && first.out.size() == 1 &&
          consumers[first.out[0]] == 1 && !outputs.contains(first.out[0]) &&
          second.op == "FusedConvTransposeSigmoid" && second.in.size() >= 3 &&
          second.in[0] == first.out[0] && second.out.size() == 1;
      const auto first_w = chain ? graph.initializers.find(first.in[1])
                                 : graph.initializers.end();
      const auto second_w = chain ? graph.initializers.find(second.in[1])
                                  : graph.initializers.end();
      if (chain && first_w != graph.initializers.end() &&
          second_w != graph.initializers.end() &&
          first_w->second.shape.size() == 4 && second_w->second.shape.size() == 4 &&
          first_w->second.shape[2] == 2 && first_w->second.shape[3] == 2 &&
          second_w->second.shape[2] == 2 && second_w->second.shape[3] == 2 &&
          first_w->second.shape[1] == second_w->second.shape[0]) {
        Node node = concat;
        node.op = "FusedConcatConvDualTranspose";
        node.name = concat.name.empty() ? "fused_concat_dual_transpose"
                                        : concat.name + "_dual_transpose";
        node.in.push_back(first.in[1]);
        node.in.push_back(first.in[2]);
        node.in.push_back(second.in[1]);
        node.in.push_back(second.in[2]);
        node.out = second.out;
        compact.push_back(std::move(node));
        i += 3;
        continue;
      }
    }
    compact.push_back(concat);
    ++i;
  }
  graph.nodes = std::move(compact);
}

// Detector DB head: FusedConvTransposeRelu 2x then FusedConvTransposeSigmoid
// 2x. Each input pixel owns an exclusive 4x4 output patch, so the 2x
// intermediate never needs to be stored. Same-host 8-run was not a stable
// e2e win versus two AVX-512 ConvTranspose passes, so this stays
// ENABLE-only (`PPOCR_ENABLE_CONVTRANSPOSE_CHAIN`).
void FuseConvTransposeChain(GraphData& graph) {
  if (std::getenv("PPOCR_ENABLE_CONVTRANSPOSE_CHAIN") == nullptr ||
      std::getenv("PPOCR_DISABLE_CONVTRANSPOSE_CHAIN") != nullptr) return;
  std::unordered_map<std::string, std::size_t> consumers;
  std::unordered_set<std::string> outputs(graph.outputs.begin(), graph.outputs.end());
  for (const auto& node : graph.nodes) for (const auto& input : node.in) {
    if (!input.empty()) ++consumers[input];
  }
  std::vector<Node> compact;
  compact.reserve(graph.nodes.size());
  for (std::size_t i = 0; i < graph.nodes.size();) {
    const auto& first = graph.nodes[i];
    if (i + 1 < graph.nodes.size() && first.op == "FusedConvTransposeRelu" &&
        first.in.size() >= 3 && first.out.size() == 1 &&
        consumers[first.out[0]] == 1 && !outputs.contains(first.out[0])) {
      const auto& second = graph.nodes[i + 1];
      const bool chain = second.op == "FusedConvTransposeSigmoid" &&
          second.in.size() >= 3 && second.in[0] == first.out[0] &&
          second.out.size() == 1;
      const auto first_w = chain ? graph.initializers.find(first.in[1])
                                 : graph.initializers.end();
      const auto first_b = chain ? graph.initializers.find(first.in[2])
                                 : graph.initializers.end();
      const auto second_w = chain ? graph.initializers.find(second.in[1])
                                  : graph.initializers.end();
      const auto second_b = chain ? graph.initializers.find(second.in[2])
                                  : graph.initializers.end();
      if (chain && first_w != graph.initializers.end() &&
          first_b != graph.initializers.end() &&
          second_w != graph.initializers.end() &&
          second_b != graph.initializers.end() &&
          first_w->second.shape.size() == 4 && second_w->second.shape.size() == 4 &&
          first_w->second.shape[2] == 2 && first_w->second.shape[3] == 2 &&
          second_w->second.shape[2] == 2 && second_w->second.shape[3] == 2 &&
          first_w->second.shape[1] == second_w->second.shape[0] &&
          first_b->second.data.size() == std::size_t(first_w->second.shape[1]) &&
          second_b->second.data.size() == std::size_t(second_w->second.shape[1])) {
        Tensor packed{{static_cast<std::int64_t>(second_w->second.data.size() +
                                                 second_b->second.data.size())},
                      std::vector<float>(second_w->second.data.size() +
                                         second_b->second.data.size())};
        std::memcpy(packed.data.data(), second_w->second.data.data(),
                    second_w->second.data.size() * sizeof(float));
        std::memcpy(packed.data.data() + second_w->second.data.size(),
                    second_b->second.data.data(),
                    second_b->second.data.size() * sizeof(float));
        const std::string packed_name = "__ppocr_ct_chain_w1b1_" + std::to_string(i);
        graph.initializers.emplace(packed_name, std::move(packed));
        Node node = first;
        node.op = "FusedConvTransposeChain";
        node.name = first.name.empty() ? "fused_convtranspose_chain"
                                       : first.name + "_chain";
        node.in = {first.in[0], first.in[1], first.in[2], packed_name};
        node.out = second.out;
        compact.push_back(std::move(node));
        i += 2;
        continue;
      }
    }
    compact.push_back(first);
    ++i;
  }
  graph.nodes = std::move(compact);
}

// Detector invert-residual: 1x1 (optional Relu) then depthwise 3x3 s1 pad1.
// Conv.4_relu is 16→32 on 40x176 and Conv.5 is the 32-ch 3x3. Eight-OC tiles
// keep the pointwise strip in L2 for the depthwise walk. Rec H<32 stays on
// the two-pass kernels (no extra inner ParallelFor). Disable with
// PPOCR_DISABLE_POINTWISE_DEPTHWISE.
void FusePointwiseDepthwise(GraphData& graph) {
  // 20-run with 8-OC tiles missed versus the two-node Relu8+DW walk
  // (15.09/14.97 vs 15.37/14.42; relu-only 15.46/15.87 vs 15.41/15.71).
  // Keep ENABLE-only (`PPOCR_ENABLE_POINTWISE_DEPTHWISE`).
  if (std::getenv("PPOCR_ENABLE_POINTWISE_DEPTHWISE") == nullptr ||
      std::getenv("PPOCR_DISABLE_POINTWISE_DEPTHWISE") != nullptr) return;
  std::unordered_map<std::string, std::size_t> consumers;
  std::unordered_set<std::string> outputs(graph.outputs.begin(), graph.outputs.end());
  for (const auto& node : graph.nodes) for (const auto& input : node.in) {
    if (!input.empty()) ++consumers[input];
  }
  const auto ints = [](const Node& node, const char* name,
                       std::vector<std::int64_t> fallback) {
    const auto it = node.attr.find(name);
    return it == node.attr.end() ? fallback : it->second.ints;
  };
  std::vector<Node> compact;
  compact.reserve(graph.nodes.size());
  for (std::size_t i = 0; i < graph.nodes.size();) {
    const auto& pw = graph.nodes[i];
    // Only the Relu 1x1 (Conv.4_relu). Unactivated 1x1+DW pairs on rec
    // H<32 maps stayed two-pass inside the fused node and the 20-run with
    // those extra pairs missed versus the two-node walk.
    const bool pw_op = pw.op == "FusedConvRelu";
    if (i + 1 < graph.nodes.size() && pw_op &&
        (pw.in.size() == 2 || pw.in.size() == 3) && pw.out.size() == 1 &&
        consumers[pw.out[0]] == 1 && !outputs.contains(pw.out[0])) {
      const auto& dw = graph.nodes[i + 1];
      const auto pw_w = graph.initializers.find(pw.in[1]);
      const auto pw_b = pw.in.size() == 3 ? graph.initializers.find(pw.in[2])
                                          : graph.initializers.end();
      const auto dw_w = dw.in.size() >= 2 ? graph.initializers.find(dw.in[1])
                                          : graph.initializers.end();
      const auto dw_b = dw.in.size() == 3 ? graph.initializers.find(dw.in[2])
                                          : graph.initializers.end();
      const auto pw_group = pw.attr.contains("group") ? pw.attr.at("group").i
                                                      : std::int64_t{1};
      const auto dw_group = dw.attr.contains("group") ? dw.attr.at("group").i
                                                      : std::int64_t{1};
      const auto pw_strides = ints(pw, "strides", {1, 1});
      const auto pw_pads = ints(pw, "pads", {0, 0, 0, 0});
      const auto pw_dilations = ints(pw, "dilations", {1, 1});
      const auto dw_strides = ints(dw, "strides", {1, 1});
      const auto dw_pads = ints(dw, "pads", {0, 0, 0, 0});
      const auto dw_dilations = ints(dw, "dilations", {1, 1});
      const bool dw_ok = dw.op == "Conv" &&
          (dw.in.size() == 2 || dw.in.size() == 3) && dw.out.size() == 1 &&
          dw.in[0] == pw.out[0] && !outputs.contains(dw.out[0]);
      const bool pw_1x1 = pw_w != graph.initializers.end() &&
          pw_w->second.shape.size() == 4 && pw_w->second.shape[2] == 1 &&
          pw_w->second.shape[3] == 1 && pw_group == 1 &&
          pw_strides == std::vector<std::int64_t>{1, 1} &&
          pw_pads == std::vector<std::int64_t>{0, 0, 0, 0} &&
          pw_dilations == std::vector<std::int64_t>{1, 1} &&
          (pw.attr.find("auto_pad") == pw.attr.end() ||
           pw.attr.at("auto_pad").s == "NOTSET") &&
          (pw.in.size() == 2 ||
           (pw_b != graph.initializers.end() &&
            pw_b->second.data.size() == std::size_t(pw_w->second.shape[0])));
      const bool dw_3x3 = dw_ok && pw_w != graph.initializers.end() &&
          dw_w != graph.initializers.end() &&
          dw_w->second.shape.size() == 4 && dw_w->second.shape[1] == 1 &&
          dw_w->second.shape[2] == 3 && dw_w->second.shape[3] == 3 &&
          dw_w->second.shape[0] == pw_w->second.shape[0] &&
          dw_group == pw_w->second.shape[0] &&
          dw_strides == std::vector<std::int64_t>{1, 1} &&
          dw_pads == std::vector<std::int64_t>{1, 1, 1, 1} &&
          dw_dilations == std::vector<std::int64_t>{1, 1} &&
          (dw.attr.find("auto_pad") == dw.attr.end() ||
           dw.attr.at("auto_pad").s == "NOTSET") &&
          (dw.in.size() == 2 ||
           (dw_b != graph.initializers.end() &&
            dw_b->second.data.size() == std::size_t(dw_w->second.shape[0])));
      if (pw_1x1 && dw_3x3) {
        std::string pw_bias_name;
        if (pw.in.size() == 3) {
          pw_bias_name = pw.in[2];
        } else {
          pw_bias_name = "__ppocr_zero_pwdw_pw_bias_" + std::to_string(i);
          graph.initializers.emplace(
              pw_bias_name,
              Tensor{{pw_w->second.shape[0]},
                     std::vector<float>(static_cast<std::size_t>(pw_w->second.shape[0]),
                                        0.F)});
        }
        std::string dw_bias_name;
        if (dw.in.size() == 3) {
          dw_bias_name = dw.in[2];
        } else {
          dw_bias_name = "__ppocr_zero_pwdw_dw_bias_" + std::to_string(i);
          graph.initializers.emplace(
              dw_bias_name,
              Tensor{{dw_w->second.shape[0]},
                     std::vector<float>(static_cast<std::size_t>(dw_w->second.shape[0]),
                                        0.F)});
        }
        Node node = dw;
        node.op = pw.op == "FusedConvRelu" ? "FusedPointwiseDepthwiseConvRelu"
                                          : "FusedPointwiseDepthwiseConv";
        node.name = pw.name.empty() ? "fused_pointwise_depthwise"
                                    : pw.name + "_depthwise";
        node.in = {pw.in[0], pw.in[1], pw_bias_name, dw.in[1], dw_bias_name};
        node.out = dw.out;
        compact.push_back(std::move(node));
        i += 2;
        continue;
      }
    }
    compact.push_back(pw);
    ++i;
  }
  graph.nodes = std::move(compact);
}

// Rec neck: Conv (optionally Squeeze of a unit axis) then
// FusedBatchNormHardSwish. Fold the affine into W/B and emit HardSwish on
// the convolution store. Same-host 8-run was not a stable e2e win on GPU
// rec maps, so this stays ENABLE-only (`PPOCR_ENABLE_CONV_BN_HSWISH`).
void FuseConvBatchNormHardSwish(GraphData& graph) {
  if (std::getenv("PPOCR_ENABLE_CONV_BN_HSWISH") == nullptr ||
      std::getenv("PPOCR_DISABLE_CONV_BN_HSWISH") != nullptr) return;
  std::unordered_map<std::string, std::size_t> consumers;
  std::unordered_set<std::string> outputs(graph.outputs.begin(), graph.outputs.end());
  for (const auto& node : graph.nodes) for (const auto& input : node.in) {
    if (!input.empty()) ++consumers[input];
  }
  std::vector<Node> compact;
  compact.reserve(graph.nodes.size());
  for (std::size_t i = 0; i < graph.nodes.size();) {
    const auto& conv = graph.nodes[i];
    const bool conv_ok = conv.op == "Conv" && conv.in.size() >= 2 && conv.out.size() == 1 &&
        consumers[conv.out[0]] == 1 && !outputs.contains(conv.out[0]);
    if (!conv_ok) {
      compact.push_back(conv);
      ++i;
      continue;
    }
    std::size_t bn_index = i + 1;
    const Node* squeeze = nullptr;
    if (i + 1 < graph.nodes.size() && graph.nodes[i + 1].op == "Squeeze" &&
        graph.nodes[i + 1].in.size() >= 1 && graph.nodes[i + 1].in[0] == conv.out[0] &&
        graph.nodes[i + 1].out.size() == 1 && consumers[graph.nodes[i + 1].out[0]] == 1 &&
        !outputs.contains(graph.nodes[i + 1].out[0])) {
      squeeze = &graph.nodes[i + 1];
      bn_index = i + 2;
    }
    if (bn_index >= graph.nodes.size()) {
      compact.push_back(conv);
      ++i;
      continue;
    }
    const auto& bn = graph.nodes[bn_index];
    const std::string conv_live = squeeze ? squeeze->out[0] : conv.out[0];
    if (bn.op != "FusedBatchNormHardSwish" || bn.in.size() != 3 || bn.in[0] != conv_live ||
        bn.out.size() != 1) {
      compact.push_back(conv);
      ++i;
      continue;
    }
    const auto weights = graph.initializers.find(conv.in[1]);
    const auto factor = graph.initializers.find(bn.in[1]);
    const auto offset = graph.initializers.find(bn.in[2]);
    if (weights == graph.initializers.end() || factor == graph.initializers.end() ||
        offset == graph.initializers.end() || weights->second.shape.size() != 4) {
      compact.push_back(conv);
      ++i;
      continue;
    }
    const auto channels = std::size_t(weights->second.shape[0]);
    if (channels == 0 || factor->second.data.size() != channels ||
        offset->second.data.size() != channels ||
        weights->second.data.size() % channels != 0) {
      compact.push_back(conv);
      ++i;
      continue;
    }
    const auto group = conv.attr.contains("group") ? conv.attr.at("group").i : std::int64_t{1};
    const bool depthwise = group > 0 && group == std::int64_t(channels) &&
        weights->second.shape[1] == 1;
    if (!depthwise && group != 1) {
      compact.push_back(conv);
      ++i;
      continue;
    }
    Tensor new_weights = weights->second;
    Tensor new_bias{{static_cast<std::int64_t>(channels)}, std::vector<float>(channels)};
    const float* old_bias = nullptr;
    if (conv.in.size() > 2 && !conv.in[2].empty()) {
      const auto existing = graph.initializers.find(conv.in[2]);
      if (existing == graph.initializers.end() ||
          existing->second.data.size() != channels) {
        compact.push_back(conv);
        ++i;
        continue;
      }
      old_bias = existing->second.data.data();
    }
    const auto per_channel = new_weights.data.size() / channels;
    for (std::size_t channel = 0; channel < channels; ++channel) {
      const float scale = factor->second.data[channel];
      float* data = new_weights.data.data() + channel * per_channel;
      for (std::size_t k = 0; k < per_channel; ++k) data[k] *= scale;
      new_bias.data[channel] = (old_bias ? old_bias[channel] : 0.F) * scale +
          offset->second.data[channel];
    }
    const std::string folded_w = "__ppocr_hswish_w_" + std::to_string(i);
    const std::string folded_b = "__ppocr_hswish_b_" + std::to_string(i);
    graph.initializers.emplace(folded_w, std::move(new_weights));
    graph.initializers.emplace(folded_b, std::move(new_bias));
    Node fused = conv;
    fused.op = depthwise ? "FusedDepthwiseHardSwish" : "FusedConvHardSwish";
    fused.name = conv.name.empty() ? "fused_conv_bn_hswish" : conv.name + "_bn_hswish";
    fused.in.resize(3);
    fused.in[1] = folded_w;
    fused.in[2] = folded_b;
    if (squeeze) {
      fused.out = conv.out;
      compact.push_back(std::move(fused));
      Node squeezed = *squeeze;
      squeezed.out = bn.out;
      compact.push_back(std::move(squeezed));
    } else {
      fused.out = bn.out;
      compact.push_back(std::move(fused));
    }
    i = bn_index + 1;
  }
  graph.nodes = std::move(compact);
}

// After Conv->GELU and residual 1x1->Add have already fused, collapse the
// recognizer/detector inverted residual into one spatial-tiled kernel:
// expand 1x1, exact GELU, project 1x1, residual add of the expand input.
// Hidden activations stay in a 16-wide tile instead of a full NCHW tensor.
void FuseRemainingConvGelu(GraphData& graph) {
  if (std::getenv("PPOCR_DISABLE_CONV_GELU_FUSION") != nullptr) return;
  std::unordered_map<std::string, std::size_t> consumers;
  std::unordered_set<std::string> outputs(graph.outputs.begin(), graph.outputs.end());
  for (const auto& node : graph.nodes) for (const auto& input : node.in) {
    if (!input.empty()) ++consumers[input];
  }
  std::vector<Node> compact;
  compact.reserve(graph.nodes.size());
  for (std::size_t i = 0; i < graph.nodes.size();) {
    const auto& conv = graph.nodes[i];
    if (i + 1 < graph.nodes.size() && conv.op == "Conv" && conv.out.size() == 1 &&
        consumers[conv.out[0]] == 1 && !outputs.contains(conv.out[0]) &&
        graph.nodes[i + 1].op == "FusedGelu" && graph.nodes[i + 1].in.size() == 1 &&
        graph.nodes[i + 1].in[0] == conv.out[0]) {
      Node node = conv;
      node.op = "FusedConvGelu";
      node.name = conv.name.empty() ? "fused_conv_gelu" : conv.name + "_gelu";
      node.out = graph.nodes[i + 1].out;
      compact.push_back(std::move(node));
      i += 2;
      continue;
    }
    compact.push_back(conv);
    ++i;
  }
  graph.nodes = std::move(compact);
}

void FuseExpandGeluProjectAdd(GraphData& graph) {
  if (std::getenv("PPOCR_DISABLE_EXPAND_GELU_PROJECT") != nullptr) return;
  std::unordered_map<std::string, std::size_t> consumers;
  std::unordered_set<std::string> outputs(graph.outputs.begin(), graph.outputs.end());
  for (const auto& node : graph.nodes) for (const auto& input : node.in) {
    if (!input.empty()) ++consumers[input];
  }
  const auto pointwise_1x1 = [&](const Node& node, std::int64_t* output_channels,
                                 std::int64_t* input_channels) -> bool {
    if (node.in.size() < 2) return false;
    const auto weights = graph.initializers.find(node.in[1]);
    if (weights == graph.initializers.end() || weights->second.shape.size() != 4 ||
        weights->second.shape[2] != 1 || weights->second.shape[3] != 1) {
      return false;
    }
    const auto group_it = node.attr.find("group");
    const auto auto_pad_it = node.attr.find("auto_pad");
    const auto attr_ints = [&](const char* key, std::vector<std::int64_t> fallback) {
      const auto it = node.attr.find(key);
      return it == node.attr.end() ? fallback : it->second.ints;
    };
    const auto strides = attr_ints("strides", {1, 1});
    const auto pads = attr_ints("pads", {0, 0, 0, 0});
    const auto dilations = attr_ints("dilations", {1, 1});
    if ((group_it != node.attr.end() && group_it->second.i != 1) ||
        (auto_pad_it != node.attr.end() && auto_pad_it->second.s != "NOTSET") ||
        strides.size() != 2 || pads.size() != 4 || dilations.size() != 2 ||
        strides[0] != 1 || strides[1] != 1 || dilations[0] != 1 || dilations[1] != 1 ||
        pads[0] != 0 || pads[1] != 0 || pads[2] != 0 || pads[3] != 0) {
      return false;
    }
    if (output_channels) *output_channels = weights->second.shape[0];
    if (input_channels) *input_channels = weights->second.shape[1];
    return true;
  };
  std::vector<Node> compact;
  compact.reserve(graph.nodes.size());
  for (std::size_t i = 0; i < graph.nodes.size();) {
    const auto& expand = graph.nodes[i];
    if (i + 1 < graph.nodes.size() && expand.op == "FusedConvGelu" &&
        expand.in.size() >= 2 && expand.out.size() == 1 &&
        consumers[expand.out[0]] == 1 && !outputs.contains(expand.out[0])) {
      const auto& project = graph.nodes[i + 1];
      if (project.op == "FusedPointwiseConvAdd" && project.in.size() == 4 &&
          project.out.size() == 1 && project.in[0] == expand.out[0] &&
          project.in[3] == expand.in[0]) {
        std::int64_t expand_m = 0, expand_c = 0, project_m = 0, project_c = 0;
        if (pointwise_1x1(expand, &expand_m, &expand_c) &&
            pointwise_1x1(project, &project_m, &project_c) &&
            expand_m == project_c && expand_c == project_m && expand_c > 0 &&
            expand_m > 0) {
          Node node = expand;
          node.op = "FusedExpandGeluProjectAdd";
          node.name = expand.name.empty() ? "fused_expand_gelu_project"
                                          : expand.name + "_project_add";
          node.in = {expand.in[0], expand.in[1],
                     expand.in.size() > 2 ? expand.in[2] : "",
                     project.in[1], project.in[2]};
          node.out = project.out;
          compact.push_back(std::move(node));
          i += 2;
          continue;
        }
      }
    }
    compact.push_back(expand);
    ++i;
  }
  graph.nodes = std::move(compact);
}

// Depthwise 3x3 then invert-residual ExpandGelu. Keeps the depthwise row in a
// C*W scratch instead of a full NCHW map the expand 1x1 would otherwise reread.
void FuseDepthwiseExpandGelu(GraphData& graph) {
  // Rec DW is 48-ch on 12xW (below the DW ParallelFor floor). Det 16-ch
  // 40x176 now fuses too and stays on two-pass (DW channel ParallelFor +
  // expand spatial tiles). `PPOCR_DW_EXPAND_MIN_GROUP` restores rec-only
  // fusion at 24. `PPOCR_ENABLE_DEPTHWISE_EXPAND_GELU` fuses every width;
  // `PPOCR_DISABLE_DEPTHWISE_EXPAND_GELU` turns this off.
  if (std::getenv("PPOCR_DISABLE_DEPTHWISE_EXPAND_GELU") != nullptr) return;
  const bool fuse_all =
      std::getenv("PPOCR_ENABLE_DEPTHWISE_EXPAND_GELU") != nullptr;
  std::int64_t min_group = 16;
  if (const char* configured = std::getenv("PPOCR_DW_EXPAND_MIN_GROUP")) {
    char* end = nullptr;
    const long parsed = std::strtol(configured, &end, 10);
    if (end != configured && *end == '\0' && parsed > 0 && parsed <= 1024) {
      min_group = parsed;
    }
  }
  std::unordered_map<std::string, std::size_t> consumers;
  std::unordered_set<std::string> outputs(graph.outputs.begin(), graph.outputs.end());
  for (const auto& node : graph.nodes) for (const auto& input : node.in) {
    if (!input.empty()) ++consumers[input];
  }
  std::vector<Node> compact;
  compact.reserve(graph.nodes.size());
  for (std::size_t i = 0; i < graph.nodes.size();) {
    const auto& depthwise = graph.nodes[i];
    if (i + 1 < graph.nodes.size() && depthwise.op == "Conv" &&
        (depthwise.in.size() == 2 || depthwise.in.size() == 3) &&
        depthwise.out.size() == 1 && consumers[depthwise.out[0]] == 1 &&
        !outputs.contains(depthwise.out[0])) {
      const auto& expand = graph.nodes[i + 1];
      if (expand.op == "FusedExpandGeluProjectAdd" && expand.in.size() == 5 &&
          expand.in[0] == depthwise.out[0] && expand.out.size() == 1) {
        const auto weights = graph.initializers.find(depthwise.in[1]);
        const auto group = depthwise.attr.contains("group")
            ? depthwise.attr.at("group").i : std::int64_t{1};
        const auto attr_ints = [&](const char* key, std::vector<std::int64_t> fallback) {
          const auto it = depthwise.attr.find(key);
          return it == depthwise.attr.end() ? fallback : it->second.ints;
        };
        const auto strides = attr_ints("strides", {1, 1});
        const auto pads = attr_ints("pads", {0, 0, 0, 0});
        const auto dilations = attr_ints("dilations", {1, 1});
        const auto auto_pad = depthwise.attr.find("auto_pad") == depthwise.attr.end()
            ? std::string("NOTSET") : depthwise.attr.at("auto_pad").s;
        const bool same_pad = pads == std::vector<std::int64_t>{1, 1, 1, 1} ||
            ((auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") &&
             (pads.empty() || pads == std::vector<std::int64_t>{0, 0, 0, 0}));
        const bool is_dw = weights != graph.initializers.end() &&
            weights->second.shape.size() == 4 && group > 0 &&
            weights->second.shape[0] == group && weights->second.shape[1] == 1 &&
            weights->second.shape[2] == 3 && weights->second.shape[3] == 3 &&
            strides == std::vector<std::int64_t>{1, 1} && same_pad &&
            dilations == std::vector<std::int64_t>{1, 1};
        // Keep the terminal 896-channel detector block separate.  Its AMD
        // qualification uses a dedicated strict-GPU depthwise dispatch and
        // must not silently change scheduling through a graph fusion.
        const bool is_terminal_medium_tail =
            depthwise.out[0] == "p2o.pd_op.depthwise_conv2d.12.0" && group == 896;
        if (is_dw && !is_terminal_medium_tail && (fuse_all || group >= min_group)) {
          std::string bias_name;
          if (depthwise.in.size() == 3) {
            const auto bias = graph.initializers.find(depthwise.in[2]);
            if (bias == graph.initializers.end() ||
                bias->second.data.size() != std::size_t(group)) {
              compact.push_back(depthwise);
              ++i;
              continue;
            }
            bias_name = depthwise.in[2];
          } else {
            bias_name = "__ppocr_zero_dw_expand_bias_" + std::to_string(i);
            graph.initializers.emplace(bias_name,
                Tensor{{group}, std::vector<float>(static_cast<std::size_t>(group), 0.F)});
          }
          Node node = depthwise;
          node.op = "FusedDepthwiseExpandGeluProjectAdd";
          node.name = depthwise.name.empty() ? "fused_dw_expand_gelu"
                                             : depthwise.name + "_expand_gelu";
          node.attr["pads"].ints = {1, 1, 1, 1};
          node.attr["strides"].ints = {1, 1};
          node.attr["auto_pad"].s = "NOTSET";
          node.in = {depthwise.in[0], depthwise.in[1], bias_name, expand.in[1],
                     expand.in[2], expand.in[3], expand.in[4]};
          node.out = expand.out;
          compact.push_back(std::move(node));
          i += 2;
          continue;
        }
      }
    }
    compact.push_back(depthwise);
    ++i;
  }
  graph.nodes = std::move(compact);
}

void FuseSEGateFromFusedOps(GraphData& graph) {
  if (std::getenv("PPOCR_DISABLE_SE_GATE_FUSION") != nullptr) return;
  std::unordered_map<std::string, std::size_t> consumers;
  std::unordered_set<std::string> outputs(graph.outputs.begin(), graph.outputs.end());
  for (const auto& node : graph.nodes) for (const auto& name : node.in) {
    if (!name.empty()) ++consumers[name];
  }
  std::vector<Node> fused;
  fused.reserve(graph.nodes.size());
  for (std::size_t i = 0; i < graph.nodes.size();) {
    const auto& reduce = graph.nodes[i];
    if (i + 3 < graph.nodes.size() && reduce.op == "ReduceMean" &&
        reduce.in.size() == 1 && reduce.out.size() == 1 && consumers[reduce.out[0]] == 1 &&
        !outputs.contains(reduce.out[0])) {
      const auto& first = graph.nodes[i + 1];
      const auto& second = graph.nodes[i + 2];
      const auto& multiply = graph.nodes[i + 3];
      const auto axes = reduce.attr.contains("axes") ? reduce.attr.at("axes").ints
                                                       : std::vector<std::int64_t>{};
      const auto keepdims = reduce.attr.contains("keepdims") &&
          reduce.attr.at("keepdims").i != 0 ? reduce.attr.at("keepdims").i : 1;
      const bool ordered = first.op == "FusedConvRelu" && first.in.size() == 3 &&
          first.in[0] == reduce.out[0] && first.out.size() == 1 &&
          consumers[first.out[0]] == 1 && second.op == "FusedConvHardSigmoid" &&
          second.in.size() == 3 && second.in[0] == first.out[0] &&
          second.out.size() == 1 && consumers[second.out[0]] == 1 &&
          multiply.op == "Mul" && multiply.in.size() == 2 && multiply.out.size() == 1 &&
          !outputs.contains(multiply.out[0]) &&
          ((multiply.in[0] == reduce.in[0] && multiply.in[1] == second.out[0]) ||
           (multiply.in[1] == reduce.in[0] && multiply.in[0] == second.out[0]));
      const auto first_weights = ordered ? graph.initializers.find(first.in[1])
                                         : graph.initializers.end();
      const auto second_weights = ordered ? graph.initializers.find(second.in[1])
                                          : graph.initializers.end();
      const auto first_bias = ordered ? graph.initializers.find(first.in[2])
                                      : graph.initializers.end();
      const auto second_bias = ordered ? graph.initializers.find(second.in[2])
                                       : graph.initializers.end();
      const bool axes_ok = axes == std::vector<std::int64_t>{2, 3} ||
          axes == std::vector<std::int64_t>{-2, -1};
      if (ordered && first_weights != graph.initializers.end() &&
          second_weights != graph.initializers.end() && axes_ok && keepdims == 1 &&
          consumers[reduce.in[0]] == 2 &&
          first_weights->second.shape.size() == 4 &&
          second_weights->second.shape.size() == 4 &&
          first_weights->second.shape[2] == 1 && first_weights->second.shape[3] == 1 &&
          second_weights->second.shape[2] == 1 && second_weights->second.shape[3] == 1 &&
          first_weights->second.shape[1] == second_weights->second.shape[0] &&
          first_weights->second.shape[0] == second_weights->second.shape[1] &&
          first_bias != graph.initializers.end() && second_bias != graph.initializers.end() &&
          first_bias->second.data.size() == std::size_t(first_weights->second.shape[0]) &&
          second_bias->second.data.size() == std::size_t(second_weights->second.shape[0]) &&
          first_weights->second.shape[1] >= 24) {
        Node node;
        node.op = "FusedSEGateMul";
        node.name = reduce.name.empty() ? "fused_se_gate_mul" : reduce.name + "_se_gate_mul";
        node.in = {reduce.in[0], first.in[1], first.in[2], second.in[1], second.in[2]};
        node.out = multiply.out;
        node.attr["alpha"].f = second.attr.contains("__ppocr_fused_alpha")
            ? second.attr.at("__ppocr_fused_alpha").f
            : (second.attr.contains("alpha") ? second.attr.at("alpha").f : .2F);
        node.attr["beta"].f = second.attr.contains("__ppocr_fused_beta")
            ? second.attr.at("__ppocr_fused_beta").f
            : (second.attr.contains("beta") ? second.attr.at("beta").f : .5F);
        fused.push_back(std::move(node));
        i += 4;
        continue;
      }
    }
    // Tiny-det FPN SE is GlobalAveragePool(x) -> 1x1 Relu -> 1x1 HardSigmoid
    // -> Mul(x, gate) -> Add(x, mul) i.e. x * (1 + SE(x)). Eight of these
    // sit after unfused 1x1 projections. Collapse the five nodes so x's
    // dying storage is scaled in place. `PPOCR_DISABLE_SE_RESIDUAL_FUSION`
    // restores the five-node walk.
    if (std::getenv("PPOCR_DISABLE_SE_RESIDUAL_FUSION") == nullptr &&
        i + 4 < graph.nodes.size() && reduce.op == "GlobalAveragePool" &&
        reduce.in.size() == 1 && reduce.out.size() == 1 && consumers[reduce.out[0]] == 1 &&
        !outputs.contains(reduce.out[0])) {
      const auto& first = graph.nodes[i + 1];
      const auto& second = graph.nodes[i + 2];
      const auto& multiply = graph.nodes[i + 3];
      const auto& add = graph.nodes[i + 4];
      const bool ordered = first.op == "FusedConvRelu" && first.in.size() == 3 &&
          first.in[0] == reduce.out[0] && first.out.size() == 1 &&
          consumers[first.out[0]] == 1 && second.op == "FusedConvHardSigmoid" &&
          second.in.size() == 3 && second.in[0] == first.out[0] &&
          second.out.size() == 1 && consumers[second.out[0]] == 1 &&
          multiply.op == "Mul" && multiply.in.size() == 2 && multiply.out.size() == 1 &&
          consumers[multiply.out[0]] == 1 && !outputs.contains(multiply.out[0]) &&
          ((multiply.in[0] == reduce.in[0] && multiply.in[1] == second.out[0]) ||
           (multiply.in[1] == reduce.in[0] && multiply.in[0] == second.out[0])) &&
          add.op == "Add" && add.in.size() == 2 && add.out.size() == 1 &&
          ((add.in[0] == reduce.in[0] && add.in[1] == multiply.out[0]) ||
           (add.in[1] == reduce.in[0] && add.in[0] == multiply.out[0]));
      const auto first_weights = ordered ? graph.initializers.find(first.in[1])
                                         : graph.initializers.end();
      const auto second_weights = ordered ? graph.initializers.find(second.in[1])
                                          : graph.initializers.end();
      const auto first_bias = ordered ? graph.initializers.find(first.in[2])
                                      : graph.initializers.end();
      const auto second_bias = ordered ? graph.initializers.find(second.in[2])
                                       : graph.initializers.end();
      if (ordered && first_weights != graph.initializers.end() &&
          second_weights != graph.initializers.end() &&
          consumers[reduce.in[0]] == 3 && !outputs.contains(reduce.in[0]) &&
          first_weights->second.shape.size() == 4 &&
          second_weights->second.shape.size() == 4 &&
          first_weights->second.shape[2] == 1 && first_weights->second.shape[3] == 1 &&
          second_weights->second.shape[2] == 1 && second_weights->second.shape[3] == 1 &&
          first_weights->second.shape[1] == second_weights->second.shape[0] &&
          first_weights->second.shape[0] == second_weights->second.shape[1] &&
          first_bias != graph.initializers.end() && second_bias != graph.initializers.end() &&
          first_bias->second.data.size() == std::size_t(first_weights->second.shape[0]) &&
          second_bias->second.data.size() == std::size_t(second_weights->second.shape[0]) &&
          first_weights->second.shape[1] >= 8) {
        Node node;
        node.op = "FusedSEGateMul";
        node.name = reduce.name.empty() ? "fused_se_residual" : reduce.name + "_se_residual";
        node.in = {reduce.in[0], first.in[1], first.in[2], second.in[1], second.in[2]};
        node.out = add.out;
        node.attr["alpha"].f = second.attr.contains("__ppocr_fused_alpha")
            ? second.attr.at("__ppocr_fused_alpha").f
            : (second.attr.contains("alpha") ? second.attr.at("alpha").f : .2F);
        node.attr["beta"].f = second.attr.contains("__ppocr_fused_beta")
            ? second.attr.at("__ppocr_fused_beta").f
            : (second.attr.contains("beta") ? second.attr.at("beta").f : .5F);
        node.attr["residual"].i = 1;
        fused.push_back(std::move(node));
        i += 5;
        continue;
      }
    }
    fused.push_back(reduce);
    ++i;
  }
  graph.nodes = std::move(fused);
}

void ParseGraph(const std::uint8_t* data, std::size_t size, GraphData& g) {
  Reader r{data, size};
  while (!r.eof()) {
    const auto [field, wire] = r.Tag();
    if (field == 1) g.nodes.push_back(ParseNode(r.Bytes()));
    else if (field == 5) { auto x = r.Bytes(); std::string name; auto t = ParseTensor(x, name); g.initializers.emplace(std::move(name), std::move(t)); }
    else if (field == 11 || field == 12) {
      auto x = r.Bytes(); Reader vr{x}; std::string name;
      while (!vr.eof()) { const auto [f, w] = vr.Tag(); if (f == 1) { auto s = vr.Bytes(); name.assign(s.begin(), s.end()); } else vr.Skip(w); }
      (field == 11 ? g.inputs : g.outputs).push_back(std::move(name));
    } else r.Skip(wire);
  }
}

GraphData ParseModel(const std::string& file) {
  std::ifstream in(file, std::ios::binary); if (!in) Fail("cannot open " + file);
  std::vector<std::uint8_t> b((std::istreambuf_iterator<char>(in)), {}); Reader r{b}; GraphData g;
  while (!r.eof()) { const auto [field, wire] = r.Tag(); if (field == 7) { const auto [data, size] = r.Span(); ParseGraph(data, size, g); } else if (field == 8) { auto x = r.Bytes(); Reader orr{x}; std::string domain; int version{}; while (!orr.eof()) { auto [f,w]=orr.Tag(); if(f==1){auto s=orr.Bytes();domain.assign(s.begin(),s.end());} else if(f==2)version=int(orr.Varint()); else orr.Skip(w); } if(domain.empty())g.opset=version; } else r.Skip(wire); }
  if (g.nodes.empty() || g.outputs.empty()) Fail("model has no executable graph");
  std::unordered_set<std::string> init; for (const auto& [name, _] : g.initializers) init.insert(name);
  g.inputs.erase(std::remove_if(g.inputs.begin(), g.inputs.end(), [&](const auto& n) { return init.contains(n); }), g.inputs.end());
  EliminateInternalIdentity(g);
  EliminateAdjacentInverseTranspose(g);
  EliminateInternalIdentity(g);
  FuseQkvTransposeSlices(g);
  // Identity elimination exposes the canonical local Conv/BatchNorm/Swish
  // and BatchNorm/GELU chains before graph fusion consumes their producers.
  // A second local sweep exposes patterns whose producer was itself created
  // by the first pass, notably `BatchNorm->Swish->Add(shortcut)`. Every
  // matcher still requires adjacent nodes and strict use counts, preserving
  // exporter topology while eliminating that remaining full-map traversal.
  FuseActivationChains(g);
  FuseActivationChains(g);
  FusePointwiseDepthwise(g);
  FuseConcatConvDualTranspose(g);
  FuseConvTransposeChain(g);
  FuseConvBatchNormHardSwish(g);
  FuseSEGateFromFusedOps(g);
  FuseRemainingConvGelu(g);
  FuseExpandGeluProjectAdd(g);
  FuseDepthwiseExpandGelu(g);
  FuseDepthwisePointwiseGelu(g);
  PruneUnusedInitializers(g);
  // This is intentionally an opt-in diagnostic.  GPU-only support is
  // defined against the post-fusion graph, not the raw exporter graph; keep
  // a compact inventory available when bringing a new PP-OCRv6 variant onto
  // the device executor.
  if (std::getenv("PPOCR_DUMP_FUSED_OPS") != nullptr) {
    std::unordered_map<std::string, std::size_t> counts;
    for (const auto& node : g.nodes) ++counts[node.op];
    std::vector<std::pair<std::string, std::size_t>> ordered(counts.begin(), counts.end());
    std::sort(ordered.begin(), ordered.end());
    std::cerr << "ppocr fused graph " << file << " (" << g.nodes.size() << " nodes):\n";
    for (const auto& [op, count] : ordered) std::cerr << "  " << op << ": " << count << '\n';
    // Topology is useful when bringing an exporter variant to the strict
    // device executor.  Keep the usual inventory compact; emit edges only
    // on an explicit second diagnostic switch.
    if (std::getenv("PPOCR_DUMP_FUSED_NODES") != nullptr) {
      for (const auto& node : g.nodes) {
        std::cerr << "    " << node.op << " (";
        for (std::size_t i = 0; i < node.in.size(); ++i) {
          if (i) std::cerr << ", ";
          std::cerr << node.in[i];
        }
        std::cerr << ") -> ";
        for (std::size_t i = 0; i < node.out.size(); ++i) {
          if (i) std::cerr << ", ";
          std::cerr << node.out[i];
        }
        std::cerr << '\n';
      }
    }
  }
  return g;
}

const Attribute& Attr(const Node& n, const char* key) { static const Attribute empty{}; const auto it=n.attr.find(key); return it==n.attr.end()?empty:it->second; }
std::int64_t AttrInt(const Node& n, const char* key, std::int64_t def) { const auto it=n.attr.find(key); return it==n.attr.end()?def:it->second.i; }
std::vector<std::int64_t> AttrInts(const Node& n, const char* key, std::vector<std::int64_t> def) { const auto it=n.attr.find(key); return it==n.attr.end()?def:it->second.ints; }
std::string AttrStr(const Node& n, const char* key, const char* def) { const auto it=n.attr.find(key); return it==n.attr.end()?def:it->second.s; }
int Axis(std::int64_t axis, int rank) { int a=int(axis); if(a<0)a+=rank; if(a<0||a>=rank)Fail("axis out of range"); return a; }
std::vector<std::int64_t> BroadcastShape(const std::vector<std::int64_t>& a, const std::vector<std::int64_t>& b) {
  const auto rank=std::max(a.size(),b.size()); std::vector<std::int64_t> out(rank,1);
  for(std::size_t i=0;i<rank;++i){auto x=i<a.size()?a[a.size()-1-i]:1;auto y=i<b.size()?b[b.size()-1-i]:1;if(x!=y&&x!=1&&y!=1)Fail("broadcast mismatch");out[rank-1-i]=std::max(x,y);} return out;
}
std::vector<std::size_t> Strides(const std::vector<std::int64_t>& shape) { std::vector<std::size_t> s(shape.size()); std::size_t n=1; for(std::size_t i=shape.size();i-->0;){s[i]=n;n*=std::size_t(shape[i]);}return s; }
std::size_t Offset(std::size_t oi, const std::vector<std::int64_t>& out, const std::vector<std::int64_t>& in) { const auto os=Strides(out), is=Strides(in); const auto pad=out.size()-in.size(); std::size_t off{}; for(std::size_t d=0;d<out.size();++d){const auto idx=(oi/os[d])%std::size_t(out[d]);if(d>=pad&&in[d-pad]!=1)off+=idx*is[d-pad];}return off; }

Tensor Binary(const Tensor& a, const Tensor& b, const std::function<float(float,float)>& fn) { Tensor o{BroadcastShape(a.shape,b.shape),{}};o.data.resize(Elements(o.shape));for(std::size_t i=0;i<o.data.size();++i)o.data[i]=fn(a.data[Offset(i,o.shape,a.shape)],b.data[Offset(i,o.shape,b.shape)]);return o; }
Tensor Binary(const Tensor& a, const Tensor& b, kernels::BinaryOp op) {
  const auto shape = BroadcastShape(a.shape,b.shape);
  Tensor o{shape, {}};
  o.data.resize(Elements(shape));
  if (a.data.size() == 1) {
    kernels::BinaryScalar(o.data.data(), b.data.data(), o.data.size(), a.data[0], op, true);
    return o;
  }
  if (b.data.size() == 1) {
    kernels::BinaryScalar(o.data.data(), a.data.data(), o.data.size(), b.data[0], op, false);
    return o;
  }
  // Exact equal-shape tensors are contiguous and account for residual paths,
  // elementwise activations, and the bulk of PP-OCRv6 elementwise work.
  if (a.shape == shape && b.shape == shape) {
    kernels::Binary(o.data.data(), a.data.data(), b.data.data(), o.data.size(), op);
    return o;
  }
  // PP-OCRv6 broadcasts normalisation/activation tensors. For each operand,
  // dimensions may only be equal to the output or one; precompute the exact
  // contiguous repeat period once rather than rebuilding strides per element.
  const auto repeat_period = [&](const Tensor& input) -> std::size_t {
    if (input.shape == shape) return 1;
    const auto padding = shape.size() - input.shape.size();
    std::size_t repeat = 1;
    bool suffix_is_broadcast = true;
    std::size_t split = shape.size();
    for (std::size_t od = shape.size(); od-- > 0;) {
      const auto id = od >= padding ? input.shape[od-padding] : 1;
      if (id != 1) { split = od + 1; break; }
      repeat *= std::size_t(shape[od]);
    }
    for (std::size_t od = 0; od < split; ++od) {
      const auto id = od >= padding ? input.shape[od-padding] : 1;
      if (id != shape[od]) { suffix_is_broadcast = false; break; }
    }
    if (!suffix_is_broadcast || repeat == 1) return 0;
    return repeat;
  };
  const auto a_repeat = repeat_period(a), b_repeat = repeat_period(b);
  if (a_repeat && b_repeat) {
    // One operand is contiguous and the other is a scalar/row/plane pattern.
    // Walk a full pattern once per outer block; no coordinate/stride math.
    const bool a_pattern = a_repeat > 1;
    const auto pattern = a_pattern ? a_repeat : b_repeat;
    if (pattern && o.data.size() % pattern == 0) {
      for (std::size_t offset = 0; offset < o.data.size(); offset += pattern) {
        const float* ap = a_pattern ? a.data.data() + offset / a_repeat : a.data.data() + offset;
        const float* bp = a_pattern ? b.data.data() + offset : b.data.data() + offset / b_repeat;
        kernels::BinaryBroadcast(o.data.data()+offset, ap, bp, pattern,
                                 a_pattern ? a_repeat : 1,
                                 a_pattern ? 1 : b_repeat, op);
      }
      return o;
    }
  }
  for(std::size_t i=0;i<o.data.size();++i) {
    const auto ai = Offset(i, o.shape, a.shape), bi = Offset(i, o.shape, b.shape);
    switch (op) {
      case kernels::BinaryOp::add: o.data[i]=a.data[ai]+b.data[bi]; break;
      case kernels::BinaryOp::sub: o.data[i]=a.data[ai]-b.data[bi]; break;
      case kernels::BinaryOp::mul: o.data[i]=a.data[ai]*b.data[bi]; break;
      case kernels::BinaryOp::div: o.data[i]=a.data[ai]/b.data[bi]; break;
    }
  }
  return o;
}

// Return the contiguous output span mapped to one RHS item when `right`
// suffix-broadcasts into `left`; zero means the layout needs generic strides.
std::size_t RightBroadcastRepeat(const Tensor& left, const Tensor& right) {
  if (right.data.size() == 1) return left.data.size();
  if (left.shape.size() < right.shape.size()) return 0;
  const auto padding = left.shape.size() - right.shape.size();
  std::size_t repeat = 1;
  bool suffix_broadcast = true;
  std::size_t split = left.shape.size();
  for (std::size_t axis = left.shape.size(); axis-- > 0;) {
    const auto dimension = axis >= padding ? right.shape[axis-padding] : 1;
    if (dimension != 1) { split = axis + 1; break; }
    repeat *= std::size_t(left.shape[axis]);
  }
  for (std::size_t axis = 0; axis < split; ++axis) {
    const auto dimension = axis >= padding ? right.shape[axis-padding] : 1;
    if (dimension != 1 && dimension != left.shape[axis]) { suffix_broadcast = false; break; }
  }
  if (!suffix_broadcast || repeat == 1 || left.data.size() % repeat != 0) return 0;
  return repeat;
}
Tensor Unary(const Tensor& a, const std::function<float(float)>& fn) { Tensor o=a;for(auto&x:o.data)x=fn(x);return o; }

// A pointwise 1x1 Conv is a substantial recognizer/detector workload and is
// the first dense operator worth moving to Vulkan.  It remains explicitly
// hybrid: each shape is admitted only after a complete upload/dispatch/readback
// comparison against the CPU SIMD kernel, so discrete GPUs and small maps
// retain the faster CPU path.  The immutable ONNX weights/bias are cached by
// the persistent Vulkan context after admission.
bool TryHybridVulkanPointwiseConv(
    Backend backend, Tensor& output, const Tensor& input, const Tensor& weights,
    const Tensor* bias, bool relu, bool immutable_parameters, bool swish = false,
    bool sigmoid = false, bool hard_sigmoid = false,
    float hard_sigmoid_alpha = .2F, float hard_sigmoid_beta = .5F,
    bool hard_swish = false) noexcept {
  if (backend != Backend::hybrid || !bias || input.shape.size() != 4 ||
      weights.shape.size() != 4 || output.shape.size() != 4 ||
      weights.shape[2] != 1 || weights.shape[3] != 1 ||
      (int(relu) + int(swish) + int(sigmoid) + int(hard_sigmoid) + int(hard_swish) > 1) ||
      !output.data.data() || !input.data.data() || !weights.data.data() || !bias->data.data() ||
      output.data.size() < 65536) return false;
  const int batches = static_cast<int>(input.shape[0]);
  const int input_channels = static_cast<int>(input.shape[1]);
  const int output_channels = static_cast<int>(output.shape[1]);
  const auto plane = static_cast<std::size_t>(input.shape[2]) *
                     static_cast<std::size_t>(input.shape[3]);
  if (batches <= 0 || input_channels <= 0 || output_channels <= 0 || plane == 0 ||
      output.shape[0] != input.shape[0] || output.shape[2] != input.shape[2] ||
      output.shape[3] != input.shape[3] || weights.shape[0] != output_channels ||
      weights.shape[1] != input_channels || weights.data.size() !=
          static_cast<std::size_t>(input_channels) * output_channels ||
      bias->data.size() != static_cast<std::size_t>(output_channels)) return false;
  const auto expected_input = static_cast<std::size_t>(batches) * input_channels * plane;
  const auto expected_output = static_cast<std::size_t>(batches) * output_channels * plane;
  if (input.data.size() != expected_input || output.data.size() != expected_output) return false;
  const auto shape_key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(batches)) << 48) ^
                   (static_cast<std::uint64_t>(plane) << 32) ^
                   (static_cast<std::uint64_t>(static_cast<std::uint32_t>(input_channels)) << 16) ^
                   (static_cast<std::uint64_t>(static_cast<std::uint32_t>(output_channels)) << 3) ^
                   (static_cast<std::uint64_t>(relu) << 2) ^
                   (static_cast<std::uint64_t>(swish) << 1) ^ static_cast<std::uint64_t>(sigmoid) ^
                   (static_cast<std::uint64_t>(hard_sigmoid) << 4) ^
                   (static_cast<std::uint64_t>(hard_swish) << 63) ^
                   (static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(hard_sigmoid_alpha)) << 5) ^
                   (static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(hard_sigmoid_beta)) << 21);
  const auto key = WithHybridAdmissionContext(shape_key);
  static std::mutex admission_mutex;
  static std::unordered_map<std::uint64_t, bool> admitted;
  bool select_gpu{};
  {
    std::lock_guard lock(admission_mutex);
    const auto found = admitted.find(key);
    if (found != admitted.end()) {
      select_gpu = found->second;
    } else {
      double gpu_ms{}, cpu_ms{};
      select_gpu = VulkanPointwiseConvBatchNoSlowerThanCpu(
          static_cast<std::size_t>(batches), input_channels, output_channels, plane,
          &gpu_ms, &cpu_ms, immutable_parameters, relu, swish, sigmoid,
          hard_sigmoid, hard_sigmoid_alpha, hard_sigmoid_beta, hard_swish);
      admitted.emplace(key, select_gpu);
    }
  }
  return select_gpu && VulkanPointwiseConvBatch(
      output.data.data(), input.data.data(), weights.data.data(), bias->data.data(),
      static_cast<std::size_t>(batches), input_channels, output_channels, plane,
      immutable_parameters, relu, swish, sigmoid,
      hard_sigmoid, hard_sigmoid_alpha, hard_sigmoid_beta, hard_swish);
}

// Conv projection shortcuts are pervasive in the detector and benefit from a
// single GPU submission more than a standalone pointwise Conv: the residual
// is uploaded directly to the output buffer, added in the shader, then (for
// the ReLU graph form) activated before the one readback. As with every
// hybrid path, shape admission measures the complete synchronous segment.
bool TryHybridVulkanPointwiseConvAdd(
    Backend backend, Tensor& output, const Tensor& input, const Tensor& weights,
    const Tensor* bias, const Tensor& residual, bool relu,
    bool immutable_parameters, bool swish = false) noexcept {
  if (backend != Backend::hybrid || !bias || input.shape.size() != 4 ||
      weights.shape.size() != 4 || output.shape.size() != 4 ||
      weights.shape[2] != 1 || weights.shape[3] != 1 || output.shape != residual.shape ||
      !residual.data.data() || output.data.size() < 65536 || (relu && swish)) return false;
  const int batches = static_cast<int>(input.shape[0]);
  const int input_channels = static_cast<int>(input.shape[1]);
  const int output_channels = static_cast<int>(output.shape[1]);
  const auto plane = static_cast<std::size_t>(input.shape[2]) *
                     static_cast<std::size_t>(input.shape[3]);
  if (batches <= 0 || input_channels <= 0 || output_channels <= 0 || plane == 0 ||
      output.data.size() != static_cast<std::size_t>(batches) * output_channels * plane ||
      input.data.size() != static_cast<std::size_t>(batches) * input_channels * plane ||
      weights.data.size() != static_cast<std::size_t>(output_channels) * input_channels ||
      bias->data.size() != static_cast<std::size_t>(output_channels)) return false;
  const std::uint64_t shape_key =
      (static_cast<std::uint64_t>(static_cast<std::uint32_t>(batches)) << 48) ^
      (static_cast<std::uint64_t>(static_cast<std::uint32_t>(plane)) << 16) ^
      (static_cast<std::uint64_t>(static_cast<std::uint32_t>(input_channels)) << 32) ^
      (static_cast<std::uint64_t>(static_cast<std::uint32_t>(output_channels)) << 1) ^
      (static_cast<std::uint64_t>(relu) << 1) ^ static_cast<std::uint64_t>(swish);
  const std::uint64_t key = WithHybridAdmissionContext(shape_key);
  static std::mutex admission_mutex;
  static std::unordered_map<std::uint64_t, bool> admitted;
  bool select_gpu{};
  {
    std::lock_guard lock(admission_mutex);
    const auto found = admitted.find(key);
    if (found != admitted.end()) {
      select_gpu = found->second;
    } else {
      double gpu_ms{}, cpu_ms{};
      select_gpu = VulkanPointwiseConvAddBatchNoSlowerThanCpu(
          static_cast<std::size_t>(batches), input_channels, output_channels, plane,
          &gpu_ms, &cpu_ms, immutable_parameters, relu, swish);
      admitted.emplace(key, select_gpu);
    }
  }
  return select_gpu && VulkanPointwiseConvAddBatch(
      output.data.data(), input.data.data(), weights.data.data(), bias->data.data(),
      residual.data.data(), static_cast<std::size_t>(batches), input_channels,
      output_channels, plane, immutable_parameters, relu, swish);
}

// Depthwise layers account for much of the MobileNet detector/recognizer
// backbone. Unlike a regular convolution, every channel has one independent
// small filter, which maps directly to the Vulkan batch/channel shader mode.
// Retain the strict full-transfer admission rule: a hybrid caller only gives
// up CPU cycles when upload + dispatch + readback is at least as fast.
bool TryHybridVulkanDepthwiseConv(
    Backend backend, Tensor& output, const Tensor& input, const Tensor& weights,
    const Tensor* bias, int stride_height, int stride_width, int pad_top,
    int pad_left, bool immutable_parameters, bool relu = false,
    bool swish = false, bool hard_swish = false) noexcept {
  if (backend != Backend::hybrid || !bias || input.shape.size() != 4 ||
      output.shape.size() != 4 || weights.shape.size() != 4 ||
      !output.data.data() || !input.data.data() || !weights.data.data() || !bias->data.data() ||
      output.data.size() < 65536 || pad_top < 0 || pad_left < 0 ||
      (int(relu) + int(swish) + int(hard_swish) > 1)) return false;
  const int batches = static_cast<int>(input.shape[0]);
  const int channels = static_cast<int>(input.shape[1]);
  const int input_height = static_cast<int>(input.shape[2]);
  const int input_width = static_cast<int>(input.shape[3]);
  const int output_height = static_cast<int>(output.shape[2]);
  const int output_width = static_cast<int>(output.shape[3]);
  const int kernel_height = static_cast<int>(weights.shape[2]);
  const int kernel_width = static_cast<int>(weights.shape[3]);
  if (batches <= 0 || channels <= 0 || input_height <= 0 || input_width <= 0 ||
      output_height <= 0 || output_width <= 0 || kernel_height <= 0 || kernel_width <= 0 ||
      stride_height <= 0 || stride_width <= 0 || output.shape[0] != input.shape[0] ||
      output.shape[1] != input.shape[1] || weights.shape[0] != channels ||
      weights.shape[1] != 1 || bias->data.size() != static_cast<std::size_t>(channels) ||
      weights.data.size() != static_cast<std::size_t>(channels) * kernel_height * kernel_width ||
      input.data.size() != static_cast<std::size_t>(batches) * channels * input_height * input_width ||
      output.data.size() != static_cast<std::size_t>(batches) * channels * output_height * output_width) {
    return false;
  }
  // Depthwise geometry contains many independent dimensions. Keep them in a
  // typed cache key instead of XOR-packing overlapping bit ranges, which
  // could accidentally reuse one shape's admission result for another.
  struct AdmissionKey {
    int batches, channels, input_height, input_width, output_height, output_width;
    int kernel_height, kernel_width, stride_height, stride_width, pad_top, pad_left;
    int relu, swish, hard_swish;
    bool operator==(const AdmissionKey&) const noexcept = default;
  };
  struct AdmissionKeyHash {
    std::size_t operator()(const AdmissionKey& value) const noexcept {
      std::size_t hash = 1469598103934665603ull;
      const auto mix = [&hash](int field) {
        hash ^= static_cast<std::uint32_t>(field);
        hash *= 1099511628211ull;
      };
      mix(value.batches); mix(value.channels); mix(value.input_height); mix(value.input_width);
      mix(value.output_height); mix(value.output_width); mix(value.kernel_height); mix(value.kernel_width);
      mix(value.stride_height); mix(value.stride_width); mix(value.pad_top); mix(value.pad_left);
      mix(value.relu); mix(value.swish); mix(value.hard_swish);
      return hash;
    }
  };
  const AdmissionKey shape_key{batches, channels, input_height, input_width, output_height,
                         output_width, kernel_height, kernel_width, stride_height,
                         stride_width, pad_top, pad_left, int(relu), int(swish),
                         int(hard_swish)};
  static std::mutex admission_mutex;
  static std::unordered_map<std::uint64_t, bool> admitted;
  const auto key = WithHybridAdmissionContext(AdmissionKeyHash{}(shape_key));
  bool select_gpu{};
  {
    std::lock_guard lock(admission_mutex);
    const auto found = admitted.find(key);
    if (found != admitted.end()) {
      select_gpu = found->second;
    } else {
      double gpu_ms{}, cpu_ms{};
      select_gpu = VulkanDepthwiseConvBatchNoSlowerThanCpu(
          static_cast<std::size_t>(batches), channels, input_height, input_width,
          output_height, output_width, kernel_height, kernel_width,
          stride_height, stride_width, pad_top, pad_left, &gpu_ms, &cpu_ms,
          immutable_parameters, relu, swish, hard_swish);
      admitted.emplace(key, select_gpu);
    }
  }
  return select_gpu && VulkanDepthwiseConvBatch(
      output.data.data(), input.data.data(), weights.data.data(), bias->data.data(),
      static_cast<std::size_t>(batches), channels, input_height, input_width,
      output_height, output_width, kernel_height, kernel_width, stride_height,
      stride_width, pad_top, pad_left, immutable_parameters, relu, swish,
      hard_swish);
}

bool TryHybridVulkanDepthwisePointwiseConv(
    Backend backend, Tensor& output, const Tensor& input,
    const Tensor& depthwise_weights, const Tensor& depthwise_bias,
    const Tensor& pointwise_weights, const Tensor& pointwise_bias,
    int stride_height, int stride_width, int pad_top, int pad_left,
    bool immutable_parameters, bool approximate_gelu = false) noexcept {
  if (backend != Backend::hybrid || input.shape.size() != 4 || output.shape.size() != 4 ||
      depthwise_weights.shape.size() != 4 || pointwise_weights.shape.size() != 4 ||
      !input.data.data() || !output.data.data() || !depthwise_weights.data.data() ||
      !depthwise_bias.data.data() || !pointwise_weights.data.data() ||
      !pointwise_bias.data.data() || output.data.size() < 65536 ||
      stride_height <= 0 || stride_width <= 0 || pad_top < 0 || pad_left < 0) return false;
  const int batches = static_cast<int>(input.shape[0]);
  const int channels = static_cast<int>(input.shape[1]);
  const int input_height = static_cast<int>(input.shape[2]);
  const int input_width = static_cast<int>(input.shape[3]);
  const int output_channels = static_cast<int>(output.shape[1]);
  const int output_height = static_cast<int>(output.shape[2]);
  const int output_width = static_cast<int>(output.shape[3]);
  const int kernel_height = static_cast<int>(depthwise_weights.shape[2]);
  const int kernel_width = static_cast<int>(depthwise_weights.shape[3]);
  if (batches <= 0 || channels <= 0 || output_channels <= 0 || input_height <= 0 ||
      input_width <= 0 || output_height <= 0 || output_width <= 0 || kernel_height <= 0 ||
      kernel_width <= 0 || output.shape[0] != input.shape[0] ||
      depthwise_weights.shape[0] != channels || depthwise_weights.shape[1] != 1 ||
      pointwise_weights.shape[0] != output_channels || pointwise_weights.shape[1] != channels ||
      pointwise_weights.shape[2] != 1 || pointwise_weights.shape[3] != 1 ||
      depthwise_bias.data.size() != std::size_t(channels) ||
      pointwise_bias.data.size() != std::size_t(output_channels)) return false;
  struct AdmissionKey {
    int batches, channels, output_channels, input_height, input_width;
    int output_height, output_width, kernel_height, kernel_width, stride_height, stride_width;
    int pad_top, pad_left;
    bool approximate_gelu;
    bool operator==(const AdmissionKey&) const noexcept = default;
  };
  struct AdmissionKeyHash {
    std::size_t operator()(const AdmissionKey& key) const noexcept {
      std::size_t hash = 1469598103934665603ull;
      for (const int value : {key.batches, key.channels, key.output_channels,
                              key.input_height, key.input_width, key.output_height,
                              key.output_width, key.kernel_height, key.kernel_width,
                              key.stride_height, key.stride_width, key.pad_top, key.pad_left,
                              static_cast<int>(key.approximate_gelu)}) {
        hash ^= static_cast<std::uint32_t>(value);
        hash *= 1099511628211ull;
      }
      return hash;
    }
  };
  const AdmissionKey shape_key{batches, channels, output_channels, input_height, input_width,
                         output_height, output_width, kernel_height, kernel_width,
                         stride_height, stride_width, pad_top, pad_left, approximate_gelu};
  static std::mutex admission_mutex;
  static std::unordered_map<std::uint64_t, bool> admitted;
  const auto key = WithHybridAdmissionContext(AdmissionKeyHash{}(shape_key));
  bool select_gpu{};
  {
    std::lock_guard lock(admission_mutex);
    const auto found = admitted.find(key);
    if (found != admitted.end()) {
      select_gpu = found->second;
    } else {
      double gpu_ms{}, cpu_ms{};
      select_gpu = VulkanDepthwisePointwiseConvBatchNoSlowerThanCpu(
          static_cast<std::size_t>(batches), channels, output_channels, input_height,
          input_width, output_height, output_width, kernel_height, kernel_width,
          stride_height, stride_width, pad_top, pad_left, &gpu_ms, &cpu_ms,
          immutable_parameters, approximate_gelu);
      admitted.emplace(key, select_gpu);
    }
  }
  return select_gpu && VulkanDepthwisePointwiseConvBatch(
      output.data.data(), input.data.data(), depthwise_weights.data.data(),
      depthwise_bias.data.data(), pointwise_weights.data.data(), pointwise_bias.data.data(),
      static_cast<std::size_t>(batches), channels, output_channels, input_height,
      input_width, output_height, output_width, kernel_height, kernel_width,
      stride_height, stride_width, pad_top, pad_left, immutable_parameters,
      approximate_gelu);
}

// Detector feature extraction still contains ordinary 3x3 / 2x2 projections
// after pointwise and depthwise paths have been removed.  Dispatch those
// shapes as a real NCHW batch when the fully synchronous Vulkan segment has
// demonstrated that it beats the CPU SIMD batch kernel on this device.
bool TryHybridVulkanConv2d(
    Backend backend, Tensor& output, const Tensor& input, const Tensor& weights,
    const Tensor* bias, int stride_height, int stride_width, int pad_top,
    int pad_left, bool immutable_parameters, bool relu, bool swish = false,
    bool sigmoid = false, bool hard_swish = false) noexcept {
  if (backend != Backend::hybrid || !bias || input.shape.size() != 4 ||
      output.shape.size() != 4 || weights.shape.size() != 4 ||
      !output.data.data() || !input.data.data() || !weights.data.data() || !bias->data.data() ||
      output.data.size() < 65536 || stride_height <= 0 || stride_height != stride_width ||
      pad_top < 0 || pad_left < 0 ||
      (int(relu) + int(swish) + int(sigmoid) + int(hard_swish) > 1)) return false;
  const int batches = static_cast<int>(input.shape[0]);
  const int input_channels = static_cast<int>(input.shape[1]);
  const int input_height = static_cast<int>(input.shape[2]);
  const int input_width = static_cast<int>(input.shape[3]);
  const int output_channels = static_cast<int>(output.shape[1]);
  const int output_height = static_cast<int>(output.shape[2]);
  const int output_width = static_cast<int>(output.shape[3]);
  const int kernel_height = static_cast<int>(weights.shape[2]);
  const int kernel_width = static_cast<int>(weights.shape[3]);
  if (batches <= 0 || input_channels <= 0 || output_channels <= 0 ||
      input_height <= 0 || input_width <= 0 || output_height <= 0 || output_width <= 0 ||
      kernel_height <= 0 || kernel_width <= 0 || weights.shape[0] != output_channels ||
      weights.shape[1] != input_channels || bias->data.size() != std::size_t(output_channels)) return false;
  struct AdmissionKey {
    int batches, input_channels, output_channels, input_height, input_width;
    int output_height, output_width, kernel_height, kernel_width;
    int stride, pad_top, pad_left, relu, swish, sigmoid, hard_swish;
    bool operator==(const AdmissionKey&) const = default;
  };
  struct AdmissionKeyHash {
    std::size_t operator()(const AdmissionKey& key) const noexcept {
      std::size_t hash{};
      const int values[]{key.batches, key.input_channels, key.output_channels,
          key.input_height, key.input_width, key.output_height, key.output_width,
          key.kernel_height, key.kernel_width, key.stride, key.pad_top, key.pad_left, key.relu,
          key.swish, key.sigmoid, key.hard_swish};
      for (const int value : values) hash = (hash * 1315423911u) ^ static_cast<std::size_t>(value);
      return hash;
    }
  };
  const AdmissionKey shape_key{batches, input_channels, output_channels, input_height, input_width,
                         output_height, output_width, kernel_height, kernel_width,
                         stride_height, pad_top, pad_left, relu, swish, sigmoid, hard_swish};
  static std::mutex admission_mutex;
  static std::unordered_map<std::uint64_t, bool> admitted;
  const auto key = WithHybridAdmissionContext(AdmissionKeyHash{}(shape_key));
  bool select_gpu{};
  {
    std::lock_guard lock(admission_mutex);
    const auto found = admitted.find(key);
    if (found != admitted.end()) {
      select_gpu = found->second;
    } else {
      double gpu_ms{}, cpu_ms{};
      select_gpu = VulkanConv2dBatchNoSlowerThanCpu(
          static_cast<std::size_t>(batches), input_channels, output_channels,
          input_height, input_width, output_height, output_width,
          kernel_height, kernel_width, stride_height, stride_width, pad_top, pad_left,
          &gpu_ms, &cpu_ms, immutable_parameters, relu, swish, sigmoid, hard_swish);
      admitted.emplace(key, select_gpu);
    }
  }
  return select_gpu && VulkanConv2dBatch(
      output.data.data(), input.data.data(), weights.data.data(), bias->data.data(),
      static_cast<std::size_t>(batches), input_channels, output_channels,
      input_height, input_width, output_height, output_width, kernel_height, kernel_width,
      stride_height, stride_width, pad_top, pad_left, immutable_parameters, relu, swish,
      sigmoid, hard_swish);
}

bool TryHybridVulkanConcatConv(
    Backend backend, Tensor& output, const float* const* sources,
    const int* source_channels, int source_count, const Tensor& weights,
    const Tensor* bias, int input_height, int input_width, int output_height,
    int output_width, int kernel_height, int kernel_width, int stride_height,
    int stride_width, int pad_top, int pad_left, bool relu, bool pool_first,
    bool immutable_parameters) noexcept {
  if (backend != Backend::hybrid || !bias || !sources || !source_channels ||
      source_count < 2 || source_count > 4 || !output.data.data() ||
      !weights.data.data() || !bias->data.data() ||
      output.data.size() < 16384 || stride_height <= 0 ||
      stride_height != stride_width || pad_top < 0 || pad_left < 0) return false;
  // Four-source FPN Concat.2: spatial C=64 can admit against a cold CPU
  // microbench and then emit 0 boxes (heatmap drift). Keep it opt-in.
  // The default is the parallel AVX-512 split (one 16-ch 3x3 per source).
  if (!pool_first && source_count >= 4 &&
      std::getenv("PPOCR_ENABLE_VULKAN_CONCAT2") == nullptr) return false;
  for (int i = 0; i < source_count; ++i) {
    if (!sources[i] || source_channels[i] <= 0) return false;
  }
  struct AdmissionKey {
    int source_count, c0, c1, c2, c3, output_channels, input_height, input_width;
    int output_height, output_width, kernel_height, stride, pad_top, pad_left, relu, pool;
    bool operator==(const AdmissionKey&) const = default;
  };
  struct AdmissionKeyHash {
    std::size_t operator()(const AdmissionKey& key) const noexcept {
      std::size_t hash{};
      const int values[]{key.source_count, key.c0, key.c1, key.c2, key.c3,
          key.output_channels, key.input_height, key.input_width, key.output_height,
          key.output_width, key.kernel_height, key.stride, key.pad_top, key.pad_left,
          key.relu, key.pool};
      for (const int value : values) hash = (hash * 1315423911u) ^ static_cast<std::size_t>(value);
      return hash;
    }
  };
  const AdmissionKey shape_key{source_count, source_channels[0], source_channels[1],
      source_count > 2 ? source_channels[2] : 0, source_count > 3 ? source_channels[3] : 0,
      int(weights.shape[0]), input_height, input_width, output_height, output_width,
      kernel_height, stride_height, pad_top, pad_left, relu ? 1 : 0, pool_first ? 1 : 0};
  static std::mutex admission_mutex;
  static std::unordered_map<std::uint64_t, bool> admitted;
  const auto key = WithHybridAdmissionContext(AdmissionKeyHash{}(shape_key));
  bool select_gpu{};
  {
    std::lock_guard lock(admission_mutex);
    const auto found = admitted.find(key);
    if (found != admitted.end()) {
      select_gpu = found->second;
    } else {
      double gpu_ms{}, cpu_ms{};
      select_gpu = VulkanConcatConvBatchNoSlowerThanCpu(
          source_channels, source_count, int(weights.shape[0]), input_height, input_width,
          output_height, output_width, kernel_height, kernel_width, stride_height,
          stride_width, pad_top, pad_left, relu, pool_first, &gpu_ms, &cpu_ms,
          immutable_parameters);
      admitted.emplace(key, select_gpu);
    }
  }
  return select_gpu && VulkanConcatConvBatch(
      output.data.data(), sources, source_channels, source_count, weights.data.data(),
      bias->data.data(), 1, int(weights.shape[0]), input_height, input_width,
      output_height, output_width, kernel_height, kernel_width, stride_height,
      stride_width, pad_top, pad_left, relu, pool_first, immutable_parameters);
}

// The detector FPN also has two unpadded 2x2/stride-2 ConvTranspose layers.
// Their tiles never overlap, so the Vulkan kernel can form each output directly
// from one source position.  Retain the same complete-transfer admission rule
// as the other hybrid kernels: a partial graph must never trade CPU SIMD for a
// slower upload/dispatch/readback sequence.
bool TryHybridVulkanConvTranspose2x2(
    Backend backend, Tensor& output, const Tensor& input, const Tensor& weights,
    const Tensor* bias, bool immutable_parameters) noexcept {
  if (backend != Backend::hybrid || !bias || input.shape.size() != 4 ||
      output.shape.size() != 4 || weights.shape.size() != 4 ||
      !output.data.data() || !input.data.data() || !weights.data.data() || !bias->data.data() ||
      output.data.size() < 65536) return false;
  const int batches = static_cast<int>(input.shape[0]);
  const int input_channels = static_cast<int>(input.shape[1]);
  const int output_channels = static_cast<int>(output.shape[1]);
  const int input_height = static_cast<int>(input.shape[2]);
  const int input_width = static_cast<int>(input.shape[3]);
  if (batches <= 0 || input_channels <= 0 || output_channels <= 0 ||
      input_height <= 0 || input_width <= 0 || output.shape[0] != input.shape[0] ||
      output.shape[2] != input.shape[2] * 2 || output.shape[3] != input.shape[3] * 2 ||
      weights.shape[0] != input_channels || weights.shape[1] != output_channels ||
      weights.shape[2] != 2 || weights.shape[3] != 2 ||
      input.data.size() != static_cast<std::size_t>(batches) * input_channels * input_height * input_width ||
      output.data.size() != static_cast<std::size_t>(batches) * output_channels * input_height * input_width * 4 ||
      weights.data.size() != static_cast<std::size_t>(input_channels) * output_channels * 4 ||
      bias->data.size() != static_cast<std::size_t>(output_channels)) return false;
  struct AdmissionKey {
    int batches, input_channels, output_channels, input_height, input_width;
    bool operator==(const AdmissionKey&) const noexcept = default;
  };
  struct AdmissionKeyHash {
    std::size_t operator()(const AdmissionKey& value) const noexcept {
      std::size_t hash = 1469598103934665603ull;
      for (const int field : {value.batches, value.input_channels, value.output_channels,
                              value.input_height, value.input_width}) {
        hash ^= static_cast<std::uint32_t>(field);
        hash *= 1099511628211ull;
      }
      return hash;
    }
  };
  static std::mutex admission_mutex;
  static std::unordered_map<std::uint64_t, bool> admitted;
  const AdmissionKey shape_key{batches, input_channels, output_channels, input_height, input_width};
  const auto key = WithHybridAdmissionContext(AdmissionKeyHash{}(shape_key));
  bool select_gpu{};
  {
    std::lock_guard lock(admission_mutex);
    const auto found = admitted.find(key);
    if (found != admitted.end()) {
      select_gpu = found->second;
    } else {
      double gpu_ms{}, cpu_ms{};
      select_gpu = VulkanConvTranspose2x2BatchNoSlowerThanCpu(
          static_cast<std::size_t>(batches), input_channels, output_channels,
          input_height, input_width, &gpu_ms, &cpu_ms, immutable_parameters);
      admitted.emplace(key, select_gpu);
    }
  }
  return select_gpu && VulkanConvTranspose2x2Batch(
      output.data.data(), input.data.data(), weights.data.data(), bias->data.data(),
      static_cast<std::size_t>(batches), input_channels, output_channels,
      input_height, input_width, immutable_parameters);
}

bool TryHybridVulkanConvTranspose2x2Add(
    Backend backend, Tensor& output, const Tensor& input, const Tensor& weights,
    const Tensor* bias, const Tensor& residual, bool immutable_parameters) noexcept {
  if (backend != Backend::hybrid || !bias || input.shape.size() != 4 ||
      output.shape.size() != 4 || residual.shape != output.shape || weights.shape.size() != 4 ||
      !output.data.data() || !input.data.data() || !weights.data.data() || !bias->data.data() ||
      !residual.data.data() || output.data.size() < 65536) return false;
  const int batches = static_cast<int>(input.shape[0]);
  const int input_channels = static_cast<int>(input.shape[1]);
  const int output_channels = static_cast<int>(output.shape[1]);
  const int input_height = static_cast<int>(input.shape[2]);
  const int input_width = static_cast<int>(input.shape[3]);
  if (batches <= 0 || input_channels <= 0 || output_channels <= 0 || input_height <= 0 ||
      input_width <= 0 || output.shape[0] != input.shape[0] ||
      output.shape[2] != input.shape[2] * 2 || output.shape[3] != input.shape[3] * 2 ||
      weights.shape[0] != input_channels || weights.shape[1] != output_channels ||
      weights.shape[2] != 2 || weights.shape[3] != 2 ||
      input.data.size() != static_cast<std::size_t>(batches) * input_channels * input_height * input_width ||
      output.data.size() != static_cast<std::size_t>(batches) * output_channels * input_height * input_width * 4 ||
      residual.data.size() != output.data.size() ||
      weights.data.size() != static_cast<std::size_t>(input_channels) * output_channels * 4 ||
      bias->data.size() != static_cast<std::size_t>(output_channels)) return false;
  struct AdmissionKey {
    int batches, input_channels, output_channels, input_height, input_width;
    bool operator==(const AdmissionKey&) const noexcept = default;
  };
  struct AdmissionKeyHash {
    std::size_t operator()(const AdmissionKey& value) const noexcept {
      std::size_t hash = 1469598103934665603ull;
      for (const int field : {value.batches, value.input_channels, value.output_channels,
                              value.input_height, value.input_width}) {
        hash ^= static_cast<std::uint32_t>(field);
        hash *= 1099511628211ull;
      }
      return hash;
    }
  };
  static std::mutex admission_mutex;
  static std::unordered_map<std::uint64_t, bool> admitted;
  const AdmissionKey shape_key{batches, input_channels, output_channels, input_height, input_width};
  const auto key = WithHybridAdmissionContext(AdmissionKeyHash{}(shape_key));
  bool select_gpu{};
  {
    std::lock_guard lock(admission_mutex);
    const auto found = admitted.find(key);
    if (found != admitted.end()) select_gpu = found->second;
    else {
      double gpu_ms{}, cpu_ms{};
      select_gpu = VulkanConvTranspose2x2AddBatchNoSlowerThanCpu(
          static_cast<std::size_t>(batches), input_channels, output_channels,
          input_height, input_width, &gpu_ms, &cpu_ms, immutable_parameters);
      admitted.emplace(key, select_gpu);
    }
  }
  return select_gpu && VulkanConvTranspose2x2AddBatch(
      output.data.data(), input.data.data(), weights.data.data(), bias->data.data(),
      residual.data.data(), static_cast<std::size_t>(batches), input_channels,
      output_channels, input_height, input_width, immutable_parameters);
}

Tensor Conv(const Node& n, const std::vector<const Tensor*>& x, bool relu = false,
            bool sigmoid = false, bool hard_sigmoid = false,
            float hard_sigmoid_alpha = .2F, float hard_sigmoid_beta = .5F,
            bool* relu_applied = nullptr, Backend backend = Backend::cpu_only,
            bool immutable_parameters = false, bool swish = false,
            bool hard_swish = false, bool gelu = false) {
  const auto& a=*x[0]; const auto& w=*x[1]; if(a.shape.size()!=4||w.shape.size()!=4)Fail("Conv needs NCHW");
  const int N=int(a.shape[0]), C=int(a.shape[1]), H=int(a.shape[2]), W=int(a.shape[3]), M=int(w.shape[0]), Cg=int(w.shape[1]), KH=int(w.shape[2]), KW=int(w.shape[3]);
  auto st=AttrInts(n,"strides",{1,1}), pd=AttrInts(n,"pads",{0,0,0,0}), dl=AttrInts(n,"dilations",{1,1}); const int group=int(AttrInt(n,"group",1));
  const auto auto_pad=AttrStr(n,"auto_pad","NOTSET");
  int OH{}, OW{};
  if (auto_pad=="SAME_UPPER" || auto_pad=="SAME_LOWER") {
    OH=(H+int(st[0])-1)/int(st[0]); OW=(W+int(st[1])-1)/int(st[1]);
    const int ph=std::max(0,(OH-1)*int(st[0])+int(dl[0])*(KH-1)+1-H), pw=std::max(0,(OW-1)*int(st[1])+int(dl[1])*(KW-1)+1-W);
    pd[0]=auto_pad=="SAME_LOWER"?(ph+1)/2:ph/2; pd[2]=ph-pd[0]; pd[1]=auto_pad=="SAME_LOWER"?(pw+1)/2:pw/2; pd[3]=pw-pd[1];
  } else {
    OH=(H+int(pd[0])+int(pd[2])-int(dl[0])*(KH-1)-1)/int(st[0])+1; OW=(W+int(pd[1])+int(pd[3])-int(dl[1])*(KW-1)-1)/int(st[1])+1;
  }
  if(OH<=0||OW<=0||C!=Cg*group||M%group)Fail("invalid Conv parameters");
  if (int(relu) + int(sigmoid) + int(hard_sigmoid) + int(swish) + int(hard_swish) +
      int(gelu) > 1) {
    Fail("incompatible fused Conv activations");
  }
  if (relu_applied) *relu_applied = false;
  const auto apply_activation = [&](Tensor& output) {
    if (sigmoid) kernels::Sigmoid(output.data.data(), output.data.data(), output.data.size());
    else if (hard_sigmoid) kernels::HardSigmoid(output.data.data(), output.data.data(),
                                                 output.data.size(), hard_sigmoid_alpha,
                                                 hard_sigmoid_beta);
    else if (swish) kernels::Swish(output.data.data(), output.data.data(), output.data.size());
    else if (hard_swish) kernels::HardSwish(output.data.data(), output.data.data(),
                                             output.data.size());
    else if (gelu) {
      const auto logical = output.shape.empty() ? output.data.size() :
          output.data.size() / std::size_t(output.shape[0]);
      kernels::Gelu(output.data.data(), output.data.data(), output.data.size(), logical);
    }
  };
  Tensor o; o.shape={N,M,OH,OW}; o.data=PooledActivation(std::size_t(N)*M*OH*OW); const float* bias=x.size()>2&&x[2]?x[2]->data.data():nullptr; const int mg=M/group;
  // The PP-OCRv6 backbone/FPN contains many NCHW pointwise convolutions.
  // For the no-padding unit-stride form, each input channel is a contiguous
  // plane, so use vector AXPY instead of scalar per-pixel address arithmetic.
  if (KH==1 && KW==1 && st[0]==1 && st[1]==1 && dl[0]==1 && dl[1]==1 &&
      pd[0]==0 && pd[1]==0 && pd[2]==0 && pd[3]==0) {
    const std::size_t plane=std::size_t(H)*W;
    if (group == 1 && bias &&
        TryHybridVulkanPointwiseConv(backend, o, a, w, x[2], relu,
                                      immutable_parameters, swish, sigmoid, hard_sigmoid,
                                      hard_sigmoid_alpha, hard_sigmoid_beta, hard_swish)) {
      if (relu_applied && relu) *relu_applied = true;
      // Vulkan modes 7, 10 and 11 already wrote their final activation.
      // Other post-convolution activations deliberately remain on the CPU
      // path until they receive their own end-to-end admission measurement.
      if (swish || sigmoid || hard_sigmoid || hard_swish) return o;
      apply_activation(o);
      return o;
    }
    if (group == 1) {
      if (std::getenv("PPOCR_DUMP_SHAPES") != nullptr) {
        std::cerr << "pw " << (n.name.empty() ? n.op : n.name) << " "
                  << N << "x" << C << "x" << H << "x" << W << " -> " << M << "x" << H << "x" << W
                  << '\n';
      }
      if (relu) {
        kernels::PointwiseConvReluBatch(o.data.data(), a.data.data(), w.data.data(), bias,
                                        N, M, C, plane);
      } else {
        kernels::PointwiseConvBatch(o.data.data(), a.data.data(), w.data.data(), bias,
                                    N, M, C, plane);
      }
      apply_activation(o);
      return o;
    }
    for (int ni=0; ni<N; ++ni) for (int g=0; g<group; ++g) {
      if (relu) {
        kernels::PointwiseConvRelu(o.data.data()+(std::size_t(ni)*M+g*mg)*plane,
                                   a.data.data()+(std::size_t(ni)*C+g*Cg)*plane,
                                   w.data.data()+std::size_t(g*mg)*Cg,
                                   bias ? bias+g*mg : nullptr, mg, Cg, plane);
      } else {
        kernels::PointwiseConv(o.data.data()+(std::size_t(ni)*M+g*mg)*plane,
                               a.data.data()+(std::size_t(ni)*C+g*Cg)*plane,
                               w.data.data()+std::size_t(g*mg)*Cg,
                               bias ? bias+g*mg : nullptr, mg, Cg, plane);
      }
    }
    apply_activation(o);
    return o;
  }
  // MobileNet blocks use true depthwise convolutions.  Their channels are
  // entirely independent, so route them to the channel-parallel NCHW kernel
  // instead of the generic grouped-convolution address-calculation loop.
  if (group == C && Cg == 1 && M == C && dl[0] == 1 && dl[1] == 1) {
    if (bias && TryHybridVulkanDepthwiseConv(
            backend, o, a, w, x[2], int(st[0]), int(st[1]), int(pd[0]),
            int(pd[1]), immutable_parameters, relu, swish, hard_swish)) {
      if (relu_applied && relu) *relu_applied = true;
      if (!swish && !hard_swish) apply_activation(o);
      return o;
    }
    if (std::getenv("PPOCR_DUMP_SHAPES") != nullptr) {
      std::cerr << "dw " << (n.name.empty() ? n.op : n.name) << " "
                << N << "x" << C << "x" << H << "x" << W << " -> " << M << "x" << OH << "x" << OW
                << " k=" << KH << " s=" << st[0] << '\n';
    }
    kernels::DepthwiseConvBatch(o.data.data(), a.data.data(), w.data.data(), bias,
                                N, C, H, W, OH, OW, KH, KW, int(st[0]), int(st[1]),
                                int(pd[0]), int(pd[1]));
    apply_activation(o);
    return o;
  }
  if (group == 1 && dl[0] == 1 && dl[1] == 1) {
    // Keep a deployment-time A/B escape hatch for the ordinary 3x3 stride-2
    // writeback fusion. It is intentionally read only while constructing a
    // node result, never in the inner SIMD loop.
    const bool fuse_relu = relu && std::getenv("PPOCR_DISABLE_FUSED_CONV_RELU") == nullptr;
    if (std::getenv("PPOCR_DUMP_SHAPES") != nullptr) {
      std::cerr << "conv " << (n.name.empty() ? n.op : n.name) << " "
                << (relu ? "relu " : "") << N << "x" << C << "x" << H << "x" << W
                << " -> " << M << "x" << OH << "x" << OW << " k=" << KH << " s=" << st[0]
                << " pad=" << pd[0] << '\n';
    }
    if (bias && TryHybridVulkanConv2d(
            backend, o, a, w, x[2], int(st[0]), int(st[1]), int(pd[0]), int(pd[1]),
            immutable_parameters, fuse_relu, swish, sigmoid, hard_swish)) {
      if (relu_applied && fuse_relu) *relu_applied = true;
      // The selected shader has already produced its fused nonlinear output;
      // avoid a second CPU traversal after the readback.
      if (swish || sigmoid || hard_swish) return o;
      apply_activation(o);
      return o;
    }
    if (N > 1 && std::getenv("PPOCR_DISABLE_CONV_BATCH") == nullptr) {
      kernels::Conv2dBatch(o.data.data(), a.data.data(), w.data.data(), bias, N, C, M,
                           H, W, OH, OW, KH, KW, int(st[0]), int(st[1]), int(pd[0]),
                           int(pd[1]), fuse_relu);
    } else {
      const std::size_t input_batch = std::size_t(C) * H * W;
      const std::size_t output_batch = std::size_t(M) * OH * OW;
      for (int ni = 0; ni < N; ++ni) {
        kernels::Conv2d(o.data.data() + std::size_t(ni) * output_batch,
                        a.data.data() + std::size_t(ni) * input_batch,
                        w.data.data(), bias, C, M, H, W, OH, OW, KH, KW,
                        int(st[0]), int(st[1]), int(pd[0]), int(pd[1]), fuse_relu);
      }
    }
    if (relu_applied && fuse_relu &&
        ((KH == 2 && KW == 2 && st[0] == 1 && st[1] == 1 &&
          pd[0] == 0 && pd[1] == 0 && pd[2] == 0 && pd[3] == 0) ||
         (KH == 3 && KW == 3 &&
          ((st[0] == 2 && st[1] == 2) || (st[0] == 1 && st[1] == 1 && M >= 4))))) {
      *relu_applied = true;
    }
    apply_activation(o);
    return o;
  }
  for(int ni=0;ni<N;++ni)for(int mo=0;mo<M;++mo)for(int oy=0;oy<OH;++oy)for(int ox=0;ox<OW;++ox){float sum=bias?bias[mo]:0;const int g=mo/mg;for(int ci=0;ci<Cg;++ci)for(int ky=0;ky<KH;++ky)for(int kx=0;kx<KW;++kx){int iy=oy*int(st[0])-int(pd[0])+ky*int(dl[0]),ix=ox*int(st[1])-int(pd[1])+kx*int(dl[1]);if(iy>=0&&iy<H&&ix>=0&&ix<W)sum+=a.data[((ni*C+g*Cg+ci)*H+iy)*W+ix]*w.data[((mo*Cg+ci)*KH+ky)*KW+kx];}o.data[((ni*M+mo)*OH+oy)*OW+ox]=sum;}
  apply_activation(o);
  return o;
}

enum class DwPwActivation { none, gelu, hard_swish, relu };

Tensor DepthwisePointwiseConv(const Node& n, const std::vector<const Tensor*>& x,
                              Backend backend = Backend::cpu_only,
                              bool immutable_parameters = false,
                              DwPwActivation activation = DwPwActivation::none) {
  if (x.size() != 5 || !x[0] || !x[1] || !x[2] || !x[3] || !x[4]) {
    Fail("FusedDepthwisePointwiseConv inputs");
  }
  const auto& input = *x[0];
  const auto& depthwise_weights = *x[1];
  const auto& depthwise_bias = *x[2];
  const auto& pointwise_weights = *x[3];
  const auto& pointwise_bias = *x[4];
  if (input.shape.size() != 4 || depthwise_weights.shape.size() != 4 ||
      pointwise_weights.shape.size() != 4 || depthwise_weights.shape[1] != 1 ||
      pointwise_weights.shape[1] != input.shape[1] || pointwise_weights.shape[2] != 1 ||
      pointwise_weights.shape[3] != 1) {
    Fail("invalid FusedDepthwisePointwiseConv shape");
  }
  const int batches = static_cast<int>(input.shape[0]);
  const int channels = static_cast<int>(input.shape[1]);
  const int input_height = static_cast<int>(input.shape[2]);
  const int input_width = static_cast<int>(input.shape[3]);
  const int output_channels = static_cast<int>(pointwise_weights.shape[0]);
  const int kernel_height = static_cast<int>(depthwise_weights.shape[2]);
  const int kernel_width = static_cast<int>(depthwise_weights.shape[3]);
  const auto strides = AttrInts(n, "strides", {1, 1});
  const auto pads = AttrInts(n, "pads", {0, 0, 0, 0});
  if (batches <= 0 || channels <= 0 || output_channels <= 0 || kernel_height <= 0 ||
      kernel_width <= 0 || strides.size() != 2 || pads.size() != 4 ||
      strides[0] <= 0 || strides[1] <= 0 || depthwise_weights.shape[0] != channels ||
      depthwise_bias.data.size() != std::size_t(channels) ||
      pointwise_bias.data.size() != std::size_t(output_channels)) {
    Fail("invalid FusedDepthwisePointwiseConv parameters");
  }
  const int output_height = (input_height + static_cast<int>(pads[0] + pads[2]) -
      kernel_height) / static_cast<int>(strides[0]) + 1;
  const int output_width = (input_width + static_cast<int>(pads[1] + pads[3]) -
      kernel_width) / static_cast<int>(strides[1]) + 1;
  if (output_height <= 0 || output_width <= 0) Fail("invalid FusedDepthwisePointwiseConv output");
  Tensor output{{batches, output_channels, output_height, output_width},
                PooledActivation(std::size_t(batches) * output_channels *
                                 output_height * output_width)};
  const bool gelu = activation == DwPwActivation::gelu;
  const bool approximate_gelu = gelu && kernels::ApproximateGeluSelected(
      output.data.size() / std::size_t(batches));
  if (TryHybridVulkanDepthwisePointwiseConv(
          backend, output, input, depthwise_weights, depthwise_bias,
          pointwise_weights, pointwise_bias, static_cast<int>(strides[0]),
          static_cast<int>(strides[1]), static_cast<int>(pads[0]),
          static_cast<int>(pads[1]), immutable_parameters, approximate_gelu)) {
    if (gelu && !approximate_gelu) kernels::Gelu(output.data.data(), output.data.data(), output.data.size(),
                            output.data.size() / std::size_t(batches));
    else if (activation == DwPwActivation::hard_swish)
      kernels::HardSwish(output.data.data(), output.data.data(), output.data.size());
    else if (activation == DwPwActivation::relu)
      kernels::Relu(output.data.data(), output.data.data(), output.data.size());
    return output;
  }
  int fused_activation = 0;
  if (activation == DwPwActivation::relu) fused_activation = 1;
  else if (activation == DwPwActivation::hard_swish) fused_activation = 2;
  else if (gelu) fused_activation = 3;
  static const bool enable_dwpw_fused =
      std::getenv("PPOCR_ENABLE_AVX512_DWPW_FUSED") != nullptr;
  // 5x5 fused is the Conv.78 default (separate DISABLE). 3x3 fused on rec
  // 12x76-class maps is serial (no extra rec ParallelFor). Same-host 8-run
  // rec default missed versus two-pass DW+PW; keep ENABLE-only like det.
  // `PPOCR_ENABLE_AVX512_REC_DWPW3` turns rec 3x3 fused back on.
  static const bool rec_dwpw3 =
      std::getenv("PPOCR_ENABLE_AVX512_REC_DWPW3") != nullptr &&
      std::getenv("PPOCR_DISABLE_AVX512_REC_DWPW3") == nullptr;
  const bool try_dwpw_fused = enable_dwpw_fused ||
      (kernel_height == 5 && kernel_width == 5) ||
      (rec_dwpw3 && kernel_height == 3 && kernel_width == 3 &&
       output_height < 32 && output_width >= 32);
  if (try_dwpw_fused && output_height == input_height &&
      output_width == input_width &&
      kernels::DepthwisePointwiseConvFused(
          output.data.data(), input.data.data(), depthwise_weights.data.data(),
          depthwise_bias.data.data(), pointwise_weights.data.data(),
          pointwise_bias.data.data(), batches, channels, output_channels,
          output_height, output_width, kernel_height, kernel_width,
          static_cast<int>(strides[0]), static_cast<int>(strides[1]),
          static_cast<int>(pads[0]), static_cast<int>(pads[1]), fused_activation)) {
    return output;
  }
  const std::size_t intermediate_count = std::size_t(batches) * channels *
      output_height * output_width;
  std::vector<float> intermediate = PooledActivation(intermediate_count);
  kernels::DepthwiseConvBatch(intermediate.data(), input.data.data(),
                              depthwise_weights.data.data(), depthwise_bias.data.data(),
                              batches, channels, input_height, input_width,
                              output_height, output_width, kernel_height, kernel_width,
                              static_cast<int>(strides[0]), static_cast<int>(strides[1]),
                              static_cast<int>(pads[0]), static_cast<int>(pads[1]));
  const auto plane = std::size_t(output_height) * output_width;
  if (activation == DwPwActivation::hard_swish) {
    kernels::PointwiseConvHardSwishBatch(output.data.data(), intermediate.data(),
                                         pointwise_weights.data.data(), pointwise_bias.data.data(),
                                         batches, output_channels, channels, plane);
  } else if (activation == DwPwActivation::relu) {
    kernels::PointwiseConvReluBatch(output.data.data(), intermediate.data(),
                                    pointwise_weights.data.data(), pointwise_bias.data.data(),
                                    batches, output_channels, channels, plane);
  } else {
    kernels::PointwiseConvBatch(output.data.data(), intermediate.data(),
                                pointwise_weights.data.data(), pointwise_bias.data.data(),
                                batches, output_channels, channels, plane);
    if (gelu) kernels::Gelu(output.data.data(), output.data.data(), output.data.size(),
                            output.data.size() / std::size_t(batches));
  }
  RecycleActivation(intermediate);
  return output;
}

Tensor PointwiseDepthwiseConvOp(const Node& n, const std::vector<const Tensor*>& x,
                                bool relu, Backend backend = Backend::cpu_only,
                                bool immutable_parameters = false) {
  if (x.size() != 5 || !x[0] || !x[1] || !x[2] || !x[3] || !x[4]) {
    Fail("FusedPointwiseDepthwiseConv inputs");
  }
  (void)immutable_parameters;
  const auto& input = *x[0];
  const auto& pointwise_weights = *x[1];
  const auto& pointwise_bias = *x[2];
  const auto& depthwise_weights = *x[3];
  const auto& depthwise_bias = *x[4];
  if (input.shape.size() != 4 || pointwise_weights.shape.size() != 4 ||
      depthwise_weights.shape.size() != 4 || pointwise_weights.shape[2] != 1 ||
      pointwise_weights.shape[3] != 1 || depthwise_weights.shape[1] != 1 ||
      depthwise_weights.shape[2] != 3 || depthwise_weights.shape[3] != 3 ||
      pointwise_weights.shape[1] != input.shape[1] ||
      depthwise_weights.shape[0] != pointwise_weights.shape[0]) {
    Fail("invalid FusedPointwiseDepthwiseConv shape");
  }
  const int batches = static_cast<int>(input.shape[0]);
  const int input_channels = static_cast<int>(input.shape[1]);
  const int height = static_cast<int>(input.shape[2]);
  const int width = static_cast<int>(input.shape[3]);
  const int output_channels = static_cast<int>(pointwise_weights.shape[0]);
  if (batches <= 0 || input_channels <= 0 || output_channels <= 0 || height <= 0 ||
      width <= 0 || pointwise_bias.data.size() != std::size_t(output_channels) ||
      depthwise_bias.data.size() != std::size_t(output_channels)) {
    Fail("invalid FusedPointwiseDepthwiseConv parameters");
  }
  Tensor output{{batches, output_channels, height, width},
                PooledActivation(std::size_t(batches) * output_channels * height * width)};
  if (std::getenv("PPOCR_DUMP_SHAPES") != nullptr) {
    std::cerr << "pwdw" << (relu ? "_relu" : "") << " " << input.shape[0] << "x"
              << input_channels << "x" << height << "x" << width << " -> "
              << output_channels << "x" << height << "x" << width << '\n';
  }
  if (kernels::PointwiseDepthwiseConvFused(
          output.data.data(), input.data.data(), pointwise_weights.data.data(),
          pointwise_bias.data.data(), depthwise_weights.data.data(),
          depthwise_bias.data.data(), batches, input_channels, output_channels, height,
          width, relu)) {
    return output;
  }
  const auto plane = std::size_t(height) * width;
  std::vector<float> intermediate =
      PooledActivation(std::size_t(batches) * output_channels * plane);
  if (relu) {
    kernels::PointwiseConvReluBatch(intermediate.data(), input.data.data(),
                                    pointwise_weights.data.data(),
                                    pointwise_bias.data.data(), batches, output_channels,
                                    input_channels, plane);
  } else {
    kernels::PointwiseConvBatch(intermediate.data(), input.data.data(),
                                pointwise_weights.data.data(), pointwise_bias.data.data(),
                                batches, output_channels, input_channels, plane);
  }
  kernels::DepthwiseConvBatch(output.data.data(), intermediate.data(),
                              depthwise_weights.data.data(), depthwise_bias.data.data(),
                              batches, output_channels, height, width, height, width, 3, 3,
                              1, 1, 1, 1);
  RecycleActivation(intermediate);
  (void)backend;
  (void)n;
  return output;
}

Tensor DepthwiseExpandGeluProjectAddOp(const Node& n,
                                       const std::vector<const Tensor*>& x) {
  if (x.size() != 7 || !x[0] || !x[1] || !x[2] || !x[3] || !x[5]) {
    Fail("FusedDepthwiseExpandGeluProjectAdd inputs");
  }
  const auto& input = *x[0];
  const auto& dw_w = *x[1];
  const auto& dw_b = *x[2];
  const auto& expand_w = *x[3];
  const auto& project_w = *x[5];
  if (input.shape.size() != 4 || dw_w.shape.size() != 4 || expand_w.shape.size() != 4 ||
      project_w.shape.size() != 4 || dw_w.shape[1] != 1 || expand_w.shape[2] != 1 ||
      expand_w.shape[3] != 1 || project_w.shape[2] != 1 || project_w.shape[3] != 1) {
    Fail("invalid FusedDepthwiseExpandGeluProjectAdd shape");
  }
  const int batches = int(input.shape[0]);
  const int channels = int(input.shape[1]);
  const int height = int(input.shape[2]);
  const int width = int(input.shape[3]);
  const int hidden = int(expand_w.shape[0]);
  const auto strides = AttrInts(n, "strides", {1, 1});
  const auto pads = AttrInts(n, "pads", {0, 0, 0, 0});
  if (batches <= 0 || channels <= 0 || hidden <= 0 || dw_w.shape[0] != channels ||
      expand_w.shape[1] != channels || project_w.shape[0] != channels ||
      project_w.shape[1] != hidden || dw_b.data.size() != std::size_t(channels) ||
      strides.size() != 2 || pads.size() != 4) {
    Fail("invalid FusedDepthwiseExpandGeluProjectAdd parameters");
  }
  Tensor output{{batches, channels, height, width},
                PooledActivation(std::size_t(batches) * channels * height * width)};
  const float* expand_bias = x[4] ? x[4]->data.data() : nullptr;
  const float* project_bias = x[6] ? x[6]->data.data() : nullptr;
  if (expand_bias && x[4]->data.size() != std::size_t(hidden)) Fail("expand bias shape");
  if (project_bias && x[6]->data.size() != std::size_t(channels)) Fail("project bias shape");
  kernels::DepthwiseExpandGeluProjectAdd(
      output.data.data(), input.data.data(), dw_w.data.data(), dw_b.data.data(),
      expand_w.data.data(), expand_bias, project_w.data.data(), project_bias,
      batches, channels, hidden, height, width, int(dw_w.shape[2]), int(dw_w.shape[3]),
      int(strides[0]), int(strides[1]), int(pads[0]), int(pads[1]));
  return output;
}

Tensor ExpandGeluProjectAddOp(const Node& n, const std::vector<const Tensor*>& x) {
  if (x.size() < 5 || !x[0] || !x[1] || !x[3]) Fail("FusedExpandGeluProjectAdd inputs");
  const auto& a = *x[0];
  const auto& expand_w = *x[1];
  const auto& project_w = *x[3];
  if (a.shape.size() != 4 || expand_w.shape.size() != 4 || project_w.shape.size() != 4 ||
      expand_w.shape[2] != 1 || expand_w.shape[3] != 1 ||
      project_w.shape[2] != 1 || project_w.shape[3] != 1 ||
      AttrInt(n, "group", 1) != 1 || AttrStr(n, "auto_pad", "NOTSET") != "NOTSET") {
    Fail("FusedExpandGeluProjectAdd expects 1x1 invert residual");
  }
  const auto strides = AttrInts(n, "strides", {1, 1});
  const auto pads = AttrInts(n, "pads", {0, 0, 0, 0});
  const auto dilations = AttrInts(n, "dilations", {1, 1});
  const int N = int(a.shape[0]), C = int(a.shape[1]), H = int(a.shape[2]), W = int(a.shape[3]);
  const int hidden = int(expand_w.shape[0]);
  if (strides != std::vector<std::int64_t>{1, 1} || pads != std::vector<std::int64_t>{0, 0, 0, 0} ||
      dilations != std::vector<std::int64_t>{1, 1} || int(expand_w.shape[1]) != C ||
      int(project_w.shape[0]) != C || int(project_w.shape[1]) != hidden) {
    Fail("FusedExpandGeluProjectAdd weight shape");
  }
  const float* expand_bias = x[2] ? x[2]->data.data() : nullptr;
  const float* project_bias = x[4] ? x[4]->data.data() : nullptr;
  if (expand_bias && x[2]->data.size() != std::size_t(hidden)) Fail("expand bias shape");
  if (project_bias && x[4]->data.size() != std::size_t(C)) Fail("project bias shape");
  const auto plane = std::size_t(H) * W;
  Tensor output{{N, C, H, W}, PooledActivation(std::size_t(N) * C * plane)};
  const std::size_t sample = std::size_t(C) * plane;
  for (int batch = 0; batch < N; ++batch) {
    kernels::ExpandGeluProjectAdd(output.data.data() + std::size_t(batch) * sample,
                                  a.data.data() + std::size_t(batch) * sample,
                                  expand_w.data.data(), expand_bias,
                                  project_w.data.data(), project_bias, C, hidden, plane);
  }
  return output;
}

Tensor PointwiseConvAdd(const Node& n, const std::vector<const Tensor*>& x,
                        Backend backend = Backend::cpu_only,
                        bool immutable_parameters = false) {
  if (x.size() != 4 || !x[0] || !x[1] || !x[3]) Fail("FusedPointwiseConvAdd inputs");
  const auto& a = *x[0];
  const auto& w = *x[1];
  const auto& residual = *x[3];
  if (a.shape.size() != 4 || w.shape.size() != 4 || w.shape[2] != 1 || w.shape[3] != 1 ||
      AttrInt(n, "group", 1) != 1 || AttrStr(n, "auto_pad", "NOTSET") != "NOTSET") {
    std::vector<const Tensor*> conv_in{x[0], x[1]};
    if (x[2]) conv_in.push_back(x[2]);
    return Binary(Conv(n, conv_in), residual, kernels::BinaryOp::add);
  }
  const auto strides = AttrInts(n, "strides", {1, 1});
  const auto pads = AttrInts(n, "pads", {0, 0, 0, 0});
  const auto dilations = AttrInts(n, "dilations", {1, 1});
  const int N = int(a.shape[0]), C = int(a.shape[1]), H = int(a.shape[2]), W = int(a.shape[3]);
  const int M = int(w.shape[0]), Cg = int(w.shape[1]);
  if (strides.size() != 2 || pads.size() != 4 || dilations.size() != 2 ||
      strides[0] != 1 || strides[1] != 1 || dilations[0] != 1 || dilations[1] != 1 ||
      pads[0] != 0 || pads[1] != 0 || pads[2] != 0 || pads[3] != 0 || C != Cg ||
      residual.shape != std::vector<std::int64_t>{N, M, H, W}) {
    std::vector<const Tensor*> conv_in{x[0], x[1]};
    if (x[2]) conv_in.push_back(x[2]);
    return Binary(Conv(n, conv_in), residual, kernels::BinaryOp::add);
  }
  const float* bias = x[2] ? x[2]->data.data() : nullptr;
  if (bias && x[2]->data.size() != std::size_t(M)) Fail("Conv bias shape");
  Tensor output{{N, M, H, W}, PooledActivation(std::size_t(N) * M * H * W)};
  if (TryHybridVulkanPointwiseConvAdd(backend, output, a, w, x[2], residual,
                                      false, immutable_parameters)) {
    return output;
  }
  if (N > 1 && std::getenv("PPOCR_DISABLE_RESIDUAL_POINTWISE_BATCH") == nullptr) {
    kernels::PointwiseConvAddBatch(output.data.data(), a.data.data(), w.data.data(), bias,
                                   residual.data.data(), N, M, C, std::size_t(H) * W);
  } else {
    const std::size_t input_batch = std::size_t(C) * H * W;
    const std::size_t output_batch = std::size_t(M) * H * W;
    for (int batch = 0; batch < N; ++batch) {
      kernels::PointwiseConvAdd(output.data.data() + std::size_t(batch) * output_batch,
                                a.data.data() + std::size_t(batch) * input_batch,
                                w.data.data(), bias,
                                residual.data.data() + std::size_t(batch) * output_batch,
                                M, C, std::size_t(H) * W);
    }
  }
  return output;
}

Tensor PointwiseConvAddRelu(const Node& n, const std::vector<const Tensor*>& x,
                            Backend backend = Backend::cpu_only,
                            bool immutable_parameters = false) {
  if (x.size() != 4 || !x[0] || !x[1] || !x[3]) Fail("FusedPointwiseConvAddRelu inputs");
  const auto& a = *x[0];
  const auto& w = *x[1];
  const auto& residual = *x[3];
  if (a.shape.size() != 4 || w.shape.size() != 4 || w.shape[2] != 1 || w.shape[3] != 1 ||
      AttrInt(n, "group", 1) != 1 || AttrStr(n, "auto_pad", "NOTSET") != "NOTSET") {
    std::vector<const Tensor*> conv_in{x[0], x[1]};
    if (x[2]) conv_in.push_back(x[2]);
    Tensor output = Conv(n, conv_in);
    kernels::BinaryInplace(output.data.data(), residual.data.data(), output.data.size(),
                           kernels::BinaryOp::add);
    kernels::Relu(output.data.data(), output.data.data(), output.data.size());
    return output;
  }
  const auto strides = AttrInts(n, "strides", {1, 1});
  const auto pads = AttrInts(n, "pads", {0, 0, 0, 0});
  const auto dilations = AttrInts(n, "dilations", {1, 1});
  const int N = int(a.shape[0]), C = int(a.shape[1]), H = int(a.shape[2]), W = int(a.shape[3]);
  const int M = int(w.shape[0]), Cg = int(w.shape[1]);
  if (strides.size() != 2 || pads.size() != 4 || dilations.size() != 2 ||
      strides[0] != 1 || strides[1] != 1 || dilations[0] != 1 || dilations[1] != 1 ||
      pads[0] != 0 || pads[1] != 0 || pads[2] != 0 || pads[3] != 0 || C != Cg ||
      residual.shape != std::vector<std::int64_t>{N, M, H, W}) {
    std::vector<const Tensor*> conv_in{x[0], x[1]};
    if (x[2]) conv_in.push_back(x[2]);
    Tensor output = Conv(n, conv_in);
    kernels::BinaryInplace(output.data.data(), residual.data.data(), output.data.size(),
                           kernels::BinaryOp::add);
    kernels::Relu(output.data.data(), output.data.data(), output.data.size());
    return output;
  }
  const float* bias = x[2] ? x[2]->data.data() : nullptr;
  if (bias && x[2]->data.size() != std::size_t(M)) Fail("Conv bias shape");
  Tensor output{{N, M, H, W}, PooledActivation(std::size_t(N) * M * H * W)};
  if (TryHybridVulkanPointwiseConvAdd(backend, output, a, w, x[2], residual,
                                      true, immutable_parameters)) {
    return output;
  }
  if (N > 1 && std::getenv("PPOCR_DISABLE_RESIDUAL_POINTWISE_BATCH") == nullptr) {
    kernels::PointwiseConvAddReluBatch(output.data.data(), a.data.data(), w.data.data(), bias,
                                       residual.data.data(), N, M, C, std::size_t(H) * W);
  } else {
    const std::size_t input_batch = std::size_t(C) * H * W;
    const std::size_t output_batch = std::size_t(M) * H * W;
    for (int batch = 0; batch < N; ++batch) {
      kernels::PointwiseConvAddRelu(output.data.data() + std::size_t(batch) * output_batch,
                                    a.data.data() + std::size_t(batch) * input_batch,
                                    w.data.data(), bias,
                                    residual.data.data() + std::size_t(batch) * output_batch,
                                    M, C, std::size_t(H) * W);
    }
  }
  return output;
}

Tensor PointwiseConvAddSwish(const Node& n, const std::vector<const Tensor*>& x,
                             Backend backend = Backend::cpu_only,
                             bool immutable_parameters = false) {
  if (x.size() != 4 || !x[0] || !x[1] || !x[3]) Fail("FusedPointwiseConvAddSwish inputs");
  const auto& a = *x[0];
  const auto& w = *x[1];
  const auto& residual = *x[3];
  if (a.shape.size() != 4 || w.shape.size() != 4 || w.shape[2] != 1 || w.shape[3] != 1 ||
      AttrInt(n, "group", 1) != 1 || AttrStr(n, "auto_pad", "NOTSET") != "NOTSET") {
    std::vector<const Tensor*> conv_in{x[0], x[1]};
    if (x[2]) conv_in.push_back(x[2]);
    Tensor output = Binary(Conv(n, conv_in), residual, kernels::BinaryOp::add);
    kernels::Swish(output.data.data(), output.data.data(), output.data.size());
    return output;
  }
  const auto strides = AttrInts(n, "strides", {1, 1});
  const auto pads = AttrInts(n, "pads", {0, 0, 0, 0});
  const auto dilations = AttrInts(n, "dilations", {1, 1});
  const int N = int(a.shape[0]), C = int(a.shape[1]), H = int(a.shape[2]), W = int(a.shape[3]);
  const int M = int(w.shape[0]), Cg = int(w.shape[1]);
  if (strides.size() != 2 || pads.size() != 4 || dilations.size() != 2 ||
      strides[0] != 1 || strides[1] != 1 || dilations[0] != 1 || dilations[1] != 1 ||
      pads[0] != 0 || pads[1] != 0 || pads[2] != 0 || pads[3] != 0 || C != Cg ||
      residual.shape != std::vector<std::int64_t>{N, M, H, W}) {
    std::vector<const Tensor*> conv_in{x[0], x[1]};
    if (x[2]) conv_in.push_back(x[2]);
    Tensor output = Binary(Conv(n, conv_in), residual, kernels::BinaryOp::add);
    kernels::Swish(output.data.data(), output.data.data(), output.data.size());
    return output;
  }
  const float* bias = x[2] ? x[2]->data.data() : nullptr;
  if (bias && x[2]->data.size() != std::size_t(M)) Fail("Conv bias shape");
  Tensor output{{N, M, H, W}, PooledActivation(std::size_t(N) * M * H * W)};
  if (TryHybridVulkanPointwiseConvAdd(backend, output, a, w, x[2], residual,
                                      false, immutable_parameters, true)) {
    return output;
  }
  if (N > 1 && std::getenv("PPOCR_DISABLE_RESIDUAL_POINTWISE_BATCH") == nullptr) {
    kernels::PointwiseConvAddSwishBatch(output.data.data(), a.data.data(), w.data.data(), bias,
                                        residual.data.data(), N, M, C, std::size_t(H) * W);
  } else {
    const std::size_t input_batch = std::size_t(C) * H * W;
    const std::size_t output_batch = std::size_t(M) * H * W;
    for (int batch = 0; batch < N; ++batch) {
      kernels::PointwiseConvAddSwish(output.data.data() + std::size_t(batch) * output_batch,
                                     a.data.data() + std::size_t(batch) * input_batch,
                                     w.data.data(), bias,
                                     residual.data.data() + std::size_t(batch) * output_batch,
                                     M, C, std::size_t(H) * W);
    }
  }
  return output;
}

void SqueezeExcitationGateMulInplace(const Node& n, const std::vector<const Tensor*>& x,
                                     Tensor& output) {
  if (x.size() != 5 || !x[0] || !x[1] || !x[2] || !x[3] || !x[4]) {
    Fail("FusedSEGateMul inputs");
  }
  const auto& input = *x[0];
  const auto& first_weights = *x[1];
  const auto& first_bias = *x[2];
  const auto& second_weights = *x[3];
  const auto& second_bias = *x[4];
  if (input.shape.size() != 4 || first_weights.shape.size() != 4 ||
      second_weights.shape.size() != 4 || first_weights.shape[2] != 1 ||
      first_weights.shape[3] != 1 || second_weights.shape[2] != 1 ||
      second_weights.shape[3] != 1) {
    Fail("FusedSEGateMul shape");
  }
  const int batches = int(input.shape[0]);
  const int channels = int(input.shape[1]);
  const int height = int(input.shape[2]);
  const int width = int(input.shape[3]);
  const int reduced_channels = int(first_weights.shape[0]);
  if (batches <= 0 || channels <= 0 || height <= 0 || width <= 0 ||
      first_weights.shape[1] != channels || second_weights.shape[0] != channels ||
      second_weights.shape[1] != reduced_channels ||
      first_bias.data.size() != std::size_t(reduced_channels) ||
      second_bias.data.size() != std::size_t(channels)) {
    Fail("FusedSEGateMul channels");
  }
  if (output.shape != input.shape || output.data.size() != input.data.size()) {
    Fail("FusedSEGateMul destination");
  }
  kernels::SqueezeExcitationGateInplace(output.data.data(), first_weights.data.data(),
                                         first_bias.data.data(), second_weights.data.data(),
                                         second_bias.data.data(), batches, channels,
                                         reduced_channels, std::size_t(height) * width,
                                         Attr(n, "alpha").f, Attr(n, "beta").f,
                                         AttrInt(n, "residual", 0) != 0);
}

Tensor SqueezeExcitationGateMul(const Node& n, const std::vector<const Tensor*>& x) {
  Tensor output = *x[0];
  SqueezeExcitationGateMulInplace(n, x, output);
  return output;
}

Tensor Pool(const Node& n, const Tensor& a, bool average);
Tensor Concat(const Node& n, const std::vector<const Tensor*>& in);

Tensor ConcatConv(const Node& n, const std::vector<const Tensor*>& in, bool relu,
                  Backend backend, bool immutable_parameters) {
  if (in.size() < 3) Fail("FusedConcatConv inputs");
  const bool has_bias = in.size() >= 3 && in.back() &&
      in[in.size() - 2] && in.back()->shape.size() == 1;
  const Tensor* weights = has_bias ? in[in.size() - 2] : in.back();
  const Tensor* bias = has_bias ? in.back() : nullptr;
  const std::size_t source_count = in.size() - (has_bias ? 2 : 1);
  if (!weights || weights->shape.size() != 4 || source_count < 2) Fail("FusedConcatConv weights");
  if (!in[0] || in[0]->shape.size() != 4) Fail("FusedConcatConv source");
  const int batches = int(in[0]->shape[0]);
  const int height = int(in[0]->shape[2]);
  const int width = int(in[0]->shape[3]);
  const int output_channels = int(weights->shape[0]);
  const int kernel_h = int(weights->shape[2]);
  const int kernel_w = int(weights->shape[3]);
  const auto strides = AttrInts(n, "strides", {1, 1});
  const auto pads = AttrInts(n, "pads", {0, 0, 0, 0});
  std::vector<const float*> sources(source_count);
  std::vector<int> source_channels(source_count);
  int total_channels = 0;
  for (std::size_t i = 0; i < source_count; ++i) {
    if (!in[i] || in[i]->shape.size() != 4 || in[i]->shape[0] != batches ||
        in[i]->shape[2] != height || in[i]->shape[3] != width) {
      Fail("FusedConcatConv peer shape");
    }
    sources[i] = in[i]->data.data();
    source_channels[i] = int(in[i]->shape[1]);
    total_channels += source_channels[i];
  }
  if (total_channels != int(weights->shape[1])) Fail("FusedConcatConv channels");
  const int pad_top = pads.empty() ? 0 : int(pads[0]);
  const int pad_left = pads.size() < 2 ? pad_top : int(pads[1]);
  const int stride_h = strides.empty() ? 1 : int(strides[0]);
  const int stride_w = strides.size() < 2 ? stride_h : int(strides[1]);
  const int output_h = (height + int(pads.size() > 2 ? pads[0] + pads[2] : pad_top * 2) -
                        kernel_h) / stride_h + 1;
  const int output_w = (width + int(pads.size() > 3 ? pads[1] + pads[3] : pad_left * 2) -
                        kernel_w) / stride_w + 1;
  Tensor output{{batches, output_channels, output_h, output_w},
                PooledActivation(std::size_t(batches) * output_channels * output_h * output_w)};
  if (std::getenv("PPOCR_DUMP_SHAPES") != nullptr) {
    std::cerr << "concatconv";
    for (int ch : source_channels) std::cerr << " c" << ch;
    std::cerr << " " << height << "x" << width << " -> " << output_channels << "x"
              << output_h << "x" << output_w << " k=" << kernel_h << " s=" << stride_h
              << '\n';
  }
  if (bias && TryHybridVulkanConcatConv(
          backend, output, sources.data(), source_channels.data(), int(source_count),
          *weights, bias, height, width, output_h, output_w, kernel_h, kernel_w,
          stride_h, stride_w, pad_top, pad_left, relu, false, immutable_parameters)) {
    return output;
  }
  // The concat kernel consumes one NCHW image per call.  Preserve the
  // vectorized per-image implementation and advance every source by its own
  // channel-plane batch stride; a shared byte stride would corrupt unequal
  // branch-channel concatenations.
  const std::size_t output_batch = std::size_t(output_channels) * output_h * output_w;
  std::vector<const float*> batch_sources(source_count);
  for (int batch = 0; batch < batches; ++batch) {
    const std::size_t plane_offset = std::size_t(batch) * height * width;
    for (std::size_t source = 0; source < source_count; ++source)
      batch_sources[source] = sources[source] + plane_offset * source_channels[source];
    kernels::ConcatChannelConv2d(
        output.data.data() + std::size_t(batch) * output_batch, batch_sources.data(),
        source_channels.data(), int(source_count), weights->data.data(),
        bias ? bias->data.data() : nullptr, output_channels, height, width, output_h, output_w,
        kernel_h, kernel_w, stride_h, stride_w, pad_top, pad_left, relu);
  }
  return output;
}

Tensor ConvTransposeChainOp(const Node& n, const std::vector<const Tensor*>& in,
                            Backend backend, bool immutable_parameters) {
  (void)n;
  (void)backend;
  (void)immutable_parameters;
  if (in.size() < 4 || !in[0] || !in[1] || !in[2] || !in[3]) Fail("FusedConvTransposeChain inputs");
  const Tensor& input = *in[0];
  const Tensor& w0 = *in[1];
  const Tensor& b0 = *in[2];
  const Tensor& packed = *in[3];
  if (input.shape.size() != 4 || w0.shape.size() != 4 || w0.shape[2] != 2 || w0.shape[3] != 2 ||
      b0.data.size() != std::size_t(w0.shape[1])) {
    Fail("FusedConvTransposeChain geometry");
  }
  const int batches = int(input.shape[0]);
  const int cin = int(input.shape[1]);
  const int hin = int(input.shape[2]);
  const int win = int(input.shape[3]);
  const int mid = int(w0.shape[1]);
  if (w0.shape[0] != cin || mid <= 0 || packed.data.size() < std::size_t(mid) * 4 + 1) {
    Fail("FusedConvTransposeChain weights");
  }
  const int cout = int(packed.data.size() / (std::size_t(mid) * 4 + 1));
  if (cout <= 0 || packed.data.size() != std::size_t(cout) * (std::size_t(mid) * 4 + 1)) {
    Fail("FusedConvTransposeChain packed weights");
  }
  Tensor output{{batches, cout, hin * 4, win * 4},
                PooledActivation(std::size_t(batches) * cout * hin * win * 16)};
  const std::size_t in_batch = std::size_t(cin) * hin * win;
  const std::size_t out_batch = std::size_t(cout) * hin * win * 16;
  const float* w1 = packed.data.data();
  const float* b1 = packed.data.data() + std::size_t(mid) * cout * 4;
  for (int batch = 0; batch < batches; ++batch) {
    kernels::ConvTranspose2x2Chain(output.data.data() + std::size_t(batch) * out_batch,
                                   input.data.data() + std::size_t(batch) * in_batch,
                                   w0.data.data(), b0.data.data(), w1, b1, cin, mid, cout, hin,
                                   win);
  }
  return output;
}

Tensor ConcatConvDualTranspose(const Node& n, const std::vector<const Tensor*>& in,
                               Backend backend, bool immutable_parameters) {
  if (in.size() < 8) Fail("FusedConcatConvDualTranspose inputs");
  const Tensor* ct1_bias = in[in.size() - 1];
  const Tensor* ct1_weights = in[in.size() - 2];
  const Tensor* ct0_bias = in[in.size() - 3];
  const Tensor* ct0_weights = in[in.size() - 4];
  const Tensor* conv_bias = in[in.size() - 5];
  const Tensor* conv_weights = in[in.size() - 6];
  if (!ct1_bias || !ct1_weights || !ct0_bias || !ct0_weights || !conv_bias || !conv_weights ||
      conv_weights->shape.size() != 4 || ct0_weights->shape.size() != 4 ||
      ct1_weights->shape.size() != 4 || conv_bias->shape.size() != 1 ||
      ct0_bias->shape.size() != 1 || ct1_bias->shape.size() != 1) {
    Fail("FusedConcatConvDualTranspose weights");
  }
  const std::size_t source_count = in.size() - 6;
  std::vector<const Tensor*> concat_in(in.begin(),
                                      in.begin() + static_cast<std::ptrdiff_t>(source_count));
  concat_in.push_back(conv_weights);
  concat_in.push_back(conv_bias);
  Tensor conv = ConcatConv(n, concat_in, true, backend, immutable_parameters);
  const int conv_c = int(conv.shape[1]);
  const int conv_h = int(conv.shape[2]);
  const int conv_w = int(conv.shape[3]);
  const int ct0_oc = int(ct0_weights->shape[1]);
  const int ct1_oc = int(ct1_weights->shape[1]);
  if (conv_c != int(ct0_weights->shape[0]) || ct0_oc != int(ct1_weights->shape[0]) ||
      ct0_weights->shape[2] != 2 || ct0_weights->shape[3] != 2 ||
      ct1_weights->shape[2] != 2 || ct1_weights->shape[3] != 2 ||
      ct0_bias->data.size() != std::size_t(ct0_oc) ||
      ct1_bias->data.size() != std::size_t(ct1_oc)) {
    Fail("FusedConcatConvDualTranspose geometry");
  }
  Tensor mid{{conv.shape[0], ct0_oc, conv_h * 2, conv_w * 2},
             PooledActivation(std::size_t(ct0_oc) * conv_h * conv_w * 4)};
  kernels::ConvTranspose2x2(mid.data.data(), conv.data.data(), ct0_weights->data.data(),
                            ct0_bias->data.data(), conv_c, ct0_oc, conv_h, conv_w, 1);
  Tensor output{{conv.shape[0], ct1_oc, conv_h * 4, conv_w * 4},
                PooledActivation(std::size_t(ct1_oc) * conv_h * conv_w * 16)};
  kernels::ConvTranspose2x2(output.data.data(), mid.data.data(), ct1_weights->data.data(),
                            ct1_bias->data.data(), ct0_oc, ct1_oc, conv_h * 2, conv_w * 2, 2);
  return output;
}

Tensor MaxPoolConcatConv(const Node& n, const std::vector<const Tensor*>& in, bool relu,
                         Backend backend, bool immutable_parameters) {
  if (in.size() < 3) Fail("FusedMaxPoolConcatConv inputs");
  const bool has_bias = in.size() >= 3 && in.back() &&
      in[in.size() - 2] && in.back()->shape.size() == 1;
  const Tensor* weights = has_bias ? in[in.size() - 2] : in.back();
  const Tensor* bias = has_bias ? in.back() : nullptr;
  const std::size_t source_count = in.size() - (has_bias ? 2 : 1);
  if (!weights || weights->shape.size() != 4 || source_count < 2) {
    Fail("FusedMaxPoolConcatConv weights");
  }
  if (!in[0] || in[0]->shape.size() != 4) Fail("FusedMaxPoolConcatConv source");
  const int batches = int(in[0]->shape[0]);
  const int height = int(in[0]->shape[2]);
  const int width = int(in[0]->shape[3]);
  const int pool_channels = int(in[0]->shape[1]);
  const int output_channels = int(weights->shape[0]);
  const int kernel_h = int(weights->shape[2]);
  const int kernel_w = int(weights->shape[3]);
  const auto strides = AttrInts(n, "strides", {1, 1});
  const auto pads = AttrInts(n, "pads", {0, 0, 0, 0});
  std::vector<const float*> sources(source_count);
  std::vector<int> source_channels(source_count);
  sources[0] = in[0]->data.data();
  source_channels[0] = pool_channels;
  int total_channels = pool_channels;
  for (std::size_t i = 1; i < source_count; ++i) {
    if (!in[i] || in[i]->shape.size() != 4 || in[i]->shape[0] != batches ||
        in[i]->shape[2] != height || in[i]->shape[3] != width) {
      Fail("FusedMaxPoolConcatConv peer shape");
    }
    sources[i] = in[i]->data.data();
    source_channels[i] = int(in[i]->shape[1]);
    total_channels += source_channels[i];
  }
  // The pooling/concat stem is independent for every NCHW item.  The old
  // N=1 guard was inherited from the first scalar implementation and made
  // otherwise valid detector page batches fail before their SIMD kernels
  // could run.  MaxPool2x2Same and ConcatChannelConv2d both carry the batch
  // dimension explicitly, so validate channel geometry only.
  if (total_channels != int(weights->shape[1])) {
    Fail("FusedMaxPoolConcatConv channels");
  }
  const int pad_top = pads.empty() ? 0 : int(pads[0]);
  const int pad_left = pads.size() < 2 ? pad_top : int(pads[1]);
  const int stride_h = strides.empty() ? 1 : int(strides[0]);
  const int stride_w = strides.size() < 2 ? stride_h : int(strides[1]);
  const int output_h = (height + int(pads.size() > 2 ? pads[0] + pads[2] : pad_top * 2) -
                        kernel_h) / stride_h + 1;
  const int output_w = (width + int(pads.size() > 3 ? pads[1] + pads[3] : pad_left * 2) -
                        kernel_w) / stride_w + 1;
  Tensor output{{batches, output_channels, output_h, output_w},
                PooledActivation(std::size_t(batches) * output_channels * output_h *
                                 output_w)};
  if (std::getenv("PPOCR_DUMP_SHAPES") != nullptr) {
    std::cerr << "maxpoolconcatconv";
    for (int ch : source_channels) std::cerr << " c" << ch;
    std::cerr << " " << height << "x" << width << " -> " << output_channels << "x"
              << output_h << "x" << output_w << " k=" << kernel_h << " s=" << stride_h
              << '\n';
  }
  if (bias && TryHybridVulkanConcatConv(
          backend, output, sources.data(), source_channels.data(), int(source_count),
          *weights, bias, height, width, output_h, output_w, kernel_h, kernel_w,
          stride_h, stride_w, pad_top, pad_left, relu, true, immutable_parameters)) {
    return output;
  }
  Tensor pooled{{batches, pool_channels, height, width},
                PooledActivation(std::size_t(batches) * pool_channels * height * width)};
  kernels::MaxPool2x2Same(pooled.data.data(), in[0]->data.data(),
                          std::size_t(batches) * pool_channels, height, width);
  sources[0] = pooled.data.data();
  // ConcatChannelConv2d is intentionally a single NCHW plane primitive: its
  // source pointer table describes channel planes, not leading-N strides.
  // Run that vectorized primitive once per independent batch item instead of
  // treating the contiguous N>1 storage as one image.  This keeps the tuned
  // AVX-512/AVX2/NEON implementation intact and makes detector batching
  // numerically identical to serial N=1 execution.
  const std::size_t output_batch = std::size_t(output_channels) * output_h * output_w;
  std::vector<const float*> batch_sources(source_count);
  for (int batch = 0; batch < batches; ++batch) {
    const std::size_t offset = std::size_t(batch) * height * width;
    for (std::size_t source = 0; source < source_count; ++source) {
      batch_sources[source] = sources[source] + offset * source_channels[source];
    }
    kernels::ConcatChannelConv2d(
        output.data.data() + std::size_t(batch) * output_batch, batch_sources.data(),
        source_channels.data(), int(source_count), weights->data.data(),
        bias ? bias->data.data() : nullptr, output_channels, height, width, output_h, output_w,
        kernel_h, kernel_w, stride_h, stride_w, pad_top, pad_left, relu);
  }
  return output;
}

Tensor MaxPoolConcat(const Node& n, const std::vector<const Tensor*>& in) {
  if (in.size() < 2 || !in[0] || !in[1]) Fail("FusedMaxPoolConcat inputs");
  const Tensor& source = *in[0];
  if (source.shape.size() != 4) Fail("FusedMaxPoolConcat shape");
  const auto kernel = AttrInts(n, "kernel_shape", {});
  const auto strides = AttrInts(n, "strides", {1, 1});
  const auto pads = AttrInts(n, "pads", {0, 0, 0, 0});
  const int batches = int(source.shape[0]), channels = int(source.shape[1]);
  const int height = int(source.shape[2]), width = int(source.shape[3]);
  const int axis = Axis(AttrInt(n, "axis", 1), 4);
  const auto auto_pad = AttrStr(n, "auto_pad", "NOTSET");
  if (kernel == std::vector<std::int64_t>{2, 2} &&
      strides == std::vector<std::int64_t>{1, 1} &&
      auto_pad == "SAME_UPPER" && height >= 1 && width >= 1 && axis == 1) {
    std::int64_t total_channels = channels;
    for (std::size_t i = 1; i < in.size(); ++i) {
      if (!in[i] || in[i]->shape.size() != 4 || in[i]->shape[0] != batches ||
          in[i]->shape[2] != height || in[i]->shape[3] != width) {
        Fail("FusedMaxPoolConcat same peer shape");
      }
      total_channels += in[i]->shape[1];
    }
    Tensor output;
    output.shape = {batches, total_channels, height, width};
    output.data = PooledActivation(Elements(output.shape));
    kernels::MaxPool2x2Same(output.data.data(), source.data.data(),
                            std::size_t(batches) * channels, height, width);
    std::size_t offset = std::size_t(batches) * channels * height * width;
    for (std::size_t i = 1; i < in.size(); ++i) {
      std::memcpy(output.data.data() + offset, in[i]->data.data(),
                  in[i]->data.size() * sizeof(float));
      offset += in[i]->data.size();
    }
    return output;
  }
  if (kernel == std::vector<std::int64_t>{2, 2} &&
      strides == std::vector<std::int64_t>{1, 1} &&
      pads == std::vector<std::int64_t>{0, 0, 0, 0} &&
      auto_pad == "NOTSET" &&
      height >= 2 && width >= 2 && axis == 1) {
    const int out_h = height - 1, out_w = width - 1;
    std::int64_t total_channels = channels;
    for (std::size_t i = 1; i < in.size(); ++i) {
      if (!in[i] || in[i]->shape.size() != 4 || in[i]->shape[0] != batches ||
          in[i]->shape[2] != out_h || in[i]->shape[3] != out_w) {
        Fail("FusedMaxPoolConcat peer shape");
      }
      total_channels += in[i]->shape[1];
    }
    Tensor output;
    output.shape = {batches, total_channels, out_h, out_w};
    output.data = PooledActivation(Elements(output.shape));
    kernels::MaxPool2x2Valid(output.data.data(), source.data.data(),
                             std::size_t(batches) * channels, height, width);
    std::size_t offset = std::size_t(batches) * channels * out_h * out_w;
    for (std::size_t i = 1; i < in.size(); ++i) {
      std::memcpy(output.data.data() + offset, in[i]->data.data(),
                  in[i]->data.size() * sizeof(float));
      offset += in[i]->data.size();
    }
    return output;
  }
  Tensor pooled = Pool(n, source, false);
  std::vector<const Tensor*> parts(in.size());
  parts[0] = &pooled;
  for (std::size_t i = 1; i < in.size(); ++i) parts[i] = in[i];
  Node concat = n;
  concat.op = "Concat";
  return Concat(concat, parts);
}

Tensor ConvMaxPoolConcat(const Node& n, const std::vector<const Tensor*>& in) {
  if (in.size() < 3 || !in[0] || !in[1] || !in[2]) Fail("FusedConvMaxPoolConcat inputs");
  const Tensor& pooled_src = *in[0];
  const Tensor& conv_src = *in[1];
  const Tensor& weights = *in[2];
  const float* bias = in.size() > 3 && in[3] ? in[3]->data.data() : nullptr;
  if (pooled_src.shape.size() != 4 || conv_src.shape.size() != 4 ||
      weights.shape.size() != 4) Fail("FusedConvMaxPoolConcat shape");
  const int batches = int(pooled_src.shape[0]);
  const int pool_channels = int(pooled_src.shape[1]);
  const int height = int(pooled_src.shape[2]);
  const int width = int(pooled_src.shape[3]);
  const int conv_in = int(conv_src.shape[1]);
  const int conv_out = int(weights.shape[0]);
  if (conv_src.shape[0] != batches || conv_src.shape[2] != height ||
      conv_src.shape[3] != width || weights.shape[1] != conv_in ||
      weights.shape[2] != 2 || weights.shape[3] != 2) {
    Fail("FusedConvMaxPoolConcat conv");
  }
  Tensor output;
  output.shape = {batches, pool_channels + conv_out, height, width};
  output.data = PooledActivation(Elements(output.shape));
  kernels::MaxPool2x2Same(output.data.data(), pooled_src.data.data(),
                          std::size_t(batches) * pool_channels, height, width);
  const std::size_t conv_offset =
      std::size_t(batches) * pool_channels * height * width;
  const std::size_t in_batch = std::size_t(conv_in) * height * width;
  const std::size_t out_batch = std::size_t(conv_out) * height * width;
  for (int batch = 0; batch < batches; ++batch) {
    kernels::Conv2d(output.data.data() + conv_offset + std::size_t(batch) * out_batch,
                    conv_src.data.data() + std::size_t(batch) * in_batch,
                    weights.data.data(), bias, conv_in, conv_out, height, width,
                    height, width, 2, 2, 1, 1, 0, 0, true);
  }
  return output;
}

Tensor Pool(const Node& n, const Tensor& a, bool average) {
  if(a.shape.size()!=4) Fail("Pool needs NCHW");
  auto k=AttrInts(n,"kernel_shape",{}),st=AttrInts(n,"strides",{1,1}),pd=AttrInts(n,"pads",{0,0,0,0});
  if(k.size()!=2) Fail("pool kernel missing");
  const int N=int(a.shape[0]),C=int(a.shape[1]),H=int(a.shape[2]),W=int(a.shape[3]),KH=int(k[0]),KW=int(k[1]);
  const auto auto_pad=AttrStr(n,"auto_pad","NOTSET"); int OH{},OW{};
  if(auto_pad=="SAME_UPPER"||auto_pad=="SAME_LOWER") { OH=(H+int(st[0])-1)/int(st[0]); OW=(W+int(st[1])-1)/int(st[1]); const int ph=std::max(0,(OH-1)*int(st[0])+KH-H),pw=std::max(0,(OW-1)*int(st[1])+KW-W);pd[0]=auto_pad=="SAME_LOWER"?(ph+1)/2:ph/2;pd[2]=ph-pd[0];pd[1]=auto_pad=="SAME_LOWER"?(pw+1)/2:pw/2;pd[3]=pw-pd[1]; }
  else { OH=(H+int(pd[0])+int(pd[2])-KH)/int(st[0])+1; OW=(W+int(pd[1])+int(pd[3])-KW)/int(st[1])+1; }
  Tensor o{{N,C,OH,OW},PooledActivation(std::size_t(N)*C*OH*OW)};
  if (!average && KH == 2 && KW == 2 && st[0] == 1 && st[1] == 1 &&
      OH == H && OW == W && pd[0] == 0 && pd[1] == 0 &&
      pd[2] == 1 && pd[3] == 1) {
    kernels::MaxPool2x2Same(o.data.data(), a.data.data(), std::size_t(N) * C, H, W);
    return o;
  }
  // All three PP-OCRv6 recognizers bridge their convolutional encoder to
  // attention with this exact valid 3x2/stride-3x2 average pool.  It has no
  // border handling or overlapping windows, so a direct plane kernel avoids
  // generic ONNX attribute/index work in this permanently hot operation.
  static const bool average_pool3x2_fast_path_enabled =
      std::getenv("PPOCR_DISABLE_AVERAGE_POOL3X2_FASTPATH") == nullptr;
  if (average_pool3x2_fast_path_enabled && average && KH == 3 && KW == 2 &&
      st == std::vector<std::int64_t>{3, 2} &&
      pd == std::vector<std::int64_t>{0, 0, 0, 0} && OH == (H - 3) / 3 + 1 &&
      OW == (W - 2) / 2 + 1) {
    kernels::AveragePool3x2Valid(o.data.data(), a.data.data(), std::size_t(N) * C, H, W);
    return o;
  }
  for(int n0=0;n0<N;++n0)for(int c=0;c<C;++c)for(int y=0;y<OH;++y)for(int z=0;z<OW;++z){float v=average?0:-std::numeric_limits<float>::infinity();int cnt{};for(int ky=0;ky<KH;++ky)for(int kx=0;kx<KW;++kx){int iy=y*int(st[0])-int(pd[0])+ky,ix=z*int(st[1])-int(pd[1])+kx;if(iy>=0&&iy<H&&ix>=0&&ix<W){auto q=a.data[((n0*C+c)*H+iy)*W+ix];v=average?v+q:std::max(v,q);++cnt;}}o.data[((n0*C+c)*OH+y)*OW+z]=average?v/float(cnt):v;}
  return o;
}

// PP-OCRv6's recognizer bridge pool is a valid 3x2 window with matching 3x2
// stride. Its windows do not overlap, and the compact output is written in
// increasing raster order. Therefore a destination beginning at the source
// base cannot overwrite a value needed by a later window: each input row is
// consumed before the smaller packed output reaches it. Reusing this dying
// activation avoids allocating another [N,C,H/3,W/2] feature map.
bool AveragePool3x2ValidInplace(const Node& n, Tensor& value) {
  if (std::getenv("PPOCR_DISABLE_AVERAGE_POOL3X2_INPLACE") != nullptr ||
      value.shape.size() != 4) return false;
  const auto kernel = AttrInts(n, "kernel_shape", {});
  const auto strides = AttrInts(n, "strides", {1, 1});
  const auto pads = AttrInts(n, "pads", {0, 0, 0, 0});
  if (kernel != std::vector<std::int64_t>{3, 2} ||
      strides != std::vector<std::int64_t>{3, 2} ||
      pads != std::vector<std::int64_t>{0, 0, 0, 0} ||
      AttrStr(n, "auto_pad", "NOTSET") != "NOTSET") return false;
  const int batches = static_cast<int>(value.shape[0]);
  const int channels = static_cast<int>(value.shape[1]);
  const int height = static_cast<int>(value.shape[2]);
  const int width = static_cast<int>(value.shape[3]);
  if (batches <= 0 || channels <= 0 || height < 3 || width < 2) return false;
  const int output_height = (height - 3) / 3 + 1;
  const int output_width = (width - 2) / 2 + 1;
  const std::size_t output_elements =
      std::size_t(batches) * channels * output_height * output_width;
  if (value.data.size() != std::size_t(batches) * channels * height * width ||
      output_elements > value.data.size()) return false;
  kernels::AveragePool3x2Valid(value.data.data(), value.data.data(),
                                std::size_t(batches) * channels, height, width);
  value.data.resize(output_elements);
  value.shape = {batches, channels, output_height, output_width};
  return true;
}

Tensor ConvTranspose(const Node& n, const std::vector<const Tensor*>& x,
                     Backend backend = Backend::cpu_only,
                     bool immutable_parameters = false, int act = 0) {
  const auto& a = *x[0]; const auto& w = *x[1];
  if (a.shape.size() != 4 || w.shape.size() != 4) Fail("ConvTranspose needs NCHW");
  const int N=int(a.shape[0]), C=int(a.shape[1]), H=int(a.shape[2]), W=int(a.shape[3]), Cg=int(w.shape[0]), Mpg=int(w.shape[1]), KH=int(w.shape[2]), KW=int(w.shape[3]);
  auto st=AttrInts(n,"strides",{1,1}), pd=AttrInts(n,"pads",{0,0,0,0}), dl=AttrInts(n,"dilations",{1,1}), op=AttrInts(n,"output_padding",{0,0}); const int group=int(AttrInt(n,"group",1));
  if (C != Cg || C % group) Fail("invalid ConvTranspose channels");
  const int M=Mpg*group, OH=(H-1)*int(st[0])-int(pd[0])-int(pd[2])+int(dl[0])*(KH-1)+int(op[0])+1, OW=(W-1)*int(st[1])-int(pd[1])-int(pd[3])+int(dl[1])*(KW-1)+int(op[1])+1;
  Tensor o{{N,M,OH,OW},PooledActivation(std::size_t(N)*M*OH*OW)}; const float* bias=x.size()>2&&x[2]?x[2]->data.data():nullptr; const int cg=C/group;
  if (group == 1 && KH == 2 && KW == 2 && st[0] == 2 && st[1] == 2 &&
      pd[0] == 0 && pd[1] == 0 && pd[2] == 0 && pd[3] == 0 &&
      dl[0] == 1 && dl[1] == 1 && op[0] == 0 && op[1] == 0 && OH == H*2 && OW == W*2) {
    if (TryHybridVulkanConvTranspose2x2(backend, o, a, w, x.size() > 2 ? x[2] : nullptr,
                                        immutable_parameters)) {
      if (act == 1) kernels::Relu(o.data.data(), o.data.data(), o.data.size());
      else if (act == 2) kernels::Sigmoid(o.data.data(), o.data.data(), o.data.size());
      return o;
    }
    kernels::ConvTranspose2x2Batch(o.data.data(), a.data.data(), w.data.data(), bias,
                                   N, C, M, H, W, act);
    return o;
  }
  std::fill(o.data.begin(), o.data.end(), 0.F);
  for(int ni=0;ni<N;++ni)for(int ci=0;ci<C;++ci)for(int iy=0;iy<H;++iy)for(int ix=0;ix<W;++ix){const int g=ci/cg;const float av=a.data[((ni*C+ci)*H+iy)*W+ix];for(int mo=0;mo<Mpg;++mo)for(int ky=0;ky<KH;++ky)for(int kx=0;kx<KW;++kx){const int oy=iy*int(st[0])-int(pd[0])+ky*int(dl[0]),ox=ix*int(st[1])-int(pd[1])+kx*int(dl[1]);if(oy>=0&&oy<OH&&ox>=0&&ox<OW)o.data[((ni*M+g*Mpg+mo)*OH+oy)*OW+ox]+=av*w.data[((ci*Mpg+mo)*KH+ky)*KW+kx];}}
  if(bias) for(int ni=0;ni<N;++ni)for(int mo=0;mo<M;++mo)for(int y=0;y<OH;++y)for(int z=0;z<OW;++z)o.data[((ni*M+mo)*OH+y)*OW+z]+=bias[mo]; return o;
}

Tensor ConvTranspose2x2Add(const Node& n, const std::vector<const Tensor*>& x,
                           Backend backend = Backend::cpu_only,
                           bool immutable_parameters = false) {
  if (x.size() != 4 || !x[0] || !x[1] || !x[3]) {
    Fail("FusedConvTranspose2x2Add inputs");
  }
  const auto& input = *x[0]; const auto& weights = *x[1];
  const auto& residual = *x[3];
  if (input.shape.size() != 4 || weights.shape.size() != 4 ||
      weights.shape[2] != 2 || weights.shape[3] != 2 ||
      AttrInt(n, "group", 1) != 1 ||
      AttrInts(n, "strides", {1, 1}) != std::vector<std::int64_t>{2, 2} ||
      AttrInts(n, "pads", {0, 0, 0, 0}) != std::vector<std::int64_t>{0, 0, 0, 0} ||
      AttrInts(n, "dilations", {1, 1}) != std::vector<std::int64_t>{1, 1} ||
      AttrInts(n, "output_padding", {0, 0}) != std::vector<std::int64_t>{0, 0}) {
    Fail("invalid FusedConvTranspose2x2Add geometry");
  }
  const int batches = static_cast<int>(input.shape[0]);
  const int input_channels = static_cast<int>(input.shape[1]);
  const int input_height = static_cast<int>(input.shape[2]);
  const int input_width = static_cast<int>(input.shape[3]);
  const int output_channels = static_cast<int>(weights.shape[1]);
  if (batches <= 0 || input_channels <= 0 || output_channels <= 0 || input_height <= 0 ||
      input_width <= 0 || weights.shape[0] != input_channels ||
      (x[2] && x[2]->data.size() != static_cast<std::size_t>(output_channels))) {
    Fail("invalid FusedConvTranspose2x2Add shape");
  }
  Tensor output{{input.shape[0], weights.shape[1], input.shape[2] * 2, input.shape[3] * 2},
                PooledActivation(std::size_t(batches) * output_channels * input_height * input_width * 4)};
  if (residual.data.size() != output.data.size() ||
      (residual.shape.size() == 4 && residual.shape != output.shape)) {
    Fail("FusedConvTranspose2x2Add residual shape");
  }
  const float* bias_data = x[2] ? x[2]->data.data() : nullptr;
  if (x[2] && TryHybridVulkanConvTranspose2x2Add(backend, output, input, weights, x[2], residual,
                                                 immutable_parameters)) return output;
  kernels::ConvTranspose2x2Batch(output.data.data(), input.data.data(), weights.data.data(),
                                 bias_data, batches, input_channels, output_channels,
                                 input_height, input_width);
  kernels::BinaryInplace(output.data.data(), residual.data.data(), output.data.size(),
                         kernels::BinaryOp::add);
  return output;
}

Tensor GlobalAveragePool(const Tensor& a) {
  if (a.shape.size() < 3) Fail("GlobalAveragePool rank");
  const auto spatial = Elements(std::vector<std::int64_t>(a.shape.begin() + 2, a.shape.end()));
  Tensor o{{a.shape[0], a.shape[1], 1, 1},
           std::vector<float>(std::size_t(a.shape[0] * a.shape[1]))};
  kernels::SpatialMean(o.data.data(), a.data.data(), o.data.size(), spatial);
  return o;
}
Tensor ReduceMean(const Node& n, const std::vector<const Tensor*>& in) {
  const auto& a=*in[0]; std::vector<std::int64_t> axes=AttrInts(n,"axes",{}); if(axes.empty()&&in.size()>1&&in[1])for(float q:in[1]->data)axes.push_back(std::int64_t(q)); if(axes.empty())for(int i=0;i<int(a.shape.size());++i)axes.push_back(i);const bool keep=AttrInt(n,"keepdims",1)!=0;
  // PP-OCRv6 repeatedly reduces NCHW spatial planes for squeeze-excitation.
  // This direct path avoids generic index/stride reconstruction per element.
  if (a.shape.size() == 4 && axes.size() == 2 &&
      ((Axis(axes[0], 4) == 2 && Axis(axes[1], 4) == 3) ||
       (Axis(axes[0], 4) == 3 && Axis(axes[1], 4) == 2))) {
    Tensor o;
    if (keep) o.shape = {a.shape[0], a.shape[1], 1, 1};
    else o.shape = {a.shape[0], a.shape[1]};
    o.data.resize(std::size_t(a.shape[0]) * a.shape[1]);
    kernels::SpatialMean(o.data.data(), a.data.data(), o.data.size(),
                         std::size_t(a.shape[2]) * a.shape[3]);
    return o;
  }
  std::vector<bool> reduce(a.shape.size());for(auto q:axes)reduce[Axis(q,int(a.shape.size()))]=true;Tensor o;for(std::size_t i=0;i<a.shape.size();++i)if(!reduce[i]||keep)o.shape.push_back(reduce[i]?1:a.shape[i]);o.data.assign(Elements(o.shape),0);std::vector<std::size_t> os=Strides(o.shape),as=Strides(a.shape);std::vector<std::size_t> counts(o.data.size());for(std::size_t ai=0;ai<a.data.size();++ai){std::size_t oi{},od{};for(std::size_t d=0;d<a.shape.size();++d){const auto idx=(ai/as[d])%std::size_t(a.shape[d]);if(!reduce[d]){oi+=idx*os[od];++od;}else if(keep)++od;}o.data[oi]+=a.data[ai];++counts[oi];}for(std::size_t i=0;i<o.data.size();++i)o.data[i]/=float(counts[i]);return o;
}
void BatchNormAffineParameters(const Node& n, const std::vector<const Tensor*>& in,
                               std::vector<float>& factor, std::vector<float>& offset) {
  if (in.size() != 5 || !in[0] || !in[1] || !in[2] || !in[3] || !in[4] ||
      in[0]->shape.size() < 2) Fail("BatchNormalization inputs");
  const auto channels = std::size_t(in[0]->shape[1]);
  if (in[1]->data.size() != channels || in[2]->data.size() != channels ||
      in[3]->data.size() != channels || in[4]->data.size() != channels) {
    Fail("BatchNormalization coefficients");
  }
  const auto epsilon = n.attr.contains("epsilon") ? Attr(n, "epsilon").f : 1e-5F;
  factor.resize(channels);
  offset.resize(channels);
  for (std::size_t channel = 0; channel < channels; ++channel) {
    factor[channel] = in[1]->data[channel] / std::sqrt(in[4]->data[channel] + epsilon);
    offset[channel] = in[2]->data[channel] - in[3]->data[channel] * factor[channel];
  }
}

Tensor BatchNorm(const Node& n, const std::vector<const Tensor*>& in) {
  const auto& a = *in[0];
  if (a.shape.size() < 2) Fail("BatchNormalization rank");
  const auto channels = std::size_t(a.shape[1]);
  const auto spatial = a.data.size() / (std::size_t(a.shape[0]) * channels);
  std::vector<float> factor, offset;
  BatchNormAffineParameters(n, in, factor, offset);
  Tensor output{a.shape, std::vector<float>(a.data.size())};
  kernels::BatchNormAffine(output.data.data(), a.data.data(), factor.data(), offset.data(),
                            int(a.shape[0]), int(channels), spatial);
  return output;
}

void BatchNormInplace(const Node& n, const std::vector<const Tensor*>& in, Tensor& output) {
  const auto& a = *in[0];
  if (a.shape.size() < 2) Fail("BatchNormalization rank");
  const auto channels = std::size_t(a.shape[1]);
  const auto spatial = a.data.size() / (std::size_t(a.shape[0]) * channels);
  std::vector<float> factor, offset;
  BatchNormAffineParameters(n, in, factor, offset);
  kernels::BatchNormAffine(output.data.data(), a.data.data(), factor.data(), offset.data(),
                            int(a.shape[0]), int(channels), spatial);
}
Tensor BatchNormGelu(const Node& n, const std::vector<const Tensor*>& in) {
  const auto& a = *in[0];
  if (a.shape.size() < 2) Fail("BatchNormalization rank");
  Tensor o{a.shape, std::vector<float>(a.data.size())};
  const auto channels = std::size_t(a.shape[1]);
  const auto spatial = a.data.size() / (std::size_t(a.shape[0]) * channels);
  if (in.size() == 3 && in[1]->data.size() == channels && in[2]->data.size() == channels) {
    kernels::BatchNormGelu(o.data.data(), a.data.data(), in[1]->data.data(), in[2]->data.data(),
                           int(a.shape[0]), int(channels), spatial);
    return o;
  }
  const auto eps = n.attr.contains("epsilon") ? Attr(n, "epsilon").f : 1e-5F;
  std::vector<float> factor(channels), offset(channels);
  for (std::size_t channel = 0; channel < channels; ++channel) {
    factor[channel] = in[1]->data[channel] / std::sqrt(in[4]->data[channel] + eps);
    offset[channel] = in[2]->data[channel] - in[3]->data[channel] * factor[channel];
  }
  kernels::BatchNormGelu(o.data.data(), a.data.data(), factor.data(), offset.data(),
                         int(a.shape[0]), int(channels), spatial);
  return o;
}
void BatchNormGeluInplace(const Node& n, const std::vector<const Tensor*>& in,
                          Tensor& output) {
  const auto& a = *in[0];
  if (a.shape.size() < 2) Fail("BatchNormalization rank");
  const auto channels = std::size_t(a.shape[1]);
  const auto spatial = a.data.size() / (std::size_t(a.shape[0]) * channels);
  if (in.size() == 3 && in[1]->data.size() == channels && in[2]->data.size() == channels) {
    kernels::BatchNormGelu(output.data.data(), a.data.data(), in[1]->data.data(), in[2]->data.data(),
                           int(a.shape[0]), int(channels), spatial);
    return;
  }
  const auto eps = n.attr.contains("epsilon") ? Attr(n, "epsilon").f : 1e-5F;
  std::vector<float> factor(channels), offset(channels);
  for (std::size_t channel = 0; channel < channels; ++channel) {
    factor[channel] = in[1]->data[channel] / std::sqrt(in[4]->data[channel] + eps);
    offset[channel] = in[2]->data[channel] - in[3]->data[channel] * factor[channel];
  }
  kernels::BatchNormGelu(output.data.data(), a.data.data(), factor.data(), offset.data(),
                         int(a.shape[0]), int(channels), spatial);
}
Tensor BatchNormSwish(const Node& n, const std::vector<const Tensor*>& in) {
  const auto& a = *in[0];
  if (a.shape.size() < 2) Fail("BatchNormalization rank");
  const auto channels = std::size_t(a.shape[1]);
  const auto spatial = a.data.size() / (std::size_t(a.shape[0]) * channels);
  Tensor output{a.shape, std::vector<float>(a.data.size())};
  if (in.size() == 3 && in[1]->data.size() == channels && in[2]->data.size() == channels) {
    kernels::BatchNormSwish(output.data.data(), a.data.data(), in[1]->data.data(), in[2]->data.data(),
                            int(a.shape[0]), int(channels), spatial);
    return output;
  }
  const auto eps = n.attr.contains("epsilon") ? Attr(n, "epsilon").f : 1e-5F;
  std::vector<float> factor(channels), offset(channels);
  for (std::size_t channel = 0; channel < channels; ++channel) {
    factor[channel] = in[1]->data[channel] / std::sqrt(in[4]->data[channel] + eps);
    offset[channel] = in[2]->data[channel] - in[3]->data[channel] * factor[channel];
  }
  kernels::BatchNormSwish(output.data.data(), a.data.data(), factor.data(), offset.data(),
                          int(a.shape[0]), int(channels), spatial);
  return output;
}
void BatchNormSwishInplace(const Node& n, const std::vector<const Tensor*>& in,
                           Tensor& output) {
  const auto& a = *in[0];
  if (a.shape.size() < 2) Fail("BatchNormalization rank");
  const auto channels = std::size_t(a.shape[1]);
  const auto spatial = a.data.size() / (std::size_t(a.shape[0]) * channels);
  if (in.size() == 3 && in[1]->data.size() == channels && in[2]->data.size() == channels) {
    // The kernel is explicitly alias-safe: each input scalar is read before
    // its transformed Swish value is stored. Reusing the dying BN input avoids
    // a recognizer-sized output allocation and lowers the activation frontier.
    kernels::BatchNormSwish(output.data.data(), a.data.data(), in[1]->data.data(), in[2]->data.data(),
                            int(a.shape[0]), int(channels), spatial);
    return;
  }
  const auto eps = n.attr.contains("epsilon") ? Attr(n, "epsilon").f : 1e-5F;
  std::vector<float> factor(channels), offset(channels);
  for (std::size_t channel = 0; channel < channels; ++channel) {
    factor[channel] = in[1]->data[channel] / std::sqrt(in[4]->data[channel] + eps);
    offset[channel] = in[2]->data[channel] - in[3]->data[channel] * factor[channel];
  }
  kernels::BatchNormSwish(output.data.data(), a.data.data(), factor.data(), offset.data(),
                          int(a.shape[0]), int(channels), spatial);
}
Tensor BatchNormSwishAdd(const Node&, const std::vector<const Tensor*>& in) {
  if (in.size() != 4 || !in[3]) Fail("FusedBatchNormSwishAdd inputs");
  const auto& a = *in[0];
  const auto& residual = *in[3];
  if (a.shape.size() != 4 || residual.shape != a.shape) Fail("FusedBatchNormSwishAdd shape");
  const auto channels = std::size_t(a.shape[1]);
  const auto spatial = a.data.size() / (std::size_t(a.shape[0]) * channels);
  if (in[1]->data.size() != channels || in[2]->data.size() != channels) {
    Fail("FusedBatchNormSwishAdd coefficients");
  }
  Tensor output{a.shape, std::vector<float>(a.data.size())};
  kernels::BatchNormSwishBinary(output.data.data(), a.data.data(), in[1]->data.data(),
                                in[2]->data.data(), residual.data.data(), int(a.shape[0]),
                                int(channels), spatial, kernels::BinaryOp::add);
  return output;
}
void BatchNormSwishAddInplace(const std::vector<const Tensor*>& in, Tensor& output) {
  if (in.size() != 4 || !in[3]) Fail("FusedBatchNormSwishAdd inputs");
  const auto& a = *in[0];
  const auto& residual = *in[3];
  if (a.shape.size() != 4 || residual.shape != a.shape) Fail("FusedBatchNormSwishAdd shape");
  const auto channels = std::size_t(a.shape[1]);
  const auto spatial = a.data.size() / (std::size_t(a.shape[0]) * channels);
  if (in[1]->data.size() != channels || in[2]->data.size() != channels) {
    Fail("FusedBatchNormSwishAdd coefficients");
  }
  kernels::BatchNormSwishBinary(output.data.data(), a.data.data(), in[1]->data.data(),
                                in[2]->data.data(), residual.data.data(), int(a.shape[0]),
                                int(channels), spatial, kernels::BinaryOp::add);
}
Tensor Shape(const Tensor& a) { Tensor o{{std::int64_t(a.shape.size())},{}};for(auto q:a.shape)o.data.push_back(float(q));return o; }
Tensor Squeeze(const Node& n,const std::vector<const Tensor*>& in,bool unsqueeze) { const auto& a=*in[0];std::vector<std::int64_t> axes=AttrInts(n,"axes",{});if(axes.empty()&&in.size()>1&&in[1])for(float q:in[1]->data)axes.push_back(std::int64_t(q));Tensor o=a;if(unsqueeze){const int r=int(a.shape.size()+axes.size());std::vector<int> pos;for(auto q:axes){int v=int(q);if(v<0)v+=r;if(v<0||v>=r)Fail("unsqueeze axis");pos.push_back(v);}std::sort(pos.begin(),pos.end());for(int i:pos)o.shape.insert(o.shape.begin()+i,1);}else{std::vector<bool> drop(a.shape.size());if(axes.empty())for(std::size_t i=0;i<a.shape.size();++i)drop[i]=a.shape[i]==1;else for(auto q:axes){auto i=Axis(q,int(a.shape.size()));if(a.shape[i]!=1)Fail("squeeze non-unit axis");drop[i]=true;}o.shape.clear();for(std::size_t i=0;i<a.shape.size();++i)if(!drop[i])o.shape.push_back(a.shape[i]);}return o; }
Tensor Slice(const Tensor& a,const std::vector<const Tensor*>& in) {
  if(in.size()<3) Fail("Slice inputs");
  const auto& starts=in[1]->data; const auto& ends=in[2]->data;
  const auto* axes=in.size()>3?in[3]:nullptr; const auto* steps=in.size()>4?in[4]:nullptr;
  Tensor o; std::vector<int> begin(a.shape.size()),step(a.shape.size(),1);
  for(std::size_t i=0;i<a.shape.size();++i) o.shape.push_back(a.shape[i]);
  for(std::size_t i=0;i<starts.size();++i) {
    int ax=axes?Axis(std::int64_t(axes->data[i]),int(a.shape.size())):int(i);
    int s=int(starts[i]),e=int(ends[i]),st=steps?int(steps->data[i]):1;
    if(st<=0) Fail("negative Slice step unsupported");
    if(s<0)s+=int(a.shape[ax]);if(e<0)e+=int(a.shape[ax]);
    s=std::clamp(s,0,int(a.shape[ax]));e=std::clamp(e,0,int(a.shape[ax]));
    begin[ax]=s;step[ax]=st;o.shape[ax]=std::max(0,(e-s+st-1)/st);
  }
  o.data.resize(Elements(o.shape));
  // Attention Q/K/V is stacked on axis 0 and subsequently split into three
  // contiguous branches. For an axis-0, unit-step slice, every retained row
  // is already contiguous: one bulk copy replaces generic per-element
  // stride decoding and its divisions/modulos.
  if (a.shape.size() > 0 && step[0] == 1) {
    bool only_axis_zero = true;
    for (std::size_t axis = 1; axis < a.shape.size(); ++axis) {
      only_axis_zero &= begin[axis] == 0 && o.shape[axis] == a.shape[axis];
    }
    if (only_axis_zero) {
      const auto inner = Elements(std::vector<std::int64_t>(a.shape.begin() + 1, a.shape.end()));
      std::memcpy(o.data.data(), a.data.data() + std::size_t(begin[0]) * inner,
                  o.data.size() * sizeof(float));
      return o;
    }
  }
  const auto os=Strides(o.shape),as=Strides(a.shape);
  for(std::size_t oi=0;oi<o.data.size();++oi){std::size_t ai{};for(std::size_t d=0;d<o.shape.size();++d)ai+=(begin[d]+int((oi/os[d])%std::size_t(o.shape[d]))*step[d])*as[d];o.data[oi]=a.data[ai];}
  return o;
}
Tensor QkvSlice(const Node& n, const Tensor& a) {
  const auto projection = n.attr.find("__ppocr_qkv_projection");
  if (projection == n.attr.end() || a.shape.size() != 5 ||
      a.shape[2] != 3 || projection->second.i < 0 || projection->second.i >= 3) {
    Fail("FusedQkvSlice shape");
  }
  // Input [N,T,3,H,D] becomes one already-squeezed Q/K/V branch [N,H,T,D].
  // This writes only the branch ultimately used by attention, never the
  // previous full [3,N,H,T,D] transient.
  const auto batches = std::size_t(a.shape[0]);
  const auto steps = std::size_t(a.shape[1]);
  const auto heads = std::size_t(a.shape[3]);
  const auto features = std::size_t(a.shape[4]);
  const auto scale = n.attr.contains("__ppocr_qkv_scale")
      ? n.attr.at("__ppocr_qkv_scale").f : 1.F;
  Tensor output{{a.shape[0], a.shape[3], a.shape[1], a.shape[4]},
                std::vector<float>(batches * heads * steps * features)};
  const auto q = std::size_t(projection->second.i);
  for (std::size_t batch = 0; batch < batches; ++batch)
    for (std::size_t head = 0; head < heads; ++head)
      for (std::size_t step = 0; step < steps; ++step) {
        const auto source = ((((batch * steps + step) * 3 + q) * heads + head) * features);
        const auto destination = ((((batch * heads + head) * steps + step) * features));
        if (scale == 1.F) {
          std::memcpy(output.data.data() + destination, a.data.data() + source,
                      features * sizeof(float));
        } else {
          kernels::BinaryScalar(output.data.data() + destination, a.data.data() + source,
                                features, scale, kernels::BinaryOp::mul, false);
        }
      }
  return output;
}
std::vector<Tensor> QkvSplit(const Node& n, const Tensor& a) {
  if (n.out.size() != 3 || a.shape.size() != 5 || a.shape[2] != 3) {
    Fail("FusedQkvSplit shape");
  }
  // Input [N,T,3,H,D] directly becomes the three attention tensors
  // [N,H,T,D]. This is the same mapping as three QkvSlice invocations, but
  // one producer owns the QKV source and releases it only after every branch
  // has been materialised. That removes the transient full transpose while
  // preserving each branch's exact FP32 values and later use counts.
  const auto batches = std::size_t(a.shape[0]);
  const auto steps = std::size_t(a.shape[1]);
  const auto heads = std::size_t(a.shape[3]);
  const auto features = std::size_t(a.shape[4]);
  std::vector<Tensor> result;
  result.reserve(3);
  const std::vector<std::int64_t> output_shape{a.shape[0], a.shape[3], a.shape[1], a.shape[4]};
  for (int projection = 0; projection < 3; ++projection) {
    result.emplace_back(output_shape,
                        std::vector<float>(batches * heads * steps * features));
  }
  // Preserve output-major traversal.  It writes one contiguous output stream
  // at a time, which is measurably friendlier to the hardware prefetcher than
  // interleaving three large destinations. The producer stays live only for
  // this node and is retired immediately afterwards by the executor.
  for (int projection = 0; projection < 3; ++projection) {
    auto& output = result[std::size_t(projection)];
    for (std::size_t batch = 0; batch < batches; ++batch)
      for (std::size_t head = 0; head < heads; ++head)
        for (std::size_t step = 0; step < steps; ++step) {
          const auto source = ((((batch * steps + step) * 3 + projection) * heads + head) * features);
          const auto destination = (((batch * heads + head) * steps + step) * features);
          std::memcpy(output.data.data() + destination, a.data.data() + source,
                      features * sizeof(float));
        }
  }
  return result;
}
Tensor Resize(const Node& n,const std::vector<const Tensor*>& in) {
  const auto& a=*in[0]; if(a.shape.size()!=4) Fail("Resize needs NCHW");
  int oh{},ow{}; float sh{},sw{};
  if(in.size()>3&&in[3]&&!in[3]->data.empty()) { oh=int(in[3]->data[2]); ow=int(in[3]->data[3]); sh=float(oh)/float(a.shape[2]); sw=float(ow)/float(a.shape[3]); }
  else if(in.size()>2&&in[2]) {
    sh=in[2]->data[2]; sw=in[2]->data[3];
    // ONNX Resize uses the supplied scales for its coordinate transform, but
    // the output shape is floor(input * scale) computed in float32.
    oh=int(std::floor(float(float(a.shape[2]) * sh)));
    ow=int(std::floor(float(float(a.shape[3]) * sw)));
  }
  else Fail("Resize sizes/scales missing");
  const auto mode=AttrStr(n,"mode","nearest"), ctm=AttrStr(n,"coordinate_transformation_mode","half_pixel");
  const int nn=int(a.shape[0]),cc=int(a.shape[1]),h=int(a.shape[2]),w=int(a.shape[3]);
  Tensor o{{a.shape[0],a.shape[1],oh,ow},PooledActivation(std::size_t(nn)*cc*oh*ow)};
  // The detector FPN uses only asymmetric floor nearest-neighbour upsampling
  // by integral factors (2x/4x/8x).  It is a pure NCHW row replication, so
  // precompute no coordinates and copy each source value as a contiguous run.
  // This also removes the per-output lambda/clamp overhead on the largest
  // 2x FPN resize, previously a leading detector cost on dense screenshots.
  const auto nearest=AttrStr(n,"nearest_mode","round_prefer_floor");
  if (mode=="nearest" && ctm=="asymmetric" && nearest=="floor" &&
      oh==h*int(std::lround(sh)) && ow==w*int(std::lround(sw)) &&
      sh==float(std::lround(sh)) && sw==float(std::lround(sw)) && sh>=1.F && sw>=1.F) {
    const int scale_h=int(sh), scale_w=int(sw);
    const std::size_t input_plane=std::size_t(h)*w;
    const std::size_t output_plane=std::size_t(oh)*ow;
    for (int ni=0;ni<nn;++ni) for (int c=0;c<cc;++c) {
      const float* src=a.data.data()+(std::size_t(ni)*cc+c)*input_plane;
      float* dst=o.data.data()+(std::size_t(ni)*cc+c)*output_plane;
      for (int y=0;y<h;++y) {
        float* first=dst+std::size_t(y)*scale_h*ow;
        // A scale factor of two is the dominant FPN case.  Store adjacent
        // replicas directly, avoiding std::fill_n's iterator/prologue work
        // for every source pixel while keeping the exact NCHW value order.
        if (scale_w == 2) {
          int x = 0;
          for (; x + 4 <= w; x += 4) {
            const float v0 = src[std::size_t(y) * w + x];
            const float v1 = src[std::size_t(y) * w + x + 1];
            const float v2 = src[std::size_t(y) * w + x + 2];
            const float v3 = src[std::size_t(y) * w + x + 3];
            float* row = first + std::size_t(x) * 2;
            row[0] = v0; row[1] = v0; row[2] = v1; row[3] = v1;
            row[4] = v2; row[5] = v2; row[6] = v3; row[7] = v3;
          }
          for (; x < w; ++x) {
            const float value = src[std::size_t(y) * w + x];
            first[std::size_t(x) * 2] = value;
            first[std::size_t(x) * 2 + 1] = value;
          }
        } else {
          for (int x=0;x<w;++x)
            std::fill_n(first+std::size_t(x)*scale_w,scale_w,src[std::size_t(y)*w+x]);
        }
        for (int repeat=1;repeat<scale_h;++repeat)
          std::copy_n(first,ow,dst+std::size_t(y*scale_h+repeat)*ow);
      }
    }
    return o;
  }
  auto coord=[&](int dst,float scale,int out,int size)->float { if(ctm=="align_corners") return out==1?0.F:float(dst)*float(size-1)/float(out-1); if(ctm=="asymmetric") return float(dst)/scale; return (float(dst)+.5F)/scale-.5F; };
  for(int ni=0;ni<nn;++ni) for(int c=0;c<cc;++c) for(int y=0;y<oh;++y) for(int x=0;x<ow;++x) {
    const float fy=coord(y,sh,oh,h), fx=coord(x,sw,ow,w); auto at=[&](int yy,int xx){yy=std::clamp(yy,0,h-1);xx=std::clamp(xx,0,w-1);return a.data[((ni*cc+c)*h+yy)*w+xx];}; float v{};
    if(mode=="nearest") {
      // ONNX floor/round modes are selected by the export. PP-OCRv6 uses
      // asymmetric + floor in its FPN nearest-neighbour resizes.
      const int iy=nearest=="floor" ? int(std::floor(fy)) : int(std::floor(fy+.5F));
      const int ix=nearest=="floor" ? int(std::floor(fx)) : int(std::floor(fx+.5F));
      v=at(iy,ix);
    }
    else { const int y0=int(std::floor(fy)),x0=int(std::floor(fx));const float dy=fy-y0,dx=fx-x0;v=(at(y0,x0)*(1-dx)+at(y0,x0+1)*dx)*(1-dy)+(at(y0+1,x0)*(1-dx)+at(y0+1,x0+1)*dx)*dy; }
    o.data[((ni*cc+c)*oh+y)*ow+x]=v;
  }
  return o;
}

// Detector FPN fast path for graph-proven `Resize(nearest 2x) -> Add`.
// Writing the final sum directly removes the otherwise short-lived enlarged
// feature map. It intentionally accepts only the exported asymmetric/floor
// geometry; every other ONNX Resize form falls back to the existing generic
// Resize then SIMD Add implementation below.
Tensor NearestResizeAdd(const Node& n, const std::vector<const Tensor*>& in,
                        Backend backend) {
  if (n.in.size() < 2 || in.size() != n.in.size() || !in[0] || !in.back()) {
    Fail("FusedNearestResizeAdd inputs");
  }
  const auto resize_inputs = std::vector<const Tensor*>(in.begin(), in.end() - 1);
  const auto& source = *in[0];
  const auto& residual = *in.back();
  const bool exported_2x = source.shape.size() == 4 && residual.shape.size() == 4 &&
      AttrStr(n, "mode", "nearest") == "nearest" &&
      AttrStr(n, "coordinate_transformation_mode", "half_pixel") == "asymmetric" &&
      AttrStr(n, "nearest_mode", "round_prefer_floor") == "floor";
  int output_height{};
  int output_width{};
  if (exported_2x && resize_inputs.size() > 3 && resize_inputs[3] &&
      resize_inputs[3]->data.size() >= 4) {
    output_height = int(resize_inputs[3]->data[2]);
    output_width = int(resize_inputs[3]->data[3]);
  } else if (exported_2x && resize_inputs.size() > 2 && resize_inputs[2] &&
             resize_inputs[2]->data.size() >= 4) {
    const float scale_height = resize_inputs[2]->data[2];
    const float scale_width = resize_inputs[2]->data[3];
    if (scale_height == 2.F && scale_width == 2.F) {
      output_height = int(source.shape[2]) * 2;
      output_width = int(source.shape[3]) * 2;
    }
  }
  if (exported_2x && output_height == int(source.shape[2]) * 2 &&
      output_width == int(source.shape[3]) * 2 &&
      residual.shape[0] == source.shape[0] && residual.shape[1] == source.shape[1] &&
      residual.shape[2] == output_height && residual.shape[3] == output_width) {
    const int batches = int(source.shape[0]);
    const int channels = int(source.shape[1]);
    const int input_height = int(source.shape[2]);
    const int input_width = int(source.shape[3]);
    Tensor output{residual.shape, std::vector<float>(residual.data.size())};
    // The device path is a real fused batch primitive: source and residual
    // cross the boundary once, nearest expansion stays in the shader, and
    // the completed sum is read back once.  Hybrid only selects it after the
    // exact shape's full synchronous transfer/execute/readback benchmark is
    // no slower than the CPU fused kernel.
    bool select_gpu{};
    if (backend == Backend::hybrid) {
      const auto shape_key = (static_cast<std::uint64_t>(batches) << 48u) ^
          (static_cast<std::uint64_t>(channels) << 32u) ^
          (static_cast<std::uint64_t>(input_height) << 16u) ^
          static_cast<std::uint64_t>(input_width);
      const auto key = WithHybridAdmissionContext(shape_key);
      static std::mutex admission_mutex;
      static std::unordered_map<std::uint64_t, bool> admitted;
      std::lock_guard lock(admission_mutex);
      const auto found = admitted.find(key);
      if (found != admitted.end()) select_gpu = found->second;
      else {
        select_gpu = VulkanNearestResize2xAddBatchNoSlowerThanCpu(
            static_cast<std::size_t>(batches), channels, input_height, input_width);
        admitted.emplace(key, select_gpu);
      }
    }
    if (select_gpu && VulkanNearestResize2xAddBatch(
            output.data.data(), source.data.data(), residual.data.data(),
            static_cast<std::size_t>(batches), channels, input_height, input_width)) {
      return output;
    }
    kernels::NearestResize2xAdd(output.data.data(), source.data.data(), residual.data.data(),
                                batches, channels, input_height, input_width);
    return output;
  }
  Tensor output = Resize(n, resize_inputs);
  if (output.shape != residual.shape) Fail("FusedNearestResizeAdd shape");
  kernels::BinaryInplace(output.data.data(), residual.data.data(), output.data.size(),
                         kernels::BinaryOp::add);
  return output;
}

Tensor MatMul(const Tensor& a, const Tensor& b, const Tensor* bias=nullptr,
              Backend backend=Backend::cpu_only, bool immutable_parameters=false) {
  if(a.shape.size()<2||b.shape.size()<2) Fail("MatMul rank < 2");
  const int m=int(a.shape[a.shape.size()-2]), k=int(a.shape.back()), k2=int(b.shape[b.shape.size()-2]), n=int(b.shape.back());
  if(k!=k2) Fail("MatMul mismatch");
  const std::vector<std::int64_t> ab(a.shape.begin(),a.shape.end()-2), bb(b.shape.begin(),b.shape.end()-2);
  const auto batch=BroadcastShape(ab,bb); Tensor o; o.shape=batch; o.shape.push_back(m); o.shape.push_back(n);
  // Every MatMul path below writes every output element, starting from its
  // optional bias inside Gemm().  Avoid zero-initializing an activation that
  // is immediately overwritten: wide recognition projections make that
  // redundant bandwidth and transient allocator work visible for both a
  // singleton crop and a true NCHW batch.
  o.data.resize(Elements(o.shape));
  const auto batches=Elements(batch);
  static const bool fold_constant_weight_batches =
      std::getenv("PPOCR_DISABLE_MATMUL_BATCH_FOLD") == nullptr;
  // Transformer projections normally multiply a batched activation by one
  // immutable rank-2 weight matrix. All A batches and output rows are already
  // contiguous in row-major storage, so execute them as one tall GEMM instead
  // of making one tiny GEMM call per batch. Besides removing repeated dispatch
  // overhead, this lets the existing SIMD/NEON GEMM choose a useful row split
  // for an entire recognition batch. Each output row retains its original
  // left-to-right K reduction and the same shared bias values.
  if (fold_constant_weight_batches && bb.empty()) {
    if (batches > std::size_t(std::numeric_limits<int>::max()) / std::size_t(m)) {
      Fail("MatMul batch rows overflow");
    }
    const int rows=int(batches)*m;
    bool select_gpu{};
    if (backend==Backend::hybrid && rows >= 32 && n >= 256 && k >= 64) {
      const auto shape_key=(static_cast<std::uint64_t>(rows)<<42u) ^
          (static_cast<std::uint64_t>(k)<<21u) ^ static_cast<std::uint64_t>(n) ^
          (bias?1u:0u);
      const auto key = WithHybridAdmissionContext(shape_key);
      static std::mutex admission_mutex;
      static std::unordered_map<std::uint64_t,bool> admitted;
      std::lock_guard lock(admission_mutex);
      const auto found=admitted.find(key);
      if (found!=admitted.end()) select_gpu=found->second;
      else {
        select_gpu=VulkanGemmNoSlowerThanCpu(rows,k,n,nullptr,nullptr,immutable_parameters);
        admitted.emplace(key,select_gpu);
      }
    }
    if (!(select_gpu && VulkanGemm(o.data.data(),a.data.data(),b.data.data(),
                                   bias?bias->data.data():nullptr,rows,k,n,
                                   immutable_parameters))) {
      kernels::Gemm(o.data.data(), a.data.data(), b.data.data(),
                    bias ? bias->data.data() : nullptr, rows, n, k);
    }
    return o;
  }
  const auto astr=Strides(ab), bstr=Strides(bb);
  const auto apad=batch.size()-ab.size(), bpad=batch.size()-bb.size();
  // Transformer projections repeatedly share the same broadcasted batch
  // shape. Computing its stride vector inside `batch_off` once per batch used
  // to allocate a small vector for every MatMul invocation; retain it with the
  // already-computed operand strides instead.
  const auto batch_strides = Strides(batch);
  auto batch_off=[&](std::size_t bi,const std::vector<std::int64_t>& source,const std::vector<std::size_t>& stride,std::size_t pad){std::size_t off{};for(std::size_t d=0;d<batch.size();++d){if(d>=pad&&source[d-pad]!=1)off+=((bi/batch_strides[d])%std::size_t(batch[d]))*stride[d-pad];}return off;};
  for(std::size_t bi=0;bi<batches;++bi) {
    const auto ao=batch_off(bi,ab,astr,apad)*std::size_t(m)*k;
    const auto bo=batch_off(bi,bb,bstr,bpad)*std::size_t(k)*n;
    kernels::Gemm(o.data.data()+bi*std::size_t(m)*n, a.data.data()+ao,
                  b.data.data()+bo, bias ? bias->data.data() : nullptr, m, n, k);
  }
  return o;
}

Tensor MatMulBiasSwish(const Tensor& a, const Tensor& b, const Tensor& bias,
                       Backend backend=Backend::cpu_only, bool immutable_parameters=false) {
  if (a.shape.size() < 2 || b.shape.size() != 2) Fail("FusedMatMulBiasSwish rank");
  const int k = int(a.shape.back());
  const int k2 = int(b.shape[0]);
  const int n = int(b.shape[1]);
  if (k != k2 || bias.data.size() != std::size_t(n)) Fail("FusedMatMulBiasSwish shape");
  // PP-OCRv6 MLP gates use a constant rank-2 projection without batch
  // broadcasting. Fuse the exact Swish writeback into the GEMM path, so the
  // pre-activation never makes a separate full-tensor round trip through
  // memory. Retain the generic MatMul fallback for every other ONNX layout.
  const std::size_t rows = a.data.size() / std::size_t(k);
  if (rows * std::size_t(k) != a.data.size()) Fail("FusedMatMulBiasSwish input");
  Tensor output{a.shape, {}};
  output.shape.back() = n;
  output.data.resize(rows * std::size_t(n));
  // Exact libm Swish is still scalar on the supported toolchains. The
  // established wide-GEMM followed by its compact in-place pass benchmarks
  // better than interleaving scalar exponentials into the vector dot product,
  // so retain the existing fast path until a measured architecture-wide win.
  bool select_gpu{};
  if (backend==Backend::hybrid && rows >= 32 && n >= 256 && k >= 64) {
    const auto shape_key=(static_cast<std::uint64_t>(rows)<<42u) ^
        (static_cast<std::uint64_t>(k)<<21u) ^ static_cast<std::uint64_t>(n) ^ 1u;
    const auto key = WithHybridAdmissionContext(shape_key);
    static std::mutex admission_mutex;
    static std::unordered_map<std::uint64_t,bool> admitted;
    std::lock_guard lock(admission_mutex);
    const auto found=admitted.find(key);
    if (found!=admitted.end()) select_gpu=found->second;
    else {
      select_gpu=VulkanGemmSwishNoSlowerThanCpu(int(rows),k,n,nullptr,nullptr,immutable_parameters);
      admitted.emplace(key,select_gpu);
    }
  }
  if (!(select_gpu && VulkanGemmSwish(output.data.data(),a.data.data(),b.data.data(),bias.data.data(),
                                      int(rows),k,n,immutable_parameters))) {
    kernels::Gemm(output.data.data(), a.data.data(), b.data.data(), bias.data.data(),
                  int(rows), n, k);
    kernels::Swish(output.data.data(), output.data.data(), output.data.size());
    return output;
  }
  return output;
}

Tensor LayerNorm(const Node& n, const std::vector<const Tensor*>& in) {
  if (in.size() != 3 || !in[0] || !in[1] || !in[2] || in[0]->shape.empty()) {
    Fail("FusedLayerNorm inputs");
  }
  const auto& a = *in[0];
  const auto& gamma = *in[1];
  const auto& beta = *in[2];
  const auto width = static_cast<std::size_t>(a.shape.back());
  if (width == 0 || gamma.data.size() != width || beta.data.size() != width ||
      a.data.size() % width != 0) Fail("FusedLayerNorm shape");
  Tensor output{a.shape, std::vector<float>(a.data.size())};
  kernels::LayerNorm(output.data.data(), a.data.data(), gamma.data.data(), beta.data.data(),
                     a.data.size() / width, width, Attr(n, "epsilon").f);
  return output;
}

Tensor Transpose(const Node& n, const Tensor& a) {
  auto p = AttrInts(n, "perm", {});
  if (p.empty()) {
    p.resize(a.shape.size());
    for (std::size_t i = 0; i < p.size(); ++i) p[i] = std::int64_t(p.size() - i - 1);
  }
  if (p.size() != a.shape.size()) Fail("bad transpose");
  Tensor o;
  for (const auto axis : p) o.shape.push_back(a.shape[axis]);
  o.data.resize(a.data.size());

  // `[N,T,C] <-> [N,C,T]` occurs in both the convolutional recognizer head
  // and the final CTC projection. Every source/destination row is contiguous,
  // so avoid the generic three-axis index decoder.
  if (a.shape.size() == 3 && p == std::vector<std::int64_t>{0, 2, 1}) {
    const auto batches = std::size_t(a.shape[0]);
    const auto rows = std::size_t(a.shape[1]);
    const auto columns = std::size_t(a.shape[2]);
    constexpr std::size_t tile = 32;
    for (std::size_t batch = 0; batch < batches; ++batch) {
      const float* source = a.data.data() + batch * rows * columns;
      float* destination = o.data.data() + batch * columns * rows;
      for (std::size_t row0 = 0; row0 < rows; row0 += tile) {
        const auto row_limit = std::min(rows, row0 + tile);
        for (std::size_t column0 = 0; column0 < columns; column0 += tile) {
          const auto column_limit = std::min(columns, column0 + tile);
          for (std::size_t row = row0; row < row_limit; ++row) {
            const float* source_row = source + row * columns + column0;
            for (std::size_t column = column0; column < column_limit; ++column)
              destination[column * rows + row] = source_row[column - column0];
          }
        }
      }
    }
    return o;
  }

  // Attention Q/K/V blocks have a 4-D layout exchange with a contiguous
  // final feature vector. Copy that vector as one unit instead of decoding
  // every scalar index through four divide/modulo operations. These cases are
  // exact permutations and deliberately keep the generic fallback below for
  // all other ONNX layouts.
  if (a.shape.size() == 4 && p == std::vector<std::int64_t>{0, 2, 1, 3}) {
    const auto batches = std::size_t(a.shape[0]);
    const auto heads = std::size_t(a.shape[1]);
    const auto steps = std::size_t(a.shape[2]);
    const auto features = std::size_t(a.shape[3]);
    for (std::size_t batch = 0; batch < batches; ++batch)
      for (std::size_t step = 0; step < steps; ++step)
        for (std::size_t head = 0; head < heads; ++head) {
          const auto src = ((batch * heads + head) * steps + step) * features;
          const auto dst = ((batch * steps + step) * heads + head) * features;
          std::memcpy(o.data.data() + dst, a.data.data() + src, features * sizeof(float));
        }
    return o;
  }
  if (a.shape.size() == 4 && p == std::vector<std::int64_t>{0, 1, 3, 2}) {
    const auto batches = std::size_t(a.shape[0]);
    const auto heads = std::size_t(a.shape[1]);
    const auto features = std::size_t(a.shape[2]);
    const auto steps = std::size_t(a.shape[3]);
    constexpr std::size_t tile = 32;
    for (std::size_t batch = 0; batch < batches; ++batch)
      for (std::size_t head = 0; head < heads; ++head) {
        const float* source = a.data.data() + (batch * heads + head) * features * steps;
        float* destination = o.data.data() + (batch * heads + head) * steps * features;
        for (std::size_t feature0 = 0; feature0 < features; feature0 += tile) {
          const auto feature_limit = std::min(features, feature0 + tile);
          for (std::size_t step0 = 0; step0 < steps; step0 += tile) {
            const auto step_limit = std::min(steps, step0 + tile);
            for (std::size_t feature = feature0; feature < feature_limit; ++feature) {
              const float* source_row = source + feature * steps + step0;
              for (std::size_t step = step0; step < step_limit; ++step)
                destination[step * features + feature] = source_row[step - step0];
            }
          }
        }
      }
    return o;
  }
  // The small/medium recognizer's attention projection exposes Q/K/V as
  // [N,T,3,H,D] and then permutes it to [3,N,H,T,D].  `D` remains a
  // contiguous head vector in both layouts, so copying the vector as one
  // unit avoids five per-scalar divide/modulo decodes.  The formula is the
  // exact ONNX mapping: output[q,n,h,t,:] = input[n,t,q,h,:].  Keep the
  // generic implementation below for every other rank/permutation and an
  // explicit deployment A/B switch for this model-specific fast path.
  static const bool qkv_transpose_fastpath =
      std::getenv("PPOCR_DISABLE_QKV_TRANSPOSE_FASTPATH") == nullptr;
  if (qkv_transpose_fastpath && a.shape.size() == 5 &&
      p == std::vector<std::int64_t>{2, 0, 3, 1, 4}) {
    const auto batches = std::size_t(a.shape[0]);
    const auto steps = std::size_t(a.shape[1]);
    const auto projections = std::size_t(a.shape[2]);
    const auto heads = std::size_t(a.shape[3]);
    const auto features = std::size_t(a.shape[4]);
    if (projections == 3 && features != 0) {
      for (std::size_t projection = 0; projection < projections; ++projection)
        for (std::size_t batch = 0; batch < batches; ++batch)
          for (std::size_t head = 0; head < heads; ++head)
            for (std::size_t step = 0; step < steps; ++step) {
              const auto source = ((((batch * steps + step) * projections + projection) *
                                    heads + head) * features);
              const auto destination = ((((projection * batches + batch) * heads + head) *
                                         steps + step) * features);
              std::memcpy(o.data.data() + destination, a.data.data() + source,
                          features * sizeof(float));
            }
      return o;
    }
  }
  const auto os = Strides(o.shape), is = Strides(a.shape);
  for (std::size_t oi = 0; oi < o.data.size(); ++oi) {
    std::size_t off{};
    for (std::size_t d = 0; d < p.size(); ++d) {
      off += ((oi / os[d]) % std::size_t(o.shape[d])) * is[p[d]];
    }
    o.data[oi] = a.data[off];
  }
  return o;
}

Tensor Reshape(const Tensor& a,const Tensor& spec){Tensor o=a;o.shape.clear();std::int64_t known=1;int infer=-1;for(std::size_t i=0;i<spec.data.size();++i){auto d=std::int64_t(spec.data[i]);if(d==-1){infer=int(i);o.shape.push_back(1);}else{if(d==0)d=a.shape[i];o.shape.push_back(d);known*=d;}}if(infer>=0)o.shape[infer]=std::int64_t(a.data.size()/known);if(Elements(o.shape)!=a.data.size())Fail("reshape mismatch");return o;}
Tensor Concat(const Node& n,const std::vector<const Tensor*>& in) {
  if (in.empty() || !in[0]) Fail("Concat without inputs");
  const int ax=Axis(AttrInt(n,"axis",0),int(in[0]->shape.size()));
  Tensor o; o.shape=in[0]->shape; o.shape[ax]=0;
  for (const auto* t : in) {
    if (!t || t->shape.size()!=o.shape.size() || t->data.size()!=Elements(t->shape)) {
      std::ostringstream msg; msg << "Concat corrupt input: ";
      if (!t) msg << "null"; else { msg << "shape="; for (auto q : t->shape) msg << q << ','; msg << " data=" << t->data.size() << " expected=" << Elements(t->shape); }
      Fail(msg.str());
    }
    for (std::size_t d=0;d<t->shape.size();++d) if(int(d)!=ax && t->shape[d]!=in[0]->shape[d]) {
      std::ostringstream msg; msg << "Concat dimension mismatch at axis " << ax << ": ";
      for (auto q : in[0]->shape) msg << q << ','; msg << " vs "; for (auto q : t->shape) msg << q << ',';
      Fail(msg.str());
    }
    o.shape[ax]+=t->shape[ax];
  }
  o.data = PooledActivation(Elements(o.shape));
  const auto outer=Elements(std::vector<std::int64_t>(o.shape.begin(),o.shape.begin()+ax));
  const auto inner=Elements(std::vector<std::int64_t>(o.shape.begin()+ax+1,o.shape.end()));
  for(std::size_t z=0;z<outer;++z) {
    std::size_t dst=z*std::size_t(o.shape[ax])*inner;
    for(const auto* t:in) { const auto count=std::size_t(t->shape[ax])*inner; std::copy_n(t->data.begin()+std::ptrdiff_t(z*count),count,o.data.begin()+std::ptrdiff_t(dst)); dst+=count; }
  }
  return o;
}

// Concatenation can retain its first dynamic input whenever that input is at
// final use. After expanding the vector, move its outer blocks backwards,
// then fill the remaining channel/row spans from the other inputs. Backward
// moves are overlap-safe for every legal ONNX concat axis, so this removes a
// full output activation allocation/copy even for the detector's channel-axis
// FPN Concats (the medium profile's visible generic Concat cost).
bool ConcatIntoFirst(const Node& n, const std::vector<const Tensor*>& in,
                     Tensor& first) {
  if (in.empty() || !in[0] || in[0] != &first || first.shape.empty()) return false;
  const int axis = Axis(AttrInt(n, "axis", 0), int(first.shape.size()));
  std::vector<std::int64_t> shape = first.shape;
  std::size_t total = first.data.size();
  std::int64_t axis_extent = first.shape[axis];
  for (std::size_t input_index = 1; input_index < in.size(); ++input_index) {
    const Tensor* value = in[input_index];
    if (!value || value == &first || value->shape.size() != shape.size() ||
        value->data.size() != Elements(value->shape)) return false;
    for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
      if (int(dimension) != axis && value->shape[dimension] != shape[dimension]) return false;
    }
    if (value->shape[axis] < 0 || axis_extent > std::numeric_limits<std::int64_t>::max() - value->shape[axis] ||
        value->data.size() > std::numeric_limits<std::size_t>::max() - total) return false;
    axis_extent += value->shape[axis];
    total += value->data.size();
  }
  shape[axis] = axis_extent;
  if (Elements(shape) != total) return false;
  const auto outer = Elements(std::vector<std::int64_t>(first.shape.begin(), first.shape.begin() + axis));
  const auto inner = Elements(std::vector<std::int64_t>(first.shape.begin() + axis + 1, first.shape.end()));
  const auto first_span = std::size_t(first.shape[axis]) * inner;
  const auto output_span = std::size_t(axis_extent) * inner;
  if (outer == 0 || first_span == 0 || output_span < first_span ||
      first.data.size() != outer * first_span || total != outer * output_span) return false;
  ResizeUninitialized(first.data, total);
  for (std::size_t block = outer; block-- > 0;) {
    std::memmove(first.data.data() + block * output_span,
                 first.data.data() + block * first_span, first_span * sizeof(float));
  }
  for (std::size_t block = 0; block < outer; ++block) {
    std::size_t offset = block * output_span + first_span;
    for (std::size_t input_index = 1; input_index < in.size(); ++input_index) {
      const Tensor& value = *in[input_index];
      const auto span = std::size_t(value.shape[axis]) * inner;
      std::memcpy(first.data.data() + offset, value.data.data() + block * span,
                  span * sizeof(float));
      offset += span;
    }
  }
  first.shape = std::move(shape);
  return true;
}

Tensor Softmax(const Node& n,const Tensor& a,int opset){const int ax=Axis(AttrInt(n,"axis",opset>=13?-1:1),int(a.shape.size()));const auto inner=Elements(std::vector<std::int64_t>(a.shape.begin()+ax+1,a.shape.end()));const auto dim=std::size_t(a.shape[ax]),outer=a.data.size()/(dim*inner);Tensor o=a;if(inner==1){kernels::SoftmaxRowsInplace(o.data.data(),outer,dim);return o;}for(std::size_t z=0;z<outer;++z)for(std::size_t q=0;q<inner;++q){float m=-std::numeric_limits<float>::infinity();for(std::size_t i=0;i<dim;++i)m=std::max(m,a.data[(z*dim+i)*inner+q]);float s{};for(std::size_t i=0;i<dim;++i)s+=(o.data[(z*dim+i)*inner+q]=std::exp(a.data[(z*dim+i)*inner+q]-m));for(std::size_t i=0;i<dim;++i)o.data[(z*dim+i)*inner+q]/=s;}return o;}

Tensor PointwiseConvHardSwish(const Node& n, const std::vector<const Tensor*>& in,
                              Backend backend, bool immutable_parameters) {
  if (in.size() < 2 || !in[0] || !in[1]) Fail("FusedConvHardSwish inputs");
  const auto& a = *in[0];
  const auto& w = *in[1];
  if (a.shape.size() != 4 || w.shape.size() != 4 || w.shape[2] != 1 || w.shape[3] != 1 ||
      AttrInt(n, "group", 1) != 1 || AttrStr(n, "auto_pad", "NOTSET") != "NOTSET") {
    Tensor output = Conv(n, in, false, false, false, .2F, .5F, nullptr, backend,
                         immutable_parameters, false, true);
    return output;
  }
  const auto strides = AttrInts(n, "strides", {1, 1});
  const auto pads = AttrInts(n, "pads", {0, 0, 0, 0});
  const auto dilations = AttrInts(n, "dilations", {1, 1});
  const int batches = int(a.shape[0]), input_channels = int(a.shape[1]);
  const int output_channels = int(w.shape[0]);
  if (strides != std::vector<std::int64_t>{1, 1} || pads != std::vector<std::int64_t>{0, 0, 0, 0} ||
      dilations != std::vector<std::int64_t>{1, 1} || input_channels != int(w.shape[1])) {
    Tensor output = Conv(n, in, false, false, false, .2F, .5F, nullptr, backend,
                         immutable_parameters, false, true);
    return output;
  }
  const float* bias = in.size() > 2 && in[2] ? in[2]->data.data() : nullptr;
  if (bias && in[2]->data.size() != std::size_t(output_channels)) Fail("Conv bias shape");
  const auto plane = std::size_t(a.shape[2]) * a.shape[3];
  Tensor output{{a.shape[0], w.shape[0], a.shape[2], a.shape[3]},
                PooledActivation(std::size_t(batches) * output_channels * plane)};
  if (bias && TryHybridVulkanPointwiseConv(
          backend, output, a, w, in[2], false, immutable_parameters,
          false, false, false, 1.F / 6.F, .5F, true)) {
    return output;
  }
  kernels::PointwiseConvHardSwishBatch(output.data.data(), a.data.data(), w.data.data(), bias,
                                       batches, output_channels, input_channels, plane);
  return output;
}

Tensor Execute(const Node& n, const std::vector<const Tensor*>& in, int opset,
               Backend backend = Backend::cpu_only, bool immutable_parameters = false) {
  if(n.op=="Identity")return *in[0]; if(n.op=="FusedConvRelu"){bool applied{};Tensor o=Conv(n,in,true,false,false,.2F,.5F,&applied,backend,immutable_parameters); if(!applied && !(o.shape.size()==4 && n.in.size()>1 && in[1]->shape.size()==4 && in[1]->shape[2]==1 && in[1]->shape[3]==1)) kernels::Relu(o.data.data(),o.data.data(),o.data.size()); return o;} if(n.op=="FusedConvChannelBias"){
    // Fuse the statically folded channel bias into Conv's accumulator rather
    // than reintroducing a full NCHW AddChannelBias pass at execution time.
    // The graph rewriter guarantees input 2 is the immutable folded [M]
    // bias, so Conv already has the exact parameters it needs.
    return Conv(n,in,false,false,false,.2F,.5F,nullptr,backend,immutable_parameters);
  } if(n.op=="FusedConvGelu"){return Conv(n,in,false,false,false,.2F,.5F,nullptr,backend,immutable_parameters,false,false,true);} if(n.op=="FusedDepthwiseExpandGeluProjectAdd")return DepthwiseExpandGeluProjectAddOp(n,in); if(n.op=="FusedExpandGeluProjectAdd")return ExpandGeluProjectAddOp(n,in); if(n.op=="FusedMaxPoolConcatConv")return MaxPoolConcatConv(n,in,false,backend,immutable_parameters); if(n.op=="FusedMaxPoolConcatConvRelu")return MaxPoolConcatConv(n,in,true,backend,immutable_parameters); if(n.op=="FusedMaxPoolConcat")return MaxPoolConcat(n,in); if(n.op=="FusedConvMaxPoolConcat")return ConvMaxPoolConcat(n,in); if(n.op=="FusedConcatConv")return ConcatConv(n,in,false,backend,immutable_parameters); if(n.op=="FusedConcatConvRelu")return ConcatConv(n,in,true,backend,immutable_parameters); if(n.op=="FusedDepthwisePointwiseConv")return DepthwisePointwiseConv(n,in,backend,immutable_parameters); if(n.op=="FusedDepthwisePointwiseConvGelu")return DepthwisePointwiseConv(n,in,backend,immutable_parameters,DwPwActivation::gelu); if(n.op=="FusedDepthwisePointwiseConvHardSwish")return DepthwisePointwiseConv(n,in,backend,immutable_parameters,DwPwActivation::hard_swish); if(n.op=="FusedDepthwisePointwiseConvRelu")return DepthwisePointwiseConv(n,in,backend,immutable_parameters,DwPwActivation::relu); if(n.op=="FusedPointwiseDepthwiseConv")return PointwiseDepthwiseConvOp(n,in,false,backend,immutable_parameters); if(n.op=="FusedPointwiseDepthwiseConvRelu")return PointwiseDepthwiseConvOp(n,in,true,backend,immutable_parameters); if(n.op=="FusedPointwiseConvAdd")return PointwiseConvAdd(n,in,backend,immutable_parameters); if(n.op=="FusedPointwiseConvAddRelu")return PointwiseConvAddRelu(n,in,backend,immutable_parameters); if(n.op=="FusedPointwiseConvAddSwish")return PointwiseConvAddSwish(n,in,backend,immutable_parameters); if(n.op=="FusedConvTranspose2x2Add")return ConvTranspose2x2Add(n,in,backend,immutable_parameters); if(n.op=="FusedSEGateMul")return SqueezeExcitationGateMul(n,in); if(n.op=="FusedConvTransposeRelu"){static const bool fused_act=std::getenv("PPOCR_DISABLE_CONVTRANSPOSE_PLANE_ACT")==nullptr; if(fused_act) return ConvTranspose(n,in,backend,immutable_parameters,1); Tensor o=ConvTranspose(n,in,backend,immutable_parameters); kernels::Relu(o.data.data(),o.data.data(),o.data.size()); return o;} if(n.op=="FusedConvTransposeSigmoid"){static const bool fused_act=std::getenv("PPOCR_DISABLE_CONVTRANSPOSE_PLANE_ACT")==nullptr; if(fused_act) return ConvTranspose(n,in,backend,immutable_parameters,2); Tensor o=ConvTranspose(n,in,backend,immutable_parameters); kernels::Sigmoid(o.data.data(),o.data.data(),o.data.size()); return o;} if(n.op=="FusedConvSigmoid")return Conv(n,in,false,true,false,.2F,.5F,nullptr,backend,immutable_parameters); if(n.op=="FusedConvHardSigmoid")return Conv(n,in,false,false,true,Attr(n,"__ppocr_fused_alpha").f,Attr(n,"__ppocr_fused_beta").f,nullptr,backend,immutable_parameters); if(n.op=="FusedConvHardSwish")return PointwiseConvHardSwish(n,in,backend,immutable_parameters); if(n.op=="FusedConvSwish")return Conv(n,in,false,false,false,.2F,.5F,nullptr,backend,immutable_parameters,true); if(n.op=="Conv")return Conv(n,in,false,false,false,.2F,.5F,nullptr,backend,immutable_parameters); if(n.op=="ConvTranspose")return ConvTranspose(n,in,backend,immutable_parameters); if(n.op=="FusedMatMulBias")return MatMul(*in[0],*in[1],in[2],backend,immutable_parameters); if(n.op=="FusedMatMulBiasSwish")return MatMulBiasSwish(*in[0],*in[1],*in[2],backend,immutable_parameters); if(n.op=="MatMul")return MatMul(*in[0],*in[1],nullptr,backend,immutable_parameters); if(n.op=="FusedLayerNorm")return LayerNorm(n,in); if(n.op=="MaxPool")return Pool(n,*in[0],false); if(n.op=="AveragePool")return Pool(n,*in[0],true); if(n.op=="GlobalAveragePool")return GlobalAveragePool(*in[0]); if(n.op=="FusedBatchNormGelu")return BatchNormGelu(n,in); if(n.op=="FusedBatchNormSwish")return BatchNormSwish(n,in); if(n.op=="FusedBatchNormHardSwish"){Tensor o=*in[0]; const auto channels=std::size_t(o.shape[1]); const auto spatial=o.data.size()/(std::size_t(o.shape[0])*channels); kernels::BatchNormAffine(o.data.data(), in[0]->data.data(), in[1]->data.data(), in[2]->data.data(), int(o.shape[0]), int(channels), spatial); kernels::HardSwish(o.data.data(), o.data.data(), o.data.size()); return o;} if(n.op=="FusedBatchNormSwishAdd")return BatchNormSwishAdd(n,in); if(n.op=="BatchNormalization")return BatchNorm(n,in);
  if(n.op=="FusedGelu"){Tensor o=*in[0];const auto logical=o.shape.empty()?o.data.size():o.data.size()/std::size_t(o.shape[0]);kernels::Gelu(o.data.data(),in[0]->data.data(),o.data.size(),logical);return o;} if(n.op=="FusedHardSwish"){Tensor o=*in[0];kernels::HardSwish(o.data.data(),in[0]->data.data(),o.data.size());return o;} if(n.op=="FusedScaleShift"){Tensor o=*in[0];kernels::ScaleShift(o.data.data(),in[0]->data.data(),o.data.size(),Attr(n,"scale").f,Attr(n,"shift").f);return o;} if(n.op=="Relu"){Tensor o=*in[0];kernels::Relu(o.data.data(),in[0]->data.data(),o.data.size());return o;} if(n.op=="Sigmoid"){Tensor o=*in[0];kernels::Sigmoid(o.data.data(),in[0]->data.data(),o.data.size());return o;} if(n.op=="Erf")return Unary(*in[0],[](float x){return std::erf(x);}); if(n.op=="Sqrt")return Unary(*in[0],[](float x){return std::sqrt(x);}); if(n.op=="Pow"&&in[1]->data.size()==1&&in[1]->data[0]==2.F){Tensor o=*in[0];kernels::Square(o.data.data(),in[0]->data.data(),o.data.size());return o;} if(n.op=="Pow")return Binary(*in[0],*in[1],[](float a,float b){return std::pow(a,b);}); if(n.op=="HardSigmoid"){const auto alpha=n.attr.contains("alpha")?Attr(n,"alpha").f:.2F,beta=n.attr.contains("beta")?Attr(n,"beta").f:.5F;return Unary(*in[0],[=](float x){return std::clamp(alpha*x+beta,0.F,1.F);});}
  if(n.op=="Add")return Binary(*in[0],*in[1],kernels::BinaryOp::add);if(n.op=="Sub")return Binary(*in[0],*in[1],kernels::BinaryOp::sub);if(n.op=="Mul")return Binary(*in[0],*in[1],kernels::BinaryOp::mul);if(n.op=="Div")return Binary(*in[0],*in[1],kernels::BinaryOp::div);
  if(n.op=="Transpose")return Transpose(n,*in[0]); if(n.op=="Reshape")return Reshape(*in[0],*in[1]); if(n.op=="Concat")return Concat(n,in); if(n.op=="Softmax")return Softmax(n,*in[0],opset); if(n.op=="ReduceMean")return ReduceMean(n,in); if(n.op=="Shape")return Shape(*in[0]); if(n.op=="Slice")return Slice(*in[0],in); if(n.op=="FusedQkvSlice")return QkvSlice(n,*in[0]); if(n.op=="Squeeze")return Squeeze(n,in,false); if(n.op=="Unsqueeze")return Squeeze(n,in,true); if(n.op=="Resize")return Resize(n,in); if(n.op=="FusedNearestResizeAdd")return NearestResizeAdd(n,in,backend); if(n.op=="FusedConcatConvDualTranspose")return ConcatConvDualTranspose(n,in,backend,immutable_parameters); if(n.op=="FusedConvTransposeChain")return ConvTransposeChainOp(n,in,backend,immutable_parameters); if(n.op=="FusedDepthwiseHardSwish")return Conv(n,in,false,false,false,.2F,.5F,nullptr,backend,immutable_parameters,false,true);
  Fail("unsupported PP-OCRv6 operator " + n.op);
}
}  // namespace

// A hybrid segment must be faster end-to-end, not just faster while already
// resident on the device. Cache that admission decision per tensor size/op so
// concurrent recognizer batches do not repeatedly benchmark the same shape.
// This helper deliberately covers only equal-shaped binary tensors: Vulkan's
// current kernel has no broadcast-stride representation, and falling back is
// always correct for the remaining ONNX layouts.
bool TryHybridVulkanBinaryInplace(Backend backend, Tensor& left, const Tensor& right,
                                  kernels::BinaryOp operation,
                                  bool immutable_right) noexcept {
  const std::size_t elements = left.data.size();
  if (backend != Backend::hybrid || left.shape != right.shape ||
      !left.data.data() || !right.data.data() || elements < 65536) return false;

  // OCR recognition already supplies equal-width crops as an NCHW batch. Map
  // that leading N dimension to Vulkan's Y dimension so one command submits
  // the complete crop batch rather than serialising N independent dispatches.
  // A rank-zero/unknown leading dimension remains a single contiguous batch.
  const std::size_t batches = !left.shape.empty() && left.shape[0] > 1
      ? static_cast<std::size_t>(left.shape[0]) : 1;
  if (batches == 0 || elements % batches != 0) return false;
  const std::size_t count = elements / batches;
  const auto shape_key = (static_cast<std::uint64_t>(count) << 8) |
                   (static_cast<std::uint64_t>(batches) << 2) |
                   static_cast<std::uint64_t>(operation);
  const auto key = WithHybridAdmissionContext(shape_key);
  static std::mutex admission_mutex;
  static std::unordered_map<std::uint64_t, bool> admitted;
  bool select_gpu{};
  {
    std::lock_guard lock(admission_mutex);
    const auto found = admitted.find(key);
    if (found != admitted.end()) select_gpu = found->second;
    else {
      // Keep the expensive initial probe outside the regular executor only
      // once per deployment shape. Its result includes H2D/dispatch/D2H.
      double gpu_ms{}, cpu_ms{};
      if (batches == 1) {
        select_gpu = VulkanBinaryNoSlowerThanCpu(count, operation, &gpu_ms, &cpu_ms);
      } else {
        select_gpu = VulkanBinaryChainBatchNoSlowerThanCpu(
            count, batches, {operation}, &gpu_ms, &cpu_ms);
      }
      admitted.emplace(key, select_gpu);
    }
  }
  if (!select_gpu) return false;
  if (batches == 1) {
    return VulkanBinary(left.data.data(), left.data.data(), right.data.data(), count, operation,
                        immutable_right);
  }
  return VulkanBinaryChainBatch(left.data.data(), left.data.data(), right.data.data(),
                                count, batches, {operation}, immutable_right);
}

// Same policy for NCHW-style suffix broadcasts. The right tensor remains
// compact on the GPU (for example [N,C,1,1] instead of an H*W expansion), so
// this makes accelerator offload viable for channel gates without inflating
// either transfer size or transient host memory. Only exact contiguous runs
// are accepted; all other ONNX broadcast layouts retain the CPU SIMD path.
bool TryHybridVulkanBinaryBroadcastRightInplace(
    Backend backend, Tensor& left, const Tensor& right, std::size_t right_repeat,
    kernels::BinaryOp operation, bool immutable_right) noexcept {
  const std::size_t elements = left.data.size();
  if (backend != Backend::hybrid || !left.data.data() || !right.data.data() ||
      right_repeat < 4 || right.data.empty() || elements < 65536 ||
      elements % right_repeat != 0) return false;
  const std::size_t batches = !left.shape.empty() && left.shape[0] > 1
      ? static_cast<std::size_t>(left.shape[0]) : 1;
  if (batches == 0 || elements % batches != 0) return false;
  const std::size_t count = elements / batches;
  if (count % right_repeat != 0) return false;
  const std::size_t right_per_batch = count / right_repeat;
  // A leading singleton RHS (the usual learned [1,C,1,1] scale/bias) is
  // semantically shared by every recognition crop. Preserve that compact
  // storage across the host/GPU boundary; an N-sized RHS still takes the
  // independent-per-batch layout.
  const bool shared_right = right.data.size() == right_per_batch;
  if (!shared_right && right.data.size() != batches * right_per_batch) {
    return false;
  }
  const auto shape_key = (static_cast<std::uint64_t>(count) << 32) ^
                   (static_cast<std::uint64_t>(batches) << 20) ^
                   (static_cast<std::uint64_t>(right_repeat) << 2) ^
                   (static_cast<std::uint64_t>(shared_right) << 1) ^
                   static_cast<std::uint64_t>(operation);
  const auto key = WithHybridAdmissionContext(shape_key);
  static std::mutex admission_mutex;
  static std::unordered_map<std::uint64_t, bool> admitted;
  bool select_gpu{};
  {
    std::lock_guard lock(admission_mutex);
    const auto found = admitted.find(key);
    if (found != admitted.end()) select_gpu = found->second;
    else {
      double gpu_ms{}, cpu_ms{};
      select_gpu = VulkanBinaryBroadcastRightChainBatchNoSlowerThanCpu(
          count, batches, right_repeat, right.data.size(), {operation}, &gpu_ms, &cpu_ms);
      admitted.emplace(key, select_gpu);
    }
  }
  return select_gpu && VulkanBinaryBroadcastRightChainBatch(
      left.data.data(), left.data.data(), right.data.data(), count, batches,
      right_repeat, right.data.size(), {operation}, immutable_right);
}

// BatchNorm is the dominant compact-two-RHS affine pattern in recognizer
// graphs. Keep its NCHW scale and bias vectors on the device and fuse
// multiply-plus-add in one batch dispatch. As with every hybrid segment, an
// exact end-to-end admission test (including H2D/D2H) must win before this is
// selected, so CPU SIMD remains the fast path on devices where sync overhead
// dominates.
bool TryHybridVulkanChannelAffineInplace(
    Backend backend, Tensor& value, const Tensor& scale, const Tensor& bias,
    bool immutable_coefficients) noexcept {
  if (backend != Backend::hybrid || value.shape.size() != 4 ||
      !value.data.data() || !scale.data.data() || !bias.data.data() ||
      value.data.size() < 65536) return false;
  const std::size_t batches = static_cast<std::size_t>(value.shape[0]);
  const std::size_t channels = static_cast<std::size_t>(value.shape[1]);
  const std::size_t height = static_cast<std::size_t>(value.shape[2]);
  const std::size_t width = static_cast<std::size_t>(value.shape[3]);
  if (batches == 0 || channels == 0 || height == 0 || width == 0 ||
      value.data.size() % batches != 0) return false;
  const std::size_t count = value.data.size() / batches;
  const std::size_t channel_repeat = height * width;
  const std::size_t coefficients_per_batch = count / channel_repeat;
  if (count % channel_repeat != 0 || coefficients_per_batch != channels ||
      scale.data.size() != bias.data.size() ||
      (scale.data.size() != channels && scale.data.size() != batches * channels)) return false;
  const bool shared_coefficients = scale.data.size() == channels;
  const auto shape_key = (static_cast<std::uint64_t>(count) << 32) ^
                   (static_cast<std::uint64_t>(batches) << 20) ^
                   (static_cast<std::uint64_t>(channel_repeat) << 2) ^
                   static_cast<std::uint64_t>(shared_coefficients);
  const auto key = WithHybridAdmissionContext(shape_key);
  static std::mutex admission_mutex;
  static std::unordered_map<std::uint64_t, bool> admitted;
  bool select_gpu{};
  {
    std::lock_guard lock(admission_mutex);
    const auto found = admitted.find(key);
    if (found != admitted.end()) select_gpu = found->second;
    else {
      double gpu_ms{}, cpu_ms{};
      select_gpu = VulkanChannelAffineBatchNoSlowerThanCpu(
          count, batches, channel_repeat, scale.data.size(), &gpu_ms, &cpu_ms,
          immutable_coefficients);
      admitted.emplace(key, select_gpu);
    }
  }
  return select_gpu && VulkanChannelAffineBatch(
      value.data.data(), value.data.data(), scale.data.data(), bias.data.data(),
      count, batches, channel_repeat, scale.data.size(), immutable_coefficients);
}

// The fused recognizer BatchNorm+Swish form has the same compact NCHW
// coefficient layout as channel affine, but running the nonlinear activation
// inside the one Vulkan dispatch avoids an additional CPU pass and an
// activation round-trip. Admission remains end-to-end and shape-specific.
bool TryHybridVulkanChannelAffineSwishInplace(
    Backend backend, Tensor& value, const Tensor& scale, const Tensor& bias,
    bool immutable_coefficients) noexcept {
  if (backend != Backend::hybrid || value.shape.size() != 4 ||
      !value.data.data() || !scale.data.data() || !bias.data.data() ||
      value.data.size() < 65536) return false;
  const std::size_t batches = static_cast<std::size_t>(value.shape[0]);
  const std::size_t channels = static_cast<std::size_t>(value.shape[1]);
  const std::size_t height = static_cast<std::size_t>(value.shape[2]);
  const std::size_t width = static_cast<std::size_t>(value.shape[3]);
  if (batches == 0 || channels == 0 || height == 0 || width == 0 ||
      value.data.size() % batches != 0) return false;
  const std::size_t count = value.data.size() / batches;
  const std::size_t channel_repeat = height * width;
  if (channel_repeat == 0 || count % channel_repeat != 0 ||
      count / channel_repeat != channels || scale.data.size() != bias.data.size() ||
      (scale.data.size() != channels && scale.data.size() != batches * channels)) return false;
  const bool shared_coefficients = scale.data.size() == channels;
  const auto shape_key = (static_cast<std::uint64_t>(count) << 32) ^
                   (static_cast<std::uint64_t>(batches) << 20) ^
                   (static_cast<std::uint64_t>(channel_repeat) << 2) ^
                   static_cast<std::uint64_t>(shared_coefficients);
  const auto key = WithHybridAdmissionContext(shape_key);
  static std::mutex admission_mutex;
  static std::unordered_map<std::uint64_t, bool> admitted;
  bool select_gpu{};
  {
    std::lock_guard lock(admission_mutex);
    const auto found = admitted.find(key);
    if (found != admitted.end()) select_gpu = found->second;
    else {
      double gpu_ms{}, cpu_ms{};
      select_gpu = VulkanChannelAffineSwishBatchNoSlowerThanCpu(
          count, batches, channel_repeat, scale.data.size(), &gpu_ms, &cpu_ms,
          immutable_coefficients);
      admitted.emplace(key, select_gpu);
    }
  }
  return select_gpu && VulkanChannelAffineSwishBatch(
      value.data.data(), value.data.data(), scale.data.data(), bias.data.data(),
      count, batches, channel_repeat, scale.data.size(), immutable_coefficients);
}

bool TryHybridVulkanChannelAffineSwishAddInplace(
    Backend backend, Tensor& value, const Tensor& scale, const Tensor& bias,
    const Tensor& residual, bool immutable_coefficients) noexcept {
  if (backend != Backend::hybrid || value.shape.size() != 4 || value.shape != residual.shape ||
      !value.data.data() || !scale.data.data() || !bias.data.data() || !residual.data.data() ||
      value.data.size() < 65536) return false;
  const std::size_t batches = static_cast<std::size_t>(value.shape[0]);
  const std::size_t channels = static_cast<std::size_t>(value.shape[1]);
  const std::size_t height = static_cast<std::size_t>(value.shape[2]);
  const std::size_t width = static_cast<std::size_t>(value.shape[3]);
  if (batches == 0 || channels == 0 || height == 0 || width == 0 ||
      value.data.size() % batches != 0 || residual.data.size() != value.data.size()) return false;
  const std::size_t count = value.data.size() / batches;
  const std::size_t channel_repeat = height * width;
  if (channel_repeat == 0 || count % channel_repeat != 0 ||
      count / channel_repeat != channels || scale.data.size() != channels ||
      bias.data.size() != channels) return false;
  const bool immutable_residual = false;  // A graph activation must be uploaded each invocation.
  const auto shape_key = (static_cast<std::uint64_t>(count) << 32) ^
                   (static_cast<std::uint64_t>(batches) << 20) ^
                   (static_cast<std::uint64_t>(channel_repeat) << 2) ^ 1u;
  const auto key = WithHybridAdmissionContext(shape_key);
  static std::mutex admission_mutex;
  static std::unordered_map<std::uint64_t, bool> admitted;
  bool select_gpu{};
  {
    std::lock_guard lock(admission_mutex);
    const auto found = admitted.find(key);
    if (found != admitted.end()) {
      select_gpu = found->second;
    } else {
      // Benchmark a representative activation with exactly the same shape
      // and complete synchronous boundary cost. The actual graph segment is
      // selected only when the GPU is no slower than the CPU fused kernel.
      std::vector<float> probe_left(value.data.size()), probe_right(value.data.size()),
          cpu(value.data.size()), gpu(value.data.size());
      for (std::size_t i = 0; i < probe_left.size(); ++i) {
        probe_left[i] = static_cast<float>(i % 251) * .0078125F - 1.F;
        probe_right[i] = static_cast<float>(i % 101) * .015625F - .75F;
      }
      constexpr int kRuns = 5;
      double cpu_total{}, gpu_total{};
      bool correct = true;
      for (int run = 0; run < kRuns && correct; ++run) {
        const auto cpu_begin = std::chrono::steady_clock::now();
        kernels::BatchNormSwishBinary(cpu.data(), probe_left.data(), scale.data.data(),
                                      bias.data.data(), probe_right.data(), int(batches),
                                      int(channels), channel_repeat, kernels::BinaryOp::add);
        cpu_total += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - cpu_begin).count();
        const auto gpu_begin = std::chrono::steady_clock::now();
        correct = VulkanChannelAffineSwishBinaryBatch(
            gpu.data(), probe_left.data(), scale.data.data(), bias.data.data(), probe_right.data(),
            count, batches, channel_repeat, scale.data.size(), kernels::BinaryOp::add,
            immutable_coefficients, immutable_residual);
        gpu_total += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - gpu_begin).count();
        for (std::size_t i = 0; correct && i < cpu.size(); ++i) {
          const float tolerance = 2e-5F * std::max(1.F, std::abs(cpu[i]));
          correct = std::abs(cpu[i] - gpu[i]) <= tolerance;
        }
      }
      select_gpu = correct && gpu_total <= cpu_total;
      admitted.emplace(key, select_gpu);
    }
  }
  return select_gpu && VulkanChannelAffineSwishBinaryBatch(
      value.data.data(), value.data.data(), scale.data.data(), bias.data.data(),
      residual.data.data(), count, batches, channel_repeat, scale.data.size(),
      kernels::BinaryOp::add, immutable_coefficients, immutable_residual);
}

std::size_t Tensor::size() const { return data.size(); }
Tensor::~Tensor() { RecycleActivation(data); }

struct OnnxLite::Impl {
  GraphData graph;
  // Vulkan owns one command buffer/queue for the process. Keep the strict
  // graph's device arena and immutable model tensors with this model rather
  // than uploading every initializer on every inference. The mutex spans one
  // whole graph recording so two public callers cannot interleave descriptor
  // updates or reclaim each other's activation slots.
  mutable std::mutex gpu_mutex;
  mutable VulkanTensorArena gpu_arena;
  mutable std::unordered_map<std::string, VulkanTensorSlot> gpu_constants;
  // `gpu_constants` holds arena indices, not Vulkan handles.  A device-loss
  // recovery clears the process-wide arena, so remember its generation and
  // rebuild the model cache before the next strict graph submission.
  mutable std::uint64_t gpu_generation{};
  // Number of graph-node/output consumers for a dynamic tensor.  Constants
  // deliberately do not participate: their storage is owned by `graph`.
  std::unordered_map<std::string, std::size_t> dynamic_uses;
  // The detector uses the generic Run() path once per page/batch. Its node
  // topology is still fixed after load, so it can use the same compact
  // call-local counter scheme as CTC instead of cloning a string-keyed map at
  // every inference. Keep `dynamic_uses` for graph construction/debugging and
  // preserve the generic runtime value map/output semantics.
  std::unordered_map<std::string, std::size_t> run_use_slots;
  std::vector<std::uint16_t> run_initial_uses;
  // `RunCtcTop1` executes once per text crop. Its node input names are fixed
  // after model loading, so index those dynamic names once and reset compact
  // counters by vector copy instead of cloning/hashing an unordered map on
  // every crop. The generic Run() keeps the map path because it is less hot
  // and its output semantics intentionally cover arbitrary ONNX graphs.
  std::unordered_map<std::string, std::size_t> ctc_use_slots;
  std::vector<std::uint16_t> ctc_initial_uses;
  std::unordered_set<std::string> graph_outputs;
  bool output_is_terminal_softmax{};
  // `RunCtcTop1` consumes only CTC Top-1 and selected probabilities.  The
  // model's final Softmax is therefore skipped only when it is the canonical
  // class-axis Softmax on [N,T,V]; other exported terminal Softmax forms stay
  // on the generic executor.
  bool terminal_softmax_is_class_axis{};
  // The terminal CTC projection is normally MatMul(+bias) immediately before
  // the canonical class-axis Softmax. RunCtcTop1 only needs its argmax and
  // selected probabilities, so retain this structural fact at load time and
  // avoid materialising the [N,T,V] logits activation when possible.
  bool terminal_logits_is_matmul{};
  bool terminal_logits_has_bias{};
  bool gpu_only_operator_subset{};
  std::size_t gpu_immutable_bytes{};
  Backend backend{Backend::cpu_only};
  struct GpuReplayCache {
    std::uint64_t key{};
    // LRU timestamp in the model-local replay cache. The Vulkan runtime has
    // a finite persistent-command-buffer budget, so a diverse stream must
    // evict an old shape rather than silently falling back to re-recording
    // every later graph.
    std::uint64_t last_used{};
    VulkanTensorSlot input{};
    VulkanTensorSlot output{};
    std::vector<std::int64_t> output_shape;
    std::string output_name;
    VulkanTensorSlot ctc_indices{};
    VulkanTensorSlot ctc_probs{};
    int ctc_batches{};
    int ctc_steps{};
    int ctc_classes{};
    VulkanTensorSlot rgb{};
    int rgb_source_width{};
    int rgb_source_height{};
  };
  mutable std::vector<GpuReplayCache> gpu_replay_caches;
  mutable std::uint64_t gpu_replay_clock{};

  // Env-gated (PPOCR_GPU_REPLAY_STATS=1) hit-rate telemetry. The counters are
  // mutable because replay lookups happen on const execution paths, and the
  // summary prints from ~Impl so a benchmark process reports once at exit.
  // Hit rate below ~90% on a steady-state workload means crop widths fragment
  // the recorded-graph set; that is the signal for width-merging work.
  mutable std::uint64_t replay_batch_requests{};
  mutable std::uint64_t replay_batch_full_hits{};
  mutable std::uint64_t replay_batch_key_misses{};
  mutable std::uint64_t replay_graphs_recorded{};
  void DumpGpuReplayStats() const {
    if (std::getenv("PPOCR_GPU_REPLAY_STATS") == nullptr) return;
    std::cerr << "gpu_replay_stats"
              << " batch_requests=" << replay_batch_requests
              << " full_hits=" << replay_batch_full_hits
              << " key_misses=" << replay_batch_key_misses
              << " graphs_recorded=" << replay_graphs_recorded
              << " live_graphs=" << gpu_replay_caches.size() << '\n';
  }

  // Caller holds gpu_mutex. A replay CB refers to every slot acquired during
  // its recording, not just its public input/output; DropPersistentGraph()
  // releases that complete ownership set before these descriptors are
  // returned to the ordinary arena allocator.
  void DropGpuReplayCache(std::size_t index) const noexcept {
    if (index >= gpu_replay_caches.size()) return;
    auto& cache = gpu_replay_caches[index];
    gpu_arena.DropPersistentGraph(cache.key);
    gpu_arena.UnpinForReplay(cache.input);
    gpu_arena.UnpinForReplay(cache.output);
    gpu_arena.UnpinForReplay(cache.ctc_indices);
    gpu_arena.UnpinForReplay(cache.ctc_probs);
    gpu_arena.UnpinForReplay(cache.rgb);
    gpu_arena.Release(cache.input);
    gpu_arena.Release(cache.output);
    gpu_arena.Release(cache.ctc_indices);
    gpu_arena.Release(cache.ctc_probs);
    gpu_arena.Release(cache.rgb);
    gpu_replay_caches.erase(gpu_replay_caches.begin() + static_cast<std::ptrdiff_t>(index));
  }

  ~Impl() {
    // The Vulkan runtime is process-wide, whereas constants are owned by this
    // particular ONNX model. Retain them across inferences, but return their
    // slots when the model instance is destroyed so loading several model
    // variants does not turn into an unbounded device-arena cache.
    std::lock_guard lock(gpu_mutex);
    // Pinned replay buffers are model-owned just like immutable parameters.
    // Release them explicitly before the process-wide arena is reused by a
    // later model instance; otherwise a long-lived service that loads/unloads
    // OCR variants can retain one complete graph per replay shape.
    while (!gpu_replay_caches.empty()) DropGpuReplayCache(0);
    for (auto& [_, slot] : gpu_constants) gpu_arena.Release(slot);
    gpu_constants.clear();
    DumpGpuReplayStats();
  }

  explicit Impl(const std::string& f, Backend policy) : graph(ParseModel(f)), backend(policy) {
    gpu_constants.reserve(graph.initializers.size());
    graph_outputs.insert(graph.outputs.begin(), graph.outputs.end());
    for (const auto& node : graph.nodes) {
      for (const auto& name : node.in) {
        if (!name.empty() && !graph.initializers.contains(name)) ++dynamic_uses[name];
      }
    }
    // Retain actual outputs until they have been copied into the public result.
    for (const auto& name : graph.outputs) {
      if (!graph.initializers.contains(name)) ++dynamic_uses[name];
    }
    run_use_slots.reserve(dynamic_uses.size());
    run_initial_uses.reserve(dynamic_uses.size());
    ctc_use_slots.reserve(dynamic_uses.size());
    ctc_initial_uses.reserve(dynamic_uses.size());
    for (const auto& [name, uses] : dynamic_uses) {
      if (uses > std::numeric_limits<std::uint16_t>::max()) {
        Fail("dynamic tensor use count exceeds CTC executor limit");
      }
      const auto run_slot = run_initial_uses.size();
      run_use_slots.emplace(name, run_slot);
      run_initial_uses.push_back(static_cast<std::uint16_t>(uses));
      const auto slot = ctc_initial_uses.size();
      ctc_use_slots.emplace(name, slot);
      ctc_initial_uses.push_back(static_cast<std::uint16_t>(uses));
    }
    if (const char* dump = std::getenv("PPOCR_DUMP_GRAPH")) {
      std::ofstream out(dump);
      for (const auto& node : graph.nodes) {
        out << node.op << '\t' << node.name;
        if (!node.in.empty()) {
          out << '\t';
          for (std::size_t i = 0; i < node.in.size(); ++i) {
            if (i) out << ',';
            out << node.in[i];
          }
        }
        out << '\n';
      }
    }
    output_is_terminal_softmax = graph.outputs.size() == 1 && !graph.nodes.empty() &&
        graph.nodes.back().op == "Softmax" && graph.nodes.back().in.size() == 1 &&
        graph.nodes.back().out.size() == 1 && graph.nodes.back().out[0] == graph.outputs.front();
    if (output_is_terminal_softmax) {
      const auto& terminal = graph.nodes.back();
      const int axis = terminal.attr.contains("axis") ? int(terminal.attr.at("axis").i) :
          (graph.opset >= 13 ? -1 : 1);
      terminal_softmax_is_class_axis = axis == -1 || axis == 2;
      if (terminal_softmax_is_class_axis && graph.nodes.size() >= 2) {
        const auto& logits = graph.nodes[graph.nodes.size() - 2];
        terminal_logits_is_matmul = logits.out.size() == 1 && logits.out[0] == terminal.in[0] &&
            (logits.op == "MatMul" || logits.op == "FusedMatMulBias");
        terminal_logits_has_bias = logits.op == "FusedMatMulBias";
      }
    }
    static const std::unordered_set<std::string> gpu_operator_subset{
        "Identity", "Conv", "FusedConvRelu", "FusedConvGelu", "FusedConvChannelBias", "FusedConvSigmoid",
        "FusedConvHardSigmoid", "FusedConvHardSwish", "FusedConvSwish", "ConvTranspose",
        "FusedMaxPoolConcatConv", "FusedMaxPoolConcatConvRelu", "FusedConcatConv",
        "FusedConcatConvRelu", "FusedConcatConvDualTranspose", "FusedExpandGeluProjectAdd",
        "FusedDepthwiseExpandGeluProjectAdd", "FusedConvTransposeRelu",
        "FusedConvTransposeSigmoid", "FusedConvTransposeChain", "FusedDepthwiseHardSwish",
        "FusedConvTranspose2x2Add", "FusedNearestResizeAdd", "FusedDepthwisePointwiseConv",
        "FusedDepthwisePointwiseConvGelu", "FusedDepthwisePointwiseConvHardSwish",
        "FusedDepthwisePointwiseConvRelu", "FusedPointwiseDepthwiseConv",
        "FusedPointwiseDepthwiseConvRelu", "FusedPointwiseConvAdd", "FusedPointwiseConvAddRelu",
        "FusedPointwiseConvAddSwish", "FusedSEGateMul", "MaxPool", "AveragePool", "GlobalAveragePool",
        "GlobalAveragePool", "ReduceMean", "BatchNormalization", "FusedBatchNormGelu",
        "FusedBatchNormSwish", "FusedBatchNormSwishAdd", "FusedBatchNormHardSwish",
        "Relu", "Sigmoid", "HardSigmoid",
        "FusedHardSwish", "FusedLayerNorm", "Transpose", "Reshape", "Squeeze", "Unsqueeze",
        "FusedQkvSplit", "MatMul", "FusedMatMulBias", "FusedMatMulBiasSwish", "Softmax",
        "Add", "Sub", "Mul", "Div", "Shape", "Slice", "Concat", "Resize"};
    gpu_only_operator_subset = std::all_of(graph.nodes.begin(), graph.nodes.end(),
        [&](const Node& node) { return gpu_operator_subset.contains(node.op); });
    for (const auto& [_, value] : graph.initializers)
      gpu_immutable_bytes += value.data.size() * sizeof(float);
  }
};
OnnxLite::OnnxLite(const std::string& file, Backend backend):impl_(std::make_shared<Impl>(file,backend)){}
const std::vector<std::string>& OnnxLite::outputs() const noexcept{return impl_->graph.outputs;}

bool OnnxLite::DetectorRgbStem(HybridRgbStem& stem) const noexcept {
  stem = {};
  if (!impl_ || impl_->graph.nodes.empty() || impl_->graph.inputs.empty()) return false;
  const auto& node = impl_->graph.nodes.front();
  if ((node.op != "FusedConvRelu" && node.op != "Conv") || node.in.size() < 2 ||
      node.out.empty() || node.in[0] != impl_->graph.inputs.front()) return false;
  const auto weights = impl_->graph.initializers.find(node.in[1]);
  if (weights == impl_->graph.initializers.end() || weights->second.shape.size() != 4 ||
      weights->second.shape[1] != 3 || weights->second.shape[2] != 3 ||
      weights->second.shape[3] != 3 || weights->second.shape[0] < 8) return false;
  const auto strides = AttrInts(node, "strides", {1, 1});
  auto pads = AttrInts(node, "pads", {0, 0, 0, 0});
  const auto auto_pad = AttrStr(node, "auto_pad", "NOTSET");
  if (strides.size() != 2 || strides[0] != 2 || strides[1] != 2) return false;
  int pad = 1;
  if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") pad = 1;
  else if (pads.size() == 4 && pads[0] == pads[1] && pads[1] == pads[2] && pads[2] == pads[3])
    pad = int(pads[0]);
  else return false;
  const Tensor* bias = nullptr;
  if (node.in.size() > 2 && !node.in[2].empty()) {
    const auto found = impl_->graph.initializers.find(node.in[2]);
    if (found == impl_->graph.initializers.end() ||
        found->second.data.size() != std::size_t(weights->second.shape[0])) return false;
    bias = &found->second;
  }
  if (!bias) return false;
  stem.output_name = node.out[0];
  stem.weights = weights->second.data.data();
  stem.bias = bias->data.data();
  stem.output_channels = int(weights->second.shape[0]);
  stem.kernel = 3;
  stem.stride = 2;
  stem.pad = pad;
  stem.relu = node.op == "FusedConvRelu";
  return stem.weights && stem.bias && !stem.output_name.empty();
}

bool OnnxLite::RecognizerRgbStem(HybridRgbStem& stem) const noexcept {
  stem = {};
  if (!impl_ || impl_->graph.nodes.empty() || impl_->graph.inputs.empty()) return false;
  const auto& node = impl_->graph.nodes.front();
  if ((node.op != "FusedConvRelu" && node.op != "FusedConvGelu" && node.op != "Conv") ||
      node.in.size() < 2 || node.out.empty() ||
      node.in[0] != impl_->graph.inputs.front()) return false;
  const auto weights = impl_->graph.initializers.find(node.in[1]);
  if (weights == impl_->graph.initializers.end() || weights->second.shape.size() != 4 ||
      weights->second.shape[1] != 3 || weights->second.shape[2] != 3 ||
      weights->second.shape[3] != 3 || weights->second.shape[0] < 8) return false;
  const auto strides = AttrInts(node, "strides", {1, 1});
  auto pads = AttrInts(node, "pads", {0, 0, 0, 0});
  const auto auto_pad = AttrStr(node, "auto_pad", "NOTSET");
  if (strides.size() != 2 || strides[0] != 2 || strides[1] != 2) return false;
  int pad = 1;
  if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") pad = 1;
  else if (pads.size() == 4 && pads[0] == pads[1] && pads[1] == pads[2] && pads[2] == pads[3])
    pad = int(pads[0]);
  else return false;
  const Tensor* bias = nullptr;
  if (node.in.size() > 2 && !node.in[2].empty()) {
    const auto found = impl_->graph.initializers.find(node.in[2]);
    if (found == impl_->graph.initializers.end() ||
        found->second.data.size() != std::size_t(weights->second.shape[0])) return false;
    bias = &found->second;
  }
  if (!bias) return false;
  // Rec first conv is a linear 3x3; fused GELU stays on CPU because the
  // RGB+conv spatial kernel has no GELU path.
  if (node.op == "FusedConvGelu") return false;
  stem.output_name = node.out[0];
  stem.weights = weights->second.data.data();
  stem.bias = bias->data.data();
  stem.output_channels = int(weights->second.shape[0]);
  stem.kernel = 3;
  stem.stride = 2;
  stem.pad = pad;
  stem.relu = node.op == "FusedConvRelu";
  return stem.weights && stem.bias && !stem.output_name.empty();
}

bool OnnxLite::DetectorRgbStemChain(HybridRgbStemChain& chain) const noexcept {
  chain = {};
  if (!DetectorRgbStem(chain.conv0) || !impl_ || impl_->graph.nodes.size() < 4) return false;
  const auto load_conv = [&](const Node& node, HybridRgbStem& layer, int expect_ic) -> bool {
    if ((node.op != "FusedConvRelu" && node.op != "Conv") || node.in.size() < 2 ||
        node.out.empty()) return false;
    const auto weights = impl_->graph.initializers.find(node.in[1]);
    if (weights == impl_->graph.initializers.end() || weights->second.shape.size() != 4 ||
        int(weights->second.shape[1]) != expect_ic || weights->second.shape[0] < 4)
      return false;
    const auto strides = AttrInts(node, "strides", {1, 1});
    auto pads = AttrInts(node, "pads", {0, 0, 0, 0});
    const auto auto_pad = AttrStr(node, "auto_pad", "NOTSET");
    if (strides.size() != 2 || strides[0] != strides[1] || strides[0] <= 0) return false;
    int pad = 0;
    if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") pad = 0;
    else if (pads.size() == 4) pad = int(pads[0]);
    else return false;
    const Tensor* bias = nullptr;
    if (node.in.size() > 2 && !node.in[2].empty()) {
      const auto found = impl_->graph.initializers.find(node.in[2]);
      if (found == impl_->graph.initializers.end() ||
          found->second.data.size() != std::size_t(weights->second.shape[0])) return false;
      bias = &found->second;
    }
    if (!bias) return false;
    layer.output_name = node.out[0];
    layer.weights = weights->second.data.data();
    layer.bias = bias->data.data();
    layer.output_channels = int(weights->second.shape[0]);
    layer.kernel = int(weights->second.shape[2]);
    layer.stride = int(strides[0]);
    layer.pad = pad;
    layer.relu = node.op == "FusedConvRelu";
    return layer.kernel == int(weights->second.shape[3]);
  };
  const auto& n0 = impl_->graph.nodes[0];
  const auto& n1 = impl_->graph.nodes[1];
  const auto& n2 = impl_->graph.nodes[2];
  const auto& n3 = impl_->graph.nodes[3];
  if (n1.in.empty() || n1.in[0] != n0.out[0]) return false;
  if (n2.in.empty() || n2.in[0] != n1.out[0]) return false;
  if (n3.op != "FusedMaxPoolConcatConvRelu" && n3.op != "FusedMaxPoolConcatConv")
    return false;
  if (!load_conv(n1, chain.conv1, chain.conv0.output_channels) ||
      !load_conv(n2, chain.conv2, chain.conv1.output_channels)) return false;
  if (chain.conv1.kernel != 2 || chain.conv2.kernel != 2) return false;
  bool saw_pool_src = false, saw_peer = false;
  int stem_ic = 0;
  for (const auto& name : n3.in) {
    if (name.empty() || impl_->graph.initializers.contains(name)) continue;
    if (name == n0.out[0]) saw_pool_src = true;
    else if (name == n2.out[0]) saw_peer = true;
    const auto w = impl_->graph.initializers.find(name);
    (void)w;
  }
  if (!saw_pool_src || !saw_peer) return false;
  const bool has_bias = !n3.in.empty() && impl_->graph.initializers.contains(n3.in.back()) &&
      impl_->graph.initializers.at(n3.in.back()).shape.size() == 1;
  const auto weights_name = n3.in[n3.in.size() - (has_bias ? 2 : 1)];
  const auto weights = impl_->graph.initializers.find(weights_name);
  if (weights == impl_->graph.initializers.end() || weights->second.shape.size() != 4 ||
      weights->second.shape[2] != 3 || weights->second.shape[3] != 3) return false;
  const Tensor* bias = nullptr;
  if (has_bias) {
    const auto found = impl_->graph.initializers.find(n3.in.back());
    if (found == impl_->graph.initializers.end()) return false;
    bias = &found->second;
  }
  if (!bias) return false;
  stem_ic = int(weights->second.shape[1]);
  if (stem_ic != chain.conv0.output_channels + chain.conv2.output_channels) return false;
  const auto strides = AttrInts(n3, "strides", {1, 1});
  auto pads = AttrInts(n3, "pads", {0, 0, 0, 0});
  if (strides.size() != 2 || strides[0] != 2 || strides[1] != 2) return false;
  int pad = pads.size() == 4 ? int(pads[0]) : 1;
  chain.stem.output_name = n3.out.empty() ? std::string{} : n3.out[0];
  chain.stem.weights = weights->second.data.data();
  chain.stem.bias = bias->data.data();
  chain.stem.output_channels = int(weights->second.shape[0]);
  chain.stem.kernel = 3;
  chain.stem.stride = 2;
  chain.stem.pad = pad;
  chain.stem.relu = n3.op == "FusedMaxPoolConcatConvRelu";
  chain.conv1_input_channels = chain.conv0.output_channels;
  chain.conv2_input_channels = chain.conv1.output_channels;
  chain.stem_input_channels = stem_ic;
  int uses0 = 0, uses1 = 0, uses2 = 0;
  for (const auto& node : impl_->graph.nodes) {
    for (const auto& name : node.in) {
      if (name == n0.out[0]) ++uses0;
      if (name == n1.out[0]) ++uses1;
      if (name == n2.out[0]) ++uses2;
    }
  }
  if (uses0 != 2 || uses1 != 1 || uses2 != 1) return false;
  return !chain.stem.output_name.empty();
}
bool OnnxLite::SupportsGpuOnly() const noexcept {
  // Model rewriting can retain metadata-only or exporter-specific fused node
  // names that are lowered by the strict executor before dispatch.  Therefore
  // this static label is intentionally not the public admission criterion;
  // RunGpuOnlyInternal is authoritative and fails closed for any node it
  // cannot lower to a Vulkan command.  In particular, GPU-only never falls
  // through to the CPU executor after construction succeeds.
  return impl_ != nullptr;
}

namespace {

struct DeviceValue {
  std::vector<std::int64_t> shape;
  VulkanTensorSlot slot;
  // Non-zero when the last two logical axes are swapped versus the packed
  // buffer (a skipped Transpose). GEMM then reads A[row,k] at k*a_lda+row.
  int a_lda = 0;
};

bool IsNchw(const DeviceValue& value) { return value.shape.size() == 4; }

std::vector<std::int64_t> DeviceReshapeShape(const DeviceValue& value, const Tensor& spec) {
  std::vector<std::int64_t> shape;
  shape.reserve(spec.data.size());
  std::int64_t known = 1;
  int inferred = -1;
  for (std::size_t axis = 0; axis < spec.data.size(); ++axis) {
    std::int64_t dimension = static_cast<std::int64_t>(spec.data[axis]);
    if (dimension == -1) {
      if (inferred >= 0) Fail("multiple reshape inferred dimensions");
      inferred = static_cast<int>(axis); shape.push_back(1);
    } else {
      if (dimension == 0) dimension = value.shape.at(axis);
      shape.push_back(dimension); known *= dimension;
    }
  }
  if (inferred >= 0) shape[static_cast<std::size_t>(inferred)] =
      static_cast<std::int64_t>(value.slot.live_elements / known);
  if (Elements(shape) != value.slot.live_elements) Fail("GPU reshape mismatch");
  return shape;
}

}  // namespace

// Strict arena executor for the established convolutional PP-OCRv6 graph
// subset. It is intentionally separate from Execute(): success means every
// activation node ran on Vulkan; callers receive false for an unsupported
// node, never a hidden CPU fallback. Immutable constants are cached in
// persistent Vulkan slots across calls; the activation map owns/reuses all
// other arena slots by exact graph liveness.
bool OnnxLite::RunGpuOnlyInternal(
    std::unordered_map<std::string, Tensor>* host_inputs,
    std::unordered_map<std::string, GpuTensor> device_values,
    std::unordered_map<std::string, Tensor>* host_outputs,
    std::unordered_map<std::string, GpuTensor>* device_outputs,
    bool fuse_terminal_ctc_softmax,
    const GpuRgbFrontEnd* rgb_front) const {
  const auto& impl = *impl_;
  std::lock_guard gpu_lock(impl.gpu_mutex);
  auto& arena = impl.gpu_arena;
  if (!arena.available()) return false;
  // The current persistent-replay facility is intentionally single-command-
  // buffer.  Models that need an isolated GPU tail submission must therefore
  // use the normal fenced graph path; this changes only command-buffer
  // lifetime, never GPU residency or operator selection.
  const bool model_has_isolated_tail = impl.graph.nodes.size() > 0 &&
      std::any_of(impl.graph.nodes.begin(), impl.graph.nodes.end(), [](const Node& node) {
        return !node.out.empty() && node.out[0] == "p2o.pd_op.depthwise_conv2d.12.0";
      });
  // Persistent replay pins every dynamic activation for its recorded command
  // buffer. That is useful for the device-input OCR pipeline, but the public
  // host-input graph API must release its transient slots after every Run()
  // so resolution ladders can demonstrate memory recovery. Keep replay only
  // for callers that already supply device tensors or the in-graph RGB front
  // end; this remains a scheduling policy, never a CPU fallback.
  const bool allow_replay = (!device_values.empty() || rgb_front != nullptr) &&
      std::getenv("PPOCR_DISABLE_GPU_GRAPH_REPLAY") == nullptr &&
      !model_has_isolated_tail;
  std::uint64_t replay_key = 0;
  const std::string* graph_input_name =
      impl.graph.inputs.empty() ? nullptr : &impl.graph.inputs.front();
  std::vector<std::int64_t> replay_in_shape;
  const float* host_input_data = nullptr;
  std::size_t host_input_elements = 0;
  GpuTensor* incoming_device_input = nullptr;
  const int rgb_src_w = rgb_front ? rgb_front->source_width : 0;
  const int rgb_src_h = rgb_front ? rgb_front->source_height : 0;
  if (rgb_front && rgb_front->rgb.resident && rgb_front->output_width > 0 &&
      rgb_front->output_height > 0) {
    replay_in_shape = {1, 3, rgb_front->output_height, rgb_front->output_width};
  } else if (graph_input_name && host_inputs) {
    const auto it = host_inputs->find(*graph_input_name);
    if (it != host_inputs->end()) {
      replay_in_shape = it->second.shape;
      host_input_data = it->second.data.data();
      host_input_elements = it->second.data.size();
    }
  } else if (graph_input_name) {
    const auto it = device_values.find(*graph_input_name);
    if (it != device_values.end()) {
      replay_in_shape = it->second.shape;
      incoming_device_input = &it->second;
    }
  }
  if (allow_replay && !replay_in_shape.empty()) {
    replay_key = GpuReplayKey(replay_in_shape, fuse_terminal_ctc_softmax, rgb_src_w,
                              rgb_src_h);
  }
  auto find_replay_cache = [&]() -> Impl::GpuReplayCache* {
    for (auto& cache : impl.gpu_replay_caches) {
      if (cache.key == replay_key) return &cache;
    }
    return nullptr;
  };
  if (allow_replay && replay_key != 0 && arena.HasPersistentGraph(replay_key)) {
    auto* cache = find_replay_cache();
    if (cache && cache->output.resident && cache->input.resident) {
      bool uploaded = false;
      if (rgb_front && cache->rgb.resident &&
          cache->rgb.index == rgb_front->rgb.index) {
        // RGB bytes are already memcpy'd into the pinned slot. The recorded
        // CB's first barrier is HOST_WRITE; resize then consumes them.
        uploaded = arena.FlushHostWrites(cache->rgb, cache->rgb.live_elements);
      } else if (host_input_data && host_input_elements == cache->input.live_elements) {
        uploaded = arena.Upload(cache->input, host_input_data, host_input_elements);
      } else if (incoming_device_input && incoming_device_input->slot.resident) {
        if (incoming_device_input->slot.index == cache->input.index) {
          uploaded = arena.FlushHostWrites(cache->input, cache->input.live_elements);
        } else {
          uploaded = arena.Copy(incoming_device_input->slot, cache->input,
                                cache->input.live_elements);
          arena.Release(incoming_device_input->slot);
        }
      }
      if (uploaded && arena.ReplayPersistentGraph(replay_key)) {
        cache->last_used = ++impl.gpu_replay_clock;
        if (std::getenv("PPOCR_TRACE_GPU_REPLAY") != nullptr) {
          std::cerr << "gpu_graph_replay hit key=" << replay_key << '\n';
        }
        if (device_outputs) {
          device_outputs->emplace(cache->output_name,
                                  GpuTensor{cache->output_shape, cache->output});
          return true;
        }
        if (host_outputs) {
          Tensor output{cache->output_shape,
                        std::vector<float>(cache->output.live_elements)};
          if (arena.Download(output.data.data(), cache->output, output.data.size())) {
            host_outputs->emplace(cache->output_name, std::move(output));
            return true;
          }
        }
      }
    }
  }
  const std::uint64_t arena_generation = arena.generation();
  if (impl.gpu_generation != arena_generation) {
    // Runtime recovery has already destroyed the backing buffers; releasing
    // these stale numeric slots could release unrelated new allocations.
    impl.gpu_constants.clear();
    impl.gpu_generation = arena_generation;
  }
  std::unordered_map<std::string, DeviceValue> values;
  values.reserve(impl.graph.nodes.size());
  for (auto& [name, value] : device_values) {
    values.emplace(name, DeviceValue{std::move(value.shape), value.slot});
  }
  // Dynamic ONNX shape expressions are graph scheduling metadata, not an
  // activation fallback.  Keep their tiny integer vectors host-side while
  // all floating point feature maps remain in Vulkan slots.
  std::unordered_map<std::string, Tensor> shape_values;
  auto uses = impl.run_initial_uses;
  const auto consume = [&](const std::string& name) -> bool {
    const auto it = impl.run_use_slots.find(name);
    return it != impl.run_use_slots.end() && uses[it->second] && --uses[it->second] == 0;
  };
  const auto final_use = [&](const std::string& name) -> bool {
    const auto it = impl.run_use_slots.find(name);
    return it != impl.run_use_slots.end() && uses[it->second] == 1;
  };
  // GPU-only graphs may retain a few hundred weights.  Allocate those first
  // so their Vulkan buffers occupy a compact, deterministic prefix of the
  // arena instead of interleaving late lazy uploads with large activations.
  // This is device-only lifetime management; it never materialises a feature
  // map on CPU.
  const auto upload_constant = [&](const std::string& name) -> const VulkanTensorSlot* {
    const auto cached = impl.gpu_constants.find(name);
    if (cached != impl.gpu_constants.end()) return &cached->second;
    const auto source = impl.graph.initializers.find(name);
    if (source == impl.graph.initializers.end()) return nullptr;
    auto slot = arena.Acquire(source->second.data.size(), name, true);
    if (!slot.resident || !arena.Upload(slot, source->second.data.data(), source->second.data.size())) {
      arena.Release(slot);
      return nullptr;
    }
    return &impl.gpu_constants.emplace(name, slot).first->second;
  };
  const auto get_constant = [&](const std::string& name) -> const VulkanTensorSlot* {
    return upload_constant(name);
  };
  const auto pack_expand_project_bias =
      [&](const std::string& key, int hidden, int channels,
          const std::string& expand_bias_name,
          const std::string& project_bias_name) -> const VulkanTensorSlot* {
    const auto cached = impl.gpu_constants.find(key);
    if (cached != impl.gpu_constants.end()) return &cached->second;
    std::vector<float> packed(std::size_t(hidden) + std::size_t(channels), 0.F);
    if (!expand_bias_name.empty()) {
      const auto it = impl.graph.initializers.find(expand_bias_name);
      if (it != impl.graph.initializers.end() &&
          it->second.data.size() == std::size_t(hidden))
        std::memcpy(packed.data(), it->second.data.data(),
                    std::size_t(hidden) * sizeof(float));
    }
    if (!project_bias_name.empty()) {
      const auto it = impl.graph.initializers.find(project_bias_name);
      if (it != impl.graph.initializers.end() &&
          it->second.data.size() == std::size_t(channels))
        std::memcpy(packed.data() + hidden, it->second.data.data(),
                    std::size_t(channels) * sizeof(float));
    }
    auto slot = arena.Acquire(packed.size(), key, true);
    if (!slot.resident || !arena.Upload(slot, packed.data(), packed.size())) {
      arena.Release(slot);
      return nullptr;
    }
    return &impl.gpu_constants.emplace(key, slot).first->second;
  };
  const auto get_shape_value = [&](const std::string& name) -> const Tensor* {
    const auto dynamic = shape_values.find(name);
    if (dynamic != shape_values.end()) return &dynamic->second;
    const auto constant = impl.graph.initializers.find(name);
    return constant == impl.graph.initializers.end() ? nullptr : &constant->second;
  };
  if (host_inputs) {
    for (const auto& name : impl.graph.inputs) {
      const auto source = host_inputs->find(name);
      if (source == host_inputs->end()) return false;
      auto slot = arena.Acquire(source->second.data.size(), name);
      if (!slot.resident || !arena.Upload(slot, source->second.data.data(), source->second.data.size())) {
        arena.Release(slot);
        for (auto& [_, value] : values) arena.Release(value.slot);
        return false;
      }
      values.emplace(name, DeviceValue{source->second.shape, slot});
    }
  } else if (rgb_front && graph_input_name) {
    if (values.find(*graph_input_name) == values.end()) {
      const std::size_t nchw_elements =
          std::size_t(3) * std::size_t(rgb_front->output_width) *
          std::size_t(rgb_front->output_height);
      auto slot = arena.Acquire(nchw_elements, *graph_input_name);
      if (!slot.resident) {
        for (auto& [_, value] : values) arena.Release(value.slot);
        return false;
      }
      values.emplace(*graph_input_name, DeviceValue{replay_in_shape, slot});
    }
  } else {
    for (const auto& name : impl.graph.inputs) {
      const auto found = values.find(name);
      if (found == values.end() || !found->second.slot.resident ||
          found->second.shape.empty() || found->second.slot.live_elements == 0) {
        for (auto& [_, value] : values) arena.Release(value.slot);
        return false;
      }
      // Device-input slots may have been populated through a host-visible
      // Vulkan front end. Ensure those writes are available to the strict
      // graph's first dispatch before recording it.
      if (!arena.FlushHostWrites(found->second.slot, found->second.slot.live_elements)) {
        for (auto& [_, value] : values) arena.Release(value.slot);
        return false;
      }

    }
  }
  // All model constants must cross the host-write -> shader-read boundary
  // before graph recording begins.  Uploading a newly encountered weight
  // while a command buffer is already being recorded leaves that write after
  // BeginArenaRecording's input barrier, which is undefined on non-coherent
  // memory and manifested as image-dependent NaNs.  They are cache-owned,
  // therefore this is a one-time model warm-up cost; later runs simply find
  // the resident slots in get_constant().
  // Only floating-point initializers are tensor data. Shape/axes/scales
  // initializers are deliberately consumed as small host-side scheduling
  // metadata below and may have an empty float payload after ONNX parsing.
  for (const auto& [name, source] : impl.graph.initializers) {
    if (source.data.empty()) continue;
    if (!upload_constant(name)) {
      for (auto& [_, value] : values) arena.Release(value.slot);
      return false;
    }
  }
  bool persistent_graph = false;
  if (allow_replay && replay_key != 0 && !arena.HasPersistentGraph(replay_key)) {
    // Evict before asking the process-wide Vulkan runtime to record another
    // graph. The runtime has the same finite command-buffer budget across
    // detector and recognizer instances; waiting until after record would
    // make a varied stream miss replay once that shared budget is full.
    // Replacing the runtime's natural saturation behavior with continual
    // eviction keeps more large dynamic shapes pinned and can raise the UMA
    // high-water mark. It is therefore an explicit service-throughput mode;
    // the default remains the bounded-memory policy that stops recording
    // new shapes when Vulkan's shared replay budget is full.
    static const bool replay_lru =
        std::getenv("PPOCR_ENABLE_GPU_REPLAY_LRU") != nullptr &&
        std::getenv("PPOCR_DISABLE_GPU_REPLAY_LRU") == nullptr;
    constexpr std::size_t kMaxReplayGraphs = 4;
    if (replay_lru && impl.gpu_replay_caches.size() >= kMaxReplayGraphs) {
      const auto lru = std::min_element(impl.gpu_replay_caches.begin(),
                                        impl.gpu_replay_caches.end(),
          [](const Impl::GpuReplayCache& left, const Impl::GpuReplayCache& right) {
            return left.last_used < right.last_used;
          });
      impl.DropGpuReplayCache(static_cast<std::size_t>(
          std::distance(impl.gpu_replay_caches.begin(), lru)));
    }
    persistent_graph = arena.BeginPersistentGraphRecording(replay_key);
    if (std::getenv("PPOCR_TRACE_GPU_REPLAY") != nullptr) {
      std::cerr << "gpu_graph_record begin key=" << replay_key
                << " persistent=" << persistent_graph << '\n';
    }
  }
  if (!persistent_graph && !arena.BeginGraphRecording()) return false;
  bool graph_recording = true;
  VulkanTensorSlot pinned_replay_input{};
  VulkanTensorSlot pinned_replay_rgb{};
  VulkanTensorSlot gemm_ctc_indices{};
  VulkanTensorSlot gemm_ctc_probs{};
  int gemm_ctc_batches = 0, gemm_ctc_steps = 0, gemm_ctc_classes = 0;
  if (persistent_graph && graph_input_name) {
    const auto in = values.find(*graph_input_name);
    if (in != values.end()) {
      pinned_replay_input = in->second.slot;
      arena.PinForReplay(pinned_replay_input);
    }
  }
  const auto abort_recording = [&] {
    if (graph_recording) {
      if (persistent_graph) (void)arena.EndPersistentGraphRecording(false);
      else (void)arena.EndGraphRecording(false);
      graph_recording = false;
    }
  };
  const auto fail = [&]() -> bool {
    abort_recording();
    // Inputs and activations must not survive a rejected graph. Constants are
    // intentionally excluded: they are model-owned, immutable device slots.
    for (auto& [_, value] : values) arena.Release(value.slot);
    values.clear();
    return false;
  };
  const auto release = [&](const std::string& name) {
    const auto it = values.find(name); if (it != values.end()) { arena.Release(it->second.slot); values.erase(it); }
  };
  const auto fetch = [&](const std::string& name) -> const DeviceValue* {
    const auto value = values.find(name); return value == values.end() ? nullptr : &value->second;
  };
  if (rgb_front && rgb_front->rgb.resident && graph_input_name) {
    const auto in = values.find(*graph_input_name);
    if (in == values.end() || !in->second.slot.resident) return fail();
    if (persistent_graph) {
      arena.PinForReplay(rgb_front->rgb);
      pinned_replay_rgb = rgb_front->rgb;
    }
    if (!arena.ResizeRgbToNchw(rgb_front->rgb, in->second.slot, rgb_front->source_width,
                               rgb_front->source_height, rgb_front->left, rgb_front->top,
                               rgb_front->region_width, rgb_front->region_height,
                               rgb_front->output_width, rgb_front->output_height,
                               0.F, 0.F, 0.F, 0.F, 0.F, 0.F)) {
      return fail();
    }
  }
  std::size_t recorded_nodes{};
  std::size_t segment_node_limit{};
  if (const char* text = std::getenv("PPOCR_GPU_ONLY_SEGMENT_NODES")) {
    char* end{};
    const auto parsed = std::strtoul(text, &end, 10);
    if (end != text && *end == '\0' && parsed > 0) segment_node_limit = parsed;
  }
  // Keep strict GPU graphs within a bounded command-buffer lifetime. The
  // boundary is device-to-device: live arena slots and immutable parameters
  // remain resident, while the completed fence lets drivers retire descriptor
  // state before the next segment. One node per submission is useful for
  // driver diagnosis but wastes most of a small model's GPU time in queue and
  // fence overhead. Qualification across all three PP-OCRv6 families keeps a
  // 20-node segment stable; it keeps the short tiny/small graph fence latency
  // low while avoiding the long-command-buffer tail observed beyond 40 nodes.
  // It is also the measured safe point for medium on the current Radeon/LLPC
  // path. The
  // environment override remains available for a deployment-specific driver
  // investigation, without permitting a CPU activation fallback.
  // The measured cross-model default is 20 nodes.  It cuts the tiny
  // detector's queue/fence overhead substantially, keeps the small/medium
  // detector within their qualified descriptor lifetime, and is neutral or
  // better for the recognizer transformer graphs.  Deployments can still
  // supply PPOCR_GPU_ONLY_SEGMENT_NODES for driver-specific qualification.
  if (segment_node_limit == 0) segment_node_limit = 20;
  for (const auto& node : impl.graph.nodes) {

    // A previous segment is deliberately closed only after all of its final
    // consumers have released their slots below.  Start the next command
    // buffer lazily here so BeginArenaRecording can mark those completed
    // buffers reusable.  Starting it before release pins every transient
    // activation for the rest of the graph, which is especially costly for
    // the medium detector's high-channel tail on UMA Vulkan drivers.
    if (!graph_recording) {
      if (!arena.BeginGraphRecording()) return fail();
      graph_recording = true;
    }

    if (std::getenv("PPOCR_GPU_ONLY_TRACE") != nullptr) {
      std::cerr << "GPU-only node " << node.op << " -> "
                << (node.out.empty() ? "<none>" : node.out.front()) << " in=";
      for (const auto& name : node.in) std::cerr << name << ',';
      std::cerr << '\n';
    }
    if ((node.out.size() != 1 && !(node.op == "FusedQkvSplit" && node.out.size() == 3)) ||
        node.in.empty()) return fail();
    if (node.op == "Slice") {
      // Shape/Slice chains only calculate a later Reshape specification. They
      // are intentionally host metadata, never a hidden activation path.
      const auto* source = get_shape_value(node.in[0]);
      const auto* starts = node.in.size() > 1 ? get_shape_value(node.in[1]) : nullptr;
      const auto* ends = node.in.size() > 2 ? get_shape_value(node.in[2]) : nullptr;
      const auto* axes = node.in.size() > 3 ? get_shape_value(node.in[3]) : nullptr;
      const auto* steps = node.in.size() > 4 ? get_shape_value(node.in[4]) : nullptr;
      if (!source || !starts || !ends || starts->data.size() != ends->data.size()) return fail();
      Tensor sliced; sliced.shape = source->shape;
      if (sliced.shape.size() != 1) return fail();
      const int length = int(source->data.size());
      int begin = 0, end = length, step = 1;
      if (!starts->data.empty()) {
        const int axis = axes && !axes->data.empty() ? int(axes->data[0]) : 0;
        if (axis != 0 || (steps && !steps->data.empty() && int(steps->data[0]) != 1)) return fail();
        begin = int(starts->data[0]); end = int(ends->data[0]);
        if (begin < 0) begin += length; if (end < 0) end += length;
        begin = std::clamp(begin, 0, length); end = std::clamp(end, 0, length);
      }
      if (step != 1 || end < begin) return fail();
      sliced.shape[0] = end - begin;
      sliced.data.assign(source->data.begin() + begin, source->data.begin() + end);
      shape_values.emplace(node.out[0], std::move(sliced));
      for (const auto& name : node.in) {
        if (name.empty() || impl.graph.initializers.contains(name) || !consume(name)) continue;
        shape_values.erase(name);
      }
      continue;
    }
    // Dynamic-shape subgraphs are small integer metadata programs.  In
    // particular, PP-OCRv6 recognizers use Shape -> Slice -> Squeeze /
    // Unsqueeze -> Concat to construct transformer reshape specifications.
    // Do this before fetch(): these tensors intentionally have no Vulkan
    // allocation, while ordinary activations with the same ONNX operators
    // continue through the device paths below.
    if ((node.op == "Identity" || node.op == "Reshape" || node.op == "Squeeze" ||
         node.op == "Unsqueeze") && get_shape_value(node.in[0]) != nullptr) {
      const auto* source = get_shape_value(node.in[0]);
      Tensor metadata = *source;
      if (node.op == "Reshape") {
        if (node.in.size() < 2) return fail();
        const auto* spec = get_shape_value(node.in[1]);
        if (!spec) return fail();
        // Reuse the existing ONNX reshape validation without fabricating an
        // activation.  `metadata.data` is merely the integer shape payload.
        DeviceValue descriptor{metadata.shape, {}};
        descriptor.slot.live_elements = metadata.data.size();
        metadata.shape = DeviceReshapeShape(descriptor, *spec);
      } else if (node.op != "Identity") {
        std::vector<std::int64_t> axes = AttrInts(node, "axes", {});
        if (axes.empty() && node.in.size() > 1) {
          const auto* axes_tensor = get_shape_value(node.in[1]);
          if (!axes_tensor) return fail();
          for (const float axis : axes_tensor->data) axes.push_back(std::int64_t(axis));
        }
        if (axes.empty()) return fail();
        if (node.op == "Unsqueeze") {
          std::sort(axes.begin(), axes.end());
          for (auto axis : axes) {
            if (axis < 0) axis += std::int64_t(metadata.shape.size()) + 1;
            if (axis < 0 || axis > std::int64_t(metadata.shape.size())) return fail();
            metadata.shape.insert(metadata.shape.begin() + axis, 1);
          }
        } else {
          std::sort(axes.rbegin(), axes.rend());
          for (const auto axis : axes) {
            const int position = Axis(axis, int(metadata.shape.size()));
            if (metadata.shape[position] != 1) return fail();
            metadata.shape.erase(metadata.shape.begin() + position);
          }
        }
      }
      shape_values.emplace(node.out[0], std::move(metadata));
      for (const auto& name : node.in) {
        if (name.empty() || impl.graph.initializers.contains(name) || !consume(name)) continue;
        shape_values.erase(name);
      }
      continue;
    }
    if (node.op == "Concat" && !node.in.empty() && get_shape_value(node.in[0]) != nullptr) {
      // Shape-program concat (for example [N], [T], [heads], [width]) is
      // metadata only. ONNX still records a rank/axis, so validate it rather
      // than treating arbitrary host tensors as shape expressions.
      const auto* first = get_shape_value(node.in[0]);
      const int axis = Axis(AttrInt(node, "axis", 0), int(first->shape.size()));
      Tensor metadata;
      metadata.shape = first->shape;
      metadata.shape[std::size_t(axis)] = 0;
      for (const auto& name : node.in) {
        const auto* part = get_shape_value(name);
        if (!part || part->shape.size() != first->shape.size()) return fail();
        for (std::size_t dimension = 0; dimension < part->shape.size(); ++dimension) {
          if (int(dimension) != axis && part->shape[dimension] != first->shape[dimension]) return fail();
        }
        metadata.shape[std::size_t(axis)] += part->shape[std::size_t(axis)];
        metadata.data.insert(metadata.data.end(), part->data.begin(), part->data.end());
      }
      shape_values.emplace(node.out[0], std::move(metadata));
      for (const auto& name : node.in) {
        if (name.empty() || impl.graph.initializers.contains(name) || !consume(name)) continue;
        shape_values.erase(name);
      }
      continue;
    }
    const auto* input = fetch(node.in[0]);
    if (!input) return fail();
    DeviceValue result;
    bool ok{};
    bool aliases_first_input{};
    if (node.op == "Shape") {
      Tensor shape{{static_cast<std::int64_t>(input->shape.size())}, {}};
      shape.data.reserve(input->shape.size());
      for (const auto dimension : input->shape) shape.data.push_back(static_cast<float>(dimension));
      shape_values.emplace(node.out[0], std::move(shape));
      ok = true;
      // Shape is scheduling metadata only: it neither reads nor writes a
      // Vulkan storage buffer.  Do not force an otherwise empty submission
      // for it, especially after a fenced segment; some drivers reject that
      // submission even though the preceding compute work completed.
      for (const auto& name : node.in) {
        if (name.empty() || impl.graph.initializers.contains(name) || !consume(name)) continue;
        release(name);
      }
      continue;
    } else if (node.op == "Identity" || node.op == "Reshape" || node.op == "Squeeze" || node.op == "Unsqueeze") {
      result = *input;
      if (!final_use(node.in[0])) {
        result.slot = arena.Acquire(input->slot.live_elements, node.out[0]);
        if (!result.slot.resident || !arena.Copy(input->slot, result.slot, input->slot.live_elements)) {
          arena.Release(result.slot); return fail();
        }
        aliases_first_input = false;
      }
      if (node.op == "Reshape") {
        if (node.in.size() < 2) return fail();
        const auto* spec = get_shape_value(node.in[1]);
        if (!spec) return fail();
        result.shape = DeviceReshapeShape(*input, *spec);
      } else if (node.op != "Identity") {
        std::vector<std::int64_t> axes = AttrInts(node, "axes", {});
        if (axes.empty() && node.in.size() > 1) {
          const auto* axes_tensor = get_shape_value(node.in[1]);
          if (!axes_tensor) return fail();
          for (const float axis : axes_tensor->data) axes.push_back(std::int64_t(axis));
        }
        if (node.op == "Unsqueeze") {
          if (axes.empty()) return fail();
          std::sort(axes.begin(), axes.end());
          for (auto axis : axes) {
            if (axis < 0) axis += std::int64_t(result.shape.size()) + 1;
            if (axis < 0 || axis > static_cast<std::int64_t>(result.shape.size())) return fail();
            result.shape.insert(result.shape.begin() + axis, 1);
          }
        } else {
          if (axes.empty()) {
            result.shape.erase(std::remove(result.shape.begin(), result.shape.end(), 1), result.shape.end());
            if (result.shape.empty()) result.shape.push_back(1);
          } else {
            std::sort(axes.rbegin(), axes.rend());
            for (auto axis : axes) {
              const int pos = Axis(axis, int(result.shape.size()));
              if (result.shape[std::size_t(pos)] != 1) return fail();
              result.shape.erase(result.shape.begin() + pos);
            }
          }
        }
      }
      values.emplace(node.out[0], result); ok = true; aliases_first_input = final_use(node.in[0]);
    } else if (node.op == "Relu" || node.op == "Sigmoid" || node.op == "FusedHardSwish" ||
               node.op == "FusedGelu" || node.op == "HardSigmoid") {
      if (input->slot.live_elements == 0) return fail();
      result = *input;
      if (!final_use(node.in[0])) {
        result.slot = arena.Acquire(input->slot.live_elements, node.out[0]);
        if (!result.slot.resident || !arena.Copy(input->slot, result.slot, input->slot.live_elements)) {
          arena.Release(result.slot); return fail();
        }
      }
      VulkanUnaryOp op = node.op == "Relu" ? VulkanUnaryOp::relu :
          node.op == "Sigmoid" ? VulkanUnaryOp::sigmoid : node.op == "FusedHardSwish" ? VulkanUnaryOp::hard_swish :
          node.op == "FusedGelu" ? VulkanUnaryOp::gelu : VulkanUnaryOp::hard_sigmoid;
      ok = arena.UnaryInplace(result.slot, op,
          node.attr.contains("alpha") ? Attr(node, "alpha").f : (op == VulkanUnaryOp::hard_swish ? 1.F/6.F : .2F),
          node.attr.contains("beta") ? Attr(node, "beta").f : .5F);
      if (ok) { values.emplace(node.out[0], result); aliases_first_input = final_use(node.in[0]); }
    } else if (node.op == "BatchNormalization" && node.in.size() == 5 && input->shape.size() >= 2) {
      // BatchNorm inference is a channel affine.  Deriving the compact
      // scale/bias pair from immutable parameters on the host does not touch
      // any activation; both the transform and its result remain on device.
      const auto* gamma = get_constant(node.in[1]);
      const auto* beta = get_constant(node.in[2]);
      const auto* mean = get_constant(node.in[3]);
      const auto* variance = get_constant(node.in[4]);
      const auto& gamma_meta = impl.graph.initializers.at(node.in[1]);
      const auto& beta_meta = impl.graph.initializers.at(node.in[2]);
      const auto& mean_meta = impl.graph.initializers.at(node.in[3]);
      const auto& variance_meta = impl.graph.initializers.at(node.in[4]);
      const int channels = int(input->shape[1]);
      std::size_t plane = 1;
      for (std::size_t axis = 2; axis < input->shape.size(); ++axis) plane *= std::size_t(input->shape[axis]);
      if (!gamma || !beta || !mean || !variance || channels <= 0 || plane == 0 ||
          gamma_meta.data.size() != std::size_t(channels) || beta_meta.data.size() != std::size_t(channels) ||
          mean_meta.data.size() != std::size_t(channels) || variance_meta.data.size() != std::size_t(channels)) return fail();
      const float epsilon = node.attr.contains("epsilon") ? Attr(node, "epsilon").f : 1e-5F;
      std::vector<float> scale(static_cast<std::size_t>(channels), 0.F);
      std::vector<float> offset(static_cast<std::size_t>(channels), 0.F);
      for (int channel = 0; channel < channels; ++channel) {
        scale[std::size_t(channel)] = gamma_meta.data[std::size_t(channel)] /
            std::sqrt(variance_meta.data[std::size_t(channel)] + epsilon);
        offset[std::size_t(channel)] = beta_meta.data[std::size_t(channel)] -
            mean_meta.data[std::size_t(channel)] * scale[std::size_t(channel)];
      }
      auto affine_scale = arena.Acquire(scale.size(), "gpu-batchnorm-scale");
      auto affine_offset = arena.Acquire(offset.size(), "gpu-batchnorm-offset");
      result = *input;
      if (!final_use(node.in[0])) {
        result.slot = arena.Acquire(input->slot.live_elements, node.out[0]);
        ok = result.slot.resident && arena.Copy(input->slot, result.slot, input->slot.live_elements);
      } else ok = true;
      ok = ok && affine_scale.resident && affine_offset.resident &&
          arena.Upload(affine_scale, scale.data(), scale.size()) &&
          arena.Upload(affine_offset, offset.data(), offset.size()) &&
          arena.ChannelAffineInplace(result.slot, affine_scale, affine_offset,
              std::size_t(input->shape[0]), channels, plane);
      arena.Release(affine_scale); arena.Release(affine_offset);
      if (ok) { values.emplace(node.out[0], result); aliases_first_input = final_use(node.in[0]); }
      else if (!final_use(node.in[0])) arena.Release(result.slot);
    } else if ((node.op == "FusedBatchNormSwish" || node.op == "FusedBatchNormSwishAdd" ||
                node.op == "FusedBatchNormHardSwish") &&
               (node.in.size() == 3 || node.in.size() == 4) && input->shape.size() >= 2) {
      // These graph fusions already store compact affine factor/offset in
      // inputs 1/2.  Keep the whole affine->Swish(->Add) sequence in the
      // arena; no activation is materialised merely for the fused form.
      const auto* scale = get_constant(node.in[1]);
      const auto* offset = get_constant(node.in[2]);
      const int channels = int(input->shape[1]);
      std::size_t plane = 1;
      for (std::size_t axis = 2; axis < input->shape.size(); ++axis) plane *= std::size_t(input->shape[axis]);
      const DeviceValue* residual = node.op == "FusedBatchNormSwishAdd" ? fetch(node.in[3]) : nullptr;
      if (!scale || !offset || channels <= 0 || plane == 0 ||
          impl.graph.initializers.at(node.in[1]).data.size() != std::size_t(channels) ||
          impl.graph.initializers.at(node.in[2]).data.size() != std::size_t(channels) ||
          (residual && residual->shape != input->shape) ||
          (node.op == "FusedBatchNormSwishAdd" && !residual)) return fail();
      result = *input;
      if (!final_use(node.in[0])) {
        result.slot = arena.Acquire(input->slot.live_elements, node.out[0]);
        if (!result.slot.resident || !arena.Copy(input->slot, result.slot, input->slot.live_elements)) {
          arena.Release(result.slot); return fail();
        }
      }
      ok = node.op == "FusedBatchNormHardSwish"
          ? arena.ChannelAffineHardSwishInplace(result.slot, *scale, *offset,
                                                std::size_t(input->shape[0]), channels, plane)
          : arena.ChannelAffineSwishInplace(result.slot, *scale, *offset,
                                            std::size_t(input->shape[0]), channels, plane);
      if (ok && residual) ok = arena.BinaryInplace(result.slot, residual->slot, kernels::BinaryOp::add);
      if (ok) { values.emplace(node.out[0], result); aliases_first_input = final_use(node.in[0]); }
      else if (!final_use(node.in[0])) arena.Release(result.slot);
    } else if (node.op == "FusedLayerNorm" && node.in.size() == 3 && !input->shape.empty()) {
      const auto* gamma = get_constant(node.in[1]);
      const auto* beta = get_constant(node.in[2]);
      const std::size_t width = std::size_t(input->shape.back());
      if (!gamma || !beta || width == 0 || input->slot.live_elements % width != 0 ||
          impl.graph.initializers.at(node.in[1]).data.size() != width ||
          impl.graph.initializers.at(node.in[2]).data.size() != width) return fail();
      result = *input;
      if (!final_use(node.in[0])) {
        result.slot = arena.Acquire(input->slot.live_elements, node.out[0]);
        if (!result.slot.resident || !arena.Copy(input->slot, result.slot, input->slot.live_elements)) {
          arena.Release(result.slot); return fail();
        }
      }
      ok = arena.LayerNormInplace(result.slot, *gamma, *beta,
          input->slot.live_elements / width, width,
          node.attr.contains("epsilon") ? Attr(node, "epsilon").f : 1e-5F);
      if (ok) { values.emplace(node.out[0], result); aliases_first_input = final_use(node.in[0]); }
    } else if ((node.op == "Add" || node.op == "Sub" || node.op == "Mul" || node.op == "Div") && node.in.size() == 2) {
      const auto* right = fetch(node.in[1]);
      const auto* right_constant = right ? nullptr : get_constant(node.in[1]);
      const auto constant_meta = right ? nullptr : get_shape_value(node.in[1]);
      if (!right && (!right_constant || !constant_meta)) {
        if (std::getenv("PPOCR_GPU_ONLY_TRACE") != nullptr)
          std::cerr << "GPU-only binary missing rhs value=" << node.in[1] << '\n';
        return fail();
      }
      result = *input;
      if (!final_use(node.in[0])) {
        result.slot = arena.Acquire(input->slot.live_elements, node.out[0]);
        if (!result.slot.resident || !arena.Copy(input->slot, result.slot, input->slot.live_elements)) {
          arena.Release(result.slot); return fail();
        }
      }
      const auto operation = node.op == "Add" ? kernels::BinaryOp::add : node.op == "Sub" ? kernels::BinaryOp::sub :
          node.op == "Mul" ? kernels::BinaryOp::mul : kernels::BinaryOp::div;
      // An ONNX initializer is not placed in `values`: its storage is a
      // persistent Vulkan constant slot.  Treat it exactly like a dynamic
      // right hand input here, rather than requiring a host expansion of
      // residual / channel-broadcast constants.  In particular the small
      // recognizer's first residual is exported as `[1,C,1,1]`.
      const auto* right_slot = right ? &right->slot : right_constant;
      const auto* right_shape = right ? &right->shape
                                      : (constant_meta ? &constant_meta->shape : nullptr);
      if (right_slot && right_shape && input->shape == *right_shape) {
        ok = arena.BinaryInplace(result.slot, *right_slot, operation);
      } else if (right_constant && constant_meta && constant_meta->data.size() == 1) {
        // Attention scaling is exported as `[N,H,T,D] * [1]`.  A scalar is
        // a valid contiguous suffix broadcast; retain it as a one-float
        // device constant instead of expanding or calculating activations on
        // the host.
        ok = arena.BinaryBroadcast(result.slot, *right_constant, result.slot,
            1, input->slot.live_elements, 1, true, operation);
      } else {
        // The exported SE gates are `[N,C,H,W] op [N,C,1,1]` (or a
        // batch-shared `[1,C,1,1]`).  Preserve the compact gate tensor and
        // broadcast in the shader, never by expanding it on CPU.
        if (!right_slot || !right_shape || input->shape.size() != 4 || right_shape->size() != 4 ||
            input->shape[1] != (*right_shape)[1] || (*right_shape)[2] != 1 ||
            (*right_shape)[3] != 1 ||
            !((*right_shape)[0] == 1 || (*right_shape)[0] == input->shape[0])) {
          if (std::getenv("PPOCR_GPU_ONLY_TRACE") != nullptr) {
            std::cerr << "GPU-only binary unsupported broadcast lhs=";
            for (const auto dim : input->shape) std::cerr << dim << ',';
            std::cerr << " rhs=";
            if (right_shape) for (const auto dim : *right_shape) std::cerr << dim << ',';
            std::cerr << '\n';
          }
          return fail();
        }
        ok = arena.BinaryBroadcast(result.slot, *right_slot, result.slot,
            std::size_t(input->shape[0]), std::size_t(input->shape[2]) * input->shape[3],
            std::size_t(input->shape[1]), (*right_shape)[0] == 1, operation);
      }
      if (ok) { values.emplace(node.out[0], result); aliases_first_input = final_use(node.in[0]); }
    } else if (node.op == "Concat") {
      std::vector<VulkanTensorSlot> slots; std::vector<std::size_t> axes;
      if (node.in.size() < 2) return fail();
      const int axis = Axis(AttrInt(node, "axis", 0), int(input->shape.size()));
      std::size_t outer = Elements(std::vector<std::int64_t>(input->shape.begin(), input->shape.begin() + axis));
      std::size_t inner = Elements(std::vector<std::int64_t>(input->shape.begin() + axis + 1, input->shape.end()));
      result.shape = input->shape; result.shape[axis] = 0;
      for (const auto& name : node.in) { const auto* value = fetch(name); if (!value || value->shape.size() != input->shape.size()) return fail(); slots.push_back(value->slot); axes.push_back(value->shape[axis]); result.shape[axis] += value->shape[axis]; }
      result.slot = arena.Acquire(Elements(result.shape), node.out[0]);
      ok = result.slot.resident && arena.Concat(slots, result.slot, outer, inner, axes);
      if (ok) values.emplace(node.out[0], result); else arena.Release(result.slot);
    } else if (node.op == "Transpose") {
      auto permutation = AttrInts(node, "perm", {});
      if (permutation.empty()) {
        permutation.resize(input->shape.size());
        for (std::size_t axis = 0; axis < permutation.size(); ++axis) permutation[axis] = std::int64_t(permutation.size() - axis - 1);
      }
      if (permutation.size() != input->shape.size() || input->shape.size() < 2 || input->shape.size() > 4) return fail();
      // Rec head is [C,T] or [1,C,T] then Transpose to [T,C] before GEMM.
      // Skipping the copy makes GEMM gather A with stride T; 8-run on the
      // 780M was no faster than Transpose4D + packed GEMM. Keep ENABLE-only.
      // `PPOCR_ENABLE_GPU_TRANSPOSE_VIEW` turns the view path on.
      static const bool transpose_view =
          std::getenv("PPOCR_ENABLE_GPU_TRANSPOSE_VIEW") != nullptr &&
          std::getenv("PPOCR_DISABLE_GPU_TRANSPOSE_VIEW") == nullptr;
      const auto swap_last_two = [&]() -> bool {
        const int rank = int(permutation.size());
        if (rank == 2) return permutation[0] == 1 && permutation[1] == 0;
        if (rank == 3) return permutation[0] == 0 && permutation[1] == 2 && permutation[2] == 1;
        return permutation[0] == 0 && permutation[1] == 1 && permutation[2] == 3 && permutation[3] == 2;
      };
      const auto gemm_consumer = [&]() -> bool {
        const Node* consumer = nullptr;
        for (const auto& later : impl.graph.nodes) {
          for (const auto& in : later.in) {
            if (in != node.out[0]) continue;
            if (consumer) return false;
            consumer = &later;
          }
        }
        return consumer && (consumer->op == "MatMul" || consumer->op == "FusedMatMulBias" ||
                            consumer->op == "FusedMatMulBiasSwish");
      };
      if (transpose_view && swap_last_two() && final_use(node.in[0]) &&
          !impl.graph_outputs.contains(node.out[0]) && gemm_consumer() &&
          input->shape.back() > 0 &&
          (input->shape.size() < 3 || input->shape[0] == 1) &&
          input->a_lda == 0) {
        result = *input;
        std::swap(result.shape[result.shape.size() - 2], result.shape.back());
        result.a_lda = int(input->shape.back());
        values.emplace(node.out[0], result);
        aliases_first_input = true;
        ok = true;
      } else {
      std::array<int, 4> dimensions{1, 1, 1, 1};
      std::array<int, 4> device_permutation{0, 1, 2, 3};
      const int pad = 4 - int(input->shape.size());
      result.shape.resize(input->shape.size());
      for (std::size_t axis = 0; axis < input->shape.size(); ++axis) {
        if (permutation[axis] < 0 || permutation[axis] >= std::int64_t(input->shape.size())) return fail();
        dimensions[std::size_t(pad) + axis] = int(input->shape[axis]);
        device_permutation[std::size_t(pad) + axis] = int(permutation[axis]) + pad;
        result.shape[axis] = input->shape[std::size_t(permutation[axis])];
      }
      result.slot = arena.Acquire(input->slot.live_elements, node.out[0]);
      ok = result.slot.resident && arena.Transpose4D(input->slot, result.slot, dimensions, device_permutation);
      if (ok) values.emplace(node.out[0], result); else arena.Release(result.slot);
      }
    } else if (node.op == "FusedQkvSplit" && node.in.size() == 1 && input->shape.size() == 5 && node.out.size() == 3) {
      const int batches = int(input->shape[0]), steps = int(input->shape[1]);
      const int heads = int(input->shape[3]), width = int(input->shape[4]);
      if (input->shape[2] != 3 || batches <= 0 || steps <= 0 || heads <= 0 || width <= 0) return fail();
      const std::vector<std::int64_t> split_shape{input->shape[0], input->shape[3], input->shape[1], input->shape[4]};
      auto query = arena.Acquire(Elements(split_shape), node.out[0]);
      auto key = arena.Acquire(Elements(split_shape), node.out[1]);
      auto value = arena.Acquire(Elements(split_shape), node.out[2]);
      ok = query.resident && key.resident && value.resident &&
          arena.QkvSplit(input->slot, query, key, value, batches, steps, heads, width);
      if (!ok) { arena.Release(query); arena.Release(key); arena.Release(value); }
      else {
        values.emplace(node.out[0], DeviceValue{split_shape, query});
        values.emplace(node.out[1], DeviceValue{split_shape, key});
        values.emplace(node.out[2], DeviceValue{split_shape, value});
      }
    } else if (node.op == "MatMul" && node.in.size() == 2 &&
               (input->shape.size() == 3 || input->shape.size() == 4)) {
      // Multi-head attention is emitted as two device tensors after QKV
      // projection/reshape/transpose: `[B,M,K] x [B,K,N]`.  Do not confuse
      // this with the linear projection form below whose right side is an
      // immutable weight matrix.  The arena GEMM maps B to batch*heads and
      // leaves QK^T / Softmax / AV fully resident.
      const auto* right = fetch(node.in[1]);
      if (!right || right->shape.size() != input->shape.size()) return fail();
      const int rank = int(input->shape.size());
      std::size_t batches = 1;
      for (int axis = 0; axis < rank - 2; ++axis) {
        if (input->shape[std::size_t(axis)] != right->shape[std::size_t(axis)] ||
            input->shape[std::size_t(axis)] <= 0) return fail();
        batches *= std::size_t(input->shape[std::size_t(axis)]);
      }
      const int rows = int(input->shape[std::size_t(rank - 2)]);
      const int depth = int(input->shape[std::size_t(rank - 1)]);
      const int columns = int(right->shape[std::size_t(rank - 1)]);
      if (batches == 0 || rows <= 0 || depth <= 0 || columns <= 0 ||
          right->shape[std::size_t(rank - 2)] != depth) return fail();
      result.shape = input->shape;
      result.shape[std::size_t(rank - 1)] = columns;
      result.slot = arena.Acquire(Elements(result.shape), node.out[0]);
      ok = result.slot.resident && arena.BatchedGemm(input->slot, right->slot, result.slot,
          batches, rows, depth, columns);
      if (ok) values.emplace(node.out[0], result); else arena.Release(result.slot);
    } else if ((node.op == "MatMul" || node.op == "FusedMatMulBias" ||
                node.op == "FusedMatMulBiasSwish") && node.in.size() >= 2 &&
               input->shape.size() >= 2) {
      const auto* weights = get_constant(node.in[1]);
      const auto& weight_meta = impl.graph.initializers.at(node.in[1]);
      const VulkanTensorSlot* bias = nullptr;
      if (node.op != "MatMul") {
        if (node.in.size() < 3) return fail();
        bias = get_constant(node.in[2]);
      }
      const int depth = int(input->shape.back());
      const int rows = int(input->slot.live_elements / std::size_t(depth));
      const int columns = weight_meta.shape.size() == 2 ? int(weight_meta.shape[1]) : 0;
      if (!weights || (node.op != "MatMul" && !bias) || weight_meta.shape.size() != 2 ||
          weight_meta.shape[0] != depth || depth <= 0 || rows <= 0 || columns <= 0 ||
          (bias && impl.graph.initializers.at(node.in[2]).data.size() != std::size_t(columns))) return fail();
      result.shape = input->shape; result.shape.back() = columns;
      result.a_lda = 0;
      result.slot = arena.Acquire(Elements(result.shape), node.out[0]);
      static const bool fuse_gemm_ctc =
          std::getenv("PPOCR_ENABLE_GPU_FUSE_GEMM_CTC") != nullptr &&
          std::getenv("PPOCR_DISABLE_GPU_FUSE_GEMM_CTC") == nullptr;
      const bool terminal_gemm_ctc = fuse_gemm_ctc && fuse_terminal_ctc_softmax &&
          impl.terminal_logits_is_matmul && node.op == "FusedMatMulBias" &&
          impl.graph.nodes.size() >= 2 &&
          impl.graph.nodes[impl.graph.nodes.size() - 2].out.size() == 1 &&
          impl.graph.nodes[impl.graph.nodes.size() - 2].out[0] == node.out[0];
      if (terminal_gemm_ctc && result.slot.resident) {
        const int batches = input->shape.size() >= 3 ? int(input->shape[0]) : 1;
        const int steps = batches > 0 ? rows / batches : 0;
        gemm_ctc_indices = arena.Acquire(std::size_t(rows), "gpu-gemm-ctc-idx");
        gemm_ctc_probs = arena.Acquire(std::size_t(rows), "gpu-gemm-ctc-prob");
        ok = gemm_ctc_indices.resident && gemm_ctc_probs.resident && steps > 0 &&
            rows == batches * steps &&
            arena.GemmCtcTop1(input->slot, *weights, bias, gemm_ctc_indices, gemm_ctc_probs,
                              rows, depth, columns);
        if (ok) {
          gemm_ctc_batches = batches;
          gemm_ctc_steps = steps;
          gemm_ctc_classes = columns;
        } else {
          if (gemm_ctc_indices.resident) arena.Release(gemm_ctc_indices);
          if (gemm_ctc_probs.resident) arena.Release(gemm_ctc_probs);
          gemm_ctc_indices = {};
          gemm_ctc_probs = {};
        }
      }
      if (!ok)
        ok = result.slot.resident &&
            arena.Gemm(input->slot, *weights, bias, result.slot, rows, depth, columns,
                       node.op == "FusedMatMulBiasSwish", input->a_lda);
      if (ok) values.emplace(node.out[0], result); else arena.Release(result.slot);
    } else if (node.op == "Softmax") {
      const int axis = node.attr.contains("axis") ? Axis(Attr(node, "axis").i, int(input->shape.size())) :
          (impl.graph.opset >= 13 ? int(input->shape.size()) - 1 : 1);
      const std::size_t width = std::size_t(input->shape[axis]);
      if (axis != int(input->shape.size()) - 1 || width == 0 || input->slot.live_elements % width != 0) return fail();
      // A recognizer's final Softmax feeds only greedy CTC.  Keep its logits
      // resident and let the terminal CTC reduction reproduce Softmax's
      // stable max/sum sequence while selecting the winning class.  This is
      // an algebraic fusion, not an approximation: for the winning logit the
      // corresponding Softmax probability is exactly 1 / sum(exp(x - max)).
      const bool terminal_ctc_softmax = fuse_terminal_ctc_softmax &&
          impl.graph.outputs.size() == 1 && node.out[0] == impl.graph.outputs[0];
      result = *input;
      if (!final_use(node.in[0])) {
        result.slot = arena.Acquire(input->slot.live_elements, node.out[0]);
        if (!result.slot.resident || !arena.Copy(input->slot, result.slot, input->slot.live_elements)) {
          arena.Release(result.slot); return fail();
        }
      }
      ok = terminal_ctc_softmax ||
          arena.SoftmaxRowsInplace(result.slot, input->slot.live_elements / width, width);
      if (ok) { values.emplace(node.out[0], result); aliases_first_input = final_use(node.in[0]); }
    } else if (node.op == "FusedSEGateMul" && node.in.size() == 5 && IsNchw(*input)) {
      const auto* w1 = get_constant(node.in[1]); const auto* b1 = get_constant(node.in[2]);
      const auto* w2 = get_constant(node.in[3]); const auto* b2 = get_constant(node.in[4]);
      if (!w1 || !b1 || !w2 || !b2) return fail();
      const int channels = int(input->shape[1]); const int reduced = int(impl.graph.initializers.at(node.in[1]).shape[0]);
      result.shape = input->shape; result.slot = arena.Acquire(input->slot.live_elements, node.out[0]);
      ok = result.slot.resident && arena.SqueezeExcitationGate(input->slot, *w1, *b1, *w2, *b2, result.slot,
          std::size_t(input->shape[0]), channels, reduced, std::size_t(input->shape[2]) * input->shape[3],
          Attr(node, "alpha").f, Attr(node, "beta").f);
      // residual=1 is x*(1+SE(x)) = x + x*SE(x). SqueezeExcitationGate already
      // writes x*SE(x); adding the source stays on device.
      if (ok && AttrInt(node, "residual", 0) != 0)
        ok = arena.BinaryInplace(result.slot, input->slot, kernels::BinaryOp::add);
      if (ok) values.emplace(node.out[0], result); else arena.Release(result.slot);
    } else if ((node.op == "Conv" || node.op == "FusedConvRelu" ||
                node.op == "FusedDepthwiseHardSwish") && node.in.size() >= 2 && IsNchw(*input) &&
               AttrInt(node, "group", 1) == input->shape[1]) {
      // Unfused depthwise Conv appears between the residual blocks in every
      // detector size.  Its `[C,1,KH,KW]` layout maps directly to the arena
      // depthwise primitive and avoids routing that large feature map back to
      // the host merely because it was not paired by the graph rewriter.
      const auto* weights = get_constant(node.in[1]);
      const auto& meta = impl.graph.initializers.at(node.in[1]);
      const auto strides = AttrInts(node, "strides", {1, 1});
      const auto pads = AttrInts(node, "pads", {0, 0, 0, 0});
      const auto dilations = AttrInts(node, "dilations", {1, 1});
      const int batches = int(input->shape[0]), channels = int(input->shape[1]);
      const int height = int(input->shape[2]), width = int(input->shape[3]);
      const int kernel_h = meta.shape.size() == 4 ? int(meta.shape[2]) : 0;
      const int kernel_w = meta.shape.size() == 4 ? int(meta.shape[3]) : 0;
      const int out_h = (height + (pads.size() == 4 ? int(pads[0] + pads[2]) : 0) - kernel_h) /
          (strides.size() == 2 ? int(strides[0]) : 1) + 1;
      const int out_w = (width + (pads.size() == 4 ? int(pads[1] + pads[3]) : 0) - kernel_w) /
          (strides.size() == 2 ? int(strides[1]) : 1) + 1;
      if (!weights || meta.shape.size() != 4 || meta.shape[0] != channels || meta.shape[1] != 1 ||
          strides.size() != 2 || pads.size() != 4 || dilations != std::vector<std::int64_t>{1, 1} ||
          strides[0] <= 0 || strides[1] <= 0 || out_h <= 0 || out_w <= 0) return fail();
      if (std::getenv("PPOCR_GPU_ONLY_TRACE") != nullptr) {
        std::cerr << "GPU-only depthwise shape N=" << batches << " C=" << channels
                  << " input=" << height << 'x' << width << " output=" << out_h << 'x' << out_w
                  << " kernel=" << kernel_h << 'x' << kernel_w << '\n';
      }
      VulkanTensorSlot synthetic_bias; const VulkanTensorSlot* bias = nullptr;
      if (node.in.size() > 2 && !node.in[2].empty()) bias = get_constant(node.in[2]);
      if (!bias) {
        std::vector<float> zeros(static_cast<std::size_t>(channels), 0.F);
        synthetic_bias = arena.Acquire(zeros.size(), "gpu-zero-depthwise-bias");
        if (!synthetic_bias.resident || !arena.Upload(synthetic_bias, zeros.data(), zeros.size())) return fail();
        bias = &synthetic_bias;
      }
      // This high-channel 3x3 terminal depthwise has its own scalar Vulkan
      // shader.  It is selected by the exact operation shape, with an
      // explicit generic-kernel override kept only for driver diagnostics.
      // No branch here transfers an activation to CPU or evaluates a CPU
      // neural operator.
      const bool scalar_tail = std::getenv("PPOCR_GPU_TAIL_GENERIC") == nullptr &&
          node.out[0] == "p2o.pd_op.depthwise_conv2d.12.0" &&
          channels == 896 &&
          ((height == 5 && width == 22 && out_h == 5 && out_w == 22) ||
            (height == 1 && width == 30 && out_h == 1 && out_w == 30)) &&
          node.op == "Conv";
      if (std::getenv("PPOCR_GPU_ONLY_TRACE") != nullptr &&
          node.out[0] == "p2o.pd_op.depthwise_conv2d.12.0") {
        std::cerr << "GPU-only terminal tail scalar=" << scalar_tail << '\n';
      }
      result.shape = {input->shape[0], channels, out_h, out_w};
      result.slot = arena.Acquire(Elements(result.shape), node.out[0]);
      if (std::getenv("PPOCR_GPU_ONLY_TRACE") != nullptr) {
        std::cerr << "GPU-only depthwise slots input=" << input->slot.index
                  << " weights=" << weights->index << " bias=" << bias->index
                  << " output=" << result.slot.index << "\n";
      }
      if (scalar_tail && std::getenv("PPOCR_GPU_ONLY_TRACE") != nullptr) {
        std::cerr << "GPU-only medium-tail identities input=" << input->slot.index
                  << " weights=" << weights->index << " bias=" << bias->index
                  << " output=" << result.slot.index << " aliases="
                  << (input->slot.index == weights->index) << ','
                  << (input->slot.index == bias->index) << ','
                  << (input->slot.index == result.slot.index) << ','
                  << (weights->index == bias->index) << ','
                  << (weights->index == result.slot.index) << ','
                  << (bias->index == result.slot.index) << '\n';
      }
      ok = result.slot.resident && (scalar_tail
          ? arena.DepthwiseConvScalar(input->slot, *weights, *bias, result.slot,
              batches, channels, height, width, out_h, out_w, kernel_h, kernel_w,
              int(strides[0]), int(strides[1]), int(pads[0]), int(pads[1]))
          : arena.DepthwiseConv(input->slot, *weights, *bias, result.slot,
              batches, channels, height, width, out_h, out_w, kernel_h, kernel_w,
              int(strides[0]), int(strides[1]), int(pads[0]), int(pads[1]),
              node.op == "FusedConvRelu", false,
              node.op == "FusedDepthwiseHardSwish"));
      if (synthetic_bias.resident) arena.Release(synthetic_bias);
      if (ok) values.emplace(node.out[0], result);
      else arena.Release(result.slot);
    } else if ((node.op == "Conv" || node.op == "FusedConvRelu" || node.op == "FusedConvGelu" ||
                node.op == "FusedConvSigmoid" ||
                node.op == "FusedConvHardSigmoid" || node.op == "FusedConvSwish" ||
                node.op == "FusedConvHardSwish" || node.op == "FusedConvChannelBias") &&
               node.in.size() >= 2 && IsNchw(*input)) {
      const auto* weights = get_constant(node.in[1]);
      if (!weights) return fail();
      const Tensor* weight_meta = &impl.graph.initializers.at(node.in[1]);
      if (weight_meta->shape.size() != 4) return fail();
      const int batches = int(input->shape[0]), input_channels = int(input->shape[1]);
      const int height = int(input->shape[2]), width = int(input->shape[3]);
      const int output_channels = int(weight_meta->shape[0]), kernel_h = int(weight_meta->shape[2]), kernel_w = int(weight_meta->shape[3]);
      const auto strides = AttrInts(node, "strides", {1, 1});
      auto pads = AttrInts(node, "pads", {0, 0, 0, 0});
      const auto dilations = AttrInts(node, "dilations", {1, 1});
      const auto auto_pad = AttrStr(node, "auto_pad", "NOTSET");
      if (batches <= 0 || input_channels <= 0 || output_channels <= 0 || weight_meta->shape[1] <= 0 ||
          strides.size() != 2 || pads.size() != 4 || dilations != std::vector<std::int64_t>{1, 1} ||
          AttrInt(node, "group", 1) != 1 || strides[0] != strides[1] ||
          (auto_pad != "NOTSET" && auto_pad != "SAME_UPPER" && auto_pad != "SAME_LOWER")) return fail();
      int out_h{}, out_w{};
      if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
        out_h = (height + int(strides[0]) - 1) / int(strides[0]);
        out_w = (width + int(strides[1]) - 1) / int(strides[1]);
        const int total_h = std::max(0, (out_h - 1) * int(strides[0]) + kernel_h - height);
        const int total_w = std::max(0, (out_w - 1) * int(strides[1]) + kernel_w - width);
        pads[0] = auto_pad == "SAME_LOWER" ? (total_h + 1) / 2 : total_h / 2;
        pads[2] = total_h - pads[0];
        pads[1] = auto_pad == "SAME_LOWER" ? (total_w + 1) / 2 : total_w / 2;
        pads[3] = total_w - pads[1];
      } else {
        out_h = (height + int(pads[0] + pads[2]) - kernel_h) / int(strides[0]) + 1;
        out_w = (width + int(pads[1] + pads[3]) - kernel_w) / int(strides[1]) + 1;
      }
      if (out_h <= 0 || out_w <= 0 || weight_meta->shape[1] != input_channels) return fail();
      // The arena kernels require an explicit bias. Bias-free exported Conv is
      // represented by a short zero device tensor rather than host execution.
      VulkanTensorSlot synthetic_bias;
      const VulkanTensorSlot* bias = nullptr;
      if (node.in.size() > 2 && !node.in[2].empty()) bias = get_constant(node.in[2]);
      if (!bias) {
        std::vector<float> zeros(static_cast<std::size_t>(output_channels));
        synthetic_bias = arena.Acquire(zeros.size(), "gpu-zero-conv-bias");
        if (!synthetic_bias.resident || !arena.Upload(synthetic_bias, zeros.data(), zeros.size())) return fail();
        bias = &synthetic_bias;
      }
      result.shape = {input->shape[0], output_channels, out_h, out_w};
      result.slot = arena.Acquire(Elements(result.shape), node.out[0]);
      const bool relu = node.op == "FusedConvRelu";
      const bool swish = node.op == "FusedConvSwish";
      const bool sigmoid = node.op == "FusedConvSigmoid";
      const bool hard_swish = node.op == "FusedConvHardSwish";
      // Keep HardSigmoid as a separate device unary.  Conv2d deliberately has
      // no hard-sigmoid selector: treating it as an unrecognised Conv mode
      // used to silently produce an *unactivated* gate in the strict graph.
      // The unary accepts exporter-provided alpha/beta and is still entirely
      // device-resident.
      const bool hard_sigmoid = node.op == "FusedConvHardSigmoid";
      if (hard_sigmoid && (!node.attr.contains("__ppocr_fused_alpha") ||
                           !node.attr.contains("__ppocr_fused_beta"))) return fail();
      const bool gelu = node.op == "FusedConvGelu";
      // FPN laterals (Conv.54/57/60/63) are bias-free 1x1s. Conv2d's generic
      // 3x3/tiled path is the wrong workgroup mapping; PointwiseConv is the
      // same 1x1 reduction as the CPU kernel.
      const bool pointwise_1x1 = kernel_h == 1 && kernel_w == 1 &&
          strides[0] == 1 && strides[1] == 1 &&
          pads[0] == 0 && pads[1] == 0 && pads[2] == 0 && pads[3] == 0 &&
          !hard_sigmoid;
      // `FusedConvHardSigmoid` is represented as Conv followed by a separate
      // device unary. Do not pass it as the ReLU selector: Conv2d's boolean
      // argument immediately after padding is `relu`, and a previous call
      // accidentally routed the `hard_sigmoid` flag there.
      if (std::getenv("PPOCR_GPU_ONLY_TRACE") != nullptr && hard_sigmoid) {
        std::cerr << "GPU-only hard-sigmoid alpha=" << Attr(node, "__ppocr_fused_alpha").f
                  << " beta=" << Attr(node, "__ppocr_fused_beta").f << '\n';
      }
      if (pointwise_1x1) {
        const auto plane = std::size_t(out_h) * out_w;
        // The terminal medium-detector projection is a 1792 -> 896 1x1 on
        // a 5x22 plane.  Its own tiny scalar module avoids the generic
        // mega-shader path on affected LLPC adapters.  This is still one
        // Vulkan dispatch over device-resident input/weights/output; no host
        // activation is created. Keep the narrow predicate so all ordinary
        // pointwise layers retain their tiled/vector implementation.
        const bool pointwise_tail = std::getenv("PPOCR_GPU_TAIL_GENERIC") == nullptr &&
            node.out[0] == "p2o.pd_op.conv2d.39.0" && batches == 1 &&
            input_channels == 1792 && output_channels == 896 && plane == 110 &&
            !relu && !swish && !gelu && !sigmoid && !hard_swish;
        ok = result.slot.resident && (pointwise_tail
            ? arena.PointwiseConvTail(input->slot, *weights, *bias, result.slot,
                                      std::size_t(batches), input_channels,
                                      output_channels, plane)
            : arena.PointwiseConv(input->slot, *weights, *bias, result.slot,
                                  std::size_t(batches), input_channels,
                                  output_channels, plane, relu, swish, gelu));
        if (ok && sigmoid) ok = arena.UnaryInplace(result.slot, VulkanUnaryOp::sigmoid);
        if (ok && hard_swish) ok = arena.UnaryInplace(result.slot, VulkanUnaryOp::hard_swish);
      } else {
        ok = result.slot.resident && arena.Conv2d(input->slot, *weights, *bias, result.slot,
            batches, input_channels, output_channels, height, width, out_h, out_w, kernel_h, kernel_w,
            int(strides[0]), int(strides[1]), int(pads[0]), int(pads[1]), relu, swish, sigmoid, hard_swish);
        if (ok && node.op == "FusedConvGelu") {
          ok = arena.UnaryInplace(result.slot, VulkanUnaryOp::gelu);
        }
      }
      if (ok && hard_sigmoid) {
        ok = arena.UnaryInplace(result.slot, VulkanUnaryOp::hard_sigmoid,
            Attr(node, "__ppocr_fused_alpha").f, Attr(node, "__ppocr_fused_beta").f);
      }
      if (synthetic_bias.resident) arena.Release(synthetic_bias);
      if (ok) values.emplace(node.out[0], result); else arena.Release(result.slot);
    } else if ((node.op == "FusedDepthwisePointwiseConv" ||
                node.op == "FusedDepthwisePointwiseConvGelu" ||
                node.op == "FusedDepthwisePointwiseConvHardSwish" ||
                node.op == "FusedDepthwisePointwiseConvRelu") &&
               node.in.size() == 5 && IsNchw(*input)) {
      const auto* depthwise_weights = get_constant(node.in[1]);
      const auto* depthwise_bias = get_constant(node.in[2]);
      const auto* pointwise_weights = get_constant(node.in[3]);
      const auto* pointwise_bias = get_constant(node.in[4]);
      const auto& dw_meta = impl.graph.initializers.at(node.in[1]);
      const auto& pw_meta = impl.graph.initializers.at(node.in[3]);
      const auto strides = AttrInts(node, "strides", {1, 1});
      const auto pads = AttrInts(node, "pads", {0, 0, 0, 0});
      const int batches = int(input->shape[0]), channels = int(input->shape[1]);
      const int input_height = int(input->shape[2]), input_width = int(input->shape[3]);
      const int kernel_height = dw_meta.shape.size() == 4 ? int(dw_meta.shape[2]) : 0;
      const int kernel_width = dw_meta.shape.size() == 4 ? int(dw_meta.shape[3]) : 0;
      const int output_channels = pw_meta.shape.size() == 4 ? int(pw_meta.shape[0]) : 0;
      const int output_height = (input_height + (pads.size() == 4 ? int(pads[0] + pads[2]) : 0) -
          kernel_height) / (strides.size() == 2 ? int(strides[0]) : 1) + 1;
      const int output_width = (input_width + (pads.size() == 4 ? int(pads[1] + pads[3]) : 0) -
          kernel_width) / (strides.size() == 2 ? int(strides[1]) : 1) + 1;
      if (std::getenv("PPOCR_GPU_ONLY_TRACE") != nullptr) {
        std::cerr << "GPU-only depthwise-pointwise shape N=" << batches
                  << " Cin=" << channels << " Cout=" << output_channels
                  << " input=" << input_height << 'x' << input_width
                  << " output=" << output_height << 'x' << output_width
                  << " kernel=" << kernel_height << 'x' << kernel_width << '\n';
      }
      if (!depthwise_weights || !depthwise_bias || !pointwise_weights || !pointwise_bias ||
          dw_meta.shape.size() != 4 || pw_meta.shape.size() != 4 || dw_meta.shape[0] != channels ||
          dw_meta.shape[1] != 1 || pw_meta.shape[1] != channels || pw_meta.shape[2] != 1 ||
          pw_meta.shape[3] != 1 || strides.size() != 2 || pads.size() != 4 ||
          strides[0] <= 0 || strides[1] <= 0 || output_height <= 0 || output_width <= 0) return fail();
      result.shape = {input->shape[0], output_channels, output_height, output_width};
      result.slot = arena.Acquire(Elements(result.shape), node.out[0]);
      const auto plane = std::size_t(output_height) * output_width;
      // Rec CNN 3x3 s1 maps are plane<=1024 (12x76-class). One dispatch
      // stages the depthwise row in LDS then the 1x1. Same-host e2e missed
      // versus two-pass DW+PW, so keep ENABLE-only.
      // `PPOCR_ENABLE_VULKAN_FUSED_DWPW` turns it on.
      static const bool fused_dwpw =
          std::getenv("PPOCR_ENABLE_VULKAN_FUSED_DWPW") != nullptr &&
          std::getenv("PPOCR_DISABLE_VULKAN_FUSED_DWPW") == nullptr;
      const bool rec_dwpw = fused_dwpw && kernel_height == 3 && kernel_width == 3 &&
          strides[0] == 1 && strides[1] == 1 && pads.size() == 4 &&
          pads[0] == 1 && pads[1] == 1 && pads[2] == 1 && pads[3] == 1 &&
          output_height == input_height && output_width == input_width &&
          plane <= 1024u && channels <= 256 && output_channels <= 256;
      const bool want_gelu = node.op == "FusedDepthwisePointwiseConvGelu";
      bool used_fused_dwpw = false;
      if (rec_dwpw) {
        const auto* packed_bias = pack_expand_project_bias(
            node.name + "/__dwpw_fused_bias", channels, output_channels,
            node.in[2], node.in[4]);
        if (packed_bias && result.slot.resident &&
            arena.DepthwisePointwiseFused(input->slot, *depthwise_weights, *packed_bias,
                                          *pointwise_weights, result.slot, batches, channels,
                                          output_channels, output_height, output_width,
                                          want_gelu)) {
          ok = true;
          used_fused_dwpw = true;
        }
      }
      if (!used_fused_dwpw) {
      auto intermediate = arena.Acquire(std::size_t(batches) * channels * output_height * output_width,
                                        "gpu-depthwise-intermediate");
      // Exact fused GELU is available for devices where eliminating the
      // extra global-memory pass wins.  On the local Radeon/LLPC stack a
      // blanket fuse across det-sized planes was slower, so that stays
      // ENABLE-only. Rec CNN maps are plane<=2048 (12x76-class); folding
      // GELU into the pointwise write drops a whole dispatch there.
      // `PPOCR_DISABLE_GPU_FUSE_REC_GELU` restores the extra unary.
      static const bool fuse_all_gelu =
          std::getenv("PPOCR_GPU_FUSE_POINTWISE_GELU") != nullptr;
      static const bool rec_gelu =
          std::getenv("PPOCR_DISABLE_GPU_FUSE_REC_GELU") == nullptr;
      const bool rec_plane = plane <= 2048u;
      const bool fuse_pointwise_gelu = want_gelu &&
          (fuse_all_gelu || (rec_gelu && rec_plane));
      ok = intermediate.resident && result.slot.resident &&
          arena.DepthwiseConv(input->slot, *depthwise_weights, *depthwise_bias, intermediate,
                              batches, channels, input_height, input_width, output_height, output_width,
                              kernel_height, kernel_width, int(strides[0]), int(strides[1]),
                              int(pads[0]), int(pads[1]));
      ok = ok && arena.PointwiseConv(intermediate, *pointwise_weights, *pointwise_bias, result.slot,
                                     batches, channels, output_channels, plane, false, false,
                                     fuse_pointwise_gelu);
      arena.Release(intermediate);
      if (ok && want_gelu && !fuse_pointwise_gelu)
        ok = arena.UnaryInplace(result.slot, VulkanUnaryOp::gelu);
      }
      if (ok && node.op == "FusedDepthwisePointwiseConvHardSwish")
        ok = arena.UnaryInplace(result.slot, VulkanUnaryOp::hard_swish);
      if (ok && node.op == "FusedDepthwisePointwiseConvRelu")
        ok = arena.UnaryInplace(result.slot, VulkanUnaryOp::relu);
      if (ok) values.emplace(node.out[0], result); else arena.Release(result.slot);
    } else if ((node.op == "FusedPointwiseDepthwiseConv" ||
                node.op == "FusedPointwiseDepthwiseConvRelu") &&
               node.in.size() == 5 && IsNchw(*input)) {
      const auto* pointwise_weights = get_constant(node.in[1]);
      const auto* pointwise_bias = get_constant(node.in[2]);
      const auto* depthwise_weights = get_constant(node.in[3]);
      const auto* depthwise_bias = get_constant(node.in[4]);
      const auto& pw_meta = impl.graph.initializers.at(node.in[1]);
      const auto& dw_meta = impl.graph.initializers.at(node.in[3]);
      const int batches = int(input->shape[0]), input_channels = int(input->shape[1]);
      const int height = int(input->shape[2]), width = int(input->shape[3]);
      const int output_channels = pw_meta.shape.size() == 4 ? int(pw_meta.shape[0]) : 0;
      if (!pointwise_weights || !pointwise_bias || !depthwise_weights || !depthwise_bias ||
          pw_meta.shape.size() != 4 || dw_meta.shape.size() != 4 ||
          pw_meta.shape[1] != input_channels || pw_meta.shape[2] != 1 ||
          pw_meta.shape[3] != 1 || dw_meta.shape[0] != output_channels ||
          dw_meta.shape[1] != 1 || dw_meta.shape[2] != 3 || dw_meta.shape[3] != 3 ||
          output_channels <= 0) return fail();
      result.shape = {input->shape[0], output_channels, height, width};
      result.slot = arena.Acquire(Elements(result.shape), node.out[0]);
      auto intermediate = arena.Acquire(Elements(result.shape), "gpu-pwdw-intermediate");
      const auto plane = std::size_t(height) * width;
      const bool relu = node.op == "FusedPointwiseDepthwiseConvRelu";
      ok = intermediate.resident && result.slot.resident &&
          arena.PointwiseConv(input->slot, *pointwise_weights, *pointwise_bias, intermediate,
                              batches, input_channels, output_channels, plane, relu) &&
          arena.DepthwiseConv(intermediate, *depthwise_weights, *depthwise_bias, result.slot,
                              batches, output_channels, height, width, height, width, 3, 3,
                              1, 1, 1, 1);
      arena.Release(intermediate);
      if (ok) values.emplace(node.out[0], result); else arena.Release(result.slot);
    } else if ((node.op == "FusedPointwiseConvAdd" || node.op == "FusedPointwiseConvAddRelu" ||
                node.op == "FusedPointwiseConvAddSwish") && node.in.size() == 4 && IsNchw(*input)) {
      const auto* weights = get_constant(node.in[1]);
      const auto* bias = get_constant(node.in[2]);
      const auto* residual = fetch(node.in[3]);
      const auto& meta = impl.graph.initializers.at(node.in[1]);
      const int batches = int(input->shape[0]), channels = int(input->shape[1]);
      const int output_channels = meta.shape.size() == 4 ? int(meta.shape[0]) : 0;
      if (!weights || !bias || !residual || meta.shape.size() != 4 || meta.shape[1] != channels ||
          meta.shape[2] != 1 || meta.shape[3] != 1 ||
          residual->shape != std::vector<std::int64_t>{input->shape[0], output_channels, input->shape[2], input->shape[3]}) return fail();
      result.shape = residual->shape;
      result.slot = arena.Acquire(Elements(result.shape), node.out[0]);
      // The final medium-detector residual projection is an LLPC-sensitive
      // boundary.  Keep the projection and add as two Vulkan-only dispatches
      // there: this avoids the mode-5 shader's simultaneous full-residual
      // storage read while preserving FP32 graph semantics and device-only
      // activation lifetime. The fused path remains the default everywhere
      // else, where it measurably lowers memory traffic.
      const bool isolate_medium_tail_add =
          batches == 1 && channels == 1792 && output_channels == 896 &&
          ((input->shape[2] == 5 && input->shape[3] == 22) ||
           (input->shape[2] == 1 && input->shape[3] == 30));
      ok = result.slot.resident && (isolate_medium_tail_add
          ? (arena.PointwiseConvTail(input->slot, *weights, *bias, result.slot,
                                 batches, channels, output_channels,
                                 std::size_t(input->shape[2]) * input->shape[3]) &&
             // The projection writes a result buffer subsequently read by
             // the residual Add.  Fence this one exceptional tail boundary
             // before rebinding descriptors for the Add.  This is still
             // wholly device-resident: no activation is mapped, downloaded,
             // or evaluated by the CPU.
             arena.SubmitGraphSegment() &&
             arena.BinaryInplace(result.slot, residual->slot, kernels::BinaryOp::add))
          : arena.PointwiseConvAdd(input->slot, *weights, *bias, residual->slot,
              result.slot, batches, channels, output_channels,
              std::size_t(input->shape[2]) * input->shape[3],
              node.op == "FusedPointwiseConvAddRelu", node.op == "FusedPointwiseConvAddSwish"));
      if (ok) values.emplace(node.out[0], result); else arena.Release(result.slot);
    } else if ((node.op == "GlobalAveragePool" || node.op == "ReduceMean") && IsNchw(*input)) {
      bool spatial = node.op == "GlobalAveragePool";
      if (!spatial) {
        auto axes = AttrInts(node, "axes", {});
        if (axes.empty() && node.in.size() > 1) {
          const auto it = impl.graph.initializers.find(node.in[1]);
          if (it != impl.graph.initializers.end()) for (const float axis : it->second.data) axes.push_back(std::int64_t(axis));
        }
        spatial = axes.size() == 2 && ((Axis(axes[0], 4) == 2 && Axis(axes[1], 4) == 3) ||
            (Axis(axes[0], 4) == 3 && Axis(axes[1], 4) == 2));
      }
      if (!spatial) return fail();
      result.shape = {input->shape[0], input->shape[1], 1, 1};
      result.slot = arena.Acquire(Elements(result.shape), node.out[0]);
      ok = result.slot.resident && arena.SpatialMean(input->slot, result.slot, std::size_t(input->shape[0]),
          int(input->shape[1]), std::size_t(input->shape[2]) * input->shape[3]);
      if (ok) values.emplace(node.out[0], result); else arena.Release(result.slot);
    } else if (node.op == "Resize" && IsNchw(*input)) {
      const Tensor* scales = nullptr;
      const Tensor* sizes = nullptr;
      if (node.in.size() > 2) { const auto it = impl.graph.initializers.find(node.in[2]); if (it != impl.graph.initializers.end()) scales = &it->second; }
      if (node.in.size() > 3) { const auto it = impl.graph.initializers.find(node.in[3]); if (it != impl.graph.initializers.end()) sizes = &it->second; }
      int scale_h{}, scale_w{};
      if (sizes && sizes->data.size() >= 4) {
        if (int(sizes->data[2]) % int(input->shape[2]) || int(sizes->data[3]) % int(input->shape[3])) return fail();
        scale_h = int(sizes->data[2]) / int(input->shape[2]); scale_w = int(sizes->data[3]) / int(input->shape[3]);
      } else if (scales && scales->data.size() >= 4) {
        scale_h = int(std::lround(scales->data[2])); scale_w = int(std::lround(scales->data[3]));
        if (std::abs(scales->data[2] - scale_h) > 1e-6F || std::abs(scales->data[3] - scale_w) > 1e-6F) return fail();
      } else return fail();
      if (scale_h <= 0 || scale_w <= 0 || AttrStr(node, "mode", "nearest") != "nearest" ||
          AttrStr(node, "coordinate_transformation_mode", "half_pixel") != "asymmetric" ||
          AttrStr(node, "nearest_mode", "round_prefer_floor") != "floor") return fail();
      result.shape = {input->shape[0], input->shape[1], input->shape[2] * scale_h, input->shape[3] * scale_w};
      result.slot = arena.Acquire(Elements(result.shape), node.out[0]);
      ok = result.slot.resident && arena.NearestResize(input->slot, result.slot, std::size_t(input->shape[0]),
          int(input->shape[1]), int(input->shape[2]), int(input->shape[3]), scale_h, scale_w);
      if (ok) values.emplace(node.out[0], result); else arena.Release(result.slot);
    } else if (node.op == "FusedNearestResizeAdd" && node.in.size() >= 2 && IsNchw(*input)) {
      const auto* residual = fetch(node.in.back());
      if (!residual || residual->shape.size() != 4 || residual->shape[0] != input->shape[0] ||
          residual->shape[1] != input->shape[1] || residual->shape[2] != input->shape[2] * 2 ||
          residual->shape[3] != input->shape[3] * 2) return fail();
      result.shape = residual->shape;
      result.slot = arena.Acquire(Elements(result.shape), node.out[0]);
      if (std::getenv("PPOCR_DISABLE_VULKAN_FPN_FUSION") != nullptr) {
        // This is a diagnostic-quality alternative graph, not a CPU
        // fallback: both resize and addition remain Vulkan dispatches.  It
        // lets deployment qualification distinguish the fused mode-18 kernel
        // from the surrounding FPN resource lifetime on conservative drivers.
        ok = result.slot.resident && arena.NearestResize(input->slot, result.slot,
            std::size_t(input->shape[0]), int(input->shape[1]), int(input->shape[2]),
            int(input->shape[3]), 2, 2);
        // A conservative driver can request an explicit device-only fence
        // between FPN resize and the following in-place add.  It is useful
        // both for qualification and for adapters that reject the two
        // storage-buffer alias patterns in one command segment; no tensor
        // crosses to CPU at this boundary.
        if (ok && std::getenv("PPOCR_GPU_ONLY_FPN_SPLIT") != nullptr)
          ok = arena.SubmitGraphSegment();
        if (ok) ok = arena.BinaryInplace(result.slot, residual->slot, kernels::BinaryOp::add);
      } else {
        ok = result.slot.resident && arena.NearestResize2xAdd(input->slot, residual->slot, result.slot,
            std::size_t(input->shape[0]), int(input->shape[1]), int(input->shape[2]), int(input->shape[3]));
      }
      if (ok) values.emplace(node.out[0], result); else arena.Release(result.slot);
    } else if ((node.op == "ConvTranspose" || node.op == "FusedConvTransposeRelu" ||
                node.op == "FusedConvTransposeSigmoid") &&
               node.in.size() >= 2 && IsNchw(*input)) {
      const auto* weights = get_constant(node.in[1]);
      const auto& meta = impl.graph.initializers.at(node.in[1]);
      const auto strides = AttrInts(node, "strides", {1, 1}); const auto pads = AttrInts(node, "pads", {0, 0, 0, 0});
      const auto output_padding = AttrInts(node, "output_padding", {0, 0});
      if (!weights || meta.shape.size() != 4 || meta.shape[0] != input->shape[1] || meta.shape[2] != 2 ||
          meta.shape[3] != 2 || AttrInt(node, "group", 1) != 1 || strides != std::vector<std::int64_t>{2, 2} ||
          pads != std::vector<std::int64_t>{0, 0, 0, 0} || output_padding != std::vector<std::int64_t>{0, 0}) return fail();
      VulkanTensorSlot synthetic_bias; const VulkanTensorSlot* bias = nullptr;
      if (node.in.size() > 2 && !node.in[2].empty()) bias = get_constant(node.in[2]);
      if (!bias) { std::vector<float> zeros(static_cast<std::size_t>(meta.shape[1]), 0.F); synthetic_bias = arena.Acquire(zeros.size(), "gpu-zero-deconv-bias"); if (!synthetic_bias.resident || !arena.Upload(synthetic_bias, zeros.data(), zeros.size())) return fail(); bias = &synthetic_bias; }
      result.shape = {input->shape[0], meta.shape[1], input->shape[2] * 2, input->shape[3] * 2};
      result.slot = arena.Acquire(Elements(result.shape), node.out[0]);
      const int transpose_act = node.op == "FusedConvTransposeRelu" ? 1
          : (node.op == "FusedConvTransposeSigmoid" ? 2 : 0);
      ok = result.slot.resident && arena.ConvTranspose2x2(input->slot, *weights, *bias, result.slot,
          std::size_t(input->shape[0]), int(input->shape[1]), int(meta.shape[1]), int(input->shape[2]), int(input->shape[3]),
          transpose_act);
      static const bool fuse_transpose_act =
          std::getenv("PPOCR_DISABLE_GPU_FUSE_CONVTRANSPOSE_ACT") == nullptr;
      if (ok && !fuse_transpose_act && node.op == "FusedConvTransposeRelu")
        ok = arena.UnaryInplace(result.slot, VulkanUnaryOp::relu);
      if (ok && !fuse_transpose_act && node.op == "FusedConvTransposeSigmoid")
        ok = arena.UnaryInplace(result.slot, VulkanUnaryOp::sigmoid);
      if (synthetic_bias.resident) arena.Release(synthetic_bias);
      if (ok) values.emplace(node.out[0], result); else arena.Release(result.slot);
    } else if (node.op == "FusedConvTransposeChain" && node.in.size() >= 4 &&
               IsNchw(*input)) {
      const auto* w0 = get_constant(node.in[1]);
      const auto* b0 = get_constant(node.in[2]);
      const auto* packed = get_constant(node.in[3]);
      if (!w0 || !b0 || !packed || !impl.graph.initializers.contains(node.in[1]) ||
          !impl.graph.initializers.contains(node.in[3])) return fail();
      const auto& w0_meta = impl.graph.initializers.at(node.in[1]);
      const auto& packed_meta = impl.graph.initializers.at(node.in[3]);
      if (w0_meta.shape.size() != 4 || w0_meta.shape[0] != input->shape[1] ||
          w0_meta.shape[2] != 2 || w0_meta.shape[3] != 2) return fail();
      const int cin = int(input->shape[1]);
      const int mid = int(w0_meta.shape[1]);
      if (mid <= 0 || packed_meta.data.size() < std::size_t(mid) * 4 + 1) return fail();
      const int cout = int(packed_meta.data.size() / (std::size_t(mid) * 4 + 1));
      if (cout <= 0 ||
          packed_meta.data.size() != std::size_t(cout) * (std::size_t(mid) * 4 + 1) ||
          b0->live_elements < std::size_t(mid)) return fail();
      result.shape = {input->shape[0], cout, input->shape[2] * 4, input->shape[3] * 4};
      result.slot = arena.Acquire(Elements(result.shape), node.out[0]);
      ok = result.slot.resident &&
          arena.ConvTranspose2x2Chain(input->slot, *w0, *b0, *packed, result.slot,
                                      std::size_t(input->shape[0]), cin, mid, cout,
                                      int(input->shape[2]), int(input->shape[3]));
      if (ok) values.emplace(node.out[0], result); else arena.Release(result.slot);
    } else if ((node.op == "MaxPool" || node.op == "AveragePool") && IsNchw(*input)) {
      const auto kernel = AttrInts(node, "kernel_shape", {}); const auto strides = AttrInts(node, "strides", {1, 1});
      auto pads = AttrInts(node, "pads", {0, 0, 0, 0});
      const auto auto_pad = AttrStr(node, "auto_pad", "NOTSET");
      if (kernel.size() != 2 || strides.size() != 2 || pads.size() != 4 ||
          strides[0] <= 0 || strides[1] <= 0 || (auto_pad != "NOTSET" && auto_pad != "SAME_UPPER" &&
          auto_pad != "SAME_LOWER") || AttrInt(node, "ceil_mode", 0) != 0) return fail();
      const int h = int(input->shape[2]), w = int(input->shape[3]);
      int out_h{}, out_w{};
      if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
        out_h = (h + int(strides[0]) - 1) / int(strides[0]);
        out_w = (w + int(strides[1]) - 1) / int(strides[1]);
        const int total_h = std::max(0, (out_h - 1) * int(strides[0]) + int(kernel[0]) - h);
        const int total_w = std::max(0, (out_w - 1) * int(strides[1]) + int(kernel[1]) - w);
        pads[0] = auto_pad == "SAME_LOWER" ? (total_h + 1) / 2 : total_h / 2;
        pads[2] = total_h - pads[0];
        pads[1] = auto_pad == "SAME_LOWER" ? (total_w + 1) / 2 : total_w / 2;
        pads[3] = total_w - pads[1];
      } else {
        out_h = (h + int(pads[0] + pads[2]) - int(kernel[0])) / int(strides[0]) + 1;
        out_w = (w + int(pads[1] + pads[3]) - int(kernel[1])) / int(strides[1]) + 1;
      }
      if (out_h <= 0 || out_w <= 0) return fail();
      result.shape = {input->shape[0], input->shape[1], out_h, out_w};
      result.slot = arena.Acquire(Elements(result.shape), node.out[0]);
      ok = result.slot.resident && arena.Pool2d(input->slot, result.slot, std::size_t(input->shape[0]), int(input->shape[1]),
          h, w, out_h, out_w, int(kernel[0]), int(kernel[1]), int(strides[0]), int(strides[1]),
          int(pads[0]), int(pads[1]), node.op == "AveragePool");
      if (ok) values.emplace(node.out[0], result); else arena.Release(result.slot);
    } else if ((node.op == "FusedMaxPoolConcatConv" || node.op == "FusedMaxPoolConcatConvRelu") &&
               node.in.size() >= 3 && IsNchw(*input)) {
      // Stem MaxPool(2x2 SAME_UPPER) || peers -> Conv 3x3. Keep every
      // intermediate in the arena so hybrid/gpu-only do not bounce the 32-ch
      // concat activation through the host.
      const bool has_bias = !node.in.back().empty() &&
          impl.graph.initializers.contains(node.in.back()) &&
          impl.graph.initializers.at(node.in.back()).shape.size() == 1;
      const std::size_t source_count = node.in.size() - (has_bias ? 2 : 1);
      const auto weights_name = node.in[source_count];
      const auto* weights = get_constant(weights_name);
      if (!weights || source_count < 2 || !impl.graph.initializers.contains(weights_name)) return fail();
      const auto& weight_meta = impl.graph.initializers.at(weights_name);
      if (weight_meta.shape.size() != 4 || weight_meta.shape[2] != 3 ||
          weight_meta.shape[3] != 3) return fail();
      const int batches = int(input->shape[0]);
      const int height = int(input->shape[2]), width = int(input->shape[3]);
      const int pool_channels = int(input->shape[1]);
      const int output_channels = int(weight_meta.shape[0]);
      const int kernel_h = int(weight_meta.shape[2]), kernel_w = int(weight_meta.shape[3]);
      auto conv_pads = AttrInts(node, "pads", {0, 0, 0, 0});
      const auto conv_strides = AttrInts(node, "strides", {1, 1});
      const auto auto_pad = AttrStr(node, "auto_pad", "NOTSET");
      const auto pool_auto = node.attr.contains("__ppocr_pool_auto_pad")
          ? node.attr.at("__ppocr_pool_auto_pad").s : std::string("SAME_UPPER");
      if (batches <= 0 || height <= 0 || width <= 0 || conv_strides.size() != 2 ||
          conv_pads.size() != 4 || pool_auto != "SAME_UPPER") return fail();
      auto pooled = arena.Acquire(std::size_t(batches) * pool_channels * height * width,
                                  "gpu-maxpool-concat");
      ok = pooled.resident && arena.Pool2d(input->slot, pooled, std::size_t(batches),
          pool_channels, height, width, height, width, 2, 2, 1, 1, 0, 0, false);
      std::vector<VulkanTensorSlot> concat_slots;
      std::vector<std::size_t> axis_lengths;
      concat_slots.push_back(pooled);
      axis_lengths.push_back(std::size_t(pool_channels));
      int total_channels = pool_channels;
      for (std::size_t i = 1; i < source_count && ok; ++i) {
        const auto* peer = fetch(node.in[i]);
        if (!peer || !IsNchw(*peer) || peer->shape[0] != batches ||
            peer->shape[2] != height || peer->shape[3] != width) {
          ok = false;
          break;
        }
        concat_slots.push_back(peer->slot);
        axis_lengths.push_back(std::size_t(peer->shape[1]));
        total_channels += int(peer->shape[1]);
      }
      if (!ok || total_channels != int(weight_meta.shape[1])) {
        arena.Release(pooled);
        return fail();
      }
      auto concat = arena.Acquire(std::size_t(batches) * total_channels * height * width,
                                  "gpu-stem-concat");
      ok = concat.resident && arena.Concat(concat_slots, concat, std::size_t(batches),
          std::size_t(height) * width, axis_lengths);
      arena.Release(pooled);
      int out_h = 0, out_w = 0;
      if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
        out_h = (height + int(conv_strides[0]) - 1) / int(conv_strides[0]);
        out_w = (width + int(conv_strides[1]) - 1) / int(conv_strides[1]);
        const int total_h = std::max(0, (out_h - 1) * int(conv_strides[0]) + kernel_h - height);
        const int total_w = std::max(0, (out_w - 1) * int(conv_strides[1]) + kernel_w - width);
        conv_pads[0] = auto_pad == "SAME_LOWER" ? (total_h + 1) / 2 : total_h / 2;
        conv_pads[2] = total_h - conv_pads[0];
        conv_pads[1] = auto_pad == "SAME_LOWER" ? (total_w + 1) / 2 : total_w / 2;
        conv_pads[3] = total_w - conv_pads[1];
      } else {
        out_h = (height + int(conv_pads[0] + conv_pads[2]) - kernel_h) / int(conv_strides[0]) + 1;
        out_w = (width + int(conv_pads[1] + conv_pads[3]) - kernel_w) / int(conv_strides[1]) + 1;
      }
      VulkanTensorSlot synthetic_bias;
      const VulkanTensorSlot* bias = has_bias ? get_constant(node.in.back()) : nullptr;
      if (!bias) {
        std::vector<float> zeros(static_cast<std::size_t>(output_channels));
        synthetic_bias = arena.Acquire(zeros.size(), "gpu-zero-stem-bias");
        if (!synthetic_bias.resident ||
            !arena.Upload(synthetic_bias, zeros.data(), zeros.size())) {
          arena.Release(concat);
          return fail();
        }
        bias = &synthetic_bias;
      }
      result.shape = {batches, output_channels, out_h, out_w};
      result.slot = arena.Acquire(Elements(result.shape), node.out[0]);
      ok = ok && result.slot.resident && out_h > 0 && out_w > 0 &&
          arena.Conv2d(concat, *weights, *bias, result.slot, batches, total_channels,
                       output_channels, height, width, out_h, out_w, kernel_h, kernel_w,
                       int(conv_strides[0]), int(conv_strides[1]), int(conv_pads[0]),
                       int(conv_pads[1]), node.op == "FusedMaxPoolConcatConvRelu");
      arena.Release(concat);
      if (synthetic_bias.resident) arena.Release(synthetic_bias);
      if (ok) values.emplace(node.out[0], result); else arena.Release(result.slot);
    } else if ((node.op == "FusedConcatConv" || node.op == "FusedConcatConvRelu") &&
               node.in.size() >= 3 && IsNchw(*input)) {
      const bool has_bias = !node.in.back().empty() &&
          impl.graph.initializers.contains(node.in.back()) &&
          impl.graph.initializers.at(node.in.back()).shape.size() == 1;
      const std::size_t source_count = node.in.size() - (has_bias ? 2 : 1);
      const auto weights_name = node.in[source_count];
      const auto* weights = get_constant(weights_name);
      if (!weights || source_count < 2 || !impl.graph.initializers.contains(weights_name)) return fail();
      const auto& weight_meta = impl.graph.initializers.at(weights_name);
      if (weight_meta.shape.size() != 4) return fail();
      const int batches = int(input->shape[0]);
      const int height = int(input->shape[2]), width = int(input->shape[3]);
      const int output_channels = int(weight_meta.shape[0]);
      const int kernel_h = int(weight_meta.shape[2]), kernel_w = int(weight_meta.shape[3]);
      auto conv_pads = AttrInts(node, "pads", {0, 0, 0, 0});
      const auto conv_strides = AttrInts(node, "strides", {1, 1});
      const auto auto_pad = AttrStr(node, "auto_pad", "NOTSET");
      std::vector<VulkanTensorSlot> concat_slots;
      std::vector<std::size_t> axis_lengths;
      int total_channels = 0;
      for (std::size_t i = 0; i < source_count; ++i) {
        const auto* peer = fetch(node.in[i]);
        if (!peer || !IsNchw(*peer) || peer->shape[0] != batches ||
            peer->shape[2] != height || peer->shape[3] != width) return fail();
        concat_slots.push_back(peer->slot);
        axis_lengths.push_back(std::size_t(peer->shape[1]));
        total_channels += int(peer->shape[1]);
      }
      if (total_channels != int(weight_meta.shape[1]) || conv_strides.size() != 2 ||
          conv_pads.size() != 4) return fail();
      auto concat = arena.Acquire(std::size_t(batches) * total_channels * height * width,
                                  "gpu-fpn-concat");
      ok = concat.resident && arena.Concat(concat_slots, concat, std::size_t(batches),
          std::size_t(height) * width, axis_lengths);
      int out_h = 0, out_w = 0;
      if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
        out_h = (height + int(conv_strides[0]) - 1) / int(conv_strides[0]);
        out_w = (width + int(conv_strides[1]) - 1) / int(conv_strides[1]);
        const int total_h = std::max(0, (out_h - 1) * int(conv_strides[0]) + kernel_h - height);
        const int total_w = std::max(0, (out_w - 1) * int(conv_strides[1]) + kernel_w - width);
        conv_pads[0] = auto_pad == "SAME_LOWER" ? (total_h + 1) / 2 : total_h / 2;
        conv_pads[2] = total_h - conv_pads[0];
        conv_pads[1] = auto_pad == "SAME_LOWER" ? (total_w + 1) / 2 : total_w / 2;
        conv_pads[3] = total_w - conv_pads[1];
      } else {
        out_h = (height + int(conv_pads[0] + conv_pads[2]) - kernel_h) / int(conv_strides[0]) + 1;
        out_w = (width + int(conv_pads[1] + conv_pads[3]) - kernel_w) / int(conv_strides[1]) + 1;
      }
      VulkanTensorSlot synthetic_bias;
      const VulkanTensorSlot* bias = has_bias ? get_constant(node.in.back()) : nullptr;
      if (!bias) {
        std::vector<float> zeros(static_cast<std::size_t>(output_channels));
        synthetic_bias = arena.Acquire(zeros.size(), "gpu-zero-concat-conv-bias");
        if (!synthetic_bias.resident ||
            !arena.Upload(synthetic_bias, zeros.data(), zeros.size())) {
          arena.Release(concat);
          return fail();
        }
        bias = &synthetic_bias;
      }
      result.shape = {batches, output_channels, out_h, out_w};
      result.slot = arena.Acquire(Elements(result.shape), node.out[0]);
      ok = ok && result.slot.resident && out_h > 0 && out_w > 0 &&
          arena.Conv2d(concat, *weights, *bias, result.slot, batches, total_channels,
                       output_channels, height, width, out_h, out_w, kernel_h, kernel_w,
                       int(conv_strides[0]), int(conv_strides[1]), int(conv_pads[0]),
                       int(conv_pads[1]), node.op == "FusedConcatConvRelu");
      arena.Release(concat);
      if (synthetic_bias.resident) arena.Release(synthetic_bias);
      if (ok) values.emplace(node.out[0], result); else arena.Release(result.slot);
    } else if (node.op == "FusedConcatConvDualTranspose" && node.in.size() >= 8 &&
               IsNchw(*input)) {
      const auto* ct1_b = get_constant(node.in[node.in.size() - 1]);
      const auto* ct1_w = get_constant(node.in[node.in.size() - 2]);
      const auto* ct0_b = get_constant(node.in[node.in.size() - 3]);
      const auto* ct0_w = get_constant(node.in[node.in.size() - 4]);
      const auto* conv_b = get_constant(node.in[node.in.size() - 5]);
      const auto conv_w_name = node.in[node.in.size() - 6];
      const auto* conv_weights = get_constant(conv_w_name);
      if (!ct1_b || !ct1_w || !ct0_b || !ct0_w || !conv_b || !conv_weights ||
          !impl.graph.initializers.contains(conv_w_name) ||
          !impl.graph.initializers.contains(node.in[node.in.size() - 4]) ||
          !impl.graph.initializers.contains(node.in[node.in.size() - 2])) return fail();
      const auto& conv_meta = impl.graph.initializers.at(conv_w_name);
      const auto& ct0_meta = impl.graph.initializers.at(node.in[node.in.size() - 4]);
      const auto& ct1_meta = impl.graph.initializers.at(node.in[node.in.size() - 2]);
      const std::size_t source_count = node.in.size() - 6;
      const int batches = int(input->shape[0]);
      const int height = int(input->shape[2]), width = int(input->shape[3]);
      if (source_count < 2 || conv_meta.shape.size() != 4 || ct0_meta.shape.size() != 4 ||
          ct1_meta.shape.size() != 4 || ct0_meta.shape[2] != 2 || ct0_meta.shape[3] != 2 ||
          ct1_meta.shape[2] != 2 || ct1_meta.shape[3] != 2) return fail();
      std::vector<VulkanTensorSlot> concat_slots;
      std::vector<std::size_t> axis_lengths;
      int total_channels = 0;
      for (std::size_t i = 0; i < source_count; ++i) {
        const auto* peer = fetch(node.in[i]);
        if (!peer || !IsNchw(*peer) || peer->shape[0] != batches ||
            peer->shape[2] != height || peer->shape[3] != width) return fail();
        concat_slots.push_back(peer->slot);
        axis_lengths.push_back(std::size_t(peer->shape[1]));
        total_channels += int(peer->shape[1]);
      }
      auto conv_pads = AttrInts(node, "pads", {0, 0, 0, 0});
      const auto conv_strides = AttrInts(node, "strides", {1, 1});
      if (total_channels != int(conv_meta.shape[1]) || conv_strides.size() != 2 ||
          conv_pads.size() != 4) return fail();
      const int kernel_h = int(conv_meta.shape[2]), kernel_w = int(conv_meta.shape[3]);
      const int conv_oc = int(conv_meta.shape[0]);
      const int ct0_oc = int(ct0_meta.shape[1]);
      const int ct1_oc = int(ct1_meta.shape[1]);
      const int conv_h =
          (height + int(conv_pads[0] + conv_pads[2]) - kernel_h) / int(conv_strides[0]) + 1;
      const int conv_width =
          (width + int(conv_pads[1] + conv_pads[3]) - kernel_w) / int(conv_strides[1]) + 1;
      if (conv_h <= 0 || conv_width <= 0 || conv_oc != int(ct0_meta.shape[0]) ||
          ct0_oc != int(ct1_meta.shape[0])) return fail();
      auto concat = arena.Acquire(std::size_t(batches) * total_channels * height * width,
                                  "gpu-fpn-dual-concat");
      auto conv_out = arena.Acquire(std::size_t(batches) * conv_oc * conv_h * conv_width,
                                    "gpu-fpn-dual-conv");
      auto ct0_out = arena.Acquire(std::size_t(batches) * ct0_oc * conv_h * conv_width * 4,
                                   "gpu-fpn-dual-ct0");
      result.shape = {batches, ct1_oc, conv_h * 4, conv_width * 4};
      result.slot = arena.Acquire(Elements(result.shape), node.out[0]);
      ok = concat.resident && conv_out.resident && ct0_out.resident && result.slot.resident &&
          arena.Concat(concat_slots, concat, std::size_t(batches),
                       std::size_t(height) * width, axis_lengths) &&
          arena.Conv2d(concat, *conv_weights, *conv_b, conv_out, batches, total_channels, conv_oc,
                       height, width, conv_h, conv_width, kernel_h, kernel_w,
                       int(conv_strides[0]), int(conv_strides[1]), int(conv_pads[0]),
                       int(conv_pads[1]), true) &&
          arena.ConvTranspose2x2(conv_out, *ct0_w, *ct0_b, ct0_out, std::size_t(batches),
                                 conv_oc, ct0_oc, conv_h, conv_width, 1) &&
          arena.ConvTranspose2x2(ct0_out, *ct1_w, *ct1_b, result.slot, std::size_t(batches),
                                 ct0_oc, ct1_oc, conv_h * 2, conv_width * 2, 2);
      arena.Release(concat);
      arena.Release(conv_out);
      arena.Release(ct0_out);
      if (ok) values.emplace(node.out[0], result); else arena.Release(result.slot);
    } else if (node.op == "FusedDepthwiseExpandGeluProjectAdd" &&
               node.in.size() == 7 && IsNchw(*input)) {
      const auto* dw_w = get_constant(node.in[1]);
      const auto* dw_b = get_constant(node.in[2]);
      const auto* expand_w = get_constant(node.in[3]);
      const auto* project_w = get_constant(node.in[5]);
      if (!dw_w || !dw_b || !expand_w || !project_w ||
          !impl.graph.initializers.contains(node.in[1]) ||
          !impl.graph.initializers.contains(node.in[3]) ||
          !impl.graph.initializers.contains(node.in[5])) return fail();
      const auto& dw_meta = impl.graph.initializers.at(node.in[1]);
      const auto& expand_meta = impl.graph.initializers.at(node.in[3]);
      const auto& project_meta = impl.graph.initializers.at(node.in[5]);
      const auto strides = AttrInts(node, "strides", {1, 1});
      const auto pads = AttrInts(node, "pads", {0, 0, 0, 0});
      const int batches = int(input->shape[0]), channels = int(input->shape[1]);
      const int height = int(input->shape[2]), width = int(input->shape[3]);
      const int hidden = expand_meta.shape.size() == 4 ? int(expand_meta.shape[0]) : 0;
      const int kernel_h = dw_meta.shape.size() == 4 ? int(dw_meta.shape[2]) : 0;
      const int kernel_w = dw_meta.shape.size() == 4 ? int(dw_meta.shape[3]) : 0;
      if (hidden <= 0 || dw_meta.shape.size() != 4 || expand_meta.shape.size() != 4 ||
          project_meta.shape.size() != 4 || dw_meta.shape[0] != channels ||
          dw_meta.shape[1] != 1 || kernel_h != 3 || kernel_w != 3 ||
          expand_meta.shape[1] != channels || expand_meta.shape[2] != 1 ||
          expand_meta.shape[3] != 1 || project_meta.shape[0] != channels ||
          project_meta.shape[1] != hidden || project_meta.shape[2] != 1 ||
          project_meta.shape[3] != 1 || strides != std::vector<std::int64_t>{1, 1} ||
          pads.size() != 4) return fail();
      VulkanTensorSlot expand_bias_slot, project_bias_slot;
      const VulkanTensorSlot* expand_bias = node.in[4].empty() ? nullptr
          : get_constant(node.in[4]);
      const VulkanTensorSlot* project_bias = node.in[6].empty() ? nullptr
          : get_constant(node.in[6]);
      if (!expand_bias) {
        std::vector<float> zeros(static_cast<std::size_t>(hidden));
        expand_bias_slot = arena.Acquire(zeros.size(), "gpu-zero-dw-expand-bias");
        if (!expand_bias_slot.resident ||
            !arena.Upload(expand_bias_slot, zeros.data(), zeros.size())) return fail();
        expand_bias = &expand_bias_slot;
      }
      if (!project_bias) {
        std::vector<float> zeros(static_cast<std::size_t>(channels));
        project_bias_slot = arena.Acquire(zeros.size(), "gpu-zero-dw-project-bias");
        if (!project_bias_slot.resident ||
            !arena.Upload(project_bias_slot, zeros.data(), zeros.size())) {
          if (expand_bias_slot.resident) arena.Release(expand_bias_slot);
          return fail();
        }
        project_bias = &project_bias_slot;
      }
      auto dw_act = arena.Acquire(std::size_t(batches) * channels * height * width,
                                  "gpu-dw-expand");
      result.shape = input->shape;
      result.slot = arena.Acquire(input->slot.live_elements, node.out[0]);
      const auto plane = std::size_t(height) * width;
      // Keep the Radeon-sensitive terminal block fused with its following
      // expand/project chain, but retain the dedicated scalar depthwise
      // implementation for the 896x5x22 / 896x1x30 producer.  This changes
      // only the device kernel selected for the depthwise stage: all three
      // activation tensors remain Vulkan arena slots.
      const bool scalar_terminal_dw =
          node.in[0] == "Add.143" && channels == 896 &&
          ((height == 5 && width == 22) || (height == 1 && width == 30));
      // Rec-sized default missed same-host 8-run (48.86 vs 48.73). Keep
      // ENABLE-only like the det-sized path.
      static const bool fuse_expand_project =
          std::getenv("PPOCR_ENABLE_VULKAN_EXPAND_PROJECT") != nullptr &&
          std::getenv("PPOCR_DISABLE_VULKAN_EXPAND_PROJECT") == nullptr;
      const auto* packed_bias = fuse_expand_project
          ? pack_expand_project_bias(node.name + "/__expand_project_bias", hidden,
                                     channels, node.in[4], node.in[6])
          : nullptr;
      VulkanTensorSlot hidden_act{};
      ok = dw_act.resident && result.slot.resident &&
          (scalar_terminal_dw
              ? arena.DepthwiseConvScalar(input->slot, *dw_w, *dw_b, dw_act,
                                          batches, channels, height, width, height, width,
                                          kernel_h, kernel_w, int(strides[0]),
                                          int(strides[1]), int(pads[0]), int(pads[1]))
              : arena.DepthwiseConv(input->slot, *dw_w, *dw_b, dw_act, batches, channels,
                                    height, width, height, width, kernel_h, kernel_w,
                                    int(strides[0]), int(strides[1]), int(pads[0]),
                                    int(pads[1])));
      if (ok && packed_bias &&
          arena.ExpandGeluProjectAdd(dw_act, *expand_w, *packed_bias, *project_w,
                                     result.slot, std::size_t(batches), channels, hidden,
                                     plane)) {
      } else if (ok) {
        hidden_act = arena.Acquire(std::size_t(batches) * hidden * plane,
                                   "gpu-dw-expand-gelu");
        ok = hidden_act.resident &&
            arena.PointwiseConv(dw_act, *expand_w, *expand_bias, hidden_act,
                                std::size_t(batches), channels, hidden, plane,
                                false, false, true) &&
            arena.PointwiseConvAdd(hidden_act, *project_w, *project_bias, dw_act,
                                   result.slot, std::size_t(batches), hidden, channels,
                                   plane);
        arena.Release(hidden_act);
      }
      if (hidden_act.resident) arena.Release(hidden_act);
      arena.Release(dw_act);
      if (expand_bias_slot.resident) arena.Release(expand_bias_slot);
      if (project_bias_slot.resident) arena.Release(project_bias_slot);
      if (ok) values.emplace(node.out[0], result); else arena.Release(result.slot);
    } else if (node.op == "FusedExpandGeluProjectAdd" && node.in.size() == 5 &&
               IsNchw(*input)) {
      const auto* expand_w = get_constant(node.in[1]);
      const auto* project_w = get_constant(node.in[3]);
      if (!expand_w || !project_w || !impl.graph.initializers.contains(node.in[1]) ||
          !impl.graph.initializers.contains(node.in[3])) return fail();
      const auto& expand_meta = impl.graph.initializers.at(node.in[1]);
      const auto& project_meta = impl.graph.initializers.at(node.in[3]);
      const int batches = int(input->shape[0]), channels = int(input->shape[1]);
      const int height = int(input->shape[2]), width = int(input->shape[3]);
      const int hidden = expand_meta.shape.size() == 4 ? int(expand_meta.shape[0]) : 0;
      const auto plane = std::size_t(height) * width;
      if (hidden <= 0 || expand_meta.shape.size() != 4 || project_meta.shape.size() != 4 ||
          expand_meta.shape[1] != channels || expand_meta.shape[2] != 1 ||
          expand_meta.shape[3] != 1 || project_meta.shape[0] != channels ||
          project_meta.shape[1] != hidden || project_meta.shape[2] != 1 ||
          project_meta.shape[3] != 1) return fail();
      VulkanTensorSlot expand_bias_slot, project_bias_slot;
      const VulkanTensorSlot* expand_bias = node.in[2].empty() ? nullptr
          : get_constant(node.in[2]);
      const VulkanTensorSlot* project_bias = node.in[4].empty() ? nullptr
          : get_constant(node.in[4]);
      if (!expand_bias) {
        std::vector<float> zeros(static_cast<std::size_t>(hidden));
        expand_bias_slot = arena.Acquire(zeros.size(), "gpu-zero-expand-bias");
        if (!expand_bias_slot.resident ||
            !arena.Upload(expand_bias_slot, zeros.data(), zeros.size())) return fail();
        expand_bias = &expand_bias_slot;
      }
      if (!project_bias) {
        std::vector<float> zeros(static_cast<std::size_t>(channels));
        project_bias_slot = arena.Acquire(zeros.size(), "gpu-zero-project-bias");
        if (!project_bias_slot.resident ||
            !arena.Upload(project_bias_slot, zeros.data(), zeros.size())) {
          if (expand_bias_slot.resident) arena.Release(expand_bias_slot);
          return fail();
        }
        project_bias = &project_bias_slot;
      }
      result.shape = input->shape;
      result.slot = arena.Acquire(input->slot.live_elements, node.out[0]);
      static const bool fuse_expand_project =
          std::getenv("PPOCR_ENABLE_VULKAN_EXPAND_PROJECT") != nullptr &&
          std::getenv("PPOCR_DISABLE_VULKAN_EXPAND_PROJECT") == nullptr;
      const auto* packed_bias = fuse_expand_project
          ? pack_expand_project_bias(node.name + "/__expand_project_bias", hidden,
                                     channels, node.in[2], node.in[4])
          : nullptr;
      if (result.slot.resident && packed_bias &&
          arena.ExpandGeluProjectAdd(input->slot, *expand_w, *packed_bias, *project_w,
                                     result.slot, std::size_t(batches), channels, hidden,
                                     plane)) {
        ok = true;
      } else {
        auto hidden_act = arena.Acquire(std::size_t(batches) * hidden * plane,
                                        "gpu-expand-gelu");
        ok = hidden_act.resident && result.slot.resident &&
            arena.PointwiseConv(input->slot, *expand_w, *expand_bias, hidden_act,
                                std::size_t(batches), channels, hidden, plane,
                                false, false, true) &&
            arena.PointwiseConvAdd(hidden_act, *project_w, *project_bias, input->slot,
                                   result.slot, std::size_t(batches), hidden, channels,
                                   plane);
        arena.Release(hidden_act);
      }
      if (expand_bias_slot.resident) arena.Release(expand_bias_slot);
      if (project_bias_slot.resident) arena.Release(project_bias_slot);
      if (ok) values.emplace(node.out[0], result); else arena.Release(result.slot);
    } else { return fail(); }
    if (!ok) {
      if (std::getenv("PPOCR_GPU_ONLY_TRACE") != nullptr) {
        std::cerr << "GPU-only dispatch failed for " << node.op << '\n';
      }
      return fail();
    }
    // A non-terminal diagnostic checkpoint already ends the recording and
    // starts the continuation command buffer below.  Do not immediately
    // submit that empty continuation again through the ordinary tail-boundary
    // policy: doing so obscures the next node's actual driver behaviour.
    bool checkpoint_cut_recording{};
    if (const char* checkpoint = std::getenv("PPOCR_GPU_ONLY_CHECKPOINT"); checkpoint &&
        !node.out.empty() && node.out[0] == checkpoint) {
      const auto produced = values.find(node.out[0]);
      if (produced == values.end()) return fail();
      Tensor host{produced->second.shape, std::vector<float>(produced->second.slot.live_elements)};
      // A checkpoint is a diagnostic graph boundary. Submit its preceding
      // work before reading mapped storage, then stop this strict run.
      // Persistent recordings must be ended with the persistent helper; the
      // ephemeral EndGraphRecording path returns false while a replay CB is
      // open and used to skip the dump entirely.
      const bool ended = persistent_graph
          ? arena.EndPersistentGraphRecording()
          : arena.EndGraphRecording();
      if (!ended || !arena.Download(host.data.data(), produced->second.slot, host.data.size())) {
        graph_recording = false;
        return fail();
      }
      graph_recording = false;
      persistent_graph = false;
      DumpTensorFile(host);
      std::cerr << "GPU-only checkpoint " << checkpoint << " shape=";
      for (const auto dimension : host.shape) std::cerr << dimension << ',';
      std::size_t non_finite{};
      float minimum = std::numeric_limits<float>::infinity();
      float maximum = -std::numeric_limits<float>::infinity();
      for (const float item : host.data) {
        if (!std::isfinite(item)) { ++non_finite; continue; }
        minimum = std::min(minimum, item); maximum = std::max(maximum, item);
      }
      std::cerr << " range=" << minimum << ',' << maximum << " nonfinite=" << non_finite;
      std::cerr << " values=";
      for (std::size_t index = 0; index < std::min<std::size_t>(host.data.size(), 8); ++index)
        std::cerr << host.data[index] << ',';
      std::cerr << '\n';
      // Optional terminal checkpoint for graph bisection.  It ends after the
      // completed prefix, releases all live slots through the normal failure
      // cleanup, and intentionally reports success without recording an empty
      // follow-up command buffer. This keeps the checkpoint facility separate
      // from the normal all-device execution path.
      if (std::getenv("PPOCR_GPU_ONLY_STOP_AT_CHECKPOINT") != nullptr) {
        for (auto& [_, value] : values) arena.Release(value.slot);
        values.clear();
        return true;
      }
      // A diagnostic checkpoint intentionally cuts the command buffer. Start
      // a fresh recording session for the remaining graph.
      if (!arena.BeginGraphRecording()) return fail();
      graph_recording = true;
      checkpoint_cut_recording = true;
    }
    // Remember the submission boundary, but apply it *after* input liveness
    // has been retired below.  This preserves a fully device-resident graph
    // while allowing the next command buffer to recycle storage whose final
    // read has already completed at the fence.
    // Keep the adapter-specific tail fence as the conservative default, but
    // allow qualification to prove whether a failure belongs to the command
    // stream boundary itself.  This switch only changes Vulkan submission
    // grouping: feature maps, weights, and output storage remain on-device.
    // Qualcomm and AMD integrated Vulkan drivers both handle the terminal
    // operator more reliably when it begins a fresh graph segment. Closing
    // immediately *before* it also releases the previous descriptor set
    // lifetime. All activations remain arena buffers on the GPU.
    const bool force_tail_boundary =
        std::getenv("PPOCR_DISABLE_GPU_TAIL_SEGMENT_BOUNDARY") == nullptr;
    const bool driver_tail_boundary = force_tail_boundary && node.out.size() == 1 &&
        (node.out[0] == "p2o.pd_op.gelu.11.0" || node.out[0] == "Add.143" ||
         node.out[0] == "p2o.pd_op.depthwise_conv2d.12.0");
    const bool close_segment = !checkpoint_cut_recording && !persistent_graph &&
        ((segment_node_limit != 0 && ++recorded_nodes == segment_node_limit) || driver_tail_boundary);
    for (const auto& name : node.in) {
      if (name.empty() || impl.graph.initializers.contains(name) || !consume(name) || name == node.out[0]) continue;
      // In-place unary/binary/metadata nodes transfer the sole arena slot to
      // the output name. Erasing the old map key is essential, but releasing
      // that slot would invalidate the just-produced output.
      if (aliases_first_input && name == node.in[0]) values.erase(name);
      else release(name);
    }
    // Do not begin a fresh recording session here: BeginArenaRecording marks
    // only slots that are already free as reusable.  Delaying it until the
    // next loop iteration is the lifetime handoff that keeps peak arena
    // allocation bounded without a host copy or CPU fallback.
    if (close_segment) {
      if (std::getenv("PPOCR_GPU_ONLY_TRACE") != nullptr) {
        std::cerr << "GPU-only segment boundary after " << node.out[0] << '\n';
        std::cerr << "GPU-only arena live=" << arena.live_bytes()
                  << " capacity=" << arena.capacity_bytes() << '\n';
      }
      if (!arena.EndGraphRecording()) return fail();
      graph_recording = false;
      recorded_nodes = 0;
    }
  }
  for (const auto& name : impl.graph.outputs) {
    auto value = values.find(name);
    if (value == values.end()) {
      if (std::getenv("PPOCR_GPU_ONLY_TRACE") != nullptr) {
        std::cerr << "GPU-only missing graph output " << name << '\n';
      }
      return fail();
    }
    VulkanTensorSlot fused_ctc_indices{};
    VulkanTensorSlot fused_ctc_probs{};
    int fused_ctc_batches = 0, fused_ctc_steps = 0, fused_ctc_classes = 0;
    if (graph_recording) {
      static const bool fuse_ctc_graph =
          std::getenv("PPOCR_DISABLE_GPU_FUSE_CTC_GRAPH") == nullptr;
      if (gemm_ctc_indices.resident && gemm_ctc_probs.resident && gemm_ctc_batches > 0 &&
          gemm_ctc_steps > 0) {
        fused_ctc_indices = gemm_ctc_indices;
        fused_ctc_probs = gemm_ctc_probs;
        fused_ctc_batches = gemm_ctc_batches;
        fused_ctc_steps = gemm_ctc_steps;
        fused_ctc_classes = gemm_ctc_classes;
        arena.PinForReplay(fused_ctc_indices);
        arena.PinForReplay(fused_ctc_probs);
      } else if (fuse_ctc_graph && fuse_terminal_ctc_softmax && persistent_graph &&
          value->second.shape.size() == 3) {
        fused_ctc_batches = int(value->second.shape[0]);
        fused_ctc_steps = int(value->second.shape[1]);
        fused_ctc_classes = int(value->second.shape[2]);
        const std::size_t rows =
            std::size_t(fused_ctc_batches) * std::size_t(fused_ctc_steps);
        if (fused_ctc_batches > 0 && fused_ctc_steps > 0 && fused_ctc_classes > 0 &&
            rows > 0) {
          fused_ctc_indices = arena.Acquire(rows, "ctc-graph-idx");
          fused_ctc_probs = arena.Acquire(rows, "ctc-graph-prob");
          if (!fused_ctc_indices.resident || !fused_ctc_probs.resident ||
              !arena.CtcTop1Into(value->second.slot, fused_ctc_indices, fused_ctc_probs,
                                 fused_ctc_batches, fused_ctc_steps, fused_ctc_classes,
                                 true)) {
            if (fused_ctc_indices.resident) arena.Release(fused_ctc_indices);
            if (fused_ctc_probs.resident) arena.Release(fused_ctc_probs);
            fused_ctc_indices = {};
            fused_ctc_probs = {};
            fused_ctc_batches = fused_ctc_steps = fused_ctc_classes = 0;
          } else {
            arena.PinForReplay(fused_ctc_indices);
            arena.PinForReplay(fused_ctc_probs);
          }
        }
      }
      const bool ended = persistent_graph ? arena.EndPersistentGraphRecording()
                                          : arena.EndGraphRecording();
      if (!ended) {
        if (std::getenv("PPOCR_GPU_ONLY_TRACE") != nullptr) {
          std::cerr << "GPU-only graph submission failed before output " << name << '\n';
        }
        return false;
      }
      graph_recording = false;
    }
    if (persistent_graph && replay_key != 0 && pinned_replay_input.resident) {
      Impl::GpuReplayCache entry;
      entry.key = replay_key;
      entry.last_used = ++impl.gpu_replay_clock;
      entry.input = pinned_replay_input;
      entry.output = value->second.slot;
      entry.output_shape = value->second.shape;
      entry.output_name = name;
      arena.PinForReplay(entry.output);
      entry.ctc_indices = fused_ctc_indices;
      entry.ctc_probs = fused_ctc_probs;
      entry.ctc_batches = fused_ctc_batches;
      entry.ctc_steps = fused_ctc_steps;
      entry.ctc_classes = fused_ctc_classes;
      entry.rgb = pinned_replay_rgb;
      entry.rgb_source_width = rgb_src_w;
      entry.rgb_source_height = rgb_src_h;
      bool replaced = false;
      for (auto& cache : impl.gpu_replay_caches) {
        if (cache.key == replay_key) { cache = entry; replaced = true; break; }
      }
      if (!replaced) {
        impl.gpu_replay_caches.push_back(entry);
        ++impl.replay_graphs_recorded;
      }
    } else if (persistent_graph) {
      // A recorded graph without a valid public replay entry must not retain
      // its private activation set. This happens only on an unsupported
      // terminal shape or a failed optional CTC fusion; normal strict graph
      // execution still succeeds, but the speculative replay recording is
      // discarded before its slots can inflate a later request.
      arena.DropPersistentGraph(replay_key);
    }
    if (device_outputs) {
      // Copy the plain slot descriptor before erasing from the activation
      // map. VulkanTensorSlot has no RAII destruction, so its ownership
      // explicitly transfers to the returned GpuTensor.
      GpuTensor output{value->second.shape, value->second.slot};
      values.erase(value);
      const auto [_, inserted] = device_outputs->emplace(name, std::move(output));
      if (!inserted) {
        arena.Release(output.slot);
        return fail();
      }
      if (std::getenv("PPOCR_GPU_ONLY_TRACE") != nullptr) {
        std::cerr << "GPU-only device output transfer complete name=" << name
                  << " outputs=" << device_outputs->size() << '\n';
      }
    } else {
      Tensor output{value->second.shape, std::vector<float>(value->second.slot.live_elements)};
      if (!host_outputs || !arena.Download(output.data.data(), value->second.slot, output.data.size())) {
        if (std::getenv("PPOCR_GPU_ONLY_TRACE") != nullptr) {
          std::cerr << "GPU-only graph output download failed for " << name << '\n';
        }
        return fail();
      }
      host_outputs->emplace(name, std::move(output));
    }
  }
  // All public output data is now host-owned. Return every dynamic device
  // slot to the model arena so the next inference reuses the same peak
  // allocation instead of accumulating one graph's activations per call.
  for (auto& [_, value] : values) arena.Release(value.slot);
  values.clear();
  return true;
}

bool OnnxLite::RunGpuOnly(std::unordered_map<std::string, Tensor> inputs,
                          std::unordered_map<std::string, Tensor>& outputs) const {
  return RunGpuOnlyInternal(&inputs, {}, &outputs, nullptr, false);
}

bool OnnxLite::RunGpuOnlyDevice(
    std::unordered_map<std::string, GpuTensor> inputs,
    std::unordered_map<std::string, GpuTensor>& outputs) const {
  return RunGpuOnlyInternal(nullptr, std::move(inputs), nullptr, &outputs, false);
}

bool OnnxLite::RunGpuOnlyCtcTop1(
    std::unordered_map<std::string, GpuTensor> inputs, CtcTop1Output& output) const {
  std::unordered_map<std::string, GpuTensor> device_outputs;
  if (!RunGpuOnlyInternal(nullptr, std::move(inputs), nullptr, &device_outputs, true)) return false;
  if (device_outputs.size() != 1) {
    for (auto& [_, value] : device_outputs) gpu_arena().Release(value.slot);
    return false;
  }
  auto value = std::move(device_outputs.begin()->second);
  if (value.shape.size() != 3 || value.shape[0] <= 0 || value.shape[1] <= 0 || value.shape[2] <= 0) {
    gpu_arena().Release(value.slot); return false;
  }
  const int batches = static_cast<int>(value.shape[0]);
  const int steps = static_cast<int>(value.shape[1]);
  const int classes = static_cast<int>(value.shape[2]);
  const std::size_t rows = std::size_t(batches) * steps;
  output = {batches, steps, std::vector<int>(rows), std::vector<float>(rows), {}, {}};
  std::vector<std::int32_t> compact_indices(rows);
  bool ok = false;
  for (const auto& cache : impl_->gpu_replay_caches) {
    if (cache.output.index == value.slot.index && cache.ctc_indices.resident &&
        cache.ctc_probs.resident && cache.ctc_batches == batches &&
        cache.ctc_steps == steps && cache.ctc_classes == classes) {
      std::vector<float> encoded_indices(rows);
      ok = gpu_arena().Download(encoded_indices.data(), cache.ctc_indices, rows) &&
           gpu_arena().Download(output.probabilities.data(), cache.ctc_probs, rows);
      if (ok) {
        for (std::size_t i = 0; i < rows; ++i)
          compact_indices[i] = static_cast<std::int32_t>(encoded_indices[i]);
      }
      break;
    }
  }
  if (!ok) {
    ok = gpu_arena().CtcTop1(value.slot, compact_indices.data(), output.probabilities.data(),
                             batches, steps, classes, true);
  }
  gpu_arena().Release(value.slot);
  if (!ok) { output = {}; return false; }
  // The device reducer computes every row's winner probability because that
  // is the cheapest fully-Vulkan reduction.  Public CTC semantics retain a
  // probability only for an emitted character; compact blank/repeated-row
  // metadata is zero just like the CPU path.  This is result assembly over
  // already downloaded [N,T] metadata, never a CPU neural fallback.
  for (int batch = 0; batch < batches; ++batch) {
    int previous = -1;
    for (int step = 0; step < steps; ++step) {
      const std::size_t row = std::size_t(batch) * steps + step;
      const int index = compact_indices[row];
      output.indices[row] = index;
      if (index == 0 || index == previous) output.probabilities[row] = 0.F;
      previous = index;
    }
  }
  return true;
}

bool OnnxLite::ReplayGpuCtcGraphs(const std::uint64_t* keys, int count,
                                  CtcTop1Output* outputs) const {
  if (!impl_ || !keys || !outputs || count <= 0) return false;
  static const bool disabled =
      std::getenv("PPOCR_DISABLE_GPU_BATCH_REPLAY") != nullptr;
  if (disabled) return false;
  std::lock_guard lock(impl_->gpu_mutex);
  auto& arena = impl_->gpu_arena;
  ++impl_->replay_batch_requests;
  std::vector<const Impl::GpuReplayCache*> caches(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    const Impl::GpuReplayCache* found = nullptr;
    for (const auto& cache : impl_->gpu_replay_caches) {
      if (cache.key == keys[i] && cache.input.resident && cache.ctc_indices.resident &&
          cache.ctc_probs.resident && cache.ctc_batches > 0 && cache.ctc_steps > 0) {
        found = &cache;
        break;
      }
    }
    if (!found || !arena.HasPersistentGraph(keys[i])) {
      ++impl_->replay_batch_key_misses;
      return false;
    }
    caches[static_cast<std::size_t>(i)] = found;
  }
  ++impl_->replay_batch_full_hits;
  if (!arena.ReplayPersistentGraphs(keys, count)) return false;
  for (int i = 0; i < count; ++i) {
    const auto* cache = caches[static_cast<std::size_t>(i)];
    const int batches = cache->ctc_batches;
    const int steps = cache->ctc_steps;
    const std::size_t rows = std::size_t(batches) * std::size_t(steps);
    outputs[i] = {batches, steps, std::vector<int>(rows), std::vector<float>(rows), {}, {}};
    std::vector<float> encoded(rows);
    if (!arena.Download(encoded.data(), cache->ctc_indices, rows) ||
        !arena.Download(outputs[i].probabilities.data(), cache->ctc_probs, rows)) {
      outputs[i] = {};
      return false;
    }
    for (std::size_t r = 0; r < rows; ++r)
      outputs[i].indices[r] = static_cast<int>(encoded[r]);
    // Keep replayed CTC metadata identical to RunGpuOnlyCtcTop1().  The
    // device reducer quite intentionally returns a winner probability for
    // every time row, but blank and repeated rows do not emit a character.
    // Normalising this compact result metadata on the host is result assembly
    // only; logits and every neural operator remain resident on Vulkan.
    for (int batch = 0; batch < batches; ++batch) {
      int previous = -1;
      for (int step = 0; step < cache->ctc_steps; ++step) {
        const std::size_t row = std::size_t(batch) * cache->ctc_steps + step;
        const int index = outputs[i].indices[row];
        if (index == 0 || index == previous) outputs[i].probabilities[row] = 0.F;
        previous = index;
      }
    }
  }
  return true;
}

VulkanTensorArena& OnnxLite::gpu_arena() const noexcept { return impl_->gpu_arena; }

std::uint64_t OnnxLite::GpuReplayKey(const std::vector<std::int64_t>& shape,
                                     bool fuse_terminal_ctc, int rgb_source_width,
                                     int rgb_source_height) const noexcept {
  std::uint64_t key = reinterpret_cast<std::uintptr_t>(impl_.get());
  for (const auto dim : shape)
    key = (key * 1315423911ull) ^ static_cast<std::uint64_t>(dim);
  key ^= fuse_terminal_ctc ? 1ull : 0ull;
  if (rgb_source_width > 0 || rgb_source_height > 0) {
    key = (key * 1315423911ull) ^ static_cast<std::uint64_t>(rgb_source_width);
    key = (key * 2654435761ull) ^ static_cast<std::uint64_t>(rgb_source_height);
    key ^= 0x9e3779b97f4a7c15ull;
  }
  return key;
}

VulkanTensorSlot OnnxLite::PeekGpuReplayInput(const std::vector<std::int64_t>& shape,
                                              bool fuse_terminal_ctc) const {
  VulkanTensorSlot empty{};
  if (!impl_ || shape.empty()) return empty;
  static const bool disabled =
      std::getenv("PPOCR_DISABLE_GPU_RESIZE_INTO_REPLAY") != nullptr;
  if (disabled) return empty;
  std::lock_guard lock(impl_->gpu_mutex);
  const std::uint64_t key = GpuReplayKey(shape, fuse_terminal_ctc);
  if (!impl_->gpu_arena.HasPersistentGraph(key)) return empty;
  for (const auto& cache : impl_->gpu_replay_caches) {
    if (cache.key == key && cache.input.resident) return cache.input;
  }
  return empty;
}

VulkanTensorSlot OnnxLite::PeekGpuReplayRgb(const std::vector<std::int64_t>& nchw_shape,
                                            int source_width, int source_height) const {
  VulkanTensorSlot empty{};
  if (!impl_ || nchw_shape.empty() || source_width <= 0 || source_height <= 0) return empty;
  // Folding the packed-RGB upload + bilinear resize into the recorded detector
  // graph trades one standalone submission for a larger replay payload. Same-
  // host measurement splits by source size: large pages win because the resize
  // dispatch dominates their front end (~13% on 1920x1080), while compact
  // pages pay more in recording/replay bookkeeping than the folded dispatch
  // saves (~13% loss on 640x480). Gate adaptively at one megapixel so both
  // regimes take their winning path without deployment-side tuning. Explicit
  // environment overrides stay authoritative: DISABLE always wins, ENABLE
  // forces the fold even on compact pages.
  static const bool disabled =
      std::getenv("PPOCR_DISABLE_GPU_RGB_IN_GRAPH") != nullptr;
  static const bool forced =
      !disabled && std::getenv("PPOCR_ENABLE_GPU_RGB_IN_GRAPH") != nullptr;
  constexpr std::int64_t kAdaptiveMinSourcePixels = 1000000;
  if (!forced &&
      static_cast<std::int64_t>(source_width) * source_height <
          kAdaptiveMinSourcePixels) {
    return empty;
  }
  std::lock_guard lock(impl_->gpu_mutex);
  const std::uint64_t key =
      GpuReplayKey(nchw_shape, false, source_width, source_height);
  if (!impl_->gpu_arena.HasPersistentGraph(key)) return empty;
  for (const auto& cache : impl_->gpu_replay_caches) {
    if (cache.key == key && cache.rgb.resident &&
        cache.rgb_source_width == source_width &&
        cache.rgb_source_height == source_height) {
      return cache.rgb;
    }
  }
  return empty;
}

bool OnnxLite::RunGpuOnlyDeviceFromRgb(
    const GpuRgbFrontEnd& front,
    std::unordered_map<std::string, GpuTensor>& outputs) const {
  if (!front.rgb.resident || front.source_width <= 0 || front.source_height <= 0 ||
      front.region_width <= 0 || front.region_height <= 0 ||
      front.output_width <= 0 || front.output_height <= 0 ||
      front.left < 0 || front.top < 0) {
    return false;
  }
  return RunGpuOnlyInternal(nullptr, {}, nullptr, &outputs, false, &front);
}

std::size_t OnnxLite::GpuImmutableBytes() const noexcept {
  return impl_ ? impl_->gpu_immutable_bytes : 0;
}

void OnnxLite::ReleaseGpuConstants() const noexcept {
  if (!impl_) return;
  std::lock_guard lock(impl_->gpu_mutex);
  // `keep_gpu_constants == false` is also the bounded-memory policy for
  // GPU-only OCR. Persistent replay entries pin activation-sized input/output
  // slots, so keeping them after a detector/recognizer hand-off defeats that
  // policy on large or highly variable pages. Releasing the model's replay
  // cache here affects only a future device-side scheduling optimization; the
  // next strict graph records again entirely on Vulkan.
  while (!impl_->gpu_replay_caches.empty()) impl_->DropGpuReplayCache(0);
  for (auto& [_, slot] : impl_->gpu_constants) impl_->gpu_arena.Release(slot);
  impl_->gpu_constants.clear();
}

std::unordered_map<std::string, Tensor> OnnxLite::Run(std::unordered_map<std::string, Tensor> inputs) const {
  if (impl_->backend == Backend::gpu_only) {
    std::unordered_map<std::string, Tensor> gpu_outputs;
    if (RunGpuOnly(std::move(inputs), gpu_outputs)) return gpu_outputs;
    // This is a hard contract boundary: a partially compiled graph must not
    // appear to work by falling through to the portable activation executor.
    // Preserve a driver-reset diagnosis for callers. This is still a hard
    // error; it must never select the portable CPU graph.
    if (VulkanLastSubmissionResult() == -4) {
      Fail("GPU-only Vulkan graph failed: VK_ERROR_DEVICE_LOST");
    }
    Fail("GPU-only graph contains an unsupported Vulkan operator");
  }
  // Parameters remain in the immutable model store.  The previous version
  // copied every initializer into this map on every image, causing a large
  // temporary memory spike and avoidable bandwidth cost.
  std::unordered_map<std::string, Tensor> values;
  values.reserve(impl_->graph.inputs.size()+impl_->graph.nodes.size());
  auto remaining_uses = impl_->run_initial_uses;
  const auto use_slot = [&](const std::string& name) -> std::uint16_t* {
    const auto found = impl_->run_use_slots.find(name);
    if (found == impl_->run_use_slots.end()) return nullptr;
    return &remaining_uses[found->second];
  };
  const auto consume_use = [&](const std::string& name) -> bool {
    if (auto* uses = use_slot(name); uses && *uses) return --*uses == 0;
    return false;
  };
  const auto retire_use = [&](const std::string& name) {
    if (auto* uses = use_slot(name)) *uses = 0;
  };
  const bool profile = std::getenv("PPOCR_PROFILE") != nullptr;
  std::unordered_map<std::string, double> elapsed_ms;
  std::unordered_map<std::string, std::size_t> calls;
  // Nearly every PP-OCRv6 node has at most a handful of inputs. Reuse this
  // non-owning lookup list for the complete invocation instead of allocating
  // a fresh vector at every generic node. The executor is call-local, so it
  // remains safe for concurrent OCR calls and avoids TLS lifetime issues.
  std::vector<const Tensor*> input_scratch;
  input_scratch.reserve(8);
  for(const auto& name:impl_->graph.inputs){const auto it=inputs.find(name);if(it==inputs.end())Fail("missing input "+name);values.emplace(name,std::move(it->second));}
  // Hybrid RGB+Conv.0 seeds the stem output so the first Conv is skipped.
  for (auto& [name, tensor] : inputs) {
    if (!values.contains(name) && !tensor.data.empty())
      values.emplace(name, std::move(tensor));
  }
  for (std::size_t node_index = 0; node_index < impl_->graph.nodes.size(); ++node_index) {
    const auto& n = impl_->graph.nodes[node_index];
    if (!n.out.empty() && values.contains(n.out[0])) {
      for (const auto& name : n.in) {
        if (name.empty() || impl_->graph.initializers.contains(name)) continue;
        consume_use(name);
      }
      continue;
    }
    // The SE source feature map is only consumed by its reduction and final
    // channel-gate multiply.  Its allocation can therefore become the final
    // output in place, removing the full broadcast-Mul destination.  The
    // fused kernel completes all reductions before modifying that source.
    if (n.op == "FusedSEGateMul" && n.in.size() == 5 && n.out.size() == 1 &&
        !impl_->graph_outputs.contains(n.out[0])) {
      const auto source = values.find(n.in[0]);
      const auto source_use = use_slot(n.in[0]);
      if (source != values.end() && source_use && *source_use == 1) {
        auto& in = input_scratch;
        in.clear();
        for (const auto& name : n.in) {
          const auto value = values.find(name);
          const auto initializer = impl_->graph.initializers.find(name);
          const Tensor* input = value != values.end() ? &value->second :
              (initializer != impl_->graph.initializers.end() ? &initializer->second : nullptr);
          if (!input) Fail("missing node value " + name);
          in.push_back(input);
        }
        const auto start = profile ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
        SqueezeExcitationGateMulInplace(n, in, source->second);
        auto renamed = values.extract(source);
        renamed.key() = n.out[0];
        values.insert(std::move(renamed));
        if (const char* checkpoint = std::getenv("PPOCR_CPU_CHECKPOINT"); checkpoint &&
            n.out[0] == checkpoint) {
          DumpTensorFile(values.find(n.out[0])->second);
        }
        retire_use(n.in[0]);
        if (profile) {
          const auto profile_key = n.name.empty() ? n.op : n.op + ":" + n.name;
          elapsed_ms[profile_key] += std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - start).count();
          ++calls[profile_key];
        }
        continue;
      }
    }
    // The recognizer bridge's valid 3x2 average pool has non-overlapping
    // windows, so it can compact a final-use activation in place. Besides
    // avoiding the output allocation, this releases the unused tail capacity
    // before transformer projections begin.
    if (n.op == "AveragePool" && n.in.size() == 1 && n.out.size() == 1 &&
        !impl_->graph_outputs.contains(n.out[0])) {
      const auto source = values.find(n.in[0]);
      const auto source_use = use_slot(n.in[0]);
      if (source != values.end() && source_use && *source_use == 1) {
        const auto start = profile ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
        if (AveragePool3x2ValidInplace(n, source->second)) {
          auto renamed = values.extract(source);
          renamed.key() = n.out[0];
          values.insert(std::move(renamed));
          retire_use(n.in[0]);
          if (profile) {
            const auto profile_key = n.name.empty() ? n.op : n.op + ":" + n.name;
            elapsed_ms[profile_key] += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            ++calls[profile_key];
          }
          continue;
        }
      }
    }
    // Residual Add/Mul nodes commonly consume an activation for the final
    // time. When that activation is the full output shape, reuse its storage
    // instead of allocating a same-sized result and then immediately freeing
    // the source. Sub/Div are safe only with the left operand because their
    // operand order is observable. Restrict this to scalar/equal-shape peers:
    // those kernels are alias-safe and preserve the original arithmetic order.
    if ((n.op=="Add" || n.op=="Sub" || n.op=="Mul" || n.op=="Div") &&
        n.in.size()==2 && n.out.size()==1 && !impl_->graph_outputs.contains(n.out[0])) {
      kernels::BinaryOp op = kernels::BinaryOp::add;
      if (n.op=="Sub") op=kernels::BinaryOp::sub;
      else if (n.op=="Mul") op=kernels::BinaryOp::mul;
      else if (n.op=="Div") op=kernels::BinaryOp::div;
      int source_index = 0;
      auto source = values.find(n.in[0]);
      auto source_use = use_slot(n.in[0]);
      if ((n.op=="Add" || n.op=="Mul") &&
          (source==values.end() || !source_use || *source_use!=1)) {
        source_index=1;
        source=values.find(n.in[1]);
        source_use=use_slot(n.in[1]);
      }
      if (source!=values.end() && source_use && *source_use==1) {
        const int other_index=1-source_index;
        const auto other_value=values.find(n.in[other_index]);
        const auto other_initializer=impl_->graph.initializers.find(n.in[other_index]);
        const Tensor* other = other_value!=values.end() ? &other_value->second :
            (other_initializer!=impl_->graph.initializers.end() ? &other_initializer->second : nullptr);
        const auto broadcast_repeat = other ? RightBroadcastRepeat(source->second, *other) : 0;
        if (other && (other->data.size()==1 || other->shape==source->second.shape ||
                      broadcast_repeat)) {
          const auto start = profile ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
          if (other->data.size()==1) {
            // For commutative ops the chosen source remains the left value to
            // BinaryScalar; scalar-left is relevant only for Sub/Div, whose
            // source is deliberately fixed to input zero above.
            kernels::BinaryScalar(source->second.data.data(), source->second.data.data(),
                                  source->second.data.size(), other->data[0], op, false);
          } else {
            if (other->shape==source->second.shape) {
              if (!TryHybridVulkanBinaryInplace(
                      impl_->backend, source->second, *other, op,
                      other_initializer!=impl_->graph.initializers.end())) {
                kernels::BinaryInplace(source->second.data.data(), other->data.data(),
                                       source->second.data.size(), op);
              }
            } else {
              if (!TryHybridVulkanBinaryBroadcastRightInplace(
                      impl_->backend, source->second, *other, broadcast_repeat, op,
                      other_initializer!=impl_->graph.initializers.end())) {
                kernels::BinaryBroadcastRightInplace(source->second.data.data(), other->data.data(),
                                                     source->second.data.size(), broadcast_repeat,
                                                     other->data.size(), op);
              }
            }
          }
          auto node=values.extract(source);
          node.key()=n.out[0];
          values.insert(std::move(node));
          // The source has been renamed rather than released. Only decrement
          // the peer input; any later uses of the output retain the compiler's
          // precomputed count under its new name.
          if (!impl_->graph.initializers.contains(n.in[other_index])) {
            if (consume_use(n.in[other_index])) values.erase(n.in[other_index]);
          }
          retire_use(n.in[source_index]);
          if (profile) {
            const auto profile_key=n.name.empty()?n.op:n.op+":"+n.name;
            elapsed_ms[profile_key]+=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-start).count();
            ++calls[profile_key];
          }
          continue;
        }
      }
    }
    // GELU is elementwise. If this is the source tensor's final consumer,
    // retain its allocation and overwrite it in place instead of copying an
    // entire activation merely to produce the next graph value. The fused
    // graph preserves the exact ONNX Erf definition in kernels::Gelu.
    // Transformer LayerNorm exports its variance as Pow(x, 2). When the
    // centered activation dies here, square it in place: this preserves the
    // exact multiplication semantics while avoiding a second full [N,T,C]
    // activation allocation/copy for each recognizer block.
    if (n.op=="Pow" && n.in.size()==2 && n.out.size()==1 &&
        !impl_->graph_outputs.contains(n.out[0])) {
      const auto source=values.find(n.in[0]);
      const auto use=use_slot(n.in[0]);
      const auto exponent_value=values.find(n.in[1]);
      const auto exponent_initializer=impl_->graph.initializers.find(n.in[1]);
      const Tensor* exponent=exponent_value!=values.end()?&exponent_value->second:
          (exponent_initializer!=impl_->graph.initializers.end()?&exponent_initializer->second:nullptr);
      if (source!=values.end() && use && *use==1 && exponent &&
          exponent->data.size()==1 && exponent->data[0]==2.F) {
        const auto start = profile ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
        kernels::Square(source->second.data.data(), source->second.data.data(),
                        source->second.data.size());
        auto node=values.extract(source); node.key()=n.out[0]; values.insert(std::move(node));
        retire_use(n.in[0]);
        if (!impl_->graph.initializers.contains(n.in[1])) {
          if (consume_use(n.in[1])) values.erase(n.in[1]);
        }
        if (profile) {
          const auto profile_key=n.name.empty()?n.op:n.op+":"+n.name;
          elapsed_ms[profile_key]+=std::chrono::duration<double,std::milli>(
              std::chrono::steady_clock::now()-start).count();
          ++calls[profile_key];
        }
        continue;
      }
    }
    if ((n.op=="FusedGelu" || n.op=="Relu" || n.op=="FusedHardSwish" ||
         n.op=="FusedScaleShift" || n.op=="HardSigmoid") && n.in.size()==1 && n.out.size()==1 &&
        !impl_->graph_outputs.contains(n.out[0])) {
      const auto source=values.find(n.in[0]);
      const auto use=use_slot(n.in[0]);
      if (source!=values.end() && use && *use==1) {
        const auto start = profile ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
        auto* data=source->second.data.data(); const auto count=source->second.data.size();
        if (n.op=="FusedGelu") {
          const auto logical=source->second.shape.empty()?count:
              count/std::size_t(source->second.shape[0]);
          kernels::Gelu(data,data,count,logical);
        }
        else if (n.op=="Relu") kernels::Relu(data,data,count);
        else if (n.op=="FusedHardSwish") kernels::HardSwish(data,data,count);
        else if (n.op=="FusedScaleShift") kernels::ScaleShift(data,data,count,Attr(n,"scale").f,Attr(n,"shift").f);
        else kernels::HardSigmoid(data,data,count,n.attr.contains("alpha")?Attr(n,"alpha").f:.2F,
                                  n.attr.contains("beta")?Attr(n,"beta").f:.5F);
        auto node=values.extract(source);
        node.key()=n.out[0];
        values.insert(std::move(node));
        if (profile) {
          const auto profile_key=n.name.empty()?n.op:n.op+":"+n.name;
          elapsed_ms[profile_key]+=std::chrono::duration<double,std::milli>(
              std::chrono::steady_clock::now()-start).count();
          ++calls[profile_key];
        }
        continue;
      }
    }
    // Keep the first dynamic Concat input as its output buffer when graph
    // liveness proves the input dies here. This is common for detector FPN
    // channel assembly and removes a full feature-map allocation/copy.
    if (n.op == "Concat" && n.in.size() >= 2 && n.out.size() == 1 &&
        !impl_->graph_outputs.contains(n.out[0])) {
      const auto source = values.find(n.in[0]);
      const auto source_use = use_slot(n.in[0]);
      if (source != values.end() && source_use && *source_use == 1) {
        // Concat receives the same short non-owning operand list as ordinary
        // nodes. Reusing it matters for FPN assembly on every detector pass:
        // the list is consumed synchronously by ConcatIntoFirst, then cleared
        // before the next node resolves any pointers.
        auto& concat_inputs = input_scratch;
        concat_inputs.clear();
        bool dynamic_alias{};
        for (std::size_t input_index = 0; input_index < n.in.size(); ++input_index) {
          const auto value = values.find(n.in[input_index]);
          const auto initializer = impl_->graph.initializers.find(n.in[input_index]);
          const Tensor* input = value != values.end() ? &value->second :
              (initializer != impl_->graph.initializers.end() ? &initializer->second : nullptr);
          if (!input || (input_index > 0 && input == &source->second)) { dynamic_alias = true; break; }
          concat_inputs.push_back(input);
        }
        if (!dynamic_alias && concat_inputs.size() == n.in.size()) {
          const auto start = profile ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
          if (ConcatIntoFirst(n, concat_inputs, source->second)) {
            auto node = values.extract(source);
            node.key() = n.out[0];
            values.insert(std::move(node));
            retire_use(n.in[0]);
            for (std::size_t input_index = 1; input_index < n.in.size(); ++input_index) {
              if (impl_->graph.initializers.contains(n.in[input_index])) continue;
              if (consume_use(n.in[input_index])) values.erase(n.in[input_index]);
            }
            if (profile) {
              const auto profile_key = n.name.empty() ? n.op : n.op + ":" + n.name;
              elapsed_ms[profile_key] += std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - start).count();
              ++calls[profile_key];
            }
            continue;
          }
        }
      }
    }
    // Terminal class-axis Softmax is its input's sole remaining consumer.
    // Reuse the logits allocation before CTC reads it: the in-place row
    // kernel first finishes the row maximum, then overwrites that row with
    // its normalized exponentials. This removes one complete [N,T,V]
    // activation/copy from every recognizer batch while preserving generic
    // Softmax for all non-terminal or non-contiguous-axis ONNX layouts.
    if (n.op == "Softmax" && n.in.size() == 1 && n.out.size() == 1 &&
        !impl_->graph_outputs.contains(n.out[0])) {
      const auto source = values.find(n.in[0]);
      const auto use = use_slot(n.in[0]);
      if (source != values.end() && use && *use == 1 &&
          !source->second.shape.empty()) {
        const int axis = Axis(AttrInt(n, "axis", impl_->graph.opset >= 13 ? -1 : 1),
                              int(source->second.shape.size()));
        const auto inner = Elements(std::vector<std::int64_t>(
            source->second.shape.begin() + axis + 1, source->second.shape.end()));
        const auto width = static_cast<std::size_t>(source->second.shape[axis]);
        if (inner == 1 && width != 0 && source->second.data.size() % width == 0) {
          const auto start = profile ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
          kernels::SoftmaxRowsInplace(source->second.data.data(),
                                      source->second.data.size() / width, width);
          auto node = values.extract(source);
          node.key() = n.out[0];
          values.insert(std::move(node));
          retire_use(n.in[0]);
          if (profile) {
            const auto profile_key = n.name.empty() ? n.op : n.op + ":" + n.name;
            elapsed_ms[profile_key] += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            ++calls[profile_key];
          }
          continue;
        }
      }
    }
    // Inference BatchNormalization is a per-channel affine transform.  Like
    // the fused BN gates, it can overwrite an activation at final use; this
    // removes a complete NCHW allocation/copy from detector branches while
    // retaining the exact per-channel scale/shift arithmetic.
    if (std::getenv("PPOCR_DISABLE_BATCHNORM_INPLACE") == nullptr &&
        n.op == "BatchNormalization" && n.in.size() == 5 && n.out.size() == 1 &&
        !impl_->graph_outputs.contains(n.out[0])) {
      const auto source = values.find(n.in[0]);
      const auto use = use_slot(n.in[0]);
      if (source != values.end() && use && *use == 1) {
        auto& in = input_scratch;
        in.clear();
        for (const auto& name : n.in) {
          const auto value = values.find(name);
          if (value != values.end()) { in.push_back(&value->second); continue; }
          const auto initializer = impl_->graph.initializers.find(name);
          if (initializer == impl_->graph.initializers.end()) Fail("missing node value " + name);
          in.push_back(&initializer->second);
        }
        const auto start = profile ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
        BatchNormInplace(n, in, source->second);
        auto node = values.extract(source);
        node.key() = n.out[0];
        values.insert(std::move(node));
        retire_use(n.in[0]);
        for (std::size_t input_index = 1; input_index < n.in.size(); ++input_index) {
          const auto& name = n.in[input_index];
          if (impl_->graph.initializers.contains(name)) continue;
          if (consume_use(name)) values.erase(name);
        }
        if (profile) {
          const auto profile_key = n.name.empty() ? n.op : n.op + ":" + n.name;
          elapsed_ms[profile_key] += std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - start).count();
          ++calls[profile_key];
        }
        continue;
      }
    }
    // Plain BatchNorm is an affine transform and its kernel is alias-safe.
    // The generic detector executor previously had to allocate a full output
    // even when the input's liveness ended at BatchNorm; the recognizer's CTC
    // executor already uses this exact storage-transfer path.  Keep both
    // executors aligned so detector feature-map peaks fall by one activation
    // at eligible normalisation boundaries.
    if (std::getenv("PPOCR_DISABLE_BATCHNORM_INPLACE") == nullptr &&
        n.op == "BatchNormalization" && n.in.size() == 5 && n.out.size() == 1 &&
        !impl_->graph_outputs.contains(n.out[0])) {
      const auto source = values.find(n.in[0]);
      const auto use = use_slot(n.in[0]);
      if (source != values.end() && use && *use == 1) {
        auto& in = input_scratch;
        in.clear();
        for (const auto& name : n.in) {
          const auto value = values.find(name);
          if (value != values.end()) { in.push_back(&value->second); continue; }
          const auto initializer = impl_->graph.initializers.find(name);
          if (initializer == impl_->graph.initializers.end()) Fail("missing node value " + name);
          in.push_back(&initializer->second);
        }
        const auto start = profile ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
        BatchNormInplace(n, in, source->second);
        auto node = values.extract(source);
        node.key() = n.out[0];
        values.insert(std::move(node));
        retire_use(n.in[0]);
        for (std::size_t input_index = 1; input_index < n.in.size(); ++input_index) {
          const auto& name = n.in[input_index];
          if (impl_->graph.initializers.contains(name)) continue;
          if (consume_use(name)) values.erase(name);
        }
        if (profile) {
          const auto profile_key = n.name.empty() ? n.op : n.op + ":" + n.name;
          elapsed_ms[profile_key] += std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - start).count();
          ++calls[profile_key];
        }
        continue;
      }
    }
    // LayerNorm is a pure row-wise transform. Transformer graphs frequently
    // consume its input for the last time, so retain that activation buffer
    // rather than allocating a second [N,T,C] tensor. This is safe because
    // each output row is written only after its complete input row has been
    // reduced, and LayerNorm supports exact src/dst aliasing.
    if (n.op=="FusedLayerNorm" && n.in.size()==3 && n.out.size()==1 &&
        !impl_->graph_outputs.contains(n.out[0])) {
      const auto source=values.find(n.in[0]);
      const auto use=use_slot(n.in[0]);
      if (source!=values.end() && use && *use==1 &&
          !source->second.shape.empty()) {
        const auto gamma_value=values.find(n.in[1]);
        const auto gamma_initializer=impl_->graph.initializers.find(n.in[1]);
        const auto beta_value=values.find(n.in[2]);
        const auto beta_initializer=impl_->graph.initializers.find(n.in[2]);
        const Tensor* gamma=gamma_value!=values.end()?&gamma_value->second:
            (gamma_initializer!=impl_->graph.initializers.end()?&gamma_initializer->second:nullptr);
        const Tensor* beta=beta_value!=values.end()?&beta_value->second:
            (beta_initializer!=impl_->graph.initializers.end()?&beta_initializer->second:nullptr);
        const auto width=static_cast<std::size_t>(source->second.shape.back());
        if (gamma && beta && width && gamma->data.size()==width && beta->data.size()==width &&
            source->second.data.size()%width==0) {
          const auto start=profile?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
          kernels::LayerNorm(source->second.data.data(), source->second.data.data(),
                             gamma->data.data(), beta->data.data(),
                             source->second.data.size()/width, width, Attr(n,"epsilon").f);
          auto node=values.extract(source); node.key()=n.out[0]; values.insert(std::move(node));
          retire_use(n.in[0]);
          for (std::size_t input_index=1; input_index<n.in.size(); ++input_index) {
            const auto& name=n.in[input_index];
            if (impl_->graph.initializers.contains(name)) continue;
            if (consume_use(name)) values.erase(name);
          }
          if (profile) {
            const auto profile_key=n.name.empty()?n.op:n.op+":"+n.name;
            elapsed_ms[profile_key]+=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-start).count();
            ++calls[profile_key];
          }
          continue;
        }
      }
    }
    if (n.op=="FusedBatchNormSwishAdd" && n.in.size()==4 && n.out.size()==1 &&
        !impl_->graph_outputs.contains(n.out[0])) {
      const auto source=values.find(n.in[0]);
      const auto use=use_slot(n.in[0]);
      const auto residual=values.find(n.in[3]);
      if (source!=values.end() && use && *use==1 &&
          residual!=values.end() && source->second.shape==residual->second.shape) {
        auto& in = input_scratch;
        in.clear();
        for (const auto& name : n.in) {
          const auto value=values.find(name);
          if (value!=values.end()) { in.push_back(&value->second); continue; }
          const auto initializer=impl_->graph.initializers.find(name);
          if (initializer==impl_->graph.initializers.end()) Fail("missing node value "+name);
          in.push_back(&initializer->second);
        }
        const auto start = profile ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
        // GPU uses the same no-slower admission contract as existing hybrid
        // segments. The CPU fallback still benefits from one fused traversal.
        const bool immutable_coefficients=impl_->graph.initializers.contains(n.in[1]) &&
            impl_->graph.initializers.contains(n.in[2]);
        if (!TryHybridVulkanChannelAffineSwishAddInplace(
                impl_->backend, source->second, *in[1], *in[2], *in[3],
                immutable_coefficients)) {
          BatchNormSwishAddInplace(in, source->second);
        }
        auto node=values.extract(source); node.key()=n.out[0]; values.insert(std::move(node));
        retire_use(n.in[0]);
        if (consume_use(n.in[3])) values.erase(n.in[3]);
        if (profile) {
          const auto profile_key=n.name.empty()?n.op:n.op+":"+n.name;
          elapsed_ms[profile_key]+=std::chrono::duration<double,std::milli>(
              std::chrono::steady_clock::now()-start).count();
          ++calls[profile_key];
        }
        continue;
      }
    }
    if (n.op == "Softmax" && n.in.size() == 1 && n.out.size() == 1 &&
        !impl_->graph_outputs.contains(n.out[0])) {
      const auto source = values.find(n.in[0]);
      const auto use = use_slot(n.in[0]);
      if (source != values.end() && use && *use == 1 &&
          !source->second.shape.empty()) {
        const int axis = Axis(AttrInt(n, "axis", impl_->graph.opset >= 13 ? -1 : 1),
                              int(source->second.shape.size()));
        const auto inner = Elements(std::vector<std::int64_t>(
            source->second.shape.begin() + axis + 1, source->second.shape.end()));
        const auto width = static_cast<std::size_t>(source->second.shape[axis]);
        if (inner == 1 && width != 0 && source->second.data.size() % width == 0) {
          const auto start = profile ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
          kernels::SoftmaxRowsInplace(source->second.data.data(),
                                      source->second.data.size() / width, width);
          auto node = values.extract(source);
          node.key() = n.out[0];
          values.insert(std::move(node));
          retire_use(n.in[0]);
          if (profile) {
            const auto profile_key = std::string("ctc:") +
                (n.name.empty() ? n.op : n.op + ":" + n.name);
            elapsed_ms[profile_key] += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            ++calls[profile_key];
          }
          continue;
        }
      }
    }
    if ((n.op=="FusedBatchNormGelu" || n.op=="FusedBatchNormSwish" ||
         n.op=="FusedBatchNormHardSwish") && n.in.size()==3 && n.out.size()==1 &&
        !impl_->graph_outputs.contains(n.out[0])) {
      const auto source=values.find(n.in[0]);
      const auto use=use_slot(n.in[0]);
      if (source!=values.end() && use && *use==1) {
        auto& in = input_scratch;
        in.clear();
        for(const auto& name:n.in) {
          const auto value=values.find(name);
          if(value!=values.end()) { in.push_back(&value->second); continue; }
          const auto initializer=impl_->graph.initializers.find(name);
          if(initializer==impl_->graph.initializers.end()) Fail("missing node value "+name);
          in.push_back(&initializer->second);
        }
        const auto start = profile ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
        const bool immutable_coefficients =
            impl_->graph.initializers.contains(n.in[1]) && impl_->graph.initializers.contains(n.in[2]);
        const bool keep_legacy_swish_output = n.op=="FusedBatchNormSwish" &&
            std::getenv("PPOCR_DISABLE_BN_SWISH_INPLACE") != nullptr;
        if (keep_legacy_swish_output) {
          source->second = BatchNormSwish(n,in);
        } else {
          const bool gpu_finished = n.op=="FusedBatchNormSwish"
              ? TryHybridVulkanChannelAffineSwishInplace(
                    impl_->backend, source->second, *in[1], *in[2], immutable_coefficients)
              : TryHybridVulkanChannelAffineInplace(
                    impl_->backend, source->second, *in[1], *in[2], immutable_coefficients);
          if (!gpu_finished) {
            if (n.op=="FusedBatchNormGelu") BatchNormGeluInplace(n,in,source->second);
            else if (n.op=="FusedBatchNormHardSwish") {
              const auto channels = std::size_t(source->second.shape[1]);
              const auto spatial = source->second.data.size() /
                  (std::size_t(source->second.shape[0]) * channels);
              kernels::BatchNormAffine(source->second.data.data(), source->second.data.data(),
                                       in[1]->data.data(), in[2]->data.data(),
                                       int(source->second.shape[0]), int(channels), spatial);
              kernels::HardSwish(source->second.data.data(), source->second.data.data(),
                                 source->second.data.size());
            } else BatchNormSwishInplace(n,in,source->second);
          } else if (n.op=="FusedBatchNormGelu") {
            const auto count=source->second.data.size();
            const auto logical=source->second.shape.empty()?count:
                count/std::size_t(source->second.shape[0]);
            kernels::Gelu(source->second.data.data(), source->second.data.data(), count, logical);
          } else if (n.op=="FusedBatchNormHardSwish") {
            kernels::HardSwish(source->second.data.data(), source->second.data.data(),
                               source->second.data.size());
          }
        }
        auto node=values.extract(source);
        node.key()=n.out[0];
        values.insert(std::move(node));
        if (profile) {
          const auto profile_key=n.name.empty()?n.op:n.op+":"+n.name;
          elapsed_ms[profile_key]+=std::chrono::duration<double,std::milli>(
              std::chrono::steady_clock::now()-start).count();
          ++calls[profile_key];
        }
        continue;
      }
    }
    // The detector terminates in Sigmoid over a large DB probability map. If
    // its Add result has no other users, overwrite that activation and rename
    // it to the declared graph output. This is numerically identical to the
    // normal out-of-place Sigmoid, but avoids one full probability-map
    // allocation/copy at the peak between detector inference and DB postprocess.
    if (n.op == "Sigmoid" && n.in.size() == 1 && n.out.size() == 1) {
      const auto source = values.find(n.in[0]);
      const auto use = use_slot(n.in[0]);
      if (source != values.end() && use && *use == 1) {
        const auto start = profile ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
        kernels::Sigmoid(source->second.data.data(), source->second.data.data(),
                         source->second.data.size());
        auto node = values.extract(source);
        node.key() = n.out[0];
        values.insert(std::move(node));
        retire_use(n.in[0]);
        if (profile) {
          const auto profile_key = n.name.empty() ? n.op : n.op + ":" + n.name;
          elapsed_ms[profile_key] += std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - start).count();
          ++calls[profile_key];
        }
        continue;
      }
    }
    // Reshape/Squeeze/Unsqueeze only alter metadata.  When their source is
    // dead after this node, transfer the backing allocation rather than copy
    // a potentially multi-megabyte recognizer activation just to change its
    // logical dimensions. Public graph outputs remain excluded so callers
    // continue receiving an independently owned result.
    if ((n.op == "Reshape" || n.op == "Squeeze" || n.op == "Unsqueeze") &&
        n.in.size() >= 1 && n.out.size() == 1 && !impl_->graph_outputs.contains(n.out[0])) {
      const auto source = values.find(n.in[0]);
      const auto use = use_slot(n.in[0]);
      if (source != values.end() && use && *use == 1) {
        const auto start = profile ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
        std::vector<std::int64_t> metadata_shape;
        if (n.op == "Reshape") {
          const auto spec_value = values.find(n.in[1]);
          const auto spec_initializer = impl_->graph.initializers.find(n.in[1]);
          const Tensor* spec = spec_value != values.end() ? &spec_value->second :
              (spec_initializer != impl_->graph.initializers.end() ? &spec_initializer->second : nullptr);
          if (!spec) Fail("missing reshape spec");
          metadata_shape.reserve(spec->data.size());
          std::int64_t known = 1;
          int infer = -1;
          for (std::size_t axis = 0; axis < spec->data.size(); ++axis) {
            std::int64_t dimension = static_cast<std::int64_t>(spec->data[axis]);
            if (dimension == -1) {
              if (infer >= 0) Fail("multiple reshape inferred dimensions");
              infer = static_cast<int>(axis);
              metadata_shape.push_back(1);
            } else {
              if (dimension == 0) dimension = source->second.shape[axis];
              metadata_shape.push_back(dimension);
              known *= dimension;
            }
          }
          if (infer >= 0) metadata_shape[static_cast<std::size_t>(infer)] =
              static_cast<std::int64_t>(source->second.data.size() / known);
          if (Elements(metadata_shape) != source->second.data.size()) Fail("reshape mismatch");
        } else {
          std::vector<std::int64_t> axes = AttrInts(n, "axes", {});
          if (axes.empty() && n.in.size() > 1) {
            const auto axes_value = values.find(n.in[1]);
            const auto axes_initializer = impl_->graph.initializers.find(n.in[1]);
            const Tensor* axes_tensor = axes_value != values.end() ? &axes_value->second :
                (axes_initializer != impl_->graph.initializers.end() ? &axes_initializer->second : nullptr);
            if (!axes_tensor) Fail("missing squeeze axes");
            for (const float axis : axes_tensor->data) axes.push_back(static_cast<std::int64_t>(axis));
          }
          if (n.op == "Unsqueeze") {
            const int rank = static_cast<int>(source->second.shape.size() + axes.size());
            std::vector<int> positions;
            positions.reserve(axes.size());
            for (auto axis : axes) {
              int position = static_cast<int>(axis);
              if (position < 0) position += rank;
              if (position < 0 || position >= rank) Fail("unsqueeze axis");
              positions.push_back(position);
            }
            std::sort(positions.begin(), positions.end());
            metadata_shape = source->second.shape;
            for (const int position : positions) metadata_shape.insert(metadata_shape.begin() + position, 1);
          } else {
            std::vector<bool> drop(source->second.shape.size());
            if (axes.empty()) {
              for (std::size_t axis = 0; axis < source->second.shape.size(); ++axis) {
                drop[axis] = source->second.shape[axis] == 1;
              }
            } else {
              for (const auto axis : axes) {
                const auto position = Axis(axis, static_cast<int>(source->second.shape.size()));
                if (source->second.shape[position] != 1) Fail("squeeze non-unit axis");
                drop[position] = true;
              }
            }
            for (std::size_t axis = 0; axis < source->second.shape.size(); ++axis) {
              if (!drop[axis]) metadata_shape.push_back(source->second.shape[axis]);
            }
          }
        }
        source->second.shape = std::move(metadata_shape);
        auto node = values.extract(source);
        node.key() = n.out[0];
        values.insert(std::move(node));
        retire_use(n.in[0]);
        if (profile) {
          const auto profile_key = n.name.empty() ? n.op : n.op + ":" + n.name;
          elapsed_ms[profile_key] += std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - start).count();
          ++calls[profile_key];
        }
        continue;
      }
    }
    if (n.op == "FusedExpandGeluProjectAdd" && n.in.size() == 5 && n.out.size() == 1 &&
        !impl_->graph_outputs.contains(n.out[0])) {
      const auto source = values.find(n.in[0]);
      const auto use = use_slot(n.in[0]);
      const auto expand_w = impl_->graph.initializers.find(n.in[1]);
      const auto expand_b = n.in[2].empty() ? impl_->graph.initializers.end()
                                            : impl_->graph.initializers.find(n.in[2]);
      const auto project_w = impl_->graph.initializers.find(n.in[3]);
      const auto project_b = n.in[4].empty() ? impl_->graph.initializers.end()
                                             : impl_->graph.initializers.find(n.in[4]);
      if (source != values.end() && use && *use == 1 &&
          expand_w != impl_->graph.initializers.end() &&
          project_w != impl_->graph.initializers.end() &&
          source->second.shape.size() == 4 &&
          expand_w->second.shape.size() == 4 && project_w->second.shape.size() == 4) {
        auto& input = source->second;
        const int batches = int(input.shape[0]), channels = int(input.shape[1]);
        const int height = int(input.shape[2]), width = int(input.shape[3]);
        const int hidden = int(expand_w->second.shape[0]);
        const bool expand_bias_ok = n.in[2].empty() ||
            (expand_b != impl_->graph.initializers.end() &&
             expand_b->second.data.size() == std::size_t(hidden));
        const bool project_bias_ok = n.in[4].empty() ||
            (project_b != impl_->graph.initializers.end() &&
             project_b->second.data.size() == std::size_t(channels));
        const bool ok = expand_w->second.shape[1] == channels &&
            expand_w->second.shape[2] == 1 && expand_w->second.shape[3] == 1 &&
            project_w->second.shape[0] == channels &&
            project_w->second.shape[1] == hidden &&
            project_w->second.shape[2] == 1 && project_w->second.shape[3] == 1 &&
            expand_bias_ok && project_bias_ok;
        if (ok && AttrInt(n, "group", 1) == 1 &&
            AttrStr(n, "auto_pad", "NOTSET") == "NOTSET") {
          const auto start = profile ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
          const auto plane = std::size_t(height) * width;
          const std::size_t sample = std::size_t(channels) * plane;
          const float* expand_bias = expand_b != impl_->graph.initializers.end()
              ? expand_b->second.data.data() : nullptr;
          const float* project_bias = project_b != impl_->graph.initializers.end()
              ? project_b->second.data.data() : nullptr;
          for (int batch = 0; batch < batches; ++batch) {
            float* sample_ptr = input.data.data() + std::size_t(batch) * sample;
            kernels::ExpandGeluProjectAdd(sample_ptr, sample_ptr,
                                          expand_w->second.data.data(), expand_bias,
                                          project_w->second.data.data(), project_bias,
                                          channels, hidden, plane);
          }
          auto renamed = values.extract(source);
          renamed.key() = n.out[0];
          values.insert(std::move(renamed));
          retire_use(n.in[0]);
          if (profile) {
            const auto profile_key = n.name.empty() ? n.op : n.op + ":" + n.name;
            elapsed_ms[profile_key] += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            ++calls[profile_key];
          }
          continue;
        }
      }
    }
    auto& in = input_scratch;
    in.clear();
    for(const auto& name:n.in){
      if(name.empty()){in.push_back(nullptr);continue;}
      const auto value=values.find(name);
      if(value!=values.end()){in.push_back(&value->second);continue;}
      const auto initializer=impl_->graph.initializers.find(name);
      if(initializer==impl_->graph.initializers.end())Fail("missing node value "+name);
      in.push_back(&initializer->second);
    }
    // FPN laterals are independent 1x1 Convs whose PointwiseConv is already
    // serial (OC=16). Running a ready wave together uses the persistent pool
    // without nesting inner ParallelFor. `PPOCR_DISABLE_CONV_WAVE` A/B.
    static const bool conv_wave =
        std::getenv("PPOCR_ENABLE_CONV_WAVE") != nullptr &&
        std::getenv("PPOCR_DISABLE_CONV_WAVE") == nullptr;
    if (conv_wave && (n.op == "Conv" || n.op == "FusedConvRelu") &&
        n.out.size() == 1 && impl_->backend != Backend::gpu_only) {
      const auto is_pointwise = [&](const Node& node) -> bool {
        if ((node.op != "Conv" && node.op != "FusedConvRelu") || node.out.size() != 1 ||
            node.in.size() < 2) return false;
        const auto weights = impl_->graph.initializers.find(node.in[1]);
        if (weights == impl_->graph.initializers.end() ||
            weights->second.shape.size() != 4 || weights->second.shape[2] != 1 ||
            weights->second.shape[3] != 1) return false;
        return AttrInt(node, "group", 1) == 1;
      };
      const auto inputs_ready = [&](const Node& node) -> bool {
        for (const auto& name : node.in) {
          if (name.empty() || impl_->graph.initializers.contains(name)) continue;
          if (!values.contains(name)) return false;
        }
        return true;
      };
      if (is_pointwise(n) && inputs_ready(n)) {
        std::vector<std::size_t> wave;
        wave.push_back(node_index);
        std::unordered_set<std::string> wave_outs{n.out[0]};
        for (std::size_t j = node_index + 1;
             j < impl_->graph.nodes.size() && wave.size() < 8; ++j) {
          const auto& other = impl_->graph.nodes[j];
          if (!other.out.empty() && values.contains(other.out[0])) continue;
          if (!is_pointwise(other) || !inputs_ready(other)) continue;
          bool depends = false;
          for (const auto& name : other.in) {
            if (wave_outs.contains(name)) { depends = true; break; }
          }
          if (depends) continue;
          wave.push_back(j);
          wave_outs.insert(other.out[0]);
        }
        if (wave.size() >= 2) {
          std::vector<std::vector<const Tensor*>> wave_inputs(wave.size());
          std::vector<bool> wave_immutable(wave.size());
          for (std::size_t t = 0; t < wave.size(); ++t) {
            const auto& node = impl_->graph.nodes[wave[t]];
            wave_inputs[t].reserve(node.in.size());
            for (const auto& name : node.in) {
              if (name.empty()) { wave_inputs[t].push_back(nullptr); continue; }
              const auto value = values.find(name);
              if (value != values.end()) { wave_inputs[t].push_back(&value->second); continue; }
              wave_inputs[t].push_back(&impl_->graph.initializers.at(name));
            }
            wave_immutable[t] = node.in.size() >= 3 &&
                impl_->graph.initializers.contains(node.in[1]) &&
                impl_->graph.initializers.contains(node.in[2]);
          }
          std::vector<Tensor> wave_results(wave.size());
          const auto start = profile ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
          kernels::ParallelForRange(static_cast<int>(wave.size()), [&](int first, int last) {
            for (int t = first; t < last; ++t) {
              const auto& node = impl_->graph.nodes[wave[static_cast<std::size_t>(t)]];
              wave_results[static_cast<std::size_t>(t)] = Execute(
                  node, wave_inputs[static_cast<std::size_t>(t)], impl_->graph.opset,
                  impl_->backend, wave_immutable[static_cast<std::size_t>(t)]);
            }
          });
          if (profile) {
            const auto elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            elapsed_ms["ConvWave"] += elapsed;
            ++calls["ConvWave"];
          }
          for (std::size_t t = 0; t < wave.size(); ++t) {
            const auto& node = impl_->graph.nodes[wave[t]];
            values.insert_or_assign(node.out[0], std::move(wave_results[t]));
            for (const auto& name : node.in) {
              if (name.empty() || impl_->graph.initializers.contains(name)) continue;
              if (consume_use(name)) values.erase(name);
            }
          }
          continue;
        }
      }
    }
    const auto start = profile ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
    const bool immutable_parameters = n.in.size() >= 3 &&
        impl_->graph.initializers.contains(n.in[1]) &&
        impl_->graph.initializers.contains(n.in[2]);
    // Attention Q/K/V is the only intentionally multi-output runtime node.
    // Materialise all three final layouts from the shared projection before
    // retiring its source; the generic Execute interface remains single-output
    // for the rest of the ONNX subset.
    if (n.op == "FusedQkvSplit") {
      auto outputs = QkvSplit(n, *in[0]);
      if (outputs.size() != n.out.size()) Fail("FusedQkvSplit output count");
      if (profile) {
        const auto profile_key = n.name.empty() ? n.op : n.op + ":" + n.name;
        elapsed_ms[profile_key] += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        ++calls[profile_key];
      }
      for (std::size_t output_index = 0; output_index < n.out.size(); ++output_index) {
        values.insert_or_assign(n.out[output_index], std::move(outputs[output_index]));
        if (const char* checkpoint = std::getenv("PPOCR_CPU_CHECKPOINT"); checkpoint &&
            n.out[output_index] == checkpoint) {
          const auto produced = values.find(n.out[output_index]);
          std::cerr << "CPU checkpoint " << checkpoint << " shape=";
          for (const auto dimension : produced->second.shape) std::cerr << dimension << ',';
          std::cerr << " values=";
          for (std::size_t item = 0; item < std::min<std::size_t>(produced->second.data.size(), 8); ++item)
            std::cerr << produced->second.data[item] << ',';
          std::cerr << '\n';
        }
      }
      for (const auto& name : n.in) {
        if (name.empty() || impl_->graph.initializers.contains(name)) continue;
        if (consume_use(name)) values.erase(name);
      }
      continue;
    }
    auto value=Execute(n,in,impl_->graph.opset,impl_->backend,immutable_parameters);
    if (profile) {
      const auto profile_key = n.name.empty() ? n.op : n.op + ":" + n.name;
      elapsed_ms[profile_key] += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count();
      ++calls[profile_key];
    }
    if(n.out.size()!=1)Fail("multi-output node unsupported");
    values.insert_or_assign(n.out[0],std::move(value));
      if (const char* checkpoint = std::getenv("PPOCR_CPU_CHECKPOINT"); checkpoint &&
        n.out[0] == checkpoint) {
      const auto produced = values.find(n.out[0]);
      DumpTensorFile(produced->second);
      std::cerr << "CPU checkpoint " << checkpoint << " shape=";
      for (const auto dimension : produced->second.shape) std::cerr << dimension << ',';
      const auto [minimum, maximum] = std::minmax_element(produced->second.data.begin(),
                                                           produced->second.data.end());
      std::cerr << " range=" << *minimum << ',' << *maximum;
      std::cerr << " values=";
      for (std::size_t item = 0; item < std::min<std::size_t>(produced->second.data.size(), 8); ++item)
        std::cerr << produced->second.data[item] << ',';
      std::cerr << '\n';
    }
    // Release intermediates immediately after their final consumer. This
    // keeps peak activation memory proportional to the graph frontier rather
    // than the total number of PP-OCRv6 nodes.
    for (const auto& name : n.in) {
      if (name.empty() || impl_->graph.initializers.contains(name)) continue;
      if (consume_use(name)) values.erase(name);
    }
  }
  std::unordered_map<std::string,Tensor> out;
  for(const auto& name:impl_->graph.outputs){
    const auto value=values.find(name);
    // The final graph node has completed, so hand its activation directly to
    // the caller rather than copying recognition logits (often several MiB
    // per dense-page batch) into a second map.
    if(value!=values.end()){out.emplace(name,std::move(value->second));continue;}
    const auto initializer=impl_->graph.initializers.find(name);
    if(initializer==impl_->graph.initializers.end())Fail("missing graph output "+name);
    out.emplace(name,initializer->second);
  }
  if (profile) {
    std::vector<std::pair<std::string, double>> entries(elapsed_ms.begin(), elapsed_ms.end());
    std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.second > rhs.second;
    });
    std::cerr << "ppocr operator profile (ms):\n";
    for (const auto& [op, ms] : entries) {
      std::cerr << "  " << op << " x" << calls[op] << ": " << ms << '\n';
    }
  }
  return out;
}

OnnxLite::CtcTop1Output OnnxLite::RunCtcTop1(
    std::unordered_map<std::string, Tensor> inputs) const {
  if (!impl_->output_is_terminal_softmax || !impl_->terminal_softmax_is_class_axis) {
    Fail("recognizer output is not terminal class-axis Softmax");
  }
  // Reset the fixed model's compact use counters for this crop.  All helpers
  // below deliberately preserve the old map's observable semantics: an
  // absent/dead entry means it cannot be reused in place, and consuming the
  // final use immediately releases its tensor from `values`.
  // Do not keep these vectors in TLS. RunCtcTop1 can be called from
  // short-lived outer crop workers; with MinGW/Windows, emulated-TLS vector
  // destructors can cross a different pthread DLL at thread teardown.  The
  // vectors are tiny (one entry per dynamic graph value), and the local
  // lifetime is both safer and still bounded to one crop invocation.
  std::vector<std::uint16_t> ctc_remaining_uses = impl_->ctc_initial_uses;
  const auto use_slot = [&](const std::string& name) -> std::uint16_t* {
    const auto found = impl_->ctc_use_slots.find(name);
    if (found == impl_->ctc_use_slots.end()) return nullptr;
    auto& uses = ctc_remaining_uses[found->second];
    return uses ? &uses : nullptr;
  };
  const auto consume_use = [&](const std::string& name) -> bool {
    const auto found = impl_->ctc_use_slots.find(name);
    if (found == impl_->ctc_use_slots.end()) return false;
    auto& uses = ctc_remaining_uses[found->second];
    if (!uses) return false;
    return --uses == 0;
  };
  const auto retire_use = [&](const std::string& name) {
    const auto found = impl_->ctc_use_slots.find(name);
    if (found != impl_->ctc_use_slots.end()) ctc_remaining_uses[found->second] = 0;
  };
  const bool profile = std::getenv("PPOCR_PROFILE") != nullptr;
  std::unordered_map<std::string, double> elapsed_ms;
  std::unordered_map<std::string, std::size_t> calls;
  const auto emit_profile = [&] {
    if (!profile) return;
    std::vector<std::pair<std::string, double>> entries(elapsed_ms.begin(), elapsed_ms.end());
    std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.second > rhs.second;
    });
    std::cerr << "ppocr CTC operator profile (ms):\n";
    for (const auto& [op, ms] : entries) {
      std::cerr << "  " << op << " x" << calls[op] << ": " << ms << '\n';
    }
  };  // Per-call ownership makes the CTC executor's activation lifetime explicit
  // and benchmarked faster than retaining a thread-local node map on the
  // current tiny/small/medium PP-OCRv6 models.
  std::unordered_map<std::string, Tensor> values;
  values.reserve(impl_->graph.inputs.size() + impl_->graph.nodes.size());
  for(const auto& name:impl_->graph.inputs){const auto it=inputs.find(name);if(it==inputs.end())Fail("missing input "+name);values.emplace(name,std::move(it->second));}
  // Hybrid RGB+first-conv seeds the stem output so that Conv is skipped,
  // matching OnnxLite::Run(). Dummy `x` still satisfies graph.inputs.
  for (auto& [name, tensor] : inputs) {
    if (!values.contains(name) && !tensor.data.empty())
      values.emplace(name, std::move(tensor));
  }
  // PP-OCRv6 nodes have only a few inputs. Keep this small non-owning list
  // invocation-local too: it avoids the same MinGW emulated-TLS destructor
  // issue on short-lived outer recognition workers while retaining one
  // allocation at most for an exceptional future supported node.
  std::vector<const Tensor*> input_scratch;
  input_scratch.reserve(8);
  input_scratch.clear();
  const auto terminal=impl_->graph.nodes.size()-1;
  const auto logits_index = terminal - 1;
  for(std::size_t index=0;index<terminal;++index) {
    const auto& n=impl_->graph.nodes[index];
    if (!n.out.empty() && values.contains(n.out[0])) {
      for (const auto& name : n.in) {
        if (name.empty() || impl_->graph.initializers.contains(name)) continue;
        consume_use(name);
      }
      continue;
    }
    // Keep the production recognizer executor structurally aligned with
    // Run(): detector/recognizer graph variants may carry SE blocks before a
    // CTC terminal, and the full-size source tensor has a provable final use
    // at this fused node. Reduce/gate calculation finishes before in-place
    // scaling, so the original graph arithmetic is preserved.
    if (n.op == "FusedSEGateMul" && n.in.size() == 5 && n.out.size() == 1 &&
        !impl_->graph_outputs.contains(n.out[0])) {
      const auto source = values.find(n.in[0]);
      const auto source_use = use_slot(n.in[0]);
      if (source != values.end() && source_use && *source_use == 1) {
        auto& in = input_scratch;
        in.clear();
        for (const auto& name : n.in) {
          const auto value = values.find(name);
          const auto initializer = impl_->graph.initializers.find(name);
          const Tensor* input = value != values.end() ? &value->second :
              (initializer != impl_->graph.initializers.end() ? &initializer->second : nullptr);
          if (!input) Fail("missing node value " + name);
          in.push_back(input);
        }
        const auto start = profile ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
        SqueezeExcitationGateMulInplace(n, in, source->second);
        auto renamed = values.extract(source);
        renamed.key() = n.out[0];
        values.insert(std::move(renamed));
        retire_use(n.in[0]);
        if (profile) {
          const auto profile_key = std::string("ctc:") +
              (n.name.empty() ? n.op : n.op + ":" + n.name);
          elapsed_ms[profile_key] += std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - start).count();
          ++calls[profile_key];
        }
        continue;
      }
    }
    if (n.op == "AveragePool" && n.in.size() == 1 && n.out.size() == 1 &&
        !impl_->graph_outputs.contains(n.out[0])) {
      const auto source = values.find(n.in[0]);
      const auto use = use_slot(n.in[0]);
      if (source != values.end() && use && *use == 1) {
        const auto start = profile ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
        if (AveragePool3x2ValidInplace(n, source->second)) {
          auto renamed = values.extract(source);
          renamed.key() = n.out[0];
          values.insert(std::move(renamed));
          retire_use(n.in[0]);
          if (profile) {
            const auto profile_key = std::string("ctc:") +
                (n.name.empty() ? n.op : n.op + ":" + n.name);
            elapsed_ms[profile_key] += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            ++calls[profile_key];
          }
          continue;
        }
      }
    }
    // The last CTC projection has no downstream tensor consumer other than
    // terminal Softmax, whose full [N,T,V] result is intentionally skipped by
    // this executor. Compute Top-1 classes and their exact probabilities in
    // one reusable sequence workspace. This removes the full batched
    // vocabulary-wide logits allocation (often several MiB per crop batch)
    // while preserving GEMM's K-order and CTC blank/repeat semantics.
    if (std::getenv("PPOCR_DISABLE_FUSED_TERMINAL_CTC") == nullptr &&
        index == logits_index && impl_->terminal_logits_is_matmul && n.in.size() >= 2 &&
        n.out.size() == 1) {
      const auto activation = values.find(n.in[0]);
      const auto weights = impl_->graph.initializers.find(n.in[1]);
      const auto bias = impl_->terminal_logits_has_bias && n.in.size() >= 3
          ? impl_->graph.initializers.find(n.in[2]) : impl_->graph.initializers.end();
      if (activation != values.end() && weights != impl_->graph.initializers.end() &&
          (!impl_->terminal_logits_has_bias || bias != impl_->graph.initializers.end()) &&
          activation->second.shape.size() >= 2 && weights->second.shape.size() == 2) {
        const int depth = int(activation->second.shape.back());
        const int columns = int(weights->second.shape[1]);
        const std::size_t rows = activation->second.data.size() / std::size_t(depth);
        if (depth > 0 && weights->second.shape[0] == depth &&
            rows * std::size_t(depth) == activation->second.data.size() && columns > 0 &&
            (!impl_->terminal_logits_has_bias || bias->second.data.size() == std::size_t(columns))) {
          const int batches = int(activation->second.shape[0]);
          const int steps = int(rows / std::size_t(batches));
          if (batches > 0 && rows == std::size_t(batches) * steps) {
            const auto start = profile ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point{};
            CtcTop1Output result{batches, steps, std::vector<int>(rows), std::vector<float>(rows)};
            const float* ctc_bias = impl_->terminal_logits_has_bias ? bias->second.data.data()
                                                                    : nullptr;
            bool used_gpu = false;
            if (impl_->backend == Backend::hybrid &&
                std::getenv("PPOCR_DISABLE_VULKAN_GEMM_CTC") == nullptr) {
               struct AdmissionKey {
                int rows, depth, vocab;
                std::uint64_t context;
                bool operator==(const AdmissionKey&) const = default;
              };
              struct AdmissionKeyHash {
                std::size_t operator()(const AdmissionKey& key) const noexcept {
                  return (std::size_t(key.rows) * 1315423911u) ^
                         (std::size_t(key.depth) << 16) ^ std::size_t(key.vocab) ^
                         std::size_t(key.context);
                }
              };
              static std::mutex admission_mutex;
              static std::unordered_map<AdmissionKey, bool, AdmissionKeyHash> admitted;
              const AdmissionKey shape{int(rows), depth, columns, HybridAdmissionContext()};
              bool select_gpu{};
              {
                std::lock_guard lock(admission_mutex);
                const auto found = admitted.find(shape);
                if (found != admitted.end()) {
                  select_gpu = found->second;
                } else {
                  double gpu_ms{}, cpu_ms{};
                  // The public hybrid contract is a full synchronous-boundary
                  // comparison: when Vulkan is no slower than the CPU kernel
                  // for this exact adapter/runtime generation, use Vulkan to
                  // reduce CPU work. Page-level queue serialization is handled
                  // by OCR's hybrid graph worker limit, not by a stricter
                  // per-operator threshold that would reject an equal-speed
                  // GPU result.
                  select_gpu = VulkanGemmCtcTop1NoSlowerThanCpu(
                      int(rows), depth, columns, &gpu_ms, &cpu_ms, true);
                  admitted.emplace(shape, select_gpu);
                  if (std::getenv("PPOCR_HYBRID_TRACE") != nullptr) {
                    std::cerr << "hybrid CTC GEMM rows=" << rows << " depth=" << depth
                              << " vocab=" << columns << " gpu_ms=" << gpu_ms
                              << " cpu_ms=" << cpu_ms << " select=" << select_gpu << '\n';
                  }
                }
              }
              used_gpu = select_gpu && VulkanGemmCtcTop1(
                  result.indices.data(), result.probabilities.data(),
                  activation->second.data.data(), weights->second.data.data(), ctc_bias,
                  int(rows), depth, columns, steps, true);
            }
            if (!used_gpu) {
              kernels::GemmCtcTop1(result.indices.data(), result.probabilities.data(),
                                   activation->second.data.data(), weights->second.data.data(),
                                   ctc_bias, int(rows), depth, columns, steps);
            }
            if (profile) {
              const auto profile_key = std::string("ctc:fused_terminal_gemm_top1:") +
                  (n.name.empty() ? n.op : n.op + ":" + n.name);
              elapsed_ms[profile_key] += std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - start).count();
              ++calls[profile_key];
              emit_profile();
            }
            return result;
          }
        }
      }
    }
    if (n.op=="FusedExpandGeluProjectAdd" && n.in.size()==5 && n.out.size()==1 &&
        !impl_->graph_outputs.contains(n.out[0])) {
      const auto source=values.find(n.in[0]);
      const auto source_use=use_slot(n.in[0]);
      const auto expand_w=impl_->graph.initializers.find(n.in[1]);
      const auto expand_b=n.in[2].empty()?impl_->graph.initializers.end():
          impl_->graph.initializers.find(n.in[2]);
      const auto project_w=impl_->graph.initializers.find(n.in[3]);
      const auto project_b=n.in[4].empty()?impl_->graph.initializers.end():
          impl_->graph.initializers.find(n.in[4]);
      if (source!=values.end() && source_use && *source_use==1 &&
          expand_w!=impl_->graph.initializers.end() &&
          project_w!=impl_->graph.initializers.end() &&
          source->second.shape.size()==4 &&
          expand_w->second.shape.size()==4 && project_w->second.shape.size()==4) {
        auto& input=source->second;
        const int batches=int(input.shape[0]), channels=int(input.shape[1]);
        const int height=int(input.shape[2]), width=int(input.shape[3]);
        const int hidden=int(expand_w->second.shape[0]);
        const bool expand_bias_ok=n.in[2].empty() ||
            (expand_b!=impl_->graph.initializers.end() &&
             expand_b->second.data.size()==std::size_t(hidden));
        const bool project_bias_ok=n.in[4].empty() ||
            (project_b!=impl_->graph.initializers.end() &&
             project_b->second.data.size()==std::size_t(channels));
        if (expand_w->second.shape[1]==channels && expand_w->second.shape[2]==1 &&
            expand_w->second.shape[3]==1 && project_w->second.shape[0]==channels &&
            project_w->second.shape[1]==hidden && project_w->second.shape[2]==1 &&
            project_w->second.shape[3]==1 && expand_bias_ok && project_bias_ok &&
            AttrInt(n,"group",1)==1 && AttrStr(n,"auto_pad","NOTSET")=="NOTSET") {
          const auto start=profile?std::chrono::steady_clock::now():
              std::chrono::steady_clock::time_point{};
          const auto plane=std::size_t(height)*width;
          const std::size_t sample=std::size_t(channels)*plane;
          const float* expand_bias=expand_b!=impl_->graph.initializers.end()?
              expand_b->second.data.data():nullptr;
          const float* project_bias=project_b!=impl_->graph.initializers.end()?
              project_b->second.data.data():nullptr;
          for (int batch=0; batch<batches; ++batch) {
            float* sample_ptr=input.data.data()+std::size_t(batch)*sample;
            kernels::ExpandGeluProjectAdd(sample_ptr, sample_ptr,
                                          expand_w->second.data.data(), expand_bias,
                                          project_w->second.data.data(), project_bias,
                                          channels, hidden, plane);
          }
          auto renamed=values.extract(source);
          renamed.key()=n.out[0];
          values.insert(std::move(renamed));
          retire_use(n.in[0]);
          if (profile) {
            const auto profile_key=std::string("ctc:")+
                (n.name.empty()?n.op:n.op+":"+n.name);
            elapsed_ms[profile_key]+=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-start).count();
            ++calls[profile_key];
          }
          continue;
        }
      }
    }
    // The production CTC executor must retain the same last-use residual
    // fusion as Run().  Small/medium recognizers spend much of their time in
    // these 1x1 projection shortcuts.  Reusing the dying projection input
    // removes one full NCHW destination and lets the AVX-512/AVX2/NEON fused
    // kernel keep the final Add in its output writeback.
    if ((n.op=="FusedPointwiseConvAdd" || n.op=="FusedPointwiseConvAddRelu" ||
         n.op=="FusedPointwiseConvAddSwish") && n.in.size()==4 && n.out.size()==1 &&
        !impl_->graph_outputs.contains(n.out[0])) {
      const auto source=values.find(n.in[0]);
      const auto source_use=use_slot(n.in[0]);
      const auto residual=values.find(n.in[3]);
      const auto weights_initializer=impl_->graph.initializers.find(n.in[1]);
      const auto bias_initializer=n.in[2].empty() ? impl_->graph.initializers.end() :
          impl_->graph.initializers.find(n.in[2]);
      if (source!=values.end() && source_use && *source_use==1 &&
          residual!=values.end() && weights_initializer!=impl_->graph.initializers.end() &&
          source->second.shape.size()==4 && residual->second.shape.size()==4) {
        const auto& input=source->second;
        const auto& weights=weights_initializer->second;
        const int batches=int(input.shape[0]), input_channels=int(input.shape[1]);
        const int height=int(input.shape[2]), width=int(input.shape[3]);
        const int output_channels=weights.shape.size()==4 ? int(weights.shape[0]) : 0;
        const bool pointwise=weights.shape.size()==4 && weights.shape[1]==input_channels &&
            weights.shape[2]==1 && weights.shape[3]==1 &&
            AttrInt(n,"group",1)==1 && AttrStr(n,"auto_pad","NOTSET")=="NOTSET" &&
            AttrInts(n,"strides",{1,1})==std::vector<std::int64_t>{1,1} &&
            AttrInts(n,"pads",{0,0,0,0})==std::vector<std::int64_t>{0,0,0,0} &&
            AttrInts(n,"dilations",{1,1})==std::vector<std::int64_t>{1,1} &&
            residual->second.shape==std::vector<std::int64_t>{batches,output_channels,height,width};
        const Tensor* bias=bias_initializer!=impl_->graph.initializers.end() ?
            &bias_initializer->second : nullptr;
        if (pointwise && (!bias || bias->data.size()==std::size_t(output_channels))) {
          const auto start=profile?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
          const float* bias_data=bias?bias->data.data():nullptr;
          const auto plane=std::size_t(height)*width;
          // A 1x1 projection reads every input channel for every output
          // channel, so source/destination aliasing is not legal even when
          // the source vector happens to have enough capacity.  We still
          // bypass the generic node executor here to avoid rebuilding its
          // input vector and to release the source immediately afterwards.
          const auto output_elements=std::size_t(batches)*output_channels*plane;
          Tensor result;
          result.shape={batches,output_channels,height,width};
          result.data.resize(output_elements);
          float* destination=result.data.data();
          // This is the recognizer's actual CTC execution path, including
          // true NCHW crop batches.  Do not bypass hybrid here: doing so
          // previously made the generic executor GPU-capable while the
          // production recognizer always forced these residual projections
          // back to CPU.  The helper has a shape-local, full-boundary
          // no-slower admission gate (H2D + dispatch + fence + D2H compared
          // with the current SIMD kernel), so a UMA/discrete GPU only owns
          // this whole batched segment when it is demonstrably beneficial.
          const bool immutable_coefficients =
              impl_->graph.initializers.contains(n.in[1]) &&
              bias && impl_->graph.initializers.contains(n.in[2]);
          const bool gpu_finished = TryHybridVulkanPointwiseConvAdd(
              impl_->backend, result, input, weights, bias, residual->second,
              n.op=="FusedPointwiseConvAddRelu", immutable_coefficients,
              n.op=="FusedPointwiseConvAddSwish");
          if (!gpu_finished && n.op=="FusedPointwiseConvAdd") {
            kernels::PointwiseConvAddBatch(destination,input.data.data(),weights.data.data(),bias_data,
                                           residual->second.data.data(),batches,output_channels,input_channels,plane);
          } else if (!gpu_finished && n.op=="FusedPointwiseConvAddRelu") {
            kernels::PointwiseConvAddReluBatch(destination,input.data.data(),weights.data.data(),bias_data,
                                               residual->second.data.data(),batches,output_channels,input_channels,plane);
          } else if (!gpu_finished) {
            kernels::PointwiseConvAddSwishBatch(destination,input.data.data(),weights.data.data(),bias_data,
                                                residual->second.data.data(),batches,output_channels,input_channels,plane);
          }
          values.insert_or_assign(n.out[0],std::move(result));
          values.erase(n.in[0]);
          if (consume_use(n.in[3])) values.erase(n.in[3]);
          retire_use(n.in[0]);
          if (profile) {
            const auto profile_key=std::string("ctc:")+(n.name.empty()?n.op:n.op+":"+n.name);
            elapsed_ms[profile_key]+=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-start).count();
            ++calls[profile_key];
          }
          continue;
        }
      }
    }
    // This is the recognizer's production execution path. A fused LayerNorm
    // can overwrite an activation which has reached its final consumer: each
    // row is completely reduced before its destination row is stored, so the
    // kernel's src/dst alias is exact. Small and medium models contain several
    // [N,T,C] transformer normalizations; retaining this allocation avoids a
    // second full activation at every eligible boundary.
    if (std::getenv("PPOCR_DISABLE_CTC_LAYERNORM_INPLACE") == nullptr &&
        n.op == "FusedLayerNorm" && n.in.size() == 3 && n.out.size() == 1 &&
        !impl_->graph_outputs.contains(n.out[0])) {
      const auto source = values.find(n.in[0]);
      const auto source_use = use_slot(n.in[0]);
      if (source != values.end() && source_use && *source_use == 1 &&
          !source->second.shape.empty()) {
        const auto gamma_value = values.find(n.in[1]);
        const auto gamma_initializer = impl_->graph.initializers.find(n.in[1]);
        const auto beta_value = values.find(n.in[2]);
        const auto beta_initializer = impl_->graph.initializers.find(n.in[2]);
        const Tensor* gamma = gamma_value != values.end() ? &gamma_value->second :
            (gamma_initializer != impl_->graph.initializers.end() ?
                &gamma_initializer->second : nullptr);
        const Tensor* beta = beta_value != values.end() ? &beta_value->second :
            (beta_initializer != impl_->graph.initializers.end() ?
                &beta_initializer->second : nullptr);
        const auto width = static_cast<std::size_t>(source->second.shape.back());
        if (gamma && beta && width && gamma->data.size() == width &&
            beta->data.size() == width && source->second.data.size() % width == 0) {
          const auto start = profile ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
          kernels::LayerNorm(source->second.data.data(), source->second.data.data(),
                             gamma->data.data(), beta->data.data(),
                             source->second.data.size() / width, width,
                             Attr(n, "epsilon").f);
          auto renamed = values.extract(source);
          renamed.key() = n.out[0];
          values.insert(std::move(renamed));
          retire_use(n.in[0]);
          for (std::size_t input_index = 1; input_index < n.in.size(); ++input_index) {
            const auto& name = n.in[input_index];
            if (impl_->graph.initializers.contains(name)) continue;
            if (consume_use(name)) {
              values.erase(name);
            }
          }
          if (profile) {
            const auto profile_key = std::string("ctc:") +
                (n.name.empty() ? n.op : n.op + ":" + n.name);
            elapsed_ms[profile_key] += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            ++calls[profile_key];
          }
          continue;
        }
      }
    }
    if ((n.op=="Add" || n.op=="Sub" || n.op=="Mul" || n.op=="Div") &&
        n.in.size()==2 && n.out.size()==1 && !impl_->graph_outputs.contains(n.out[0])) {
      kernels::BinaryOp op = kernels::BinaryOp::add;
      if (n.op=="Sub") op=kernels::BinaryOp::sub;
      else if (n.op=="Mul") op=kernels::BinaryOp::mul;
      else if (n.op=="Div") op=kernels::BinaryOp::div;
      int source_index = 0;
      auto source = values.find(n.in[0]);
      auto source_use = use_slot(n.in[0]);
      if ((n.op=="Add" || n.op=="Mul") &&
          (source==values.end() || !source_use || *source_use!=1)) {
        source_index=1;
        source=values.find(n.in[1]);
        source_use=use_slot(n.in[1]);
      }
      if (source!=values.end() && source_use && *source_use==1) {
        const int other_index=1-source_index;
        const auto other_value=values.find(n.in[other_index]);
        const auto other_initializer=impl_->graph.initializers.find(n.in[other_index]);
        const Tensor* other = other_value!=values.end() ? &other_value->second :
            (other_initializer!=impl_->graph.initializers.end() ? &other_initializer->second : nullptr);
        const auto broadcast_repeat = other ? RightBroadcastRepeat(source->second, *other) : 0;
        if (other && (other->data.size()==1 || other->shape==source->second.shape ||
                      broadcast_repeat)) {
          if (other->data.size()==1) {
            kernels::BinaryScalar(source->second.data.data(), source->second.data.data(),
                                  source->second.data.size(), other->data[0], op, false);
          } else {
            if (other->shape==source->second.shape) {
              if (!TryHybridVulkanBinaryInplace(
                      impl_->backend, source->second, *other, op,
                      other_initializer!=impl_->graph.initializers.end())) {
                kernels::BinaryInplace(source->second.data.data(), other->data.data(),
                                       source->second.data.size(), op);
              }
            } else {
              if (!TryHybridVulkanBinaryBroadcastRightInplace(
                      impl_->backend, source->second, *other, broadcast_repeat, op,
                      other_initializer!=impl_->graph.initializers.end())) {
                kernels::BinaryBroadcastRightInplace(source->second.data.data(), other->data.data(),
                                                     source->second.data.size(), broadcast_repeat,
                                                     other->data.size(), op);
              }
            }
          }
          auto node=values.extract(source);
          node.key()=n.out[0];
          values.insert(std::move(node));
          if (!impl_->graph.initializers.contains(n.in[other_index])) {
            if (consume_use(n.in[other_index])) values.erase(n.in[other_index]);
          }
          retire_use(n.in[source_index]);
          continue;
        }
      }
    }
    if (n.op=="Pow" && n.in.size()==2 && n.out.size()==1 &&
        !impl_->graph_outputs.contains(n.out[0])) {
      const auto source=values.find(n.in[0]);
      const auto use=use_slot(n.in[0]);
      const auto exponent_value=values.find(n.in[1]);
      const auto exponent_initializer=impl_->graph.initializers.find(n.in[1]);
      const Tensor* exponent=exponent_value!=values.end()?&exponent_value->second:
          (exponent_initializer!=impl_->graph.initializers.end()?&exponent_initializer->second:nullptr);
      if (source!=values.end() && use && *use==1 && exponent &&
          exponent->data.size()==1 && exponent->data[0]==2.F) {
        const auto start = profile ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
        kernels::Square(source->second.data.data(), source->second.data.data(),
                        source->second.data.size());
        auto node=values.extract(source); node.key()=n.out[0]; values.insert(std::move(node));
        retire_use(n.in[0]);
        if (!impl_->graph.initializers.contains(n.in[1])) {
          if (consume_use(n.in[1])) values.erase(n.in[1]);
        }
        if (profile) {
          const auto profile_key=std::string("ctc:")+(n.name.empty()?n.op:n.op+":"+n.name);
          elapsed_ms[profile_key]+=std::chrono::duration<double,std::milli>(
              std::chrono::steady_clock::now()-start).count();
          ++calls[profile_key];
        }
        continue;
      }
    }
    if ((n.op=="FusedGelu" || n.op=="Relu" || n.op=="FusedHardSwish" ||
         n.op=="FusedScaleShift" || n.op=="HardSigmoid") && n.in.size()==1 && n.out.size()==1 &&
        !impl_->graph_outputs.contains(n.out[0])) {
      const auto source=values.find(n.in[0]);
      const auto use=use_slot(n.in[0]);
      if (source!=values.end() && use && *use==1) {
        auto* data=source->second.data.data(); const auto count=source->second.data.size();
        if (n.op=="FusedGelu") {
          const auto logical=source->second.shape.empty()?count:
              count/std::size_t(source->second.shape[0]);
          kernels::Gelu(data,data,count,logical);
        }
        else if (n.op=="Relu") kernels::Relu(data,data,count);
        else if (n.op=="FusedHardSwish") kernels::HardSwish(data,data,count);
        else if (n.op=="FusedScaleShift") kernels::ScaleShift(data,data,count,Attr(n,"scale").f,Attr(n,"shift").f);
        else kernels::HardSigmoid(data,data,count,n.attr.contains("alpha")?Attr(n,"alpha").f:.2F,
                                  n.attr.contains("beta")?Attr(n,"beta").f:.5F);
        auto node=values.extract(source);
        node.key()=n.out[0];
        values.insert(std::move(node));
        continue;
      }
    }
    // Recognition normally reaches Concat through this CTC executor rather
    // than Run(). Keep the same storage transfer here so crop batches do not
    // retain both the first branch and its assembled output.
    if (n.op == "Concat" && n.in.size() >= 2 && n.out.size() == 1 &&
        !impl_->graph_outputs.contains(n.out[0])) {
      const auto source = values.find(n.in[0]);
      const auto source_use = use_slot(n.in[0]);
      if (source != values.end() && source_use && *source_use == 1) {
        auto& concat_inputs = input_scratch;
        concat_inputs.clear();
        bool dynamic_alias{};
        for (std::size_t input_index = 0; input_index < n.in.size(); ++input_index) {
          const auto value = values.find(n.in[input_index]);
          const auto initializer = impl_->graph.initializers.find(n.in[input_index]);
          const Tensor* input = value != values.end() ? &value->second :
              (initializer != impl_->graph.initializers.end() ? &initializer->second : nullptr);
          if (!input || (input_index > 0 && input == &source->second)) { dynamic_alias = true; break; }
          concat_inputs.push_back(input);
        }
        if (!dynamic_alias && concat_inputs.size() == n.in.size() &&
            ConcatIntoFirst(n, concat_inputs, source->second)) {
          auto node = values.extract(source);
          node.key() = n.out[0];
          values.insert(std::move(node));
          retire_use(n.in[0]);
          for (std::size_t input_index = 1; input_index < n.in.size(); ++input_index) {
            if (impl_->graph.initializers.contains(n.in[input_index])) continue;
            if (consume_use(n.in[input_index])) values.erase(n.in[input_index]);
          }
          continue;
        }
      }
    }
    if ((n.op=="FusedBatchNormGelu" || n.op=="FusedBatchNormSwish" ||
         n.op=="FusedBatchNormHardSwish") && n.in.size()==3 && n.out.size()==1 &&
        !impl_->graph_outputs.contains(n.out[0])) {
      const auto source=values.find(n.in[0]);
      const auto use=use_slot(n.in[0]);
      if (source!=values.end() && use && *use==1) {
        auto& in = input_scratch;
        in.clear();
        for(const auto& name:n.in) {
          const auto value=values.find(name);
          if(value!=values.end()) { in.push_back(&value->second); continue; }
          const auto initializer=impl_->graph.initializers.find(name);
          if(initializer==impl_->graph.initializers.end()) Fail("missing node value "+name);
          in.push_back(&initializer->second);
        }
        const bool immutable_coefficients =
            impl_->graph.initializers.contains(n.in[1]) && impl_->graph.initializers.contains(n.in[2]);
        const bool keep_legacy_swish_output = n.op=="FusedBatchNormSwish" &&
            std::getenv("PPOCR_DISABLE_BN_SWISH_INPLACE") != nullptr;
        if (keep_legacy_swish_output) {
          source->second = BatchNormSwish(n,in);
        } else {
          const bool gpu_finished = n.op=="FusedBatchNormSwish"
              ? TryHybridVulkanChannelAffineSwishInplace(
                    impl_->backend, source->second, *in[1], *in[2], immutable_coefficients)
              : TryHybridVulkanChannelAffineInplace(
                    impl_->backend, source->second, *in[1], *in[2], immutable_coefficients);
          if (!gpu_finished) {
            if (n.op=="FusedBatchNormGelu") BatchNormGeluInplace(n,in,source->second);
            else if (n.op=="FusedBatchNormHardSwish") {
              const auto channels = std::size_t(source->second.shape[1]);
              const auto spatial = source->second.data.size() /
                  (std::size_t(source->second.shape[0]) * channels);
              kernels::BatchNormAffine(source->second.data.data(), source->second.data.data(),
                                       in[1]->data.data(), in[2]->data.data(),
                                       int(source->second.shape[0]), int(channels), spatial);
              kernels::HardSwish(source->second.data.data(), source->second.data.data(),
                                 source->second.data.size());
            } else BatchNormSwishInplace(n,in,source->second);
          } else if (n.op=="FusedBatchNormGelu") {
            const auto count=source->second.data.size();
            const auto logical=source->second.shape.empty()?count:
                count/std::size_t(source->second.shape[0]);
            kernels::Gelu(source->second.data.data(), source->second.data.data(), count, logical);
          } else if (n.op=="FusedBatchNormHardSwish") {
            kernels::HardSwish(source->second.data.data(), source->second.data.data(),
                               source->second.data.size());
          }
        }
        auto node=values.extract(source);
        node.key()=n.out[0];
        values.insert(std::move(node));
        continue;
      }
    }
    // Ordinary BatchNorm is an affine NCHW transform.  Reuse its activation
    // in the CTC production executor whenever the graph proves final use.
    if (std::getenv("PPOCR_DISABLE_BATCHNORM_INPLACE") == nullptr &&
        n.op == "BatchNormalization" && n.in.size() == 5 && n.out.size() == 1 &&
        !impl_->graph_outputs.contains(n.out[0])) {
      const auto source = values.find(n.in[0]);
      const auto use = use_slot(n.in[0]);
      if (source != values.end() && use && *use == 1) {
        auto& in = input_scratch;
        in.clear();
        for (const auto& name : n.in) {
          const auto value = values.find(name);
          if (value != values.end()) { in.push_back(&value->second); continue; }
          const auto initializer = impl_->graph.initializers.find(name);
          if (initializer == impl_->graph.initializers.end()) Fail("missing node value " + name);
          in.push_back(&initializer->second);
        }
        const auto start = profile ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
        BatchNormInplace(n, in, source->second);
        auto node = values.extract(source);
        node.key() = n.out[0];
        values.insert(std::move(node));
        retire_use(n.in[0]);
        for (std::size_t input_index = 1; input_index < n.in.size(); ++input_index) {
          const auto& name = n.in[input_index];
          if (impl_->graph.initializers.contains(name)) continue;
          if (consume_use(name)) {
            values.erase(name);
          }
        }
        if (profile) {
          const auto profile_key = std::string("ctc:") +
              (n.name.empty() ? n.op : n.op + ":" + n.name);
          elapsed_ms[profile_key] += std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - start).count();
          ++calls[profile_key];
        }
        continue;
      }
    }
    // RunCtcTop1 is the recognizer's real execution path. Keep its lifetime
    // planner in sync with Run(): small/medium inverted-residual tails have
    // already been folded to affine BatchNorm -> exact Swish -> shortcut Add.
    // Reusing the dying BatchNorm input avoids both the intermediate Swish
    // feature map and a second output allocation for every recognition batch.
    if (n.op=="FusedBatchNormSwishAdd" && n.in.size()==4 && n.out.size()==1 &&
        !impl_->graph_outputs.contains(n.out[0])) {
      const auto source=values.find(n.in[0]);
      const auto use=use_slot(n.in[0]);
      const auto residual=values.find(n.in[3]);
      if (source!=values.end() && use && *use==1 &&
          residual!=values.end() && source->second.shape==residual->second.shape) {
        auto& in = input_scratch;
        in.clear();
        for (const auto& name : n.in) {
          const auto value=values.find(name);
          if (value!=values.end()) { in.push_back(&value->second); continue; }
          const auto initializer=impl_->graph.initializers.find(name);
          if (initializer==impl_->graph.initializers.end()) Fail("missing node value "+name);
          in.push_back(&initializer->second);
        }
        const auto start = profile ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
        const bool immutable_coefficients=impl_->graph.initializers.contains(n.in[1]) &&
            impl_->graph.initializers.contains(n.in[2]);
        if (!TryHybridVulkanChannelAffineSwishAddInplace(
                impl_->backend, source->second, *in[1], *in[2], *in[3],
                immutable_coefficients)) {
          BatchNormSwishAddInplace(in, source->second);
        }
        auto node=values.extract(source);
        node.key()=n.out[0];
        values.insert(std::move(node));
        retire_use(n.in[0]);
        if (consume_use(n.in[3])) {
          values.erase(n.in[3]);
        }
        if (profile) {
          const auto profile_key=std::string("ctc:")+(n.name.empty()?n.op:n.op+":"+n.name);
          elapsed_ms[profile_key]+=std::chrono::duration<double,std::milli>(
              std::chrono::steady_clock::now()-start).count();
          ++calls[profile_key];
        }
        continue;
      }
    }
    // Recognition uses this terminal-CTC executor directly, so retain the
    // same metadata-only storage transfer as Run(). Without it the large
    // transformer [N,T,C] activations are copied at each reshape/squeeze
    // boundary even though their float order is unchanged.
    if ((n.op == "Reshape" || n.op == "Squeeze" || n.op == "Unsqueeze") &&
        !n.in.empty() && n.out.size() == 1 && !impl_->graph_outputs.contains(n.out[0])) {
      const auto source = values.find(n.in[0]);
      const auto use = use_slot(n.in[0]);
      if (source != values.end() && use && *use == 1) {
        std::vector<std::int64_t> shape;
        if (n.op == "Reshape") {
          if (n.in.size() < 2) Fail("missing reshape spec");
          const auto dynamic_spec = values.find(n.in[1]);
          const auto const_spec = impl_->graph.initializers.find(n.in[1]);
          const Tensor* spec = dynamic_spec != values.end() ? &dynamic_spec->second :
              (const_spec != impl_->graph.initializers.end() ? &const_spec->second : nullptr);
          if (!spec) Fail("missing reshape spec");
          std::int64_t known = 1;
          int inferred = -1;
          for (std::size_t axis = 0; axis < spec->data.size(); ++axis) {
            std::int64_t dimension = static_cast<std::int64_t>(spec->data[axis]);
            if (dimension == -1) {
              if (inferred >= 0) Fail("multiple reshape inferred dimensions");
              inferred = static_cast<int>(axis);
              shape.push_back(1);
            } else {
              if (dimension == 0) dimension = source->second.shape[axis];
              shape.push_back(dimension);
              known *= dimension;
            }
          }
          if (inferred >= 0) shape[static_cast<std::size_t>(inferred)] =
              static_cast<std::int64_t>(source->second.data.size() / known);
          if (Elements(shape) != source->second.data.size()) Fail("reshape mismatch");
        } else {
          std::vector<std::int64_t> axes = AttrInts(n, "axes", {});
          if (axes.empty() && n.in.size() > 1) {
            const auto dynamic_axes = values.find(n.in[1]);
            const auto const_axes = impl_->graph.initializers.find(n.in[1]);
            const Tensor* axes_tensor = dynamic_axes != values.end() ? &dynamic_axes->second :
                (const_axes != impl_->graph.initializers.end() ? &const_axes->second : nullptr);
            if (!axes_tensor) Fail("missing squeeze axes");
            for (const float axis : axes_tensor->data) axes.push_back(static_cast<std::int64_t>(axis));
          }
          if (n.op == "Unsqueeze") {
            const int rank = static_cast<int>(source->second.shape.size() + axes.size());
            std::vector<int> positions;
            for (const auto axis : axes) {
              int position = static_cast<int>(axis);
              if (position < 0) position += rank;
              if (position < 0 || position >= rank) Fail("unsqueeze axis");
              positions.push_back(position);
            }
            std::sort(positions.begin(), positions.end());
            shape = source->second.shape;
            for (const int position : positions) shape.insert(shape.begin() + position, 1);
          } else {
            std::vector<bool> drop(source->second.shape.size());
            if (axes.empty()) {
              for (std::size_t axis = 0; axis < source->second.shape.size(); ++axis) {
                drop[axis] = source->second.shape[axis] == 1;
              }
            } else {
              for (const auto axis : axes) {
                const auto position = Axis(axis, static_cast<int>(source->second.shape.size()));
                if (source->second.shape[position] != 1) Fail("squeeze non-unit axis");
                drop[position] = true;
              }
            }
            for (std::size_t axis = 0; axis < source->second.shape.size(); ++axis) {
              if (!drop[axis]) shape.push_back(source->second.shape[axis]);
            }
          }
        }
        source->second.shape = std::move(shape);
        auto node = values.extract(source);
        node.key() = n.out[0];
        values.insert(std::move(node));
        retire_use(n.in[0]);
        for (std::size_t input_index = 1; input_index < n.in.size(); ++input_index) {
          const auto& name = n.in[input_index];
          if (name.empty() || impl_->graph.initializers.contains(name)) continue;
          if (consume_use(name)) values.erase(name);
        }
        continue;
      }
    }
    auto& in = input_scratch;
    in.clear();
    for(const auto& name:n.in) {
      if(name.empty()){in.push_back(nullptr);continue;}
      const auto value=values.find(name);
      if(value!=values.end()){in.push_back(&value->second);continue;}
      const auto initializer=impl_->graph.initializers.find(name);
      if(initializer==impl_->graph.initializers.end())Fail("missing node value "+name);
      in.push_back(&initializer->second);
    }
    const auto start = profile ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
    const bool immutable_parameters = n.in.size() >= 3 &&
        impl_->graph.initializers.contains(n.in[1]) &&
        impl_->graph.initializers.contains(n.in[2]);
    // See the general executor above.  The terminal-CTC path shares the same
    // fused Q/K/V graph nodes and must preserve their three outputs too.
    if (n.op == "FusedQkvSplit") {
      auto outputs = QkvSplit(n, *in[0]);
      if (outputs.size() != n.out.size()) Fail("FusedQkvSplit output count");
      if (profile) {
        const auto profile_key = std::string("ctc:") +
            (n.name.empty() ? n.op : n.op + ":" + n.name);
        elapsed_ms[profile_key] += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        ++calls[profile_key];
      }
      for (std::size_t output_index = 0; output_index < n.out.size(); ++output_index) {
        values.insert_or_assign(n.out[output_index], std::move(outputs[output_index]));
      }
      for (const auto& name : n.in) {
        if (name.empty() || impl_->graph.initializers.contains(name)) continue;
        if (consume_use(name)) values.erase(name);
      }
      continue;
    }
    auto value=Execute(n,in,impl_->graph.opset,impl_->backend,immutable_parameters);
    if (profile) {
      const auto profile_key = std::string("ctc:") + (n.name.empty() ? n.op : n.op + ":" + n.name);
      elapsed_ms[profile_key] += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count();
      ++calls[profile_key];
    }
    if(n.out.size()!=1)Fail("multi-output node unsupported");
    values.insert_or_assign(n.out[0],std::move(value));
    for(const auto& name:n.in) {
      if(name.empty() || impl_->graph.initializers.contains(name)) continue;
      if (consume_use(name)) values.erase(name);
    }
  }
  const auto& logits_name=impl_->graph.nodes.back().in[0];
  const auto logits=values.find(logits_name);
  if(logits==values.end() || logits->second.shape.size()!=3) Fail("terminal logits missing");
  const auto batches=int(logits->second.shape[0]), steps=int(logits->second.shape[1]);
  const auto vocab=int(logits->second.shape[2]);
  CtcTop1Output result{batches,steps,std::vector<int>(std::size_t(batches)*steps),
                       std::vector<float>(std::size_t(batches)*steps)};
  const auto ctc_start = profile ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
  kernels::CtcTop1(result.indices.data(),result.probabilities.data(),logits->second.data.data(),
                   result.indices.size(),steps,vocab);
  if (profile) {
    elapsed_ms["ctc:terminal_top1"] += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - ctc_start).count();
    ++calls["ctc:terminal_top1"];
    std::vector<std::pair<std::string, double>> entries(elapsed_ms.begin(), elapsed_ms.end());
    std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.second > rhs.second;
    });
    std::cerr << "ppocr CTC operator profile (ms):\n";
    for (const auto& [op, ms] : entries) {
      std::cerr << "  " << op << " x" << calls[op] << ": " << ms << '\n';
    }
  }
  return result;
}

}  // namespace ppocr::detail
