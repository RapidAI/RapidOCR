// Native Metal execution for the PP-OCRv6 GPU graph on Apple Silicon.
//
// The GPU graph, its operator fusions, the arena and the replay logic live in
// vulkan_backend.cpp and talk to the device only through a small compute
// subset of the Vulkan entry points, resolved by name at runtime. This file
// implements that subset directly on Metal, so the same graph runs without
// MoltenVK or a Vulkan loader:
//
// - every allocation is one MTLBuffer in shared storage (Apple unified
//   memory): "mapping" returns its contents pointer, so weights stay resident
//   and uploads/downloads are plain memcpy without staging copies;
// - a vkQueueSubmit becomes one MTLCommandBuffer with one compute encoder;
// - kernels are the same GLSL compute shaders, translated to MSL by
//   SPIRV-Cross at build time and compiled with fast math off. Each uber-shader
//   mode gets its own specialized pipeline (function constant), so a dispatch
//   only carries the code and threadgroup memory of the operator it runs.
//
// Only what vulkan_backend.cpp calls is implemented. Anything else returns
// VK_ERROR_FEATURE_NOT_PRESENT, which makes the runtime keep the CPU path.

#include <vulkan/vulkan.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "vulkan_binary_spv.hpp"
#include "vulkan_depthwise_tail_spv.hpp"
#include "vulkan_pointwise_tail_spv.hpp"
#include "metal_binary_msl.hpp"
#include "metal_depthwise_tail_msl.hpp"
#include "metal_pointwise_tail_msl.hpp"
#include "metal_native_kernels.hpp"

namespace {

constexpr std::uint32_t kPushBufferIndex = 8;
constexpr std::uint32_t kPushBytes = 128;
constexpr std::uint32_t kBindings = 5;

std::atomic<int> g_metal_errors{0};

bool Verbose() {
  static const bool verbose = std::getenv("PPOCR_METAL_VERBOSE") != nullptr;
  return verbose;
}

struct MtlInstance { int unused{}; };
struct MtlPhysicalDevice { id<MTLDevice> device{nil}; };
struct MtlQueue;
struct MtlDevice {
  id<MTLDevice> device{nil};
  id<MTLCommandQueue> queue{nil};
  MtlQueue* queue_handle{};
};
struct MtlQueue { MtlDevice* device{}; id<MTLCommandBuffer> last{nil}; };
struct MtlMemory { id<MTLBuffer> buffer{nil}; VkDeviceSize size{}; };
struct MtlBuffer { VkDeviceSize size{}; MtlMemory* memory{}; VkDeviceSize offset{}; };
struct MtlShaderModule { int kind{-1}; std::array<std::uint32_t, 3> local{1, 1, 1}; };

// One shader kind (uber-shader or a tail shader) with its MSL library and
// lazily specialized pipelines keyed by the push-constant mode word.
struct MtlProgram {
  id<MTLLibrary> library{nil};
  id<MTLComputePipelineState> generic{nil};
  std::unordered_map<std::uint32_t, id<MTLComputePipelineState>> by_mode;
  bool specializable{};
  int kind{-1};
};

struct MtlPipeline {
  MtlProgram* program{};
  std::array<std::uint32_t, 3> local{1, 1, 1};
};

struct MtlDescriptorSet {
  std::array<MtlBuffer*, kBindings> buffers{};
  std::array<VkDeviceSize, kBindings> offsets{};
};
struct MtlDescriptorPool { std::vector<std::unique_ptr<MtlDescriptorSet>> sets; std::size_t used{}; };

enum class CmdType : std::uint8_t { dispatch, barrier, copy };
struct Cmd {
  CmdType type{};
  MtlPipeline* pipeline{};
  std::array<MtlBuffer*, kBindings> buffers{};
  std::array<VkDeviceSize, kBindings> offsets{};
  std::array<std::uint8_t, kPushBytes> push{};
  std::uint32_t push_size{};
  std::uint32_t x{}, y{}, z{};
  MtlBuffer* src{};
  MtlBuffer* dst{};
  std::vector<VkBufferCopy> regions;
};
struct MtlCommandBuffer {
  std::vector<Cmd> commands;
  MtlPipeline* pipeline{};
  MtlDescriptorSet bound{};
  std::array<std::uint8_t, kPushBytes> push{};
  std::uint32_t push_size{};
};
struct MtlCommandPool { std::vector<std::unique_ptr<MtlCommandBuffer>> buffers; };
struct MtlFence { id<MTLCommandBuffer> pending{nil}; bool signaled{}; };
struct MtlDummy { int unused{}; };

template <class T, class H> T* As(H handle) { return reinterpret_cast<T*>(handle); }
template <class H, class T> H Handle(T* object) { return reinterpret_cast<H>(object); }

std::mutex g_program_mutex;
std::array<MtlProgram, 3> g_programs;
std::array<bool, 3> g_program_attempted{};

const std::vector<std::uint8_t>* MslSource(int kind) {
  static const std::vector<std::uint8_t> sources[3] = {
      {ppocr::detail::kMetalBinaryMsl.begin(), ppocr::detail::kMetalBinaryMsl.end()},
      {ppocr::detail::kMetalDepthwiseTailMsl.begin(), ppocr::detail::kMetalDepthwiseTailMsl.end()},
      {ppocr::detail::kMetalPointwiseTailMsl.begin(), ppocr::detail::kMetalPointwiseTailMsl.end()}};
  return kind >= 0 && kind < 3 ? &sources[kind] : nullptr;
}

// The uber-shader branches on params.mode. A function constant replaces that
// load when a pipeline is specialized, so the Metal compiler drops every
// other operator (and its threadgroup arrays) from that pipeline.
std::string PrepareSource(int kind, bool* specializable) {
  const auto* bytes = MslSource(kind);
  std::string source(bytes->begin(), bytes->end());
  *specializable = false;
  if (kind != 0 || std::getenv("PPOCR_METAL_DISABLE_SPECIALIZATION") != nullptr) return source;
  const std::string needle = "params.mode";
  std::string out;
  out.reserve(source.size() + 4096);
  std::size_t position = 0;
  std::size_t replaced = 0;
  while (true) {
    const std::size_t next = source.find(needle, position);
    if (next == std::string::npos) { out.append(source, position, std::string::npos); break; }
    const bool word_end = next + needle.size() >= source.size() ||
        !(std::isalnum(static_cast<unsigned char>(source[next + needle.size()])) ||
          source[next + needle.size()] == '_');
    out.append(source, position, next - position);
    out.append(word_end ? "ppocr_mode(params)" : needle);
    if (word_end) ++replaced;
    position = next + needle.size();
  }
  if (replaced == 0) return source;
  const std::size_t kernel = out.find("kernel void main0");
  if (kernel == std::string::npos) return source;
  // Params is declared before the kernel by SPIRV-Cross.
  static const char* helper =
      "constant uint ppocr_fixed_mode [[function_constant(0)]];\n"
      "constant bool ppocr_has_fixed_mode = is_function_constant_defined(ppocr_fixed_mode);\n"
      "static inline uint ppocr_mode(constant Params& p) {\n"
      // Only the low 16 bits select the code path; the high bits carry per-call
      // data (content width, pads, flags) and stay dynamic.
      "  return ppocr_has_fixed_mode ? (ppocr_fixed_mode | (p.mode & 0xffff0000u)) : p.mode;\n"
      "}\n\n";
  out.insert(kernel, helper);
  *specializable = true;
  return out;
}

MtlProgram* ProgramFor(id<MTLDevice> device, int kind) {
  std::lock_guard lock(g_program_mutex);
  auto& program = g_programs[kind];
  if (g_program_attempted[kind]) return program.library ? &program : nullptr;
  g_program_attempted[kind] = true;
  program.kind = kind;
  @autoreleasepool {
    bool specializable = false;
    const std::string source = PrepareSource(kind, &specializable);
    MTLCompileOptions* options = [MTLCompileOptions new];
    // Fast math may reassociate or flush denormals; recognition text must
    // match the CPU graph, so compile with IEEE semantics.
    // mathMode is new in the macOS 15 SDK; older SDKs (macos-14 runners)
    // only have fastMathEnabled.
#if defined(__MAC_15_0) && __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_15_0
    if (@available(macOS 15.0, *)) {
      options.mathMode = MTLMathModeSafe;
      options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    } else
#endif
    {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
      options.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
    options.languageVersion = MTLLanguageVersion2_4;
    NSError* error = nil;
    NSString* text = [[NSString alloc] initWithBytes:source.data() length:source.size()
                                            encoding:NSUTF8StringEncoding];
    id<MTLLibrary> library = [device newLibraryWithSource:text options:options error:&error];
    if (!library) {
      std::fprintf(stderr, "ppocr: Metal shader %d did not compile: %s\n", kind,
                   error ? error.localizedDescription.UTF8String : "unknown error");
      g_metal_errors.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
    program.library = library;
    program.specializable = specializable;
  }
  return &program;
}

id<MTLComputePipelineState> MakePipeline(id<MTLDevice> device, MtlProgram* program,
                                         const std::uint32_t* mode) {
  @autoreleasepool {
    NSError* error = nil;
    id<MTLFunction> function = nil;
    if (program->specializable) {
      // A library with function constants must be specialized even when
      // the constant stays undefined (the generic pipeline reads the mode
      // from push constants).
      MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
      if (mode) {
        const std::uint32_t value = *mode;
        [values setConstantValue:&value type:MTLDataTypeUInt atIndex:0];
      }
      function = [program->library newFunctionWithName:@"main0" constantValues:values error:&error];
    } else {
      function = [program->library newFunctionWithName:@"main0"];
    }
    if (!function) {
      std::fprintf(stderr, "ppocr: Metal function lookup failed: %s\n",
                   error ? error.localizedDescription.UTF8String : "main0 missing");
      g_metal_errors.fetch_add(1, std::memory_order_relaxed);
      return nil;
    }
    id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function error:&error];
    if (!state) {
      std::fprintf(stderr, "ppocr: Metal pipeline failed: %s\n",
                   error ? error.localizedDescription.UTF8String : "unknown error");
      g_metal_errors.fetch_add(1, std::memory_order_relaxed);
    }
    return state;
  }
}

id<MTLComputePipelineState> PipelineForDispatch(id<MTLDevice> device, MtlPipeline* pipeline,
                                                const Cmd& command) {
  MtlProgram* program = pipeline->program;
  if (program->specializable && command.push_size >= 44) {
    std::uint32_t mode{};
    std::memcpy(&mode, command.push.data() + 40, sizeof(mode));
    mode &= 0xffffu;  // high bits are per-call data, see PrepareSource
    std::lock_guard lock(g_program_mutex);
    auto found = program->by_mode.find(mode);
    if (found != program->by_mode.end()) return found->second;
    id<MTLComputePipelineState> state = MakePipeline(device, program, &mode);
    program->by_mode.emplace(mode, state);
    if (Verbose()) std::fprintf(stderr, "ppocr: Metal pipeline for mode %u: %s\n", mode, state ? "ok" : "failed");
    return state;
  }
  std::lock_guard lock(g_program_mutex);
  if (!program->generic) program->generic = MakePipeline(device, program, nullptr);
  return program->generic;
}

// ---- SPIR-V identification -------------------------------------------------

int ShaderKind(const VkShaderModuleCreateInfo* info) {
  const auto matches = [&](const auto& spv) {
    return info->codeSize == spv.size() && std::memcmp(info->pCode, spv.data(), spv.size()) == 0;
  };
  if (matches(ppocr::detail::kVulkanBinarySpv)) return 0;
  if (matches(ppocr::detail::kVulkanDepthwiseTailSpv)) return 1;
  if (matches(ppocr::detail::kVulkanPointwiseTailSpv)) return 2;
  return -1;
}

std::array<std::uint32_t, 3> LocalSize(const VkShaderModuleCreateInfo* info) {
  std::array<std::uint32_t, 3> local{1, 1, 1};
  const std::uint32_t* words = info->pCode;
  const std::size_t count = info->codeSize / 4;
  for (std::size_t i = 5; i < count;) {
    const std::uint32_t word = words[i];
    const std::uint32_t length = word >> 16;
    const std::uint32_t opcode = word & 0xffffu;
    if (length == 0) break;
    // OpExecutionMode %entry LocalSize x y z
    if (opcode == 16 && length >= 6 && words[i + 2] == 17) {
      local = {words[i + 3], words[i + 4], words[i + 5]};
    }
    i += length;
  }
  return local;
}

// ---- Instance / physical device ------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL CreateInstance(const VkInstanceCreateInfo*, const VkAllocationCallbacks*,
                                              VkInstance* instance) {
  *instance = Handle<VkInstance>(new MtlInstance{});
  return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL DestroyInstance(VkInstance instance, const VkAllocationCallbacks*) {
  delete As<MtlInstance>(instance);
}
VKAPI_ATTR VkResult VKAPI_CALL EnumerateInstanceLayerProperties(std::uint32_t* count, VkLayerProperties*) {
  *count = 0;
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL EnumerateInstanceExtensionProperties(const char*, std::uint32_t* count,
                                                                    VkExtensionProperties*) {
  *count = 0;
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL EnumerateDeviceExtensionProperties(VkPhysicalDevice, const char*,
                                                                  std::uint32_t* count,
                                                                  VkExtensionProperties*) {
  *count = 0;
  return VK_SUCCESS;
}

MtlPhysicalDevice* SystemPhysicalDevice() {
  static MtlPhysicalDevice* physical = [] {
    auto* result = new MtlPhysicalDevice{};
    result->device = MTLCreateSystemDefaultDevice();
    return result;
  }();
  return physical->device ? physical : nullptr;
}

VKAPI_ATTR VkResult VKAPI_CALL EnumeratePhysicalDevices(VkInstance, std::uint32_t* count,
                                                        VkPhysicalDevice* devices) {
  MtlPhysicalDevice* physical = SystemPhysicalDevice();
  if (!physical) { *count = 0; return VK_SUCCESS; }
  if (!devices) { *count = 1; return VK_SUCCESS; }
  if (*count < 1) return VK_INCOMPLETE;
  devices[0] = Handle<VkPhysicalDevice>(physical);
  *count = 1;
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceProperties(VkPhysicalDevice physical_device,
                                                       VkPhysicalDeviceProperties* properties) {
  *properties = {};
  id<MTLDevice> device = As<MtlPhysicalDevice>(physical_device)->device;
  properties->apiVersion = VK_API_VERSION_1_2;
  properties->vendorID = 0x106B;
  properties->deviceType = device.hasUnifiedMemory ? VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
                                                   : VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
  std::snprintf(properties->deviceName, sizeof(properties->deviceName), "%s (Metal)",
                device.name.UTF8String);
  auto& limits = properties->limits;
  limits.maxComputeSharedMemorySize = static_cast<std::uint32_t>(device.maxThreadgroupMemoryLength);
  limits.maxComputeWorkGroupCount[0] = limits.maxComputeWorkGroupCount[1] =
      limits.maxComputeWorkGroupCount[2] = 65535;
  const MTLSize threads = device.maxThreadsPerThreadgroup;
  limits.maxComputeWorkGroupSize[0] = static_cast<std::uint32_t>(threads.width);
  limits.maxComputeWorkGroupSize[1] = static_cast<std::uint32_t>(threads.height);
  limits.maxComputeWorkGroupSize[2] = static_cast<std::uint32_t>(threads.depth);
  limits.maxComputeWorkGroupInvocations = static_cast<std::uint32_t>(threads.width);
  limits.maxStorageBufferRange = static_cast<std::uint32_t>(
      std::min<NSUInteger>(device.maxBufferLength, UINT32_MAX));
  limits.minStorageBufferOffsetAlignment = 16;
  limits.nonCoherentAtomSize = 1;
  limits.maxPushConstantsSize = kPushBytes;
  limits.maxBoundDescriptorSets = 4;
  limits.timestampPeriod = 1.F;
  limits.timestampComputeAndGraphics = VK_FALSE;
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice, std::uint32_t* count,
                                                                  VkQueueFamilyProperties* properties) {
  if (!properties) { *count = 1; return; }
  if (*count < 1) return;
  properties[0] = {};
  properties[0].queueFlags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
  properties[0].queueCount = 1;
  properties[0].timestampValidBits = 0;  // no per-dispatch timestamps
  properties[0].minImageTransferGranularity = {1, 1, 1};
  *count = 1;
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceMemoryProperties(VkPhysicalDevice physical_device,
                                                             VkPhysicalDeviceMemoryProperties* properties) {
  *properties = {};
  id<MTLDevice> device = As<MtlPhysicalDevice>(physical_device)->device;
  properties->memoryTypeCount = 1;
  properties->memoryTypes[0].propertyFlags =
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  properties->memoryTypes[0].heapIndex = 0;
  properties->memoryHeapCount = 1;
  properties->memoryHeaps[0].size = device.recommendedMaxWorkingSetSize;
  properties->memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
}

// ---- Device ----------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL CreateDevice(VkPhysicalDevice physical_device, const VkDeviceCreateInfo*,
                                            const VkAllocationCallbacks*, VkDevice* device_out) {
  auto* device = new MtlDevice{};
  device->device = As<MtlPhysicalDevice>(physical_device)->device;
  device->queue = [device->device newCommandQueue];
  if (!device->queue) { delete device; return VK_ERROR_INITIALIZATION_FAILED; }
  device->queue_handle = new MtlQueue{device, nil};
  *device_out = Handle<VkDevice>(device);
  return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL DestroyDevice(VkDevice device_handle, const VkAllocationCallbacks*) {
  auto* device = As<MtlDevice>(device_handle);
  if (!device) return;
  if (device->queue_handle && device->queue_handle->last) [device->queue_handle->last waitUntilCompleted];
  delete device->queue_handle;
  delete device;
}
VKAPI_ATTR void VKAPI_CALL GetDeviceQueue(VkDevice device, std::uint32_t, std::uint32_t, VkQueue* queue) {
  *queue = Handle<VkQueue>(As<MtlDevice>(device)->queue_handle);
}

// ---- Memory / buffers ------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL CreateBuffer(VkDevice, const VkBufferCreateInfo* info,
                                            const VkAllocationCallbacks*, VkBuffer* buffer) {
  *buffer = Handle<VkBuffer>(new MtlBuffer{info->size, nullptr, 0});
  return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL DestroyBuffer(VkDevice, VkBuffer buffer, const VkAllocationCallbacks*) {
  delete As<MtlBuffer>(buffer);
}
VKAPI_ATTR void VKAPI_CALL GetBufferMemoryRequirements(VkDevice, VkBuffer buffer,
                                                       VkMemoryRequirements* requirements) {
  const VkDeviceSize size = std::max<VkDeviceSize>(As<MtlBuffer>(buffer)->size, 16);
  requirements->size = (size + 255) & ~VkDeviceSize(255);
  requirements->alignment = 256;
  requirements->memoryTypeBits = 1;
}
VKAPI_ATTR VkResult VKAPI_CALL AllocateMemory(VkDevice device, const VkMemoryAllocateInfo* info,
                                              const VkAllocationCallbacks*, VkDeviceMemory* memory) {
  if (info->memoryTypeIndex != 0) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
  id<MTLBuffer> buffer = [As<MtlDevice>(device)->device
      newBufferWithLength:std::max<VkDeviceSize>(info->allocationSize, 16)
                  options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache];
  if (!buffer) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
  *memory = Handle<VkDeviceMemory>(new MtlMemory{buffer, info->allocationSize});
  return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL FreeMemory(VkDevice, VkDeviceMemory memory, const VkAllocationCallbacks*) {
  delete As<MtlMemory>(memory);
}
VKAPI_ATTR VkResult VKAPI_CALL BindBufferMemory(VkDevice, VkBuffer buffer, VkDeviceMemory memory,
                                                VkDeviceSize offset) {
  As<MtlBuffer>(buffer)->memory = As<MtlMemory>(memory);
  As<MtlBuffer>(buffer)->offset = offset;
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL MapMemory(VkDevice, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize,
                                         VkMemoryMapFlags, void** data) {
  *data = static_cast<std::uint8_t*>(As<MtlMemory>(memory)->buffer.contents) + offset;
  return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL UnmapMemory(VkDevice, VkDeviceMemory) {}
VKAPI_ATTR VkResult VKAPI_CALL FlushMappedMemoryRanges(VkDevice, std::uint32_t, const VkMappedMemoryRange*) {
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL InvalidateMappedMemoryRanges(VkDevice, std::uint32_t,
                                                            const VkMappedMemoryRange*) {
  return VK_SUCCESS;
}

// ---- Layouts, shaders, pipelines -----------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorSetLayout(VkDevice, const VkDescriptorSetLayoutCreateInfo*,
                                                         const VkAllocationCallbacks*,
                                                         VkDescriptorSetLayout* layout) {
  *layout = Handle<VkDescriptorSetLayout>(new MtlDummy{});
  return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL DestroyDescriptorSetLayout(VkDevice, VkDescriptorSetLayout layout,
                                                      const VkAllocationCallbacks*) {
  delete As<MtlDummy>(layout);
}
VKAPI_ATTR VkResult VKAPI_CALL CreatePipelineLayout(VkDevice, const VkPipelineLayoutCreateInfo* info,
                                                    const VkAllocationCallbacks*, VkPipelineLayout* layout) {
  for (std::uint32_t i = 0; i < info->pushConstantRangeCount; ++i) {
    const auto& range = info->pPushConstantRanges[i];
    if (range.offset + range.size > kPushBytes) return VK_ERROR_FEATURE_NOT_PRESENT;
  }
  *layout = Handle<VkPipelineLayout>(new MtlDummy{});
  return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL DestroyPipelineLayout(VkDevice, VkPipelineLayout layout,
                                                 const VkAllocationCallbacks*) {
  delete As<MtlDummy>(layout);
}
VKAPI_ATTR VkResult VKAPI_CALL CreateShaderModule(VkDevice, const VkShaderModuleCreateInfo* info,
                                                  const VkAllocationCallbacks*, VkShaderModule* module) {
  const int kind = ShaderKind(info);
  if (kind < 0) return VK_ERROR_FEATURE_NOT_PRESENT;
  *module = Handle<VkShaderModule>(new MtlShaderModule{kind, LocalSize(info)});
  return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL DestroyShaderModule(VkDevice, VkShaderModule module, const VkAllocationCallbacks*) {
  delete As<MtlShaderModule>(module);
}
VKAPI_ATTR VkResult VKAPI_CALL CreateComputePipelines(VkDevice device_handle, VkPipelineCache,
                                                      std::uint32_t count,
                                                      const VkComputePipelineCreateInfo* infos,
                                                      const VkAllocationCallbacks*, VkPipeline* pipelines) {
  auto* device = As<MtlDevice>(device_handle);
  for (std::uint32_t i = 0; i < count; ++i) {
    auto* module = As<MtlShaderModule>(infos[i].stage.module);
    MtlProgram* program = module ? ProgramFor(device->device, module->kind) : nullptr;
    if (!program) return VK_ERROR_INITIALIZATION_FAILED;
    // Compile the generic pipeline now so an unusable device is rejected at
    // initialization (the runtime then stays on CPU) rather than mid-graph.
    {
      std::lock_guard lock(g_program_mutex);
      if (!program->generic) program->generic = MakePipeline(device->device, program, nullptr);
      if (!program->generic) return VK_ERROR_INITIALIZATION_FAILED;
      const auto threads = module->local[0] * module->local[1] * module->local[2];
      if (program->generic.maxTotalThreadsPerThreadgroup < threads) {
        std::fprintf(stderr, "ppocr: Metal pipeline allows %lu threads, shader needs %u\n",
                     static_cast<unsigned long>(program->generic.maxTotalThreadsPerThreadgroup), threads);
        return VK_ERROR_INITIALIZATION_FAILED;
      }
    }
    pipelines[i] = Handle<VkPipeline>(new MtlPipeline{program, module->local});
  }
  return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL DestroyPipeline(VkDevice, VkPipeline pipeline, const VkAllocationCallbacks*) {
  delete As<MtlPipeline>(pipeline);
}

// ---- Descriptors -----------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorPool(VkDevice, const VkDescriptorPoolCreateInfo*,
                                                    const VkAllocationCallbacks*, VkDescriptorPool* pool) {
  *pool = Handle<VkDescriptorPool>(new MtlDescriptorPool{});
  return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL DestroyDescriptorPool(VkDevice, VkDescriptorPool pool, const VkAllocationCallbacks*) {
  delete As<MtlDescriptorPool>(pool);
}
VKAPI_ATTR VkResult VKAPI_CALL ResetDescriptorPool(VkDevice, VkDescriptorPool pool, VkDescriptorPoolResetFlags) {
  As<MtlDescriptorPool>(pool)->used = 0;
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL AllocateDescriptorSets(VkDevice, const VkDescriptorSetAllocateInfo* info,
                                                      VkDescriptorSet* sets) {
  auto* pool = As<MtlDescriptorPool>(info->descriptorPool);
  for (std::uint32_t i = 0; i < info->descriptorSetCount; ++i) {
    if (pool->used == pool->sets.size()) pool->sets.push_back(std::make_unique<MtlDescriptorSet>());
    MtlDescriptorSet* set = pool->sets[pool->used++].get();
    *set = {};
    sets[i] = Handle<VkDescriptorSet>(set);
  }
  return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL UpdateDescriptorSets(VkDevice, std::uint32_t write_count,
                                                const VkWriteDescriptorSet* writes, std::uint32_t copy_count,
                                                const VkCopyDescriptorSet* copies) {
  for (std::uint32_t i = 0; i < write_count; ++i) {
    auto* set = As<MtlDescriptorSet>(writes[i].dstSet);
    for (std::uint32_t j = 0; j < writes[i].descriptorCount; ++j) {
      const std::uint32_t binding = writes[i].dstBinding + writes[i].dstArrayElement + j;
      if (binding >= kBindings || !writes[i].pBufferInfo) continue;
      set->buffers[binding] = As<MtlBuffer>(writes[i].pBufferInfo[j].buffer);
      set->offsets[binding] = writes[i].pBufferInfo[j].offset;
    }
  }
  for (std::uint32_t i = 0; i < copy_count; ++i) {
    auto* source = As<MtlDescriptorSet>(copies[i].srcSet);
    auto* target = As<MtlDescriptorSet>(copies[i].dstSet);
    for (std::uint32_t j = 0; j < copies[i].descriptorCount; ++j) {
      const std::uint32_t from = copies[i].srcBinding + j;
      const std::uint32_t to = copies[i].dstBinding + j;
      if (from >= kBindings || to >= kBindings) continue;
      target->buffers[to] = source->buffers[from];
      target->offsets[to] = source->offsets[from];
    }
  }
}

// ---- Command buffers -------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL CreateCommandPool(VkDevice, const VkCommandPoolCreateInfo*,
                                                 const VkAllocationCallbacks*, VkCommandPool* pool) {
  *pool = Handle<VkCommandPool>(new MtlCommandPool{});
  return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL DestroyCommandPool(VkDevice, VkCommandPool pool, const VkAllocationCallbacks*) {
  delete As<MtlCommandPool>(pool);
}
VKAPI_ATTR VkResult VKAPI_CALL AllocateCommandBuffers(VkDevice, const VkCommandBufferAllocateInfo* info,
                                                      VkCommandBuffer* buffers) {
  auto* pool = As<MtlCommandPool>(info->commandPool);
  for (std::uint32_t i = 0; i < info->commandBufferCount; ++i) {
    pool->buffers.push_back(std::make_unique<MtlCommandBuffer>());
    buffers[i] = Handle<VkCommandBuffer>(pool->buffers.back().get());
  }
  return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL FreeCommandBuffers(VkDevice, VkCommandPool, std::uint32_t, const VkCommandBuffer*) {}
VKAPI_ATTR VkResult VKAPI_CALL ResetCommandBuffer(VkCommandBuffer buffer, VkCommandBufferResetFlags) {
  auto* command_buffer = As<MtlCommandBuffer>(buffer);
  command_buffer->commands.clear();
  command_buffer->pipeline = nullptr;
  command_buffer->bound = {};
  command_buffer->push_size = 0;
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL BeginCommandBuffer(VkCommandBuffer buffer, const VkCommandBufferBeginInfo*) {
  return ResetCommandBuffer(buffer, 0);
}
VKAPI_ATTR VkResult VKAPI_CALL EndCommandBuffer(VkCommandBuffer) { return VK_SUCCESS; }
VKAPI_ATTR void VKAPI_CALL CmdBindPipeline(VkCommandBuffer buffer, VkPipelineBindPoint, VkPipeline pipeline) {
  As<MtlCommandBuffer>(buffer)->pipeline = As<MtlPipeline>(pipeline);
}
VKAPI_ATTR void VKAPI_CALL CmdBindDescriptorSets(VkCommandBuffer buffer, VkPipelineBindPoint, VkPipelineLayout,
                                                 std::uint32_t first_set, std::uint32_t count,
                                                 const VkDescriptorSet* sets, std::uint32_t, const std::uint32_t*) {
  // Descriptor contents are read when the command executes in Vulkan, and
  // the runtime never updates a set between binding and submission, so a
  // snapshot at bind time is equivalent.
  if (first_set == 0 && count >= 1) As<MtlCommandBuffer>(buffer)->bound = *As<MtlDescriptorSet>(sets[0]);
}
VKAPI_ATTR void VKAPI_CALL CmdPushConstants(VkCommandBuffer buffer, VkPipelineLayout, VkShaderStageFlags,
                                            std::uint32_t offset, std::uint32_t size, const void* values) {
  auto* command_buffer = As<MtlCommandBuffer>(buffer);
  if (offset + size > kPushBytes) return;
  std::memcpy(command_buffer->push.data() + offset, values, size);
  command_buffer->push_size = std::max(command_buffer->push_size, offset + size);
}
VKAPI_ATTR void VKAPI_CALL CmdDispatch(VkCommandBuffer buffer, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
  auto* command_buffer = As<MtlCommandBuffer>(buffer);
  if (!command_buffer->pipeline || x == 0 || y == 0 || z == 0) return;
  Cmd command{};
  command.type = CmdType::dispatch;
  command.pipeline = command_buffer->pipeline;
  command.buffers = command_buffer->bound.buffers;
  command.offsets = command_buffer->bound.offsets;
  command.push = command_buffer->push;
  command.push_size = command_buffer->push_size;
  command.x = x; command.y = y; command.z = z;
  command_buffer->commands.push_back(std::move(command));
}
VKAPI_ATTR void VKAPI_CALL CmdCopyBuffer(VkCommandBuffer buffer, VkBuffer source, VkBuffer target,
                                         std::uint32_t count, const VkBufferCopy* regions) {
  Cmd command{};
  command.type = CmdType::copy;
  command.src = As<MtlBuffer>(source);
  command.dst = As<MtlBuffer>(target);
  command.regions.assign(regions, regions + count);
  As<MtlCommandBuffer>(buffer)->commands.push_back(std::move(command));
}
VKAPI_ATTR void VKAPI_CALL CmdPipelineBarrier(VkCommandBuffer buffer, VkPipelineStageFlags, VkPipelineStageFlags,
                                              VkDependencyFlags, std::uint32_t, const VkMemoryBarrier*,
                                              std::uint32_t, const VkBufferMemoryBarrier*, std::uint32_t,
                                              const VkImageMemoryBarrier*) {
  auto& commands = As<MtlCommandBuffer>(buffer)->commands;
  if (!commands.empty() && commands.back().type == CmdType::barrier) return;
  Cmd command{};
  command.type = CmdType::barrier;
  commands.push_back(std::move(command));
}
VKAPI_ATTR void VKAPI_CALL CmdResetQueryPool(VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL CmdWriteTimestamp(VkCommandBuffer, VkPipelineStageFlagBits, VkQueryPool, std::uint32_t) {}

// ---- Synchronization -------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL CreateFence(VkDevice, const VkFenceCreateInfo* info, const VkAllocationCallbacks*,
                                           VkFence* fence) {
  *fence = Handle<VkFence>(new MtlFence{nil, (info->flags & VK_FENCE_CREATE_SIGNALED_BIT) != 0});
  return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL DestroyFence(VkDevice, VkFence fence, const VkAllocationCallbacks*) {
  auto* object = As<MtlFence>(fence);
  if (object && object->pending) [object->pending waitUntilCompleted];
  delete object;
}
VKAPI_ATTR VkResult VKAPI_CALL ResetFences(VkDevice, std::uint32_t count, const VkFence* fences) {
  for (std::uint32_t i = 0; i < count; ++i) {
    As<MtlFence>(fences[i])->pending = nil;
    As<MtlFence>(fences[i])->signaled = false;
  }
  return VK_SUCCESS;
}

VkResult CompleteFence(MtlFence* fence, bool wait) {
  if (fence->signaled && !fence->pending) return VK_SUCCESS;
  if (!fence->pending) return VK_NOT_READY;
  if (wait) [fence->pending waitUntilCompleted];
  const MTLCommandBufferStatus status = fence->pending.status;
  if (status == MTLCommandBufferStatusError) {
    NSError* error = fence->pending.error;
    std::fprintf(stderr, "ppocr: Metal command buffer failed: %s\n",
                 error ? error.localizedDescription.UTF8String : "unknown error");
    g_metal_errors.fetch_add(1, std::memory_order_relaxed);
    fence->pending = nil;
    return VK_ERROR_DEVICE_LOST;
  }
  if (status != MTLCommandBufferStatusCompleted) return VK_TIMEOUT;
  fence->pending = nil;
  fence->signaled = true;
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL WaitForFences(VkDevice, std::uint32_t count, const VkFence* fences, VkBool32,
                                             std::uint64_t timeout) {
  VkResult result = VK_SUCCESS;
  for (std::uint32_t i = 0; i < count; ++i) {
    const VkResult one = CompleteFence(As<MtlFence>(fences[i]), timeout != 0);
    if (one != VK_SUCCESS) result = one;
  }
  return result;
}
VKAPI_ATTR VkResult VKAPI_CALL GetFenceStatus(VkDevice, VkFence fence) {
  return CompleteFence(As<MtlFence>(fence), false);
}
VKAPI_ATTR VkResult VKAPI_CALL CreateSemaphore(VkDevice, const VkSemaphoreCreateInfo*, const VkAllocationCallbacks*,
                                               VkSemaphore* semaphore) {
  // One in-order queue: submission order already provides the dependency.
  *semaphore = Handle<VkSemaphore>(new MtlDummy{});
  return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL DestroySemaphore(VkDevice, VkSemaphore semaphore, const VkAllocationCallbacks*) {
  delete As<MtlDummy>(semaphore);
}
VKAPI_ATTR VkResult VKAPI_CALL CreateQueryPool(VkDevice, const VkQueryPoolCreateInfo*, const VkAllocationCallbacks*,
                                               VkQueryPool*) {
  return VK_ERROR_FEATURE_NOT_PRESENT;
}
VKAPI_ATTR void VKAPI_CALL DestroyQueryPool(VkDevice, VkQueryPool, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL GetQueryPoolResults(VkDevice, VkQueryPool, std::uint32_t, std::uint32_t, std::size_t,
                                                   void*, VkDeviceSize, VkQueryResultFlags) {
  return VK_ERROR_FEATURE_NOT_PRESENT;
}

// GPU time of every submitted command buffer, reported by PPOCR_GPU_PROFILE.
std::atomic<std::uint64_t> g_gpu_ns{0};
std::atomic<std::uint64_t> g_gpu_submits{0};
std::atomic<std::uint64_t> g_gpu_dispatches{0};

// ---- Native kernels ---------------------------------------------------------
// Compiled once from kMetalNativeKernels with the same IEEE options as the
// translated shaders. A missing kernel leaves the uber-shader path in place.
struct NativeKernels {
  id<MTLComputePipelineState> pointwise_gemm{nil};
  id<MTLComputePipelineState> conv3x3_gemm{nil};
  id<MTLComputePipelineState> rows_gemm{nil};
  bool attempted{};
};
NativeKernels g_native;
std::mutex g_native_mutex;

MTLCompileOptions* SafeCompileOptions() {
  MTLCompileOptions* options = [MTLCompileOptions new];
#if defined(__MAC_15_0) && __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_15_0
  if (@available(macOS 15.0, *)) {
    options.mathMode = MTLMathModeSafe;
    options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
  } else
#endif
  {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    options.fastMathEnabled = NO;
#pragma clang diagnostic pop
  }
  options.languageVersion = MTLLanguageVersion2_4;
  return options;
}

const NativeKernels& Native(id<MTLDevice> device) {
  std::lock_guard lock(g_native_mutex);
  if (g_native.attempted) return g_native;
  g_native.attempted = true;
  if (std::getenv("PPOCR_METAL_DISABLE_NATIVE") != nullptr) return g_native;
  @autoreleasepool {
    NSError* error = nil;
    NSString* text = [NSString stringWithUTF8String:ppocr::detail::kMetalNativeKernels];
    id<MTLLibrary> library = [device newLibraryWithSource:text options:SafeCompileOptions() error:&error];
    if (!library) {
      std::fprintf(stderr, "ppocr: Metal native kernels did not compile: %s\n",
                   error ? error.localizedDescription.UTF8String : "unknown error");
      return g_native;
    }
    const auto make = [&](NSString* name) -> id<MTLComputePipelineState> {
      id<MTLFunction> function = [library newFunctionWithName:name];
      if (!function) return nil;
      NSError* pipeline_error = nil;
      id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function
                                                                                 error:&pipeline_error];
      if (!state && Verbose())
        std::fprintf(stderr, "ppocr: Metal native %s failed: %s\n", name.UTF8String,
                     pipeline_error ? pipeline_error.localizedDescription.UTF8String : "unknown");
      return state;
    };
    g_native.pointwise_gemm = make(@"pointwise_gemm");
    if (std::getenv("PPOCR_METAL_DISABLE_NATIVE_CONV") == nullptr) g_native.conv3x3_gemm = make(@"conv3x3_gemm");
    if (std::getenv("PPOCR_METAL_DISABLE_NATIVE_ROWS_GEMM") == nullptr) g_native.rows_gemm = make(@"rows_gemm");
  }
  return g_native;
}

struct PushView {
  std::uint32_t count, batches, operation0, operation1, operation2, operation3;
  std::uint32_t steps, right_repeat, right_per_batch, right_batch_stride, mode;
};

// Returns a native pipeline and its grid for a flagged uber-shader dispatch.
id<MTLComputePipelineState> NativeForDispatch(id<MTLDevice> device, const Cmd& command,
                                              MTLSize* groups, MTLSize* threads) {
  if (command.push_size < sizeof(PushView) || !command.pipeline || !command.pipeline->program ||
      command.pipeline->program->kind != 0) return nil;
  PushView push{};
  std::memcpy(&push, command.push.data(), sizeof(push));
  if ((push.mode & 0x7fffu) == 50u || (push.mode & 0x7fffu) == 51u) {
    // 3x3 Conv: the uber-shader reads every field, so the driver selects
    // the native kernel by mode (PPOCR_METAL_DISABLE_NATIVE_CONV=1 opts out).
    id<MTLComputePipelineState> state = Native(device).conv3x3_gemm;
    const std::uint64_t plane = std::uint64_t(push.operation3) * push.steps;
    if (!state || plane == 0 || push.count % plane != 0 ||
        (push.right_batch_stride != 1u && push.right_batch_stride != 2u)) return nil;
    const std::uint64_t outputs = push.count / plane;
    *groups = MTLSizeMake(((outputs + 31) / 32) * ((plane + 31) / 32), command.y, 1);
    *threads = MTLSizeMake(128, 1, 1);
    return state;
  }
  if ((push.mode & 0xffffu) == 24u || (push.mode & 0xffffu) == 25u) {
    // Row GEMM (recognizer head / MLP); selected by mode.
    id<MTLComputePipelineState> state = Native(device).rows_gemm;
    if (!state || push.operation0 == 0 || push.operation2 == 0) return nil;
    *groups = MTLSizeMake(((push.operation0 + 31) / 32) * ((push.operation2 + 31) / 32), 1, 1);
    *threads = MTLSizeMake(128, 1, 1);
    return state;
  }
  if (push.operation3 != 1u) return nil;
  const std::uint32_t mode = push.mode & 0xffffu;
  switch (mode) {
    case 3: case 4: case 5: case 6: case 7: case 9: case 10: case 11: case 12: case 55: {
      id<MTLComputePipelineState> state = Native(device).pointwise_gemm;
      if (!state || push.right_repeat == 0 || push.right_batch_stride == 0) return nil;
      const NSUInteger tiles = ((push.right_batch_stride + 31) / 32) * ((push.right_repeat + 31) / 32);
      *groups = MTLSizeMake(tiles, command.y, 1);
      *threads = MTLSizeMake(128, 1, 1);
      return state;
    }
    default: return nil;
  }
}

bool EncodeCommands(MtlDevice* device, id<MTLCommandBuffer> metal_buffer, const MtlCommandBuffer& recorded) {
  static const bool serial = std::getenv("PPOCR_METAL_SERIAL_DISPATCH") != nullptr;
  id<MTLComputeCommandEncoder> encoder = nil;
  id<MTLComputePipelineState> current = nil;
  std::array<id<MTLBuffer>, kBindings> bound_buffers{};
  std::array<NSUInteger, kBindings> bound_offsets{};
  std::uint64_t dispatches = 0;
  bool barrier_pending = false;
  for (const Cmd& command : recorded.commands) {
    if (command.type == CmdType::copy) {
      if (encoder) { [encoder endEncoding]; encoder = nil; current = nil; bound_buffers = {}; }
      id<MTLBlitCommandEncoder> blit = [metal_buffer blitCommandEncoder];
      for (const auto& region : command.regions) {
        if (!command.src->memory || !command.dst->memory) return false;
        [blit copyFromBuffer:command.src->memory->buffer
                sourceOffset:command.src->offset + region.srcOffset
                    toBuffer:command.dst->memory->buffer
           destinationOffset:command.dst->offset + region.dstOffset
                        size:region.size];
      }
      [blit endEncoding];
      barrier_pending = false;
      continue;
    }
    if (command.type == CmdType::barrier) { barrier_pending = true; continue; }
    if (!encoder) {
      encoder = [metal_buffer computeCommandEncoderWithDispatchType:
                                  serial ? MTLDispatchTypeSerial : MTLDispatchTypeConcurrent];
      barrier_pending = false;
    } else if (barrier_pending) {
      if (!serial) [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      barrier_pending = false;
    }
    MTLSize native_groups{}, native_threads{};
    id<MTLComputePipelineState> native = NativeForDispatch(device->device, command, &native_groups, &native_threads);
    id<MTLComputePipelineState> state =
        native ? native : PipelineForDispatch(device->device, command.pipeline, command);
    if (!state) { [encoder endEncoding]; return false; }
    if (state != current) { [encoder setComputePipelineState:state]; current = state; }
    for (std::uint32_t binding = 0; binding < kBindings; ++binding) {
      MtlBuffer* buffer = command.buffers[binding];
      if (!buffer || !buffer->memory) continue;
      id<MTLBuffer> metal = buffer->memory->buffer;
      const NSUInteger offset = static_cast<NSUInteger>(buffer->offset + command.offsets[binding]);
      if (bound_buffers[binding] == metal) {
        if (bound_offsets[binding] != offset) {
          [encoder setBufferOffset:offset atIndex:binding];
          bound_offsets[binding] = offset;
        }
        continue;
      }
      [encoder setBuffer:metal offset:offset atIndex:binding];
      bound_buffers[binding] = metal;
      bound_offsets[binding] = offset;
    }
    [encoder setBytes:command.push.data() length:std::max<std::uint32_t>(command.push_size, 4)
              atIndex:kPushBufferIndex];
    const auto& local = command.pipeline->local;
    if (native) {
      [encoder dispatchThreadgroups:native_groups threadsPerThreadgroup:native_threads];
    } else {
      [encoder dispatchThreadgroups:MTLSizeMake(command.x, command.y, command.z)
              threadsPerThreadgroup:MTLSizeMake(local[0], local[1], local[2])];
    }
    ++dispatches;
  }
  if (encoder) [encoder endEncoding];
  g_gpu_dispatches.fetch_add(dispatches, std::memory_order_relaxed);
  return true;
}

// PPOCR_METAL_KERNEL_PROFILE=1: run every recorded command in its own
// command buffer and attribute its GPU time to the shader kind and mode.
// Execution stays in submission order, so results are unchanged; only the
// timing of the run is distorted by the per-command commits.
struct KernelStat { double ms{}; std::uint64_t calls{}; std::uint64_t groups{}; NSUInteger tg_memory{}; NSUInteger max_threads{}; };
std::mutex g_kernel_mutex;
std::map<std::string, KernelStat> g_kernel_stats;

bool KernelProfileEnabled() {
  static const bool enabled = std::getenv("PPOCR_METAL_KERNEL_PROFILE") != nullptr;
  return enabled;
}

std::string KernelKey(const Cmd& command) {
  if (command.type == CmdType::copy) return "copy";
  std::uint32_t mode = 0;
  if (command.push_size >= 44) std::memcpy(&mode, command.push.data() + 40, sizeof(mode));
  char text[64];
  std::uint32_t op3 = 0;
  if (command.push_size >= 24) std::memcpy(&op3, command.push.data() + 20, sizeof(op3));
  std::snprintf(text, sizeof(text), op3 == 1u ? "n%d.mode%u" : "k%d.mode%u", command.pipeline && command.pipeline->program
                ? command.pipeline->program->kind : -1, mode & 0xffffu);
  static const bool shapes = std::getenv("PPOCR_METAL_KERNEL_SHAPES") != nullptr;
  if (!shapes) return text;
  std::uint32_t fields[10]{};
  std::memcpy(fields, command.push.data(), std::min<std::size_t>(sizeof(fields), command.push_size));
  std::string key = text;
  for (std::uint32_t field : fields) key += "," + std::to_string(field);
  key += " g" + std::to_string(command.x) + "x" + std::to_string(command.y);
  return key;
}

bool SubmitProfiledCommands(MtlQueue* queue, const MtlCommandBuffer& recorded) {
  MtlCommandBuffer single;
  for (const Cmd& command : recorded.commands) {
    if (command.type == CmdType::barrier) continue;
    single.commands.assign(1, command);
    @autoreleasepool {
      id<MTLCommandBuffer> metal_buffer = [queue->device->queue commandBuffer];
      if (!EncodeCommands(queue->device, metal_buffer, single)) return false;
      [metal_buffer commit];
      [metal_buffer waitUntilCompleted];
      if (metal_buffer.status == MTLCommandBufferStatusError) return false;
      const double ms = (metal_buffer.GPUEndTime - metal_buffer.GPUStartTime) * 1e3;
      std::lock_guard lock(g_kernel_mutex);
      auto& stat = g_kernel_stats[KernelKey(command)];
      stat.ms += ms; ++stat.calls; stat.groups += std::uint64_t(command.x) * command.y * command.z;
      if (command.type == CmdType::dispatch) {
        id<MTLComputePipelineState> state = PipelineForDispatch(queue->device->device, command.pipeline, command);
        if (state) { stat.tg_memory = state.staticThreadgroupMemoryLength; stat.max_threads = state.maxTotalThreadsPerThreadgroup; }
      }
    }
  }
  return true;
}

VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit(VkQueue queue_handle, std::uint32_t count, const VkSubmitInfo* submits,
                                           VkFence fence) {
  auto* queue = As<MtlQueue>(queue_handle);
  @autoreleasepool {
    id<MTLCommandBuffer> metal_buffer = nil;
    static const bool profile = std::getenv("PPOCR_GPU_PROFILE") != nullptr;
    for (std::uint32_t i = 0; i < count; ++i) {
      for (std::uint32_t j = 0; j < submits[i].commandBufferCount; ++j) {
        if (KernelProfileEnabled()) {
          if (!SubmitProfiledCommands(queue, *As<MtlCommandBuffer>(submits[i].pCommandBuffers[j])))
            return VK_ERROR_DEVICE_LOST;
          continue;
        }
        if (!metal_buffer) metal_buffer = [queue->device->queue commandBufferWithUnretainedReferences];
        if (!EncodeCommands(queue->device, metal_buffer, *As<MtlCommandBuffer>(submits[i].pCommandBuffers[j]))) {
          return VK_ERROR_DEVICE_LOST;
        }
      }
    }
    if (!metal_buffer) metal_buffer = [queue->device->queue commandBufferWithUnretainedReferences];
    if (profile) {
      [metal_buffer addCompletedHandler:^(id<MTLCommandBuffer> done) {
        const double seconds = done.GPUEndTime - done.GPUStartTime;
        if (seconds > 0) g_gpu_ns.fetch_add(static_cast<std::uint64_t>(seconds * 1e9), std::memory_order_relaxed);
        g_gpu_submits.fetch_add(1, std::memory_order_relaxed);
      }];
    }
    [metal_buffer commit];
    queue->last = metal_buffer;
    if (fence) {
      As<MtlFence>(fence)->pending = metal_buffer;
      As<MtlFence>(fence)->signaled = false;
    }
  }
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL QueueWaitIdle(VkQueue queue_handle) {
  auto* queue = As<MtlQueue>(queue_handle);
  if (queue->last) {
    [queue->last waitUntilCompleted];
    const bool failed = queue->last.status == MTLCommandBufferStatusError;
    queue->last = nil;
    if (failed) { g_metal_errors.fetch_add(1, std::memory_order_relaxed); return VK_ERROR_DEVICE_LOST; }
  }
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL DeviceWaitIdle(VkDevice device) {
  return QueueWaitIdle(Handle<VkQueue>(As<MtlDevice>(device)->queue_handle));
}

PFN_vkVoidFunction Lookup(const char* name);
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetInstanceProcAddr(VkInstance, const char* name) { return Lookup(name); }
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice, const char* name) { return Lookup(name); }

PFN_vkVoidFunction Lookup(const char* name) {
  if (!name) return nullptr;
#define PPOCR_METAL_ENTRY(fn) {"vk" #fn, reinterpret_cast<PFN_vkVoidFunction>(&fn)}
  static const std::unordered_map<std::string, PFN_vkVoidFunction> table = {
      PPOCR_METAL_ENTRY(GetInstanceProcAddr), PPOCR_METAL_ENTRY(GetDeviceProcAddr),
      PPOCR_METAL_ENTRY(CreateInstance), PPOCR_METAL_ENTRY(DestroyInstance),
      PPOCR_METAL_ENTRY(EnumerateInstanceLayerProperties), PPOCR_METAL_ENTRY(EnumerateInstanceExtensionProperties),
      PPOCR_METAL_ENTRY(EnumerateDeviceExtensionProperties), PPOCR_METAL_ENTRY(EnumeratePhysicalDevices),
      PPOCR_METAL_ENTRY(GetPhysicalDeviceProperties), PPOCR_METAL_ENTRY(GetPhysicalDeviceQueueFamilyProperties),
      PPOCR_METAL_ENTRY(GetPhysicalDeviceMemoryProperties), PPOCR_METAL_ENTRY(CreateDevice),
      PPOCR_METAL_ENTRY(DestroyDevice), PPOCR_METAL_ENTRY(GetDeviceQueue), PPOCR_METAL_ENTRY(CreateBuffer),
      PPOCR_METAL_ENTRY(DestroyBuffer), PPOCR_METAL_ENTRY(GetBufferMemoryRequirements),
      PPOCR_METAL_ENTRY(AllocateMemory), PPOCR_METAL_ENTRY(FreeMemory), PPOCR_METAL_ENTRY(BindBufferMemory),
      PPOCR_METAL_ENTRY(MapMemory), PPOCR_METAL_ENTRY(UnmapMemory), PPOCR_METAL_ENTRY(FlushMappedMemoryRanges),
      PPOCR_METAL_ENTRY(InvalidateMappedMemoryRanges), PPOCR_METAL_ENTRY(CreateDescriptorSetLayout),
      PPOCR_METAL_ENTRY(DestroyDescriptorSetLayout), PPOCR_METAL_ENTRY(CreatePipelineLayout),
      PPOCR_METAL_ENTRY(DestroyPipelineLayout), PPOCR_METAL_ENTRY(CreateShaderModule),
      PPOCR_METAL_ENTRY(DestroyShaderModule), PPOCR_METAL_ENTRY(CreateComputePipelines),
      PPOCR_METAL_ENTRY(DestroyPipeline), PPOCR_METAL_ENTRY(CreateDescriptorPool),
      PPOCR_METAL_ENTRY(DestroyDescriptorPool), PPOCR_METAL_ENTRY(ResetDescriptorPool),
      PPOCR_METAL_ENTRY(AllocateDescriptorSets), PPOCR_METAL_ENTRY(UpdateDescriptorSets),
      PPOCR_METAL_ENTRY(CreateCommandPool), PPOCR_METAL_ENTRY(DestroyCommandPool),
      PPOCR_METAL_ENTRY(AllocateCommandBuffers), PPOCR_METAL_ENTRY(FreeCommandBuffers),
      PPOCR_METAL_ENTRY(ResetCommandBuffer), PPOCR_METAL_ENTRY(BeginCommandBuffer),
      PPOCR_METAL_ENTRY(EndCommandBuffer), PPOCR_METAL_ENTRY(CmdBindPipeline),
      PPOCR_METAL_ENTRY(CmdBindDescriptorSets), PPOCR_METAL_ENTRY(CmdPushConstants),
      PPOCR_METAL_ENTRY(CmdDispatch), PPOCR_METAL_ENTRY(CmdCopyBuffer), PPOCR_METAL_ENTRY(CmdPipelineBarrier),
      PPOCR_METAL_ENTRY(CmdResetQueryPool), PPOCR_METAL_ENTRY(CmdWriteTimestamp), PPOCR_METAL_ENTRY(CreateFence),
      PPOCR_METAL_ENTRY(DestroyFence), PPOCR_METAL_ENTRY(ResetFences), PPOCR_METAL_ENTRY(WaitForFences),
      PPOCR_METAL_ENTRY(GetFenceStatus), PPOCR_METAL_ENTRY(CreateSemaphore), PPOCR_METAL_ENTRY(DestroySemaphore),
      PPOCR_METAL_ENTRY(CreateQueryPool), PPOCR_METAL_ENTRY(DestroyQueryPool),
      PPOCR_METAL_ENTRY(GetQueryPoolResults), PPOCR_METAL_ENTRY(QueueSubmit), PPOCR_METAL_ENTRY(QueueWaitIdle),
      PPOCR_METAL_ENTRY(DeviceWaitIdle)};
#undef PPOCR_METAL_ENTRY
  const auto found = table.find(name);
  if (found == table.end()) {
    if (Verbose()) std::fprintf(stderr, "ppocr: Metal driver has no %s\n", name);
    return nullptr;
  }
  return found->second;
}

}  // namespace

extern "C" PFN_vkGetInstanceProcAddr ppocr_metal_vk_get_instance_proc_addr(void) {
  return SystemPhysicalDevice() ? &GetInstanceProcAddr : nullptr;
}

extern "C" int ppocr_metal_device_available(void) {
  @autoreleasepool { return SystemPhysicalDevice() != nullptr; }
}

extern "C" int ppocr_metal_error_count(void) { return g_metal_errors.load(std::memory_order_relaxed); }

extern "C" void ppocr_metal_print_profile(void) {
  const auto ns = g_gpu_ns.load(std::memory_order_relaxed);
  std::fprintf(stderr, "metal_profile command_buffers=%llu dispatches=%llu gpu_ms=%.3f\n",
               static_cast<unsigned long long>(g_gpu_submits.load(std::memory_order_relaxed)),
               static_cast<unsigned long long>(g_gpu_dispatches.load(std::memory_order_relaxed)),
               static_cast<double>(ns) / 1e6);
  if (!KernelProfileEnabled()) return;
  std::lock_guard lock(g_kernel_mutex);
  std::vector<std::pair<std::string, KernelStat>> rows(g_kernel_stats.begin(), g_kernel_stats.end());
  std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
  for (const auto& [key, stat] : rows) {
    std::fprintf(stderr, "metal_kernel %-16s gpu_ms=%9.3f calls=%7llu groups=%10llu tg_bytes=%5lu max_threads=%4lu\n",
                 key.c_str(), stat.ms, static_cast<unsigned long long>(stat.calls),
                 static_cast<unsigned long long>(stat.groups), static_cast<unsigned long>(stat.tg_memory),
                 static_cast<unsigned long>(stat.max_threads));
  }
}
