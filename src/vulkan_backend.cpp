#include "vulkan_backend.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <bit>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>
#include <unordered_map>
#include <vector>

#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <windows.h>
#elif defined(PPOCR_HAS_VULKAN_HEADERS) && (defined(__linux__) || defined(__APPLE__))
#include <vulkan/vulkan.h>
#include <dlfcn.h>

// The execution runtime below deliberately uses the Win32-shaped loader API
// so it can keep Vulkan as an optional runtime dependency.  Supply a tiny
// private POSIX adapter rather than linking against libvulkan or duplicating
// the substantial executor for Linux x86/ARM.  Defining `_WIN32` *after* the
// Vulkan headers is confined to this translation unit and makes the existing
// platform-neutral executor gates select this adapter; it never changes ABI
// definitions from the Vulkan headers.  This compatibility macro is temporary
// and is removed again after the implementation block below.
using HMODULE = void*;
inline HMODULE LoadLibraryA(const char* name) noexcept {
  (void)name;
#if defined(__APPLE__)
  return dlopen("libvulkan.1.dylib", RTLD_NOW | RTLD_LOCAL);
#else
  return dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
}
inline void* GetProcAddress(HMODULE module, const char* name) noexcept {
  return module ? dlsym(module, name) : nullptr;
}
inline int FreeLibrary(HMODULE module) noexcept {
  return module ? dlclose(module) == 0 : 0;
}
#define PPOCR_POSIX_VULKAN_LOADER_ADAPTER 1
#define _WIN32 1
#endif
#if defined(PPOCR_HAS_VULKAN_KERNELS)
#include "vulkan_binary_spv.hpp"
#include "vulkan_depthwise_tail_spv.hpp"
#include "vulkan_pointwise_tail_spv.hpp"
#endif

namespace ppocr::detail {

#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
namespace {

// This is deliberately a small, reusable Vulkan execution context rather than
// a second inference runtime.  Keeping device state, the pipeline, descriptor
// set, command buffer and host-visible buffers alive removes the large
// per-dispatch setup cost that made a future hybrid segment uncompetitive.
class VulkanBinaryRuntime {
  struct PersistentGraph {
    std::uint64_t key{};
    VkCommandBuffer command_buffer{VK_NULL_HANDLE};
    VkDescriptorPool descriptor_pool{VK_NULL_HANDLE};
    std::vector<VkDescriptorSet> descriptor_sets;
    // Every activation acquired while recording is pinned: a later replay
    // command buffer still addresses that exact storage.  Keep the complete
    // set, rather than only the public input/output slots, so dropping an LRU
    // replay graph returns all of its intermediate feature maps to the arena.
    std::vector<std::uint32_t> arena_slots;
    bool ready{};
  };

 public:
  VulkanBinaryRuntime() = default;
  ~VulkanBinaryRuntime() { Cleanup(); }
  VulkanBinaryRuntime(const VulkanBinaryRuntime&) = delete;
  VulkanBinaryRuntime& operator=(const VulkanBinaryRuntime&) = delete;

  bool Initialize() noexcept {
    if (restart_required_.exchange(false, std::memory_order_acq_rel)) {
      // The preceding failed submission invalidated the logical device. Tear
      // it down before attempting a new GPU-only graph; this never permits a
      // CPU activation fallback.
      Cleanup();
    }
    std::lock_guard lock(mutex_);
    if (attempted_) return ready_;
    attempted_ = true;

    loader_ = LoadLibraryA("vulkan-1.dll");
    if (!loader_) return false;
    get_instance_proc_ = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        GetProcAddress(loader_, "vkGetInstanceProcAddr"));
    create_instance_ = reinterpret_cast<PFN_vkCreateInstance>(
        get_instance_proc_ ? get_instance_proc_(VK_NULL_HANDLE, "vkCreateInstance") : nullptr);
    if (!create_instance_) { CleanupUnlocked(); return false; }

    const VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr,
                                "ppocr_cpp", 1, "ppocr_cpp", 1, VK_API_VERSION_1_0};
    const VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr,
                                              0, &app, 0, nullptr, 0, nullptr};
    if (create_instance_(&instance_info, nullptr, &instance_) != VK_SUCCESS) {
      CleanupUnlocked();
      return false;
    }

    destroy_instance_ = Global<PFN_vkDestroyInstance>("vkDestroyInstance");
    enumerate_devices_ = Global<PFN_vkEnumeratePhysicalDevices>("vkEnumeratePhysicalDevices");
    get_properties_ = Global<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties");
    get_queues_ = Global<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
        "vkGetPhysicalDeviceQueueFamilyProperties");
    get_memory_properties_ = Global<PFN_vkGetPhysicalDeviceMemoryProperties>(
        "vkGetPhysicalDeviceMemoryProperties");
    get_device_proc_ = Global<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");
    create_device_ = Global<PFN_vkCreateDevice>("vkCreateDevice");
    if (!destroy_instance_ || !enumerate_devices_ || !get_properties_ || !get_queues_ ||
        !get_memory_properties_ || !get_device_proc_ || !create_device_ || !SelectPhysicalDevice()) {
      CleanupUnlocked();
      return false;
    }

    const float priority = 1.F;
    const VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr,
                                              0, queue_family_, 1, &priority};
    const VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0,
                                          1, &queue_info, 0, nullptr, 0, nullptr, nullptr};
    if (create_device_(physical_, &device_info, nullptr, &device_) != VK_SUCCESS) {
      CleanupUnlocked();
      return false;
    }

    destroy_device_ = Device<PFN_vkDestroyDevice>("vkDestroyDevice");
    get_queue_ = Device<PFN_vkGetDeviceQueue>("vkGetDeviceQueue");
    create_buffer_ = Device<PFN_vkCreateBuffer>("vkCreateBuffer");
    destroy_buffer_ = Device<PFN_vkDestroyBuffer>("vkDestroyBuffer");
    get_buffer_memory_requirements_ = Device<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements");
    allocate_memory_ = Device<PFN_vkAllocateMemory>("vkAllocateMemory");
    free_memory_ = Device<PFN_vkFreeMemory>("vkFreeMemory");
    bind_buffer_memory_ = Device<PFN_vkBindBufferMemory>("vkBindBufferMemory");
    map_memory_ = Device<PFN_vkMapMemory>("vkMapMemory");
    unmap_memory_ = Device<PFN_vkUnmapMemory>("vkUnmapMemory");
    flush_mapped_memory_ranges_ = Device<PFN_vkFlushMappedMemoryRanges>("vkFlushMappedMemoryRanges");
    invalidate_mapped_memory_ranges_ = Device<PFN_vkInvalidateMappedMemoryRanges>("vkInvalidateMappedMemoryRanges");
    create_descriptor_set_layout_ = Device<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout");
    destroy_descriptor_set_layout_ = Device<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout");
    create_pipeline_layout_ = Device<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout");
    destroy_pipeline_layout_ = Device<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout");
    create_shader_module_ = Device<PFN_vkCreateShaderModule>("vkCreateShaderModule");
    destroy_shader_module_ = Device<PFN_vkDestroyShaderModule>("vkDestroyShaderModule");
    create_compute_pipelines_ = Device<PFN_vkCreateComputePipelines>("vkCreateComputePipelines");
    destroy_pipeline_ = Device<PFN_vkDestroyPipeline>("vkDestroyPipeline");
    create_descriptor_pool_ = Device<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool");
    destroy_descriptor_pool_ = Device<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool");
    reset_descriptor_pool_ = Device<PFN_vkResetDescriptorPool>("vkResetDescriptorPool");
    allocate_descriptor_sets_ = Device<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets");
    update_descriptor_sets_ = Device<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets");
    create_command_pool_ = Device<PFN_vkCreateCommandPool>("vkCreateCommandPool");
    destroy_command_pool_ = Device<PFN_vkDestroyCommandPool>("vkDestroyCommandPool");
    allocate_command_buffers_ = Device<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers");
    create_fence_ = Device<PFN_vkCreateFence>("vkCreateFence");
    destroy_fence_ = Device<PFN_vkDestroyFence>("vkDestroyFence");
    create_semaphore_ = Device<PFN_vkCreateSemaphore>("vkCreateSemaphore");
    destroy_semaphore_ = Device<PFN_vkDestroySemaphore>("vkDestroySemaphore");
    reset_fences_ = Device<PFN_vkResetFences>("vkResetFences");
    wait_for_fences_ = Device<PFN_vkWaitForFences>("vkWaitForFences");
    reset_command_buffer_ = Device<PFN_vkResetCommandBuffer>("vkResetCommandBuffer");
    begin_command_buffer_ = Device<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer");
    end_command_buffer_ = Device<PFN_vkEndCommandBuffer>("vkEndCommandBuffer");
    cmd_bind_pipeline_ = Device<PFN_vkCmdBindPipeline>("vkCmdBindPipeline");
    cmd_bind_descriptor_sets_ = Device<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets");
    cmd_push_constants_ = Device<PFN_vkCmdPushConstants>("vkCmdPushConstants");
    cmd_dispatch_ = Device<PFN_vkCmdDispatch>("vkCmdDispatch");
    cmd_copy_buffer_ = Device<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer");
    cmd_pipeline_barrier_ = Device<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
    queue_submit_ = Device<PFN_vkQueueSubmit>("vkQueueSubmit");
    queue_wait_idle_ = Device<PFN_vkQueueWaitIdle>("vkQueueWaitIdle");
    if (!HaveDeviceFunctions()) { CleanupUnlocked(); return false; }
    get_queue_(device_, queue_family_, 0, &queue_);
    if (!queue_ || !CreateFixedResources()) { CleanupUnlocked(); return false; }
    ready_ = true;
    last_submission_result_.store(VK_SUCCESS, std::memory_order_release);
    return true;
  }

  bool Run(float* output, const float* left, const float* right, std::size_t count,
           std::size_t batches, std::size_t right_repeat, std::size_t right_elements,
           std::size_t right_per_batch, std::size_t right_batch_stride,
           const kernels::BinaryOp* operations, std::size_t steps,
           bool immutable_right) noexcept {
    if (!output || !left || !right || !operations || count == 0 || batches == 0 ||
        count > UINT32_MAX || batches > UINT32_MAX || count > UINT32_MAX / batches ||
        right_repeat == 0 || right_elements == 0 ||
        right_repeat > UINT32_MAX || right_elements > UINT32_MAX ||
        right_per_batch > UINT32_MAX || right_batch_stride > UINT32_MAX ||
        steps == 0 || steps > 4 || !Initialize()) return false;
    std::lock_guard lock(mutex_);
    const auto elements = count * batches;
    const VkDeviceSize output_bytes = static_cast<VkDeviceSize>(elements) * sizeof(float);
    const VkDeviceSize right_bytes = static_cast<VkDeviceSize>(right_elements) * sizeof(float);
    if (!EnsureCapacity(output_bytes, right_bytes)) return false;
    BindScratchParameters(right_bytes, sizeof(float));
    std::memcpy(mapped_[0], left, static_cast<std::size_t>(output_bytes));
    // Model initializer storage is immutable and has a stable address for the
    // lifetime of an OnnxLite instance. Retain its compact RHS in the
    // persistent mapped buffer and avoid a redundant host write on every
    // hybrid dispatch. Dynamic tensors always copy.
    if (!immutable_right || !right_cached_ || right_cached_bytes_ != right_bytes ||
        right_cached_source_ != right) {
      std::memcpy(mapped_[1], right, static_cast<std::size_t>(right_bytes));
      chain_cached_ = false;
      right_cached_ = immutable_right;
      right_cached_bytes_ = immutable_right ? right_bytes : 0;
      right_cached_source_ = immutable_right ? right : nullptr;
      // Binding one is shared with pointwise weights.
      pointwise_cached_ = false;
    }
    if (reset_command_buffer_(command_buffer_, 0) != VK_SUCCESS) return false;
    const VkCommandBufferUsageFlags usage =
        std::getenv("PPOCR_GPU_ONLY_DISABLE_ONE_TIME_SUBMIT") != nullptr ? 0u :
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                         usage, nullptr};
    if (begin_command_buffer_(command_buffer_, &begin) != VK_SUCCESS) return false;
    cmd_bind_pipeline_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
                              0, 1, &descriptor_set_, 0, nullptr);
    struct Push {
      std::uint32_t count, batches, operation0, operation1, operation2, operation3, steps,
                    right_repeat, right_per_batch, right_batch_stride, mode;
    } push{static_cast<std::uint32_t>(count), static_cast<std::uint32_t>(batches),
           static_cast<std::uint32_t>(operations[0]),
           static_cast<std::uint32_t>(steps > 1 ? operations[1] : kernels::BinaryOp::add),
           static_cast<std::uint32_t>(steps > 2 ? operations[2] : kernels::BinaryOp::add),
           static_cast<std::uint32_t>(steps > 3 ? operations[3] : kernels::BinaryOp::add),
           static_cast<std::uint32_t>(steps), static_cast<std::uint32_t>(right_repeat),
           static_cast<std::uint32_t>(right_per_batch),
           static_cast<std::uint32_t>(right_batch_stride), 0};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                        0, sizeof(push), &push);
    // The SPIR-V kernel processes four contiguous floats per invocation.
    // Retain a scalar tail in the shader for shapes not divisible by four.
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>((count + 1023) / 1024),
                  static_cast<std::uint32_t>(batches), 1);
    if (end_command_buffer_(command_buffer_) != VK_SUCCESS) return false;
    const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
                              1, &command_buffer_, 0, nullptr};
    // Waiting on this submission's fence avoids idling the whole queue.  This
    // matters when a caller supplies multiple recognition batches: completion
    // remains synchronous for mapped readback, but unrelated queue work is no
    // longer included in each batch's critical path.
    if (reset_fences_(device_, 1, &submission_fence_) != VK_SUCCESS ||
        queue_submit_(queue_, 1, &submit, submission_fence_) != VK_SUCCESS ||
        wait_for_fences_(device_, 1, &submission_fence_, VK_TRUE,
                         std::numeric_limits<std::uint64_t>::max()) != VK_SUCCESS) return false;
    // The shader updates mapped_[0] in place.  This keeps the CPU-visible
    // activation/result in one allocation instead of copying a third mapped
    // output buffer after every batched submission.
    std::memcpy(output, mapped_[0], static_cast<std::size_t>(output_bytes));
    return true;
  }

  bool RunChannelAffine(float* output, const float* left, const float* scale,
                        const float* bias, std::size_t count, std::size_t batches,
                        std::size_t channel_repeat, std::size_t coefficient_elements,
                        bool immutable_coefficients, bool swish) noexcept {
    if (!output || !left || !scale || !bias || count == 0 || batches == 0 ||
        count > UINT32_MAX || batches > UINT32_MAX || count > UINT32_MAX / batches ||
        channel_repeat == 0 || coefficient_elements == 0 ||
        channel_repeat > UINT32_MAX || coefficient_elements > UINT32_MAX || !Initialize()) return false;
    const auto coefficients_per_batch = count / channel_repeat;
    if (count % channel_repeat != 0 || coefficients_per_batch == 0 ||
        (coefficient_elements != coefficients_per_batch &&
         (batches > UINT32_MAX / coefficients_per_batch ||
          coefficient_elements != batches * coefficients_per_batch))) return false;
    std::lock_guard lock(mutex_);
    const auto elements = count * batches;
    const VkDeviceSize output_bytes = static_cast<VkDeviceSize>(elements) * sizeof(float);
    const VkDeviceSize coefficient_bytes = static_cast<VkDeviceSize>(coefficient_elements) * sizeof(float);
    // Keep binding one's capacity stable for binary-chain RHS caching. Binding
    // two is the optional affine bias buffer; allocating it independently must
    // not make the scale descriptor point beyond its actual mapped buffer.
    if (!EnsureCapacity(output_bytes, coefficient_bytes, coefficient_bytes)) return false;
    BindScratchParameters(coefficient_bytes, coefficient_bytes);
    std::memcpy(mapped_[0], left, static_cast<std::size_t>(output_bytes));
    const bool reuse_coefficients = immutable_coefficients && coefficients_cached_ &&
        coefficients_cached_bytes_ == coefficient_bytes &&
        coefficients_cached_scale_ == scale && coefficients_cached_bias_ == bias;
    if (!reuse_coefficients) {
      std::memcpy(mapped_[1], scale, static_cast<std::size_t>(coefficient_bytes));
      std::memcpy(mapped_[2], bias, static_cast<std::size_t>(coefficient_bytes));
      chain_cached_ = false;
      coefficients_cached_ = immutable_coefficients;
      coefficients_cached_bytes_ = immutable_coefficients ? coefficient_bytes : 0;
      coefficients_cached_scale_ = immutable_coefficients ? scale : nullptr;
      coefficients_cached_bias_ = immutable_coefficients ? bias : nullptr;
      // Bindings one/two are also pointwise Conv's weight/bias buffers.
      right_cached_ = false;
      right_cached_bytes_ = 0;
      right_cached_source_ = nullptr;
      pointwise_cached_ = false;
    }
    if (reset_command_buffer_(command_buffer_, 0) != VK_SUCCESS) return false;
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                         VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    if (begin_command_buffer_(command_buffer_, &begin) != VK_SUCCESS) return false;
    cmd_bind_pipeline_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
                              0, 1, &descriptor_set_, 0, nullptr);
    struct Push {
      std::uint32_t count, batches, operation0, operation1, operation2, operation3, steps,
                    right_repeat, right_per_batch, right_batch_stride, mode;
    } push{static_cast<std::uint32_t>(count), static_cast<std::uint32_t>(batches), 0, 0, 0, 0, 1,
           static_cast<std::uint32_t>(channel_repeat),
           static_cast<std::uint32_t>(coefficients_per_batch),
           static_cast<std::uint32_t>(coefficient_elements == coefficients_per_batch ? 0 : coefficients_per_batch),
           swish ? 2u : 1u};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                        0, sizeof(push), &push);
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>((count + 1023) / 1024),
                  static_cast<std::uint32_t>(batches), 1);
    if (end_command_buffer_(command_buffer_) != VK_SUCCESS) return false;
    const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
                              1, &command_buffer_, 0, nullptr};
    if (reset_fences_(device_, 1, &submission_fence_) != VK_SUCCESS ||
        queue_submit_(queue_, 1, &submit, submission_fence_) != VK_SUCCESS ||
        wait_for_fences_(device_, 1, &submission_fence_, VK_TRUE,
                         std::numeric_limits<std::uint64_t>::max()) != VK_SUCCESS) return false;
    std::memcpy(output, mapped_[0], static_cast<std::size_t>(output_bytes));
    return true;
  }

  bool RunChannelAffineSwishBinary(float* output, const float* left,
                                   const float* scale, const float* bias,
                                   const float* right, std::size_t count,
                                   std::size_t batches, std::size_t channel_repeat,
                                   std::size_t coefficient_elements,
                                   kernels::BinaryOp operation,
                                   bool immutable_coefficients,
                                   bool immutable_right) noexcept {
    if (!output || !left || !scale || !bias || !right || count == 0 || batches == 0 ||
        count > UINT32_MAX || batches > UINT32_MAX || count > UINT32_MAX / batches ||
        channel_repeat == 0 || coefficient_elements == 0 || channel_repeat > UINT32_MAX ||
        coefficient_elements > UINT32_MAX || !Initialize()) return false;
    const auto coefficients_per_batch = count / channel_repeat;
    if (count % channel_repeat != 0 || coefficients_per_batch == 0 ||
        (coefficient_elements != coefficients_per_batch &&
         (batches > UINT32_MAX / coefficients_per_batch ||
          coefficient_elements != batches * coefficients_per_batch))) return false;
    std::lock_guard lock(mutex_);
    const auto elements = count * batches;
    const VkDeviceSize output_bytes = static_cast<VkDeviceSize>(elements) * sizeof(float);
    const VkDeviceSize coefficient_bytes = static_cast<VkDeviceSize>(coefficient_elements) * sizeof(float);
    // Binding three is the dynamic residual activation. Keep the affine result
    // in binding zero, so the entire affine->Swish->binary chain uses one
    // submission and one readback.
    if (!EnsureCapacity(output_bytes, coefficient_bytes, coefficient_bytes, output_bytes)) return false;
    BindScratchParameters(coefficient_bytes, coefficient_bytes);
    std::memcpy(mapped_[0], left, static_cast<std::size_t>(output_bytes));
    const bool reuse_coefficients = immutable_coefficients && coefficients_cached_ &&
        coefficients_cached_bytes_ == coefficient_bytes &&
        coefficients_cached_scale_ == scale && coefficients_cached_bias_ == bias;
    if (!reuse_coefficients) {
      std::memcpy(mapped_[1], scale, static_cast<std::size_t>(coefficient_bytes));
      std::memcpy(mapped_[2], bias, static_cast<std::size_t>(coefficient_bytes));
      chain_cached_ = false;
      coefficients_cached_ = immutable_coefficients;
      coefficients_cached_bytes_ = immutable_coefficients ? coefficient_bytes : 0;
      coefficients_cached_scale_ = immutable_coefficients ? scale : nullptr;
      coefficients_cached_bias_ = immutable_coefficients ? bias : nullptr;
      right_cached_ = false;
      right_cached_bytes_ = 0;
      right_cached_source_ = nullptr;
      pointwise_cached_ = false;
    }
    const bool reuse_right = immutable_right && right_cached_ &&
        right_cached_bytes_ == output_bytes && right_cached_source_ == right;
    if (!reuse_right) {
      std::memcpy(mapped_[3], right, static_cast<std::size_t>(output_bytes));
      right_cached_ = immutable_right;
      right_cached_bytes_ = immutable_right ? output_bytes : 0;
      right_cached_source_ = immutable_right ? right : nullptr;
    }
    if (reset_command_buffer_(command_buffer_, 0) != VK_SUCCESS) return false;
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                         VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    if (begin_command_buffer_(command_buffer_, &begin) != VK_SUCCESS) return false;
    cmd_bind_pipeline_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
                              0, 1, &descriptor_set_, 0, nullptr);
    struct Push {
      std::uint32_t count, batches, operation0, operation1, operation2, operation3, steps,
                    right_repeat, right_per_batch, right_batch_stride, mode;
    } push{static_cast<std::uint32_t>(count), static_cast<std::uint32_t>(batches),
           static_cast<std::uint32_t>(operation), 0, 0, 0, 1,
           static_cast<std::uint32_t>(channel_repeat),
           static_cast<std::uint32_t>(coefficients_per_batch),
           static_cast<std::uint32_t>(coefficient_elements == coefficients_per_batch ? 0 : coefficients_per_batch),
           8u};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                        0, sizeof(push), &push);
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>((count + 1023) / 1024),
                  static_cast<std::uint32_t>(batches), 1);
    if (end_command_buffer_(command_buffer_) != VK_SUCCESS) return false;
    const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
                              1, &command_buffer_, 0, nullptr};
    if (reset_fences_(device_, 1, &submission_fence_) != VK_SUCCESS ||
        queue_submit_(queue_, 1, &submit, submission_fence_) != VK_SUCCESS ||
        wait_for_fences_(device_, 1, &submission_fence_, VK_TRUE,
                         std::numeric_limits<std::uint64_t>::max()) != VK_SUCCESS) return false;
    std::memcpy(output, mapped_[0], static_cast<std::size_t>(output_bytes));
    return true;
  }

  bool RunPointwiseConv(float* output, const float* input, const float* weights,
                        const float* bias, const float* residual, std::size_t batches,
                        int input_channels, int output_channels, std::size_t plane,
                        bool immutable_parameters, bool relu, bool swish = false,
                        bool sigmoid = false, bool hard_sigmoid = false,
                        float hard_sigmoid_alpha = .2F,
                        float hard_sigmoid_beta = .5F,
                        bool hard_swish = false) noexcept {
    if (!output || !input || !weights || !bias || batches == 0 || input_channels <= 0 ||
        output_channels <= 0 || plane == 0 || batches > UINT32_MAX ||
        plane > UINT32_MAX ||
        (int(relu) + int(swish) + int(sigmoid) + int(hard_sigmoid) + int(hard_swish) > 1) ||
        !Initialize()) return false;
    const auto in_channels = static_cast<std::size_t>(input_channels);
    const auto out_channels = static_cast<std::size_t>(output_channels);
    if (in_channels > UINT32_MAX || out_channels > UINT32_MAX ||
        plane > UINT32_MAX / in_channels || plane > UINT32_MAX / out_channels ||
        batches > UINT32_MAX / (plane * out_channels) ||
        out_channels > std::numeric_limits<std::size_t>::max() / in_channels) return false;
    std::lock_guard lock(mutex_);
    const VkDeviceSize input_bytes = static_cast<VkDeviceSize>(batches) * in_channels * plane * sizeof(float);
    const VkDeviceSize output_bytes = static_cast<VkDeviceSize>(batches) * out_channels * plane * sizeof(float);
    const VkDeviceSize weight_bytes = static_cast<VkDeviceSize>(out_channels) * in_channels * sizeof(float);
    const VkDeviceSize bias_bytes = static_cast<VkDeviceSize>(out_channels) * sizeof(float);
    if (!EnsureCapacity(input_bytes, weight_bytes, bias_bytes, output_bytes,
                        residual ? output_bytes : sizeof(float))) return false;
    // Binding three remains the output. A residual projection uploads its
    // second live activation to binding four, avoiding any output alias.
    std::memcpy(mapped_[0], input, static_cast<std::size_t>(input_bytes));
    if (residual) {
      std::memcpy(mapped_[4], residual, static_cast<std::size_t>(output_bytes));
      // Binding four is also the packed pointwise data for the
      // depthwise->pointwise chain.
      chain_cached_ = false;
    }
    BindScratchParameters(weight_bytes, bias_bytes);
    const bool reuse_parameters = immutable_parameters && pointwise_cached_ &&
        pointwise_weight_bytes_ == weight_bytes && pointwise_bias_bytes_ == bias_bytes &&
        pointwise_cached_weights_ == weights && pointwise_cached_bias_ == bias;
    if (!reuse_parameters) {
      std::memcpy(mapped_[1], weights, static_cast<std::size_t>(weight_bytes));
      std::memcpy(mapped_[2], bias, static_cast<std::size_t>(bias_bytes));
      chain_cached_ = false;
      pointwise_cached_ = immutable_parameters;
      pointwise_weight_bytes_ = immutable_parameters ? weight_bytes : 0;
      pointwise_bias_bytes_ = immutable_parameters ? bias_bytes : 0;
      pointwise_cached_weights_ = immutable_parameters ? weights : nullptr;
      pointwise_cached_bias_ = immutable_parameters ? bias : nullptr;
      // Pointwise parameters overwrite the binary RHS and affine coefficients.
      right_cached_ = false;
      coefficients_cached_ = false;
    }
    if (reset_command_buffer_(command_buffer_, 0) != VK_SUCCESS) return false;
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                         VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    if (begin_command_buffer_(command_buffer_, &begin) != VK_SUCCESS) return false;
    cmd_bind_pipeline_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
                              0, 1, &descriptor_set_, 0, nullptr);
    struct Push {
      std::uint32_t count, batches, operation0, operation1, operation2, operation3, steps,
                    right_repeat, right_per_batch, right_batch_stride, mode;
    } push{static_cast<std::uint32_t>(out_channels * plane), static_cast<std::uint32_t>(batches),
           (hard_sigmoid || hard_swish) ? std::bit_cast<std::uint32_t>(hard_sigmoid_alpha) : 0u,
           (hard_sigmoid || hard_swish) ? std::bit_cast<std::uint32_t>(hard_sigmoid_beta) : 0u,
           0, 0, 1, static_cast<std::uint32_t>(plane),
           static_cast<std::uint32_t>(in_channels), static_cast<std::uint32_t>(out_channels),
           residual ? (swish ? 9u : (relu ? 6u : 5u)) :
                      (hard_swish ? 12u : (hard_sigmoid ? 11u :
                          (sigmoid ? 10u : (swish ? 7u : (relu ? 4u : 3u))))) };
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                        0, sizeof(push), &push);
    // The tiled shader maps one workgroup to four output channels and 256
    // spatial positions per channel, allowing its lanes to reuse a staged
    // [C] weight row. Wider-than-1024-channel projections retain the generic
    // one-vec4-per-invocation mapping used by the shader fallback.
    const auto dispatch_x = in_channels <= 1024
        ? ((out_channels + 3) / 4) * ((plane + 255) / 256)
        : (out_channels * plane + 1023) / 1024;
    if (dispatch_x == 0 || dispatch_x > UINT32_MAX) return false;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch_x),
                  static_cast<std::uint32_t>(batches), 1);
    if (end_command_buffer_(command_buffer_) != VK_SUCCESS) return false;
    const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
                              1, &command_buffer_, 0, nullptr};
    if (reset_fences_(device_, 1, &submission_fence_) != VK_SUCCESS ||
        queue_submit_(queue_, 1, &submit, submission_fence_) != VK_SUCCESS ||
        wait_for_fences_(device_, 1, &submission_fence_, VK_TRUE,
                         std::numeric_limits<std::uint64_t>::max()) != VK_SUCCESS) return false;
    std::memcpy(output, mapped_[3], static_cast<std::size_t>(output_bytes));
    return true;
  }

  bool RunPointwiseConvAdd(float* output, const float* input, const float* weights,
                           const float* bias, const float* residual,
                           std::size_t batches, int input_channels,
                           int output_channels, std::size_t plane,
                           bool immutable_parameters, bool relu,
                           bool swish = false) noexcept {
    if (!residual) return false;
    return RunPointwiseConv(output, input, weights, bias, residual, batches,
                            input_channels, output_channels, plane,
                            immutable_parameters, relu, swish);
  }

  bool RunDepthwiseConv(float* output, const float* input, const float* weights,
                        const float* bias, std::size_t batches, int channels,
                        int input_height, int input_width, int output_height,
                        int output_width, int kernel_height, int kernel_width,
                        int stride_height, int stride_width, int pad_top,
                        int pad_left, bool immutable_parameters, bool relu = false,
                        bool swish = false, bool hard_swish = false) noexcept {
    if (!output || !input || !weights || !bias || batches == 0 || channels <= 0 ||
        input_height <= 0 || input_width <= 0 || output_height <= 0 || output_width <= 0 ||
        kernel_height <= 0 || kernel_width <= 0 || stride_height <= 0 || stride_width <= 0 ||
        pad_top < 0 || pad_left < 0 || pad_top > 255 || pad_left > 255 ||
        batches > UINT32_MAX || (int(relu) + int(swish) + int(hard_swish) > 1) ||
        !Initialize()) return false;
    const auto c = static_cast<std::size_t>(channels);
    const auto ih = static_cast<std::size_t>(input_height);
    const auto iw = static_cast<std::size_t>(input_width);
    const auto oh = static_cast<std::size_t>(output_height);
    const auto ow = static_cast<std::size_t>(output_width);
    const auto kh = static_cast<std::size_t>(kernel_height);
    const auto kw = static_cast<std::size_t>(kernel_width);
    if (c > UINT32_MAX || ih > UINT32_MAX || iw > UINT32_MAX || oh > UINT32_MAX ||
        ow > UINT32_MAX || kh > UINT32_MAX || kw > UINT32_MAX ||
        ih > UINT32_MAX / iw || oh > UINT32_MAX / ow ||
        c > UINT32_MAX / (ih * iw) || c > UINT32_MAX / (oh * ow) ||
        c > UINT32_MAX / (kh * kw) || batches > UINT32_MAX / (c * oh * ow)) return false;
    const auto input_per_batch = c * ih * iw;
    const auto output_per_batch = c * oh * ow;
    const auto weight_elements = c * kh * kw;
    std::lock_guard lock(mutex_);
    const VkDeviceSize input_bytes = static_cast<VkDeviceSize>(batches) * input_per_batch * sizeof(float);
    const VkDeviceSize output_bytes = static_cast<VkDeviceSize>(batches) * output_per_batch * sizeof(float);
    const VkDeviceSize weight_bytes = static_cast<VkDeviceSize>(weight_elements) * sizeof(float);
    const VkDeviceSize bias_bytes = static_cast<VkDeviceSize>(c) * sizeof(float);
    if (!EnsureCapacity(input_bytes, weight_bytes, bias_bytes, output_bytes)) return false;
    std::memcpy(mapped_[0], input, static_cast<std::size_t>(input_bytes));
    BindScratchParameters(weight_bytes, bias_bytes);
    const bool reuse_parameters = immutable_parameters && pointwise_cached_ &&
        pointwise_weight_bytes_ == weight_bytes && pointwise_bias_bytes_ == bias_bytes &&
        pointwise_cached_weights_ == weights && pointwise_cached_bias_ == bias;
    if (!reuse_parameters) {
      std::memcpy(mapped_[1], weights, static_cast<std::size_t>(weight_bytes));
      std::memcpy(mapped_[2], bias, static_cast<std::size_t>(bias_bytes));
      chain_cached_ = false;
      pointwise_cached_ = immutable_parameters;
      pointwise_weight_bytes_ = immutable_parameters ? weight_bytes : 0;
      pointwise_bias_bytes_ = immutable_parameters ? bias_bytes : 0;
      pointwise_cached_weights_ = immutable_parameters ? weights : nullptr;
      pointwise_cached_bias_ = immutable_parameters ? bias : nullptr;
      right_cached_ = false;
      coefficients_cached_ = false;
    }
    if (reset_command_buffer_(command_buffer_, 0) != VK_SUCCESS) return false;
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                         VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    if (begin_command_buffer_(command_buffer_, &begin) != VK_SUCCESS) return false;
    cmd_bind_pipeline_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
                              0, 1, &descriptor_set_, 0, nullptr);
    struct Push {
      std::uint32_t count, batches, operation0, operation1, operation2, operation3, steps,
                    right_repeat, right_per_batch, right_batch_stride, mode;
    } push{static_cast<std::uint32_t>(output_per_batch), static_cast<std::uint32_t>(batches),
           static_cast<std::uint32_t>(kernel_height), static_cast<std::uint32_t>(kernel_width),
           static_cast<std::uint32_t>(stride_height), static_cast<std::uint32_t>(stride_width),
           static_cast<std::uint32_t>(output_width), static_cast<std::uint32_t>(input_height),
           static_cast<std::uint32_t>(input_width), static_cast<std::uint32_t>(output_height),
           (hard_swish ? 22u : (swish ? 21u : (relu ? 20u : 13u))) |
               (static_cast<std::uint32_t>(pad_top) << 16u) |
               (static_cast<std::uint32_t>(pad_left) << 24u)};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                        0, sizeof(push), &push);
    // Mode 13 maps each workgroup to one channel and up to 1,024 spatial
    // values.  This makes the channel's compact depthwise filter shared
    // memory, so a batched MobileNet block no longer rereads [KH,KW] for every
    // output pixel.  The Y dimension remains the true NCHW batch dimension.
    const auto groups_per_channel = (oh * ow + 1023) / 1024;
    if (groups_per_channel == 0 || c > std::numeric_limits<std::size_t>::max() / groups_per_channel) {
      return false;
    }
    const auto dispatch_x = c * groups_per_channel;
    if (dispatch_x == 0 || dispatch_x > UINT32_MAX) return false;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch_x),
                  static_cast<std::uint32_t>(batches), 1);
    if (end_command_buffer_(command_buffer_) != VK_SUCCESS) return false;
    const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
                              1, &command_buffer_, 0, nullptr};
    if (reset_fences_(device_, 1, &submission_fence_) != VK_SUCCESS ||
        queue_submit_(queue_, 1, &submit, submission_fence_) != VK_SUCCESS ||
        wait_for_fences_(device_, 1, &submission_fence_, VK_TRUE,
                         std::numeric_limits<std::uint64_t>::max()) != VK_SUCCESS) return false;
    std::memcpy(output, mapped_[3], static_cast<std::size_t>(output_bytes));
    return true;
  }

  bool RunDepthwisePointwiseConv(
      float* output, const float* input, const float* depthwise_weights,
      const float* depthwise_bias, const float* pointwise_weights,
      const float* pointwise_bias, std::size_t batches, int channels,
      int output_channels, int input_height, int input_width, int output_height,
      int output_width, int kernel_height, int kernel_width, int stride_height,
      int stride_width, int pad_top, int pad_left,
      bool immutable_parameters, bool approximate_gelu) noexcept {
    if (!output || !input || !depthwise_weights || !depthwise_bias ||
        !pointwise_weights || !pointwise_bias || batches == 0 || channels <= 0 ||
        output_channels <= 0 || input_height <= 0 || input_width <= 0 ||
        output_height <= 0 || output_width <= 0 || kernel_height <= 0 ||
        kernel_width <= 0 || stride_height <= 0 || stride_width <= 0 ||
        pad_top < 0 || pad_left < 0 || pad_top > 255 || pad_left > 255 ||
        batches > UINT32_MAX || !Initialize()) return false;
    const auto c = static_cast<std::size_t>(channels);
    const auto m = static_cast<std::size_t>(output_channels);
    const auto ih = static_cast<std::size_t>(input_height);
    const auto iw = static_cast<std::size_t>(input_width);
    const auto oh = static_cast<std::size_t>(output_height);
    const auto ow = static_cast<std::size_t>(output_width);
    const auto kh = static_cast<std::size_t>(kernel_height);
    const auto kw = static_cast<std::size_t>(kernel_width);
    if (c > UINT32_MAX || m > UINT32_MAX || ih > UINT32_MAX || iw > UINT32_MAX ||
        oh > UINT32_MAX || ow > UINT32_MAX || kh > UINT32_MAX || kw > UINT32_MAX ||
        ih > UINT32_MAX / iw || oh > UINT32_MAX / ow ||
        c > UINT32_MAX / (ih * iw) || c > UINT32_MAX / (oh * ow) ||
        c > UINT32_MAX / (kh * kw) || m > UINT32_MAX / c ||
        batches > UINT32_MAX / (m * oh * ow)) return false;
    const auto input_per_batch = c * ih * iw;
    const auto intermediate_per_batch = c * oh * ow;
    const auto output_per_batch = m * oh * ow;
    const auto depthwise_weight_elements = c * kh * kw;
    const auto pointwise_weight_elements = m * c;
    if (pointwise_weight_elements > std::numeric_limits<std::size_t>::max() - m) return false;
    const auto packed_pointwise_elements = pointwise_weight_elements + m;
    std::lock_guard lock(mutex_);
    const VkDeviceSize input_bytes = static_cast<VkDeviceSize>(batches) * input_per_batch * sizeof(float);
    const VkDeviceSize intermediate_bytes =
        static_cast<VkDeviceSize>(batches) * intermediate_per_batch * sizeof(float);
    const VkDeviceSize output_bytes = static_cast<VkDeviceSize>(batches) * output_per_batch * sizeof(float);
    const VkDeviceSize depthwise_weight_bytes =
        static_cast<VkDeviceSize>(depthwise_weight_elements) * sizeof(float);
    const VkDeviceSize depthwise_bias_bytes = static_cast<VkDeviceSize>(c) * sizeof(float);
    const VkDeviceSize packed_pointwise_bytes =
        static_cast<VkDeviceSize>(packed_pointwise_elements) * sizeof(float);
    // Binding 0 is both chain input and final result; binding 3 carries the
    // sole device-resident intermediate. Binding 4 packs the second Conv's
    // [M,C] weights and [M] bias, keeping both parameter sets live at once.
    if (!EnsureCapacity(std::max(input_bytes, output_bytes), depthwise_weight_bytes,
                        depthwise_bias_bytes, intermediate_bytes, packed_pointwise_bytes)) return false;
    std::memcpy(mapped_[0], input, static_cast<std::size_t>(input_bytes));
    BindScratchParameters(depthwise_weight_bytes, depthwise_bias_bytes);
    // Bindings one/two/four are shared scratch buffers. The pointer-and-size
    // key below is valid only until another primitive uses those bindings;
    // every such upload invalidates chain_cached_. Consecutive immutable
    // executions of this graph node can therefore skip three model uploads.
    const bool reuse_parameters = immutable_parameters && chain_cached_ &&
        chain_depthwise_weight_bytes_ == depthwise_weight_bytes &&
        chain_depthwise_bias_bytes_ == depthwise_bias_bytes &&
        chain_pointwise_bytes_ == packed_pointwise_bytes &&
        chain_depthwise_weights_ == depthwise_weights &&
        chain_depthwise_bias_ == depthwise_bias &&
        chain_pointwise_weights_ == pointwise_weights &&
        chain_pointwise_bias_ == pointwise_bias;
    if (!reuse_parameters) {
      std::memcpy(mapped_[1], depthwise_weights, static_cast<std::size_t>(depthwise_weight_bytes));
      std::memcpy(mapped_[2], depthwise_bias, static_cast<std::size_t>(depthwise_bias_bytes));
      std::memcpy(mapped_[4], pointwise_weights, static_cast<std::size_t>(pointwise_weight_elements * sizeof(float)));
      std::memcpy(static_cast<float*>(mapped_[4]) + pointwise_weight_elements, pointwise_bias,
                  static_cast<std::size_t>(m * sizeof(float)));
      chain_cached_ = immutable_parameters;
      chain_depthwise_weights_ = immutable_parameters ? depthwise_weights : nullptr;
      chain_depthwise_bias_ = immutable_parameters ? depthwise_bias : nullptr;
      chain_pointwise_weights_ = immutable_parameters ? pointwise_weights : nullptr;
      chain_pointwise_bias_ = immutable_parameters ? pointwise_bias : nullptr;
      chain_depthwise_weight_bytes_ = immutable_parameters ? depthwise_weight_bytes : 0;
      chain_depthwise_bias_bytes_ = immutable_parameters ? depthwise_bias_bytes : 0;
      chain_pointwise_bytes_ = immutable_parameters ? packed_pointwise_bytes : 0;
      pointwise_cached_ = false;
      right_cached_ = false;
      coefficients_cached_ = false;
    }
    if (reset_command_buffer_(command_buffer_, 0) != VK_SUCCESS) return false;
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                         VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    if (begin_command_buffer_(command_buffer_, &begin) != VK_SUCCESS) return false;
    cmd_bind_pipeline_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
                              0, 1, &descriptor_set_, 0, nullptr);
    struct Push {
      std::uint32_t count, batches, operation0, operation1, operation2, operation3, steps,
                    right_repeat, right_per_batch, right_batch_stride, mode;
    } depthwise_push{static_cast<std::uint32_t>(intermediate_per_batch),
                     static_cast<std::uint32_t>(batches),
                     static_cast<std::uint32_t>(kernel_height),
                     static_cast<std::uint32_t>(kernel_width),
                     static_cast<std::uint32_t>(stride_height),
                     static_cast<std::uint32_t>(stride_width),
                     static_cast<std::uint32_t>(output_width),
                     static_cast<std::uint32_t>(input_height),
                     static_cast<std::uint32_t>(input_width),
                     static_cast<std::uint32_t>(output_height),
                     13u | (static_cast<std::uint32_t>(pad_top) << 16u) |
                         (static_cast<std::uint32_t>(pad_left) << 24u)};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                        sizeof(depthwise_push), &depthwise_push);
    const auto groups_per_channel = (oh * ow + 1023) / 1024;
    const auto depthwise_dispatch_x = c * groups_per_channel;
    if (depthwise_dispatch_x == 0 || depthwise_dispatch_x > UINT32_MAX) return false;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(depthwise_dispatch_x),
                  static_cast<std::uint32_t>(batches), 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier,
                          0, nullptr, 0, nullptr);
    const auto plane = oh * ow;
    Push pointwise_push{static_cast<std::uint32_t>(output_per_batch),
                        static_cast<std::uint32_t>(batches), 0, 0, 0, 0, 1,
                        static_cast<std::uint32_t>(plane), static_cast<std::uint32_t>(c),
                        static_cast<std::uint32_t>(m), approximate_gelu ? 30u : 29u};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                        sizeof(pointwise_push), &pointwise_push);
    const auto pointwise_dispatch_x = c <= 1024
        ? ((m + 3) / 4) * ((plane + 255) / 256)
        : (m * plane + 1023) / 1024;
    if (pointwise_dispatch_x == 0 || pointwise_dispatch_x > UINT32_MAX) return false;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(pointwise_dispatch_x),
                  static_cast<std::uint32_t>(batches), 1);
    if (end_command_buffer_(command_buffer_) != VK_SUCCESS) return false;
    const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
                              1, &command_buffer_, 0, nullptr};
    if (reset_fences_(device_, 1, &submission_fence_) != VK_SUCCESS ||
        queue_submit_(queue_, 1, &submit, submission_fence_) != VK_SUCCESS ||
        wait_for_fences_(device_, 1, &submission_fence_, VK_TRUE,
                         std::numeric_limits<std::uint64_t>::max()) != VK_SUCCESS) return false;
    std::memcpy(output, mapped_[0], static_cast<std::size_t>(output_bytes));
    return true;
  }

  bool RunConv2d(float* output, const float* input, const float* weights,
                 const float* bias, std::size_t batches, int input_channels,
                 int output_channels, int input_height, int input_width,
                 int output_height, int output_width, int kernel_height,
                 int kernel_width, int stride_height, int stride_width,
                 int pad_top, int pad_left, bool immutable_parameters,
                 bool relu, bool swish, bool sigmoid, bool hard_swish,
                 const float* const* concat_sources = nullptr,
                 const int* concat_channels = nullptr,
                 int concat_source_count = 0,
                 bool pool_first = false) noexcept {
    // Push-constant space deliberately encodes one common stride. PP-OCRv6
    // uses square ordinary convolutions; asymmetric strides retain CPU SIMD.
    const bool pack_concat = concat_sources != nullptr;
    if (!output || !weights || !bias || batches == 0 || input_channels <= 0 ||
        output_channels <= 0 || input_height <= 0 || input_width <= 0 ||
        output_height <= 0 || output_width <= 0 || kernel_height <= 0 ||
        kernel_width <= 0 || stride_height <= 0 || stride_height != stride_width ||
        pad_top < 0 || pad_left < 0 || pad_top > 255 || pad_left > 255 ||
        batches > UINT32_MAX ||
        (int(relu) + int(swish) + int(sigmoid) + int(hard_swish) > 1) ||
        (pack_concat
            ? (!concat_channels || concat_source_count < 2 || concat_source_count > 4)
            : !input) ||
        !Initialize()) return false;
    const auto in_channels = static_cast<std::size_t>(input_channels);
    const auto out_channels = static_cast<std::size_t>(output_channels);
    const auto ih = static_cast<std::size_t>(input_height);
    const auto iw = static_cast<std::size_t>(input_width);
    const auto oh = static_cast<std::size_t>(output_height);
    const auto ow = static_cast<std::size_t>(output_width);
    const auto kh = static_cast<std::size_t>(kernel_height);
    const auto kw = static_cast<std::size_t>(kernel_width);
    if (in_channels > UINT32_MAX || out_channels > UINT32_MAX || ih > UINT32_MAX ||
        iw > UINT32_MAX || oh > UINT32_MAX || ow > UINT32_MAX || kh > UINT32_MAX ||
        kw > UINT32_MAX || ih > UINT32_MAX / iw || oh > UINT32_MAX / ow ||
        kh > UINT32_MAX / kw || in_channels > UINT32_MAX / (ih * iw) ||
        out_channels > UINT32_MAX / (oh * ow) ||
        out_channels > UINT32_MAX / (in_channels * kh * kw) ||
        batches > UINT32_MAX / (out_channels * oh * ow)) return false;
    const auto input_per_batch = in_channels * ih * iw;
    const auto output_per_batch = out_channels * oh * ow;
    const auto weight_elements = out_channels * in_channels * kh * kw;
    std::lock_guard lock(mutex_);
    const VkDeviceSize input_bytes = static_cast<VkDeviceSize>(batches) * input_per_batch * sizeof(float);
    const VkDeviceSize output_bytes = static_cast<VkDeviceSize>(batches) * output_per_batch * sizeof(float);
    const VkDeviceSize weight_bytes = static_cast<VkDeviceSize>(weight_elements) * sizeof(float);
    const VkDeviceSize bias_bytes = static_cast<VkDeviceSize>(out_channels) * sizeof(float);
    if (!EnsureCapacity(input_bytes, weight_bytes, bias_bytes, output_bytes)) return false;
    if (pack_concat) {
      // Pack Concat / MaxPool+Concat directly into the mapped upload buffer.
      // The previous host vector plus a second memcpy doubled Concat.2's
      // 1.8 MB transfer before the GPU conv even started.
      auto* dest = static_cast<float*>(mapped_[0]);
      const std::size_t plane = ih * iw;
      int packed_channels = 0;
      for (int i = 0; i < concat_source_count; ++i) {
        if (!concat_sources[i] || concat_channels[i] <= 0) return false;
        packed_channels += concat_channels[i];
      }
      if (packed_channels != input_channels) return false;
      const std::size_t first_n = batches * std::size_t(concat_channels[0]) * plane;
      // Pack the unpooled first source when the GPU MaxPool (mode 40) can
      // own it. Batched Concat still pools on the host because mode 40's
      // count is per-batch and this hybrid path is N=1 for PP-OCR.
      if (pool_first && batches != 1) {
        kernels::MaxPool2x2Same(dest, concat_sources[0],
                                batches * std::size_t(concat_channels[0]), input_height,
                                input_width);
      } else {
        std::memcpy(dest, concat_sources[0], first_n * sizeof(float));
      }
      dest += first_n;
      for (int i = 1; i < concat_source_count; ++i) {
        const std::size_t n = batches * std::size_t(concat_channels[i]) * plane;
        std::memcpy(dest, concat_sources[i], n * sizeof(float));
        dest += n;
      }
    } else {
      std::memcpy(mapped_[0], input, static_cast<std::size_t>(input_bytes));
    }
    BindScratchParameters(weight_bytes, bias_bytes);
    const bool reuse_parameters = immutable_parameters && pointwise_cached_ &&
        pointwise_weight_bytes_ == weight_bytes && pointwise_bias_bytes_ == bias_bytes &&
        pointwise_cached_weights_ == weights && pointwise_cached_bias_ == bias;
    if (!reuse_parameters) {
      std::memcpy(mapped_[1], weights, static_cast<std::size_t>(weight_bytes));
      std::memcpy(mapped_[2], bias, static_cast<std::size_t>(bias_bytes));
      chain_cached_ = false;
      pointwise_cached_ = immutable_parameters;
      pointwise_weight_bytes_ = immutable_parameters ? weight_bytes : 0;
      pointwise_bias_bytes_ = immutable_parameters ? bias_bytes : 0;
      pointwise_cached_weights_ = immutable_parameters ? weights : nullptr;
      pointwise_cached_bias_ = immutable_parameters ? bias : nullptr;
      right_cached_ = false;
      coefficients_cached_ = false;
    }
    // Large 3x3 maps use a 16x16 spatial tile with a shared input halo.
    // Conv.0 is C<=8. Concat+Conv (C=32/64 s1, C=32 s2) also takes this
    // path via pack_concat; other mid-C 3x3 maps stay on the tiled 4-OC
    // kernel because spatial admission was noisier and slower on those
    // shapes. Modes 50/51 are unused elsewhere; 40 is Pool2D and 41 is
    // broadcast. Concat.2 C=64 s1 stages 12 IC/pass in the 4096-float halo.
    const bool use_spatial_3x3 = kh == 3 && kw == 3 && oh * ow >= 2048 &&
        !swish && !sigmoid && !hard_swish &&
        std::getenv("PPOCR_DISABLE_VULKAN_CONV2D_SPATIAL") == nullptr &&
        ((stride_height == 1 && (in_channels <= 8 || (pack_concat && in_channels <= 64)) &&
          in_channels <= 64) ||
         (stride_height == 2 &&
          (in_channels <= 8 || (pack_concat && in_channels <= 32))));
    const std::uint32_t conv_mode = use_spatial_3x3
        ? (relu ? 51u : 50u)
        : (hard_swish ? 28u : (sigmoid ? 27u : (swish ? 26u : (relu ? 15u : 14u))));
    const std::uint32_t mode_bits = conv_mode |
        (static_cast<std::uint32_t>(pad_top) << 16u) |
        (static_cast<std::uint32_t>(pad_left) << 24u);
    struct Push {
      std::uint32_t count, batches, operation0, operation1, operation2, operation3, steps,
                    right_repeat, right_per_batch, right_batch_stride, mode;
    } push{static_cast<std::uint32_t>(output_per_batch), static_cast<std::uint32_t>(batches),
           static_cast<std::uint32_t>(in_channels), static_cast<std::uint32_t>(ih),
           static_cast<std::uint32_t>(iw), static_cast<std::uint32_t>(oh),
           static_cast<std::uint32_t>(ow), static_cast<std::uint32_t>(kh),
           static_cast<std::uint32_t>(kw), static_cast<std::uint32_t>(stride_height),
           mode_bits};
    const auto weights_per_channel = in_channels * kh * kw;
    // Must match the shader: one workgroup is 256 lanes x 4 floats. The old
    // 16K hybrid cutoff left FPN Concat.2 (40x176=7040) on the generic
    // path with (N+3)/4 workgroups — 256x over-dispatch. Keep the same
    // 256-plane floor the shader uses so SE maps stay compact.
    // Shader tiled path requires plane>=2048 (SE maps below that take the
    // generic vec4 dispatch). Host used 256 and mis-dispatched 33x33.
    constexpr std::size_t kConv2dTiledMinimumOutputPlane = 2048u;
    const bool use_tiled = !use_spatial_3x3 && stride_height == 1 &&
        weights_per_channel <= 4096 &&
        output_per_batch / out_channels >= kConv2dTiledMinimumOutputPlane &&
        std::getenv("PPOCR_DISABLE_VULKAN_CONV2D_TILE") == nullptr;
    const bool tile4oc = use_tiled && kh == 3 && kw == 3 &&
        weights_per_channel * 4 <= 4096;
    const auto dispatch_x = use_spatial_3x3
        ? ((ow + 15) / 16) * ((oh + 15) / 16) * ((out_channels + 3) / 4)
        : (use_tiled
            ? (tile4oc ? ((out_channels + 3) / 4) : out_channels) *
                  ((oh * ow + 1023) / 1024)
            : (output_per_batch + 1023) / 1024);
    if (dispatch_x == 0 || dispatch_x > UINT32_MAX || !conv2d_command_buffer_) return false;
    const std::size_t pool_count = (pool_first && pack_concat && batches == 1)
        ? std::size_t(concat_channels[0]) * ih * iw : 0;
    const bool reuse_cmd = conv2d_cmd_ready_ &&
        conv2d_dispatch_x_ == static_cast<std::uint32_t>(dispatch_x) &&
        conv2d_batches_ == static_cast<std::uint32_t>(batches) &&
        conv2d_input_bytes_ == input_bytes && conv2d_output_bytes_ == output_bytes &&
        conv2d_weight_bytes_ == weight_bytes && conv2d_bias_bytes_ == bias_bytes &&
        conv2d_mode_ == mode_bits && conv2d_weights_ == weights && conv2d_bias_ == bias &&
        conv2d_pool_first_ == pool_first;
    if (!reuse_cmd) {
      if (reset_command_buffer_(conv2d_command_buffer_, 0) != VK_SUCCESS) return false;
      const VkCommandBufferBeginInfo persistent_begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
          nullptr, 0, nullptr};
      if (begin_command_buffer_(conv2d_command_buffer_, &persistent_begin) != VK_SUCCESS) return false;
      const VkMemoryBarrier host_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
          VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
      const VkMemoryBarrier shader_to_shader{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
          VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
      cmd_pipeline_barrier_(conv2d_command_buffer_, VK_PIPELINE_STAGE_HOST_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &host_barrier, 0, nullptr, 0, nullptr);
      cmd_bind_pipeline_(conv2d_command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
      cmd_bind_descriptor_sets_(conv2d_command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
          pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);
      if (pool_count > 0) {
        // MaxPool2x2 SAME: k=2 s=1 pad=0, output spatial equals input.
        Push pool{static_cast<std::uint32_t>(pool_count),
                  static_cast<std::uint32_t>(batches), 0u,
                  static_cast<std::uint32_t>(concat_channels[0]),
                  static_cast<std::uint32_t>(ih), static_cast<std::uint32_t>(iw),
                  static_cast<std::uint32_t>(ih), static_cast<std::uint32_t>(iw),
                  2u, 2u, 40u | (1u << 16u) | (1u << 24u)};
        cmd_push_constants_(conv2d_command_buffer_, pipeline_layout_,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pool), &pool);
        cmd_dispatch_(conv2d_command_buffer_,
            static_cast<std::uint32_t>((pool_count + 1023) / 1024),
            static_cast<std::uint32_t>(batches), 1);
        cmd_pipeline_barrier_(conv2d_command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &shader_to_shader, 0, nullptr,
            0, nullptr);
        Push copy{static_cast<std::uint32_t>(pool_count), 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                  61u};
        cmd_push_constants_(conv2d_command_buffer_, pipeline_layout_,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(copy), &copy);
        cmd_dispatch_(conv2d_command_buffer_,
            static_cast<std::uint32_t>((pool_count + 1023) / 1024), 1, 1);
        cmd_pipeline_barrier_(conv2d_command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &shader_to_shader, 0, nullptr,
            0, nullptr);
      }
      cmd_push_constants_(conv2d_command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
          0, sizeof(push), &push);
      cmd_dispatch_(conv2d_command_buffer_, static_cast<std::uint32_t>(dispatch_x),
                    static_cast<std::uint32_t>(batches), 1);
      if (end_command_buffer_(conv2d_command_buffer_) != VK_SUCCESS) return false;
      conv2d_cmd_ready_ = true;
      conv2d_pool_first_ = pool_first;
      conv2d_dispatch_x_ = static_cast<std::uint32_t>(dispatch_x);
      conv2d_batches_ = static_cast<std::uint32_t>(batches);
      conv2d_input_bytes_ = input_bytes;
      conv2d_output_bytes_ = output_bytes;
      conv2d_weight_bytes_ = weight_bytes;
      conv2d_bias_bytes_ = bias_bytes;
      conv2d_mode_ = mode_bits;
      conv2d_weights_ = weights;
      conv2d_bias_ = bias;
    }
    const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
                              1, &conv2d_command_buffer_, 0, nullptr};
    if (reset_fences_(device_, 1, &submission_fence_) != VK_SUCCESS ||
        queue_submit_(queue_, 1, &submit, submission_fence_) != VK_SUCCESS ||
        wait_for_fences_(device_, 1, &submission_fence_, VK_TRUE,
                         std::numeric_limits<std::uint64_t>::max()) != VK_SUCCESS) return false;
    std::memcpy(output, mapped_[3], static_cast<std::size_t>(output_bytes));
    return true;
  }

  bool RunRgbResizeAndConv2d(
      float* output, const std::uint8_t* rgb, std::size_t rgb_bytes,
      int source_width, int source_height, int nchw_width, int nchw_height,
      const float* weights, const float* bias, int output_channels,
      int kernel, int stride, int pad, bool relu, int crop_x = 0, int crop_y = 0,
      int crop_w = 0, int crop_h = 0, bool recognition = false) noexcept {
    if (!output || !rgb || !weights || !bias || rgb_bytes == 0 ||
        source_width <= 0 || source_height <= 0 || nchw_width <= 0 || nchw_height <= 0 ||
        output_channels <= 0 || kernel != 3 || stride <= 0 || pad < 0 ||
        rgb_bytes < std::size_t(source_width) * source_height * 3 || !Initialize()) {
      return false;
    }
    if (crop_w <= 0) crop_w = source_width;
    if (crop_h <= 0) crop_h = source_height;
    if (crop_x < 0 || crop_y < 0 || crop_x + crop_w > source_width ||
        crop_y + crop_h > source_height) {
      return false;
    }
    const int out_h = (nchw_height + 2 * pad - kernel) / stride + 1;
    const int out_w = (nchw_width + 2 * pad - kernel) / stride + 1;
    if (out_h <= 0 || out_w <= 0) return false;
    const auto nchw_elements = std::size_t(3) * nchw_width * nchw_height;
    const auto weight_elements = std::size_t(output_channels) * 3 * kernel * kernel;
    const auto output_elements = std::size_t(output_channels) * out_h * out_w;
    const VkDeviceSize nchw_bytes = static_cast<VkDeviceSize>(nchw_elements) * sizeof(float);
    const VkDeviceSize weight_bytes = static_cast<VkDeviceSize>(weight_elements) * sizeof(float);
    const VkDeviceSize bias_bytes = static_cast<VkDeviceSize>(output_channels) * sizeof(float);
    const VkDeviceSize output_bytes = static_cast<VkDeviceSize>(output_elements) * sizeof(float);
    const VkDeviceSize rgb_store = static_cast<VkDeviceSize>(rgb_bytes);
    // Default: resize A(RGB)->D(NCHW) and Conv.0 swap_ad D->A in one submit.
    // The previous two-fence path host-copied the NCHW plane between submits.
    const bool one_submit =
        std::getenv("PPOCR_DISABLE_VULKAN_RGB_CONV_ONE_SUBMIT") == nullptr;
    std::lock_guard lock(mutex_);
    if (!EnsureCapacity(
            one_submit ? std::max(rgb_store, output_bytes) : std::max(rgb_store, nchw_bytes),
            weight_bytes, bias_bytes,
            one_submit ? nchw_bytes : std::max(nchw_bytes, output_bytes))) {
      return false;
    }
    std::memcpy(mapped_[0], rgb, rgb_bytes);
    BindScratchParameters(weight_bytes, bias_bytes);
    std::memcpy(mapped_[1], weights, static_cast<std::size_t>(weight_bytes));
    std::memcpy(mapped_[2], bias, static_cast<std::size_t>(bias_bytes));
    pointwise_cached_ = false;
    chain_cached_ = false;
    conv2d_cmd_ready_ = false;
    stem_cmd_ready_ = false;
    const auto begin_one_shot = [&]() {
      if (reset_command_buffer_(command_buffer_, 0) != VK_SUCCESS) return false;
      const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
          VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
      return begin_command_buffer_(command_buffer_, &begin) == VK_SUCCESS;
    };
    const auto submit_one_shot = [&]() {
      if (end_command_buffer_(command_buffer_) != VK_SUCCESS) return false;
      const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
                                1, &command_buffer_, 0, nullptr};
      return reset_fences_(device_, 1, &submission_fence_) == VK_SUCCESS &&
          queue_submit_(queue_, 1, &submit, submission_fence_) == VK_SUCCESS &&
          wait_for_fences_(device_, 1, &submission_fence_, VK_TRUE,
                           std::numeric_limits<std::uint64_t>::max()) == VK_SUCCESS;
    };
    const VkMemoryBarrier host_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    const VkMemoryBarrier shader_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_HOST_READ_BIT};
    const VkMemoryBarrier shader_to_shader{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    struct Push {
      std::uint32_t count, batches, operation0, operation1, operation2, operation3, steps,
                    right_repeat, right_per_batch, right_batch_stride, mode;
    } resize_push{static_cast<std::uint32_t>(nchw_elements), 1u,
                  static_cast<std::uint32_t>(source_width),
                  static_cast<std::uint32_t>(source_height),
                  static_cast<std::uint32_t>(crop_x),
                  static_cast<std::uint32_t>(crop_y),
                  static_cast<std::uint32_t>(crop_w),
                  static_cast<std::uint32_t>(crop_h),
                  static_cast<std::uint32_t>(nchw_width),
                  static_cast<std::uint32_t>(nchw_height),
                  recognition ? 47u : 46u};
    const bool use_spatial = kernel == 3 && (stride == 1 || stride == 2) &&
        std::size_t(out_h) * out_w >= 2048 && output_channels <= 16;
    const std::uint32_t conv_mode = use_spatial ? (relu ? 51u : 50u) : (relu ? 15u : 14u);
    const std::uint32_t pad_bits =
        (static_cast<std::uint32_t>(pad) << 16u) | (static_cast<std::uint32_t>(pad) << 24u);
    constexpr std::size_t kTiledMinPlane = 2048u;
    const auto weights_per_channel = std::size_t(3) * kernel * kernel;
    const bool use_tiled = !use_spatial && stride == 1 && weights_per_channel <= 4096 &&
        output_elements / std::size_t(output_channels) >= kTiledMinPlane;
    const bool tile4oc = use_tiled && kernel == 3 && weights_per_channel * 4 <= 4096;
    const auto dispatch_x = use_spatial
        ? ((out_w + 15) / 16) * ((out_h + 15) / 16) * ((output_channels + 3) / 4)
        : (use_tiled
            ? (tile4oc ? ((output_channels + 3) / 4) : output_channels) *
                  ((out_h * out_w + 1023) / 1024)
            : (output_elements + 1023) / 1024);
    if (dispatch_x == 0) return false;
    const auto bind_compute = [&]() {
      cmd_bind_pipeline_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
      cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
          0, 1, &descriptor_set_, 0, nullptr);
    };
    if (one_submit) {
      Push conv_push{static_cast<std::uint32_t>(output_elements), 1u, 3u,
                     static_cast<std::uint32_t>(nchw_height),
                     static_cast<std::uint32_t>(nchw_width),
                     static_cast<std::uint32_t>(out_h),
                     static_cast<std::uint32_t>(out_w),
                     static_cast<std::uint32_t>(kernel),
                     static_cast<std::uint32_t>(kernel),
                     static_cast<std::uint32_t>(stride),
                     conv_mode | 0x8000u | pad_bits};
      const bool persist =
          std::getenv("PPOCR_DISABLE_VULKAN_RGB_CONV_PERSIST") == nullptr &&
          stem_command_buffer_ != VK_NULL_HANDLE;
      const bool reuse = persist && rgb_conv_cmd_ready_ &&
          rgb_conv_sw_ == source_width && rgb_conv_sh_ == source_height &&
          rgb_conv_nw_ == nchw_width && rgb_conv_nh_ == nchw_height &&
          rgb_conv_oc_ == output_channels && rgb_conv_stride_ == stride &&
          rgb_conv_pad_ == pad && rgb_conv_relu_ == relu &&
          rgb_conv_recognition_ == recognition && rgb_conv_weights_ == weights &&
          rgb_conv_crop_x_ == crop_x && rgb_conv_crop_y_ == crop_y &&
          rgb_conv_crop_w_ == crop_w && rgb_conv_crop_h_ == crop_h;
      if (!reuse) {
        if (!persist) {
          if (!begin_one_shot()) return false;
          cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_HOST_BIT,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &host_barrier, 0, nullptr, 0, nullptr);
          bind_compute();
          cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
              0, sizeof(resize_push), &resize_push);
          cmd_dispatch_(command_buffer_,
              static_cast<std::uint32_t>((nchw_elements + 1023) / 1024), 1, 1);
          cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &shader_to_shader, 0, nullptr, 0,
              nullptr);
          cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
              0, sizeof(conv_push), &conv_push);
          cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch_x), 1, 1);
          cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &shader_barrier, 0, nullptr, 0, nullptr);
          if (!submit_one_shot()) return false;
          std::memcpy(output, mapped_[0], static_cast<std::size_t>(output_bytes));
          return true;
        }
        if (reset_command_buffer_(stem_command_buffer_, 0) != VK_SUCCESS) return false;
        const VkCommandBufferBeginInfo persistent_begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            nullptr, 0, nullptr};
        if (begin_command_buffer_(stem_command_buffer_, &persistent_begin) != VK_SUCCESS)
          return false;
        cmd_pipeline_barrier_(stem_command_buffer_, VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &host_barrier, 0, nullptr, 0, nullptr);
        cmd_bind_pipeline_(stem_command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        cmd_bind_descriptor_sets_(stem_command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);
        cmd_push_constants_(stem_command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(resize_push), &resize_push);
        cmd_dispatch_(stem_command_buffer_,
            static_cast<std::uint32_t>((nchw_elements + 1023) / 1024), 1, 1);
        cmd_pipeline_barrier_(stem_command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &shader_to_shader, 0, nullptr, 0,
            nullptr);
        cmd_push_constants_(stem_command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(conv_push), &conv_push);
        cmd_dispatch_(stem_command_buffer_, static_cast<std::uint32_t>(dispatch_x), 1, 1);
        cmd_pipeline_barrier_(stem_command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &shader_barrier, 0, nullptr, 0, nullptr);
        if (end_command_buffer_(stem_command_buffer_) != VK_SUCCESS) return false;
        rgb_conv_cmd_ready_ = true;
        rgb_conv_sw_ = source_width; rgb_conv_sh_ = source_height;
        rgb_conv_nw_ = nchw_width; rgb_conv_nh_ = nchw_height;
        rgb_conv_oc_ = output_channels; rgb_conv_stride_ = stride;
        rgb_conv_pad_ = pad; rgb_conv_relu_ = relu;
        rgb_conv_recognition_ = recognition; rgb_conv_weights_ = weights;
        rgb_conv_crop_x_ = crop_x; rgb_conv_crop_y_ = crop_y;
        rgb_conv_crop_w_ = crop_w; rgb_conv_crop_h_ = crop_h;
      }
      const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
                                1, &stem_command_buffer_, 0, nullptr};
      if (reset_fences_(device_, 1, &submission_fence_) != VK_SUCCESS ||
          queue_submit_(queue_, 1, &submit, submission_fence_) != VK_SUCCESS ||
          wait_for_fences_(device_, 1, &submission_fence_, VK_TRUE,
                           std::numeric_limits<std::uint64_t>::max()) != VK_SUCCESS)
        return false;
      std::memcpy(output, mapped_[0], static_cast<std::size_t>(output_bytes));
      return true;
    }
    if (!begin_one_shot()) return false;
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &host_barrier, 0, nullptr, 0, nullptr);
    bind_compute();
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(resize_push), &resize_push);
    cmd_dispatch_(command_buffer_,
        static_cast<std::uint32_t>((nchw_elements + 1023) / 1024), 1, 1);
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &shader_barrier, 0, nullptr, 0, nullptr);
    if (!submit_one_shot()) return false;
    std::memcpy(mapped_[0], mapped_[3], static_cast<std::size_t>(nchw_bytes));
    Push conv_push{static_cast<std::uint32_t>(output_elements), 1u, 3u,
                   static_cast<std::uint32_t>(nchw_height),
                   static_cast<std::uint32_t>(nchw_width),
                   static_cast<std::uint32_t>(out_h),
                   static_cast<std::uint32_t>(out_w),
                   static_cast<std::uint32_t>(kernel),
                   static_cast<std::uint32_t>(kernel),
                   static_cast<std::uint32_t>(stride), conv_mode | pad_bits};
    if (!begin_one_shot()) return false;
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &host_barrier, 0, nullptr, 0, nullptr);
    bind_compute();
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(conv_push), &conv_push);
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch_x), 1, 1);
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &shader_barrier, 0, nullptr, 0, nullptr);
    if (!submit_one_shot()) return false;
    std::memcpy(output, mapped_[3], static_cast<std::size_t>(output_bytes));
    return true;
  }

  bool RunRgbStem(
      float* output, const std::uint8_t* rgb, std::size_t rgb_bytes,
      int source_width, int source_height, int nchw_width, int nchw_height,
      const VulkanStemLayer& conv0, const VulkanStemLayer& conv1,
      const VulkanStemLayer& conv2, const VulkanStemLayer* stem_conv,
      float* conv2_out = nullptr) noexcept {
    const bool full_stem = stem_conv != nullptr;
    if (!output || !rgb || rgb_bytes == 0 || !Initialize()) return false;
    if (!conv0.weights || !conv0.bias || !conv1.weights || !conv1.bias ||
        !conv2.weights || !conv2.bias) return false;
    if (full_stem && (!stem_conv->weights || !stem_conv->bias)) return false;
    if (!full_stem && !conv2_out) return false;
    if (conv0.kernel != 3 || conv0.stride != 2 || conv0.input_channels != 3) return false;
    if (conv1.kernel != 2 || conv1.stride != 1 || conv2.kernel != 2 || conv2.stride != 1)
      return false;
    if (full_stem && (stem_conv->kernel != 3 || stem_conv->stride != 2)) return false;
    const int c0_h = (nchw_height + 2 * conv0.pad - conv0.kernel) / conv0.stride + 1;
    const int c0_w = (nchw_width + 2 * conv0.pad - conv0.kernel) / conv0.stride + 1;
    if (c0_h <= 0 || c0_w <= 0) return false;
    const int oc0 = conv0.output_channels, oc1 = conv1.output_channels, oc2 = conv2.output_channels;
    const int ocs = full_stem ? stem_conv->output_channels : 0;
    if (oc0 <= 0 || oc1 <= 0 || oc2 <= 0) return false;
    if (full_stem && ocs <= 0) return false;
    if (conv1.input_channels != oc0 || conv2.input_channels != oc1) return false;
    if (full_stem && stem_conv->input_channels != oc0 + oc2) return false;
    int stem_h = 0, stem_w = 0;
    if (full_stem) {
      stem_h = (c0_h + 2 * stem_conv->pad - stem_conv->kernel) / stem_conv->stride + 1;
      stem_w = (c0_w + 2 * stem_conv->pad - stem_conv->kernel) / stem_conv->stride + 1;
      if (stem_h <= 0 || stem_w <= 0) return false;
    }
    const auto nchw_n = std::size_t(3) * nchw_width * nchw_height;
    const auto c0_n = std::size_t(oc0) * c0_h * c0_w;
    const auto c1_n = std::size_t(oc1) * c0_h * c0_w;
    const auto c2_n = std::size_t(oc2) * c0_h * c0_w;
    const auto stem_n = full_stem ? std::size_t(ocs) * stem_h * stem_w : std::size_t{0};
    const VkDeviceSize nchw_b = static_cast<VkDeviceSize>(nchw_n) * sizeof(float);
    const VkDeviceSize c0_b = static_cast<VkDeviceSize>(c0_n) * sizeof(float);
    const VkDeviceSize c1_b = static_cast<VkDeviceSize>(c1_n) * sizeof(float);
    const VkDeviceSize c2_b = static_cast<VkDeviceSize>(c2_n) * sizeof(float);
    const VkDeviceSize stem_b = static_cast<VkDeviceSize>(stem_n) * sizeof(float);
    const VkDeviceSize act_b = std::max({nchw_b, c0_b, c1_b, c2_b, stem_b,
                                         static_cast<VkDeviceSize>(rgb_bytes)});
    const std::size_t w0 = std::size_t(oc0) * 3 * 9;
    const std::size_t w1 = std::size_t(oc1) * std::size_t(oc0) * 4;
    const std::size_t w2 = std::size_t(oc2) * std::size_t(oc1) * 4;
    const std::size_t ws = full_stem ? std::size_t(ocs) * std::size_t(oc0 + oc2) * 9 : 0;
    const std::size_t packed_w = w0 + w1 + w2 + ws;
    const std::size_t packed_bias = std::size_t(oc0 + oc1 + oc2 + ocs);
    const VkDeviceSize packed_w_b = static_cast<VkDeviceSize>(packed_w) * sizeof(float);
    const VkDeviceSize packed_bias_b = static_cast<VkDeviceSize>(packed_bias) * sizeof(float);
    const VkDeviceSize save_b = std::max(c0_b, stem_b);
    std::lock_guard lock(mutex_);
    if (!EnsureCapacity(act_b, packed_w_b, packed_bias_b, act_b, save_b)) return false;
    std::memcpy(mapped_[0], rgb, rgb_bytes);
    const float* stem_weights = full_stem ? stem_conv->weights : nullptr;
    const bool reuse_stem = stem_cmd_ready_ && stem_command_buffer_ &&
        stem_src_w_ == source_width && stem_src_h_ == source_height &&
        stem_nchw_w_ == nchw_width && stem_nchw_h_ == nchw_height &&
        stem_w0_ == conv0.weights && stem_w1_ == conv1.weights &&
        stem_w2_ == conv2.weights && stem_ws_ == stem_weights &&
        stem_full_ == full_stem;
    if (!reuse_stem) {
    auto* packed_weights = static_cast<float*>(mapped_[1]);
    auto* packed_biases = static_cast<float*>(mapped_[2]);
    std::memcpy(packed_weights, conv0.weights, w0 * sizeof(float));
    std::memcpy(packed_weights + w0, conv1.weights, w1 * sizeof(float));
    std::memcpy(packed_weights + w0 + w1, conv2.weights, w2 * sizeof(float));
    if (full_stem)
      std::memcpy(packed_weights + w0 + w1 + w2, stem_conv->weights, ws * sizeof(float));
    std::memcpy(packed_biases, conv0.bias, std::size_t(oc0) * sizeof(float));
    std::memcpy(packed_biases + oc0, conv1.bias, std::size_t(oc1) * sizeof(float));
    std::memcpy(packed_biases + oc0 + oc1, conv2.bias, std::size_t(oc2) * sizeof(float));
    if (full_stem)
      std::memcpy(packed_biases + oc0 + oc1 + oc2, stem_conv->bias, std::size_t(ocs) * sizeof(float));
    BindScratchParameters(packed_w_b, packed_bias_b);
    pointwise_cached_ = false;
    chain_cached_ = false;
    conv2d_cmd_ready_ = false;
    rgb_conv_cmd_ready_ = false;
    if (!stem_command_buffer_ || reset_command_buffer_(stem_command_buffer_, 0) != VK_SUCCESS)
      return false;
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
        0, nullptr};
    if (begin_command_buffer_(stem_command_buffer_, &begin) != VK_SUCCESS) return false;
    cmd_bind_pipeline_(stem_command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    cmd_bind_descriptor_sets_(stem_command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
        0, 1, &descriptor_set_, 0, nullptr);
    const VkMemoryBarrier host_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT};
    const VkMemoryBarrier shader_to_host{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
    const VkMemoryBarrier shader_to_shader{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    struct Push {
      std::uint32_t count, batches, operation0, operation1, operation2, operation3, steps,
                    right_repeat, right_per_batch, right_batch_stride, mode;
    };
    const auto dispatch_conv = [&](int ic, int oc, int ih, int iw, int oh, int ow, int kh, int kw,
                                   int stride, int pad, bool relu, bool swap_ad,
                                   std::uint32_t w_off, std::uint32_t b_off) -> bool {
      const std::size_t out_n = std::size_t(oc) * oh * ow;
      const bool conv2x2 = kh == 2 && kw == 2 && stride == 1 && pad == 0;
      const bool spatial = !conv2x2 && kh == 3 && kw == 3 && oh * ow >= 2048 &&
          ((stride == 1 && ic <= 64) || (stride == 2 && ic <= 8));
      const std::uint32_t conv_mode = conv2x2 ? (relu ? 59u : 58u)
          : (spatial ? (relu ? 51u : 50u) : (relu ? 15u : 14u));
      const std::uint32_t mode_bits = conv_mode | (swap_ad ? 0x8000u : 0u) |
          (std::uint32_t(pad) << 16u) | (std::uint32_t(pad) << 24u);
      // Stem Conv.0 on 32x48 (plane 1536) took the shader's generic 14/15
      // fallback while the host used tiled 8-wide dispatch, leaving the
      // tail of the C0 plane stale. Always match generic vec4 coverage
      // unless the dedicated spatial / 2x2 kernels own the launch.
      const auto dispatch_x = (spatial || conv2x2)
          ? ((ow + 15) / 16) * ((oh + 15) / 16) * ((oc + 3) / 4)
          : int((out_n + 1023) / 1024);
      if (dispatch_x <= 0) return false;
      Push push{static_cast<std::uint32_t>(out_n), 1u, static_cast<std::uint32_t>(ic),
                static_cast<std::uint32_t>(ih), static_cast<std::uint32_t>(iw),
                static_cast<std::uint32_t>(oh), static_cast<std::uint32_t>(ow),
                conv2x2 ? w_off : static_cast<std::uint32_t>(kh),
                conv2x2 ? b_off : static_cast<std::uint32_t>(kw),
                static_cast<std::uint32_t>(stride), mode_bits};
      cmd_push_constants_(stem_command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
          0, sizeof(push), &push);
      cmd_dispatch_(stem_command_buffer_, static_cast<std::uint32_t>(dispatch_x), 1, 1);
      return true;
    };
    cmd_pipeline_barrier_(stem_command_buffer_, VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
        &host_barrier, 0, nullptr, 0, nullptr);
    Push resize{static_cast<std::uint32_t>(nchw_n), 1u,
                static_cast<std::uint32_t>(source_width),
                static_cast<std::uint32_t>(source_height), 0u, 0u,
                static_cast<std::uint32_t>(source_width),
                static_cast<std::uint32_t>(source_height),
                static_cast<std::uint32_t>(nchw_width),
                static_cast<std::uint32_t>(nchw_height), 46u};
    cmd_push_constants_(stem_command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(resize), &resize);
    cmd_dispatch_(stem_command_buffer_, static_cast<std::uint32_t>((nchw_n + 1023) / 1024), 1, 1);
    cmd_pipeline_barrier_(stem_command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &shader_to_shader, 0, nullptr, 0, nullptr);
    const auto dispatch_copy = [&](std::uint32_t count, std::uint32_t mode) {
      Push copy{count, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, mode};
      cmd_push_constants_(stem_command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
          0, sizeof(copy), &copy);
      cmd_dispatch_(stem_command_buffer_, (count + 1023u) / 1024u, 1, 1);
    };
    if (!full_stem) {
      // Same resident layout as Conv.0-only: resize wrote D, Conv.0 swap_ad
      // writes A, one copy parks C0 in E, Conv.1/2 stay on device. Drops the
      // old NCHW host-style copy 61 pair (two extra dispatches + a fence).
      if (!dispatch_conv(3, oc0, nchw_height, nchw_width, c0_h, c0_w, conv0.kernel, conv0.kernel,
                         conv0.stride, conv0.pad, conv0.relu, true, 0u, 0u))
        return false;
      cmd_pipeline_barrier_(stem_command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &shader_to_shader, 0, nullptr, 0, nullptr);
      dispatch_copy(static_cast<std::uint32_t>(c0_n), 60u);
      cmd_pipeline_barrier_(stem_command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &shader_to_shader, 0, nullptr, 0, nullptr);
      if (!dispatch_conv(oc0, oc1, c0_h, c0_w, c0_h, c0_w, conv1.kernel, conv1.kernel,
                         conv1.stride, conv1.pad, conv1.relu, false,
                         static_cast<std::uint32_t>(w0), static_cast<std::uint32_t>(oc0)))
        return false;
      cmd_pipeline_barrier_(stem_command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &shader_to_shader, 0, nullptr, 0, nullptr);
      if (!dispatch_conv(oc1, oc2, c0_h, c0_w, c0_h, c0_w, conv2.kernel, conv2.kernel,
                         conv2.stride, conv2.pad, conv2.relu, true,
                         static_cast<std::uint32_t>(w0 + w1),
                         static_cast<std::uint32_t>(oc0 + oc1)))
        return false;
    } else if (!dispatch_conv(3, oc0, nchw_height, nchw_width, c0_h, c0_w, conv0.kernel,
                              conv0.kernel, conv0.stride, conv0.pad, conv0.relu, true, 0u, 0u))
      return false;
    if (full_stem) {
    cmd_pipeline_barrier_(stem_command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &shader_to_shader, 0, nullptr, 0, nullptr);
    Push save_c0{static_cast<std::uint32_t>(c0_n), 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 60u};
    cmd_push_constants_(stem_command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(save_c0), &save_c0);
    cmd_dispatch_(stem_command_buffer_,
        static_cast<std::uint32_t>((c0_n + 1023) / 1024), 1, 1);
    cmd_pipeline_barrier_(stem_command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &shader_to_shader, 0, nullptr, 0, nullptr);
    if (!dispatch_conv(oc0, oc1, c0_h, c0_w, c0_h, c0_w, conv1.kernel, conv1.kernel, conv1.stride,
                       conv1.pad, conv1.relu, false, static_cast<std::uint32_t>(w0),
                       static_cast<std::uint32_t>(oc0)))
      return false;
    cmd_pipeline_barrier_(stem_command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &shader_to_shader, 0, nullptr, 0, nullptr);
    if (!dispatch_conv(oc1, oc2, c0_h, c0_w, c0_h, c0_w, conv2.kernel, conv2.kernel, conv2.stride,
                       conv2.pad, conv2.relu, true, static_cast<std::uint32_t>(w0 + w1),
                       static_cast<std::uint32_t>(oc0 + oc1)))
      return false;
    cmd_pipeline_barrier_(stem_command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &shader_to_shader, 0, nullptr, 0, nullptr);
    Push pool{static_cast<std::uint32_t>(c0_n), 1u, 0u, static_cast<std::uint32_t>(oc0),
              static_cast<std::uint32_t>(c0_h), static_cast<std::uint32_t>(c0_w),
              static_cast<std::uint32_t>(c0_h), static_cast<std::uint32_t>(c0_w), 2u, 2u,
              57u | (1u << 16u) | (1u << 24u)};
    cmd_push_constants_(stem_command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(pool), &pool);
    cmd_dispatch_(stem_command_buffer_, static_cast<std::uint32_t>((c0_n + 1023) / 1024), 1, 1);
    cmd_pipeline_barrier_(stem_command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &shader_to_shader, 0, nullptr, 0, nullptr);
    const std::uint32_t stem_mode = 56u | (std::uint32_t(stem_conv->pad) << 16u) |
        (std::uint32_t(stem_conv->pad) << 24u);
    const auto stem_groups = ((stem_h * stem_w + 1023) / 1024) * ((ocs + 3) / 4);
    if (stem_groups <= 0) return false;
    Push stem{static_cast<std::uint32_t>(stem_n), 1u, static_cast<std::uint32_t>(oc0),
              static_cast<std::uint32_t>(oc2), static_cast<std::uint32_t>(c0_h),
              static_cast<std::uint32_t>(c0_w), static_cast<std::uint32_t>(stem_h),
              static_cast<std::uint32_t>(stem_w), static_cast<std::uint32_t>(w0 + w1 + w2),
              static_cast<std::uint32_t>(oc0 + oc1 + oc2), stem_mode};
    cmd_push_constants_(stem_command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(stem), &stem);
    cmd_dispatch_(stem_command_buffer_, static_cast<std::uint32_t>(stem_groups), 1, 1);
    }
    cmd_pipeline_barrier_(stem_command_buffer_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &shader_to_host, 0, nullptr, 0, nullptr);
    if (end_command_buffer_(stem_command_buffer_) != VK_SUCCESS) return false;
    stem_cmd_ready_ = true;
    stem_src_w_ = source_width; stem_src_h_ = source_height;
    stem_nchw_w_ = nchw_width; stem_nchw_h_ = nchw_height;
    stem_w0_ = conv0.weights; stem_w1_ = conv1.weights;
    stem_w2_ = conv2.weights; stem_ws_ = stem_weights;
    stem_full_ = full_stem;
    }
    BindScratchParameters(packed_w_b, packed_bias_b);
    const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
                              1, &stem_command_buffer_, 0, nullptr};
    if (reset_fences_(device_, 1, &submission_fence_) != VK_SUCCESS ||
        queue_submit_(queue_, 1, &submit, submission_fence_) != VK_SUCCESS ||
        wait_for_fences_(device_, 1, &submission_fence_, VK_TRUE,
                         std::numeric_limits<std::uint64_t>::max()) != VK_SUCCESS)
      return false;
    if (full_stem) {
      std::memcpy(output, mapped_[4], static_cast<std::size_t>(stem_b));
    } else {
      std::memcpy(output, mapped_[4], static_cast<std::size_t>(c0_b));
      std::memcpy(conv2_out, mapped_[0], static_cast<std::size_t>(c2_b));
    }
    return true;
  }

  bool RunConvTranspose2x2(float* output, const float* input, const float* weights,
                           const float* bias, std::size_t batches, int input_channels,
                           int output_channels, int input_height, int input_width,
                           bool immutable_parameters) noexcept {
    if (!output || !input || !weights || !bias || batches == 0 || input_channels <= 0 ||
        output_channels <= 0 || input_height <= 0 || input_width <= 0 ||
        batches > UINT32_MAX || !Initialize()) return false;
    const auto in_channels = static_cast<std::size_t>(input_channels);
    const auto out_channels = static_cast<std::size_t>(output_channels);
    const auto ih = static_cast<std::size_t>(input_height);
    const auto iw = static_cast<std::size_t>(input_width);
    if (in_channels > UINT32_MAX || out_channels > UINT32_MAX || ih > UINT32_MAX ||
        iw > UINT32_MAX || ih > UINT32_MAX / iw || in_channels > UINT32_MAX / (ih * iw) ||
        out_channels > UINT32_MAX / (ih * iw * 4) ||
        batches > UINT32_MAX / (out_channels * ih * iw * 4) ||
        out_channels > std::numeric_limits<std::size_t>::max() / (in_channels * 4)) return false;
    const auto input_per_batch = in_channels * ih * iw;
    const auto output_per_batch = out_channels * ih * iw * 4;
    const auto weight_elements = in_channels * out_channels * 4;
    // Mode 17 cooperatively stages one [Cin,2,2] filter row in workgroup
    // memory. On smaller feature maps its setup/barrier cost can exceed the
    // saved global-weight reads, while FPN's larger maps amortize it well.
    // Keep the threshold explicit and deployment-tunable; mode 16 remains
    // numerically identical and is the safe generic path.
    constexpr std::size_t kTransposeTiledMinimumOutputPlane = 16u * 1024u;
    const bool use_tiled = in_channels * 4 <= 4096 &&
        output_per_batch / out_channels >= kTransposeTiledMinimumOutputPlane &&
        std::getenv("PPOCR_DISABLE_VULKAN_CONVTRANSPOSE_TILE") == nullptr;
    std::lock_guard lock(mutex_);
    const VkDeviceSize input_bytes = static_cast<VkDeviceSize>(batches) * input_per_batch * sizeof(float);
    const VkDeviceSize output_bytes = static_cast<VkDeviceSize>(batches) * output_per_batch * sizeof(float);
    const VkDeviceSize weight_bytes = static_cast<VkDeviceSize>(weight_elements) * sizeof(float);
    const VkDeviceSize bias_bytes = static_cast<VkDeviceSize>(out_channels) * sizeof(float);
    if (!EnsureCapacity(input_bytes, weight_bytes, bias_bytes, output_bytes)) return false;
    std::memcpy(mapped_[0], input, static_cast<std::size_t>(input_bytes));
    BindScratchParameters(weight_bytes, bias_bytes);
    const bool reuse_parameters = immutable_parameters && pointwise_cached_ &&
        pointwise_weight_bytes_ == weight_bytes && pointwise_bias_bytes_ == bias_bytes &&
        pointwise_cached_weights_ == weights && pointwise_cached_bias_ == bias;
    if (!reuse_parameters) {
      std::memcpy(mapped_[1], weights, static_cast<std::size_t>(weight_bytes));
      std::memcpy(mapped_[2], bias, static_cast<std::size_t>(bias_bytes));
      chain_cached_ = false;
      pointwise_cached_ = immutable_parameters;
      pointwise_weight_bytes_ = immutable_parameters ? weight_bytes : 0;
      pointwise_bias_bytes_ = immutable_parameters ? bias_bytes : 0;
      pointwise_cached_weights_ = immutable_parameters ? weights : nullptr;
      pointwise_cached_bias_ = immutable_parameters ? bias : nullptr;
      right_cached_ = false;
      coefficients_cached_ = false;
    }
    if (reset_command_buffer_(command_buffer_, 0) != VK_SUCCESS) return false;
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                         VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    if (begin_command_buffer_(command_buffer_, &begin) != VK_SUCCESS) return false;
    cmd_bind_pipeline_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
                              0, 1, &descriptor_set_, 0, nullptr);
    struct Push {
      std::uint32_t count, batches, operation0, operation1, operation2, operation3, steps,
                    right_repeat, right_per_batch, right_batch_stride, mode;
    } push{static_cast<std::uint32_t>(output_per_batch), static_cast<std::uint32_t>(batches),
           static_cast<std::uint32_t>(in_channels), static_cast<std::uint32_t>(out_channels),
           static_cast<std::uint32_t>(ih), static_cast<std::uint32_t>(iw),
           0u, 0u, 0u, 0u,
           static_cast<std::uint32_t>(use_tiled ? 17u : 16u)};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                        0, sizeof(push), &push);
    // Mode 17 gives one workgroup a compact [Cin,2,2] row for one output
    // channel. Mode 16 remains the safe generic fallback for unusually wide
    // input-channel exports. The shader consumes four values per invocation
    // and has 256 invocations per workgroup.
    const auto dispatch_x = use_tiled
        ? out_channels * ((output_per_batch / out_channels + 1023) / 1024)
        : (output_per_batch + 1023) / 1024;
    if (dispatch_x == 0 || dispatch_x > UINT32_MAX) return false;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch_x),
                  static_cast<std::uint32_t>(batches), 1);
    if (end_command_buffer_(command_buffer_) != VK_SUCCESS) return false;
    const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
                              1, &command_buffer_, 0, nullptr};
    if (reset_fences_(device_, 1, &submission_fence_) != VK_SUCCESS ||
        queue_submit_(queue_, 1, &submit, submission_fence_) != VK_SUCCESS ||
        wait_for_fences_(device_, 1, &submission_fence_, VK_TRUE,
                         std::numeric_limits<std::uint64_t>::max()) != VK_SUCCESS) return false;
    std::memcpy(output, mapped_[3], static_cast<std::size_t>(output_bytes));
    return true;
  }

  bool RunConvTranspose2x2Add(float* output, const float* input, const float* weights,
                              const float* bias, const float* residual,
                              std::size_t batches, int input_channels,
                              int output_channels, int input_height, int input_width,
                              bool immutable_parameters) noexcept {
    if (!output || !input || !weights || !bias || !residual || batches == 0 ||
        input_channels <= 0 || output_channels <= 0 || input_height <= 0 ||
        input_width <= 0 || batches > UINT32_MAX || !Initialize()) return false;
    const auto in_channels = static_cast<std::size_t>(input_channels);
    const auto out_channels = static_cast<std::size_t>(output_channels);
    const auto ih = static_cast<std::size_t>(input_height);
    const auto iw = static_cast<std::size_t>(input_width);
    if (in_channels > UINT32_MAX || out_channels > UINT32_MAX || ih > UINT32_MAX ||
        iw > UINT32_MAX || ih > UINT32_MAX / iw || in_channels > UINT32_MAX / (ih * iw) ||
        out_channels > UINT32_MAX / (ih * iw * 4) ||
        batches > UINT32_MAX / (out_channels * ih * iw * 4) ||
        out_channels > std::numeric_limits<std::size_t>::max() / (in_channels * 4)) return false;
    const auto input_per_batch = in_channels * ih * iw;
    const auto output_per_batch = out_channels * ih * iw * 4;
    const auto weight_elements = in_channels * out_channels * 4;
    constexpr std::size_t kTransposeTiledMinimumOutputPlane = 16u * 1024u;
    const bool use_tiled = in_channels * 4 <= 4096 &&
        output_per_batch / out_channels >= kTransposeTiledMinimumOutputPlane &&
        std::getenv("PPOCR_DISABLE_VULKAN_CONVTRANSPOSE_TILE") == nullptr;
    std::lock_guard lock(mutex_);
    const VkDeviceSize input_bytes = static_cast<VkDeviceSize>(batches) * input_per_batch * sizeof(float);
    const VkDeviceSize output_bytes = static_cast<VkDeviceSize>(batches) * output_per_batch * sizeof(float);
    const VkDeviceSize weight_bytes = static_cast<VkDeviceSize>(weight_elements) * sizeof(float);
    const VkDeviceSize bias_bytes = static_cast<VkDeviceSize>(out_channels) * sizeof(float);
    if (!EnsureCapacity(output_bytes, weight_bytes, bias_bytes, output_bytes, output_bytes)) return false;
    std::memcpy(mapped_[0], input, static_cast<std::size_t>(input_bytes));
    std::memcpy(mapped_[4], residual, static_cast<std::size_t>(output_bytes));
    BindScratchParameters(weight_bytes, bias_bytes);
    const bool reuse_parameters = immutable_parameters && pointwise_cached_ &&
        pointwise_weight_bytes_ == weight_bytes && pointwise_bias_bytes_ == bias_bytes &&
        pointwise_cached_weights_ == weights && pointwise_cached_bias_ == bias;
    if (!reuse_parameters) {
      std::memcpy(mapped_[1], weights, static_cast<std::size_t>(weight_bytes));
      std::memcpy(mapped_[2], bias, static_cast<std::size_t>(bias_bytes));
      chain_cached_ = false;
      pointwise_cached_ = immutable_parameters;
      pointwise_weight_bytes_ = immutable_parameters ? weight_bytes : 0;
      pointwise_bias_bytes_ = immutable_parameters ? bias_bytes : 0;
      pointwise_cached_weights_ = immutable_parameters ? weights : nullptr;
      pointwise_cached_bias_ = immutable_parameters ? bias : nullptr;
      right_cached_ = false;
      coefficients_cached_ = false;
    }
    if (reset_command_buffer_(command_buffer_, 0) != VK_SUCCESS) return false;
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                         VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    if (begin_command_buffer_(command_buffer_, &begin) != VK_SUCCESS) return false;
    cmd_bind_pipeline_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
                              0, 1, &descriptor_set_, 0, nullptr);
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; };
    Push transpose{static_cast<std::uint32_t>(output_per_batch), static_cast<std::uint32_t>(batches),
                   static_cast<std::uint32_t>(in_channels), static_cast<std::uint32_t>(out_channels),
                   static_cast<std::uint32_t>(ih), static_cast<std::uint32_t>(iw),
                   0u, 0u, 0u, 0u, static_cast<std::uint32_t>(use_tiled ? 17u : 16u)};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                        0, sizeof(transpose), &transpose);
    const auto transpose_dispatch_x = use_tiled
        ? out_channels * ((output_per_batch / out_channels + 1023) / 1024)
        : (output_per_batch + 1023) / 1024;
    if (transpose_dispatch_x == 0 || transpose_dispatch_x > UINT32_MAX) return false;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(transpose_dispatch_x),
                  static_cast<std::uint32_t>(batches), 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier,
                          0, nullptr, 0, nullptr);
    Push add{static_cast<std::uint32_t>(output_per_batch), static_cast<std::uint32_t>(batches),
             0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 31u};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                        0, sizeof(add), &add);
    const auto add_dispatch_x = (output_per_batch + 1023) / 1024;
    if (add_dispatch_x == 0 || add_dispatch_x > UINT32_MAX) return false;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(add_dispatch_x),
                  static_cast<std::uint32_t>(batches), 1);
    if (end_command_buffer_(command_buffer_) != VK_SUCCESS) return false;
    const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
                              1, &command_buffer_, 0, nullptr};
    if (reset_fences_(device_, 1, &submission_fence_) != VK_SUCCESS ||
        queue_submit_(queue_, 1, &submit, submission_fence_) != VK_SUCCESS ||
        wait_for_fences_(device_, 1, &submission_fence_, VK_TRUE,
                         std::numeric_limits<std::uint64_t>::max()) != VK_SUCCESS) return false;
    std::memcpy(output, mapped_[0], static_cast<std::size_t>(output_bytes));
    return true;
  }

  bool RunNearestResize2xAdd(float* output, const float* source, const float* residual,
                             std::size_t batches, int channels, int input_height,
                             int input_width) noexcept {
    if (!output || !source || !residual || batches == 0 || channels <= 0 ||
        input_height <= 0 || input_width <= 0 || batches > UINT32_MAX || !Initialize()) return false;
    const auto c = static_cast<std::size_t>(channels);
    const auto ih = static_cast<std::size_t>(input_height);
    const auto iw = static_cast<std::size_t>(input_width);
    if (c > UINT32_MAX || ih > UINT32_MAX || iw > UINT32_MAX || ih > UINT32_MAX / iw ||
        c > UINT32_MAX / (ih * iw) || c > UINT32_MAX / (ih * iw * 4) ||
        batches > UINT32_MAX / (c * ih * iw * 4)) return false;
    const auto input_per_batch = c * ih * iw;
    const auto output_per_batch = input_per_batch * 4;
    std::lock_guard lock(mutex_);
    const VkDeviceSize source_bytes = static_cast<VkDeviceSize>(batches) * input_per_batch * sizeof(float);
    const VkDeviceSize output_bytes = static_cast<VkDeviceSize>(batches) * output_per_batch * sizeof(float);
    if (!EnsureCapacity(source_bytes, sizeof(float), sizeof(float), output_bytes, output_bytes)) return false;
    BindScratchParameters(sizeof(float), sizeof(float));
    std::memcpy(mapped_[0], source, static_cast<std::size_t>(source_bytes));
    std::memcpy(mapped_[4], residual, static_cast<std::size_t>(output_bytes));
    chain_cached_ = false;
    if (reset_command_buffer_(command_buffer_, 0) != VK_SUCCESS) return false;
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                         VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    if (begin_command_buffer_(command_buffer_, &begin) != VK_SUCCESS) return false;
    cmd_bind_pipeline_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
                              0, 1, &descriptor_set_, 0, nullptr);
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; }
        push{static_cast<std::uint32_t>(output_per_batch), static_cast<std::uint32_t>(batches),
             0, 0, 0, 0, 1, static_cast<std::uint32_t>(ih), static_cast<std::uint32_t>(iw),
             static_cast<std::uint32_t>(c), 18u};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                        0, sizeof(push), &push);
    const auto dispatch_x = (output_per_batch + 1023) / 1024;
    if (dispatch_x == 0 || dispatch_x > UINT32_MAX) return false;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch_x),
                  static_cast<std::uint32_t>(batches), 1);
    if (end_command_buffer_(command_buffer_) != VK_SUCCESS) return false;
    const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
                              1, &command_buffer_, 0, nullptr};
    if (reset_fences_(device_, 1, &submission_fence_) != VK_SUCCESS ||
        queue_submit_(queue_, 1, &submit, submission_fence_) != VK_SUCCESS ||
        wait_for_fences_(device_, 1, &submission_fence_, VK_TRUE,
                         std::numeric_limits<std::uint64_t>::max()) != VK_SUCCESS) return false;
    std::memcpy(output, mapped_[3], static_cast<std::size_t>(output_bytes));
    return true;
  }

  bool RunGemm(float* output, const float* left, const float* right, const float* bias,
               int rows, int depth, int columns, bool immutable_parameters,
               bool swish) noexcept {
    if (!output || !left || !right || rows <= 0 || depth <= 0 || columns <= 0 || !Initialize()) return false;
    const auto r=static_cast<std::size_t>(rows), k=static_cast<std::size_t>(depth), c=static_cast<std::size_t>(columns);
    if (r > UINT32_MAX || k > UINT32_MAX || c > UINT32_MAX || r > UINT32_MAX / c ||
        r > std::numeric_limits<std::size_t>::max() / k || k > std::numeric_limits<std::size_t>::max() / c) return false;
    const auto left_elements=r*k, right_elements=k*c, output_elements=r*c;
    std::lock_guard lock(mutex_);
    const VkDeviceSize left_bytes=static_cast<VkDeviceSize>(left_elements)*sizeof(float);
    const VkDeviceSize right_bytes=static_cast<VkDeviceSize>(right_elements)*sizeof(float);
    const VkDeviceSize bias_bytes=static_cast<VkDeviceSize>(bias?c:1)*sizeof(float);
    const VkDeviceSize output_bytes=static_cast<VkDeviceSize>(output_elements)*sizeof(float);
    if (!EnsureCapacity(left_bytes,right_bytes,bias_bytes,output_bytes)) return false;
    std::memcpy(mapped_[0],left,static_cast<std::size_t>(left_bytes));
    BindScratchParameters(right_bytes, bias_bytes);
    const bool reuse_parameters=immutable_parameters && pointwise_cached_ &&
        pointwise_weight_bytes_==right_bytes && pointwise_bias_bytes_==bias_bytes &&
        pointwise_cached_weights_==right && pointwise_cached_bias_==bias;
    if (!reuse_parameters) {
      std::memcpy(mapped_[1],right,static_cast<std::size_t>(right_bytes));
      if (bias) std::memcpy(mapped_[2],bias,static_cast<std::size_t>(bias_bytes));
      else std::memset(mapped_[2],0,sizeof(float));
      chain_cached_ = false;
      pointwise_cached_=immutable_parameters;
      pointwise_weight_bytes_=immutable_parameters?right_bytes:0;
      pointwise_bias_bytes_=immutable_parameters?bias_bytes:0;
      pointwise_cached_weights_=immutable_parameters?right:nullptr;
      pointwise_cached_bias_=immutable_parameters?bias:nullptr;
      right_cached_=false;
      coefficients_cached_=false;
    }
    if (reset_command_buffer_(command_buffer_,0)!=VK_SUCCESS) return false;
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,nullptr,
                                         VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,nullptr};
    if (begin_command_buffer_(command_buffer_,&begin)!=VK_SUCCESS) return false;
    cmd_bind_pipeline_(command_buffer_,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline_);
    cmd_bind_descriptor_sets_(command_buffer_,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline_layout_,0,1,&descriptor_set_,0,nullptr);
    // Recognition MLPs use the same wide, row-major shape as the plain
    // projection GEMMs, but previously took the scalar-per-vec4 mode merely
    // because their Swish was fused.  Enable the cooperative tile only when
    // explicitly requested while it is being profiled across drivers; the
    // public hybrid admission still covers the full transfer/fence/readback.
    // Plain GEMM retains its established default tile.
    const bool tiled_batch_gemm = !swish
        ? rows >= 8 && depth >= 16 && columns >= 128
        : (std::getenv("PPOCR_ENABLE_VULKAN_GEMM_SWISH_TILE") != nullptr &&
           rows >= 8 && depth >= 16 && columns >= 128);
    struct Push { std::uint32_t count,batches,operation0,operation1,operation2,operation3,steps,
                                right_repeat,right_per_batch,right_batch_stride,mode; }
        push{static_cast<std::uint32_t>(output_elements),1u,static_cast<std::uint32_t>(r),
             static_cast<std::uint32_t>(k),static_cast<std::uint32_t>(c),bias?1u:0u,
             1u,0u,0u,0u,tiled_batch_gemm ? (swish ? 25u : 24u) : (swish ? 23u : 19u)};
    cmd_push_constants_(command_buffer_,pipeline_layout_,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(push),&push);
    const auto dispatch_x=tiled_batch_gemm ? (c+127)/128 : (output_elements+1023)/1024;
    const auto dispatch_y=tiled_batch_gemm ? (r+7)/8 : 1;
    if (dispatch_x==0 || dispatch_x>UINT32_MAX || dispatch_y==0 || dispatch_y>UINT32_MAX) return false;
    cmd_dispatch_(command_buffer_,static_cast<std::uint32_t>(dispatch_x),
                  static_cast<std::uint32_t>(dispatch_y),1);
    if (end_command_buffer_(command_buffer_)!=VK_SUCCESS) return false;
    const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO,nullptr,0,nullptr,nullptr,1,&command_buffer_,0,nullptr};
    if (reset_fences_(device_,1,&submission_fence_)!=VK_SUCCESS ||
        queue_submit_(queue_,1,&submit,submission_fence_)!=VK_SUCCESS ||
        wait_for_fences_(device_,1,&submission_fence_,VK_TRUE,std::numeric_limits<std::uint64_t>::max())!=VK_SUCCESS) return false;
    std::memcpy(output,mapped_[3],static_cast<std::size_t>(output_bytes));
    return true;
  }

  bool RunGemmCtcTop1(int* indices, float* probabilities, const float* left,
                      const float* right, const float* bias, int rows, int depth,
                      int vocab, int steps, bool immutable_parameters) noexcept {
    if (!indices || !probabilities || !left || !right || rows <= 0 || depth <= 0 ||
        vocab <= 0 || steps <= 0 || rows % steps != 0 || !Initialize()) return false;
    if (std::getenv("PPOCR_DISABLE_VULKAN_GEMM_CTC") != nullptr) return false;
    const auto r = static_cast<std::size_t>(rows);
    const auto k = static_cast<std::size_t>(depth);
    const auto v = static_cast<std::size_t>(vocab);
    if (r > UINT32_MAX || k > UINT32_MAX || v > UINT32_MAX || r > UINT32_MAX / v ||
        r > std::numeric_limits<std::size_t>::max() / k ||
        k > std::numeric_limits<std::size_t>::max() / v) return false;
    const auto left_elements = r * k;
    const auto right_elements = k * v;
    const auto logit_elements = r * v;
    std::lock_guard lock(mutex_);
    const VkDeviceSize left_bytes = static_cast<VkDeviceSize>(left_elements) * sizeof(float);
    const VkDeviceSize right_bytes = static_cast<VkDeviceSize>(right_elements) * sizeof(float);
    const VkDeviceSize bias_bytes = static_cast<VkDeviceSize>(bias ? v : 1) * sizeof(float);
    const VkDeviceSize logit_bytes = static_cast<VkDeviceSize>(logit_elements) * sizeof(float);
    const VkDeviceSize compact_bytes = static_cast<VkDeviceSize>(r) * sizeof(float);
    if (!EnsureCapacity(std::max(left_bytes, compact_bytes), right_bytes, bias_bytes, logit_bytes,
                        compact_bytes))
      return false;
    std::memcpy(mapped_[0], left, static_cast<std::size_t>(left_bytes));
    BindScratchParameters(right_bytes, bias_bytes);
    const bool reuse_parameters = immutable_parameters && pointwise_cached_ &&
        pointwise_weight_bytes_ == right_bytes && pointwise_bias_bytes_ == bias_bytes &&
        pointwise_cached_weights_ == right && pointwise_cached_bias_ == bias;
    if (!reuse_parameters) {
      std::memcpy(mapped_[1], right, static_cast<std::size_t>(right_bytes));
      if (bias) std::memcpy(mapped_[2], bias, static_cast<std::size_t>(bias_bytes));
      else std::memset(mapped_[2], 0, sizeof(float));
      chain_cached_ = false;
      pointwise_cached_ = immutable_parameters;
      pointwise_weight_bytes_ = immutable_parameters ? right_bytes : 0;
      pointwise_bias_bytes_ = immutable_parameters ? bias_bytes : 0;
      pointwise_cached_weights_ = immutable_parameters ? right : nullptr;
      pointwise_cached_bias_ = immutable_parameters ? bias : nullptr;
      right_cached_ = false;
      coefficients_cached_ = false;
    }
    conv2d_cmd_ready_ = false;
    stem_cmd_ready_ = false;
    if (reset_command_buffer_(command_buffer_, 0) != VK_SUCCESS) return false;
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    if (begin_command_buffer_(command_buffer_, &begin) != VK_SUCCESS) return false;
    cmd_bind_pipeline_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
        0, 1, &descriptor_set_, 0, nullptr);
    const VkMemoryBarrier host_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &host_barrier, 0, nullptr, 0, nullptr);
    const bool tiled = rows >= 8 && depth >= 16 && vocab >= 128;
    struct Push {
      std::uint32_t count, batches, operation0, operation1, operation2, operation3, steps,
                    right_repeat, right_per_batch, right_batch_stride, mode;
    } gemm{static_cast<std::uint32_t>(logit_elements), 1u, static_cast<std::uint32_t>(r),
           static_cast<std::uint32_t>(k), static_cast<std::uint32_t>(v), bias ? 1u : 0u,
           1u, 0u, 0u, 0u, tiled ? 24u : 19u};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(gemm), &gemm);
    const auto gemm_x = tiled ? (vocab + 127) / 128 : int((logit_elements + 1023) / 1024);
    const auto gemm_y = tiled ? (rows + 7) / 8 : 1;
    if (gemm_x <= 0 || gemm_y <= 0) return false;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(gemm_x),
                  static_cast<std::uint32_t>(gemm_y), 1);
    const VkMemoryBarrier shader_to_shader{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &shader_to_shader, 0, nullptr, 0, nullptr);
    Push ctc{static_cast<std::uint32_t>(r), 1u, static_cast<std::uint32_t>(r),
             static_cast<std::uint32_t>(v), 0u, 0u, 0u, 0u, 0u, 0u,
             48u | 0x20000000u | 0x40000000u | 0x80000000u};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(ctc), &ctc);
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(r), 1, 1);
    const VkMemoryBarrier shader_to_host{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &shader_to_host, 0, nullptr, 0, nullptr);
    if (end_command_buffer_(command_buffer_) != VK_SUCCESS) return false;
    const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
                              1, &command_buffer_, 0, nullptr};
    if (reset_fences_(device_, 1, &submission_fence_) != VK_SUCCESS ||
        queue_submit_(queue_, 1, &submit, submission_fence_) != VK_SUCCESS ||
        wait_for_fences_(device_, 1, &submission_fence_, VK_TRUE,
                         std::numeric_limits<std::uint64_t>::max()) != VK_SUCCESS)
      return false;
    const auto* gpu_indices = static_cast<const float*>(mapped_[0]);
    const auto* gpu_probs = static_cast<const float*>(mapped_[4]);
    const int sequences = rows / steps;
    for (int sequence = 0; sequence < sequences; ++sequence) {
      int previous = -1;
      for (int step = 0; step < steps; ++step) {
        const int row = sequence * steps + step;
        const int best = int(gpu_indices[row]);
        indices[row] = best;
        probabilities[row] = (best != 0 && best != previous) ? gpu_probs[row] : 0.F;
        previous = best;
      }
    }
    return true;
  }

  BackendInfo Info() const noexcept {
    BackendInfo info;
    info.vulkan_loader_available = loader_ != nullptr;
    info.vulkan_compute_available = ready_;
    info.device_name = device_name_;
    info.vulkan_runtime_generation = arena_generation_.load(std::memory_order_acquire);
    return info;
  }

  int LastSubmissionResult() const noexcept {
    return last_submission_result_.load(std::memory_order_acquire);
  }

  VulkanTensorSlot AcquireArenaSlot(std::size_t elements, bool persistent) noexcept {
    VulkanTensorSlot result{};
    if (elements == 0 || elements > std::numeric_limits<VkDeviceSize>::max() / sizeof(float) ||
        !Initialize()) return result;
    const VkDeviceSize required = static_cast<VkDeviceSize>(elements) * sizeof(float);
    std::lock_guard lock(mutex_);
    std::size_t selected = arena_buffers_.size();
    VkDeviceSize selected_capacity = std::numeric_limits<VkDeviceSize>::max();
    for (std::size_t index = 0; index < arena_buffers_.size(); ++index) {
      const auto& candidate = arena_buffers_[index];
      // A recorded command buffer has not executed yet.  Although the graph
      // planner has reached a tensor's final *logical* consumer, that earlier
      // dispatch may still read its backing buffer.  Reusing it while adding
      // later nodes overwrites data in the same pending submission and makes
      // results depend on graph shape/input values.  Defer reuse until the
      // recording is submitted and fenced.  A free slot that existed before
      // this recording began is safe to reuse; one freed during recording is
      // still potentially read by a preceding dispatch in this command buffer.
      if (candidate.persistent == persistent && !candidate.live && !candidate.replay_pinned &&
          (!arena_recording_ || candidate.reusable_in_recording) &&
          candidate.capacity >= required &&
          candidate.capacity < selected_capacity) {
        selected = index;
        selected_capacity = candidate.capacity;
      }
    }
    if (selected == arena_buffers_.size()) {
      ArenaBuffer slot{};
      if (!CreateArenaBuffer(slot, GrowCapacity(required), persistent)) return result;
      arena_buffers_.push_back(std::move(slot));
    }
    auto& slot = arena_buffers_[selected];
    slot.live = true;
    slot.reusable_in_recording = false;
    slot.live_elements = elements;
    if (persistent_recording_) {
      slot.replay_pinned = true;
      auto& recorded = persistent_recording_->arena_slots;
      const auto index = static_cast<std::uint32_t>(selected);
      if (std::find(recorded.begin(), recorded.end(), index) == recorded.end())
        recorded.push_back(index);
    }
    result.index = static_cast<std::uint32_t>(selected);
    result.capacity_elements = static_cast<std::size_t>(slot.capacity / sizeof(float));
    result.live_elements = elements;
    result.resident = true;
    return result;
  }

  void ReleaseArenaSlot(std::uint32_t index) noexcept {
    std::lock_guard lock(mutex_);
    if (index >= arena_buffers_.size()) return;
    auto& slot = arena_buffers_[index];
    // Recorded-graph activations stay live so replay can overwrite them
    // in-place. Releasing would let a later Acquire recycle the buffer and
    // corrupt the cached command buffer's bindings.
    if (slot.replay_pinned) return;
    slot.live = false;
    slot.live_elements = 0;
  }

  float* ArenaMappedData(std::uint32_t index) noexcept {
    std::lock_guard lock(mutex_);
    if (index >= arena_buffers_.size() || !arena_buffers_[index].live) return nullptr;
    auto& slot = arena_buffers_[index];
    // Callers use this escape hatch only at a public host->GPU boundary.
    // Remember that the staging contents must be copied before shader use.
    slot.staging_dirty = slot.device_local;
    return static_cast<float*>(slot.mapped);
  }

  bool UploadArena(std::uint32_t index, const float* source, std::size_t elements) noexcept {
    if (!source) return false;
    std::lock_guard lock(mutex_);
    if (index >= arena_buffers_.size()) return false;
    auto& slot = arena_buffers_[index];
    if (!slot.live || elements > slot.live_elements) return false;
    std::memcpy(slot.mapped, source, elements * sizeof(float));
    slot.staging_dirty = slot.device_local;
    if (!FlushArenaWritesLocked(slot, elements)) return false;
    // A synchronous arena primitive may consume the slot immediately.  The
    // graph executor separately batches its constant uploads before it opens
    // its recording command buffer.
    return CopyStagingToArenaLocked(slot, elements);
  }

  bool FlushArenaWrites(std::uint32_t index, std::size_t elements) noexcept {
    std::lock_guard lock(mutex_);
    if (index >= arena_buffers_.size()) return false;
    auto& slot = arena_buffers_[index];
    if (!slot.live || elements > slot.live_elements) return false;
    if (!FlushArenaWritesLocked(slot, elements)) return false;
    // mapped_data() can write a graph input after the last explicit Upload.
    // BeginArenaRecording coalesces its transfer copy before the next graph
    // consumer; flushing is only the host-memory visibility step.
    return true;
  }

  bool DownloadArena(float* destination, std::uint32_t index, std::size_t elements) noexcept {
    if (!destination) return false;
    std::lock_guard lock(mutex_);
    if (index >= arena_buffers_.size()) return false;
    auto& slot = arena_buffers_[index];
    if (!slot.live || elements > slot.live_elements) return false;
    if (!CopyArenaToStagingLocked(slot, elements) || !InvalidateArenaReadsLocked(slot, elements)) return false;
    std::memcpy(destination, slot.mapped, elements * sizeof(float));
    return true;
  }

  // The strict graph executor can record many dependent dispatches into one
  // command buffer. Each arena primitive still updates its own descriptor set
  // before binding it, so the recorded set state stays stable until execution.
  // Barriers already emitted after every write preserve the existing device
  // read-after-write rules; only the costly submit/fence boundary is deferred.
  bool BeginArenaRecording() noexcept {
    if (!Initialize()) return false;
    std::lock_guard lock(mutex_);
    if (pending_standalone_ && !FlushPendingStandaloneLocked()) return false;
    if (arena_recording_) return false;
    // Graph descriptors are immutable for the whole command-buffer lifetime.
    // Reclaim the previous graph's sets only after its fence has completed.
    if (reset_descriptor_pool_(device_, graph_descriptor_pool_, 0) != VK_SUCCESS) return false;
    // Every preceding graph segment is fenced before a new recording starts.
    // Retire large *free* activation allocations at this point when the arena
    // has crossed its bounded retention budget.  Slots keep their vector index
    // as a tombstone, so live graph metadata can never be invalidated.  This
    // avoids carrying a whole high-resolution detector frontier into a later
    // narrow tail dispatch on UMA adapters, while constants and live branch
    // tensors remain entirely device resident.
    TrimFreeArenaBuffersLocked();
    // Driver-isolation diagnostic: allocate a distinct command-buffer object
    // for each fenced graph segment. This is still a GPU-only boundary (no
    // tensor staging), and lets an adapter distinguish recorded-command
    // lifetime from shader/data hazards. It is opt-in because retaining many
    // command buffers is not a throughput-oriented default.
    if (std::getenv("PPOCR_GPU_ONLY_FRESH_COMMAND_BUFFER") != nullptr) {
      const VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
          nullptr, command_pool_, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
      VkCommandBuffer fresh{};
      if (allocate_command_buffers_(device_, &command_info, &fresh) != VK_SUCCESS) return false;
      command_buffer_ = fresh;
    }
    if (reset_command_buffer_(command_buffer_, 0) != VK_SUCCESS) return false;
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                         VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    if (begin_command_buffer_(command_buffer_, &begin) != VK_SUCCESS) return false;
    cmd_bind_pipeline_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    arena_recording_ = true;
    // Arena inputs can originate either from a host-visible mapped upload or
    // from a preceding standalone preprocessing submission.  A command-buffer
    // boundary alone does not establish either HOST_WRITE -> shader-read or
    // shader-write -> shader-read visibility.  Make both producers available
    // to the first strict-graph dispatch; later arena primitives emit their
    // own compute-to-compute barriers after each write.
    const VkMemoryBarrier input_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_,
        VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &input_barrier,
        0, nullptr, 0, nullptr);
    recording_descriptor_sets_.clear();
    // The preceding graph submission was fenced before a new recording can
    // begin, so slots released by that earlier graph are now safe to recycle.
    for (auto& slot : arena_buffers_) slot.reusable_in_recording = !slot.live;
    // Uploads completed before recording are host-visible staging writes.
    // Record their transfer copies at the beginning of this command buffer so
    // their first graph consumers observe device-local storage.  Intermediate
    // graph values never take this path because they are not staging-dirty.
    if (!RecordDirtyStagingCopiesLocked()) { arena_recording_ = false; return false; }
    return true;
  }

  bool EndArenaRecording(bool submit) noexcept {
    std::lock_guard lock(mutex_);
    if (!arena_recording_) return false;
    arena_recording_ = false;
    if (end_command_buffer_(command_buffer_) != VK_SUCCESS) return false;
    if (!submit) {
      recording_descriptor_sets_.clear();
      return true;
    }
    const VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
                            1, &command_buffer_, 0, nullptr};
    const VkResult reset = reset_fences_(device_, 1, &submission_fence_);
    // Keep the command buffer lifetime conservative on drivers which defer
    // validation/reclamation until queue idle.  This is opt-in because it is
    // a throughput trade-off, but it preserves a completely GPU-resident
    // graph and helps isolate device-loss bugs from execution overlap.
    const bool queue_idle_before_submit =
        std::getenv("PPOCR_GPU_ONLY_QUEUE_IDLE_SEGMENTS") != nullptr;
    const VkResult pre_idle = reset == VK_SUCCESS && queue_idle_before_submit
        ? queue_wait_idle_(queue_) : reset;
    const VkResult submitted = pre_idle == VK_SUCCESS
        ? queue_submit_(queue_, 1, &info, submission_fence_) : pre_idle;
    const VkResult waited = submitted == VK_SUCCESS
        ? wait_for_fences_(device_, 1, &submission_fence_, VK_TRUE,
                          std::numeric_limits<std::uint64_t>::max()) : submitted;
    last_submission_result_.store(static_cast<int>(waited), std::memory_order_release);
    const bool completed = waited == VK_SUCCESS;
    if (!completed) restart_required_.store(true, std::memory_order_release);
    if (!completed && std::getenv("PPOCR_GPU_ONLY_TRACE") != nullptr) {
      std::cerr << "Vulkan graph submission failed reset=" << reset
                << " submit=" << submitted << " wait=" << waited << '\n';
    }
    recording_descriptor_sets_.clear();
    return completed;
  }

  bool SubmitArenaSegment() noexcept {
    // A persistent recorded graph must stay in one command buffer so replay
    // is a single submit. Driver-isolation fences remain for ephemeral graphs.
    if (persistent_recording_) return true;
    if (!EndArenaRecording(true)) return false;
    return BeginArenaRecording();
  }

  PersistentGraph* FindPersistentGraphLocked(std::uint64_t key) noexcept {
    for (auto& graph : persistent_graphs_) {
      if (graph.key == key) return &graph;
    }
    return nullptr;
  }

  bool AllocatePersistentGraphLocked(PersistentGraph& graph) noexcept {
    if (graph.command_buffer && graph.descriptor_pool) return true;
    if (!command_pool_ || !create_descriptor_pool_ || !allocate_command_buffers_) return false;
    if (!graph.command_buffer) {
      const VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
          nullptr, command_pool_, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
      if (allocate_command_buffers_(device_, &command_info, &graph.command_buffer) != VK_SUCCESS)
        return false;
    }
    if (!graph.descriptor_pool) {
      // A persistent graph keeps every descriptor set alive for replay.
      // Medium PP-OCRv6 can exceed 2K dispatches after operator fusion is
      // disabled for exactness; retain the same headroom as transient graphs.
      constexpr std::uint32_t kSets = 8192;
      const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 * kSets};
      const VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
          nullptr, 0, kSets, 1, &pool_size};
      if (create_descriptor_pool_(device_, &pool_info, nullptr, &graph.descriptor_pool) != VK_SUCCESS)
        return false;
    }
    return true;
  }

  bool BeginPersistentGraphRecording(std::uint64_t key) noexcept {
    if (!Initialize()) return false;
    std::lock_guard lock(mutex_);
    if (pending_standalone_ && !FlushPendingStandaloneLocked()) return false;
    if (arena_recording_ || persistent_recording_) return false;
    // The persistent-graph budget bounds how many recorded replay command
    // buffers (and their pinned activations) stay resident. The historical
    // hard cap of 8 silently stopped recording once a dense page's width
    // buckets x batch splits exceeded it (~10-14 shapes on a 34-crop page),
    // so multi-shape chunked replay could never assemble a complete set and
    // every later page fell back to per-batch recording. 24 covers the
    // measured dense-page spread with headroom; deployments can tighten or
    // widen it via PPOCR_GPU_MAX_PERSISTENT_GRAPHS. Each entry pins its own
    // activation slots, so raising this raises the UMA high-water mark.
    static const std::size_t kMaxPersistentGraphs = [] {
      if (const char* text = std::getenv("PPOCR_GPU_MAX_PERSISTENT_GRAPHS")) {
        char* end{};
        const auto parsed = std::strtoul(text, &end, 10);
        if (end != text && *end == '\0' && parsed > 0) {
          return static_cast<std::size_t>(parsed);
        }
      }
      return static_cast<std::size_t>(24);
    }();
    if (persistent_graphs_.size() >= kMaxPersistentGraphs &&
        !FindPersistentGraphLocked(key)) {
      return false;
    }
    PersistentGraph* graph = FindPersistentGraphLocked(key);
    if (!graph) {
      persistent_graphs_.push_back(PersistentGraph{});
      graph = &persistent_graphs_.back();
      graph->key = key;
    }
    if (!AllocatePersistentGraphLocked(*graph)) return false;
    graph->ready = false;
    graph->arena_slots.clear();
    graph->descriptor_sets.clear();
    if (reset_descriptor_pool_(device_, graph->descriptor_pool, 0) != VK_SUCCESS) return false;
    if (reset_command_buffer_(graph->command_buffer, 0) != VK_SUCCESS) return false;
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                         0, nullptr};
    if (begin_command_buffer_(graph->command_buffer, &begin) != VK_SUCCESS) return false;
    saved_command_buffer_ = command_buffer_;
    command_buffer_ = graph->command_buffer;
    persistent_recording_ = graph;
    cmd_bind_pipeline_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    arena_recording_ = true;
    const VkMemoryBarrier input_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_,
        VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &input_barrier,
        0, nullptr, 0, nullptr);
    recording_descriptor_sets_.clear();
    for (auto& slot : arena_buffers_) slot.reusable_in_recording = !slot.live && !slot.replay_pinned;
    if (!RecordDirtyStagingCopiesLocked()) {
      arena_recording_ = false;
      command_buffer_ = saved_command_buffer_;
      persistent_recording_ = nullptr;
      // No replay command was completed, so any activations acquired while
      // attempting this record must not remain pinned indefinitely.
      for (const auto index : graph->arena_slots) {
        if (index >= arena_buffers_.size()) continue;
        auto& slot = arena_buffers_[index];
        slot.replay_pinned = false;
        slot.live = false;
        slot.live_elements = 0;
      }
      graph->arena_slots.clear();
      return false;
    }
    return true;
  }

  bool EndPersistentGraphRecording(bool submit) noexcept {
    std::lock_guard lock(mutex_);
    if (!arena_recording_ || !persistent_recording_) return false;
    PersistentGraph* graph = persistent_recording_;
    arena_recording_ = false;
    persistent_recording_ = nullptr;
    const bool ended = end_command_buffer_(command_buffer_) == VK_SUCCESS;
    command_buffer_ = saved_command_buffer_;
    saved_command_buffer_ = VK_NULL_HANDLE;
    if (!ended || !submit) {
      for (const auto index : graph->arena_slots) {
        if (index >= arena_buffers_.size()) continue;
        auto& slot = arena_buffers_[index];
        slot.replay_pinned = false;
        slot.live = false;
        slot.live_elements = 0;
      }
      graph->arena_slots.clear();
      return false;
    }
    const VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
                            1, &graph->command_buffer, 0, nullptr};
    const VkResult reset = reset_fences_(device_, 1, &submission_fence_);
    const VkResult submitted = reset == VK_SUCCESS
        ? queue_submit_(queue_, 1, &info, submission_fence_) : reset;
    const VkResult waited = submitted == VK_SUCCESS
        ? wait_for_fences_(device_, 1, &submission_fence_, VK_TRUE,
                           std::numeric_limits<std::uint64_t>::max()) : submitted;
    const bool completed = waited == VK_SUCCESS;
    if (!completed) restart_required_.store(true, std::memory_order_release);
    graph->ready = completed;
    return completed;
  }

  bool ReplayPersistentGraph(std::uint64_t key) noexcept {
    const std::uint64_t keys[1]{key};
    return ReplayPersistentGraphs(keys, 1);
  }

  bool ReplayPersistentGraphs(const std::uint64_t* keys, int count) noexcept {
    if (!keys || count <= 0 || !Initialize()) return false;
    std::lock_guard lock(mutex_);
    std::vector<VkCommandBuffer> command_buffers;
    command_buffers.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
      PersistentGraph* graph = FindPersistentGraphLocked(keys[i]);
      if (!graph || !graph->ready || !graph->command_buffer) return false;
      command_buffers.push_back(graph->command_buffer);
    }
    const auto mark_unready = [&] {
      restart_required_.store(true, std::memory_order_release);
      for (int i = 0; i < count; ++i) {
        if (PersistentGraph* graph = FindPersistentGraphLocked(keys[i]))
          graph->ready = false;
      }
    };
    if (pending_standalone_ && chain_semaphore_) {
      pending_standalone_ = false;
      const VkSubmitInfo first{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
                               1, &command_buffer_, 1, &chain_semaphore_};
      if (queue_submit_(queue_, 1, &first, VK_NULL_HANDLE) != VK_SUCCESS) {
        mark_unready();
        return false;
      }
      const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      const VkSubmitInfo second{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 1, &chain_semaphore_,
                                &wait_stage, static_cast<std::uint32_t>(command_buffers.size()),
                                command_buffers.data(), 0, nullptr};
      const VkResult reset = reset_fences_(device_, 1, &submission_fence_);
      const VkResult submitted = reset == VK_SUCCESS
          ? queue_submit_(queue_, 1, &second, submission_fence_) : reset;
      const VkResult waited = submitted == VK_SUCCESS
          ? wait_for_fences_(device_, 1, &submission_fence_, VK_TRUE,
                             std::numeric_limits<std::uint64_t>::max()) : submitted;
      if (waited != VK_SUCCESS) {
        mark_unready();
        return false;
      }
      return true;
    }
    if (pending_standalone_ && !FlushPendingStandaloneLocked()) {
      mark_unready();
      return false;
    }
    const VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
                            static_cast<std::uint32_t>(command_buffers.size()),
                            command_buffers.data(), 0, nullptr};
    const VkResult reset = reset_fences_(device_, 1, &submission_fence_);
    const VkResult submitted = reset == VK_SUCCESS
        ? queue_submit_(queue_, 1, &info, submission_fence_) : reset;
    const VkResult waited = submitted == VK_SUCCESS
        ? wait_for_fences_(device_, 1, &submission_fence_, VK_TRUE,
                           std::numeric_limits<std::uint64_t>::max()) : submitted;
    if (waited != VK_SUCCESS) {
      mark_unready();
      return false;
    }
    return true;
  }

  void DeferNextStandaloneSubmit() noexcept {
    if (!Initialize()) return;
    std::lock_guard lock(mutex_);
    // Chaining a resize command buffer to a replay graph is a useful
    // deployment experiment, but a few command-buffer/driver combinations
    // need the completed resize fence before consuming the newly written
    // replay slot. Keep correctness as the default and retain the feature as
    // an explicit opt-in after local qualification.
    static const bool chain =
        std::getenv("PPOCR_ENABLE_GPU_CHAIN_RESIZE_REPLAY") != nullptr &&
        std::getenv("PPOCR_DISABLE_GPU_CHAIN_RESIZE_REPLAY") == nullptr;
    if (chain && chain_semaphore_) defer_next_standalone_ = true;
  }

  bool FlushDeferredStandalone() noexcept {
    if (!Initialize()) return false;
    std::lock_guard lock(mutex_);
    defer_next_standalone_ = false;
    return FlushPendingStandaloneLocked();
  }

  bool HasPersistentGraph(std::uint64_t key) noexcept {
    std::lock_guard lock(mutex_);
    for (const auto& graph : persistent_graphs_) {
      if (graph.key == key && graph.ready) return true;
    }
    return false;
  }

  bool PersistentGraphRecording() noexcept {
    std::lock_guard lock(mutex_);
    return persistent_recording_ != nullptr;
  }

  void DropPersistentGraph(std::uint64_t key) noexcept {
    std::lock_guard lock(mutex_);
    if (persistent_recording_ && persistent_recording_->key == key) return;
    const auto found = std::find_if(persistent_graphs_.begin(), persistent_graphs_.end(),
                                    [key](const PersistentGraph& graph) {
                                      return graph.key == key;
                                    });
    if (found == persistent_graphs_.end()) return;
    // Command buffers belong to command_pool_; releasing the descriptor pool
    // returns the potentially large per-dispatch descriptor cache. The caller
    // unpins only this graph's tensor slots afterwards; other live replay
    // graphs may share the same process-wide arena.
    if (found->descriptor_pool && destroy_descriptor_pool_)
      destroy_descriptor_pool_(device_, found->descriptor_pool, nullptr);
    found->descriptor_pool = VK_NULL_HANDLE;
    found->command_buffer = VK_NULL_HANDLE;
    found->descriptor_sets.clear();
    // Releasing only the graph's externally visible input/output slots left
    // every internally recorded activation permanently pinned.  Those slots
    // are not shared with another persistent graph (pinned slots cannot be
    // reacquired), and this method runs only between fenced submissions, so
    // they can become ordinary free arena storage immediately.
    for (const auto index : found->arena_slots) {
      if (index >= arena_buffers_.size()) continue;
      auto& slot = arena_buffers_[index];
      slot.replay_pinned = false;
      slot.live = false;
      slot.live_elements = 0;
      slot.reusable_in_recording = false;
    }
    found->arena_slots.clear();
    persistent_graphs_.erase(found);
  }

  void PinArenaSlot(std::uint32_t index) noexcept {
    std::lock_guard lock(mutex_);
    if (index >= arena_buffers_.size()) return;
    arena_buffers_[index].replay_pinned = true;
    // The input RGB slot is usually acquired before persistent recording
    // begins, then pinned explicitly by the graph compiler. Include these
    // late/manual pins in the same ownership list as activations acquired
    // during recording so an aborted or evicted replay graph cannot leak it.
    if (persistent_recording_) {
      auto& recorded = persistent_recording_->arena_slots;
      if (std::find(recorded.begin(), recorded.end(), index) == recorded.end())
        recorded.push_back(index);
    }
  }


  bool FlushPendingStandaloneLocked() noexcept {
    if (!pending_standalone_) return true;
    pending_standalone_ = false;
    const VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
                            1, &command_buffer_, 0, nullptr};
    const VkResult reset = reset_fences_(device_, 1, &submission_fence_);
    const VkResult submitted = reset == VK_SUCCESS
        ? queue_submit_(queue_, 1, &info, submission_fence_) : reset;
    const VkResult waited = submitted == VK_SUCCESS
        ? wait_for_fences_(device_, 1, &submission_fence_, VK_TRUE,
                           std::numeric_limits<std::uint64_t>::max()) : submitted;
    if (waited != VK_SUCCESS) {
      restart_required_.store(true, std::memory_order_release);
      return false;
    }
    return true;
  }

  bool BeginArenaOpLocked() noexcept {
    if (arena_recording_) {
      // A dedicated tail pipeline can be bound for one operation in a graph.
      // Restore the ordinary all-operator pipeline at every subsequent graph
      // op so pipeline state never leaks across that isolated dispatch.
      cmd_bind_pipeline_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
      return true;
    }
    if (pending_standalone_ && !FlushPendingStandaloneLocked()) return false;
    if (reset_command_buffer_(command_buffer_, 0) != VK_SUCCESS) return false;
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                         VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    if (begin_command_buffer_(command_buffer_, &begin) != VK_SUCCESS) return false;
    cmd_bind_pipeline_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    if (!RecordDirtyStagingCopiesLocked()) return false;
    // Standalone front-end/boundary dispatches may consume freshly mapped
    // coherent arena data. Command-buffer creation is not a memory
    // dependency, so establish HOST_WRITE -> shader-read visibility just as
    // BeginArenaRecording does for strict graph segments.
    const VkMemoryBarrier input_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_,
        VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &input_barrier,
        0, nullptr, 0, nullptr);
    return true;
  }

  void UnpinArenaSlot(std::uint32_t index) noexcept {
    std::lock_guard lock(mutex_);
    if (index < arena_buffers_.size()) arena_buffers_[index].replay_pinned = false;
  }

  bool EndArenaOpLocked() noexcept {
    if (arena_recording_) return true;
    if (end_command_buffer_(command_buffer_) != VK_SUCCESS) {
      defer_next_standalone_ = false;
      return false;
    }
    const bool defer = defer_next_standalone_ && chain_semaphore_ != VK_NULL_HANDLE;
    defer_next_standalone_ = false;
    if (defer) {
      pending_standalone_ = true;
      return true;
    }
    const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
                              1, &command_buffer_, 0, nullptr};
    const VkResult reset = reset_fences_(device_, 1, &submission_fence_);
    const VkResult submitted = reset == VK_SUCCESS
        ? queue_submit_(queue_, 1, &submit, submission_fence_) : reset;
    const VkResult waited = submitted == VK_SUCCESS
        ? wait_for_fences_(device_, 1, &submission_fence_, VK_TRUE,
                           std::numeric_limits<std::uint64_t>::max()) : submitted;
    if (waited != VK_SUCCESS) restart_required_.store(true, std::memory_order_release);
    return waited == VK_SUCCESS;
  }

  // Execute the driver-sensitive medium detector tail in a dedicated command
  // buffer and descriptor set.  This is strictly a GPU scheduling boundary:
  // activations and parameters remain in their original Vulkan buffers.
  // Use buffer handles directly because ArenaBuffer is declared in the
  // implementation-detail section below this graph-recording helper.
  bool RunIsolatedDepthwiseTailLocked(const std::array<VkBuffer, 5>& buffers,
                                      const std::array<VkDeviceSize, 5>& ranges,
                                      std::uint32_t count, std::uint32_t batches,
                                      std::uint32_t kernel_height,
                                      std::uint32_t kernel_width,
                                      std::uint32_t stride_height,
                                      std::uint32_t stride_width,
                                      std::uint32_t output_width,
                                      std::uint32_t input_height,
                                      std::uint32_t input_width,
                                      std::uint32_t output_height,
                                      std::uint32_t mode) noexcept {
    if (!arena_recording_ || persistent_recording_ || !tail_command_buffer_ ||
        !tail_descriptor_set_ || !tail_pipeline_) return false;
    // The regular recorder contains only the continuation opened after the
    // preceding Add.143 fence. End and discard it: queue ordering is already
    // established by that fence. Do not reset the graph descriptor pool here:
    // the preceding command buffer still owns its descriptor sets until the
    // queue fence completes, and resetting that pool before the tail submit
    // violates Vulkan descriptor-pool lifetime rules on strict drivers.
    arena_recording_ = false;
    if (end_command_buffer_(command_buffer_) != VK_SUCCESS) return false;
    recording_descriptor_sets_.clear();

    std::array<VkDescriptorBufferInfo, 5> infos{};
    std::array<VkWriteDescriptorSet, 5> writes{};
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {buffers[binding], 0, ranges[binding]};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                         tail_descriptor_set_, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr,
                         &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(),
                            0, nullptr);
    if (reset_command_buffer_(tail_command_buffer_, 0) != VK_SUCCESS) return false;
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                         VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    if (begin_command_buffer_(tail_command_buffer_, &begin) != VK_SUCCESS) return false;
    const VkMemoryBarrier input_visible{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(tail_command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &input_visible,
                          0, nullptr, 0, nullptr);
    cmd_bind_pipeline_(tail_command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, tail_pipeline_);
    cmd_bind_descriptor_sets_(tail_command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                              pipeline_layout_, 0, 1, &tail_descriptor_set_, 0, nullptr);
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; };
    const Push push{count, batches, kernel_height, kernel_width, stride_height, stride_width,
                    output_width, input_height, input_width, output_height, mode};
    cmd_push_constants_(tail_command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                        0, sizeof(push), &push);
    // The dedicated tail shader maps one workgroup to one channel and uses
    // local invocations for the feature-map plane. `count / output_plane` is
    // the original C, supplied without changing the shared push ABI.
    if (output_height == 0 || output_width == 0) return false;
    const std::size_t dispatch = std::size_t(count) /
        (std::size_t(output_height) * output_width);
    if (dispatch == 0 || dispatch > UINT32_MAX) return false;
    cmd_dispatch_(tail_command_buffer_, static_cast<std::uint32_t>(dispatch), batches, 1);
    const VkMemoryBarrier output_visible{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(tail_command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &output_visible,
                          0, nullptr, 0, nullptr);
    if (end_command_buffer_(tail_command_buffer_) != VK_SUCCESS) return false;
    const VkSubmitInfo tail_submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
                                        1, &tail_command_buffer_, 0, nullptr};
    const VkResult tail_reset = reset_fences_(device_, 1, &submission_fence_);
    const VkResult tail_submit = tail_reset == VK_SUCCESS
        ? queue_submit_(queue_, 1, &tail_submit_info, submission_fence_) : tail_reset;
    const VkResult tail_wait = tail_submit == VK_SUCCESS
        ? wait_for_fences_(device_, 1, &submission_fence_, VK_TRUE,
                           std::numeric_limits<std::uint64_t>::max()) : tail_submit;
    if (tail_wait != VK_SUCCESS) {
      restart_required_.store(true, std::memory_order_release);
      if (std::getenv("PPOCR_GPU_ONLY_TRACE") != nullptr) {
        std::cerr << "Vulkan isolated depthwise tail failed reset=" << tail_reset
                  << " submit=" << tail_submit << " wait=" << tail_wait << '\n';
      }
      return false;
    }

    // Resume normal recording after the fenced GPU-only tail; values released
    // after this point are now safe to reuse without host staging. The prior
    // command buffer was fenced before the tail submission, so its descriptor
    // sets may now be reclaimed safely.
    if (reset_descriptor_pool_(device_, graph_descriptor_pool_, 0) != VK_SUCCESS ||
        reset_command_buffer_(command_buffer_, 0) != VK_SUCCESS) return false;
    if (begin_command_buffer_(command_buffer_, &begin) != VK_SUCCESS) return false;
    cmd_bind_pipeline_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &output_visible,
                          0, nullptr, 0, nullptr);
    arena_recording_ = true;
    recording_descriptor_sets_.clear();
    for (auto& slot : arena_buffers_) slot.reusable_in_recording = true;
    return true;
  }

  VkDescriptorSet DescriptorSetForArenaOpLocked() noexcept {
    if (!arena_recording_) return descriptor_set_;
    VkDescriptorPool pool = graph_descriptor_pool_;
    std::vector<VkDescriptorSet>* sets = &recording_descriptor_sets_;
    if (persistent_recording_ && persistent_recording_->descriptor_pool) {
      pool = persistent_recording_->descriptor_pool;
      sets = &persistent_recording_->descriptor_sets;
    }
    const VkDescriptorSetAllocateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        nullptr, pool, 1, &descriptor_layout_};
    VkDescriptorSet set{};
    if (allocate_descriptor_sets_(device_, &info, &set) != VK_SUCCESS) return VK_NULL_HANDLE;
    sets->push_back(set);
    return set;
  }

  bool RunArenaCopy(std::uint32_t input_index, std::uint32_t output_index,
                    std::size_t elements) noexcept {
    if (elements == 0 || elements > UINT32_MAX || !Initialize()) return false;
    std::lock_guard lock(mutex_);
    if (input_index >= arena_buffers_.size() || output_index >= arena_buffers_.size()) return false;
    const auto& input = arena_buffers_[input_index]; const auto& output = arena_buffers_[output_index];
    if (!input.live || !output.live || input.live_elements < elements || output.live_elements < elements) return false;
    if (!BeginArenaOpLocked()) return false;
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(elements) * sizeof(float);
    // A graph-local tensor copy is a Vulkan transfer, not a compute shader.
    // Besides removing one descriptor/pipeline bind from every fan-out copy,
    // this gives strict GPU-only execution a simple queue-family-neutral
    // hand-off point for drivers which are sensitive to a long run of compute
    // dispatches.  The buffers never leave device memory.
    const VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_,
        VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
    const VkBufferCopy region{0, 0, bytes};
    cmd_copy_buffer_(command_buffer_, input.buffer, output.buffer, 1, &region);
    const VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &after,
                          0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaBinary(std::uint32_t left_index, std::uint32_t right_index,
                      std::size_t elements, kernels::BinaryOp operation) noexcept {
    if (elements == 0 || elements > UINT32_MAX || !Initialize()) return false;
    std::lock_guard lock(mutex_);
    if (left_index >= arena_buffers_.size() || right_index >= arena_buffers_.size()) return false;
    const auto& left = arena_buffers_[left_index];
    const auto& right = arena_buffers_[right_index];
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(elements) * sizeof(float);
    if (!left.live || !right.live || elements > left.live_elements || elements > right.live_elements ||
        left.capacity < bytes || right.capacity < bytes) return false;
    // The graph arena supplies the two live bindings directly. The remaining
    // descriptors are valid aliases; mode zero never dereferences them.
    std::array<VkDescriptorBufferInfo, 5> infos{};
    std::array<VkWriteDescriptorSet, 5> writes{};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked();
    if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      const auto& slot = binding == 1 ? right : left;
      infos[binding] = {slot.buffer, 0, bytes};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!BeginArenaOpLocked()) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 
                              0, 1, &arena_set, 0, nullptr);
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; };
    Push push{static_cast<std::uint32_t>(elements), 1u, static_cast<std::uint32_t>(operation), 0u,
              0u, 0u, 1u, 1u, static_cast<std::uint32_t>(elements),
              static_cast<std::uint32_t>(elements), 0u};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                        sizeof(push), &push);
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>((elements + 1023) / 1024), 1, 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaBinaryChain(std::uint32_t left_index, const std::uint32_t* right_indices,
                           const kernels::BinaryOp* operations, std::size_t steps,
                           std::size_t elements) noexcept {
    if (!right_indices || !operations || steps == 0 || steps > 4 || elements == 0 ||
        elements > UINT32_MAX || !Initialize()) return false;
    // Descriptor contents are consumed at command execution, not at record
    // time. The compact Hybrid descriptor set cannot safely be rebound to a
    // different RHS several times in one pending command buffer. Submit each
    // graph operation separately for now: this preserves device residency and
    // correct barriers without ever copying an activation through host memory.
    // A later multi-descriptor-set graph compiler can collapse these fences.
    for (std::size_t step = 0; step < steps; ++step) {
      if (!RunArenaBinary(left_index, right_indices[step], elements, operations[step])) return false;
    }
    return true;
  }

  bool RunArenaBinaryBroadcast(std::uint32_t left_index, std::uint32_t right_index,
                               std::uint32_t output_index, std::size_t batches,
                               std::size_t right_repeat, std::size_t right_per_batch,
                               bool right_shared, kernels::BinaryOp operation) noexcept {
    if (batches == 0 || right_repeat == 0 || right_per_batch == 0 || !Initialize()) return false;
    const std::size_t elements = batches * right_repeat * right_per_batch;
    const std::size_t rhs_elements = (right_shared ? 1u : batches) * right_per_batch;
    if (elements == 0 || elements > UINT32_MAX || rhs_elements > UINT32_MAX) return false;
    std::lock_guard lock(mutex_);
    if (left_index >= arena_buffers_.size() || right_index >= arena_buffers_.size() ||
        output_index >= arena_buffers_.size()) return false;
    const auto& left = arena_buffers_[left_index];
    const auto& right = arena_buffers_[right_index];
    const auto& output = arena_buffers_[output_index];
    if (!left.live || !right.live || !output.live || left.live_elements < elements ||
        output.live_elements < elements || right.live_elements < rhs_elements) return false;
    std::array<VkDescriptorBufferInfo, 5> infos{}; std::array<VkWriteDescriptorSet, 5> writes{};
    const std::array<const ArenaBuffer*, 5> sources{&left, &right, &left, &output, &left};
    const std::array<VkDeviceSize, 5> ranges{
        VkDeviceSize(elements * sizeof(float)), VkDeviceSize(rhs_elements * sizeof(float)), sizeof(float),
        VkDeviceSize(elements * sizeof(float)), sizeof(float)};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked();
    if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {sources[binding]->buffer, 0, ranges[binding]};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!BeginArenaOpLocked()) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1,
                              &arena_set, 0, nullptr);
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; }
        push{static_cast<std::uint32_t>(elements / batches), static_cast<std::uint32_t>(batches),
             static_cast<std::uint32_t>(operation), 0u, 0u, 0u, 0u,
             static_cast<std::uint32_t>(right_repeat), static_cast<std::uint32_t>(right_per_batch),
             right_shared ? 0u : static_cast<std::uint32_t>(right_per_batch), 41u};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    const std::size_t dispatch = (elements / batches + 1023) / 1024;
    if (dispatch == 0 || dispatch > UINT32_MAX) return false;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch), static_cast<std::uint32_t>(batches), 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaConcat(const std::uint32_t* input_indices, std::size_t input_count,
                      std::uint32_t output_index, std::size_t outer, std::size_t inner,
                      const std::size_t* axis_lengths) noexcept {
    if (!input_indices || !axis_lengths || input_count < 2 || input_count > 4 || outer == 0 || inner == 0 ||
        !Initialize()) return false;
    std::size_t total_axis{};
    for (std::size_t i = 0; i < input_count; ++i) {
      if (axis_lengths[i] == 0 || total_axis > UINT32_MAX - axis_lengths[i]) return false;
      total_axis += axis_lengths[i];
    }
    if (total_axis > UINT32_MAX || outer > UINT32_MAX || inner > UINT32_MAX ||
        outer > UINT32_MAX / total_axis || outer * total_axis > UINT32_MAX / inner) return false;
    const std::size_t total = outer * total_axis * inner;
    std::lock_guard lock(mutex_);
    if (output_index >= arena_buffers_.size()) return false;
    const auto& output = arena_buffers_[output_index];
    if (!output.live || output.live_elements < total) return false;
    std::array<const ArenaBuffer*, 5> sources{};
    for (std::size_t i = 0; i < 4; ++i) {
      const auto source_index = input_indices[std::min(i, input_count - 1)];
      if (source_index >= arena_buffers_.size()) return false;
      sources[i] = &arena_buffers_[source_index];
      const std::size_t expected = outer * axis_lengths[std::min(i, input_count - 1)] * inner;
      if (!sources[i]->live || sources[i]->live_elements < expected) return false;
    }
    sources[4] = sources[3];
    std::array<VkDescriptorBufferInfo, 5> infos{}; std::array<VkWriteDescriptorSet, 5> writes{};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked();
    if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      const auto* source = binding == 3 ? &output : sources[binding];
      const std::size_t elements = binding == 3 ? total : outer * axis_lengths[std::min<std::size_t>(binding, input_count - 1)] * inner;
      infos[binding] = {source->buffer, 0, VkDeviceSize(elements * sizeof(float))};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!BeginArenaOpLocked()) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1,
                              &arena_set, 0, nullptr);
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; }
        push{static_cast<std::uint32_t>(total), 1u, static_cast<std::uint32_t>(axis_lengths[0]),
             static_cast<std::uint32_t>(input_count > 1 ? axis_lengths[1] : 0),
             static_cast<std::uint32_t>(input_count > 2 ? axis_lengths[2] : 0),
             static_cast<std::uint32_t>(input_count > 3 ? axis_lengths[3] : 0),
             static_cast<std::uint32_t>(outer), static_cast<std::uint32_t>(inner), 0u, 0u, 42u};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    const std::size_t dispatch = (total + 1023) / 1024;
    if (dispatch == 0 || dispatch > UINT32_MAX) return false;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch), 1, 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaNearestResize(std::uint32_t input_index, std::uint32_t output_index,
                             std::size_t batches, int channels, int input_height, int input_width,
                             int scale_height, int scale_width) noexcept {
    if (batches == 0 || channels <= 0 || input_height <= 0 || input_width <= 0 ||
        scale_height <= 0 || scale_width <= 0 || !Initialize()) return false;
    const std::size_t input_plane = std::size_t(input_height) * input_width;
    const std::size_t output_plane = input_plane * std::size_t(scale_height) * scale_width;
    const std::size_t input_elements = batches * std::size_t(channels) * input_plane;
    const std::size_t output_elements = batches * std::size_t(channels) * output_plane;
    if (input_elements > UINT32_MAX || output_elements > UINT32_MAX) return false;
    std::lock_guard lock(mutex_);
    if (input_index >= arena_buffers_.size() || output_index >= arena_buffers_.size()) return false;
    const auto& input = arena_buffers_[input_index]; const auto& output = arena_buffers_[output_index];
    if (!input.live || !output.live || input.live_elements < input_elements || output.live_elements < output_elements) return false;
    std::array<VkDescriptorBufferInfo, 5> infos{}; std::array<VkWriteDescriptorSet, 5> writes{};
    const std::array<const ArenaBuffer*, 5> sources{&input, &input, &input, &output, &input};
    const std::array<VkDeviceSize, 5> ranges{VkDeviceSize(input_elements * sizeof(float)), sizeof(float), sizeof(float),
                                              VkDeviceSize(output_elements * sizeof(float)), sizeof(float)};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked();
    if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {sources[binding]->buffer, 0, ranges[binding]};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!BeginArenaOpLocked()) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1,
                              &arena_set, 0, nullptr);
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; }
        push{static_cast<std::uint32_t>(std::size_t(channels) * output_plane), static_cast<std::uint32_t>(batches),
             static_cast<std::uint32_t>(channels), static_cast<std::uint32_t>(input_height),
             static_cast<std::uint32_t>(input_width), static_cast<std::uint32_t>(scale_height),
             static_cast<std::uint32_t>(scale_width), 0u, 0u, 0u, 43u};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    const std::size_t dispatch = (std::size_t(channels) * output_plane + 1023) / 1024;
    if (dispatch == 0 || dispatch > UINT32_MAX) return false;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch), static_cast<std::uint32_t>(batches), 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaUnary(std::uint32_t index, std::size_t elements, VulkanUnaryOp operation,
                     float alpha, float beta) noexcept {
    if (elements == 0 || elements > UINT32_MAX || !Initialize()) return false;
    std::lock_guard lock(mutex_);
    if (index >= arena_buffers_.size()) return false;
    const auto& value = arena_buffers_[index];
    if (!value.live || value.live_elements < elements) return false;
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(elements) * sizeof(float);
    std::array<VkDescriptorBufferInfo, 5> infos{};
    std::array<VkWriteDescriptorSet, 5> writes{};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked();
    if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {value.buffer, 0, bytes};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!BeginArenaOpLocked()) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
                              0, 1, &arena_set, 0, nullptr);
    std::uint32_t alpha_bits{}, beta_bits{};
    std::memcpy(&alpha_bits, &alpha, sizeof(alpha));
    std::memcpy(&beta_bits, &beta, sizeof(beta));
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; }
        push{static_cast<std::uint32_t>(elements), 1u, static_cast<std::uint32_t>(operation),
             alpha_bits, beta_bits, 0u, 0u, 0u, 0u, 0u, 32u};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    const std::size_t dispatch = (elements + 1023) / 1024;
    if (dispatch == 0 || dispatch > UINT32_MAX) return false;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch), 1, 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaChannelAffine(std::uint32_t value_index, std::uint32_t scale_index,
                             std::uint32_t bias_index, std::size_t batches, int channels,
                             std::size_t plane, int activation = 0) noexcept {
    if (batches == 0 || channels <= 0 || plane == 0 || !Initialize()) return false;
    const std::size_t elements = batches * std::size_t(channels) * plane;
    if (elements > UINT32_MAX) return false;
    std::lock_guard lock(mutex_);
    if (value_index >= arena_buffers_.size() || scale_index >= arena_buffers_.size() ||
        bias_index >= arena_buffers_.size()) return false;
    const auto& value = arena_buffers_[value_index]; const auto& scale = arena_buffers_[scale_index];
    const auto& bias = arena_buffers_[bias_index];
    if (!value.live || !scale.live || !bias.live || value.live_elements < elements ||
        scale.live_elements < std::size_t(channels) || bias.live_elements < std::size_t(channels)) return false;
    std::array<VkDescriptorBufferInfo, 5> infos{}; std::array<VkWriteDescriptorSet, 5> writes{};
    const std::array<const ArenaBuffer*, 5> sources{&value, &scale, &bias, &value, &value};
    const std::array<VkDeviceSize, 5> ranges{
        static_cast<VkDeviceSize>(elements * sizeof(float)),
        static_cast<VkDeviceSize>(std::size_t(channels) * sizeof(float)),
        static_cast<VkDeviceSize>(std::size_t(channels) * sizeof(float)),
        static_cast<VkDeviceSize>(elements * sizeof(float)),
        static_cast<VkDeviceSize>(elements * sizeof(float))};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked();
    if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {sources[binding]->buffer, 0, ranges[binding]};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!BeginArenaOpLocked()) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1,
                              &arena_set, 0, nullptr);
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; }
        push{static_cast<std::uint32_t>(std::size_t(channels) * plane), static_cast<std::uint32_t>(batches),
             0u, 0u, 0u, 0u, 0u, static_cast<std::uint32_t>(plane),
             static_cast<std::uint32_t>(channels), 0u,
             activation == 1 ? 45u : (activation == 2 ? 66u : 33u)};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    const std::size_t dispatch = (std::size_t(channels) * plane + 1023) / 1024;
    if (dispatch == 0 || dispatch > UINT32_MAX) return false;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch), static_cast<std::uint32_t>(batches), 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaRowTransform(std::uint32_t value_index, std::uint32_t gamma_index,
                            std::uint32_t beta_index, std::size_t rows, std::size_t width,
                            float epsilon, bool layer_norm) noexcept {
    if (rows == 0 || width == 0 || rows > UINT32_MAX || width > UINT32_MAX ||
        rows > UINT32_MAX / width || !Initialize()) return false;
    const std::size_t elements = rows * width;
    std::lock_guard lock(mutex_);
    if (value_index >= arena_buffers_.size() || (layer_norm &&
        (gamma_index >= arena_buffers_.size() || beta_index >= arena_buffers_.size()))) return false;
    const auto& value = arena_buffers_[value_index];
    const auto& gamma = layer_norm ? arena_buffers_[gamma_index] : value;
    const auto& beta = layer_norm ? arena_buffers_[beta_index] : value;
    if (!value.live || value.live_elements < elements ||
        (layer_norm && (!gamma.live || !beta.live || gamma.live_elements < width || beta.live_elements < width))) return false;
    std::array<VkDescriptorBufferInfo, 5> infos{}; std::array<VkWriteDescriptorSet, 5> writes{};
    const std::array<const ArenaBuffer*, 5> sources{&value, &gamma, &beta, &value, &value};
    const std::array<VkDeviceSize, 5> ranges{
        static_cast<VkDeviceSize>(elements * sizeof(float)),
        static_cast<VkDeviceSize>((layer_norm ? width : 1) * sizeof(float)),
        static_cast<VkDeviceSize>((layer_norm ? width : 1) * sizeof(float)),
        static_cast<VkDeviceSize>(elements * sizeof(float)), static_cast<VkDeviceSize>(elements * sizeof(float))};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked();
    if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {sources[binding]->buffer, 0, ranges[binding]};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!BeginArenaOpLocked()) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1,
                              &arena_set, 0, nullptr);
    std::uint32_t epsilon_bits{}; std::memcpy(&epsilon_bits, &epsilon, sizeof(epsilon));
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; }
        push{static_cast<std::uint32_t>(elements), 1u, static_cast<std::uint32_t>(rows),
             static_cast<std::uint32_t>(width), epsilon_bits, 0u, 0u, 0u, 0u, 0u,
             layer_norm ? 34u : 35u};
    // LayerNorm already used only lane 0 of a 256-wide group (exact CPU-order
    // reduction). Attention rows are tens of columns, so one workgroup per
    // row left 255 lanes idle. Compact mapping: one thread owns one row.
    // Softmax's cooperative write helps only wide vocab rows; attention
    // softmax is seq-length and stays sequential. Disable with
    // PPOCR_DISABLE_VULKAN_ROW_COMPACT.
    const bool compact = std::getenv("PPOCR_DISABLE_VULKAN_ROW_COMPACT") == nullptr &&
        (layer_norm || width <= 256);
    push.operation3 = compact ? 1u : 0u;
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    cmd_dispatch_(command_buffer_,
                  compact ? static_cast<std::uint32_t>((rows + 255) / 256)
                          : static_cast<std::uint32_t>(rows),
                  1, 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaRgbResize(std::uint32_t rgb_index, std::uint32_t output_index,
                         int source_width, int source_height, int left, int top,
                         int region_width, int region_height, int output_width,
                         int output_height, bool recognition,
                         std::size_t output_offset_elements = 0,
                         int content_width = 0) noexcept {
    if (source_width <= 0 || source_height <= 0 || left < 0 || top < 0 || region_width <= 0 ||
        region_height <= 0 || output_width <= 0 || output_height <= 0 || left + region_width > source_width ||
        top + region_height > source_height || !Initialize()) return false;
    const std::size_t rgb_elements = std::size_t(source_width) * source_height * 3;
    const std::size_t output_elements = std::size_t(output_width) * output_height * 3;
    if (rgb_elements > UINT32_MAX || output_elements > UINT32_MAX ||
        output_offset_elements > std::numeric_limits<std::size_t>::max() - output_elements) return false;
    std::lock_guard lock(mutex_);
    if (rgb_index >= arena_buffers_.size() || output_index >= arena_buffers_.size()) return false;
    const auto& rgb = arena_buffers_[rgb_index]; const auto& output = arena_buffers_[output_index];
    if (!rgb.live || !output.live || rgb.live_elements < rgb_elements ||
        output.live_elements < output_offset_elements + output_elements) return false;
    std::array<VkDescriptorBufferInfo, 5> infos{}; std::array<VkWriteDescriptorSet, 5> writes{};
    const std::array<const ArenaBuffer*, 5> sources{&rgb, &rgb, &rgb, &output, &rgb};
    const std::array<VkDeviceSize, 5> ranges{VkDeviceSize(rgb_elements), sizeof(float), sizeof(float),
                                              VkDeviceSize(output_elements * sizeof(float)), sizeof(float)};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked(); if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {sources[binding]->buffer,
                        binding == 3 ? VkDeviceSize(output_offset_elements * sizeof(float)) : 0,
                        ranges[binding]};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!BeginArenaOpLocked()) return false;
    // The RGB staging upload must be visible before its first shader read
    // when an explicitly requested device-local arena is in use. The default
    // coherent arena needs no copy and avoids a transfer submission on UMA.
    if (!CopyStagingToArenaLocked(arena_buffers_[rgb_index], rgb_elements)) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &arena_set, 0, nullptr);
    const std::uint32_t packed_content =
        (content_width > 0 && content_width < output_width)
            ? (static_cast<std::uint32_t>(content_width) << 16)
            : 0u;
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; }
        push{static_cast<std::uint32_t>(output_elements), 1u, static_cast<std::uint32_t>(source_width),
             static_cast<std::uint32_t>(source_height), static_cast<std::uint32_t>(left), static_cast<std::uint32_t>(top),
             static_cast<std::uint32_t>(region_width), static_cast<std::uint32_t>(region_height),
             static_cast<std::uint32_t>(output_width), static_cast<std::uint32_t>(output_height),
             (recognition ? 47u : 46u) | packed_content};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>((output_elements + 1023) / 1024), 1, 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 1, &barrier, 0, nullptr, 0, nullptr);
    // Shader output is the source of truth; do not later copy stale staging
    // zeros over this NCHW at graph-record time.
    arena_buffers_[output_index].staging_dirty = false;
    return EndArenaOpLocked();
  }

  bool RunArenaRgbResizeBatch(std::uint32_t rgb_index, std::uint32_t output_index,
                              int batches, int source_width, int source_height,
                              int output_width, int output_height) noexcept {
    if (batches <= 0 || source_width <= 0 || source_height <= 0 || output_width <= 0 ||
        output_height <= 0 || !Initialize()) return false;
    const std::size_t source_elements = std::size_t(source_width) * source_height * 3;
    const std::size_t output_elements = std::size_t(output_width) * output_height * 3;
    if (source_elements > UINT32_MAX || output_elements > UINT32_MAX ||
        std::size_t(batches) > UINT32_MAX) return false;
    std::lock_guard lock(mutex_);
    if (rgb_index >= arena_buffers_.size() || output_index >= arena_buffers_.size()) return false;
    const auto& rgb = arena_buffers_[rgb_index]; const auto& output = arena_buffers_[output_index];
    if (!rgb.live || !output.live || rgb.live_elements < source_elements * std::size_t(batches) ||
        output.live_elements < output_elements * std::size_t(batches)) return false;
    std::array<VkDescriptorBufferInfo, 5> infos{}; std::array<VkWriteDescriptorSet, 5> writes{};
    const std::array<const ArenaBuffer*, 5> sources{&rgb, &rgb, &rgb, &output, &rgb};
    const std::array<VkDeviceSize, 5> ranges{
        VkDeviceSize(source_elements * std::size_t(batches)), sizeof(float), sizeof(float),
        VkDeviceSize(output_elements * std::size_t(batches) * sizeof(float)), sizeof(float)};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked(); if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {sources[binding]->buffer, 0, ranges[binding]};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!BeginArenaOpLocked() || !CopyStagingToArenaLocked(arena_buffers_[rgb_index], source_elements * std::size_t(batches))) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &arena_set, 0, nullptr);
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; }
        push{static_cast<std::uint32_t>(output_elements), static_cast<std::uint32_t>(batches),
             static_cast<std::uint32_t>(source_width), static_cast<std::uint32_t>(source_height),
             0u, 0u, static_cast<std::uint32_t>(source_width), static_cast<std::uint32_t>(source_height),
             static_cast<std::uint32_t>(output_width), static_cast<std::uint32_t>(output_height), 46u};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>((output_elements + 1023) / 1024),
                  static_cast<std::uint32_t>(batches), 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaThreshold(std::uint32_t input_index, std::uint32_t output_index,
                          std::size_t elements, float threshold) noexcept {
    if (elements == 0 || elements > UINT32_MAX || !Initialize()) return false;
    std::lock_guard lock(mutex_);
    if (input_index >= arena_buffers_.size() || output_index >= arena_buffers_.size()) return false;
    const auto& input = arena_buffers_[input_index]; const auto& output = arena_buffers_[output_index];
    if (!input.live || !output.live || input.live_elements < elements || output.live_elements < elements) return false;
    std::array<VkDescriptorBufferInfo, 5> infos{}; std::array<VkWriteDescriptorSet, 5> writes{};
    const std::array<const ArenaBuffer*, 5> sources{&input, &input, &input, &output, &input};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked(); if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {sources[binding]->buffer, 0, VkDeviceSize(elements * sizeof(float))};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!BeginArenaOpLocked()) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &arena_set, 0, nullptr);
    std::uint32_t threshold_bits{}; std::memcpy(&threshold_bits, &threshold, sizeof(threshold));
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; }
        push{static_cast<std::uint32_t>(elements), 1u, threshold_bits, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 49u};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>((elements + 1023) / 1024), 1, 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaDbPostprocess(std::uint32_t probability_index,
                             std::size_t probability_offset_elements,
                             std::uint32_t state_index, std::uint32_t result_index,
                             int height, int width, int image_width, int image_height,
                             float threshold, float box_threshold,
                             float unclip_ratio, std::size_t result_elements) noexcept {
    if (height <= 0 || width <= 0 || image_width <= 0 || image_height <= 0 ||
        result_elements < 6 || !Initialize()) return false;
    const std::size_t elements = std::size_t(height) * width;
    if (elements > UINT32_MAX || probability_offset_elements >
        std::numeric_limits<std::size_t>::max() - elements ||
        result_elements > UINT32_MAX) return false;
    const std::size_t capacity = (result_elements - 1) / 5;
    if (capacity == 0 || capacity > UINT32_MAX) return false;
    std::lock_guard lock(mutex_);
    if (probability_index >= arena_buffers_.size() || state_index >= arena_buffers_.size() ||
        result_index >= arena_buffers_.size()) return false;
    const auto& probability = arena_buffers_[probability_index];
    const auto& state = arena_buffers_[state_index];
    const auto& result = arena_buffers_[result_index];
    if (!probability.live || !state.live || !result.live ||
        probability.live_elements < probability_offset_elements + elements ||
        state.live_elements < elements * 2 || result.live_elements < result_elements) return false;
    std::array<VkDescriptorBufferInfo, 5> infos{}; std::array<VkWriteDescriptorSet, 5> writes{};
    const std::array<const ArenaBuffer*, 5> sources{&probability, &state, &result, &result, &probability};
    const std::array<VkDeviceSize, 5> ranges{
        VkDeviceSize(elements * sizeof(float)), VkDeviceSize(elements * 2 * sizeof(float)),
        VkDeviceSize(result_elements * sizeof(float)), VkDeviceSize(result_elements * sizeof(float)),
        sizeof(float)};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked();
    if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {sources[binding]->buffer,
                        binding == 0 ? VkDeviceSize(probability_offset_elements * sizeof(float)) : 0,
                        ranges[binding]};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!BeginArenaOpLocked()) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1,
                              &arena_set, 0, nullptr);
    std::uint32_t threshold_bits{}, box_threshold_bits{}, unclip_bits{};
    std::memcpy(&threshold_bits, &threshold, sizeof(threshold));
    std::memcpy(&box_threshold_bits, &box_threshold, sizeof(box_threshold));
    std::memcpy(&unclip_bits, &unclip_ratio, sizeof(unclip_ratio));
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; }
        push{static_cast<std::uint32_t>(elements), 1u, threshold_bits, box_threshold_bits,
             unclip_bits, static_cast<std::uint32_t>(image_width),
             static_cast<std::uint32_t>(image_height), static_cast<std::uint32_t>(height),
             static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(capacity), 54u};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    cmd_dispatch_(command_buffer_, 1, 1, 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaCtcTop1(std::uint32_t logits_index, std::uint32_t indices_index,
                       std::uint32_t probabilities_index, int batches, int steps, int classes,
                       bool input_is_logits) noexcept {
    if (batches <= 0 || steps <= 0 || classes <= 0 || !Initialize()) return false;
    const std::size_t rows = std::size_t(batches) * steps;
    const std::size_t logits_elements = rows * classes;
    if (rows > UINT32_MAX || logits_elements > UINT32_MAX) return false;
    std::lock_guard lock(mutex_);
    if (logits_index >= arena_buffers_.size() || indices_index >= arena_buffers_.size() || probabilities_index >= arena_buffers_.size()) return false;
    const auto& logits = arena_buffers_[logits_index]; const auto& indices = arena_buffers_[indices_index];
    const auto& probabilities = arena_buffers_[probabilities_index];
    if (!logits.live || !indices.live || !probabilities.live || logits.live_elements < logits_elements ||
        indices.live_elements < rows || probabilities.live_elements < rows) return false;
    std::array<VkDescriptorBufferInfo, 5> infos{}; std::array<VkWriteDescriptorSet, 5> writes{};
    const std::array<const ArenaBuffer*, 5> sources{&logits, &logits, &logits, &indices, &probabilities};
    const std::array<VkDeviceSize, 5> ranges{VkDeviceSize(logits_elements * sizeof(float)), sizeof(float), sizeof(float),
                                              VkDeviceSize(rows * sizeof(float)), VkDeviceSize(rows * sizeof(float))};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked(); if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {sources[binding]->buffer, 0, ranges[binding]};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!BeginArenaOpLocked()) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &arena_set, 0, nullptr);
    // The scalar compatibility reduction preserves the exported CTC's
    // sequential sum order.  On short/near-tie rows that numerical stability
    // matters more than the cooperative kernel's throughput; deployments can
    // opt in after qualifying their own model/driver pair.
    const bool cooperative_ctc =
        std::getenv("PPOCR_DISABLE_GPU_CTC_COOPERATIVE") == nullptr;
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; }
        push{static_cast<std::uint32_t>(rows), 1u, static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(classes),
             0u, 0u, 0u, 0u, 0u, 0u,
             (input_is_logits ? (48u | 0x80000000u) : 48u) |
                 (cooperative_ctc ? 0x40000000u : 0u)};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    // Cooperative CTC is now the default once CTC lives inside the persistent
    // rec command buffer: one 256-lane workgroup per vocabulary row beats the
    // scalar-per-row dispatch on this UMA adapter. `PPOCR_DISABLE_GPU_CTC_COOPERATIVE`
    // restores the one-thread-per-row path.
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(
                      cooperative_ctc ? rows : (rows + 255) / 256), 1, 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaTranspose4D(std::uint32_t input_index, std::uint32_t output_index,
                           const std::array<int, 4>& dimensions,
                           const std::array<int, 4>& permutation) noexcept {
    std::size_t elements = 1; std::uint32_t permutation_bits{};
    for (std::size_t axis = 0; axis < 4; ++axis) {
      if (dimensions[axis] <= 0 || permutation[axis] < 0 || permutation[axis] >= 4 ||
          elements > UINT32_MAX / std::size_t(dimensions[axis])) return false;
      for (std::size_t previous = 0; previous < axis; ++previous)
        if (permutation[axis] == permutation[previous]) return false;
      elements *= std::size_t(dimensions[axis]);
      permutation_bits |= std::uint32_t(permutation[axis]) << (axis * 2);
    }
    if (!Initialize()) return false;
    std::lock_guard lock(mutex_);
    if (input_index >= arena_buffers_.size() || output_index >= arena_buffers_.size()) return false;
    const auto& input = arena_buffers_[input_index]; const auto& output = arena_buffers_[output_index];
    if (!input.live || !output.live || input.live_elements < elements || output.live_elements < elements) return false;
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(elements * sizeof(float));
    std::array<VkDescriptorBufferInfo, 5> infos{}; std::array<VkWriteDescriptorSet, 5> writes{};
    const std::array<const ArenaBuffer*, 5> sources{&input, &input, &input, &output, &input};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked();
    if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {sources[binding]->buffer, 0, bytes};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!BeginArenaOpLocked()) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1,
                              &arena_set, 0, nullptr);
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; }
        push{static_cast<std::uint32_t>(elements), 1u, static_cast<std::uint32_t>(dimensions[0]),
             static_cast<std::uint32_t>(dimensions[1]), static_cast<std::uint32_t>(dimensions[2]),
             static_cast<std::uint32_t>(dimensions[3]), permutation_bits, 0u, 0u, 0u, 36u};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    const std::size_t dispatch = (elements + 1023) / 1024;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch), 1, 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaSpatialMean(std::uint32_t input_index, std::uint32_t output_index,
                           std::size_t batches, int channels, std::size_t plane) noexcept {
    if (batches == 0 || channels <= 0 || plane == 0 || batches > UINT32_MAX ||
        std::size_t(channels) > UINT32_MAX / batches || !Initialize()) return false;
    const std::size_t output_elements = batches * std::size_t(channels);
    if (output_elements > UINT32_MAX || output_elements > UINT32_MAX / plane) return false;
    const std::size_t input_elements = output_elements * plane;
    std::lock_guard lock(mutex_);
    if (input_index >= arena_buffers_.size() || output_index >= arena_buffers_.size()) return false;
    const auto& input = arena_buffers_[input_index]; const auto& output = arena_buffers_[output_index];
    if (!input.live || !output.live || input.live_elements < input_elements ||
        output.live_elements < output_elements) return false;
    std::array<VkDescriptorBufferInfo, 5> infos{}; std::array<VkWriteDescriptorSet, 5> writes{};
    const std::array<const ArenaBuffer*, 5> sources{&input, &input, &input, &output, &input};
    const std::array<VkDeviceSize, 5> ranges{
        static_cast<VkDeviceSize>(input_elements * sizeof(float)), sizeof(float), sizeof(float),
        static_cast<VkDeviceSize>(output_elements * sizeof(float)), sizeof(float)};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked();
    if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {sources[binding]->buffer, 0, ranges[binding]};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!BeginArenaOpLocked()) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1,
                              &arena_set, 0, nullptr);
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; }
        push{static_cast<std::uint32_t>(output_elements), 1u, static_cast<std::uint32_t>(output_elements),
             static_cast<std::uint32_t>(plane), 0u, 0u, 0u, 0u, 0u, 0u, 37u};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    const std::size_t dispatch = (output_elements + 255) / 256;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch), 1, 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaQkvSplit(std::uint32_t input_index, std::uint32_t query_index,
                        std::uint32_t key_index, std::uint32_t value_index,
                        std::size_t batches, int steps, int heads, int head_width) noexcept {
    if (batches == 0 || steps <= 0 || heads <= 0 || head_width <= 0 || batches > UINT32_MAX ||
        !Initialize()) return false;
    const std::size_t branch_elements = batches * std::size_t(steps) * heads * head_width;
    if (branch_elements == 0 || branch_elements > UINT32_MAX / 3) return false;
    const std::size_t input_elements = branch_elements * 3;
    std::lock_guard lock(mutex_);
    if (input_index >= arena_buffers_.size() || query_index >= arena_buffers_.size() ||
        key_index >= arena_buffers_.size() || value_index >= arena_buffers_.size()) return false;
    const auto& input = arena_buffers_[input_index]; const auto& query = arena_buffers_[query_index];
    const auto& key = arena_buffers_[key_index]; const auto& value = arena_buffers_[value_index];
    if (!input.live || !query.live || !key.live || !value.live || input.live_elements < input_elements ||
        query.live_elements < branch_elements || key.live_elements < branch_elements ||
        value.live_elements < branch_elements || input_index == query_index || input_index == key_index ||
        input_index == value_index) return false;
    std::array<VkDescriptorBufferInfo, 5> infos{}; std::array<VkWriteDescriptorSet, 5> writes{};
    // Shader binding A is source, D query, E key. Binding B is a harmless
    // alias and C carries the V output (the shader never reads B/C in mode38).
    const std::array<const ArenaBuffer*, 5> sources{&input, &input, &value, &query, &key};
    const std::array<VkDeviceSize, 5> ranges{
        static_cast<VkDeviceSize>(input_elements * sizeof(float)), sizeof(float),
        static_cast<VkDeviceSize>(branch_elements * sizeof(float)),
        static_cast<VkDeviceSize>(branch_elements * sizeof(float)),
        static_cast<VkDeviceSize>(branch_elements * sizeof(float))};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked();
    if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {sources[binding]->buffer, 0, ranges[binding]};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!BeginArenaOpLocked()) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1,
                              &arena_set, 0, nullptr);
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; }
        push{static_cast<std::uint32_t>(branch_elements), 1u, static_cast<std::uint32_t>(batches),
             static_cast<std::uint32_t>(steps), static_cast<std::uint32_t>(heads),
             static_cast<std::uint32_t>(head_width), 0u, 0u, 0u, 0u, 38u};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    const std::size_t dispatch = (branch_elements + 1023) / 1024;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch), 1, 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaBatchedGemm(std::uint32_t left_index, std::uint32_t right_index,
                           std::uint32_t output_index, std::size_t matrix_batches,
                           int rows, int depth, int columns) noexcept {
    if (matrix_batches == 0 || rows <= 0 || depth <= 0 || columns <= 0 ||
        matrix_batches > UINT32_MAX || !Initialize()) return false;
    const std::size_t left_matrix = std::size_t(rows) * depth;
    const std::size_t right_matrix = std::size_t(depth) * columns;
    const std::size_t output_matrix = std::size_t(rows) * columns;
    if (left_matrix > UINT32_MAX || right_matrix > UINT32_MAX || output_matrix > UINT32_MAX ||
        matrix_batches > std::numeric_limits<std::size_t>::max() / left_matrix ||
        matrix_batches > std::numeric_limits<std::size_t>::max() / right_matrix ||
        matrix_batches > std::numeric_limits<std::size_t>::max() / output_matrix) return false;
    const std::size_t left_elements = matrix_batches * left_matrix;
    const std::size_t right_elements = matrix_batches * right_matrix;
    const std::size_t output_elements = matrix_batches * output_matrix;
    std::lock_guard lock(mutex_);
    if (left_index >= arena_buffers_.size() || right_index >= arena_buffers_.size() ||
        output_index >= arena_buffers_.size()) return false;
    const auto& left = arena_buffers_[left_index]; const auto& right = arena_buffers_[right_index];
    const auto& output = arena_buffers_[output_index];
    if (!left.live || !right.live || !output.live || left.live_elements < left_elements ||
        right.live_elements < right_elements || output.live_elements < output_elements) return false;
    std::array<VkDescriptorBufferInfo, 5> infos{}; std::array<VkWriteDescriptorSet, 5> writes{};
    const std::array<const ArenaBuffer*, 5> sources{&left, &right, &left, &output, &left};
    const std::array<VkDeviceSize, 5> ranges{
        static_cast<VkDeviceSize>(left_elements * sizeof(float)),
        static_cast<VkDeviceSize>(right_elements * sizeof(float)), sizeof(float),
        static_cast<VkDeviceSize>(output_elements * sizeof(float)), sizeof(float)};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked();
    if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {sources[binding]->buffer, 0, ranges[binding]};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!BeginArenaOpLocked()) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1,
                              &arena_set, 0, nullptr);
    // One thread owns four consecutive columns of one row and reuses each A
    // scalar across those columns. The old mapping walked output_matrix with
    // four independent K loops (no A reuse). Disable with
    // PPOCR_DISABLE_VULKAN_BATCHED_GEMM_ROW.
    // operation3==2 is the mode-24 8x128 LDS tile (Z = matrix). Disable with
    // PPOCR_DISABLE_VULKAN_BATCHED_GEMM_TILE.
    const bool row_kernel =
        std::getenv("PPOCR_DISABLE_VULKAN_BATCHED_GEMM_ROW") == nullptr;
    const bool tiled_kernel = row_kernel && rows >= 8 && depth >= 16 && columns >= 32 &&
        std::getenv("PPOCR_ENABLE_VULKAN_BATCHED_GEMM_TILE") != nullptr &&
        std::getenv("PPOCR_DISABLE_VULKAN_BATCHED_GEMM_TILE") == nullptr;
    const std::uint32_t gemm_variant = tiled_kernel ? 2u : (row_kernel ? 1u : 0u);
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; }
        push{static_cast<std::uint32_t>(output_matrix), static_cast<std::uint32_t>(matrix_batches),
             static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(depth),
             static_cast<std::uint32_t>(columns), gemm_variant, 0u, 0u, 0u, 0u, 39u};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    const std::size_t col4 = (std::size_t(columns) + 3) / 4;
    const std::size_t dispatch_x = tiled_kernel
        ? (std::size_t(columns) + 127) / 128
        : (row_kernel ? (std::size_t(rows) * col4 + 255) / 256
                      : (output_matrix + 1023) / 1024);
    const std::size_t dispatch_y = tiled_kernel
        ? (std::size_t(rows) + 7) / 8
        : matrix_batches;
    const std::size_t dispatch_z = tiled_kernel ? matrix_batches : 1;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch_x),
                  static_cast<std::uint32_t>(dispatch_y),
                  static_cast<std::uint32_t>(dispatch_z));
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaPool2d(std::uint32_t input_index, std::uint32_t output_index,
                      std::size_t batches, int channels, int input_height, int input_width,
                      int output_height, int output_width, int kernel_height, int kernel_width,
                      int stride_height, int stride_width, int pad_top, int pad_left,
                      bool average) noexcept {
    if (batches == 0 || channels <= 0 || input_height <= 0 || input_width <= 0 || output_height <= 0 ||
        output_width <= 0 || kernel_height <= 0 || kernel_width <= 0 || stride_height <= 0 ||
        stride_width <= 0 || pad_top < 0 || pad_left < 0 || stride_height > 255 || stride_width > 255 ||
        pad_top > 255 || pad_left > 255 || !Initialize()) return false;
    const std::size_t input_elements = batches * std::size_t(channels) * input_height * input_width;
    const std::size_t output_elements = batches * std::size_t(channels) * output_height * output_width;
    if (input_elements > UINT32_MAX || output_elements > UINT32_MAX) return false;
    std::lock_guard lock(mutex_);
    if (input_index >= arena_buffers_.size() || output_index >= arena_buffers_.size()) return false;
    const auto& input = arena_buffers_[input_index]; const auto& output = arena_buffers_[output_index];
    if (!input.live || !output.live || input.live_elements < input_elements || output.live_elements < output_elements) return false;
    std::array<VkDescriptorBufferInfo, 5> infos{}; std::array<VkWriteDescriptorSet, 5> writes{};
    const std::array<const ArenaBuffer*, 5> sources{&input, &input, &input, &output, &input};
    const std::array<VkDeviceSize, 5> ranges{
        static_cast<VkDeviceSize>(input_elements * sizeof(float)), sizeof(float), sizeof(float),
        static_cast<VkDeviceSize>(output_elements * sizeof(float)), sizeof(float)};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked();
    if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {sources[binding]->buffer, 0, ranges[binding]};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!BeginArenaOpLocked()) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1,
                              &arena_set, 0, nullptr);
    const std::uint32_t op0 = (average ? 1u : 0u) | (std::uint32_t(pad_top) << 8u) |
        (std::uint32_t(pad_left) << 16u);
    const std::uint32_t mode = 40u | (std::uint32_t(stride_height) << 16u) |
        (std::uint32_t(stride_width) << 24u);
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; }
        push{static_cast<std::uint32_t>(std::size_t(channels) * output_height * output_width),
             static_cast<std::uint32_t>(batches), op0, static_cast<std::uint32_t>(channels),
             static_cast<std::uint32_t>(input_height), static_cast<std::uint32_t>(input_width),
             static_cast<std::uint32_t>(output_height), static_cast<std::uint32_t>(output_width),
             static_cast<std::uint32_t>(kernel_height), static_cast<std::uint32_t>(kernel_width), mode};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    const std::size_t dispatch = (std::size_t(channels) * output_height * output_width + 1023) / 1024;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch), static_cast<std::uint32_t>(batches), 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaPointwise(std::uint32_t input_index, std::uint32_t weights_index,
                         std::uint32_t bias_index, std::uint32_t output_index,
                         std::size_t batches, int input_channels, int output_channels,
                         std::size_t plane, bool relu, bool swish, bool gelu,
                         bool hard_swish) noexcept {
    if (batches == 0 || input_channels <= 0 || output_channels <= 0 || plane == 0 ||
        (int(relu) + int(swish) + int(gelu) + int(hard_swish) > 1) || batches > UINT32_MAX ||
        plane > UINT32_MAX || !Initialize()) return false;
    const std::size_t input_elements = batches * std::size_t(input_channels) * plane;
    const std::size_t output_elements = batches * std::size_t(output_channels) * plane;
    const std::size_t weight_elements = std::size_t(input_channels) * output_channels;
    if (input_elements > UINT32_MAX || output_elements > UINT32_MAX) return false;
    std::lock_guard lock(mutex_);
    if (input_index >= arena_buffers_.size() || weights_index >= arena_buffers_.size() ||
        bias_index >= arena_buffers_.size() || output_index >= arena_buffers_.size()) return false;
    const auto& input = arena_buffers_[input_index];
    const auto& weights = arena_buffers_[weights_index];
    const auto& bias = arena_buffers_[bias_index];
    const auto& output = arena_buffers_[output_index];
    if (!input.live || !weights.live || !bias.live || !output.live ||
        input.live_elements < input_elements || weights.live_elements < weight_elements ||
        bias.live_elements < std::size_t(output_channels) || output.live_elements < output_elements) return false;
    std::array<VkDescriptorBufferInfo, 5> infos{};
    std::array<VkWriteDescriptorSet, 5> writes{};
    // Pointwise modes do not consume E, but bind a valid input-sized view so
    // all descriptor ranges match their backing VkBuffer.  A previous
    // diagnostic alias bound `weights` with an activation-sized range here;
    // that is invalid when the activation is larger than [M,C] and can make
    // permissive drivers report a later device loss.
    const std::array<const ArenaBuffer*, 5> sources{&input, &weights, &bias, &output, &input};
    const std::array<VkDeviceSize, 5> ranges{
        static_cast<VkDeviceSize>(input_elements * sizeof(float)),
        static_cast<VkDeviceSize>(weight_elements * sizeof(float)),
        static_cast<VkDeviceSize>(std::size_t(output_channels) * sizeof(float)),
        static_cast<VkDeviceSize>(output_elements * sizeof(float)),
        static_cast<VkDeviceSize>(input_elements * sizeof(float))};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked();
    if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {sources[binding]->buffer, 0, ranges[binding]};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!BeginArenaOpLocked()) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
                              0, 1, &arena_set, 0, nullptr);
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; };
    // Mode 55 is a direct pointwise write with the shader's high-accuracy
    // FP32 GELU applied before storing. It removes the otherwise separate
    // full activation read/write from fused depthwise-pointwise-GELU blocks.
    const std::uint32_t mode = relu ? 4u : (swish ? 7u : (gelu ? 55u : (hard_swish ? 12u : 3u)));
    std::uint32_t alpha_bits = 0, beta_bits = 0;
    if (hard_swish) {
      const float alpha = 1.F / 6.F, beta = .5F;
      std::memcpy(&alpha_bits, &alpha, sizeof(alpha));
      std::memcpy(&beta_bits, &beta, sizeof(beta));
    }
    Push push{static_cast<std::uint32_t>(output_channels * plane), static_cast<std::uint32_t>(batches),
              alpha_bits, beta_bits, 0u, 0u, 1u, static_cast<std::uint32_t>(plane),
              static_cast<std::uint32_t>(input_channels), static_cast<std::uint32_t>(output_channels), mode};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                        sizeof(push), &push);
    const auto dispatch = input_channels <= 1024
        ? ((static_cast<std::size_t>(output_channels) + 3) / 4) * ((plane + 255) / 256)
        : (std::size_t(output_channels) * plane + 1023) / 1024;
    if (dispatch == 0 || dispatch > UINT32_MAX) return false;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch), static_cast<std::uint32_t>(batches), 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaPointwiseTail(std::uint32_t input_index, std::uint32_t weights_index,
                             std::uint32_t bias_index, std::uint32_t output_index,
                             std::size_t batches, int input_channels, int output_channels,
                             std::size_t plane) noexcept {
    if (batches == 0 || input_channels <= 0 || output_channels <= 0 || plane == 0 ||
        batches > UINT32_MAX || plane > UINT32_MAX || !Initialize()) return false;
    const std::size_t input_elements = batches * std::size_t(input_channels) * plane;
    const std::size_t output_elements = batches * std::size_t(output_channels) * plane;
    const std::size_t weight_elements = std::size_t(input_channels) * output_channels;
    if (input_elements > UINT32_MAX || output_elements > UINT32_MAX) return false;
    std::lock_guard lock(mutex_);
    if (tail_pointwise_pipeline_ == VK_NULL_HANDLE || input_index >= arena_buffers_.size() ||
        weights_index >= arena_buffers_.size() || bias_index >= arena_buffers_.size() ||
        output_index >= arena_buffers_.size()) return false;
    const auto& input = arena_buffers_[input_index];
    const auto& weights = arena_buffers_[weights_index];
    const auto& bias = arena_buffers_[bias_index];
    const auto& output = arena_buffers_[output_index];
    if (!input.live || !weights.live || !bias.live || !output.live ||
        input.live_elements < input_elements || weights.live_elements < weight_elements ||
        bias.live_elements < std::size_t(output_channels) || output.live_elements < output_elements) return false;
    std::array<VkDescriptorBufferInfo, 5> infos{};
    std::array<VkWriteDescriptorSet, 5> writes{};
    const std::array<const ArenaBuffer*, 5> sources{&input, &weights, &bias, &output, &input};
    const std::array<VkDeviceSize, 5> ranges{
        VkDeviceSize(input_elements * sizeof(float)), VkDeviceSize(weight_elements * sizeof(float)),
        VkDeviceSize(std::size_t(output_channels) * sizeof(float)),
        VkDeviceSize(output_elements * sizeof(float)), VkDeviceSize(input_elements * sizeof(float))};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked();
    if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {sources[binding]->buffer, 0, ranges[binding]};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!BeginArenaOpLocked()) return false;
    cmd_bind_pipeline_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, tail_pointwise_pipeline_);
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
                              0, 1, &arena_set, 0, nullptr);
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; };
    Push push{static_cast<std::uint32_t>(output_channels * plane), static_cast<std::uint32_t>(batches),
              0u, 0u, 0u, 0u, 1u, static_cast<std::uint32_t>(plane),
              static_cast<std::uint32_t>(input_channels), static_cast<std::uint32_t>(output_channels), 56u};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    const std::size_t dispatch = (std::size_t(output_channels) * plane + 1023) / 1024;
    if (dispatch == 0 || dispatch > UINT32_MAX) return false;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch), static_cast<std::uint32_t>(batches), 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaPointwiseAdd(std::uint32_t input_index, std::uint32_t weights_index,
                            std::uint32_t bias_index, std::uint32_t residual_index,
                            std::uint32_t output_index, std::size_t batches,
                            int input_channels, int output_channels,
                            std::size_t plane, bool relu, bool swish) noexcept {
    if (batches == 0 || input_channels <= 0 || output_channels <= 0 || plane == 0 ||
        (relu && swish) || batches > UINT32_MAX || plane > UINT32_MAX || !Initialize()) return false;
    const std::size_t input_elements = batches * std::size_t(input_channels) * plane;
    const std::size_t output_elements = batches * std::size_t(output_channels) * plane;
    const std::size_t weight_elements = std::size_t(input_channels) * output_channels;
    if (input_elements > UINT32_MAX || output_elements > UINT32_MAX) return false;
    std::lock_guard lock(mutex_);
    if (input_index >= arena_buffers_.size() || weights_index >= arena_buffers_.size() ||
        bias_index >= arena_buffers_.size() || residual_index >= arena_buffers_.size() ||
        output_index >= arena_buffers_.size()) return false;
    const auto& input = arena_buffers_[input_index];
    const auto& weights = arena_buffers_[weights_index];
    const auto& bias = arena_buffers_[bias_index];
    const auto& residual = arena_buffers_[residual_index];
    const auto& output = arena_buffers_[output_index];
    if (!input.live || !weights.live || !bias.live || !residual.live || !output.live ||
        input.live_elements < input_elements || weights.live_elements < weight_elements ||
        bias.live_elements < std::size_t(output_channels) || residual.live_elements < output_elements ||
        output.live_elements < output_elements) return false;
    std::array<VkDescriptorBufferInfo, 5> infos{};
    std::array<VkWriteDescriptorSet, 5> writes{};
    const std::array<const ArenaBuffer*, 5> sources{&input, &weights, &bias, &output, &residual};
    const std::array<VkDeviceSize, 5> ranges{
        static_cast<VkDeviceSize>(input_elements * sizeof(float)),
        static_cast<VkDeviceSize>(weight_elements * sizeof(float)),
        static_cast<VkDeviceSize>(std::size_t(output_channels) * sizeof(float)),
        static_cast<VkDeviceSize>(output_elements * sizeof(float)),
        static_cast<VkDeviceSize>(output_elements * sizeof(float))};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked();
    if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {sources[binding]->buffer, 0, ranges[binding]};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!BeginArenaOpLocked()) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
                              0, 1, &arena_set, 0, nullptr);
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; };
    const std::uint32_t mode = relu ? 6u : (swish ? 9u : 5u);
    Push push{static_cast<std::uint32_t>(output_channels * plane), static_cast<std::uint32_t>(batches),
              0u, 0u, 0u, 0u, 1u, static_cast<std::uint32_t>(plane),
              static_cast<std::uint32_t>(input_channels), static_cast<std::uint32_t>(output_channels), mode};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                        sizeof(push), &push);
    const auto dispatch = input_channels <= 1024
        ? ((static_cast<std::size_t>(output_channels) + 3) / 4) * ((plane + 255) / 256)
        : (std::size_t(output_channels) * plane + 1023) / 1024;
    if (dispatch == 0 || dispatch > UINT32_MAX) return false;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch), static_cast<std::uint32_t>(batches), 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaExpandGeluProjectAdd(std::uint32_t input_index, std::uint32_t expand_index,
                                    std::uint32_t packed_bias_index, std::uint32_t project_index,
                                    std::uint32_t output_index, std::size_t batches, int channels,
                                    int hidden, std::size_t plane) noexcept {
    if (batches == 0 || channels <= 0 || hidden <= 0 || plane == 0 || hidden > 256 ||
        batches > UINT32_MAX || plane > UINT32_MAX || !Initialize()) return false;
    if (std::getenv("PPOCR_DISABLE_VULKAN_EXPAND_PROJECT") != nullptr) return false;
    const std::size_t input_elements = batches * std::size_t(channels) * plane;
    const std::size_t expand_elements = std::size_t(hidden) * channels;
    const std::size_t project_elements = std::size_t(channels) * hidden;
    const std::size_t bias_elements = std::size_t(hidden) + channels;
    if (input_elements > UINT32_MAX) return false;
    std::lock_guard lock(mutex_);
    if (input_index >= arena_buffers_.size() || expand_index >= arena_buffers_.size() ||
        packed_bias_index >= arena_buffers_.size() || project_index >= arena_buffers_.size() ||
        output_index >= arena_buffers_.size()) return false;
    const auto& input = arena_buffers_[input_index];
    const auto& expand = arena_buffers_[expand_index];
    const auto& packed_bias = arena_buffers_[packed_bias_index];
    const auto& project = arena_buffers_[project_index];
    const auto& output = arena_buffers_[output_index];
    if (!input.live || !expand.live || !packed_bias.live || !project.live || !output.live ||
        input.live_elements < input_elements || expand.live_elements < expand_elements ||
        packed_bias.live_elements < bias_elements || project.live_elements < project_elements ||
        output.live_elements < input_elements) return false;
    std::array<VkDescriptorBufferInfo, 5> infos{};
    std::array<VkWriteDescriptorSet, 5> writes{};
    const std::array<const ArenaBuffer*, 5> sources{&input, &expand, &packed_bias, &output, &project};
    const std::array<VkDeviceSize, 5> ranges{
        static_cast<VkDeviceSize>(input_elements * sizeof(float)),
        static_cast<VkDeviceSize>(expand_elements * sizeof(float)),
        static_cast<VkDeviceSize>(bias_elements * sizeof(float)),
        static_cast<VkDeviceSize>(input_elements * sizeof(float)),
        static_cast<VkDeviceSize>(project_elements * sizeof(float))};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked();
    if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {sources[binding]->buffer, 0, ranges[binding]};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!BeginArenaOpLocked()) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
                              0, 1, &arena_set, 0, nullptr);
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; }
        push{static_cast<std::uint32_t>(std::size_t(channels) * plane),
             static_cast<std::uint32_t>(batches), static_cast<std::uint32_t>(channels),
             static_cast<std::uint32_t>(hidden), static_cast<std::uint32_t>(plane), 0u,
             0u, 0u, 0u, 0u, 58u};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                        sizeof(push), &push);
    const std::size_t tile = std::min<std::size_t>(16, 4096 / std::size_t(hidden));
    const std::size_t dispatch = (plane + tile - 1) / tile;
    if (dispatch == 0 || dispatch > UINT32_MAX) return false;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch),
                  static_cast<std::uint32_t>(batches), 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaDepthwisePointwiseFused(std::uint32_t input_index, std::uint32_t dw_index,
                                       std::uint32_t packed_bias_index, std::uint32_t pw_index,
                                       std::uint32_t output_index, std::size_t batches, int channels,
                                       int output_channels, int height, int width,
                                       bool gelu) noexcept {
    if (batches == 0 || channels <= 0 || output_channels <= 0 || height <= 0 || width <= 0 ||
        channels > 256 || batches > UINT32_MAX || !Initialize()) return false;
    const std::size_t plane = std::size_t(height) * width;
    if (plane == 0 || plane > UINT32_MAX) return false;
    const std::size_t input_elements = batches * std::size_t(channels) * plane;
    const std::size_t output_elements = batches * std::size_t(output_channels) * plane;
    const std::size_t dw_elements = std::size_t(channels) * 9;
    const std::size_t pw_elements = std::size_t(output_channels) * channels;
    const std::size_t bias_elements = std::size_t(channels) + output_channels;
    if (input_elements > UINT32_MAX || output_elements > UINT32_MAX) return false;
    std::lock_guard lock(mutex_);
    if (input_index >= arena_buffers_.size() || dw_index >= arena_buffers_.size() ||
        packed_bias_index >= arena_buffers_.size() || pw_index >= arena_buffers_.size() ||
        output_index >= arena_buffers_.size()) return false;
    const auto& input = arena_buffers_[input_index];
    const auto& dw = arena_buffers_[dw_index];
    const auto& packed_bias = arena_buffers_[packed_bias_index];
    const auto& pw = arena_buffers_[pw_index];
    const auto& output = arena_buffers_[output_index];
    if (!input.live || !dw.live || !packed_bias.live || !pw.live || !output.live ||
        input.live_elements < input_elements || dw.live_elements < dw_elements ||
        packed_bias.live_elements < bias_elements || pw.live_elements < pw_elements ||
        output.live_elements < output_elements) return false;
    std::array<VkDescriptorBufferInfo, 5> infos{};
    std::array<VkWriteDescriptorSet, 5> writes{};
    const std::array<const ArenaBuffer*, 5> sources{&input, &dw, &packed_bias, &output, &pw};
    const std::array<VkDeviceSize, 5> ranges{
        static_cast<VkDeviceSize>(input_elements * sizeof(float)),
        static_cast<VkDeviceSize>(dw_elements * sizeof(float)),
        static_cast<VkDeviceSize>(bias_elements * sizeof(float)),
        static_cast<VkDeviceSize>(output_elements * sizeof(float)),
        static_cast<VkDeviceSize>(pw_elements * sizeof(float))};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked();
    if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {sources[binding]->buffer, 0, ranges[binding]};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!BeginArenaOpLocked()) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
                              0, 1, &arena_set, 0, nullptr);
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; }
        push{static_cast<std::uint32_t>(std::size_t(output_channels) * plane),
             static_cast<std::uint32_t>(batches), static_cast<std::uint32_t>(channels),
             static_cast<std::uint32_t>(height), static_cast<std::uint32_t>(width),
             static_cast<std::uint32_t>(output_channels), 0u, 0u, 0u, 0u,
             gelu ? 63u : 62u};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                        sizeof(push), &push);
    const std::size_t tile = std::min<std::size_t>(16, 4096 / std::size_t(channels));
    if (tile == 0) return false;
    const std::size_t dispatch = (plane + tile - 1) / tile;
    if (dispatch == 0 || dispatch > UINT32_MAX) return false;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch),
                  static_cast<std::uint32_t>(batches), 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }


  bool RunArenaDepthwise(std::uint32_t input_index, std::uint32_t weights_index,
                         std::uint32_t bias_index, std::uint32_t output_index,
                         std::size_t batches, int channels, int input_height,
                         int input_width, int output_height, int output_width,
                         int kernel_height, int kernel_width, int stride_height,
                         int stride_width, int pad_top, int pad_left, int activation_mode,
                         bool force_scalar_tail = false) noexcept {
    if (batches == 0 || channels <= 0 || input_height <= 0 || input_width <= 0 ||
        output_height <= 0 || output_width <= 0 || kernel_height <= 0 || kernel_width <= 0 ||
        stride_height <= 0 || stride_width <= 0 || pad_top < 0 || pad_left < 0 ||
        pad_top > 255 || pad_left > 255 || !Initialize()) return false;
    const std::size_t input_plane = std::size_t(input_height) * input_width;
    const std::size_t output_plane = std::size_t(output_height) * output_width;
    const std::size_t input_elements = batches * std::size_t(channels) * input_plane;
    const std::size_t output_elements = batches * std::size_t(channels) * output_plane;
    const std::size_t weight_elements = std::size_t(channels) * kernel_height * kernel_width;
    if (input_elements > UINT32_MAX || output_elements > UINT32_MAX) return false;
    std::lock_guard lock(mutex_);
    if (input_index >= arena_buffers_.size() || weights_index >= arena_buffers_.size() ||
        bias_index >= arena_buffers_.size() || output_index >= arena_buffers_.size()) return false;
    const auto& input = arena_buffers_[input_index];
    const auto& weights = arena_buffers_[weights_index];
    const auto& bias = arena_buffers_[bias_index];
    const auto& output = arena_buffers_[output_index];
    if (!input.live || !weights.live || !bias.live || !output.live ||
        input.live_elements < input_elements || weights.live_elements < weight_elements ||
        bias.live_elements < std::size_t(channels) || output.live_elements < output_elements) return false;
    std::array<VkDescriptorBufferInfo, 5> infos{};
    std::array<VkWriteDescriptorSet, 5> writes{};
    // Depthwise mode never consumes binding E.  Bind the compact immutable
    // filter view there rather than aliasing InputA through a writable
    // storage declaration; this keeps every descriptor range/buffer pair
    // self-consistent on strict Vulkan drivers.
    const std::array<const ArenaBuffer*, 5> sources{&input, &weights, &bias, &output, &weights};
    const std::array<VkDeviceSize, 5> ranges{
        static_cast<VkDeviceSize>(input_elements * sizeof(float)),
        static_cast<VkDeviceSize>(weight_elements * sizeof(float)),
        static_cast<VkDeviceSize>(std::size_t(channels) * sizeof(float)),
        static_cast<VkDeviceSize>(output_elements * sizeof(float)),
        static_cast<VkDeviceSize>(weight_elements * sizeof(float))};
    // The exported high-channel terminal depthwise has too little spatial
    // work for the all-operator vec4 shader.  Select its isolated scalar
    // module by default.  It remains a single Vulkan dispatch over arena
    // buffers; PPOCR_GPU_TAIL_GENERIC is a diagnostic opt-out only.
    const bool scalar_tail_workaround = activation_mode == 0 && force_scalar_tail;
    // The graph executor already fences immediately after Add.143.  This
    // extra command-buffer isolation is an explicit driver diagnostic: it
    // keeps all tensors device-resident and never introduces a CPU fallback.
    const bool isolate_tail = scalar_tail_workaround && arena_recording_ &&
        !persistent_recording_ &&
        std::getenv("PPOCR_ENABLE_VULKAN_ISOLATED_DEPTHWISE_TAIL") != nullptr;
    if (isolate_tail) {
      std::array<VkBuffer, 5> isolated_buffers{};
      for (std::uint32_t binding = 0; binding < 5; ++binding) {
        isolated_buffers[binding] = sources[binding]->buffer;
      }
      const std::uint32_t isolated_mode = 52u |
          (static_cast<std::uint32_t>(pad_top) << 16u) |
          (static_cast<std::uint32_t>(pad_left) << 24u);
      return RunIsolatedDepthwiseTailLocked(
          isolated_buffers, ranges, static_cast<std::uint32_t>(channels * output_plane),
          static_cast<std::uint32_t>(batches), static_cast<std::uint32_t>(kernel_height),
          static_cast<std::uint32_t>(kernel_width), static_cast<std::uint32_t>(stride_height),
          static_cast<std::uint32_t>(stride_width), static_cast<std::uint32_t>(output_width),
          static_cast<std::uint32_t>(input_height), static_cast<std::uint32_t>(input_width),
          static_cast<std::uint32_t>(output_height), isolated_mode);
    }
    if (!BeginArenaOpLocked()) return false;
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; };
    // Radeon 780M's LLPC path intermittently loses the device on the medium
    // detector's final plain depthwise layer (896x5x22 and the 896x1x30
    // extreme-aspect variant).  These tiny planes do not benefit from vec4
    // batching, so route them to a separate scalar shader with simpler
    // indexing.  It remains a Vulkan compute operation and is deliberately
    // limited to the no-activation form.
    // The regular API keeps its established packed path.  The graph executor
    // opts into the scalar form only for the qualified problematic node, so a
    // preceding same-shape depthwise block cannot accidentally change its
    // kernel or mask a driver interaction.
    // Keep the legacy scalar mode available as a diagnostic fallback, but
    // the production tail uses the dedicated module (mode is irrelevant to
    // that shader apart from its packed padding fields).
    const std::uint32_t mode = static_cast<std::uint32_t>(
        scalar_tail_workaround ? 52 : 13 + activation_mode) |
        (static_cast<std::uint32_t>(pad_top) << 16u) | (static_cast<std::uint32_t>(pad_left) << 24u);
    Push push{static_cast<std::uint32_t>(channels * output_plane), static_cast<std::uint32_t>(batches),
              static_cast<std::uint32_t>(kernel_height), static_cast<std::uint32_t>(kernel_width),
              static_cast<std::uint32_t>(stride_height), static_cast<std::uint32_t>(stride_width),
              static_cast<std::uint32_t>(output_width), static_cast<std::uint32_t>(input_height),
              static_cast<std::uint32_t>(input_width), static_cast<std::uint32_t>(output_height), mode};
    // Split the known Radeon-sensitive medium tail into four channel windows.
    // Keep the original complete descriptors for every dispatch and encode
    // the channel start in mode 53.  Earlier descriptor-rebased slices were
    // algebraically valid but still caused device loss on Radeon/LLPC after a
    // long graph.  This form varies only push constants, keeps all tensor
    // data resident, and remains valid for an arbitrary NCHW batch because
    // the shader retains the original batch stride.
    // The dedicated module is the production path. Its deliberately minimal
    // SPIR-V has no mode-53 channel-window protocol, so keep this dispatch
    // whole; all data remains resident.
    const bool dedicated_tail_pipeline = scalar_tail_workaround &&
        tail_pipeline_ != VK_NULL_HANDLE &&
        std::getenv("PPOCR_DISABLE_VULKAN_DEDICATED_DEPTHWISE_TAIL") == nullptr;
    int tail_slices = dedicated_tail_pipeline ? 1 : 4;
    // Keep the safe conservative default, while making driver qualification
    // reproducible without recompiling.  A value of one selects the original
    // single mode-52 dispatch; values above one use mode 53's whole-buffer
    // channel windows.  This controls GPU dispatch granularity only and can
    // never introduce a host fallback.
    if (!dedicated_tail_pipeline) if (const char* configured = std::getenv("PPOCR_GPU_TAIL_SLICES")) {
      char* end{};
      const long parsed = std::strtol(configured, &end, 10);
      if (end != configured && *end == '\0' && parsed >= 1 && parsed <= channels)
        tail_slices = static_cast<int>(parsed);
    }
    if (scalar_tail_workaround && batches == 1 && tail_slices > 1 && channels >= tail_slices) {
      const int channels_per_slice = (channels + tail_slices - 1) / tail_slices;
      // This optional compatibility mode establishes a GPU-only queue/fence
      // boundary after each channel window. It lets us distinguish a single
      // tail dispatch from a long command-stream interaction on LLPC, with no
      // activation staging or CPU inference. Descriptor sets are recreated
      // after every boundary because the graph descriptor pool is reset only
      // after that fence signals.
      const bool submit_each_slice =
          std::getenv("PPOCR_GPU_TAIL_SUBMIT_SLICES") != nullptr;
      const auto submit_and_continue = [&]() noexcept {
        if (!arena_recording_ || end_command_buffer_(command_buffer_) != VK_SUCCESS) return false;
        const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
                                  1, &command_buffer_, 0, nullptr};
        arena_recording_ = false;
        const VkResult reset = reset_fences_(device_, 1, &submission_fence_);
        const VkResult submitted = reset == VK_SUCCESS
            ? queue_submit_(queue_, 1, &submit, submission_fence_) : reset;
        const VkResult waited = submitted == VK_SUCCESS
            ? wait_for_fences_(device_, 1, &submission_fence_, VK_TRUE,
                               std::numeric_limits<std::uint64_t>::max()) : submitted;
        if (waited != VK_SUCCESS) {
          restart_required_.store(true, std::memory_order_release);
          if (std::getenv("PPOCR_GPU_ONLY_TRACE") != nullptr) {
            std::cerr << "Vulkan scalar depthwise tail segment failed reset=" << reset
                      << " submit=" << submitted << " wait=" << waited << '\n';
          }
          return false;
        }
        recording_descriptor_sets_.clear();
        if (reset_descriptor_pool_(device_, graph_descriptor_pool_, 0) != VK_SUCCESS ||
            reset_command_buffer_(command_buffer_, 0) != VK_SUCCESS) return false;
        const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                             VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
        if (begin_command_buffer_(command_buffer_, &begin) != VK_SUCCESS) return false;
        cmd_bind_pipeline_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        const VkMemoryBarrier visible{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
        cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &visible,
                              0, nullptr, 0, nullptr);
        arena_recording_ = true;
        return true;
      };
      for (int channel_begin = 0; channel_begin < channels; channel_begin += channels_per_slice) {
        const int slice_channels = std::min(channels_per_slice, channels - channel_begin);
        if (std::getenv("PPOCR_GPU_ONLY_TRACE") != nullptr) {
          std::cerr << "Vulkan scalar depthwise tail slice channels=" << channel_begin
                    << '+' << slice_channels << " of " << channels << '\n';
        }
        const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked();
        if (!arena_set) return false;
        Push slice_push = push;
        for (std::uint32_t binding = 0; binding < 5; ++binding) {
          infos[binding] = {sources[binding]->buffer, 0, ranges[binding]};
          writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
        }
        update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
        cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
                                  0, 1, &arena_set, 0, nullptr);
        // count stays the complete per-batch tensor span so mode 53 computes
        // the original NCHW offsets.  The dispatch itself limits work to this
        // window and the 16 high bits encode its first channel.
        slice_push.mode = 53u | (static_cast<std::uint32_t>(channel_begin) << 16u);
        cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                            sizeof(slice_push), &slice_push);
        const std::size_t slice_dispatch = (std::size_t(slice_channels) * output_plane + 255) / 256;
        cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(slice_dispatch), 1, 1);
        const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
        cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        if (submit_each_slice && channel_begin + slice_channels < channels && !submit_and_continue())
          return false;
      }
      return EndArenaOpLocked();
    }
    // A descriptor set is immutable while its command buffer is pending. In
    // particular, do not update the reusable standalone set before a graph
    // recording is opened: on LLPC that can make the dedicated terminal
    // pipeline observe the prior descriptor contents. Allocate/update it only
    // after BeginArenaOpLocked has selected the active recording context.
    const VkDescriptorSet default_arena_set = DescriptorSetForArenaOpLocked();
    if (!default_arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {sources[binding]->buffer, 0, ranges[binding]};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, default_arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    // Select the isolated module only for the exact exported tail. The next
    // arena operator restores the normal pipeline in BeginArenaOpLocked().
    if (dedicated_tail_pipeline)
      cmd_bind_pipeline_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, tail_pipeline_);
    // The tiled mapping assigns a group to a channel. The generic mapping
    // emits vec4 values per invocation, so 256 lanes cover 1,024 outputs.
    const auto groups_per_channel = (output_plane + 1023) / 1024;
    const bool tiled_dispatch = !scalar_tail_workaround && output_plane >= 1024;
    const auto dispatch = tiled_dispatch
        ? std::size_t(channels) * groups_per_channel
    // The scalar shader owns one output per invocation and has 128
    // invocations for one output channel. Mapping a complete terminal
    // channel to a workgroup keeps filter/address state uniform.
        : scalar_tail_workaround
            ? std::size_t(channels)
            : (std::size_t(channels) * output_plane + 1023) / 1024;
    if (dispatch == 0 || dispatch > UINT32_MAX) return false;
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                        sizeof(push), &push);
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
                              0, 1, &default_arena_set, 0, nullptr);
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch), static_cast<std::uint32_t>(batches), 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaConv2d(std::uint32_t input_index, std::uint32_t weights_index,
                      std::uint32_t bias_index, std::uint32_t output_index,
                      std::size_t batches, int input_channels, int output_channels,
                      int input_height, int input_width, int output_height,
                      int output_width, int kernel_height, int kernel_width,
                      int stride_height, int stride_width, int pad_top, int pad_left,
                      int activation_mode) noexcept {
    if (batches == 0 || input_channels <= 0 || output_channels <= 0 || input_height <= 0 ||
        input_width <= 0 || output_height <= 0 || output_width <= 0 || kernel_height <= 0 ||
        kernel_width <= 0 || stride_height <= 0 || stride_height != stride_width ||
        pad_top < 0 || pad_left < 0 || pad_top > 255 || pad_left > 255 ||
        !Initialize()) return false;
    const std::size_t input_plane = std::size_t(input_height) * input_width;
    const std::size_t output_plane = std::size_t(output_height) * output_width;
    const std::size_t input_elements = batches * std::size_t(input_channels) * input_plane;
    const std::size_t output_elements = batches * std::size_t(output_channels) * output_plane;
    const std::size_t weight_elements = std::size_t(output_channels) * input_channels *
        kernel_height * kernel_width;
    if (input_elements > UINT32_MAX || output_elements > UINT32_MAX) return false;
    std::lock_guard lock(mutex_);
    if (input_index >= arena_buffers_.size() || weights_index >= arena_buffers_.size() ||
        bias_index >= arena_buffers_.size() || output_index >= arena_buffers_.size()) return false;
    const auto& input = arena_buffers_[input_index];
    const auto& weights = arena_buffers_[weights_index];
    const auto& bias = arena_buffers_[bias_index];
    const auto& output = arena_buffers_[output_index];
    if (!input.live || !weights.live || !bias.live || !output.live ||
        input.live_elements < input_elements || weights.live_elements < weight_elements ||
        bias.live_elements < std::size_t(output_channels) ||
        output.live_elements < output_elements) return false;
    std::array<VkDescriptorBufferInfo, 5> infos{};
    std::array<VkWriteDescriptorSet, 5> writes{};
    const std::array<const ArenaBuffer*, 5> sources{&input, &weights, &bias, &output, &input};
    const std::array<VkDeviceSize, 5> ranges{
        static_cast<VkDeviceSize>(input_elements * sizeof(float)),
        static_cast<VkDeviceSize>(weight_elements * sizeof(float)),
        static_cast<VkDeviceSize>(std::size_t(output_channels) * sizeof(float)),
        static_cast<VkDeviceSize>(output_elements * sizeof(float)),
        static_cast<VkDeviceSize>(input_elements * sizeof(float))};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked();
    if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {sources[binding]->buffer, 0, ranges[binding]};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!BeginArenaOpLocked()) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
                              0, 1, &arena_set, 0, nullptr);
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; };
    // Must match the shader: tiled 3x3 is plane>=2048. 256..2047 used to
    // dispatch the channel-tile layout into the generic vec4 walk.
    constexpr std::size_t kTiledMinimumOutputPlane = 2048u;
    const std::size_t weights_per_channel = std::size_t(input_channels) * kernel_height * kernel_width;
    // Modes 50/51 must not reuse Pool2D(40) or broadcast(41). Spatial tiles
    // cover Conv.0-class 3x3 s1/s2 maps (C<=8); activation_mode 0/1 is none/ReLU.
    // Hybrid Concat+Conv uses the scratch Conv2d path, not this arena entry.
    // Spatial 16x16 tiles already stage Concat.2 C=64 s1 (12 IC/pass) and
    // stem C=32 s2 (3 IC/pass) in the 4096-float halo. The C<=8 gate left
    // those maps on the generic 3x3 kernel. `PPOCR_DISABLE_VULKAN_CONV2D_SPATIAL`
    // restores C<=8.
    const bool use_spatial_3x3 = kernel_height == 3 && kernel_width == 3 &&
        output_plane >= 2048 &&
        (activation_mode == 0 || activation_mode == 1) &&
        std::getenv("PPOCR_DISABLE_VULKAN_CONV2D_SPATIAL") == nullptr &&
        ((stride_height == 1 && input_channels <= 64) ||
         (stride_height == 2 && input_channels <= 32));
    const std::uint32_t mode = (use_spatial_3x3
        ? (activation_mode == 1 ? 51u : 50u)
        : static_cast<std::uint32_t>(14 + activation_mode)) |
        (static_cast<std::uint32_t>(pad_top) << 16u) |
        (static_cast<std::uint32_t>(pad_left) << 24u);
    Push push{static_cast<std::uint32_t>(output_channels * output_plane),
              static_cast<std::uint32_t>(batches), static_cast<std::uint32_t>(input_channels),
              static_cast<std::uint32_t>(input_height), static_cast<std::uint32_t>(input_width),
              static_cast<std::uint32_t>(output_height), static_cast<std::uint32_t>(output_width),
              static_cast<std::uint32_t>(kernel_height), static_cast<std::uint32_t>(kernel_width),
              static_cast<std::uint32_t>(stride_height), mode};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                        sizeof(push), &push);
    // The 4-OC tiled kernel's spatial mapping is stride-1 only. Stem
    // FusedMaxPoolConcatConv is 32-IC 3x3 s2 (40x176); using the s1 tile
    // zeroed outputs (CPU 16.4 vs GPU 0 at the first real-DetInput miss).
    const bool use_tiled = !use_spatial_3x3 && stride_height == 1 &&
        weights_per_channel <= 4096 &&
        output_plane >= kTiledMinimumOutputPlane &&
        std::getenv("PPOCR_DISABLE_VULKAN_CONV2D_TILE") == nullptr;
    const bool tile4oc = use_tiled && kernel_height == 3 && kernel_width == 3 &&
        weights_per_channel * 4 <= 4096;
    const std::size_t dispatch = use_spatial_3x3
        ? std::size_t((output_width + 15) / 16) * std::size_t((output_height + 15) / 16) *
              std::size_t((output_channels + 3) / 4)
        : (use_tiled
            ? (tile4oc ? std::size_t((output_channels + 3) / 4)
                       : std::size_t(output_channels)) *
                  ((output_plane + 1023) / 1024)
            : (std::size_t(output_channels) * output_plane + 1023) / 1024);
    if (dispatch == 0 || dispatch > UINT32_MAX) return false;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch),
                  static_cast<std::uint32_t>(batches), 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaConvTranspose2x2(std::uint32_t input_index, std::uint32_t weights_index,
                                std::uint32_t bias_index, std::uint32_t output_index,
                                std::size_t batches, int input_channels, int output_channels,
                                int input_height, int input_width, int activation = 0) noexcept {
    if (batches == 0 || input_channels <= 0 || output_channels <= 0 || input_height <= 0 ||
        input_width <= 0 || !Initialize()) return false;
    const std::size_t input_plane = std::size_t(input_height) * input_width;
    const std::size_t output_plane = input_plane * 4;
    const std::size_t input_elements = batches * std::size_t(input_channels) * input_plane;
    const std::size_t output_elements = batches * std::size_t(output_channels) * output_plane;
    const std::size_t weight_elements = std::size_t(input_channels) * output_channels * 4;
    if (input_elements > UINT32_MAX || output_elements > UINT32_MAX) return false;
    std::lock_guard lock(mutex_);
    if (input_index >= arena_buffers_.size() || weights_index >= arena_buffers_.size() ||
        bias_index >= arena_buffers_.size() || output_index >= arena_buffers_.size()) return false;
    const auto& input = arena_buffers_[input_index]; const auto& weights = arena_buffers_[weights_index];
    const auto& bias = arena_buffers_[bias_index]; const auto& output = arena_buffers_[output_index];
    if (!input.live || !weights.live || !bias.live || !output.live ||
        input.live_elements < input_elements || weights.live_elements < weight_elements ||
        bias.live_elements < std::size_t(output_channels) || output.live_elements < output_elements) return false;
    std::array<VkDescriptorBufferInfo, 5> infos{}; std::array<VkWriteDescriptorSet, 5> writes{};
    const std::array<const ArenaBuffer*, 5> sources{&input, &weights, &bias, &output, &input};
    const std::array<VkDeviceSize, 5> ranges{
        static_cast<VkDeviceSize>(input_elements * sizeof(float)),
        static_cast<VkDeviceSize>(weight_elements * sizeof(float)),
        static_cast<VkDeviceSize>(std::size_t(output_channels) * sizeof(float)),
        static_cast<VkDeviceSize>(output_elements * sizeof(float)),
        static_cast<VkDeviceSize>(input_elements * sizeof(float))};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked();
    if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {sources[binding]->buffer, 0, ranges[binding]};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!BeginArenaOpLocked()) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1,
                              &arena_set, 0, nullptr);
    constexpr std::size_t kTiledMinimumOutputPlane = 16u * 1024u;
    const bool tiled = std::size_t(input_channels) * 4 <= 4096 && output_plane >= kTiledMinimumOutputPlane &&
        std::getenv("PPOCR_DISABLE_VULKAN_CONVTRANSPOSE_TILE") == nullptr;
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; };
    const std::uint32_t act = activation == 1 ? 1u : (activation == 2 ? 2u : 0u);
    static const bool fuse_act =
        std::getenv("PPOCR_DISABLE_GPU_FUSE_CONVTRANSPOSE_ACT") == nullptr;
    const std::uint32_t packed_act = fuse_act ? (act << 16) : 0u;
    Push push{static_cast<std::uint32_t>(output_channels * output_plane), static_cast<std::uint32_t>(batches),
              static_cast<std::uint32_t>(input_channels), static_cast<std::uint32_t>(output_channels),
              static_cast<std::uint32_t>(input_height), static_cast<std::uint32_t>(input_width),
              0u, 0u, 0u, 0u, (tiled ? 17u : 16u) | packed_act};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    const std::size_t dispatch = tiled ? std::size_t(output_channels) * ((output_plane + 1023) / 1024)
                                       : (std::size_t(output_channels) * output_plane + 1023) / 1024;
    if (dispatch == 0 || dispatch > UINT32_MAX) return false;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch), static_cast<std::uint32_t>(batches), 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaConvTranspose2x2Chain(std::uint32_t input_index, std::uint32_t w0_index,
                                     std::uint32_t b0_index, std::uint32_t packed_index,
                                     std::uint32_t output_index, std::size_t batches,
                                     int input_channels, int mid_channels, int output_channels,
                                     int input_height, int input_width) noexcept {
    if (batches == 0 || input_channels <= 0 || mid_channels <= 0 || output_channels <= 0 ||
        input_height <= 0 || input_width <= 0 || input_channels > 64 || mid_channels > 64 ||
        output_channels > 16 || !Initialize()) return false;
    const std::size_t input_plane = std::size_t(input_height) * input_width;
    const std::size_t output_plane = input_plane * 16;
    const std::size_t input_elements = batches * std::size_t(input_channels) * input_plane;
    const std::size_t output_elements = batches * std::size_t(output_channels) * output_plane;
    const std::size_t w0_elements = std::size_t(input_channels) * mid_channels * 4;
    const std::size_t packed_elements =
        std::size_t(mid_channels) * output_channels * 4 + std::size_t(output_channels);
    if (input_elements > UINT32_MAX || output_elements > UINT32_MAX) return false;
    std::lock_guard lock(mutex_);
    if (input_index >= arena_buffers_.size() || w0_index >= arena_buffers_.size() ||
        b0_index >= arena_buffers_.size() || packed_index >= arena_buffers_.size() ||
        output_index >= arena_buffers_.size()) return false;
    const auto& input = arena_buffers_[input_index];
    const auto& w0 = arena_buffers_[w0_index];
    const auto& b0 = arena_buffers_[b0_index];
    const auto& packed = arena_buffers_[packed_index];
    const auto& output = arena_buffers_[output_index];
    if (!input.live || !w0.live || !b0.live || !packed.live || !output.live ||
        input.live_elements < input_elements || w0.live_elements < w0_elements ||
        b0.live_elements < std::size_t(mid_channels) ||
        packed.live_elements < packed_elements ||
        output.live_elements < output_elements) return false;
    std::array<VkDescriptorBufferInfo, 5> infos{};
    std::array<VkWriteDescriptorSet, 5> writes{};
    const std::array<const ArenaBuffer*, 5> sources{&input, &w0, &b0, &output, &packed};
    const std::array<VkDeviceSize, 5> ranges{
        static_cast<VkDeviceSize>(input_elements * sizeof(float)),
        static_cast<VkDeviceSize>(w0_elements * sizeof(float)),
        static_cast<VkDeviceSize>(std::size_t(mid_channels) * sizeof(float)),
        static_cast<VkDeviceSize>(output_elements * sizeof(float)),
        static_cast<VkDeviceSize>(packed_elements * sizeof(float))};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked();
    if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {sources[binding]->buffer, 0, ranges[binding]};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0,
                            nullptr);
    if (!BeginArenaOpLocked()) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1,
                              &arena_set, 0, nullptr);
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; }
        push{static_cast<std::uint32_t>(output_channels * output_plane),
             static_cast<std::uint32_t>(batches), static_cast<std::uint32_t>(input_channels),
             static_cast<std::uint32_t>(mid_channels), static_cast<std::uint32_t>(input_height),
             static_cast<std::uint32_t>(input_width), static_cast<std::uint32_t>(output_channels),
             0u, 0u, 0u, 67u};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                        sizeof(push), &push);
    const std::size_t dispatch = (std::size_t(output_channels) * output_plane + 1023) / 1024;
    if (dispatch == 0 || dispatch > UINT32_MAX) return false;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch),
                  static_cast<std::uint32_t>(batches), 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaNearestResize2xAdd(std::uint32_t source_index, std::uint32_t residual_index,
                                  std::uint32_t output_index, std::size_t batches, int channels,
                                  int input_height, int input_width) noexcept {
    if (batches == 0 || channels <= 0 || input_height <= 0 || input_width <= 0 || !Initialize()) return false;
    const std::size_t input_plane = std::size_t(input_height) * input_width;
    const std::size_t input_elements = batches * std::size_t(channels) * input_plane;
    const std::size_t output_elements = input_elements * 4;
    if (input_elements > UINT32_MAX || output_elements > UINT32_MAX) return false;
    std::lock_guard lock(mutex_);
    if (source_index >= arena_buffers_.size() || residual_index >= arena_buffers_.size() ||
        output_index >= arena_buffers_.size()) return false;
    const auto& source = arena_buffers_[source_index]; const auto& residual = arena_buffers_[residual_index];
    const auto& output = arena_buffers_[output_index];
    if (!source.live || !residual.live || !output.live || source.live_elements < input_elements ||
        residual.live_elements < output_elements || output.live_elements < output_elements) return false;
    std::array<VkDescriptorBufferInfo, 5> infos{}; std::array<VkWriteDescriptorSet, 5> writes{};
    const std::array<const ArenaBuffer*, 5> sources{&source, &source, &source, &output, &residual};
    const std::array<VkDeviceSize, 5> ranges{
        static_cast<VkDeviceSize>(input_elements * sizeof(float)), sizeof(float), sizeof(float),
        static_cast<VkDeviceSize>(output_elements * sizeof(float)),
        static_cast<VkDeviceSize>(output_elements * sizeof(float))};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked();
    if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {sources[binding]->buffer, 0, ranges[binding]};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!BeginArenaOpLocked()) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1,
                              &arena_set, 0, nullptr);
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; }
        push{static_cast<std::uint32_t>(std::size_t(channels) * input_plane * 4),
             static_cast<std::uint32_t>(batches), 0u, 0u, 0u, 0u, 1u,
             static_cast<std::uint32_t>(input_height), static_cast<std::uint32_t>(input_width),
             static_cast<std::uint32_t>(channels), 18u};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    const std::size_t dispatch = (std::size_t(channels) * input_plane * 4 + 1023) / 1024;
    if (dispatch == 0 || dispatch > UINT32_MAX) return false;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch), static_cast<std::uint32_t>(batches), 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaGemm(std::uint32_t left_index, std::uint32_t right_index, std::uint32_t bias_index,
                    std::uint32_t output_index, int rows, int depth, int columns,
                    bool has_bias, bool swish, int a_lda = 0) noexcept {
    if (rows <= 0 || depth <= 0 || columns <= 0 || !Initialize()) return false;
    if (a_lda < 0) return false;
    const std::size_t lda = a_lda == 0 ? std::size_t(depth) : std::size_t(a_lda);
    const std::size_t left_elements = a_lda == 0
        ? std::size_t(rows) * depth
        : (std::size_t(depth - 1) * lda + std::size_t(rows));
    const std::size_t right_elements = std::size_t(depth) * columns;
    const std::size_t output_elements = std::size_t(rows) * columns;
    if (left_elements > UINT32_MAX || right_elements > UINT32_MAX || output_elements > UINT32_MAX) return false;
    std::lock_guard lock(mutex_);
    if (left_index >= arena_buffers_.size() || right_index >= arena_buffers_.size() ||
        output_index >= arena_buffers_.size() || (has_bias && bias_index >= arena_buffers_.size())) return false;
    const auto& left = arena_buffers_[left_index]; const auto& right = arena_buffers_[right_index];
    const auto& output = arena_buffers_[output_index];
    const auto& bias = has_bias ? arena_buffers_[bias_index] : left;
    if (!left.live || !right.live || !output.live || (has_bias && !bias.live) ||
        left.live_elements < left_elements || right.live_elements < right_elements ||
        output.live_elements < output_elements || (has_bias && bias.live_elements < std::size_t(columns))) return false;
    std::array<VkDescriptorBufferInfo, 5> infos{}; std::array<VkWriteDescriptorSet, 5> writes{};
    const std::array<const ArenaBuffer*, 5> sources{&left, &right, &bias, &output, &left};
    const std::array<VkDeviceSize, 5> ranges{
        static_cast<VkDeviceSize>(left_elements * sizeof(float)),
        static_cast<VkDeviceSize>(right_elements * sizeof(float)),
        static_cast<VkDeviceSize>((has_bias ? std::size_t(columns) : 1) * sizeof(float)),
        static_cast<VkDeviceSize>(output_elements * sizeof(float)),
        static_cast<VkDeviceSize>(left_elements * sizeof(float))};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked();
    if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {sources[binding]->buffer, 0, ranges[binding]};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!BeginArenaOpLocked()) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1,
                              &arena_set, 0, nullptr);
    // The tiled GEMM was numerically verified against the generic path on
    // tiny/small/medium PP-OCRv6 recognizers.  It reuses B across eight rows
    // and lowers Radeon 780M strict-graph time, so make the safe non-Swish
    // form the default.  Swish remains opt-in until its distinct writeback is
    // covered by the same full-model regression matrix.
    const bool tiled = rows >= 8 && depth >= 16 && columns >= 128 &&
        !swish && std::getenv("PPOCR_DISABLE_VULKAN_GEMM_TILE") == nullptr;
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; }
        push{static_cast<std::uint32_t>(output_elements), 1u, static_cast<std::uint32_t>(rows),
             static_cast<std::uint32_t>(depth), static_cast<std::uint32_t>(columns), has_bias ? 1u : 0u,
             1u, static_cast<std::uint32_t>(a_lda), 0u, 0u,
             tiled ? (swish ? 25u : 24u) : (swish ? 23u : 19u)};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    const std::size_t dispatch_x = tiled ? (std::size_t(columns) + 127) / 128 : (output_elements + 1023) / 1024;
    const std::size_t dispatch_y = tiled ? (std::size_t(rows) + 7) / 8 : 1;
    if (dispatch_x == 0 || dispatch_x > UINT32_MAX || dispatch_y == 0 || dispatch_y > UINT32_MAX) return false;
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(dispatch_x), static_cast<std::uint32_t>(dispatch_y), 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    return EndArenaOpLocked();
  }

  bool RunArenaGemmCtcTop1(std::uint32_t left_index, std::uint32_t right_index,
                           std::uint32_t bias_index, std::uint32_t indices_index,
                           std::uint32_t probs_index, int rows, int depth, int vocab,
                           bool has_bias) noexcept {
    if (rows <= 0 || depth <= 0 || vocab <= 0 || !Initialize()) return false;
    const std::size_t left_elements = std::size_t(rows) * depth;
    const std::size_t right_elements = std::size_t(depth) * vocab;
    const std::size_t compact = std::size_t(rows);
    if (left_elements > UINT32_MAX || right_elements > UINT32_MAX) return false;
    std::lock_guard lock(mutex_);
    if (left_index >= arena_buffers_.size() || right_index >= arena_buffers_.size() ||
        indices_index >= arena_buffers_.size() || probs_index >= arena_buffers_.size() ||
        (has_bias && bias_index >= arena_buffers_.size())) return false;
    const auto& left = arena_buffers_[left_index];
    const auto& right = arena_buffers_[right_index];
    const auto& indices = arena_buffers_[indices_index];
    const auto& probs = arena_buffers_[probs_index];
    const auto& bias = has_bias ? arena_buffers_[bias_index] : left;
    if (!left.live || !right.live || !indices.live || !probs.live ||
        (has_bias && !bias.live) || left.live_elements < left_elements ||
        right.live_elements < right_elements || indices.live_elements < compact ||
        probs.live_elements < compact ||
        (has_bias && bias.live_elements < std::size_t(vocab))) return false;
    std::array<VkDescriptorBufferInfo, 5> infos{};
    std::array<VkWriteDescriptorSet, 5> writes{};
    const std::array<const ArenaBuffer*, 5> sources{&left, &right, &bias, &indices, &probs};
    const std::array<VkDeviceSize, 5> ranges{
        VkDeviceSize(left_elements * sizeof(float)),
        VkDeviceSize(right_elements * sizeof(float)),
        VkDeviceSize((has_bias ? std::size_t(vocab) : 1) * sizeof(float)),
        VkDeviceSize(compact * sizeof(float)), VkDeviceSize(compact * sizeof(float))};
    const VkDescriptorSet arena_set = DescriptorSetForArenaOpLocked();
    if (!arena_set) return false;
    for (std::uint32_t binding = 0; binding < 5; ++binding) {
      infos[binding] = {sources[binding]->buffer, 0, ranges[binding]};
      writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, arena_set, binding, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0,
                            nullptr);
    if (!BeginArenaOpLocked()) return false;
    cmd_bind_descriptor_sets_(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0,
                              1, &arena_set, 0, nullptr);
    struct Push { std::uint32_t count, batches, operation0, operation1, operation2, operation3,
                                steps, right_repeat, right_per_batch, right_batch_stride, mode; }
        push{static_cast<std::uint32_t>(rows), 1u, static_cast<std::uint32_t>(rows),
             static_cast<std::uint32_t>(depth), static_cast<std::uint32_t>(vocab),
             has_bias ? 1u : 0u, 0u, 0u, 0u, 0u, 65u};
    cmd_push_constants_(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                        sizeof(push), &push);
    cmd_dispatch_(command_buffer_, static_cast<std::uint32_t>(rows), 1, 1);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0,
                          nullptr);
    return EndArenaOpLocked();
  }

  void ResetArenaSlots() noexcept {
    std::lock_guard lock(mutex_);
    for (auto& slot : arena_buffers_) {
      slot.live = false;
      slot.live_elements = 0;
      slot.reusable_in_recording = false;
    }
  }

  bool ReclaimFreeArenaStorage() noexcept {
    if (!Initialize()) return false;
    std::lock_guard lock(mutex_);
    // A recorded command buffer can still reference a logically released
    // slot. Model hand-off occurs between fenced graphs, so reject misuse.
    if (arena_recording_) return false;
    ReclaimFreeArenaBuffersLocked(/*retain=*/0, /*include_persistent=*/true);
    return true;
  }

  bool ReclaimFreeTransientArenaStorage() noexcept {
    if (!Initialize()) return false;
    std::lock_guard lock(mutex_);
    // A recorded command buffer can still reference a logically released
    // activation. The OCR stage hand-off is fenced before this call.
    if (arena_recording_) return false;
    ReclaimFreeArenaBuffersLocked(/*retain=*/0, /*include_persistent=*/false);
    return true;
  }

  std::size_t ArenaCapacityBytes() const noexcept {
    std::lock_guard lock(mutex_);
    std::size_t result{};
    for (const auto& slot : arena_buffers_) result += static_cast<std::size_t>(slot.capacity);
    return result;
  }

  std::size_t ArenaLiveBytes() const noexcept {
    std::lock_guard lock(mutex_);
    std::size_t result{};
    for (const auto& slot : arena_buffers_)
      if (slot.live) result += slot.live_elements * sizeof(float);
    return result;
  }

  std::size_t ArenaAllocatedBytes() const noexcept {
    std::lock_guard lock(mutex_);
    std::size_t result{};
    for (const auto& slot : arena_buffers_) {
      result += static_cast<std::size_t>(slot.capacity);
      if (slot.device_local) result += static_cast<std::size_t>(slot.capacity);
    }
    return result;
  }

  std::uint64_t ArenaGeneration() const noexcept {
    return arena_generation_.load(std::memory_order_acquire);
  }

  std::uint64_t HybridAdmissionContext() const noexcept {
    // The adapter is selected once for this process; arena generation changes
    // on device-loss cleanup. Reading two atomics keeps hybrid's hot lookup
    // lock-free and ensures a recovered device must requalify every shape.
    const auto adapter = adapter_identity_.load(std::memory_order_acquire);
    const auto generation = arena_generation_.load(std::memory_order_acquire);
    return adapter ^ (generation + 0x9e3779b97f4a7c15ull + (adapter << 6) +
                      (adapter >> 2));
  }

 private:
  struct ArenaBuffer {
    VkBuffer buffer{};
    VkDeviceMemory memory{};
    // A device-local tensor is paired with a host-visible transfer buffer.
    // The shader always sees `buffer`; mapped staging is touched only at the
    // public upload/download boundary and never for an intermediate tensor.
    VkBuffer staging_buffer{};
    VkDeviceMemory staging_memory{};
    void* mapped{};
    VkDeviceSize capacity{};
    VkDeviceSize memory_offset{};
    VkDeviceSize staging_offset{};
    std::size_t live_elements{};
    VkMemoryPropertyFlags memory_flags{};
    VkMemoryPropertyFlags staging_memory_flags{};
    bool live{};
    bool reusable_in_recording{};
    bool persistent{};
    bool replay_pinned{};
    bool shared_memory{};
    bool device_local{};
    bool staging_dirty{};
  };

  struct ArenaMemoryBlock {
    VkDeviceMemory memory{};
    void* mapped{};
    VkDeviceSize capacity{};
    VkDeviceSize used{};
    VkMemoryPropertyFlags memory_flags{};
    std::uint32_t memory_type{UINT32_MAX};
  };

  bool FlushArenaWritesLocked(const ArenaBuffer& slot, std::size_t elements) noexcept {
    if (!slot.mapped) return false;
    if ((slot.staging_memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) return true;
    if (!flush_mapped_memory_ranges_) return false;
    const VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr,
        slot.device_local ? slot.staging_memory : slot.memory,
        slot.device_local ? slot.staging_offset : slot.memory_offset,
        static_cast<VkDeviceSize>(elements) * sizeof(float)};
    return flush_mapped_memory_ranges_(device_, 1, &range) == VK_SUCCESS;
  }

  bool InvalidateArenaReadsLocked(const ArenaBuffer& slot, std::size_t elements) const noexcept {
    if (!slot.mapped) return false;
    if ((slot.staging_memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) return true;
    if (!invalidate_mapped_memory_ranges_) return false;
    const VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr,
        slot.device_local ? slot.staging_memory : slot.memory,
        slot.device_local ? slot.staging_offset : slot.memory_offset,
        static_cast<VkDeviceSize>(elements) * sizeof(float)};
    return invalidate_mapped_memory_ranges_(device_, 1, &range) == VK_SUCCESS;
  }

  std::uint32_t FindMemoryType(std::uint32_t bits, VkMemoryPropertyFlags required,
                               VkMemoryPropertyFlags preferred = 0) const noexcept {
    std::uint32_t selected = UINT32_MAX;
    int score = -1;
    for (std::uint32_t bit = 0; bit < memory_properties_.memoryTypeCount; ++bit) {
      const auto flags = memory_properties_.memoryTypes[bit].propertyFlags;
      if ((bits & (1u << bit)) == 0 || (flags & required) != required) continue;
      const int candidate_score = std::popcount(flags & preferred);
      if (candidate_score > score) { selected = bit; score = candidate_score; }
    }
    return selected;
  }

  bool CopyStagingToArenaLocked(ArenaBuffer& slot, std::size_t elements) noexcept {
    if (!slot.device_local || !slot.staging_dirty) return true;
    if (!cmd_copy_buffer_ || !command_buffer_) return false;
    const bool standalone = !arena_recording_;
    if (standalone) {
      if (reset_command_buffer_(command_buffer_, 0) != VK_SUCCESS) return false;
      const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
          VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
      if (begin_command_buffer_(command_buffer_, &begin) != VK_SUCCESS) return false;
    }
    const VkBufferCopy region{0, 0, static_cast<VkDeviceSize>(elements) * sizeof(float)};
    cmd_copy_buffer_(command_buffer_, slot.staging_buffer, slot.buffer, 1, &region);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier,
                          0, nullptr, 0, nullptr);
    slot.staging_dirty = false;
    if (!standalone) return true;
    if (end_command_buffer_(command_buffer_) != VK_SUCCESS) return false;
    const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
        1, &command_buffer_, 0, nullptr};
    return reset_fences_(device_, 1, &submission_fence_) == VK_SUCCESS &&
        queue_submit_(queue_, 1, &submit, submission_fence_) == VK_SUCCESS &&
        wait_for_fences_(device_, 1, &submission_fence_, VK_TRUE,
                         std::numeric_limits<std::uint64_t>::max()) == VK_SUCCESS;
  }

  bool RecordDirtyStagingCopiesLocked() noexcept {
    for (auto& slot : arena_buffers_) {
      if (!slot.live || !slot.staging_dirty) continue;
      if (!CopyStagingToArenaLocked(slot, slot.live_elements)) return false;
    }
    return true;
  }

  bool CopyArenaToStagingLocked(ArenaBuffer& slot, std::size_t elements) noexcept {
    if (!slot.device_local) return true;
    if (!cmd_copy_buffer_ || !command_buffer_) return false;
    const bool standalone = !arena_recording_;
    if (standalone) {
      if (reset_command_buffer_(command_buffer_, 0) != VK_SUCCESS) return false;
      const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
          VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
      if (begin_command_buffer_(command_buffer_, &begin) != VK_SUCCESS) return false;
    }
    const VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before,
                          0, nullptr, 0, nullptr);
    const VkBufferCopy region{0, 0, static_cast<VkDeviceSize>(elements) * sizeof(float)};
    cmd_copy_buffer_(command_buffer_, slot.buffer, slot.staging_buffer, 1, &region);
    const VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
    cmd_pipeline_barrier_(command_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after,
                          0, nullptr, 0, nullptr);
    if (!standalone) return true;
    if (end_command_buffer_(command_buffer_) != VK_SUCCESS) return false;
    const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
        1, &command_buffer_, 0, nullptr};
    return reset_fences_(device_, 1, &submission_fence_) == VK_SUCCESS &&
        queue_submit_(queue_, 1, &submit, submission_fence_) == VK_SUCCESS &&
        wait_for_fences_(device_, 1, &submission_fence_, VK_TRUE,
                         std::numeric_limits<std::uint64_t>::max()) == VK_SUCCESS;
  }

  void DestroyArenaBuffer(ArenaBuffer& slot) noexcept {
    if (slot.mapped && slot.device_local) unmap_memory_(device_, slot.staging_memory);
    else if (slot.mapped && !slot.shared_memory) unmap_memory_(device_, slot.memory);
    if (slot.staging_buffer) destroy_buffer_(device_, slot.staging_buffer, nullptr);
    if (slot.staging_memory) free_memory_(device_, slot.staging_memory, nullptr);
    if (slot.buffer) destroy_buffer_(device_, slot.buffer, nullptr);
    if (slot.memory && !slot.shared_memory) free_memory_(device_, slot.memory, nullptr);
    slot = {};
  }

  bool CreateArenaBuffer(ArenaBuffer& slot, VkDeviceSize bytes, bool persistent) noexcept {
    const VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
    if (create_buffer_(device_, &info, nullptr, &slot.buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements{};
    get_buffer_memory_requirements_(device_, slot.buffer, &requirements);
    const char* device_local_env = std::getenv("PPOCR_GPU_ARENA_DEVICE_LOCAL");
    // Host-visible coherent storage avoids an extra transfer pair on UMA
    // Vulkan adapters (and gives strict GPU-only graphs a single coherent
    // ownership model). Discrete deployments can opt into device-local
    // buffers explicitly after their driver path is qualified.
    const bool use_device_local = device_local_env != nullptr && std::strcmp(device_local_env, "0") != 0;
    const auto local_type = use_device_local
        ? FindMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        : UINT32_MAX;
    if (local_type != UINT32_MAX) {
      const VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr,
                                             requirements.size, local_type};
      if (allocate_memory_(device_, &allocation, nullptr, &slot.memory) != VK_SUCCESS ||
          bind_buffer_memory_(device_, slot.buffer, slot.memory, 0) != VK_SUCCESS) {
        if (slot.memory) free_memory_(device_, slot.memory, nullptr);
        if (slot.buffer) destroy_buffer_(device_, slot.buffer, nullptr);
        slot = {};
        return false;
      }
      const VkBufferCreateInfo staging_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, bytes,
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
      if (create_buffer_(device_, &staging_info, nullptr, &slot.staging_buffer) != VK_SUCCESS) {
        DestroyArenaBuffer(slot); return false;
      }
      VkMemoryRequirements staging_requirements{};
      get_buffer_memory_requirements_(device_, slot.staging_buffer, &staging_requirements);
      const auto staging_type = FindMemoryType(staging_requirements.memoryTypeBits,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
      if (staging_type == UINT32_MAX) { DestroyArenaBuffer(slot); return false; }
      const VkMemoryAllocateInfo staging_allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr,
          staging_requirements.size, staging_type};
      if (allocate_memory_(device_, &staging_allocation, nullptr, &slot.staging_memory) != VK_SUCCESS ||
          bind_buffer_memory_(device_, slot.staging_buffer, slot.staging_memory, 0) != VK_SUCCESS ||
          map_memory_(device_, slot.staging_memory, 0, bytes, 0, &slot.mapped) != VK_SUCCESS) {
        DestroyArenaBuffer(slot); return false;
      }
      slot.capacity = bytes;
      slot.persistent = persistent;
      slot.device_local = true;
      slot.memory_flags = memory_properties_.memoryTypes[local_type].propertyFlags;
      slot.staging_memory_flags = memory_properties_.memoryTypes[staging_type].propertyFlags;
      return true;
    }
    std::uint32_t memory_type = UINT32_MAX;
    int best_score = -1;
    for (std::uint32_t bit = 0; bit < memory_properties_.memoryTypeCount; ++bit) {
      const auto flags = memory_properties_.memoryTypes[bit].propertyFlags;
      if ((requirements.memoryTypeBits & (1u << bit)) &&
          (flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
              (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        const bool device_local = (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        const bool cached = (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0;
        // Directly mapped device-local heap types are normally best for the
        // arena.  Keep a host-cached alternative for driver qualification:
        // it changes only the Vulkan memory type, never the graph or CPU
        // fallback behaviour, and is useful on UMA implementations whose
        // direct-mapped local heap has conservative residency limits.
        const bool prefer_cached = std::getenv("PPOCR_GPU_ARENA_PREFER_HOST_CACHED") != nullptr;
        const int score = prefer_cached ? (cached ? 8 : (device_local ? 2 : 1))
                                        : (device_local ? 4 : 0) + (cached ? 1 : 0);
        if (score > best_score) { memory_type = bit; best_score = score; }
      }
    }
    if (memory_type == UINT32_MAX) { destroy_buffer_(device_, slot.buffer, nullptr); slot.buffer = {}; return false; }
    if (persistent) {
      const auto align_up = [](VkDeviceSize value, VkDeviceSize alignment) noexcept {
        return alignment == 0 ? value : (value + alignment - 1) / alignment * alignment;
      };
      ArenaMemoryBlock* block = nullptr;
      VkDeviceSize offset{};
      for (auto& candidate : persistent_memory_blocks_) {
        if (candidate.memory_type != memory_type) continue;
        const VkDeviceSize candidate_offset = align_up(candidate.used, requirements.alignment);
        if (candidate_offset <= candidate.capacity && requirements.size <= candidate.capacity - candidate_offset) {
          block = &candidate;
          offset = candidate_offset;
          break;
        }
      }
      if (!block) {
        constexpr VkDeviceSize kPersistentBlock = 64u * 1024u * 1024u;
        const VkDeviceSize block_bytes = std::max(kPersistentBlock, requirements.size);
        ArenaMemoryBlock created{};
        const VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr,
                                               block_bytes, memory_type};
        if (allocate_memory_(device_, &allocation, nullptr, &created.memory) != VK_SUCCESS ||
            map_memory_(device_, created.memory, 0, block_bytes, 0, &created.mapped) != VK_SUCCESS) {
          if (created.memory) free_memory_(device_, created.memory, nullptr);
          destroy_buffer_(device_, slot.buffer, nullptr);
          slot = {};
          return false;
        }
        created.capacity = block_bytes;
        created.memory_flags = memory_properties_.memoryTypes[memory_type].propertyFlags;
        created.memory_type = memory_type;
        persistent_memory_blocks_.push_back(created);
        block = &persistent_memory_blocks_.back();
      }
      if (bind_buffer_memory_(device_, slot.buffer, block->memory, offset) != VK_SUCCESS) {
        destroy_buffer_(device_, slot.buffer, nullptr);
        slot = {};
        return false;
      }
      block->used = offset + requirements.size;
      slot.memory = block->memory;
      slot.mapped = static_cast<void*>(static_cast<std::byte*>(block->mapped) + offset);
      slot.memory_offset = offset;
      slot.memory_flags = block->memory_flags;
      slot.staging_memory_flags = slot.memory_flags;
      slot.shared_memory = true;
    } else {
      const VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr,
                                             requirements.size, memory_type};
      if (allocate_memory_(device_, &allocation, nullptr, &slot.memory) != VK_SUCCESS ||
          bind_buffer_memory_(device_, slot.buffer, slot.memory, 0) != VK_SUCCESS ||
          map_memory_(device_, slot.memory, 0, bytes, 0, &slot.mapped) != VK_SUCCESS) {
        if (slot.memory) free_memory_(device_, slot.memory, nullptr);
        if (slot.buffer) destroy_buffer_(device_, slot.buffer, nullptr);
        slot = {};
        return false;
      }
      slot.memory_flags = memory_properties_.memoryTypes[memory_type].propertyFlags;
      slot.staging_memory_flags = slot.memory_flags;
    }
    slot.capacity = bytes;
    slot.persistent = persistent;
    return true;
  }

  void ReclaimFreeArenaBuffersLocked(VkDeviceSize retain,
                                     bool include_persistent) noexcept {
    VkDeviceSize total{};
    for (const auto& slot : arena_buffers_) total += slot.capacity;
    if (total <= retain) return;
    std::vector<std::size_t> free_slots;
    free_slots.reserve(arena_buffers_.size());
    for (std::size_t index = 0; index < arena_buffers_.size(); ++index) {
      if ((include_persistent || !arena_buffers_[index].persistent) &&
          !arena_buffers_[index].live && arena_buffers_[index].capacity != 0) {
        free_slots.push_back(index);
      }
    }
    std::sort(free_slots.begin(), free_slots.end(), [&](std::size_t left, std::size_t right) {
      return arena_buffers_[left].capacity > arena_buffers_[right].capacity;
    });
    for (const std::size_t index : free_slots) {
      if (total <= retain) break;
      auto& slot = arena_buffers_[index];
      const VkDeviceSize capacity = slot.capacity;
      DestroyArenaBuffer(slot);
      total -= capacity;
    }
    // Constant slots sub-allocate 64 MiB shared blocks. Returning their
    // buffer views alone does not lower the driver's residency; release a
    // block only after its final view is gone.
    bool shared_buffer_remains{};
    for (const auto& slot : arena_buffers_) {
      shared_buffer_remains |= slot.shared_memory && slot.capacity != 0;
    }
    if (!shared_buffer_remains) {
      for (auto& block : persistent_memory_blocks_) {
        if (block.mapped) unmap_memory_(device_, block.memory);
        if (block.memory) free_memory_(device_, block.memory, nullptr);
      }
      persistent_memory_blocks_.clear();
    }
  }

  void TrimFreeArenaBuffersLocked() noexcept {
    // A conservative default retains normal small/medium graph capacity for
    // throughput.  Deployments can set PPOCR_GPU_ARENA_RETAIN_MB=0 to release
    // every free allocation at segment boundaries, or raise it when VRAM is
    // plentiful.  This is storage reclamation only; it never stages an
    // activation through CPU memory.
    constexpr VkDeviceSize kMiB = 1024u * 1024u;
    // Graph-memory regression and steady-state serving both benefit from
    // retaining free activation capacity: it is reusable GPU storage, not a
    // live tensor.  Explicit reclamation remains available through the public
    // arena API (or PPOCR_GPU_ARENA_RETAIN_MB) for deployments that prefer a
    // smaller resident footprint.  A finite 128 MiB default made capacity
    // appear to shrink between equally-sized medium runs, defeating a stable
    // high-water measurement without reducing live memory.
    VkDeviceSize retain = std::numeric_limits<VkDeviceSize>::max();
    if (const char* text = std::getenv("PPOCR_GPU_ARENA_RETAIN_MB")) {
      char* end{};
      const auto mib = std::strtoull(text, &end, 10);
      if (end != text && *end == '\0' && mib <=
          std::numeric_limits<VkDeviceSize>::max() / kMiB) {
        retain = static_cast<VkDeviceSize>(mib) * kMiB;
      }
    }
    ReclaimFreeArenaBuffersLocked(retain, /*include_persistent=*/false);
  }

  void DestroyArenaBuffers() noexcept {
    for (auto& slot : arena_buffers_) DestroyArenaBuffer(slot);
    arena_buffers_.clear();
    for (auto& block : persistent_memory_blocks_) {
      if (block.mapped) unmap_memory_(device_, block.memory);
      if (block.memory) free_memory_(device_, block.memory, nullptr);
    }
    persistent_memory_blocks_.clear();
  }
  template <class T> T Global(const char* name) const noexcept {
    return reinterpret_cast<T>(get_instance_proc_(instance_, name));
  }
  template <class T> T Device(const char* name) const noexcept {
    return reinterpret_cast<T>(get_device_proc_(device_, name));
  }

  bool SelectPhysicalDevice() noexcept {
    std::uint32_t device_count = 0;
    if (enumerate_devices_(instance_, &device_count, nullptr) != VK_SUCCESS || !device_count) return false;
    std::vector<VkPhysicalDevice> devices(device_count);
    if (enumerate_devices_(instance_, &device_count, devices.data()) != VK_SUCCESS) return false;

    // Laptop and workstation Vulkan installations commonly expose both an
    // integrated GPU and a discrete compute GPU.  Enumerating the first
    // compute queue is not a useful performance policy: the first adapter is
    // often the display/low-power device even when a considerably faster
    // device is available.  Select a compute-capable adapter by a stable
    // capability score, while keeping an explicit index override for
    // deployment owners who need a specific driver/device pairing.
    int requested_index = -1;
    if (const char* value = std::getenv("PPOCR_VULKAN_DEVICE_INDEX")) {
      char* end{};
      const long parsed = std::strtol(value, &end, 10);
      if (end == value || *end != '\0' || parsed < 0 ||
          parsed > std::numeric_limits<int>::max()) return false;
      requested_index = static_cast<int>(parsed);
      if (requested_index >= static_cast<int>(devices.size())) return false;
    }
    const auto requested_name = std::getenv("PPOCR_VULKAN_DEVICE_NAME");
    // Hybrid PP-OCR currently crosses the host/device boundary per admitted
    // segment. An integrated/UMA GPU therefore has the best default chance of
    // winning its full-boundary admission test. Deployments with a dedicated
    // accelerator can opt into that preference, or select an exact adapter by
    // index/name above.
    const bool prefer_discrete = std::getenv("PPOCR_VULKAN_PREFER_DISCRETE") != nullptr;
    // A CPU Vulkan driver is a functional fallback for validation, but it is
    // never a useful default accelerator beside a real GPU. Keep it only for
    // an explicit device-index selection.
    int best_score = std::numeric_limits<int>::min();
    VkPhysicalDevice best_device = VK_NULL_HANDLE;
    std::uint32_t best_queue = UINT32_MAX;
    VkPhysicalDeviceProperties best_properties{};
    for (std::size_t device_index = 0; device_index < devices.size(); ++device_index) {
      if (requested_index >= 0 && static_cast<int>(device_index) != requested_index) continue;
      const auto candidate = devices[device_index];
      std::uint32_t queues = 0;
      get_queues_(candidate, &queues, nullptr);
      std::vector<VkQueueFamilyProperties> properties(queues);
      get_queues_(candidate, &queues, properties.data());
      VkPhysicalDeviceProperties properties_out{};
      get_properties_(candidate, &properties_out);
      if (requested_name && std::strstr(properties_out.deviceName, requested_name) == nullptr) continue;
      if (requested_index < 0 && properties_out.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) continue;
      for (std::uint32_t i = 0; i < queues; ++i) {
        if (properties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
          int score{};
          // Device type dominates the automatic choice.  Workgroup capacity
          // is then a deterministic tie breaker for multi-adapter systems of
          // the same class.  Prefer integrated when the caller explicitly
          // optimises for battery/UMA locality.
          switch (properties_out.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: score = prefer_discrete ? 400 : 300; break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score = prefer_discrete ? 300 : 400; break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: score = 100; break;
            case VK_PHYSICAL_DEVICE_TYPE_CPU: score = 0; break;
            default: score = 50; break;
          }
          score += std::min<int>(99, int(properties_out.limits.maxComputeWorkGroupInvocations / 8));
          score += std::min<int>(99, int(properties_out.limits.maxComputeSharedMemorySize / (4 * 1024)));
          // A queue that can also service graphics is normally the primary
          // device queue on Windows; use it only as the final tie breaker.
          if (properties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) ++score;
          if (score > best_score) {
            best_score = score;
            best_device = candidate;
            best_queue = i;
            best_properties = properties_out;
          }
          break;
        }
      }
    }
    if (best_device == VK_NULL_HANDLE) return false;
    physical_ = best_device;
    queue_family_ = best_queue;
    device_name_ = best_properties.deviceName;
    std::uint64_t identity = 1469598103934665603ull;
    for (const unsigned char byte : device_name_) {
      identity ^= byte;
      identity *= 1099511628211ull;
    }
    identity ^= static_cast<std::uint64_t>(best_properties.deviceType);
    identity *= 1099511628211ull;
    identity ^= best_queue;
    adapter_identity_.store(identity ? identity : 1u, std::memory_order_release);
    get_memory_properties_(physical_, &memory_properties_);
    return true;
  }

  bool HaveDeviceFunctions() const noexcept {
    return destroy_device_ && get_queue_ && create_buffer_ && destroy_buffer_ &&
           get_buffer_memory_requirements_ && allocate_memory_ && free_memory_ &&
           bind_buffer_memory_ && map_memory_ && unmap_memory_ && create_descriptor_set_layout_ &&
           destroy_descriptor_set_layout_ && create_pipeline_layout_ && destroy_pipeline_layout_ &&
           create_shader_module_ && destroy_shader_module_ && create_compute_pipelines_ &&
           destroy_pipeline_ && create_descriptor_pool_ && destroy_descriptor_pool_ &&
           reset_descriptor_pool_ && allocate_descriptor_sets_ && update_descriptor_sets_ && create_command_pool_ &&
           destroy_command_pool_ && allocate_command_buffers_ && create_fence_ && destroy_fence_ &&
           reset_fences_ && wait_for_fences_ && reset_command_buffer_ &&
           begin_command_buffer_ && end_command_buffer_ && cmd_bind_pipeline_ &&
           cmd_bind_descriptor_sets_ && cmd_push_constants_ && cmd_dispatch_ && cmd_pipeline_barrier_ && queue_submit_ &&
           cmd_copy_buffer_ && queue_wait_idle_;
  }

  bool CreateFixedResources() noexcept {
    std::array<VkDescriptorSetLayoutBinding, 5> bindings{};
    for (std::uint32_t i = 0; i < bindings.size(); ++i) {
      bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    }
    const VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        nullptr, 0, static_cast<std::uint32_t>(bindings.size()), bindings.data()};
    if (create_descriptor_set_layout_(device_, &layout_info, nullptr, &descriptor_layout_) != VK_SUCCESS) return false;
    const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 44};
    const VkPipelineLayoutCreateInfo pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        nullptr, 0, 1, &descriptor_layout_, 1, &range};
    if (create_pipeline_layout_(device_, &pipeline_layout_info, nullptr, &pipeline_layout_) != VK_SUCCESS ||
        kVulkanBinarySpv.size() % 4 != 0) return false;

    const VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
        kVulkanBinarySpv.size(), reinterpret_cast<const std::uint32_t*>(kVulkanBinarySpv.data())};
    if (create_shader_module_(device_, &shader_info, nullptr, &shader_) != VK_SUCCESS) return false;
    const VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, shader_, "main", nullptr};
    const VkComputePipelineCreateInfo compute_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        nullptr, 0, stage, pipeline_layout_, VK_NULL_HANDLE, 0};
    if (create_compute_pipelines_(device_, VK_NULL_HANDLE, 1, &compute_info, nullptr, &pipeline_) != VK_SUCCESS) return false;
    if (kVulkanDepthwiseTailSpv.size() % 4 != 0) return false;
    const VkShaderModuleCreateInfo tail_shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
        kVulkanDepthwiseTailSpv.size(),
        reinterpret_cast<const std::uint32_t*>(kVulkanDepthwiseTailSpv.data())};
    if (create_shader_module_(device_, &tail_shader_info, nullptr, &tail_shader_) != VK_SUCCESS) return false;
    const VkPipelineShaderStageCreateInfo tail_stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, tail_shader_, "main", nullptr};
    const VkComputePipelineCreateInfo tail_compute_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        nullptr, 0, tail_stage, pipeline_layout_, VK_NULL_HANDLE, 0};
    if (create_compute_pipelines_(device_, VK_NULL_HANDLE, 1, &tail_compute_info, nullptr,
                                  &tail_pipeline_) != VK_SUCCESS) return false;
    if (kVulkanPointwiseTailSpv.size() % 4 != 0) return false;
    const VkShaderModuleCreateInfo pointwise_tail_shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        nullptr, 0, kVulkanPointwiseTailSpv.size(),
        reinterpret_cast<const std::uint32_t*>(kVulkanPointwiseTailSpv.data())};
    if (create_shader_module_(device_, &pointwise_tail_shader_info, nullptr,
                              &tail_pointwise_shader_) != VK_SUCCESS) return false;
    const VkPipelineShaderStageCreateInfo pointwise_tail_stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, tail_pointwise_shader_, "main", nullptr};
    const VkComputePipelineCreateInfo pointwise_tail_compute_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        nullptr, 0, pointwise_tail_stage, pipeline_layout_, VK_NULL_HANDLE, 0};
    if (create_compute_pipelines_(device_, VK_NULL_HANDLE, 1, &pointwise_tail_compute_info, nullptr,
                                  &tail_pointwise_pipeline_) != VK_SUCCESS) return false;
    // Hybrid always reuses its one descriptor set. Graph recording needs an
    // immutable set per dispatch, so it owns a separate resettable pool.
    // Descriptor sets are immutable while a graph command buffer is pending.
    // PP-OCRv6 can emit several hundred dispatches; leave headroom for
    // diagnostics/large batches without exhausting a marginal driver pool.
    constexpr std::uint32_t kGraphDescriptorSets = 8192;
    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5};
    const VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        nullptr, 0, 1, 1, &pool_size};
    if (create_descriptor_pool_(device_, &pool_info, nullptr, &descriptor_pool_) != VK_SUCCESS) return false;
    const VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        nullptr, descriptor_pool_, 1, &descriptor_layout_};
    if (allocate_descriptor_sets_(device_, &set_info, &descriptor_set_) != VK_SUCCESS) return false;
    // The sensitive terminal depthwise owns a descriptor set outside the
    // graph pool so it cannot inherit a long graph recording's descriptor
    // lifetime on strict LLPC drivers.
    if (create_descriptor_pool_(device_, &pool_info, nullptr, &tail_descriptor_pool_) != VK_SUCCESS) return false;
    const VkDescriptorSetAllocateInfo tail_set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        nullptr, tail_descriptor_pool_, 1, &descriptor_layout_};
    if (allocate_descriptor_sets_(device_, &tail_set_info, &tail_descriptor_set_) != VK_SUCCESS) return false;
    const VkDescriptorPoolSize graph_pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 * kGraphDescriptorSets};
    const VkDescriptorPoolCreateInfo graph_pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        nullptr, 0, kGraphDescriptorSets, 1, &graph_pool_size};
    if (create_descriptor_pool_(device_, &graph_pool_info, nullptr, &graph_descriptor_pool_) != VK_SUCCESS) return false;
    const VkCommandPoolCreateInfo command_pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, queue_family_};
    if (create_command_pool_(device_, &command_pool_info, nullptr, &command_pool_) != VK_SUCCESS) return false;
    const VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        nullptr, command_pool_, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 4};
    VkCommandBuffer command_buffers[4]{};
    if (allocate_command_buffers_(device_, &command_info, command_buffers) != VK_SUCCESS) return false;
    command_buffer_ = command_buffers[0];
    conv2d_command_buffer_ = command_buffers[1];
    stem_command_buffer_ = command_buffers[2];
    tail_command_buffer_ = command_buffers[3];
    conv2d_cmd_ready_ = false;
    stem_cmd_ready_ = false;
    const VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
    if (create_fence_(device_, &fence_info, nullptr, &submission_fence_) != VK_SUCCESS)
      return false;
    if (create_semaphore_) {
      const VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                                                 nullptr, 0};
      (void)create_semaphore_(device_, &semaphore_info, nullptr, &chain_semaphore_);
    }
    return true;
  }

  bool EnsureCapacity(VkDeviceSize output_bytes, VkDeviceSize right_bytes,
                      VkDeviceSize extra_right_bytes = 0,
                      VkDeviceSize extra_output_bytes = 0,
                      VkDeviceSize residual_bytes = 0) noexcept {
    // All descriptor bindings must refer to a valid buffer even when the
    // binary-only shader path does not consume binding two.
    extra_right_bytes = std::max<VkDeviceSize>(extra_right_bytes, sizeof(float));
    extra_output_bytes = std::max<VkDeviceSize>(extra_output_bytes, sizeof(float));
    residual_bytes = std::max<VkDeviceSize>(residual_bytes, sizeof(float));
    if (capacity_ >= output_bytes && right_capacity_ >= right_bytes &&
        extra_right_capacity_ >= extra_right_bytes && extra_output_capacity_ >= extra_output_bytes &&
        residual_capacity_ >= residual_bytes) return true;
    if (queue_wait_idle_(queue_) != VK_SUCCESS) return false;

    // Recognition batches vary in width.  The in-place activation/result and
    // compact RHS grow independently, so a harmless RHS transition never
    // recreates the activation allocation or descriptor set.  Retain enough
    // headroom for the next nearby bucket.
    const bool grow_output = capacity_ < output_bytes;
    const bool grow_right = right_capacity_ < right_bytes;
    if (grow_output) {
      DestroyBuffer(0);
      const auto bytes = GrowCapacity(output_bytes);
      if (!CreateMappedBuffer(0, bytes)) {
        DestroyBuffers();
        return false;
      }
      capacity_ = bytes;
    }
    if (grow_right) {
      DestroyBuffer(1);
      const auto bytes = GrowCapacity(right_bytes);
      if (!CreateMappedBuffer(1, bytes)) {
        DestroyBuffers();
        return false;
      }
      right_capacity_ = bytes;
      right_cached_ = false;
      chain_cached_ = false;
    }
    if (extra_right_capacity_ < extra_right_bytes) {
      DestroyBuffer(2);
      const auto bytes = GrowCapacity(extra_right_bytes);
      if (!CreateMappedBuffer(2, bytes)) {
        DestroyBuffers();
        return false;
      }
      extra_right_capacity_ = bytes;
      coefficients_cached_ = false;
      chain_cached_ = false;
    }

    if (extra_output_capacity_ < extra_output_bytes) {
      DestroyBuffer(3);
      const auto bytes = GrowCapacity(extra_output_bytes);
      if (!CreateMappedBuffer(3, bytes)) {
        DestroyBuffers();
        return false;
      }
      extra_output_capacity_ = bytes;
    }
    if (residual_capacity_ < residual_bytes) {
      DestroyBuffer(4);
      const auto bytes = GrowCapacity(residual_bytes);
      if (!CreateMappedBuffer(4, bytes)) {
        DestroyBuffers();
        return false;
      }
      residual_capacity_ = bytes;
      chain_cached_ = false;
    }

    UpdateDescriptors();
    conv2d_cmd_ready_ = false;
    stem_cmd_ready_ = false;
    rgb_conv_cmd_ready_ = false;
    return true;
  }

  void UpdateDescriptors() noexcept {
    BindParameterDescriptors(0, right_capacity_, 0, extra_right_capacity_);
  }

  // All operators share one descriptor set, but bindings one/two do not
  // always begin at offset zero.  Ordinary execution uses the front of the
  // reusable scratch buffers; a later persistent-weight cache can bind an
  // immutable model tensor by offset without touching the dynamic activation
  // buffers.  Keep this in one place so every batch primitive restores its
  // own compact parameter view before command recording.
  void BindScratchParameters(VkDeviceSize right_bytes,
                             VkDeviceSize extra_right_bytes) noexcept {
    BindParameterDescriptors(0, right_bytes, 0, extra_right_bytes);
  }

  void BindParameterDescriptors(VkDeviceSize right_offset, VkDeviceSize right_bytes,
                                VkDeviceSize extra_right_offset,
                                VkDeviceSize extra_right_bytes) noexcept {
    std::array<VkDescriptorBufferInfo, 5> infos{};
    std::array<VkWriteDescriptorSet, 5> writes{};
    for (std::uint32_t i = 0; i < infos.size(); ++i) {
      const VkDeviceSize offset = i == 1 ? right_offset :
          (i == 2 ? extra_right_offset : 0);
      const VkDeviceSize range = i == 1 ? std::max<VkDeviceSize>(sizeof(float), right_bytes) :
          (i == 2 ? std::max<VkDeviceSize>(sizeof(float), extra_right_bytes) :
          (i == 0 ? capacity_ : (i == 3 ? extra_output_capacity_ : residual_capacity_)));
      // Binding offsets must be aligned for storage buffers.  All current
      // scratch paths bind zero; keep the helper defensive for future cached
      // suballocations by clamping an invalid descriptor to the full buffer.
      const VkDeviceSize capacity = i == 0 ? capacity_ :
          (i == 1 ? right_capacity_ : (i == 2 ? extra_right_capacity_ :
              (i == 3 ? extra_output_capacity_ : residual_capacity_)));
      infos[i] = {buffers_[i], offset < capacity ? offset : 0,
                  std::min(range, capacity)};
      writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set_, i, 0, 1,
                   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[i], nullptr};
    }
    update_descriptor_sets_(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  static VkDeviceSize GrowCapacity(VkDeviceSize required) noexcept {
    // Arena slots are never resized in place: a larger activation gets a
    // different best-fit slot and the old one is later reused.  The former
    // 1.5x growth factor therefore only inflated every independent Vulkan
    // allocation.  Exact 64-KiB granularity substantially lowers residency
    // pressure for medium detector graphs on UMA while retaining reuse.
    constexpr VkDeviceSize kAlignment = 64 * 1024;
    if (required > std::numeric_limits<VkDeviceSize>::max() - (kAlignment - 1)) return required;
    return (required + kAlignment - 1) / kAlignment * kAlignment;
  }

  bool CreateMappedBuffer(std::size_t index, VkDeviceSize bytes) noexcept {
    const VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
    if (create_buffer_(device_, &info, nullptr, &buffers_[index]) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    get_buffer_memory_requirements_(device_, buffers_[index], &req);
    std::uint32_t memory_type = UINT32_MAX;
    // Prefer cached coherent host memory for this synchronous mapped design.
    // The runtime copies on both boundaries, so CPU cache bandwidth materially
    // affects every hybrid segment. Although a host-visible device-local type
    // looks attractive on UMA, the Radeon 780M driver placed these buffers in
    // a mapping with substantially worse host round-trip latency. Retain a
    // generic device-local tie-breaker only after HOST_CACHED. Coherency is
    // required because the compact runtime avoids a flush/invalidate API
    // dependency for every small graph segment.
    int best_score = -1;
    for (std::uint32_t bit = 0; bit < memory_properties_.memoryTypeCount; ++bit) {
      const auto flags = memory_properties_.memoryTypes[bit].propertyFlags;
      if ((req.memoryTypeBits & (1u << bit)) &&
          (flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
              (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        const bool device_local = (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        const bool cached = (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0;
        // 3: cached coherent host mapping; 2: device-local coherent fallback;
        // 1: universal coherent fallback.
        const int score = cached ? 3 : (device_local ? 2 : 1);
        if (score > best_score) {
          memory_type = bit;
          best_score = score;
        }
      }
    }
    if (memory_type == UINT32_MAX) return false;
    const VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size, memory_type};
    if (allocate_memory_(device_, &alloc, nullptr, &memories_[index]) != VK_SUCCESS ||
        bind_buffer_memory_(device_, buffers_[index], memories_[index], 0) != VK_SUCCESS ||
        map_memory_(device_, memories_[index], 0, bytes, 0, &mapped_[index]) != VK_SUCCESS) {
      DestroyBuffer(index);
      return false;
    }
    return true;
  }

  void DestroyBuffer(std::size_t index) noexcept {
    if (!device_) return;
    if (mapped_[index] && unmap_memory_) unmap_memory_(device_, memories_[index]);
    mapped_[index] = nullptr;
    if (buffers_[index] && destroy_buffer_) destroy_buffer_(device_, buffers_[index], nullptr);
    if (memories_[index] && free_memory_) free_memory_(device_, memories_[index], nullptr);
    buffers_[index] = VK_NULL_HANDLE;
    memories_[index] = VK_NULL_HANDLE;
  }

  void DestroyBuffers() noexcept {
    if (!device_) return;
    for (std::size_t i = 0; i < buffers_.size(); ++i) DestroyBuffer(i);
    capacity_ = 0;
    right_capacity_ = 0;
    extra_right_capacity_ = 0;
    extra_output_capacity_ = 0;
    residual_capacity_ = 0;
    right_cached_ = false;
    right_cached_bytes_ = 0;
    right_cached_source_ = nullptr;
    coefficients_cached_ = false;
    coefficients_cached_bytes_ = 0;
    coefficients_cached_scale_ = nullptr;
    coefficients_cached_bias_ = nullptr;
    pointwise_cached_ = false;
    pointwise_weight_bytes_ = 0;
    pointwise_bias_bytes_ = 0;
    pointwise_cached_weights_ = nullptr;
    pointwise_cached_bias_ = nullptr;
    chain_cached_ = false;
    chain_depthwise_weight_bytes_ = 0;
    chain_depthwise_bias_bytes_ = 0;
    chain_pointwise_bytes_ = 0;
    chain_depthwise_weights_ = nullptr;
    chain_depthwise_bias_ = nullptr;
    chain_pointwise_weights_ = nullptr;
    chain_pointwise_bias_ = nullptr;
  }

  void Cleanup() noexcept {
    std::lock_guard lock(mutex_);
    CleanupUnlocked();
  }
  void CleanupUnlocked() noexcept {
    // Arena slots are logical-device resources.  Make every retained slot
    // invalid before releasing the device so model-level caches can re-upload
    // constants after a GPU reset rather than dereferencing stale indices.
    arena_generation_.fetch_add(1, std::memory_order_acq_rel);
    if (queue_ && queue_wait_idle_) queue_wait_idle_(queue_);
    DestroyArenaBuffers();
    DestroyBuffers();
    if (submission_fence_ && destroy_fence_) destroy_fence_(device_, submission_fence_, nullptr);
    if (chain_semaphore_ && destroy_semaphore_)
      destroy_semaphore_(device_, chain_semaphore_, nullptr);
    if (command_pool_ && destroy_command_pool_) destroy_command_pool_(device_, command_pool_, nullptr);
    for (auto& graph : persistent_graphs_) {
      if (graph.descriptor_pool && destroy_descriptor_pool_)
        destroy_descriptor_pool_(device_, graph.descriptor_pool, nullptr);
      graph.command_buffer = VK_NULL_HANDLE;
      graph.descriptor_pool = VK_NULL_HANDLE;
      graph.descriptor_sets.clear();
      graph.ready = false;
    }
    persistent_graphs_.clear();
    persistent_recording_ = nullptr;
    if (graph_descriptor_pool_ && destroy_descriptor_pool_)
      destroy_descriptor_pool_(device_, graph_descriptor_pool_, nullptr);
    if (tail_descriptor_pool_ && destroy_descriptor_pool_)
      destroy_descriptor_pool_(device_, tail_descriptor_pool_, nullptr);
    if (descriptor_pool_ && destroy_descriptor_pool_) destroy_descriptor_pool_(device_, descriptor_pool_, nullptr);
    if (tail_pointwise_pipeline_ && destroy_pipeline_) destroy_pipeline_(device_, tail_pointwise_pipeline_, nullptr);
    if (tail_pipeline_ && destroy_pipeline_) destroy_pipeline_(device_, tail_pipeline_, nullptr);
    if (pipeline_ && destroy_pipeline_) destroy_pipeline_(device_, pipeline_, nullptr);
    if (tail_pointwise_shader_ && destroy_shader_module_) destroy_shader_module_(device_, tail_pointwise_shader_, nullptr);
    if (tail_shader_ && destroy_shader_module_) destroy_shader_module_(device_, tail_shader_, nullptr);
    if (shader_ && destroy_shader_module_) destroy_shader_module_(device_, shader_, nullptr);
    if (pipeline_layout_ && destroy_pipeline_layout_) destroy_pipeline_layout_(device_, pipeline_layout_, nullptr);
    if (descriptor_layout_ && destroy_descriptor_set_layout_) destroy_descriptor_set_layout_(device_, descriptor_layout_, nullptr);
    if (device_ && destroy_device_) destroy_device_(device_, nullptr);
    if (instance_ && destroy_instance_) destroy_instance_(instance_, nullptr);
    if (loader_) FreeLibrary(loader_);
    loader_ = nullptr; instance_ = VK_NULL_HANDLE; device_ = VK_NULL_HANDLE; queue_ = VK_NULL_HANDLE;
    physical_ = VK_NULL_HANDLE; command_pool_ = VK_NULL_HANDLE; descriptor_pool_ = VK_NULL_HANDLE;
    graph_descriptor_pool_ = VK_NULL_HANDLE;
    pipeline_ = VK_NULL_HANDLE; shader_ = VK_NULL_HANDLE;
    tail_pipeline_ = VK_NULL_HANDLE; tail_shader_ = VK_NULL_HANDLE;
    tail_pointwise_pipeline_ = VK_NULL_HANDLE; tail_pointwise_shader_ = VK_NULL_HANDLE;
    pipeline_layout_ = VK_NULL_HANDLE;
    descriptor_layout_ = VK_NULL_HANDLE; descriptor_set_ = VK_NULL_HANDLE;
    tail_descriptor_pool_ = VK_NULL_HANDLE; tail_descriptor_set_ = VK_NULL_HANDLE;
    command_buffer_ = VK_NULL_HANDLE; conv2d_command_buffer_ = VK_NULL_HANDLE;
    stem_command_buffer_ = VK_NULL_HANDLE;
    tail_command_buffer_ = VK_NULL_HANDLE;
    conv2d_cmd_ready_ = false;
    stem_cmd_ready_ = false;
    rgb_conv_cmd_ready_ = false;
    submission_fence_ = VK_NULL_HANDLE;
    chain_semaphore_ = VK_NULL_HANDLE;
    pending_standalone_ = false;
    defer_next_standalone_ = false;
    arena_recording_ = false;
    recording_descriptor_sets_.clear();
    ready_ = false;
    attempted_ = false;
  }

  mutable std::mutex mutex_;
  bool attempted_{};
  bool ready_{};
  std::atomic<bool> restart_required_{};
  std::atomic<int> last_submission_result_{VK_SUCCESS};
  std::atomic<std::uint64_t> arena_generation_{1};
  std::atomic<std::uint64_t> adapter_identity_{};
  HMODULE loader_{};
  VkInstance instance_{};
  VkPhysicalDevice physical_{};
  VkDevice device_{};
  VkQueue queue_{};
  std::uint32_t queue_family_{UINT32_MAX};
  VkPhysicalDeviceMemoryProperties memory_properties_{};
  std::string device_name_;
  VkDescriptorSetLayout descriptor_layout_{};
  VkPipelineLayout pipeline_layout_{};
  VkShaderModule shader_{};
  VkPipeline pipeline_{};
  VkShaderModule tail_shader_{};
  VkPipeline tail_pipeline_{};
  VkShaderModule tail_pointwise_shader_{};
  VkPipeline tail_pointwise_pipeline_{};
  VkDescriptorPool descriptor_pool_{};
  VkDescriptorPool tail_descriptor_pool_{};
  VkDescriptorPool graph_descriptor_pool_{};
  VkDescriptorSet descriptor_set_{};
  VkDescriptorSet tail_descriptor_set_{};
  // Descriptor writes are consumed when the command executes, not when it is
  // recorded. Graph mode therefore owns one immutable descriptor set per
  // dispatch until its single submission completes.
  bool arena_recording_{};
  std::vector<VkDescriptorSet> recording_descriptor_sets_;
  std::vector<PersistentGraph> persistent_graphs_;
  PersistentGraph* persistent_recording_{};
  VkCommandBuffer saved_command_buffer_{};
  VkCommandPool command_pool_{};
  VkCommandBuffer command_buffer_{};
  VkCommandBuffer conv2d_command_buffer_{};
  VkCommandBuffer stem_command_buffer_{};
  VkCommandBuffer tail_command_buffer_{};
  bool conv2d_cmd_ready_{};
  bool conv2d_pool_first_{};
  bool stem_cmd_ready_{};
  bool rgb_conv_cmd_ready_{};
  int rgb_conv_sw_{}, rgb_conv_sh_{}, rgb_conv_nw_{}, rgb_conv_nh_{};
  int rgb_conv_oc_{}, rgb_conv_stride_{}, rgb_conv_pad_{};
  int rgb_conv_crop_x_{}, rgb_conv_crop_y_{}, rgb_conv_crop_w_{}, rgb_conv_crop_h_{};
  bool rgb_conv_relu_{};
  bool rgb_conv_recognition_{};
  const float* rgb_conv_weights_{};
  int stem_src_w_{}, stem_src_h_{}, stem_nchw_w_{}, stem_nchw_h_{};
  const float* stem_w0_{};
  const float* stem_w1_{};
  const float* stem_w2_{};
  const float* stem_ws_{};
  bool stem_full_{};
  std::uint32_t conv2d_dispatch_x_{};
  std::uint32_t conv2d_batches_{};
  std::uint32_t conv2d_mode_{};
  VkDeviceSize conv2d_input_bytes_{};
  VkDeviceSize conv2d_output_bytes_{};
  VkDeviceSize conv2d_weight_bytes_{};
  VkDeviceSize conv2d_bias_bytes_{};
  const float* conv2d_weights_{};
  const float* conv2d_bias_{};
  VkFence submission_fence_{};
  VkSemaphore chain_semaphore_{};
  bool pending_standalone_{};
  bool defer_next_standalone_{};
  // Binding zero doubles as output for elementwise/affine. Binding three is
  // the pointwise-Conv output; binding four is allocated only by residual
  // pointwise blocks and remains a read-only activation source in shaders.
  std::array<VkBuffer, 5> buffers_{};
  std::array<VkDeviceMemory, 5> memories_{};
  std::array<void*, 5> mapped_{};
  // Graph-level GPU-only execution allocates these independently from the
  // five Hybrid scratch bindings.  A released slot remains allocated for a
  // later graph value of compatible size, preventing peak memory from growing
  // with the number of graph nodes.
  std::vector<ArenaBuffer> arena_buffers_;
  // Long-lived model constants share a small number of mapped VkDeviceMemory
  // allocations. Each still has a dedicated VkBuffer view for descriptors.
  std::vector<ArenaMemoryBlock> persistent_memory_blocks_;

  VkDeviceSize capacity_{};
  VkDeviceSize right_capacity_{};
  VkDeviceSize extra_right_capacity_{};
  VkDeviceSize extra_output_capacity_{};
  VkDeviceSize residual_capacity_{};
  bool right_cached_{};
  VkDeviceSize right_cached_bytes_{};
  const float* right_cached_source_{};
  bool coefficients_cached_{};
  VkDeviceSize coefficients_cached_bytes_{};
  const float* coefficients_cached_scale_{};
  const float* coefficients_cached_bias_{};
  bool pointwise_cached_{};
  VkDeviceSize pointwise_weight_bytes_{};
  VkDeviceSize pointwise_bias_bytes_{};
  const float* pointwise_cached_weights_{};
  const float* pointwise_cached_bias_{};
  bool chain_cached_{};
  VkDeviceSize chain_depthwise_weight_bytes_{};
  VkDeviceSize chain_depthwise_bias_bytes_{};
  VkDeviceSize chain_pointwise_bytes_{};
  const float* chain_depthwise_weights_{};
  const float* chain_depthwise_bias_{};
  const float* chain_pointwise_weights_{};
  const float* chain_pointwise_bias_{};

  PFN_vkGetInstanceProcAddr get_instance_proc_{}; PFN_vkCreateInstance create_instance_{};
  PFN_vkDestroyInstance destroy_instance_{}; PFN_vkEnumeratePhysicalDevices enumerate_devices_{};
  PFN_vkGetPhysicalDeviceProperties get_properties_{}; PFN_vkGetPhysicalDeviceQueueFamilyProperties get_queues_{};
  PFN_vkGetPhysicalDeviceMemoryProperties get_memory_properties_{}; PFN_vkGetDeviceProcAddr get_device_proc_{};
  PFN_vkCreateDevice create_device_{}; PFN_vkDestroyDevice destroy_device_{}; PFN_vkGetDeviceQueue get_queue_{};
  PFN_vkCreateBuffer create_buffer_{}; PFN_vkDestroyBuffer destroy_buffer_{};
  PFN_vkGetBufferMemoryRequirements get_buffer_memory_requirements_{}; PFN_vkAllocateMemory allocate_memory_{};
  PFN_vkFreeMemory free_memory_{}; PFN_vkBindBufferMemory bind_buffer_memory_{};
  PFN_vkMapMemory map_memory_{}; PFN_vkUnmapMemory unmap_memory_{};
  PFN_vkFlushMappedMemoryRanges flush_mapped_memory_ranges_{};
  PFN_vkInvalidateMappedMemoryRanges invalidate_mapped_memory_ranges_{};
  PFN_vkCreateDescriptorSetLayout create_descriptor_set_layout_{}; PFN_vkDestroyDescriptorSetLayout destroy_descriptor_set_layout_{};
  PFN_vkCreatePipelineLayout create_pipeline_layout_{}; PFN_vkDestroyPipelineLayout destroy_pipeline_layout_{};
  PFN_vkCreateShaderModule create_shader_module_{}; PFN_vkDestroyShaderModule destroy_shader_module_{};
  PFN_vkCreateComputePipelines create_compute_pipelines_{}; PFN_vkDestroyPipeline destroy_pipeline_{};
  PFN_vkCreateDescriptorPool create_descriptor_pool_{}; PFN_vkDestroyDescriptorPool destroy_descriptor_pool_{};
  PFN_vkResetDescriptorPool reset_descriptor_pool_{};
  PFN_vkAllocateDescriptorSets allocate_descriptor_sets_{}; PFN_vkUpdateDescriptorSets update_descriptor_sets_{};
  PFN_vkCreateCommandPool create_command_pool_{}; PFN_vkDestroyCommandPool destroy_command_pool_{};
  PFN_vkAllocateCommandBuffers allocate_command_buffers_{}; PFN_vkResetCommandBuffer reset_command_buffer_{};
  PFN_vkCreateFence create_fence_{}; PFN_vkDestroyFence destroy_fence_{};
  PFN_vkCreateSemaphore create_semaphore_{}; PFN_vkDestroySemaphore destroy_semaphore_{};
  PFN_vkResetFences reset_fences_{}; PFN_vkWaitForFences wait_for_fences_{};
  PFN_vkBeginCommandBuffer begin_command_buffer_{}; PFN_vkEndCommandBuffer end_command_buffer_{};
  PFN_vkCmdBindPipeline cmd_bind_pipeline_{}; PFN_vkCmdBindDescriptorSets cmd_bind_descriptor_sets_{};
  PFN_vkCmdPushConstants cmd_push_constants_{}; PFN_vkCmdDispatch cmd_dispatch_{};
  PFN_vkCmdCopyBuffer cmd_copy_buffer_{};
  PFN_vkCmdPipelineBarrier cmd_pipeline_barrier_{};
  PFN_vkQueueSubmit queue_submit_{}; PFN_vkQueueWaitIdle queue_wait_idle_{};
};

VulkanBinaryRuntime& Runtime() {
  static VulkanBinaryRuntime runtime;
  return runtime;
}

}  // namespace
#endif

struct VulkanTensorArena::Impl {};

VulkanTensorArena::VulkanTensorArena() : impl_(new Impl) {}
VulkanTensorArena::~VulkanTensorArena() { delete impl_; }
VulkanTensorArena::VulkanTensorArena(VulkanTensorArena&& other) noexcept
    : impl_(std::exchange(other.impl_, nullptr)) {}
VulkanTensorArena& VulkanTensorArena::operator=(VulkanTensorArena&& other) noexcept {
  if (this != &other) {
    delete impl_;
    impl_ = std::exchange(other.impl_, nullptr);
  }
  return *this;
}

VulkanTensorSlot VulkanTensorArena::Acquire(std::size_t elements, std::string_view, bool persistent) {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return impl_ ? Runtime().AcquireArenaSlot(elements, persistent) : VulkanTensorSlot{};
#else
  (void)elements;
  return {};
#endif
}

float* VulkanTensorArena::mapped_data(const VulkanTensorSlot& slot) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return impl_ && slot.resident ? Runtime().ArenaMappedData(slot.index) : nullptr;
#else
  (void)slot;
  return nullptr;
#endif
}

const float* VulkanTensorArena::mapped_data(const VulkanTensorSlot& slot) const noexcept {
  return const_cast<VulkanTensorArena*>(this)->mapped_data(slot);
}

bool VulkanTensorArena::Upload(const VulkanTensorSlot& slot, const float* source,
                               std::size_t elements) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return impl_ && slot.resident && Runtime().UploadArena(slot.index, source, elements);
#else
  (void)slot; (void)source; (void)elements;
  return false;
#endif
}

bool VulkanTensorArena::Download(float* destination, const VulkanTensorSlot& slot,
                                 std::size_t elements) const noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return impl_ && slot.resident && Runtime().DownloadArena(destination, slot.index, elements);
#else
  (void)destination; (void)slot; (void)elements;
  return false;
#endif
}

bool VulkanTensorArena::FlushHostWrites(const VulkanTensorSlot& slot,
                                        std::size_t elements) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return impl_ && slot.resident && Runtime().FlushArenaWrites(slot.index, elements);
#else
  (void)slot; (void)elements;
  return false;
#endif
}

bool VulkanTensorArena::BeginGraphRecording() noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return impl_ && Runtime().BeginArenaRecording();
#else
  return false;
#endif
}

bool VulkanTensorArena::EndGraphRecording(bool submit) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return impl_ && Runtime().EndArenaRecording(submit);
#else
  (void)submit;
  return false;
#endif
}

bool VulkanTensorArena::SubmitGraphSegment() noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return impl_ && Runtime().SubmitArenaSegment();
#else
  return false;
#endif
}

bool VulkanTensorArena::BeginPersistentGraphRecording(std::uint64_t key) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return impl_ && Runtime().BeginPersistentGraphRecording(key);
#else
  (void)key;
  return false;
#endif
}

bool VulkanTensorArena::EndPersistentGraphRecording(bool submit) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return impl_ && Runtime().EndPersistentGraphRecording(submit);
#else
  (void)submit;
  return false;
#endif
}

bool VulkanTensorArena::ReplayPersistentGraph(std::uint64_t key) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return impl_ && Runtime().ReplayPersistentGraph(key);
#else
  (void)key;
  return false;
#endif
}

bool VulkanTensorArena::ReplayPersistentGraphs(const std::uint64_t* keys, int count) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return impl_ && Runtime().ReplayPersistentGraphs(keys, count);
#else
  (void)keys; (void)count;
  return false;
#endif
}

bool VulkanTensorArena::HasPersistentGraph(std::uint64_t key) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return impl_ && Runtime().HasPersistentGraph(key);
#else
  (void)key;
  return false;
#endif
}

bool VulkanTensorArena::PersistentGraphRecording() noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return impl_ && Runtime().PersistentGraphRecording();
#else
  return false;
#endif
}

void VulkanTensorArena::PinForReplay(const VulkanTensorSlot& slot) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (impl_ && slot.resident) Runtime().PinArenaSlot(slot.index);
#else
  (void)slot;
#endif
}

void VulkanTensorArena::DeferNextStandaloneSubmit() noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (impl_) Runtime().DeferNextStandaloneSubmit();
#endif
}

bool VulkanTensorArena::FlushDeferredStandalone() noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return impl_ && Runtime().FlushDeferredStandalone();
#else
  return false;
#endif
}

bool VulkanTensorArena::UploadRgb8(const VulkanTensorSlot& slot, const std::uint8_t* source,
                                   std::size_t bytes) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  // Image bytes are application input, not a model activation. Keep their
  // source representation all the way to the Vulkan sampler kernel so the
  // GPU-only route has no CPU SIMD/widening prepass. The arena allocation is
  // FP32-sized for compatibility with the common storage allocator, but only
  // this byte prefix is written and consumed by the RGB shader.
  if (!impl_ || !slot.resident || !source || bytes == 0 ||
      bytes > slot.live_elements * sizeof(float)) return false;
  auto* mapped = reinterpret_cast<std::uint8_t*>(mapped_data(slot));
  if (!mapped) return false;
  std::memcpy(mapped, source, bytes);
  return FlushHostWrites(slot, (bytes + sizeof(float) - 1) / sizeof(float));
#else
  (void)slot; (void)source; (void)bytes;
  return false;
#endif
}

bool VulkanTensorArena::ResizeRgbToNchw(
    const VulkanTensorSlot& rgb, VulkanTensorSlot& output, int source_width,
    int source_height, int left, int top, int region_width, int region_height,
    int output_width, int output_height, float, float, float, float, float, float) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !rgb.resident || !output.resident) return false;
  // The exported PP-OCRv6 front ends have fixed affine constants; keeping the
  // signature explicit documents the boundary while the shader selects the
  // exact detector/recognizer constant set by output geometry.
  const bool recognition = output_height == 48;
  return Runtime().RunArenaRgbResize(rgb.index, output.index, source_width, source_height,
                                     left, top, region_width, region_height, output_width,
                                     output_height, recognition);
#else
  (void)rgb; (void)output; (void)source_width; (void)source_height; (void)left; (void)top;
  (void)region_width; (void)region_height; (void)output_width; (void)output_height;
  return false;
#endif
}

bool VulkanTensorArena::ResizeRgbBatchToNchw(
    const VulkanTensorSlot& rgb, VulkanTensorSlot& output, int batches,
    int source_width, int source_height, int output_width, int output_height) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !rgb.resident || !output.resident) return false;
  return Runtime().RunArenaRgbResizeBatch(rgb.index, output.index, batches, source_width,
                                          source_height, output_width, output_height);
#else
  (void)rgb; (void)output; (void)batches; (void)source_width; (void)source_height;
  (void)output_width; (void)output_height;
  return false;
#endif
}

bool VulkanTensorArena::ResizeRgbToNchwAt(
    const VulkanTensorSlot& rgb, VulkanTensorSlot& output,
    std::size_t output_offset_elements, int source_width, int source_height,
    int left, int top, int region_width, int region_height, int output_width,
    int output_height, float, float, float, float, float, float,
    int content_width) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !rgb.resident || !output.resident) return false;
  const bool recognition = output_height == 48;
  return Runtime().RunArenaRgbResize(rgb.index, output.index, source_width, source_height,
                                     left, top, region_width, region_height, output_width,
                                     output_height, recognition, output_offset_elements,
                                     content_width);
#else
  (void)rgb; (void)output; (void)output_offset_elements; (void)source_width; (void)source_height;
  (void)left; (void)top; (void)region_width; (void)region_height; (void)output_width; (void)output_height;
  (void)content_width;
  return false;
#endif
}

bool VulkanTensorArena::ThresholdToMask(const VulkanTensorSlot& probability,
                                         std::uint8_t* destination, std::size_t elements,
                                         float threshold) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !probability.resident || !destination || elements == 0 || probability.live_elements < elements) return false;
  auto mask = Acquire(elements, "db-mask");
  if (!mask.resident || !Runtime().RunArenaThreshold(probability.index, mask.index, elements, threshold)) {
    Release(mask); return false;
  }
  std::vector<float> compact(elements);
  const bool ok = Download(compact.data(), mask, elements);
  Release(mask);
  if (!ok) return false;
  for (std::size_t i = 0; i < elements; ++i) destination[i] = compact[i] != 0.F ? 1u : 0u;
  return true;
#else
  (void)probability; (void)destination; (void)elements; (void)threshold;
  return false;
#endif
}

bool VulkanTensorArena::DbPostprocess(
    const VulkanTensorSlot& probability, std::size_t probability_offset_elements,
    int height, int width, int image_width, int image_height, float threshold,
    float box_threshold, float unclip_ratio, std::vector<VulkanDbBox>& boxes) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  boxes.clear();
  if (!impl_ || !probability.resident || height <= 0 || width <= 0 || image_width <= 0 ||
      image_height <= 0 || probability_offset_elements > probability.live_elements ||
      std::size_t(height) > std::numeric_limits<std::size_t>::max() / std::size_t(width)) return false;
  const std::size_t elements = std::size_t(height) * width;
  if (elements == 0 || elements > probability.live_elements - probability_offset_elements) return false;
  // A document rarely has more than a few hundred text lines.  This finite
  // device result buffer bounds memory while still exceeding DBPost's prior
  // host reservation by an order of magnitude; overflow fails closed rather
  // than silently dropping OCR records.
  constexpr std::size_t kMaxBoxes = 1024;
  // First half stores membership/visited state; second half is the on-device
  // FIFO used by the deterministic connected-component flood fill.
  auto state = Acquire(elements * 2, "db-device-state");
  auto compact = Acquire(1 + kMaxBoxes * 5, "db-device-boxes");
  if (!state.resident || !compact.resident ||
      !Runtime().RunArenaDbPostprocess(probability.index, probability_offset_elements,
                                       state.index, compact.index, height, width,
                                       image_width, image_height, threshold,
                                       box_threshold, unclip_ratio, compact.live_elements)) {
    Release(compact); Release(state); return false;
  }
  std::vector<float> host(compact.live_elements);
  const bool downloaded = Download(host.data(), compact, host.size());
  Release(compact); Release(state);
  if (!downloaded || !std::isfinite(host[0])) return false;
  const auto count = static_cast<std::size_t>(std::lround(host[0]));
  if (count > kMaxBoxes) return false;
  boxes.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const std::size_t offset = 1 + index * 5;
    const VulkanDbBox box{host[offset], host[offset + 1], host[offset + 2],
                          host[offset + 3], host[offset + 4]};
    if (!std::isfinite(box.x0) || !std::isfinite(box.y0) || !std::isfinite(box.x1) ||
        !std::isfinite(box.y1) || !std::isfinite(box.score)) return false;
    boxes.push_back(box);
  }
  return true;
#else
  (void)probability; (void)probability_offset_elements; (void)height; (void)width;
  (void)image_width; (void)image_height; (void)threshold; (void)box_threshold;
  (void)unclip_ratio; (void)boxes;
  return false;
#endif
}

bool VulkanTensorArena::CtcTop1Into(const VulkanTensorSlot& logits,
                                    VulkanTensorSlot& indices,
                                    VulkanTensorSlot& probabilities,
                                    int batches, int steps, int classes,
                                    bool input_is_logits) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !logits.resident || !indices.resident || !probabilities.resident ||
      batches <= 0 || steps <= 0 || classes <= 0) return false;
  const std::size_t rows = std::size_t(batches) * steps;
  if (indices.live_elements < rows || probabilities.live_elements < rows) return false;
  return Runtime().RunArenaCtcTop1(logits.index, indices.index, probabilities.index, batches,
                                   steps, classes, input_is_logits);
#else
  (void)logits; (void)indices; (void)probabilities; (void)batches; (void)steps; (void)classes;
  (void)input_is_logits;
  return false;
#endif
}

bool VulkanTensorArena::CtcTop1(const VulkanTensorSlot& logits, std::int32_t* indices,
                                float* probabilities, int batches, int steps, int classes,
                                bool input_is_logits) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !logits.resident || !indices || !probabilities || batches <= 0 || steps <= 0 || classes <= 0) return false;
  const std::size_t rows = std::size_t(batches) * steps;
  auto index_slot = Acquire(rows, "ctc-indices");
  auto probability_slot = Acquire(rows, "ctc-probabilities");
  if (!index_slot.resident || !probability_slot.resident ||
      !Runtime().RunArenaCtcTop1(logits.index, index_slot.index, probability_slot.index, batches, steps, classes,
                                 input_is_logits)) {
    Release(probability_slot); Release(index_slot); return false;
  }
  std::vector<float> encoded_indices(rows);
  const bool ok = Download(encoded_indices.data(), index_slot, rows) && Download(probabilities, probability_slot, rows);
  Release(probability_slot); Release(index_slot);
  if (!ok) return false;
  for (std::size_t i = 0; i < rows; ++i) indices[i] = static_cast<std::int32_t>(encoded_indices[i]);
  return true;
#else
  (void)logits; (void)indices; (void)probabilities; (void)batches; (void)steps; (void)classes;
  return false;
#endif
}

bool VulkanTensorArena::Copy(const VulkanTensorSlot& input, VulkanTensorSlot& output,
                             std::size_t elements) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !input.resident || !output.resident || input.index == output.index ||
      elements > input.live_elements || elements > output.live_elements) return false;
  return Runtime().RunArenaCopy(input.index, output.index, elements);
#else
  (void)input; (void)output; (void)elements;
  return false;
#endif
}

bool VulkanTensorArena::BinaryInplace(VulkanTensorSlot& left,
                                      const VulkanTensorSlot& right,
                                      kernels::BinaryOp operation) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !left.resident || !right.resident || left.live_elements != right.live_elements) return false;
  return Runtime().RunArenaBinary(left.index, right.index, left.live_elements, operation);
#else
  (void)left; (void)right; (void)operation;
  return false;
#endif
}

bool VulkanTensorArena::BinaryBroadcast(
    const VulkanTensorSlot& left, const VulkanTensorSlot& right,
    VulkanTensorSlot& output, std::size_t batches, std::size_t right_repeat,
    std::size_t right_per_batch, bool right_shared,
    kernels::BinaryOp operation) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !left.resident || !right.resident || !output.resident || batches == 0 ||
      right_repeat == 0 || right_per_batch == 0 || left.live_elements != output.live_elements ||
      left.live_elements % batches != 0 || left.live_elements / batches != right_repeat * right_per_batch ||
      right.live_elements < (right_shared ? right_per_batch : batches * right_per_batch)) return false;
  return Runtime().RunArenaBinaryBroadcast(left.index, right.index, output.index, batches,
                                           right_repeat, right_per_batch, right_shared, operation);
#else
  (void)left; (void)right; (void)output; (void)batches; (void)right_repeat;
  (void)right_per_batch; (void)right_shared; (void)operation;
  return false;
#endif
}

bool VulkanTensorArena::Concat(const std::vector<VulkanTensorSlot>& inputs,
                               VulkanTensorSlot& output, std::size_t outer,
                               std::size_t inner,
                               const std::vector<std::size_t>& axis_lengths) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !output.resident || inputs.size() < 2 || inputs.size() > 4 ||
      inputs.size() != axis_lengths.size()) return false;
  std::array<std::uint32_t, 4> indices{};
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    if (!inputs[i].resident) return false;
    indices[i] = inputs[i].index;
  }
  return Runtime().RunArenaConcat(indices.data(), inputs.size(), output.index, outer, inner,
                                  axis_lengths.data());
#else
  (void)inputs; (void)output; (void)outer; (void)inner; (void)axis_lengths;
  return false;
#endif
}

bool VulkanTensorArena::NearestResize(const VulkanTensorSlot& input,
                                      VulkanTensorSlot& output, std::size_t batches,
                                      int channels, int input_height, int input_width,
                                      int scale_height, int scale_width) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !input.resident || !output.resident) return false;
  return Runtime().RunArenaNearestResize(input.index, output.index, batches, channels,
                                         input_height, input_width, scale_height, scale_width);
#else
  (void)input; (void)output; (void)batches; (void)channels; (void)input_height; (void)input_width;
  (void)scale_height; (void)scale_width;
  return false;
#endif
}

bool VulkanTensorArena::BinaryChainInplace(
    VulkanTensorSlot& left, const std::vector<VulkanTensorSlot>& rights,
    const std::vector<kernels::BinaryOp>& operations) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !left.resident || rights.empty() || rights.size() != operations.size() || rights.size() > 4) {
    return false;
  }
  std::array<std::uint32_t, 4> indices{};
  for (std::size_t step = 0; step < rights.size(); ++step) {
    if (!rights[step].resident || rights[step].live_elements != left.live_elements) return false;
    indices[step] = rights[step].index;
  }
  return Runtime().RunArenaBinaryChain(left.index, indices.data(), operations.data(),
                                       operations.size(), left.live_elements);
#else
  (void)left; (void)rights; (void)operations;
  return false;
#endif
}

bool VulkanTensorArena::UnaryInplace(VulkanTensorSlot& value, VulkanUnaryOp operation,
                                     float alpha, float beta) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !value.resident) return false;
  return Runtime().RunArenaUnary(value.index, value.live_elements, operation, alpha, beta);
#else
  (void)value; (void)operation; (void)alpha; (void)beta;
  return false;
#endif
}

bool VulkanTensorArena::ChannelAffineInplace(VulkanTensorSlot& value,
                                             const VulkanTensorSlot& scale,
                                             const VulkanTensorSlot& bias,
                                             std::size_t batches, int channels,
                                             std::size_t plane) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !value.resident || !scale.resident || !bias.resident) return false;
  return Runtime().RunArenaChannelAffine(value.index, scale.index, bias.index, batches, channels, plane);
#else
  (void)value; (void)scale; (void)bias; (void)batches; (void)channels; (void)plane;
  return false;
#endif
}

bool VulkanTensorArena::ChannelAffineSwishInplace(VulkanTensorSlot& value,
                                                  const VulkanTensorSlot& scale,
                                                  const VulkanTensorSlot& bias,
                                                  std::size_t batches, int channels,
                                                  std::size_t plane) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !value.resident || !scale.resident || !bias.resident) return false;
  return Runtime().RunArenaChannelAffine(value.index, scale.index, bias.index,
                                         batches, channels, plane, 1);
#else
  (void)value; (void)scale; (void)bias; (void)batches; (void)channels; (void)plane;
  return false;
#endif
}

bool VulkanTensorArena::ChannelAffineHardSwishInplace(VulkanTensorSlot& value,
                                                      const VulkanTensorSlot& scale,
                                                      const VulkanTensorSlot& bias,
                                                      std::size_t batches, int channels,
                                                      std::size_t plane) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !value.resident || !scale.resident || !bias.resident) return false;
  return Runtime().RunArenaChannelAffine(value.index, scale.index, bias.index,
                                         batches, channels, plane, 2);
#else
  (void)value; (void)scale; (void)bias; (void)batches; (void)channels; (void)plane;
  return false;
#endif
}

bool VulkanTensorArena::LayerNormInplace(VulkanTensorSlot& value, const VulkanTensorSlot& gamma,
                                         const VulkanTensorSlot& beta, std::size_t rows,
                                         std::size_t width, float epsilon) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !value.resident || !gamma.resident || !beta.resident) return false;
  return Runtime().RunArenaRowTransform(value.index, gamma.index, beta.index, rows, width, epsilon, true);
#else
  (void)value; (void)gamma; (void)beta; (void)rows; (void)width; (void)epsilon;
  return false;
#endif
}

bool VulkanTensorArena::SoftmaxRowsInplace(VulkanTensorSlot& value, std::size_t rows,
                                           std::size_t width) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !value.resident) return false;
  return Runtime().RunArenaRowTransform(value.index, 0u, 0u, rows, width, 0.F, false);
#else
  (void)value; (void)rows; (void)width;
  return false;
#endif
}

bool VulkanTensorArena::Transpose4D(const VulkanTensorSlot& input, VulkanTensorSlot& output,
                                    const std::array<int, 4>& dimensions,
                                    const std::array<int, 4>& permutation) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !input.resident || !output.resident || input.index == output.index) return false;
  return Runtime().RunArenaTranspose4D(input.index, output.index, dimensions, permutation);
#else
  (void)input; (void)output; (void)dimensions; (void)permutation;
  return false;
#endif
}

bool VulkanTensorArena::SpatialMean(const VulkanTensorSlot& input, VulkanTensorSlot& output,
                                    std::size_t batches, int channels, std::size_t plane) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !input.resident || !output.resident) return false;
  return Runtime().RunArenaSpatialMean(input.index, output.index, batches, channels, plane);
#else
  (void)input; (void)output; (void)batches; (void)channels; (void)plane;
  return false;
#endif
}

bool VulkanTensorArena::SqueezeExcitationGate(
    const VulkanTensorSlot& input, const VulkanTensorSlot& first_weights,
    const VulkanTensorSlot& first_bias, const VulkanTensorSlot& second_weights,
    const VulkanTensorSlot& second_bias, VulkanTensorSlot& output,
    std::size_t batches, int channels, int reduced_channels, std::size_t plane,
    float alpha, float beta) noexcept {
  if (!impl_ || !input.resident || !first_weights.resident || !first_bias.resident ||
      !second_weights.resident || !second_bias.resident || !output.resident ||
      batches == 0 || channels <= 0 || reduced_channels <= 0 || plane == 0 ||
      input.live_elements != batches * std::size_t(channels) * plane ||
      output.live_elements != input.live_elements) return false;
  // The gate's NCHW spatial dimensions are 1x1. Reuse arena free slots for
  // the two compact tensors and guarantee their release on every failure.
  auto mean = Acquire(batches * std::size_t(channels), "se-mean");
  auto reduced = Acquire(batches * std::size_t(reduced_channels), "se-reduced");
  auto gate = Acquire(batches * std::size_t(channels), "se-gate");
  const auto release_all = [&] { Release(gate); Release(reduced); Release(mean); };
  if (!mean.resident || !reduced.resident || !gate.resident ||
      !SpatialMean(input, mean, batches, channels, plane) ||
      !PointwiseConv(mean, first_weights, first_bias, reduced, batches, channels,
                     reduced_channels, 1, true, false) ||
      !PointwiseConv(reduced, second_weights, second_bias, gate, batches, reduced_channels,
                     channels, 1, false, false) ||
      !UnaryInplace(gate, VulkanUnaryOp::hard_sigmoid, alpha, beta) ||
      !BinaryBroadcast(input, gate, output, batches, plane, std::size_t(channels), false,
                       kernels::BinaryOp::mul)) {
    release_all();
    return false;
  }
  release_all();
  return true;
}

bool VulkanTensorArena::QkvSplit(const VulkanTensorSlot& input, VulkanTensorSlot& query,
                                 VulkanTensorSlot& key, VulkanTensorSlot& value,
                                 std::size_t batches, int steps, int heads,
                                 int head_width) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !input.resident || !query.resident || !key.resident || !value.resident) return false;
  return Runtime().RunArenaQkvSplit(input.index, query.index, key.index, value.index,
                                    batches, steps, heads, head_width);
#else
  (void)input; (void)query; (void)key; (void)value; (void)batches; (void)steps; (void)heads; (void)head_width;
  return false;
#endif
}

bool VulkanTensorArena::BatchedGemm(const VulkanTensorSlot& left, const VulkanTensorSlot& right,
                                    VulkanTensorSlot& output, std::size_t matrix_batches,
                                    int rows, int depth, int columns) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !left.resident || !right.resident || !output.resident) return false;
  return Runtime().RunArenaBatchedGemm(left.index, right.index, output.index, matrix_batches,
                                       rows, depth, columns);
#else
  (void)left; (void)right; (void)output; (void)matrix_batches; (void)rows; (void)depth; (void)columns;
  return false;
#endif
}

bool VulkanTensorArena::Pool2d(const VulkanTensorSlot& input, VulkanTensorSlot& output,
                               std::size_t batches, int channels, int input_height, int input_width,
                               int output_height, int output_width, int kernel_height, int kernel_width,
                               int stride_height, int stride_width, int pad_top, int pad_left,
                               bool average) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !input.resident || !output.resident) return false;
  return Runtime().RunArenaPool2d(input.index, output.index, batches, channels, input_height, input_width,
                                  output_height, output_width, kernel_height, kernel_width, stride_height,
                                  stride_width, pad_top, pad_left, average);
#else
  (void)input; (void)output; (void)batches; (void)channels; (void)input_height; (void)input_width;
  (void)output_height; (void)output_width; (void)kernel_height; (void)kernel_width; (void)stride_height;
  (void)stride_width; (void)pad_top; (void)pad_left; (void)average;
  return false;
#endif
}

bool VulkanTensorArena::PointwiseConv(
    const VulkanTensorSlot& input, const VulkanTensorSlot& weights,
    const VulkanTensorSlot& bias, VulkanTensorSlot& output, std::size_t batches,
    int input_channels, int output_channels, std::size_t plane,
    bool relu, bool swish, bool gelu, bool hard_swish) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !input.resident || !weights.resident || !bias.resident || !output.resident ||
      (int(relu) + int(swish) + int(gelu) + int(hard_swish) > 1)) return false;
  return Runtime().RunArenaPointwise(input.index, weights.index, bias.index, output.index,
                                     batches, input_channels, output_channels, plane, relu, swish, gelu,
                                     hard_swish);
#else
  (void)input; (void)weights; (void)bias; (void)output; (void)batches;
  (void)input_channels; (void)output_channels; (void)plane; (void)relu; (void)swish; (void)gelu;
  (void)hard_swish;
  return false;
#endif
}

bool VulkanTensorArena::PointwiseConvTail(
    const VulkanTensorSlot& input, const VulkanTensorSlot& weights,
    const VulkanTensorSlot& bias, VulkanTensorSlot& output, std::size_t batches,
    int input_channels, int output_channels, std::size_t plane) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !input.resident || !weights.resident || !bias.resident || !output.resident) return false;
  return Runtime().RunArenaPointwiseTail(input.index, weights.index, bias.index, output.index,
                                         batches, input_channels, output_channels, plane);
#else
  (void)input; (void)weights; (void)bias; (void)output; (void)batches;
  (void)input_channels; (void)output_channels; (void)plane;
  return false;
#endif
}

bool VulkanTensorArena::PointwiseConvAdd(
    const VulkanTensorSlot& input, const VulkanTensorSlot& weights,
    const VulkanTensorSlot& bias, const VulkanTensorSlot& residual,
    VulkanTensorSlot& output, std::size_t batches, int input_channels,
    int output_channels, std::size_t plane, bool relu, bool swish) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !input.resident || !weights.resident || !bias.resident || !residual.resident ||
      !output.resident || (relu && swish)) return false;
  return Runtime().RunArenaPointwiseAdd(input.index, weights.index, bias.index, residual.index,
                                        output.index, batches, input_channels, output_channels,
                                        plane, relu, swish);
#else
  (void)input; (void)weights; (void)bias; (void)residual; (void)output; (void)batches;
  (void)input_channels; (void)output_channels; (void)plane; (void)relu; (void)swish;
  return false;
#endif
}

bool VulkanTensorArena::ExpandGeluProjectAdd(
    const VulkanTensorSlot& input, const VulkanTensorSlot& expand_weights,
    const VulkanTensorSlot& packed_bias, const VulkanTensorSlot& project_weights,
    VulkanTensorSlot& output, std::size_t batches, int channels, int hidden,
    std::size_t plane) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !input.resident || !expand_weights.resident || !packed_bias.resident ||
      !project_weights.resident || !output.resident) return false;
  return Runtime().RunArenaExpandGeluProjectAdd(
      input.index, expand_weights.index, packed_bias.index, project_weights.index,
      output.index, batches, channels, hidden, plane);
#else
  (void)input; (void)expand_weights; (void)packed_bias; (void)project_weights;
  (void)output; (void)batches; (void)channels; (void)hidden; (void)plane;
  return false;
#endif
}

bool VulkanTensorArena::DepthwisePointwiseFused(
    const VulkanTensorSlot& input, const VulkanTensorSlot& dw_weights,
    const VulkanTensorSlot& packed_bias, const VulkanTensorSlot& pw_weights,
    VulkanTensorSlot& output, std::size_t batches, int channels, int output_channels,
    int height, int width, bool gelu) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !input.resident || !dw_weights.resident || !packed_bias.resident ||
      !pw_weights.resident || !output.resident) return false;
  return Runtime().RunArenaDepthwisePointwiseFused(
      input.index, dw_weights.index, packed_bias.index, pw_weights.index, output.index,
      batches, channels, output_channels, height, width, gelu);
#else
  (void)input; (void)dw_weights; (void)packed_bias; (void)pw_weights; (void)output;
  (void)batches; (void)channels; (void)output_channels; (void)height; (void)width;
  (void)gelu;
  return false;
#endif
}

bool VulkanTensorArena::DepthwiseConv(
    const VulkanTensorSlot& input, const VulkanTensorSlot& weights,
    const VulkanTensorSlot& bias, VulkanTensorSlot& output,
    std::size_t batches, int channels, int input_height, int input_width,
    int output_height, int output_width, int kernel_height, int kernel_width,
    int stride_height, int stride_width, int pad_top, int pad_left,
    bool relu, bool swish, bool hard_swish) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !input.resident || !weights.resident || !bias.resident || !output.resident ||
      (relu && (swish || hard_swish)) || (swish && hard_swish)) return false;
  const int activation_mode = relu ? 7 : (swish ? 8 : (hard_swish ? 9 : 0));
  return Runtime().RunArenaDepthwise(input.index, weights.index, bias.index, output.index,
                                     batches, channels, input_height, input_width, output_height,
                                     output_width, kernel_height, kernel_width, stride_height,
                                     stride_width, pad_top, pad_left, activation_mode);
#else
  (void)input; (void)weights; (void)bias; (void)output; (void)batches; (void)channels;
  (void)input_height; (void)input_width; (void)output_height; (void)output_width;
  (void)kernel_height; (void)kernel_width; (void)stride_height; (void)stride_width;
  (void)pad_top; (void)pad_left; (void)relu; (void)swish; (void)hard_swish;
  return false;
#endif
}

bool VulkanTensorArena::Conv2d(
    const VulkanTensorSlot& input, const VulkanTensorSlot& weights,
    const VulkanTensorSlot& bias, VulkanTensorSlot& output, std::size_t batches,
    int input_channels, int output_channels, int input_height, int input_width,
    int output_height, int output_width, int kernel_height, int kernel_width,
    int stride_height, int stride_width, int pad_top, int pad_left, bool relu,
    bool swish, bool sigmoid, bool hard_swish) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !input.resident || !weights.resident || !bias.resident || !output.resident ||
      (int(relu) + int(swish) + int(sigmoid) + int(hard_swish) > 1)) return false;
  // Shader modes 14/15/26/27/28 are intentionally contiguous only in their
  // public meanings, not numerically. Translate to the compact mode offset.
  const int activation_mode = relu ? 1 : (swish ? 12 : (sigmoid ? 13 : (hard_swish ? 14 : 0)));
  return Runtime().RunArenaConv2d(input.index, weights.index, bias.index, output.index,
                                  batches, input_channels, output_channels, input_height,
                                  input_width, output_height, output_width, kernel_height,
                                  kernel_width, stride_height, stride_width, pad_top, pad_left,
                                  activation_mode);
#else
  (void)input; (void)weights; (void)bias; (void)output; (void)batches; (void)input_channels;
  (void)output_channels; (void)input_height; (void)input_width; (void)output_height;
  (void)output_width; (void)kernel_height; (void)kernel_width; (void)stride_height;
  (void)stride_width; (void)pad_top; (void)pad_left; (void)relu; (void)swish;
  (void)sigmoid; (void)hard_swish;
  return false;
#endif
}

bool VulkanTensorArena::ConvTranspose2x2(
    const VulkanTensorSlot& input, const VulkanTensorSlot& weights,
    const VulkanTensorSlot& bias, VulkanTensorSlot& output, std::size_t batches,
    int input_channels, int output_channels, int input_height, int input_width,
    int activation) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !input.resident || !weights.resident || !bias.resident || !output.resident) return false;
  return Runtime().RunArenaConvTranspose2x2(input.index, weights.index, bias.index, output.index,
                                            batches, input_channels, output_channels,
                                            input_height, input_width, activation);
#else
  (void)input; (void)weights; (void)bias; (void)output; (void)batches;
  (void)input_channels; (void)output_channels; (void)input_height; (void)input_width;
  (void)activation;
  return false;
#endif
}

bool VulkanTensorArena::ConvTranspose2x2Chain(
    const VulkanTensorSlot& input, const VulkanTensorSlot& w0,
    const VulkanTensorSlot& b0, const VulkanTensorSlot& packed_w1_b1,
    VulkanTensorSlot& output, std::size_t batches, int input_channels,
    int mid_channels, int output_channels, int input_height,
    int input_width) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !input.resident || !w0.resident || !b0.resident ||
      !packed_w1_b1.resident || !output.resident) return false;
  return Runtime().RunArenaConvTranspose2x2Chain(
      input.index, w0.index, b0.index, packed_w1_b1.index, output.index, batches,
      input_channels, mid_channels, output_channels, input_height, input_width);
#else
  (void)input; (void)w0; (void)b0; (void)packed_w1_b1; (void)output; (void)batches;
  (void)input_channels; (void)mid_channels; (void)output_channels; (void)input_height;
  (void)input_width;
  return false;
#endif
}

bool VulkanTensorArena::NearestResize2xAdd(
    const VulkanTensorSlot& source, const VulkanTensorSlot& residual,
    VulkanTensorSlot& output, std::size_t batches, int channels, int input_height,
    int input_width) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !source.resident || !residual.resident || !output.resident) return false;
  return Runtime().RunArenaNearestResize2xAdd(source.index, residual.index, output.index,
                                              batches, channels, input_height, input_width);
#else
  (void)source; (void)residual; (void)output; (void)batches; (void)channels;
  (void)input_height; (void)input_width;
  return false;
#endif
}

bool VulkanTensorArena::Gemm(const VulkanTensorSlot& left, const VulkanTensorSlot& right,
                             const VulkanTensorSlot* bias, VulkanTensorSlot& output,
                             int rows, int depth, int columns, bool swish, int a_lda) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !left.resident || !right.resident || !output.resident ||
      (bias != nullptr && !bias->resident)) return false;
  return Runtime().RunArenaGemm(left.index, right.index, bias ? bias->index : 0u, output.index,
                                rows, depth, columns, bias != nullptr, swish, a_lda);
#else
  (void)left; (void)right; (void)bias; (void)output; (void)rows; (void)depth; (void)columns;
  (void)swish; (void)a_lda;
  return false;
#endif
}

bool VulkanTensorArena::GemmCtcTop1(const VulkanTensorSlot& left,
                                    const VulkanTensorSlot& right,
                                    const VulkanTensorSlot* bias,
                                    VulkanTensorSlot& indices,
                                    VulkanTensorSlot& probabilities, int rows,
                                    int depth, int vocab) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !left.resident || !right.resident || !indices.resident ||
      !probabilities.resident || (bias != nullptr && !bias->resident)) return false;
  return Runtime().RunArenaGemmCtcTop1(left.index, right.index, bias ? bias->index : 0u,
                                       indices.index, probabilities.index, rows, depth, vocab,
                                       bias != nullptr);
#else
  (void)left; (void)right; (void)bias; (void)indices; (void)probabilities; (void)rows;
  (void)depth; (void)vocab;
  return false;
#endif
}

void VulkanTensorArena::Release(VulkanTensorSlot& slot) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (impl_ && slot.resident) Runtime().ReleaseArenaSlot(slot.index);
#endif
  slot = {};
}

void VulkanTensorArena::Reset() noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (impl_) Runtime().ResetArenaSlots();
#endif
}

bool VulkanTensorArena::ReclaimFreeStorage() noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return impl_ && Runtime().ReclaimFreeArenaStorage();
#else
  return false;
#endif
}

bool VulkanTensorArena::ReclaimFreeTransientStorage() noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return impl_ && Runtime().ReclaimFreeTransientArenaStorage();
#else
  return false;
#endif
}

bool VulkanTensorArena::available() const noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return impl_ && Runtime().Initialize();
#else
  return false;
#endif
}

std::uint64_t VulkanTensorArena::generation() const noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return impl_ ? Runtime().ArenaGeneration() : 0;
#else
  return 0;
#endif
}

bool VulkanTensorArena::DepthwiseConvScalar(
    const VulkanTensorSlot& input, const VulkanTensorSlot& weights,
    const VulkanTensorSlot& bias, VulkanTensorSlot& output,
    std::size_t batches, int channels, int input_height, int input_width,
    int output_height, int output_width, int kernel_height, int kernel_width,
    int stride_height, int stride_width, int pad_top, int pad_left) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!impl_ || !input.resident || !weights.resident || !bias.resident || !output.resident) return false;
  return Runtime().RunArenaDepthwise(input.index, weights.index, bias.index, output.index,
                                     batches, channels, input_height, input_width, output_height,
                                     output_width, kernel_height, kernel_width, stride_height,
                                     stride_width, pad_top, pad_left, 0, true);
#else
  (void)input; (void)weights; (void)bias; (void)output; (void)batches; (void)channels;
  (void)input_height; (void)input_width; (void)output_height; (void)output_width;
  (void)kernel_height; (void)kernel_width; (void)stride_height; (void)stride_width;
  (void)pad_top; (void)pad_left;
  return false;
#endif
}

void VulkanTensorArena::UnpinForReplay(const VulkanTensorSlot& slot) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (impl_ && slot.resident) Runtime().UnpinArenaSlot(slot.index);
#else
  (void)slot;
#endif
}

void VulkanTensorArena::DropPersistentGraph(std::uint64_t key) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (impl_) Runtime().DropPersistentGraph(key);
#else
  (void)key;
#endif
}

std::size_t VulkanTensorArena::capacity_bytes() const noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return impl_ ? Runtime().ArenaCapacityBytes() : 0;
#else
  return 0;
#endif
}

std::size_t VulkanTensorArena::allocated_bytes() const noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return impl_ ? Runtime().ArenaAllocatedBytes() : 0;
#else
  return 0;
#endif
}

std::size_t VulkanTensorArena::live_bytes() const noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return impl_ ? Runtime().ArenaLiveBytes() : 0;
#else
  return 0;
#endif
}

BackendInfo QueryVulkanBackendInfo() {
  BackendInfo result;
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  auto& runtime = Runtime();
  (void)runtime.Initialize();
  result = runtime.Info();
#elif defined(PPOCR_HAS_VULKAN_HEADERS) && defined(_WIN32)
  HMODULE loader = LoadLibraryA("vulkan-1.dll");
  result.vulkan_loader_available = loader != nullptr;
  if (loader) FreeLibrary(loader);
#endif
  // Every PP-OCRv6 neural operator and the GPU OCR front-end now has a strict
  // The public GPU-only contract is eligible only after Vulkan compute has
  // initialized. Runtime graph execution remains fail-closed: unsupported
  // operators or a device submission failure return an error rather than
  // falling back to CPU activation execution.
  result.full_graph_gpu_available = result.vulkan_compute_available;
  return result;
}

int VulkanLastSubmissionResult() noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return Runtime().LastSubmissionResult();
#else
  return 0;
#endif
}

std::uint64_t VulkanHybridAdmissionContext() noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return Runtime().HybridAdmissionContext();
#else
  return 0;
#endif
}

#if defined(PPOCR_POSIX_VULKAN_LOADER_ADAPTER)
#undef _WIN32
#undef PPOCR_POSIX_VULKAN_LOADER_ADAPTER
#endif

bool VulkanBinary(float* output, const float* left, const float* right,
                  std::size_t count, kernels::BinaryOp operation,
                  bool immutable_right) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return Runtime().Run(output, left, right, count, 1, 1, count, count, count, &operation, 1,
                       immutable_right);
#else
  (void)output; (void)left; (void)right; (void)count; (void)operation; (void)immutable_right;
  return false;
#endif
}

bool VulkanBinaryChain(float* output, const float* left, const float* right,
                       std::size_t count,
                       const std::vector<kernels::BinaryOp>& operations,
                       bool immutable_right) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return Runtime().Run(output, left, right, count, 1, 1, count, count, count,
                       operations.data(), operations.size(), immutable_right);
#else
  (void)output; (void)left; (void)right; (void)count; (void)operations; (void)immutable_right;
  return false;
#endif
}

bool VulkanBinaryChainBatch(float* output, const float* left, const float* right,
                            std::size_t count, std::size_t batches,
                            const std::vector<kernels::BinaryOp>& operations,
                            bool immutable_right) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return Runtime().Run(output, left, right, count, batches, 1, count * batches,
                       count, count,
                       operations.data(), operations.size(), immutable_right);
#else
  (void)output; (void)left; (void)right; (void)count; (void)batches; (void)operations; (void)immutable_right;
  return false;
#endif
}

bool VulkanBinaryBroadcastRightChainBatch(
    float* output, const float* left, const float* right, std::size_t count,
    std::size_t batches, std::size_t right_repeat, std::size_t right_elements,
    const std::vector<kernels::BinaryOp>& operations,
    bool immutable_right) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (operations.empty() || operations.size() > 4 || right_repeat == 0 ||
      right_elements == 0 || count % right_repeat != 0 ||
      count / right_repeat > UINT32_MAX) return false;
  const auto right_per_batch = count / right_repeat;
  // ONNX broadcasts a model constant [1,C,1,1] over all crop batches. Keep
  // that RHS compact and share it in the shader instead of expanding/copying
  // it N times. Dynamic [N,C,1,1] inputs retain a separate compact slice.
  const std::size_t right_batch_stride = right_elements == right_per_batch ? 0 : right_per_batch;
  if ((right_batch_stride == 0 && right_elements != right_per_batch) ||
      (right_batch_stride != 0 &&
       (batches > UINT32_MAX / right_per_batch || right_elements != batches * right_per_batch))) {
    return false;
  }
  return Runtime().Run(output, left, right, count, batches, right_repeat, right_elements,
                       right_per_batch, right_batch_stride,
                       operations.data(), operations.size(), immutable_right);
#else
  (void)output; (void)left; (void)right; (void)count; (void)batches;
  (void)right_repeat; (void)right_elements; (void)operations; (void)immutable_right;
  return false;
#endif
}

bool VulkanChannelAffineBatch(
    float* output, const float* left, const float* scale, const float* bias,
    std::size_t count, std::size_t batches, std::size_t channel_repeat,
    std::size_t coefficient_elements, bool immutable_coefficients) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return Runtime().RunChannelAffine(output, left, scale, bias, count, batches,
                                    channel_repeat, coefficient_elements,
                                    immutable_coefficients, false);
#else
  (void)output; (void)left; (void)scale; (void)bias; (void)count; (void)batches;
  (void)channel_repeat; (void)coefficient_elements; (void)immutable_coefficients;
  return false;
#endif
}

bool VulkanChannelAffineSwishBatch(
    float* output, const float* left, const float* scale, const float* bias,
    std::size_t count, std::size_t batches, std::size_t channel_repeat,
    std::size_t coefficient_elements, bool immutable_coefficients) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return Runtime().RunChannelAffine(output, left, scale, bias, count, batches,
                                    channel_repeat, coefficient_elements,
                                    immutable_coefficients, true);
#else
  (void)output; (void)left; (void)scale; (void)bias; (void)count; (void)batches;
  (void)channel_repeat; (void)coefficient_elements; (void)immutable_coefficients;
  return false;
#endif
}

bool VulkanChannelAffineSwishBinaryBatch(
    float* output, const float* left, const float* scale, const float* bias,
    const float* right, std::size_t count, std::size_t batches,
    std::size_t channel_repeat, std::size_t coefficient_elements,
    kernels::BinaryOp operation, bool immutable_coefficients,
    bool immutable_right) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return Runtime().RunChannelAffineSwishBinary(
      output, left, scale, bias, right, count, batches, channel_repeat,
      coefficient_elements, operation, immutable_coefficients, immutable_right);
#else
  (void)output; (void)left; (void)scale; (void)bias; (void)right; (void)count;
  (void)batches; (void)channel_repeat; (void)coefficient_elements; (void)operation;
  (void)immutable_coefficients; (void)immutable_right;
  return false;
#endif
}

bool VulkanPointwiseConvBatch(
    float* output, const float* input, const float* weights, const float* bias,
    std::size_t batches, int input_channels, int output_channels,
    std::size_t plane, bool immutable_parameters, bool relu, bool swish,
    bool sigmoid, bool hard_sigmoid, float hard_sigmoid_alpha,
    float hard_sigmoid_beta, bool hard_swish) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return Runtime().RunPointwiseConv(output, input, weights, bias, nullptr, batches,
                                    input_channels, output_channels, plane,
                                    immutable_parameters, relu, swish, sigmoid,
                                    hard_sigmoid, hard_sigmoid_alpha, hard_sigmoid_beta,
                                    hard_swish);
#else
  (void)output; (void)input; (void)weights; (void)bias; (void)batches;
  (void)input_channels; (void)output_channels; (void)plane;
  (void)immutable_parameters; (void)relu; (void)swish; (void)sigmoid;
  (void)hard_sigmoid; (void)hard_sigmoid_alpha; (void)hard_sigmoid_beta; (void)hard_swish;
  return false;
#endif
}

bool VulkanPointwiseConvAddBatch(
    float* output, const float* input, const float* weights, const float* bias,
    const float* residual, std::size_t batches, int input_channels,
    int output_channels, std::size_t plane, bool immutable_parameters,
    bool relu, bool swish) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return Runtime().RunPointwiseConvAdd(output, input, weights, bias, residual, batches,
                                       input_channels, output_channels, plane,
                                       immutable_parameters, relu, swish);
#else
  (void)output; (void)input; (void)weights; (void)bias; (void)residual; (void)batches;
  (void)input_channels; (void)output_channels; (void)plane;
  (void)immutable_parameters; (void)relu; (void)swish;
  return false;
#endif
}

bool VulkanDepthwiseConvBatch(
    float* output, const float* input, const float* weights, const float* bias,
    std::size_t batches, int channels, int input_height, int input_width,
    int output_height, int output_width, int kernel_height, int kernel_width,
    int stride_height, int stride_width, int pad_top, int pad_left,
    bool immutable_parameters, bool relu, bool swish, bool hard_swish) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return Runtime().RunDepthwiseConv(output, input, weights, bias, batches, channels,
                                    input_height, input_width, output_height, output_width,
                                    kernel_height, kernel_width, stride_height, stride_width,
                                    pad_top, pad_left, immutable_parameters, relu, swish,
                                    hard_swish);
#else
  (void)output; (void)input; (void)weights; (void)bias; (void)batches; (void)channels;
  (void)input_height; (void)input_width; (void)output_height; (void)output_width;
  (void)kernel_height; (void)kernel_width; (void)stride_height; (void)stride_width;
  (void)pad_top; (void)pad_left; (void)immutable_parameters; (void)relu; (void)swish;
  (void)hard_swish;
  return false;
#endif
}

bool VulkanDepthwisePointwiseConvBatch(
    float* output, const float* input, const float* depthwise_weights,
    const float* depthwise_bias, const float* pointwise_weights,
    const float* pointwise_bias, std::size_t batches, int channels,
    int output_channels, int input_height, int input_width, int output_height,
    int output_width, int kernel_height, int kernel_width, int stride_height,
    int stride_width, int pad_top, int pad_left,
    bool immutable_parameters, bool approximate_gelu) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return Runtime().RunDepthwisePointwiseConv(
      output, input, depthwise_weights, depthwise_bias, pointwise_weights,
      pointwise_bias, batches, channels, output_channels, input_height,
      input_width, output_height, output_width, kernel_height, kernel_width,
      stride_height, stride_width, pad_top, pad_left, immutable_parameters,
      approximate_gelu);
#else
  (void)output; (void)input; (void)depthwise_weights; (void)depthwise_bias;
  (void)pointwise_weights; (void)pointwise_bias; (void)batches; (void)channels;
  (void)output_channels; (void)input_height; (void)input_width;
  (void)output_height; (void)output_width; (void)kernel_height;
  (void)kernel_width; (void)stride_height; (void)stride_width;
  (void)pad_top; (void)pad_left; (void)immutable_parameters; (void)approximate_gelu;
  return false;
#endif
}

bool VulkanConv2dBatch(
    float* output, const float* input, const float* weights, const float* bias,
    std::size_t batches, int input_channels, int output_channels,
    int input_height, int input_width, int output_height, int output_width,
    int kernel_height, int kernel_width, int stride_height, int stride_width,
    int pad_top, int pad_left, bool immutable_parameters, bool relu, bool swish,
    bool sigmoid, bool hard_swish) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return Runtime().RunConv2d(output, input, weights, bias, batches, input_channels,
                             output_channels, input_height, input_width, output_height,
                             output_width, kernel_height, kernel_width, stride_height,
                             stride_width, pad_top, pad_left, immutable_parameters, relu, swish,
                             sigmoid, hard_swish);
#else
  (void)output; (void)input; (void)weights; (void)bias; (void)batches;
  (void)input_channels; (void)output_channels; (void)input_height; (void)input_width;
  (void)output_height; (void)output_width; (void)kernel_height; (void)kernel_width;
  (void)stride_height; (void)stride_width; (void)pad_top; (void)pad_left;
  (void)immutable_parameters; (void)relu; (void)swish; (void)sigmoid; (void)hard_swish;
  return false;
#endif
}

bool VulkanResizeRgbToNchw(
    float* output, const std::uint8_t* rgb, std::size_t rgb_bytes,
    int source_width, int source_height, int output_width, int output_height,
    bool recognition) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!output || !rgb || rgb_bytes == 0 || source_width <= 0 || source_height <= 0 ||
      output_width <= 0 || output_height <= 0 ||
      rgb_bytes < std::size_t(source_width) * source_height * 3 ||
      std::getenv("PPOCR_DISABLE_VULKAN_DET_PREPROCESS") != nullptr) return false;
  VulkanTensorArena arena;
  if (!arena.available()) return false;
  const std::size_t rgb_elements = std::size_t(source_width) * source_height * 3;
  const std::size_t output_elements = std::size_t(3) * output_width * output_height;
  auto rgb_slot = arena.Acquire(rgb_elements, "hyb-rgb");
  auto out_slot = arena.Acquire(output_elements, "hyb-rgb-nchw");
  if (!rgb_slot.resident || !out_slot.resident ||
      !arena.UploadRgb8(rgb_slot, rgb, rgb_bytes) ||
      !arena.ResizeRgbToNchw(rgb_slot, out_slot, source_width, source_height, 0, 0,
                             source_width, source_height, output_width, output_height,
                             1.F / (255.F * .229F), -.485F / .229F,
                             1.F / (255.F * .224F), -.456F / .224F,
                             1.F / (255.F * .225F), -.406F / .225F) ||
      !arena.Download(output, out_slot, output_elements)) {
    arena.Release(out_slot);
    arena.Release(rgb_slot);
    return false;
  }
  (void)recognition;
  arena.Release(out_slot);
  arena.Release(rgb_slot);
  return true;
#else
  (void)output; (void)rgb; (void)rgb_bytes; (void)source_width; (void)source_height;
  (void)output_width; (void)output_height; (void)recognition;
  return false;
#endif
}

bool VulkanResizeRgbAndConv2d(
    float* output, const std::uint8_t* rgb, std::size_t rgb_bytes,
    int source_width, int source_height, int nchw_width, int nchw_height,
    const float* weights, const float* bias, int output_channels,
    int kernel, int stride, int pad, bool relu) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!output || !rgb || !weights || !bias || rgb_bytes == 0 ||
      source_width <= 0 || source_height <= 0 || nchw_width <= 0 || nchw_height <= 0 ||
      output_channels <= 0 || kernel <= 0 || stride <= 0 || pad < 0 ||
      rgb_bytes < std::size_t(source_width) * source_height * 3 ||
      std::getenv("PPOCR_DISABLE_VULKAN_RGB_CONV") != nullptr) return false;
  const int out_h = (nchw_height + 2 * pad - kernel) / stride + 1;
  const int out_w = (nchw_width + 2 * pad - kernel) / stride + 1;
  if (out_h <= 0 || out_w <= 0) return false;
  return Runtime().RunRgbResizeAndConv2d(
      output, rgb, rgb_bytes, source_width, source_height, nchw_width, nchw_height,
      weights, bias, output_channels, kernel, stride, pad, relu);
#else
  (void)output; (void)rgb; (void)rgb_bytes; (void)source_width; (void)source_height;
  (void)nchw_width; (void)nchw_height; (void)weights; (void)bias; (void)output_channels;
  (void)kernel; (void)stride; (void)pad; (void)relu;
  return false;
#endif
}

bool VulkanResizeRgbCropAndConv2d(
    float* output, const std::uint8_t* rgb, std::size_t rgb_bytes,
    int source_width, int source_height, int crop_x, int crop_y, int crop_w,
    int crop_h, int nchw_width, int nchw_height, const float* weights,
    const float* bias, int output_channels, int kernel, int stride, int pad,
    bool relu) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!output || !rgb || !weights || !bias || rgb_bytes == 0 ||
      source_width <= 0 || source_height <= 0 || nchw_width <= 0 || nchw_height <= 0 ||
      crop_w <= 0 || crop_h <= 0 || output_channels <= 0 || kernel <= 0 ||
      stride <= 0 || pad < 0 ||
      rgb_bytes < std::size_t(source_width) * source_height * 3 ||
      std::getenv("PPOCR_DISABLE_VULKAN_REC_RGB_CONV") != nullptr) return false;
  const int out_h = (nchw_height + 2 * pad - kernel) / stride + 1;
  const int out_w = (nchw_width + 2 * pad - kernel) / stride + 1;
  if (out_h <= 0 || out_w <= 0) return false;
  return Runtime().RunRgbResizeAndConv2d(
      output, rgb, rgb_bytes, source_width, source_height, nchw_width, nchw_height,
      weights, bias, output_channels, kernel, stride, pad, relu, crop_x, crop_y,
      crop_w, crop_h, true);
#else
  (void)output; (void)rgb; (void)rgb_bytes; (void)source_width; (void)source_height;
  (void)crop_x; (void)crop_y; (void)crop_w; (void)crop_h; (void)nchw_width;
  (void)nchw_height; (void)weights; (void)bias; (void)output_channels;
  (void)kernel; (void)stride; (void)pad; (void)relu;
  return false;
#endif
}

bool VulkanResizeRgbCropAndConv2dNoSlowerThanCpu(
    int source_width, int source_height, int crop_x, int crop_y, int crop_w,
    int crop_h, int nchw_width, int nchw_height, int output_channels,
    int kernel, int stride, int pad, bool relu, double* gpu_ms, double* cpu_ms) noexcept {
  if (gpu_ms) *gpu_ms = std::numeric_limits<double>::infinity();
  if (cpu_ms) *cpu_ms = std::numeric_limits<double>::infinity();
  if (source_width <= 0 || source_height <= 0 || crop_w <= 0 || crop_h <= 0 ||
      nchw_width <= 0 || nchw_height <= 0 || output_channels <= 0 || kernel != 3 ||
      stride <= 0 || pad < 0) return false;
  try {
    const int out_h = (nchw_height + 2 * pad - kernel) / stride + 1;
    const int out_w = (nchw_width + 2 * pad - kernel) / stride + 1;
    if (out_h <= 0 || out_w <= 0) return false;
    std::vector<std::uint8_t> rgb(std::size_t(source_width) * source_height * 3);
    for (std::size_t i = 0; i < rgb.size(); ++i)
      rgb[i] = static_cast<std::uint8_t>((i * 37u) % 251u);
    const std::size_t weight_n = std::size_t(output_channels) * 3 * kernel * kernel;
    std::vector<float> weights(weight_n), bias(static_cast<std::size_t>(output_channels));
    for (std::size_t i = 0; i < weights.size(); ++i)
      weights[i] = static_cast<float>(i % 29) * .015625F - .1875F;
    for (std::size_t i = 0; i < bias.size(); ++i)
      bias[i] = static_cast<float>(i % 17) * .03125F - .25F;
    std::vector<float> nchw(std::size_t(3) * nchw_width * nchw_height);
    std::vector<float> cpu(std::size_t(output_channels) * out_h * out_w), gpu = cpu;
    constexpr int kRuns = 5;
    if (!VulkanResizeRgbCropAndConv2d(
            gpu.data(), rgb.data(), rgb.size(), source_width, source_height, crop_x, crop_y,
            crop_w, crop_h, nchw_width, nchw_height, weights.data(), bias.data(),
            output_channels, kernel, stride, pad, relu)) {
      return false;
    }
    double cpu_total{};
    double gpu_total{};
    for (int run = 0; run < kRuns; ++run) {
      const auto cpu_begin = std::chrono::steady_clock::now();
      {
        const std::size_t plane = std::size_t(nchw_width) * nchw_height;
        for (int oy = 0; oy < nchw_height; ++oy) for (int ox = 0; ox < nchw_width; ++ox) {
          const float fx = (float(ox) + .5F) * float(crop_w) / float(nchw_width) - .5F;
          const float fy = (float(oy) + .5F) * float(crop_h) / float(nchw_height) - .5F;
          const int x_floor = int(std::floor(fx));
          const int y_floor = int(std::floor(fy));
          const int x0 = std::clamp(x_floor, 0, crop_w - 1);
          const int y0 = std::clamp(y_floor, 0, crop_h - 1);
          const int x1 = std::min(x0 + 1, crop_w - 1);
          const int y1 = std::min(y0 + 1, crop_h - 1);
          const float dx = fx - float(x_floor);
          const float dy = fy - float(y_floor);
          for (int channel = 0; channel < 3; ++channel) {
            const int rgb_c = 2 - channel;
            const auto sample = [&](int x, int y) {
              return float(rgb[(std::size_t(crop_y + y) * source_width + crop_x + x) * 3 + rgb_c]);
            };
            const float upper = sample(x0, y0) * (1.F - dx) + sample(x1, y0) * dx;
            const float lower = sample(x0, y1) * (1.F - dx) + sample(x1, y1) * dx;
            const float vertical = upper * (1.F - dy) + lower * dy;
            const float sampled = std::clamp(float(int(vertical + .5F)), 0.F, 255.F);
            nchw[std::size_t(channel) * plane + std::size_t(oy) * nchw_width + ox] =
                sampled * (2.F / 255.F) - 1.F;
          }
        }
      }
      kernels::Conv2d(cpu.data(), nchw.data(), weights.data(), bias.data(), 3, output_channels,
                      nchw_height, nchw_width, out_h, out_w, kernel, kernel, stride, stride,
                      pad, pad, relu);
      cpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - cpu_begin).count();
      const auto gpu_begin = std::chrono::steady_clock::now();
      if (!VulkanResizeRgbCropAndConv2d(
              gpu.data(), rgb.data(), rgb.size(), source_width, source_height, crop_x, crop_y,
              crop_w, crop_h, nchw_width, nchw_height, weights.data(), bias.data(),
              output_channels, kernel, stride, pad, relu)) {
        return false;
      }
      gpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - gpu_begin).count();
      for (std::size_t i = 0; i < gpu.size(); ++i) {
        const float tolerance = 8e-2F * std::max(1.F, std::abs(cpu[i]));
        if (std::abs(gpu[i] - cpu[i]) > tolerance) return false;
      }
    }
    if (gpu_ms) *gpu_ms = gpu_total / kRuns;
    if (cpu_ms) *cpu_ms = cpu_total / kRuns;
    return gpu_total <= cpu_total;
  } catch (...) {
    return false;
  }
}

bool VulkanResizeRgbAndStem(
    float* output, const std::uint8_t* rgb, std::size_t rgb_bytes,
    int source_width, int source_height, int nchw_width, int nchw_height,
    const VulkanStemLayer& conv0, const VulkanStemLayer& conv1,
    const VulkanStemLayer& conv2, const VulkanStemLayer& stem_conv) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!output || !rgb || rgb_bytes == 0 ||
      std::getenv("PPOCR_DISABLE_VULKAN_RGB_STEM") != nullptr) return false;
  return Runtime().RunRgbStem(output, rgb, rgb_bytes, source_width, source_height,
                              nchw_width, nchw_height, conv0, conv1, conv2, &stem_conv);
#else
  (void)output; (void)rgb; (void)rgb_bytes; (void)source_width; (void)source_height;
  (void)nchw_width; (void)nchw_height; (void)conv0; (void)conv1; (void)conv2; (void)stem_conv;
  return false;
#endif
}

bool VulkanResizeRgbAndConv012(
    float* conv0, float* conv2, const std::uint8_t* rgb, std::size_t rgb_bytes,
    int source_width, int source_height, int nchw_width, int nchw_height,
    const VulkanStemLayer& layer0, const VulkanStemLayer& layer1,
    const VulkanStemLayer& layer2) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!conv0 || !conv2 || !rgb || rgb_bytes == 0 ||
      std::getenv("PPOCR_DISABLE_VULKAN_RGB_CONV012") != nullptr) return false;
  return Runtime().RunRgbStem(conv0, rgb, rgb_bytes, source_width, source_height,
                              nchw_width, nchw_height, layer0, layer1, layer2, nullptr,
                              conv2);
#else
  (void)conv0; (void)conv2; (void)rgb; (void)rgb_bytes; (void)source_width;
  (void)source_height; (void)nchw_width; (void)nchw_height; (void)layer0;
  (void)layer1; (void)layer2;
  return false;
#endif
}

bool VulkanResizeRgbAndConv012NoSlowerThanCpu(
    int source_width, int source_height, int nchw_width, int nchw_height,
    const VulkanStemLayer& layer0, const VulkanStemLayer& layer1,
    const VulkanStemLayer& layer2, double* gpu_ms, double* cpu_ms) noexcept {
  if (gpu_ms) *gpu_ms = std::numeric_limits<double>::infinity();
  if (cpu_ms) *cpu_ms = std::numeric_limits<double>::infinity();
  if (source_width <= 0 || source_height <= 0 || nchw_width <= 0 || nchw_height <= 0 ||
      !layer0.weights || !layer0.bias || !layer1.weights || !layer1.bias ||
      !layer2.weights || !layer2.bias) return false;
  const int c0_h = (nchw_height + 2 * layer0.pad - layer0.kernel) / layer0.stride + 1;
  const int c0_w = (nchw_width + 2 * layer0.pad - layer0.kernel) / layer0.stride + 1;
  if (c0_h <= 0 || c0_w <= 0) return false;
  try {
    const std::size_t rgb_bytes = std::size_t(source_width) * source_height * 3;
    const std::size_t nchw_n = std::size_t(3) * nchw_width * nchw_height;
    const std::size_t c0_n = std::size_t(layer0.output_channels) * c0_h * c0_w;
    const std::size_t c1_n = std::size_t(layer1.output_channels) * c0_h * c0_w;
    const std::size_t c2_n = std::size_t(layer2.output_channels) * c0_h * c0_w;
    std::vector<std::uint8_t> rgb(rgb_bytes);
    for (std::size_t i = 0; i < rgb.size(); ++i)
      rgb[i] = static_cast<std::uint8_t>((i * 37u) % 251u);
    std::vector<float> nchw(nchw_n), t0(c0_n), t1(c1_n), t2(c2_n), g0(c0_n), g2(c2_n);
    constexpr float scales[3]{1.F / (255.F * .229F), 1.F / (255.F * .224F),
                              1.F / (255.F * .225F)};
    constexpr float offsets[3]{-.485F / .229F, -.456F / .224F, -.406F / .225F};
    const auto cpu_resize = [&] {
      const std::size_t plane = std::size_t(nchw_width) * nchw_height;
      for (int oy = 0; oy < nchw_height; ++oy) for (int ox = 0; ox < nchw_width; ++ox) {
        const float fx = (float(ox) + .5F) * float(source_width) / float(nchw_width) - .5F;
        const float fy = (float(oy) + .5F) * float(source_height) / float(nchw_height) - .5F;
        const int x_floor = int(std::floor(fx));
        const int y_floor = int(std::floor(fy));
        const int x0 = std::clamp(x_floor, 0, source_width - 1);
        const int y0 = std::clamp(y_floor, 0, source_height - 1);
        const int x1 = std::min(x0 + 1, source_width - 1);
        const int y1 = std::min(y0 + 1, source_height - 1);
        const float dx = fx - float(x_floor);
        const float dy = fy - float(y_floor);
        for (int channel = 0; channel < 3; ++channel) {
          const int rgb_c = 2 - channel;
          const auto sample = [&](int x, int y) {
            return float(rgb[(std::size_t(y) * source_width + x) * 3 + rgb_c]);
          };
          const float upper = sample(x0, y0) * (1.F - dx) + sample(x1, y0) * dx;
          const float lower = sample(x0, y1) * (1.F - dx) + sample(x1, y1) * dx;
          const float sampled =
              std::clamp(std::floor(upper * (1.F - dy) + lower * dy + .5F), 0.F, 255.F);
          nchw[std::size_t(channel) * plane + std::size_t(oy) * nchw_width + ox] =
              sampled * scales[channel] + offsets[channel];
        }
      }
    };
    if (!VulkanResizeRgbAndConv012(g0.data(), g2.data(), rgb.data(), rgb.size(), source_width,
                                   source_height, nchw_width, nchw_height, layer0, layer1,
                                   layer2)) {
      return false;
    }
    constexpr int kRuns = 3;
    double cpu_total{};
    double gpu_total{};
    for (int run = 0; run < kRuns; ++run) {
      const auto cpu_begin = std::chrono::steady_clock::now();
      cpu_resize();
      kernels::Conv2d(t0.data(), nchw.data(), layer0.weights, layer0.bias, 3,
                      layer0.output_channels, nchw_height, nchw_width, c0_h, c0_w,
                      layer0.kernel, layer0.kernel, layer0.stride, layer0.stride, layer0.pad,
                      layer0.pad, layer0.relu);
      kernels::Conv2d(t1.data(), t0.data(), layer1.weights, layer1.bias, layer1.input_channels,
                      layer1.output_channels, c0_h, c0_w, c0_h, c0_w, layer1.kernel,
                      layer1.kernel, layer1.stride, layer1.stride, layer1.pad, layer1.pad,
                      layer1.relu);
      kernels::Conv2d(t2.data(), t1.data(), layer2.weights, layer2.bias, layer2.input_channels,
                      layer2.output_channels, c0_h, c0_w, c0_h, c0_w, layer2.kernel,
                      layer2.kernel, layer2.stride, layer2.stride, layer2.pad, layer2.pad,
                      layer2.relu);
      cpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - cpu_begin).count();
      const auto gpu_begin = std::chrono::steady_clock::now();
      if (!VulkanResizeRgbAndConv012(g0.data(), g2.data(), rgb.data(), rgb.size(), source_width,
                                     source_height, nchw_width, nchw_height, layer0, layer1,
                                     layer2)) {
        return false;
      }
      gpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - gpu_begin).count();
      for (std::size_t i = 0; i < g0.size(); ++i) {
        if (std::abs(g0[i] - t0[i]) > 8e-2F * std::max(1.F, std::abs(t0[i]))) return false;
      }
      for (std::size_t i = 0; i < g2.size(); ++i) {
        if (std::abs(g2[i] - t2[i]) > 8e-2F * std::max(1.F, std::abs(t2[i]))) return false;
      }
    }
    if (gpu_ms) *gpu_ms = gpu_total / kRuns;
    if (cpu_ms) *cpu_ms = cpu_total / kRuns;
    return gpu_total <= cpu_total;
  } catch (...) {
    return false;
  }
}

bool VulkanResizeRgbAndConv2dNoSlowerThanCpu(
    int source_width, int source_height, int nchw_width, int nchw_height,
    int output_channels, int kernel, int stride, int pad, bool relu,
    double* gpu_ms, double* cpu_ms) noexcept {
  if (gpu_ms) *gpu_ms = std::numeric_limits<double>::infinity();
  if (cpu_ms) *cpu_ms = std::numeric_limits<double>::infinity();
  if (source_width <= 0 || source_height <= 0 || nchw_width <= 0 || nchw_height <= 0 ||
      output_channels <= 0 || kernel <= 0 || stride <= 0 || pad < 0) return false;
  const int out_h = (nchw_height + 2 * pad - kernel) / stride + 1;
  const int out_w = (nchw_width + 2 * pad - kernel) / stride + 1;
  if (out_h <= 0 || out_w <= 0) return false;
  try {
    const std::size_t rgb_bytes = std::size_t(source_width) * source_height * 3;
    const std::size_t nchw_elements = std::size_t(3) * nchw_width * nchw_height;
    const std::size_t weight_elements =
        std::size_t(output_channels) * 3 * kernel * kernel;
    const std::size_t output_elements = std::size_t(output_channels) * out_h * out_w;
    std::vector<std::uint8_t> rgb(rgb_bytes);
    for (std::size_t i = 0; i < rgb.size(); ++i)
      rgb[i] = static_cast<std::uint8_t>((i * 37u) % 251u);
    std::vector<float> weights(weight_elements), bias(static_cast<std::size_t>(output_channels)),
        cpu(output_elements), gpu(output_elements), nchw(nchw_elements);
    for (std::size_t i = 0; i < weights.size(); ++i)
      weights[i] = static_cast<float>(i % 29) * .015625F - .1875F;
    for (std::size_t i = 0; i < bias.size(); ++i)
      bias[i] = static_cast<float>(i % 17) * .03125F - .25F;
    constexpr float scales[3]{1.F / (255.F * .229F), 1.F / (255.F * .224F),
                              1.F / (255.F * .225F)};
    constexpr float offsets[3]{-.485F / .229F, -.456F / .224F, -.406F / .225F};
    const auto cpu_resize = [&] {
      const std::size_t plane = std::size_t(nchw_width) * nchw_height;
      for (int oy = 0; oy < nchw_height; ++oy) for (int ox = 0; ox < nchw_width; ++ox) {
        const float fx = (float(ox) + .5F) * float(source_width) / float(nchw_width) - .5F;
        const float fy = (float(oy) + .5F) * float(source_height) / float(nchw_height) - .5F;
        const int x_floor = int(std::floor(fx));
        const int y_floor = int(std::floor(fy));
        const int x0 = std::clamp(x_floor, 0, source_width - 1);
        const int y0 = std::clamp(y_floor, 0, source_height - 1);
        const int x1 = std::min(x0 + 1, source_width - 1);
        const int y1 = std::min(y0 + 1, source_height - 1);
        const float dx = fx - float(x_floor);
        const float dy = fy - float(y_floor);
        for (int channel = 0; channel < 3; ++channel) {
          const int rgb_c = 2 - channel;
          const auto sample = [&](int x, int y) {
            return float(rgb[(std::size_t(y) * source_width + x) * 3 + rgb_c]);
          };
          const float upper = sample(x0, y0) * (1.F - dx) + sample(x1, y0) * dx;
          const float lower = sample(x0, y1) * (1.F - dx) + sample(x1, y1) * dx;
          const float sampled =
              std::clamp(std::floor(upper * (1.F - dy) + lower * dy + .5F), 0.F, 255.F);
          nchw[std::size_t(channel) * plane + std::size_t(oy) * nchw_width + ox] =
              sampled * scales[channel] + offsets[channel];
        }
      }
    };
    if (!VulkanResizeRgbAndConv2d(gpu.data(), rgb.data(), rgb.size(), source_width,
                                  source_height, nchw_width, nchw_height, weights.data(),
                                  bias.data(), output_channels, kernel, stride, pad, relu)) {
      return false;
    }
    constexpr int kRuns = 5;
    double cpu_total{};
    double gpu_total{};
    for (int run = 0; run < kRuns; ++run) {
      const auto cpu_begin = std::chrono::steady_clock::now();
      cpu_resize();
      kernels::Conv2d(cpu.data(), nchw.data(), weights.data(), bias.data(), 3, output_channels,
                      nchw_height, nchw_width, out_h, out_w, kernel, kernel, stride, stride,
                      pad, pad, relu);
      cpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - cpu_begin).count();
      const auto gpu_begin = std::chrono::steady_clock::now();
      if (!VulkanResizeRgbAndConv2d(gpu.data(), rgb.data(), rgb.size(), source_width,
                                    source_height, nchw_width, nchw_height, weights.data(),
                                    bias.data(), output_channels, kernel, stride, pad, relu)) {
        return false;
      }
      gpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - gpu_begin).count();
      for (std::size_t i = 0; i < gpu.size(); ++i) {
        const float tolerance = 3e-2F * std::max(1.F, std::abs(cpu[i]));
        if (std::abs(gpu[i] - cpu[i]) > tolerance) return false;
      }
    }
    if (gpu_ms) *gpu_ms = gpu_total / kRuns;
    if (cpu_ms) *cpu_ms = cpu_total / kRuns;
    return gpu_total <= cpu_total;
  } catch (...) {
    return false;
  }
}

bool VulkanResizeRgbAndStemNoSlowerThanCpu(
    int source_width, int source_height, int nchw_width, int nchw_height,
    const VulkanStemLayer& conv0, const VulkanStemLayer& conv1,
    const VulkanStemLayer& conv2, const VulkanStemLayer& stem_conv,
    double* gpu_ms, double* cpu_ms) noexcept {
  if (gpu_ms) *gpu_ms = std::numeric_limits<double>::infinity();
  if (cpu_ms) *cpu_ms = std::numeric_limits<double>::infinity();
  if (source_width <= 0 || source_height <= 0 || nchw_width <= 0 || nchw_height <= 0 ||
      !conv0.weights || !conv0.bias || !conv1.weights || !conv1.bias ||
      !conv2.weights || !conv2.bias || !stem_conv.weights || !stem_conv.bias)
    return false;
  const int c0_h = (nchw_height + 2 * conv0.pad - conv0.kernel) / conv0.stride + 1;
  const int c0_w = (nchw_width + 2 * conv0.pad - conv0.kernel) / conv0.stride + 1;
  const int stem_h = (c0_h + 2 * stem_conv.pad - stem_conv.kernel) / stem_conv.stride + 1;
  const int stem_w = (c0_w + 2 * stem_conv.pad - stem_conv.kernel) / stem_conv.stride + 1;
  if (c0_h <= 0 || c0_w <= 0 || stem_h <= 0 || stem_w <= 0) return false;
  try {
    const std::size_t rgb_bytes = std::size_t(source_width) * source_height * 3;
    const std::size_t nchw_n = std::size_t(3) * nchw_width * nchw_height;
    const std::size_t c0_n = std::size_t(conv0.output_channels) * c0_h * c0_w;
    const std::size_t c1_n = std::size_t(conv1.output_channels) * c0_h * c0_w;
    const std::size_t c2_n = std::size_t(conv2.output_channels) * c0_h * c0_w;
    const std::size_t stem_n = std::size_t(stem_conv.output_channels) * stem_h * stem_w;
    std::vector<std::uint8_t> rgb(rgb_bytes);
    for (std::size_t i = 0; i < rgb.size(); ++i)
      rgb[i] = static_cast<std::uint8_t>((i * 37u) % 251u);
    std::vector<float> nchw(nchw_n), t0(c0_n), t1(c1_n), t2(c2_n), pooled(c0_n),
        cpu(stem_n), gpu(stem_n);
    constexpr float scales[3]{1.F / (255.F * .229F), 1.F / (255.F * .224F),
                              1.F / (255.F * .225F)};
    constexpr float offsets[3]{-.485F / .229F, -.456F / .224F, -.406F / .225F};
    const auto cpu_resize = [&] {
      const std::size_t plane = std::size_t(nchw_width) * nchw_height;
      for (int oy = 0; oy < nchw_height; ++oy) for (int ox = 0; ox < nchw_width; ++ox) {
        const float fx = (float(ox) + .5F) * float(source_width) / float(nchw_width) - .5F;
        const float fy = (float(oy) + .5F) * float(source_height) / float(nchw_height) - .5F;
        const int x_floor = int(std::floor(fx));
        const int y_floor = int(std::floor(fy));
        const int x0 = std::clamp(x_floor, 0, source_width - 1);
        const int y0 = std::clamp(y_floor, 0, source_height - 1);
        const int x1 = std::min(x0 + 1, source_width - 1);
        const int y1 = std::min(y0 + 1, source_height - 1);
        const float dx = fx - float(x_floor);
        const float dy = fy - float(y_floor);
        for (int channel = 0; channel < 3; ++channel) {
          const int rgb_c = 2 - channel;
          const auto sample = [&](int x, int y) {
            return float(rgb[(std::size_t(y) * source_width + x) * 3 + rgb_c]);
          };
          const float upper = sample(x0, y0) * (1.F - dx) + sample(x1, y0) * dx;
          const float lower = sample(x0, y1) * (1.F - dx) + sample(x1, y1) * dx;
          const float sampled =
              std::clamp(std::floor(upper * (1.F - dy) + lower * dy + .5F), 0.F, 255.F);
          nchw[std::size_t(channel) * plane + std::size_t(oy) * nchw_width + ox] =
              sampled * scales[channel] + offsets[channel];
        }
      }
    };
    if (!VulkanResizeRgbAndStem(gpu.data(), rgb.data(), rgb.size(), source_width, source_height,
                                nchw_width, nchw_height, conv0, conv1, conv2, stem_conv)) {
      return false;
    }
    constexpr int kRuns = 3;
    double cpu_total{};
    double gpu_total{};
    for (int run = 0; run < kRuns; ++run) {
      const auto cpu_begin = std::chrono::steady_clock::now();
      cpu_resize();
      kernels::Conv2d(t0.data(), nchw.data(), conv0.weights, conv0.bias, 3, conv0.output_channels,
                      nchw_height, nchw_width, c0_h, c0_w, conv0.kernel, conv0.kernel,
                      conv0.stride, conv0.stride, conv0.pad, conv0.pad, conv0.relu);
      kernels::Conv2d(t1.data(), t0.data(), conv1.weights, conv1.bias, conv1.input_channels,
                      conv1.output_channels, c0_h, c0_w, c0_h, c0_w, conv1.kernel, conv1.kernel,
                      conv1.stride, conv1.stride, conv1.pad, conv1.pad, conv1.relu);
      kernels::Conv2d(t2.data(), t1.data(), conv2.weights, conv2.bias, conv2.input_channels,
                      conv2.output_channels, c0_h, c0_w, c0_h, c0_w, conv2.kernel, conv2.kernel,
                      conv2.stride, conv2.stride, conv2.pad, conv2.pad, conv2.relu);
      kernels::MaxPool2x2Same(pooled.data(), t0.data(), std::size_t(conv0.output_channels), c0_h,
                              c0_w);
      const float* srcs[2] = {pooled.data(), t2.data()};
      const int chans[2] = {conv0.output_channels, conv2.output_channels};
      kernels::ConcatChannelConv2d(cpu.data(), srcs, chans, 2, stem_conv.weights, stem_conv.bias,
                                   stem_conv.output_channels, c0_h, c0_w, stem_h, stem_w,
                                   stem_conv.kernel, stem_conv.kernel, stem_conv.stride,
                                   stem_conv.stride, stem_conv.pad, stem_conv.pad,
                                   stem_conv.relu);
      cpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - cpu_begin).count();
      const auto gpu_begin = std::chrono::steady_clock::now();
      if (!VulkanResizeRgbAndStem(gpu.data(), rgb.data(), rgb.size(), source_width, source_height,
                                  nchw_width, nchw_height, conv0, conv1, conv2, stem_conv)) {
        return false;
      }
      gpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - gpu_begin).count();
      for (std::size_t i = 0; i < gpu.size(); ++i) {
        const float tolerance = 8e-2F * std::max(1.F, std::abs(cpu[i]));
        if (std::abs(gpu[i] - cpu[i]) > tolerance) return false;
      }
    }
    if (gpu_ms) *gpu_ms = gpu_total / kRuns;
    if (cpu_ms) *cpu_ms = cpu_total / kRuns;
    return gpu_total <= cpu_total;
  } catch (...) {
    return false;
  }
}

bool VulkanConcatConvBatch(
    float* output, const float* const* sources, const int* source_channels,
    int source_count, const float* weights, const float* bias,
    std::size_t batches, int output_channels, int input_height, int input_width,
    int output_height, int output_width, int kernel_height, int kernel_width,
    int stride_height, int stride_width, int pad_top, int pad_left, bool relu,
    bool pool_first, bool immutable_parameters) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  if (!output || !sources || !source_channels || !weights || !bias ||
      source_count < 2 || source_count > 4 || batches == 0 || output_channels <= 0 ||
      input_height <= 0 || input_width <= 0 || output_height <= 0 || output_width <= 0 ||
      kernel_height <= 0 || kernel_width <= 0 || stride_height <= 0 || stride_width <= 0 ||
      pad_top < 0 || pad_left < 0 ||
      std::getenv("PPOCR_DISABLE_VULKAN_CONCAT_CONV") != nullptr) return false;
  int total_channels = 0;
  for (int i = 0; i < source_count; ++i) {
    if (!sources[i] || source_channels[i] <= 0) return false;
    total_channels += source_channels[i];
  }
  return Runtime().RunConv2d(output, nullptr, weights, bias, batches, total_channels,
                             output_channels, input_height, input_width, output_height,
                             output_width, kernel_height, kernel_width, stride_height,
                             stride_width, pad_top, pad_left, immutable_parameters, relu,
                             false, false, false, sources, source_channels, source_count,
                             pool_first);
#else
  (void)output; (void)sources; (void)source_channels; (void)source_count;
  (void)weights; (void)bias; (void)batches; (void)output_channels;
  (void)input_height; (void)input_width; (void)output_height; (void)output_width;
  (void)kernel_height; (void)kernel_width; (void)stride_height; (void)stride_width;
  (void)pad_top; (void)pad_left; (void)relu; (void)pool_first; (void)immutable_parameters;
  return false;
#endif
}

bool VulkanConcatConvBatchNoSlowerThanCpu(
    const int* source_channels, int source_count, int output_channels,
    int input_height, int input_width, int output_height, int output_width,
    int kernel_height, int kernel_width, int stride_height, int stride_width,
    int pad_top, int pad_left, bool relu, bool pool_first,
    double* gpu_ms, double* cpu_ms, bool immutable_parameters) noexcept {
  if (gpu_ms) *gpu_ms = std::numeric_limits<double>::infinity();
  if (cpu_ms) *cpu_ms = std::numeric_limits<double>::infinity();
  if (!source_channels || source_count < 2 || source_count > 4 || output_channels <= 0 ||
      input_height <= 0 || input_width <= 0 || output_height <= 0 || output_width <= 0 ||
      kernel_height <= 0 || kernel_width <= 0) return false;
  try {
    int total_channels = 0;
    for (int i = 0; i < source_count; ++i) {
      if (source_channels[i] <= 0) return false;
      total_channels += source_channels[i];
    }
    const std::size_t plane = std::size_t(input_height) * input_width;
    const std::size_t output_elements =
        std::size_t(output_channels) * output_height * output_width;
    const std::size_t weight_elements =
        std::size_t(output_channels) * total_channels * kernel_height * kernel_width;
    std::vector<std::vector<float>> host_sources(static_cast<std::size_t>(source_count));
    std::vector<const float*> source_ptrs(static_cast<std::size_t>(source_count));
    for (int i = 0; i < source_count; ++i) {
      host_sources[i].resize(std::size_t(source_channels[i]) * plane);
      for (std::size_t j = 0; j < host_sources[i].size(); ++j)
        host_sources[i][j] = static_cast<float>((j + std::size_t(i) * 17) % 251) * .0078125F - 1.F;
      source_ptrs[i] = host_sources[i].data();
    }
    std::vector<float> weights(weight_elements), bias(static_cast<std::size_t>(output_channels)),
        cpu(output_elements), gpu(output_elements);
    for (std::size_t i = 0; i < weights.size(); ++i)
      weights[i] = static_cast<float>(i % 29) * .015625F - .1875F;
    for (std::size_t i = 0; i < bias.size(); ++i)
      bias[i] = static_cast<float>(i % 17) * .03125F - .25F;
    std::vector<float> pooled;
    std::vector<const float*> cpu_sources = source_ptrs;
    if (pool_first) {
      pooled.resize(host_sources[0].size());
      kernels::MaxPool2x2Same(pooled.data(), host_sources[0].data(),
                              std::size_t(source_channels[0]), input_height, input_width);
      cpu_sources[0] = pooled.data();
    }
    constexpr int kRuns = 5;
    if (!VulkanConcatConvBatch(gpu.data(), source_ptrs.data(), source_channels, source_count,
                               weights.data(), bias.data(), 1, output_channels, input_height,
                               input_width, output_height, output_width, kernel_height,
                               kernel_width, stride_height, stride_width, pad_top, pad_left,
                               relu, pool_first, immutable_parameters)) {
      return false;
    }
    double cpu_total{};
    double gpu_total{};
    for (int run = 0; run < kRuns; ++run) {
      const auto cpu_begin = std::chrono::steady_clock::now();
      if (pool_first) {
        kernels::MaxPool2x2Same(pooled.data(), host_sources[0].data(),
                                std::size_t(source_channels[0]), input_height, input_width);
      }
      kernels::ConcatChannelConv2d(cpu.data(), cpu_sources.data(), source_channels, source_count,
                                   weights.data(), bias.data(), output_channels, input_height,
                                   input_width, output_height, output_width, kernel_height,
                                   kernel_width, stride_height, stride_width, pad_top, pad_left,
                                   relu);
      cpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - cpu_begin).count();
      const auto gpu_begin = std::chrono::steady_clock::now();
      if (!VulkanConcatConvBatch(gpu.data(), source_ptrs.data(), source_channels, source_count,
                                 weights.data(), bias.data(), 1, output_channels, input_height,
                                 input_width, output_height, output_width, kernel_height,
                                 kernel_width, stride_height, stride_width, pad_top, pad_left,
                                 relu, pool_first, immutable_parameters)) {
        return false;
      }
      gpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - gpu_begin).count();
      for (std::size_t i = 0; i < gpu.size(); ++i) {
        const float tolerance = 3e-5F * std::max(1.F, std::abs(cpu[i]));
        if (std::abs(gpu[i] - cpu[i]) > tolerance) return false;
      }
    }
    const double cpu_mean = cpu_total / kRuns;
    const double gpu_mean = gpu_total / kRuns;
    if (gpu_ms) *gpu_ms = gpu_mean;
    if (cpu_ms) *cpu_ms = cpu_mean;
    return gpu_mean <= cpu_mean;
  } catch (...) {
    return false;
  }
}

bool VulkanConvTranspose2x2Batch(
    float* output, const float* input, const float* weights, const float* bias,
    std::size_t batches, int input_channels, int output_channels,
    int input_height, int input_width, bool immutable_parameters) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return Runtime().RunConvTranspose2x2(output, input, weights, bias, batches,
                                       input_channels, output_channels, input_height,
                                       input_width, immutable_parameters);
#else
  (void)output; (void)input; (void)weights; (void)bias; (void)batches;
  (void)input_channels; (void)output_channels; (void)input_height; (void)input_width;
  (void)immutable_parameters;
  return false;
#endif
}

bool VulkanConvTranspose2x2AddBatch(
    float* output, const float* input, const float* weights, const float* bias,
    const float* residual, std::size_t batches, int input_channels,
    int output_channels, int input_height, int input_width,
    bool immutable_parameters) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return Runtime().RunConvTranspose2x2Add(output, input, weights, bias, residual, batches,
                                          input_channels, output_channels, input_height,
                                          input_width, immutable_parameters);
#else
  (void)output; (void)input; (void)weights; (void)bias; (void)residual; (void)batches;
  (void)input_channels; (void)output_channels; (void)input_height; (void)input_width;
  (void)immutable_parameters;
  return false;
#endif
}

bool VulkanNearestResize2xAddBatch(
    float* output, const float* source, const float* residual,
    std::size_t batches, int channels, int input_height, int input_width) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return Runtime().RunNearestResize2xAdd(output, source, residual, batches,
                                         channels, input_height, input_width);
#else
  (void)output; (void)source; (void)residual; (void)batches; (void)channels;
  (void)input_height; (void)input_width;
  return false;
#endif
}

bool VulkanGemm(float* output, const float* left, const float* right, const float* bias,
                int rows, int depth, int columns, bool immutable_parameters) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return Runtime().RunGemm(output, left, right, bias, rows, depth, columns, immutable_parameters, false);
#else
  (void)output; (void)left; (void)right; (void)bias; (void)rows; (void)depth; (void)columns;
  (void)immutable_parameters;
  return false;
#endif
}

bool VulkanGemmSwish(float* output, const float* left, const float* right, const float* bias,
                     int rows, int depth, int columns, bool immutable_parameters) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return Runtime().RunGemm(output, left, right, bias, rows, depth, columns, immutable_parameters, true);
#else
  (void)output; (void)left; (void)right; (void)bias; (void)rows; (void)depth; (void)columns;
  (void)immutable_parameters;
  return false;
#endif
}

bool VulkanGemmCtcTop1(
    int* indices, float* probabilities, const float* left, const float* right,
    const float* bias, int rows, int depth, int vocab, int steps,
    bool immutable_parameters) noexcept {
#if defined(PPOCR_HAS_VULKAN_HEADERS) && defined(PPOCR_HAS_VULKAN_KERNELS) && defined(_WIN32)
  return Runtime().RunGemmCtcTop1(indices, probabilities, left, right, bias, rows, depth, vocab,
                                  steps, immutable_parameters);
#else
  (void)indices; (void)probabilities; (void)left; (void)right; (void)bias;
  (void)rows; (void)depth; (void)vocab; (void)steps; (void)immutable_parameters;
  return false;
#endif
}

bool VulkanGemmCtcTop1NoSlowerThanCpu(
    int rows, int depth, int vocab, double* gpu_ms, double* cpu_ms,
    bool immutable_parameters) noexcept {
  if (gpu_ms) *gpu_ms = std::numeric_limits<double>::infinity();
  if (cpu_ms) *cpu_ms = std::numeric_limits<double>::infinity();
  if (rows <= 0 || depth <= 0 || vocab <= 0) return false;
  try {
    std::vector<float> left(std::size_t(rows) * depth), right(std::size_t(depth) * vocab),
        bias(static_cast<std::size_t>(vocab));
    std::vector<int> cpu_idx(static_cast<std::size_t>(rows)), gpu_idx(cpu_idx.size());
    std::vector<float> cpu_prob(static_cast<std::size_t>(rows)), gpu_prob(cpu_prob.size());
    for (std::size_t i = 0; i < left.size(); ++i)
      left[i] = static_cast<float>(i % 29) * .015625F - .1875F;
    for (std::size_t i = 0; i < right.size(); ++i)
      right[i] = static_cast<float>(i % 17) * .03125F - .25F;
    for (int i = 0; i < vocab; ++i) bias[i] = static_cast<float>(i % 13) * .0625F - .375F;
    if (!VulkanGemmCtcTop1(gpu_idx.data(), gpu_prob.data(), left.data(), right.data(),
                           bias.data(), rows, depth, vocab, rows, immutable_parameters)) {
      return false;
    }
    constexpr int kRuns = 3;
    double cpu_total{};
    double gpu_total{};
    for (int run = 0; run < kRuns; ++run) {
      const auto cpu_begin = std::chrono::steady_clock::now();
      kernels::GemmCtcTop1(cpu_idx.data(), cpu_prob.data(), left.data(), right.data(),
                           bias.data(), rows, depth, vocab, rows);
      cpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - cpu_begin).count();
      const auto gpu_begin = std::chrono::steady_clock::now();
      if (!VulkanGemmCtcTop1(gpu_idx.data(), gpu_prob.data(), left.data(), right.data(),
                             bias.data(), rows, depth, vocab, rows, immutable_parameters)) {
        return false;
      }
      gpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - gpu_begin).count();
      for (int row = 0; row < rows; ++row) {
        if (gpu_idx[row] != cpu_idx[row]) return false;
        const float tolerance = 2e-4F * std::max(1.F, std::abs(cpu_prob[row]));
        if (std::abs(gpu_prob[row] - cpu_prob[row]) > tolerance) return false;
      }
    }
    if (gpu_ms) *gpu_ms = gpu_total / kRuns;
    if (cpu_ms) *cpu_ms = cpu_total / kRuns;
    return gpu_total <= cpu_total;
  } catch (...) {
    return false;
  }
}

bool VulkanBinaryChainNoSlowerThanCpu(
    std::size_t count, const std::vector<kernels::BinaryOp>& operations,
    double* gpu_ms, double* cpu_ms, bool immutable_right) noexcept {
  if (gpu_ms) *gpu_ms = std::numeric_limits<double>::infinity();
  if (cpu_ms) *cpu_ms = std::numeric_limits<double>::infinity();
  if (count == 0 || count > UINT32_MAX || operations.empty() || operations.size() > 4) return false;
  try {
    std::vector<float> left(count), right(count), cpu(count), gpu(count);
    for (std::size_t i = 0; i < count; ++i) {
      left[i] = static_cast<float>(i) * .03125F - 7.F;
      right[i] = static_cast<float>(i % 29) * .125F + .5F;
    }
    constexpr int kRuns = 5;
    double cpu_total{};
    double gpu_total{};
    for (int run = 0; run < kRuns; ++run) {
      const auto cpu_begin = std::chrono::steady_clock::now();
      std::memcpy(cpu.data(), left.data(), count * sizeof(float));
      for (const auto operation : operations) {
        kernels::Binary(cpu.data(), cpu.data(), right.data(), count, operation);
      }
      cpu_total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpu_begin).count();
      const auto gpu_begin = std::chrono::steady_clock::now();
      if (!VulkanBinaryChain(gpu.data(), left.data(), right.data(), count, operations,
                             immutable_right)) return false;
      gpu_total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gpu_begin).count();
      for (std::size_t i = 0; i < count; ++i) {
        const float tolerance = 2e-5F * std::max(1.F, std::abs(cpu[i]));
        if (std::abs(gpu[i] - cpu[i]) > tolerance) return false;
      }
    }
    const double cpu_mean = cpu_total / kRuns;
    const double gpu_mean = gpu_total / kRuns;
    if (gpu_ms) *gpu_ms = gpu_mean;
    if (cpu_ms) *cpu_ms = cpu_mean;
    return gpu_mean <= cpu_mean;
  } catch (...) {
    return false;
  }
}

bool VulkanBinaryChainBatchNoSlowerThanCpu(
    std::size_t count, std::size_t batches,
    const std::vector<kernels::BinaryOp>& operations,
    double* gpu_ms, double* cpu_ms, bool immutable_right) noexcept {
  if (gpu_ms) *gpu_ms = std::numeric_limits<double>::infinity();
  if (cpu_ms) *cpu_ms = std::numeric_limits<double>::infinity();
  if (count == 0 || batches == 0 || count > UINT32_MAX || batches > UINT32_MAX ||
      count > UINT32_MAX / batches || operations.empty() || operations.size() > 4) return false;
  try {
    const auto elements = count * batches;
    std::vector<float> left(elements), right(elements), cpu(elements), gpu(elements);
    for (std::size_t i = 0; i < elements; ++i) {
      left[i] = static_cast<float>(i % count) * .03125F - 7.F + static_cast<float>(i / count) * .125F;
      right[i] = static_cast<float>(i % 29) * .125F + .5F;
    }
    constexpr int kRuns = 5;
    double cpu_total{};
    double gpu_total{};
    for (int run = 0; run < kRuns; ++run) {
      const auto cpu_begin = std::chrono::steady_clock::now();
      std::memcpy(cpu.data(), left.data(), elements * sizeof(float));
      for (const auto operation : operations) {
        kernels::Binary(cpu.data(), cpu.data(), right.data(), elements, operation);
      }
      cpu_total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpu_begin).count();
      const auto gpu_begin = std::chrono::steady_clock::now();
      if (!VulkanBinaryChainBatch(gpu.data(), left.data(), right.data(), count, batches, operations,
                                  immutable_right)) return false;
      gpu_total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gpu_begin).count();
      for (std::size_t i = 0; i < elements; ++i) {
        const float tolerance = 2e-5F * std::max(1.F, std::abs(cpu[i]));
        if (std::abs(gpu[i] - cpu[i]) > tolerance) return false;
      }
    }
    const double cpu_mean = cpu_total / kRuns;
    const double gpu_mean = gpu_total / kRuns;
    if (gpu_ms) *gpu_ms = gpu_mean;
    if (cpu_ms) *cpu_ms = cpu_mean;
    return gpu_mean <= cpu_mean;
  } catch (...) {
    return false;
  }
}

bool VulkanBinaryBroadcastRightChainBatchNoSlowerThanCpu(
    std::size_t count, std::size_t batches, std::size_t right_repeat,
    std::size_t right_elements, const std::vector<kernels::BinaryOp>& operations,
    double* gpu_ms, double* cpu_ms, bool immutable_right) noexcept {
  if (gpu_ms) *gpu_ms = std::numeric_limits<double>::infinity();
  if (cpu_ms) *cpu_ms = std::numeric_limits<double>::infinity();
  if (count == 0 || batches == 0 || right_repeat == 0 || right_elements == 0 ||
      count > UINT32_MAX || batches > UINT32_MAX || count > UINT32_MAX / batches ||
      count % right_repeat != 0 || count / right_repeat > UINT32_MAX ||
      (right_elements != count / right_repeat &&
       (batches > UINT32_MAX / (count / right_repeat) ||
        right_elements != batches * (count / right_repeat))) ||
      operations.empty() || operations.size() > 4) return false;
  try {
    const auto elements = count * batches;
    std::vector<float> left(elements), right(right_elements), cpu(elements), gpu(elements);
    for (std::size_t i = 0; i < elements; ++i) {
      left[i] = static_cast<float>(i % count) * .03125F - 7.F +
                static_cast<float>(i / count) * .125F;
    }
    for (std::size_t i = 0; i < right_elements; ++i) {
      right[i] = static_cast<float>(i % 29) * .125F + .5F;
    }
    constexpr int kRuns = 5;
    double cpu_total{};
    double gpu_total{};
    for (int run = 0; run < kRuns; ++run) {
      const auto cpu_begin = std::chrono::steady_clock::now();
      std::memcpy(cpu.data(), left.data(), elements * sizeof(float));
      for (const auto operation : operations) {
        kernels::BinaryBroadcastRightInplace(cpu.data(), right.data(), elements,
                                             right_repeat, right_elements, operation);
      }
      cpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - cpu_begin).count();
      const auto gpu_begin = std::chrono::steady_clock::now();
      if (!VulkanBinaryBroadcastRightChainBatch(gpu.data(), left.data(), right.data(), count,
                                                batches, right_repeat, right_elements,
                                                operations, immutable_right)) return false;
      gpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - gpu_begin).count();
      for (std::size_t i = 0; i < elements; ++i) {
        const float tolerance = 2e-5F * std::max(1.F, std::abs(cpu[i]));
        if (std::abs(gpu[i] - cpu[i]) > tolerance) return false;
      }
    }
    const double cpu_mean = cpu_total / kRuns;
    const double gpu_mean = gpu_total / kRuns;
    if (gpu_ms) *gpu_ms = gpu_mean;
    if (cpu_ms) *cpu_ms = cpu_mean;
    return gpu_mean <= cpu_mean;
  } catch (...) {
    return false;
  }
}

bool VulkanChannelAffineBatchNoSlowerThanCpu(
    std::size_t count, std::size_t batches, std::size_t channel_repeat,
    std::size_t coefficient_elements, double* gpu_ms, double* cpu_ms,
    bool immutable_coefficients) noexcept {
  if (gpu_ms) *gpu_ms = std::numeric_limits<double>::infinity();
  if (cpu_ms) *cpu_ms = std::numeric_limits<double>::infinity();
  if (count == 0 || batches == 0 || channel_repeat == 0 || coefficient_elements == 0 ||
      count > UINT32_MAX || batches > UINT32_MAX || count > UINT32_MAX / batches ||
      count % channel_repeat != 0) return false;
  const auto coefficients_per_batch = count / channel_repeat;
  if (coefficient_elements != coefficients_per_batch &&
      (batches > UINT32_MAX / coefficients_per_batch ||
       coefficient_elements != batches * coefficients_per_batch)) return false;
  try {
    // Admission deliberately includes a fresh parameter upload. A synthetic
    // probe vector has no stable model-lifetime identity, regardless of the
    // caller's production cacheability flag.
    (void)immutable_coefficients;
    const auto elements = count * batches;
    std::vector<float> left(elements), scale(coefficient_elements), bias(coefficient_elements),
        cpu(elements), gpu(elements);
    for (std::size_t i = 0; i < elements; ++i) {
      left[i] = static_cast<float>(i % count) * .03125F - 7.F +
                static_cast<float>(i / count) * .125F;
    }
    for (std::size_t i = 0; i < coefficient_elements; ++i) {
      scale[i] = static_cast<float>(i % 29) * .03125F + .5F;
      bias[i] = static_cast<float>(i % 23) * .015625F - .25F;
    }
    constexpr int kRuns = 5;
    double cpu_total{};
    double gpu_total{};
    for (int run = 0; run < kRuns; ++run) {
      const auto cpu_begin = std::chrono::steady_clock::now();
      std::memcpy(cpu.data(), left.data(), elements * sizeof(float));
      kernels::BinaryBroadcastRightInplace(cpu.data(), scale.data(), elements,
                                           channel_repeat, coefficient_elements,
                                           kernels::BinaryOp::mul);
      kernels::BinaryBroadcastRightInplace(cpu.data(), bias.data(), elements,
                                           channel_repeat, coefficient_elements,
                                           kernels::BinaryOp::add);
      cpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - cpu_begin).count();
      const auto gpu_begin = std::chrono::steady_clock::now();
      if (!VulkanChannelAffineBatch(gpu.data(), left.data(), scale.data(), bias.data(), count,
                                    batches, channel_repeat, coefficient_elements,
                                    false)) return false;
      gpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - gpu_begin).count();
      for (std::size_t i = 0; i < elements; ++i) {
        const float tolerance = 2e-5F * std::max(1.F, std::abs(cpu[i]));
        if (std::abs(gpu[i] - cpu[i]) > tolerance) return false;
      }
    }
    const double cpu_mean = cpu_total / kRuns;
    const double gpu_mean = gpu_total / kRuns;
    if (gpu_ms) *gpu_ms = gpu_mean;
    if (cpu_ms) *cpu_ms = cpu_mean;
    return gpu_mean <= cpu_mean;
  } catch (...) {
    return false;
  }
}

bool VulkanChannelAffineSwishBatchNoSlowerThanCpu(
    std::size_t count, std::size_t batches, std::size_t channel_repeat,
    std::size_t coefficient_elements, double* gpu_ms, double* cpu_ms,
    bool immutable_coefficients) noexcept {
  if (gpu_ms) *gpu_ms = std::numeric_limits<double>::infinity();
  if (cpu_ms) *cpu_ms = std::numeric_limits<double>::infinity();
  if (count == 0 || batches == 0 || channel_repeat == 0 || coefficient_elements == 0 ||
      count > UINT32_MAX || batches > UINT32_MAX || count > UINT32_MAX / batches ||
      count % channel_repeat != 0) return false;
  const auto coefficients_per_batch = count / channel_repeat;
  if (coefficient_elements != coefficients_per_batch &&
      (batches > UINT32_MAX / coefficients_per_batch ||
       coefficient_elements != batches * coefficients_per_batch)) return false;
  try {
    (void)immutable_coefficients;
    const auto elements = count * batches;
    std::vector<float> left(elements), scale(coefficient_elements), bias(coefficient_elements),
        cpu(elements), gpu(elements);
    for (std::size_t i = 0; i < elements; ++i) {
      left[i] = static_cast<float>(i % count) * .03125F - 7.F +
                static_cast<float>(i / count) * .125F;
    }
    for (std::size_t i = 0; i < coefficient_elements; ++i) {
      scale[i] = static_cast<float>(i % 29) * .03125F + .5F;
      bias[i] = static_cast<float>(i % 23) * .015625F - .25F;
    }
    constexpr int kRuns = 5;
    double cpu_total{};
    double gpu_total{};
    for (int run = 0; run < kRuns; ++run) {
      const auto cpu_begin = std::chrono::steady_clock::now();
      std::memcpy(cpu.data(), left.data(), elements * sizeof(float));
      kernels::BatchNormSwish(cpu.data(), cpu.data(), scale.data(), bias.data(),
                              static_cast<int>(batches),
                              static_cast<int>(coefficients_per_batch), channel_repeat);
      cpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - cpu_begin).count();
      const auto gpu_begin = std::chrono::steady_clock::now();
      if (!VulkanChannelAffineSwishBatch(gpu.data(), left.data(), scale.data(), bias.data(),
                                          count, batches, channel_repeat,
                                          coefficient_elements,
                                          false)) return false;
      gpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - gpu_begin).count();
      for (std::size_t i = 0; i < elements; ++i) {
        const float tolerance = 2e-5F * std::max(1.F, std::abs(cpu[i]));
        if (std::abs(gpu[i] - cpu[i]) > tolerance) return false;
      }
    }
    const double cpu_mean = cpu_total / kRuns;
    const double gpu_mean = gpu_total / kRuns;
    if (gpu_ms) *gpu_ms = gpu_mean;
    if (cpu_ms) *cpu_ms = cpu_mean;
    return gpu_mean <= cpu_mean;
  } catch (...) {
    return false;
  }
}

bool VulkanPointwiseConvBatchNoSlowerThanCpu(
    std::size_t batches, int input_channels, int output_channels,
    std::size_t plane, double* gpu_ms, double* cpu_ms,
    bool immutable_parameters, bool relu, bool swish, bool sigmoid,
    bool hard_sigmoid, float hard_sigmoid_alpha, float hard_sigmoid_beta,
    bool hard_swish) noexcept {
  if (gpu_ms) *gpu_ms = std::numeric_limits<double>::infinity();
  if (cpu_ms) *cpu_ms = std::numeric_limits<double>::infinity();
  if (batches == 0 || input_channels <= 0 || output_channels <= 0 || plane == 0 ||
      batches > UINT32_MAX || plane > UINT32_MAX ||
      plane > UINT32_MAX / static_cast<std::size_t>(input_channels) ||
      plane > UINT32_MAX / static_cast<std::size_t>(output_channels) ||
      (int(relu) + int(swish) + int(sigmoid) + int(hard_sigmoid) + int(hard_swish) > 1)) return false;
  try {
    (void)immutable_parameters;
    const auto inputs_per_batch = static_cast<std::size_t>(input_channels) * plane;
    const auto outputs_per_batch = static_cast<std::size_t>(output_channels) * plane;
    if (batches > std::numeric_limits<std::size_t>::max() / inputs_per_batch ||
        batches > std::numeric_limits<std::size_t>::max() / outputs_per_batch) return false;
    const auto input_elements = batches * inputs_per_batch;
    const auto output_elements = batches * outputs_per_batch;
    std::vector<float> input(input_elements), weights(
        static_cast<std::size_t>(input_channels) * output_channels), bias(output_channels),
        cpu(output_elements), gpu(output_elements);
    for (std::size_t i = 0; i < input.size(); ++i) {
      input[i] = static_cast<float>(i % 251) * .0078125F - 1.F;
    }
    for (std::size_t i = 0; i < weights.size(); ++i) {
      weights[i] = static_cast<float>(i % 29) * .015625F - .1875F;
    }
    for (int i = 0; i < output_channels; ++i) bias[static_cast<std::size_t>(i)] =
        static_cast<float>(i % 17) * .03125F - .25F;
    constexpr int kRuns = 5;
    double cpu_total{};
    double gpu_total{};
    for (int run = 0; run < kRuns; ++run) {
      const auto cpu_begin = std::chrono::steady_clock::now();
      // Admission must compare against the same flattened NCHW CPU batch
      // kernels used by the executor.  A serial per-image reference can
      // incorrectly admit a GPU segment that loses once AVX/NEON work is
      // shared across the actual OCR batch.
      if (relu) {
        kernels::PointwiseConvReluBatch(cpu.data(), input.data(), weights.data(), bias.data(),
                                        static_cast<int>(batches), output_channels,
                                        input_channels, plane);
      } else {
        kernels::PointwiseConvBatch(cpu.data(), input.data(), weights.data(), bias.data(),
                                   static_cast<int>(batches), output_channels,
                                   input_channels, plane);
      }
      if (swish) kernels::Swish(cpu.data(), cpu.data(), output_elements);
      if (sigmoid) kernels::Sigmoid(cpu.data(), cpu.data(), output_elements);
      if (hard_sigmoid) kernels::HardSigmoid(cpu.data(), cpu.data(), output_elements,
                                             hard_sigmoid_alpha, hard_sigmoid_beta);
      if (hard_swish) kernels::HardSwish(cpu.data(), cpu.data(), output_elements);
      cpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - cpu_begin).count();
      const auto gpu_begin = std::chrono::steady_clock::now();
      // Probe must write model parameters freshly: a recycled vector address
      // cannot safely identify a long-lived initializer cache entry.
      if (!VulkanPointwiseConvBatch(gpu.data(), input.data(), weights.data(), bias.data(),
                                    batches, input_channels, output_channels, plane,
                                    false, relu, swish, sigmoid, hard_sigmoid,
                                    hard_sigmoid_alpha, hard_sigmoid_beta,
                                    hard_swish)) return false;
      gpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - gpu_begin).count();
      for (std::size_t i = 0; i < output_elements; ++i) {
        const float tolerance = 3e-5F * std::max(1.F, std::abs(cpu[i]));
        if (std::abs(gpu[i] - cpu[i]) > tolerance) return false;
      }
    }
    const double cpu_mean = cpu_total / kRuns;
    const double gpu_mean = gpu_total / kRuns;
    if (gpu_ms) *gpu_ms = gpu_mean;
    if (cpu_ms) *cpu_ms = cpu_mean;
    return gpu_mean <= cpu_mean;
  } catch (...) {
    return false;
  }
}

bool VulkanPointwiseConvAddBatchNoSlowerThanCpu(
    std::size_t batches, int input_channels, int output_channels,
    std::size_t plane, double* gpu_ms, double* cpu_ms,
    bool immutable_parameters, bool relu, bool swish) noexcept {
  if (gpu_ms) *gpu_ms = std::numeric_limits<double>::infinity();
  if (cpu_ms) *cpu_ms = std::numeric_limits<double>::infinity();
  if (batches == 0 || input_channels <= 0 || output_channels <= 0 || plane == 0 ||
      batches > UINT32_MAX || plane > UINT32_MAX ||
      plane > UINT32_MAX / static_cast<std::size_t>(input_channels) ||
      plane > UINT32_MAX / static_cast<std::size_t>(output_channels) ||
      (relu && swish)) return false;
  try {
    (void)immutable_parameters;
    const auto inputs_per_batch = static_cast<std::size_t>(input_channels) * plane;
    const auto outputs_per_batch = static_cast<std::size_t>(output_channels) * plane;
    if (batches > std::numeric_limits<std::size_t>::max() / inputs_per_batch ||
        batches > std::numeric_limits<std::size_t>::max() / outputs_per_batch) return false;
    const auto input_elements = batches * inputs_per_batch;
    const auto output_elements = batches * outputs_per_batch;
    std::vector<float> input(input_elements), weights(
        static_cast<std::size_t>(input_channels) * output_channels), bias(output_channels),
        residual(output_elements), cpu(output_elements), gpu(output_elements);
    for (std::size_t i = 0; i < input.size(); ++i) {
      input[i] = static_cast<float>(i % 251) * .0078125F - 1.F;
    }
    for (std::size_t i = 0; i < weights.size(); ++i) {
      weights[i] = static_cast<float>(i % 29) * .015625F - .1875F;
    }
    for (int i = 0; i < output_channels; ++i) bias[static_cast<std::size_t>(i)] =
        static_cast<float>(i % 17) * .03125F - .25F;
    for (std::size_t i = 0; i < residual.size(); ++i) {
      residual[i] = static_cast<float>(i % 101) * .015625F - .75F;
    }
    constexpr int kRuns = 5;
    double cpu_total{};
    double gpu_total{};
    for (int run = 0; run < kRuns; ++run) {
      const auto cpu_begin = std::chrono::steady_clock::now();
      // Keep the no-slower policy honest for real page/crop batches: compare
      // the GPU boundary with the executor's batched residual kernel, not a
      // weaker loop of individually scheduled convolutions.
      if (relu) {
        kernels::PointwiseConvAddReluBatch(
            cpu.data(), input.data(), weights.data(), bias.data(), residual.data(),
            static_cast<int>(batches), output_channels, input_channels, plane);
      } else if (swish) {
        kernels::PointwiseConvAddSwishBatch(
            cpu.data(), input.data(), weights.data(), bias.data(), residual.data(),
            static_cast<int>(batches), output_channels, input_channels, plane);
      } else {
        kernels::PointwiseConvAddBatch(
            cpu.data(), input.data(), weights.data(), bias.data(), residual.data(),
            static_cast<int>(batches), output_channels, input_channels, plane);
      }
      cpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - cpu_begin).count();
      const auto gpu_begin = std::chrono::steady_clock::now();
      // As with the non-residual probe, use fresh parameter uploads here. A
      // vector address is not a model-initializer identity across probes.
      if (!VulkanPointwiseConvAddBatch(gpu.data(), input.data(), weights.data(), bias.data(),
                                        residual.data(), batches, input_channels, output_channels,
                                        plane, false, relu, swish)) return false;
      gpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - gpu_begin).count();
      for (std::size_t i = 0; i < output_elements; ++i) {
        const float tolerance = 3e-5F * std::max(1.F, std::abs(cpu[i]));
        if (std::abs(gpu[i] - cpu[i]) > tolerance) return false;
      }
    }
    const double cpu_mean = cpu_total / kRuns;
    const double gpu_mean = gpu_total / kRuns;
    if (gpu_ms) *gpu_ms = gpu_mean;
    if (cpu_ms) *cpu_ms = cpu_mean;
    return gpu_mean <= cpu_mean;
  } catch (...) {
    return false;
  }
}

bool VulkanDepthwiseConvBatchNoSlowerThanCpu(
    std::size_t batches, int channels, int input_height, int input_width,
    int output_height, int output_width, int kernel_height, int kernel_width,
    int stride_height, int stride_width, int pad_top, int pad_left,
    double* gpu_ms, double* cpu_ms, bool immutable_parameters, bool relu,
    bool swish, bool hard_swish) noexcept {
  if (gpu_ms) *gpu_ms = std::numeric_limits<double>::infinity();
  if (cpu_ms) *cpu_ms = std::numeric_limits<double>::infinity();
  if (batches == 0 || channels <= 0 || input_height <= 0 || input_width <= 0 ||
      output_height <= 0 || output_width <= 0 || kernel_height <= 0 || kernel_width <= 0 ||
      stride_height <= 0 || stride_width <= 0 || pad_top < 0 || pad_left < 0 ||
      batches > UINT32_MAX || (int(relu) + int(swish) + int(hard_swish) > 1)) return false;
  try {
    const auto c = static_cast<std::size_t>(channels);
    const auto input_plane = static_cast<std::size_t>(input_height) * input_width;
    const auto output_plane = static_cast<std::size_t>(output_height) * output_width;
    const auto kernel_plane = static_cast<std::size_t>(kernel_height) * kernel_width;
    if (input_plane == 0 || output_plane == 0 || kernel_plane == 0 ||
        c > std::numeric_limits<std::size_t>::max() / input_plane ||
        c > std::numeric_limits<std::size_t>::max() / output_plane ||
        c > std::numeric_limits<std::size_t>::max() / kernel_plane) return false;
    const auto input_per_batch = c * input_plane;
    const auto output_per_batch = c * output_plane;
    if (batches > std::numeric_limits<std::size_t>::max() / input_per_batch ||
        batches > std::numeric_limits<std::size_t>::max() / output_per_batch) return false;
    const auto input_elements = batches * input_per_batch;
    const auto output_elements = batches * output_per_batch;
    std::vector<float> input(input_elements), weights(c * kernel_plane), bias(c),
        cpu(output_elements), gpu(output_elements);
    for (std::size_t i = 0; i < input.size(); ++i) {
      input[i] = static_cast<float>(i % 251) * .0078125F - 1.F;
    }
    for (std::size_t i = 0; i < weights.size(); ++i) {
      weights[i] = static_cast<float>(i % 29) * .015625F - .1875F;
    }
    for (std::size_t i = 0; i < bias.size(); ++i) {
      bias[i] = static_cast<float>(i % 17) * .03125F - .25F;
    }
    constexpr int kRuns = 5;
    double cpu_total{};
    double gpu_total{};
    for (int run = 0; run < kRuns; ++run) {
      const auto cpu_begin = std::chrono::steady_clock::now();
      // Compare with the exact batched CPU route used by the executor rather
      // than a serial loop over images. Otherwise a GPU segment could be
      // admitted against a weaker reference even though SIMD batch scheduling
      // would win in the real OCR call.
      kernels::DepthwiseConvBatch(cpu.data(), input.data(), weights.data(), bias.data(),
                                  static_cast<int>(batches), channels, input_height, input_width,
                                  output_height, output_width, kernel_height, kernel_width,
                                  stride_height, stride_width, pad_top, pad_left);
      if (relu) kernels::Relu(cpu.data(), cpu.data(), output_elements);
      if (swish) kernels::Swish(cpu.data(), cpu.data(), output_elements);
      if (hard_swish) kernels::HardSwish(cpu.data(), cpu.data(), output_elements);
      cpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - cpu_begin).count();
      const auto gpu_begin = std::chrono::steady_clock::now();
      if (!VulkanDepthwiseConvBatch(
              gpu.data(), input.data(), weights.data(), bias.data(), batches, channels,
              input_height, input_width, output_height, output_width,
              kernel_height, kernel_width, stride_height, stride_width,
              pad_top, pad_left, false, relu, swish, hard_swish)) return false;
      gpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - gpu_begin).count();
      for (std::size_t i = 0; i < output_elements; ++i) {
        const float tolerance = 3e-5F * std::max(1.F, std::abs(cpu[i]));
        if (std::abs(gpu[i] - cpu[i]) > tolerance) return false;
      }
    }
    (void)immutable_parameters;
    const double cpu_mean = cpu_total / kRuns;
    const double gpu_mean = gpu_total / kRuns;
    if (gpu_ms) *gpu_ms = gpu_mean;
    if (cpu_ms) *cpu_ms = cpu_mean;
    return gpu_mean <= cpu_mean;
  } catch (...) {
    return false;
  }
}

bool VulkanDepthwisePointwiseConvBatchNoSlowerThanCpu(
    std::size_t batches, int channels, int output_channels,
    int input_height, int input_width, int output_height, int output_width,
    int kernel_height, int kernel_width, int stride_height, int stride_width,
    int pad_top, int pad_left, double* gpu_ms, double* cpu_ms,
    bool immutable_parameters, bool approximate_gelu) noexcept {
  if (gpu_ms) *gpu_ms = std::numeric_limits<double>::infinity();
  if (cpu_ms) *cpu_ms = std::numeric_limits<double>::infinity();
  if (batches == 0 || channels <= 0 || output_channels <= 0 ||
      input_height <= 0 || input_width <= 0 || output_height <= 0 ||
      output_width <= 0 || kernel_height <= 0 || kernel_width <= 0 ||
      stride_height <= 0 || stride_width <= 0 || pad_top < 0 || pad_left < 0 ||
      batches > UINT32_MAX) return false;
  try {
    const auto c = static_cast<std::size_t>(channels);
    const auto m = static_cast<std::size_t>(output_channels);
    const auto input_plane = static_cast<std::size_t>(input_height) * input_width;
    const auto output_plane = static_cast<std::size_t>(output_height) * output_width;
    const auto kernel_plane = static_cast<std::size_t>(kernel_height) * kernel_width;
    if (input_plane == 0 || output_plane == 0 || kernel_plane == 0 ||
        c > std::numeric_limits<std::size_t>::max() / input_plane ||
        c > std::numeric_limits<std::size_t>::max() / output_plane ||
        c > std::numeric_limits<std::size_t>::max() / kernel_plane ||
        m > std::numeric_limits<std::size_t>::max() / c ||
        m > std::numeric_limits<std::size_t>::max() / output_plane) return false;
    const auto input_per_batch = c * input_plane;
    const auto intermediate_per_batch = c * output_plane;
    const auto output_per_batch = m * output_plane;
    if (batches > std::numeric_limits<std::size_t>::max() / input_per_batch ||
        batches > std::numeric_limits<std::size_t>::max() / intermediate_per_batch ||
        batches > std::numeric_limits<std::size_t>::max() / output_per_batch) return false;
    const auto input_elements = batches * input_per_batch;
    const auto intermediate_elements = batches * intermediate_per_batch;
    const auto output_elements = batches * output_per_batch;
    std::vector<float> input(input_elements), depthwise_weights(c * kernel_plane),
        depthwise_bias(c), pointwise_weights(m * c), pointwise_bias(m),
        intermediate(intermediate_elements), cpu(output_elements), gpu(output_elements);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(i % 251) * .0078125F - 1.F;
    for (std::size_t i = 0; i < depthwise_weights.size(); ++i) {
      depthwise_weights[i] = float(i % 29) * .015625F - .1875F;
    }
    for (std::size_t i = 0; i < pointwise_weights.size(); ++i) {
      pointwise_weights[i] = float(i % 31) * .015625F - .21875F;
    }
    for (std::size_t i = 0; i < c; ++i) depthwise_bias[i] = float(i % 17) * .03125F - .25F;
    for (std::size_t i = 0; i < m; ++i) pointwise_bias[i] = float(i % 13) * .03125F - .1875F;
    constexpr int kRuns = 5;
    double cpu_total{};
    double gpu_total{};
    for (int run = 0; run < kRuns; ++run) {
      const auto cpu_begin = std::chrono::steady_clock::now();
      kernels::DepthwiseConvBatch(intermediate.data(), input.data(),
                                  depthwise_weights.data(), depthwise_bias.data(),
                                  static_cast<int>(batches), channels, input_height,
                                  input_width, output_height, output_width,
                                  kernel_height, kernel_width, stride_height,
                                  stride_width, pad_top, pad_left);
      kernels::PointwiseConvBatch(cpu.data(), intermediate.data(), pointwise_weights.data(),
                                  pointwise_bias.data(), static_cast<int>(batches),
                                  output_channels, channels, output_plane);
      if (approximate_gelu) {
        kernels::Gelu(cpu.data(), cpu.data(), cpu.size(), output_per_batch);
      }
      cpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - cpu_begin).count();
      const auto gpu_begin = std::chrono::steady_clock::now();
      if (!VulkanDepthwisePointwiseConvBatch(
              gpu.data(), input.data(), depthwise_weights.data(), depthwise_bias.data(),
              pointwise_weights.data(), pointwise_bias.data(), batches, channels,
              output_channels, input_height, input_width, output_height, output_width,
              kernel_height, kernel_width, stride_height, stride_width, pad_top,
              pad_left, false, approximate_gelu)) return false;
      gpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - gpu_begin).count();
      for (std::size_t i = 0; i < output_elements; ++i) {
        const float tolerance = 4e-5F * std::max(1.F, std::abs(cpu[i]));
        if (std::abs(gpu[i] - cpu[i]) > tolerance) return false;
      }
    }
    (void)immutable_parameters;
    const double cpu_mean = cpu_total / kRuns;
    const double gpu_mean = gpu_total / kRuns;
    if (gpu_ms) *gpu_ms = gpu_mean;
    if (cpu_ms) *cpu_ms = cpu_mean;
    return gpu_mean <= cpu_mean;
  } catch (...) {
    return false;
  }
}

bool VulkanConv2dBatchNoSlowerThanCpu(
    std::size_t batches, int input_channels, int output_channels,
    int input_height, int input_width, int output_height, int output_width,
    int kernel_height, int kernel_width, int stride_height, int stride_width,
    int pad_top, int pad_left, double* gpu_ms, double* cpu_ms,
    bool immutable_parameters, bool relu, bool swish, bool sigmoid,
    bool hard_swish) noexcept {
  if (gpu_ms) *gpu_ms = std::numeric_limits<double>::infinity();
  if (cpu_ms) *cpu_ms = std::numeric_limits<double>::infinity();
  if (batches == 0 || input_channels <= 0 || output_channels <= 0 ||
      input_height <= 0 || input_width <= 0 || output_height <= 0 || output_width <= 0 ||
      kernel_height <= 0 || kernel_width <= 0 || stride_height <= 0 ||
      stride_height != stride_width || pad_top < 0 || pad_left < 0 || batches > UINT32_MAX ||
      (int(relu) + int(swish) + int(sigmoid) + int(hard_swish) > 1)) return false;
  try {
    (void)immutable_parameters;
    const auto input_per_batch = static_cast<std::size_t>(input_channels) * input_height * input_width;
    const auto output_per_batch = static_cast<std::size_t>(output_channels) * output_height * output_width;
    const auto weight_elements = static_cast<std::size_t>(output_channels) * input_channels *
                                 kernel_height * kernel_width;
    if (input_per_batch == 0 || output_per_batch == 0 || weight_elements == 0 ||
        batches > std::numeric_limits<std::size_t>::max() / input_per_batch ||
        batches > std::numeric_limits<std::size_t>::max() / output_per_batch) return false;
    std::vector<float> input(batches * input_per_batch), weights(weight_elements),
        bias(static_cast<std::size_t>(output_channels)), cpu(batches * output_per_batch),
        gpu(batches * output_per_batch);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i % 251) * .0078125F - 1.F;
    for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<float>(i % 29) * .015625F - .1875F;
    for (std::size_t i = 0; i < bias.size(); ++i) bias[i] = static_cast<float>(i % 17) * .03125F - .25F;
    constexpr int kRuns = 5;
    // One untimed GPU call pays pipeline/descriptor setup so the timed
    // transfer-inclusive mean is comparable to a warmed hybrid Conv.
    if (!VulkanConv2dBatch(gpu.data(), input.data(), weights.data(), bias.data(), batches,
                           input_channels, output_channels, input_height, input_width,
                           output_height, output_width, kernel_height, kernel_width,
                           stride_height, stride_width, pad_top, pad_left,
                           immutable_parameters, relu, swish, sigmoid, hard_swish)) {
      return false;
    }
    double cpu_total{};
    double gpu_total{};
    for (int run = 0; run < kRuns; ++run) {
      const auto cpu_begin = std::chrono::steady_clock::now();
      kernels::Conv2dBatch(cpu.data(), input.data(), weights.data(), bias.data(),
                           static_cast<int>(batches), input_channels, output_channels,
                           input_height, input_width, output_height, output_width,
                           kernel_height, kernel_width, stride_height, stride_width,
                           pad_top, pad_left, relu);
      if (swish) kernels::Swish(cpu.data(), cpu.data(), cpu.size());
      if (sigmoid) kernels::Sigmoid(cpu.data(), cpu.data(), cpu.size());
      if (hard_swish) kernels::HardSwish(cpu.data(), cpu.data(), cpu.size());
      cpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - cpu_begin).count();
      const auto gpu_begin = std::chrono::steady_clock::now();
      if (!VulkanConv2dBatch(gpu.data(), input.data(), weights.data(), bias.data(), batches,
                             input_channels, output_channels, input_height, input_width,
                             output_height, output_width, kernel_height, kernel_width,
                             stride_height, stride_width, pad_top, pad_left,
                             immutable_parameters, relu, swish,
                             sigmoid, hard_swish)) return false;
      gpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - gpu_begin).count();
      for (std::size_t i = 0; i < gpu.size(); ++i) {
        const float tolerance = 3e-5F * std::max(1.F, std::abs(cpu[i]));
        if (std::abs(gpu[i] - cpu[i]) > tolerance) return false;
      }
    }
    const double cpu_mean = cpu_total / kRuns;
    const double gpu_mean = gpu_total / kRuns;
    if (gpu_ms) *gpu_ms = gpu_mean;
    if (cpu_ms) *cpu_ms = cpu_mean;
    return gpu_mean <= cpu_mean;
  } catch (...) {
    return false;
  }
}

bool VulkanConvTranspose2x2BatchNoSlowerThanCpu(
    std::size_t batches, int input_channels, int output_channels,
    int input_height, int input_width, double* gpu_ms, double* cpu_ms,
    bool immutable_parameters) noexcept {
  if (gpu_ms) *gpu_ms = std::numeric_limits<double>::infinity();
  if (cpu_ms) *cpu_ms = std::numeric_limits<double>::infinity();
  if (batches == 0 || input_channels <= 0 || output_channels <= 0 ||
      input_height <= 0 || input_width <= 0 || batches > UINT32_MAX) return false;
  try {
    (void)immutable_parameters;
    const auto input_per_batch = static_cast<std::size_t>(input_channels) * input_height * input_width;
    const auto output_per_batch = static_cast<std::size_t>(output_channels) * input_height * input_width * 4;
    const auto weight_elements = static_cast<std::size_t>(input_channels) * output_channels * 4;
    if (input_per_batch == 0 || output_per_batch == 0 || weight_elements == 0 ||
        batches > std::numeric_limits<std::size_t>::max() / input_per_batch ||
        batches > std::numeric_limits<std::size_t>::max() / output_per_batch) return false;
    std::vector<float> input(batches * input_per_batch), weights(weight_elements),
        bias(static_cast<std::size_t>(output_channels)), cpu(batches * output_per_batch),
        gpu(batches * output_per_batch);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i % 251) * .0078125F - 1.F;
    for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<float>(i % 29) * .015625F - .1875F;
    for (std::size_t i = 0; i < bias.size(); ++i) bias[i] = static_cast<float>(i % 17) * .03125F - .25F;
    constexpr int kRuns = 5;
    double cpu_total{};
    double gpu_total{};
    for (int run = 0; run < kRuns; ++run) {
      const auto cpu_begin = std::chrono::steady_clock::now();
      kernels::ConvTranspose2x2Batch(cpu.data(), input.data(), weights.data(), bias.data(),
                                     static_cast<int>(batches), input_channels, output_channels,
                                     input_height, input_width);
      cpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - cpu_begin).count();
      const auto gpu_begin = std::chrono::steady_clock::now();
      if (!VulkanConvTranspose2x2Batch(gpu.data(), input.data(), weights.data(), bias.data(),
                                       batches, input_channels, output_channels,
                                       input_height, input_width, false)) return false;
      gpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - gpu_begin).count();
      for (std::size_t i = 0; i < gpu.size(); ++i) {
        const float tolerance = 3e-5F * std::max(1.F, std::abs(cpu[i]));
        if (std::abs(gpu[i] - cpu[i]) > tolerance) return false;
      }
    }
    const double cpu_mean = cpu_total / kRuns;
    const double gpu_mean = gpu_total / kRuns;
    if (gpu_ms) *gpu_ms = gpu_mean;
    if (cpu_ms) *cpu_ms = cpu_mean;
    return gpu_mean <= cpu_mean;
  } catch (...) {
    return false;
  }
}

bool VulkanConvTranspose2x2AddBatchNoSlowerThanCpu(
    std::size_t batches, int input_channels, int output_channels,
    int input_height, int input_width, double* gpu_ms, double* cpu_ms,
    bool immutable_parameters) noexcept {
  if (gpu_ms) *gpu_ms = std::numeric_limits<double>::infinity();
  if (cpu_ms) *cpu_ms = std::numeric_limits<double>::infinity();
  if (batches == 0 || input_channels <= 0 || output_channels <= 0 ||
      input_height <= 0 || input_width <= 0 || batches > UINT32_MAX) return false;
  try {
    (void)immutable_parameters;
    const auto input_per_batch = static_cast<std::size_t>(input_channels) * input_height * input_width;
    const auto output_per_batch = static_cast<std::size_t>(output_channels) * input_height * input_width * 4;
    const auto weight_elements = static_cast<std::size_t>(input_channels) * output_channels * 4;
    if (input_per_batch == 0 || output_per_batch == 0 || weight_elements == 0 ||
        batches > std::numeric_limits<std::size_t>::max() / input_per_batch ||
        batches > std::numeric_limits<std::size_t>::max() / output_per_batch) return false;
    std::vector<float> input(batches * input_per_batch), weights(weight_elements),
        bias(static_cast<std::size_t>(output_channels)), residual(batches * output_per_batch),
        cpu(batches * output_per_batch), gpu(batches * output_per_batch);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i % 251) * .0078125F - 1.F;
    for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<float>(i % 29) * .015625F - .1875F;
    for (std::size_t i = 0; i < bias.size(); ++i) bias[i] = static_cast<float>(i % 17) * .03125F - .25F;
    for (std::size_t i = 0; i < residual.size(); ++i) residual[i] = static_cast<float>(i % 41) * .03125F - .625F;
    constexpr int kRuns = 5;
    double cpu_total{}, gpu_total{};
    for (int run = 0; run < kRuns; ++run) {
      const auto cpu_begin = std::chrono::steady_clock::now();
      kernels::ConvTranspose2x2Batch(cpu.data(), input.data(), weights.data(), bias.data(),
                                     static_cast<int>(batches), input_channels, output_channels,
                                     input_height, input_width);
      kernels::BinaryInplace(cpu.data(), residual.data(), cpu.size(), kernels::BinaryOp::add);
      cpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - cpu_begin).count();
      const auto gpu_begin = std::chrono::steady_clock::now();
      if (!VulkanConvTranspose2x2AddBatch(gpu.data(), input.data(), weights.data(), bias.data(),
                                          residual.data(), batches, input_channels, output_channels,
                                          input_height, input_width, false)) return false;
      gpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - gpu_begin).count();
      for (std::size_t i = 0; i < gpu.size(); ++i) {
        const float tolerance = 3e-5F * std::max(1.F, std::abs(cpu[i]));
        if (std::abs(gpu[i] - cpu[i]) > tolerance) return false;
      }
    }
    const double cpu_mean = cpu_total / kRuns;
    const double gpu_mean = gpu_total / kRuns;
    if (gpu_ms) *gpu_ms = gpu_mean;
    if (cpu_ms) *cpu_ms = cpu_mean;
    return gpu_mean <= cpu_mean;
  } catch (...) {
    return false;
  }
}

bool VulkanNearestResize2xAddBatchNoSlowerThanCpu(
    std::size_t batches, int channels, int input_height, int input_width,
    double* gpu_ms, double* cpu_ms) noexcept {
  if (gpu_ms) *gpu_ms = std::numeric_limits<double>::infinity();
  if (cpu_ms) *cpu_ms = std::numeric_limits<double>::infinity();
  if (batches == 0 || channels <= 0 || input_height <= 0 || input_width <= 0 ||
      batches > UINT32_MAX) return false;
  try {
    const auto input_per_batch = static_cast<std::size_t>(channels) * input_height * input_width;
    if (input_per_batch == 0 || input_per_batch > std::numeric_limits<std::size_t>::max() / 4 ||
        batches > std::numeric_limits<std::size_t>::max() / (input_per_batch * 4)) return false;
    const auto output_per_batch = input_per_batch * 4;
    std::vector<float> source(batches * input_per_batch), residual(batches * output_per_batch),
        cpu(batches * output_per_batch), gpu(batches * output_per_batch);
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = static_cast<float>(i % 251) * .0078125F - 1.F;
    for (std::size_t i = 0; i < residual.size(); ++i) residual[i] = static_cast<float>(i % 127) * .015625F - .75F;
    constexpr int kRuns = 5;
    double cpu_total{};
    double gpu_total{};
    for (int run = 0; run < kRuns; ++run) {
      const auto cpu_begin = std::chrono::steady_clock::now();
      kernels::NearestResize2xAdd(cpu.data(), source.data(), residual.data(),
                                  static_cast<int>(batches), channels, input_height, input_width);
      cpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - cpu_begin).count();
      const auto gpu_begin = std::chrono::steady_clock::now();
      if (!VulkanNearestResize2xAddBatch(gpu.data(), source.data(), residual.data(), batches,
                                         channels, input_height, input_width)) return false;
      gpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - gpu_begin).count();
      for (std::size_t i = 0; i < gpu.size(); ++i) {
        const float tolerance = 2e-5F * std::max(1.F, std::abs(cpu[i]));
        if (std::abs(gpu[i] - cpu[i]) > tolerance) return false;
      }
    }
    const double cpu_mean = cpu_total / kRuns;
    const double gpu_mean = gpu_total / kRuns;
    if (gpu_ms) *gpu_ms = gpu_mean;
    if (cpu_ms) *cpu_ms = cpu_mean;
    return gpu_mean <= cpu_mean;
  } catch (...) {
    return false;
  }
}

bool VulkanGemmNoSlowerThanCpu(int rows, int depth, int columns,
                               double* gpu_ms, double* cpu_ms,
                               bool immutable_parameters) noexcept {
  if (gpu_ms) *gpu_ms = std::numeric_limits<double>::infinity();
  if (cpu_ms) *cpu_ms = std::numeric_limits<double>::infinity();
  if (rows <= 0 || depth <= 0 || columns <= 0) return false;
  try {
    (void)immutable_parameters;
    const auto r=static_cast<std::size_t>(rows), k=static_cast<std::size_t>(depth), c=static_cast<std::size_t>(columns);
    if (r > std::numeric_limits<std::size_t>::max()/k || k > std::numeric_limits<std::size_t>::max()/c ||
        r > std::numeric_limits<std::size_t>::max()/c) return false;
    std::vector<float> left(r*k), right(k*c), bias(c), cpu(r*c), gpu(r*c);
    for (std::size_t i=0;i<left.size();++i) left[i]=static_cast<float>(i%251)*.0078125F-1.F;
    for (std::size_t i=0;i<right.size();++i) right[i]=static_cast<float>(i%113)*.015625F-.75F;
    for (std::size_t i=0;i<bias.size();++i) bias[i]=static_cast<float>(i%31)*.03125F-.5F;
    constexpr int kRuns=5;
    double cpu_total{},gpu_total{};
    for (int run=0;run<kRuns;++run) {
      const auto cpu_begin=std::chrono::steady_clock::now();
      kernels::Gemm(cpu.data(),left.data(),right.data(),bias.data(),rows,columns,depth);
      cpu_total+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-cpu_begin).count();
      const auto gpu_begin=std::chrono::steady_clock::now();
      if (!VulkanGemm(gpu.data(),left.data(),right.data(),bias.data(),rows,depth,columns,
                      false)) return false;
      gpu_total+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-gpu_begin).count();
      for (std::size_t i=0;i<gpu.size();++i) {
        const float tolerance=3e-5F*std::max(1.F,std::abs(cpu[i]));
        if (std::abs(cpu[i]-gpu[i])>tolerance) return false;
      }
    }
    const double cpu_mean=cpu_total/kRuns, gpu_mean=gpu_total/kRuns;
    if (gpu_ms) *gpu_ms=gpu_mean;
    if (cpu_ms) *cpu_ms=cpu_mean;
    return gpu_mean<=cpu_mean;
  } catch (...) {
    return false;
  }
}

bool VulkanGemmSwishNoSlowerThanCpu(int rows, int depth, int columns,
                                    double* gpu_ms, double* cpu_ms,
                                    bool immutable_parameters) noexcept {
  if (gpu_ms) *gpu_ms = std::numeric_limits<double>::infinity();
  if (cpu_ms) *cpu_ms = std::numeric_limits<double>::infinity();
  if (rows <= 0 || depth <= 0 || columns <= 0) return false;
  try {
    const auto r = static_cast<std::size_t>(rows);
    const auto k = static_cast<std::size_t>(depth);
    const auto c = static_cast<std::size_t>(columns);
    if (r > std::numeric_limits<std::size_t>::max() / k ||
        k > std::numeric_limits<std::size_t>::max() / c ||
        r > std::numeric_limits<std::size_t>::max() / c) return false;
    std::vector<float> left(r * k), right(k * c), bias(c), cpu(r * c), gpu(r * c);
    for (std::size_t i = 0; i < left.size(); ++i) left[i] = static_cast<float>(i % 251) * .0078125F - 1.F;
    for (std::size_t i = 0; i < right.size(); ++i) right[i] = static_cast<float>(i % 113) * .015625F - .75F;
    for (std::size_t i = 0; i < bias.size(); ++i) bias[i] = static_cast<float>(i % 31) * .03125F - .5F;
    constexpr int kRuns = 5;
    double cpu_total{};
    double gpu_total{};
    for (int run = 0; run < kRuns; ++run) {
      const auto cpu_begin = std::chrono::steady_clock::now();
      kernels::Gemm(cpu.data(), left.data(), right.data(), bias.data(), rows, columns, depth);
      kernels::Swish(cpu.data(), cpu.data(), cpu.size());
      cpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - cpu_begin).count();
      const auto gpu_begin = std::chrono::steady_clock::now();
      if (!VulkanGemmSwish(gpu.data(), left.data(), right.data(), bias.data(), rows, depth,
                           columns, immutable_parameters)) return false;
      gpu_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - gpu_begin).count();
      for (std::size_t i = 0; i < gpu.size(); ++i) {
        const float tolerance = 3e-5F * std::max(1.F, std::abs(cpu[i]));
        if (std::abs(cpu[i] - gpu[i]) > tolerance) return false;
      }
    }
    const double cpu_mean = cpu_total / kRuns;
    const double gpu_mean = gpu_total / kRuns;
    if (gpu_ms) *gpu_ms = gpu_mean;
    if (cpu_ms) *cpu_ms = cpu_mean;
    return gpu_mean <= cpu_mean;
  } catch (...) {
    return false;
  }
}

bool VulkanBinaryNoSlowerThanCpu(std::size_t count, kernels::BinaryOp operation,
                                 double* gpu_ms, double* cpu_ms, bool immutable_right) noexcept {
  if (gpu_ms) *gpu_ms = std::numeric_limits<double>::infinity();
  if (cpu_ms) *cpu_ms = std::numeric_limits<double>::infinity();
  if (count == 0 || count > UINT32_MAX) return false;
  try {
    std::vector<float> left(count), right(count), cpu(count), gpu(count);
    for (std::size_t i = 0; i < count; ++i) {
      left[i] = static_cast<float>(i) * .03125F - 7.F;
      right[i] = static_cast<float>(i % 29) * .125F + .5F;
    }
    constexpr int kRuns = 5;
    double cpu_total{};
    double gpu_total{};
    for (int run = 0; run < kRuns; ++run) {
      const auto cpu_begin = std::chrono::steady_clock::now();
      kernels::Binary(cpu.data(), left.data(), right.data(), count, operation);
      cpu_total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpu_begin).count();
      const auto gpu_begin = std::chrono::steady_clock::now();
      if (!VulkanBinary(gpu.data(), left.data(), right.data(), count, operation,
                        immutable_right)) return false;
      gpu_total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gpu_begin).count();
      for (std::size_t i = 0; i < count; ++i) {
        const float tolerance = 2e-5F * std::max(1.F, std::abs(cpu[i]));
        if (std::abs(gpu[i] - cpu[i]) > tolerance) return false;
      }
    }
    const double cpu_mean = cpu_total / kRuns;
    const double gpu_mean = gpu_total / kRuns;
    if (gpu_ms) *gpu_ms = gpu_mean;
    if (cpu_ms) *cpu_ms = cpu_mean;
    return gpu_mean <= cpu_mean;
  } catch (...) {
    return false;
  }
}

}  // namespace ppocr::detail
