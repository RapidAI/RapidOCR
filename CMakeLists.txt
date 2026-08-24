cmake_minimum_required(VERSION 3.20)
project(ppocr_cpp LANGUAGES CXX)

add_library(ppocr STATIC
    src/kernels.cpp
    src/onnx_lite.cpp
    src/ppocr.cpp
    src/vulkan_backend.cpp
)
target_include_directories(ppocr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/Vulkan-Headers/include/vulkan/vulkan.h")
  target_include_directories(ppocr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/Vulkan-Headers/include)
  target_compile_definitions(ppocr PRIVATE PPOCR_HAS_VULKAN_HEADERS=1)
endif()
# A Windows compiler is vendored for hermetic local builds.  On Linux/ARM use
# a normal glslangValidator/glslang installation to embed exactly the same
# SPIR-V at build time; the runtime itself still dynamically loads Vulkan and
# has no Vulkan loader link dependency.
set(PPOCR_GLSLANG "${CMAKE_CURRENT_SOURCE_DIR}/third_party/glslang/bin/glslang.exe")
if(NOT EXISTS "${PPOCR_GLSLANG}")
  find_program(PPOCR_GLSLANG NAMES glslangValidator glslang)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/Vulkan-Headers/include/vulkan/vulkan.h" AND PPOCR_GLSLANG)
  set(PPOCR_VULKAN_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
  set(PPOCR_VULKAN_SPV "${PPOCR_VULKAN_GENERATED_DIR}/vulkan_binary.spv")
  set(PPOCR_VULKAN_SPV_HEADER "${PPOCR_VULKAN_GENERATED_DIR}/vulkan_binary_spv.hpp")
  # The high-channel detector tail has its own deliberately small shader.
  # Keeping it out of the all-operator module avoids a Radeon/LLPC compiler
  # interaction while retaining exactly the same Vulkan descriptor ABI.
  set(PPOCR_VULKAN_TAIL_SPV "${PPOCR_VULKAN_GENERATED_DIR}/vulkan_depthwise_tail.spv")
  set(PPOCR_VULKAN_TAIL_SPV_HEADER "${PPOCR_VULKAN_GENERATED_DIR}/vulkan_depthwise_tail_spv.hpp")
  set(PPOCR_VULKAN_POINTWISE_TAIL_SPV "${PPOCR_VULKAN_GENERATED_DIR}/vulkan_pointwise_tail.spv")
  set(PPOCR_VULKAN_POINTWISE_TAIL_SPV_HEADER "${PPOCR_VULKAN_GENERATED_DIR}/vulkan_pointwise_tail_spv.hpp")
  add_custom_command(
    OUTPUT "${PPOCR_VULKAN_SPV}" "${PPOCR_VULKAN_SPV_HEADER}" "${PPOCR_VULKAN_TAIL_SPV}" "${PPOCR_VULKAN_TAIL_SPV_HEADER}" "${PPOCR_VULKAN_POINTWISE_TAIL_SPV}" "${PPOCR_VULKAN_POINTWISE_TAIL_SPV_HEADER}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PPOCR_VULKAN_GENERATED_DIR}"
    COMMAND "${PPOCR_GLSLANG}" -V -S comp -o "${PPOCR_VULKAN_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan_binary.comp"
    COMMAND "${PPOCR_GLSLANG}" -V -S comp -o "${PPOCR_VULKAN_TAIL_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan_depthwise_tail.comp"
    COMMAND "${PPOCR_GLSLANG}" -V -S comp -o "${PPOCR_VULKAN_POINTWISE_TAIL_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan_pointwise_tail.comp"
    # Do not embed literal quotes in -D values: CMake otherwise treats them
    # as part of the filename when the generator invokes this script.
    COMMAND ${CMAKE_COMMAND} -DINPUT=${PPOCR_VULKAN_SPV} -DOUTPUT=${PPOCR_VULKAN_SPV_HEADER} -DNAME=kVulkanBinarySpv -P "${CMAKE_CURRENT_SOURCE_DIR}/src/embed_spirv.cmake"
    COMMAND ${CMAKE_COMMAND} -DINPUT=${PPOCR_VULKAN_TAIL_SPV} -DOUTPUT=${PPOCR_VULKAN_TAIL_SPV_HEADER} -DNAME=kVulkanDepthwiseTailSpv -P "${CMAKE_CURRENT_SOURCE_DIR}/src/embed_spirv.cmake"
    COMMAND ${CMAKE_COMMAND} -DINPUT=${PPOCR_VULKAN_POINTWISE_TAIL_SPV} -DOUTPUT=${PPOCR_VULKAN_POINTWISE_TAIL_SPV_HEADER} -DNAME=kVulkanPointwiseTailSpv -P "${CMAKE_CURRENT_SOURCE_DIR}/src/embed_spirv.cmake"
    COMMAND ${CMAKE_COMMAND} -E touch "${PPOCR_VULKAN_SPV_HEADER}"
    COMMAND ${CMAKE_COMMAND} -E touch "${PPOCR_VULKAN_TAIL_SPV_HEADER}"
    COMMAND ${CMAKE_COMMAND} -E touch "${PPOCR_VULKAN_POINTWISE_TAIL_SPV_HEADER}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan_binary.comp" "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan_depthwise_tail.comp" "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan_pointwise_tail.comp" "${CMAKE_CURRENT_SOURCE_DIR}/src/embed_spirv.cmake" "${PPOCR_GLSLANG}"
    VERBATIM)
  add_custom_target(ppocr_vulkan_shader DEPENDS "${PPOCR_VULKAN_SPV_HEADER}" "${PPOCR_VULKAN_TAIL_SPV_HEADER}" "${PPOCR_VULKAN_POINTWISE_TAIL_SPV_HEADER}")
  add_dependencies(ppocr ppocr_vulkan_shader)
  target_include_directories(ppocr PRIVATE "${PPOCR_VULKAN_GENERATED_DIR}")
  target_compile_definitions(ppocr PRIVATE PPOCR_HAS_VULKAN_KERNELS=1)
endif()
if(UNIX AND NOT APPLE)
  # dlopen/dlsym are used to keep the Vulkan loader optional at deployment.
  target_link_libraries(ppocr PRIVATE ${CMAKE_DL_LIBS})
endif()
target_compile_features(ppocr PUBLIC cxx_std_20)
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR};${CMAKE_GENERATOR_PLATFORM}" PPOCR_CPU)
if(MSVC)
  target_compile_options(ppocr PRIVATE /W4 /permissive- /O2 /fp:fast)
else()
  target_compile_options(ppocr PRIVATE -Wall -Wextra -Wpedantic -O3 -ffast-math -funroll-loops)
endif()

# AVX2 is isolated in its own object file.  kernels.cpp remains baseline x86,
# so CPUs without AVX/AVX2 can safely take its scalar fallback at runtime.
if(PPOCR_CPU MATCHES "x86_64|amd64|x64|i[3-6]86")
  target_sources(ppocr PRIVATE src/kernels_avx2.cpp)
  target_compile_definitions(ppocr PRIVATE PPOCR_HAS_AVX2_KERNELS=1)
  if(MSVC)
    set_source_files_properties(src/kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2;/fp:fast")
  else()
    set_source_files_properties(src/kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  endif()
  # AVX-512 code lives in a separate object file and is entered only after
  # CPUID/XCR0 runtime checks. Baseline x86 and AVX2 machines remain supported.
  target_sources(ppocr PRIVATE src/kernels_avx512.cpp)
  target_compile_definitions(ppocr PRIVATE PPOCR_HAS_AVX512_KERNELS=1)
  if(MSVC)
    set_source_files_properties(src/kernels_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512;/fp:fast")
  else()
    set_source_files_properties(src/kernels_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mfma")
  endif()
endif()

add_executable(ppocr_demo examples/ppocr_demo.cpp)
target_link_libraries(ppocr_demo PRIVATE ppocr)

add_executable(ppocr_bench examples/ppocr_bench.cpp)
target_link_libraries(ppocr_bench PRIVATE ppocr)

add_executable(ppocr_mem_soak examples/ppocr_mem_soak.cpp)
target_link_libraries(ppocr_mem_soak PRIVATE ppocr)
if(WIN32)
  target_link_libraries(ppocr_mem_soak PRIVATE psapi)
endif()

add_executable(ppocr_vulkan_smoke examples/ppocr_vulkan_smoke.cpp)
target_link_libraries(ppocr_vulkan_smoke PRIVATE ppocr)
target_include_directories(ppocr_vulkan_smoke PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_executable(ppocr_batch_smoke examples/ppocr_batch_smoke.cpp)
target_link_libraries(ppocr_batch_smoke PRIVATE ppocr)

# Low-level GEMM regression coverage.  This executable is intentionally
# dependency-free and exercises odd/tail shapes as well as the paired-row
# AVX-512/AVX2 paths selected dynamically at runtime.
add_executable(ppocr_kernel_smoke examples/ppocr_kernel_smoke.cpp)
target_link_libraries(ppocr_kernel_smoke PRIVATE ppocr)
target_include_directories(ppocr_kernel_smoke PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Whole neural-network CPU-vs-device regression. This tests the strict Vulkan
# graph executor directly; the public GPU-only OCR regression below additionally
# covers the device RGB front end and device CTC terminal reduction.
add_executable(ppocr_gpu_graph_smoke examples/ppocr_gpu_graph_smoke.cpp)
target_link_libraries(ppocr_gpu_graph_smoke PRIVATE ppocr)
target_include_directories(ppocr_gpu_graph_smoke PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Public front-end GPU-only numerical regression: validates RGB resize and
# normalization independently before an OCR result can be advertised as
# device-only correct.
add_executable(ppocr_gpu_frontend_smoke examples/ppocr_gpu_frontend_smoke.cpp)
target_link_libraries(ppocr_gpu_frontend_smoke PRIVATE ppocr)
target_include_directories(ppocr_gpu_frontend_smoke PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Exercises the real GPU-only detector handoff: GPU RGB preprocessing feeding
# the strict Vulkan graph, with CPU-vs-device final-output comparison.
add_executable(ppocr_gpu_pipeline_smoke examples/ppocr_gpu_pipeline_smoke.cpp)
target_link_libraries(ppocr_gpu_pipeline_smoke PRIVATE ppocr)
target_include_directories(ppocr_gpu_pipeline_smoke PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Public full OCR GPU-only contract regression.  Unlike the graph-only tools,
# it checks decoded text, detector geometry and CTC confidence on real pages.
add_executable(ppocr_gpu_ocr_smoke examples/ppocr_gpu_ocr_smoke.cpp)
target_link_libraries(ppocr_gpu_ocr_smoke PRIVATE ppocr)

# Strict Vulkan graph memory regression. It exercises a single warm model at
# ascending input sizes, records the arena high-water allocation and proves
# that dynamic activations fall back to the constants-only live baseline after
# every image/batch.
add_executable(ppocr_gpu_graph_mem_soak examples/ppocr_gpu_graph_mem_soak.cpp)
target_link_libraries(ppocr_gpu_graph_mem_soak PRIVATE ppocr)
target_include_directories(ppocr_gpu_graph_mem_soak PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
