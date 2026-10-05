cmake_minimum_required(VERSION 3.20)
project(ppocr_cpp VERSION 5.0.0 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(PPOCR_BUILD_SHARED "Build the shared library" ON)
option(PPOCR_BUILD_STATIC "Build the static library" ON)
option(PPOCR_BUILD_EXAMPLES "Build examples and smoke tools" ON)
option(PPOCR_BUILD_TESTS "Build the C API smoke test" ON)
# Shaders are compiled into the library. The loader is dlopen'd, so this does
# not add a libvulkan DT_NEEDED entry. Wheels pass OFF: manylinux images do
# not ship glslang, and the published CPU wheel must not change when the
# image happens to contain a shader compiler.
option(PPOCR_ENABLE_VULKAN "Embed Vulkan compute shaders when headers and glslangValidator are available" ON)

if(NOT PPOCR_BUILD_SHARED AND NOT PPOCR_BUILD_STATIC)
  message(FATAL_ERROR "Enable PPOCR_BUILD_SHARED and/or PPOCR_BUILD_STATIC")
endif()

set(PPOCR_SOURCES
    src/kernels.cpp
    src/onnx_lite.cpp
    src/ppocr.cpp
    src/vulkan_backend.cpp
    src/ppocr_c_api.cpp
)

add_library(ppocr_objects OBJECT ${PPOCR_SOURCES})
set_target_properties(ppocr_objects PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
target_compile_definitions(ppocr_objects PRIVATE PPOCR_BUILDING_SHARED)
target_include_directories(ppocr_objects PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(ppocr_objects PUBLIC cxx_std_20)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/Vulkan-Headers/include/vulkan/vulkan.h")
  target_include_directories(ppocr_objects PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/Vulkan-Headers/include)
  target_compile_definitions(ppocr_objects PRIVATE PPOCR_HAS_VULKAN_HEADERS=1)
endif()
# The vendored glslang.exe is a Windows compiler. On Linux/ARM use a host
# glslangValidator so the build does not try to execute a PE binary.
if(WIN32 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/glslang/bin/glslang.exe")
  set(PPOCR_GLSLANG "${CMAKE_CURRENT_SOURCE_DIR}/third_party/glslang/bin/glslang.exe")
else()
  find_program(PPOCR_GLSLANG NAMES glslangValidator glslang)
endif()
if(PPOCR_ENABLE_VULKAN AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/Vulkan-Headers/include/vulkan/vulkan.h" AND PPOCR_GLSLANG)
  set(PPOCR_VULKAN_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
  set(PPOCR_VULKAN_SPV "${PPOCR_VULKAN_GENERATED_DIR}/vulkan_binary.spv")
  set(PPOCR_VULKAN_SPV_HEADER "${PPOCR_VULKAN_GENERATED_DIR}/vulkan_binary_spv.hpp")
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
    COMMAND ${CMAKE_COMMAND} -DINPUT=${PPOCR_VULKAN_SPV} -DOUTPUT=${PPOCR_VULKAN_SPV_HEADER} -DNAME=kVulkanBinarySpv -P "${CMAKE_CURRENT_SOURCE_DIR}/src/embed_spirv.cmake"
    COMMAND ${CMAKE_COMMAND} -DINPUT=${PPOCR_VULKAN_TAIL_SPV} -DOUTPUT=${PPOCR_VULKAN_TAIL_SPV_HEADER} -DNAME=kVulkanDepthwiseTailSpv -P "${CMAKE_CURRENT_SOURCE_DIR}/src/embed_spirv.cmake"
    COMMAND ${CMAKE_COMMAND} -DINPUT=${PPOCR_VULKAN_POINTWISE_TAIL_SPV} -DOUTPUT=${PPOCR_VULKAN_POINTWISE_TAIL_SPV_HEADER} -DNAME=kVulkanPointwiseTailSpv -P "${CMAKE_CURRENT_SOURCE_DIR}/src/embed_spirv.cmake"
    COMMAND ${CMAKE_COMMAND} -E touch "${PPOCR_VULKAN_SPV_HEADER}"
    COMMAND ${CMAKE_COMMAND} -E touch "${PPOCR_VULKAN_TAIL_SPV_HEADER}"
    COMMAND ${CMAKE_COMMAND} -E touch "${PPOCR_VULKAN_POINTWISE_TAIL_SPV_HEADER}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan_binary.comp" "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan_depthwise_tail.comp" "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan_pointwise_tail.comp" "${CMAKE_CURRENT_SOURCE_DIR}/src/embed_spirv.cmake" "${PPOCR_GLSLANG}"
    VERBATIM)
  add_custom_target(ppocr_vulkan_shader DEPENDS "${PPOCR_VULKAN_SPV_HEADER}" "${PPOCR_VULKAN_TAIL_SPV_HEADER}" "${PPOCR_VULKAN_POINTWISE_TAIL_SPV_HEADER}")
  add_dependencies(ppocr_objects ppocr_vulkan_shader)
  target_include_directories(ppocr_objects PRIVATE "${PPOCR_VULKAN_GENERATED_DIR}")
  target_compile_definitions(ppocr_objects PRIVATE PPOCR_HAS_VULKAN_KERNELS=1)
elseif(NOT PPOCR_ENABLE_VULKAN)
  message(STATUS "Vulkan compute shaders disabled by PPOCR_ENABLE_VULKAN=OFF")
else()
  message(STATUS "Vulkan compute shaders disabled: need Vulkan-Headers and glslangValidator")
endif()

if(MSVC)
  target_compile_options(ppocr_objects PRIVATE /W4 /permissive- /O2 /fp:fast)
else()
  target_compile_options(ppocr_objects PRIVATE -Wall -Wextra -Wpedantic -O3 -ffast-math -funroll-loops)
endif()

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR};${CMAKE_GENERATOR_PLATFORM}" PPOCR_CPU)
if(NOT MSVC AND PPOCR_CPU MATCHES "x86_64|amd64|x64|i[3-6]86")
  # Baseline translation units stay SSE2. Auto-vectorization must not emit
  # AVX or AVX-512, or an AVX2-only machine will SIGILL inside the fallback.
  set_source_files_properties(
      src/kernels.cpp src/onnx_lite.cpp src/ppocr.cpp src/vulkan_backend.cpp src/ppocr_c_api.cpp
      PROPERTIES COMPILE_OPTIONS "-mno-avx;-mno-avx2;-mno-avx512f")
endif()

# AVX2 is isolated in its own object file. kernels.cpp remains baseline x86,
# so CPUs without AVX/AVX2 can safely take its scalar fallback at runtime.
if(PPOCR_CPU MATCHES "x86_64|amd64|x64|i[3-6]86")
  target_sources(ppocr_objects PRIVATE src/kernels_avx2.cpp)
  target_compile_definitions(ppocr_objects PRIVATE PPOCR_HAS_AVX2_KERNELS=1)
  if(MSVC)
    set_source_files_properties(src/kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2;/fp:fast")
  else()
    # -mno-avx512* is last-wins together with -mavx2 and forbids the compiler
    # from rewriting this file with EVEX/zmm encodings.
    set_source_files_properties(src/kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS
        "-mavx2;-mfma;-mno-avx512f;-mno-avx512bw;-mno-avx512vl;-mno-avx512dq")
  endif()
  target_sources(ppocr_objects PRIVATE src/kernels_avx512.cpp)
  target_compile_definitions(ppocr_objects PRIVATE PPOCR_HAS_AVX512_KERNELS=1)
  if(MSVC)
    set_source_files_properties(src/kernels_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512;/fp:fast")
  else()
    # This TU uses AVX-512F/DQ/BW/VL plus BMI tzcnt. None of those encodings
    # are enabled in kernels.cpp or kernels_avx2.cpp. HasAvx512() requires
    # the same feature bits before the TU is entered.
    set_source_files_properties(src/kernels_avx512.cpp PROPERTIES COMPILE_OPTIONS
        "-mavx512f;-mavx512dq;-mavx512bw;-mavx512vl;-mbmi;-mbmi2;-mfma")
  endif()
endif()

set(PPOCR_INSTALL_TARGETS "")

if(PPOCR_BUILD_STATIC)
  add_library(ppocr STATIC $<TARGET_OBJECTS:ppocr_objects>)
  target_include_directories(ppocr PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_compile_features(ppocr PUBLIC cxx_std_20)
  if(UNIX AND NOT APPLE)
    target_link_libraries(ppocr PUBLIC ${CMAKE_DL_LIBS})
  endif()
  set_target_properties(ppocr PROPERTIES OUTPUT_NAME ppocr EXPORT_NAME static)
  if(WIN32)
    set_target_properties(ppocr PROPERTIES OUTPUT_NAME ppocr_static)
  endif()
  add_library(ppocr::static ALIAS ppocr)
  list(APPEND PPOCR_INSTALL_TARGETS ppocr)
endif()

if(PPOCR_BUILD_SHARED)
  add_library(ppocr_shared SHARED $<TARGET_OBJECTS:ppocr_objects>)
  target_include_directories(ppocr_shared PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_compile_features(ppocr_shared PUBLIC cxx_std_20)
  target_compile_definitions(ppocr_shared PUBLIC PPOCR_SHARED)
  if(UNIX AND NOT APPLE)
    target_link_libraries(ppocr_shared PRIVATE ${CMAKE_DL_LIBS})
  endif()
  set_target_properties(ppocr_shared PROPERTIES
      OUTPUT_NAME ppocr
      EXPORT_NAME ppocr
      VERSION ${PROJECT_VERSION}
      SOVERSION ${PROJECT_VERSION_MAJOR})
  add_library(ppocr::ppocr ALIAS ppocr_shared)
  list(APPEND PPOCR_INSTALL_TARGETS ppocr_shared)
endif()

function(ppocr_link_runtime target)
  if(PPOCR_BUILD_STATIC)
    target_link_libraries(${target} PRIVATE ppocr)
  else()
    target_link_libraries(${target} PRIVATE ppocr_shared)
  endif()
endfunction()

# Smoke tools call detail:: symbols. Those symbols are hidden in the shared
# library, so the tools link the object files instead.
function(ppocr_link_objects target)
  target_link_libraries(${target} PRIVATE $<TARGET_OBJECTS:ppocr_objects>)
  target_include_directories(${target} PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_compile_features(${target} PRIVATE cxx_std_20)
  if(UNIX AND NOT APPLE)
    target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
  endif()
endfunction()

if(PPOCR_BUILD_EXAMPLES)
  add_executable(ppocr_demo examples/ppocr_demo.cpp)
  ppocr_link_runtime(ppocr_demo)

  add_executable(ppocr_cpp_api_demo examples/cpp_api_demo.cpp)
  target_link_libraries(ppocr_cpp_api_demo PRIVATE $<IF:$<BOOL:${PPOCR_BUILD_SHARED}>,ppocr_shared,ppocr>)
  if(PPOCR_BUILD_SHARED)
    set_target_properties(ppocr_cpp_api_demo PROPERTIES BUILD_RPATH "$<TARGET_FILE_DIR:ppocr_shared>")
  endif()

  add_executable(ppocr_c_api_demo examples/c_api_demo.c)
  target_link_libraries(ppocr_c_api_demo PRIVATE $<IF:$<BOOL:${PPOCR_BUILD_SHARED}>,ppocr_shared,ppocr>)
  if(PPOCR_BUILD_SHARED)
    set_target_properties(ppocr_c_api_demo PROPERTIES BUILD_RPATH "$<TARGET_FILE_DIR:ppocr_shared>")
  endif()

  add_executable(ppocr_isa_bench examples/ppocr_isa_bench.cpp)
  ppocr_link_objects(ppocr_isa_bench)

  add_executable(ppocr_bench examples/ppocr_bench.cpp)
  ppocr_link_runtime(ppocr_bench)

  add_executable(ppocr_mem_soak examples/ppocr_mem_soak.cpp)
  ppocr_link_runtime(ppocr_mem_soak)
  if(WIN32)
    target_link_libraries(ppocr_mem_soak PRIVATE psapi)
  endif()

  add_executable(ppocr_vulkan_smoke examples/ppocr_vulkan_smoke.cpp)
  ppocr_link_objects(ppocr_vulkan_smoke)

  add_executable(ppocr_batch_smoke examples/ppocr_batch_smoke.cpp)
  ppocr_link_runtime(ppocr_batch_smoke)

  add_executable(ppocr_kernel_smoke examples/ppocr_kernel_smoke.cpp)
  ppocr_link_objects(ppocr_kernel_smoke)

  add_executable(ppocr_determinism examples/ppocr_determinism.cpp)
  ppocr_link_objects(ppocr_determinism)

  add_executable(ppocr_gpu_graph_smoke examples/ppocr_gpu_graph_smoke.cpp)
  ppocr_link_objects(ppocr_gpu_graph_smoke)

  add_executable(ppocr_gpu_frontend_smoke examples/ppocr_gpu_frontend_smoke.cpp)
  ppocr_link_objects(ppocr_gpu_frontend_smoke)

  add_executable(ppocr_gpu_pipeline_smoke examples/ppocr_gpu_pipeline_smoke.cpp)
  ppocr_link_objects(ppocr_gpu_pipeline_smoke)

  add_executable(ppocr_gpu_ocr_smoke examples/ppocr_gpu_ocr_smoke.cpp)
  ppocr_link_runtime(ppocr_gpu_ocr_smoke)

  add_executable(ppocr_vulkan_robust examples/ppocr_vulkan_robust.cpp)
  ppocr_link_runtime(ppocr_vulkan_robust)

  add_executable(ppocr_gpu_graph_mem_soak examples/ppocr_gpu_graph_mem_soak.cpp)
  ppocr_link_objects(ppocr_gpu_graph_mem_soak)
endif()

if(PPOCR_BUILD_TESTS)
  add_executable(ppocr_c_api_smoke tests/c_api_smoke.c)
  target_link_libraries(ppocr_c_api_smoke PRIVATE $<IF:$<BOOL:${PPOCR_BUILD_SHARED}>,ppocr_shared,ppocr>)
  if(PPOCR_BUILD_SHARED)
    set_target_properties(ppocr_c_api_smoke PROPERTIES BUILD_RPATH "$<TARGET_FILE_DIR:ppocr_shared>")
  endif()
  enable_testing()
  add_test(NAME c_api_smoke COMMAND ppocr_c_api_smoke)
  if(NOT MSVC AND PPOCR_CPU MATCHES "x86_64|amd64|x64|i[3-6]86")
    add_test(NAME isa_encoding
        COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_isa_encoding.sh
            ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ppocr_objects.dir/src/kernels.cpp.o
            ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ppocr_objects.dir/src/kernels_avx2.cpp.o
            ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ppocr_objects.dir/src/kernels_avx512.cpp.o)
  endif()
endif()

if(NOT SKBUILD)
  install(TARGETS ${PPOCR_INSTALL_TARGETS}
      EXPORT ppocrTargets
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(DIRECTORY include/ppocr DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(EXPORT ppocrTargets
      NAMESPACE ppocr::
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ppocr)
  configure_package_config_file(
      cmake/ppocrConfig.cmake.in
      ${CMAKE_CURRENT_BINARY_DIR}/ppocrConfig.cmake
      INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ppocr)
  write_basic_package_version_file(
      ${CMAKE_CURRENT_BINARY_DIR}/ppocrConfigVersion.cmake
      COMPATIBILITY SameMajorVersion)
  install(FILES
      ${CMAKE_CURRENT_BINARY_DIR}/ppocrConfig.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/ppocrConfigVersion.cmake
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ppocr)
else()
  # scikit-build-core stages the shared library next to the Python package.
  if(NOT PPOCR_BUILD_SHARED)
    message(FATAL_ERROR "The Python wheel needs PPOCR_BUILD_SHARED=ON")
  endif()
  install(TARGETS ppocr_shared
      LIBRARY DESTINATION "${SKBUILD_PLATLIB_DIR}/rapidocr"
      RUNTIME DESTINATION "${SKBUILD_PLATLIB_DIR}/rapidocr"
      ARCHIVE DESTINATION "${SKBUILD_PLATLIB_DIR}/rapidocr")
endif()
