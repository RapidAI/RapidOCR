cmake_minimum_required(VERSION 3.1)
project(BaiPiaoOcrOnnx)

option(BAIPIAO_LIB "OcrLite Jni Support" OFF)
option(BAIPIAO_STATIC "Use Static Librarys For Build" ON)
option(BAIPIAO_OPENMP "OcrLite Enable OpenMP" ON)
option(BAIPIAO_BENCHMARK "build benchmark" ON)
#set(BAIPIAO_LIB ON)
#set(BAIPIAO_STATIC OFF)
#set(BAIPIAO_OPENMP OFF)
set(BAIPIAO_BENCHMARK ON)

set(CMAKE_CXX_STANDARD 11)
add_definitions(-DUNICODE -D_UNICODE)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions("-Wall -g -O0")
else ()
    add_definitions("-Wall")
endif ()

# OpenMP flags for MACOS
if (APPLE)
    if (CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(OpenMP_C "${CMAKE_C_COMPILER}")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY ${OpenMP_C_LIB_NAMES})
    endif ()
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY ${OpenMP_C_LIB_NAMES})
    endif ()
    link_directories("/usr/local/opt/libomp/lib")
endif ()

# OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message(FATAL_ERROR "OpenMP Not Found!")
endif ()

# OnnxRuntime
if (BAIPIAO_STATIC)
    set(OnnxRuntime_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-static")
else ()
    set(OnnxRuntime_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-shared")
endif ()
find_package(OnnxRuntime REQUIRED)
if (OnnxRuntime_FOUND)
    message(STATUS "OnnxRuntime_LIBS: ${OnnxRuntime_LIBS}")
    message(STATUS "OnnxRuntime_INCLUDE_DIRS: ${OnnxRuntime_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "onnxruntime Not Found!")
endif (OnnxRuntime_FOUND)

# OpenCV
if (BAIPIAO_STATIC)
    include(${CMAKE_CURRENT_SOURCE_DIR}/opencv-static/OpenCVWrapperConfig.cmake)
else ()
    include(${CMAKE_CURRENT_SOURCE_DIR}/opencv-shared/OpenCVWrapperConfig.cmake)
endif ()
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
    message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "opencv Not Found!")
endif (OpenCV_FOUND)

# JNI
if (BAIPIAO_LIB)
    find_package(JNI REQUIRED)
    if (JNI_FOUND)
        message("JNI FOUND")
        message(STATUS "JNI_LIBS: ${JNI_LIBS}")
        message(STATUS "JNI_INCLUDE_DIRS: ${JNI_INCLUDE_DIRS}")
        include_directories(${JNI_INCLUDE_DIRS})
    else ()
        message(FATAL_ERROR "JNI Not Found!")
    endif ()
endif ()

# project include
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# source
file(GLOB OCR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
set(OCR_COMPILE_CODE ${OCR_SRC})

if (BAIPIAO_LIB)
    add_library(BaiPiaoOcrOnnx SHARED ${OCR_COMPILE_CODE})
    target_compile_definitions(BaiPiaoOcrOnnx PRIVATE __JNI__)
    target_link_libraries(BaiPiaoOcrOnnx ${OnnxRuntime_LIBS} ${OpenCV_LIBS} ${JNI_LIBS} ${OpenMP_CXX_LIB_NAMES})
else ()
add_executable(BaiPiaoOcrOnnx ${OCR_COMPILE_CODE})
target_link_libraries(BaiPiaoOcrOnnx ${OnnxRuntime_LIBS} ${OpenCV_LIBS} ${OpenMP_CXX_LIB_NAMES})
endif ()
if (BAIPIAO_OPENMP)
    target_compile_definitions(BaiPiaoOcrOnnx PRIVATE __OPENMP__)
endif ()

# benchmark
if (BAIPIAO_BENCHMARK AND NOT BAIPIAO_LIB)
    add_executable(benchmark benchmark/benchmark.cpp
            src/AngleNet.cpp
            src/clipper.cpp
            src/CrnnNet.cpp
            src/DbNet.cpp
            src/getopt.cpp
            src/OcrLite.cpp
            src/OcrUtils.cpp)
    target_link_libraries(benchmark ${OnnxRuntime_LIBS} ${OpenCV_LIBS} ${OpenMP_CXX_LIB_NAMES})
    if (BAIPIAO_OPENMP)
        target_compile_definitions(benchmark PRIVATE __OPENMP__)
    endif ()
endif ()