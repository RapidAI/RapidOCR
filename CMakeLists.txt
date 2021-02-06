cmake_minimum_required(VERSION 3.10)
project(RapidOCR)


# Disable in-source builds to prevent source tree corruption.
if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "
FATAL: In-source builds are not allowed.
       You should create a separate directory for build files.
")
endif()


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cpp/BaiPiaoOcrOnnx/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/onnxruntime/)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)



if(WIN32)

	add_definitions(-D_BPOCR_EXPORT_API)
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/win/x64/)
	IF(CMAKE_BUILD_TYPE STREQUAL Debug)  #debug
	   	link_libraries(opencv_world3413d.lib onnxruntime.lib)
	else()
	   	link_libraries(opencv_world3413.lib onnxruntime.lib)
	endif()

elseif(UNIX)

	# set（OpenCV_DIR  /path/to/opencv/build）
	FIND_PACKAGE(OpenCV REQUIRED)
	if(OPENCV_FOUND)
		message("OpenCV FOUND")
		include_directories(${OpenCV_INCLUDE_DIRS})
		link_directories(${OpenCV_LIBRARY_DIRS})
		link_libraries(${OpenCV_LIBS})
		
	endif()

endif()

#  openmp 
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
message(${CMAKE_EXE_LINKER_FLAGS})
endif()

add_definitions(-D__OPENMP__)
aux_source_directory(cpp/BaiPiaoOcrOnnx/src/ SRC_MAIN)
list(REMOVE_ITEM SRC_MAIN "main.cpp") 
set(OCR_SRC api4cpp/baipiaoocr_api.cpp ${SRC_MAIN}  )

add_library(rapidocrapi SHARED ${OCR_SRC})

target_compile_definitions(rapidocrapi PUBLIC -D_BPOCR_LIB)

# test program 
set(BPOCRTESTSRC api4cpp/bpocrapitest.cpp)
add_executable(rapidocrtest ${BPOCRTESTSRC})


#opencv 
#set(OPENCV_LIB opencv_core opencv_imgproc opencv_imgcodecs)

if(WIN32)

target_link_directories(bpocrapi PUBLIC lib/win)
target_link_libraries(bpocrtest bpocrapi)
#INSTALL_NUGET(bpocrapi Microsoft.ML.OnnxRuntime 1.6.0)
elseif(UNIX)

message(STATUS ${CMAKE_CURRENT_SOURCE_DIR}/lib/linux)
target_link_directories(rapidocrapi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib/linux )
target_link_libraries(rapidocrapi onnxruntime ${OPENCV_LIB})

target_link_directories(rapidocrtest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib/linux)
target_link_libraries(rapidocrtest onnxruntime ${OPENCV_LIB} rapidocrapi  gomp)


endif()


