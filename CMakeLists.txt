cmake_minimum_required(VERSION 3.10)
project(BaiPiaoOCR)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cpp/BaiPiaoOcrOnnx/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/onnxruntime/core/session)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)

if(WIN32)

	add_definitions(-D_BPOCR_EXPORT_API)
	
elseif(UNIX)
	include_directories(/usr/include/opencv4/)

endif()

aux_source_directory(cpp/BaiPiaoOcrOnnx/src/ SRC_MAIN)
list(REMOVE_ITEM SRC_MAIN "main.cpp") 
set(OCR_SRC api4cpp/baipiaoocr_api.cpp ${SRC_MAIN}  )

add_library(bpocrapi SHARED ${OCR_SRC})

target_compile_definitions(bpocrapi PUBLIC -D_BPOCR_LIB)

# test program 
set(BPOCRTESTSRC api4cpp/bpocrapitest.cpp)
add_executable(bpocrtest ${BPOCRTESTSRC})


#opencv 
set(OPENCV_LIB opencv_core opencv_imgproc opencv_imgcodecs)

if(WIN32)

target_link_directories(bpocrapi PUBLIC lib/win)

elseif(UNIX)

message(STATUS ${CMAKE_CURRENT_SOURCE_DIR}/lib/linux)
target_link_directories(bpocrapi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib/linux )
target_link_libraries(bpocrapi onnxruntime ${OPENCV_LIB})

target_link_directories(bpocrtest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib/linux)
target_link_libraries(bpocrtest onnxruntime ${OPENCV_LIB} bpocrapi  gomp)


endif()


