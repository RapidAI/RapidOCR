cmake_minimum_required(VERSION 3.10)
project(BaiPiaoOCR)

include_directories(cpp/BaiPiaoOcrOnnx/include)
include_directories(include/onnxruntime/core/session)
include_directories(include/)

if(WIN32)

	add_definitions(-D_BPOCR_EXPORT_API)
elseif(UNIX)
	include_directories(/usr/include/opencv4/)

endif()

aux_source_directory(cpp/BaiPiaoOcrOnnx/src/ SRC_MAIN)
set(OCR_SRC api4cpp/baipiaoocr_api.cpp ${SRC_MAIN}  )

add_library(bpocrapi SHARED ${OCR_SRC})

target_compile_definitions(bpocrapi PUBLIC -D_BPOCR_LIB)