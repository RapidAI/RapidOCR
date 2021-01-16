cmake_minimum_required(VERSION 3.10)
project(BaiPiaoOCR)


if(WIN32)

	add_definitions(-D_BPOCR_EXPORT_API)
elseif(UNIX)

endif()

set(OCR_SRC api4cpp/baipiaoocr_api.cpp)

add_library(bpocrapi SHARED ${OCR_SRC})