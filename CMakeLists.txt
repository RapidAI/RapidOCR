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
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/ /rapidocr/include)



if(WIN32)

	add_definitions(-D_BPOCR_EXPORT_API)
	IF(CMAKE_CL_64)
		set(platform x64)
		link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/win/x64/)
		link_directories(onnxruntime-win-x64-1.6.0/lib)
	ELSE(CMAKE_CL_64)
		set(platform x86)
		link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/win/x86/)
	ENDIF(CMAKE_CL_64)
    


	set(OpenCV_DIR c:/rapidocr/opencv/opencv/build)  # 指向下载的opencv解开后的目录，下面会有bin/lib/include等目录

elseif(UNIX)

	# set（OpenCV_DIR  /path/to/opencv/build）

	link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/linux /rapidocr/lib/)
endif()

#IF(CMAKE_BUILD_TYPE STREQUAL Debug)  #debug
#   	
#else()
#   	link_libraries(opencv_world3413.lib onnxruntime.lib)
#endif()

# for onnxruntime

link_libraries(onnxruntime)

FIND_PACKAGE(OpenCV REQUIRED)
if(OPENCV_FOUND)
	message("OpenCV FOUND")
	include_directories(${OpenCV_INCLUDE_DIRS})
	link_directories(${OpenCV_LIBRARY_DIRS})
	link_libraries(${OpenCV_LIBS})
		
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

#end of openmp

add_definitions(-D__OPENMP__)
aux_source_directory(cpp/BaiPiaoOcrOnnx/src/ SRC_MAIN)
list(REMOVE_ITEM SRC_MAIN "main.cpp") 


set(OCR_SRC api4cpp/rapidocr_api.cpp ${SRC_MAIN}  )

add_library(rapidocrapi SHARED ${OCR_SRC})

target_compile_definitions(rapidocrapi PUBLIC -D_BPOCR_LIB)

# test program 
set(BPOCRTESTSRC api4cpp/rapidocrtest.cpp)
add_executable(rapidocrtest ${BPOCRTESTSRC})


target_link_libraries(rapidocrtest onnxruntime ${OPENCV_LIB} rapidocrapi)


