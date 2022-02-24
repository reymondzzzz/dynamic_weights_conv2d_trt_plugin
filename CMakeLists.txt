cmake_minimum_required(VERSION 3.21)
project(dynamic_weights_conv2d)

find_package(CUDAToolkit REQUIRED)
find_library(CUDNN_LIB NAMES cudnn PATHS ${CUDAToolkit_LIBRARY_DIR} PATH_SUFFIXES lib)
if(CUDNN_LIB)
    message(STATUS "Found CUDNN: ${CUDNN_LIB}")
else()
    message(FATAL_ERROR "CUDNN is not found")
endif()

set(trt_libs nvinfer nvonnxparser)
set(trt_home /usr/local/tensorrt)
set(TRT_INCLUDE_DIR ${trt_home}/include)
foreach(libName ${trt_libs})
    find_library(${libName}_lib NAMES ${libName} "/usr" PATHS ${trt_home} PATH_SUFFIXES lib)
    list(APPEND TRT_LIBS "${${libName}_lib}")
endforeach()

if(nvinfer_lib)
    message(STATUS "Found TRT: ${TRT_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "TRT is not found")
endif()

set(CMAKE_CXX_STANDARD 11)

add_library(dynamic_weights_conv2d_trt_plugin SHARED include/trt_plugin.h src/trt_plugin.cpp include/utils.h)
target_include_directories(dynamic_weights_conv2d_trt_plugin PUBLIC include ${TRT_INCLUDE_DIR})
target_link_libraries(dynamic_weights_conv2d_trt_plugin PUBLIC ${CUDNN_LIB} CUDA::cudart ${TRT_LIBS})


add_executable(main.cpp)
target_include_directories(main PUBLIC include /usr/local/cuda/include ${TRT_INCLUDE_DIR} )
target_link_libraries(main PUBLIC dynamic_weights_conv2d_trt_plugin ${TRT_LIBS})
set_target_properties(main PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

