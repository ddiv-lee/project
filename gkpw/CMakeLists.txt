cmake_minimum_required(VERSION 3.5)
project(gkpw)

add_definitions(-std=c++11)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-pthread;-g;-G;-gencode;arch=compute_30;code=sm_30)

include_directories(${PROJECT_SOURCE_DIR}/include)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
endif()

AUX_SOURCE_DIRECTORY(./src DIR_SRCS)



#cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR} src/prelu.cu)
find_package( OpenCV)
include_directories(OpenCV_INCLUDE_DIRS)

MESSAGE(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
add_executable( gkpw ${DIR_SRCS} Vibe.h Vibe.cpp throw_algo.cpp throw_algo.h testurl.cpp)

#set(CMAKE_CXX_FLAGS  "-lpthread")
#set(CMAKE_CXX_FLAGS  "-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
target_link_libraries( gkpw openblas)
target_link_libraries( gkpw cudart)
target_link_libraries( gkpw nvinfer)
target_link_libraries( gkpw nvparsers)
target_link_libraries( gkpw ${OpenCV_LIBS} )

