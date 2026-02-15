cmake_minimum_required(VERSION 3.18)

project(GeMM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 120)

find_package(CUDAToolkit REQUIRED)
include_directories(${PROJECT_SOURCE_DIR}/inc)

add_executable(gemm
    src/main.cpp
    src/kernel_runner.cpp
    src/matrix_data.cpp
    src/benchmark_collector.cpp
    src/device/matrix.cpp
    src/host/matrix.cpp
    cuda/cublas.cu
    cuda/naive.cu
    cuda/memory_coalescing.cu
    cuda/smem_tiling.cu
    cuda/1d_blocktiling.cu
    )

#set(CMAKE_CUDA_FLAGS_DEBUG "-G -lineinfo")
#set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(gemm PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(gemm PRIVATE CUDA::cublas CUDA::cudart)
