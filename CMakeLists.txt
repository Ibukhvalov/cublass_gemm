cmake_minimum_required(VERSION 3.18)

project(GeMM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 120)

find_package(CUDAToolkit REQUIRED)
include_directories(${PROJECT_SOURCE_DIR}/inc)

add_executable(gemm_test
    src/main.cpp
    src/kernel_runner.cpp
    src/matrix_data.cpp
    src/benchmark_collector.cpp
    src/device/matrix.cpp
    src/host/matrix.cpp
    cuda/cublass.cu
    cuda/naive.cu
    cuda/memory_coalescing.cu
    cuda/smem_tiling.cu
    )

target_include_directories(gemm_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(gemm_test PRIVATE CUDA::cublas CUDA::cudart)
