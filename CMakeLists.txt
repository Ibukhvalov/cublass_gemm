cmake_minimum_required(VERSION 3.18)

project(GeMM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set CUDA architectures manually (CMake doesn't support "native" for CUDA)
# You can check supported archs by: nvcc --list-gpu-arch
set(CMAKE_CUDA_ARCHITECTURES 120) # 86 = Ampere, change to match your GPU

find_package(CUDAToolkit REQUIRED)
include_directories(${PROJECT_SOURCE_DIR}/inc)

add_executable(gemm_test
    src/main.cpp
    cuda/kernel.cu)

target_include_directories(gemm_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(gemm_test PRIVATE CUDA::cublas CUDA::cudart)
