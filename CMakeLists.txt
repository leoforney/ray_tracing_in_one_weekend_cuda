cmake_minimum_required(VERSION 3.21)
project(ray_tracing_in_one_weekend_cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=compute_86")


add_executable(ray_tracing_in_one_weekend_cuda main.cu
        vec3.cuh
        color.cuh
        ray.cuh)

set_target_properties(ray_tracing_in_one_weekend_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
