cmake_minimum_required(VERSION 3.24)
project(xurcroh_native LANGUAGES CUDA)

set(CMAKE_CUDA_ARCHITECTURES 90 CACHE STRING "CUDA architecture")
add_library(xurcroh_burn_alloc SHARED xurcroh_burn_alloc.cu)
add_library(otp_mask_kernels SHARED otp_mask_kernels.cu)
set_target_properties(xurcroh_burn_alloc otp_mask_kernels PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)
