cmake_minimum_required(VERSION 3.18)
project(gpu-engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11
)
FetchContent_MakeAvailable(nlohmann_json doctest)

# ---------------------------------------------------------------------------
# CPU oracle + host-only KV cache bookkeeping: pure C++17, no CUDA needed.
# Buildable and testable on any machine (this is M1's whole point).
# ---------------------------------------------------------------------------
add_library(cpu_oracle STATIC
    src/model.cpp
    src/kv_cache_pool.cpp
    src/reference/weights_loader.cpp
    src/reference/cpu_model.cpp
)
target_include_directories(cpu_oracle PUBLIC include src)
target_link_libraries(cpu_oracle PUBLIC nlohmann_json::nlohmann_json)

add_executable(gpu_engine_tests
    tests/doctest_main.cpp
    tests/test_end_to_end.cpp
    tests/test_kv_cache.cpp
)
target_link_libraries(gpu_engine_tests PRIVATE cpu_oracle doctest::doctest)

enable_testing()
add_test(NAME gpu_engine_tests COMMAND gpu_engine_tests)

# ---------------------------------------------------------------------------
# CUDA path: cuBLAS GEMMs, custom kernels, paged attention, CUDA Graphs.
# Only enabled when a CUDA toolchain is actually present. Source files
# still live in src/kernels/*.cu and src/engine.cu regardless -- this guard
# only controls whether they get compiled on this particular machine.
# ---------------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3060 (sm_86)

    find_package(CUDAToolkit REQUIRED)

    add_library(gpu_kernels STATIC
        src/loader.cpp
        src/model.cpp
        src/kv_cache_pool.cpp
        src/reference/weights_loader.cpp
        src/kv_cache.cu
        src/kernels/embedding.cu
        src/kernels/layernorm.cu
        src/kernels/gelu.cu
        src/kernels/attention.cu
        src/kernels/logits.cu
        src/engine.cu
    )
    target_include_directories(gpu_kernels PUBLIC include src)
    target_link_libraries(gpu_kernels PUBLIC CUDA::cublas CUDA::cudart nlohmann_json::nlohmann_json)
    set_target_properties(gpu_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    add_executable(gpu_kernel_tests tests/doctest_main.cpp tests/test_kernels.cpp src/reference/cpu_model.cpp src/reference/weights_loader.cpp src/model.cpp)
    target_link_libraries(gpu_kernel_tests PRIVATE gpu_kernels doctest::doctest nlohmann_json::nlohmann_json)
    add_test(NAME gpu_kernel_tests COMMAND gpu_kernel_tests)

    add_executable(bench_decode bench/bench_decode.cpp)
    target_link_libraries(bench_decode PRIVATE gpu_kernels)

    message(STATUS "CUDA toolchain found -- GPU kernel targets enabled")
else()
    message(WARNING "No CUDA toolchain found -- gpu_kernels/gpu_kernel_tests/bench_decode are "
                     "SKIPPED on this machine. Source for them still lives under src/ and bench/; "
                     "they need to be built and run on a machine with an NVIDIA GPU + CUDA 12.x "
                     "before any M2-M7 claim can be considered verified.")
endif()
