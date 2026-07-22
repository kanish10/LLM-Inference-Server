cmake_minimum_required(VERSION 3.16)
project(inference-engine CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find dependencies
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)
find_package(Threads REQUIRED)

# Optional: llama.cpp (only needed for --mode llama)
option(WITH_LLAMA "Build with llama.cpp support" OFF)
if(WITH_LLAMA)
    add_subdirectory(third_party/llama.cpp)
endif()

# Generate protobuf/gRPC C++ sources for engine.proto at configure time so
# `cmake --build` alone is sufficient (no separate `make proto` step needed).
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)
set(PROTO_DIR ${CMAKE_SOURCE_DIR}/../proto)
set(GEN_DIR ${CMAKE_SOURCE_DIR}/gen)
file(MAKE_DIRECTORY ${GEN_DIR})

set(PROTO_SRCS ${GEN_DIR}/engine.pb.cc)
set(PROTO_HDRS ${GEN_DIR}/engine.pb.h)
set(GRPC_SRCS ${GEN_DIR}/engine.grpc.pb.cc)
set(GRPC_HDRS ${GEN_DIR}/engine.grpc.pb.h)

add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS} ${GRPC_SRCS} ${GRPC_HDRS}
    COMMAND protobuf::protoc
    ARGS --proto_path=${PROTO_DIR}
         --cpp_out=${GEN_DIR}
         --grpc_out=${GEN_DIR}
         --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
         ${PROTO_DIR}/engine.proto
    DEPENDS ${PROTO_DIR}/engine.proto
    COMMENT "Generating C++ protobuf/gRPC code for engine.proto"
    VERBATIM
)
add_custom_target(generate_proto DEPENDS ${PROTO_SRCS} ${GRPC_SRCS})

# Shared library: everything except main(), so tests can link it too.
add_library(engine_core STATIC
    src/scheduler.cpp
    src/kv_cache.cpp
    src/mock_generator.cpp
    src/grpc_service.cpp
    ${PROTO_SRCS}
    ${GRPC_SRCS}
)
add_dependencies(engine_core generate_proto)

if(WITH_LLAMA)
    target_sources(engine_core PRIVATE src/llama_generator.cpp)
    target_compile_definitions(engine_core PUBLIC WITH_LLAMA)
    target_link_libraries(engine_core PUBLIC llama)
endif()

target_include_directories(engine_core PUBLIC src ${GEN_DIR})
target_link_libraries(engine_core PUBLIC
    gRPC::grpc++
    protobuf::libprotobuf
    Threads::Threads
)

# Engine executable
add_executable(engine src/main.cpp)
target_link_libraries(engine PRIVATE engine_core)

# Tests
enable_testing()
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

add_executable(tests
    tests/scheduler_test.cpp
    tests/kv_cache_test.cpp
)
target_include_directories(tests PRIVATE src ${GEN_DIR})
target_link_libraries(tests PRIVATE
    engine_core
    GTest::gtest_main
    Threads::Threads
)
include(GoogleTest)
gtest_discover_tests(tests)
