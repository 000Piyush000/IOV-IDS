cmake_minimum_required(VERSION 3.16)
project(iov_ids CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

# AddressSanitizer + UndefinedBehaviorSanitizer: catch memory bugs and UB in tests (CI job)
option(ENABLE_SANITIZERS "Build with ASan/UBSan" OFF)
if(ENABLE_SANITIZERS)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
endif()

# ---- ONNX Runtime (prebuilt release unpacked into third_party/onnxruntime)
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH "ONNX Runtime root")
if(NOT EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "ONNX Runtime not found in ${ONNXRUNTIME_DIR}. Run tools/get_onnxruntime.sh first.")
endif()
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_DIR}/include")
# let executables find libonnxruntime.so at runtime without LD_LIBRARY_PATH
set(CMAKE_BUILD_RPATH "${ONNXRUNTIME_DIR}/lib")

# ---- Shared code (no ONNX dependency)
add_library(iov_common common/features.cpp common/file_source.cpp common/protocol.cpp)
target_include_directories(iov_common PUBLIC common agent)

# ---- Detector (ONNX Runtime wrapper)
add_library(iov_detector common/detector.cpp)
target_link_libraries(iov_detector PUBLIC iov_common PRIVATE onnxruntime)

add_executable(agent agent/main.cpp agent/socketcan_source.cpp agent/tcp_client.cpp)
target_link_libraries(agent PRIVATE iov_detector)

add_executable(edge_server server/edge_server.cpp)
target_link_libraries(edge_server PRIVATE iov_common)

add_executable(fl_client fl/fl_client.cpp)
target_link_libraries(fl_client PRIVATE iov_common)

add_executable(feature_dump tools/feature_dump.cpp)
target_link_libraries(feature_dump PRIVATE iov_common)

add_executable(bench_infer tools/bench_infer.cpp)
target_link_libraries(bench_infer PRIVATE iov_detector)

# ---- Unit tests (GoogleTest from libgtest-dev; download as fallback)
find_package(GTest QUIET)
if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz)
  FetchContent_MakeAvailable(googletest)
endif()
enable_testing()
add_executable(unit_tests tests/test_sliding_window.cpp tests/test_features.cpp tests/test_protocol.cpp tests/test_zero_day.cpp)
target_link_libraries(unit_tests PRIVATE iov_common GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(unit_tests)
