cmake_minimum_required(VERSION 3.16)
project(iov_ids CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

# Shared code used by the agent, tools and tests
add_library(iov_common common/features.cpp common/file_source.cpp)
target_include_directories(iov_common PUBLIC common agent)

add_executable(agent agent/main.cpp agent/socketcan_source.cpp)
target_link_libraries(agent PRIVATE iov_common)

add_executable(feature_dump tools/feature_dump.cpp)
target_link_libraries(feature_dump PRIVATE iov_common)

# ---- Unit tests (GoogleTest). Uses the system package (libgtest-dev);
# falls back to downloading it if not installed.
find_package(GTest QUIET)
if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz)
  FetchContent_MakeAvailable(googletest)
endif()
enable_testing()
add_executable(unit_tests tests/test_sliding_window.cpp tests/test_features.cpp)
target_link_libraries(unit_tests PRIVATE iov_common GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(unit_tests)
