cmake_minimum_required(VERSION 3.3)

SET (CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++2a -mcmodel=large")
add_definitions(-DUSE_CPU_ONLY)

# ソースコードの検索
file(GLOB_RECURSE SOURCES src/context.cpp src/decode.cpp src/main.cpp src/tensor.cpp src/vocab.cpp src/weight.cpp src/context.hpp src/decode.hpp src/tensor.hpp src/vocab.hpp src/weight.hpp)
message("# SOURCES: ${SOURCES}")

include_directories(src)

# プロジェクトの設定
project(swan CXX)
add_executable(swan ${SOURCES})
