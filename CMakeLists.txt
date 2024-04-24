cmake_minimum_required(VERSION 3.13)

SET (CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++2a -mcmodel=large -DUSE_CPU_ONLY")

# ソースコードの検索
file(GLOB_RECURSE SOURCES src/*.cpp)
message("# SOURCES: ${SOURCES}")

include_directories(src)

# プロジェクトの設定
project(swan CXX)
add_executable(swan ${SOURCES})
