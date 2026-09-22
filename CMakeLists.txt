cmake_minimum_required(VERSION 3.14)
project(engine)

find_package(Vulkan REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(engine)
add_subdirectory(src)
add_subdirectory(external)
