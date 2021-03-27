cmake_minimum_required(VERSION 3.12.0)

project(VulkanLearningProject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

add_executable(VulkanLearningProject src/main.cpp)

target_link_libraries(VulkanLearningProject)
