cmake_minimum_required(VERSION 3.12.0)

project(VulkanLearningProject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

find_package(Vulkan REQUIRED FATAL_ERROR)
find_package(glfw REQUIRED FATAL_ERROR) 

add_executable(VulkanLearningProject src/main.cpp glfw ${VULKAN_LIB_LIST})

target_link_libraries(VulkanLearningProject)
