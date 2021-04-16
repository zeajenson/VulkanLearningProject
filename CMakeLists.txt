cmake_minimum_required(VERSION 3.12.0)

project(VulkanLearningProject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

find_package(Vulkan REQUIRED FATAL_ERROR)
find_package(glfw3 3.3 REQUIRED FATAL_ERROR) 

add_executable(VulkanLearningProject main.cpp) 

target_link_libraries(VulkanLearningProject glfw Vulkan::Vulkan)
