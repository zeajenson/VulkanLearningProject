#include<iostream>
#include<ranges>

#include<vulkan/vulkan.hpp>
#include<GLFW/glfw3.h>

#include"VulkanStuff.hpp"

int main(){

    if(!glfwInit() && !glfwVulkanSupported()){
        std::terminate();
    }

    auto window = glfwCreateWindow(690, 420, "WeeWoo", nullptr, nullptr);


    auto const vulkanState = initVulkanState();



    for(;!glfwWindowShouldClose(window);){
        glfwPollEvents();
    }

    glfwTerminate();
}
