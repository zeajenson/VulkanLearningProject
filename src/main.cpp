#include<iostream>
#include<ranges>

#include<vulkan/vulkan.hpp>
#include<GLFW/glfw3.h>

int main(){

    if(!glfwInit() && !glfwVulkanSupported()){
        std::terminate();
    }

    auto window = glfwCreateWindow(690, 420, "WeeWoo", nullptr, nullptr);

    auto const appInfo = vk::ApplicationInfo("",0,"",0,VK_API_VERSION_1_2);
    auto const instance =  vk::createInstanceUnique(vk::InstanceCreateInfo({}, &appInfo));



    for(;!glfwWindowShouldClose(window);){
        glfwPollEvents();
    }

    glfwTerminate();
}
