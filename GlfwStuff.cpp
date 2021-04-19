#ifndef GlfwStuff_cpp
#define GlfwStuff_cpp

#include<memory>

#include<GLFW/glfw3.h>

struct UniqueGlfwWindowDestroyer{
    void operator()(GLFWwindow * window){
         glfwDestroyWindow(window);
         glfwTerminate();
    }
};

using UniqueGlfwWindow = std::unique_ptr<GLFWwindow, UniqueGlfwWindowDestroyer>;

auto createGlfwWindowUnique(){
    if(glfwInit() != GLFW_TRUE) throw std::exception();
    if(glfwVulkanSupported() != GLFW_TRUE) throw std::exception{};
    
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    return UniqueGlfwWindow(glfwCreateWindow(690, 420, "WeeWoo", nullptr, nullptr));
}

#endif



