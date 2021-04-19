#include<iostream>
#include<ranges>
#include<fstream>
#include<filesystem>

#include<vulkan/vulkan.hpp>
#include<GLFW/glfw3.h>
#include<glm/glm.hpp>

#include"VulkanStuff.cpp"
#include"GlfwStuff.cpp"

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
}

int main(){
    auto const window = createGlfwWindowUnique();

    auto const appInfo = vk::ApplicationInfo("",0,"",0,VK_API_VERSION_1_2);

    uint32_t glfwExtensionCount = 0;
    auto const glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    auto const layers = std::vector<const char *>{"VK_LAYER_KHRONOS_validation"};

    auto const extensions = [&glfwExtensions, &glfwExtensionCount]{
        auto extensions = std::vector<const char *>(glfwExtensions, glfwExtensions + glfwExtensionCount);
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        return extensions;
    }();

    auto const instance = vk::createInstanceUnique(vk::InstanceCreateInfo({}, &appInfo, layers, extensions));

    auto const mesenger = instance->createDebugUtilsMessengerEXTUnique(
        vk::DebugUtilsMessengerCreateInfoEXT(
            {}, 
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning, 
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation, 
            debugCallback),
        nullptr, 
        vk::DispatchLoaderDynamic(instance.get(), vkGetInstanceProcAddr));


    auto const surface = [&]{
        VkSurfaceKHR surface;
        
        if(glfwCreateWindowSurface(instance.get(), window.get(), nullptr, &surface) != VK_SUCCESS){
            std::cout << "unalbe to create window surface" << std::endl;
            std::terminate();
        }

        return vk::UniqueSurfaceKHR(
                vk::SurfaceKHR(surface), 
                vk::ObjectDestroy<vk::Instance, vk::DispatchLoaderStatic>(instance.get()));
    }();
    
    auto const gpu = instance->enumeratePhysicalDevices().back();

    auto const [
        graphicsIndex,
        presentIndex
    ] = findQueueIndices(surface, gpu);

    auto const graphicsQueuesPrioraties = std::vector{1.0f};

    auto const queueCreateInfos = std::vector{
        vk::DeviceQueueCreateInfo({}, graphicsIndex, graphicsQueuesPrioraties),
    };
    
    //TODO: make sure the device supports this before going any further
    auto const deviceExtensions = std::vector<const char *>{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    auto const deviceFeatures = vk::PhysicalDeviceFeatures{};

    auto const device = gpu.createDeviceUnique(vk::DeviceCreateInfo({}, 
        queueCreateInfos, 
        layers, 
        deviceExtensions, 
        &deviceFeatures));

    auto const graphicsQueue = device->getQueue(graphicsIndex, 0);

    auto renderState = createVulkanRenderState(device, gpu, surface, window, graphicsIndex, presentIndex);
    auto frameResized = false;
    
    glfwSetWindowUserPointer(window.get(), &frameResized);
    glfwSetFramebufferSizeCallback(window.get(), [](GLFWwindow * window, int width, int height){
        (*(bool *)glfwGetWindowUserPointer(window)) = true;
    });

    auto const maxFramesInFlight = 2;
    auto perFrameSync = createSynchronization(device, maxFramesInFlight);
    auto imagesInFlight = std::vector<vk::Fence>(renderState->swapchainImageViews.size(), nullptr);
    auto currentFrame = 0;

    auto const presentQueue = device->getQueue(presentIndex, 0);



    auto drawFrame = [&]{
        if(device->waitForFences(1, &perFrameSync[currentFrame].inFlightFence.get(), VK_TRUE, UINT64_MAX) != vk::Result::eSuccess)
            std::cout << "unable to wait for fence: " << currentFrame << std::endl;

        auto const imageIndex = device->acquireNextImageKHR(
                renderState->swapchain.get(), 
                UINT64_MAX, 
                perFrameSync[currentFrame].imageAvailableSemaphore.get(), 
                perFrameSync[currentFrame].inFlightFence.get());
        if(imageIndex.result not_eq vk::Result::eSuccess)
            throw std::runtime_error("failed to present swapchain image.");

        if(imageIndex.result == vk::Result::eErrorOutOfDateKHR or imageIndex.result == vk::Result::eSuboptimalKHR or frameResized){
            renderState = createVulkanRenderState(device, gpu, surface, window, graphicsIndex, presentIndex);
            frameResized = false;
            return;
        }
        
        if(imagesInFlight[imageIndex.value])
            if(device->waitForFences(1, &imagesInFlight[imageIndex.value], VK_TRUE, UINT64_MAX) != vk::Result::eSuccess)
                std::cout << "Unable to wait for image in flight fence: " << imageIndex.value << std::endl;

        //TODO: this is a weird hack instead two different indapendendet sets of fences should exist for sync.
        imagesInFlight[imageIndex.value] = perFrameSync[currentFrame].inFlightFence.get();

        vk::Semaphore waitSemaphores[] = {perFrameSync[currentFrame].imageAvailableSemaphore.get()};
        vk::PipelineStageFlags waitDstStageMasks[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};

        vk::Semaphore signalSemaphores[] = {perFrameSync[currentFrame].renderFinishedSemaphore.get()};

        vk::CommandBuffer graphicsQueueCommandBuffers[] = {renderState->commandBuffers[imageIndex.value].get()};

        auto const submitInfo = vk::SubmitInfo(
                1, waitSemaphores, 
                waitDstStageMasks,
                1, graphicsQueueCommandBuffers,
                1, signalSemaphores);

        if(device->resetFences(1, &perFrameSync[currentFrame].inFlightFence.get()) != vk::Result::eSuccess)
            std::cout << "Unable to reset fence: " << currentFrame << std::endl;

        if(graphicsQueue.submit(1, &submitInfo, perFrameSync[currentFrame].inFlightFence.get()) != vk::Result::eSuccess)
            std::cerr << "Bad submit" << std::endl;
        
        auto const presentInfo = vk::PresentInfoKHR(
                1, signalSemaphores, 
                1, &renderState->swapchain.get(), 
                &imageIndex.value);

        if(presentQueue.presentKHR(presentInfo) != vk::Result::eSuccess){
            std::cerr << "Bad present" << std::endl;
        }

        presentQueue.waitIdle();

        currentFrame = (currentFrame + 1) % maxFramesInFlight;
    };

    for(;not glfwWindowShouldClose(window.get());){
        drawFrame();
        glfwPollEvents();
    }
}

