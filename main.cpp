#include<iostream>
#include<ranges>
#include<fstream>
#include<filesystem>

#include<vulkan/vulkan.hpp>
#include<GLFW/glfw3.h>

#include"VulkanStuff.cpp"

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
}

int main(){

    if(!glfwInit() && !glfwVulkanSupported()){
        std::terminate();
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    auto window = glfwCreateWindow(690, 420, "WeeWoo", nullptr, nullptr);

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
        
        if(glfwCreateWindowSurface(instance.get(), window, nullptr, &surface) != VK_SUCCESS){
            std::cout << "unalbe to create window surface" << std::endl;
            std::terminate();
        }

        return vk::UniqueSurfaceKHR(vk::SurfaceKHR(surface));
    }();
    
    auto const gpu = instance->enumeratePhysicalDevices().back();

    auto const [
        graphicsIndex,
        presentIndex
    ] = [&]{
        std::optional<int> graphicsIndex{std::nullopt};
        std::optional<int> presentIndex{std::nullopt};
        auto const queueFamilyProperties = gpu.getQueueFamilyProperties();

        for(int i =0; i < queueFamilyProperties.size(); i++){
            auto const & property = queueFamilyProperties[i];
            if(!graphicsIndex && property.queueFlags & vk::QueueFlagBits::eGraphics){
                graphicsIndex = i;
            }

            if(!presentIndex && gpu.getSurfaceSupportKHR(i, surface.get())){
                presentIndex = i;
            }

            if(graphicsIndex && presentIndex){
                break;
            }
        }

        struct{
            int graphicsIndex, displayIndex;
        } indices{graphicsIndex.value(), presentIndex.value()};

        return indices;
    }(); 

    auto const graphicsQueuesPrioraties = std::vector<float>{1.0f};

    auto const queueCreateInfos = std::vector<vk::DeviceQueueCreateInfo>{
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

    auto const capabilities = gpu.getSurfaceCapabilitiesKHR(surface.get());

    //TODO: write functions to find these.
    auto const surfaceFormat = gpu.getSurfaceFormatsKHR(surface.get()).back(); 
    auto const presentMode = gpu.getSurfacePresentModesKHR(surface.get()).back(); 

    auto const extent = [&]{
        if(capabilities.currentExtent.width != UINT32_MAX) return capabilities.currentExtent;

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        auto const & minImageExtent = capabilities.minImageExtent;
        auto const & maxImageExtent = capabilities.maxImageExtent;

        return vk::Extent2D{
            std::clamp(static_cast<uint32_t>(width), minImageExtent.width, maxImageExtent.width),
            std::clamp(static_cast<uint32_t>(height), minImageExtent.height, maxImageExtent.height)
        };
    }();

    auto const imageCount = capabilities.minImageCount + 1;

    auto const swapchainImageFormat = surfaceFormat.format;

    auto const createSwapchain = [&]()->vk::UniqueSwapchainKHR{
        auto info = vk::SwapchainCreateInfoKHR{};
        info.surface = surface.get();
        info.minImageCount = imageCount;
        info.imageFormat = swapchainImageFormat;
        info.imageColorSpace = surfaceFormat.colorSpace;
        info.imageExtent = extent;
        info.imageArrayLayers = 1;
        info.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

        if(graphicsIndex == presentIndex){
            auto const indices = std::vector{
                static_cast<uint32_t>(graphicsIndex), 
                static_cast<uint32_t>(presentIndex)
            };

            info.imageSharingMode = vk::SharingMode::eConcurrent;
            info.setQueueFamilyIndices(indices);  
        }else{
            info.imageSharingMode = vk::SharingMode::eExclusive;
        }

        info.preTransform = capabilities.currentTransform;
        info.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
        info.presentMode = presentMode;
        info.clipped = VK_TRUE;

        info.oldSwapchain = nullptr;

        return device->createSwapchainKHRUnique(info);
    };

    auto const swapchain = createSwapchain();

    auto const swapchainImages = device->getSwapchainImagesKHR(swapchain.get());

    auto const swapchainImageViews = [&]{
        auto imageViews = std::vector<vk::UniqueImageView>(swapchainImages.size());

        auto const swizIdent = vk::ComponentSwizzle::eIdentity;

        for(auto const & image: swapchainImages)
            imageViews.push_back(device->createImageViewUnique(vk::ImageViewCreateInfo(
                {}, 
                image, 
                vk::ImageViewType::e2D, 
                surfaceFormat.format, 
                vk::ComponentMapping(
                    swizIdent, 
                    swizIdent, 
                    swizIdent, 
                    swizIdent), 
                vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1))));

        return imageViews;
    }();

    auto const renderPass = createRenderPass(device, swapchainImageFormat);
    auto const pipelineLayout = device->createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo());
    auto const renderPipeline = createGraphicsPipeline(device, renderPass, pipelineLayout, extent);

    for(;!glfwWindowShouldClose(window);){
        
        glfwPollEvents();
    }

    glfwTerminate();
}
