#include<variant>
#include<vector>
#include<ranges>
#include<string>
#include<memory>

#include<vulkan/vulkan.hpp>

struct VulkanState{
    vk::UniqueInstance const instance;
    std::vector<std::string> const requiredLayerNames;
    std::vector<std::string> const requiredExtensionNames;
    vk::PhysicalDevice const gpu;
    vk::UniqueDevice const device;
};

auto createInstance(std::vector<std::string> const & requiredLayerNames, std::vector<std::string> const & requiredExtensionNames){
    auto const layerProperties = vk::enumerateInstanceLayerProperties();

    if(requiredLayerNames | std::views::all([](auto name){return true;}))

    auto const extensions = vk::enumerateInstanceExtensionProperties();

    auto const appInfo = vk::ApplicationInfo("weewoo wo", 0, "", 0, VK_API_VERSION_1_2);
    return vk::createInstanceUnique(vk::InstanceCreateInfo({}, &appInfo));
}

vk::PhysicalDevice chooseGpu();

auto initVulkanState(){
    return VulkanState{
        createInstance(std::vector<std::string>(), std::vector<std::string>())
    };
}
