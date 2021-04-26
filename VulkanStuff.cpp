#include<vector>
#include<iostream>
#include<fstream>
#include<filesystem>

#include<vulkan/vulkan.hpp>
#include<GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include<stb_image.h>

#include<tiny_obj_loader.h>

#include"GlfwStuff.cpp"

auto createDebugMessanger(
        vk::UniqueInstance const & instance, 
        PFN_vkDebugUtilsMessengerCallbackEXT debugCallback)
{
    return instance->createDebugUtilsMessengerEXTUnique(
        vk::DebugUtilsMessengerCreateInfoEXT(
            {}, 
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning, 
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation, 
            debugCallback),
        nullptr, 
        vk::DispatchLoaderDynamic(instance.get(), vkGetInstanceProcAddr));
}

auto createSurface(vk::UniqueInstance const & instance, UniqueGlfwWindow const & window){
    VkSurfaceKHR surface;
    if(glfwCreateWindowSurface(instance.get(), window.get(), nullptr, &surface) != VK_SUCCESS)
        throw std::runtime_error("unalbe to create window surface");
    
    return vk::UniqueSurfaceKHR(
            vk::SurfaceKHR(surface), 
            vk::ObjectDestroy<vk::Instance, vk::DispatchLoaderStatic>(instance.get()));
}

auto findQueueIndices(vk::UniqueSurfaceKHR const & surface, vk::PhysicalDevice const & gpu){
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
}

auto createExtent(vk::SurfaceCapabilitiesKHR const & capabilities, UniqueGlfwWindow const & window){
    if(capabilities.currentExtent.width != UINT32_MAX) return capabilities.currentExtent;
    
    auto const & minImageExtent = capabilities.minImageExtent;
    auto const & maxImageExtent = capabilities.maxImageExtent;
   
    int width, height;
    glfwGetFramebufferSize(window.get(), &width, &height);

    return vk::Extent2D{
        std::clamp(static_cast<uint32_t>(width), minImageExtent.width, maxImageExtent.width),
        std::clamp(static_cast<uint32_t>(height), minImageExtent.height, maxImageExtent.height)
    };
}

auto createSwapchain(
        vk::UniqueDevice const & device,
        vk::Extent2D const & extent,
        vk::UniqueSurfaceKHR const & surface,
        uint32_t imageCount,
        vk::SurfaceFormatKHR const & surfaceFormat,
        vk::PresentModeKHR const & presentMode,
        vk::SurfaceCapabilitiesKHR const & capabilities,
        uint32_t graphicsIndex,
        uint32_t presentIndex)
{
    auto info = vk::SwapchainCreateInfoKHR{};
    info.surface = surface.get();
    info.minImageCount = imageCount;
    info.imageFormat = surfaceFormat.format;
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
}

auto create_image_view(
        vk::Device const device, 
        vk::Image const image, 
        vk::Format const format, 
        vk::ImageAspectFlags const aspectFlags){
    return device.createImageViewUnique(
            vk::ImageViewCreateInfo({}, 
                image, 
                vk::ImageViewType::e2D, 
                format, 
                {}, 
                vk::ImageSubresourceRange(
                    aspectFlags, 
                    0, 
                    1, 
                    0, 
                    1)));
}

auto createSwapchainImageViews(
        vk::UniqueDevice const & device,
        std::vector<vk::Image> const & swapchainImages, 
        vk::SurfaceFormatKHR surfaceFormat){
    auto imageViews = std::vector<vk::UniqueImageView>();
    imageViews.reserve(swapchainImages.size());
    
    auto const swizIdent = vk::ComponentSwizzle::eIdentity;
    
    for(auto const & image: swapchainImages)
        imageViews.push_back(create_image_view(
                    device.get(), 
                    image, 
                    surfaceFormat.format, 
                    vk::ImageAspectFlagBits::eColor));
    
    return imageViews;
}

auto loadShader(std::filesystem::path path){
    auto file = std::ifstream(path.string(), std::ios::ate | std::ios::binary);
    auto const fileSize = (size_t) file.tellg();
    auto buffer = std::vector<char>(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);

    return buffer;
}

auto createShaderModule(std::filesystem::path path, vk::Device const device){
    auto const shaderCode = loadShader(path);
    return device.createShaderModuleUnique(vk::ShaderModuleCreateInfo({}, 
            shaderCode.size(), 
            reinterpret_cast<uint32_t const *>(shaderCode.data())));
}

auto find_supported_format(
        vk::PhysicalDevice const gpu,
        std::vector<vk::Format> const & formats, 
        vk::ImageTiling tiling, 
        vk::FormatFeatureFlags features) 
    -> std::optional<vk::Format>
{
    for(auto const & format : formats){
        auto const props = gpu.getFormatProperties(format);

        if(tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features)
            return format;

        if(tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features)
            return format;
    }

    return std::nullopt;
}

auto createRenderPass(vk::UniqueDevice const & device, vk::PhysicalDevice const gpu, vk::Format format){
    auto const collorAttachment = vk::AttachmentDescription({}, 
            format, 
            vk::SampleCountFlagBits::e1, 
            vk::AttachmentLoadOp::eClear, 
            vk::AttachmentStoreOp::eStore, 
            //Stencil
            vk::AttachmentLoadOp::eDontCare, 
            vk::AttachmentStoreOp::eDontCare, 
            vk::ImageLayout::eUndefined, 
            vk::ImageLayout::ePresentSrcKHR);

    auto const colorAttachmentRef = vk::AttachmentReference(0, vk::ImageLayout::eColorAttachmentOptimal);

    auto const depthFormat = find_supported_format(
            gpu,
            {
                vk::Format::eD32Sfloat,
                vk::Format::eD32SfloatS8Uint,
                vk::Format::eD24UnormS8Uint
            },
            vk::ImageTiling::eOptimal,
            vk::FormatFeatureFlagBits::eDepthStencilAttachment);

    if(not depthFormat) throw std::runtime_error("Missing depth format");

    auto const depthAttachment = vk::AttachmentDescription({}, 
            depthFormat.value(), 
            vk::SampleCountFlagBits::e1, 
            vk::AttachmentLoadOp::eClear, 
            vk::AttachmentStoreOp::eStore, 
            vk::AttachmentLoadOp::eDontCare, 
            vk::AttachmentStoreOp::eDontCare, 
            vk::ImageLayout::eUndefined, 
            vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal);

    auto const depthAttachmentRef = vk::AttachmentReference(
            1, 
            vk::ImageLayout::eDepthStencilAttachmentOptimal);
    
    auto const subpass = vk::SubpassDescription({}, 
            vk::PipelineBindPoint::eGraphics, 
            {}, 
            colorAttachmentRef, 
            {}, 
            &depthAttachmentRef);  

    auto const subpassDependency = vk::SubpassDependency(
            VK_SUBPASS_EXTERNAL, 
            0, 
            vk::PipelineStageFlagBits::eColorAttachmentOutput 
            | vk::PipelineStageFlagBits::eEarlyFragmentTests, 
            vk::PipelineStageFlagBits::eColorAttachmentOutput
            | vk::PipelineStageFlagBits::eEarlyFragmentTests, 
            {}, 
            vk::AccessFlagBits::eColorAttachmentWrite
            | vk::AccessFlagBits::eDepthStencilAttachmentWrite);

    auto const attachments = std::array{collorAttachment, depthAttachment};
    return device->createRenderPassUnique(vk::RenderPassCreateInfo({}, 
                attachments, 
                subpass, 
                subpassDependency));
}

struct Vertex{
    glm::vec3 position;
    glm::vec3 color;
    glm::vec2 texCoord;
};

auto createVertexBindingDescritptions(){
    
    auto const bindingDescription = vk::VertexInputBindingDescription(0, sizeof(Vertex), vk::VertexInputRate::eVertex);

    auto attributes = std::array<vk::VertexInputAttributeDescription, 3>();
    auto & position = attributes[0];
    auto & color = attributes[1];
    auto & texCoord = attributes[2];

    position.binding = 0;
    position.location = 0;
    position.format = vk::Format::eR32G32B32Sfloat;
    position.offset = offsetof(Vertex, position);

    color.binding = 0;
    color.location = 1;
    color.format = vk::Format::eR32G32B32Sfloat;
    color.offset = offsetof(Vertex, color);

    texCoord.binding = 0;
    texCoord.location = 2;
    texCoord.format = vk::Format::eR32G32Sfloat;
    texCoord.offset = offsetof(Vertex, texCoord);


    struct{
        std::array<vk::VertexInputBindingDescription, 1> bindingDescriptions;
        std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions;
    } bindings {
        std::array{bindingDescription},
        attributes
    };

    return bindings;
}

struct UniformBufferObject{
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
}; 

auto create_descriptor_set_layout(vk::Device const device) {
    auto const uboBinding = vk::DescriptorSetLayoutBinding(
            0, 
            vk::DescriptorType::eUniformBuffer, 
            1,
            vk::ShaderStageFlagBits::eVertex,
            nullptr);

    auto const samplerBinding = vk::DescriptorSetLayoutBinding(
            1,
            vk::DescriptorType::eCombinedImageSampler,
            1, vk::ShaderStageFlagBits::eFragment, 
            nullptr);

    auto const descriptorSets = std::array{uboBinding, samplerBinding};

    return device.createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo({}, descriptorSets));
}

auto createGraphicsPipeline(
        vk::Device const & device, 
        vk::RenderPass const & renderPass, 
        vk::PipelineLayout const & layout,
        vk::Extent2D const & swapchainExtent) 
{
    auto const vertShaderModule = createShaderModule("./vert.spv", device);
    auto const vertShaderStageInfo = vk::PipelineShaderStageCreateInfo({}, 
            vk::ShaderStageFlagBits::eVertex, 
            vertShaderModule.get(), 
            "main");

    auto const fragShaderModule = createShaderModule("./frag.spv", device);
    auto const fragShaderStageInfo = vk::PipelineShaderStageCreateInfo({}, 
            vk::ShaderStageFlagBits::eFragment, 
            fragShaderModule.get(), 
            "main");

    auto const shaderStages = std::vector{vertShaderStageInfo, fragShaderStageInfo};

    auto const [
        bindingDescriptions,
        attributeDescriptions
    ] = createVertexBindingDescritptions();

    auto const vertexInputInfo = vk::PipelineVertexInputStateCreateInfo({}, bindingDescriptions, attributeDescriptions);

    auto const inputAssemblyInfo = vk::PipelineInputAssemblyStateCreateInfo({}, vk::PrimitiveTopology::eTriangleList, VK_FALSE);

    auto const viewport = vk::Viewport(0.0f, 0.0f, swapchainExtent.width, swapchainExtent.height, 0.0f, 1.0f);
    auto const scissor = vk::Rect2D({0,0}, swapchainExtent);
    auto const viewportStateInfo = vk::PipelineViewportStateCreateInfo({}, 1, &viewport, 1, &scissor);


    auto const rasterizer = vk::PipelineRasterizationStateCreateInfo({}, 
            VK_FALSE, 
            VK_FALSE, 
            vk::PolygonMode::eFill, 
            vk::CullModeFlagBits::eBack, 
            vk::FrontFace::eCounterClockwise, 
            VK_FALSE);

    auto const multisampleing = vk::PipelineMultisampleStateCreateInfo({},
            vk::SampleCountFlagBits::e1, VK_FALSE, 1.0f, nullptr, VK_FALSE, VK_FALSE);

    //per frame buffer.
    auto const colorBlendAttachment = vk::PipelineColorBlendAttachmentState(
            VK_FALSE, 
            vk::BlendFactor::eOne, 
            {}, 
            {}, 
            vk::BlendFactor::eOne, 
            {}, 
            {}, 
            vk::ColorComponentFlagBits::eR 
            | vk::ColorComponentFlagBits::eB 
            | vk::ColorComponentFlagBits::eG 
            | vk::ColorComponentFlagBits::eA);


    auto const blendConstants = std::array{0.0f, 0.0f, 0.0f, 0.0f};
    auto const colorBlending = vk::PipelineColorBlendStateCreateInfo({}, 
            VK_FALSE, 
            vk::LogicOp::eCopy, 
            1, & colorBlendAttachment, 
            blendConstants);

    auto const dynamicStates = std::array{
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
        vk::DynamicState::eLineWidth
    };
    auto const dynamicState = vk::PipelineDynamicStateCreateInfo({}, dynamicStates);

    auto const depthStencil = vk::PipelineDepthStencilStateCreateInfo({}, 
            VK_TRUE, 
            VK_TRUE, 
            vk::CompareOp::eLess, 
            VK_FALSE, 
            VK_FALSE, 
            {}, 
            {}, 
            0.0f, 
            1.0f);

    auto const pipelineCreateInfo = [&]{
        auto pipelineCreateInfo = vk::GraphicsPipelineCreateInfo{};
        pipelineCreateInfo
            .setStages(shaderStages)
            .setPVertexInputState(&vertexInputInfo)
            .setPInputAssemblyState(&inputAssemblyInfo)
            .setPViewportState(&viewportStateInfo)
            .setPRasterizationState(&rasterizer)
            .setPMultisampleState(&multisampleing)
            .setPColorBlendState(&colorBlending)
            .setPDynamicState(&dynamicState)
            .setPDepthStencilState(&depthStencil)
            .setLayout(layout)
            .setRenderPass(renderPass);

        return pipelineCreateInfo;
    }();

    auto const createInfos = std::array{pipelineCreateInfo};

    //TODO: check return result and see why creation may have failed.
    return device.createGraphicsPipelinesUnique({}, createInfos).value;
}

auto createFrameBuffers(
        vk::UniqueDevice const & device, 
        std::vector<vk::UniqueImageView> const & swapchainImageViews,
        vk::UniqueImageView const & depthImageView,
        vk::UniqueRenderPass const & renderPass,
        vk::Extent2D const & extent)
{
    auto frameBuffers = std::vector<vk::UniqueFramebuffer>();
    frameBuffers.reserve(swapchainImageViews.size());
    for(auto const & imageView : swapchainImageViews){
        
        auto const attachments = std::vector{ 
            imageView.get(), 
            depthImageView.get() 
        };

        frameBuffers.push_back(device->createFramebufferUnique(
                    vk::FramebufferCreateInfo({}, 
                        renderPass.get(), 
                        attachments, 
                        extent.width, 
                        extent.height, 
                        1)));
    }

    return frameBuffers;
}

struct Synchronization{
    vk::UniqueSemaphore imageAvailableSemaphore;
    vk::UniqueSemaphore renderFinishedSemaphore;
    vk::UniqueFence inFlightFence;
};

auto createSynchronization(vk::UniqueDevice const & device, int32_t maxFramesInFlight){
    auto frameSync = std::vector<Synchronization>();
    frameSync.reserve(maxFramesInFlight);

    for(int i = 0; i < maxFramesInFlight; i++){
        frameSync.push_back(Synchronization{
                    device->createSemaphoreUnique(vk::SemaphoreCreateInfo()),
                    device->createSemaphoreUnique(vk::SemaphoreCreateInfo()),
                    device->createFenceUnique(vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled))
                });
    }

    return frameSync;
}

auto findMemoryType(vk::PhysicalDevice const & gpu, uint32_t memoryBitsRequirement, vk::MemoryPropertyFlags properties){
    auto memoryProperties = gpu.getMemoryProperties();

    for(uint32_t memoryIndex = 0; memoryIndex < memoryProperties.memoryTypeCount; memoryIndex++){
        if(memoryBitsRequirement & (1 << memoryIndex) && (memoryProperties.memoryTypes[memoryIndex].propertyFlags & properties) == properties)
            return memoryIndex;
    }

    throw std::runtime_error("failed to find memoryType");
}

auto createBuffer(
        vk::UniqueDevice const & device,
        vk::PhysicalDevice const & gpu,
        vk::DeviceSize const size, 
        vk::BufferUsageFlags const usage, 
        vk::MemoryPropertyFlags properties)
{
    auto buffer = device.get().createBufferUnique(vk::BufferCreateInfo({}, size, usage, vk::SharingMode::eExclusive));

    auto const memoryRequirements = device.get().getBufferMemoryRequirements(buffer.get());

    auto memory = device.get().allocateMemoryUnique(vk::MemoryAllocateInfo(
                memoryRequirements.size, 
                findMemoryType(gpu, memoryRequirements.memoryTypeBits, properties)));

    device.get().bindBufferMemory(buffer.get(), memory.get(), 0);

    struct handles{
        vk::UniqueBuffer buffer;
        vk::UniqueDeviceMemory bufferMemory;
    };

    return handles{
        std::move(buffer),
        std::move(memory)
    };
}

struct CommandScope{
    CommandScope(
            vk::UniqueDevice const & device, 
            vk::UniqueCommandPool const & commandPool, 
            vk::Queue const & graphicsQueue): 
        device(device),
        graphicsQueue(graphicsQueue)
    {

        commandBuffer = std::move(device->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(
                    commandPool.get(), 
                    vk::CommandBufferLevel::ePrimary, 
                    1)).back());

        commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    }
    ~CommandScope(){
        commandBuffer->end();

        graphicsQueue.submit(std::array{vk::SubmitInfo({}, {}, {}, 1, &commandBuffer.get())});

        device->waitIdle();
    }

    vk::UniqueDevice const & device;
    vk::Queue const & graphicsQueue;
    vk::UniqueCommandBuffer commandBuffer;
};

auto copyBuffer(
        vk::UniqueDevice const & device,
        vk::UniqueCommandPool const & commandPool,
        vk::Queue const & graphicsQueue,
        vk::UniqueBuffer const & srcBuffer, 
        vk::UniqueBuffer const & dstBuffer, 
        vk::DeviceSize size)
{
    auto commandScope = CommandScope(device, commandPool, graphicsQueue);
    commandScope.commandBuffer->copyBuffer(
            srcBuffer.get(),
            dstBuffer.get(), 
            std::array{vk::BufferCopy({}, {}, size)});

}

auto createVertexBuffer(
        vk::UniqueDevice const & device, 
        vk::PhysicalDevice const & gpu, 
        vk::UniqueCommandPool const & commandPool,
        vk::Queue const & graphicsQueue,
        std::vector<Vertex> const & vertices)
{ 
    auto const bufferSize = vk::DeviceSize(sizeof(Vertex) * vertices.size());

    auto const [
        hostBuffer,
        hostBufferMemory
    ] = createBuffer(
            device,
            gpu,
            bufferSize, 
            vk::BufferUsageFlagBits::eTransferSrc, 
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    //TODO: load vertices directly into staging memory, instead of allocating it twice.
    auto const memory = device->mapMemory(hostBufferMemory.get(), 0, bufferSize, {});
    memcpy(memory, vertices.data(), (size_t)bufferSize);
    device->unmapMemory(hostBufferMemory.get());

    auto bufferHandles = createBuffer(
            device,
            gpu,
            bufferSize, 
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, 
            vk::MemoryPropertyFlagBits::eDeviceLocal);

    copyBuffer(
            device, 
            commandPool, 
            graphicsQueue, 
            hostBuffer, 
            bufferHandles.buffer, 
            bufferSize);

    return bufferHandles;
}

auto createIndexBuffer(
        vk::UniqueDevice const & device, 
        vk::PhysicalDevice const & gpu, 
        vk::UniqueCommandPool const & commandPool,
        vk::Queue const & graphicsQueue,
        std::vector<uint32_t> const & indices)
{
    auto const bufferSize = vk::DeviceSize(sizeof(Vertex) * indices.size());

    auto const [
        hostBuffer,
        hostBufferMemory
    ] = createBuffer(
            device,
            gpu,
            bufferSize, 
            vk::BufferUsageFlagBits::eTransferSrc, 
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    //TODO: load indices directly into staging memory, instead of allocating it twice.
    auto const memory = device->mapMemory(hostBufferMemory.get(), 0, bufferSize, {});
    memcpy(memory, indices.data(), (size_t)bufferSize);
    device->unmapMemory(hostBufferMemory.get());

    auto bufferHandles = createBuffer(
            device,
            gpu,
            bufferSize, 
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, 
            vk::MemoryPropertyFlagBits::eDeviceLocal);

    copyBuffer(
            device, 
            commandPool, 
            graphicsQueue, 
            hostBuffer, 
            bufferHandles.buffer, 
            bufferSize);

    return bufferHandles;
}

auto createUniformBuffers(
        vk::UniqueDevice const & device, 
        vk::PhysicalDevice const & gpu, 
        vk::UniqueCommandPool const & commandPool,
        vk::Queue const & graphicsQueue,
        size_t swapchainImageCount)
{
    auto uniformBuffers = std::vector<std::pair<vk::UniqueBuffer, vk::UniqueDeviceMemory>>(); 
    uniformBuffers.reserve(swapchainImageCount);

    for(size_t i = 0; i < swapchainImageCount; i++){
        auto const bufferSize = vk::DeviceSize(sizeof(UniformBufferObject));

        auto [
            buffer,
            memory
        ] = createBuffer(
                device, 
                gpu, 
                bufferSize, 
                vk::BufferUsageFlagBits::eUniformBuffer, 
                vk::MemoryPropertyFlagBits::eHostVisible 
                | vk::MemoryPropertyFlagBits::eHostCoherent);
        
        uniformBuffers.push_back({
                std::move(buffer),
                std::move(memory)
            });
    }

    return uniformBuffers;
}

auto create_descriptor_pool(vk::Device const device, uint32_t imageCount) noexcept{
    auto const unieformBufferPoolSize = vk::DescriptorPoolSize(
            vk::DescriptorType::eUniformBuffer, imageCount);
    auto const textureSamplerPoolSize = vk::DescriptorPoolSize(
            vk::DescriptorType::eCombinedImageSampler, imageCount);

    auto const poolSizes = std::array{
        unieformBufferPoolSize, 
        textureSamplerPoolSize
    };

    auto const poolInfo = vk::DescriptorPoolCreateInfo(
            vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 
            imageCount, 
            poolSizes);
    
    return device.createDescriptorPoolUnique(poolInfo);
}

auto create_descriptor_sets(
        vk::Device const device, 
        vk::DescriptorPool const pool, 
        uint32_t imageCount, 
        vk::DescriptorSetLayout const layout,
        std::vector<std::pair<vk::UniqueBuffer, vk::UniqueDeviceMemory>> const & uniformBuffers,
        vk::ImageView const textureImageView,
        vk::Sampler const textureSampler)
{
    auto const layouts = std::vector(imageCount, layout);
    
    auto descriptorSets = device.allocateDescriptorSetsUnique(vk::DescriptorSetAllocateInfo(pool, layouts));

    for(size_t i =0; i < imageCount; i++){

        auto const uniformBuffer = uniformBuffers[i].first.get();
        auto const bufferInfo = vk::DescriptorBufferInfo(uniformBuffer, 0, sizeof(UniformBufferObject));

        auto const uniformWriteDescriptorSet = vk::WriteDescriptorSet(
                descriptorSets[i].get(), 
                0, 
                0, 
                vk::DescriptorType::eUniformBuffer, 
                {}, bufferInfo, {});

        auto const imageInfo = vk::DescriptorImageInfo(
                textureSampler, 
                textureImageView, 
                vk::ImageLayout::eShaderReadOnlyOptimal);

        auto const samplerWriteDescriptorSet = vk::WriteDescriptorSet(
                descriptorSets[i].get(), 
                1, 
                0, 
                vk::DescriptorType::eCombinedImageSampler, 
                imageInfo, {}, {}); 

        device.updateDescriptorSets({
                    uniformWriteDescriptorSet,
                    samplerWriteDescriptorSet
                }, {});
    }

    return std::move(descriptorSets);
}

struct ImageHandles{
    vk::UniqueImage image;
    vk::UniqueDeviceMemory memory;
};

auto transition_image_layout(
        vk::UniqueDevice const & device,
        vk::UniqueCommandBuffer const & commandBuffer,
        vk::UniqueImage const & image,
        vk::Format const & format,
        vk::ImageLayout const & oldLayout,
        vk::ImageLayout const & newLayout)
{
    auto barrier = vk::ImageMemoryBarrier(
            {},
            {},
            oldLayout,
            newLayout,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            image.get(),
            vk::ImageSubresourceRange(
                vk::ImageAspectFlagBits::eColor,
                0, 
                1, 
                0, 
                1)
            );

    auto sourceStage = vk::PipelineStageFlags{};
    auto destinationStage = vk::PipelineStageFlags{};

    if(oldLayout == vk::ImageLayout::eUndefined and newLayout == vk::ImageLayout::eTransferDstOptimal){
        barrier
            .setSrcAccessMask({})
            .setDstAccessMask(vk::AccessFlagBits::eTransferWrite);

        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;

    }
    else if(oldLayout == vk::ImageLayout::eTransferDstOptimal and newLayout == vk::ImageLayout::eShaderReadOnlyOptimal){
        barrier
            .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask(vk::AccessFlagBits::eShaderRead);

        sourceStage = vk::PipelineStageFlagBits::eTransfer;
        destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    }else{
        std::cerr << "Unsoported layout transition" << std::endl;
        std::terminate();
    }

    commandBuffer->pipelineBarrier(
            sourceStage, 
            destinationStage, 
            {}, 
            0, nullptr, 
            0, nullptr, 
            1, &barrier);
}

void copy_buffer_to_image(
        vk::UniqueCommandBuffer const & commandBuffer,
        vk::UniqueBuffer const & buffer, 
        vk::UniqueImage const & image, 
        uint32_t width, 
        uint32_t height)
{
    auto region = vk::BufferImageCopy(
            0, 
            0, 
            0, 
            vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1), 
            {0, 0, 0}, 
            {width, height, 1});

    commandBuffer->copyBufferToImage(
            buffer.get(), 
            image.get(), 
            vk::ImageLayout::eTransferDstOptimal, 
            1, &region);
}

auto create_image(        
        vk::UniqueDevice const & device,
        vk::PhysicalDevice const & gpu,
        vk::Format format,
        vk::ImageTiling tiling,
        vk::ImageUsageFlags usage,
        vk::MemoryPropertyFlags properties,
        int const width,
        int const height) 
{
    auto const imageInfo = vk::ImageCreateInfo({}, 
            vk::ImageType::e2D, 
            format, 
            vk::Extent3D(width, height, 1), 
            1, 
            1, 
            vk::SampleCountFlagBits::e1, 
            tiling,
            usage,
            vk::SharingMode::eExclusive, 
            {}, 
            vk::ImageLayout::eUndefined);

    auto image = device->createImageUnique(imageInfo);

    auto const memoryRequirements = device->getImageMemoryRequirements(image.get());
    auto imageMemory = device->allocateMemoryUnique(vk::MemoryAllocateInfo(
                memoryRequirements.size, 
                findMemoryType(
                    gpu,
                    memoryRequirements.memoryTypeBits, 
                    properties)));

    device->bindImageMemory(image.get(), imageMemory.get(), 0);

    return ImageHandles{
        std::move(image),
        std::move(imageMemory)
    };
}

struct pixelDeleter { void operator() (uint8_t * pixels){stbi_image_free(pixels);} };
using pixelsRef = std::unique_ptr<uint8_t, pixelDeleter>;

auto create_texture_image(        
        vk::UniqueDevice const & device,
        vk::PhysicalDevice const & gpu,
        vk::UniqueCommandPool const & commandPool,
        vk::Queue const & graphicsQueue,
        char const * filename) noexcept
{
    int width, height, channels;
    auto pixels = pixelsRef(stbi_load(filename, &width, &height, &channels, STBI_rgb_alpha));
    if(not pixels){
        std::cerr << "No pixels for texture: " << filename << std::endl;
    }

    auto const size = vk::DeviceSize(width * height * 4);
    auto [
        stagingBuffer,
        memory
    ] = createBuffer(
            device, 
            gpu, 
            size, 
            vk::BufferUsageFlagBits::eTransferSrc, 
            vk::MemoryPropertyFlagBits::eHostVisible 
            | vk::MemoryPropertyFlagBits::eHostCoherent);

    auto data = device->mapMemory(memory.get(), 0, size);
    memcpy(data, pixels.get(), static_cast<size_t>(size));
    device->unmapMemory(memory.get());

    auto [
        image,
        imageMemory
    ] = create_image(
            device,
            gpu,
            vk::Format::eR8G8B8A8Srgb,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst 
            | vk::ImageUsageFlagBits::eSampled, 
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            width, height);
    {
        auto commandScope = CommandScope(device, commandPool, graphicsQueue);
        transition_image_layout(
                device,
                commandScope.commandBuffer,
                image, 
                vk::Format::eR8G8B8A8Srgb, 
                vk::ImageLayout::eUndefined, 
                vk::ImageLayout::eTransferDstOptimal);

        copy_buffer_to_image(commandScope.commandBuffer, stagingBuffer, image, width, height);

        transition_image_layout(
                device, 
                commandScope.commandBuffer,
                image, 
                vk::Format::eR8G8B8A8Srgb, 
                vk::ImageLayout::eTransferDstOptimal, 
                vk::ImageLayout::eShaderReadOnlyOptimal);
    }

    return ImageHandles{
        std::move(image),
        std::move(imageMemory)
    };
}

auto create_texture_sampler(vk::Device const device, vk::PhysicalDevice const gpu){
    return device.createSamplerUnique(vk::SamplerCreateInfo()
            .setMagFilter(vk::Filter::eLinear)
            .setMinFilter(vk::Filter::eLinear)
            .setAddressModeU(vk::SamplerAddressMode::eRepeat)
            .setAddressModeV(vk::SamplerAddressMode::eRepeat)
            .setAddressModeW(vk::SamplerAddressMode::eRepeat)
            .setAnisotropyEnable(VK_TRUE)
            .setMaxAnisotropy(gpu.getProperties().limits.maxSamplerAnisotropy)
            .setBorderColor(vk::BorderColor::eIntOpaqueBlack)
            .setUnnormalizedCoordinates(VK_FALSE)
            .setCompareEnable(VK_FALSE)
            .setCompareOp(vk::CompareOp::eAlways)
            .setMipmapMode(vk::SamplerMipmapMode::eLinear)
            .setMipLodBias(0).setMinLod(0).setMaxLod(0));
}

auto has_stencil_component(vk::Format format){
    return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
}

struct DepthImageHandles{
    vk::UniqueImage image;
    vk::UniqueDeviceMemory imageMemory;
    vk::UniqueImageView imageView;
};

auto create_depth_resource(
        vk::UniqueDevice const & device, 
        vk::PhysicalDevice const & gpu,
        vk::Extent2D const & swapchainExtent)
{
    auto const depthFormat = find_supported_format(
            gpu,
            {
                vk::Format::eD32Sfloat,
                vk::Format::eD32SfloatS8Uint,
                vk::Format::eD24UnormS8Uint
            },
            vk::ImageTiling::eOptimal,
            vk::FormatFeatureFlagBits::eDepthStencilAttachment);

    if(not depthFormat) throw std::runtime_error("no format for depth image");

    //TODO: figure out why the tutorial pases in a memory usage bit
    auto [
        image,
        imageMemory
    ] = create_image(
            device, 
            gpu, 
            depthFormat.value(), 
            vk::ImageTiling::eOptimal, 
            vk::ImageUsageFlagBits::eDepthStencilAttachment, 
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            swapchainExtent.width, 
            swapchainExtent.height);
    
    auto imageView = create_image_view(
            device.get(), 
            image.get(), 
            depthFormat.value(), vk::ImageAspectFlagBits::eDepth);

    return DepthImageHandles{
        std::move(image),
        std::move(imageMemory),
        std::move(imageView)
    };
}

//TODO: don't use tiny obj loader. it's kinda bad.
auto load_model(char const * filename){
    auto attrib = tinyobj::attrib_t{};
    auto shapes = std::vector<tinyobj::shape_t>{};
    auto materials = std::vector<tinyobj::material_t>{};
    std::string warn, err;

    if(!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename))
        throw std::runtime_error(warn + err);

    auto vertices = std::vector<Vertex>{};
    vertices.reserve(attrib.vertices.size() / 3);

    auto indices = std::vector<uint32_t>{};
    auto currentIndex = uint32_t{0};

    for(auto const & shape: shapes){
        for(auto const & index: shape.mesh.indices){
            auto vertex = Vertex{};

            vertex.position = {
                attrib.vertices[3 * index.vertex_index + 0],
                attrib.vertices[3 * index.vertex_index + 1],
                attrib.vertices[3 * index.vertex_index + 2]
            };

            vertex.texCoord = {
                attrib.texcoords[2 * index.texcoord_index + 0],
                attrib.texcoords[2 * index.texcoord_index + 1]
            };

            vertices.push_back(vertex);
            
        }
    }

    struct {
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
    } meshData{
        vertices,
        indices
    };

    return meshData;
}

struct VulkanRenderState{
    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::UniqueImageView> swapchainImageViews;
    vk::UniqueRenderPass renderPass;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline graphicsPipeline;
    DepthImageHandles depthImageHandles;
    std::vector<vk::UniqueFramebuffer> frameBuffers;
    vk::UniqueCommandPool commandPool;
    vk::UniqueDescriptorPool descriptorPool;
    std::vector<vk::UniqueDescriptorSet> descriptorSets;
    std::vector<vk::UniqueCommandBuffer> commandBuffers;
    vk::UniqueBuffer vertexBuffer;
    vk::UniqueDeviceMemory vertexBufferMemory;
    vk::UniqueBuffer indexBuffer;
    vk::UniqueDeviceMemory indexBufferMemory;
    std::vector<std::pair<vk::UniqueBuffer, vk::UniqueDeviceMemory>> uniformBuffers;
    ImageHandles imageHandles;
    vk::UniqueImageView textureImageView;
    vk::UniqueSampler textureSampler;
    vk::Extent2D swapchainExtent;
};

[[nodiscard]] 
auto createVulkanRenderState(
        vk::UniqueDevice const & device,
        vk::PhysicalDevice const & gpu,
        vk::UniqueSurfaceKHR const & surface,
        UniqueGlfwWindow const & window,
        int32_t const graphicsIndex,
        int32_t const presentIndex)
{
    
    device->waitIdle();

    auto const capabilities = gpu.getSurfaceCapabilitiesKHR(surface.get());

    //TODO: write functions to find these.
    auto const surfaceFormat = gpu.getSurfaceFormatsKHR(surface.get()).back(); 
    auto const presentMode = gpu.getSurfacePresentModesKHR(surface.get()).back(); 

    auto const extent = createExtent(capabilities, window);

    auto const imageCount = capabilities.minImageCount + 1;

    auto const swapchainImageFormat = surfaceFormat.format;

    auto swapchain = createSwapchain(
            device, 
            extent, 
            surface, 
            imageCount, 
            surfaceFormat, 
            presentMode, 
            capabilities, 
            graphicsIndex, 
            presentIndex);

    auto swapchainImageViews = createSwapchainImageViews(
            device, 
            device->getSwapchainImagesKHR(swapchain.get()), 
            surfaceFormat);

    auto renderPass = createRenderPass(device, gpu, swapchainImageFormat);
    auto descriptorSetLayout = create_descriptor_set_layout(device.get());
    auto pipelineLayout = device->createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo({}, 
                1,
                &descriptorSetLayout.get()));

    auto graphicsPipeline = std::move(createGraphicsPipeline(
                device.get(), 
                renderPass.get(), 
                pipelineLayout.get(), 
                extent)
            .back());

    auto depthImageHandles = create_depth_resource(device, gpu, extent);
    auto frameBuffers = createFrameBuffers(
            device, 
            swapchainImageViews, 
            depthImageHandles.imageView, 
            renderPass, 
            extent);

    auto commandPool = device->createCommandPoolUnique(vk::CommandPoolCreateInfo({}, graphicsIndex));

    auto descriptorPool = create_descriptor_pool(device.get(), swapchainImageViews.size());

//    auto const vertices = std::vector<Vertex>{
//        {{0,    0.5,    0.0},     {1,0,1},    {1, 0}},
//        {{-0.5, 0,      0.0},     {0,1,0},    {0, 0}},
//        {{0.5,  0,      0.0},     {1,0,1},    {0, 1}},
//        {{1,    1,      0.0},     {1,1,0},    {1, 1}},
//
//        {{0,    0.5,    -0.5},     {1,0,1},    {1, 0}},
//        {{-0.5, 0,      -0.5},     {0,1,0},    {0, 0}},
//        {{0.5,  0,      -0.5},     {1,0,1},    {0, 1}},
//        {{1,    1,      -0.5},     {1,1,0},    {1, 1}},
//    };
//
//    auto const indices = std::vector<uint16_t>{
//        0, 1, 2, 2, 3, 0,
//        4, 5, 6, 6, 7, 4
//    };

    auto const graphicsQueue = device->getQueue(graphicsIndex,0);


    auto [
        vertices,
        indices
    ] = load_model("./assets/viking_room.obj");

    auto [
        vertexBuffer,
        vertexBufferMemory
    ] = createVertexBuffer(device, gpu, commandPool, graphicsQueue, vertices);

    auto [
        indexBuffer,
        indexBufferMemory
    ] = createIndexBuffer(device, gpu, commandPool, graphicsQueue, indices);

    auto uniformBuffers = createUniformBuffers(device, gpu, commandPool, graphicsQueue, swapchainImageViews.size());

    auto imageHandles = create_texture_image(device, gpu, commandPool, graphicsQueue, "Image.jpg");
    auto textureImageView = create_image_view(
            device.get(), 
            imageHandles.image.get(), 
            vk::Format::eR8G8B8A8Srgb, 
            vk::ImageAspectFlagBits::eColor);
    auto textureSampler = create_texture_sampler(device.get(), gpu);

    auto descriptorSets = create_descriptor_sets(
            device.get(), 
            descriptorPool.get(), 
            swapchainImageViews.size(), 
            descriptorSetLayout.get(), 
            uniformBuffers, 
            textureImageView.get(), 
            textureSampler.get());

    for(size_t i = 0; i < swapchainImageViews.size(); i++){
        auto const bufferInfo = vk::DescriptorBufferInfo(
                uniformBuffers[i].first.get(), 
                0, 
                sizeof(UniformBufferObject));

        auto const descriptorWrite = vk::WriteDescriptorSet(
                descriptorSets[i].get(), 
                0, 
                0,
                1, 
                vk::DescriptorType::eUniformBuffer, 
                nullptr, 
                &bufferInfo, 
                nullptr);

        device->updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
    }

    auto commandBuffers = device->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(
                commandPool.get(), 
                vk::CommandBufferLevel::ePrimary, 
                frameBuffers.size()));

    for(auto i = 0; i < commandBuffers.size(); i++){
        //TODO: just zip these together.
        auto const & commandBuffer = commandBuffers[i];
        auto const & frameBuffer = frameBuffers[i];
        commandBuffer->begin(vk::CommandBufferBeginInfo());

        auto const clearColor = std::vector{
            vk::ClearValue(vk::ClearColorValue(std::array{0.0f, 0.0f, 0.0f ,0.0f})),
            vk::ClearValue().setDepthStencil({1.0f, 0})
        };

        auto const renderArea = vk::Rect2D({0,0},extent);
        commandBuffer->beginRenderPass(
                vk::RenderPassBeginInfo(
                    renderPass.get(), 
                    frameBuffer.get(), 
                    renderArea, 
                    clearColor), 
                vk::SubpassContents::eInline);

        auto const viewport = vk::Viewport(0.0f, 0.0f, extent.width, extent.height, 0.0f, 1.0f);
        commandBuffer->setViewport(0, viewport);

        auto const scissor = vk::Rect2D({0,0}, extent);
        commandBuffer->setScissor(0, scissor);

        commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline.get());

        vk::DeviceSize offsets[] = {0};
        commandBuffer->bindVertexBuffers(0, 1, &vertexBuffer.get(), offsets);
        commandBuffer->bindIndexBuffer(indexBuffer.get(), 0, vk::IndexType::eUint32);

        commandBuffer->bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics, 
                pipelineLayout.get(), 
                0, 
                1, 
                &descriptorSets[i].get(), 
                0, 
                nullptr);

        //commandBuffer->draw(static_cast<uint32_t>(vertices.size()),1,0,0);
        commandBuffer->drawIndexed(indices.size(), 1, 0, 0, 0);
        commandBuffer->endRenderPass();
        commandBuffer->end();

    }

    return std::make_unique<VulkanRenderState>(
        std::move(swapchain),
        std::move(swapchainImageViews),
        std::move(renderPass),
        std::move(descriptorSetLayout),
        std::move(pipelineLayout),
        std::move(graphicsPipeline),
        std::move(depthImageHandles),
        std::move(frameBuffers),
        std::move(commandPool),
        std::move(descriptorPool),
        std::move(descriptorSets),
        std::move(commandBuffers),
        std::move(vertexBuffer),
        std::move(vertexBufferMemory),
        std::move(indexBuffer),
        std::move(indexBufferMemory),
        std::move(uniformBuffers),
        std::move(imageHandles),
        std::move(textureImageView),
        std::move(textureSampler),
        extent
    );
}

void update_uniformBuffer(
        vk::Device const device,
        vk::Buffer const uniformBuffer, 
        vk::DeviceMemory const unformBufferMemory,
        vk::Extent2D const swapchainExtent)
{
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    auto ubo = UniformBufferObject{};
    
    ubo.model = glm::rotate(
            glm::mat4(1.0f), 
            time * glm::radians(90.0f), 
            glm::vec3(0.0f, 0.0f, 1.0f));
    
    ubo.view = glm::lookAt(
            glm::vec3(2.0f,2.0f,2.0f), 
            glm::vec3(0.0f,0.0f,0.0f),
            glm::vec3(0.0f,0.0f,1.0f));

    ubo.proj = glm::perspective(
            glm::radians(45.0f), 
            (float)swapchainExtent.width/ (float)swapchainExtent.height,
            0.1f,
            10.0f);

    ubo.proj[1][1] *= -1;

    auto data = device.mapMemory(unformBufferMemory, 0, sizeof(ubo));
    memcpy(data, &ubo, sizeof(ubo));
    device.unmapMemory(unformBufferMemory);
}


