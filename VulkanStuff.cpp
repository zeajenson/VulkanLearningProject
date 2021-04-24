#include<filesystem>
#include<fstream>
#include<vector>
#include<iostream>

#include<vulkan/vulkan.hpp>
#include<GLFW/glfw3.h>
#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include<stb_image.h>

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

auto createSwapchainImageViews(
        vk::UniqueDevice const & device,
        std::vector<vk::Image> const & swapchainImages, 
        vk::SurfaceFormatKHR surfaceFormat){
    auto imageViews = std::vector<vk::UniqueImageView>();
    imageViews.reserve(swapchainImages.size());
    
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

auto createRenderPass(vk::UniqueDevice const & device, vk::Format format){
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

    auto const subpass = vk::SubpassDescription({}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &colorAttachmentRef);

    auto const subpassDependency = vk::SubpassDependency(
            VK_SUBPASS_EXTERNAL, 
            0, 
            vk::PipelineStageFlagBits::eColorAttachmentOutput, 
            vk::PipelineStageFlagBits::eColorAttachmentOutput, 
            {}, 
            vk::AccessFlagBits::eColorAttachmentWrite);

    return device->createRenderPassUnique(vk::RenderPassCreateInfo({}, 
                1, &collorAttachment, 
                1, &subpass, 
                1, &subpassDependency));
}

struct Vertex{
    glm::vec2 position;
    glm::vec3 color;
};

auto createVertexBindingDescritptions(){
    
    auto const bindingDescription = vk::VertexInputBindingDescription(0, sizeof(Vertex), vk::VertexInputRate::eVertex);

    auto attributes = std::array<vk::VertexInputAttributeDescription, 2>();
    auto & position = attributes[0];
    auto & color = attributes[1];

    position.binding = 0;
    position.location = 0;
    position.format = vk::Format::eR32G32Sfloat;
    position.offset = offsetof(Vertex, position);

    color.binding = 0;
    color.location = 1;
    color.format = vk::Format::eR32G32B32Sfloat;
    color.offset = offsetof(Vertex, color);

    struct{
        std::array<vk::VertexInputBindingDescription, 1> bindingDescriptions;
        std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions;
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

auto create_descriptor_set_layout(vk::Device const device) noexcept{
    auto const uboBinding = vk::DescriptorSetLayoutBinding(
            0, 
            vk::DescriptorType::eUniformBuffer, 
            1,
            vk::ShaderStageFlagBits::eVertex,
            nullptr);

    return device.createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo({}, 1, &uboBinding));
}

auto createGraphicsPipeline(
        vk::Device const & device, 
        vk::RenderPass const & renderPass, 
        vk::PipelineLayout const & layout,
        vk::Extent2D const & swapchainExtent) 
noexcept {
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
        vk::UniqueRenderPass const & renderPass,
        vk::Extent2D const & extent)
{
    auto frameBuffers = std::vector<vk::UniqueFramebuffer>();
    frameBuffers.reserve(swapchainImageViews.size());
    for(auto const & imageView : swapchainImageViews){
        
        auto const attachments = std::vector{ imageView.get() };

        frameBuffers.push_back(device->createFramebufferUnique(vk::FramebufferCreateInfo({}, 
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
        std::vector<uint16_t> const & indices)
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
    auto const poolSize = vk::DescriptorPoolSize(
            vk::DescriptorType::eUniformBuffer,
            imageCount);

    auto const poolInfo = vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, imageCount, 1, &poolSize);
    
    return device.createDescriptorPoolUnique(poolInfo);
}

auto create_descriptor_sets(
        vk::Device const device, 
        vk::DescriptorPool const pool, 
        uint32_t imageCount, 
        vk::DescriptorSetLayout const layout) noexcept
{
    auto const layouts = std::vector(imageCount, layout);
    
    return device.allocateDescriptorSetsUnique(vk::DescriptorSetAllocateInfo(pool, layouts));
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

struct pixelDeleter { void operator() (uint8_t * pixels){stbi_image_free(pixels);} };
using pixelsRef = std::unique_ptr<uint8_t, pixelDeleter>;

auto create_image(        
        vk::UniqueDevice const & device,
        vk::PhysicalDevice const & gpu,
        vk::Format format,
        vk::ImageTiling tiling,
        vk::ImageUsageFlags usage,
        int const width,
        int const height,
        pixelsRef pixels) 
{
    auto const size = vk::DeviceSize(width * height * 4);

    auto [
        buffer,
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
    auto textureMemory = device->allocateMemoryUnique(vk::MemoryAllocateInfo(
                memoryRequirements.size, 
                findMemoryType(
                    gpu, 
                    memoryRequirements.memoryTypeBits, 
                    vk::MemoryPropertyFlagBits::eDeviceLocal)));

    device->bindImageMemory(image.get(), textureMemory.get(), 0);

    struct Stuff{
        vk::UniqueBuffer stagingBuffer;
        vk::UniqueImage image;
        vk::UniqueDeviceMemory imageMemory;
    };

    return Stuff{
        std::move(buffer),
        std::move(image),
        std::move(memory)
    };
}

auto create_texture_image(        
        vk::UniqueDevice const & device,
        vk::PhysicalDevice const & gpu,
        vk::UniqueCommandPool const & commandPool,
        vk::Queue const & graphicsQueue,
        std::filesystem::path path) noexcept
{
    int width, height, channels;
    auto const filename = path.c_str();
    auto pixels = pixelsRef(stbi_load("./Image.png", &width, &height, &channels, STBI_rgb_alpha));
    if(not pixels){
        std::cerr << "No pixels for texture: " << filename << std::endl;
    }

    auto [
        stagingBuffer,
        image,
        bufferMemory
    ] = create_image(
            device,
            gpu,
            vk::Format::eR8G8B8A8Srgb,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst 
            | vk::ImageUsageFlagBits::eSampled, 
            width, height,
            std::move(pixels));

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

    return ImageHandles{
        std::move(image),
        std::move(bufferMemory)
    };
}


struct VulkanRenderState{
    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::UniqueImageView> swapchainImageViews;
    vk::UniqueRenderPass renderPass;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline graphicsPipeline;
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
    vk::Extent2D swapchainExtent;
};

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

    auto renderPass = createRenderPass(device, swapchainImageFormat);
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

    auto frameBuffers = createFrameBuffers(device, swapchainImageViews, renderPass, extent);

    auto commandPool = device->createCommandPoolUnique(vk::CommandPoolCreateInfo({}, graphicsIndex));

    auto descriptorPool = create_descriptor_pool(device.get(), swapchainImageViews.size());

    auto descriptorSets = create_descriptor_sets(
            device.get(), 
            descriptorPool.get(), 
            swapchainImageViews.size(), 
            descriptorSetLayout.get());



    auto const vertices = std::vector<Vertex>{
        {{0,    0.5},   {1,0,1}},
        {{-0.5, 0},     {0,1,0}},
        {{0.5,  0},     {1,0,1}},
        {{1,    1},     {1,1,0}}
    };

    auto const indices = std::vector<uint16_t>{
        0, 1, 2, 2, 3, 0
    };

    auto const graphicsQueue = device->getQueue(graphicsIndex,0);

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
        auto const clearColor = std::vector{vk::ClearValue(vk::ClearColorValue(std::array{0.0f, 0.0f, 0.0f ,0.0f}))};
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
        commandBuffer->bindIndexBuffer(indexBuffer.get(), 0, vk::IndexType::eUint16);

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


