#include<filesystem>
#include<fstream>
#include<vector>

#include<vulkan/vulkan.hpp>
#include<GLFW/glfw3.h>

#include"GlfwStuff.cpp"


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

auto createShaderModule(std::filesystem::path path, vk::UniqueDevice const & device){
    auto const shaderCode = loadShader(path);
    return device->createShaderModuleUnique(vk::ShaderModuleCreateInfo({}, 
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

auto createGraphicsPipeline(
        vk::UniqueDevice const & device, 
        vk::UniqueRenderPass const & renderPass, 
        vk::UniquePipelineLayout const & layout,
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

    auto const vertexInputInfo = vk::PipelineVertexInputStateCreateInfo();

    auto const inputAssemblyInfo = vk::PipelineInputAssemblyStateCreateInfo({}, vk::PrimitiveTopology::eTriangleList, VK_FALSE);

    auto const viewport = vk::Viewport(0.0f, 0.0f, swapchainExtent.width, swapchainExtent.height, 0.0f, 1.0f);

    auto const scissor = vk::Rect2D({0,0}, swapchainExtent);

    auto const viewportStateInfo = vk::PipelineViewportStateCreateInfo({}, 1, &viewport, 1, &scissor);


    auto const rasterizer = vk::PipelineRasterizationStateCreateInfo({}, 
            VK_FALSE, 
            VK_FALSE, 
            vk::PolygonMode::eFill, 
            vk::CullModeFlagBits::eBack, 
            vk::FrontFace::eClockwise, 
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

    //TODO: dynamic viewport.
    auto const dynamicStates = std::array{
        //vk::DynamicState::eViewport, 
        vk::DynamicState::eLineWidth
    };
    auto const dynamicState = vk::PipelineDynamicStateCreateInfo({}, dynamicStates);

    auto const createInfos = std::array{
        vk::GraphicsPipelineCreateInfo({}, 
                shaderStages, 
                &vertexInputInfo, 
                &inputAssemblyInfo, 
                nullptr,
                &viewportStateInfo,
                &rasterizer,
                &multisampleing,
                nullptr,
                &colorBlending,
                &dynamicState,
                layout.get(),
                renderPass.get(),
                0) 
    };

    //TODO: check return result and see why creation may have failed.
    return device->createGraphicsPipelinesUnique({}, createInfos).value;
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



struct VulkanRenderState{
    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::UniqueImageView> swapchainImageViews;
    vk::UniqueRenderPass renderPass;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline graphicsPipeline;
    std::vector<vk::UniqueFramebuffer> frameBuffers;
    vk::UniqueCommandPool commandPool;
    std::vector<vk::UniqueCommandBuffer> commandBuffers;
};

auto createVulkanRenderState(
        vk::UniqueDevice const & device,
        vk::PhysicalDevice const & gpu,
        vk::UniqueSurfaceKHR const & surface,
        UniqueGlfwWindow const & window,
        int32_t const graphicsIndex,
        int32_t const presentIndex)
{
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
    auto pipelineLayout = device->createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo());
    auto graphicsPipeline = std::move(createGraphicsPipeline(device, renderPass, pipelineLayout, extent).back());

    auto frameBuffers = createFrameBuffers(device, swapchainImageViews, renderPass, extent);

    auto commandPool = device->createCommandPoolUnique(vk::CommandPoolCreateInfo({}, graphicsIndex));

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

        commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline.get());
        commandBuffer->draw(3,1,0,0);
        commandBuffer->endRenderPass();
        commandBuffer->end();
    } 

    return std::make_shared<VulkanRenderState>(
        std::move(swapchain),
        std::move(swapchainImageViews),
        std::move(renderPass),
        std::move(pipelineLayout),
        std::move(graphicsPipeline),
        std::move(frameBuffers),
        std::move(commandPool),
        std::move(commandBuffers)
    );
}

void drawFrame(){

}


