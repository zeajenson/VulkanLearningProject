#include<filesystem>
#include<fstream>
#include<vector>

#include<vulkan/vulkan.hpp>

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

    return device->createRenderPassUnique(vk::RenderPassCreateInfo({}, 1, &collorAttachment, 1, &subpass));
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

    auto const viewport = vk::Viewport(0.0f, 0.0f, (float) swapchainExtent.width, swapchainExtent.height, 0.0f, 1.0f);

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

    auto const dynamicStates = std::array{vk::DynamicState::eViewport, vk::DynamicState::eLineWidth};
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

    return device->createGraphicsPipelinesUnique({}, createInfos);
}

