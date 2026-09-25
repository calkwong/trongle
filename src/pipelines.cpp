#include "pipelines.h"
#include "common.h"

#include <volk.h>
#include <fmt/core.h>

#include <ios>
#include <vector>
#include <fstream>
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <memory>
#include <array>
#include <string>

// REVIEW
bool load_shader_module(const char* path, VkDevice device, VkShaderModule* out_shader_module)
{
    // cursor at the end
    std::ifstream file(path, std::ios::ate | std::ios::binary);

    if (!file.is_open())
    {
        return false;
    }

    // find what the size of the file is by looking up the location of the cursor
    // because the cursor is at the end, it gives the size directly in bytes
    const size_t file_size = file.tellg();

    // spirv expects the buffer to be on uint32, so make sure to reserve a int
    // vector big enough for the entire file
    std::vector<uint32_t> buffer(file_size / sizeof(uint32_t));

    // put file cursor at beginning
    file.seekg(0);

    // load the entire file into the buffer
    file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(file_size));

    // now that the file is loaded into the buffer, we can close it
    file.close();

    // create a new shader module, using the buffer we loaded
    VkShaderModuleCreateInfo create_info{ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };

    // codeSize has to be in bytes, so multply the ints in the buffer by size of
    // int to know the real size of the buffer
    create_info.codeSize = buffer.size() * sizeof(uint32_t);
    create_info.pCode = buffer.data();

    // check that the creation goes well.
    VkShaderModule shader_module{};
    VK_CHECK(vkCreateShaderModule(device, &create_info, nullptr, &shader_module));

    *out_shader_module = shader_module;
    return true;
}

ShaderProgram load_shader_program(const char* path, VkDevice device)
{
    VkShaderModule module{};

    std::string shader_path = "../shaders/compiled/";
    shader_path += path;
    shader_path += ".spv";

    fmt::println("Shader path: {}", shader_path);

    if (!load_shader_module(shader_path.c_str(), device, &module))
    {
        assert(0 && "Load shader failed");
    }

    ShaderProgram program{ .module = module, .name = path };
    return program;
}

// TODO: implement proper error check for invalid pipeline_layout + p_desc_set_and_binding_mapping_info combination
std::unique_ptr<ShaderPass> create_graphics_pipeline(
    VkDevice device,
    ShaderProgram* program,
    ShaderStages stages,
    VkShaderDescriptorSetAndBindingMappingInfoEXT* p_desc_set_and_binding_mapping_info,
    VkPipelineLayout* pipeline_layout,
    ColorAttachmentFormats color_attachment_formats,
    VkCullModeFlags cull_mode /*= VK_CULL_MODE_BACK_BIT*/,
    SpecConstants spec_constants /*= {}*/
)
{
    std::unique_ptr<ShaderPass> result = std::make_unique<ShaderPass>();
    VkPipeline pipeline{};

    VkGraphicsPipelineCreateInfo info{ .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };

    VkPipelineRenderingCreateInfo rendering_create_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    rendering_create_info.colorAttachmentCount = color_attachment_formats.size();
    rendering_create_info.pColorAttachmentFormats = data(color_attachment_formats); // 0?
    rendering_create_info.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
    rendering_create_info.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

    VkPipelineCreateFlags2CreateInfo create_flags_2_create_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO };
    create_flags_2_create_info.pNext = &rendering_create_info;
    create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    if (p_desc_set_and_binding_mapping_info == nullptr)
    {
        info.pNext = &rendering_create_info;
    }
    else
    {
        info.pNext = &create_flags_2_create_info;
    }

    std::vector<VkSpecializationMapEntry> spec_map_entries{};
    VkSpecializationInfo specialization_info{};

    auto specialization_count = spec_constants.size();
    if (specialization_count > 0)
    {
        for (uint32_t i = 0; i < specialization_count; i++)
        {
            VkSpecializationMapEntry map_entry{};
            map_entry.constantID = i;
            map_entry.offset = i * sizeof(uint32_t);
            map_entry.size = sizeof(uint32_t);
            spec_map_entries.push_back(map_entry);
        }

        specialization_info.mapEntryCount = static_cast<uint32_t>(spec_map_entries.size());
        specialization_info.pMapEntries = spec_map_entries.data();
        specialization_info.dataSize = spec_map_entries.size() * sizeof(uint32_t);
        specialization_info.pData = data(spec_constants);
    }

    std::vector<VkPipelineShaderStageCreateInfo> shader_stages{};
    const char* vs_entry = "vs_main";
    const char* ps_entry = "ps_main";
    const char* ms_entry = "mesh_main";
    for (auto it = stages.begin(); it != stages.end(); it++)
    {
        VkPipelineShaderStageCreateInfo shader_stage_create_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        shader_stage_create_info.pNext = p_desc_set_and_binding_mapping_info;
        shader_stage_create_info.flags = 0;
        shader_stage_create_info.stage = *it;
        shader_stage_create_info.module = program->module;

        switch (shader_stage_create_info.stage)
        {
        case VK_SHADER_STAGE_VERTEX_BIT:
            shader_stage_create_info.pName = vs_entry;
            break;
        case VK_SHADER_STAGE_FRAGMENT_BIT:
            shader_stage_create_info.pName = ps_entry;
            break;
        case VK_SHADER_STAGE_MESH_BIT_EXT:
            shader_stage_create_info.pName = ms_entry;
            break;
        default:
            assert(0);
        }

        shader_stage_create_info.pSpecializationInfo = spec_constants.size() > 0 ? &specialization_info : 0;
        shader_stages.push_back(shader_stage_create_info);
    }

    VkPipelineVertexInputStateCreateInfo vertex_input_state_create_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };

    VkPipelineInputAssemblyStateCreateInfo input_assembly_state_create_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    input_assembly_state_create_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewport_state_create_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewport_state_create_info.viewportCount = 1;
    viewport_state_create_info.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterization_state_create_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rasterization_state_create_info.pNext = 0;
    rasterization_state_create_info.flags = 0;
    rasterization_state_create_info.depthClampEnable = VK_FALSE;
    rasterization_state_create_info.rasterizerDiscardEnable = 0;
    rasterization_state_create_info.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization_state_create_info.cullMode = cull_mode;
    rasterization_state_create_info.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization_state_create_info.depthBiasEnable = VK_FALSE;
    rasterization_state_create_info.depthBiasConstantFactor = 0.0f;
    rasterization_state_create_info.depthBiasClamp = 0.0f;
    rasterization_state_create_info.depthBiasSlopeFactor = 0.0f;
    rasterization_state_create_info.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample_state_create_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    multisample_state_create_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisample_state_create_info.minSampleShading = 1.0f;

    VkPipelineDepthStencilStateCreateInfo depth_stencil_state_create_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    depth_stencil_state_create_info.depthTestEnable = VK_TRUE;
    depth_stencil_state_create_info.depthWriteEnable = VK_TRUE;
    depth_stencil_state_create_info.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL; // reverse z
    depth_stencil_state_create_info.minDepthBounds = 0.0f;
    depth_stencil_state_create_info.maxDepthBounds = 1.0f;

    std::vector<VkPipelineColorBlendAttachmentState> color_blend_attachment_state{};
    for (size_t i = 0; i < color_attachment_formats.size(); i++)
    {
        VkPipelineColorBlendAttachmentState state{};
        state.blendEnable = VK_FALSE;
        state.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        color_blend_attachment_state.push_back(state);
    }

    VkPipelineColorBlendStateCreateInfo color_blend_state_create_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    color_blend_state_create_info.flags = 0;
    color_blend_state_create_info.logicOpEnable = VK_FALSE;
    color_blend_state_create_info.logicOp = VK_LOGIC_OP_COPY;
    color_blend_state_create_info.attachmentCount = color_blend_attachment_state.size();
    color_blend_state_create_info.pAttachments = color_blend_attachment_state.data();

    std::array<VkDynamicState, 2> dynamic_states = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

    VkPipelineDynamicStateCreateInfo dynamic_state_create_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynamic_state_create_info.dynamicStateCount = dynamic_states.size();
    dynamic_state_create_info.pDynamicStates = dynamic_states.data();

    info.stageCount = shader_stages.size();
    info.pStages = shader_stages.data();
    info.pVertexInputState = &vertex_input_state_create_info;
    info.pInputAssemblyState = &input_assembly_state_create_info;
    info.pTessellationState = nullptr;
    info.pViewportState = &viewport_state_create_info;
    info.pRasterizationState = &rasterization_state_create_info;
    info.pMultisampleState = &multisample_state_create_info;
    info.pDepthStencilState = &depth_stencil_state_create_info;
    info.pColorBlendState = &color_blend_state_create_info;
    info.pDynamicState = &dynamic_state_create_info;
    info.layout = pipeline_layout != nullptr ? *pipeline_layout : nullptr;

    VK_CHECK(vkCreateGraphicsPipelines(device, 0, 1, &info, nullptr, &pipeline));

    result->pipeline = pipeline;

    return result;
}
