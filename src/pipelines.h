#pragma once

#include <volk.h>

#include <string>
#include <initializer_list>
#include <cstdint>

struct ShaderProgram
{
    VkShaderModule module;
    std::string name;
};

bool load_shader_module(const char* path, VkDevice device, VkShaderModule& out_shader_module);
ShaderProgram load_shader_program(const char* path, VkDevice device);

using SpecConstants = std::initializer_list<uint32_t>;
using ShaderStages = std::initializer_list<VkShaderStageFlagBits>;
using ColorAttachmentFormats = std::initializer_list<VkFormat>;

VkPipeline create_graphics_pipeline(
    VkDevice device,
    const ShaderProgram* program,
    ShaderStages stages,
    const VkShaderDescriptorSetAndBindingMappingInfoEXT* p_desc_set_and_binding_mapping_info,
    const VkPipelineLayout* pipeline_layout,
    ColorAttachmentFormats color_attachment_formats,
    VkCullModeFlags cull_mode = VK_CULL_MODE_BACK_BIT,
    SpecConstants spec_constants = {}
);

VkPipeline create_compute_pipeline(
    VkDevice device,
    const ShaderProgram* program,
    const VkShaderDescriptorSetAndBindingMappingInfoEXT* p_desc_set_and_binding_mapping_info,
    const VkPipelineLayout* pipeline_layout,
    SpecConstants spec_constants = {}
);
