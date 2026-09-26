#include "descriptors.h"
#include "common.h"

#include <volk.h>

#include <cstdint>

VkDescriptorPool create_descriptor_pool(VkDevice device, const VkDescriptorPoolSize* pool_sizes, uint32_t pool_size_count)
{
    VkDescriptorPool pool{};

    VkDescriptorPoolCreateInfo info{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    info.maxSets = 4;
    info.poolSizeCount = pool_size_count;
    info.pPoolSizes = pool_sizes;

    VK_CHECK(vkCreateDescriptorPool(device, &info, nullptr, &pool));

    return pool;
}

VkDescriptorSetLayout create_descriptor_set_layout(VkDevice device, const VkDescriptorSetLayoutBinding* bindings, uint32_t binding_count, const VkDescriptorSetLayoutBindingFlagsCreateInfo* p_next /*= 0*/, VkDescriptorSetLayoutCreateFlags flags /*= 0*/)
{
    VkDescriptorSetLayout layout{};

    VkDescriptorSetLayoutCreateInfo info{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    info.pNext = p_next;
    info.flags = flags;
    info.bindingCount = binding_count;
    info.pBindings = bindings;

    vkCreateDescriptorSetLayout(device, &info, nullptr, &layout);

    return layout;
}

VkDescriptorSet create_descriptor_set(VkDevice device, VkDescriptorPool pool, const VkDescriptorSetLayout* layout, const uint32_t* descriptor_counts)
{
    VkDescriptorSet set{};

    VkDescriptorSetVariableDescriptorCountAllocateInfo variable_info{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO };
    variable_info.pDescriptorCounts = descriptor_counts;
    variable_info.descriptorSetCount = 1;

    VkDescriptorSetAllocateInfo info{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    info.pNext = &variable_info;
    info.descriptorPool = pool;
    info.descriptorSetCount = 1;
    info.pSetLayouts = layout;

    vkAllocateDescriptorSets(device, &info, &set);

    return set;
}
