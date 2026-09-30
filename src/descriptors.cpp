#include "descriptors.h"
#include "common.h"

#include <volk.h>

#include <cstdint>

VkDescriptorPool create_descriptor_pool(VkDevice device, const VkDescriptorPoolSize* pool_sizes, uint32_t pool_size_count, uint32_t max_sets, VkDescriptorPoolCreateFlags flags /*= 0*/)
{
    VkDescriptorPool pool{};

    VkDescriptorPoolCreateInfo info{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    info.maxSets = 4;
    info.flags = flags;
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

VkWriteDescriptorSet write_image_descriptor(VkDescriptorSet set, VkDescriptorType type, uint32_t handle, const VkDescriptorImageInfo* info)
{
    VkWriteDescriptorSet write{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write.dstSet = set;
    write.dstBinding = 0;
    write.dstArrayElement = handle;
    write.descriptorCount = 1;
    write.descriptorType = type;
    write.pImageInfo = info;

    return write;
}

VkWriteDescriptorSet write_buffer_descriptor(VkDescriptorSet set, VkDescriptorType type, uint32_t handle, const VkDescriptorBufferInfo* info)
{
    VkWriteDescriptorSet write{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write.dstSet = set;
    write.dstBinding = 0;
    write.dstArrayElement = handle;
    write.descriptorCount = 1;
    write.descriptorType = type;
    write.pBufferInfo = info;

    return write;
}

VkWriteDescriptorSet write_sampler_descriptor(VkDescriptorSet set, uint32_t handle, const VkDescriptorImageInfo* info)
{
    VkWriteDescriptorSet write{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write.dstSet = set;
    write.dstBinding = 0;
    write.dstArrayElement = handle;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    write.pImageInfo = info;

    return write;
}

uint32_t ImageManager::add(VkImageView view, VkImageLayout layout)
{
    uint32_t index = static_cast<uint32_t>(infos.size());

    infos.emplace_back(VkDescriptorImageInfo{ .imageView = view, .imageLayout = layout });

    return index;
}

uint32_t ImageManager::register_image(VkDevice device, VkDescriptorSet sampled_set, VkDescriptorSet storage_set, VkImageView view, VkImageLayout layout)
{
    uint32_t handle = add(view, layout);

    VkWriteDescriptorSet writes[2] = { write_image_descriptor(sampled_set, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, handle, &infos[handle]),
                                       write_image_descriptor(storage_set, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, handle, &infos[handle]) };
    vkUpdateDescriptorSets(device, 2, &writes[0], 0, nullptr);
    return handle;
}

uint32_t ImageManager::register_sampled_image(VkDevice device, VkDescriptorSet set, VkImageView view, VkImageLayout layout)
{
    uint32_t handle = add(view, layout);

    VkWriteDescriptorSet write{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write.dstSet = set;
    write.dstBinding = 0;
    write.dstArrayElement = handle;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = &infos[handle];

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    return handle;
}
