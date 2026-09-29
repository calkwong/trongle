#pragma once

#include <volk.h>

#include <cstdint>
#include <vector>

struct ImageManager
{
    // TODO: update and/or free ids

    std::vector<VkDescriptorImageInfo> infos;
    // std::vector<uint32_t> unused;

    uint32_t add(VkImageView view, VkImageLayout layout);
};

// Traditional
VkDescriptorPool create_descriptor_pool(VkDevice device, const VkDescriptorPoolSize* pool_sizes, uint32_t pool_size_count, uint32_t max_sets, VkDescriptorPoolCreateFlags flags = 0);

VkDescriptorSetLayout create_descriptor_set_layout(
    VkDevice device,
    const VkDescriptorSetLayoutBinding* bindings,
    uint32_t binding_count,
    const VkDescriptorSetLayoutBindingFlagsCreateInfo* p_next = 0,
    VkDescriptorSetLayoutCreateFlags flags = 0
);

VkDescriptorSet create_descriptor_set(VkDevice device, VkDescriptorPool pool, const VkDescriptorSetLayout* layout, const uint32_t* descriptor_counts);

VkWriteDescriptorSet write_image_descriptor(VkDescriptorSet set, VkDescriptorType type, uint32_t handle, const VkDescriptorImageInfo* info);
VkWriteDescriptorSet write_buffer_descriptor(VkDescriptorSet set, VkDescriptorType type, uint32_t handle, const VkDescriptorBufferInfo* info);
VkWriteDescriptorSet write_sampler_descriptor(VkDescriptorSet set, VkDescriptorType type, uint32_t handle, const VkDescriptorImageInfo* info);

// Descriptor Heap
