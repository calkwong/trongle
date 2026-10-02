#pragma once

#include <volk.h>

#include <cstdint>
#include <vector>

// This assigns the same bindless id to both storage and sampled descriptor sets. This provides convenience at the cost of memory and holes
// in our descriptor sets.
struct ImageManager
{
    VkDescriptorSet set;
    std::vector<VkDescriptorImageInfo> infos;
    std::vector<uint32_t> free_ids;

    uint32_t add(VkImageView view, VkImageLayout layout);
    uint32_t register_image(VkDevice device, VkImageView view, VkImageLayout layout);
    uint32_t register_sampled_image(VkDevice device, VkImageView view, VkImageLayout layout);
};

// Traditional
VkDescriptorPool create_descriptor_pool(VkDevice device, const VkDescriptorPoolSize* pool_sizes, uint32_t pool_size_count, uint32_t max_sets, VkDescriptorPoolCreateFlags flags = 0);

VkDescriptorSetLayout create_descriptor_set_layout(
    VkDevice device,
    const VkDescriptorSetLayoutBinding* bindings,
    uint32_t binding_count,
    const VkDescriptorBindingFlags* binding_flags = nullptr,
    uint32_t binding_flags_count = 0,
    VkDescriptorSetLayoutCreateFlags layout_flags = 0
);

VkDescriptorSet create_descriptor_set(VkDevice device, VkDescriptorPool pool, const VkDescriptorSetLayout* layout, const uint32_t* descriptor_counts);

VkWriteDescriptorSet write_image_descriptor(VkDescriptorSet set, VkDescriptorType type, uint32_t handle, const VkDescriptorImageInfo* info, uint32_t binding);
VkWriteDescriptorSet write_buffer_descriptor(VkDescriptorSet set, VkDescriptorType type, uint32_t handle, const VkDescriptorBufferInfo* info);
VkWriteDescriptorSet write_sampler_descriptor(VkDescriptorSet set, uint32_t handle, const VkDescriptorImageInfo* info);

// Descriptor Heap
