#pragma once

#include <volk.h>

#include <cstdint>

// Traditional
VkDescriptorPool create_descriptor_pool(VkDevice device, const VkDescriptorPoolSize* pool_sizes, uint32_t pool_size_count);

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
