#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>

#include <cstddef>
#include <functional>

struct Image
{
    VkImage image;
    VkImageView view;
    VmaAllocation allocation;
    VkExtent3D extent;
    VkFormat format;
};

struct Buffer
{
    VkBuffer buffer;
    VkDeviceAddress address;
    VmaAllocation allocation;
    VmaAllocationInfo info;
    VkDeviceSize size;
};

void stage_barrier(
    VkCommandBuffer cmd,
    VkImage image,
    VkImageLayout old_layout,
    VkImageLayout new_layout,
    VkPipelineStageFlags2 src_stage_mask,
    VkPipelineStageFlags2 dst_stage_mask,
    VkAccessFlags2 src_access_mask,
    VkAccessFlags2 dst_access_mask,
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT
);

void stage_barrier(
    VkCommandBuffer cmd,
    VkPipelineStageFlags2 src_stage_mask,
    VkPipelineStageFlags2 dst_stage_mask,
    VkAccessFlags2 src_access_mask,
    VkAccessFlags2 dst_access_mask
);

Buffer create_buffer(
    VkDevice device,
    VmaAllocator allocator,
    size_t alloc_size,
    VmaAllocationCreateFlags allocation_flags,
    VkBufferUsageFlags usage_flags,
    VkDeviceSize alignment = 0
);

Image create_image(
    VkDevice device,
    VmaAllocator allocator,
    VkExtent3D extent,
    VkFormat format,
    VkImageUsageFlags usage,
    VkImageAspectFlags aspect,
    VmaAllocationCreateFlags allocation_flags = 0,
    bool mipmapped = false
);

void destroy_buffer(VmaAllocator allocator, const Buffer& buffer);
void destroy_image(VkDevice device, VmaAllocator allocator, const Image& image);
VkDeviceAddress get_buffer_address(VkDevice device, VkBuffer buffer);

void immediate_submit(VkDevice device, VkQueue queue, VkFence fence, VkCommandPool command_pool, VkCommandBuffer cmd, std::function<void(VkCommandBuffer cmd)>&& func);
