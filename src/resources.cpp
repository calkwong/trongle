#include "resources.h"
#include "common.h"

#include <volk.h>
#include <vk_mem_alloc.h>

#include <cstddef>
#include <cassert>

void stage_barrier(
    VkCommandBuffer cmd,
    VkImage image,
    VkImageLayout old_layout,
    VkImageLayout new_layout,
    VkPipelineStageFlags2 src_stage_mask,
    VkPipelineStageFlags2 dst_stage_mask,
    VkAccessFlags2 src_access_mask,
    VkAccessFlags2 dst_access_mask,
    VkImageAspectFlags aspect /*= VK_IMAGE_ASPECT_COLOR_BIT*/
)
{
    VkImageMemoryBarrier2 barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    barrier.srcStageMask = src_stage_mask;
    barrier.dstStageMask = dst_stage_mask;
    barrier.srcAccessMask = src_access_mask;
    barrier.dstAccessMask = dst_access_mask;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;

    VkImageSubresourceRange subresource_range{};
    subresource_range.aspectMask = aspect;
    subresource_range.baseMipLevel = 0;
    subresource_range.levelCount = VK_REMAINING_MIP_LEVELS;
    subresource_range.baseArrayLayer = 0;
    subresource_range.layerCount = VK_REMAINING_ARRAY_LAYERS;

    barrier.subresourceRange = subresource_range;
    barrier.image = image;

    VkDependencyInfo info{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    info.imageMemoryBarrierCount = 1;
    info.pImageMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2(cmd, &info);
}

void stage_barrier(
    VkCommandBuffer cmd,
    VkPipelineStageFlags2 src_stage_mask,
    VkPipelineStageFlags2 dst_stage_mask,
    VkAccessFlags2 src_access_mask,
    VkAccessFlags2 dst_access_mask
)
{
    VkMemoryBarrier2 barrier{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    barrier.srcStageMask = src_stage_mask;
    barrier.srcAccessMask = src_access_mask;
    barrier.dstStageMask = dst_stage_mask;
    barrier.dstAccessMask = dst_access_mask;

    VkDependencyInfo info{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    info.memoryBarrierCount = 1;
    info.pMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2(cmd, &info);
}

// REVIEW: alignment
Buffer create_buffer(VkDevice device, VmaAllocator allocator, size_t alloc_size, VmaAllocationCreateFlags allocation_flags, VkBufferUsageFlags usage_flags, VkDeviceSize alignment /* = 0 */)
{
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = alloc_size;
    buffer_info.usage = usage_flags;

    VmaAllocationCreateInfo alloc_info{};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_info.flags = allocation_flags;

    Buffer buffer{};
    buffer.size = alloc_size;

    if (alignment != 0)
        VK_CHECK(vmaCreateBufferWithAlignment(allocator, &buffer_info, &alloc_info, alignment, &buffer.buffer, &buffer.allocation, &buffer.info));
    else
        VK_CHECK(vmaCreateBuffer(allocator, &buffer_info, &alloc_info, &buffer.buffer, &buffer.allocation, &buffer.info));

    buffer.address = get_buffer_address(device, buffer.buffer);

    return buffer;
}

Image create_image(
    VkDevice device,
    VmaAllocator allocator,
    VkExtent3D extent,
    VkFormat format,
    VkImageUsageFlags usage,
    VkImageAspectFlags aspect,
    VmaAllocationCreateFlags allocation_flags /*= 0*/,
    bool mipmapped /*= false*/
)
{
    Image image{};
    image.extent = extent;
    image.format = format;

    VkImageCreateInfo img_info{};
    img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_info.imageType = VK_IMAGE_TYPE_2D;
    img_info.format = format;
    img_info.extent = extent;
    img_info.mipLevels = 1;
    img_info.arrayLayers = 1;
    img_info.samples = VK_SAMPLE_COUNT_1_BIT;
    img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    img_info.usage = usage;

    if (mipmapped)
    {
        assert(0 && "IMPLEMENT!");
        // img_info.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(extent.width, extent.height)))) + 1;
        // img_info.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    }

    VmaAllocationCreateInfo alloc_info{};
    alloc_info.flags = allocation_flags;
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VK_CHECK(vmaCreateImage(allocator, &img_info, &alloc_info, &image.image, &image.allocation, nullptr));

    return image;
}

void destroy_buffer(VmaAllocator allocator, const Buffer& buffer)
{
    vmaDestroyBuffer(allocator, buffer.buffer, buffer.allocation);
}

void destroy_image(VkDevice device, VmaAllocator allocator, const Image& image)
{
    if (image.view != VK_NULL_HANDLE)
        vkDestroyImageView(device, image.view, nullptr);
    vmaDestroyImage(allocator, image.image, image.allocation);
}

VkDeviceAddress get_buffer_address(VkDevice device, VkBuffer buffer)
{
    VkBufferDeviceAddressInfo info{ .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
    info.buffer = buffer;
    return vkGetBufferDeviceAddress(device, &info);
}
