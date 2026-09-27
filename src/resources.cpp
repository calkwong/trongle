#include "resources.h"
#include "common.h"

#include <cstdint>
#include <volk.h>
#include <vk_mem_alloc.h>

#include <cstddef>
#include <cassert>
#include <functional>
#include <cstring>
#include <cmath>
#include <vector>
#include <initializer_list>

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

Buffer create_staging_buffer(VkDevice device, VmaAllocator allocator, size_t alloc_size, VmaAllocationCreateFlags allocation_flags, VkBufferUsageFlags usage_flags, VkDeviceSize alignment /* = 0 */)
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
        auto larger_extent = extent.width >= extent.height ? extent.width : extent.height;
        img_info.mipLevels = static_cast<uint32_t>(std::floor(std::log2(larger_extent))) + 1;
        img_info.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    }

    VmaAllocationCreateInfo alloc_info{};
    alloc_info.flags = allocation_flags;
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VK_CHECK(vmaCreateImage(allocator, &img_info, &alloc_info, &image.image, &image.allocation, nullptr));

    VkImageViewCreateInfo img_view_info{};
    img_view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    img_view_info.image = image.image;
    img_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    img_view_info.format = format;
    VkImageSubresourceRange subresource_range{};
    subresource_range.aspectMask = aspect;
    subresource_range.baseMipLevel = 0;
    subresource_range.levelCount = VK_REMAINING_MIP_LEVELS;
    subresource_range.baseArrayLayer = 0;
    subresource_range.layerCount = VK_REMAINING_ARRAY_LAYERS;
    img_view_info.subresourceRange = subresource_range;

    VK_CHECK(vkCreateImageView(device, &img_view_info, nullptr, &image.view));

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

void immediate_submit(VkDevice device, VkQueue queue, VkFence fence, VkCommandPool command_pool, VkCommandBuffer cmd, std::function<void(VkCommandBuffer cmd)>&& func)
{
    VK_CHECK(vkResetFences(device, 1, &fence));
    VK_CHECK(vkResetCommandPool(device, command_pool, 0));

    VkCommandBufferBeginInfo cmd_begin_info{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    cmd_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_begin_info));

    func(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmd_submit_info{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
    cmd_submit_info.commandBuffer = cmd;

    VkSubmitInfo2 submit{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    submit.pCommandBufferInfos = &cmd_submit_info;
    submit.commandBufferInfoCount = 1;

    VK_CHECK(vkQueueSubmit2(queue, 1, &submit, fence));

    VK_CHECK(vkWaitForFences(device, 1, &fence, true, WAIT_TIME));
}

Buffer create_buffer_with_data(
    VkDevice device,
    VkQueue queue,
    VkFence fence,
    VkCommandPool command_pool,
    VkCommandBuffer cmd,
    VmaAllocator allocator,
    const void* data,
    size_t data_size,
    VkBufferUsageFlags flags /*= 0*/
)
{
    Buffer buffer = create_buffer(
        device,
        allocator,
        data_size,
        0,
        VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | flags
    );

    Buffer staging = create_staging_buffer(
        device,
        allocator,
        data_size,
        VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT
    );

    void* staging_data = staging.info.pMappedData;
    memcpy(staging_data, data, data_size);

    immediate_submit(
        device,
        queue,
        fence,
        command_pool,
        cmd,
        [&](VkCommandBuffer cmd)
        {
            VkBufferCopy copy{};
            copy.dstOffset = 0;
            copy.srcOffset = 0;
            copy.size = data_size;

            vkCmdCopyBuffer(cmd, staging.buffer, buffer.buffer, 1, &copy);
        }
    );

    destroy_buffer(allocator, staging);

    return buffer;
}

VkImageMemoryBarrier2 image_barrier(
    VkImage image,
    VkImageLayout old_layout,
    VkImageLayout new_layout,
    VkPipelineStageFlags2 src_stage_mask,
    VkPipelineStageFlags2 dst_stage_mask,
    VkImageAspectFlags aspect /*= VK_IMAGE_ASPECT_COLOR_BIT*/
)
{
    VkImageMemoryBarrier2 barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    barrier.srcStageMask = src_stage_mask;
    barrier.dstStageMask = dst_stage_mask;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.image = image;

    VkImageSubresourceRange subresource_range{};
    subresource_range.aspectMask = aspect;
    subresource_range.baseMipLevel = 0;
    subresource_range.levelCount = VK_REMAINING_MIP_LEVELS;
    subresource_range.baseArrayLayer = 0;
    subresource_range.layerCount = VK_REMAINING_ARRAY_LAYERS;
    barrier.subresourceRange = subresource_range;

    return barrier;
}

VkSampler create_sampler(VkDevice device, VkFilter filter, VkSamplerMipmapMode mipmap, VkSamplerAddressMode address)
{
    VkSamplerCreateInfo info{ .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    info.magFilter = filter;
    info.minFilter = filter;
    info.mipmapMode = mipmap;
    info.addressModeU = address;
    info.addressModeV = address;
    info.addressModeW = address;
    info.maxLod = VK_LOD_CLAMP_NONE;

    VkSampler sampler{};

    VK_CHECK(vkCreateSampler(device, &info, nullptr, &sampler));

    return sampler;
}

void transition_images(
    VkCommandBuffer cmd,
    std::initializer_list<VkImage> images,
    std::initializer_list<VkImage> depth_images
)
{
    std::vector<VkImageMemoryBarrier2> barriers(images.size());

    VkImageSubresourceRange subresource_range{};
    subresource_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource_range.baseMipLevel = 0;
    subresource_range.levelCount = VK_REMAINING_MIP_LEVELS;
    subresource_range.baseArrayLayer = 0;
    subresource_range.layerCount = VK_REMAINING_ARRAY_LAYERS;

    for (auto image : images)
    {
        barriers.emplace_back(VkImageMemoryBarrier2{

            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .image = image,
            .subresourceRange = subresource_range,
        });
    }

    subresource_range.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    for (auto image : depth_images)
    {
        barriers.emplace_back(VkImageMemoryBarrier2{

            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .image = image,
            .subresourceRange = subresource_range,
        });
    }

    VkDependencyInfo info{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    info.imageMemoryBarrierCount = barriers.size();
    info.pImageMemoryBarriers = barriers.data();

    vkCmdPipelineBarrier2(cmd, &info);
}
