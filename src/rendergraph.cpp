#include "rendergraph.h"
#include "resources.h"
#include "descriptors.h"
#include "sync.h"

#include <volk.h>
#include <fmt/core.h>

#include <cstdint>
#include <cassert>
#include <string>
#include <functional>
#include <utility>
#include <cstddef>

// REVIEW
size_t std::hash<ImageResourceDesc>::operator()(const ImageResourceDesc& key) const noexcept
{
    size_t ret = 0;

    auto combine = [&](uint32_t value)
    {
        ret ^= hash<uint32_t>()(value)
            + static_cast<size_t>(0x9e3779b9)
            + (ret << 6)
            + (ret >> 2);
    };

    combine(key.extent.width);
    combine(key.extent.height);
    combine(key.extent.depth);
    combine(static_cast<uint32_t>(key.format));
    combine(static_cast<uint32_t>(key.usage));
    combine(static_cast<uint32_t>(key.aspect));

    return ret;
}

// REVIEW
size_t std::hash<BufferResourceDesc>::operator()(const BufferResourceDesc& key) const noexcept
{
    size_t ret = 0;

    auto combine = [&](uint32_t value)
    {
        ret ^= hash<uint32_t>()(value)
            + static_cast<size_t>(0x9e3779b9)
            + (ret << 6)
            + (ret >> 2);
    };

    combine(key.alloc_size);
    combine(static_cast<uint32_t>(key.usage));

    return ret;
}

void Pass::read(uint32_t handle, AccessType access_type)
{
    reads.push_back(PassInfo{ handle, access_type });
}

void Pass::write(uint32_t handle, AccessType access_type)
{
    writes.push_back(PassInfo{ handle, access_type });
}

bool ImageResourceDesc::operator==(const ImageResourceDesc& desc) const
{
    return extent.width == desc.extent.width
        && extent.height == desc.extent.height
        && extent.depth == desc.extent.depth
        && format == desc.format && usage == desc.usage && aspect == desc.aspect;
}

bool BufferResourceDesc::operator==(const BufferResourceDesc& desc) const
{
    return alloc_size == desc.alloc_size
        && usage == desc.usage;
}

uint32_t Rendergraph::create_task_image(ImageResourceDesc desc)
{
    uint32_t id = static_cast<uint32_t>(resource_infos.size());

    auto info = ResourceInfo{};
    info.type = ResourceType::ImageResourceDesc;
    info.res_desc.image = desc;

    resource_infos.push_back(info);

    return id;
}

uint32_t Rendergraph::create_task_buffer(BufferResourceDesc desc)
{
    uint32_t id = static_cast<uint32_t>(resource_infos.size());

    auto info = ResourceInfo{};
    info.type = ResourceType::BufferResourceDesc;
    info.res_desc.buffer = desc;

    resource_infos.push_back(info);

    return id;
}

void Rendergraph::resolve_resources(VkDescriptorSet sampled_set, VkDescriptorSet storage_set)
{
    auto create_new_resource = [&](ResourceInfo info, ResourceList& res_list)
    {
        switch (info.type)
        {
        case ResourceType::ImageResourceDesc:
        {
            auto image = create_image(device, allocator, info.res_desc.image.extent, info.res_desc.image.format, info.res_desc.image.usage, info.res_desc.image.aspect);

            TrackedResource tracked_res{};
            tracked_res.type = TrackedResourceType::Image;
            tracked_res.res.image = image;
            tracked_res.handle = image_manager->register_image(device, sampled_set, storage_set, image.view, VK_IMAGE_LAYOUT_GENERAL);
            tracked_res.unused = 0;
            tracked_res.state = AccessInfo{ VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE };

            res_list.resources.push_back(tracked_res);
            images_to_transition.push_back(res_list.resources[res_list.cursor++].res.image);
            physical_resources.push_back(tracked_res);
            break;
        }
        case ResourceType::BufferResourceDesc:
        {
            auto buffer = create_buffer(device, allocator, info.res_desc.buffer.alloc_size, 0, info.res_desc.buffer.usage);

            TrackedResource tracked_res{};
            tracked_res.type = TrackedResourceType::Buffer;
            tracked_res.res.buffer = buffer;
            tracked_res.handle = -1;
            tracked_res.unused = 0;
            tracked_res.state = AccessInfo{ VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE };

            res_list.resources.push_back(tracked_res);
            physical_resources.push_back(tracked_res);
            break;
        }
        default:
            assert(0 && "Imported resources not yet implemented");
        }
    };

    auto create_or_extract_resource = [&](ResourceInfo info, ResourceList& res_list, bool inserted)
    {
        if (inserted)
        {
            create_new_resource(info, res_list);
        }
        else
        {
            // Resource exists
            if (res_list.cursor < res_list.resources.size())
            {
                physical_resources.push_back(res_list.resources[res_list.cursor++]);
            }
            // Create new resource
            else
            {
                create_new_resource(info, res_list);
            }
        }
    };

    for (uint32_t handle = 0; handle < resource_infos.size(); handle++)
    {
        const auto& info = resource_infos[handle];

        switch (info.type)
        {
        case ResourceType::ImageResourceDesc:
        {
            auto [iter, inserted] = image_resource_cache.try_emplace(info.res_desc.image);
            ResourceList& res_list = iter->second;
            create_or_extract_resource(info, res_list, inserted);
            break;
        }
        case ResourceType::BufferResourceDesc:
        {
            auto [iter, inserted] = buffer_resource_cache.try_emplace(info.res_desc.buffer);
            ResourceList& res_list = iter->second;
            create_or_extract_resource(info, res_list, inserted);
            break;
        }
        default:
        {
            assert(0 && "Imported resources not yet implemented");
            break;
        }
        }
    }
}

void Rendergraph::transition_image_layouts(VkCommandBuffer cmd)
{
    transition_images(cmd, images_to_transition);
}

void Rendergraph::reset()
{
    resource_infos.clear();
    physical_resources.clear();
    images_to_transition.clear();
    passes.clear();
    executes.clear();
    barriers.clear();

    // Reset cursor to the beginning of resources and recycle zombie descriptor IDs
    for (auto& [_, res_list] : image_resource_cache)
    {
        for (auto i = res_list.cursor; i < res_list.resources.size(); i++)
        {
            auto& tracked_res = res_list.resources[i];
            tracked_res.unused += 1;

            if (tracked_res.unused > FRAMES_UNUSED)
            {
                destroy_image(device, allocator, tracked_res.res.image);

                image_manager->free_ids.push_back(tracked_res.handle);

                // TODO: refactor - we don't want to be shifting vector elements
                res_list.resources.erase(res_list.resources.begin() + i);
            }
        }
        res_list.cursor = 0;
    }

    for (auto& [_, res_list] : buffer_resource_cache)
    {
        for (auto i = res_list.cursor; i < res_list.resources.size(); i++)
        {
            auto& tracked_res = res_list.resources[i];
            tracked_res.unused += 1;

            if (tracked_res.unused > FRAMES_UNUSED)
            {
                destroy_buffer(allocator, tracked_res.res.buffer);

                // TODO: refactor - we don't want to be shifting vector elements
                res_list.resources.erase(res_list.resources.begin() + i);
            }
        }
        res_list.cursor = 0;
    }
}

void Rendergraph::add_pass(const std::string& name, std::function<void(Pass& pass)> setup, std::function<void()> execute)
{
    auto& pass = passes.emplace_back(Pass{ .name = name });

    setup(pass);

    // REVIEW
    executes.push_back(std::move(execute));
}

void Rendergraph::cleanup()
{
    for (const auto& [_, res_list] : image_resource_cache)
    {
        for (const auto& res : res_list.resources)
        {
            destroy_image(device, allocator, res.res.image);
        }
    }

    for (const auto& [_, res_list] : buffer_resource_cache)
    {
        for (const auto& res : res_list.resources)
        {
            destroy_buffer(allocator, res.res.buffer);
        }
    }
}

// TODO: refactor parameters
void Rendergraph::compile(VkCommandBuffer cmd, VkDescriptorSet sampled_set, VkDescriptorSet storage_set)
{
    resolve_resources(sampled_set, storage_set);
    transition_image_layouts(cmd);
    prepare_barriers();
}

// TODO: remove gigabarriers, then refactor execute()
void Rendergraph::prepare_barriers()
{
    for (const auto& _ : passes)
    {
        barriers.emplace_back(
            VkMemoryBarrier2{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT }
        );
    }
}

void Rendergraph::execute(VkCommandBuffer cmd)
{
    for (auto i = 0; i < executes.size(); i++)
    {
        auto& barrier = barriers[i];

        VkDependencyInfo info{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        // TODO: refactor likely needed after removing gigabarriers
        info.memoryBarrierCount = 1;
        info.pMemoryBarriers = &barrier;

        vkCmdPipelineBarrier2(cmd, &info);

        auto& callback = executes[i];
        callback();
    }
}

uint32_t get_image_id(const Rendergraph& graph, uint32_t handle)
{
    return graph.physical_resources[handle].handle;
}

VkImageView get_image_view(const Rendergraph& graph, uint32_t handle)
{
    return graph.physical_resources[handle].res.image.view;
}

VkImage get_image(const Rendergraph& graph, uint32_t handle)
{
    return graph.physical_resources[handle].res.image.image;
}

VkDeviceAddress get_buffer_address(const Rendergraph& graph, uint32_t handle)
{
    return graph.physical_resources[handle].res.buffer.address;
}

VkBuffer get_buffer(const Rendergraph& graph, uint32_t handle)
{
    return graph.physical_resources[handle].res.buffer.buffer;
}

VkDeviceSize get_buffer_size(const Rendergraph& graph, uint32_t handle)
{
    return graph.physical_resources[handle].res.buffer.size;
}
