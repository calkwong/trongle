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

void Pass::read_image(uint32_t handle, AccessType access_type)
{
    reads.push_back(PassInfo{ handle, access_type, true });
}

void Pass::write_image(uint32_t handle, AccessType access_type)
{
    writes.push_back(PassInfo{ handle, access_type, true });
}

void Pass::read_buffer(uint32_t handle, AccessType access_type)
{
    reads.push_back(PassInfo{ handle, access_type, false });
}

void Pass::write_buffer(uint32_t handle, AccessType access_type)
{
    writes.push_back(PassInfo{ handle, access_type, false });
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
    uint32_t id = static_cast<uint32_t>(image_descs.size());
    image_descs.push_back(desc);

    return id;
}

uint32_t Rendergraph::create_task_buffer(BufferResourceDesc desc)
{
    uint32_t id = static_cast<uint32_t>(buffer_descs.size());
    buffer_descs.push_back(desc);

    return id;
}

void Rendergraph::resolve_images()
{
    auto create_new_image_resource = [&](const ImageResourceDesc& desc, ResourceList<TrackedImage>& res_list)
    {
        auto image = create_image(device, allocator, desc.extent, desc.format, desc.usage, desc.aspect);
        auto index = static_cast<uint32_t>(res_list.resources.size());
        auto& tracked_res = res_list.resources.emplace_back(
            TrackedImage{
                .image = image,
                .handle = image_manager->register_image(device, image.view, VK_IMAGE_LAYOUT_GENERAL),
                .unused = 0,
                .state = AccessState{} }
        );
        images_to_transition.push_back(res_list.resources[res_list.cursor++].image);
        physical_images.push_back(tracked_res);

        image_states.emplace_back(FrameLocalAccessState<TrackedImage>{ .state = AccessState{}, .res_list = &res_list, .index = index });
    };

    auto create_or_extract_resource = [&](const ImageResourceDesc& desc, ResourceList<TrackedImage>& res_list, bool inserted)
    {
        if (inserted)
        {
            create_new_image_resource(desc, res_list);
        }
        else
        {
            // Resource exists
            if (res_list.cursor < res_list.resources.size())
            {
                auto index = res_list.cursor;
                const auto& res = res_list.resources[res_list.cursor++];
                physical_images.push_back(res);

                image_states.emplace_back(FrameLocalAccessState<TrackedImage>{ .state = res.state, .res_list = &res_list, .index = index });
            }
            // Create new resource
            else
            {
                create_new_image_resource(desc, res_list);
            }
        }
    };

    for (const auto& desc : image_descs)
    {
        auto [iter, inserted] = image_resource_cache.try_emplace(desc);
        ResourceList<TrackedImage>& res_list = iter->second;
        create_or_extract_resource(desc, res_list, inserted);
    }

    assert(image_states.size() == image_descs.size());
}

void Rendergraph::resolve_buffers()
{
    auto create_new_buffer_resource = [&](const BufferResourceDesc& desc, ResourceList<TrackedBuffer>& res_list)
    {
        auto buffer = create_buffer(device, allocator, desc.alloc_size, 0, desc.usage);
        auto index = static_cast<uint32_t>(res_list.resources.size());
        auto& tracked_res = res_list.resources.emplace_back(
            TrackedBuffer{
                .buffer = buffer,
                .unused = 0,
                .state = AccessState{} }
        );
        physical_buffers.push_back(tracked_res);

        buffer_states.emplace_back(FrameLocalAccessState<TrackedBuffer>{ .state = AccessState{}, .res_list = &res_list, .index = index });
    };

    auto create_or_extract_resource = [&](const BufferResourceDesc& desc, ResourceList<TrackedBuffer>& res_list, bool inserted)
    {
        if (inserted)
        {
            create_new_buffer_resource(desc, res_list);
        }
        else
        {
            // Resource exists
            if (res_list.cursor < res_list.resources.size())
            {
                auto index = res_list.cursor;
                const auto& res = res_list.resources[res_list.cursor++];
                physical_buffers.push_back(res);

                buffer_states.emplace_back(FrameLocalAccessState<TrackedBuffer>{ .state = res.state, .res_list = &res_list, .index = index });
            }
            // Create new resource
            else
            {
                create_new_buffer_resource(desc, res_list);
            }
        }
    };

    for (const auto& desc : buffer_descs)
    {
        auto [iter, inserted] = buffer_resource_cache.try_emplace(desc);
        ResourceList<TrackedBuffer>& res_list = iter->second;
        create_or_extract_resource(desc, res_list, inserted);
    }

    assert(buffer_states.size() == buffer_descs.size());
}

void Rendergraph::transition_image_layouts(VkCommandBuffer cmd)
{
    transition_images(cmd, images_to_transition);
}

void Rendergraph::reset()
{
    image_descs.clear();
    buffer_descs.clear();
    physical_images.clear();
    physical_buffers.clear();
    images_to_transition.clear();
    passes.clear();
    executes.clear();
    barriers.clear();
    image_states.clear();
    buffer_states.clear();

    // Reset unused counter, set cursor back to the beginning of resource list and recycle zombie descriptor IDs
    for (auto& [_, res_list] : image_resource_cache)
    {
        auto tracked_res = res_list.resources.begin();
        while (tracked_res != res_list.resources.begin() + res_list.cursor)
        {
            tracked_res->unused = 0;
            tracked_res++;
        }

        // Advances from cursor
        while (tracked_res != res_list.resources.end())
        {
            tracked_res->unused += 1;

            if (tracked_res->unused > FRAMES_UNUSED)
            {
                destroy_image(device, allocator, tracked_res->image);
                image_manager->free_ids.push_back(tracked_res->handle);

                // TODO: refactor - we don't want to be shifting vector elements
                tracked_res = res_list.resources.erase(tracked_res);
            }
            else
            {
                tracked_res++;
            }
        }

        res_list.cursor = 0;
    }

    for (auto& [_, res_list] : buffer_resource_cache)
    {
        auto tracked_res = res_list.resources.begin();
        while (tracked_res != res_list.resources.begin() + res_list.cursor)
        {
            tracked_res->unused = 0;
            tracked_res++;
        }

        // Advances from cursor
        while (tracked_res != res_list.resources.end())
        {
            tracked_res->unused += 1;

            if (tracked_res->unused > FRAMES_UNUSED)
            {
                destroy_buffer(allocator, tracked_res->buffer);

                // TODO: refactor - we don't want to be shifting vector elements
                tracked_res = res_list.resources.erase(tracked_res);
            }
            else
            {
                tracked_res++;
            }
        }

        res_list.cursor = 0;
    }
}

void Rendergraph::add_pass(const std::string& name, std::function<void(Pass& pass)> setup, std::function<void()> execute)
{
    auto& pass = passes.emplace_back(Pass{ .name = name });

    setup(pass);

    executes.push_back(std::move(execute));
}

void Rendergraph::cleanup()
{
    for (const auto& [_, res_list] : image_resource_cache)
    {
        for (const auto& res : res_list.resources)
        {
            destroy_image(device, allocator, res.image);
        }
    }

    for (const auto& [_, res_list] : buffer_resource_cache)
    {
        for (const auto& res : res_list.resources)
        {
            destroy_buffer(allocator, res.buffer);
        }
    }
}

void Rendergraph::compile(VkCommandBuffer cmd)
{
    resolve_images();
    resolve_buffers();

    // TODO: we could possibly combine this into prepare_barriers
    transition_image_layouts(cmd);

    prepare_barriers();
    update_cached_resource_states();
}

void Rendergraph::prepare_barriers()
{
#if 1
    auto process_write_dependencies = [&](AccessState& res_state, VkMemoryBarrier2& barrier, const PassInfo& write)
    {
        if (!res_state.readers.empty()) // Src is readers
        {
            for (const auto& reader : res_state.readers)
            {
                auto reader_access_info = get_access_info(reader);
                barrier.srcStageMask |= reader_access_info.stage_mask;
                barrier.srcAccessMask |= reader_access_info.access_mask;
            }
            res_state.readers.clear();
        }
        else // Src is last writer
        {
            auto last_writer_access_info = get_access_info(res_state.last_writer);
            barrier.srcStageMask |= last_writer_access_info.stage_mask;
            barrier.srcAccessMask |= last_writer_access_info.access_mask;
        }
        res_state.last_writer = write.access_type;

        auto dst_access_info = get_access_info(write.access_type);
        barrier.dstStageMask |= dst_access_info.stage_mask;
        barrier.dstAccessMask |= dst_access_info.access_mask;
    };

    auto process_read_dependencies = [&](AccessState& res_state, VkMemoryBarrier2& barrier, const PassInfo& read)
    {
        // Append self to list of readers
        res_state.readers.push_back(read.access_type);

        // Record last writer in barrier srcStage & srcAccess
        auto last_writer_access_info = get_access_info(res_state.last_writer);
        barrier.srcStageMask |= last_writer_access_info.stage_mask;
        barrier.srcAccessMask |= last_writer_access_info.access_mask;

        auto dst_access_info = get_access_info(read.access_type);
        barrier.dstStageMask |= dst_access_info.stage_mask;
        barrier.dstAccessMask |= dst_access_info.access_mask;
    };

    for (const auto& pass : passes)
    {
        VkMemoryBarrier2 barrier{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };

        for (const auto& write : pass.writes)
        {
            auto id = write.handle;
            if (write.is_image)
            {
                AccessState& res_state = image_states[id].state;
                process_write_dependencies(res_state, barrier, write);
            }
            else
            {
                AccessState& res_state = buffer_states[id].state;
                process_write_dependencies(res_state, barrier, write);
            }
        }

        for (const auto& read : pass.reads)
        {
            auto id = read.handle;
            if (read.is_image)
            {
                AccessState& res_state = image_states[id].state;
                process_write_dependencies(res_state, barrier, read);
            }
            else
            {
                AccessState& res_state = buffer_states[id].state;
                process_write_dependencies(res_state, barrier, read);
            }
        }

        barriers.push_back(barrier);
    }
#else
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
#endif
}

void Rendergraph::execute(VkCommandBuffer cmd)
{
    assert(barriers.size() == executes.size());
    for (auto i = 0; i < executes.size(); i++)
    {
        auto& barrier = barriers[i];

        VkDependencyInfo info{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        // TODO: review if this needs to be refactored for non gigabarrier path
        info.memoryBarrierCount = 1;
        info.pMemoryBarriers = &barrier;

        vkCmdPipelineBarrier2(cmd, &info);

        auto& callback = executes[i];
        callback();
    }
}

void Rendergraph::update_cached_resource_states()
{
    for (const auto& state : image_states)
    {
        state.res_list->resources[state.index].state = state.state;
    }

    for (const auto& state : buffer_states)
    {
        state.res_list->resources[state.index].state = state.state;
    }
}

uint32_t get_image_id(const Rendergraph& graph, uint32_t handle)
{
    return graph.physical_images[handle].handle;
}

VkImageView get_image_view(const Rendergraph& graph, uint32_t handle)
{
    return graph.physical_images[handle].image.view;
}

VkImage get_image(const Rendergraph& graph, uint32_t handle)
{
    return graph.physical_images[handle].image.image;
}

VkDeviceAddress get_buffer_address(const Rendergraph& graph, uint32_t handle)
{
    return graph.physical_buffers[handle].buffer.address;
}

VkBuffer get_buffer(const Rendergraph& graph, uint32_t handle)
{
    return graph.physical_buffers[handle].buffer.buffer;
}

VkDeviceSize get_buffer_size(const Rendergraph& graph, uint32_t handle)
{
    return graph.physical_buffers[handle].buffer.size;
}
