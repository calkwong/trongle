#include "rendergraph.h"
#include "resources.h"
#include "descriptors.h"

#include <volk.h>
#include <fmt/core.h>

#include <cstdint>
#include <cassert>
#include <string>
#include <functional>
#include <utility>

void Pass::read_image(uint32_t handle)
{
    reads.push_back(handle);
}

void Pass::write_image(uint32_t handle)
{
    writes.push_back(handle);
}

bool ResourceInfo::operator==(const ResourceInfo& info) const
{
    return extent.width == info.extent.width
        && extent.height == info.extent.height
        && extent.depth == info.extent.depth
        && format == info.format && usage == info.usage && aspect == info.aspect;
}

uint32_t Rendergraph::register_resource(ResourceInfo info)
{
    uint32_t id = static_cast<uint32_t>(resource_infos.size());
    resource_infos.push_back(info);
    return id;
}

void Rendergraph::resolve_resources(VkDescriptorSet sampled_set, VkDescriptorSet storage_set)
{
    for (uint32_t handle = 0; handle < resource_infos.size(); handle++)
    {
        const auto& info = resource_infos[handle];

        auto [iter, inserted] = resource_cache.try_emplace(info);
        ResourceList& res_list = iter->second;

        auto create_new_resource = [&]()
        {
            auto image = create_image(device, allocator, info.extent, info.format, info.usage, info.aspect);

            auto tracked_res = res_list.resources.emplace_back(
                TrackedResource{
                    .image = image,
                    .handle = image_manager->register_image(device, sampled_set, storage_set, image.view, VK_IMAGE_LAYOUT_GENERAL),
                    .unused = 0 }
            );

            physical_resources.push_back(tracked_res);
            images_to_transition.push_back(res_list.resources[res_list.cursor++].image);
        };

        if (inserted)
        {
            create_new_resource();
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
                create_new_resource();
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
    for (auto& [_, res_list] : resource_cache)
    {
        for (auto i = res_list.cursor; i < res_list.resources.size(); i++)
        {
            auto& tracked_res = res_list.resources[i];
            tracked_res.unused += 1;

            if (tracked_res.unused > FRAMES_UNUSED)
            {
                destroy_image(device, allocator, tracked_res.image);

                image_manager->free_ids.push_back(tracked_res.handle);

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
    for (const auto& [_, res_list] : resource_cache)
    {
        for (const auto& res : res_list.resources)
        {
            destroy_image(device, allocator, res.image);
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
    return graph.physical_resources[handle].image.view;
}

VkImage get_image(const Rendergraph& graph, uint32_t handle)
{
    return graph.physical_resources[handle].image.image;
}
