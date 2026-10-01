#include "rendergraph.h"
#include "resources.h"
#include "descriptors.h"

#include <volk.h>
#include <fmt/core.h>

#include <cstdint>
#include <cassert>

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
            if (res_list.cursor < res_list.resources.size()) // Resource exists
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

    // Reset cursor to the beginning of resources
    for (auto& [_, res_list] : resource_cache)
    {
        for (auto i = res_list.cursor; i < res_list.resources.size(); i++)
        {
            auto& tracked_res = res_list.resources[i];
            tracked_res.unused += 1;

            if (tracked_res.unused > FRAMES_UNUSED)
            {
                destroy_image(device, allocator, tracked_res.image);

                // Recycle descriptor ID
                image_manager->free_ids.push_back(tracked_res.handle);

                // TODO: refactor - we don't want to be shifting vector elements
                res_list.resources.erase(res_list.resources.begin() + i);
            }
        }
        res_list.cursor = 0;
    }
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

uint32_t get_image_id(const Rendergraph& graph, uint32_t handle)
{
    return graph.physical_resources[handle].handle;
}

VkImageView get_image_view(const Rendergraph& graph, uint32_t handle)
{
    return graph.physical_resources[handle].image.view;
}
