#pragma once

#include "resources.h"
#include "sync.h"

#include <volk.h>

#include <vector>
#include <cstdint>
#include <unordered_map>
#include <functional>
#include <cstddef>
#include <string>

constexpr uint32_t FRAMES_UNUSED = 4;

struct ResourceInfo
{
    VkExtent3D extent;
    VkFormat format;
    VkImageUsageFlags usage;
    VkImageAspectFlags aspect;

    bool operator==(const ResourceInfo& info) const;
};

template<>
struct std::hash<ResourceInfo>
{
    size_t operator()(const ResourceInfo& key) const noexcept;
};

struct TrackedResource
{
    Image image;
    uint32_t handle;
    uint32_t unused;
    AccessInfo state;
};

struct ResourceList
{
    std::vector<TrackedResource> resources;
    uint32_t cursor = 0;
};

struct ImageManager;

struct PassInfo
{
    uint32_t handle;
    AccessType access_type;
};

struct Pass
{
    std::string name;
    std::vector<PassInfo> reads;
    std::vector<PassInfo> writes;

    // TODO: AccessType
    void read_image(uint32_t handle, AccessType access_type);
    void write_image(uint32_t handle, AccessType access_type);
};

// Immediate mode
struct Rendergraph
{
    VkDevice device;
    VmaAllocator allocator;
    ImageManager* image_manager;

    std::unordered_map<ResourceInfo, ResourceList> resource_cache;
    std::vector<ResourceInfo> resource_infos;
    std::vector<TrackedResource> physical_resources;
    std::vector<Image> images_to_transition;
    std::vector<Pass> passes;
    std::vector<std::function<void()>> executes;
    std::vector<VkMemoryBarrier2> barriers;

    uint32_t register_resource(ResourceInfo info);

    // Grab an existing physical resource, or create one + assign bindless ID
    void resolve_resources(VkDescriptorSet sampled_set, VkDescriptorSet storage_set);

    // Transition newly created images from UNDEFINED -> GENERAL
    void transition_image_layouts(VkCommandBuffer cmd);

    void add_pass(const std::string& name, std::function<void(Pass& pass)> setup, std::function<void()> execute);

    void compile(VkCommandBuffer cmd, VkDescriptorSet sampled_set, VkDescriptorSet storage_set);
    void prepare_barriers();
    void execute(VkCommandBuffer cmd);
    void reset();
    void cleanup();
};

// Returns the bindless descriptor handle
uint32_t get_image_id(const Rendergraph& graph, uint32_t handle);

VkImageView get_image_view(const Rendergraph& graph, uint32_t handle);
VkImage get_image(const Rendergraph& graph, uint32_t handle);

// TODO:
// - read/write, AccessType
// - auto barriers, need a global resource state per resource
