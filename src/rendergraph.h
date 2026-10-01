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

struct ImageResourceDesc
{
    VkExtent3D extent;
    VkFormat format;
    VkImageUsageFlags usage;
    VkImageAspectFlags aspect;

    bool operator==(const ImageResourceDesc& desc) const;
};

struct BufferResourceDesc
{
    uint64_t alloc_size;
    VkBufferUsageFlags usage;

    bool operator==(const BufferResourceDesc& desc) const;
};

template<>
struct std::hash<ImageResourceDesc>
{
    size_t operator()(const ImageResourceDesc& key) const noexcept;
};

template<>
struct std::hash<BufferResourceDesc>
{
    size_t operator()(const BufferResourceDesc& key) const noexcept;
};

struct TrackedImage
{
    Image image;
    uint32_t handle;
    uint32_t unused;
    AccessInfo state;
};

struct TrackedBuffer
{
    Buffer buffer;
    uint32_t unused;
    AccessInfo state;
};

template<typename T>
struct ResourceList
{
    std::vector<T> resources;
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

    // TODO: likely need image and buffer variants
    void read(uint32_t handle, AccessType access_type);
    void write(uint32_t handle, AccessType access_type);
};

// Immediate mode
struct Rendergraph
{
    VkDevice device;
    VmaAllocator allocator;
    ImageManager* image_manager;

    std::vector<Pass> passes;
    std::vector<ImageResourceDesc> image_descs;
    std::vector<BufferResourceDesc> buffer_descs;
    std::unordered_map<ImageResourceDesc, ResourceList<TrackedImage>> image_resource_cache;
    std::unordered_map<BufferResourceDesc, ResourceList<TrackedBuffer>> buffer_resource_cache;
    std::vector<Image> images_to_transition;
    std::vector<TrackedImage> physical_images;
    std::vector<TrackedBuffer> physical_buffers;
    std::vector<std::function<void()>> executes;
    std::vector<VkMemoryBarrier2> barriers;

    uint32_t create_task_image(ImageResourceDesc info);
    uint32_t create_task_buffer(BufferResourceDesc info);

    // Grab an existing physical resource, or create one + assign bindless ID
    void resolve_images(VkDescriptorSet sampled_set, VkDescriptorSet storage_set);
    void resolve_buffers(VkDescriptorSet sampled_set, VkDescriptorSet storage_set);

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
VkDeviceAddress get_buffer_address(const Rendergraph& graph, uint32_t handle);
VkBuffer get_buffer(const Rendergraph& graph, uint32_t handle);
VkDeviceSize get_buffer_size(const Rendergraph& graph, uint32_t handle);

// TODO:
// - read/write, AccessType
// - auto barriers, need a global resource state per resource
// - fix cases where unused needs to be set back to 0
// - possible vector shifting bug
