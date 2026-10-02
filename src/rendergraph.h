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

enum class ResourceType
{
    Owned,
    External,
};

struct ImageResourceDesc
{
    VkExtent3D extent;
    VkFormat format;
    VkImageUsageFlags usage;
    VkImageAspectFlags aspect;

    bool operator==(const ImageResourceDesc& desc) const;
};

template<>
struct std::hash<ImageResourceDesc>
{
    size_t operator()(const ImageResourceDesc& key) const noexcept;
};

struct BufferResourceDesc
{
    uint64_t alloc_size;
    VkBufferUsageFlags usage;

    bool operator==(const BufferResourceDesc& desc) const;
};

template<>
struct std::hash<BufferResourceDesc>
{
    size_t operator()(const BufferResourceDesc& key) const noexcept;
};

struct ImageResource
{
    Image image;
    uint32_t id;
};

struct ImageResourceInfo
{
    ResourceType type;

    union
    {
        ImageResourceDesc desc;
        ImageResource data;
    } res;
};

struct BufferResource
{
    Buffer buffer;
    uint32_t id;
};

struct BufferResourceInfo
{
    ResourceType type;

    union
    {
        BufferResourceDesc desc;
        BufferResource data;
    } res;
};

template<typename T>
struct ResourceList
{
    std::vector<T> resources;
    uint32_t cursor = 0;
};

struct AccessState
{
    AccessType last_writer;
    std::vector<AccessType> readers;
};

template<typename T>
struct FrameLocalAccessState
{
    AccessState state;
    ResourceList<T>* res_list;
    uint32_t index; // index into the vector res_list holds
    bool imported = false;
};

struct PhysicalImage
{
    VkImage image;
    VkImageView view;
    uint32_t handle;
};

struct PhysicalBuffer
{
    VkBuffer buffer;
    VkDeviceAddress address;
    VkDeviceSize size;
};

struct TrackedImage
{
    Image image;
    uint32_t handle;
    uint32_t unused;
    AccessState state;
};

struct TrackedBuffer
{
    Buffer buffer;
    uint32_t unused;
    AccessState state;
};

struct ImageManager;

struct PassInfo
{
    uint32_t handle;
    AccessType access_type;
    bool is_image;
};

struct Pass
{
    std::string name;
    std::vector<PassInfo> reads;
    std::vector<PassInfo> writes;

    void read_image(uint32_t handle, AccessType access_type);
    void write_image(uint32_t handle, AccessType access_type);

    void read_buffer(uint32_t handle, AccessType access_type);
    void write_buffer(uint32_t handle, AccessType access_type);
};

// Immediate mode
struct Rendergraph
{
    VkDevice device;
    VmaAllocator allocator;
    ImageManager* image_manager;
    uint32_t swapchain_index = -1u;

    std::vector<Pass> passes;
    std::vector<ImageResourceInfo> image_infos;
    std::vector<BufferResourceInfo> buffer_infos;
    std::unordered_map<ImageResourceDesc, ResourceList<TrackedImage>> image_resource_cache;
    std::unordered_map<BufferResourceDesc, ResourceList<TrackedBuffer>> buffer_resource_cache;

    // TODO: handle unused -> remove zombie key/value pair from cache
    std::unordered_map<uint64_t, TrackedImage> persistent_image_cache;
    std::unordered_map<uint64_t, TrackedBuffer> persistent_buffer_cache;
    std::vector<VkImageMemoryBarrier2> image_barriers;

    // Physical resources are for resolving handles, views, ids when executing Pass callback functions
    std::vector<PhysicalImage> physical_images;
    std::vector<PhysicalBuffer> physical_buffers;

    std::vector<std::function<void()>> executes;
    std::vector<VkMemoryBarrier2> barriers;

    // Rebuilt each frame from persistent hashmap which stores final state from last frame (or initial state on initialization)
    std::vector<FrameLocalAccessState<TrackedImage>> image_states;
    std::vector<FrameLocalAccessState<TrackedBuffer>> buffer_states;

    uint32_t create_task_image(ImageResourceDesc info);

    // TODO: implement mapping buffer to descriptor
    uint32_t create_task_buffer(BufferResourceDesc info);

    // TODO: these 2 need testing
    uint32_t import_image(Image image, uint32_t descriptor_handle);
    uint32_t import_buffer(Buffer buffer, uint32_t descriptor_handle);

    uint32_t import_swapchain(VkImage image, VkImageView view);

    // Grab an existing physical resource, or create one + assign bindless ID
    // Also assigns a slot in image/buffer_states for per pass update of FrameLocalAccessState
    void resolve_images();
    void resolve_buffers();

    void add_pass(const std::string& name, std::function<void(Pass& pass)> setup, std::function<void()> execute);

    void update_cached_resource_states();

    void compile(VkCommandBuffer cmd);

    // TODO - can we combine readers into bitflags instead of using vector?
    // Automatic barriers - determines dependencies and updates FrameLocalAccessState to build per pass barrier
    // REVIEW - Assumes RW resources are W only; this may cause correctness issue when implementing pass reordering and/or pass merging
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
