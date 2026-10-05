#pragma once

#include "swapchain.h"
#include "descriptors.h"
#include "rendergraph.h"

#include <volk.h>
#include <vk_mem_alloc.h>
#include <glm/ext/matrix_float4x4.hpp>

#include <cstdint>
#include <array>
#include <vector>

constexpr uint32_t FRAMES_IN_FLIGHT = 2;

struct FrameData
{
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkFence fence;
    VkSemaphore image_acquired_semaphore;
    VkQueryPool query_pool_timestamp;
    VkQueryPool query_pool_mesh_pipeline;
};

struct ObjectData;
struct Node;
struct MeshData;
struct GltfMesh;
struct Meshlet;

struct EngineStats
{
    double gpu_time;
};

class Engine
{
public:
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VmaAllocator allocator;
    VkQueue graphics_queue;
    uint32_t graphics_queue_family;
    VkSurfaceKHR surface;
    VkDebugUtilsMessengerEXT debug_messenger;

    VkPipelineLayout pipeline_layout;

    VkFence imm_fence;
    VkCommandPool imm_pool;
    VkCommandBuffer imm_buf;

    SDL_Window* window;
    bool relative_mouse_mode = true;
    bool swapchain_dirty;
    Swapchain swapchain;

    uint32_t frame_number;
    std::array<FrameData, FRAMES_IN_FLIGHT> frames;

    EngineStats stats;

    // Descriptors
    VkDescriptorPool desc_pool;
    VkDescriptorSetLayout buffer_layout;
    VkDescriptorSetLayout image_layout;
    VkDescriptorSetLayout sampler_layout;
    VkDescriptorSet buffer_set;
    VkDescriptorSet image_set;
    VkDescriptorSet sampler_set;
    VkDescriptorPool imgui_pool;

    ImageManager image_manager;

    Rendergraph graph;

    // Scene data
    std::vector<ObjectData> renderables;
    std::vector<MeshData> meshes;

    FrameData& get_current_frame()
    {
        return frames[frame_number % FRAMES_IN_FLIGHT];
    }

    void init_vulkan();
    void cleanup();
    void init_commands();
    void init_sync();
    void init_query_pool();
    void init_descriptors();
    void init_imgui();
    void destroy_imgui();
    void run();
    void register_object(const Node& node, const glm::mat4& top_matrix, const std::vector<Node>& children, const std::vector<GltfMesh>& meshes);
};
