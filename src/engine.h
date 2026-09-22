#pragma once

#include "swapchain.h"

#include <volk.h>
#include <vk_mem_alloc.h>
#include <glm/ext/matrix_float4x4.hpp>

#include <cstdint>
#include <array>
#include <memory>
#include <vector>

constexpr uint32_t FRAMES_IN_FLIGHT = 2;

struct FrameData
{
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkFence fence;
    VkSemaphore image_acquired_semaphore;

    // TODO: deletion queue
};

struct ShaderPass;
struct ObjectData;
struct Node;
struct MeshData;

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
    std::unique_ptr<ShaderPass> mesh_pass;

    VkFence imm_fence;
    VkCommandPool imm_pool;
    VkCommandBuffer imm_buf;

    SDL_Window* window;
    bool swapchain_dirty;
    Swapchain swapchain;

    uint32_t frame_number;
    std::array<FrameData, FRAMES_IN_FLIGHT> frames;

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
    void run();
    void register_object(const Node* node, const glm::mat4& top_matrix);
};
