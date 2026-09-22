#pragma once

#include "swapchain.h"

#include <volk.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <array>

constexpr uint32_t FRAMES_IN_FLIGHT = 2;

struct FrameData
{
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkFence fence;
    VkSemaphore image_acquired_semaphore;

    // TODO: deletion queue
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

    VkFence imm_fence;
    VkCommandPool imm_pool;
    VkCommandBuffer imm_buf;

    SDL_Window* window;
    bool swapchain_dirty;
    Swapchain swapchain;

    uint32_t frame_number;
    std::array<FrameData, FRAMES_IN_FLIGHT> frames;

    FrameData& get_current_frame()
    {
        return frames[frame_number % FRAMES_IN_FLIGHT];
    }

    void init_vulkan();
    void cleanup();
    void init_commands();
    void init_sync();
    void run();
};
