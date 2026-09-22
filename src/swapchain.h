#pragma once

#include <volk.h>

#include <vector>
#include <cstdint>

struct Swapchain
{
    VkSwapchainKHR swapchain;
    VkExtent2D extent;
    std::vector<VkImage> images;
    std::vector<VkImageView> image_views;
    std::vector<VkSemaphore> render_done_semaphores;
};

Swapchain create_swapchain(VkPhysicalDevice physical_device, VkDevice device, VkSurfaceKHR surface, uint32_t width, uint32_t height, VkSwapchainKHR old_swapchain = 0);

struct SDL_Window;
void update_swapchain(Swapchain& swapchain, SDL_Window* window, VkPhysicalDevice physical_device, VkDevice device, VkSurfaceKHR surface);

void destroy_swapchain(VkDevice device, Swapchain swapchain);
