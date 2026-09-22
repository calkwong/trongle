#include "swapchain.h"
#include "common.h"

#include <volk.h>
#include <fmt/core.h>
#include <VkBootstrap.h>
#include "SDL3/SDL_video.h"

#include <cstdlib>
#include <cstdint>

Swapchain create_swapchain(VkPhysicalDevice physical_device, VkDevice device, VkSurfaceKHR surface, uint32_t width, uint32_t height, VkSwapchainKHR old_swapchain /*= 0*/)
{
    vkb::SwapchainBuilder swapchain_builder{ physical_device, device, surface };

    swapchain_builder = swapchain_builder
                            .set_desired_format(VkSurfaceFormatKHR{ .format = VK_FORMAT_B8G8R8A8_UNORM, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
                            .set_desired_present_mode(VK_PRESENT_MODE_MAILBOX_KHR)
                            // .set_desired_present_mode(VK_PRESENT_MODE_IMMEDIATE_KHR)
                            .set_desired_extent(width, height)
                            .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    if (old_swapchain)
    {
        swapchain_builder.set_old_swapchain(old_swapchain);
    }

    auto swapchain_ret = swapchain_builder.build();

    if (!swapchain_ret)
    {
        fmt::println("Failed to create swapchain.");
        std::abort();
    }
    vkb::Swapchain vkb_swapchain = swapchain_ret.value();

    Swapchain swapchain{};

    swapchain.swapchain = vkb_swapchain.swapchain;
    swapchain.extent = vkb_swapchain.extent;
    swapchain.images = vkb_swapchain.get_images().value();
    swapchain.image_views = vkb_swapchain.get_image_views().value();

    auto swapchain_count = swapchain.images.size();
    swapchain.render_done_semaphores.resize(swapchain_count);
    VkSemaphoreCreateInfo semaphore_info{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    for (auto i = 0; i < swapchain_count; i++)
    {
        VK_CHECK(vkCreateSemaphore(device, &semaphore_info, nullptr, &swapchain.render_done_semaphores[i]));
    }

    return swapchain;
}

void update_swapchain(Swapchain& swapchain, SDL_Window* window, VkPhysicalDevice physical_device, VkDevice device, VkSurfaceKHR surface)
{
    int width{};
    int height{};
    SDL_GetWindowSizeInPixels(window, &width, &height);

    vkDeviceWaitIdle(device);

    Swapchain old_swapchain = swapchain;
    swapchain = create_swapchain(physical_device, device, surface, width, height, old_swapchain.swapchain);
    destroy_swapchain(device, old_swapchain);

    fmt::println("Swapchain resized: {} {}", width, height);
}

void destroy_swapchain(VkDevice device, Swapchain swapchain)
{
    for (auto image_view : swapchain.image_views)
    {
        vkDestroyImageView(device, image_view, nullptr);
    }

    vkDestroySwapchainKHR(device, swapchain.swapchain, nullptr);

    for (auto semaphore : swapchain.render_done_semaphores)
    {
        vkDestroySemaphore(device, semaphore, nullptr);
    }
}
