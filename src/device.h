#pragma once

#include <volk.h>
#include <VkBootstrap.h>
#include <SDL3/SDL_video.h>

#include <cstdint>

vkb::Instance create_instance();
SDL_Window* create_window(VkInstance instance, uint32_t window_width, uint32_t window_height, SDL_WindowFlags window_flags);
void create_surface(SDL_Window* window, VkInstance instance, VkSurfaceKHR* surface);
vkb::Result<vkb::PhysicalDevice> create_physical_device(vkb::Instance& vkb_inst, VkSurfaceKHR surface);

// VkBootstrap specific - checking for extension support, this may be incorrect.
// Without bootstrap, the right way to do this is query extension name -> query feature -> set feature -> device creation using extensions & features set
vkb::Device create_device(vkb::Result<vkb::PhysicalDevice> vkb_physical_device);
void get_graphics_queue_and_family(vkb::Device vkb_device, VkQueue& graphics_queue, uint32_t& graphics_queue_family);
