#include "device.h"

#include <volk.h>
#include <VkBootstrap.h>
#include <fmt/core.h>
#include <SDL3/SDL_hints.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_video.h>
#include <SDL3/SDL_vulkan.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace
{
VkBool32 custom_debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    VkDebugUtilsMessageTypeFlagsEXT message_type,
    const VkDebugUtilsMessengerCallbackDataEXT* p_callback_data,
    void* p_user_data
)
{
    auto ms = vkb::to_string_message_severity(message_severity);
    auto mt = vkb::to_string_message_type(message_type);
    if (message_type & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT)
    {
        if (strcmp(p_callback_data->pMessageIdName, "VUID-RuntimeSpirv-OpVariable-08746") == 0)
            return VK_FALSE;
        fmt::println("[{}: {}] - {}\n{}\n", ms, mt, p_callback_data->pMessageIdName, p_callback_data->pMessage);
    }
    else
        fmt::println("[{}: {}]\n{}\n", ms, mt, p_callback_data->pMessage);

    return VK_FALSE;
}
} // namespace

vkb::Instance create_instance()
{
    vkb::InstanceBuilder builder{};
    auto inst_ret = builder.set_app_name("trongle")
                        .request_validation_layers()
                        .set_debug_callback(custom_debug_callback)
                        .require_api_version(1, 4)
                        .build();
    if (!inst_ret)
    {
        fmt::println("Failed to create instance.");
        std::abort();
    }
    vkb::Instance vkb_inst = inst_ret.value();
    return vkb_inst;
}

SDL_Window* create_window(VkInstance instance, uint32_t window_width, uint32_t window_height, SDL_WindowFlags window_flags)
{
#ifndef _WIN32
    if (dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD))
    {
        SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "x11");
    }
    else
    {
        SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "wayland");
    }
#endif

    // SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "x11");

    SDL_SetHint(SDL_HINT_APP_ID, "trongle-engine");
    SDL_Init(SDL_INIT_VIDEO);

    #ifdef NDEBUG
        fmt::println("Release mode, {}", SDL_GetCurrentVideoDriver());
    #else
        fmt::println("Debug mode, {}", SDL_GetCurrentVideoDriver());
    #endif


    SDL_Window* window = SDL_CreateWindow("trongle", window_width, window_height, window_flags);
    return window;
}

void create_surface(SDL_Window* window, VkInstance instance, VkSurfaceKHR* surface)
{
    bool surface_result = SDL_Vulkan_CreateSurface(window, instance, nullptr, surface);
    if (!surface_result)
    {
        fmt::println("Failed to create window surface.");
        std::abort();
    }
}

vkb::Result<vkb::PhysicalDevice> create_physical_device(vkb::Instance& vkb_inst, VkSurfaceKHR surface)
{
    VkPhysicalDeviceVulkan14Features features_14{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES };

    VkPhysicalDeviceVulkan13Features features_13{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    features_13.dynamicRendering = true;
    features_13.synchronization2 = true;
    // features_13.maintenance4 = true;

    VkPhysicalDeviceVulkan12Features features_12{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    features_12.bufferDeviceAddress = true;
    features_12.shaderFloat16 = true;
    features_12.descriptorBindingPartiallyBound = true;
    features_12.descriptorBindingVariableDescriptorCount = true;
    features_12.descriptorIndexing = true;
    features_12.descriptorBindingSampledImageUpdateAfterBind = true;
    features_12.descriptorBindingStorageImageUpdateAfterBind = true;
    features_12.descriptorBindingStorageBufferUpdateAfterBind = true;
    features_12.runtimeDescriptorArray = true;
    features_12.scalarBlockLayout = true;
    features_12.shaderSampledImageArrayNonUniformIndexing = true;
    features_12.shaderStorageImageArrayNonUniformIndexing = true;
    features_12.drawIndirectCount = true;
    features_12.hostQueryReset = true;
    features_12.shaderInt8 = true;
    features_12.shaderBufferInt64Atomics = true;
    features_12.samplerFilterMinmax = true;

    VkPhysicalDeviceVulkan11Features features_11{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };

    VkPhysicalDeviceFeatures features_10{};
    features_10.shaderInt16 = true;
    features_10.pipelineStatisticsQuery = true;
    features_10.shaderInt64 = true;
    features_10.depthClamp = true;

    vkb::PhysicalDeviceSelector selector{ vkb_inst };
    auto vkb_physical_device = selector.set_surface(surface)
                                   .set_minimum_version(1, 4)
                                   .set_required_features_14(features_14)
                                   .set_required_features_13(features_13)
                                   .set_required_features_12(features_12)
                                   .set_required_features_11(features_11)
                                   .set_required_features(features_10)
                                   .select();
    if (!vkb_physical_device)
    {
        fmt::println("Failed to select physical device.");
        std::abort();
    }

    return vkb_physical_device;
}

vkb::Device create_device(vkb::Result<vkb::PhysicalDevice> vkb_physical_device)
{
    // VkBootstrap specific - checking for extension support, this may be incorrect.
    // Without bootstrap, the right way to do this is query extension name -> query feature -> set feature -> device creation using extensions & features set
    bool desc_heap_supported = vkb_physical_device->enable_extension_if_present(VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME);
    bool unified_layouts_supported = vkb_physical_device->enable_extension_if_present(VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME);
    if (unified_layouts_supported)
    {
        VkPhysicalDeviceUnifiedImageLayoutsFeaturesKHR unified_layouts_features{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR };
        unified_layouts_features.unifiedImageLayouts = true;
        unified_layouts_supported = vkb_physical_device->enable_extension_features_if_present(unified_layouts_features);
    }

    bool mesh_shader_supported = vkb_physical_device->enable_extension_if_present(VK_EXT_MESH_SHADER_EXTENSION_NAME);
    if (mesh_shader_supported)
    {
        VkPhysicalDeviceMeshShaderFeaturesEXT mesh_shader_features{};
        mesh_shader_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
        mesh_shader_features.meshShader = true;
        mesh_shader_features.meshShaderQueries = true;
        mesh_shader_supported &= vkb_physical_device->enable_extension_features_if_present(mesh_shader_features);
    }

    assert(mesh_shader_supported);

    vkb::DeviceBuilder device_builder{ vkb_physical_device.value() };
    auto dev_ret = device_builder.build();
    if (!dev_ret)
    {
        fmt::println("Failed to select logical device.");
        std::abort();
    }
    vkb::Device vkb_device = dev_ret.value();
    return vkb_device;
}

void get_graphics_queue_and_family(vkb::Device vkb_device, VkQueue& graphics_queue, uint32_t& graphics_queue_family)
{
    auto graphics_queue_ret = vkb_device.get_queue(vkb::QueueType::graphics);
    if (!graphics_queue_ret)
    {
        fmt::println("Failed to select graphics queue.");
        std::abort();
    }
    graphics_queue = graphics_queue_ret.value();
    graphics_queue_family = vkb_device.get_queue_index(vkb::QueueType::graphics).value();
}
