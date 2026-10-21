#include "engine.h"
#include "common.h"
#include "swapchain.h"
#include "resources.h"
#include "asset_loader.h"
#include "pipelines.h"
#include "shared_cpu_gpu.h"
#include "camera.h"
#include "descriptors.h"
#include "rendergraph.h"
#include "sync.h"
#include "config.h"

#include <volk.h>
#include <vk_mem_alloc.h>
#include <fmt/core.h>
#include <VkBootstrap.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <SDL3/SDL_events.h>
#include <glm/ext/matrix_float4x4.hpp>
#include <glm/common.hpp>
#include <glm/exponential.hpp>
#include <glm/ext/matrix_float3x3.hpp>
#include <glm/ext/quaternion_float.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/geometric.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/matrix.hpp>
#include <glm/trigonometric.hpp>
#include <glm/ext/vector_uint2.hpp>
#include <glm/ext/vector_float2.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>

#include <vector>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <array>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <random>

#define MAX_TIMESTAMP_QUERIES 22
#define MAX_MESH_PIPELINE_QUERIES 2

const char* ASSET_NAME = "Sponza/Sponza.gltf";
// const char* ASSET_NAME = "chinesedragon.gltf";
// const char* ASSET_NAME = "de_dust2/de_dust2.gltf";
// const char* ASSET_NAME = "BistroInterior_Wine/BistroInterior_Wine.gltf";
int DEBUG_ID = 8;

namespace
{
// Taken directly from https://github.com/zeux/niagara/blob/master/src/scene.cpp
void decompose_transform(const glm::mat4& m, glm::vec3& t, glm::vec3& s, glm::vec4& rotation)
{
    t.x = m[3][0];
    t.y = m[3][1];
    t.z = m[3][2];

    float det = glm::determinant(glm::mat3(m));
    float sign = (det < 0.0f) ? -1.0f : 1.0f;

    s.x = glm::sqrt(m[0][0] * m[0][0] + m[0][1] * m[0][1] + m[0][2] * m[0][2]) * sign;
    s.y = glm::sqrt(m[1][0] * m[1][0] + m[1][1] * m[1][1] + m[1][2] * m[1][2]) * sign;
    s.z = glm::sqrt(m[2][0] * m[2][0] + m[2][1] * m[2][1] + m[2][2] * m[2][2]) * sign;

    float rsx = (s[0] == 0.f) ? 0.f : 1.f / s[0];
    float rsy = (s[1] == 0.f) ? 0.f : 1.f / s[1];
    float rsz = (s[2] == 0.f) ? 0.f : 1.f / s[2];

    // mat = rotation * scale, we want a pure rotation matrix hence normalize axes
    float r00 = m[0][0] * rsx, r10 = m[1][0] * rsy, r20 = m[2][0] * rsz;
    float r01 = m[0][1] * rsx, r11 = m[1][1] * rsy, r21 = m[2][1] * rsz;
    float r02 = m[0][2] * rsx, r12 = m[1][2] * rsy, r22 = m[2][2] * rsz;

    // "branchless" version of Mike Day's matrix to quaternion conversion, no attempt was made to understand quats :)
    int qc = r22 < 0 ? (r00 > r11 ? 0 : 1) : (r00 < -r11 ? 2 : 3);
    float qs1 = qc & 2 ? -1.f : 1.f;
    float qs2 = qc & 1 ? -1.f : 1.f;
    float qs3 = (qc - 1) & 2 ? -1.f : 1.f;

    float qt = 1.f - qs3 * r00 - qs2 * r11 - qs1 * r22;
    float qs = 0.5f / glm::sqrt(qt);

    rotation[qc ^ 0] = qs * qt;
    rotation[qc ^ 1] = qs * (r01 + qs1 * r10);
    rotation[qc ^ 2] = qs * (r20 + qs2 * r02);
    rotation[qc ^ 3] = qs * (r12 + qs3 * r21);
}

uint32_t get_group_count(uint32_t size, uint32_t threads)
{
    return (size + threads - 1) / threads;
}

// TODO: if extent is alr pow2?
uint32_t nearest_pow2(uint32_t extent)
{
    return 1 << static_cast<uint32_t>(glm::floor(glm::log2(static_cast<float>(extent))));
}

// TODO: if extent is alr pow2?
uint32_t next_pow2(uint32_t extent)
{
    return nearest_pow2(extent) << 1;
}

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

void Engine::init_vulkan()
{
    VK_CHECK(volkInitialize());

    vkb::InstanceBuilder builder{};
    auto inst_ret = builder.set_app_name("Untitled vulkan renderer")
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
    instance = vkb_inst.instance;
    debug_messenger = vkb_inst.debug_messenger;

    volkLoadInstance(instance);

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
    auto window_flags = static_cast<SDL_WindowFlags>(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    window = SDL_CreateWindow("trongle", 1280u, 720u, window_flags);
    bool surface_result = SDL_Vulkan_CreateSurface(window, vkb_inst, nullptr, &surface);
    if (!surface_result)
    {
        fmt::println("Failed to create window surface.");
        std::abort();
    }
    SDL_SetWindowRelativeMouseMode(window, relative_mouse_mode);

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
    auto phys_ret = selector.set_surface(surface)
                        .set_minimum_version(1, 4)
                        .set_required_features_14(features_14)
                        .set_required_features_13(features_13)
                        .set_required_features_12(features_12)
                        .set_required_features_11(features_11)
                        .set_required_features(features_10)
                        .select();
    if (!phys_ret)
    {
        fmt::println("Failed to select physical device.");
        std::abort();
    }
    physical_device = phys_ret->physical_device;

    // VkBootstrap specific - checking for extension support, this may be incorrect.
    // Without bootstrap, the right way to do this is query extension name -> query feature -> set feature -> device creation using extensions & features set
    bool desc_heap_supported = phys_ret->enable_extension_if_present(VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME);
    bool unified_layouts_supported = phys_ret->enable_extension_if_present(VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME);
    if (unified_layouts_supported)
    {
        VkPhysicalDeviceUnifiedImageLayoutsFeaturesKHR unified_layouts_features{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR };
        unified_layouts_features.unifiedImageLayouts = true;
        unified_layouts_supported = phys_ret->enable_extension_features_if_present(unified_layouts_features);
    }

    bool mesh_shader_supported = phys_ret->enable_extension_if_present(VK_EXT_MESH_SHADER_EXTENSION_NAME);
    if (mesh_shader_supported)
    {
        VkPhysicalDeviceMeshShaderFeaturesEXT mesh_shader_features{};
        mesh_shader_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
        mesh_shader_features.meshShader = true;
        mesh_shader_features.meshShaderQueries = true;
        mesh_shader_supported &= phys_ret->enable_extension_features_if_present(mesh_shader_features);
    }

    assert(mesh_shader_supported);

    vkb::DeviceBuilder device_builder{ phys_ret.value() };
    auto dev_ret = device_builder.build();
    if (!dev_ret)
    {
        fmt::println("Failed to select logical device.");
        std::abort();
    }
    vkb::Device vkb_device = dev_ret.value();
    device = vkb_device.device;

    volkLoadDevice(vkb_device);

    auto graphics_queue_ret = vkb_device.get_queue(vkb::QueueType::graphics);
    if (!graphics_queue_ret)
    {
        fmt::println("Failed to select graphics queue.");
        std::abort();
    }
    graphics_queue = graphics_queue_ret.value();
    graphics_queue_family = vkb_device.get_queue_index(vkb::QueueType::graphics).value();

    auto window_width = 1280u;
    auto window_height = 720u;
    swapchain = create_swapchain(physical_device, device, surface, window_width, window_height);

    VmaAllocatorCreateInfo allocator_info{};
    allocator_info.physicalDevice = physical_device;
    allocator_info.device = device;
    allocator_info.instance = instance;
    allocator_info.vulkanApiVersion = VK_API_VERSION_1_4;
    allocator_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    VmaVulkanFunctions vulkan_functions{};
    VK_CHECK(vmaImportVulkanFunctionsFromVolk(&allocator_info, &vulkan_functions));
    allocator_info.pVulkanFunctions = &vulkan_functions;
    VK_CHECK(vmaCreateAllocator(&allocator_info, &allocator));

#ifdef NDEBUG
    fmt::println("Release mode, {}", SDL_GetCurrentVideoDriver());
#else
    fmt::println("Debug mode, {}", SDL_GetCurrentVideoDriver());
#endif
}

void Engine::cleanup()
{
    vkDeviceWaitIdle(device);

    destroy_imgui();

    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);

    vkDestroyDescriptorPool(device, desc_pool, nullptr);
    vkDestroyDescriptorSetLayout(device, buffer_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, image_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, sampler_layout, nullptr);

    for (auto& frame : frames)
    {
        vkDestroyCommandPool(device, frame.command_pool, nullptr);
        vkDestroyFence(device, frame.fence, nullptr);
        vkDestroySemaphore(device, frame.image_acquired_semaphore, nullptr);
        vkDestroyQueryPool(device, frame.query_pool_timestamp, nullptr);
        vkDestroyQueryPool(device, frame.query_pool_mesh_pipeline, nullptr);
    }

    vkDestroyCommandPool(device, imm_pool, nullptr);
    vkDestroyFence(device, imm_fence, nullptr);

    vkDestroyDebugUtilsMessengerEXT(instance, debug_messenger, nullptr);

    destroy_swapchain(device, swapchain);

    vmaDestroyAllocator(allocator);

    vkDestroyDevice(device, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
}

void Engine::init_commands()
{
    VkCommandPoolCreateInfo pool_info{ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pool_info.queueFamilyIndex = graphics_queue_family;

    for (auto& frame : frames)
    {
        VK_CHECK(vkCreateCommandPool(device, &pool_info, nullptr, &frame.command_pool));

        VkCommandBufferAllocateInfo allocate_info{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        allocate_info.commandPool = frame.command_pool;
        allocate_info.commandBufferCount = 1;
        allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

        VK_CHECK(vkAllocateCommandBuffers(device, &allocate_info, &frame.command_buffer));
    }

    // Create immediate command pool & buffer
    VK_CHECK(vkCreateCommandPool(device, &pool_info, nullptr, &imm_pool));

    VkCommandBufferAllocateInfo allocate_info{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocate_info.commandPool = imm_pool;
    allocate_info.commandBufferCount = 1;
    allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

    VK_CHECK(vkAllocateCommandBuffers(device, &allocate_info, &imm_buf));
}

void Engine::init_sync()
{
    VkFenceCreateInfo fence_info{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkSemaphoreCreateInfo semaphore_info{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };

    for (auto& frame : frames)
    {
        VK_CHECK(vkCreateFence(device, &fence_info, nullptr, &frame.fence));
        VK_CHECK(vkCreateSemaphore(device, &semaphore_info, nullptr, &frame.image_acquired_semaphore));
    }

    // Create immediate fence
    VK_CHECK(vkCreateFence(device, &fence_info, nullptr, &imm_fence));
}

void Engine::init_query_pool()
{
    VkQueryPoolCreateInfo query_pool_info{ .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };

    for (auto& frame : frames)
    {
        query_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        query_pool_info.queryCount = MAX_TIMESTAMP_QUERIES;
        VK_CHECK(vkCreateQueryPool(device, &query_pool_info, nullptr, &frame.query_pool_timestamp));

        query_pool_info.queryType = VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT;
        query_pool_info.queryCount = MAX_MESH_PIPELINE_QUERIES;
        query_pool_info.pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT;
        VK_CHECK(vkCreateQueryPool(device, &query_pool_info, nullptr, &frame.query_pool_mesh_pipeline));

        vkResetQueryPool(device, frame.query_pool_timestamp, 0, MAX_TIMESTAMP_QUERIES);
        vkResetQueryPool(device, frame.query_pool_mesh_pipeline, 0, MAX_MESH_PIPELINE_QUERIES);
    }
}

void Engine::run()
{
    // Init rendergraph
    graph.device = device;
    graph.allocator = allocator;
    graph.image_manager = &image_manager;

    VkPhysicalDeviceMeshShaderPropertiesEXT mesh_shader_properties{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT };
    VkPhysicalDeviceProperties2 properties2{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
    properties2.pNext = &mesh_shader_properties;

    vkGetPhysicalDeviceProperties2(physical_device, &properties2);
    assert(properties2.properties.limits.timestampComputeAndGraphics);
    auto max_push_constant_size = properties2.properties.limits.maxPushConstantsSize;
    auto timestamp_period = properties2.properties.limits.timestampPeriod;
    assert(mesh_shader_properties.maxMeshWorkGroupCount[0] == MAX_MESHLETS_DRAW); // Only checking workgroup x

    // Init camera
    Camera camera{};
    camera.position = glm::vec3(0, 0, 5);
    camera.far = 1000.0f;
    camera.near = 0.01f;

    frame_context.proj = camera.set_perspective_matrix(glm::radians(camera.fov), static_cast<float>(swapchain.extent.width) / swapchain.extent.height, camera.near);
    frame_context.debug_proj = frame_context.proj;

    AssetLoader asset_loader{ .device = device, .allocator = allocator };
    bool loaded = asset_loader.load_gltf(graphics_queue, imm_fence, imm_pool, imm_buf, ASSET_NAME, static_cast<uint32_t>(image_manager.infos.size()));
    if (!loaded)
    {
        assert(0 && "load_gltf failed");
    }

#if 1
    for (const auto& node : asset_loader.parent_nodes)
    {
        register_object(node, glm::mat4(1.0), asset_loader.children_nodes, asset_loader.meshes);
    }
#else
    for (const auto& node : asset_loader.parent_nodes)
    {
        register_object(node, glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, 2)), asset_loader.children_nodes, asset_loader.meshes);
    }

    std::mt19937 mt(42);
    auto draw_radius = 400.0f;
    auto draw_count = 500'000;

    std::vector<glm::mat4> transforms(draw_count);
    for (size_t i = 0; i < draw_count; i++)
    {
        const float x = static_cast<float>(mt()) / static_cast<float>(mt.max()) * draw_radius - draw_radius * 0.5f;
        const float y = static_cast<float>(mt()) / static_cast<float>(mt.max()) * draw_radius - draw_radius * 0.5f;
        const float z = static_cast<float>(mt()) / static_cast<float>(mt.max()) * -draw_radius;

        glm::mat4 t = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));
        glm::vec3 axis = glm::normalize(
            glm::vec3(
                static_cast<float>(mt()) / static_cast<float>(mt.max()),
                static_cast<float>(mt()) / static_cast<float>(mt.max()),
                static_cast<float>(mt()) / static_cast<float>(mt.max())
            )
        );
        glm::mat4 r = glm::rotate(glm::mat4(1.0f), glm::radians(static_cast<float>(mt()) / static_cast<float>(mt.max()) * 360.0f), axis);
        glm::mat4 s = glm::scale(glm::mat4(1.0f), glm::vec3(static_cast<float>(mt()) / static_cast<float>(mt.max())) + 1.0f);
        transforms[i] = t * r * s;
    }

    for (auto& transform : transforms)
    {
        for (const auto& node : asset_loader.parent_nodes)
        {
            register_object(node, transform, asset_loader.children_nodes, asset_loader.meshes);
        }
    }
#endif

    // Set meshlet bits offset, needed for 2 pass culling
    assert(!renderables.empty());
    auto meshlet_bits = 0;
    for (auto& renderable : renderables)
    {
        meshlet_bits += meshes[renderable.mesh_id].mesh_lods[0].meshlet_count;
    }
    auto meshlet_offset = get_group_count(renderables.size(), 32);

    // Update descriptors
    for (const auto& image : asset_loader.images)
    {
        image_manager.register_sampled_image(device, image.view, VK_IMAGE_LAYOUT_GENERAL);
    }

    // Create sampler
    std::vector<VkWriteDescriptorSet> writes{};
    std::vector<VkDescriptorImageInfo> desc_info{};
    VkSampler linear_samp = create_sampler(device, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT);
    VkSampler depth_reduce_samp = create_depth_reduce_sampler(device);
    VkSampler nearest_samp = create_sampler(device, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    // TODO: clamp to border? use compareEnable
    // VkSampler shadow_samp = create_sampler(device, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, true);
    VkSampler shadow_samp = create_sampler(device, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    desc_info.emplace_back(VkDescriptorImageInfo{ .sampler = linear_samp });
    desc_info.emplace_back(VkDescriptorImageInfo{ .sampler = depth_reduce_samp });
    desc_info.emplace_back(VkDescriptorImageInfo{ .sampler = nearest_samp });
    desc_info.emplace_back(VkDescriptorImageInfo{ .sampler = shadow_samp });

    for (auto i = 0; i < desc_info.size(); i++)
    {
        writes.emplace_back(write_sampler_descriptor(sampler_set, i, &desc_info[i]));
    }

    vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);

    writes.clear();
    desc_info.clear();

    // Load GPU data
    Buffer vertex_buffer = create_buffer_with_data(device, graphics_queue, imm_fence, imm_pool, imm_buf, allocator, asset_loader.m_vertices.data(), asset_loader.m_vertices.size() * sizeof(Vertex));
    Buffer object_buffer = create_buffer_with_data(device, graphics_queue, imm_fence, imm_pool, imm_buf, allocator, renderables.data(), renderables.size() * sizeof(ObjectData));
    Buffer material_buffer = create_buffer_with_data(device, graphics_queue, imm_fence, imm_pool, imm_buf, allocator, asset_loader.materials.data(), asset_loader.materials.size() * sizeof(MaterialData));
    Buffer mesh_buffer = create_buffer_with_data(device, graphics_queue, imm_fence, imm_pool, imm_buf, allocator, meshes.data(), meshes.size() * sizeof(MeshData));
    Buffer meshlet_buffer = create_buffer_with_data(device, graphics_queue, imm_fence, imm_pool, imm_buf, allocator, asset_loader.m_meshlets.data(), asset_loader.m_meshlets.size() * sizeof(Meshlet));
    Buffer meshlet_indices_buffer = create_buffer_with_data(device, graphics_queue, imm_fence, imm_pool, imm_buf, allocator, asset_loader.meshlet_indices.data(), asset_loader.meshlet_indices.size() * sizeof(uint32_t));
    Buffer spd_counter_buffer = create_buffer(device, allocator, sizeof(uint32_t), 0, VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT_KHR);
    Buffer late_pass_candidates_buffer = create_buffer(device, allocator, (meshlet_offset + get_group_count(meshlet_bits, 32)) * sizeof(uint32_t), 0, VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT_KHR);

    std::array<Image, SUN_CASCADE_COUNT> shadowmap_images{};
    std::array<uint32_t, SUN_CASCADE_COUNT> shadowmap_descriptor_handles{};
    for (auto i = 0; i < SUN_CASCADE_COUNT; i++)
    {
        shadowmap_images[i] = create_image(
            device,
            allocator,
            VkExtent3D{ SHADOWMAP_EXTENT, SHADOWMAP_EXTENT, 1 },
            VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_USAGE_2_DEPTH_STENCIL_ATTACHMENT_BIT_KHR | VK_IMAGE_USAGE_2_SAMPLED_BIT_KHR | VK_IMAGE_USAGE_2_STORAGE_BIT_KHR,
            VK_IMAGE_ASPECT_DEPTH_BIT
        );

        shadowmap_descriptor_handles[i] = image_manager.register_image(device, shadowmap_images[i].view, VK_IMAGE_LAYOUT_GENERAL);
    }

    VkExtent3D depth_pyramid_extent{};
    depth_pyramid_extent.width = nearest_pow2(swapchain.extent.width);
    depth_pyramid_extent.height = nearest_pow2(swapchain.extent.height);
    depth_pyramid_extent.depth = 1;
    Image depth_pyramid_image = create_image(device, allocator, depth_pyramid_extent, VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, true);

    {
        std::vector<VkImage> color_images = {depth_pyramid_image.image};
        std::vector<VkImage> depth_images{};
        for (const auto& image : shadowmap_images)
        {
            depth_images.push_back(image.image);
        }

        immediate_submit(device, graphics_queue, imm_fence, imm_pool, imm_buf, [&](VkCommandBuffer cmd)
                         {
                             vkCmdFillBuffer(cmd, spd_counter_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
                             vkCmdFillBuffer(cmd, late_pass_candidates_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
                             transition_images(cmd, color_images, depth_images);
                         });
    }

    const auto depth_pyramid_descriptor_handle = image_manager.register_image(device, depth_pyramid_image.view, VK_IMAGE_LAYOUT_GENERAL);

    // Returns depth pyramid mip levels for next PoT
    auto get_depth_pyramid_image_count = [&](const Image& image) -> uint32_t
    {
        const auto& depth_pyramid_extent = image.extent;
        return static_cast<uint32_t>(glm::floor(glm::log2(static_cast<float>(glm::max(depth_pyramid_extent.width, depth_pyramid_extent.height))))) + 1;
    };

    // Reserves 11 descriptor handles to support up to 2048x2048
    auto create_depth_pyramid_image_views = [&](const Image& image, std::vector<VkImageView>& views)
    {
        if (!views.empty())
        {
            for (const auto& view : views)
            {
                vkDestroyImageView(device, view, nullptr);
            }

            views.clear();
        }

        VkImageViewCreateInfo img_view_info{};
        img_view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        img_view_info.image = image.image;
        img_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        img_view_info.format = VK_FORMAT_R32_SFLOAT;
        VkImageSubresourceRange subresource_range{};
        subresource_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresource_range.levelCount = 1;
        subresource_range.baseArrayLayer = 0;
        subresource_range.layerCount = VK_REMAINING_ARRAY_LAYERS;

        auto count = get_depth_pyramid_image_count(image);

        views.resize(count);
        for (auto i = 0; i < count; i++)
        {
            subresource_range.baseMipLevel = i;
            img_view_info.subresourceRange = subresource_range;

            VK_CHECK(vkCreateImageView(device, &img_view_info, nullptr, &views[i]));
            image_manager.register_image(device, views[i], VK_IMAGE_LAYOUT_GENERAL);
        }
        for (auto i = count; i < 11; i++)
        {
            image_manager.register_image(device, views[count - 1], VK_IMAGE_LAYOUT_GENERAL);
        }
    };

    auto update_depth_pyramid = [&](const Image& image, std::vector<VkImageView>& views, uint32_t depth_pyramid_first_mip)
    {
        assert(!views.empty());

        for (const auto& view : views)
        {
            vkDestroyImageView(device, view, nullptr);
        }

        views.clear();

        VkImageViewCreateInfo img_view_info{};
        img_view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        img_view_info.image = image.image;
        img_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        img_view_info.format = VK_FORMAT_R32_SFLOAT;
        VkImageSubresourceRange subresource_range{};
        subresource_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresource_range.levelCount = 1;
        subresource_range.baseArrayLayer = 0;
        subresource_range.layerCount = VK_REMAINING_ARRAY_LAYERS;

        auto count = get_depth_pyramid_image_count(image);

        views.resize(count);
        for (auto i = 0; i < count; i++)
        {
            subresource_range.baseMipLevel = i;
            img_view_info.subresourceRange = subresource_range;

            VK_CHECK(vkCreateImageView(device, &img_view_info, nullptr, &views[i]));
            image_manager.update(device, depth_pyramid_first_mip + i, views[i], VK_IMAGE_LAYOUT_GENERAL);
        }
    };

    const auto depth_pyramid_first_mip = static_cast<uint32_t>(image_manager.infos.size());
    std::vector<VkImageView> depth_pyramid_image_views{};
    create_depth_pyramid_image_views(depth_pyramid_image, depth_pyramid_image_views);

    // Create pipeline layout
    VkPushConstantRange pc_range{ .stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = max_push_constant_size };
    std::array<VkDescriptorSetLayout, 3> desc_set_layouts{ buffer_layout, image_layout, sampler_layout };
    VkPipelineLayoutCreateInfo pipeline_layout_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipeline_layout_info.pPushConstantRanges = &pc_range;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pSetLayouts = desc_set_layouts.data();
    pipeline_layout_info.setLayoutCount = desc_set_layouts.size();
    VK_CHECK(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout));

    VkPipeline cull_mesh_pipeline{};
    VkPipeline cull_meshlet_pipeline{};
    VkPipeline compact_dispatch_pipeline{};
    VkPipeline hiz_pipeline{};
    VkPipeline vbuffer_pipeline{};
    VkPipeline lighting_pipeline{};
    VkPipeline shadow_matrices_pipeline{};
    VkPipeline cull_shadow_pipeline{};
    VkPipeline cull_shadow_meshlet_pipeline{};
    VkPipeline depth_pipeline{};

    std::vector<VkPipeline> pipelines{};

    auto build_pipelines = [&]()
    {
        auto cull_program = load_shader_program("cull_mesh.slang", device);
        auto cull_meshlets_program = load_shader_program("cull_meshlet.slang", device);
        auto compact_dispatch_program = load_shader_program("compact_dispatch.slang", device);
        auto hiz_program = load_shader_program("hiz_spd.slang", device);
        auto vbuffer_program = load_shader_program("vbuffer.slang", device);
        auto lighting_program = load_shader_program("resolve_vbuffer.slang", device);
        auto shadow_matrices_program = load_shader_program("shadow_matrices.slang", device);
        auto cull_shadow_program = load_shader_program("cull_shadow.slang", device);
        auto cull_shadow_meshlet_program = load_shader_program("cull_shadow_meshlet.slang", device);
        auto depth_program = load_shader_program("depth.slang", device);

        pipelines.clear();

        auto replace_pipeline = [&](VkPipeline& old_pipeline, VkPipeline new_pipeline)
        {
            assert(new_pipeline);

            if (old_pipeline)
            {
                vkDestroyPipeline(device, old_pipeline, nullptr);
            }

            old_pipeline = new_pipeline;
            pipelines.push_back(old_pipeline);
        };

        replace_pipeline(
            vbuffer_pipeline,
            create_graphics_pipeline(
                device,
                &vbuffer_program,
                { VK_SHADER_STAGE_MESH_BIT_EXT, VK_SHADER_STAGE_FRAGMENT_BIT },
                nullptr,
                &pipeline_layout,
                { VK_FORMAT_R32G32_UINT }
            )
        );

        replace_pipeline(
            depth_pipeline,
            create_graphics_pipeline(
                device,
                &depth_program,
                { VK_SHADER_STAGE_MESH_BIT_EXT },
                nullptr,
                &pipeline_layout,
                {},
                {},
                GraphicsPipelineConfig{ .cull_mode = VK_CULL_MODE_FRONT_BIT, .depth_clamp = VK_TRUE }
            )
        );

        replace_pipeline(cull_mesh_pipeline, create_compute_pipeline(device, &cull_program, nullptr, &pipeline_layout));
        replace_pipeline(cull_meshlet_pipeline, create_compute_pipeline(device, &cull_meshlets_program, nullptr, &pipeline_layout));
        replace_pipeline(compact_dispatch_pipeline, create_compute_pipeline(device, &compact_dispatch_program, nullptr, &pipeline_layout));
        replace_pipeline(hiz_pipeline, create_compute_pipeline(device, &hiz_program, nullptr, &pipeline_layout));
        replace_pipeline(lighting_pipeline, create_compute_pipeline(device, &lighting_program, nullptr, &pipeline_layout));
        replace_pipeline(shadow_matrices_pipeline, create_compute_pipeline(device, &shadow_matrices_program, nullptr, &pipeline_layout));
        replace_pipeline(cull_shadow_pipeline, create_compute_pipeline(device, &cull_shadow_program, nullptr, &pipeline_layout));
        replace_pipeline(cull_shadow_meshlet_pipeline, create_compute_pipeline(device, &cull_shadow_meshlet_program, nullptr, &pipeline_layout));
    };

    build_pipelines();

    std::array<uint64_t, MAX_TIMESTAMP_QUERIES> timestamp_results{};
    std::array<uint64_t, MAX_MESH_PIPELINE_QUERIES> mesh_pipeline_results{};

    bool skip_occlusion = true;
    bool b_shadows = true;
    float azimuth = 0.0f;
    float elevation = 80.0f;

    auto last_frame = SDL_GetTicks();
    bool quit{ false };

    while (!quit)
    {
        auto start = SDL_GetTicks();
        auto delta_time = (start - last_frame) / 1000.0f;
        last_frame = start;

        // Poll events
        SDL_Event event{};
        while (SDL_PollEvent(&event))
        {
            switch (event.type)
            {
            case SDL_EVENT_QUIT:
                quit = true;
                break;
            case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
                if (event.window.data1 > 0 && event.window.data2 > 0)
                {
                    fmt::println("swapchain resize - SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED {} {}", event.window.data1, event.window.data2);

                    for (const auto& image : swapchain.images)
                    {
                        invalidate_imported_image(graph, image);
                    }
                    update_swapchain(swapchain, window, physical_device, device, surface);

                    destroy_image(device, allocator, depth_pyramid_image);
                    depth_pyramid_extent.width = nearest_pow2(swapchain.extent.width);
                    depth_pyramid_extent.height = nearest_pow2(swapchain.extent.height);
                    depth_pyramid_image = create_image(device, allocator, depth_pyramid_extent, VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, true);
                    image_manager.update(device, depth_pyramid_descriptor_handle, depth_pyramid_image.view, VK_IMAGE_LAYOUT_GENERAL);
                    update_depth_pyramid(depth_pyramid_image, depth_pyramid_image_views, depth_pyramid_first_mip);
                    skip_occlusion = true;
                    immediate_submit(device, graphics_queue, imm_fence, imm_pool, imm_buf, [&](VkCommandBuffer cmd)
                                     {
                                         stage_barrier(cmd, depth_pyramid_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, 0, 0, 0);
                                     });

                    frame_context.proj = camera.set_perspective_matrix(glm::radians(camera.fov), static_cast<float>(swapchain.extent.width) / swapchain.extent.height, camera.near);
                    frame_context.debug_proj = frame_context.proj;
                }
                break;
            case SDL_EVENT_KEY_DOWN:
                if (event.key.key == SDLK_SPACE && event.key.repeat == 0)
                {
                    relative_mouse_mode = !relative_mouse_mode;
                    SDL_SetWindowRelativeMouseMode(window, relative_mouse_mode);
                }
                if (event.key.key == SDLK_Y && event.key.repeat == 0)
                {
                    int recompile = std::system("ninja Shaders");

                    if (recompile == 0)
                    {
                        vkDeviceWaitIdle(device);
                        build_pipelines();
                    }
                }
                if (event.key.key == SDLK_F && event.key.repeat == 0)
                {
                    freeze_matrices = !freeze_matrices;
                }
                break;
            default:
                break;
            }

            camera.process_sdl_event(event, SDL_GetWindowRelativeMouseMode(window));
            ImGui_ImplSDL3_ProcessEvent(&event);
        }

        if (swapchain_dirty)
        {
            for (const auto& image : swapchain.images)
            {
                invalidate_imported_image(graph, image);
            }
            update_swapchain(swapchain, window, physical_device, device, surface);
            swapchain_dirty = false;

            destroy_image(device, allocator, depth_pyramid_image);
            depth_pyramid_extent.width = nearest_pow2(swapchain.extent.width);
            depth_pyramid_extent.height = nearest_pow2(swapchain.extent.height);
            depth_pyramid_image = create_image(device, allocator, depth_pyramid_extent, VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, true);
            image_manager.update(device, depth_pyramid_descriptor_handle, depth_pyramid_image.view, VK_IMAGE_LAYOUT_GENERAL);
            update_depth_pyramid(depth_pyramid_image, depth_pyramid_image_views, depth_pyramid_first_mip);
            skip_occlusion = true;
            immediate_submit(device, graphics_queue, imm_fence, imm_pool, imm_buf, [&](VkCommandBuffer cmd)
                             {
                                 stage_barrier(cmd, depth_pyramid_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, 0, 0, 0);
                             });

            frame_context.proj = camera.set_perspective_matrix(glm::radians(camera.fov), static_cast<float>(swapchain.extent.width) / swapchain.extent.height, camera.near);
            frame_context.debug_proj = frame_context.proj;
        }

        // Update camera
        camera.update(delta_time);
        if (!freeze_matrices)
        {
            frame_context.prev_view = frame_context.view;
            frame_context.prev_view_proj = frame_context.view_proj;
            frame_context.view = camera.get_view_matrix();
            frame_context.view_proj = frame_context.proj * frame_context.view;
        }
        else
        {
            frame_context.debug_view = camera.get_view_matrix();
            frame_context.debug_view_proj = frame_context.debug_proj * frame_context.debug_view;
        }

        float az = glm::radians(azimuth);
        float el = glm::radians(elevation);
        float3 sunlight_dir{};
        float3 sunlight_color{};
        sunlight_dir.x = glm::cos(el) * glm::sin(az);
        sunlight_dir.y = glm::sin(el);
        sunlight_dir.z = -glm::cos(el) * glm::cos(az);
        sunlight_dir = glm::normalize(glm::vec3(sunlight_dir));

        // Wait on fence
        auto frame = get_current_frame();
        VK_CHECK(vkWaitForFences(device, 1, &frame.fence, true, WAIT_TIME));

        // Acquire next image
        uint32_t swapchain_image_idx{};
        VkResult acquire_result = vkAcquireNextImageKHR(device, swapchain.swapchain, WAIT_TIME, frame.image_acquired_semaphore, nullptr, &swapchain_image_idx);
        if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            fmt::println("Acquire out of date / suboptimal");
            swapchain_dirty = true;
            continue;
        }
        VK_CHECK_SWAPCHAIN(acquire_result);
        VK_CHECK(vkResetFences(device, 1, &frame.fence));

        // Get query pool results
        auto timestamp_results_size = timestamp_results.size();
        auto mesh_pipeline_results_size = mesh_pipeline_results.size();

        if (frame_number >= FRAMES_IN_FLIGHT)
        {
            VK_CHECK(vkGetQueryPoolResults(
                device,
                frame.query_pool_timestamp,
                0,
                timestamp_results_size,
                timestamp_results_size * sizeof(uint64_t),
                timestamp_results.data(),
                sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT
            ));

            VK_CHECK(vkGetQueryPoolResults(
                device,
                frame.query_pool_mesh_pipeline,
                0,
                mesh_pipeline_results_size,
                mesh_pipeline_results_size * sizeof(uint64_t),
                mesh_pipeline_results.data(),
                sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT
            ));
        }

        double new_gpu_time = static_cast<double>(timestamp_results[1] - timestamp_results[0]) * timestamp_period * 1e-6;
        stats.gpu_time = new_gpu_time + 0.95 * (stats.gpu_time - new_gpu_time);
        auto triangles_early = mesh_pipeline_results[0];
        auto triangles_late = mesh_pipeline_results[1];

        vkResetQueryPool(device, frame.query_pool_timestamp, 0, MAX_TIMESTAMP_QUERIES);
        vkResetQueryPool(device, frame.query_pool_mesh_pipeline, 0, MAX_MESH_PIPELINE_QUERIES);

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Stats");
        ImGui::Text("GPU time:           %.3f ms", stats.gpu_time);
        ImGui::Text("Cull mesh early:    %.3f ms", static_cast<double>(timestamp_results[3] - timestamp_results[2]) * timestamp_period * 1e-6);
        ImGui::Text("Cull meshlet early: %.3f ms", static_cast<double>(timestamp_results[5] - timestamp_results[4]) * timestamp_period * 1e-6);
        ImGui::Text("Vbuffer early:      %.3f ms", static_cast<double>(timestamp_results[7] - timestamp_results[6]) * timestamp_period * 1e-6);
        ImGui::Text("Hiz:                %.3f ms", static_cast<double>(timestamp_results[9] - timestamp_results[8]) * timestamp_period * 1e-6);
        ImGui::Text("Cull mesh late:     %.3f ms", static_cast<double>(timestamp_results[11] - timestamp_results[10]) * timestamp_period * 1e-6);
        ImGui::Text("Cull meshlet late:  %.3f ms", static_cast<double>(timestamp_results[13] - timestamp_results[12]) * timestamp_period * 1e-6);
        ImGui::Text("Vbuffer late:       %.3f ms", static_cast<double>(timestamp_results[15] - timestamp_results[14]) * timestamp_period * 1e-6);
        ImGui::Text("Hiz temporal:       %.3f ms", static_cast<double>(timestamp_results[17] - timestamp_results[16]) * timestamp_period * 1e-6);
        ImGui::Text("Zero buffers early: %.3f ms", static_cast<double>(timestamp_results[19] - timestamp_results[18]) * timestamp_period * 1e-6);
        ImGui::Text("Zero buffers late:  %.3f ms", static_cast<double>(timestamp_results[21] - timestamp_results[20]) * timestamp_period * 1e-6);

        ImGui::Text("Triangles early:   %u", static_cast<unsigned int>(triangles_early));
        ImGui::Text("Triangles late:    %u", static_cast<unsigned int>(triangles_late));
        // ImGui::Text("Triangles:   %.1fM", static_cast<double>(triangles) * 1e-6);
        ImGui::Checkbox("Freeze camera", &freeze_matrices);
        ImGui::Checkbox("Shadows", &b_shadows);
        ImGui::SliderInt("Debug", &DEBUG_ID, 0, MAX_DEBUG_ID);
        ImGui::SliderFloat("Azimuth", &azimuth, 0.0, 360.0);
        ImGui::SliderFloat("Elevation", &elevation, 0.0, 90.0);

        ImGui::End();
        ImGui::Render();

        auto swapchain_image = graph.import_swapchain(swapchain.images[swapchain_image_idx], swapchain.image_views[swapchain_image_idx]);

        auto depth_image = graph.create_task_image(
            ImageResourceDesc{
                VkExtent3D{ swapchain.extent.width, swapchain.extent.height, 1 },
                VK_FORMAT_D32_SFLOAT,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                VK_IMAGE_ASPECT_DEPTH_BIT }
        );
        auto draw_image = graph.create_task_image(
            ImageResourceDesc{
                VkExtent3D{ swapchain.extent.width, swapchain.extent.height, 1 },
                VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT }
        );
        auto vbuffer = graph.create_task_image(
            ImageResourceDesc{
                VkExtent3D{ swapchain.extent.width, swapchain.extent.height, 1 },
                VK_FORMAT_R32G32_UINT,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT }
        );

        std::array<uint32_t, SUN_CASCADE_COUNT> shadowmaps{};
        for (uint32_t i = 0; i < SUN_CASCADE_COUNT; i++)
        {
            shadowmaps[i] = graph.import_image(shadowmap_images[i], shadowmap_descriptor_handles[i]);
        }

        auto dispatch_buffer = graph.create_task_buffer(BufferResourceDesc{ .alloc_size = sizeof(Dispatch) * 2, .usage = VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT });
        auto prefix_sum_buffer = graph.create_task_buffer(BufferResourceDesc{ .alloc_size = sizeof(PrefixSums), .usage = VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT });
        auto meshlet_draw_buffer = graph.create_task_buffer(BufferResourceDesc{ .alloc_size = sizeof(uint32_t) * MAX_MESHLETS_DRAW, .usage = VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT });
        auto global_buffer = graph.create_task_buffer(BufferResourceDesc{ .alloc_size = sizeof(GlobalData) * FRAMES_IN_FLIGHT, .usage = VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT });

        auto depth_pyramid = graph.import_image(depth_pyramid_image, depth_pyramid_descriptor_handle);
        auto spd_counter = graph.import_buffer(spd_counter_buffer, -1);
        auto late_pass_candidates = graph.import_buffer(late_pass_candidates_buffer, -1);

        // Record command buffer
        VkCommandBuffer cmd = frame.command_buffer;
        VK_CHECK(vkResetCommandPool(device, frame.command_pool, 0));

        VkCommandBufferBeginInfo cmd_begin_info{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        cmd_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_begin_info));
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, frame.query_pool_timestamp, 0);

        std::array<VkDescriptorSet, 3> sets{ buffer_set, image_set, sampler_set };

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, sets.size(), sets.data(), 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, sets.size(), sets.data(), 0, nullptr);

        // TODO: dispatch may not need to be zeroed since its already done in compact dispatch
        graph.add_pass(
            Pass("Zero buffers early")
                .write_buffer(dispatch_buffer, AccessType::TransferWrite)
                .write_buffer(prefix_sum_buffer, AccessType::TransferWrite),
            [&]()
            {
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, frame.query_pool_timestamp, 18);
                vkCmdFillBuffer(cmd, get_buffer(graph, dispatch_buffer), 0, VK_WHOLE_SIZE, 0);
                // TODO: refactor
                vkCmdFillBuffer(cmd, get_buffer(graph, prefix_sum_buffer), 0, VK_WHOLE_SIZE, 0);
                // vkCmdFillBuffer(cmd, get_buffer(graph, prefix_sum_buffer), 0, sizeof(uint64_t), 0);
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, frame.query_pool_timestamp, 19);
            }
        );

        graph.add_pass(
            Pass("Compute cull mesh early")
                .read_image(depth_pyramid, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer)
                .write_buffer(prefix_sum_buffer, AccessType::ComputeShaderWrite)
                .write_buffer(late_pass_candidates, AccessType::ComputeShaderWrite),
            [&]()
            {
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, frame.query_pool_timestamp, 2);

                struct PushData
                {
                    glm::mat4 view;
                    glm::mat4 prev_view;
                    VkDeviceAddress mesh_buffer;
                    VkDeviceAddress object_buffer;
                    VkDeviceAddress prefix_sum_buffer;
                    VkDeviceAddress late_pass_candidates_buffer;
                    glm::vec4 planes;
                    glm::vec2 depth_pyramid_extent;
                    glm::vec2 depth_buffer_extent;
                    float p00;
                    float p11;
                    float near;
                    float far;
                    uint32_t count;
                    float lod_distance_factor;
                    uint32_t late;
                    float depth_pyramid_lod;
                    uint32_t depth_pyramid_id;
                    uint32_t skip_occlusion;
                };

                auto proj_t = glm::transpose(frame_context.proj);
                auto m0 = proj_t[0];
                auto m1 = proj_t[1];
                auto m3 = proj_t[3];
                auto left_plane = glm::normalize(glm::vec3(m3 + m0));
                auto bottom_plane = glm::normalize(glm::vec3(m3 + m1));

                PushData data{};
                data.view = frame_context.view;
                data.prev_view = frame_context.prev_view;
                data.mesh_buffer = mesh_buffer.address;
                data.object_buffer = object_buffer.address;
                data.prefix_sum_buffer = get_buffer_address(graph, prefix_sum_buffer);
                data.late_pass_candidates_buffer = late_pass_candidates_buffer.address;
                data.planes = glm::vec4(left_plane.x, left_plane.z, bottom_plane.y, bottom_plane.z);
                data.depth_pyramid_extent = glm::vec2(depth_pyramid_image.extent.width, depth_pyramid_image.extent.height);
                data.depth_buffer_extent = glm::vec2(swapchain.extent.width, swapchain.extent.height);
                data.p00 = frame_context.proj[0][0];
                data.p11 = frame_context.proj[1][1];
                data.near = camera.near;
                data.far = camera.far;
                data.count = renderables.size();
                data.lod_distance_factor = 2.0f / (data.p11 * swapchain.extent.height);
                data.late = 0;
                data.depth_pyramid_lod = static_cast<float>(glm::floor(glm::log2(static_cast<float>(glm::max(depth_pyramid_image.extent.width, depth_pyramid_image.extent.height)))) + 1);
                data.depth_pyramid_id = depth_pyramid_descriptor_handle;
                data.skip_occlusion = skip_occlusion ? 1 : 0;

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cull_mesh_pipeline);
                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                vkCmdDispatch(cmd, get_group_count(renderables.size(), 256), 1, 1);
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, frame.query_pool_timestamp, 3);
            }
        );

        graph.add_pass(
            Pass("Compact dispatch early")
                .read_buffer(prefix_sum_buffer, AccessType::ComputeShaderReadOther)
                .write_buffer(dispatch_buffer, AccessType::ComputeShaderWrite),
            [&]()
            {
                struct PushData
                {
                    VkDeviceAddress prefix_sum_buffer;
                    VkDeviceAddress dispatch_buffer;
                    uint32_t prefix_sum_idx;
                };

                PushData data{};
                data.prefix_sum_buffer = get_buffer_address(graph, prefix_sum_buffer);
                data.dispatch_buffer = get_buffer_address(graph, dispatch_buffer);
                data.prefix_sum_idx = 0;

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compact_dispatch_pipeline);
                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                vkCmdDispatch(cmd, 1, 1, 1);
            }
        );

        graph.add_pass(
            Pass("Compute cull meshlets early")
                .read_buffer(dispatch_buffer, AccessType::IndirectBuffer)
                .read_buffer(prefix_sum_buffer, AccessType::ComputeShaderReadOther)
                .read_image(depth_pyramid, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer)
                .write_buffer(dispatch_buffer, AccessType::ComputeShaderWrite)
                .write_buffer(meshlet_draw_buffer, AccessType::ComputeShaderWrite)
                .write_buffer(late_pass_candidates, AccessType::ComputeShaderWrite),
            [&]()
            {
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, frame.query_pool_timestamp, 4);
                struct PushData
                {
                    glm::mat4 view;
                    glm::mat4 prev_view;
                    VkDeviceAddress meshlet_buffer;
                    VkDeviceAddress object_buffer;
                    VkDeviceAddress prefix_sum_buffer;
                    VkDeviceAddress meshlet_draw_buffer;
                    VkDeviceAddress dispatch_buffer;
                    VkDeviceAddress late_pass_candidates_buffer;
                    glm::vec4 planes;
                    glm::vec2 depth_pyramid_extent;
                    glm::vec2 depth_buffer_extent;
                    float p00;
                    float p11;
                    float near;
                    float far;
                    uint32_t late;
                    float depth_pyramid_lod;
                    uint32_t depth_pyramid_id;
                    uint32_t skip_occlusion;
                };

                auto proj_t = glm::transpose(frame_context.proj);
                auto m0 = proj_t[0];
                auto m1 = proj_t[1];
                auto m3 = proj_t[3];
                auto left_plane = glm::normalize(glm::vec3(m3 + m0));
                auto bottom_plane = glm::normalize(glm::vec3(m3 + m1));

                PushData data{};
                data.view = frame_context.view;
                data.prev_view = frame_context.prev_view;
                data.meshlet_buffer = meshlet_buffer.address;
                data.object_buffer = object_buffer.address;
                data.prefix_sum_buffer = get_buffer_address(graph, prefix_sum_buffer);
                data.meshlet_draw_buffer = get_buffer_address(graph, meshlet_draw_buffer);
                data.dispatch_buffer = get_buffer_address(graph, dispatch_buffer);
                data.late_pass_candidates_buffer = late_pass_candidates_buffer.address + meshlet_offset * sizeof(uint32_t);
                data.planes = glm::vec4(left_plane.x, left_plane.z, bottom_plane.y, bottom_plane.z);
                data.depth_pyramid_extent = glm::vec2(depth_pyramid_image.extent.width, depth_pyramid_image.extent.height);
                data.depth_buffer_extent = glm::vec2(swapchain.extent.width, swapchain.extent.height);
                data.p00 = frame_context.proj[0][0];
                data.p11 = frame_context.proj[1][1];
                data.near = camera.near;
                data.far = camera.far;
                data.late = 0;
                data.depth_pyramid_lod = static_cast<float>(glm::floor(glm::log2(static_cast<float>(glm::max(depth_pyramid_image.extent.width, depth_pyramid_image.extent.height)))) + 1);
                data.depth_pyramid_id = depth_pyramid_descriptor_handle;
                data.skip_occlusion = skip_occlusion ? 1 : 0;

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cull_meshlet_pipeline);
                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                vkCmdDispatchIndirect(cmd, get_buffer(graph, dispatch_buffer), 0);
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, frame.query_pool_timestamp, 5);
            }
        );

        graph.add_pass(
            Pass("Vbuffer early")
                .read_buffer(dispatch_buffer, AccessType::IndirectBuffer)
                .read_buffer(prefix_sum_buffer, AccessType::MeshShaderReadSampledImageOrUniformTexelBuffer)
                .read_buffer(meshlet_draw_buffer, AccessType::MeshShaderReadSampledImageOrUniformTexelBuffer)
                .write_image(vbuffer, AccessType::ColorAttachmentWrite)
                .write_image(depth_image, AccessType::DepthStencilAttachmentWrite),
            [&]()
            {
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, frame.query_pool_timestamp, 6);

                VkClearValue clear_value{};
                clear_value.color.uint32[0] = 0xFFFFFFFFu;
                clear_value.color.uint32[1] = 0xFFFFFFFFu;
                clear_value.color.uint32[2] = 0;
                clear_value.color.uint32[3] = 0;

                VkRenderingAttachmentInfo rendering_attachment_info = VkRenderingAttachmentInfo{ .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
                rendering_attachment_info.imageView = get_image_view(graph, vbuffer);
                rendering_attachment_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                rendering_attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                rendering_attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                rendering_attachment_info.clearValue = clear_value;

                VkRenderingAttachmentInfo depth_attachment_info{ .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
                depth_attachment_info.imageView = get_image_view(graph, depth_image);
                depth_attachment_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                depth_attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                depth_attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

                VkRenderingInfo rendering_info{ .sType = VK_STRUCTURE_TYPE_RENDERING_INFO };
                rendering_info.renderArea = VkRect2D{ VkOffset2D{ 0, 0 }, swapchain.extent };
                rendering_info.layerCount = 1;
                rendering_info.colorAttachmentCount = 1;
                rendering_info.pColorAttachments = &rendering_attachment_info;
                rendering_info.pDepthAttachment = &depth_attachment_info;

                vkCmdBeginRendering(cmd, &rendering_info);
                vkCmdBeginQuery(cmd, frame.query_pool_mesh_pipeline, 0, 0);

                VkViewport viewport{};
                viewport.x = 0;
                viewport.y = static_cast<float>(swapchain.extent.height);
                viewport.width = static_cast<float>(swapchain.extent.width);
                viewport.height = -static_cast<float>(swapchain.extent.height);
                viewport.minDepth = 0.0f;
                viewport.maxDepth = 1.0f;
                vkCmdSetViewport(cmd, 0, 1, &viewport);

                VkRect2D scissor{};
                scissor.offset.x = 0;
                scissor.offset.y = 0;
                scissor.extent.width = swapchain.extent.width;
                scissor.extent.height = swapchain.extent.height;
                vkCmdSetScissor(cmd, 0, 1, &scissor);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vbuffer_pipeline);

                struct PushData
                {
                    glm::mat4 view_proj;
                    VkDeviceAddress vertex_buffer;
                    VkDeviceAddress object_buffer;
                    VkDeviceAddress material_buffer;
                    VkDeviceAddress meshlet_buffer;
                    VkDeviceAddress meshlet_indices_buffer;
                    VkDeviceAddress prefix_sum_buffer;
                    VkDeviceAddress meshlet_draw_buffer;
                };

                PushData data{};
                data.view_proj = !freeze_matrices ? frame_context.view_proj : frame_context.debug_view_proj;
                data.vertex_buffer = vertex_buffer.address;
                data.object_buffer = object_buffer.address;
                data.material_buffer = material_buffer.address;
                data.meshlet_buffer = meshlet_buffer.address;
                data.meshlet_indices_buffer = meshlet_indices_buffer.address;
                data.prefix_sum_buffer = get_buffer_address(graph, prefix_sum_buffer);
                data.meshlet_draw_buffer = get_buffer_address(graph, meshlet_draw_buffer);

                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(data), &data);
                vkCmdDrawMeshTasksIndirectEXT(cmd, get_buffer(graph, dispatch_buffer), sizeof(Dispatch), 1, 0);
                vkCmdEndQuery(cmd, frame.query_pool_mesh_pipeline, 0);
                vkCmdEndRendering(cmd);
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, frame.query_pool_timestamp, 7);
            }
        );

        graph.add_pass(
            Pass("Hiz")
                .read_image(depth_image, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer)
                .write_image(depth_pyramid, AccessType::ComputeShaderWrite)
                .write_buffer(spd_counter, AccessType::ComputeShaderWrite),
            [&]()
            {
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, frame.query_pool_timestamp, 8);
                struct PushData
                {
                    VkDeviceAddress spd_counter_buffer;
                    glm::vec2 rcp_resolution; // extent of depth buffer
                    uint32_t mips;
                    uint32_t num_wgs;
                    uint32_t src_id; // depth texture to sample
                    uint32_t dst_id; // mip level 0 of hi-z we write to, offset accordingly!
                };

                auto width = next_pow2(swapchain.extent.width);
                auto height = next_pow2(swapchain.extent.height);
                auto group_count_x = get_group_count(width, 64);
                auto group_count_y = get_group_count(height, 64);

                PushData data{};
                data.spd_counter_buffer = spd_counter_buffer.address;
                data.rcp_resolution = glm::vec2(1.0) / glm::vec2(swapchain.extent.width, swapchain.extent.height);
                data.mips = get_depth_pyramid_image_count(depth_pyramid_image);
                data.num_wgs = group_count_x * group_count_y;
                data.src_id = get_image_id(graph, depth_image);
                data.dst_id = depth_pyramid_first_mip;

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hiz_pipeline);
                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                vkCmdDispatch(cmd, group_count_x, group_count_y, 1);
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, frame.query_pool_timestamp, 9);
            }
        );

        skip_occlusion = false;

        graph.add_pass(
            Pass("Zero buffers late")
                .write_buffer(dispatch_buffer, AccessType::TransferWrite),
            [&]()
            {
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, frame.query_pool_timestamp, 20);
                vkCmdFillBuffer(cmd, get_buffer(graph, dispatch_buffer), 0, VK_WHOLE_SIZE, 0);
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, frame.query_pool_timestamp, 21);
            }
        );

        graph.add_pass(
            Pass("Compute cull mesh late")
                .read_image(depth_pyramid, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer)
                .read_buffer(prefix_sum_buffer, AccessType::ComputeShaderReadOther)
                .read_buffer(late_pass_candidates, AccessType::ComputeShaderReadOther)
                .write_buffer(prefix_sum_buffer, AccessType::ComputeShaderWrite),
            [&]()
            {
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, frame.query_pool_timestamp, 10);
                struct PushData
                {
                    glm::mat4 view;
                    glm::mat4 prev_view;
                    VkDeviceAddress mesh_buffer;
                    VkDeviceAddress object_buffer;
                    VkDeviceAddress prefix_sum_buffer;
                    VkDeviceAddress late_pass_candidates_buffer;
                    glm::vec4 planes;
                    glm::vec2 depth_pyramid_extent;
                    glm::vec2 depth_buffer_extent;
                    float p00;
                    float p11;
                    float near;
                    float far;
                    uint32_t count;
                    float lod_distance_factor;
                    uint32_t late;
                    float depth_pyramid_lod;
                    uint32_t depth_pyramid_id;
                    uint32_t skip_occlusion;
                };

                auto proj_t = glm::transpose(frame_context.proj);
                auto m0 = proj_t[0];
                auto m1 = proj_t[1];
                auto m3 = proj_t[3];
                auto left_plane = glm::normalize(glm::vec3(m3 + m0));
                auto bottom_plane = glm::normalize(glm::vec3(m3 + m1));

                PushData data{};
                data.view = frame_context.view;
                data.prev_view = frame_context.prev_view;
                data.mesh_buffer = mesh_buffer.address;
                data.object_buffer = object_buffer.address;
                data.prefix_sum_buffer = get_buffer_address(graph, prefix_sum_buffer);
                data.late_pass_candidates_buffer = late_pass_candidates_buffer.address;
                data.planes = glm::vec4(left_plane.x, left_plane.z, bottom_plane.y, bottom_plane.z);
                data.depth_pyramid_extent = glm::vec2(depth_pyramid_image.extent.width, depth_pyramid_image.extent.height);
                data.depth_buffer_extent = glm::vec2(swapchain.extent.width, swapchain.extent.height);
                data.p00 = frame_context.proj[0][0];
                data.p11 = frame_context.proj[1][1];
                data.near = camera.near;
                data.far = camera.far;
                data.count = renderables.size();
                data.lod_distance_factor = 2.0f / (data.p11 * swapchain.extent.height);
                data.late = 1;
                data.depth_pyramid_lod = static_cast<float>(glm::floor(glm::log2(static_cast<float>(glm::max(depth_pyramid_image.extent.width, depth_pyramid_image.extent.height)))) + 1);
                data.depth_pyramid_id = depth_pyramid_descriptor_handle;
                data.skip_occlusion = skip_occlusion ? 1 : 0;

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cull_mesh_pipeline);
                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                vkCmdDispatch(cmd, get_group_count(renderables.size(), 256), 1, 1);
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, frame.query_pool_timestamp, 11);
            }
        );

        graph.add_pass(
            Pass("Compact dispatch late")
                .read_buffer(prefix_sum_buffer, AccessType::ComputeShaderReadOther)
                .write_buffer(dispatch_buffer, AccessType::ComputeShaderWrite),
            [&]()
            {
                struct PushData
                {
                    VkDeviceAddress prefix_sum_buffer;
                    VkDeviceAddress dispatch_buffer;
                    uint32_t prefix_sum_idx;
                };

                PushData data{};
                data.prefix_sum_buffer = get_buffer_address(graph, prefix_sum_buffer);
                data.dispatch_buffer = get_buffer_address(graph, dispatch_buffer);
                data.prefix_sum_idx = 0;

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compact_dispatch_pipeline);
                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                vkCmdDispatch(cmd, 1, 1, 1);
            }
        );

        graph.add_pass(
            Pass("Compute cull meshlets late")
                .read_buffer(dispatch_buffer, AccessType::IndirectBuffer)
                .read_buffer(prefix_sum_buffer, AccessType::ComputeShaderReadOther)
                .read_buffer(late_pass_candidates, AccessType::ComputeShaderReadOther)
                .read_image(depth_pyramid, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer)
                .write_buffer(dispatch_buffer, AccessType::ComputeShaderWrite)
                .write_buffer(meshlet_draw_buffer, AccessType::ComputeShaderWrite),
            [&]()
            {
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, frame.query_pool_timestamp, 12);

                struct PushData
                {
                    glm::mat4 view;
                    glm::mat4 prev_view;
                    VkDeviceAddress meshlet_buffer;
                    VkDeviceAddress object_buffer;
                    VkDeviceAddress prefix_sum_buffer;
                    VkDeviceAddress meshlet_draw_buffer;
                    VkDeviceAddress dispatch_buffer;
                    VkDeviceAddress late_pass_candidates_buffer;
                    glm::vec4 planes;
                    glm::vec2 depth_pyramid_extent;
                    glm::vec2 depth_buffer_extent;
                    float p00;
                    float p11;
                    float near;
                    float far;
                    uint32_t late;
                    float depth_pyramid_lod;
                    uint32_t depth_pyramid_id;
                    uint32_t skip_occlusion;
                };

                auto proj_t = glm::transpose(frame_context.proj);
                auto m0 = proj_t[0];
                auto m1 = proj_t[1];
                auto m3 = proj_t[3];
                auto left_plane = glm::normalize(glm::vec3(m3 + m0));
                auto bottom_plane = glm::normalize(glm::vec3(m3 + m1));

                PushData data{};
                data.view = frame_context.view;
                data.prev_view = frame_context.prev_view;
                data.meshlet_buffer = meshlet_buffer.address;
                data.object_buffer = object_buffer.address;
                data.prefix_sum_buffer = get_buffer_address(graph, prefix_sum_buffer);
                data.meshlet_draw_buffer = get_buffer_address(graph, meshlet_draw_buffer);
                data.dispatch_buffer = get_buffer_address(graph, dispatch_buffer);
                data.late_pass_candidates_buffer = late_pass_candidates_buffer.address + meshlet_offset * sizeof(uint32_t);
                data.planes = glm::vec4(left_plane.x, left_plane.z, bottom_plane.y, bottom_plane.z);
                data.depth_pyramid_extent = glm::vec2(depth_pyramid_image.extent.width, depth_pyramid_image.extent.height);
                data.depth_buffer_extent = glm::vec2(swapchain.extent.width, swapchain.extent.height);
                data.p00 = frame_context.proj[0][0];
                data.p11 = frame_context.proj[1][1];
                data.near = camera.near;
                data.far = camera.far;
                data.late = 1;
                data.depth_pyramid_lod = static_cast<float>(glm::floor(glm::log2(static_cast<float>(glm::max(depth_pyramid_image.extent.width, depth_pyramid_image.extent.height)))) + 1);
                data.depth_pyramid_id = depth_pyramid_descriptor_handle;
                data.skip_occlusion = skip_occlusion ? 1 : 0;

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cull_meshlet_pipeline);
                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                vkCmdDispatchIndirect(cmd, get_buffer(graph, dispatch_buffer), 0);
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, frame.query_pool_timestamp, 13);
            }
        );

        // TODO: should we declare read_image for attachments?
        graph.add_pass(
            Pass("Vbuffer late")
                .read_buffer(dispatch_buffer, AccessType::IndirectBuffer)
                .read_buffer(prefix_sum_buffer, AccessType::MeshShaderReadSampledImageOrUniformTexelBuffer)
                .read_buffer(meshlet_draw_buffer, AccessType::MeshShaderReadSampledImageOrUniformTexelBuffer)
                .write_image(vbuffer, AccessType::ColorAttachmentReadWrite)
                .write_image(depth_image, AccessType::DepthStencilAttachmentReadWrite),
            [&]()
            {
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, frame.query_pool_timestamp, 14);

                VkRenderingAttachmentInfo rendering_attachment_info = VkRenderingAttachmentInfo{ .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
                rendering_attachment_info.imageView = get_image_view(graph, vbuffer);
                rendering_attachment_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                rendering_attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                rendering_attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

                VkRenderingAttachmentInfo depth_attachment_info{ .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
                depth_attachment_info.imageView = get_image_view(graph, depth_image);
                depth_attachment_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                depth_attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                depth_attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

                VkRenderingInfo rendering_info{ .sType = VK_STRUCTURE_TYPE_RENDERING_INFO };
                rendering_info.renderArea = VkRect2D{ VkOffset2D{ 0, 0 }, swapchain.extent };
                rendering_info.layerCount = 1;
                rendering_info.colorAttachmentCount = 1;
                rendering_info.pColorAttachments = &rendering_attachment_info;
                rendering_info.pDepthAttachment = &depth_attachment_info;

                vkCmdBeginRendering(cmd, &rendering_info);
                vkCmdBeginQuery(cmd, frame.query_pool_mesh_pipeline, 1, 0);

                VkViewport viewport{};
                viewport.x = 0;
                viewport.y = static_cast<float>(swapchain.extent.height);
                viewport.width = static_cast<float>(swapchain.extent.width);
                viewport.height = -static_cast<float>(swapchain.extent.height);
                viewport.minDepth = 0.0f;
                viewport.maxDepth = 1.0f;
                vkCmdSetViewport(cmd, 0, 1, &viewport);

                VkRect2D scissor{};
                scissor.offset.x = 0;
                scissor.offset.y = 0;
                scissor.extent.width = swapchain.extent.width;
                scissor.extent.height = swapchain.extent.height;
                vkCmdSetScissor(cmd, 0, 1, &scissor);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vbuffer_pipeline);

                struct PushData
                {
                    glm::mat4 view_proj;
                    VkDeviceAddress vertex_buffer;
                    VkDeviceAddress object_buffer;
                    VkDeviceAddress material_buffer;
                    VkDeviceAddress meshlet_buffer;
                    VkDeviceAddress meshlet_indices_buffer;
                    VkDeviceAddress prefix_sum_buffer;
                    VkDeviceAddress meshlet_draw_buffer;
                };

                PushData data{};
                data.view_proj = !freeze_matrices ? frame_context.view_proj : frame_context.debug_view_proj;
                data.vertex_buffer = vertex_buffer.address;
                data.object_buffer = object_buffer.address;
                data.material_buffer = material_buffer.address;
                data.meshlet_buffer = meshlet_buffer.address;
                data.meshlet_indices_buffer = meshlet_indices_buffer.address;
                data.prefix_sum_buffer = get_buffer_address(graph, prefix_sum_buffer);
                data.meshlet_draw_buffer = get_buffer_address(graph, meshlet_draw_buffer);

                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(data), &data);
                vkCmdDrawMeshTasksIndirectEXT(cmd, get_buffer(graph, dispatch_buffer), sizeof(Dispatch), 1, 0);
                vkCmdEndQuery(cmd, frame.query_pool_mesh_pipeline, 1);
                vkCmdEndRendering(cmd);
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, frame.query_pool_timestamp, 15);
            }
        );

        if (b_shadows)
        {
            graph.add_pass(
                Pass("Zero buffers for shadow pass")
                    .write_buffer(prefix_sum_buffer, AccessType::TransferWrite),
                [&]()
                {
                    vkCmdFillBuffer(cmd, get_buffer(graph, prefix_sum_buffer), 0, sizeof(uint64_t), 0);
                }
            );

            graph.add_pass(
                Pass("Generate shadow matrices")
                    .write_buffer(global_buffer, AccessType::ComputeShaderWrite),
                [&]()
                {
                    struct PushData
                    {
                        VkDeviceAddress global_buffer;
                        float4x4 inverse_view_proj;
                        float3 sunlight_dir; // normalized
                        float shadow_near;
                        float shadow_far;
                        float lambda;
                    };
                    PushData data{};
                    data.global_buffer = get_buffer_address(graph, global_buffer, sizeof(GlobalData) * (frame_number % FRAMES_IN_FLIGHT));
                    auto proj = glm::perspective(glm::radians(camera.fov), static_cast<float>(swapchain.extent.width) / static_cast<float>(swapchain.extent.height), SHADOW_DRAW_DISTANCE, camera.near);
                    data.inverse_view_proj = glm::inverse(proj * frame_context.view);
                    data.sunlight_dir = sunlight_dir;
                    data.shadow_near = camera.near;
                    data.shadow_far = SHADOW_DRAW_DISTANCE;
                    data.lambda = 0.95;

                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shadow_matrices_pipeline);
                    vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                    vkCmdDispatch(cmd, 1, 1, 1);
                }
            );

            graph.add_pass(
                Pass("Cull shadow instances")
                    .read_buffer(prefix_sum_buffer, AccessType::ComputeShaderReadOther)
                    .read_buffer(global_buffer, AccessType::ComputeShaderReadOther)
                    .write_buffer(prefix_sum_buffer, AccessType::ComputeShaderWrite),
                [&]()
                {
                    struct PushData
                    {
                        VkDeviceAddress global_buffer;
                        VkDeviceAddress object_buffer;
                        VkDeviceAddress mesh_buffer;
                        VkDeviceAddress prefix_sum_buffer;
                        uint32_t count;
                        uint32_t shadowmap_height;
                        float lod_distance_factor;
                    };

                    PushData data{};
                    data.global_buffer = get_buffer_address(graph, global_buffer, sizeof(GlobalData) * (frame_number % FRAMES_IN_FLIGHT));
                    data.object_buffer = object_buffer.address;
                    data.mesh_buffer = mesh_buffer.address;
                    data.prefix_sum_buffer = get_buffer_address(graph, prefix_sum_buffer);
                    data.count = static_cast<uint32_t>(renderables.size());
                    data.shadowmap_height = SHADOWMAP_EXTENT;
                    auto p11 = frame_context.proj[1][1];
                    data.lod_distance_factor = 2.0f / (p11 * swapchain.extent.height);

                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cull_shadow_pipeline);
                    vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                    vkCmdDispatch(cmd, get_group_count(renderables.size(), 256), 1, 1);
                }
            );

            for (uint32_t cascade_index = 0; cascade_index < SUN_CASCADE_COUNT; cascade_index++)
            {
                graph.add_pass(
                    Pass("Compact dispatch for cascade " + std::to_string(cascade_index + 1))
                        .read_buffer(prefix_sum_buffer, AccessType::ComputeShaderReadOther)
                        .write_buffer(dispatch_buffer, AccessType::ComputeShaderWrite),
                    [&, cascade_index]()
                    {
                        struct PushData
                        {
                            VkDeviceAddress prefix_sum_buffer;
                            VkDeviceAddress dispatch_buffer;
                            uint32_t prefix_sum_idx;
                        };

                        PushData data{};
                        data.prefix_sum_buffer = get_buffer_address(graph, prefix_sum_buffer);
                        data.dispatch_buffer = get_buffer_address(graph, dispatch_buffer);
                        data.prefix_sum_idx = cascade_index;

                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compact_dispatch_pipeline);
                        vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                        vkCmdDispatch(cmd, 1, 1, 1);
                    }
                );

                graph.add_pass(
                    Pass("Cull shadow meshlets")
                        .read_buffer(dispatch_buffer, AccessType::IndirectBuffer)
                        .read_buffer(prefix_sum_buffer, AccessType::ComputeShaderReadOther)
                        .read_buffer(global_buffer, AccessType::ComputeShaderReadOther)
                        .write_buffer(dispatch_buffer, AccessType::ComputeShaderWrite)
                        .write_buffer(meshlet_draw_buffer, AccessType::ComputeShaderWrite),
                    [&, cascade_index]()
                    {
                        struct PushData
                        {
                            VkDeviceAddress global_buffer;
                            VkDeviceAddress object_buffer;
                            VkDeviceAddress meshlet_buffer;
                            VkDeviceAddress prefix_sum_buffer;
                            VkDeviceAddress dispatch_buffer;
                            VkDeviceAddress meshlet_draw_buffer;
                            uint32_t count;
                            uint32_t shadowmap_height;
                            uint32_t cascade_index;
                        };

                        PushData data{};
                        data.global_buffer = get_buffer_address(graph, global_buffer, sizeof(GlobalData) * (frame_number % FRAMES_IN_FLIGHT));
                        data.object_buffer = object_buffer.address;
                        data.meshlet_buffer = meshlet_buffer.address;
                        data.prefix_sum_buffer = get_buffer_address(graph, prefix_sum_buffer);
                        data.dispatch_buffer = get_buffer_address(graph, dispatch_buffer);
                        data.meshlet_draw_buffer = get_buffer_address(graph, meshlet_draw_buffer);
                        data.count = static_cast<uint32_t>(renderables.size());
                        data.shadowmap_height = SHADOWMAP_EXTENT;

                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cull_shadow_meshlet_pipeline);
                        vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                        vkCmdDispatchIndirect(cmd, get_buffer(graph, dispatch_buffer), 0);
                    }
                );

                graph.add_pass(
                    Pass("Shadowmap " + std::to_string(cascade_index + 1))
                        .read_buffer(dispatch_buffer, AccessType::IndirectBuffer)
                        .read_buffer(prefix_sum_buffer, AccessType::MeshShaderReadSampledImageOrUniformTexelBuffer)
                        .read_buffer(meshlet_draw_buffer, AccessType::MeshShaderReadSampledImageOrUniformTexelBuffer)
                        .read_buffer(global_buffer, AccessType::ComputeShaderReadOther)
                        .read_buffer(prefix_sum_buffer, AccessType::ComputeShaderReadOther)
                        .write_image(shadowmaps[cascade_index], AccessType::DepthStencilAttachmentReadWrite),
                    [&, cascade_index]()
                    {
                        VkRenderingAttachmentInfo depth_attachment_info{ .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
                        depth_attachment_info.imageView = get_image_view(graph, shadowmaps[cascade_index]);
                        depth_attachment_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                        depth_attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                        depth_attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

                        VkRenderingInfo rendering_info{ .sType = VK_STRUCTURE_TYPE_RENDERING_INFO };
                        rendering_info.renderArea = VkRect2D{ VkOffset2D{ 0, 0 }, VkExtent2D{ SHADOWMAP_EXTENT, SHADOWMAP_EXTENT } };
                        rendering_info.layerCount = 1;
                        rendering_info.pDepthAttachment = &depth_attachment_info;

                        vkCmdBeginRendering(cmd, &rendering_info);

                        VkViewport viewport{};
                        viewport.x = 0;
                        viewport.y = static_cast<float>(SHADOWMAP_EXTENT);
                        viewport.width = static_cast<float>(SHADOWMAP_EXTENT);
                        viewport.height = -static_cast<float>(SHADOWMAP_EXTENT);
                        viewport.minDepth = 0.0f;
                        viewport.maxDepth = 1.0f;
                        vkCmdSetViewport(cmd, 0, 1, &viewport);

                        VkRect2D scissor{};
                        scissor.offset.x = 0;
                        scissor.offset.y = 0;
                        scissor.extent.width = SHADOWMAP_EXTENT;
                        scissor.extent.height = SHADOWMAP_EXTENT;
                        vkCmdSetScissor(cmd, 0, 1, &scissor);

                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, depth_pipeline);

                        struct PushData
                        {
                            VkDeviceAddress global_buffer;
                            VkDeviceAddress object_buffer;
                            VkDeviceAddress vertex_buffer;
                            VkDeviceAddress meshlet_buffer;
                            VkDeviceAddress prefix_sum_buffer;
                            VkDeviceAddress meshlet_indices_buffer;
                            VkDeviceAddress meshlet_draw_buffer;
                            uint32_t cascade_index;
                        };

                        PushData data{};
                        data.global_buffer = get_buffer_address(graph, global_buffer, sizeof(GlobalData) * (frame_number % FRAMES_IN_FLIGHT));
                        data.object_buffer = object_buffer.address;
                        data.vertex_buffer = vertex_buffer.address;
                        data.meshlet_buffer = meshlet_buffer.address;
                        data.prefix_sum_buffer = get_buffer_address(graph, prefix_sum_buffer);
                        data.meshlet_indices_buffer = meshlet_indices_buffer.address;
                        data.meshlet_draw_buffer = get_buffer_address(graph, meshlet_draw_buffer);
                        data.cascade_index = cascade_index;

                        vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(data), &data);
                        vkCmdDrawMeshTasksIndirectEXT(cmd, get_buffer(graph, dispatch_buffer), sizeof(Dispatch), 1, 0);
                        vkCmdEndRendering(cmd);
                    }
                );
            }
        }

        graph.add_pass(
            Pass("Lighting")
                .read_image(vbuffer, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer)
                .read_image(global_buffer, AccessType::ComputeShaderReadOther)
                // TODO: handle other cascades, use a texture array?
                .read_image(shadowmaps[0], AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer)
                .write_image(draw_image, AccessType::ComputeShaderWrite),
            [&]()
            {
                struct PushData
                {
                    glm::mat4 view_proj;
                    VkDeviceAddress vertex_buffer;
                    VkDeviceAddress object_buffer;
                    VkDeviceAddress meshlet_buffer;
                    VkDeviceAddress meshlet_indices_buffer;
                    VkDeviceAddress material_buffer;
                    VkDeviceAddress global_buffer;
                    glm::vec3 camera_pos;
                    uint32_t visbuffer_id;
                    glm::vec2 screen_extent;
                    uint32_t debug_id;
                    uint32_t output_id;
                    glm::vec3 light_dir;
                    uint32_t shadowmap_id; // Offset accordingly
                };

                PushData data{};
                data.view_proj = frame_context.view_proj;
                data.vertex_buffer = vertex_buffer.address;
                data.object_buffer = object_buffer.address;
                data.meshlet_buffer = meshlet_buffer.address;
                data.meshlet_indices_buffer = meshlet_indices_buffer.address;
                data.material_buffer = material_buffer.address;
                data.global_buffer = get_buffer_address(graph, global_buffer, sizeof(GlobalData) * (frame_number % FRAMES_IN_FLIGHT));
                data.camera_pos = camera.position;
                data.visbuffer_id = get_image_id(graph, vbuffer);
                data.screen_extent = glm::vec2(swapchain.extent.width, swapchain.extent.height);
                data.debug_id = DEBUG_ID;
                data.output_id = get_image_id(graph, draw_image);
                data.light_dir = sunlight_dir;
                data.shadowmap_id = b_shadows ? get_image_id(graph, shadowmaps[0]) : -1u;

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lighting_pipeline);
                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                vkCmdDispatch(cmd, get_group_count(swapchain.extent.width, 8), get_group_count(swapchain.extent.height, 8), 1);
            }
        );

        graph.add_pass(
            Pass("Hiz temporal")
                .read_image(depth_image, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer)
                .write_image(depth_pyramid, AccessType::ComputeShaderWrite)
                .write_buffer(spd_counter, AccessType::ComputeShaderWrite),
            [&]()
            {
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, frame.query_pool_timestamp, 16);
                struct PushData
                {
                    VkDeviceAddress spd_counter_buffer;
                    glm::vec2 rcp_resolution; // extent of depth buffer
                    uint32_t mips;
                    uint32_t num_wgs;
                    uint32_t src_id; // depth texture to sample
                    uint32_t dst_id; // mip level 0 of hi-z we write to, offset accordingly!
                };

                auto width = next_pow2(swapchain.extent.width);
                auto height = next_pow2(swapchain.extent.height);
                auto group_count_x = get_group_count(width, 64);
                auto group_count_y = get_group_count(height, 64);

                PushData data{};
                data.spd_counter_buffer = spd_counter_buffer.address;
                data.rcp_resolution = glm::vec2(1.0) / glm::vec2(swapchain.extent.width, swapchain.extent.height);
                data.mips = get_depth_pyramid_image_count(depth_pyramid_image);
                data.num_wgs = group_count_x * group_count_y;
                data.src_id = get_image_id(graph, depth_image);
                data.dst_id = depth_pyramid_first_mip;

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hiz_pipeline);
                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                vkCmdDispatch(cmd, group_count_x, group_count_y, 1);
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, frame.query_pool_timestamp, 17);
            }
        );

        graph.add_pass(
            Pass("Copy to swapchain")
                .read_image(draw_image, AccessType::TransferRead)
                .write_image(swapchain_image, AccessType::TransferWrite),
            [&]()
            {
                copy_image(cmd, get_image(graph, draw_image), get_image(graph, swapchain_image), swapchain.extent, swapchain.extent);
            }
        );

        graph.add_pass(
            Pass("ImGui")
                .write_image(swapchain_image, AccessType::ColorAttachmentReadWrite),
            [&]()
            {
                VkRenderingAttachmentInfo color_attachment{};
                color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                color_attachment.imageView = swapchain.image_views[swapchain_image_idx];
                color_attachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

                VkRenderingInfo render_info{};
                render_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                render_info.renderArea = VkRect2D{ VkOffset2D{ 0, 0 }, swapchain.extent };
                render_info.layerCount = 1;
                render_info.colorAttachmentCount = 1;
                render_info.pColorAttachments = &color_attachment;

                vkCmdBeginRendering(cmd, &render_info);
                ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
                vkCmdEndRendering(cmd);
            }
        );

        // Rendergraph compilation
        graph.compile(cmd);
        graph.execute(cmd);

        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, frame.query_pool_timestamp, 1);
        VK_CHECK(vkEndCommandBuffer(cmd));

        // Submit command buffer
        VkCommandBufferSubmitInfo cmd_submit_info{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        cmd_submit_info.commandBuffer = cmd;

        VkSemaphoreSubmitInfo wait_info{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
        wait_info.semaphore = frame.image_acquired_semaphore;
        wait_info.value = 1;
        wait_info.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

        VkSemaphoreSubmitInfo signal_info{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
        signal_info.semaphore = swapchain.render_done_semaphores[swapchain_image_idx];
        signal_info.value = 1;
        signal_info.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;

        VkSubmitInfo2 submit{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        submit.pSignalSemaphoreInfos = &signal_info;
        submit.signalSemaphoreInfoCount = 1;
        submit.pWaitSemaphoreInfos = &wait_info;
        submit.waitSemaphoreInfoCount = 1;
        submit.pCommandBufferInfos = &cmd_submit_info;
        submit.commandBufferInfoCount = 1;
        VK_CHECK(vkQueueSubmit2(graphics_queue, 1, &submit, frame.fence));

        // Present image
        VkPresentInfoKHR present_info{ .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = &swapchain.render_done_semaphores[swapchain_image_idx];
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &swapchain.swapchain;
        present_info.pImageIndices = &swapchain_image_idx;

        VkResult present_result = vkQueuePresentKHR(graphics_queue, &present_info);
        VK_CHECK_SWAPCHAIN(present_result);

        graph.reset();

        frame_number++;
    }

    vkDeviceWaitIdle(device);

    destroy_buffer(allocator, vertex_buffer);
    destroy_buffer(allocator, object_buffer);
    destroy_buffer(allocator, material_buffer);
    destroy_buffer(allocator, mesh_buffer);
    destroy_buffer(allocator, meshlet_buffer);
    destroy_buffer(allocator, meshlet_indices_buffer);
    destroy_buffer(allocator, spd_counter_buffer);
    destroy_buffer(allocator, late_pass_candidates_buffer);

    destroy_image(device, allocator, depth_pyramid_image);
    for (const auto& img : shadowmap_images)
    {
        destroy_image(device, allocator, img);
    }

    vkDestroySampler(device, linear_samp, nullptr);
    vkDestroySampler(device, depth_reduce_samp, nullptr);
    vkDestroySampler(device, nearest_samp, nullptr);
    vkDestroySampler(device, shadow_samp, nullptr);

    for (const auto& view : depth_pyramid_image_views)
    {
        vkDestroyImageView(device, view, nullptr);
    }

    for (const auto& pipeline : pipelines)
    {
        vkDestroyPipeline(device, pipeline, nullptr);
    }

    asset_loader.cleanup();
    graph.cleanup();
}

void Engine::register_object(const Node& node, const glm::mat4& top_matrix, const std::vector<Node>& children, const std::vector<GltfMesh>& mesh_assets)
{
    auto world_matrix = top_matrix * node.world_transform;

    if (node.mesh_index != -1)
    {
        for (const auto& mesh : mesh_assets[node.mesh_index].mesh)
        {
            ObjectData obj{};

            glm::vec3 translation{};
            glm::vec3 scale{};
            glm::vec4 rotation{};

            decompose_transform(world_matrix, translation, scale, rotation);

            // TODO: handle non uniform scaling
            obj.translation = translation;
            obj.scale = glm::max(glm::max(scale.x, scale.y), scale.z);
            obj.orientation = rotation;
            obj.material_id = mesh.material_id;

            // TODO: test req - do we need a mesh cache?
            obj.mesh_id = static_cast<uint32_t>(meshes.size());

            renderables.push_back(obj);

            auto& m = meshes.emplace_back(
                MeshData{
                    .center = mesh.center,
                    .radius = mesh.radius,
                    .lod_count = mesh.lod_count }
            );

            memcpy(&m.mesh_lods, &mesh.mesh_lods, sizeof(mesh.mesh_lods));
        }
    }

    for (auto i = 0; i < node.child_count; i++)
    {
        const auto& child = children[node.first_child];
        register_object(child, top_matrix, children, mesh_assets);
    }
}

void Engine::init_descriptors()
{
    uint32_t buffer_count = 10;
    uint32_t image_count = MAX_DESCRIPTORS_IMAGE_COUNT;
    uint32_t sampler_count = 5;

    std::array<VkDescriptorPoolSize, 4> pool_sizes{
        VkDescriptorPoolSize{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = buffer_count },
        VkDescriptorPoolSize{ .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = image_count },
        VkDescriptorPoolSize{ .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .descriptorCount = image_count },
        VkDescriptorPoolSize{ .type = VK_DESCRIPTOR_TYPE_SAMPLER, .descriptorCount = sampler_count },
    };

    desc_pool = create_descriptor_pool(device, pool_sizes.data(), pool_sizes.size(), 4, VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT);

    // Set up bindless
    std::array<VkDescriptorSetLayoutBinding, 2> image_bindings{
        VkDescriptorSetLayoutBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = image_count, .stageFlags = VK_SHADER_STAGE_ALL },
        VkDescriptorSetLayoutBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .descriptorCount = image_count, .stageFlags = VK_SHADER_STAGE_ALL }
    };
    VkDescriptorSetLayoutBinding buffer_binding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = buffer_count, .stageFlags = VK_SHADER_STAGE_ALL };
    VkDescriptorSetLayoutBinding sampler_binding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, .descriptorCount = sampler_count, .stageFlags = VK_SHADER_STAGE_ALL };
    VkDescriptorBindingFlags binding_flags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    std::array<VkDescriptorBindingFlags, 2> image_binding_flags{
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
    };

    buffer_layout = create_descriptor_set_layout(device, &buffer_binding, 1, &binding_flags, 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT);
    image_layout = create_descriptor_set_layout(device, image_bindings.data(), image_bindings.size(), image_binding_flags.data(), image_binding_flags.size(), VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT);
    sampler_layout = create_descriptor_set_layout(device, &sampler_binding, 1, &binding_flags, 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT);

    buffer_set = create_descriptor_set(device, desc_pool, &buffer_layout, &buffer_count);
    image_set = create_descriptor_set(device, desc_pool, &image_layout, &image_count);
    sampler_set = create_descriptor_set(device, desc_pool, &sampler_layout, &sampler_count);

    image_manager.set = image_set;
    assert(image_manager.set != nullptr);
}

void Engine::init_imgui()
{
    std::array<VkDescriptorPoolSize, 2> pool_sizes{
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE },
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_SAMPLER, IMGUI_IMPL_VULKAN_MINIMUM_SAMPLER_POOL_SIZE }
    };

    imgui_pool = create_descriptor_pool(device, pool_sizes.data(), pool_sizes.size(), 1, VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls

    // Setup Platform/Renderer backends
    ImGui_ImplSDL3_InitForVulkan(window);
    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.ApiVersion = VK_API_VERSION_1_4;
    init_info.Instance = instance;
    init_info.PhysicalDevice = physical_device;
    init_info.Device = device;
    init_info.QueueFamily = graphics_queue_family;
    init_info.Queue = graphics_queue;
    init_info.DescriptorPool = imgui_pool;
    init_info.MinImageCount = 2;
    init_info.ImageCount = 2;
    init_info.UseDynamicRendering = true;

    VkPipelineRenderingCreateInfo render_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    render_info.colorAttachmentCount = 1;

    auto swapchain_image_format = VK_FORMAT_B8G8R8A8_UNORM;
    render_info.pColorAttachmentFormats = &swapchain_image_format;
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo = render_info;

    ImGui_ImplVulkan_Init(&init_info);
}

void Engine::destroy_imgui()
{
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    vkDestroyDescriptorPool(device, imgui_pool, nullptr);
}

int main()
{
    Engine engine{};

    engine.init_vulkan();
    engine.init_commands();
    engine.init_sync();
    engine.init_query_pool();
    engine.init_descriptors();
    engine.init_imgui();

    engine.run();

    engine.cleanup();

    fmt::println("Closing app...");
}
