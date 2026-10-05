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
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>

#include <vector>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <array>
#include <cstring>

#define MAX_TIMESTAMP_QUERIES 2
#define MAX_PIPELINE_QUERIES 1

// const char* ASSET_NAME = "Sponza/Sponza.gltf";

const char* ASSET_NAME = "DamagedHelmet/DamagedHelmet.gltf";
int GBUFFER_DEBUG_ID = 4; // color, normal, metal, roughness, debug

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
    return (size + threads + 1) / threads;
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
        // if (strcmp(p_callback_data->pMessageIdName, "VUID-RuntimeSpirv-OpVariable-08746") == 0)
        //     return VK_FALSE;
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

#ifdef USE_RENDERDOC
    SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "x11");
#else
    SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "wayland");
#endif

    SDL_Init(SDL_INIT_VIDEO);
    auto window_flags = static_cast<SDL_WindowFlags>(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    window = SDL_CreateWindow("Untitled vulkan renderer", 1280u, 720u, window_flags);
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

    VkPhysicalDeviceVulkan11Features features_11{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };

    VkPhysicalDeviceFeatures features_10{};
    features_10.shaderInt16 = true;
    features_10.pipelineStatisticsQuery = true;

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
    fmt::println("Initialized Vulkan - Release mode");
#else
    fmt::println("Initialized Vulkan - Debug mode");
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
        vkDestroyQueryPool(device, frame.query_pool_pipeline, nullptr);
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

        query_pool_info.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
        query_pool_info.queryCount = MAX_PIPELINE_QUERIES;
        query_pool_info.pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT;
        VK_CHECK(vkCreateQueryPool(device, &query_pool_info, nullptr, &frame.query_pool_pipeline));

        vkResetQueryPool(device, frame.query_pool_timestamp, 0, MAX_TIMESTAMP_QUERIES);
        vkResetQueryPool(device, frame.query_pool_pipeline, 0, MAX_PIPELINE_QUERIES);
    }
}

void Engine::run()
{
    // Init rendergraph
    graph.device = device;
    graph.allocator = allocator;
    graph.image_manager = &image_manager;

    VkPhysicalDeviceProperties2 properties2{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
    vkGetPhysicalDeviceProperties2(physical_device, &properties2);
    assert(properties2.properties.limits.timestampComputeAndGraphics);
    auto max_push_constant_size = properties2.properties.limits.maxPushConstantsSize;
    auto timestamp_period = properties2.properties.limits.timestampPeriod;

    // Init camera
    Camera camera{};
    camera.position = glm::vec3(0, 0, 5);
    camera.far = 100.0f;
    camera.near = 0.01f;

    auto proj = camera.set_perspective_matrix(glm::radians(camera.fov), static_cast<float>(swapchain.extent.width) / swapchain.extent.height, camera.near);

    AssetLoader asset_loader{ .device = device, .allocator = allocator };
    bool loaded = asset_loader.load_gltf(graphics_queue, imm_fence, imm_pool, imm_buf, ASSET_NAME, static_cast<uint32_t>(image_manager.infos.size()));
    if (!loaded)
    {
        assert(0 && "load_gltf failed");
    }

    // TODO: handle meshlets

    for (const auto& node : asset_loader.parent_nodes)
    {
        register_object(node, glm::mat4(1.0), asset_loader.children_nodes, asset_loader.meshes);
    }

    // Update descriptors
    for (const auto& image : asset_loader.images)
    {
        image_manager.register_sampled_image(device, image.view, VK_IMAGE_LAYOUT_GENERAL);
    }

    // Create sampler
    std::vector<VkWriteDescriptorSet> writes{};
    std::vector<VkDescriptorImageInfo> desc_info{};
    VkSampler linear_samp = create_sampler(device, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT);
    desc_info.emplace_back(VkDescriptorImageInfo{ .sampler = linear_samp });
    writes.emplace_back(write_sampler_descriptor(sampler_set, 0, &desc_info[0]));
    vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);

    writes.clear();
    desc_info.clear();

    // Load GPU data
    Buffer vertex_buffer = create_buffer_with_data(device, graphics_queue, imm_fence, imm_pool, imm_buf, allocator, asset_loader.m_vertices.data(), asset_loader.m_vertices.size() * sizeof(Vertex));
    Buffer index_buffer = create_buffer_with_data(device, graphics_queue, imm_fence, imm_pool, imm_buf, allocator, asset_loader.m_indices.data(), asset_loader.m_indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT);
    Buffer object_buffer = create_buffer_with_data(device, graphics_queue, imm_fence, imm_pool, imm_buf, allocator, renderables.data(), renderables.size() * sizeof(ObjectData));
    Buffer material_buffer = create_buffer_with_data(device, graphics_queue, imm_fence, imm_pool, imm_buf, allocator, asset_loader.materials.data(), asset_loader.materials.size() * sizeof(MaterialData));
    Buffer mesh_buffer = create_buffer_with_data(device, graphics_queue, imm_fence, imm_pool, imm_buf, allocator, meshes.data(), meshes.size() * sizeof(MeshData));
    Buffer meshlet_buffer = create_buffer_with_data(device, graphics_queue, imm_fence, imm_pool, imm_buf, allocator, asset_loader.m_meshlets.data(), asset_loader.m_meshlets.size() * sizeof(Meshlet));
    Buffer meshlet_indices_buffer = create_buffer_with_data(device, graphics_queue, imm_fence, imm_pool, imm_buf, allocator, asset_loader.meshlet_indices.data(), asset_loader.meshlet_indices.size() * sizeof(uint32_t));

    // Create pipeline layout
    VkPushConstantRange pc_range{ .stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = max_push_constant_size };
    std::array<VkDescriptorSetLayout, 3> desc_set_layouts{ buffer_layout, image_layout, sampler_layout };
    VkPipelineLayoutCreateInfo pipeline_layout_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipeline_layout_info.pPushConstantRanges = &pc_range;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pSetLayouts = desc_set_layouts.data();
    pipeline_layout_info.setLayoutCount = desc_set_layouts.size();
    VK_CHECK(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout));

    VkPipeline cull_pipeline{};
    VkPipeline gbuffer_pipeline{};
    VkPipeline debug_pipeline{};

    std::vector<VkPipeline> pipelines{};

    auto build_pipelines = [&]()
    {
        auto cull_program = load_shader_program("culling.slang", device);
        auto gbuffer_program = load_shader_program("gbuffer.slang", device);
        auto debug_program = load_shader_program("debug.slang", device);

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
            gbuffer_pipeline,
            create_graphics_pipeline(
                device,
                &gbuffer_program,
                { VK_SHADER_STAGE_MESH_BIT_EXT, VK_SHADER_STAGE_FRAGMENT_BIT },
                nullptr,
                &pipeline_layout,
                { VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R32_UINT }
            )
        );
        replace_pipeline(cull_pipeline, create_compute_pipeline(device, &cull_program, nullptr, &pipeline_layout));
        replace_pipeline(debug_pipeline, create_compute_pipeline(device, &debug_program, nullptr, &pipeline_layout));
    };

    build_pipelines();

    std::array<uint64_t, MAX_TIMESTAMP_QUERIES> timestamp_results{};
    std::array<uint64_t, MAX_PIPELINE_QUERIES> pipeline_results{};

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
                    for (const auto& image : swapchain.images)
                    {
                        invalidate_imported_image(graph, image);
                    }
                    update_swapchain(swapchain, window, physical_device, device, surface);
                    proj = camera.set_perspective_matrix(glm::radians(camera.fov), static_cast<float>(swapchain.extent.width) / swapchain.extent.height, camera.near);
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
            proj = camera.set_perspective_matrix(glm::radians(camera.fov), static_cast<float>(swapchain.extent.width) / swapchain.extent.height, camera.near);
        }

        // Update camera
        camera.update(delta_time);
        auto view = camera.get_view_matrix();
        auto view_proj = proj * view;

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
        auto pipeline_results_size = pipeline_results.size();

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

            // VK_CHECK(vkGetQueryPoolResults(
            //     device,
            //     frame.query_pool_pipeline,
            //     0,
            //     pipeline_results_size,
            //     pipeline_results_size * sizeof(uint64_t),
            //     pipeline_results.data(),
            //     sizeof(uint64_t),
            //     VK_QUERY_RESULT_64_BIT
            // ));
        }

        double new_gpu_time = static_cast<double>(timestamp_results[1] - timestamp_results[0]) * timestamp_period * 1e-6;
        stats.gpu_time = new_gpu_time + 0.95 * (stats.gpu_time - new_gpu_time);
        auto triangles = pipeline_results[0];

        vkResetQueryPool(device, frame.query_pool_timestamp, 0, MAX_TIMESTAMP_QUERIES);
        vkResetQueryPool(device, frame.query_pool_pipeline, 0, MAX_PIPELINE_QUERIES);

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Stats");
        ImGui::Text("GPU time:    %.3f ms", stats.gpu_time);
        ImGui::Text("Triangles:   %u", static_cast<unsigned int>(triangles));
        ImGui::Text("Triangles:   %.1fM", static_cast<double>(triangles) * 1e-6);
        ImGui::SliderInt("Debug gbuffers", &GBUFFER_DEBUG_ID, 0, 4);

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
        auto gbuffer_color = graph.create_task_image(
            ImageResourceDesc{
                VkExtent3D{ swapchain.extent.width, swapchain.extent.height, 1 },
                VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT }
        );
        auto gbuffer_normal = graph.create_task_image(
            ImageResourceDesc{
                VkExtent3D{ swapchain.extent.width, swapchain.extent.height, 1 },
                VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT }
        );
        auto gbuffer_mr = graph.create_task_image(
            ImageResourceDesc{
                VkExtent3D{ swapchain.extent.width, swapchain.extent.height, 1 },
                VK_FORMAT_R8G8_UNORM,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT }
        );
        auto gbuffer_debug = graph.create_task_image(
            ImageResourceDesc{
                VkExtent3D{ swapchain.extent.width, swapchain.extent.height, 1 },
                VK_FORMAT_R32_UINT,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT }
        );

        auto draw_indirect_buffer = graph.create_task_buffer(BufferResourceDesc{ .alloc_size = asset_loader.m_meshlets.size() * sizeof(DrawIndirect), .usage = VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT });
        auto dispatch_buffer = graph.create_task_buffer(BufferResourceDesc{ .alloc_size = sizeof(Dispatch), .usage = VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT });

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

        graph.add_pass(
            "Zero buffers",
            [&](Pass& pass)
            {
                pass.write_buffer(dispatch_buffer, AccessType::TransferWrite);
            },
            [&]()
            {
                vkCmdFillBuffer(cmd, get_buffer(graph, dispatch_buffer), 0, get_buffer_size(graph, dispatch_buffer), 0);
            }
        );

        graph.add_pass(
            "Compute cull",
            [&](Pass& pass)
            {
                pass.write_buffer(dispatch_buffer, AccessType::ComputeShaderWrite);
                pass.write_buffer(draw_indirect_buffer, AccessType::ComputeShaderWrite);
            },
            [&]()
            {
                struct PushData
                {
                    glm::mat4 view;
                    VkDeviceAddress mesh_buffer;
                    VkDeviceAddress object_buffer;
                    VkDeviceAddress draw_indirect_buffer;
                    VkDeviceAddress dispatch_buffer;
                    glm::vec4 planes;
                    float p00;
                    float p11;
                    float near;
                    float far;
                    uint32_t count;
                    float lod_distance_factor;
                };

                auto proj_t = glm::transpose(proj);
                auto m0 = proj_t[0];
                auto m1 = proj_t[1];
                auto m3 = proj_t[3];
                auto left_plane = glm::normalize(glm::vec3(m3 + m0));
                auto bottom_plane = glm::normalize(glm::vec3(m3 + m1));

                PushData data{};
                data.view = view;
                data.mesh_buffer = mesh_buffer.address;
                data.object_buffer = object_buffer.address;
                data.draw_indirect_buffer = get_buffer_address(graph, draw_indirect_buffer);
                data.dispatch_buffer = get_buffer_address(graph, dispatch_buffer);
                data.planes = glm::vec4(left_plane.x, left_plane.z, bottom_plane.y, bottom_plane.z);
                data.p00 = proj[0][0];
                data.p11 = proj[1][1];
                data.near = camera.near;
                data.far = camera.far;
                data.count = renderables.size();
                data.lod_distance_factor = 2.0f / (data.p11 * swapchain.extent.height);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cull_pipeline);
                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                vkCmdDispatch(cmd, get_group_count(renderables.size(), 256), 1, 1);
            }
        );

        graph.add_pass(
            "Gbuffers",
            [&](Pass& pass)
            {
                pass.read_buffer(dispatch_buffer, AccessType::IndirectBuffer);
                pass.read_buffer(draw_indirect_buffer, AccessType::IndirectBuffer);
                pass.write_image(gbuffer_color, AccessType::ColorAttachmentWrite);
                pass.write_image(gbuffer_normal, AccessType::ColorAttachmentWrite);
                pass.write_image(gbuffer_mr, AccessType::ColorAttachmentWrite);
                pass.write_image(gbuffer_debug, AccessType::ColorAttachmentWrite);
            },
            [&]()
            {
                VkClearColorValue clear_color_value = { 1.f, 0.f, 0.f, 1.f };
                VkClearValue clear_value{ .color = clear_color_value };

                std::array<VkRenderingAttachmentInfo, 4> rendering_attachment_infos{};
                std::array<VkImageView, 4> gbuffer_views{
                    get_image_view(graph, gbuffer_color),
                    get_image_view(graph, gbuffer_normal),
                    get_image_view(graph, gbuffer_mr),
                    get_image_view(graph, gbuffer_debug)
                };

                for (auto i = 0; i < rendering_attachment_infos.size(); i++)
                {
                    rendering_attachment_infos[i] =
                        VkRenderingAttachmentInfo{
                            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                            .imageView = gbuffer_views[i],
                            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                            .storeOp = VK_ATTACHMENT_STORE_OP_STORE
                        };
                }

                VkRenderingAttachmentInfo depth_attachment_info{ .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
                depth_attachment_info.imageView = get_image_view(graph, depth_image);
                depth_attachment_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                depth_attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                depth_attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

                VkRenderingInfo rendering_info{ .sType = VK_STRUCTURE_TYPE_RENDERING_INFO };
                rendering_info.renderArea = VkRect2D{ VkOffset2D{ 0, 0 }, swapchain.extent };
                rendering_info.layerCount = 1;
                rendering_info.colorAttachmentCount = static_cast<uint32_t>(rendering_attachment_infos.size());
                rendering_info.pColorAttachments = rendering_attachment_infos.data();
                rendering_info.pDepthAttachment = &depth_attachment_info;

                vkCmdBeginRendering(cmd, &rendering_info);

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

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gbuffer_pipeline);

                struct PushData
                {
                    glm::mat4 view_proj;
                    VkDeviceAddress vertex_buffer;
                    VkDeviceAddress object_buffer;
                    VkDeviceAddress material_buffer;
                    VkDeviceAddress meshlet_buffer;
                    VkDeviceAddress draw_indirect_buffer;
                    VkDeviceAddress meshlet_indices_buffer;
                };

                PushData data{};
                data.view_proj = view_proj;
                data.vertex_buffer = vertex_buffer.address;
                data.object_buffer = object_buffer.address;
                data.material_buffer = material_buffer.address;
                data.meshlet_buffer = meshlet_buffer.address;
                data.draw_indirect_buffer = get_buffer_address(graph, draw_indirect_buffer);
                data.meshlet_indices_buffer = meshlet_indices_buffer.address;

                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(data), &data);
                vkCmdDrawMeshTasksIndirectEXT(cmd, get_buffer(graph, dispatch_buffer), 0, 1, 0);
                vkCmdEndRendering(cmd);
            }
        );

        graph.add_pass(
            "Debug gbuffers",
            [&](Pass& pass)
            {
                pass.read_image(gbuffer_color, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer);
                pass.read_image(gbuffer_normal, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer);
                pass.read_image(gbuffer_mr, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer);
                pass.read_image(gbuffer_debug, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer);
                pass.write_image(draw_image, AccessType::ComputeShaderWrite);
            },
            [&]()
            {
                struct PushData
                {
                    glm::uvec2 extent;
                    uint32_t debug_id;
                    uint32_t color_id;
                    uint32_t normal_id;
                    uint32_t metal_roughness_id;
                    uint32_t draw_id;
                    uint32_t debug_tex_id;
                };

                PushData data{};
                data.extent = glm::uvec2(swapchain.extent.width, swapchain.extent.height);
                data.debug_id = GBUFFER_DEBUG_ID;
                data.color_id = get_image_id(graph, gbuffer_color);
                data.normal_id = get_image_id(graph, gbuffer_normal);
                data.metal_roughness_id = get_image_id(graph, gbuffer_mr);
                data.draw_id = get_image_id(graph, draw_image);
                data.debug_tex_id = get_image_id(graph, gbuffer_debug);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, debug_pipeline);
                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                vkCmdDispatch(cmd, get_group_count(swapchain.extent.width, 8), get_group_count(swapchain.extent.height, 8), 1);
            }
        );

        graph.add_pass(
            "Copy to swapchain",
            [&](Pass& pass)
            {
                pass.read_image(draw_image, AccessType::TransferRead);
                pass.write_image(swapchain_image, AccessType::TransferWrite);
            },
            [&]()
            {
                copy_image(cmd, get_image(graph, draw_image), get_image(graph, swapchain_image), swapchain.extent, swapchain.extent);
            }
        );

        graph.add_pass(
            "ImGui",
            [&](Pass& pass)
            {
                pass.write_image(swapchain_image, AccessType::ColorAttachmentReadWrite);
            },
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
    destroy_buffer(allocator, index_buffer);
    destroy_buffer(allocator, object_buffer);
    destroy_buffer(allocator, material_buffer);
    destroy_buffer(allocator, mesh_buffer);
    destroy_buffer(allocator, meshlet_buffer);
    destroy_buffer(allocator, meshlet_indices_buffer);
    vkDestroySampler(device, linear_samp, nullptr);

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
    uint32_t image_count = 300;
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
