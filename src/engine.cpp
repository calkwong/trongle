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
#include "utils.h"
#include "device.h"

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
#include <glm/ext/vector_uint3.hpp>
#include <glm/vector_relational.hpp>
#include <glm/ext/matrix_transform.hpp>
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
#include <memory>

#if STRESS_TEST == 1 || LOCAL_LIGHTS == 1
#include <random>
#endif

#if STRESS_TEST == 1
bool b_shadows = false;
#else
bool b_shadows = true;
#endif

// Fog settings
float scattering_factor = 0.1;
float noise_pos_mult = 0.0;
float noise_speed_mult = 0.0;
float density_modifier = 0.2;
float height_fog_density = 0.0;
float height_fog_falloff = 0.5;
float box_fog_density = 3.0;
float noise_scale = 0.5;
float temporal_reprojection_jitter_scale = 0.2;
bool b_fog_spatial_filter = true;
bool b_fog_temporal_filter = true;

const char* ASSET_NAME = "Sponza/Sponza.gltf";
// const char* ASSET_NAME = "chinesedragon.gltf";
// const char* ASSET_NAME = "sphere.gltf";
// const char* ASSET_NAME = "BistroInterior_Wine/BistroInterior_Wine.gltf";
int DEBUG_ID = 0;

void Engine::init_vulkan()
{
    VK_CHECK(volkInitialize());

    auto vkb_inst = create_instance();
    instance = vkb_inst.instance;
    debug_messenger = vkb_inst.debug_messenger;
    volkLoadInstance(instance);

    auto window_flags = static_cast<SDL_WindowFlags>(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    auto window_width = 1920u;
    auto window_height = 1080u;

    window = create_window(instance, window_width, window_height, window_flags);
    create_surface(window, instance, &surface);
    SDL_SetWindowRelativeMouseMode(window, relative_mouse_mode);

    auto vkb_physical_device = create_physical_device(vkb_inst, surface);
    physical_device = vkb_physical_device.value();
    auto vkb_device = create_device(vkb_physical_device);
    device = vkb_device.device;
    volkLoadDevice(device);
    get_graphics_queue_and_family(vkb_device, graphics_queue, graphics_queue_family);

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

void Engine::run()
{
    // Init rendergraph
    graph = std::make_unique<Rendergraph>(device, allocator, &image_manager);

    VkPhysicalDeviceMeshShaderPropertiesEXT mesh_shader_properties{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT };
    VkPhysicalDeviceProperties2 properties2{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
    properties2.pNext = &mesh_shader_properties;

    vkGetPhysicalDeviceProperties2(physical_device, &properties2);
    assert(properties2.properties.limits.timestampComputeAndGraphics);
    auto max_push_constant_size = properties2.properties.limits.maxPushConstantsSize;
    auto timestamp_period = properties2.properties.limits.timestampPeriod;
    // fmt::println("maxMeshWorkGroupSize {} {} {}", mesh_shader_properties.maxMeshWorkGroupSize[0], mesh_shader_properties.maxMeshWorkGroupSize[1], mesh_shader_properties.maxMeshWorkGroupSize[2]);
    // fmt::println("maxMeshWorkGroupInvocations {}",  mesh_shader_properties.maxMeshWorkGroupInvocations);
    // fmt::println("maxMeshWorkGroupCount {} {} {}", mesh_shader_properties.maxMeshWorkGroupCount[0], mesh_shader_properties.maxMeshWorkGroupCount[1], mesh_shader_properties.maxMeshWorkGroupCount[2]);
    // fmt::println("maxMeshWorkGroupTotalCount {}", mesh_shader_properties.maxMeshWorkGroupTotalCount);

    // Init camera
    Camera camera{};
    camera.position = glm::vec3(0, 0, 5);
    camera.far = 1000.0f;
    camera.near = 0.01f;

    frame_context.proj = camera.set_perspective_matrix(glm::radians(camera.fov), static_cast<float>(swapchain.extent.width) / swapchain.extent.height, camera.near);

    AssetLoader asset_loader{ .device = device, .allocator = allocator };
    bool loaded = asset_loader.load_gltf(graphics_queue, imm_fence, imm_pool, imm_buf, ASSET_NAME, static_cast<uint32_t>(image_manager.infos.size()));
    if (!loaded)
    {
        assert(0 && "load_gltf failed");
    }

    // Register material textures
    for (const auto& image : asset_loader.images)
    {
        image_manager.register_sampled_image(device, image.view, VK_IMAGE_LAYOUT_GENERAL);
    }

#if STRESS_TEST == 0
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

    // Create sampler
    std::vector<VkWriteDescriptorSet> writes{};
    std::vector<VkDescriptorImageInfo> desc_info{};
    std::vector<VkSampler> samplers{};
    samplers.emplace_back(create_sampler(device, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT));
    samplers.emplace_back(create_depth_reduce_sampler(device));
    samplers.emplace_back(create_sampler(device, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE));
    samplers.emplace_back(create_sampler(device, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, true));
    samplers.emplace_back(create_sampler(device, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE));

    for (const auto& samp : samplers)
    {
        desc_info.emplace_back(VkDescriptorImageInfo{ .sampler = samp });
    }

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
    Buffer froxels_buffer = create_buffer(device, allocator, sizeof(FroxelAABB) * FROXELS_X * FROXELS_Y * FROXELS_Z, 0, VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT);
    Buffer global_buffer = create_buffer(device, allocator, sizeof(GlobalData) * FRAMES_IN_FLIGHT, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT);
    Buffer sh_buffer = create_buffer(device, allocator, sizeof(float) * 27, 0, VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT);

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
    const auto depth_pyramid_descriptor_handle = image_manager.register_image(device, depth_pyramid_image.view, VK_IMAGE_LAYOUT_GENERAL);

    // Returns depth pyramid mip levels for next PoT
    auto get_depth_pyramid_image_count = [&](const Image& image) -> uint32_t
    {
        const auto& depth_pyramid_extent = image.extent;
        return get_image_mip_level(depth_pyramid_extent.width, depth_pyramid_extent.height);
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
    VkPipeline vbuffer_masked_pipeline{};
    VkPipeline lighting_pipeline{};
    VkPipeline shadow_matrices_pipeline{};
    VkPipeline cull_shadow_pipeline{};
    VkPipeline cull_shadow_meshlet_pipeline{};
    VkPipeline depth_pipeline{};
    VkPipeline depth_masked_pipeline{};
    VkPipeline exr_to_cubemap_pipeline{};
    VkPipeline envmap_pipeline{};
    VkPipeline froxels_pipeline{};
    VkPipeline binning_pipeline{};
    VkPipeline brdf_lut_pipeline{};
    VkPipeline sh_pipeline{};
    VkPipeline reset_buffers_pipeline{};
    VkPipeline data_injection_pipeline{};
    VkPipeline light_scattering_pipeline{};
    VkPipeline light_integration_pipeline{};
    VkPipeline volumetric_noise_pipeline{};
    VkPipeline fog_spatial_pipeline{};
    VkPipeline fog_temporal_pipeline{};

    std::vector<VkPipeline> pipelines{};

    auto build_pipelines = [&]()
    {
        auto cull_program = load_shader_program("cull_mesh.slang", device);
        auto cull_meshlets_program = load_shader_program("cull_meshlet.slang", device);
        auto compact_dispatch_program = load_shader_program("compact_dispatch.slang", device);
        auto hiz_program = load_shader_program("hiz_spd.slang", device);
        auto vbuffer_program = load_shader_program("vbuffer.slang", device);
        auto vbuffer_masked_program = load_shader_program("vbuffer.slang", device);
        auto lighting_program = load_shader_program("resolve_vbuffer.slang", device);
        auto shadow_matrices_program = load_shader_program("shadow_matrices.slang", device);
        auto cull_shadow_program = load_shader_program("cull_shadow.slang", device);
        auto cull_shadow_meshlet_program = load_shader_program("cull_shadow_meshlet.slang", device);
        auto depth_program = load_shader_program("depth.slang", device);
        auto depth_masked_program = load_shader_program("depth.slang", device);
        auto exr_to_cubemap_program = load_shader_program("exr_to_cubemap.slang", device);
        auto envmap_program = load_shader_program("envmap.slang", device);
        auto froxels_program = load_shader_program("froxels.slang", device);
        auto binning_program = load_shader_program("binning.slang", device);
        auto brdf_lut_program = load_shader_program("brdf_lut.slang", device);
        auto sh_program = load_shader_program("sh.slang", device);
        auto reset_buffers_program = load_shader_program("reset_buffers.slang", device);
        auto data_injection_program = load_shader_program("data_injection.slang", device);
        auto light_scattering_program = load_shader_program("light_scattering.slang", device);
        auto light_integration_program = load_shader_program("light_integration.slang", device);
        auto volumetric_noise_program = load_shader_program("volumetric_noise.slang", device);
        auto fog_spatial_program = load_shader_program("fog_spatial_filter.slang", device);
        auto fog_temporal_program = load_shader_program("fog_temporal_filter.slang", device);

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
                { VK_FORMAT_R32_UINT },
                { 1 }
            )
        );

        replace_pipeline(
            vbuffer_masked_pipeline,
            create_graphics_pipeline(
                device,
                &vbuffer_masked_program,
                { VK_SHADER_STAGE_MESH_BIT_EXT, VK_SHADER_STAGE_FRAGMENT_BIT },
                nullptr,
                &pipeline_layout,
                { VK_FORMAT_R32_UINT },
                { 0 },
                GraphicsPipelineConfig{ .cull_mode = VK_CULL_MODE_NONE }
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
                { 1 },
                GraphicsPipelineConfig{ .cull_mode = VK_CULL_MODE_FRONT_BIT, .depth_clamp = VK_TRUE }
            )
        );

        replace_pipeline(
            depth_masked_pipeline,
            create_graphics_pipeline(
                device,
                &depth_masked_program,
                { VK_SHADER_STAGE_MESH_BIT_EXT, VK_SHADER_STAGE_FRAGMENT_BIT },
                nullptr,
                &pipeline_layout,
                {},
                { 0 },
                GraphicsPipelineConfig{ .cull_mode = VK_CULL_MODE_NONE, .depth_clamp = VK_TRUE }
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
        replace_pipeline(exr_to_cubemap_pipeline, create_compute_pipeline(device, &exr_to_cubemap_program, nullptr, &pipeline_layout));
        replace_pipeline(envmap_pipeline, create_compute_pipeline(device, &envmap_program, nullptr, &pipeline_layout));
        replace_pipeline(froxels_pipeline, create_compute_pipeline(device, &froxels_program, nullptr, &pipeline_layout));
        replace_pipeline(binning_pipeline, create_compute_pipeline(device, &binning_program, nullptr, &pipeline_layout));
        replace_pipeline(brdf_lut_pipeline, create_compute_pipeline(device, &brdf_lut_program, nullptr, &pipeline_layout));
        replace_pipeline(sh_pipeline, create_compute_pipeline(device, &sh_program, nullptr, &pipeline_layout));
        replace_pipeline(reset_buffers_pipeline, create_compute_pipeline(device, &reset_buffers_program, nullptr, &pipeline_layout));
        replace_pipeline(data_injection_pipeline, create_compute_pipeline(device, &data_injection_program, nullptr, &pipeline_layout));
        replace_pipeline(light_scattering_pipeline, create_compute_pipeline(device, &light_scattering_program, nullptr, &pipeline_layout));
        replace_pipeline(light_integration_pipeline, create_compute_pipeline(device, &light_integration_program, nullptr, &pipeline_layout));
        replace_pipeline(volumetric_noise_pipeline, create_compute_pipeline(device, &volumetric_noise_program, nullptr, &pipeline_layout));
        replace_pipeline(fog_spatial_pipeline, create_compute_pipeline(device, &fog_spatial_program, nullptr, &pipeline_layout));
        replace_pipeline(fog_temporal_pipeline, create_compute_pipeline(device, &fog_temporal_program, nullptr, &pipeline_layout));
    };

    build_pipelines();

#if LOCAL_LIGHTS == 1
    std::mt19937 mt(42);
    std::uniform_real_distribution<float> pos(-1.0f, 1.0f);
    std::uniform_real_distribution<float> color(0.f, 1.0f);

    std::vector<PointLight> light_data(MAX_POINT_LIGHTS);

    float light_area = 10.f; // in radius
    float light_radius = 1.f;

    for (size_t i = 0; i < 1000; i++)
    {
        light_data[i].pos = glm::vec3(pos(mt) * light_area, std::abs(pos(mt) * light_area), pos(mt) * light_area);
        light_data[i].radius = light_radius;
        glm::vec3 light_color = glm::vec3(color(mt), color(mt), color(mt));
        light_data[i].color = glm::vec4(light_color, 1.0);
    }

    Buffer light_buffer = create_buffer_with_data(device, graphics_queue, imm_fence, imm_pool, imm_buf, allocator, light_data.data(), MAX_POINT_LIGHTS * sizeof(PointLight));
#endif

    Image brdf_lut_image = create_image(device, allocator, VkExtent3D{ 128, 128, 1 }, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_USAGE_2_SAMPLED_BIT_KHR | VK_IMAGE_USAGE_2_STORAGE_BIT_KHR);
    auto brdf_lut_id = image_manager.register_image(device, brdf_lut_image.view, VK_IMAGE_LAYOUT_GENERAL);

    std::vector<VkImage> images_to_transition{};
    images_to_transition.push_back(brdf_lut_image.image);
    images_to_transition.push_back(depth_pyramid_image.image);

#if VOLUMETRIC_FOG == 1
    assert(b_shadows);
    Image volumetric_noise_image = create_3d_image(device, allocator, VkExtent3D{ 64, 64, 64 }, VK_FORMAT_R16_SFLOAT, VK_IMAGE_USAGE_2_SAMPLED_BIT_KHR | VK_IMAGE_USAGE_2_STORAGE_BIT_KHR);
    auto volumetric_noise_id = image_manager.register_image(device, volumetric_noise_image.view, VK_IMAGE_LAYOUT_GENERAL);
    images_to_transition.push_back(volumetric_noise_image.image);
#endif

    std::vector<VkImage> depth_images{};
    for (const auto& image : shadowmap_images)
    {
        depth_images.push_back(image.image);
    }

    // TODO: If converting to rendergraph, need to see if Profiler implementation will cause errors as we don't care for profiling here
    immediate_submit(device, graphics_queue, imm_fence, imm_pool, imm_buf, [&](VkCommandBuffer cmd)
                     {
                         std::array<VkDescriptorSet, 3> sets{ buffer_set, image_set, sampler_set };
                         vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, sets.size(), sets.data(), 0, nullptr);

                         vkCmdFillBuffer(cmd, spd_counter_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
                         vkCmdFillBuffer(cmd, late_pass_candidates_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
                         transition_images(cmd, images_to_transition, depth_images);

                         {
                             struct PushData
                             {
                                 glm::vec2 image_size;
                                 uint32_t brdf_lut_id;
                             };

                             PushData data{};
                             data.image_size = glm::vec2(brdf_lut_image.extent.width, brdf_lut_image.extent.height);
                             data.brdf_lut_id = brdf_lut_id;

                             vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, brdf_lut_pipeline);
                             vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                             auto wg_x = get_group_count(brdf_lut_image.extent.width, 32);
                             auto wg_y = get_group_count(brdf_lut_image.extent.height, 32);
                             vkCmdDispatch(cmd, wg_x, wg_y, 1);
                         }

                         {
                             struct PushData
                             {
                                 float4x4 inverse_proj;
                                 VkDeviceAddress froxels_buffer;
                                 glm::vec2 screen_size;
                                 glm::vec2 froxel_dim;
                                 float near_z;
                                 float far_z;
                                 uint32_t depth_slices;
                             };

                             PushData data{};
                             data.inverse_proj = glm::inverse(frame_context.proj);
                             data.froxels_buffer = froxels_buffer.address;
                             data.screen_size = glm::vec2(swapchain.extent.width, swapchain.extent.height);
                             auto froxel_x = glm::ceil(static_cast<float>(swapchain.extent.width) / FROXELS_X);
                             auto froxel_y = glm::ceil(static_cast<float>(swapchain.extent.height) / FROXELS_Y);
                             data.froxel_dim = glm::vec2(froxel_x, froxel_y);
                             // TODO: We can inflate near plane otherwise we get lots of intersections close to near plane
                             data.near_z = camera.near;
                             data.far_z = FROXELS_FAR_PLANE;
                             data.depth_slices = FROXELS_Z;

                             vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, froxels_pipeline);
                             vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                             vkCmdDispatch(cmd, 1, 1, FROXELS_Z);
                         }

#if VOLUMETRIC_FOG == 1
                         {
                             struct PushData
                             {
                                 uint32_t noise_id;
                             };

                             PushData data{};
                             data.noise_id = volumetric_noise_id;

                             vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, volumetric_noise_pipeline);
                             vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                             auto group_count_x = get_group_count(64, 8);
                             auto group_count_y = get_group_count(64, 8);
                             vkCmdDispatch(cmd, group_count_x, group_count_y, 64);
                         }
#endif
                     });

    float shadow_draw_distance = SHADOW_DRAW_DISTANCE;
    bool skip_occlusion = true;
    float azimuth = 0.0f;
    float elevation = 90.0f;
    // TODO: imgui handles turning off/on temporal filtering; turning off/on volumetric fog still needs special handling
    bool fog_temporal_first_frame = true;

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
                        graph->invalidate_imported_image(image);
                    }
                    update_swapchain(swapchain, window, physical_device, device, surface);

                    destroy_image(device, allocator, depth_pyramid_image);
                    depth_pyramid_extent.width = nearest_pow2(swapchain.extent.width);
                    depth_pyramid_extent.height = nearest_pow2(swapchain.extent.height);
                    depth_pyramid_image = create_image(device, allocator, depth_pyramid_extent, VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, true);
                    image_manager.update(device, depth_pyramid_descriptor_handle, depth_pyramid_image.view, VK_IMAGE_LAYOUT_GENERAL);
                    update_depth_pyramid(depth_pyramid_image, depth_pyramid_image_views, depth_pyramid_first_mip);

                    skip_occlusion = true;
                    frame_context.proj = camera.set_perspective_matrix(glm::radians(camera.fov), static_cast<float>(swapchain.extent.width) / swapchain.extent.height, camera.near);

                    immediate_submit(device, graphics_queue, imm_fence, imm_pool, imm_buf, [&](VkCommandBuffer cmd)
                                     {
                                         stage_barrier(cmd, depth_pyramid_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, 0, 0, 0);

                                         std::array<VkDescriptorSet, 3> sets{ buffer_set, image_set, sampler_set };
                                         vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, sets.size(), sets.data(), 0, nullptr);

                                         {
                                             struct FroxelsPushConstants
                                             {
                                                 float4x4 inverse_proj;
                                                 VkDeviceAddress froxels_buffer;
                                                 glm::vec2 screen_size;
                                                 glm::vec2 froxel_dim;
                                                 float near_z;
                                                 float far_z;
                                                 uint32_t depth_slices;
                                             };

                                             FroxelsPushConstants data{};
                                             data.inverse_proj = glm::inverse(frame_context.proj);
                                             data.froxels_buffer = froxels_buffer.address;
                                             data.screen_size = glm::vec2(swapchain.extent.width, swapchain.extent.height);
                                             auto froxel_x = glm::ceil(static_cast<float>(swapchain.extent.width) / FROXELS_X);
                                             auto froxel_y = glm::ceil(static_cast<float>(swapchain.extent.height) / FROXELS_Y);
                                             data.froxel_dim = glm::vec2(froxel_x, froxel_y);
                                             // TODO: We can inflate near plane otherwise we get lots of intersections close to near plane
                                             data.near_z = camera.near;
                                             data.far_z = FROXELS_FAR_PLANE;
                                             data.depth_slices = FROXELS_Z;

                                             vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, froxels_pipeline);
                                             vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(FroxelsPushConstants), &data);
                                             vkCmdDispatch(cmd, 1, 1, FROXELS_Z);
                                         }
                                     });
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
                graph->invalidate_imported_image(image);
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
            frame_context.proj = camera.set_perspective_matrix(glm::radians(camera.fov), static_cast<float>(swapchain.extent.width) / swapchain.extent.height, camera.near);

            immediate_submit(device, graphics_queue, imm_fence, imm_pool, imm_buf, [&](VkCommandBuffer cmd)
                             {
                                 stage_barrier(cmd, depth_pyramid_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, 0, 0, 0);

                                 std::array<VkDescriptorSet, 3> sets{ buffer_set, image_set, sampler_set };
                                 vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, sets.size(), sets.data(), 0, nullptr);

                                 {
                                     struct FroxelsPushConstants
                                     {
                                         float4x4 inverse_proj;
                                         VkDeviceAddress froxels_buffer;
                                         glm::vec2 screen_size;
                                         glm::vec2 froxel_dim;
                                         float near_z;
                                         float far_z;
                                         uint32_t depth_slices;
                                     };

                                     FroxelsPushConstants data{};
                                     data.inverse_proj = glm::inverse(frame_context.proj);
                                     data.froxels_buffer = froxels_buffer.address;
                                     data.screen_size = glm::vec2(swapchain.extent.width, swapchain.extent.height);
                                     auto froxel_x = glm::ceil(static_cast<float>(swapchain.extent.width) / FROXELS_X);
                                     auto froxel_y = glm::ceil(static_cast<float>(swapchain.extent.height) / FROXELS_Y);
                                     data.froxel_dim = glm::vec2(froxel_x, froxel_y);
                                     // TODO: We can inflate near plane otherwise we get lots of intersections close to near plane
                                     data.near_z = camera.near;
                                     data.far_z = FROXELS_FAR_PLANE;
                                     data.depth_slices = FROXELS_Z;

                                     vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, froxels_pipeline);
                                     vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(FroxelsPushConstants), &data);
                                     vkCmdDispatch(cmd, 1, 1, FROXELS_Z);
                                 }
                             });
        }

        camera.update(delta_time);
        if (!freeze_matrices)
        {
            frame_context.prev_view = frame_context.view;
            frame_context.prev_view_proj = frame_context.view_proj;
            frame_context.view = camera.get_view_matrix();
            frame_context.view_proj = frame_context.proj * frame_context.view;
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
        auto frame_index = frame_number % FRAMES_IN_FLIGHT;
        auto ping_pong_index = frame_number % 2;
        auto next_ping_pong_index = (frame_number + 1) % 2;
        VK_CHECK(vkWaitForFences(device, 1, &frame.fence, true, WAIT_TIME));

        // Acquire next image
        uint32_t swapchain_image_idx{};
        VkResult acquire_result = vkAcquireNextImageKHR(device, swapchain.swapchain, WAIT_TIME, frame.image_acquired_semaphore, nullptr, &swapchain_image_idx);
        VK_CHECK_SWAPCHAIN(acquire_result);
        if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            swapchain_dirty = true;
            continue;
        }
        VK_CHECK(vkResetFences(device, 1, &frame.fence));

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        if (frame_number >= FRAMES_IN_FLIGHT)
        {
            graph->gpu_profiler.get_timestamp_results(device, frame_index, timestamp_period);
            graph->gpu_profiler.draw(frame_index);
        }
        graph->gpu_profiler.reset(device, frame_index);

        ImGui::Begin("Settings");
        ImGui::Checkbox("Freeze camera", &freeze_matrices);
        ImGui::Checkbox("Shadows", &b_shadows);
        ImGui::SliderFloat("Shadow draw distance", &shadow_draw_distance, SHADOW_DRAW_DISTANCE, 400.0);
        ImGui::SliderFloat("scattering_factor", &scattering_factor, 0.0, 1.0);
        ImGui::SliderFloat("noise_pos_mult", &noise_pos_mult, 0.0, 1.0);
        ImGui::SliderFloat("noise_speed_mult", &noise_speed_mult, 0.0, 1.0);
        ImGui::SliderFloat("density_modifier", &density_modifier, 0.0, 1.0);
        ImGui::SliderFloat("height_fog_density", &height_fog_density, 0.0, 10.0);
        ImGui::SliderFloat("height_fog_falloff", &height_fog_falloff, 0.0, 10.0);
        ImGui::SliderFloat("box_fog_density", &box_fog_density, 0.0, 10.0);
        ImGui::SliderFloat("noise_scale", &noise_scale, 0.0, 1.0);
        ImGui::SliderFloat("temporal_reprojection_jitter_scale", &temporal_reprojection_jitter_scale, 0.0, 1.0);
        ImGui::Checkbox("Fog spatial filter", &b_fog_spatial_filter);
        // This does not appear to trigger on setup, so it's safe
        fog_temporal_first_frame = ImGui::Checkbox("Fog temporal filter", &b_fog_temporal_filter);
        ImGui::SliderFloat("Azimuth", &azimuth, 0.0, 360.0);
        ImGui::SliderFloat("Elevation", &elevation, 0.0, 90.0);
        ImGui::SliderInt("Debug channels", &DEBUG_ID, 0, MAX_DEBUG_ID);
        ImGui::Text("0:  Default");
        ImGui::Text("1:  Albedo");
        ImGui::Text("2:  Shading normal");
        ImGui::Text("3:  Tangent");
        ImGui::Text("4:  Tangent sign");
        ImGui::Text("5:  Metallic");
        ImGui::Text("6:  Roughness");
        ImGui::Text("7:  AO");
        ImGui::Text("8:  UV");
        ImGui::Text("9:  Shadow cascades");
        ImGui::Text("10: Triangles");
        ImGui::Text("11: Meshlets");
        ImGui::Text("12: Radiance");
        ImGui::Text("13: Irradiance");

        ImGui::End();
        ImGui::Render();

        auto swapchain_image = graph->import_swapchain(swapchain.images[swapchain_image_idx], swapchain.image_views[swapchain_image_idx]);

        auto depth_image = graph->create_task_image(
            ImageResourceDesc{
                VkExtent3D{ swapchain.extent.width, swapchain.extent.height, 1 },
                VK_FORMAT_D32_SFLOAT,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                VK_IMAGE_ASPECT_DEPTH_BIT }
        );
        auto draw_image = graph->create_task_image(
            ImageResourceDesc{
                VkExtent3D{ swapchain.extent.width, swapchain.extent.height, 1 },
                VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT }
        );
        auto vbuffer = graph->create_task_image(
            ImageResourceDesc{
                VkExtent3D{ swapchain.extent.width, swapchain.extent.height, 1 },
                VK_FORMAT_R32_UINT,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT }
        );
        auto fog_froxel_data = graph->create_task_image(
            ImageResourceDesc{
                VkExtent3D{ FOG_WIDTH, FOG_HEIGHT, FOG_DEPTH },
                VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_TYPE_3D }
        );
        auto raw_light_scattering = graph->create_task_image(
            ImageResourceDesc{
                VkExtent3D{ FOG_WIDTH, FOG_HEIGHT, FOG_DEPTH },
                VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_TYPE_3D }
        );
        std::array<uint32_t, 2> temporal_filtered_scattering{};
        for (auto& id : temporal_filtered_scattering)
        {
            id = graph->create_task_image(
                ImageResourceDesc{
                    VkExtent3D{ FOG_WIDTH, FOG_HEIGHT, FOG_DEPTH },
                    VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_TYPE_3D }
            );
        }

        std::array<uint32_t, SUN_CASCADE_COUNT> shadowmaps{};
        for (uint32_t i = 0; i < SUN_CASCADE_COUNT; i++)
        {
            shadowmaps[i] = graph->import_image(shadowmap_images[i], shadowmap_descriptor_handles[i]);
        }

        auto dispatch_buffer = graph->create_task_buffer(BufferResourceDesc{ .alloc_size = sizeof(Dispatch) * 5, .usage = VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT });
        auto prefix_sum_buffer = graph->create_task_buffer(BufferResourceDesc{ .alloc_size = sizeof(PrefixSums), .usage = VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT });
        auto meshlet_instance_buffer = graph->create_task_buffer(BufferResourceDesc{ .alloc_size = sizeof(MeshletInstance) * MAX_MESHLETS_DRAW, .usage = VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT });
        auto shadow_meshlet_instance_buffer = graph->create_task_buffer(BufferResourceDesc{ .alloc_size = sizeof(MeshletInstance) * MAX_MESHLETS_DRAW, .usage = VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT });
        auto froxel_offset_buffer = graph->create_task_buffer(BufferResourceDesc{ .alloc_size = sizeof(FroxelOffset) * TOTAL_FROXELS, .usage = VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT });
        auto froxel_item_buffer = graph->create_task_buffer(BufferResourceDesc{ .alloc_size = sizeof(uint32_t) * MAX_LIGHTS_PER_FROXEL * TOTAL_FROXELS, .usage = VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT });
        auto light_count_buffer = graph->create_task_buffer(BufferResourceDesc{ .alloc_size = sizeof(uint32_t), .usage = VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT });

        auto depth_pyramid = graph->import_image(depth_pyramid_image, depth_pyramid_descriptor_handle);
        auto spd_counter = graph->import_buffer(spd_counter_buffer, -1);
        auto late_pass_candidates = graph->import_buffer(late_pass_candidates_buffer, -1);
        auto globals = graph->import_buffer(global_buffer, -1);

        // Record command buffer
        VkCommandBuffer cmd = frame.command_buffer;
        VK_CHECK(vkResetCommandPool(device, frame.command_pool, 0));

        VkCommandBufferBeginInfo cmd_begin_info{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        cmd_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_begin_info));

        auto* global_data = static_cast<GlobalData*>(global_buffer.info.pMappedData);
        GlobalData data{};
        data.view = frame_context.view;
        data.view_proj = frame_context.view_proj;
        data.inverse_view_proj = glm::inverse(data.view_proj);
        data.previous_view = frame_context.prev_view;
        data.previous_view_proj = frame_context.prev_view_proj;
        global_data[frame_index] = data;

        auto jitter_index = frame_number % 8;
        float halton_x = 2.0f * Halton(jitter_index + 1, 2) - 1.0f;
        float halton_y = 2.0f * Halton(jitter_index + 1, 3) - 1.0f;

        std::array<VkDescriptorSet, 3> sets{ buffer_set, image_set, sampler_set };

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, sets.size(), sets.data(), 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, sets.size(), sets.data(), 0, nullptr);

        if (b_shadows)
        {
            graph->add_pass(
                Pass("Reset buffers for shadow pass")
                    .write_buffer(prefix_sum_buffer, AccessType::TransferWrite),
                [&]()
                {
                    struct PushData
                    {
                        VkDeviceAddress prefix_sum_buffer;
                        VkDeviceAddress light_count_buffer;
                        uint32_t zero_lights;
                    };

                    PushData data{};
                    data.prefix_sum_buffer = graph->buffer_address(prefix_sum_buffer);
                    data.light_count_buffer = graph->buffer_address(light_count_buffer);
                    data.zero_lights = 0;

                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, reset_buffers_pipeline);
                    vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                    vkCmdDispatch(cmd, 1, 1, 1);
                }
            );

            graph->add_pass(
                Pass("Generate shadow matrices")
                    .write_buffer(globals, AccessType::ComputeShaderWrite),
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
                    data.global_buffer = graph->buffer_address(globals, sizeof(GlobalData) * (frame_index));
                    auto proj = glm::perspective(glm::radians(camera.fov), static_cast<float>(swapchain.extent.width) / static_cast<float>(swapchain.extent.height), shadow_draw_distance, camera.near);
                    data.inverse_view_proj = glm::inverse(proj * frame_context.view);
                    data.sunlight_dir = sunlight_dir;
                    data.shadow_near = camera.near;
                    data.shadow_far = shadow_draw_distance;
                    data.lambda = 0.95;

                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shadow_matrices_pipeline);
                    vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                    vkCmdDispatch(cmd, 1, 1, 1);
                }
            );

            graph->add_pass(
                Pass("Cull shadow instances")
                    .read_buffer(prefix_sum_buffer, AccessType::ComputeShaderReadOther)
                    .read_buffer(globals, AccessType::ComputeShaderReadOther)
                    .write_buffer(prefix_sum_buffer, AccessType::ComputeShaderWrite),
                [&]()
                {
                    struct PushData
                    {
                        VkDeviceAddress global_buffer;
                        VkDeviceAddress object_buffer;
                        VkDeviceAddress mesh_buffer;
                        VkDeviceAddress prefix_sum_buffer;
                        float3 camera_pos;
                        uint32_t count;
                        uint32_t shadowmap_height;
                        float lod_distance_factor;
                    };

                    PushData data{};
                    data.global_buffer = graph->buffer_address(globals, sizeof(GlobalData) * (frame_index));
                    data.object_buffer = object_buffer.address;
                    data.mesh_buffer = mesh_buffer.address;
                    data.prefix_sum_buffer = graph->buffer_address(prefix_sum_buffer);
                    data.camera_pos = camera.position;
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
                graph->add_pass(
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
                            uint32_t late;
                        };

                        PushData data{};
                        data.prefix_sum_buffer = graph->buffer_address(prefix_sum_buffer);
                        data.dispatch_buffer = graph->buffer_address(dispatch_buffer);
                        data.prefix_sum_idx = cascade_index;
                        data.late = 0;

                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compact_dispatch_pipeline);
                        vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                        vkCmdDispatch(cmd, 1, 1, 1);
                    }
                );

                graph->add_pass(
                    Pass("Cull shadow meshlets")
                        .read_buffer(dispatch_buffer, AccessType::IndirectBuffer)
                        .read_buffer(prefix_sum_buffer, AccessType::ComputeShaderReadOther)
                        .read_buffer(globals, AccessType::ComputeShaderReadOther)
                        .write_buffer(dispatch_buffer, AccessType::ComputeShaderWrite)
                        .write_buffer(meshlet_instance_buffer, AccessType::ComputeShaderWrite),
                    [&, cascade_index]()
                    {
                        struct PushData
                        {
                            VkDeviceAddress global_buffer;
                            VkDeviceAddress object_buffer;
                            VkDeviceAddress meshlet_buffer;
                            VkDeviceAddress prefix_sum_buffer;
                            VkDeviceAddress dispatch_buffer;
                            VkDeviceAddress meshlet_instance_buffer;
                            uint32_t count;
                            uint32_t shadowmap_height;
                            uint32_t cascade_index;
                        };

                        PushData data{};
                        data.global_buffer = graph->buffer_address(globals, sizeof(GlobalData) * (frame_index));
                        data.object_buffer = object_buffer.address;
                        data.meshlet_buffer = meshlet_buffer.address;
                        data.prefix_sum_buffer = graph->buffer_address(prefix_sum_buffer);
                        data.dispatch_buffer = graph->buffer_address(dispatch_buffer);
                        data.meshlet_instance_buffer = graph->buffer_address(meshlet_instance_buffer);
                        data.count = static_cast<uint32_t>(renderables.size());
                        data.shadowmap_height = SHADOWMAP_EXTENT;
                        data.cascade_index = cascade_index;

                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cull_shadow_meshlet_pipeline);
                        vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                        vkCmdDispatchIndirect(cmd, graph->buffer(dispatch_buffer), 0);
                    }
                );

                graph->add_pass(
                    Pass("Shadowmap " + std::to_string(cascade_index + 1))
                        .read_buffer(dispatch_buffer, AccessType::IndirectBuffer)
                        .read_buffer(meshlet_instance_buffer, AccessType::MeshShaderReadSampledImageOrUniformTexelBuffer)
                        .read_buffer(globals, AccessType::ComputeShaderReadOther)
                        .write_image(shadowmaps[cascade_index], AccessType::DepthStencilAttachmentReadWrite),
                    [&, cascade_index]()
                    {
                        VkRenderingAttachmentInfo depth_attachment_info{ .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
                        depth_attachment_info.imageView = graph->image_view(shadowmaps[cascade_index]);
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

                        struct PushData
                        {
                            VkDeviceAddress global_buffer;
                            VkDeviceAddress object_buffer;
                            VkDeviceAddress vertex_buffer;
                            VkDeviceAddress meshlet_buffer;
                            VkDeviceAddress meshlet_indices_buffer;
                            VkDeviceAddress meshlet_instance_buffer;
                            VkDeviceAddress material_buffer;
                            uint32_t cascade_index;
                        };

                        PushData data{};
                        data.global_buffer = graph->buffer_address(globals, sizeof(GlobalData) * (frame_index));
                        data.object_buffer = object_buffer.address;
                        data.vertex_buffer = vertex_buffer.address;
                        data.meshlet_buffer = meshlet_buffer.address;
                        data.meshlet_indices_buffer = meshlet_indices_buffer.address;
                        data.meshlet_instance_buffer = graph->buffer_address(meshlet_instance_buffer);
                        data.material_buffer = material_buffer.address;
                        data.cascade_index = cascade_index;

                        vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(data), &data);

                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, depth_pipeline);
                        vkCmdDrawMeshTasksIndirectEXT(cmd, graph->buffer(dispatch_buffer), sizeof(Dispatch), 1, 0);

                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, depth_masked_pipeline);
                        vkCmdDrawMeshTasksIndirectEXT(cmd, graph->buffer(dispatch_buffer), sizeof(Dispatch) * DISPATCH_EARLY_MASKED, 1, 0);

                        vkCmdEndRendering(cmd);
                    }
                );
            }
        }

        graph->add_pass(
            Pass("Reset buffers early")
                .write_buffer(prefix_sum_buffer, AccessType::TransferWrite)
                .write_buffer(light_count_buffer, AccessType::TransferWrite),
            [&]()
            {
                struct PushData
                {
                    VkDeviceAddress prefix_sum_buffer;
                    VkDeviceAddress light_count_buffer;
                    uint32_t zero_lights;
                };

                PushData data{};
                data.prefix_sum_buffer = graph->buffer_address(prefix_sum_buffer);
                data.light_count_buffer = graph->buffer_address(light_count_buffer);
                data.zero_lights = 1;

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, reset_buffers_pipeline);
                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                vkCmdDispatch(cmd, 1, 1, 1);
            }
        );

#if LOCAL_LIGHTS == 1
        graph->add_pass(
            Pass("Light culling")
                .write_buffer(froxel_offset_buffer, AccessType::ComputeShaderWrite)
                .write_buffer(froxel_item_buffer, AccessType::ComputeShaderWrite)
                .write_buffer(light_count_buffer, AccessType::ComputeShaderWrite),
            [&]()
            {
                struct PushData
                {
                    float4x4 view;
                    VkDeviceAddress froxels_buffer;
                    VkDeviceAddress light_buffer;
                    VkDeviceAddress froxel_offset_list;
                    VkDeviceAddress froxel_item_list;
                    VkDeviceAddress light_count_buffer;
                    uint32_t light_count;
                };

                PushData data{};
                data.view = frame_context.view;
                data.froxels_buffer = froxels_buffer.address;
                data.light_buffer = light_buffer.address;
                data.froxel_offset_list = graph->buffer_address(froxel_offset_buffer);
                data.froxel_item_list = graph->buffer_address(froxel_item_buffer);
                data.light_count_buffer = graph->buffer_address(light_count_buffer);
                data.light_count = MAX_POINT_LIGHTS;

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, binning_pipeline);
                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                vkCmdDispatch(cmd, 1, 1, 6);
            }
        );
#endif

        graph->add_pass(
            Pass("Compute cull mesh early")
                .read_image(depth_pyramid, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer)
                .write_buffer(prefix_sum_buffer, AccessType::ComputeShaderWrite)
                .write_buffer(late_pass_candidates, AccessType::ComputeShaderWrite),
            [&, skip_occlusion]()
            {
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
                    float num_mips;
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
                data.prefix_sum_buffer = graph->buffer_address(prefix_sum_buffer);
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
                data.num_mips = static_cast<float>(glm::floor(glm::log2(static_cast<float>(glm::max(depth_pyramid_image.extent.width, depth_pyramid_image.extent.height)))) + 1);
                data.depth_pyramid_id = depth_pyramid_descriptor_handle;
                data.skip_occlusion = skip_occlusion ? 1 : 0;

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cull_mesh_pipeline);
                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                vkCmdDispatch(cmd, get_group_count(renderables.size(), 256), 1, 1);
            }
        );

        graph->add_pass(
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
                    uint32_t late;
                };

                PushData data{};
                data.prefix_sum_buffer = graph->buffer_address(prefix_sum_buffer);
                data.dispatch_buffer = graph->buffer_address(dispatch_buffer);
                data.prefix_sum_idx = 0;
                data.late = 0;

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compact_dispatch_pipeline);
                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                vkCmdDispatch(cmd, 1, 1, 1);
            }
        );

        graph->add_pass(
            Pass("Compute cull meshlets early")
                .read_buffer(dispatch_buffer, AccessType::IndirectBuffer)
                .read_buffer(prefix_sum_buffer, AccessType::ComputeShaderReadOther)
                .read_image(depth_pyramid, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer)
                .write_buffer(dispatch_buffer, AccessType::ComputeShaderWrite)
                .write_buffer(meshlet_instance_buffer, AccessType::ComputeShaderWrite)
                .write_buffer(late_pass_candidates, AccessType::ComputeShaderWrite),
            [&, skip_occlusion]()
            {
                struct PushData
                {
                    glm::mat4 view;
                    glm::mat4 prev_view;
                    VkDeviceAddress meshlet_buffer;
                    VkDeviceAddress object_buffer;
                    VkDeviceAddress prefix_sum_buffer;
                    VkDeviceAddress meshlet_instance_buffer;
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
                    float num_mips;
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
                data.prefix_sum_buffer = graph->buffer_address(prefix_sum_buffer);
                data.meshlet_instance_buffer = graph->buffer_address(meshlet_instance_buffer);
                data.dispatch_buffer = graph->buffer_address(dispatch_buffer);
                data.late_pass_candidates_buffer = late_pass_candidates_buffer.address + meshlet_offset * sizeof(uint32_t);
                data.planes = glm::vec4(left_plane.x, left_plane.z, bottom_plane.y, bottom_plane.z);
                data.depth_pyramid_extent = glm::vec2(depth_pyramid_image.extent.width, depth_pyramid_image.extent.height);
                data.depth_buffer_extent = glm::vec2(swapchain.extent.width, swapchain.extent.height);
                data.p00 = frame_context.proj[0][0];
                data.p11 = frame_context.proj[1][1];
                data.near = camera.near;
                data.far = camera.far;
                data.late = 0;
                data.num_mips = static_cast<float>(glm::floor(glm::log2(static_cast<float>(glm::max(depth_pyramid_image.extent.width, depth_pyramid_image.extent.height)))) + 1);
                data.depth_pyramid_id = depth_pyramid_descriptor_handle;
                data.skip_occlusion = skip_occlusion ? 1 : 0;

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cull_meshlet_pipeline);
                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                vkCmdDispatchIndirect(cmd, graph->buffer(dispatch_buffer), 0);
            }
        );

        graph->add_pass(
            Pass("Vbuffer early")
                .read_buffer(dispatch_buffer, AccessType::IndirectBuffer)
                .read_buffer(meshlet_instance_buffer, AccessType::MeshShaderReadSampledImageOrUniformTexelBuffer)
                .write_image(vbuffer, AccessType::ColorAttachmentWrite)
                .write_image(depth_image, AccessType::DepthStencilAttachmentWrite),
            [&]()
            {
                VkClearValue clear_value{};
                clear_value.color.uint32[0] = 0xFFFFFFFFu;
                clear_value.color.uint32[1] = 0;
                clear_value.color.uint32[2] = 0;
                clear_value.color.uint32[3] = 0;

                VkRenderingAttachmentInfo rendering_attachment_info = VkRenderingAttachmentInfo{ .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
                rendering_attachment_info.imageView = graph->image_view(vbuffer);
                rendering_attachment_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                rendering_attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                rendering_attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                rendering_attachment_info.clearValue = clear_value;

                VkRenderingAttachmentInfo depth_attachment_info{ .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
                depth_attachment_info.imageView = graph->image_view(depth_image);
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

                struct PushData
                {
                    glm::mat4 view_proj;
                    VkDeviceAddress vertex_buffer;
                    VkDeviceAddress object_buffer;
                    VkDeviceAddress material_buffer;
                    VkDeviceAddress meshlet_buffer;
                    VkDeviceAddress meshlet_indices_buffer;
                    VkDeviceAddress meshlet_instance_buffer;
                };

                PushData data{};
                data.view_proj = frame_context.proj * camera.get_view_matrix();
                data.vertex_buffer = vertex_buffer.address;
                data.object_buffer = object_buffer.address;
                data.material_buffer = material_buffer.address;
                data.meshlet_buffer = meshlet_buffer.address;
                data.meshlet_indices_buffer = meshlet_indices_buffer.address;
                data.meshlet_instance_buffer = graph->buffer_address(meshlet_instance_buffer);
                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(data), &data);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vbuffer_pipeline);
                vkCmdDrawMeshTasksIndirectEXT(cmd, graph->buffer(dispatch_buffer), sizeof(Dispatch), 1, 0);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vbuffer_masked_pipeline);
                vkCmdDrawMeshTasksIndirectEXT(cmd, graph->buffer(dispatch_buffer), sizeof(Dispatch) * DISPATCH_EARLY_MASKED, 1, 0);

                vkCmdEndRendering(cmd);
            }
        );

        if (!freeze_matrices)
        {
            graph->add_pass(
                Pass("Hiz")
                    .read_image(depth_image, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer)
                    .write_image(depth_pyramid, AccessType::ComputeShaderWrite)
                    .write_buffer(spd_counter, AccessType::ComputeShaderWrite),
                [&]()
                {
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
                    data.src_id = graph->image_id(depth_image);
                    data.dst_id = depth_pyramid_first_mip;

                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hiz_pipeline);
                    vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                    vkCmdDispatch(cmd, group_count_x, group_count_y, 1);
                }
            );
        }

        skip_occlusion = false;

        graph->add_pass(
            Pass("Compute cull mesh late")
                .read_image(depth_pyramid, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer)
                .read_buffer(prefix_sum_buffer, AccessType::ComputeShaderReadOther)
                .read_buffer(late_pass_candidates, AccessType::ComputeShaderReadOther)
                .write_buffer(prefix_sum_buffer, AccessType::ComputeShaderWrite)
                .write_buffer(late_pass_candidates, AccessType::ComputeShaderWrite),
            [&, skip_occlusion]()
            {
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
                    float num_mips;
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
                data.prefix_sum_buffer = graph->buffer_address(prefix_sum_buffer);
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
                data.num_mips = static_cast<float>(glm::floor(glm::log2(static_cast<float>(glm::max(depth_pyramid_image.extent.width, depth_pyramid_image.extent.height)))) + 1);
                data.depth_pyramid_id = depth_pyramid_descriptor_handle;
                data.skip_occlusion = skip_occlusion ? 1 : 0;

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cull_mesh_pipeline);
                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                vkCmdDispatch(cmd, get_group_count(renderables.size(), 256), 1, 1);
            }
        );

        graph->add_pass(
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
                    uint32_t late;
                };

                PushData data{};
                data.prefix_sum_buffer = graph->buffer_address(prefix_sum_buffer);
                data.dispatch_buffer = graph->buffer_address(dispatch_buffer);
                data.prefix_sum_idx = 0;
                data.late = 1;

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compact_dispatch_pipeline);
                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                vkCmdDispatch(cmd, 1, 1, 1);
            }
        );

        graph->add_pass(
            Pass("Compute cull meshlets late")
                .read_buffer(dispatch_buffer, AccessType::IndirectBuffer)
                .read_buffer(prefix_sum_buffer, AccessType::ComputeShaderReadOther)
                .read_buffer(late_pass_candidates, AccessType::ComputeShaderReadOther)
                .read_image(depth_pyramid, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer)
                .write_buffer(dispatch_buffer, AccessType::ComputeShaderWrite)
                .write_buffer(meshlet_instance_buffer, AccessType::ComputeShaderWrite)
                .write_buffer(late_pass_candidates, AccessType::ComputeShaderWrite),
            [&, skip_occlusion]()
            {
                struct PushData
                {
                    glm::mat4 view;
                    glm::mat4 prev_view;
                    VkDeviceAddress meshlet_buffer;
                    VkDeviceAddress object_buffer;
                    VkDeviceAddress prefix_sum_buffer;
                    VkDeviceAddress meshlet_instance_buffer;
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
                    float num_mips;
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
                data.prefix_sum_buffer = graph->buffer_address(prefix_sum_buffer);
                data.meshlet_instance_buffer = graph->buffer_address(meshlet_instance_buffer);
                data.dispatch_buffer = graph->buffer_address(dispatch_buffer);
                data.late_pass_candidates_buffer = late_pass_candidates_buffer.address + meshlet_offset * sizeof(uint32_t);
                data.planes = glm::vec4(left_plane.x, left_plane.z, bottom_plane.y, bottom_plane.z);
                data.depth_pyramid_extent = glm::vec2(depth_pyramid_image.extent.width, depth_pyramid_image.extent.height);
                data.depth_buffer_extent = glm::vec2(swapchain.extent.width, swapchain.extent.height);
                data.p00 = frame_context.proj[0][0];
                data.p11 = frame_context.proj[1][1];
                data.near = camera.near;
                data.far = camera.far;
                data.late = 1;
                data.num_mips = static_cast<float>(glm::floor(glm::log2(static_cast<float>(glm::max(depth_pyramid_image.extent.width, depth_pyramid_image.extent.height)))) + 1);
                data.depth_pyramid_id = depth_pyramid_descriptor_handle;
                data.skip_occlusion = skip_occlusion ? 1 : 0;

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cull_meshlet_pipeline);
                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                vkCmdDispatchIndirect(cmd, graph->buffer(dispatch_buffer), 0);
            }
        );

        graph->add_pass(
            Pass("Vbuffer late")
                .read_buffer(dispatch_buffer, AccessType::IndirectBuffer)
                .read_buffer(meshlet_instance_buffer, AccessType::MeshShaderReadSampledImageOrUniformTexelBuffer)
                .read_image(vbuffer, AccessType::ColorAttachmentRead)
                .read_image(depth_image, AccessType::DepthStencilAttachmentRead)
                .write_image(vbuffer, AccessType::ColorAttachmentWrite)
                .write_image(depth_image, AccessType::DepthStencilAttachmentWrite),
            [&]()
            {
                VkRenderingAttachmentInfo rendering_attachment_info = VkRenderingAttachmentInfo{ .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
                rendering_attachment_info.imageView = graph->image_view(vbuffer);
                rendering_attachment_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                rendering_attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                rendering_attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

                VkRenderingAttachmentInfo depth_attachment_info{ .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
                depth_attachment_info.imageView = graph->image_view(depth_image);
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
                    VkDeviceAddress meshlet_instance_buffer;
                };

                PushData data{};
                data.view_proj = frame_context.proj * camera.get_view_matrix();
                data.vertex_buffer = vertex_buffer.address;
                data.object_buffer = object_buffer.address;
                data.material_buffer = material_buffer.address;
                data.meshlet_buffer = meshlet_buffer.address;
                data.meshlet_indices_buffer = meshlet_indices_buffer.address;
                data.meshlet_instance_buffer = graph->buffer_address(meshlet_instance_buffer);
                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(data), &data);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vbuffer_pipeline);
                vkCmdDrawMeshTasksIndirectEXT(cmd, graph->buffer(dispatch_buffer), sizeof(Dispatch) * DISPATCH_LATE_OPAQUE, 1, 0);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vbuffer_masked_pipeline);
                vkCmdDrawMeshTasksIndirectEXT(cmd, graph->buffer(dispatch_buffer), sizeof(Dispatch) * DISPATCH_LATE_MASKED, 1, 0);

                vkCmdEndRendering(cmd);
            }
        );

#if VOLUMETRIC_FOG == 1
        graph->add_pass(
            Pass("DataInjection")
                .write_image(fog_froxel_data, AccessType::ComputeShaderWrite),
            [&]()
            {
                struct PushData
                {
                    float4x4 inverse_view_proj;
                    glm::uvec3 froxel_dims;
                    uint32_t current_frame;
                    glm::vec2 halton;
                    float froxel_near;
                    float froxel_far;
                    uint32_t noise_tex_id;
                    uint32_t froxel_data_tex_id;
                    float noise_pos_mult;
                    float noise_speed_mult;
                    float density_modifier;
                    float height_fog_density;
                    float height_fog_falloff;
                    float box_fog_density;
                    float noise_scale;
                    float temporal_reprojection_jitter_scale;
                    float scattering_factor;
                };

                PushData data{};
                data.inverse_view_proj = glm::inverse(frame_context.proj * camera.get_view_matrix());
                data.froxel_dims = glm::uvec3(FOG_WIDTH, FOG_HEIGHT, FOG_DEPTH);
                data.current_frame = frame_number;
                data.halton = glm::vec2(halton_x, halton_y);
                data.froxel_near = camera.near;
                data.froxel_far = FROXELS_FAR_PLANE;
                data.noise_tex_id = volumetric_noise_id;
                data.froxel_data_tex_id = graph->image_id(fog_froxel_data);
                data.noise_pos_mult = noise_pos_mult;
                data.noise_speed_mult = noise_speed_mult;
                data.density_modifier = density_modifier;
                data.height_fog_density = height_fog_density;
                data.height_fog_falloff = height_fog_falloff;
                data.box_fog_density = box_fog_density;
                data.noise_scale = noise_scale;
                data.temporal_reprojection_jitter_scale = temporal_reprojection_jitter_scale;
                data.scattering_factor = scattering_factor;

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, data_injection_pipeline);
                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                auto group_count_x = get_group_count(FOG_WIDTH, 8);
                auto group_count_y = get_group_count(FOG_HEIGHT, 8);
                vkCmdDispatch(cmd, group_count_x, group_count_y, FOG_DEPTH);
            }
        );

        auto light_scattering_pass = Pass("LightScattering")
                                         .read_buffer(globals, AccessType::ComputeShaderReadOther)
                                         .read_image(fog_froxel_data, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer)
                                         .write_image(raw_light_scattering, AccessType::ComputeShaderWrite);
        for (const auto shadowmap : shadowmaps)
        {
            light_scattering_pass.read_image(shadowmap, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer);
        }
        graph->add_pass(
            light_scattering_pass,
            [&]()
            {
                struct PushData
                {
                    float4x4 inverse_view_proj;
                    glm::uvec3 froxel_dims;
                    uint32_t light_scattering_id;
                    glm::vec3 camera_pos;
                    uint32_t froxel_data_tex_id;
                    glm::vec2 halton;
                    float froxel_near;
                    float froxel_far;
                    float3 sunlight_dir;
                    uint32_t current_frame;
                    VkDeviceAddress global_buffer;
                    uint32_t shadowmap_id;
                    float temporal_reprojection_jitter_scale;
                    float noise_scale;
                };

                PushData data{};
                data.inverse_view_proj = glm::inverse(frame_context.proj * camera.get_view_matrix());
                data.froxel_dims = glm::uvec3(FOG_WIDTH, FOG_HEIGHT, FOG_DEPTH);
                data.light_scattering_id = graph->image_id(raw_light_scattering);
                data.camera_pos = camera.position;
                data.froxel_data_tex_id = graph->image_id(fog_froxel_data);
                data.halton = glm::vec2(halton_x, halton_y);
                data.froxel_near = camera.near;
                data.froxel_far = FROXELS_FAR_PLANE;
                data.sunlight_dir = sunlight_dir;
                data.current_frame = frame_number;
                data.global_buffer = graph->buffer_address(globals, sizeof(GlobalData) * (frame_index));
                data.shadowmap_id = graph->image_id(shadowmaps[0]);
                data.temporal_reprojection_jitter_scale = temporal_reprojection_jitter_scale;
                data.noise_scale = noise_scale;

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, light_scattering_pipeline);
                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                auto group_count_x = get_group_count(FOG_WIDTH, 8);
                auto group_count_y = get_group_count(FOG_HEIGHT, 8);
                vkCmdDispatch(cmd, group_count_x, group_count_y, FOG_DEPTH);
            }
        );

        if (b_fog_spatial_filter)
        {
            graph->add_pass(
                Pass("FogSpatialFilterHorizontal")
                    .read_image(raw_light_scattering, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer)
                    .write_image(fog_froxel_data, AccessType::ComputeShaderWrite),
                [&]()
                {
                    struct PushData
                    {
                        glm::uvec3 froxel_dims;
                        uint32_t input_id;
                        uint32_t output_id;
                        uint32_t horizontal;
                    };

                    PushData data{};
                    data.froxel_dims = glm::uvec3(FOG_WIDTH, FOG_HEIGHT, FOG_DEPTH);
                    data.input_id = graph->image_id(raw_light_scattering);
                    data.output_id = graph->image_id(fog_froxel_data);
                    data.horizontal = 1;

                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fog_spatial_pipeline);
                    vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                    auto group_count_x = get_group_count(FOG_WIDTH, 8);
                    auto group_count_y = get_group_count(FOG_HEIGHT, 8);
                    vkCmdDispatch(cmd, group_count_x, group_count_y, FOG_DEPTH);
                }
            );

            graph->add_pass(
                Pass("FogSpatialFilterVertical")
                    .read_image(fog_froxel_data, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer)
                    .write_image(raw_light_scattering, AccessType::ComputeShaderWrite),
                [&]()
                {
                    struct PushData
                    {
                        glm::uvec3 froxel_dims;
                        uint32_t input_id;
                        uint32_t output_id;
                        uint32_t horizontal;
                    };

                    PushData data{};
                    data.froxel_dims = glm::uvec3(FOG_WIDTH, FOG_HEIGHT, FOG_DEPTH);
                    data.input_id = graph->image_id(fog_froxel_data);
                    data.output_id = graph->image_id(raw_light_scattering);
                    data.horizontal = 0;

                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fog_spatial_pipeline);
                    vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                    auto group_count_x = get_group_count(FOG_WIDTH, 8);
                    auto group_count_y = get_group_count(FOG_HEIGHT, 8);
                    vkCmdDispatch(cmd, group_count_x, group_count_y, FOG_DEPTH);
                }
            );
        }

        if (b_fog_temporal_filter)
        {
            graph->add_pass(
                Pass("FogTemporalFilter")
                    .read_image(raw_light_scattering, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer)
                    .read_image(temporal_filtered_scattering[next_ping_pong_index], AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer)
                    .write_image(temporal_filtered_scattering[ping_pong_index], AccessType::ComputeShaderWrite),
                [&, fog_temporal_first_frame]()
                {
                    struct PushData
                    {
                        glm::mat4 inverse_view_proj;
                        glm::mat4 previous_view_proj;
                        glm::uvec3 froxel_dims;
                        uint32_t first_frame;
                        glm::vec2 halton;
                        float temporal_reprojection_jitter_scale;
                        float noise_scale;
                        float near;
                        float far;
                        uint32_t raw_scattering_id;
                        uint32_t temporal_previous_id;
                        uint32_t temporal_current_id;
                        float scale;
                        float bias;
                        float reprojection_factor;
                        uint32_t blue_noise_tex;
                        uint32_t current_frame;
                    };

                    PushData data{};
                    // TODO: take into account frozen matrices
                    data.inverse_view_proj = glm::inverse(frame_context.view_proj);
                    data.previous_view_proj = frame_context.prev_view_proj;
                    data.froxel_dims = glm::uvec3(FOG_WIDTH, FOG_HEIGHT, FOG_DEPTH);
                    data.first_frame = fog_temporal_first_frame ? 1 : 0;
                    data.halton = glm::vec2(halton_x, halton_y);
                    data.temporal_reprojection_jitter_scale = temporal_reprojection_jitter_scale;
                    data.noise_scale = noise_scale;
                    data.near = camera.near;
                    data.far = FROXELS_FAR_PLANE;
                    data.raw_scattering_id = graph->image_id(raw_light_scattering);
                    data.temporal_previous_id = graph->image_id(temporal_filtered_scattering[next_ping_pong_index]);
                    data.temporal_current_id = graph->image_id(temporal_filtered_scattering[ping_pong_index]);
                    const float ratio = FROXELS_FAR_PLANE / camera.near;
                    data.scale = static_cast<float>(FOG_DEPTH) / glm::log(ratio);
                    data.bias = static_cast<float>(FOG_DEPTH) * glm::log(camera.near) / glm::log(ratio);
                    data.reprojection_factor = 0.1;
                    data.blue_noise_tex = -1;
                    data.current_frame = -1;

                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fog_temporal_pipeline);
                    vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                    auto group_count_x = get_group_count(FOG_WIDTH, 8);
                    auto group_count_y = get_group_count(FOG_HEIGHT, 8);
                    vkCmdDispatch(cmd, group_count_x, group_count_y, FOG_DEPTH);
                }
            );

            fog_temporal_first_frame = false;
        }

        auto light_integration_input_id = b_fog_temporal_filter ? temporal_filtered_scattering[ping_pong_index] : raw_light_scattering;

        graph->add_pass(
            Pass("LightIntegration")
                .read_image(light_integration_input_id, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer)
                .write_image(fog_froxel_data, AccessType::ComputeShaderWrite),
            [&]()
            {
                struct PushData
                {
                    glm::mat4 inverse_view_proj;
                    glm::uvec3 froxel_dims;
                    uint32_t light_scattering_id;
                    glm::vec3 camera_pos;
                    uint32_t integrated_light_scattering_id;
                    float froxel_near;
                    float froxel_far;
                };

                PushData data{};
                data.inverse_view_proj = glm::inverse(frame_context.proj * camera.get_view_matrix());
                data.froxel_dims = glm::uvec3(FOG_WIDTH, FOG_HEIGHT, FOG_DEPTH);
                data.light_scattering_id = graph->image_id(light_integration_input_id);
                data.camera_pos = camera.position;
                data.integrated_light_scattering_id = graph->image_id(fog_froxel_data);
                data.froxel_near = camera.near;
                data.froxel_far = FROXELS_FAR_PLANE;

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, light_integration_pipeline);
                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                auto group_count_x = get_group_count(FOG_WIDTH, 8);
                auto group_count_y = get_group_count(FOG_HEIGHT, 8);
                vkCmdDispatch(cmd, group_count_x, group_count_y, 1);
            }
        );
#endif

        auto lighting_pass = Pass("Lighting")
                                 .read_image(vbuffer, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer)
                                 .read_buffer(globals, AccessType::ComputeShaderReadOther)
                                 .read_image(depth_image, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer)
                                 .read_buffer(froxel_offset_buffer, AccessType::ComputeShaderReadOther)
                                 .read_buffer(froxel_item_buffer, AccessType::ComputeShaderReadOther)
                                 .read_buffer(meshlet_instance_buffer, AccessType::ComputeShaderReadOther)
                                 .read_image(fog_froxel_data, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer)
                                 .write_image(draw_image, AccessType::ComputeShaderWrite);
        for (const auto shadowmap : shadowmaps)
        {
            lighting_pass.read_image(shadowmap, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer);
        }
        graph->add_pass(
            lighting_pass,
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
                    VkDeviceAddress light_buffer;
                    VkDeviceAddress froxel_offset_list;
                    VkDeviceAddress froxel_item_list;
                    VkDeviceAddress sh_buffer;
                    VkDeviceAddress meshlet_instance_buffer;
                    glm::vec2 screen_extent;
                    glm::vec3 light_dir;
                    uint32_t shadowmap_id; // Offset accordingly
                    glm::vec3 camera_pos;
                    uint32_t visbuffer_id;
                    glm::vec2 pixels_per_cluster;
                    uint32_t debug_id;
                    uint32_t output_id;
                    uint32_t depth_id;
                    float near;
                    float scale;
                    float bias;
                    float fog_far;
                    float fog_scale;
                    float fog_bias;
                    uint32_t fog_id;
                };

                PushData data{};
                data.view_proj = frame_context.proj * camera.get_view_matrix();
                data.vertex_buffer = vertex_buffer.address;
                data.object_buffer = object_buffer.address;
                data.meshlet_buffer = meshlet_buffer.address;
                data.meshlet_indices_buffer = meshlet_indices_buffer.address;
                data.material_buffer = material_buffer.address;
                data.global_buffer = graph->buffer_address(globals, sizeof(GlobalData) * (frame_index));
#if LOCAL_LIGHTS == 1
                data.light_buffer = light_buffer.address;
#endif
                data.froxel_offset_list = graph->buffer_address(froxel_offset_buffer);
                data.froxel_item_list = graph->buffer_address(froxel_item_buffer);
                data.sh_buffer = sh_buffer.address;
                data.meshlet_instance_buffer = graph->buffer_address(meshlet_instance_buffer);
                data.light_dir = sunlight_dir;
                data.shadowmap_id = b_shadows ? graph->image_id(shadowmaps[0]) : -1u;
                data.camera_pos = camera.position;
                data.visbuffer_id = graph->image_id(vbuffer);
                data.screen_extent = glm::vec2(swapchain.extent.width, swapchain.extent.height);
                auto cluster_x = glm::ceil(static_cast<float>(swapchain.extent.width) / FROXELS_X);
                auto cluster_y = glm::ceil(static_cast<float>(swapchain.extent.height) / FROXELS_Y);
                data.pixels_per_cluster = glm::vec2(cluster_x, cluster_y);
                data.debug_id = DEBUG_ID;
                data.output_id = graph->image_id(draw_image);
                data.depth_id = graph->image_id(depth_image);
                data.near = camera.near;
                const float ratio = FROXELS_FAR_PLANE / camera.near;
                data.scale = static_cast<float>(FROXELS_Z) / glm::log(ratio);
                data.bias = static_cast<float>(FROXELS_Z) * glm::log(camera.near) / glm::log(ratio);
                data.fog_far = FROXELS_FAR_PLANE;
                data.fog_scale = static_cast<float>(FOG_DEPTH) / glm::log(ratio);
                data.fog_bias = static_cast<float>(FOG_DEPTH) * glm::log(camera.near) / glm::log(ratio);
                data.fog_id = graph->image_id(fog_froxel_data);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lighting_pipeline);
                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                vkCmdDispatch(cmd, get_group_count(swapchain.extent.width, 8), get_group_count(swapchain.extent.height, 8), 1);
            }
        );

        if (!freeze_matrices)
        {
            graph->add_pass(
                Pass("Hiz temporal")
                    .read_image(depth_image, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer)
                    .write_image(depth_pyramid, AccessType::ComputeShaderWrite)
                    .write_buffer(spd_counter, AccessType::ComputeShaderWrite),
                [&]()
                {
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
                    data.src_id = graph->image_id(depth_image);
                    data.dst_id = depth_pyramid_first_mip;

                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hiz_pipeline);
                    vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushData), &data);
                    vkCmdDispatch(cmd, group_count_x, group_count_y, 1);
                }
            );
        }

        graph->add_pass(
            Pass("Copy to swapchain")
                .read_image(draw_image, AccessType::TransferRead)
                .write_image(swapchain_image, AccessType::TransferWrite),
            [&]()
            {
                copy_image(cmd, graph->image(draw_image), graph->image(swapchain_image), swapchain.extent, swapchain.extent);
            }
        );

        graph->add_pass(
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
        graph->compile(cmd);
        graph->execute(cmd, frame_index);

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

        graph->reset();

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
    destroy_buffer(allocator, froxels_buffer);
#if LOCAL_LIGHTS == 1
    destroy_buffer(allocator, light_buffer);
#endif
    destroy_buffer(allocator, global_buffer);
    destroy_buffer(allocator, sh_buffer);

    destroy_image(device, allocator, depth_pyramid_image);
    destroy_image(device, allocator, brdf_lut_image);

#if VOLUMETRIC_FOG == 1
    destroy_image(device, allocator, volumetric_noise_image);
#endif

    for (const auto& img : shadowmap_images)
    {
        destroy_image(device, allocator, img);
    }

    for (const auto& samp : samplers)
    {
        vkDestroySampler(device, samp, nullptr);
    }

    for (const auto& view : depth_pyramid_image_views)
    {
        vkDestroyImageView(device, view, nullptr);
    }

    for (const auto& pipeline : pipelines)
    {
        vkDestroyPipeline(device, pipeline, nullptr);
    }

    asset_loader.cleanup();
    graph->cleanup();
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
            // TODO: test req - do we need a mesh cache?
            obj.mesh_id = static_cast<uint32_t>(meshes.size());
            obj.material_id = mesh.material_id;
            obj.masked = mesh.masked;

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
    uint32_t sampler_count = MAX_DESCRIPTORS_SAMPLER_COUNT;

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
    engine.init_descriptors();
    engine.init_imgui();

    engine.run();

    engine.cleanup();

    fmt::println("Closing app...");
}
