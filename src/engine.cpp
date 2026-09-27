#include "engine.h"
#include "common.h"
#include "swapchain.h"
#include "resources.h"
#include "asset_loader.h"
#include "pipelines.h"
#include "shared_cpu_gpu.h"
#include "camera.h"
#include "descriptors.h"

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
#include <glm/ext/vector_float4.hpp>
#include <glm/matrix.hpp>
#include <glm/trigonometric.hpp>

#include <vector>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <array>

const char* ASSET_NAME = "Sponza/Sponza.gltf";

namespace
{
// tTken directly from https://github.com/zeux/niagara/blob/master/src/scene.cpp
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

    // SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "wayland");
    SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "x11");
    SDL_Init(SDL_INIT_VIDEO);
    auto window_flags = static_cast<SDL_WindowFlags>(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    window = SDL_CreateWindow("Untitled vulkan renderer", 1280u, 720u, window_flags);
    bool surface_result = SDL_Vulkan_CreateSurface(window, vkb_inst, nullptr, &surface);
    if (!surface_result)
    {
        fmt::println("Failed to create window surface.");
        std::abort();
    }
    SDL_SetWindowRelativeMouseMode(window, true);

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
    features_12.runtimeDescriptorArray = true;

    VkPhysicalDeviceVulkan11Features features_11{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };

    VkPhysicalDeviceFeatures features_10{};
    features_10.shaderInt16 = true;

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

    fmt::println("Initialized Vulkan");
}

void Engine::cleanup()
{
    vkDeviceWaitIdle(device);

    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyPipeline(device, mesh_pass->pipeline, nullptr);

    vkDestroyDescriptorPool(device, desc_pool, nullptr);
    vkDestroyDescriptorSetLayout(device, buffer_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, storage_image_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, sampled_image_layout, nullptr);
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
    Camera camera{};
    camera.position = glm::vec3(0, 0, 5);
    camera.far = 100.0f;
    camera.near = 0.01f;

    auto proj = camera.set_perspective_matrix(glm::radians(camera.fov), static_cast<float>(swapchain.extent.width) / swapchain.extent.height, camera.near);

    // Init scene
    AssetLoader asset_loader{ .device = device, .allocator = allocator };
    bool loaded = asset_loader.load_gltf(graphics_queue, imm_fence, imm_pool, imm_buf, ASSET_NAME);
    if (!loaded)
    {
        assert(0 && "load_gltf failed");
    }

    for (const auto& node : asset_loader.parent_nodes)
    {
        register_object(node, glm::mat4(1.0), asset_loader.children_nodes, asset_loader.meshes);
    }

    // Update descriptors
    std::vector<VkWriteDescriptorSet> writes{};
    std::vector<VkDescriptorImageInfo> image_infos(asset_loader.images.size());
    for (uint32_t handle = 0; handle < asset_loader.images.size(); handle++)
    {
        image_infos[handle] = VkDescriptorImageInfo{ .imageView = asset_loader.images[handle].view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
        writes.emplace_back(write_image_descriptor(sampled_image_desc_set, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, handle, &image_infos[handle]));
    }
    vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);

    writes.clear();
    image_infos.clear();

    // Create sampler
    VkSampler linear_samp = create_sampler(device, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT);
    image_infos.emplace_back(VkDescriptorImageInfo{ .sampler = linear_samp });
    writes.emplace_back(write_sampler_descriptor(sampler_desc_set, VK_DESCRIPTOR_TYPE_SAMPLER, 0, &image_infos[0]));
    vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);

    // Load GPU data
    Buffer vertex_buffer = create_buffer_with_data(device, graphics_queue, imm_fence, imm_pool, imm_buf, allocator, asset_loader.m_vertices.data(), asset_loader.m_vertices.size() * sizeof(Vertex));
    Buffer index_buffer = create_buffer_with_data(device, graphics_queue, imm_fence, imm_pool, imm_buf, allocator, asset_loader.m_indices.data(), asset_loader.m_indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    Buffer object_buffer = create_buffer_with_data(device, graphics_queue, imm_fence, imm_pool, imm_buf, allocator, renderables.data(), renderables.size() * sizeof(ObjectData));
    Buffer material_buffer = create_buffer_with_data(device, graphics_queue, imm_fence, imm_pool, imm_buf, allocator, asset_loader.materials.data(), asset_loader.materials.size() * sizeof(MaterialData));

    Image depth_image = create_image(device, allocator, VkExtent3D{ swapchain.extent.width, swapchain.extent.height, 1 }, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);

    // TODO: single pipeline barrier for all images instead of individual calls
    // UNDEFINED -> GENERAL once, then keep in GENERAL in render loop
    immediate_submit(device, graphics_queue, imm_fence, imm_pool, imm_buf, [&](VkCommandBuffer cmd)
                     {
                         stage_barrier(
                             cmd,
                             depth_image.image,
                             VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_GENERAL,
                             0,
                             0,
                             0,
                             0,
                             VK_IMAGE_ASPECT_DEPTH_BIT
                         );
                     });

    // Init PSO
    auto swapchain_format = VK_FORMAT_B8G8R8A8_UNORM;
    auto program = load_shader_program("mesh.slang", device);

    VkPhysicalDeviceProperties2 properties2{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
    vkGetPhysicalDeviceProperties2(physical_device, &properties2);
    auto max_push_constant_size = properties2.properties.limits.maxPushConstantsSize;
    VkPushConstantRange pc_range{ .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = max_push_constant_size };

    std::array<VkDescriptorSetLayout, 4> desc_set_layouts{ buffer_layout, storage_image_layout, sampled_image_layout, sampler_layout };
    VkPipelineLayoutCreateInfo pipeline_layout_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipeline_layout_info.pPushConstantRanges = &pc_range;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pSetLayouts = desc_set_layouts.data();
    pipeline_layout_info.setLayoutCount = desc_set_layouts.size();
    VK_CHECK(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout));

    mesh_pass = create_graphics_pipeline(device, &program, { VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT }, nullptr, &pipeline_layout, { swapchain_format });
    vkDestroyShaderModule(device, program.module, nullptr);

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
                    update_swapchain(swapchain, window, physical_device, device, surface);
                }
                break;
            default:
                break;
            }

            camera.process_sdl_event(event, SDL_GetWindowRelativeMouseMode(window));
        }

        if (swapchain_dirty)
        {
            update_swapchain(swapchain, window, physical_device, device, surface);
            swapchain_dirty = false;
        }

        // Update camera
        camera.update(delta_time);
        auto view = camera.get_view_matrix();
        auto view_proj = proj * view;

        // Wait on fence
        auto frame = get_current_frame();
        VK_CHECK(vkWaitForFences(device, 1, &frame.fence, true, WAIT_TIME));

        // TODO: flush frame deletion queue

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

        // Record command buffer
        VkCommandBuffer cmd = frame.command_buffer;
        VK_CHECK(vkResetCommandPool(device, frame.command_pool, 0));

        VkCommandBufferBeginInfo cmd_begin_info{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        cmd_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_begin_info));

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &buffer_desc_set, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 1, 1, &storage_image_desc_set, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 2, 1, &sampled_image_desc_set, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 3, 1, &sampler_desc_set, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &buffer_desc_set, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 1, 1, &storage_image_desc_set, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 2, 1, &sampled_image_desc_set, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 3, 1, &sampler_desc_set, 0, nullptr);

        stage_barrier(
            cmd,
            swapchain.images[swapchain_image_idx],
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            0,
            0,
            0,
            0
        );

        VkClearColorValue clear_color_value = { 0.f, 0.f, 0.f, 1.f };
        VkClearValue clear_value{ .color = clear_color_value };

        VkRenderingAttachmentInfo rendering_attachment_info{ .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        rendering_attachment_info.imageView = swapchain.image_views[swapchain_image_idx];
        rendering_attachment_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        rendering_attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        rendering_attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingAttachmentInfo depth_attachment_info{ .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        depth_attachment_info.imageView = depth_image.view;
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

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh_pass->pipeline);

        struct PushData
        {
            glm::mat4 view_proj;
            VkDeviceAddress vb;
            VkDeviceAddress ob;
            VkDeviceAddress mb;
        };

        PushData data{ view_proj, vertex_buffer.address, object_buffer.address, material_buffer.address };
        VkPushConstantsInfo pc{ .sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO };
        pc.layout = pipeline_layout;
        pc.size = sizeof(PushData);
        pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pc.pValues = &data;
        vkCmdPushConstants2(cmd, &pc);
        vkCmdBindIndexBuffer2(cmd, index_buffer.buffer, 0, index_buffer.size, VK_INDEX_TYPE_UINT32);

        uint32_t i = 0;
        for (auto i = 0; i < meshes.size(); i++)
        {
            auto& mesh = meshes[i];
            vkCmdDrawIndexed(cmd, mesh.index_count, 1, mesh.first_index, mesh.vertex_offset, i);
        }

        vkCmdEndRendering(cmd);

        stage_barrier(
            cmd,
            swapchain.images[swapchain_image_idx],
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            0,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            0
        );

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

        frame_number++;
    }

    vkDeviceWaitIdle(device);

    destroy_buffer(allocator, vertex_buffer);
    destroy_buffer(allocator, index_buffer);
    destroy_buffer(allocator, object_buffer);
    destroy_buffer(allocator, material_buffer);
    destroy_image(device, allocator, depth_image);
    vkDestroySampler(device, linear_samp, nullptr);

    asset_loader.cleanup();
}

// TODO: do we need to cache to deduplicate?
void Engine::register_object(const Node& node, const glm::mat4& top_matrix, const std::vector<Node>& children, const std::vector<MeshAsset>& mesh_assets)
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
            obj.orientation = glm::quat(rotation.w, rotation.x, rotation.y, rotation.z);
            obj.material_id = mesh.material_id;
            // obj.mesh_id = -1;

            renderables.push_back(obj);
            meshes.push_back(mesh);
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
    uint32_t storage_image_count = 300;
    uint32_t sampled_image_count = 300;
    uint32_t sampler_count = 5;

    std::array<VkDescriptorPoolSize, 4> pool_sizes{
        VkDescriptorPoolSize{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = buffer_count },
        VkDescriptorPoolSize{ .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = storage_image_count },
        VkDescriptorPoolSize{ .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .descriptorCount = sampled_image_count },
        VkDescriptorPoolSize{ .type = VK_DESCRIPTOR_TYPE_SAMPLER, .descriptorCount = sampler_count },
    };

    desc_pool = create_descriptor_pool(device, pool_sizes.data(), pool_sizes.size());

    // Set up bindless
    VkDescriptorBindingFlags binding_flags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
    VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags_info{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
    binding_flags_info.pBindingFlags = &binding_flags;
    binding_flags_info.bindingCount = 1;

    VkDescriptorSetLayoutBinding buffer_binding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = buffer_count, .stageFlags = VK_SHADER_STAGE_ALL };
    VkDescriptorSetLayoutBinding storage_image_binding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = storage_image_count, .stageFlags = VK_SHADER_STAGE_ALL };
    VkDescriptorSetLayoutBinding sampled_image_binding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .descriptorCount = sampled_image_count, .stageFlags = VK_SHADER_STAGE_ALL };
    VkDescriptorSetLayoutBinding sampler_binding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, .descriptorCount = sampler_count, .stageFlags = VK_SHADER_STAGE_ALL };

    buffer_layout = create_descriptor_set_layout(device, &buffer_binding, 1, &binding_flags_info, 0);
    storage_image_layout = create_descriptor_set_layout(device, &storage_image_binding, 1, &binding_flags_info, 0);
    sampled_image_layout = create_descriptor_set_layout(device, &sampled_image_binding, 1, &binding_flags_info, 0);
    sampler_layout = create_descriptor_set_layout(device, &sampler_binding, 1, &binding_flags_info, 0);

    buffer_desc_set = create_descriptor_set(device, desc_pool, &buffer_layout, &buffer_count);
    storage_image_desc_set = create_descriptor_set(device, desc_pool, &storage_image_layout, &storage_image_count);
    sampled_image_desc_set = create_descriptor_set(device, desc_pool, &sampled_image_layout, &sampled_image_count);
    sampler_desc_set = create_descriptor_set(device, desc_pool, &sampler_layout, &sampler_count);
}

int main()
{
    Engine engine{};

    engine.init_vulkan();
    engine.init_commands();
    engine.init_sync();
    engine.init_descriptors();

    engine.run();

    engine.cleanup();

    fmt::println("Closing app...");
}
