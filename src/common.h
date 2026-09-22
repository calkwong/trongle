#pragma once

#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <fmt/core.h>

#define VK_CHECK(x) \
    do \
    { \
        VkResult err = x; \
        if (err) \
        { \
            fmt::println("{} {}", __FILE__, __LINE__); \
            fmt::println("Detected Vulkan error: {}", string_VkResult(err)); \
            std::abort(); \
        } \
    } while (0)

#define VK_CHECK_SWAPCHAIN(x) \
    do \
    { \
        VkResult result = x; \
        assert(result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR || result == VK_ERROR_OUT_OF_DATE_KHR); \
    } while (0)
