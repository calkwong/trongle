#pragma once

#include "config.h"

#include <volk.h>

#include <vector>
#include <cstdint>
#include <string>

struct Profiler
{
    struct FrameData
    {
        std::vector<std::string> tasks;
        std::vector<uint64_t> timestamps;
        std::vector<double> profiled_times;
    };

    FrameData frames[FRAMES_IN_FLIGHT];
    double last_gpu_time;

    void add_task(const std::string& task, uint32_t frame_index);
    void write_timestamp(VkCommandBuffer cmd, VkQueryPool query_pool, uint32_t frame_index);
    void reset(uint32_t frame_index);
    void get_timestamp_results(VkDevice device, VkQueryPool query_pool, uint32_t frame_index, float timestamp_period);
    void draw(uint32_t frame_index);
};
