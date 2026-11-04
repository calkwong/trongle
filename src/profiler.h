#pragma once

#include "config.h"

#include <volk.h>

#include <vector>
#include <cstdint>
#include <string>

#define MAX_TIMESTAMP_QUERIES 100

struct Profiler
{
    struct FrameData
    {
        std::vector<std::string> tasks;
        std::vector<uint64_t> timestamps;
        std::vector<double> profiled_times;
        VkQueryPool query_pool;
    };

    FrameData frames[FRAMES_IN_FLIGHT];
    double last_gpu_time;

    void add_task(const std::string& task, uint32_t frame_index);
    void write_timestamp(VkCommandBuffer cmd, uint32_t frame_index);
    void reset(VkDevice device, uint32_t frame_index);
    void get_timestamp_results(VkDevice device, uint32_t frame_index, float timestamp_period);
    void draw(uint32_t frame_index);
};
