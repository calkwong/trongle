#include "profiler.h"
#include "common.h"

#include <volk.h>
#include <imgui.h>

#include <string>
#include <cstdint>
#include <vector>

void Profiler::add_task(const std::string& task, uint32_t frame_index)
{
    frames[frame_index].tasks.push_back(task);
}

void Profiler::write_timestamp(VkCommandBuffer cmd, uint32_t frame_index)
{
    auto& frame = frames[frame_index];
    auto query = frame.timestamps.size();
    frame.timestamps.push_back(query);
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, frame.query_pool, query);
}

void Profiler::reset(VkDevice device, uint32_t frame_index)
{
    auto& frame = frames[frame_index];

    frame.tasks.clear();
    frame.timestamps.clear();
    frame.profiled_times.clear();

    vkResetQueryPool(device, frame.query_pool, 0, MAX_TIMESTAMP_QUERIES);
}

void Profiler::get_timestamp_results(VkDevice device, uint32_t frame_index, float timestamp_period)
{
    auto& frame = frames[frame_index];
    auto& timestamps = frame.timestamps;
    auto timestamp_size = static_cast<uint32_t>(timestamps.size());
    auto& profiled_times = frame.profiled_times;

    VK_CHECK(vkGetQueryPoolResults(
        device,
        frame.query_pool,
        0,
        timestamp_size,
        timestamp_size * sizeof(uint64_t),
        timestamps.data(),
        sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT
    ));

    double gpu_time = (timestamps[timestamp_size - 1] - timestamps[0]) * timestamp_period * 1e-6;
    last_gpu_time = last_gpu_time * 0.95 + gpu_time * 0.05;
    profiled_times.push_back(last_gpu_time);

    for (auto i = 0; i < frame.tasks.size(); i++)
    {
        profiled_times.push_back((timestamps[(i * 2) + 2] - timestamps[(i * 2) + 1]) * timestamp_period * 1e-6);
    }
}

void Profiler::draw(uint32_t frame_index)
{
    const auto& frame = frames[frame_index];

    ImGui::Begin("Stats");

    ImGui::Text("GPU Render time");
    ImGui::SameLine();
    ImGui::SetCursorPosX(220.0f);
    ImGui::Text("%.3f ms", frame.profiled_times[0]);

    for (auto i = 1; i < frame.profiled_times.size(); i++)
    {
        ImGui::Text("%s", frame.tasks[i-1].c_str());
        ImGui::SameLine();
        ImGui::SetCursorPosX(220.0f);
        ImGui::Text("%.3f ms", frame.profiled_times[i]);
    }

    ImGui::End();
}
