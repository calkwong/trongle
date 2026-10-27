#include "cs2.h"
#include "shared_cpu_gpu.h"
#include "resources.h"

#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/matrix.hpp>
#include <glm/vector_relational.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <filesystem>
#include <cassert>
#include <algorithm>

namespace
{
const char* WS = " \t\r\n";

std::string trim(const std::string& s)
{
    size_t b = s.find_first_not_of(WS);
    if (b == std::string::npos)
        return "";
    size_t e = s.find_last_not_of(WS);
    return s.substr(b, e - b + 1);
}

bool is_header(const std::string& line)
{
    return line.size() >= 8 && line.compare(0, 4, "====") == 0 && line.compare(line.size() - 4, 4, "====") == 0;
}

std::vector<Entity> parse(std::istream& in)
{
    std::vector<Entity> entities;
    std::string line;

    while (std::getline(in, line))
    {
        std::string t = trim(line);
        if (t.empty())
            continue;

        if (is_header(t))
        {
            entities.emplace_back();
            continue;
        }

        // Defensive
        if (entities.empty())
            entities.emplace_back();

        // Key is the first string, value is everything after it
        size_t keyEnd = t.find_first_of(" \t");
        std::string key = t.substr(0, keyEnd);
        std::string value =
            (keyEnd == std::string::npos) ? "" : trim(t.substr(keyEnd));

        // Multi-line value: opened by """ and closed by a line of exactly """
        if (value == "\"\"\"")
        {
            std::string block = value;
            while (std::getline(in, line))
            {
                std::string text = trim(line);
                block += "\n" + line;
                if (text == "\"\"\"")
                    break;
            }
            value = block;
        }

        entities.back()[key] = value;
    }
    return entities;
}

bool is_light_probe_volume(const Entity& e)
{
    auto it = e.find("classname");
    return (it != e.end()) && (it->second.find("light_probe_volume") != std::string::npos);
}
} // namespace

std::vector<Entity> parse_and_extract_lpvs(std::string entity_file)
{
    std::filesystem::path entity_path = "../assets/" + entity_file;

    std::ifstream f(entity_path);
    std::vector<Entity> probes{};
    std::vector<Entity> entities = parse(f);

    // Light probe volumes reduced to only the lpv_keys fields
    for (const Entity& e : entities)
    {
        if (!is_light_probe_volume(e))
            continue;
        Entity p;
        for (const std::string& k : lpv_keys)
        {
            auto it = e.find(k);
            if (it != e.end())
                p[k] = it->second;
        }
        probes.push_back(std::move(p));
    }

    return probes;
}

std::vector<LightProbeVolume> calculate_light_probe_binding(std::vector<LpvData>& lpv_cpu_data, const Image& lpv_atlas)
{
    std::vector<LightProbeVolume> lpv_gpu_data{};

    // Sort by indoor_outdoor_level, then by volume size
    std::sort(lpv_cpu_data.begin(), lpv_cpu_data.end(), [&](const LpvData& a, const LpvData& b)
              {
                  if (a.indoor_outdoor_level != b.indoor_outdoor_level)
                      return a.indoor_outdoor_level > b.indoor_outdoor_level;

                  auto volume = [&](const LpvData& lpv)
                  {
                      const auto size = lpv.box_maxs - lpv.box_mins;
                      return size[0] * size[1] * size[2];
                  };

                  return volume(a) < volume(b);
              });

    for (const auto& lpv : lpv_cpu_data)
    {
        lpv_gpu_data.emplace_back(calculate_gpu_probe_data(lpv, lpv_atlas.extent.width, lpv_atlas.extent.height, lpv_atlas.extent.depth));
    }

    return lpv_gpu_data;
}

LightProbeVolume calculate_gpu_probe_data(const LpvData& probe_data, uint32_t probe_width, uint32_t probe_height, uint32_t probe_depth)
{
    auto transform = glm::translate(glm::mat4(1.0f), probe_data.origin);

    if (glm::determinant(transform) == 0.0)
        assert(0 && "Matrix is not invertible");

    glm::mat4 world_to_local = glm::inverse(transform);

    if (any(glm::lessThanEqual(probe_data.box_maxs, probe_data.box_mins)))
        assert(0 && "Handle");

    auto local_bb_size = probe_data.box_maxs - probe_data.box_mins;

    LightProbeVolume lpv{};
    lpv.world_to_local_volume_normalized = glm::scale(glm::mat4(1.0f), glm::vec3(1.0) / local_bb_size) * glm::translate(glm::mat4(1.0f), -probe_data.box_mins) * world_to_local;
    auto half = glm::vec3(0.5);
    auto depth_divide = glm::vec3(1.0, 1.0, 1.0 / 6.0);

    auto texture_dims = glm::vec3(probe_width, probe_height, probe_depth) * depth_divide;

    auto border_min = half * depth_divide / probe_data.atlas_size;
    auto border_max = (probe_data.atlas_size - half) * depth_divide / probe_data.atlas_size;
    auto scale = probe_data.atlas_size / texture_dims;
    auto offset = probe_data.atlas_offset / texture_dims;

    lpv.border_min = glm::vec4(border_min, 1.0);
    lpv.border_max = glm::vec4(border_max, 1.0);
    lpv.atlas_scale = glm::vec4(scale, 0);
    lpv.atlas_offset = glm::vec4(offset, 0);

    return lpv;
}

glm::vec3 convert_to_s2(glm::vec3 v)
{
    return glm::vec3(-v.z, -v.x, v.y);
}

glm::vec3 convert_from_s2(glm::vec3 v)
{
    return glm::vec3(-v.y, v.z, -v.x);
}
