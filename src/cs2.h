#pragma once

#include "shared_cpu_gpu.h"
#include "resources.h"

#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_uint3.hpp>
#include <glm/ext/matrix_float4x4.hpp>

#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

using Entity = std::unordered_map<std::string, std::string>;

const std::vector<std::string> lpv_keys = {
    "box_maxs",
    "box_mins",
    "origin",
    "indoor_outdoor_level",
    "light_probe_size_x",
    "light_probe_size_y",
    "light_probe_size_z",
    "light_probe_atlas_x",
    "light_probe_atlas_y",
    "light_probe_atlas_z",
};

struct LpvData
{
    glm::vec3 box_maxs;
    glm::vec3 box_mins;
    glm::vec3 box_center;
    glm::vec3 origin;
    glm::vec3 atlas_size;
    glm::vec3 atlas_offset;
    int indoor_outdoor_level;
    uint32_t shader_index;
};

std::vector<Entity> parse_and_extract_lpvs(std::string entity_file);
LightProbeVolume calculate_gpu_probe_data(const LpvData& probe_data, uint32_t probe_width, uint32_t probe_height, uint32_t probe_depth);
std::vector<LightProbeVolume> calculate_light_probe_binding(std::vector<LpvData>& lpv_cpu_data, const Image& lpv_atlas);
glm::vec3 convert_to_s2(glm::vec3 v);
glm::vec3 convert_from_s2(glm::vec3 v);
