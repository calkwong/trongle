#pragma once

#include <glm/ext/vector_float4.hpp>
#include <glm/ext/quaternion_float.hpp>
#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/vector_float3.hpp>

#include <cstdint>
#include <array>

using vec3 = glm::vec3;

struct Vertex
{
    uint16_t px;
    uint16_t py;
    uint16_t pz;
    uint16_t tangent;

    uint32_t normal;

    uint16_t uv_x;
    uint16_t uv_y;
};

struct ObjectData
{
    vec3 translation;
    float scale;
    glm::quat orientation;

    uint32_t mesh_id;
    uint32_t material_id;
};

struct MaterialData
{
    glm::vec4 base_color_factor = glm::vec4(1.0f);

    float metallic_factor = 1.0f;
    float roughness_factor = 1.0f;

    uint32_t diffuse_id = -1;
    uint32_t metal_roughness_id = -1;

    vec3 emissive_factor;
    uint32_t normal_id = -1;

    uint32_t emissive_id = -1;
    uint32_t occlusion_id = -1;
};

struct MeshLod
{
    uint32_t first_index;
    uint32_t index_count;
    float error;
};

struct MeshData
{
    vec3 center;
    float radius;

    uint32_t lod_count;
    uint32_t vertex_offset;

    std::array<MeshLod, 8> mesh_lods;
};

struct DrawIndirect
{
    uint32_t index_count;
    uint32_t instance_count;
    uint32_t first_index;
    int vertex_offset;
    uint32_t first_instance;
};

struct Dispatch
{
    uint32_t x;
    uint32_t y;
    uint32_t z;
};
