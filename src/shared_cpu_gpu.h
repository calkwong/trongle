#pragma once

#include <glm/ext/quaternion_float.hpp>
#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/vector_float3.hpp>

#include <cstdint>

using vec3 = glm::vec3;

struct MeshData
{
    uint32_t first_index;
    uint32_t index_count;
    uint32_t vertex_offset;
};

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
};
