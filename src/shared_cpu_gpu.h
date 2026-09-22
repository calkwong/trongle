#pragma once

#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/vector_float3.hpp>

#include <cstdint>

using vec3 = glm::vec3;

struct MeshLod
{
    uint32_t first_index;
    uint32_t count;
};

struct MeshData
{
    MeshLod mesh_lods;
};

struct Vertex
{
    vec3 pos;
    float uv_x;

    vec3 normal;
    float uv_y;

    vec3 tangent;
    uint32_t _padding;
};

struct ObjectData
{
    glm::mat4 transform;
};
