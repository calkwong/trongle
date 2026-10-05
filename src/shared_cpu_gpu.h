#pragma once

// clang-format off
#ifdef __cplusplus
    #include <glm/ext/vector_float4.hpp>
    #include <glm/ext/quaternion_float.hpp>
    #include <glm/ext/vector_float3.hpp>
    #include <cstdint>

    #define PUBLIC_ACCESS
    using float3 = glm::vec3;
    using float4 = glm::vec4;
#else // Slang
    #define PUBLIC_ACCESS public
#endif
// clang-format on

PUBLIC_ACCESS struct Vertex
{
#ifdef __cplusplus
    uint16_t px;
    uint16_t py;
    uint16_t pz;
#else
    float16_t px;
    float16_t py;
    float16_t pz;
#endif

    uint16_t tangent;
    uint32_t normal;

#ifdef __cplusplus
    uint16_t uv_x;
    uint16_t uv_y;
#else
    float16_t uv_x;
    float16_t uv_y;
#endif
};

PUBLIC_ACCESS struct ObjectData
{
    float3 translation;
    float scale;
    float4 orientation;

    uint32_t mesh_id;
    uint32_t material_id;
};

PUBLIC_ACCESS struct MaterialData
{
    float4 base_color_factor = float4(1.0f);

    float metallic_factor = 1.0f;
    float roughness_factor = 1.0f;

    uint32_t diffuse_id = -1;
    uint32_t metal_roughness_id = -1;

    float3 emissive_factor;
    uint32_t normal_id = -1;

    uint32_t emissive_id = -1;
    uint32_t occlusion_id = -1;
};

PUBLIC_ACCESS struct MeshLod
{
    uint32_t first_index;
    uint32_t index_count;
    float error;

    uint32_t meshlet_offset; // Offset into meshlet buffer
    uint32_t meshlet_count;
};

PUBLIC_ACCESS struct MeshData
{
    float3 center;
    float radius;

    uint32_t lod_count;

    MeshLod mesh_lods[8];
};

PUBLIC_ACCESS struct DrawIndirect
{
    uint32_t meshlet_id;
    uint32_t instance_id;
};

PUBLIC_ACCESS struct Dispatch
{
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

PUBLIC_ACCESS struct Meshlet
{
#ifdef __cplusplus
    uint16_t cx, cy, cz;
    uint16_t radius;
#else
    float16_t cx, cy, cz;
    float16_t radius;
#endif

    uint32_t base_vertex;
    uint32_t data_offset;

    uint8_t vertex_count;
    uint8_t triangle_count;
    uint16_t padding;
};
