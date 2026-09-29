module types;

#language slang 2026

public struct Vertex
{
    float16_t px;
    float16_t py;
    float16_t pz;
    uint16_t tangent;
    uint32_t normal;
    float16_t uv_x;
    float16_t uv_y;
};

public struct ObjectData
{
    float3 translation;
    float scale;
    float4 orientation;

    uint mesh_id;
    uint material_id;
}

public struct MaterialData
{
    float4 base_color_factor;

    float metallic_factor;
    float roughness_factor;
    uint32_t diffuse_id;
    uint32_t metal_roughness_id;

    float3 emissive_factor;
    uint32_t normal_id;

    uint32_t emissive_id;
    uint32_t occlusion_id;
}

public struct MeshLod
{
    uint32_t first_index;
    uint32_t index_count;
    float error;
}

public struct MeshData
{
    float3 center;
    float radius;

    uint32_t lod_count;
    uint32_t vertex_offset;

    MeshLod mesh_lods[8];
}

public struct DrawIndirect
{
    uint index_count;
    uint instance_count;
    uint first_index;
    int vertex_offset;
    uint first_instance;
}

public struct Dispatch
{
    uint x;
    uint y;
    uint z;
}

