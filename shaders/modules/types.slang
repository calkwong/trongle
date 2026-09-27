module types;

public struct Vertex
{
    public float16_t px;
    public float16_t py;
    public float16_t pz;
    public uint16_t tangent;

    public uint32_t normal;

    public float16_t uv_x;
    public float16_t uv_y;
};

// TODO: fix
public struct MeshLod
{
    public uint first_index;
    public uint count;
}

public struct ObjectData
{
    public float3 translation;
    public float scale;
    public float4 orientation;

    public uint mesh_id;
    public uint material_id;
}

public struct MaterialData
{
    public float4 base_color_factor;

    public float metallic_factor;
    public float roughness_factor;
    public uint32_t diffuse_id;
    public uint32_t metal_roughness_id;

    public uint32_t normal_id;
    public float3 emissive_factor;

    public uint32_t emissive_id;
    public uint32_t occlusion_id;
}

