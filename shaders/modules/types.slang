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
    // public uint mesh_id;
    // public uint material_id;
    // uint padding[2];
}

