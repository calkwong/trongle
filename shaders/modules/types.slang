module types;

public struct Vertex
{
    public float3 pos;
    public float uv_x;
    public float3 normal;
    public float uv_y;
    public float3 tangent;
    public uint _padding;
}

public struct MeshLod
{
    public uint first_index;
    public uint count;
}

public struct ObjectData
{
    public float4x4 world_transform;
    // public uint mesh_id;
    // public uint material_id;
    // uint padding[2];
}

