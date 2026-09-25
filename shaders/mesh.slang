import types;
import util;

struct Interp
{
    float4 position : SV_Position;
    float3 normal;
    float4 tangent;
    float2 uv;
}

struct PushConstants
{
    float4x4 view_proj;
    Vertex* vertex_buffer;
    ObjectData* object_buffer;
}

[[vk::push_constant]]
PushConstants pc;

[shader("vertex")]
Interp vs_main(uint instance_id: SV_VulkanInstanceID, uint vertex_id: SV_VulkanVertexID)
{
    Interp interp;

    Vertex v = pc.vertex_buffer[vertex_id];
    ObjectData o = pc.object_buffer[instance_id];

    float3 pos = float3(v.px, v.py, v.pz);
    float4 world_pos = float4(rotate_quat(pos, o.orientation) * o.scale + o.translation, 1.0);
    interp.position = mul(pc.view_proj, world_pos);

    float3 normal;
    float4 tangent;
    unpack_tbn(v.normal, uint(v.tangent), normal, tangent);

    interp.normal = rotate_quat(normal, o.orientation);
    interp.tangent = float4(rotate_quat(tangent.xyz, o.orientation), tangent.w);

    interp.uv = float2(v.uv_x, v.uv_y);

    return interp;
}

[shader("pixel")]
void ps_main(Interp in, out float4 out_color: SV_Target0)
{
    float3 N = normalize(in.normal);
    out_color = float4(N, 1.0);
    out_color.xyz = out_color.xyz * 0.5 + 0.5;
}

