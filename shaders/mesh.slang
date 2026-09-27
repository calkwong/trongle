import types;
import util;
import bindings;

struct Interp
{
    float4 position : SV_Position;
    float3 normal;
    float4 tangent;
    float2 uv;
    nointerpolation uint material_id;
}

struct PushConstants
{
    float4x4 view_proj;
    Vertex* vertex_buffer;
    ObjectData* object_buffer;
    MaterialData* material_buffer;
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
    interp.material_id = o.material_id;

    return interp;
}

[shader("pixel")]
void ps_main(Interp in, out float4 out_color: SV_Target0)
{
    MaterialData m = pc.material_buffer[in.material_id];

    SamplerState samp = get_sampler<SamplerState>(LINEAR_SAMPLER);

    float4 albedo = m.base_color_factor;
    if (m.diffuse_id != INVALID_ID)
    {
        Texture2D tex = get_nonuniform_sampled_image<Texture2D>(m.diffuse_id);
        albedo.xyz *= tex.Sample(samp, in.uv).xyz;
    }

    out_color = albedo;

    //     float perceptual_roughness = m.roughness_factor;
    //     if (m.metal_roughness_id != INVALID_ID)
    //     {
    //         perceptual_roughness *= get_nonuniform_sampled_image<Texture2D>(m.metal_roughness_id).Sample(samp, in.uv).g;
    //     }
    //
    //     out_color = float4(float3(perceptual_roughness), 1.0);
}

