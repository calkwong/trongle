import types;
import bindings;
import util;

struct PushConstants
{
    uint2 extent;
    uint debug_id;
    uint color_id;
    uint normal_id;
    uint metal_roughness_id;
    uint output_id;
    uint debug_tex_id;
}

[[vk::push_constant]]
PushConstants pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 global_id: SV_DispatchThreadID)
{
    let tex_coord = global_id.xy;
    if (any(tex_coord >= pc.extent))
        return;

    float4 out_color = float4(0, 0, 0, 1);

    switch (pc.debug_id)
    {
    case 0:
    {
        Texture2D tex = get_sampled_image<Texture2D>(pc.color_id);
        out_color = tex.Load(int3(tex_coord, 0));
        break;
    }
    case 1:
    {
        Texture2D tex = get_sampled_image<Texture2D>(pc.normal_id);
        out_color.xyz = tex.Load(int3(tex_coord, 0)).xyz;
        break;
    }
    case 2:
    {
        Texture2D<float2> tex = get_sampled_image<Texture2D<float2>>(pc.metal_roughness_id);
        float metallic = tex.Load(int3(tex_coord, 0)).r;
        out_color = float4(float3(metallic), 1.0);
        break;
    }
    case 3:
    {
        Texture2D<float2> tex = get_sampled_image<Texture2D<float2>>(pc.metal_roughness_id);
        float roughness = tex.Load(int3(tex_coord, 0)).g;
        out_color = float4(float3(roughness), 1.0);
        break;
    }
    case 4:
    {
        Texture2D<uint> tex = get_sampled_image<Texture2D<uint>>(pc.debug_tex_id);
        uint id = tex.Load(int3(tex_coord, 0)).r;

        // TODO: remove hack
        if (id == 0)
            break;

        uint mhash = hash(id);
        out_color = float4(float3(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255)) / 255.0, 1.0);
        break;
    }
    default:
        break;
    }

    RWTexture2D out = get_storage_image<RWTexture2D>(pc.output_id);
    out.Store(tex_coord, out_color);
}

