import bindings;

struct PushConstants
{
    float2 image_size;
    uint src_id;
    uint cubemap_id;
};

[[vk::push_constant]]
PushConstants pc;

static const float PI = 3.14159265359;

float2 sample_equirectangular_map(float3 dir)
{
    float u = atan2(dir.z, dir.x);
    float v = asin(dir.y);

    u = u / (2.0 * PI) + 0.5;
    v = v / PI + 0.5;

    v = 1.0 - v;

    return float2(u, v);
}

[shader("compute")]
[numthreads(32, 32, 1)]
void main(
    uint3 global_id: SV_DispatchThreadID,
    uint3 wg_id: SV_GroupID
)
{
    int2 cubemap_size;
    int layers;
    RWTexture2DArray<float4> cubemap = get_storage_image<RWTexture2DArray<float4>>(pc.cubemap_id);
    cubemap.GetDimensions(cubemap_size.x, cubemap_size.y, layers);

    int2 texel_coord = int2(global_id.xy);
    if (any(texel_coord.xy >= float2(cubemap_size)))
        return;

    float2 uv = (float2(texel_coord) + float2(0.5)) / pc.image_size;
    uv = uv * 2.0 - 1.0;
    uv.y *= -1.0;

    float3 direction = float3(0.0);
    uint face = wg_id.z;

    // +X, -X, +Y, -Y, +Z, -Z
    switch (face)
    {
    case 0:
        direction = normalize(float3(1.0, uv.y, -uv.x));
        break;
    case 1:
        direction = normalize(float3(-1.0, uv.y, uv.x));
        break;
    case 2:
        direction = normalize(float3(uv.x, 1.0, -uv.y));
        break;
    case 3:
        direction = normalize(float3(uv.x, -1.0, uv.y));
        break;
    case 4:
        direction = normalize(float3(uv.x, uv.y, 1.0));
        break;
    case 5:
        direction = normalize(float3(-uv.x, uv.y, -1.0));
        break;
    default:
        direction = float3(0.0);
        break;
    }

    float2 sample_uv = sample_equirectangular_map(direction);
    SamplerState samp = get_sampler<SamplerState>(LINEAR_SAMPLER);
    float4 color = get_sampled_image<Texture2D>(pc.src_id).SampleLevel(samp, sample_uv, 0.0);
    RWTexture2DArray<float4> tex = get_storage_image<RWTexture2DArray<float4>>(pc.cubemap_id);
    tex.Store(int3(texel_coord, face), color);
}
