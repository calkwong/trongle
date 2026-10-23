import bindings;
import sampling;
import pbr;

struct PushConstants
{
    uint src_id;
    uint envmap_id;
    float roughness;
};

[[vk::push_constant]]
PushConstants pc;

static const float PI = 3.14159265359;

[shader("compute")]
[numthreads(8, 8, 6)]
void main(
    uint3 global_id: SV_DispatchThreadID,
)
{
    int2 envmap_size;
    int layers;
    RWTexture2DArray<float4> envmap = get_storage_image<RWTexture2DArray<float4>>(pc.envmap_id);
    envmap.GetDimensions(envmap_size.x, envmap_size.y, layers);

    int2 texel_coord = int2(global_id.xy);
    if (any(texel_coord.xy >= float2(envmap_size)))
        return;

    float2 uv = (float2(texel_coord) + float2(0.5)) / float2(envmap_size);
    uv = uv * 2.0 - 1.0;

    float perceptual_roughness = pc.roughness;
    int2 src_size;
    get_sampled_image<TextureCube>(pc.src_id).GetDimensions(0, src_size.x, src_size.y, layers);

    float3 direction = float3(0.0);
    uint face = global_id.z;

    // +X, -X, +Y, -Y, +Z, -Z
    switch (face)
    {
    case 0:
        direction = normalize(float3(1.0, -uv.y, -uv.x));
        break;
    case 1:
        direction = normalize(float3(-1.0, -uv.y, uv.x));
        break;
    case 2:
        direction = normalize(float3(uv.x, 1.0, uv.y));
        break;
    case 3:
        direction = normalize(float3(uv.x, -1.0, -uv.y));
        break;
    case 4:
        direction = normalize(float3(uv.x, -uv.y, 1.0));
        break;
    case 5:
        direction = normalize(float3(-uv.x, -uv.y, -1.0));
        break;
    default:
        direction = float3(0.0);
        break;
    }

    float3 N = direction;
    float3 V = N;
    float linear_roughness = perceptual_roughness * perceptual_roughness;

    uint sample_count = 32;
    float total_weight = 0.0;
    float3 prefiltered_color = float3(0.0);
    for (uint i = 0; i < sample_count; i++)
    {
        float2 Xi = Hammersley(i, sample_count);
        float3 H = ImportanceSampleGGX(Xi, N, linear_roughness);
        float3 L = normalize(2.0 * dot(V, H) * H - V);

        float NoL = max(dot(N, L), 0.0);
        float NoH = max(dot(N, H), 0.0);
        if (NoL > 0.0)
        {
            // https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling
            // https://chetanjags.wordpress.com/2015/08/26/image-based-lighting/
            float pdf = D_GGX(NoH, linear_roughness) / 4.0 + 0.0001; // NoH and VdotH (from Jacobian) cancel out
            float omegaS = 1.0 / (float(sample_count) * pdf); // solid angle of a sample
            float omegaP = 4.0 * PI / (6.0 * src_size.x * src_size.y); // solid angle of one pixel from cubemap/sphere
            float mipLevel = perceptual_roughness == 0.0 ? 0.0 : max(0.5 * log2(omegaS / omegaP) + 1.0, 0.0);
            SamplerState samp = get_sampler<SamplerState>(CUBE_SAMPLER);
            prefiltered_color += get_sampled_image<TextureCube>(pc.src_id).SampleLevel(samp, L, mipLevel).rgb * NoL;
            total_weight += NoL;
        }
    }

    prefiltered_color = prefiltered_color / total_weight;

    envmap.Store(int3(texel_coord, face), float4(prefiltered_color, 1.0));
}
