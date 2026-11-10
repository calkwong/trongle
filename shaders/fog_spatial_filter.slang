import bindings;

struct PushConstants
{
    uint3 froxel_dims;
    uint input_id;
    uint output_id;
    uint horizontal;
};

[[vk::push_constant]]
PushConstants pc;

#define SIGMA_FILTER 4.0
#define RADIUS 2

float gaussian(float radius, float sigma)
{
    float v = radius / sigma;
    return exp(-(v * v));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(
    uint3 global_id: SV_DispatchThreadID
)
{
    uint3 froxel_coord = global_id;

    if (any(froxel_coord >= pc.froxel_dims))
        return;

    Texture3D light_scattering_tex = get_sampled_image<Texture3D>(pc.input_id);

    float accumulated_weight = 0.0;
    float4 accumulated_scattering_extinction = float4(0.0);

    for (int i = -RADIUS; i <= RADIUS; i++)
    {
        int3 offset = pc.horizontal == 1 ? int3(i, 0, 0) : int3(0, i, 0);
        int3 coord = int3(froxel_coord) + offset;

        if (all(coord >= 0) && all(coord < int3(pc.froxel_dims)))
        {
            float weight = gaussian(length(offset.xy), SIGMA_FILTER);
            float4 sampled_value = light_scattering_tex.Load(int4(coord, 0));
            accumulated_scattering_extinction += sampled_value * weight;
            accumulated_weight += weight;
        }
    }

    float4 scattering_extinction = accumulated_scattering_extinction / accumulated_weight;

    RWTexture3D scattering_extinction_tex = get_storage_image<RWTexture3D>(pc.output_id);
    scattering_extinction_tex.Store(froxel_coord, scattering_extinction);
}
