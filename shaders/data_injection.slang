import bindings;
import volumetrics;

#include "../src/config.h"

struct PushConstants
{
    float4x4 inverse_view_proj;
    uint3 froxel_dims;
    uint current_frame;
    float2 halton;
    float froxel_near;
    float froxel_far;
    uint noise_tex_id;
    uint froxel_data_tex_id;
    float noise_pos_mult;
    float noise_speed_mult;
    float density_modifier;
    float height_fog_density;
    float height_fog_falloff;
    float box_fog_density;
    float noise_scale;
    float temporal_reprojection_jitter_scale;
    float scattering_factor;
};

[[vk::push_constant]]
PushConstants pc;

// Converts color and density to scattering extinction
float4 scattering_extinction_from_color_density(float3 color, float density)
{
    float extinction = pc.scattering_factor * density;
    return float4(color * extinction, extinction);
}

float sd_box(float3 p, float3 b)
{
    float3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// One thread per froxel
[shader("compute")]
[numthreads(8, 8, 1)]
void main(
    uint3 global_id: SV_DispatchThreadID
)
{
    uint3 froxel_coord = global_id;
    if (any(froxel_coord >= pc.froxel_dims))
        return;

    float3 world_pos = world_from_froxel(
        froxel_coord,
        pc.froxel_dims,
        pc.halton,
        pc.temporal_reprojection_jitter_scale,
        pc.current_frame,
        pc.noise_scale,
        pc.froxel_near,
        pc.froxel_far,
        pc.inverse_view_proj
    );
    float4 scattering_extinction = float4(0);

    // Optional noise to animate fog and break constant density
    // TODO: fix noise speed mult & current_frame
    float3 sampling_coord = world_pos * pc.noise_pos_mult + float3(1, 0.1, 2) * pc.current_frame * pc.noise_speed_mult;

    SamplerState samp = get_sampler<SamplerState>(LINEAR_SAMPLER);
    float fog_noise = get_sampled_image<Texture3D>(pc.noise_tex_id).SampleLevel(samp, sampling_coord, 0).r;
    fog_noise = saturate(fog_noise * fog_noise);

    // Add and accumulate constant fog
    float fog_density = pc.density_modifier * fog_noise;
    scattering_extinction += scattering_extinction_from_color_density(float3(0.5), fog_density);

    // Add and accumulate height fog
    float height_fog = pc.height_fog_density * exp(-pc.height_fog_falloff * max(world_pos.y, 0)) * fog_noise;
    scattering_extinction += scattering_extinction_from_color_density(float3(0.5), height_fog);

    // Add density from a box
    float3 box_position = float3(2.25, 0.8, 0);
    float3 box_size = float3(1);
    float4 box_fog_color = float4(1, 1, 0, 1);

#if 1
    float3 box = abs(world_pos - box_position);
    if (all(box <= box_size))
    {
        scattering_extinction += scattering_extinction_from_color_density(box_fog_color.rgb, pc.box_fog_density * fog_noise);
    }
#else
    float cube_fog = 1.0 - smoothstep(0.0, 0.25, sd_box(float3(world_pos - box_position), float3(0.75)));
    if (cube_fog != 0.0)
    {
        scattering_extinction.rgb += box_fog_color.rgb * cube_fog;
        scattering_extinction.a += cube_fog;
    }
#endif

    get_storage_image<RWTexture3D>(pc.froxel_data_tex_id).Store(froxel_coord, scattering_extinction);
}
