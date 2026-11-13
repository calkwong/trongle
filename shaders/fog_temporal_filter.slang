import bindings;
import volumetrics;
import util;

#include "../src/config.h"

struct PushConstants
{
    float4x4 inverse_view_proj;
    float4x4 previous_view_proj;
    uint3 froxel_dims;
    uint first_frame;
    float2 halton;
    float temporal_reprojection_jitter_scale;
    float noise_scale;
    float near;
    float far;
    uint raw_scattering_id;
    uint temporal_previous_id;
    uint temporal_current_id;
    float scale;
    float bias;
    float reprojection_factor;
    uint blue_noise_id;
    uint current_frame;
};

[[vk::push_constant]]
PushConstants pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(
    uint3 global_id: SV_DispatchThreadID
)
{
    uint3 froxel_coord = global_id;

    if (any(froxel_coord >= pc.froxel_dims))
        return;

    SamplerState samp = get_sampler<SamplerState>(LINEAR_CLAMP_SAMPLER);
    Texture3D temporal_previous_tex = get_sampled_image<Texture3D>(pc.temporal_previous_id);
    Texture3D raw_scattering_tex = get_sampled_image<Texture3D>(pc.raw_scattering_id);
    RWTexture3D<float4> temporal_current_tex = get_storage_image<RWTexture3D>(pc.temporal_current_id);

    float3 uvw = (float3(froxel_coord) + float3(0.5)) / float3(pc.froxel_dims);

    float4 scattering_extinction = raw_scattering_tex.SampleLevel(samp, uvw, 0.0);
    if (pc.first_frame == 1)
    {
        temporal_current_tex.Store(froxel_coord, scattering_extinction);
        return;
    }

    float3 world_pos_no_jitter = world_from_froxel_no_jitter(froxel_coord, pc.froxel_dims, pc.near, pc.far, pc.inverse_view_proj);
    // float3 world_pos_no_jitter = world_from_froxel(froxel_coord, pc.froxel_dims, pc.halton, pc.temporal_reprojection_jitter_scale, pc.current_frame, pc.noise_scale, pc.near, pc.far, pc.inverse_view_proj);

    float4 clip_pos = mul(pc.previous_view_proj, float4(world_pos_no_jitter, 1.0));
    float3 ndc = clip_pos.xyz / clip_pos.w;
    float depth = ndc.z;

    float linear_depth = linearize_depth(pc.near, depth);
    // Do not clamp slice, we want a continuous slice here
    float slice = linear_depth_to_exponential_slice(linear_depth, pc.scale, pc.bias);
    float w = slice / float(FOG_DEPTH);

    float2 history_uv = ndc.xy * float2(0.5, -0.5) + float2(0.5);
    float3 history_uvw = float3(history_uv, w);

    if (all(history_uvw >= 0.0) && all(history_uvw <= 1.0))
    {
        float4 history = temporal_previous_tex.SampleLevel(samp, history_uvw, 0.0);
        // TODO: Proper neighbourhood clamp? Leaving naive clamp disabled yields better result and removes some flickering on fog volume edges
        // history = max(history, scattering_extinction);
        scattering_extinction.rgb = lerp(history.rgb, scattering_extinction.rgb, pc.reprojection_factor);
        scattering_extinction.a = lerp(history.a, scattering_extinction.a, pc.reprojection_factor);
    }

    temporal_current_tex.Store(froxel_coord, scattering_extinction);
}
