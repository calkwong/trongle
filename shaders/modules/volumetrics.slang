module volumetrics;

import bindings;
import util;

#include "../../src/config.h"

public float2 uv_from_froxels(float2 froxel_coord, uint width, uint height)
{
    return froxel_coord / float2(width, height);
}

// TODO: use better noise?
public float generate_noise(uint2 pixel, uint frame, float noise_scale)
{
    return (interleaved_gradient_noise(pixel, frame) * noise_scale) - (noise_scale * 0.5f);
}

// https://advances.realtimerendering.com/s2016/Siggraph2016_idTech6.pdf
public float slice_to_exponential_depth(float near, float far, float slice, float num_slices)
{
    return near * pow(far / near, slice / num_slices);
}

// This will provide jittered world position (first adjusted for froxel center) for a given froxel index
public float3 world_from_froxel(
    uint3 froxel_coord,
    uint3 froxel_dimensions,
    float2 halton_xy,
    float temporal_reprojection_jitter_scale,
    uint current_frame,
    float noise_scale,
    float froxel_near,
    float froxel_far,
    float4x4 inverse_view_proj
)
{
    // TODO: use halton
    halton_xy = float2(0);
    float2 uv = uv_from_froxels(froxel_coord.xy + float2(0.5) + halton_xy * temporal_reprojection_jitter_scale, froxel_dimensions.x, froxel_dimensions.y);

    float jitter = generate_noise(froxel_coord.xy, current_frame, noise_scale);
    // TODO: use jittering
    jitter = 0.0;
    float linear_depth = slice_to_exponential_depth(froxel_near, froxel_far, float(froxel_coord.z) + 0.5 + jitter, float(froxel_dimensions.z));
    float depth = linear_depth_to_raw_depth(froxel_near, linear_depth);

    return reconstruct_world_pos(depth, inverse_view_proj, uv);
}

public float3 apply_volumetric_fog(float2 screen_uv, float near, float far, float scale, float bias, float depth, float3 color, uint fog_texture_id)
{
    float linear_depth = linearize_depth(near, depth);
    float depth_uv = linear_depth_to_exponential_slice(linear_depth, scale, bias) / float(FOG_DEPTH);

    float3 froxel_uvw = float3(screen_uv.xy, depth_uv);
    SamplerState samp = get_sampler<SamplerState>(LINEAR_CLAMP_SAMPLER);
    float4 scattering_transmittance = get_sampled_image<Texture3D>(fog_texture_id).SampleLevel(samp, froxel_uvw, 0);

    // TODO: opacity anti aliasing?
    color.rgb = color.rgb * scattering_transmittance.a + scattering_transmittance.rgb;
    return color;
}
