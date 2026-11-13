module volumetrics;

import bindings;
import util;

#include "../../src/config.h"

public float2 uv_from_froxels(float2 froxel_coord, uint width, uint height)
{
    return froxel_coord / float2(width, height);
}

// Output is in range [0,1].
// Blue noise texture size is 128x128.
float2 animated_blue_noise_golden_ratio(uint2 pixel, uint frame, uint blue_noise_id)
{
    Texture2D blue_noise_tex = get_sampled_image<Texture2D>(blue_noise_id);
    pixel = pixel % 128;
    float2 blue_noise = blue_noise_tex.Load(int3(pixel, 0)).rg;

    const float golden_ratio = 0.61803398875;
    frame = frame % 64;
    float blue_noise0 = frac((blue_noise.r) + float(frame) * golden_ratio);
    float blue_noise1 = frac((blue_noise.g) + float(frame) * golden_ratio);

    return float2(blue_noise0, blue_noise1);
}

public float generate_noise(uint2 pixel, uint frame, float noise_scale, uint blue_noise_id)
{
#if 1
    float2 blue_noise = animated_blue_noise_golden_ratio(pixel, frame, blue_noise_id);
    return triangular_noise(blue_noise.x, blue_noise.y) * noise_scale;
#else
    return (interleaved_gradient_noise(pixel, frame) * noise_scale) - (noise_scale * 0.5f);
#endif
}

// https://advances.realtimerendering.com/s2016/Siggraph2016_idTech6.pdf
public float slice_to_exponential_depth(float near, float far, float slice, float num_slices)
{
    return near * pow(far / near, slice / num_slices);
}

// Reconstructs world position from froxel center + jitter/noise
public float3 world_from_froxel(
    uint3 froxel_coord,
    uint3 froxel_dimensions,
    float2 halton,
    float temporal_reprojection_jitter_scale,
    uint current_frame,
    float noise_scale,
    float froxel_near,
    float froxel_far,
    float4x4 inverse_view_proj,
    uint blue_noise_id
)
{
    float jitter = generate_noise(froxel_coord.xy, current_frame, noise_scale, blue_noise_id);
    // jitter = 0.0;
    // halton = float2(0);

    float2 uv = uv_from_froxels(froxel_coord.xy + float2(0.5) + halton * temporal_reprojection_jitter_scale, froxel_dimensions.x, froxel_dimensions.y);
    float linear_depth = slice_to_exponential_depth(froxel_near, froxel_far, float(froxel_coord.z) + 0.5 + jitter, float(froxel_dimensions.z));
    float depth = linear_depth_to_raw_depth(froxel_near, linear_depth);

    return reconstruct_world_pos(depth, inverse_view_proj, uv);
}

// Reconstructs world position from froxel center
public float3 world_from_froxel_no_jitter(uint3 froxel_coord, uint3 froxel_dimensions, float froxel_near, float froxel_far, float4x4 inverse_view_proj)
{
    float2 uv = uv_from_froxels(froxel_coord.xy + float2(0.5), froxel_dimensions.x, froxel_dimensions.y);
    float linear_depth = slice_to_exponential_depth(froxel_near, froxel_far, float(froxel_coord.z) + 0.5, float(froxel_dimensions.z));
    float depth = linear_depth_to_raw_depth(froxel_near, linear_depth);

    return reconstruct_world_pos(depth, inverse_view_proj, uv);
}

public float3 apply_volumetric_fog(
    float2 screen_uv,
    float near,
    float far,
    float scale,
    float bias,
    float depth,
    float3 color,
    uint fog_texture_id,
    SamplerState samp
)
{
    float linear_depth = linearize_depth(near, depth);
    float depth_uv = linear_depth_to_exponential_slice(linear_depth, scale, bias) / float(FOG_DEPTH);

    float3 froxel_uvw = float3(screen_uv.xy, depth_uv);
    float4 scattering_transmittance = get_sampled_image<Texture3D>(fog_texture_id).SampleLevel(samp, froxel_uvw, 0);

    color.rgb = color.rgb * scattering_transmittance.a + scattering_transmittance.rgb;
    return color;
}
