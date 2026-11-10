import bindings;
import util;
import volumetrics;

#include "../src/config.h"

struct PushConstants
{
    float4x4 inverse_view_proj;
    uint3 froxel_dims;
    uint light_scattering_id;
    float3 camera_pos;
    uint integrated_light_scattering_id;
    float froxel_near;
    float froxel_far;
};

[[vk::push_constant]]
PushConstants pc;

// Dispatch with workgroup z = 1 as we perform integration
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

    float3 integrated_scattering = float3(0);
    float integrated_transmittance = 1.0;

    float3 rcp_froxel_dims = float3(1.0) / float3(pc.froxel_dims);
    uint froxel_dimension_z = pc.froxel_dims.z;
    float3 sampling_coord = (float3(froxel_coord) + float3(0.5)) * rcp_froxel_dims;

    float current_z = slice_to_exponential_depth(pc.froxel_near, pc.froxel_far, float(0), float(froxel_dimension_z));
    float raw_depth = linear_depth_to_raw_depth(pc.froxel_near, current_z);
    float3 current_pos_ws = reconstruct_world_pos(raw_depth, pc.inverse_view_proj, sampling_coord.xy);

    for (int z = 0; z < froxel_dimension_z; z++)
    {
        froxel_coord.z = z;
        sampling_coord.z = float(z) / FOG_DEPTH;

        // TODO: apply jitter?
        float jitter = 0.0;
        float next_z = slice_to_exponential_depth(pc.froxel_near, pc.froxel_far, float(z + 1), float(froxel_dimension_z));
        raw_depth = linear_depth_to_raw_depth(pc.froxel_near, next_z);
        float3 next_pos_ws = reconstruct_world_pos(raw_depth, pc.inverse_view_proj, sampling_coord.xy);
        float z_step = length(next_pos_ws - current_pos_ws);
        current_pos_ws = next_pos_ws;

        // Following equations from Physically Based Sky, Atmosphere and Cloud Rendering by Hillaire
        float4 sampled_scattering_extinction = get_sampled_image<Texture3D>(pc.light_scattering_id).SampleLevel(samp, sampling_coord, 0);
        float3 sampled_scattering = sampled_scattering_extinction.xyz;
        float sampled_extinction = sampled_scattering_extinction.w;
        float clamped_extinction = max(sampled_extinction, 0.00001f);

        float transmittance = exp(-sampled_extinction * z_step);
        float3 scattering = (sampled_scattering - (sampled_scattering * transmittance)) / clamped_extinction;

        integrated_scattering += scattering * integrated_transmittance;
        integrated_transmittance *= transmittance;

        // TODO: opacity anti aliasing?

        get_storage_image<RWTexture3D>(pc.integrated_light_scattering_id).Store(froxel_coord, float4(integrated_scattering, integrated_transmittance));
    }
}
