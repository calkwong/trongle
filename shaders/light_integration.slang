import bindings;
import util;
import volumetrics;

#include "../src/config.h"

struct PushConstants
{
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

    // TODO: use froxel_near?
    float current_z = 0;

    float3 rcp_froxel_dims = float3(1.0) / float3(pc.froxel_dims);
    uint froxel_dimension_z = pc.froxel_dims.z;

    for (int z = 0; z < froxel_dimension_z; z++)
    {
        froxel_coord.z = z;

        // TODO: apply jitter?
        float jitter = 0.0;
        float next_z = slice_to_exponential_depth_jittered(pc.froxel_near, pc.froxel_far, jitter, z + 1, froxel_dimension_z);

        float z_step = abs(next_z - current_z);
        current_z = next_z;

        // Following equations from Physically Based Sky, Atmosphere and Cloud Rendering by Hillaire
        float4 sampled_scattering_extinction = get_sampled_image<Texture3D>(pc.light_scattering_id).SampleLevel(samp, float3(froxel_coord) * rcp_froxel_dims, 0);
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
