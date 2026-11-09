import bindings;
import types;
import util;
import volumetrics;
import shadow_mapping;

#include "../src/config.h"

struct PushConstants
{
    float4x4 inverse_view_proj;
    uint3 froxel_dims;
    uint light_scattering_id;
    float3 camera_pos;
    uint froxel_data_tex_id;
    float2 halton;
    float froxel_near;
    float froxel_far;
    float3 sunlight_dir;
    uint current_frame;
    GlobalData* global_buffer;
    uint shadowmap_id;
    float temporal_reprojection_jitter_scale;
    float noise_scale;
};

[[vk::push_constant]]
PushConstants pc;

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

    SamplerState samp = get_sampler<SamplerState>(LINEAR_CLAMP_SAMPLER);

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

    float3 rcp_froxel_dims = float3(1.0) / float3(pc.froxel_dims);

    float4 scattering_extinction = get_sampled_image<Texture3D>(pc.froxel_data_tex_id).SampleLevel(samp, float3(froxel_coord) * rcp_froxel_dims, 0);
    float extinction = scattering_extinction.a;
    float3 lighting = float3(0);

    if (extinction >= 0.01f)
    {
        float3 V = normalize(pc.camera_pos - world_pos);
        float3 L = pc.sunlight_dir;
        float VoL = dot(V, L);

        float shadow_factor = calculate_shadow(world_pos, pc.shadowmap_id, pc.global_buffer);
        float3 sunlight_color = float3(1);
        float sunlight_intensity = 1;
        float3 sunlight = shadow_factor * sunlight_color * sunlight_intensity;
        lighting += sunlight;

        // TODO: handle local lights
    }

    float3 scattering = scattering_extinction.rgb * lighting;
    get_storage_image<RWTexture3D>(pc.light_scattering_id).Store(froxel_coord, float4(scattering, extinction));
}
