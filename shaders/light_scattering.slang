import bindings;
import types;
import util;
import volumetrics;
import shadow_mapping;
import pbr;

#include "../src/config.h"

struct PushConstants
{
    float4x4 inverse_view_proj;
    GlobalData* global_buffer;
    PointLight* light_buffer;
    FroxelOffset* froxel_offset_list;
    uint* froxel_item_list;
    uint3 froxel_dims;
    uint light_scattering_id;
    float3 camera_pos;
    uint froxel_data_tex_id;
    float2 halton;
    float froxel_near;
    float froxel_far;
    float3 sunlight_dir;
    uint current_frame;
    uint shadowmap_id;
    float temporal_reprojection_jitter_scale;
    float noise_scale;
    float light_scale;
    float light_bias;
    uint blue_noise_id;
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
        pc.inverse_view_proj,
        pc.blue_noise_id
    );

    float3 rcp_froxel_dims = float3(1.0) / float3(pc.froxel_dims);

    float4 scattering_extinction = get_sampled_image<Texture3D>(pc.froxel_data_tex_id).SampleLevel(samp, float3(froxel_coord) * rcp_froxel_dims, 0);
    float extinction = scattering_extinction.a;
    float3 lighting = float3(0);

    if (extinction >= 0.01f)
    {
        float3 V = normalize(pc.camera_pos - world_pos);
        float3 L = pc.sunlight_dir;

        // TODO: phase
        float VoL = dot(V, L);

        // TODO: experiment with better PCF
        float shadow_factor = calculate_shadow(world_pos, pc.shadowmap_id, pc.global_buffer);
        float3 sunlight_color = float3(1);
        float sunlight_intensity = 1;
        float3 sunlight = shadow_factor * sunlight_color * sunlight_intensity;
        lighting += sunlight;

#if LOCAL_LIGHTS == 1
        float view_z = slice_to_exponential_depth(pc.froxel_near, pc.froxel_far, froxel_coord.z + 0.5, FOG_DEPTH);
        uint slice = uint(linear_depth_to_exponential_slice(view_z, pc.light_scale, pc.light_bias));

        let x = ceil(float(FOG_WIDTH) / float(FROXELS_X));
        let y = ceil(float(FOG_HEIGHT) / float(FROXELS_Y));
        float2 pixels_per_cluster = float2(x, y);
        uint cluster_index = get_light_cluster_index(float2(froxel_coord.x, pc.froxel_dims.y - froxel_coord.y), pixels_per_cluster, slice);

        FroxelOffset froxel_offset = pc.froxel_offset_list[cluster_index];
        for (int i = 0; i < froxel_offset.count; i++)
        {
            uint index = pc.froxel_item_list[froxel_offset.offset + i];

            PointLight point_light = pc.light_buffer[index];
            float3 light_pos = point_light.pos;
            float light_radius = point_light.radius;

            // Local light shadow ignored for now
            float3 light_to_world = light_pos - world_pos;
            float length_light_to_world = length(light_to_world);
            if (length_light_to_world < light_radius)
            {
                float3 light_color = point_light.color.xyz;
                float attenuation = get_square_falloff_attenuation(light_to_world, light_radius);
                // TODO: phase
                light_color *= attenuation;
                lighting += light_color;
            }
        }
#endif
    }

    float3 scattering = scattering_extinction.rgb * lighting;
    get_storage_image<RWTexture3D>(pc.light_scattering_id).Store(froxel_coord, float4(scattering, extinction));
}
