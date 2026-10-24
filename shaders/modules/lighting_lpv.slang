// https://github.com/ValveResourceFormat/ValveResourceFormat/tree/master/Renderer/Shaders

module lighting_lpv;

import bindings;
import types;

public struct LightProbeSample_t
{
    float3 irradiance;
    float coverage;
}

// Replaces call to ComputeLightProbeShading
public float3 SampleLightProbeIrradiance(LightProbeVolume* lpvs, float3 pos_ws, float3 normal, uint probe_id)
{
    LightProbeSample_t probe = SampleLightProbeVolumes(lpvs, pos_ws, normal, probe_id);

    // Sky from above and its ground bounce from below, for what no volume covers
    float sky_weight = 1.0 - probe.coverage;

    // SKY_FILLS_UNCOVERED path not currently supported
    return probe.irradiance + GetSkyAmbientLighting(normal) * sky_weight;
}

public LightProbeSample_t SampleLightProbeVolumes(LightProbeVolume* lpvs, float3 pos_ws, float3 normal, uint probe_id)
{
    LightProbeSample_t probe;
    probe.irradiance = float3(0);
    probe.coverage = 0.0f;

    // Probe blending not currently supported
    // TODO: look up froxel for our volume
    uint n_volume = 0;
    AddLightProbeVolume(probe, lpvs[n_volume], CalculateProbeSampleCoords(lpvs[n_volume].world_to_local_volume_normalized, pos_ws), normal, 1.0, probe_id);

    return probe;
}

public void AddLightProbeVolume(inout LightProbeSample_t probe, LightProbeVolume lpv, float3 local_coords, float3 normal, float weight, uint probe_id)
{
    probe.irradiance += SampleProbeAmbientCube(CalculateProbeIndirectCoords(lpv, local_coords), normal, probe_id) * weight;
    probe.coverage += weight;
}

public float3 CalculateProbeSampleCoords(float4x4 world_to_local_volume_normalized, float3 frag_pos)
{
    float3 light_probe_local_pos = mul(world_to_local_volume_normalized, float4(frag_pos, 1.0)).xyz;
    return light_probe_local_pos;
}

public float3 CalculateProbeIndirectCoords(LightProbeVolume lpv, float3 indirect_coords)
{
    indirect_coords.z /= 6;
    indirect_coords = clamp(indirect_coords, lpv.border_min.xyz, lpv.border_max.xyz);

    indirect_coords.z *= 6;
    indirect_coords = fma(indirect_coords, lpv.atlas_scale.xyz, lpv.atlas_offset.xyz);

    indirect_coords.z /= 6;

    return indirect_coords;
}

public float3 SampleProbeAmbientCube(float3 indirect_sample_coords, float3 normal, uint probe_id)
{
    // Take up to 3 samples along the normal direction
    float3 depth_slice_offsets = lerp(float3(0, 1, 2) / 6.0, float3(3, 4, 5) / 6.0, step(normal, float3(0.0)));
    float3 ambient[3];

    float3 normal_squared = normal * normal;

    float3 indirect_diffuse = float3(0.0);

    SamplerState tex_3d_samp = get_sampler<SamplerState>(CUBE_SAMPLER);

    for (int i = 0; i < 3; i++)
    {
        ambient[i] = get_sampled_image<Texture3D>(probe_id).SampleLevel(tex_3d_samp, indirect_sample_coords + float3(0, 0, depth_slice_offsets[i]), 0.0).rgb;
        indirect_diffuse += ambient[i] * normal_squared[i];
    }

    return indirect_diffuse;
}

// TODO: spherical harmonics
public float3 GetSkyAmbientLighting(float3 direction)
{
    return float3(0.0);
}
