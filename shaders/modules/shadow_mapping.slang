module shadow_mapping;

import types;
import bindings;

#include "../../src/config.h"

public float calculate_shadow(float3 world_pos, uint shadowmap_id, GlobalData* global_buffer)
{
    float3 ndc;
    int cascade_index = 0;

    for (int i = 0; i < SUN_CASCADE_COUNT; i++)
    {
        ndc = mul(global_buffer.shadow_matrices[i], float4(world_pos, 1.0)).xyz;

        // Z plane check may require tweaking based on cascade selection strategy and potentially if we're depth clamping
        if (abs(ndc.x) < 1.0 && abs(ndc.y) < 1.0 && ndc.z > 0.0 && ndc.z < 1.0)
        {
            cascade_index = i;
            break;
        }
    }

    float current_depth = ndc.z;

    // Anything beyond far plane is considered lit
    if (current_depth < 0.0)
        return 1.0;

    float2 uv = ndc.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;

    float2 offset = float2(1.0) / float2(SHADOWMAP_EXTENT);

    float sum = 0.0;
    SamplerComparisonState samp = get_sampler<SamplerComparisonState>(SHADOW_SAMPLER);

    // TODO: bias + better sampling
    for (int y = -1; y <= 1; y++)
    {
        for (int x = -1; x <= 1; x++)
        {
            float2 sample_uv = float2(uv.x + x * offset.x, uv.y + y * offset.y);
            sum += get_nonuniform_sampled_image<Texture2D>(shadowmap_id + cascade_index).SampleCmpLevelZero(samp, sample_uv, current_depth).r;
        }
    }

    sum /= 9.0;
    return sum;
}
