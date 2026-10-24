import bindings;
import types;

#include "../src/config.h"

struct PushConstants
{
    float4x4 view;
    FroxelAABB* froxels_buffer;
    LightProbeVolume* lpv_buffer;
    PointLight* light_buffer;
    FroxelOffset* froxel_offset_list;
    uint* froxel_item_list;
    uint* light_count_buffer;
    uint light_count;
};

[[vk::push_constant]]
PushConstants pc;

float distance_squared(float3 a, float3 b)
{
    float3 d = a - b;
    return dot(d, d);
}

bool sphere_aabb_intersect(FroxelAABB aabb, float3 center_vs, float radius)
{
    float3 closest = clamp(center_vs, aabb.min.xyz, aabb.max.xyz);

    float d = distance_squared(closest, center_vs);
    return d <= (radius * radius);
}

bool aabb_aabb_intersect(FroxelAABB a, FroxelAABB b)
{
    return (a.min.x <= b.max.x && a.max.x >= b.min.x)
        && (a.min.y <= b.max.y && a.max.y >= b.min.y)
        && (a.min.z <= b.max.z && a.max.z >= b.min.z);
}

#define WG_SIZE_Z 4
#define THREADS (FROXELS_X * FROXELS_Y * WG_SIZE_Z)

groupshared float4 shared_lights[THREADS];

[shader("compute")]
[numthreads(FROXELS_X, FROXELS_Y, WG_SIZE_Z)]
void main(
    uint3 wg_id: SV_GroupID,
    uint thread_id: SV_GroupIndex,
)
{
    uint froxel_index = wg_id.z * THREADS + thread_id;

    FroxelAABB froxel_aabb = pc.froxels_buffer[froxel_index];

    uint visible_light_count = 0;
    uint light_count = pc.light_count;

    // Per thread list
    uint visible_lights[MAX_LIGHTS_PER_FROXEL];

    for (uint light_offset = 0; light_offset < light_count; light_offset = light_offset + THREADS)
    {
        uint batch_size = min(THREADS, light_count - light_offset);

        if (thread_id < batch_size)
        {
            PointLight light = pc.light_buffer[light_offset + thread_id];
            float3 light_vs = mul(pc.view, float4(light.pos, 1.0)).xyz;
            shared_lights[thread_id] = float4(light_vs, light.radius);
        }

        GroupMemoryBarrierWithGroupSync();

        // Note that overflow will be silent
        for (uint i = 0; i < batch_size && visible_light_count < MAX_LIGHTS_PER_FROXEL; i++)
        {
            float4 light_data = shared_lights[i];
            float3 center_vs = light_data.xyz;
            float radius = light_data.w;

            bool intersect = sphere_aabb_intersect(froxel_aabb, center_vs, radius);
            if (intersect)
            {
                // Record into per thread visible light list
                visible_lights[visible_light_count++] = light_offset + i;
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    uint offset;
    InterlockedAdd(*pc.light_count_buffer, visible_light_count, offset);

    pc.froxel_offset_list[froxel_index].offset = offset;
    pc.froxel_offset_list[froxel_index].count = visible_light_count;

    for (uint i = 0; i < visible_light_count; i++)
    {
        pc.froxel_item_list[offset + i] = visible_lights[i];
    }
}
