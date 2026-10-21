import bindings;
import types;
import util;
import prefix_sum;

#include "../src/config.h"

struct PushConstants
{
    GlobalData* global_buffer;
    ObjectData* object_buffer;
    MeshData* mesh_buffer;
    PrefixSums* prefix_sum_buffer;
    float3 camera_pos;
    uint count;
    uint shadowmap_height;
    float lod_distance_factor;
};

[[vk::push_constant]]
PushConstants pc;

bool partially_inside_cascade(float3 center_ws, float radius, uint cascade_index)
{
    float3 center_vs = mul(pc.global_buffer.shadow_views[cascade_index], float4(center_ws, 1.0)).xyz; // light space
    float cascade_radius = pc.global_buffer.shadow_radius[cascade_index];

    bool visible = true;

    visible = visible && ((abs(center_vs.x) - radius) <= cascade_radius);
    visible = visible && ((abs(center_vs.y) - radius) <= cascade_radius);

    // No near plane check as we're using depth clamp/pancaking
    visible = visible && (center_vs.z + radius >= (-2.0 * cascade_radius));

    return visible;
}

[shader("compute")]
[numthreads(256, SUN_CASCADE_COUNT, 1)]
void main(
    uint3 global_id: SV_DispatchThreadID,
    uint3 thread_id: SV_GroupThreadID
)
{
    uint instance_id = global_id.x;
    uint count = pc.count;
    if (instance_id >= count)
        return;

    ObjectData o = pc.object_buffer[instance_id];
    uint mesh_id = o.mesh_id;
    MeshData mesh = pc.mesh_buffer[mesh_id];

    float radius = mesh.radius;
    float max_scale = o.scale;
    radius *= max_scale;

    uint cascade_index = thread_id.y;
    float cascade_radius = pc.global_buffer.shadow_radius[cascade_index];

    float3 center_ws = rotate_quat(mesh.center, o.orientation) * o.scale + o.translation;

    if (!partially_inside_cascade(center_ws, radius, cascade_index))
        return;

    // Uses main camera view for LOD selection
    uint lod_index = 0;
    float d = max(0.0, distance(center_ws.xyz, pc.camera_pos) - radius);
    float e = d * pc.lod_distance_factor / max_scale;

    for (uint i = 0; i < mesh.lod_count; i++)
    {
        lod_index = e >= mesh.mesh_lods[i].error ? i : lod_index;
    }

    MeshLod lod = mesh.mesh_lods[lod_index];
    uint meshlet_count = lod.meshlet_count;
    uint meshlet_offset = lod.meshlet_offset;

    prefix_sum_inclusive_append(pc.prefix_sum_buffer, cascade_index, instance_id, meshlet_count, meshlet_offset);
}
