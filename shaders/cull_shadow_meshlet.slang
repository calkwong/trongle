import bindings;
import types;
import util;
import prefix_sum;

#include "../src/config.h"

struct PushConstants
{
    GlobalData* global_buffer;
    ObjectData* object_buffer;
    Meshlet* meshlet_buffer;
    PrefixSums* prefix_sum_buffer;
    Dispatch* dispatch_buffer;
    uint* meshlet_draw_buffer;
    uint count;
    uint shadowmap_height;
    uint cascade_index;
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
[numthreads(256, 1, 1)]
void main(
    uint3 global_id: SV_DispatchThreadID,
    uint3 thread_id: SV_GroupThreadID
)
{
    uint index = global_id.x;

    // lower 32 bits only
    uint total = uint(get_prefix_sum_counter(pc.prefix_sum_buffer, pc.cascade_index));
    if (index >= total)
    {
        return;
    }

    uint prefix_sum_idx = prefix_sum_binary_search(pc.prefix_sum_buffer, pc.cascade_index, index);
    PrefixSumData prefix_sum_data = pc.prefix_sum_buffer.prefix_sums[pc.cascade_index].data[prefix_sum_idx];
    uint meshlet_offset = prefix_sum_data.meshlet_offset;
    uint meshlet_count = prefix_sum_data.meshlet_count;
    uint instance_id = prefix_sum_data.instance_id;
    uint sum = prefix_sum_data.sum;

    uint local_meshlet_id = index - (sum - meshlet_count);
    ObjectData o = pc.object_buffer[instance_id];

    uint global_meshlet_id = local_meshlet_id + meshlet_offset;
    Meshlet meshlet = pc.meshlet_buffer[global_meshlet_id];

    float3 center_ws = rotate_quat(float3(meshlet.cx, meshlet.cy, meshlet.cz), o.orientation) * o.scale + o.translation;
    float radius = float(meshlet.radius);
    radius *= o.scale;

    // if (!partially_inside_cascade(center_ws, radius, pc.cascade_index))
    //     return;

    uint count;
    InterlockedAdd(pc.dispatch_buffer[1].x, 1, count);

    pc.meshlet_draw_buffer[count] = index;
}
