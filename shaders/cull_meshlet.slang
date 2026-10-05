import types;
import util;
import prefix_sum;

struct PushConstants
{
    float4x4 view;
    Meshlet* meshlet_buffer;
    ObjectData* object_buffer;
    PrefixSum* prefix_sum_buffer;
    uint* meshlet_draw_buffer;
    Dispatch* dispatch_buffer;
    float4 planes;
    float p00;
    float p11;
    float near;
    float far;
}

[[vk::push_constant]]
PushConstants pc;

// Symmetrical sphere-plane distance test
bool frustum_culled(float3 center, float radius)
{
    bool visible = true;

    visible = visible && (center.z * pc.planes.y - abs(center.x) * pc.planes.x) >= -radius;
    visible = visible && (center.z * pc.planes.w - abs(center.y) * pc.planes.z) >= -radius;
    visible = visible && (center.z - radius <= -pc.near) && (center.z + radius >= -pc.far);

    return !visible;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 global_id: SV_DispatchThreadID)
{
    uint index = global_id.x;

    // lower 32 bits only
    uint total = uint(get_prefix_sum_counter(pc.prefix_sum_buffer));
    if (index >= total)
    {
        return;
    }

    uint prefix_sum_idx = prefix_sum_binary_search(pc.prefix_sum_buffer, index);
    PrefixSumData prefix_sum_data = pc.prefix_sum_buffer.data[prefix_sum_idx];
    uint meshlet_offset = prefix_sum_data.meshlet_offset;
    uint meshlet_count = prefix_sum_data.meshlet_count;
    uint instance_id = prefix_sum_data.instance_id;
    uint sum = prefix_sum_data.sum;

    uint local_meshlet_id = index - (sum - meshlet_count);
    uint global_meshlet_id = local_meshlet_id + meshlet_offset;
    Meshlet meshlet = pc.meshlet_buffer[global_meshlet_id];

    ObjectData o = pc.object_buffer[instance_id];
    float4 center = float4(rotate_quat(float3(meshlet.cx, meshlet.cy, meshlet.cz), o.orientation) * o.scale + o.translation, 1.0);
    center = mul(pc.view, center);

    float radius = float(meshlet.radius);
    radius *= o.scale;

    if (!frustum_culled(center.xyz, radius))
    {
        uint count;
        InterlockedAdd(pc.dispatch_buffer[1].x, 1, count);

        pc.meshlet_draw_buffer[count] = index;
    }
}

