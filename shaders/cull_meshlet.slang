import types;
import bindings;
import util;
import prefix_sum;
import culling;

struct PushConstants
{
    float4x4 view;
    float4x4 prev_view;
    Meshlet* meshlet_buffer;
    ObjectData* object_buffer;
    PrefixSum* prefix_sum_buffer;
    uint* meshlet_draw_buffer;
    Dispatch* dispatch_buffer;
    uint* late_pass_candidates_buffer;
    float4 planes;
    float2 depth_pyramid_extent;
    float p00;
    float p11;
    float near;
    float far;
    uint late;
    float depth_pyramid_lod;
    uint depth_pyramid_id;
    uint skip_occlusion;
}

[[vk::push_constant]]
PushConstants pc;

// (LATE) look up bitfield for early return
// frustum test
// HZB test
// (EARLY) set bitfield if culled from HZB

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
    ObjectData o = pc.object_buffer[instance_id];

    bool late = pc.late == 1;
    if (late)
    {
        uint word = pc.late_pass_candidates_buffer[index >> 5];
        uint bit_set = (word >> (index & 31u)) & 1u;
        if (bit_set == 1u)
            return;
    }

    uint global_meshlet_id = local_meshlet_id + meshlet_offset;
    Meshlet meshlet = pc.meshlet_buffer[global_meshlet_id];

    float4 world_center = float4(rotate_quat(float3(meshlet.cx, meshlet.cy, meshlet.cz), o.orientation) * o.scale + o.translation, 1.0);
    float4 center = mul(pc.view, world_center);
    float4 occlusion_center = center;
    if (!late)
        occlusion_center = mul(pc.prev_view, world_center);

    float radius = float(meshlet.radius);
    radius *= o.scale;

    bool visible = !frustum_culled(center.xyz, radius, pc.planes, pc.near, pc.far);

    if (!visible)
        return;

    float4 aabb;
    // TODO: do we need this conditional check?
    if (pc.skip_occlusion == 0 && project_sphere(float3(occlusion_center.x, occlusion_center.y, -occlusion_center.z), radius, pc.near, pc.p00, pc.p11, aabb))
    {
        float mip = get_occlusion_mip(aabb, pc.depth_pyramid_extent, pc.depth_pyramid_lod);
        SamplerState samp = get_sampler<SamplerState>(DEPTH_REDUCE);
        float depth = get_sampled_image<Texture2D>(pc.depth_pyramid_id).SampleLevel(samp, (aabb.zw + aabb.xy) * 0.5, mip).r;

        float depth_sphere = pc.near / (-occlusion_center.z - radius);

        visible &= depth_sphere > depth; // reverse depth
    }

    if (visible)
    {
        uint count;
        InterlockedAdd(pc.dispatch_buffer[1].x, 1, count);

        pc.meshlet_draw_buffer[count] = index;

        if (!late)
        {
            InterlockedOr(pc.late_pass_candidates_buffer[index >> 5], 1u << (index & 31u));
        }
    }
}
