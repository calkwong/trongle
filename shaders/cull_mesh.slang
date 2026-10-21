import types;
import bindings;
import util;
import prefix_sum;
import culling;

struct PushConstants
{
    float4x4 view;
    float4x4 prev_view;
    MeshData* mesh_buffer;
    ObjectData* object_buffer;
    PrefixSums* prefix_sum_buffer;
    uint* late_pass_candidates_buffer;
    float4 planes;
    float2 depth_pyramid_extent;
    float2 depth_buffer_extent;
    float p00;
    float p11;
    float near;
    float far;
    uint count;
    float lod_distance_factor;
    uint late;
    float depth_pyramid_lod;
    uint depth_pyramid_id;
    uint skip_occlusion;
}

[[vk::push_constant]]
PushConstants pc;

// (LATE) look up bitfield for early return
// (EARLY ONLY) frustum test
// LOD test
// HZB test
// (EARLY) set bitfield if culled from HZB

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 global_id: SV_DispatchThreadID)
{
    uint index = global_id.x;
    if (index >= pc.count)
        return;

    bool late = pc.late == 1;
    if (late)
    {
        uint word = pc.late_pass_candidates_buffer[index >> 5];
        uint bit_set = (word >> (index & 31u)) & 1u;
        if (bit_set != 1u)
            return;
        // This seems to be more efficient than zeroing the entire buffer each frame
        InterlockedAnd(pc.late_pass_candidates_buffer[index >> 5], ~(1u << (index & 31u)));
    }

    ObjectData o = pc.object_buffer[index];
    uint mesh_id = o.mesh_id;

    MeshData mesh = pc.mesh_buffer[mesh_id];

    float4 world_center = float4(rotate_quat(mesh.center, o.orientation) * o.scale + o.translation, 1.0);
    float4 center = mul(pc.view, world_center);
    float4 occlusion_center = center;
    float scale = o.scale;
    float radius = scale * mesh.radius;
    bool visible = true;

    // We can skip frustum cull for late pass. Late pass will only process meshes that pass frustum but fail HZB.
    if (!late)
    {
        occlusion_center = mul(pc.prev_view, world_center);
        visible &= !frustum_culled(center.xyz, radius, pc.planes, pc.near, pc.far);
    }

    if (!visible)
        return;

    float4 aabb;
    // TODO: do we need this conditional check?
    if (pc.skip_occlusion == 0 && project_sphere(float3(occlusion_center.x, occlusion_center.y, -occlusion_center.z), radius, pc.near, pc.p00, pc.p11, aabb))
    {
        // TODO: this can be optimized as a push constant, leaving as is for clarity
        float2 valid_extent = pc.depth_buffer_extent * 0.5;
        float2 hzb_uv_scale = valid_extent / pc.depth_pyramid_extent;

        float mip = get_occlusion_mip(aabb, pc.depth_pyramid_extent, pc.depth_pyramid_lod);
        SamplerState samp = get_sampler<SamplerState>(DEPTH_REDUCE);

        float2 adjusted_uv = (aabb.zw + aabb.xy) * 0.5 * hzb_uv_scale;
        float depth = get_sampled_image<Texture2D>(pc.depth_pyramid_id).SampleLevel(samp, adjusted_uv, mip).r;
        float depth_sphere = pc.near / (-occlusion_center.z - radius);

        visible &= depth_sphere > depth; // reverse depth
    }

    if (!late && !visible)
    {
        InterlockedOr(pc.late_pass_candidates_buffer[index >> 5], 1u << (index & 31u));
        return;
    }

    if (visible)
    {
        uint lod_index = 0;
        float d = max(0.0, length(center.xyz) - radius);
        float e = d * pc.lod_distance_factor / scale;

        for (uint i = 0; i < mesh.lod_count; i++)
        {
            lod_index = e >= mesh.mesh_lods[i].error ? i : lod_index;
        }

        MeshLod lod = mesh.mesh_lods[lod_index];
        uint meshlet_count = lod.meshlet_count;
        uint meshlet_offset = lod.meshlet_offset;

        prefix_sum_inclusive_append(pc.prefix_sum_buffer, 0, index, meshlet_count, meshlet_offset);
    }
}
