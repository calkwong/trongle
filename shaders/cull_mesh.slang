import types;
import util;
import prefix_sum;

struct PushConstants
{
    float4x4 view;
    MeshData* mesh_buffer;
    ObjectData* object_buffer;
    PrefixSum* prefix_sum_buffer;
    float4 planes;
    float p00;
    float p11;
    float near;
    float far;
    uint count;
    float lod_distance_factor;
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
    if (index >= pc.count)
        return;

    ObjectData o = pc.object_buffer[index];
    uint mesh_id = o.mesh_id;

    MeshData mesh = pc.mesh_buffer[mesh_id];

    float4 center = float4(rotate_quat(mesh.center, o.orientation) * o.scale + o.translation, 1.0);
    center = mul(pc.view, center);

    float scale = o.scale;
    float radius = scale * mesh.radius;

    if (!frustum_culled(center.xyz, radius))
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

        prefix_sum_inclusive_append(pc.prefix_sum_buffer, index, meshlet_count, meshlet_offset);
    }
}

