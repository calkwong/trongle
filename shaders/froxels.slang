import bindings;
import types;

struct PushConstants
{
    float4x4 inverse_proj;
    FroxelAABB* froxels_buffer;
    float2 screen_size;
    float2 froxel_dim;
    float near_z;
    float far_z;
    uint depth_slices;
};

[[vk::push_constant]]
PushConstants pc;

float3 screen_to_view(float2 screen_pos, float2 screen_size, float4x4 inverse_proj)
{
    // Clamp to screen_size
    float2 pos = min(screen_pos, screen_size);

    // Reverse z
    float4 ndc = float4(float2(pos / screen_size), 1.0, 1.0);
    ndc.xy = ndc.xy * 2.0 - 1.0;

    float4 view_pos = mul(inverse_proj, ndc);
    view_pos.xyz /= view_pos.w;

    return view_pos.xyz;
}

// Calculates view space position of one of the cluster's corners
// This is modified for reverse z infinite far plane
float3 line_intersection_to_z_plane(float3 point, float z)
{
    float t = z / point.z;

    return t * point;
}

#define WG_SIZE_X 16
#define WG_SIZE_Y 8

[shader("compute")]
[numthreads(WG_SIZE_X, WG_SIZE_Y, 1)]
void main(
    uint3 t_id: SV_GroupThreadID,
    uint t_index: SV_GroupIndex,
    uint3 wg_id: SV_GroupID
)
{
    // Left-bot & right-top froxel corners in screen space
    float2 min_screen_pos = float2(t_id.x, t_id.y) * pc.froxel_dim;
    float2 max_screen_pos = float2(t_id.x + 1, t_id.y + 1) * pc.froxel_dim;

    float3 min_view_pos = screen_to_view(min_screen_pos, pc.screen_size, pc.inverse_proj);
    float3 max_view_pos = screen_to_view(max_screen_pos, pc.screen_size, pc.inverse_proj);

    uint slice = wg_id.z;
    uint num_slices = pc.depth_slices;

    float froxel_near_z = -pc.near_z * pow(pc.far_z / pc.near_z, float(slice) / float(num_slices));
    float froxel_far_z = -pc.near_z * pow(pc.far_z / pc.near_z, float(slice + 1) / float(num_slices));

    // AABB is conservative and simplified, and is slightly larger than the actual froxel
    float3 min_near = line_intersection_to_z_plane(min_view_pos, froxel_near_z);
    float3 min_far = line_intersection_to_z_plane(min_view_pos, froxel_far_z);
    float3 max_near = line_intersection_to_z_plane(max_view_pos, froxel_near_z);
    float3 max_far = line_intersection_to_z_plane(max_view_pos, froxel_far_z);

    float3 aabb_min = min(min(min_near, min_far), min(max_near, max_far));
    float3 aabb_max = max(max(min_near, min_far), max(max_near, max_far));

    uint froxels_per_slice = WG_SIZE_X * WG_SIZE_Y;
    uint froxel_index = wg_id.z * froxels_per_slice + t_index;

    FroxelAABB aabb;
    aabb.min = float4(aabb_min, 1.0);
    aabb.max = float4(aabb_max, 1.0);

    pc.froxels_buffer[froxel_index] = aabb;
}
