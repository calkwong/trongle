// Based on https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-10-parallel-split-shadow-maps-programmable-gpus

import bindings;
import types;

#include "../src/config.h"

struct PushConstants
{
    GlobalData* global_buffer;
    float4x4 inverse_view_proj;
    float3 sunlight_dir; // normalized
    float shadow_near;
    float shadow_far;
    float lambda;
};

[[vk::push_constant]]
PushConstants pc;

float4x4 create_view_matrix(float3 eye, float3 center, float3 up)
{
    float3 f = normalize(center - eye);
    float3 r = normalize(cross(f, up));
    float3 u = cross(r, f);

    return float4x4(
        float4(r.x, r.y, r.z, -dot(r, eye)),
        float4(u.x, u.y, u.z, -dot(u, eye)),
        float4(-f.x, -f.y, -f.z, dot(f, eye)),
        float4(0.0, 0.0, 0.0, 1.0)
    );
}

float4x4 create_ortho_projection_matrix(float left, float right, float bottom, float top, float z_near, float z_far)
{
    float rl = right - left;
    float tb = top - bottom;
    float fn = z_far - z_near;

    return float4x4(
        float4(2.0 / rl, 0.0, 0.0, -(right + left) / rl),
        float4(0.0, 2.0 / tb, 0.0, -(top + bottom) / tb),
        float4(0.0, 0.0, -1.0 / fn, -z_near / fn),
        float4(0.0, 0.0, 0.0, 1.0)
    );
}

[shader("compute")]
[numthreads(SUN_CASCADE_COUNT, 1, 1)]
void main(
    uint thread_id: SV_GroupIndex
)
{
    float3 frustum_corners[8] = {
        float3(-1, 1, 1),
        float3(1, 1, 1),
        float3(-1, -1, 1),
        float3(1, -1, 1),
        float3(-1, 1, 0),
        float3(1, 1, 0),
        float3(-1, -1, 0),
        float3(1, -1, 0),
    };

    float m = (float)SUN_CASCADE_COUNT;
    float3 light_dir = pc.sunlight_dir;
    float near = pc.shadow_near;
    float far = pc.shadow_far;

    float range = far - near;
    float ratio = far / near;
    float lambda = pc.lambda;

    float i = thread_id + 1.0; // i = [1, m]
    float log_split = near * pow(ratio, i / m);
    float uniform_split = near + range * i / m;
    float d = lambda * (log_split - uniform_split) + uniform_split; // world space
    float split_ratio = (d - near) / range;

    for (int i = 0; i < 8; i++)
    {
        float4 corner = mul(pc.inverse_view_proj, float4(frustum_corners[i], 1.0));
        frustum_corners[i] = corner.xyz / corner.w;
    }

    float current_split = split_ratio;

    // TODO: I don't know if this mapping is safe or implementation defined
    uint lane = thread_id;
    float prev = WaveShuffle(current_split, lane == 0 ? 0 : lane - 1);
    float last_split = lane == 0 ? 0.0 : prev;

    printf("near: %f", near);
    printf("far: %f", far);
    printf("range: %f", range);
    printf("ratio: %f", ratio);
    printf("lambda: %f", lambda);
    printf("last_split: %f", last_split);


    float3 center = float3(0.0);
    for (int corner_i = 0; corner_i < 4; corner_i++)
    {
        float3 distance = frustum_corners[corner_i + 4] - frustum_corners[corner_i];
        frustum_corners[corner_i + 4] = frustum_corners[corner_i] + distance * current_split; // far plane
        frustum_corners[corner_i] = frustum_corners[corner_i] + distance * last_split; // near plane

        center += frustum_corners[corner_i + 4];
        center += frustum_corners[corner_i];
    }
    center /= 8.0;

    // (optional) Ritter's algorithm for tighter bounding sphere

    float radius = 0.0;

    for (int i = 0; i < 8; i++)
    {
        float dist = distance(frustum_corners[i], center);
        radius = max(dist, radius);
    }

    // (optional) stable cascades
    // https://github.com/TheRealMJP/Shadows/blob/master/Shadows/SetupShadows.hlsl
    // https://alextardif.com/shadowmapping.html

    float4x4 shadow_view = create_view_matrix(center + radius * light_dir, center, float3(0, 1, 0));
    float4x4 shadow_proj = create_ortho_projection_matrix(-radius, radius, -radius, radius, radius * 2.0, 0.0);

    // For shadow culling
    pc.global_buffer.shadow_matrices[thread_id] = mul(shadow_proj, shadow_view);
    pc.global_buffer.shadow_views[thread_id] = shadow_view;
    pc.global_buffer.shadow_radius[thread_id] = radius;
}
