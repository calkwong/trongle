module culling;

// 2D Polyhedral Bounds of a Clipped, Perspective-Projected 3D Sphere. Michael Mara, Morgan McGuire. 2013
// https://zeux.io/2023/01/12/approximate-projected-bounds/
public bool project_sphere(float3 c, float r, float near, float p00, float p11, inout float4 aabb)
{
    if (c.z < r + near)
        return false;

    float3 cr = c * r;
    float czr2 = c.z * c.z - r * r;

    float vx = sqrt(c.x * c.x + czr2);
    float minx = (vx * c.x - cr.z) / (vx * c.z + cr.x);
    float maxx = (vx * c.x + cr.z) / (vx * c.z - cr.x);

    float vy = sqrt(c.y * c.y + czr2);
    float miny = (vy * c.y - cr.z) / (vy * c.z + cr.y);
    float maxy = (vy * c.y + cr.z) / (vy * c.z - cr.y);

    aabb = float4(minx * p00, miny * p11, maxx * p00, maxy * p11);
    // clip space -> uv space
    aabb = aabb.xwzy * float4(0.5, -0.5, 0.5, -0.5) + float4(0.5);

    return true;
}

// symmetrical sphere-plane distance test
public bool frustum_culled(float3 center, float radius, float4 plane, float near, float far)
{
    bool visible = true;

    visible = visible && (center.z * plane.y - abs(center.x) * plane.x) >= -radius;
    visible = visible && (center.z * plane.w - abs(center.y) * plane.z) >= -radius;
    visible = visible && (center.z - radius <= -near) && (center.z + radius >= -far);

    return !visible;
}

public float get_occlusion_mip(float4 aabb, float2 valid_extent, float texture_lod)
{
    float2 size = (aabb.zw - aabb.xy) * valid_extent;
    float max_dim = max(size.x, size.y);
    float mip = ceil(log2(max_dim));
    mip = min(mip, texture_lod);

    return mip;
}
