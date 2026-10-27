module util;

#include "../../src/config.h"

// 10-10-10-2
public float3 decode_normal(uint n)
{
    return normalize(((int3(n) >> int3(20, 10, 0)) & int3(1023)) / 511.0 - 1.0);
}

// 8-8
public float2 decode_tangent(uint t)
{
    return ((int2(t) >> int2(8, 0)) & int2(255)) / 127.0 - 1.0;
}

public float2 encode_oct(float3 n)
{
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    float2 s = float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    n.xy = n.z >= 0.0 ? n.xy : (1.0 - abs(n.yx)) * s;
    //n.xy = n.xy * 0.5 + 0.5;
    return n.xy;
}

// https://x.com/Stubbesaurus/status/937994790553227264/
public float3 decode_oct(float2 f)
{
    float3 n = float3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = max(-n.z, 0.0);
    n.xy += float2(n.x >= 0.0 ? -t : t, n.y >= 0.0 ? -t : t);
    return normalize(n);
}

public void unpack_tbn(uint n, uint t, out float3 normal, out float4 tangent)
{
    normal = decode_normal(n);
    float2 tp = decode_tangent(t);
    tangent.xyz = decode_oct(tp);
    tangent.w = (n & (1 << 30)) != 0 ? 1.0 : -1.0;
}

public float3 rotate_quat(float3 v, float4 q)
{
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

public uint hash(uint a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

public float3 tangent_to_world_space(float3 sample, float4 T, float3 N)
{
    N = normalize(N);
    T.xyz = normalize(T.xyz);
    float3 B = cross(N, T.xyz) * T.w;

    return normalize(T.xyz * sample.x + B * sample.y + N * sample.z);
}

public float3 tangent_to_world_space_mikktspace(float3 sample, float4 T, float3 N)
{
    // Use interpolated T and N directly, no normalization
    float3 B = cross(N, T.xyz) * T.w;

    return normalize(T.xyz * sample.x + B * sample.y + N * sample.z);
}

// Optimized filmic operator by Jim Hejl and Richard Burgess-Dawson, no gamma correction needed
// http://filmicworlds.com/blog/filmic-tonemapping-operators/
public float3 tonemap(float3 color)
{
    float3 x = max(0, color - 0.004);
    return (x * (6.2 * x + .5)) / (x * (6.2 * x + 1.7) + 0.06);
}

// https://github.com/ValveResourceFormat/ValveResourceFormat/blob/86aae0564839303d53aa0b079e92103fe5ac5703/Renderer/Shaders/common/utils.slang#L225C1-L229C2
float3 hemi_oct_to_float32x3(float2 e)
{
    float3 v = float3(e.xy, 1.0 - abs(e.x) - abs(e.y));
    return normalize(v);
}

// https://github.com/ValveResourceFormat/ValveResourceFormat/blob/86aae0564839303d53aa0b079e92103fe5ac5703/Renderer/Shaders/common/utils.slang#L260
// Unpack HemiOct normal map
public float3 decode_hemi_octahedron_normal(float2 v_hemi_oct)
{
    // Reconstruct the tangent vector from the map
    float2 temp = float2(v_hemi_oct.x + v_hemi_oct.y - 1.003922, v_hemi_oct.x - v_hemi_oct.y);
    float3 tangent_normal = hemi_oct_to_float32x3(temp);

    // This is free, it gets compiled into the TS->WS matrix mul
    tangent_normal.y = -tangent_normal.y;
    return tangent_normal;
}

// Assumes infinite far plane, reverse z
// Returns positive z, negate where necessary
public float linearize_depth(float near, float depth)
{
	return near / depth;
}

// https://advances.realtimerendering.com/s2016/Siggraph2016_idTech6.pdf, Slide 5
// view_z must be positive!
// Do not use this for volumetrics!
public uint get_cluster_z_slice(float view_z, float scale, float bias)
{
    uint slice = uint(floor(log(view_z) * scale - bias));
    return clamp(slice, 0, FROXELS_Z - 1);
}

public uint get_cluster_index(float2 frag_coord, float2 pixels_per_cluster, uint slice)
{
    uint2 clusters = uint2(floor(frag_coord / pixels_per_cluster));
    return clusters.x + FROXELS_X * clusters.y + slice * FROXELS_X * FROXELS_Y;
}

public float distance_squared(float3 a, float3 b)
{
    float3 d = a - b;
    return dot(d, d);
}

public float3 srgb_to_linear(float3 c)
{
	return pow(c, float3(2.2));
}

public float3 linear_to_srgb(float3 c)
{
	return pow(c, float3(1.0 / 2.2));
}
