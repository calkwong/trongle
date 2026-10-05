module util;

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

