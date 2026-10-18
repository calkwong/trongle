import types;
import bindings;
import util;
import barycentrics;

struct PushConstants
{
    float4x4 view_proj;
    Vertex* vertex_buffer;
    ObjectData* object_buffer;
    Meshlet* meshlet_buffer;
    uint32_t* meshlet_indices_buffer;
    float3 camera_pos;
    uint visbuffer_id;
    float2 screen_extent;
    uint debug_id;
    uint output_id;
}

[[vk::push_constant]]
PushConstants pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 global_id: SV_DispatchThreadID)
{
    let tex_coord = global_id.xy;
    if (any(tex_coord >= pc.screen_extent))
        return;

    RWTexture2D out = get_storage_image<RWTexture2D>(pc.output_id);
    Texture2D<uint2> vbuffer = get_sampled_image<Texture2D<uint2>>(pc.visbuffer_id);
    uint packed = vbuffer.Load(int3(tex_coord, 0)).r;
    uint meshlet_id = vbuffer.Load(int3(tex_coord, 0)).g;

    float4 out_color = float4(0, 0, 0, 1);
    if (packed == 0xFFFFFFFFu && meshlet_id == 0xFFFFFFFFu)
    {
        out.Store(tex_coord, out_color);
        return;
    }

    uint instance_id = packed >> 7;
    uint triangle_id = packed & 127u;

    Meshlet meshlet = pc.meshlet_buffer[meshlet_id];
    bool pack_vertex = meshlet.pack_vertex == 1;
    uint vertex_offset = meshlet.data_offset;
    uint vertex_count = meshlet.vertex_count;
    uint base_vertex = meshlet.base_vertex;

    uint index_offset = vertex_offset + (pack_vertex ? (vertex_count + 1) / 2 : vertex_count);
    uint triangle_offset = meshlet.data_offset + vertex_count;

    // This may not work outside of Slang and SPIRV
    uint8_t* meshlet_triangles = (uint8_t*)pc.meshlet_indices_buffer;
    let triangles_offset = index_offset * 4 + triangle_id * 3;

    uint idx0 = meshlet_triangles[triangles_offset];
    uint idx1 = meshlet_triangles[triangles_offset + 1];
    uint idx2 = meshlet_triangles[triangles_offset + 2];

    uint vertex_index0 = pc.meshlet_indices_buffer[vertex_offset + (pack_vertex ? (idx0 / 2) : idx0)];
    uint vertex_index1 = pc.meshlet_indices_buffer[vertex_offset + (pack_vertex ? (idx1 / 2) : idx1)];
    uint vertex_index2 = pc.meshlet_indices_buffer[vertex_offset + (pack_vertex ? (idx2 / 2) : idx2)];

    if (pack_vertex)
    {
        // Check for odd/even
        vertex_index0 = (idx0 & 1u) != 0 ? vertex_index0 >> 16 : vertex_index0 & (0xFFFF);
        vertex_index1 = (idx1 & 1u) != 0 ? vertex_index1 >> 16 : vertex_index1 & (0xFFFF);
        vertex_index2 = (idx2 & 1u) != 0 ? vertex_index2 >> 16 : vertex_index2 & (0xFFFF);
    }

    Vertex v0 = pc.vertex_buffer[base_vertex + vertex_index0];
    Vertex v1 = pc.vertex_buffer[base_vertex + vertex_index1];
    Vertex v2 = pc.vertex_buffer[base_vertex + vertex_index2];

    ObjectData obj = pc.object_buffer[instance_id];

    float3 wp0 = rotate_quat(float3(v0.px, v0.py, v0.pz), obj.orientation) * obj.scale + obj.translation;
    float3 wp1 = rotate_quat(float3(v1.px, v1.py, v1.pz), obj.orientation) * obj.scale + obj.translation;
    float3 wp2 = rotate_quat(float3(v2.px, v2.py, v2.pz), obj.orientation) * obj.scale + obj.translation;

    float4 p0 = mul(pc.view_proj, float4(wp0, 1.0));
    float4 p1 = mul(pc.view_proj, float4(wp1, 1.0));
    float4 p2 = mul(pc.view_proj, float4(wp2, 1.0));

    float2 in_uv = (float2(tex_coord) + float2(0.5)) / pc.screen_extent;

    // we want y up here, not y with top left origin
    float2 pNdc = in_uv * 2.0 - 1.0;
    pNdc.y = -pNdc.y;

    BarycentricDeriv bary = calculate_barycentric(p0, p1, p2, pNdc, pc.screen_extent);
    float2 uv;
    float2 uv_ddx;
    float2 uv_ddy;
    interpolate_with_deriv(bary, float2(v0.uv_x, v0.uv_y), float2(v1.uv_x, v1.uv_y), float2(v2.uv_x, v2.uv_y), uv, uv_ddx, uv_ddy);

    float3 world_pos = interpolate(bary, wp0, wp1, wp2);

    float3 np0, np1, np2;
    float4 tp0, tp1, tp2;
    unpack_tbn(v0.normal, uint(v0.tangent), np0, tp0);
    unpack_tbn(v1.normal, uint(v1.tangent), np1, tp1);
    unpack_tbn(v2.normal, uint(v2.tangent), np2, tp2);

    float3 V = normalize(pc.camera_pos - world_pos);

    // alternative: store triangle face bit in visibility buffer
    float3 e1 = wp1 - wp0;
    float3 e2 = wp2 - wp0;
    float3 gN = cross(e1, e2);
    bool is_back_face = dot(gN, V) < 0.0;

    float3 n0 = rotate_quat(np0, obj.orientation);
    float3 n1 = rotate_quat(np1, obj.orientation);
    float3 n2 = rotate_quat(np2, obj.orientation);

    float3 N = interpolate(bary, n0, n1, n2); // mikktspace
    N = normalize(N);
    N = is_back_face ? -N : N;

    switch (pc.debug_id)
    {
    case 0: // GEOMETRIC NORMAL
    {
        out_color = float4(float3(N) * 0.5 + 0.5, 1);
        break;
    }
    case 1: // TRIANGLES
    {
        uint mhash = hash(triangle_id);
        out_color.xyz = float3(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255)) / 255.0;
        break;
    }
    case 2: // MESHLETS
    {
        uint mhash = hash(meshlet_id);
        out_color.xyz = float3(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255)) / 255.0;
        break;
    }
    default:
        break;
    }

    out.Store(tex_coord, out_color);
}
