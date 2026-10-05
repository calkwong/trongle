import types;
import util;
import bindings;
import prefix_sum;

struct Interp
{
    float4 position : SV_Position;
    float3 normal;
    float4 tangent;
    float2 uv;
    nointerpolation uint material_id;
    nointerpolation uint meshlet_id;
}

struct PushConstants
{
    float4x4 view_proj;
    Vertex* vertex_buffer;
    ObjectData* object_buffer;
    MaterialData* material_buffer;
    Meshlet* meshlet_buffer;
    uint32_t* meshlet_indices_buffer;
    PrefixSum* prefix_sum_buffer;
    uint32_t* meshlet_draw_buffer;
}

[[vk::push_constant]]
PushConstants pc;

#define WG_SIZE_X 64

[shader("mesh")]
[numthreads(WG_SIZE_X, 1, 1)]
[outputtopology("triangle")]
void mesh_main(
    uint3 wg_id: SV_GroupID,
    uint t_index: SV_GroupIndex,
    out vertices Interp verts[64],
    out indices uint3 tris[126],
)
{
    uint index = pc.meshlet_draw_buffer[wg_id.x];

    uint prefix_sum_idx = prefix_sum_binary_search(pc.prefix_sum_buffer, index);
    PrefixSumData prefix_sum_data = pc.prefix_sum_buffer.data[prefix_sum_idx];
    uint meshlet_offset = prefix_sum_data.meshlet_offset;
    uint meshlet_count = prefix_sum_data.meshlet_count;
    uint instance_id = prefix_sum_data.instance_id;
    uint sum = prefix_sum_data.sum;

    uint local_meshlet_id = index - (sum - meshlet_count);
    uint global_meshlet_id = local_meshlet_id + meshlet_offset;
    Meshlet meshlet = pc.meshlet_buffer[global_meshlet_id];

    uint vertex_count = meshlet.vertex_count;
    uint triangle_count = meshlet.triangle_count;

    SetMeshOutputCounts(vertex_count, triangle_count);

    bool pack_vertex = meshlet.pack_vertex == 1;
    uint vertex_offset = meshlet.data_offset;
    uint index_offset = vertex_offset + (pack_vertex ? (vertex_count + 1) / 2 : vertex_count);

    ObjectData o = pc.object_buffer[instance_id];
    uint base_vertex = meshlet.base_vertex;

    for (uint i = t_index; i < vertex_count;)
    {
        uint vertex_index = pc.meshlet_indices_buffer[vertex_offset + (pack_vertex ? (i / 2) : i)];

        if (pack_vertex)
        {
            vertex_index = (i & 1u) != 0 ? vertex_index >> 16 : vertex_index & (0xFFFF);
        }

        Vertex v = pc.vertex_buffer[base_vertex + vertex_index];

        float3 pos = float3(v.px, v.py, v.pz);
        float4 world_pos = float4(rotate_quat(pos, o.orientation) * o.scale + o.translation, 1.0);

        float3 normal;
        float4 tangent;
        unpack_tbn(v.normal, uint(v.tangent), normal, tangent);

        verts[i].position = mul(pc.view_proj, world_pos);
        verts[i].normal = rotate_quat(normal, o.orientation);
        verts[i].tangent = float4(rotate_quat(tangent.xyz, o.orientation), tangent.w);
        verts[i].uv = float2(v.uv_x, v.uv_y);
        verts[i].material_id = o.material_id;
        verts[i].meshlet_id = global_meshlet_id;

        break;
    }

    for (uint i = t_index; i < triangle_count; i += WG_SIZE_X)
    {
        // This may not work outside of Slang and SPIRV
        uint8_t* meshlet_triangles = (uint8_t*)pc.meshlet_indices_buffer;
        let triangles_offset = index_offset * 4 + i * 3;

        uint idx0 = meshlet_triangles[triangles_offset];
        uint idx1 = meshlet_triangles[triangles_offset + 1];
        uint idx2 = meshlet_triangles[triangles_offset + 2];

        tris[i] = uint3(idx0, idx1, idx2);
    }
}

[shader("pixel")]
void ps_main(
    Interp in,
    out float4 out_color: SV_Target0,
    out float4 out_normal: SV_Target1,
    out float2 out_mr: SV_Target2,
    out uint out_debug: SV_Target3
)
{
    MaterialData m = pc.material_buffer[in.material_id];

    SamplerState samp = get_sampler<SamplerState>(LINEAR_SAMPLER);

    float4 albedo = m.base_color_factor;
    if (m.diffuse_id != INVALID_ID)
    {
        Texture2D tex = get_nonuniform_sampled_image<Texture2D>(m.diffuse_id);
        albedo.xyz *= tex.Sample(samp, in.uv).xyz;
    }

    float3 N;
    if (m.normal_id != INVALID_ID)
    {
        Texture2D tex = get_nonuniform_sampled_image<Texture2D>(m.normal_id);
        N = tex.Sample(samp, in.uv).xyz;
    }
    else
    {
        N = normalize(in.normal);
    }

    float2 metal_roughness = float2(m.metallic_factor, m.roughness_factor);
    if (m.metal_roughness_id != INVALID_ID)
    {
        Texture2D tex = get_nonuniform_sampled_image<Texture2D>(m.metal_roughness_id);
        metal_roughness *= tex.Sample(samp, in.uv).bg;
    }

    out_color = albedo;
    out_normal = float4(N, 1.0);
    out_mr = float2(metal_roughness);
    out_debug = in.meshlet_id;
}

