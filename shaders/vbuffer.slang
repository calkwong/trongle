import bindings;
import types;
import util;
import prefix_sum;

#include "../src/config.h"

struct Interp
{
    float4 position : SV_Position;
    nointerpolation uint meshlet_id;
};

struct PerPrimitive
{
    uint data; // (instance_id << 7) | triangle_id;
};

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
    out vertices Interp verts[MAX_MESH_VERTS],
    out indices uint3 tris[MAX_MESH_TRIS],
    out primitives PerPrimitive prims[MAX_MESH_TRIS],
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

    for (uint i = t_index; i < vertex_count; i += WG_SIZE_X)
    {
        uint vertex_index = pc.meshlet_indices_buffer[vertex_offset + (pack_vertex ? (i / 2) : i)];

        if (pack_vertex)
        {
            // Check for odd/even
            vertex_index = (i & 1u) != 0 ? vertex_index >> 16 : vertex_index & (0xFFFF);
        }

        Vertex v = pc.vertex_buffer[base_vertex + vertex_index];

        float3 pos = float3(v.px, v.py, v.pz);
        float4 world_pos = float4(rotate_quat(pos, o.orientation) * o.scale + o.translation, 1.0);

        float3 normal;
        float4 tangent;
        unpack_tbn(v.normal, uint(v.tangent), normal, tangent);

        verts[i].position = mul(pc.view_proj, world_pos);
        verts[i].meshlet_id = global_meshlet_id;
    }

    for (uint i = t_index; i < triangle_count; i += WG_SIZE_X)
    {
        let triangle_id = i;
        prims[i].data = (instance_id << 7) | triangle_id;

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
    PerPrimitive prim,
    out uint2 out_id: SV_Target0,
)
{
    out_id = uint2(prim.data, in.meshlet_id);
}
