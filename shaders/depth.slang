import bindings;
import types;
import util;
import prefix_sum;

#include "../src/config.h"

struct Interp
{
    float4 position : SV_Position;
    float2 uv;
    nointerpolation uint material_id;
};

struct PushConstants
{
    GlobalData* global_buffer;
    ObjectData* object_buffer;
    Vertex* vertex_buffer;
    Meshlet* meshlet_buffer;
    uint* meshlet_indices_buffer;
    MeshletInstance* meshlet_instance_buffer;
    MaterialData* material_buffer;
    uint cascade_index;
};

[[vk::push_constant]]
PushConstants pc;

[vk::constant_id(0)]
const int OPAQUE = 1;

#define WG_SIZE_X 64

[shader("mesh")]
[numthreads(WG_SIZE_X, 1, 1)]
[outputtopology("triangle")]
void mesh_main(
    uint3 wg_id: SV_GroupID,
    uint t_index: SV_GroupIndex,
    out vertices Interp verts[MAX_MESH_VERTS],
    out indices uint3 tris[MAX_MESH_TRIS],
)
{
    MeshletInstance meshlet_instance;
    if (OPAQUE == 1)
        meshlet_instance = pc.meshlet_instance_buffer[wg_id.x];
    else
        meshlet_instance = pc.meshlet_instance_buffer[MESHLET_DRAW_MASKED_OFFSET + wg_id.x];

    uint instance_id = meshlet_instance.instance_id;
    uint meshlet_id = meshlet_instance.meshlet_id;
    Meshlet meshlet = pc.meshlet_buffer[meshlet_id];

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
        verts[i].position = mul(pc.global_buffer.shadow_matrices[pc.cascade_index], world_pos);
        verts[i].uv = float2(v.uv_x, v.uv_y);
        verts[i].material_id = o.material_id;
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
    Interp in
)
{
   	if (OPAQUE == 0)
	{
	    SamplerState samp = get_sampler<SamplerState>(LINEAR_SAMPLER);

		MaterialData m = pc.material_buffer[in.material_id];

		float4 albedo = m.base_color_factor;
		if (m.diffuse_id != 0)
		    albedo *= get_sampled_image<Texture2D>(m.diffuse_id).Sample(samp, in.uv);

		if (albedo.a < 0.5)
			discard;
	}
}
