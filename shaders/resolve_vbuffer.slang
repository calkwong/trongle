import types;
import bindings;
import util;
import barycentrics;
import pbr;
import shadow_mapping;
import lighting_lpv;

struct PushConstants
{
    float4x4 view_proj;
    Vertex* vertex_buffer;
    ObjectData* object_buffer;
    Meshlet* meshlet_buffer;
    uint32_t* meshlet_indices_buffer;
    MaterialData* material_buffer;
    GlobalData* global_buffer;
    LightProbeVolume* lpv_buffer;
    PointLight* light_buffer;
    FroxelOffset* froxel_offset_list;
    uint* froxel_item_list;
    float3 light_dir;
    uint shadowmap_id;
    float3 camera_pos;
    uint visbuffer_id;
    float2 screen_extent;
    float2 pixels_per_cluster;
    uint debug_id;
    uint output_id;
    uint envmap_id;
    uint lpv_atlas_id;
    uint depth_id;
    float near;
    float scale;
    float bias;
    uint debug_probe;
}

[[vk::push_constant]]
PushConstants pc;

enum Debug : uint
{
    Default,
    Albedo,
    Normal, // Shading normal (or geometric normal if no normal mapping)
    Tangent,
    TangentSign,
    Metallic,
    Roughness,
    AO,
    Uv,
    Shadow,
    Triangles,
    Meshlets,
    IndirectSpecular,
    IndirectDiffuse
};

#define CS2 1

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

    float3 e1 = wp1 - wp0;
    float3 e2 = wp2 - wp0;
    float3 gN = cross(e1, e2);
    bool is_back_face = dot(gN, V) < 0.0;

    float3 n0 = rotate_quat(np0, obj.orientation);
    float3 n1 = rotate_quat(np1, obj.orientation);
    float3 n2 = rotate_quat(np2, obj.orientation);

    float3 N = interpolate(bary, n0, n1, n2);
    N = is_back_face ? -N : N;

    float4 t0 = float4(rotate_quat(tp0.xyz, obj.orientation), tp0.w);
    float4 t1 = float4(rotate_quat(tp1.xyz, obj.orientation), tp1.w);
    float4 t2 = float4(rotate_quat(tp2.xyz, obj.orientation), tp2.w);

    float4 T = interpolate(bary, t0, t1, t2);
    T.w = is_back_face ? -T.w : T.w;

    SamplerState samp = get_sampler<SamplerState>(LINEAR_SAMPLER);

    MaterialData m = pc.material_buffer[obj.material_id];

    float4 albedo = m.base_color_factor;
    if (m.diffuse_id != INVALID_ID)
    {
        albedo *= get_nonuniform_sampled_image<Texture2D>(m.diffuse_id).SampleGrad(samp, uv, uv_ddx, uv_ddy);
    }

#if CS2 == 1
    float metallic = m.metallic_factor;
    float perceptual_roughness = m.roughness_factor;

    float4 normal_roughness;

    // Normal roughness texture
    if (m.normal_id != INVALID_ID)
    {
        normal_roughness = get_nonuniform_sampled_image<Texture2D>(m.normal_id).SampleGrad(samp, uv, uv_ddx, uv_ddy);
        float3 tangent_normal = decode_hemi_octahedron_normal(normal_roughness.rg);
        N = tangent_to_world_space_mikktspace(tangent_normal, T, N);

        perceptual_roughness *= normal_roughness.a;
    }
    else
    {
        N = normalize(N);
    }

    // Where metallic is stored varies based on material
    // height.a: https://github.com/ValveResourceFormat/ValveResourceFormat/blob/86aae0564839303d53aa0b079e92103fe5ac5703/Renderer/Shaders/csgo_environment.frag.slang#L1053
    // metallicTex.g: https://github.com/ValveResourceFormat/ValveResourceFormat/blob/master/Renderer/Shaders/complex.frag.slang
    if (m.metal_roughness_id != INVALID_ID)
    {
        metallic *= get_nonuniform_sampled_image<Texture2D>(m.metal_roughness_id).SampleGrad(samp, uv, uv_ddx, uv_ddy).g;
    }
#else
    float3 normal;
    if (m.normal_id != INVALID_ID)
    {
        normal = get_nonuniform_sampled_image<Texture2D>(m.normal_id).SampleGrad(samp, uv, uv_ddx, uv_ddy).xyz;
        normal = normal * 2.0 - 1.0;
        N = tangent_to_world_space_mikktspace(normal, T, N);
    }
    else
    {
        N = normalize(N);
    }

    float metallic = m.metallic_factor;
    float perceptual_roughness = m.roughness_factor;
    if (m.metal_roughness_id != INVALID_ID)
    {
        float2 metal_roughness = get_nonuniform_sampled_image<Texture2D>(m.metal_roughness_id).SampleGrad(samp, uv, uv_ddx, uv_ddy).bg;
        metallic *= metal_roughness.x;
        perceptual_roughness *= metal_roughness.y;
    }
#endif

    perceptual_roughness = max(perceptual_roughness, 0.045);
    float roughness = perceptual_roughness * perceptual_roughness;

    float ao = 1.0;
    // Occlusion strength ignored
    if (m.occlusion_id != INVALID_ID)
    {
        ao = get_nonuniform_sampled_image<Texture2D>(m.occlusion_id).SampleGrad(samp, uv, uv_ddx, uv_ddy).r;
    }

    float3 L = pc.light_dir;
    float3 H = normalize(L + V);
    float3 light_color = float3(1.0, 0.73046076, 0.5088813);

    BRDFData brdf_data = prepare_brdf_data(N, L, V, albedo.xyz, roughness, metallic);
    out_color.xyz = eval_brdf(brdf_data, albedo.xyz, roughness, metallic) * light_color * brdf_data.NoL;

    if (pc.shadowmap_id != INVALID_ID)
    {
        float shadow_factor = calculate_shadow(world_pos, pc.shadowmap_id, pc.global_buffer);
        out_color.xyz *= shadow_factor;
    }

#if 1
    SamplerState cube_samp = get_sampler<SamplerState>(CUBE_SAMPLER);
    // TODO: mip based on roughness
    float mip = 0.0;
    float3 indirect_specular = get_nonuniform_sampled_image<TextureCube>(pc.envmap_id + obj.probe_id).SampleLevel(cube_samp, N, mip).rgb;
    // let probe_id = pc.debug_probe;
    let probe_id = obj.probe_id;
    float3 source_N = float3(-N.z, -N.x, N.y);
    float3 indirect_diffuse = SampleLightProbeIrradiance(pc.lpv_buffer, world_pos, source_N, pc.lpv_atlas_id, probe_id);
#endif

// LOCAL LIGHTS
#if 0
    let frag_coord = tex_coord + float2(0.5);
    float depth = get_sampled_image<Texture2D>(pc.depth_id).Load(int3(tex_coord, 0)).r;
    float view_z = linearize_depth(pc.near, depth);
    uint slice = get_cluster_z_slice(view_z, pc.scale, pc.bias);
    uint cluster_index = get_cluster_index(float2(frag_coord.x, pc.screen_extent.y - frag_coord.y), pc.pixels_per_cluster.xy, slice);

    FroxelOffset froxel_offset = pc.froxel_offset_list[cluster_index];
    for (int i = 0; i < froxel_offset.count; i++)
    {
        uint index = pc.froxel_item_list[froxel_offset.offset + i];

        PointLight point_light = pc.light_buffer[index];
        float3 light_pos = point_light.pos;
        float light_radius = point_light.radius;
        float3 light_color = point_light.color.xyz;

        float3 distance = light_pos - world_pos;
        float3 L = normalize(distance);

        update_brdf_data(brdf_data, L);

        float attenuation = get_square_falloff_attenuation(distance, light_radius);
        out_color.xyz += eval_brdf(brdf_data, albedo.xyz, roughness, metallic) * light_color * attenuation * brdf_data.NoL;
    }
#endif

    // TODO: indirect lighting
    out_color.xyz += albedo.xyz * 0.1;

    // TODO: gradient noise from http://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare
    out_color.xyz = tonemap(out_color.xyz);

    switch (pc.debug_id)
    {
    case Debug::Albedo:
    {
        out_color = albedo;
        break;
    }
    case Debug::Normal:
    {
        out_color = float4(float3(N) * 0.5 + 0.5, 1);
        break;
    }
    case Debug::Tangent:
    {
        T.xyz = T.xyz * 0.5 + 0.5;
        out_color = float4(T.xyz, 1.0);
        break;
    }
    case Debug::TangentSign:
    {
        float T_sign = T.w > 0.0 ? 1.0 : 0.0;
        out_color = float4(float3(T_sign), 1.0);
        break;
    }
    case Debug::Metallic:
    {
        out_color = float4(float3(metallic), 1.0);
        break;
    }
    case Debug::Roughness:
    {
        out_color = float4(float3(perceptual_roughness), 1.0);
        break;
    }
    case Debug::AO:
    {
        out_color = float4(float3(ao), 1.0);
        break;
    }
    case Debug::Uv:
    {
        out_color = float4(uv, 0.0, 1.0);
        break;
    }
    case Debug::Shadow:
    {
        SamplerState nearest_samp = get_sampler<SamplerState>(NEAREST_SAMPLER);

        if (in_uv.x < 0.5 && in_uv.y < 0.5)
        {
            uint cascade = 0;
            cascade = in_uv.x < 0.25 ? cascade : cascade + 1;
            cascade = in_uv.y < 0.25 ? cascade : cascade + 2;

            float2 shadow_uv = in_uv * 4.0;
            shadow_uv = fmod(shadow_uv, float2(1.0));

            out_color.xyz = get_nonuniform_sampled_image<Texture2D>(pc.shadowmap_id + cascade).SampleLevel(nearest_samp, shadow_uv, 0.0).rrr;
        }
        break;
    }
    case Debug::Triangles:
    {
        uint mhash = hash(triangle_id);
        out_color.xyz = float3(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255)) / 255.0;
        break;
    }
    case Debug::Meshlets:
    {
        uint mhash = hash(meshlet_id);
        out_color.xyz = float3(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255)) / 255.0;
        break;
    }
    case Debug::IndirectSpecular:
    {
        out_color.xyz = indirect_specular;
        break;
    }
    case Debug::IndirectDiffuse:
    {
        out_color.xyz = indirect_diffuse;
        break;
    }
    default:
        break;
    }

    out.Store(tex_coord, out_color);
}
