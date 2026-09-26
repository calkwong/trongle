#include "asset_loader.h"
#include "resources.h"
#include "shared_cpu_gpu.h"

#include <volk.h>
#include <fastgltf/math.hpp>
#include <fastgltf/util.hpp>
#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
// #include <stb_image.h>
#include <vk_mem_alloc.h>
// #include <basisu_transcoder.h>
#include <fmt/core.h>
#include <meshoptimizer.h>

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace
{
bool read_raw_image_data_from_file(const char* filename, std::vector<uint8_t>& ktx_data)
{
    // cursor at the end
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open())
    {
        return false;
    }

    // find what the size of the file is by looking up the location of the cursor
    // because the cursor is at the end, it gives the size directly in bytes
    const size_t file_size = file.tellg();

    // spirv expects the buffer to be on uint32, so make sure to reserve a int
    // vector big enough for the entire file
    std::vector<uint8_t> buffer(file_size);

    // put file cursor at beginning
    file.seekg(0);

    // load the entire file into the buffer
    file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(file_size));

    // now that the file is loaded into the buffer, we can close it
    file.close();

    ktx_data = std::move(buffer);

    return true;
}

// TODO: optimize mesh
// TODO: load_images
} // namespace

bool AssetLoader::load_gltf(
    VkQueue queue,
    VkFence fence,
    VkCommandPool command_pool,
    VkCommandBuffer cmd,
    const std::string& file_path
)
{
    auto asset_path = "../assets/" + file_path;
    fmt::println("loading glTF: {}", asset_path);

    // TODO: implement enums to select at runtime
    constexpr auto supported_extensions =
        fastgltf::Extensions::KHR_texture_basisu
        | fastgltf::Extensions::KHR_materials_transmission
        | fastgltf::Extensions::KHR_materials_volume;

    fastgltf::Parser parser(supported_extensions);

    // TODO: look up options
    constexpr auto gltf_options{
        fastgltf::Options::DontRequireValidAssetMember |
        // fastgltf::Options::LoadGLBBuffers | // now default behaviour
        fastgltf::Options::AllowDouble | fastgltf::Options::LoadExternalBuffers
    };

    std::filesystem::path path = asset_path;
    auto gltf_file = fastgltf::GltfDataBuffer::FromPath(path);

    if (gltf_file.error() != fastgltf::Error::None)
    {
        fmt::println("Error: {}", fastgltf::getErrorName(gltf_file.error()));
        return false;
    }

    fastgltf::Asset asset{};

    auto type = fastgltf::determineGltfFileType(gltf_file.get());
    if (type == fastgltf::GltfType::glTF)
    {
        auto load = parser.loadGltf(gltf_file.get(), path.parent_path(), gltf_options);
        if (load)
        {
            asset = std::move(load.get());
        }
        else
        {
            fmt::println("Failed to load gltf: {}", fastgltf::to_underlying(load.error()));
            return false;
        }
    }
    else if (type == fastgltf::GltfType::GLB)
    {
        auto load{ parser.loadGltfBinary(gltf_file.get(), path.parent_path(), gltf_options) };
        if (load)
        {
            asset = std::move(load.get());
        }
        else
        {
            fmt::println("Failed to load gltf: {}", fastgltf::to_underlying(load.error()));
            return false;
        }
    }
    else
    {
        fmt::println("Failed to determine gltf container");
        return false;
    }

    /*
    // note: handle another way
    assert(!asset.materials.empty() && "No materials found");
    auto& materials_data = scene->materials;

    // TODO: currently supports ktx2 in URI only

    std::vector<Image> images{};
    if (!asset.images.empty())
        images = load_images(asset, device, queue, fence, command_pool, cmd, allocator, heap_manager, file.asset_path);

    file.images.insert(file.images.end(), images.begin(), images.end());

    for (fastgltf::Material& mat : asset.materials)
    {
        MaterialData mat_data{};
        mat_data.base_color_factor.x = mat.pbrData.baseColorFactor[0];
        mat_data.base_color_factor.y = mat.pbrData.baseColorFactor[1];
        mat_data.base_color_factor.z = mat.pbrData.baseColorFactor[2];
        mat_data.base_color_factor.w = mat.pbrData.baseColorFactor[3];
        mat_data.metallic_factor = mat.pbrData.metallicFactor;
        mat_data.roughness_factor = mat.pbrData.roughnessFactor;

        // note: emissive strength currently unaccounted for
        mat_data.emissive_factor.x = mat.emissiveFactor[0];
        mat_data.emissive_factor.y = mat.emissiveFactor[1];
        mat_data.emissive_factor.z = mat.emissiveFactor[2];

        if (mat.pbrData.baseColorTexture.has_value())
        {
            size_t image_index =
                asset.textures[mat.pbrData.baseColorTexture.value().textureIndex].imageIndex
                ? asset.textures[mat.pbrData.baseColorTexture.value().textureIndex].imageIndex.value()
                : asset.textures[mat.pbrData.baseColorTexture.value().textureIndex].basisuImageIndex.value();

            mat_data.diffuse_id = static_cast<uint32_t>(texture_offset + image_index);
        }

        if (mat.pbrData.metallicRoughnessTexture.has_value())
        {
            size_t image_index =
                asset.textures[mat.pbrData.metallicRoughnessTexture.value().textureIndex].imageIndex
                ? asset.textures[mat.pbrData.metallicRoughnessTexture.value().textureIndex].imageIndex.value()
                : asset.textures[mat.pbrData.metallicRoughnessTexture.value().textureIndex].basisuImageIndex.value();

            mat_data.metal_roughness_id = static_cast<uint32_t>(texture_offset + image_index);
        }

        if (mat.normalTexture.has_value())
        {
            size_t image_index =
                asset.textures[mat.normalTexture.value().textureIndex].imageIndex
                ? asset.textures[mat.normalTexture.value().textureIndex].imageIndex.value()
                : asset.textures[mat.normalTexture.value().textureIndex].basisuImageIndex.value();

            mat_data.normal_id = static_cast<uint32_t>(texture_offset + image_index);
        }

        if (mat.occlusionTexture.has_value())
        {
            size_t image_index =
                asset.textures[mat.occlusionTexture.value().textureIndex].imageIndex
                ? asset.textures[mat.occlusionTexture.value().textureIndex].imageIndex.value()
                : asset.textures[mat.occlusionTexture.value().textureIndex].basisuImageIndex.value();

            mat_data.occlusion_id = static_cast<uint32_t>(texture_offset + image_index);
        }

        if (mat.emissiveTexture.has_value())
        {
            size_t image_index =
                asset.textures[mat.emissiveTexture.value().textureIndex].imageIndex
                ? asset.textures[mat.emissiveTexture.value().textureIndex].imageIndex.value()
                : asset.textures[mat.emissiveTexture.value().textureIndex].basisuImageIndex.value();

            mat_data.emissive_id = static_cast<uint32_t>(texture_offset + image_index);
        }

        if (mat.transmission.get())
        {
            auto* transmission_material = mat.transmission.get();
            mat_data.transmission_factor = transmission_material->transmissionFactor;
            if (transmission_material->transmissionTexture.has_value())
            {
                size_t image_index =
                    asset.textures[transmission_material->transmissionTexture.value().textureIndex].imageIndex
                    ? asset.textures[transmission_material->transmissionTexture.value().textureIndex].imageIndex.value()
                    : asset.textures[transmission_material->transmissionTexture.value().textureIndex].basisuImageIndex.value();
                mat_data.transmission_id = static_cast<uint32_t>(texture_offset + image_index);
            }
        }

        if (mat.volume.get())
        {
            auto* volume_material = mat.volume.get();
            mat_data.thickness_factor = volume_material->thicknessFactor;
            mat_data.attenuation_color.x = volume_material->attenuationColor[0];
            mat_data.attenuation_color.y = volume_material->attenuationColor[1];
            mat_data.attenuation_color.z = volume_material->attenuationColor[2];
            mat_data.attenuation_distance =
                volume_material->attenuationDistance == std::numeric_limits<float>::infinity()
                ? 0.0
                : volume_material->attenuationDistance;
            if (volume_material->thicknessTexture.has_value())
            {
                size_t image_index =
                    asset.textures[volume_material->thicknessTexture.value().textureIndex].imageIndex
                    ? asset.textures[volume_material->thicknessTexture.value().textureIndex].imageIndex.value()
                    : asset.textures[volume_material->thicknessTexture.value().textureIndex].basisuImageIndex.value();
                mat_data.thickness_id = static_cast<uint32_t>(texture_offset + image_index);
            }
        }

        materials_data.push_back(mat_data);
    }
    */

    for (fastgltf::Mesh& mesh : asset.meshes)
    {
        MeshAsset new_mesh{};

        for (int i = 0; i < mesh.primitives.size(); ++i)
        {
            const auto& p = mesh.primitives[i];

            std::vector<uint32_t> indices{};
            {
                auto& index_accessor = asset.accessors[p.indicesAccessor.value()];
                indices.resize(index_accessor.count);
                fastgltf::iterateAccessorWithIndex<std::uint32_t>(
                    asset,
                    index_accessor,
                    [&](std::uint32_t index, size_t idx)
                    {
                        indices[idx] = index;
                    }
                );
            }

            using Position = std::array<uint16_t, 3>;
            std::vector<Position> positions{};

            if (auto it = p.findAttribute("POSITION"); it != p.attributes.end())
            {
                auto& position_accessor = asset.accessors[it->accessorIndex];
                positions.resize(position_accessor.count);
                fastgltf::iterateAccessorWithIndex<glm::vec3>(
                    asset,
                    position_accessor,
                    [&](glm::vec3 pos, size_t index)
                    {
                        uint16_t px = meshopt_quantizeHalf(pos.x);
                        uint16_t py = meshopt_quantizeHalf(pos.y);
                        uint16_t pz = meshopt_quantizeHalf(pos.z);

                        positions[index] = Position{ px, py, pz };
                    }
                );
            }

            std::vector<uint32_t> normals{};
            if (auto it = p.findAttribute("NORMAL"); it != p.attributes.end())
            {
                auto& normals_accessor = asset.accessors[it->accessorIndex];
                normals.resize(normals_accessor.count);
                fastgltf::iterateAccessorWithIndex<glm::vec3>(
                    asset,
                    normals_accessor,
                    [&](glm::vec3 n, size_t index)
                    {
                        uint32_t normal =
                            (meshopt_quantizeSnorm(n.x, 10) + 511) << 20
                            | (meshopt_quantizeSnorm(n.y, 10) + 511) << 10
                            | (meshopt_quantizeSnorm(n.z, 10) + 511);

                        normals[index] = normal;
                    }
                );
            }

            using UV = std::array<uint16_t, 2>;
            std::vector<UV> uvs{};
            if (auto it = p.findAttribute("TEXCOORD_0"); it != p.attributes.end())
            {
                auto& uv_accessor = asset.accessors[it->accessorIndex];
                uvs.resize(uv_accessor.count);
                fastgltf::iterateAccessorWithIndex<glm::vec2>(
                    asset,
                    uv_accessor,
                    [&](glm::vec2 uv, size_t index)
                    {
                        uint16_t uv_x = meshopt_quantizeHalf(uv.x);
                        uint16_t uv_y = meshopt_quantizeHalf(uv.y);

                        uvs[index] = UV{ uv_x, uv_y };
                    }
                );
            }
            else
            {
                uvs.resize(positions.size());
            }

            auto encode_oct = [&](glm::vec3 n) -> glm::vec2
            {
                n /= (fabs(n.x) + fabs(n.y) + fabs(n.z));
                float u = n.z >= 0.0f ? n.x : (1.0f - fabs(n.y)) * (n.x >= 0.0f ? 1.0f : -1.0f);
                float v = n.z >= 0.0f ? n.y : (1.0f - fabs(n.x)) * (n.y >= 0.0f ? 1.0f : -1.0f);

                // optional mapping to [0, 1]?
                return glm::vec2(u, v);
            };

            // bool generate_mikkt_tangents = false;
            std::vector<uint16_t> tangents{};
            if (auto it = p.findAttribute("TANGENT"); it != p.attributes.end())
            {
                auto& tangent_accessor = asset.accessors[it->accessorIndex];
                tangents.resize(tangent_accessor.count);
                fastgltf::iterateAccessorWithIndex<glm::vec4>(
                    asset,
                    tangent_accessor,
                    [&](glm::vec4 tangent, size_t index)
                    {
                        glm::vec2 t_encoded = encode_oct(glm::vec3(tangent));

                        uint16_t t = (meshopt_quantizeSnorm(t_encoded.x, 8) + 127) << 8 | (meshopt_quantizeSnorm(t_encoded.y, 8) + 127);

                        tangents[index] = t;
                        normals[index] |= (tangent.w >= 0 ? 1 : 0) << 30;
                    }
                );
            }
            else
            {
                // assert(0 && "IMPLEMENT TANGENTS");
                tangents.resize(positions.size());
                // generate_mikkt_tangents = true;
                // tangents.resize(positions.size());
            }

            assert(positions.size() == normals.size() && positions.size() == tangents.size() && positions.size() == uvs.size());

            std::vector<Vertex> vertices{};
            for (size_t idx = 0; idx < positions.size(); ++idx)
            {
                vertices.emplace_back(Vertex{ positions[idx][0], positions[idx][1], positions[idx][2], tangents[idx], normals[idx], uvs[idx][0], uvs[idx][1] });
            }

            // if (generate_mikkt_tangents)
            // {
            //     // fmt::println("generating tangents manually");
            //     MikkMesh mikk_mesh{ &vertices, &indices };
            //     mikk_calculate_tangents(mikk_mesh);
            // }

            MeshData mesh_data{};
            // // note: if we implement multithreading, this likely needs to be computed post meshoptimizing
            // mesh_data.vertex_offset = static_cast<uint32_t>(scene->vertices.size());
            // // for multiple gltf compatibility
            // auto material_offset = materials_data.size() - asset.materials.size();
            // if (p.materialIndex.has_value())
            // {
            //     size_t idx = p.materialIndex.value();
            //     mesh_data.material_id = static_cast<uint32_t>(idx + material_offset);
            //     auto alpha_mode = asset.materials[idx].alphaMode;
            //     // if (asset.materials[idx].doubleSided)
            //     //     fmt::println("{}", asset.materials[idx].name);
            //     switch (alpha_mode)
            //     {
            //     case fastgltf::AlphaMode::Mask:
            //         mesh_data.pass = MaterialPass::Mask;
            //         break;
            //     case fastgltf::AlphaMode::Blend:
            //         mesh_data.pass = MaterialPass::Blend;
            //         break;
            //     default:
            //         break;
            //     }

            //     if (asset.materials[idx].transmission.get())
            //         mesh_data.pass = MaterialPass::Transmission;
            // }
            // else
            // {
            //     assert(0);
            // }

            // auto& meshlet_indices = scene->meshlet_indices;
            // auto& meshlets = scene->meshlets;

            // optimize_mesh(vertices, indices, meshlet_indices, meshlets, mesh_data, scene->vertices, scene->indices);

            mesh_data.first_index = static_cast<uint32_t>(m_indices.size());
            mesh_data.index_count = static_cast<uint32_t>(indices.size());
            mesh_data.vertex_offset = static_cast<uint32_t>(m_vertices.size());

            new_mesh.mesh.push_back(mesh_data);

            m_vertices.insert(m_vertices.end(), vertices.begin(), vertices.end());
            m_indices.insert(m_indices.end(), indices.begin(), indices.end());
        }

        meshes.push_back(new_mesh);
    }

    struct NodeWork
    {
        Node* node;
        size_t index;
    };

    std::vector<NodeWork> work{};

    parent_nodes.reserve(asset.scenes[0].nodeIndices.size());
    for (auto node_index : asset.scenes[0].nodeIndices) // We handle 1 scene per gltf for simplicity
    {
        auto& nd = parent_nodes.emplace_back(Node{});
        work.emplace_back(NodeWork{ &nd, node_index });
    }

    while (work.size() > 0)
    {
        auto [node, node_index] = work.back();
        work.pop_back();
        const auto& gltf_node = asset.nodes[node_index];

        std::visit(
            fastgltf::visitor{
                [&](fastgltf::math::fmat4x4 matrix)
                {
                    memcpy(&node->local_transform, matrix.data(), sizeof(matrix));
                },
                [&](fastgltf::TRS transform)
                {
                    const glm::vec3 tl(
                        transform.translation[0],
                        transform.translation[1],
                        transform.translation[2]
                    );
                    const glm::quat rot(
                        transform.rotation[3],
                        transform.rotation[0],
                        transform.rotation[1],
                        transform.rotation[2]
                    );
                    const glm::vec3 sc(transform.scale[0], transform.scale[1], transform.scale[2]);

                    const glm::mat4 tm = glm::translate(glm::mat4(1.f), tl);
                    const glm::mat4 rm = glm::mat4_cast(rot);
                    const glm::mat4 sm = glm::scale(glm::mat4(1.f), sc);

                    node->local_transform = tm * rm * sm;
                } },
            gltf_node.transform
        );

        if (gltf_node.meshIndex.has_value())
        {
            node->mesh_index = gltf_node.meshIndex.value();
        }

        node->child_count = static_cast<uint32_t>(gltf_node.children.size());
        node->first_child = static_cast<uint32_t>(children_nodes.size());

        children_nodes.reserve(children_nodes.size() + node->child_count);
        for (auto child_index : gltf_node.children)
        {
            auto& nd = children_nodes.emplace_back(Node{});
            work.emplace_back(NodeWork{ &nd, child_index });
        }
    }

    for (auto& node : parent_nodes)
    {
        node.refresh_transform(glm::mat4(1.0f), children_nodes);
    }

    return true;
}

void AssetLoader::cleanup()
{
    for (auto& img : images)
    {
        destroy_image(device, allocator, img);
    }
}

void Node::refresh_transform(const glm::mat4& parent_matrix, std::vector<Node>& children)
{
    world_transform = parent_matrix * local_transform;

    for (auto i = 0; i < child_count; i++)
    {
        auto& child = children[first_child];
        child.refresh_transform(world_transform, children);
    }
}
