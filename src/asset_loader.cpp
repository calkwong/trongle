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
#include <glm/common.hpp>
#include <vk_mem_alloc.h>
#include <basisu_transcoder.h>
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
#include <memory>
#include <algorithm>
#include <execution>

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

std::vector<Image> load_images(
    const fastgltf::Asset& asset,
    VkDevice device,
    VkQueue queue,
    VkFence fence,
    VkCommandPool command_pool,
    VkCommandBuffer cmd,
    VmaAllocator allocator,
    std::string_view asset_path
)
{
    std::vector<Image> images{};

    struct RawImageData
    {
        int width;
        int height;
        int components;
        uint32_t mips;
        uint32_t size;
        VkFormat format;

        std::unique_ptr<unsigned char[]> ktx;
        std::unique_ptr<basist::ktx2_image_level_info[]> ktx_info;
    };

    auto create_raw_image_data = [&](const void* data, size_t data_size) -> RawImageData
    {
        RawImageData raw_image_data{};

        // Create the KTX2 transcoder object
        basist::ktx2_transcoder transcoder{};

        // Initialize the transcoder
        if (!transcoder.init(data, static_cast<uint32_t>(data_size)))
            assert(0 && "Failed to initialize transcoder");

        // TODO: refactor when we stop using BC7 across the board
        auto target_format = basist::transcoder_texture_format::cTFBC7_RGBA;
        uint32_t bytes_per_block_or_pixel = basist::basis_get_bytes_per_block_or_pixel(target_format);

        auto transfer_func = transcoder.get_dfd_transfer_func();
        switch (transfer_func)
        {
        case basist::KTX2_KHR_DF_TRANSFER_SRGB:
            raw_image_data.format = VK_FORMAT_BC7_SRGB_BLOCK;
            break;
        case basist::KTX2_KHR_DF_TRANSFER_LINEAR:
            raw_image_data.format = VK_FORMAT_BC7_UNORM_BLOCK;
            break;
        default:
            assert(0);
            break;
        }

        raw_image_data.mips = transcoder.get_levels();
        raw_image_data.ktx_info = std::make_unique<basist::ktx2_image_level_info[]>(raw_image_data.mips);

        for (uint32_t i = 0; i < raw_image_data.mips; i++)
        {
            transcoder.get_image_level_info(raw_image_data.ktx_info[i], i, 0, 0);
        }

        uint32_t num_blocks_or_pixels{};
        uint64_t buffer_size{};
        for (uint32_t i = 0; i < raw_image_data.mips; i++)
        {
            num_blocks_or_pixels = raw_image_data.ktx_info[i].m_total_blocks;
            buffer_size += bytes_per_block_or_pixel * num_blocks_or_pixels;
        }

        auto header = transcoder.get_header();
        auto supercompression_scheme = header.m_supercompression_scheme;
        switch (supercompression_scheme)
        {
        case basist::KTX2_SS_BASISLZ:
        case basist::KTX2_SS_ZSTANDARD:
            break;
        default:
            assert(0 && "Unsupported supercompression scheme");
            break;
        }

        transcoder.start_transcoding();

        raw_image_data.ktx = std::make_unique<unsigned char[]>(buffer_size);

        unsigned char* ktx_data = raw_image_data.ktx.get();

        raw_image_data.size = 0;
        for (uint32_t i = 0; i < raw_image_data.mips; i++)
        {
            num_blocks_or_pixels = raw_image_data.ktx_info[i].m_total_blocks;
            uint32_t output_size = bytes_per_block_or_pixel * num_blocks_or_pixels;
            if (!transcoder.transcode_image_level(i, 0, 0, ktx_data, output_size, target_format))
                assert(0 && "Transcoding image failed");
            ktx_data += output_size;
            raw_image_data.size += output_size;
        }

        return raw_image_data;
    };

    std::vector<size_t> indices(asset.images.size());
    for (size_t i = 0; i < indices.size(); i++)
        indices[i] = i;

    auto has_ktx2_format = [&](std::string_view file_path) -> bool
    {
        auto pos = file_path.find_last_of('.');
        auto format = file_path.substr(pos + 1);
        return (format == "ktx2");
    };

    auto raw_images = std::vector<RawImageData>(asset.images.size());

    std::filesystem::path path = asset_path;
    basist::basisu_transcoder_init();

    std::transform(
        std::execution::par,
        indices.begin(),
        indices.end(),
        raw_images.begin(),
        [&](size_t index)
        {
            const fastgltf::Image& image = asset.images[index];

            if (const auto* file_path = std::get_if<fastgltf::sources::URI>(&image.data))
            {
                assert(file_path->fileByteOffset == 0); // we don't support offsets with stbi
                assert(file_path->uri.isLocalPath()); // only load local files
                std::filesystem::path full_path = path / file_path->uri.path();

                std::vector<uint8_t> buffer{};
                if (!read_raw_image_data_from_file(full_path.string().c_str(), buffer))
                    assert(0);

                return create_raw_image_data(buffer.data(), buffer.size());
            }
            if (const auto* vector = std::get_if<fastgltf::sources::Vector>(&image.data))
            {
                assert(0 && "fastgltf::sources::Vector not implemented");
            }
            if (const auto* view = std::get_if<fastgltf::sources::BufferView>(&image.data))
            {
                // Printing this so I know when this will run
                fmt::println("BufferView");

                auto& buffer_view = asset.bufferViews[view->bufferViewIndex];
                auto& buffer = asset.buffers[buffer_view.bufferIndex];
                if (const auto* arr = std::get_if<fastgltf::sources::Array>(&buffer.data))
                {
                    return create_raw_image_data(arr->bytes.data() + buffer_view.byteOffset, buffer_view.byteLength);
                }
                else
                {
                    assert(0 && "fastgltf::sources::BufferView not implemented");
                }
            }
            std::abort();
            return RawImageData{};
        }
    );

    Buffer scratch = create_buffer(
        device,
        allocator,
        1000000000,
        VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
    );

    struct ImageUploadInfo
    {
        const void* data;
        uint32_t size;
        uint32_t buffer_offset;
        uint32_t image_index;
        uint32_t mips;
        VkExtent3D extent;
    };

    // TODO: refactor when we stop using BC7 across the board
    auto target_format = basist::transcoder_texture_format::cTFBC7_RGBA;
    const uint32_t bytes_per_block_or_pixel = basist::basis_get_bytes_per_block_or_pixel(target_format);

    std::vector<ImageUploadInfo> image_upload_info{};

    // Offset for each image upload
    uint32_t buffer_offset{};

    for (const auto& raw_image_data : raw_images)
    {
        uint32_t upload_offset{};
        for (uint32_t mip = 0; mip < raw_image_data.mips; mip++)
        {
            uint32_t num_blocks_or_pixels = raw_image_data.ktx_info[mip].m_total_blocks;
            uint32_t output_size = bytes_per_block_or_pixel * num_blocks_or_pixels;

            image_upload_info.emplace_back(
                ImageUploadInfo{
                    .data = static_cast<void*>(raw_image_data.ktx.get() + upload_offset),
                    .size = output_size,
                    .buffer_offset = buffer_offset,
                    .image_index = static_cast<uint32_t>(images.size()),
                    .mips = mip,
                    .extent = { raw_image_data.ktx_info[mip].m_orig_width,
                                raw_image_data.ktx_info[mip].m_orig_height,
                                1 } }
            );

            upload_offset += output_size;
            buffer_offset += output_size;
        }

        images.emplace_back(create_image(
            device,
            allocator,
            { raw_image_data.ktx_info[0].m_orig_width, raw_image_data.ktx_info[0].m_orig_height, 1 },
            raw_image_data.format,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            0,
            true
        ));
    }

    auto flush_uploads = [&]()
    {
        if (scratch.info.size < buffer_offset)
        {
            destroy_buffer(allocator, scratch);
            scratch = create_buffer(
                device,
                allocator,
                static_cast<size_t>(buffer_offset * 1.5),
                VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            );
        }

        // Copy to staging in parallel
        std::for_each(
            std::execution::par,
            image_upload_info.begin(),
            image_upload_info.end(),
            [&](const ImageUploadInfo& upload_info)
            {
                auto p = static_cast<std::byte*>(scratch.info.pMappedData) + upload_info.buffer_offset;
                memcpy(p, upload_info.data, upload_info.size);
            }
        );

        std::vector<VkBufferImageCopy2> buffer_image_copies(image_upload_info.size());
        std::vector<VkCopyBufferToImageInfo2> buffer_to_image_info(image_upload_info.size());
        for (int i = 0; i < image_upload_info.size(); i++)
        {
            const auto& upload_info = image_upload_info[i];
            auto& copy = buffer_image_copies[i];
            auto& info = buffer_to_image_info[i];

            copy.sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2;
            copy.pNext = nullptr;
            copy.bufferOffset = upload_info.buffer_offset;
            copy.bufferRowLength = 0;
            copy.bufferImageHeight = 0;
            copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy.imageSubresource.mipLevel = upload_info.mips;
            copy.imageSubresource.baseArrayLayer = 0;
            copy.imageSubresource.layerCount = 1;
            copy.imageOffset = VkOffset3D{ 0, 0, 0 };
            copy.imageExtent = upload_info.extent;

            info.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2;
            info.pNext = nullptr;
            info.srcBuffer = scratch.buffer;
            info.dstImage = images[upload_info.image_index].image;
            info.dstImageLayout = VK_IMAGE_LAYOUT_GENERAL;
            info.regionCount = 1;
            info.pRegions = &copy;
        }

        for (const auto& info : buffer_to_image_info)
        {
            vkCmdCopyBufferToImage2(cmd, &info);
        }
    };

    std::vector<VkImageMemoryBarrier2> image_barriers(asset.images.size());
    for (int i = 0; i < image_barriers.size(); i++)
    {
        image_barriers[i] = image_barrier(
            images[i].image,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            0,
            0,
            0,
            0
        );
    }

    immediate_submit(
        device,
        queue,
        fence,
        command_pool,
        cmd,
        [&](VkCommandBuffer cmd)
        {
            VkDependencyInfo info{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            info.imageMemoryBarrierCount = image_barriers.size();
            info.pImageMemoryBarriers = image_barriers.data();

            vkCmdPipelineBarrier2(cmd, &info);

            flush_uploads();
        }
    );

    destroy_buffer(allocator, scratch);

    return images;
}

void optimize_mesh(
    std::vector<Vertex>& vertices,
    std::vector<uint32_t>& indices,
    GltfPrimitive& mesh,
    std::vector<Vertex>& m_vertices,
    std::vector<uint32_t>& m_indices,
    std::vector<Meshlet>& m_meshlets,
    std::vector<uint32_t>& meshlet_indices
)
{
    // Index filtering
    indices.resize(meshopt_filterIndexBuffer(indices.data(), indices.data(), indices.size(), vertices.data(), vertices.size(), sizeof(uint16_t) * 3, sizeof(Vertex)));

    // Indexing
    std::vector<unsigned int> remap(vertices.size());
    size_t vertex_count = meshopt_generateVertexRemap(remap.data(), indices.data(), indices.size(), vertices.data(), vertices.size(), sizeof(Vertex));

    meshopt_remapIndexBuffer(indices.data(), indices.data(), indices.size(), remap.data());
    meshopt_remapVertexBuffer(vertices.data(), vertices.data(), vertices.size(), sizeof(Vertex), remap.data());

    vertices.resize(vertex_count);

    // Vertex cache optimization
    meshopt_optimizeVertexCache(indices.data(), indices.data(), indices.size(), vertices.size());

    // TODO: (Optional) Overdraw optimization

    // Vertex fetch optimization
    meshopt_optimizeVertexFetch(vertices.data(), indices.data(), indices.size(), vertices.data(), vertices.size(), sizeof(Vertex));

    // Dequantize for mesh center and radius
    glm::vec3 center{};

    std::vector<glm::vec3> positions(vertex_count);
    for (size_t i = 0; i < vertex_count; i++)
    {
        float px = meshopt_dequantizeHalf(vertices[i].px);
        float py = meshopt_dequantizeHalf(vertices[i].py);
        float pz = meshopt_dequantizeHalf(vertices[i].pz);

        positions[i] = glm::vec3(px, py, pz);
        center += positions[i];
    }

    std::vector<glm::vec3> normals(vertex_count);
    for (size_t i = 0; i < vertex_count; i++)
    {
        auto n = vertices[i].normal;
        glm::vec3 normal = glm::vec3((n >> 20) & 1023, (n >> 10) & 1023, n & 1023) / glm::vec3(511.0) - glm::vec3(1.0);
        normals[i] = normal;
    }

    center /= vertices.size();
    float radius = 0.0;

    for (size_t i = 0; i < vertex_count; i++)
    {
        radius = std::max(radius, glm::distance(center, positions[i]));
    }

    mesh.center = center;
    mesh.radius = radius;

    // Simplification
    constexpr uint32_t max_lod = 8;
    float lod_error_scale = meshopt_simplifyScale(&positions[0].x, vertices.size(), sizeof(glm::vec3));
    float target_error = 1e-1f;
    float lod_error = 0.f;

    constexpr float attr_weights[3] = { 1.f, 1.f, 1.f }; // For normals
    float next_error{};
    float simplify_threshold = 0.6f;

    constexpr size_t max_vertices = 64;
    constexpr size_t max_triangles = 126;
    constexpr float cone_weight = 0.f; // 0 if not cone culling; 0.25 otherwise for a good default

    auto base_vertex = static_cast<uint32_t>(m_vertices.size());
    auto data_offset = static_cast<uint32_t>(meshlet_indices.size());
    auto meshlets_offset = static_cast<uint32_t>(m_meshlets.size());

    while (mesh.lod_count < max_lod)
    {
        uint32_t first_index = static_cast<uint32_t>(m_indices.size());
        uint32_t index_count = static_cast<uint32_t>(indices.size());

        m_indices.insert(m_indices.end(), indices.begin(), indices.end());

        size_t target_index_count = (static_cast<size_t>(indices.size() * simplify_threshold) / 3) * 3;

        auto max_meshlets = meshopt_buildMeshletsBound(indices.size(), max_vertices, max_triangles);
        std::vector<meshopt_Meshlet> meshlets(max_meshlets);
        std::vector<unsigned int> meshlet_vertices(indices.size());
        std::vector<unsigned char> meshlet_triangles(indices.size());

        size_t meshlet_count = meshopt_buildMeshlets(
            meshlets.data(),
            meshlet_vertices.data(),
            meshlet_triangles.data(),
            indices.data(),
            indices.size(),
            &positions[0].x,
            vertices.size(),
            sizeof(float) * 3,
            max_vertices,
            max_triangles,
            cone_weight
        );

        const meshopt_Meshlet& last = meshlets[meshlet_count - 1];

        meshlet_vertices.resize(last.vertex_offset + last.vertex_count);
        meshlet_triangles.resize(last.triangle_offset + last.triangle_count * 3);
        meshlets.resize(meshlet_count);

        for (auto& m : meshlets)
        {
            // Further optimizing each meshlet in isolation for better triangle and vertex locality
            meshopt_optimizeMeshlet(&meshlet_vertices[m.vertex_offset], &meshlet_triangles[m.triangle_offset], m.triangle_count, m.vertex_count);

            meshopt_Bounds bounds = meshopt_computeMeshletBounds(
                &meshlet_vertices[m.vertex_offset],
                &meshlet_triangles[m.triangle_offset],
                m.triangle_count,
                &positions[0].x,
                vertices.size(),
                sizeof(float) * 3
            );

            uint32_t min_vertex = -1u;
            uint32_t max_vertex = 0;

            // Compress vertices into uint16_t if possible
            for (auto i = 0; i < m.vertex_count; i++)
            {
                auto idx = m.vertex_offset + i;
                min_vertex = std::min(meshlet_vertices[idx], min_vertex);
                max_vertex = std::max(meshlet_vertices[idx], max_vertex);
            }

            bool pack_vertex = (max_vertex - min_vertex) < (1u << 16);
            for (auto i = 0; i < m.vertex_count; i++)
            {
                auto v = meshlet_vertices[m.vertex_offset + i] - min_vertex;
                if (pack_vertex && (i & 1u))
                {
                    meshlet_indices.back() |= v << 16;
                }
                else
                {
                    meshlet_indices.push_back(v);
                }
            }

            // Compress triangles into uint8_t
            auto index_group = reinterpret_cast<unsigned int*>(meshlet_triangles.data() + m.triangle_offset);
            auto index_group_count = (m.triangle_count * 3 + 3) / 4;

            for (auto i = 0; i < index_group_count; i++)
            {
                meshlet_indices.push_back(index_group[i]);
            }

            Meshlet new_meshlet{};
            new_meshlet.cx = meshopt_quantizeHalf(bounds.center[0]);
            new_meshlet.cy = meshopt_quantizeHalf(bounds.center[1]);
            new_meshlet.cz = meshopt_quantizeHalf(bounds.center[2]);
            new_meshlet.radius = meshopt_quantizeHalf(bounds.radius);
            new_meshlet.base_vertex = base_vertex + min_vertex;
            new_meshlet.data_offset = data_offset;
            new_meshlet.pack_vertex = pack_vertex;
            new_meshlet.vertex_count = m.vertex_count;
            new_meshlet.triangle_count = m.triangle_count;
            m_meshlets.push_back(new_meshlet);

            data_offset = static_cast<uint32_t>(meshlet_indices.size());
        }

        MeshLod lod{
            .first_index = first_index,
            .index_count = index_count,
            .error = lod_error * lod_error_scale,
            .meshlet_offset = meshlets_offset,
            .meshlet_count = static_cast<uint32_t>(meshlet_count),
        };

        mesh.mesh_lods[mesh.lod_count++] = lod;

        meshlets_offset = static_cast<uint32_t>(m_meshlets.size());

        unsigned int options = meshopt_SimplifyErrorClamped;

        if (mesh.lod_count < max_lod)
        {
            size_t new_size = meshopt_simplifyWithAttributes(
                indices.data(),
                indices.data(),
                indices.size(),
                &positions[0].x,
                vertices.size(),
                sizeof(glm::vec3),
                &normals[0].x,
                sizeof(glm::vec3),
                &attr_weights[0],
                3,
                nullptr,
                target_index_count,
                target_error,
                options,
                &next_error
            );

            assert(new_size <= indices.size());

            // Reached error bound
            if (new_size == indices.size() || new_size == 0)
            {
                break;
            }

            // Discard LOD if too similar to previous LOD, saves memory
            if (new_size >= static_cast<size_t>(indices.size() * 0.85f))
            {
                break;
            }

            indices.resize(new_size);

            // REVIEW - Since we start from last LOD, we need to accumulate error?
            lod_error = glm::max(lod_error * 1.5f, next_error);

            meshopt_optimizeVertexCache(indices.data(), indices.data(), indices.size(), vertices.size());
        }
    }

    m_vertices.insert(m_vertices.end(), vertices.begin(), vertices.end());
}
} // namespace

bool AssetLoader::load_gltf(
    VkQueue queue,
    VkFence fence,
    VkCommandPool command_pool,
    VkCommandBuffer cmd,
    const std::string& gltf_file,
    uint32_t texture_offset
)
{
    std::filesystem::path asset_path = "../assets/" + gltf_file;
    fmt::println("loading glTF: {}", asset_path.string());

    // TODO: implement enums to select at runtime
    constexpr auto supported_extensions =
        fastgltf::Extensions::KHR_texture_basisu
        | fastgltf::Extensions::KHR_materials_transmission
        | fastgltf::Extensions::KHR_materials_volume;

    fastgltf::Parser parser(supported_extensions);

    constexpr auto gltf_options{
        fastgltf::Options::DontRequireValidAssetMember | fastgltf::Options::AllowDouble | fastgltf::Options::LoadExternalBuffers
    };

    auto gltf_data_buffer = fastgltf::GltfDataBuffer::FromPath(asset_path);

    if (gltf_data_buffer.error() != fastgltf::Error::None)
    {
        fmt::println("Error: {}", fastgltf::getErrorName(gltf_data_buffer.error()));
        return false;
    }

    fastgltf::Asset asset{};

    auto gltf_path = asset_path.parent_path();
    auto type = fastgltf::determineGltfFileType(gltf_data_buffer.get());
    if (type == fastgltf::GltfType::glTF)
    {
        auto load = parser.loadGltf(gltf_data_buffer.get(), gltf_path, gltf_options);
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
        auto load{ parser.loadGltfBinary(gltf_data_buffer.get(), gltf_path, gltf_options) };
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

    if (!asset.images.empty())
    {
        images = load_images(asset, device, queue, fence, command_pool, cmd, allocator, gltf_path.string());
    }

    assert(!asset.materials.empty() && "No materials found");

    for (fastgltf::Material& mat : asset.materials)
    {
        MaterialData mat_data{};
        mat_data.base_color_factor.x = mat.pbrData.baseColorFactor[0];
        mat_data.base_color_factor.y = mat.pbrData.baseColorFactor[1];
        mat_data.base_color_factor.z = mat.pbrData.baseColorFactor[2];
        mat_data.base_color_factor.w = mat.pbrData.baseColorFactor[3];
        mat_data.metallic_factor = mat.pbrData.metallicFactor;
        mat_data.roughness_factor = mat.pbrData.roughnessFactor;

        // TODO: handle emissiveStrength
        mat_data.emissive_factor.x = mat.emissiveFactor[0];
        mat_data.emissive_factor.y = mat.emissiveFactor[1];
        mat_data.emissive_factor.z = mat.emissiveFactor[2];

        if (mat.pbrData.baseColorTexture.has_value())
        {
            size_t image_index =
                asset.textures[mat.pbrData.baseColorTexture.value().textureIndex].imageIndex
                ? asset.textures[mat.pbrData.baseColorTexture.value().textureIndex].imageIndex.value()
                : asset.textures[mat.pbrData.baseColorTexture.value().textureIndex].basisuImageIndex.value();

            mat_data.diffuse_id = static_cast<uint32_t>(image_index) + texture_offset;
        }

        if (mat.pbrData.metallicRoughnessTexture.has_value())
        {
            size_t image_index =
                asset.textures[mat.pbrData.metallicRoughnessTexture.value().textureIndex].imageIndex
                ? asset.textures[mat.pbrData.metallicRoughnessTexture.value().textureIndex].imageIndex.value()
                : asset.textures[mat.pbrData.metallicRoughnessTexture.value().textureIndex].basisuImageIndex.value();

            mat_data.metal_roughness_id = static_cast<uint32_t>(image_index) + texture_offset;
        }

        if (mat.normalTexture.has_value())
        {
            size_t image_index =
                asset.textures[mat.normalTexture.value().textureIndex].imageIndex
                ? asset.textures[mat.normalTexture.value().textureIndex].imageIndex.value()
                : asset.textures[mat.normalTexture.value().textureIndex].basisuImageIndex.value();

            mat_data.normal_id = static_cast<uint32_t>(image_index) + texture_offset;
        }

        if (mat.occlusionTexture.has_value())
        {
            size_t image_index =
                asset.textures[mat.occlusionTexture.value().textureIndex].imageIndex
                ? asset.textures[mat.occlusionTexture.value().textureIndex].imageIndex.value()
                : asset.textures[mat.occlusionTexture.value().textureIndex].basisuImageIndex.value();

            mat_data.occlusion_id = static_cast<uint32_t>(image_index) + texture_offset;
        }

        if (mat.emissiveTexture.has_value())
        {
            size_t image_index =
                asset.textures[mat.emissiveTexture.value().textureIndex].imageIndex
                ? asset.textures[mat.emissiveTexture.value().textureIndex].imageIndex.value()
                : asset.textures[mat.emissiveTexture.value().textureIndex].basisuImageIndex.value();

            mat_data.emissive_id = static_cast<uint32_t>(image_index) + texture_offset;
        }

        materials.push_back(mat_data);
    }

    for (fastgltf::Mesh& mesh : asset.meshes)
    {
        GltfMesh new_mesh{};

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

            GltfPrimitive mesh_data{};
            if (p.materialIndex.has_value())
            {
                mesh_data.material_id = static_cast<uint32_t>(p.materialIndex.value());
            }
            else
            {
                assert(0);
            }

            optimize_mesh(vertices, indices, mesh_data, m_vertices, m_indices, m_meshlets, meshlet_indices);
            new_mesh.mesh.push_back(mesh_data);
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
