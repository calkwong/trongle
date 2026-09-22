#pragma once

#include "resources.h"
#include "shared_cpu_gpu.h"

#include <glm/ext/matrix_float4x4.hpp>
#include <volk.h>

#include <vector>
#include <string>
#include <memory>
#include <cstdint>

struct MeshAsset
{
    std::vector<MeshData> mesh;
};

struct Node
{
    std::shared_ptr<MeshAsset> mesh_asset;
    std::vector<std::shared_ptr<Node>> children;

    glm::mat4 local_transform;
    glm::mat4 world_transform;

    void refresh_transform(const glm::mat4& parent_matrix);
};

struct AssetLoader
{
    VkDevice device;
    VmaAllocator allocator;

    std::vector<Image> images;
    std::vector<MeshData> meshes;
    std::vector<Vertex> m_vertices;
    std::vector<uint32_t> m_indices;

    std::vector<std::shared_ptr<Node>> top_nodes;

    void cleanup();
    bool load_gltf(VkQueue queue, VkFence fence, VkCommandPool command_pool, VkCommandBuffer cmd, const std::string& file_path);
};
