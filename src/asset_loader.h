#pragma once

#include "resources.h"
#include "shared_cpu_gpu.h"

#include <glm/ext/vector_float3.hpp>
#include <glm/ext/matrix_float4x4.hpp>
#include <volk.h>

#include <vector>
#include <string>
#include <cstdint>

struct GltfPrimitive
{
    MeshLod mesh_lods[8];

    glm::vec3 center;
    float radius;

    uint32_t lod_count;
    uint32_t material_id;
};

struct GltfMesh
{
    std::vector<GltfPrimitive> mesh;
};

struct Node
{
    glm::mat4 local_transform;
    glm::mat4 world_transform;

    uint32_t mesh_index = -1;
    uint32_t first_child;
    uint32_t child_count;

    void refresh_transform(const glm::mat4& parent_matrix, std::vector<Node>& children);
};

struct AssetLoader
{
    VkDevice device;
    VmaAllocator allocator;

    std::vector<Image> images;
    std::vector<GltfMesh> meshes;
    std::vector<Vertex> m_vertices;
    std::vector<uint32_t> m_indices;
    std::vector<MaterialData> materials;

    std::vector<Meshlet> m_meshlets;
    std::vector<uint32_t> meshlet_indices;

    std::vector<Node> parent_nodes;
    std::vector<Node> children_nodes;

    void cleanup();
    bool load_gltf(VkQueue queue, VkFence fence, VkCommandPool command_pool, VkCommandBuffer cmd, const std::string& file_path, uint32_t texture_offset);
};
