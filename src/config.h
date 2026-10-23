#pragma once

#define FRAMES_IN_FLIGHT 2

#define MAX_LOD 8
#define MAX_MESH_VERTS 64
#define MAX_MESH_TRIS 126

#define MAX_DESCRIPTORS_IMAGE_COUNT 2000

// Check limit: VkPhysicalDeviceMeshShaderPropertiesEXT -> maxMeshWorkGroupCount
#define MAX_MESHLETS_DRAW 4194304

#define STRESS_TEST 0
#if STRESS_TEST == 1
#define MAX_INSTANCES 1000000
#else
#define MAX_INSTANCES 4000
#endif

#define SUN_CASCADE_COUNT 4
#define SHADOWMAP_EXTENT 4096
#define SHADOW_DRAW_DISTANCE 48.0f

#define MAX_DEBUG_ID 12
