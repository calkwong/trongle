#pragma once

#define FRAMES_IN_FLIGHT 2

#define MAX_LOD 8
#define MAX_MESH_VERTS 64
#define MAX_MESH_TRIS 126

#define MAX_DESCRIPTORS_IMAGE_COUNT 2000
#define MAX_DESCRIPTORS_SAMPLER_COUNT 10

// This should be set to maxMeshWorkGroupCount, whilst also respecting maxMeshWorkGroupTotalCount
// We are exceeding the limit here and likely operating under UB
#define MAX_MESHLETS_DRAW (4194304 * 4)
#define MESHLET_DRAW_MASKED_OFFSET (MAX_MESHLETS_DRAW / 2)

#define LOCAL_LIGHTS 1
#define STRESS_TEST 0
#define VOLUMETRIC_FOG 1
#define FOG_WIDTH 128
#define FOG_HEIGHT 128
#define FOG_DEPTH 128

#if STRESS_TEST == 1
#define MAX_INSTANCES 1000000
#else
#define MAX_INSTANCES 4000
#endif

#define SUN_CASCADE_COUNT 4
#define SHADOWMAP_EXTENT 4096
#define SHADOW_DRAW_DISTANCE 48.0f

#define MAX_DEBUG_ID 13

// TODO: make this configurable
#define FROXELS_FAR_PLANE 40.0

#define FROXELS_X 16
#define FROXELS_Y 8
#define FROXELS_Z 24
#define TOTAL_FROXELS (FROXELS_X * FROXELS_Y * FROXELS_Z)

#define MAX_POINT_LIGHTS 1000
#define MAX_LIGHTS_PER_FROXEL 100

#define DISPATCH_EARLY_OPAQUE 1
#define DISPATCH_EARLY_MASKED 2
#define DISPATCH_LATE_OPAQUE 3
#define DISPATCH_LATE_MASKED 4
