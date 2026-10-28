A week and a half attempt at rendering a map from CS2. Fully bindless and GPU driven, the entire scene is rendered in 4 draw calls for main camera view, and 8 draw calls for 4 cascaded shadow maps (opaque + masked).

![image](https://github.com/calkwong/trongle/blob/cs2/media/cs2.png)

Meshes and meshlets are culled against frustum and hierarchical z buffer. A HiZ is built based on last frame's depth buffer, and we test visibility against this HiZ in the early pass. Meshlets that survived are rasterized into a visibility buffer where we store triangle ID, instance ID and meshlet ID, and a new HiZ is built based on this early pass. In the late pass, we cull only those that are within our frustum view and did not pass occlusion culling in the early pass. This variant of two pass culling is based on last frame depth buffer, differing from the variant in mew-engine that uses bitfields per camera view to mark last frame visibility. It scales better with scene complexity.

# Complications (unsupported)
* Material Blending
* Environment map combined Light Probe Volumes (implemented a singular skybox fallback instead)

# Tools
* Source 2 Viewer, Source 2 Viewer CLI
* KTX-Software
* meshoptimizer
* gltfpack
