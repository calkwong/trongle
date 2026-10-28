import bindings;
import util;

struct PushConstants
{
    float2 screen_size;
    uint src_id;
    uint dst_id;
    uint swapchain_id;
};

[[vk::push_constant]]
PushConstants pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(
    uint3 global_id: SV_DispatchThreadID
)
{
    int2 texel_coord = int2(global_id.xy);

    if (texel_coord.x >= pc.screen_size.x || texel_coord.y >= pc.screen_size.y)
        return;

    float3 src_color = get_sampled_image<Texture2D>(pc.src_id).Load(int3(texel_coord, 0)).xyz;
    get_storage_image<RWTexture2D>(pc.dst_id).Store(int2(texel_coord), float4(src_color, 1.0));

    float3 final_color = tonemap(src_color);
    get_storage_image<RWTexture2D>(pc.src_id).Store(int2(texel_coord), float4(final_color, 1.0));
}
