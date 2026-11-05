import bindings;
import types;
import prefix_sum;

#include "../src/config.h"

struct PushConstants
{
    PrefixSums* prefix_sum_buffer;
    Dispatch* dispatch_buffer;
    uint prefix_sum_idx;
    uint late;
};

[[vk::push_constant]]
PushConstants pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    // lower 32 bits only
    uint count = uint(get_prefix_sum_counter(pc.prefix_sum_buffer, pc.prefix_sum_idx));

    const uint threads = 256;

    pc.dispatch_buffer[0].x = (count + threads - 1) / threads;
    pc.dispatch_buffer[0].y = 1;
    pc.dispatch_buffer[0].z = 1;

    uint first = pc.late == 0 ? DISPATCH_EARLY_OPAQUE : DISPATCH_LATE_OPAQUE;

    for (uint i = first; i < first + 2; i++)
    {
        pc.dispatch_buffer[i].x = 0;
        pc.dispatch_buffer[i].y = 1;
        pc.dispatch_buffer[i].z = 1;
    }
}
