import bindings;
import types;

#include "../src/config.h"

struct PushConstants
{
    PrefixSums* prefix_sum_buffer;
    uint* light_count_buffer;
    uint zero_lights;
};

[[vk::push_constant]]
PushConstants pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    for (uint i = 0; i < SUN_CASCADE_COUNT; i++)
    {
        pc.prefix_sum_buffer.prefix_sums[i].counter = 0;
    }

    if (pc.zero_lights == 1)
        *pc.light_count_buffer = 0;
}
