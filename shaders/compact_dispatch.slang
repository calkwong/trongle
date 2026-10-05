import bindings;
import types;
import prefix_sum;

struct PushConstants
{
    PrefixSum* prefix_sum_buffer;
    Dispatch* dispatch_buffer;
};

[[vk::push_constant]]
PushConstants pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    // lower 32 bits only
    uint count = uint(get_prefix_sum_counter(pc.prefix_sum_buffer));

    const uint threads = 256;

    pc.dispatch_buffer[0].x = (count + threads - 1) / threads;
    pc.dispatch_buffer[0].y = 1;
    pc.dispatch_buffer[0].z = 1;

    pc.dispatch_buffer[1].x = 0;
    pc.dispatch_buffer[1].y = 1;
    pc.dispatch_buffer[1].z = 1;
}

