module prefix_sum;

#language slang 2026

import types;

public void prefix_sum_inclusive_append(PrefixSum* buf, uint index, uint value, uint offset)
{
    uint64_t upper_bits_one = uint64_t(1) << uint64_t(32);
    uint64_t sum_and_counter;
    InterlockedAdd(buf.counter, upper_bits_one | uint64_t(value), sum_and_counter);
    uint instance_index = uint(sum_and_counter >> 32);

    buf.data[instance_index].instance_id = index;
    buf.data[instance_index].sum = uint(sum_and_counter) + value; // inclusive
    // buf.data[instance_index].sum = uint(sum_and_counter); // exclusive
    buf.data[instance_index].meshlet_offset = offset;
    buf.data[instance_index].meshlet_count = value;
}

public uint prefix_sum_binary_search(PrefixSum* buf, uint target)
{
    uint count = uint(buf.counter >> 32);
    uint first = 0;

    while (count > 0)
    {
        uint step = count / 2;
        uint current = first + step;
        bool greater = target >= buf.data[current].sum;
        first = greater ? (current + 1) : first; // left or right path
        count = greater ? (count - (step + 1)) : step; // size of left or right path
    }

    return first; // inclusive
    // return first - 1; // exclusive
}

public uint64_t get_prefix_sum_counter(PrefixSum* buf)
{
    return buf.counter;
}

