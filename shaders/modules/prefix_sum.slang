module prefix_sum;

#language slang 2026

import types;

public void prefix_sum_inclusive_append(PrefixSums* buf, uint prefix_sum_index, uint index, uint value, uint offset)
{
    uint64_t upper_bits_one = uint64_t(1) << uint64_t(32);
    uint64_t sum_and_counter;
    InterlockedAdd(buf.prefix_sums[prefix_sum_index].counter, upper_bits_one | uint64_t(value), sum_and_counter);
    uint instance_index = uint(sum_and_counter >> 32);

    buf.prefix_sums[prefix_sum_index].data[instance_index].sum = uint(sum_and_counter) + value;
    buf.prefix_sums[prefix_sum_index].data[instance_index].meshlet_offset = offset;
    buf.prefix_sums[prefix_sum_index].data[instance_index].meshlet_count = value;
    buf.prefix_sums[prefix_sum_index].data[instance_index].instance_id = index;
}

public uint prefix_sum_binary_search(PrefixSums* buf, uint prefix_sum_index, uint target)
{
    uint count = uint(buf.prefix_sums[prefix_sum_index].counter >> 32);
    uint first = 0;

    while (count > 0)
    {
        uint step = count / 2;
        uint current = first + step;
        bool greater = target >= buf.prefix_sums[prefix_sum_index].data[current].sum;
        first = greater ? (current + 1) : first; // left or right path
        count = greater ? (count - (step + 1)) : step; // size of left or right path
    }

    return first;
}

public uint64_t get_prefix_sum_counter(PrefixSums* buf, uint prefix_sum_index)
{
    return buf.prefix_sums[prefix_sum_index].counter;
}
