// References
// https://www.elopezr.com/temporal-aa-and-the-quest-for-the-holy-trail/
// https://alextardif.com/TAA.html
// https://github.com/playdeadgames/temporal/blob/master/Assets/Shaders/TemporalReprojection.shader
// https://advances.realtimerendering.com/s2014/#_HIGH-QUALITY_TEMPORAL_SUPERSAMPLING
// https://developer.download.nvidia.com/gameworks/events/GDC2016/msalvi_temporal_supersampling.pdf

import bindings;
import types;
import util;

struct PushConstants
{
    float4 jitter_offset;
    GlobalData* global_buffer;
    float2 screen_size;
    uint current_id;
    uint history_id;
    uint resolve_id;
    uint depth_id;
    // uint velocity_id;
    uint variance_clipping;
    uint history_filter;
    uint ycocg;
    uint valid_history; // valid when first_frame==false
    uint dynamic_blend;
};

[[vk::push_constant]]
PushConstants pc;

// https://github.com/TheRealMJP/MSAAFilter/blob/master/MSAAFilter/Resolve.hlsl
// note: there is some scaling we may have to do
float filter_cubic(float x, float B, float C)
{
    float y = 0.0f;
    float x2 = x * x;
    float x3 = x * x * x;
    if (x < 1)
        y = (12 - 9 * B - 6 * C) * x3 + (-18 + 12 * B + 6 * C) * x2 + (6 - 2 * B);
    else if (x <= 2)
        y = (-B - 6 * C) * x3 + (6 * B + 30 * C) * x2 + (-12 * B - 48 * C) * x + (8 * B + 24 * C);

    return y / 6.0;
}

// note: there is some scaling we may have to do
// filter per axis and multiply their respective weights (see MJP link above),
// or use length? https://github.com/h3r2tic/rtoy-samples/blob/main/assets/shaders/taa.glsl
float filter_mitchell(float value)
{
    return filter_cubic(value, 1.0 / 3.0, 1.0 / 3.0);
}

// TODO: account for exposure
// https://www.youtube.com/watch?v=nj4puag4hwc
// https://gpuopen.com/learn/optimized-reversible-tonemapper-for-resolve/
float3 simple_tonemap(float3 c)
{
    return c;

    // float w = max(c.x, max(c.y, c.z));
    // return c / (1.0 + w);
}

float3 reverse_tonemap(float3 c)
{
    return c;

    // float w = max(c.x, max(c.y, c.z));
    // return c / (1.0 - w);
}

float3 maybe_rgb_to_ycocg(float3 c)
{
    if (pc.ycocg == 0)
        return c;

    float y = c.x / 4.0 + c.y / 2.0 + c.z / 4.0;
    float co = c.x / 2.0 - c.z / 2.0;
    float cg = -c.x / 4.0 + c.y / 2.0 - c.z / 4.0;

    return float3(y, co, cg);
}

float3 maybe_ycocg_to_rgb(float3 c)
{
    if (pc.ycocg == 0)
        return c;

    float tmp = c.x - c.z;
    float r = tmp + c.y;
    float g = c.r + c.z;
    float b = tmp - c.y;

    return clamp(float3(r, g, b), float3(0.0), float3(1.0));
}

float3 sample_color(uint texture_index, SamplerState samp, float2 uv)
{
    return simple_tonemap(get_sampled_image<Texture2D>(texture_index).SampleLevel(samp, uv, 0.0).xyz);
}

float luminance(float3 c)
{
    return (pc.ycocg == 1) ? c.x : maybe_rgb_to_ycocg(c).x;
}

// 9 tap exploiting bilinear filtering
// https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1
// http://float3.ca/bicubic-filtering-in-fewer-taps/
float3 sample_texture_catmull_rom(float2 uv, uint texture_index, float2 resolution)
{
    // We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding
    // down the sample location to get the exact center of our "starting" texel. The starting texel will be at
    // location [1, 1] in the grid, where [0, 0] is the top left corner.
    float2 sample_position = uv * resolution;
    float2 tex_pos_1 = floor(sample_position - 0.5) + 0.5;

    // Compute the fractional offset from our starting texel to our original sample location, which we'll
    // feed into the Catmull-Rom spline function to get our filter weights.
    float2 f = sample_position - tex_pos_1;

    // Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
    // These equations are pre-expanded based on our knowledge of where the texels will be located,
    // which lets us avoid having to evaluate a piece-wise function.
    float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    float2 w3 = f * f * (-0.5 + 0.5 * f);

    // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
    // simultaneously evaluate the middle 2 samples from the 4x4 grid.
    float2 w12 = w1 + w2;
    float2 offset_12 = w2 / w12;

    // Compute the final UV coordinates we'll use for sampling the texture
    float2 tex_pos_0 = tex_pos_1 - 1;
    float2 tex_pos_3 = tex_pos_1 + 2;
    float2 tex_pos_12 = tex_pos_1 + offset_12;

    tex_pos_0 /= resolution;
    tex_pos_3 /= resolution;
    tex_pos_12 /= resolution;

    SamplerState linear_clamp_samp = get_sampler<SamplerState>(LINEAR_CLAMP_SAMPLER);

    float3 result = float3(0);
    Texture2D tex = get_sampled_image<Texture2D>(texture_index);

    result += simple_tonemap(tex.SampleLevel(linear_clamp_samp, float2(tex_pos_0.x, tex_pos_0.y), 0.0).rgb) * w0.x * w0.y;
    result += simple_tonemap(tex.SampleLevel(linear_clamp_samp, float2(tex_pos_12.x, tex_pos_0.y), 0.0).rgb) * w12.x * w0.y;
    result += simple_tonemap(tex.SampleLevel(linear_clamp_samp, float2(tex_pos_3.x, tex_pos_0.y), 0.0).rgb) * w3.x * w0.y;
    result += simple_tonemap(tex.SampleLevel(linear_clamp_samp, float2(tex_pos_0.x, tex_pos_12.y), 0.0).rgb) * w0.x * w12.y;
    result += simple_tonemap(tex.SampleLevel(linear_clamp_samp, float2(tex_pos_12.x, tex_pos_12.y), 0.0).rgb) * w12.x * w12.y;
    result += simple_tonemap(tex.SampleLevel(linear_clamp_samp, float2(tex_pos_3.x, tex_pos_12.y), 0.0).rgb) * w3.x * w12.y;
    result += simple_tonemap(tex.SampleLevel(linear_clamp_samp, float2(tex_pos_0.x, tex_pos_3.y), 0.0).rgb) * w0.x * w3.y;
    result += simple_tonemap(tex.SampleLevel(linear_clamp_samp, float2(tex_pos_12.x, tex_pos_3.y), 0.0).rgb) * w12.x * w3.y;
    result += simple_tonemap(tex.SampleLevel(linear_clamp_samp, float2(tex_pos_3.x, tex_pos_3.y), 0.0).rgb) * w3.x * w3.y;

    return result;
}

// Optimized ver from Inside/playdeadgames
float4 clip_aabb(float3 aabb_min, float3 aabb_max, float4 q, float average_alpha)
{
    // Only clips towards aabb center (but fast!)
    float3 p_clip = 0.5 * (aabb_max + aabb_min);
    float3 e_clip = 0.5 * (aabb_max - aabb_min) + 0.00000001;

    float4 v_clip = q - float4(p_clip, average_alpha);
    float3 v_unit = v_clip.xyz / e_clip;
    float3 a_unit = abs(v_unit);
    float ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));

    if (ma_unit > 1.0)
        return float4(p_clip, average_alpha) + v_clip / ma_unit;
    else
    {
        // Point is inside AABB
        return q;
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(
    uint3 global_id: SV_DispatchThreadID
)
{
    int2 texel_coord = int2(global_id.xy);

    if (texel_coord.x >= pc.screen_size.x || texel_coord.y >= pc.screen_size.y)
        return;

    SamplerState nearest_samp = get_sampler<SamplerState>(NEAREST_SAMPLER);
    SamplerState linear_clamp_samp = get_sampler<SamplerState>(LINEAR_CLAMP_SAMPLER);
    RWTexture2D tex = get_storage_image<RWTexture2D>(pc.resolve_id);

    float2 uv = float2(texel_coord) + float2(0.5);
    uv /= pc.screen_size;

    // TODO: on swapchain resize, we should ideally not reset this
    // Frame 0 or on swapchain resize
    if (pc.valid_history == 0)
    {
        float3 history_color = get_sampled_image<Texture2D>(pc.current_id).SampleLevel(nearest_samp, uv, 0.0).xyz;
        tex.Store(texel_coord, float4(history_color, 1.0));
        return;
    }

    // Ignore or accumulate clamped edges for variance clipping?
    float3 min_color = float3(99999.0);
    float3 max_color = float3(-99999.0);

    // Used in variance clipping
    float3 m1 = float3(0.);
    float3 m2 = float3(0.);

    // Neighbourhood sampling
    float3 current_color;
    for (int x = -1; x <= 1; x++)
    {
        for (int y = -1; y <= 1; y++)
        {
            float3 col = sample_color(pc.current_id, nearest_samp, uv + (float2(x, y) / pc.screen_size));
            col = maybe_rgb_to_ycocg(col);

            if (x == 0 && y == 0)
                current_color = col;

            // If mitchell is implemented, remember to ignore weights when computing min/max_color
            min_color = min(col, min_color);
            max_color = max(col, max_color);
            m1 += col;
            m2 += (col * col);
        }
    }

    // Depth dilation - during reprojection, velocity itself is aliased, and we could be reintroducing edge aliasing indirectly.
    // To better account for edges, dilate aliased info by taking pixel with nearest depth in neighbourhood
    float closest_depth = 0.0;
    float2 closest_uv_offset;
    for (int x = -1; x <= 1; x++)
    {
        for (int y = -1; y <= 1; y++)
        {
            float2 uv_offset = float2(x, y) / pc.screen_size;
            float neighbour_depth = get_sampled_image<Texture2D>(pc.depth_id).SampleLevel(nearest_samp, uv + uv_offset, 0.0).r;
            if (neighbour_depth > closest_depth)
            {
                closest_depth = neighbour_depth;
                closest_uv_offset = uv_offset;
            }
        }
    }

    // note: our scene is entirely static now, so we recompute it here
    float current_depth = get_sampled_image<Texture2D>(pc.depth_id).SampleLevel(nearest_samp, uv, 0.0).r;
    float2 current_ndc = uv * 2.0 - 1.0;
    current_ndc.y *= -1.0;
    float4 world_pos = mul(pc.global_buffer.inverse_view_proj, float4(current_ndc, current_depth, 1.0));
    world_pos /= world_pos.w;
    float4 previous_clip = mul(pc.global_buffer.previous_view_proj, world_pos);
    float2 previous_ndc = previous_clip.xy / previous_clip.w;
    current_ndc -= pc.jitter_offset.xy;
    previous_ndc -= pc.jitter_offset.zw;
    float2 camera_velocity = (current_ndc - previous_ndc) * float2(0.5, -0.5);

    // REVIEW - without this, setting blend_factor to 1.0 is not sufficient as nan has propogated to the blending eq
    camera_velocity = current_depth != 0.0 ? camera_velocity : float2(0.0);

    // TODO: object motion - we are not currently using true motion vectors
    // float2 object_velocity = texture(sampler2D(get_sampled_image<Texture2D>(velocity_id), nearest_samp), uv + closest_uv_offset).rg; // uv space
    float2 reprojected_uv = uv - camera_velocity; // - object_velocity;

    // TODO: catmull 5-taps from COD?
    float3 history_color;
    if (pc.history_filter == 1)
        history_color = sample_texture_catmull_rom(reprojected_uv, pc.history_id, pc.screen_size);
    else
        history_color = sample_color(pc.history_id, linear_clamp_samp, reprojected_uv);

    history_color = maybe_rgb_to_ycocg(history_color);

    // https://developer.download.nvidia.com/gameworks/events/GDC2016/msalvi_temporal_supersampling.pdf
    float3 history_color_clamped;
    if (pc.variance_clipping == 1)
    {
        float rcp_samples = 1.0 / 9.0;
        // large gamma - temporally stable results at the cost of increased ghosting; small gamma - lose ability to integrate data over time
        float gamma = 1.0;
        float3 mu = m1 * rcp_samples;
        float3 sigma = sqrt(abs((m2 * rcp_samples) - (mu * mu)));
        float3 minc = mu - gamma * sigma;
        float3 maxc = mu + gamma * sigma;
        history_color_clamped = clip_aabb(minc, maxc, float4(clamp(history_color, min_color, max_color), 1.0), 1.0).xyz;
    }
    else
    {
        // Neighbourhood clamping trades off ghosting for flickering - toggle to see.
        history_color_clamped = history_color;
    }

    float blend_factor = 0.05;

    // TODO: explore other blending heuristics; a straightforward one is when reprojected uv is OOB
    // https://www.youtube.com/watch?v=WzpLWzGvFK4&t=18m
    if (pc.dynamic_blend == 1)
    {
        float min_blend_factor = 0.03;
        float max_blend_factor = 0.12;

        float luma_history = luminance(history_color_clamped);
        float luma_current = luminance(current_color);

        float diff = 1.0 - abs(luma_history - luma_current) / max(max(luma_history, luma_current), 0.2);
        float w = diff * diff; // w == 0, low similarity -> lower history contribution; w == 1, high similarity -> higher history contribution

        blend_factor = max_blend_factor + (min_blend_factor - max_blend_factor) * w;
    }

    float3 final_color = history_color_clamped + (current_color - history_color_clamped) * blend_factor;

    final_color = maybe_ycocg_to_rgb(final_color);
    final_color = reverse_tonemap(final_color);

    // tex.Store(texel_coord, float4(tonemap(final_color), 1.0));
    tex.Store(texel_coord, float4(final_color, 1.0));
}
