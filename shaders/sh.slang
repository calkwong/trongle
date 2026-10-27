import bindings;
import sh_util;

struct PushConstants
{
    SH* sh_buffer;
    uint cubemap_id;
};

[[vk::push_constant]]
PushConstants pc;

#define WG_SIZE_X 32

groupshared SH9 shared_r[WG_SIZE_X];
groupshared SH9 shared_g[WG_SIZE_X];
groupshared SH9 shared_b[WG_SIZE_X];

static const int samples = 100;
static const float PI = 3.14159265359;

[shader("compute")]
[numthreads(WG_SIZE_X, 1, 1)]
void main(
    uint t_index: SV_GroupIndex,

)
{
    SH9 R = sh_zero();
    SH9 G = sh_zero();
    SH9 B = sh_zero();

    // TODO: use more workgroups
    for (float phi = t_index + 0.5; phi < samples; phi = phi + WG_SIZE_X)
    {
        for (float theta = 0.5; theta < samples; theta++)
        {
            float3 dir = uniform_sample_sphere(phi * (1.0 / samples), theta * (1.0 / samples));

            SH9 base = sh_basis(dir);

            SamplerState samp = get_sampler<SamplerState>(CUBE_SAMPLER);
            float3 color = get_sampled_image<TextureCube>(pc.cubemap_id).SampleLevel(samp, dir, 0.0).xyz;

            R = sh_sum(R, sh_scale(base, color.r));
            G = sh_sum(G, sh_scale(base, color.g));
            B = sh_sum(B, sh_scale(base, color.b));
        }
    }

    shared_r[t_index] = R;
    shared_g[t_index] = G;
    shared_b[t_index] = B;

    GroupMemoryBarrierWithGroupSync();

    if (t_index == 0)
    {
        R = shared_r[0];
        G = shared_g[0];
        B = shared_b[0];

        for (int i = 1; i < WG_SIZE_X; i++)
        {
            R = sh_sum(shared_r[i], R);
            G = sh_sum(shared_g[i], G);
            B = sh_sum(shared_b[i], B);
        }

        float factor = (4 * PI) / (samples * samples);
        R = sh_scale(R, factor);
        G = sh_scale(G, factor);
        B = sh_scale(B, factor);

        // we bake 1 / PI into this instead of computing at runtime
        SH9 coefficients;
        coefficients.c[0] = 0.5 * sqrt(1.0 / PI);
        coefficients.c[1] = (2.0 / 3.0) * -0.5 * sqrt(3.0 / PI);
        coefficients.c[2] = (2.0 / 3.0) * 0.5 * sqrt(3.0 / PI);
        coefficients.c[3] = (2.0 / 3.0) * -0.5 * sqrt(3.0 / PI);
        coefficients.c[4] = 0.25 * 0.5 * sqrt(15.0 / PI);
        coefficients.c[5] = 0.25 * -0.5 * sqrt(15.0 / PI);
        coefficients.c[6] = 0.25 * 0.25 * sqrt(5.0 / PI);
        coefficients.c[7] = 0.25 * -0.5 * sqrt(15.0 / PI);
        coefficients.c[8] = 0.25 * 0.25 * sqrt(15.0 / PI);

        R = sh_mul(R, coefficients);
        G = sh_mul(G, coefficients);
        B = sh_mul(B, coefficients);

        pc.sh_buffer->r_coefficients = R;
        pc.sh_buffer->g_coefficients = G;
        pc.sh_buffer->b_coefficients = B;
    }
}
