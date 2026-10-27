module pbr;

#define DISNEY_DIFFUSE 0

static const float PI = 3.14159265359;

public float D_GGX(float NoH, float roughness)
{
    float one_minus_NoH_squared = 1.0 - NoH * NoH;
    float a = NoH * roughness;
    float k = roughness / (one_minus_NoH_squared + a * a);
    float d = k * k * (1.0 / PI);
    return k * k * (1.0 / PI);
}

public float V_SmithGGXCorrelated(float NoV, float NoL, float roughness)
{
    float a2 = roughness * roughness;
    float GGXV = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
    float GGXL = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
    return 0.5 / (GGXV + GGXL);
}

public float3 F_Schlick(float u, float3 f0)
{
    float f = pow(1.0 - u, 5.0);
    return f + f0 * (1.0 - f);
}

public float F_Schlick(float u, float f0, float f90)
{
    return f0 + (f90 - f0) * pow(1.0 - u, 5.0);
}

public float3 F_SchlickRoughness(float u, float3 f0, float roughness)
{
	return f0 + (max(float3(1.0 - roughness), f0) - f0) * pow(1.0 - u, 5.0);
}

// TODO: verify
public float get_square_falloff_attenuation(float3 distance, float radius)
{
    float distance_square = dot(distance, distance);
    float light_inv_radius = 1.0 / radius;
    float factor = distance_square * light_inv_radius * light_inv_radius;
    float smooth_factor = max(1.0 - factor * factor, 0.0);
    return (smooth_factor * smooth_factor) / max(distance_square, 1e-4);
}

public float Fd_Lambert()
{
    return 1.0 / PI;
}

public float Fd_Burley(float NoV, float NoL, float LoH, float roughness)
{
    float f90 = 0.5 + 2.0 * roughness * LoH * LoH;
    float light_scatter = F_Schlick(NoL, 1.0, f90);
    float view_scatter = F_Schlick(NoV, 1.0, f90);
    return light_scatter * view_scatter * (1.0 / PI);
}

public struct BRDFData
{
    public float3 N;
    public float3 L;
    public float3 V;
    public float3 H;
    public float3 f0;

    public float NoL;
    public float NoH;
    public float NoV;
    public float VoH;
    public float LoH;
}

public BRDFData prepare_brdf_data(float3 N, float3 L, float3 V, float3 albedo, float roughness, float metallic)
{
    BRDFData data;

    data.N = N;
    data.L = L;
    data.V = V;

    float3 H = normalize(L + V);
    data.H = H;

    data.NoL = max(dot(N, L), 0.0);
    data.NoH = max(dot(N, H), 0.0);
    data.NoV = max(dot(N, V), 0.001);
    data.VoH = max(dot(V, H), 0.0);
    data.LoH = max(dot(L, H), 0.0);

    // TODO: this should be a param
    float3 f0 = float3(0.04);
    data.f0 = lerp(f0, albedo, metallic);

    return data;
}

// for point lights only
public void update_brdf_data(inout BRDFData data_out, float3 L)
{
    data_out.L = L;
    data_out.H = normalize(data_out.L + data_out.V);

    data_out.NoL = max(dot(data_out.N, data_out.L), 0.0);
    data_out.NoH = max(dot(data_out.N, data_out.H), 0.0);
    data_out.VoH = max(dot(data_out.V, data_out.H), 0.0);
    data_out.LoH = max(dot(data_out.L, data_out.H), 0.0);
}

public float3 eval_brdf(BRDFData brdf_data, float3 albedo, float roughness, float metallic)
{
    float3 diffuse = eval_diffuse_brdf(brdf_data, albedo, roughness, metallic);
    float3 F_out;
    float3 specular = eval_specular_brdf(brdf_data, roughness, F_out);

#if DISNEY_DIFFUSE == 0
    diffuse *= (1.0 - F_out);
#endif
    return diffuse + specular;
}

public float3 eval_diffuse_brdf(BRDFData brdf_data, float3 albedo, float roughness, float metallic)
{
#if DISNEY_DIFFUSE == 1
    return Fd_Burley(brdf_data.NoV, brdf_data.NoL, brdf_data.LoH, roughness) * albedo.xyz * (1.0 - metallic);
#else
    return Fd_Lambert() * albedo.xyz * (1.0 - metallic);
#endif
}

public float3 eval_specular_brdf(BRDFData brdf_data, float roughness, inout float3 F_out)
{
    float D = D_GGX(brdf_data.NoH, roughness);
    float V = V_SmithGGXCorrelated(brdf_data.NoV, brdf_data.NoL, roughness);

    // TODO: VoH or LoH? Filament is using LoH.
    F_out = F_Schlick(brdf_data.VoH, brdf_data.f0);

    return D * V * F_out;
}
