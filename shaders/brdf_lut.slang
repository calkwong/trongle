import bindings;
import sampling;
import pbr;

struct PushConstants
{
	float2 image_size;
	uint brdf_lut_id;
};

[[vk::push_constant]] PushConstants pc;

float2 IntegratedBRDF(float NoV, float perceptual_roughness)
{
	float3 V = float3(0.0);
	V.x = sqrt(1.0 - NoV * NoV);
    V.y = 0.0;
    V.z = NoV;

	float3 N = float3(0.0, 0.0, 1.0);
	float linear_roughness = perceptual_roughness * perceptual_roughness;

	float scale = 0.0;
	float bias = 0.0;

	uint sample_count = 1024;
	for (uint i = 0; i < sample_count; i++)
	{
		float2 Xi = Hammersley(i, sample_count);
		float3 H = ImportanceSampleGGX(Xi, N, linear_roughness);
		float3 L = normalize(2.0 * dot(V,H) * H - V);

		float NoL = max(L.z, 0.0);
		float NoH = max(H.z, 0.0);
		float VoH = max(dot(V, H), 0.0);

		if (NoL > 0.0)
		{
			float V_pdf = V_SmithGGXCorrelated(NoV, NoL, linear_roughness) * VoH * NoL / NoH;
			float Fc = pow(1.0 - VoH, 5.0);
			scale += V_pdf * (1.0 - Fc);
			bias += V_pdf * Fc;
		}
	}

	return 4.0 * float2(scale, bias) / float(sample_count);
}

[shader("compute")]
[numthreads(32, 32, 1)]
void main(
    uint3 global_id : SV_DispatchThreadID
)
{
	float2 texel_coord = float2(global_id.xy);
	float2 uv = (texel_coord + float2(0.5)) / pc.image_size;

	float2 color = IntegratedBRDF(uv.x, uv.y);

	get_storage_image<RWTexture2D<float2>>(pc.brdf_lut_id).Store(int2(texel_coord), color);
}
