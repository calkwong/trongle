// From http://filmicworlds.com/blog/visibility-buffer-rendering-with-material-graphs/
module barycentrics;

public struct BarycentricDeriv
{
	public float3 lambda;
	public float3 ddx;
	public float3 ddy;
};

public BarycentricDeriv calculate_barycentric(float4 p0, float4 p1, float4 p2, float2 pNdc, float2 winSize)
{
	BarycentricDeriv ret;

	float3 inv_w = 1.0 / float3(p0.w, p1.w, p2.w);

	// perspective divide
	float2 ndc0 = p0.xy * inv_w.x;
	float2 ndc1 = p1.xy * inv_w.y;
	float2 ndc2 = p2.xy * inv_w.z;

	// HLSL is row major, GLSL is column major. Determinant M == Determinant transposeM.
	float inv_det = 1.0 / determinant(float2x2(ndc2 - ndc1, ndc0 - ndc1));
	// change in barycentric coords given a unit change in x/y in ndc space, perspective corrected
	// review formula, why do we multiply with inv_det again?
	ret.ddx = float3(ndc1.y - ndc2.y, ndc2.y - ndc0.y, ndc0.y - ndc1.y) * inv_det * inv_w;
	ret.ddy = float3(ndc2.x - ndc1.x, ndc0.x - ndc2.x, ndc1.x - ndc0.x) * inv_det * inv_w;
	// denominator for perspective correct barycentrics
	float ddx_sum = ret.ddx.x + ret.ddx.y + ret.ddx.z;
	float ddy_sum = ret.ddy.x + ret.ddy.y + ret.ddy.z;

	float2 delta_vec = pNdc - ndc0;
	float interp_inv_w = inv_w.x + delta_vec.x * ddx_sum + delta_vec.y * ddy_sum;
	float interp_w = 1.0 / interp_inv_w;

	ret.lambda.x = interp_w * (inv_w[0] + delta_vec.x * ret.ddx.x + delta_vec.y * ret.ddy.x);
	ret.lambda.y = interp_w * (0.0     + delta_vec.x * ret.ddx.y + delta_vec.y * ret.ddy.y);
	ret.lambda.z = interp_w * (0.0     + delta_vec.x * ret.ddx.z + delta_vec.y * ret.ddy.z);

	// ndc to pixel
	ret.ddx *= (2.0 / winSize.x);
	ret.ddy *= (2.0 / winSize.y);
	ddx_sum *= (2.0 / winSize.x);
	ddy_sum *= (2.0 / winSize.y);

	// flip as ndc is bottom-top while window coords are top down
	ret.ddy *= -1.0;
	ddy_sum *= -1.0;

	float interp_w_ddx = 1.0 / (interp_inv_w + ddx_sum);
	float interp_w_ddy = 1.0 / (interp_inv_w + ddy_sum);

	ret.ddx = interp_w_ddx * (ret.lambda * interp_inv_w + ret.ddx) - ret.lambda;
	ret.ddy = interp_w_ddy * (ret.lambda * interp_inv_w + ret.ddy) - ret.lambda;

	return ret;
}

public float4 interpolate(BarycentricDeriv bary, float4 v0, float4 v1, float4 v2)
{
	return bary.lambda.x * v0 + bary.lambda.y * v1 + bary.lambda.z * v2;
}

public float3 interpolate(BarycentricDeriv bary, float3 v0, float3 v1, float3 v2)
{
	return bary.lambda.x * v0 + bary.lambda.y * v1 + bary.lambda.z * v2;
}

public float2 interpolate(BarycentricDeriv bary, float2 v0, float2 v1, float2 v2)
{
	return bary.lambda.x * v0 + bary.lambda.y * v1 + bary.lambda.z * v2;
}

public float interpolate(BarycentricDeriv bary, float v0, float v1, float v2)
{
	return bary.lambda.x * v0 + bary.lambda.y * v1 + bary.lambda.z * v2;
}

public void interpolate_with_deriv(BarycentricDeriv bary, float2 v0, float2 v1, float2 v2, inout float2 v, inout float2 ddx, inout float2 ddy)
{
	v = bary.lambda.x * v0 + bary.lambda.y * v1 + bary.lambda.z * v2;
	ddx = bary.ddx.x * v0 + bary.ddx.y * v1 + bary.ddx.z * v2;
	ddy = bary.ddy.x * v0 + bary.ddy.y * v1 + bary.ddy.z * v2;
}
