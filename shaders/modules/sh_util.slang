module sh_util;

static const float PI = 3.14159265359;

public struct SH9
{
	public float c[9];
};

public struct SH
{
	public SH9 r_coefficients;
	public SH9 g_coefficients;
	public SH9 b_coefficients;
};

public SH9 sh_mul(SH9 a, SH9 b)
{
	for (int i = 0; i < 9; i++)
	{
		a.c[i] = a.c[i] * b.c[i];
	}

	return a;
}

public SH9 sh_sum(SH9 a, SH9 b)
{
	for (int i = 0; i < 9; i++)
	{
		a.c[i] = a.c[i] + b.c[i];
	}

	return a;
}

public SH9 sh_scale(SH9 a, float value)
{
	for (int i = 0; i < 9; i++)
	{
		a.c[i] *= value;
	}

	return a;
}

public float sh_dot(SH9 a, SH9 b)
{
	float value = 0.0;

	for (int i = 0; i < 9; i++)
	{
		value += a.c[i] * b.c[i];
	}

	return value;
}

public SH9 sh_basis(float3 dir)
{
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	SH9 base;
	base.c[0] = 1;           // band 0
	base.c[1] = y;           // band 1
	base.c[2] = z;
	base.c[3] = x;
	base.c[4] = x * y;       // band 2
	base.c[5] = y * z;
	base.c[6] = 3 * z * z - 1;
	base.c[7] = x * z;
	base.c[8] = x * x - y * y;

	return base;
}

public SH9 sh_zero()
{
	SH9 sh;
	for (int i = 0; i < 9; i++)
	{
		sh.c[i] = 0.0;
	}

	return sh;
}

// https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#fragment-SamplingFunctionDefinitions-6
// no sin(theta) correction needed
public float3 uniform_sample_sphere(float az, float ze)
{
	float phi = 2.0 * PI * az;
	float z = 1.0 - 2.0 * ze; // mapping from hemisphere -> sphere
	float r = sqrt(max(0.0, 1.0 - z * z));

	return float3(r * cos(phi), z, r * sin(phi));
}

public float3 evaluate_sh(SH9 r_coefficients, SH9 g_coefficients, SH9 b_coefficients, float3 direction)
{
	SH9 coefficients;
	coefficients.c[0] =  0.5 * sqrt(1.0 / PI);
	coefficients.c[1] = -0.5 * sqrt(3.0 / PI);
	coefficients.c[2] =  0.5 * sqrt(3.0 / PI);
	coefficients.c[3] = -0.5 * sqrt(3.0 / PI);
	coefficients.c[4] =  0.5 * sqrt(15.0 / PI);
	coefficients.c[5] = -0.5 * sqrt(15.0 / PI);
	coefficients.c[6] = 0.25 * sqrt(5.0 / PI);
	coefficients.c[7] = -0.5 * sqrt(15.0 / PI);
	coefficients.c[8] = 0.25 * sqrt(15.0 / PI);

	SH9 base = sh_basis(direction);

	base = sh_mul(coefficients, base);

	float r = max(sh_dot(r_coefficients, base), 0.0);
	float g = max(sh_dot(g_coefficients, base), 0.0);
	float b = max(sh_dot(b_coefficients, base), 0.0);

	return float3(r, g, b);
}
