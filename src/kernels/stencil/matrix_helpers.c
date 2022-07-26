#include "matrix_helpers.h"
#include <stdio.h>
#include "../../utils.h"
#include "sys/time.h"
#include <stdlib.h>
#include <string.h>
#include <cblas.h>
#include <math.h>

void test()
{
	int iterations = 2;
	int n = 3;

	float *S = calloc(sizeof(float), n * n);
	S[0] = 1;
	S[1] = 1;
	S[2] = 1;
	S[3] = 1;
	S[4] = 2;
	S[5] = 2;
	S[6] = 1;
	S[7] = 2;
	S[8] = 2;

	float *V = calloc(sizeof(float), n * n);
	V[0] = 0.25;
	V[1] = 0.1;
	V[2] = 0;
	V[3] = 0.1;
	V[4] = 0.25;
	V[5] = 0.1;
	V[6] = 0;
	V[7] = 0.1;
	V[8] = 0.25;

	float *H = calloc(sizeof(float), n * n);
	H[0] = 0.25;
	H[1] = 0.1;
	H[2] = 0;
	H[3] = 0.1;
	H[4] = 0.25;
	H[5] = 0.1;
	H[6] = 0;
	H[7] = 0.1;
	H[8] = 0.25;

	float *PHT = calloc(sizeof(float), n * n);
	float *DH = calloc(sizeof(float), n * n);
	float *PHINV = calloc(sizeof(float), n * n);

	float *PVT = calloc(sizeof(float), n * n);
	float *DV = calloc(sizeof(float), n * n);
	float *PVINV = calloc(sizeof(float), n * n);

	float *temp1 = calloc(sizeof(float), n * n);
	float *temp2 = calloc(sizeof(float), n * n);
	float *result = calloc(sizeof(float), n * n);

	diagonalize_matrix(V, n, n, PVT, DV, PVINV);
	diagonalize_matrix(H, n, n, PHT, DH, PHINV);

	float *T = calloc(sizeof(float), n * n);

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, PVINV, n, S, n, 0.0, temp1, n);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, n, n, 1.0, temp1, n, PHT, n, 0.0, T, n);

	float *b_table = generate_binomial_table(iterations);
	float *H_eigen = malloc(sizeof(float) * n);
	float *V_eigen = malloc(sizeof(float) * n);
	float *HN_eigen = malloc(sizeof(float) * n);
	for (int i = 0; i < n; i++)
	{
		H_eigen[i] = DH[i + i * n];
		V_eigen[i] = DV[i + i * n];
		HN_eigen[i] = powf(DH[i + i * n], iterations);
	}
	float l;
	float cumL;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			l = V_eigen[i] / H_eigen[j];
			cumL = 1;
			for (int k = 0; k < iterations + 1; k++)
			{
				result[i + n * j] += b_table[k] * cumL;
				cumL *= l;
			}
			result[i + n * j] *= T[i + n * j] * HN_eigen[j];
		}
	}

	cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, n, n, 1.0, PHT, n, result, n, 0.0, temp1, n);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, temp1, n, PVINV, n, 0.0, temp2, n);

	printf("actual output\n");
	print_matrix(temp2, n, n);
}

// A = (PT)^T * D * PINV
void diagonalize_matrix(float *A, int n, int m, float *PT, float *D, float *PINV)
{

	float real_eigen_values[n];
	float im_eigen_values[n];
	int lwork = -1;
	float wkopt;
	float *work;
	int info;

	sgeev_("N", "N", &n, A, &n, real_eigen_values, im_eigen_values, PT, &n, PINV, &n, &wkopt, &lwork, &info);
	lwork = (int)wkopt;
	work = (float *)malloc(lwork * sizeof(float));
	sgeev_("V", "V", &n, A, &n, real_eigen_values, im_eigen_values, PT, &n, PINV, &n, work, &lwork, &info);

	int column_spacing = 1;
	float p;
	// Ok this isn't great but not actually sure about the maths of why exactly it's scaled by the dot product here to denormalise but stack overflow told me to do it and it works: https://stackoverflow.com/questions/72069026/matrix-diagonalization-and-basis-change-with-geev
	for (int i = 0; i < n; i++)
	{
		p = 1 / sdot_(&n, PT + i * n, &column_spacing, PINV + i * n, &column_spacing);
		PT[i * n] *= p;
		PT[i * n + 1] *= p;
		PT[i * n + 2] *= p;
	}

	for (int i = 0; i < n; i++)
	{
		D[i + n * i] = real_eigen_values[i];
	}

	free((void *)work);
}

// Beware of overflow here
float *generate_binomial_table(int n)
{
	float *table = malloc(sizeof(float) * n + 1);
	table[0] = 1;
	for (int k = 0; k < n; ++k)
		table[k + 1] = (table[k] * (n - k)) / (k + 1);
	return table;
}
