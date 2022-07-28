#define _USE_MATH_DEFINES
#include <math.h>
#include "matrix_helpers.h"
#include <stdio.h>
#include "../../utils.h"
#include "sys/time.h"
#include <stdlib.h>
#include <string.h>
#include <cblas.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif


// void test()
// {
// 	int iterations = 1;
// 	int n = 5;
// 	float a = 0.1;
// 	float b = 0.5;
// 	float c = b/2;

// 	float *S = calloc(sizeof(float), n * n);
// 	for (int i = 0; i < n*n; i++) {
// 		S[i] = 1;
// 	}
// 	S[n + 1] = 2;
// 	S[n + 2] = 2;
// 	S[2*n + 1] = 2;
// 	S[2*n + 2] = 2;

// 	// Create Vertical Transform               
// 	float* V = calloc(sizeof(float),(n)*(n));
// 	V[0] = c;
// 	V[1] = a;
// 	for (int i = 1; i < n - 1; i ++) {
// 		V[(i)*(n+1)-1] = a;
// 		V[(i)*(n+1)] = c;
// 		V[(i)*(n+1)+1] = a;
// 	}
// 	V[n*n-2] = a;
// 	V[n*n-1] = c;

// 	// Create Horizontal Transform
// 	float* H = calloc(sizeof(float),(n)*(n));
// 	H[0] = c;
// 	H[n] = a;
// 	for (int i = 1; i < n - 1; i ++) {
// 		H[(i-1)*(n+1)+1] = a;
// 		H[i*(n+1)] = c;
// 		H[(i+1)*(n+1)-1] = a;
// 	}
// 	H[n*(n-1) -1] = a;
// 	H[n*(n) -1] = c;


// 	float *PHT = calloc(sizeof(float), n * n);
// 	float *DH = calloc(sizeof(float), n * n);
// 	float *PHINV = calloc(sizeof(float), n * n);

// 	float *PVT = calloc(sizeof(float), n * n);
// 	float *DV = calloc(sizeof(float), n * n);
// 	float *PVINV = calloc(sizeof(float), n * n);

// 	float *temp1 = calloc(sizeof(float), n * n);
// 	float *result1 = calloc(sizeof(float), n * n);

// 	float *temp2 = calloc(sizeof(float), n * n);
// 	float *result2 = calloc(sizeof(float), n * n);


// 	diagonalize_matrix(V, n, n, PVT, DV, PVINV);
// 	diagonalize_matrix2(H, n, n, PHT, DH, PHINV);

// 	printf("Lapack method\n");
// 	print_matrix(PVT,n,n);
// 	print_matrix(DV,n,n);
// 	print_matrix(PVINV,n,n);
// 	cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, n, n, 1.0, PVT, n, DV, n, 0.0, temp1, n);
// 	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, temp1, n, PVINV, n, 0.0, result1, n);
// 	printf("Lapack result\n");
// 	print_matrix(result1,n,n);

// 	printf("custom method\n");
// 	print_matrix(PHT,n,n);
// 	print_matrix(DH,n,n);
// 	print_matrix(PHINV,n,n);
// 	cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, n, n, 1.0, PHT, n, DH, n, 0.0, temp1, n);
// 	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, temp1, n, PHINV, n, 0.0, result1, n);
// 	printf("Custom result\n");
// 	print_matrix(result1,n,n);

// }

// // A = (PT)^T * D * PINV
// void diagonalize_matrix(float *A, int n, int m, float *PT, float *D, float *PINV)
// {

// 	float real_eigen_values[n];
// 	float im_eigen_values[n];
// 	int lwork = -1;
// 	float wkopt;
// 	float *work;
// 	int info;

// 	sgeev_("N", "N", &n, A, &n, real_eigen_values, im_eigen_values, PT, &n, PINV, &n, &wkopt, &lwork, &info);
// 	lwork = (int)wkopt;
// 	work = (float *)malloc(lwork * sizeof(float));
// 	sgeev_("V", "V", &n, A, &n, real_eigen_values, im_eigen_values, PT, &n, PINV, &n, work, &lwork, &info);

// 	int column_spacing = 1;
// 	float p;
// 	// Ok this isn't great but not actually sure about the maths of why exactly it's scaled by the dot product here to denormalise but stack overflow told me to do it and it works: https://stackoverflow.com/questions/72069026/matrix-diagonalization-and-basis-change-with-geev
// 	for (int i = 0; i < n; i++)
// 	{
// 		p = 1 / sdot_(&n, PT + i * n, &column_spacing, PINV + i * n, &column_spacing);
// 		PT[i * n] *= p;
// 		PT[i * n + 1] *= p;
// 		PT[i * n + 2] *= p;
// 	}

// 	for (int i = 0; i < n; i++)
// 	{
// 		D[i + n * i] = real_eigen_values[i];
// 	}

// 	free((void *)work);
// }

// void diagonalize_matrix2(float *A, int n, int m, float *PT, float *D, float *PINV)
// {
// 	float a = A[n];
// 	float b = A[n + 1];
// 	float c = A[n + 2];
// 	// printf("%f %f %f\n",a,b,c);
// 	float *eigen_values = malloc(sizeof(float) * n);
// 	for (int i = 0; i < n; i++)
// 	{
// 		eigen_values[i] = b + 2 * sqrtf(a * c) * cosf(((n - (i)) * M_PI) / (n + 1));
// 	}

// 	for (int j = 0; j < n; j++)
// 	{
// 		for (int i = 0; i < n; i++)
// 		{
// 			PT[j * n + i] = powf((a/c),((n-i)/2.0))*sinf(((n-i)*(n-j)*M_PI)/(n+1));
// 			PINV[i * n + j] = powf((c/a),((n-i)/2.0))*sinf(((n-i)*(n-j)*M_PI)/(n+1));
// 		}
// 	}

// 	int column_spacing = 1;
// 	float p;
// 	// scalling
// 	for (int i = 0; i < n; i++)
// 	{
// 		p = 1 / sdot_(&n, PT + i * n, &column_spacing, PINV + i * n, &column_spacing);
// 		for (int k = 0; k < n; k++){
// 			PT[i * n + k] *= p;
// 		}
// 	}

// 	for (int i = 0; i < n; i++)
// 	{
// 		D[i + n * i] = eigen_values[i];
// 	}
// }

// Beware of overflow here
float *generate_binomial_table(int n)
{
	float *table = malloc(sizeof(float) * n + 1);
	table[0] = 1;
	for (int k = 0; k < n; ++k)
		table[k + 1] = (table[k] * (n - k)) / (k + 1);
	return table;
}
