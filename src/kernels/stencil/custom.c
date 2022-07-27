#define START_TIMER(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP_TIMER(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include "custom.h"
#include <stdio.h>
#include "../../utils.h"
#include "sys/time.h"
#include <stdlib.h>
#include <string.h>
#include <cblas.h>
#include "matrix_helpers.h"
#include <math.h>


int custom_linear_convection_kernel(struct dataobj *restrict u_vec, const float dt, const float h_x, const float h_y, const int x0_blk0_size, const int y0_blk0_size, const int time_M, const int time_m, const int x_M, const int x_m, const int y_M, const int y_m, struct profiler * timers) {
	
	float a = 0.1;
	float b = 0.5;
	float c = b/2;
	int n = u_vec->size[1];
	int iterations = time_M;

	float* S = u_vec->data;

	// Create Vertical Transform               
	float* V = calloc(sizeof(float),(n)*(n));
	V[0] = c;
	V[1] = a;
	for (int i = 1; i < n - 1; i ++) {
		V[(i)*(n+1)-1] = a;
		V[(i)*(n+1)] = c;
		V[(i)*(n+1)+1] = a;
	}
	V[n*n-2] = a;
	V[n*n-1] = c;

	// Create Horizontal Transform
	float* H = calloc(sizeof(float),(n)*(n));
	H[0] = c;
	H[n] = a;
	for (int i = 1; i < n - 1; i ++) {
		H[(i-1)*(n+1)+1] = a;
		H[i*(n+1)] = c;
		H[(i+1)*(n+1)-1] = a;
	}
	H[n*(n-1) -1] = a;
	H[n*(n) -1] = c;


	float *PHT = calloc(sizeof(float), n * n);
	float *DH = calloc(sizeof(float), n * n);
	float *PHINV = calloc(sizeof(float), n * n);

	float *PVT = calloc(sizeof(float), n * n);
	float *DV = calloc(sizeof(float), n * n);
	float *PVINV = calloc(sizeof(float), n * n);

	float *temp1 = calloc(sizeof(float), n * n);
	float *temp2 = calloc(sizeof(float), n * n);
	float *result = calloc(sizeof(float), n * n);


	START_TIMER(section0)
	diagonalize_matrix(V, n, n, PVT, DV, PVINV);
	diagonalize_matrix(H, n, n, PHT, DH, PHINV);

	float *T = calloc(sizeof(float), n * n);

  //Combine PV^-1 * S * PH = T
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, PVINV, n, S, n, 0.0, temp1, n);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, n, n, 1.0, temp1, n, PHT, n, 0.0, T, n);

	float *b_table = generate_binomial_table(iterations);
	float *H_eigen = malloc(sizeof(float) * n);
	float *V_eigen = malloc(sizeof(float) * n);
	float *HN_eigen = malloc(sizeof(float) * n);
	
	// Converts the diagonal to eigen vectors
	for (int i = 0; i < n; i++)
	{
		H_eigen[i] = DH[i + i * n];
		V_eigen[i] = DV[i + i * n];
	}

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			result[i + n * j] = T[i + n * j] * powf(H_eigen[i] + V_eigen[j], iterations);
		}
	}

	cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, n, n, 1.0, PHT, n, result, n, 0.0, temp1, n);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, temp1, n, PVINV, n, 0.0, temp2, n);
	STOP_TIMER(section0,timers)

	if (n < 25) {
		printf("actual output\n");
		print_matrix(temp2, n, n);
	}


	
	free(V);
	free(H);	
	return 0;
}