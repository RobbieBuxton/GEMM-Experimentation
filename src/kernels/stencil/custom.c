#define START_TIMER(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP_TIMER(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;
#define THREAD_NUMBER 8

#include "custom.h"
#include <stdio.h>
#include "../../utils.h"
#include "sys/time.h"
#include <stdlib.h>
#include <string.h>
#include <cblas.h>
// #include "matrix_helpers.h"
#include <math.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif


int custom_linear_convection_kernel(struct dataobj *restrict u_vec, const float dt, const float h_x, const float h_y, const int x0_blk0_size, const int y0_blk0_size, const int time_M, const int time_m, const int x_M, const int x_m, const int y_M, const int y_m, struct profiler * timers, float * result) {
	START_TIMER(section0)
	float a = 0.13;
	float c = 0.25;
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

	
	diagonalize_matrix(V, n, n, PVT, DV, PVINV);
	diagonalize_matrix(H, n, n, PHT, DH, PHINV);


	float *T = calloc(sizeof(float), n * n);

	
  //Combine PV^-1 * S * PH = T
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, PVINV, n, S, n, 0.0, temp1, n);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, n, n, 1.0, temp1, n, PHT, n, 0.0, T, n);


	float *H_eigen = malloc(sizeof(float) * n);
	float *V_eigen = malloc(sizeof(float) * n);
	float *HN_eigen = malloc(sizeof(float) * n);
	
	// Converts the diagonal to eigen vectors
	for (int i = 0; i < n; i++)
	{
		H_eigen[i] = DH[i + i * n];
		V_eigen[i] = DV[i + i * n];
	}

	
	
	#pragma omp parallel num_threads(THREAD_NUMBER)
	{
		#pragma omp for collapse(1) schedule(static,1)
		for (int i = 0; i < n; i++)
			{
			for (int j = 0; j < n; j++)
			{
				temp2[i + n * j] = T[i + n * j] * powf(H_eigen[i] + V_eigen[j], iterations);
			}
		}
	}

	

	cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, n, n, 1.0, PHT, n, temp2, n, 0.0, temp1, n);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, temp1, n, PVINV, n, 0.0, result, n);


	free(H);	
	free(PHT);
	free(DH);
	free(PHINV);
	free(V);
	free(PVT);
	free(DV);
	free(PVINV);
	free(temp1);
	free(temp2);
	STOP_TIMER(section0,timers)
	return 0;
}

// This is not generalised atm and only works for square matrices with that are toepliz tridiagonal
void diagonalize_matrix(float *A, int n, int m, float *PT, float *D, float *PINV)
{
	float a = A[n];
	float b = A[n + 1];
	float c = A[n + 2];
	// printf("%f %f %f\n",a,b,c);
	float *eigen_values = malloc(sizeof(float) * n);

	float pwr;
	float trig;
	#pragma omp parallel num_threads(THREAD_NUMBER)
	{
		#pragma omp for collapse(1) schedule(static,1)
		for (int j = 0; j < n; j++)
		{
			eigen_values[j] = b + 2 * sqrtf(a * c) * cosf(((n - j) * M_PI) / (n + 1));
			for (int i = 0; i < n; i++)
			{
				pwr = (n-i)/2.0;
				trig = sinf(((n-i)*(n-j)*M_PI)/(n+1));

				PT[j * n + i] = powf(a/c,pwr)*trig;
				PINV[i * n + j] = powf(c/a,pwr)*trig;
			}
		}
	}
	int column_spacing = 1;
	float* p = calloc(sizeof(float),n);
	// scalling
	for (int i = 0; i < n; i++)
	{
		p[i] = 1 / sdot_(&n, PT + i * n, &column_spacing, PINV + i * n, &column_spacing);
	}

	#pragma omp parallel num_threads(THREAD_NUMBER)
	{
		#pragma omp for collapse(1) schedule(static,1)
		for (int i = 0; i < n; i++)
		{
			D[i + n * i] = eigen_values[i];
			for (int k = 0; k < n; k++){
				PT[i * n + k] *= p[i];
			}
		}
	}
}