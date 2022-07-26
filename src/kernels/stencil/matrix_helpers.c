#include "matrix_helpers.h"
#include "custom.h"
#include <stdio.h>
#include "../../utils.h"
#include "sys/time.h"
#include <stdlib.h>
#include <string.h>
#include <cblas.h>

void test() {

	int n = 3;

	float* H = calloc(sizeof(float),n*n);
	H[0] = 0.25;
	H[1] = 0.1;
	H[2] = 0;
	H[3] = 0.1;
	H[4] = 0.25;
	H[5] = 0.1;
	H[6] = 0;
	H[7] = 0.1;
	H[8] = 0.25;

	float *PT = calloc(sizeof(float),n*n);
	float *D = calloc(sizeof(float),n*n);
	float* PINV = calloc(sizeof(float),n*n);

	diagonalize_matrix(H,n,n,PT,D,PINV);

	float* temp = calloc(sizeof(float),n*n);
	float* output = calloc(sizeof(float),n*n);

	cblas_sgemm(
	CblasRowMajor,					
	CblasTrans,						
	CblasNoTrans,						
	n,			
	n,								
	n,							
	1.0,										
	PT, 	
	n,	
	D,		
	n, 								
	0.0, 										
	temp, 	
	n);

	cblas_sgemm(
	CblasRowMajor,					
	CblasNoTrans,						
	CblasNoTrans,						
	n,			
	n,								
	n,							
	1.0,										
	temp, 	
	n,	
	PINV,		
	n, 								
	0.0, 										
	output, 	
	n);

	printf("Output\n");
	print_matrix(output,n,n);
}

// A = (PT)^T * D * PINV
void diagonalize_matrix(float* A, int n, int m, float* PT, float* D, float* PINV) {

	float real_eigen_values[n];
	float im_eigen_values[n];
	int lwork = -1;
	float wkopt; 
	float* work; 
	int info;

	print_matrix(A,n,n);
  sgeev_("N", "N", &n, A, &n, real_eigen_values, im_eigen_values, PT, &n, PINV, &n, &wkopt, &lwork, &info );
	lwork = (int)wkopt;
  work = (float*)malloc( lwork*sizeof(float) );
	sgeev_( "V", "V", &n, A, &n, real_eigen_values, im_eigen_values, PT, &n, PINV, &n, work, &lwork, &info );

	int column_spacing = 1;
	float p;
	//Ok this isn't great but not actually sure about the maths of why exactly it's scaled by the dot product here to denormalise but stack overflow told me to do it and it works: https://stackoverflow.com/questions/72069026/matrix-diagonalization-and-basis-change-with-geev 
	for (int i = 0; i < n; i++) {
		p = 1/sdot_(&n,PT+i*n,&column_spacing,PINV+i*n,&column_spacing);
		PT[i*n] *= p;
		PT[i*n+1] *= p;
		PT[i*n+2] *= p;
	}

	for (int i = 0; i < n; i ++) {
		D[i + n*i] = real_eigen_values[i];
	}
	
	free((void*)work);
}