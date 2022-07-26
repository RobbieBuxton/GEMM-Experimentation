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

	float* target = calloc(sizeof(float),n*n);
	target[0] = 1;
	target[3] = 2;
	target[6] = 3;
	target[1] = 4;
	target[4] = 5;
	target[7] = 6;
	target[2] = 7;
	target[5] = 8;
	target[8] = 9;

	char jobvl = 'N';
	char jobvr = 'N';
	int lda = n;
	float wr[n];
	float wi[n];
	int ldvl = n;
	float* vl = calloc(sizeof(float),ldvl*n);
	int ldvr = n;
	float *vr = calloc(sizeof(float),ldvr*n);
	int lwork = -1;
	float wkopt; 
	float* work; 
	int info;

	print_matrix(target,n,n);
  sgeev_("N", "N", &n, target, &lda, wr, wi, vl, &ldvl, vr, &ldvr,&wkopt, &lwork, &info );
	lwork = (int)wkopt;
  work = (float*)malloc( lwork*sizeof(float) );
	sgeev_( "V", "V", &n, target, &lda, wr, wi, vl, &ldvl, vr, &ldvr,work, &lwork, &info );
	
	printf("Eiegen values\n");
	print_matrix(wr,n,1);
	printf("Eiegen vectors left\n");
	print_matrix(vl,n,n);
	printf("Eiegen vectors right\n");
	print_matrix(vr,n,n);

	int t = 1;
	float p = 0;
	//Ok this isn't great but not actually sure about the maths of why exactly it's scaled by the dot product here to denormalise but stack overflow told me to do it and it works: https://stackoverflow.com/questions/72069026/matrix-diagonalization-and-basis-change-with-geev 
	for (int i = 0; i < n; i++) {
		p = 1/sdot_(&n,vl+i*n,&t,vr+i*n,&t);
		vl[i*n] *= p;
		vl[i*n+1] *= p;
		vl[i*n+2] *= p;
	}

	printf("Eiegen vectors right scaled\n");
	print_matrix(vr,n,n);


	float* diag = calloc(sizeof(float),n*n);

	for (int i = 0; i < n; i ++) {
		diag[i + n*i] = wr[i];
	}

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
	vr, 	
	n,	
	diag,		
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
	vl,		
	n, 								
	0.0, 										
	output, 	
	n);

	printf("Output\n");
	print_matrix(output,n,n);

	free((void*)work);
}