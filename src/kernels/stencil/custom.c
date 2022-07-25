#define START_TIMER(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP_TIMER(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include "custom.h"
#include <stdio.h>
#include "../../utils.h"
#include "sys/time.h"
#include <stdlib.h>
#include <string.h>
#include <cblas.h>

int custom_linear_convection_kernel(struct dataobj *restrict u_vec, const float dt, const float h_x, const float h_y, const int x0_blk0_size, const int y0_blk0_size, const int time_M, const int time_m, const int x_M, const int x_m, const int y_M, const int y_m, struct profiler * timers) {
	
	float a = 0.1;
	float b = 0.5;
	float c = b/2;
	int n = u_vec->size[1];

	// Alternating stencils like how devito does it
	float* stencils[2];
	stencils[0] = u_vec->data;
	stencils[1] = u_vec->data +  n*sizeof(float) * ((n));

	
	float** transform = malloc(sizeof(float*)*2);

	// Vertical Transform               
	transform[0] = calloc(sizeof(float),(n)*(n));
	transform[0][0] = c;
	transform[0][1] = a;
	for (int i = 1; i < n - 1; i ++) {
		transform[0][(i)*(n+1)-1] = a;
		transform[0][(i)*(n+1)] = c;
		transform[0][(i)*(n+1)+1] = a;
	}
	transform[0][n*n-2] = a;
	transform[0][n*n-1] = c;

	// printf("Vertical Transform \n");
	// print_matrix(transform[0],n,n);

	// printf("Stencil\n");
	// print_matrix(stencils[0],n,n);

	// Horizontal Transform
	transform[1] = calloc(sizeof(float),(n)*(n));
	transform[1][0] = c;
	transform[1][n] = a;
	for (int i = 1; i < n - 1; i ++) {
		transform[1][(i-1)*(n+1)+1] = a;
		transform[1][i*(n+1)] = c;
		transform[1][(i+1)*(n+1)-1] = a;
	}
	transform[1][n*(n-1) -1] = a;
	transform[1][n*(n) -1] = c;

	// printf("Horizontal Transform\n");
 	// print_matrix(transform[1],n,n);

	// print_matrix(stencils[1],n,n);


	START_TIMER(section0)
	for (int t = time_m, t0 = t%2, t1 = (t+1)%2; t <= time_M; t++, t0 = t%2, t1 = (t+1)%2) {
		// printf("t0: %d t1: %d\n",t0,t1);
		// printf("Stencil %d\n",t);
		// print_matrix(stencils[t0],n,n);
		
		

		//Multiply vertical  
		cblas_sgemm(
		CblasRowMajor,					
		CblasNoTrans,						
		CblasNoTrans,						
		n,			
		n,								
		n,							
		1.0,										
		transform[0], 	
		n,	
		stencils[t0],		
		n, 								
		0.0, 										
		stencils[t1], 	
		n);

		//Multiply horizontal
		cblas_sgemm(
		CblasRowMajor,					
		CblasNoTrans,						
		CblasNoTrans,						
		n,								
		n,								
		n,							
		1.0,										
		stencils[t0], 	
		n,								
		transform[1],		
		n, 								
		1.0, 										
		stencils[t1], 	
		n);
	}
	STOP_TIMER(section0,timers)
	
	// if (n < 30) {
	// 	printf("Stencil %d\n",time_M+1);
	// 	print_matrix(stencils[(time_M+1)%2],n,n);
	// }

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

	print_matrix(transform[0],n,n);
  sgeev_("N", "N", &n, transform[0], &lda, wr, wi, vl, &ldvl, vr, &ldvr,&wkopt, &lwork, &info );
	lwork = (int)wkopt;
  work = (float*)malloc( lwork*sizeof(float) );
	sgeev_( "V", "V", &n, transform[0], &lda, wr, wi, vl, &ldvl, vr, &ldvr,work, &lwork, &info );
	
	printf("Eiegen values\n");
	print_matrix(wr,n,1);
	printf("Eiegen vectors left\n");
	print_matrix(vl,n,n);
	printf("Eiegen vectors right\n");
	print_matrix(vr,n,n);

	printf("sdot %f\n",sdot_(&n,vl,&n,vr,&n));


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

	print_matrix(output,n,n);

	free((void*)work);
	free(transform[0]);
	free(transform[1]);
	free(transform);
	
	return 0;
}