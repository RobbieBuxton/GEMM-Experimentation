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

	// Alternating stencils like how devito does it
	float* stencils[2];
	stencils[0] = u_vec->data;
	stencils[1] = u_vec->data +  u_vec->size[1]*sizeof(float) * ((u_vec->size[1]));

	
	float** transform = malloc(sizeof(float*)*2);

	// Vertical Transform               
	transform[0] = calloc(sizeof(float),(u_vec->size[1])*(u_vec->size[1]));
	transform[0][0] = c;
	transform[0][1] = a;
	for (int i = 1; i < u_vec->size[1] - 1; i ++) {
		transform[0][(i)*(u_vec->size[1]+1)-1] = a;
		transform[0][(i)*(u_vec->size[1]+1)] = c;
		transform[0][(i)*(u_vec->size[1]+1)+1] = a;
	}
	transform[0][u_vec->size[1]*u_vec->size[1]-2] = a;
	transform[0][u_vec->size[1]*u_vec->size[1]-1] = c;

	// printf("Vertical Transform \n");
	// print_matrix(transform[0],u_vec->size[1],u_vec->size[1]);

	// printf("Stencil\n");
	// print_matrix(stencils[0],u_vec->size[1],u_vec->size[1]);

	// Horizontal Transform
	transform[1] = calloc(sizeof(float),(u_vec->size[1])*(u_vec->size[1]));
	transform[1][0] = c;
	transform[1][u_vec->size[1]] = a;
	for (int i = 1; i < u_vec->size[1] - 1; i ++) {
		transform[1][(i-1)*(u_vec->size[1]+1)+1] = a;
		transform[1][i*(u_vec->size[1]+1)] = c;
		transform[1][(i+1)*(u_vec->size[1]+1)-1] = a;
	}
	transform[1][u_vec->size[1]*(u_vec->size[1]-1) -1] = a;
	transform[1][u_vec->size[1]*(u_vec->size[1]) -1] = c;

	// printf("Horizontal Transform\n");
 	// print_matrix(transform[1],u_vec->size[1],u_vec->size[1]);

	// print_matrix(stencils[1],u_vec->size[1],u_vec->size[1]);


	START_TIMER(section0)
	for (int t = time_m, t0 = t%2, t1 = (t+1)%2; t <= time_M; t++, t0 = t%2, t1 = (t+1)%2) {
		// printf("t0: %d t1: %d\n",t0,t1);
		// printf("Stencil %d\n",t);
		// print_matrix(stencils[t0],u_vec->size[1],u_vec->size[1]);
		
		

		//Multiply vertical  
		cblas_sgemm(
		CblasRowMajor,					
		CblasNoTrans,						
		CblasNoTrans,						
		u_vec->size[1],			
		u_vec->size[1],								
		u_vec->size[1],							
		1.0,										
		transform[0], 	
		u_vec->size[1],	
		stencils[t0],		
		u_vec->size[1], 								
		0.0, 										
		stencils[t1], 	
		u_vec->size[1]);

		//Multiply horizontal
		cblas_sgemm(
		CblasRowMajor,					
		CblasNoTrans,						
		CblasNoTrans,						
		u_vec->size[1],								
		u_vec->size[1],								
		u_vec->size[1],							
		1.0,										
		stencils[t0], 	
		u_vec->size[1],								
		transform[1],		
		u_vec->size[1], 								
		1.0, 										
		stencils[t1], 	
		u_vec->size[1]);
	}
	STOP_TIMER(section0,timers)
	
	if (u_vec->size[1] < 30) {
		printf("Stencil %d\n",time_M+1);
		print_matrix(stencils[(time_M+1)%2],u_vec->size[1],u_vec->size[1]);

	}

	char jobvl = 'N';
	char jobvr = 'N';
	int n = u_vec->size[1];
	int lda = n;
	float wr[n];
	float wi[n];
	int ldvl = n;
	float vl[ldvl*n];
	int ldvr = n;
	float vr[ldvr*n];
	int lwork = -1;
	float wkopt; 
	float* work; 
	int info;

  sgeev_( "Vectors", "Vectors", &n, transform[1], &lda, wr, wi, vl, &ldvl, vr, &ldvr,&wkopt, &lwork, &info );
	lwork = (int)wkopt;
  work = (float*)malloc( lwork*sizeof(float) );
	printf("Info: %d\n",info);
	
	
	free(transform[0]);
	free(transform[1]);
	free(transform);
	
	return 0;
}