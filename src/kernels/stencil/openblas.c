#define START_TIMER(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP_TIMER(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include "openblas.h"
#include <stdio.h>
#include "../../utils.h"
#include <stdlib.h>
#include <string.h>
#include <cblas.h>

int openblas_linear_convection_kernel(struct dataobj *restrict u_vec, const float dt, const float h_x, const float h_y, const int i0x0_blk0_size, const int i0x_ltkn, const int i0x_rtkn, const int i0y0_blk0_size, const int i0y_ltkn, const int i0y_rtkn, const int time_M, const int time_m, const int x_M, const int x_m, const int y_M, const int y_m, struct profiler * timers) {
	
	int row_size =  u_vec->size[1]*sizeof(float);
	
	float** transform = malloc(sizeof(float*)*2);
	// -1 shifted index
	transform[0] = calloc(sizeof(float),(u_vec->size[1])*(u_vec->size[1]));
	transform[0][u_vec->size[1]+1] = 0.2;
	transform[0][2*(u_vec->size[1]+1)] = 0.2;
	transform[0][3*(u_vec->size[1]+1)] = 0.2;
	transform[0][4*(u_vec->size[1]+1)] = 0.2;
	transform[0][5*(u_vec->size[1]+1)] = 0.2;

	// printf("Transformation 0\n");
	// print_matrix(transform[0],u_vec->size[1],u_vec->size[1]);

	// 0 shifted index
	transform[1] = calloc(sizeof(float),(u_vec->size[1])*(u_vec->size[1]));
	transform[1][1] = -0.2;
	transform[1][(u_vec->size[1] + 1)] = 0.6;
	transform[1][(u_vec->size[1] + 1) + 1] = 0.2;
	transform[1][2*(u_vec->size[1] + 1)] = 0.6;
	transform[1][2*(u_vec->size[1] + 1) + 1] = 0.2;
	transform[1][3*(u_vec->size[1] + 1)] = 0.6;
	transform[1][3*(u_vec->size[1] + 1) + 1] = 0.2;
	transform[1][4*(u_vec->size[1] + 1)] = 0.6;
	transform[1][4*(u_vec->size[1] + 1) + 1] = 0.2;
	transform[1][5*(u_vec->size[1] + 1)] = 0.6;

	// printf("Transformation 1\n");
 	// print_matrix(transform[1],u_vec->size[1],u_vec->size[1]);

	// printf("Extended Stencil\n");
	// print_matrix(u_vec->data,u_vec->size[1],u_vec->size[2]);

	

	float* stencils[2];
	stencils[0] = u_vec->data + row_size ;
	stencils[1] = u_vec->data + row_size * ((u_vec->size[1]) + 2);
	for (int t = time_m, t0 = t%2, t1 = (t+1)%2; t <= time_M; t++, t0 = t%2, t1 = (t+1)%2) {
		printf("t0: %d t1: %d\n",t0,t1);
		printf("Stencil %d\n",t);
		print_matrix(stencils[t0],u_vec->size[1],u_vec->size[1]);
		//Multiply
		cblas_sgemm(
		CblasRowMajor,					
		CblasNoTrans,						
		CblasNoTrans,						
		u_vec->size[1]-1,	// Each time you go up you cut off a row							
		u_vec->size[1],								
		u_vec->size[1],							
		1.0,										
		stencils[t0] - row_size, 	
		u_vec->size[1],								
		transform[0],		
		u_vec->size[1], 								
		0.0, 										
		stencils[t1] , 	
		u_vec->size[1]);

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
		0.0, 										
		stencils[t1], 	
		u_vec->size[1]);
	}
	printf("Stencil %d\n",time_M+1);
	print_matrix(stencils[(time_M+1)%2],u_vec->size[1],u_vec->size[1]);


	free(transform[0]);
	free(transform[1]);
	free(transform);
	return 0;
}