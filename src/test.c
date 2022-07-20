#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <limits.h> 
#include <stdlib.h>
#include <cblas.h>
#include "utils.h"
#include "test.h"
#include "kernels/matrix/nvblas.h"
#include "kernels/matrix/openblas.h"
#include "kernels/matrix/devito.h"
#include "kernels/stencil/devito.h"
#include "kernels/stencil/openblas.h"


int main (int argc, char* argv[]) {

	// test_matrix_kernel(&devito_chain_contraction_kernel,11,250,5,0.05);
	// double results[2];
	// test_chain_contraction(&openblas_chain_contraction_kernel, 9, 200, 0.05, results);
	
	//Switching order causes malloc assersion problem :shrug:
	int size = 5;
	int iterations = 10;
	printf("Size: %d Iterations: %d\n",size,iterations);
	printf("####DEVITO####\n");
	test_devito_stencil_kernel(1,1,iterations,size);
	printf("####OPENBLAS####\n");
	test_openblas_stencil_kernel(1,1,iterations,size);
	
	return 0;
}

void test_devito_stencil_kernel(int steps, int step, int iterations, int size) {
	
	const float dt = 0.1;
	const float h_x = 0.5;
	const float h_y = 0.5;
	const int x0_blk0_size = 1; 
	const int y0_blk0_size = 1;
	// const int time_M = 5;
	const int time_M = iterations;
	const int time_m = 0; 
	const int x_M = size -3 ;
	const int x_m = 0;
	const int y_M = size -3; 
	const int y_m = 0; 
	struct profiler timers = {.section0 = 0};

	int width = size;
	int height = size;

	// //Init devito
	struct dataobj devito_u_vec;
	init_vector(&devito_u_vec,width,height);
	fill_stencil(devito_u_vec.data,width,height,1);
	((float *)devito_u_vec.data)[width * 2 + 2] = 2; 
	((float *)devito_u_vec.data)[width * 2 + 3] = 2; 
	((float *)devito_u_vec.data)[width * 3 + 2] = 2; 
	((float *)devito_u_vec.data)[width * 3 + 3] = 2; 

	fill_stencil(&(((float *) devito_u_vec.data)[(width)*(height)]),width,height,1);
	((float *)devito_u_vec.data)[width*height + width * 2 + 2] = 2; 
	((float *)devito_u_vec.data)[width*height + width * 2 + 3] = 2; 
	((float *)devito_u_vec.data)[width*height + width * 3 + 2] = 2; 
	((float *)devito_u_vec.data)[width*height + width * 3 + 3] = 2; 

	devito_linear_convection_kernel(&devito_u_vec, dt, h_x, h_y, x0_blk0_size, y0_blk0_size, time_M, time_m, x_M, x_m, y_M, y_m, &timers);

	printf("devito timer: %f\n",timers.section0);
	free(devito_u_vec.data);
}

void test_openblas_stencil_kernel(int steps, int step, int iterations, int size) {
	const float dt = 0.1;
	const float h_x = 0.5;
	const float h_y = 0.5;
	const int x0_blk0_size = 1; 
	const int y0_blk0_size = 1;
	// const int time_M = 5;
	const int time_M = iterations;
	const int time_m = 0; 
	const int x_M = 4;
	const int x_m = 0;
	const int y_M = 4; 
	const int y_m = 0; 
	struct profiler timers = {.section0 = 0};

	int width = size;
	int height = size;

	//Init openblas
	struct dataobj openblas_u_vec;
	init_vector(&openblas_u_vec,width,(height+1)*2);
	fill_stencil(openblas_u_vec.data+width*sizeof(float),width,height,1);
	((float *)openblas_u_vec.data)[width*3 + 2] = 2; 
	((float *)openblas_u_vec.data)[width*3 + 3] = 2; 
	((float *)openblas_u_vec.data)[width*4 + 2] = 2; 
	((float *)openblas_u_vec.data)[width*4 + 3] = 2;

	fill_stencil(openblas_u_vec.data+(height+2)*(width*sizeof(float)),width,height,1);
	((float *)openblas_u_vec.data)[(height+1)*width + width*2 + 1] = 2; 
	((float *)openblas_u_vec.data)[(height+1)*width + width*2 + 2] = 2; 
	((float *)openblas_u_vec.data)[(height+1)*width + width*3 + 1] = 2; 
	((float *)openblas_u_vec.data)[(height+1)*width + width*3 + 2] = 2;

	openblas_linear_convection_kernel(&openblas_u_vec, dt, h_x, h_y, x0_blk0_size, y0_blk0_size, time_M, time_m, x_M, x_m, y_M, y_m, &timers);
	printf("openblas timer: %f\n",timers.section0);
	free(openblas_u_vec.data);
}

void test_matrix_kernel(matrixKernel kernel,int steps, int step, int iterations, float sparcity) {
	double results[steps][2];
	
	FILE* fp1 = fopen("results.csv", "w");
	if (fp1 == NULL)
	{
			printf("Error while opening the file.\n");
			return;
	}

	for (int i = 0; i < steps; i++) {
		results[i][0] = i*step;
		test_chain_contraction(kernel, i*step, iterations, sparcity, results[i]);
		fprintf(fp1,"%.0f, %f\n",results[i][0],results[i][1]);
	}

	fclose(fp1);
}

// Test for given size
void test_chain_contraction(matrixKernel kernel,int size, int iterations, float sparcity, double *results) {
	//Dimentions of the matrix's 
	int i = size;
	int j = size;
	int k = size;
	int l = size;
	
	struct dataobj A_vec, B_vec, C_vec, D_vec, E_vec, F_vec;
	init_vector(&A_vec,i,j);
	init_vector(&B_vec,j,k);
	init_vector(&C_vec,j,k);
	init_vector(&E_vec,k,l);

	//Non Static
	init_vector(&D_vec,i,k);
	init_vector(&F_vec,i,l);


	//Fills matrix with data (Should update it to be random)
	sparse_fill_matrix(A_vec.data,A_vec.size[0],A_vec.size[1],sparcity);
	sparse_fill_matrix(B_vec.data,B_vec.size[0],B_vec.size[1],sparcity);
	sparse_fill_matrix(C_vec.data,C_vec.size[0],C_vec.size[1],sparcity);
	sparse_fill_matrix(E_vec.data,E_vec.size[0],E_vec.size[1],sparcity);

	//Init timers
	struct profiler timers = {.section0 = 0};

	int block_size = 32;
	int thread_number = 8;

	printf("For size: %.0f\n", results[0]);
	printf("Started chain contraction\n");
	(*kernel)(&A_vec, &B_vec, &C_vec, &D_vec, &E_vec, &F_vec, block_size, i-1,0,j-1,0,k-1,0,l-1,0,thread_number,&timers,iterations);
	printf("Chain contraction %f seconds\n\n",timers.section0);

	// PrintArrays
	if (((i < 10) && (j < 10)) && (k < 10)) {
		printf("A\n");
		print_matrix(A_vec.data,A_vec.size[0],A_vec.size[1]);
		printf("B\n");
		print_matrix(B_vec.data,B_vec.size[0],B_vec.size[1]);
		printf("C\n");
		print_matrix(C_vec.data,C_vec.size[0],C_vec.size[1]);
		printf("D\n");
		print_matrix(D_vec.data,D_vec.size[0],D_vec.size[1]);
		printf("E\n");
		print_matrix(E_vec.data,E_vec.size[0],E_vec.size[1]);
		printf("F\n");
		print_matrix(F_vec.data,F_vec.size[0],F_vec.size[1]);
	}

	results[1] = timers.section0;

	//Free array storing matrices 
	destroy_vector(&A_vec);
	destroy_vector(&B_vec);
	destroy_vector(&C_vec);
	destroy_vector(&E_vec);
	destroy_vector(&D_vec);
	destroy_vector(&F_vec);
}

// Init and destroy vector helpers
void init_vector(struct dataobj *restrict vect, int n, int m) {
	float* data = calloc(sizeof(float),n*m*2);
	vect->data = data;
	vect->size = calloc(sizeof(long),3);
	vect->size[0] = 2;
	vect->size[1] = n;
	vect->size[2] = m;
}

void destroy_vector(struct dataobj *restrict vect) {
	free(vect->data); 
	free(vect->size);
}




