#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <limits.h> 
#include <stdlib.h>
#include <cblas.h>
#include "utils.h"
#include "test.h"
#include "kernels/nvblas.h"
#include "kernels/openblas.h"
#include "kernels/devito.h"

int main (int argc, char* argv[]) {

	test_kernel(21,250,5);
	// double results[4];
	// test_chain_contraction(5, 200, results);
}


void test_kernel(int steps, int step, int iterations) {
	double results[steps][4];
	
	FILE* fp1 = fopen("results.csv", "w");
	if (fp1 == NULL)
	{
			printf("Error while opening the file.\n");
			return;
	}

	for (int i = 0; i < steps; i++) {
		results[i][0] = i*step;
		test_chain_contraction(i*step, iterations, results[i]);
		fprintf(fp1,"%.0f, %f, %f, %f\n",results[i][0],results[i][1],results[i][2],results[i][3]);
	}

	fclose(fp1);
}

// Test for given size
void test_chain_contraction(int size, int iterations, double *results) {
	//Dimentions of the matrix's 
	//They are the same for below 320

	int i = size;
	int j = size;
	int k = size;
	int l = size;
	float sparcity = 0.25;
	
	struct dataobj A_vec, B_vec, C_vec, nvblas_D_vec, openblas_D_vec, devito_D_vec, E_vec, nvblas_F_vec, openblas_F_vec, devito_F_vec;
	init_vector(&A_vec,i,j);
	init_vector(&B_vec,j,k);
	init_vector(&C_vec,j,k);
	init_vector(&E_vec,k,l);
	init_vector(&nvblas_D_vec,i,k);
	init_vector(&openblas_D_vec,i,k);
	init_vector(&devito_D_vec,i,k);
	init_vector(&nvblas_F_vec,i,l);
	init_vector(&openblas_F_vec,i,l);
	init_vector(&devito_F_vec,i,l);

	//Fills matrix with data (Should update it to be random)
	sparse_fill_matrix(A_vec.data,A_vec.size[0],A_vec.size[1],sparcity);
	sparse_fill_matrix(B_vec.data,B_vec.size[0],B_vec.size[1],sparcity);
	sparse_fill_matrix(C_vec.data,C_vec.size[0],C_vec.size[1],sparcity);
	sparse_fill_matrix(E_vec.data,E_vec.size[0],E_vec.size[1],sparcity);

	//Init timers
	struct profiler nvblas_timers = {.section0 = 0};
	struct profiler openblas_timers = {.section0 = 0};
	struct profiler devito_timers = {.section0 = 0};

	int block_size = 32;
	int thread_number = 8;

	printf("For size: %.0f\n", results[0]);
	printf("Started nvblas\n");
	nvblas_chain_contraction_kernel(&A_vec, &B_vec, &C_vec, &nvblas_D_vec, &E_vec, &nvblas_F_vec, block_size, i-1,0,j-1,0,k-1,0,l-1,0,thread_number,&nvblas_timers,iterations);
	printf("nvblas Multiplication took %f seconds\n",nvblas_timers.section0);
	printf("Started openblas\n");
	openblas_chain_contraction_kernel(&A_vec, &B_vec, &C_vec, &openblas_D_vec, &E_vec, &openblas_F_vec, block_size, i-1,0,j-1,0,k-1,0,l-1,0,thread_number,&openblas_timers,iterations);
	printf("openblas Multiplication took %f seconds\n",openblas_timers.section0);
	// printf("Started Devito\n");
	// devito_chain_contraction_kernel(&A_vec, &B_vec, &C_vec, &devito_D_vec, &E_vec, &devito_F_vec, block_size, i-1,0,j-1,0,k-1,0,l-1,0,thread_number,&devito_timers,iterations);
	// printf("Devito Multiplication took %f seconds\n\n",devito_timers.section0);

	
	// PrintArrays
	if (((i < 10) && (j < 10)) && (k < 10)) {
		printf("A\n");
		print_matrix(A_vec.data,A_vec.size[0],A_vec.size[1]);
		printf("B\n");
		print_matrix(B_vec.data,B_vec.size[0],B_vec.size[1]);
		printf("C\n");
		print_matrix(C_vec.data,C_vec.size[0],C_vec.size[1]);
		printf("nvblas D\n");
		print_matrix(nvblas_D_vec.data,nvblas_D_vec.size[0],nvblas_D_vec.size[1]);
		printf("openblas D\n");
		print_matrix(openblas_D_vec.data,openblas_D_vec.size[0],openblas_D_vec.size[1]);
		printf("devito D\n");
		print_matrix(devito_D_vec.data,devito_D_vec.size[0],devito_D_vec.size[1]);
		printf("E\n");
		print_matrix(E_vec.data,E_vec.size[0],E_vec.size[1]);
		printf("nvblas F\n");
		print_matrix(nvblas_F_vec.data,nvblas_F_vec.size[0],nvblas_F_vec.size[1]);
		printf("openblas F\n");
		print_matrix(openblas_F_vec.data,openblas_F_vec.size[0],openblas_F_vec.size[1]);
		printf("devito F\n");
		print_matrix(devito_F_vec.data,devito_F_vec.size[0],devito_F_vec.size[1]);
	}
	// printf("The matrices are the same: %s\n",equal_matrix(openblas_F_vec.data,devito_F_vec.data,i,l) ? "true" : "false");

	results[1] = nvblas_timers.section0;
	results[2] = openblas_timers.section0;
	results[3] = devito_timers.section0;
	

	//Free array storing matrices 
	destroy_vector(&A_vec);
	destroy_vector(&B_vec);
	destroy_vector(&C_vec);
	destroy_vector(&E_vec);
	destroy_vector(&openblas_D_vec);
	destroy_vector(&devito_D_vec);
	destroy_vector(&openblas_F_vec);
	destroy_vector(&devito_F_vec);
}

// Init and destroy vector helpers
void init_vector(struct dataobj *restrict vect, int n, int m) {
	float* data = malloc(sizeof(float)*n*m);
	vect->data = data;
	vect->size = malloc(sizeof(long)*2);
	vect->size[0] = n;
	vect->size[1] = m;
}

void destroy_vector(struct dataobj *restrict vect) {
	free(vect->data); 
	free(vect->size);
}




