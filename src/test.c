#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <limits.h> 
#include <stdlib.h>
#include <cblas.h>
#include "utils.h"
#include "test.h"
#include "sys/time.h"
#include "math.h"
#include "xmmintrin.h"
#include "pmmintrin.h"
#include "omp.h"
#include "kernels/gemm.h"


	// i, j, k, l = dimensions('i j k l')
	// A = Function(name='A', shape=mat_shape, dimensions=(i, j))
	// B = Function(name='B', shape=mat_shape, dimensions=(j, k))
	// C = Function(name='C', shape=mat_shape, dimensions=(j, k))
	// D = Function(name='D', shape=mat_shape, dimensions=(i, k))
	// E = Function(name='E', shape=mat_shape, dimensions=(k, l))
	// F = Function(name='F', shape=mat_shape, dimensions=(i, l))
	// chain_contractions(A, B, C, D, E, F, optimize)

int main (int argc, char* argv[] ) {

	//Dimentions of the matrix's 
	//They are the same for below 320

	int size = 1000;
	int i = size;
	int j = size;
	int k = size;
	int l = size;
	
	struct dataobj A_vec, B_vec, C_vec, gemm_D_vec, devito_D_vec, E_vec, gemm_F_vec, devito_F_vec;
	init_vector(&A_vec,i,j);
	init_vector(&B_vec,j,k);
	init_vector(&C_vec,j,k);
	init_vector(&E_vec,k,l);
	init_vector(&gemm_D_vec,i,k);
	init_vector(&devito_D_vec,i,k);
	init_vector(&gemm_F_vec,i,l);
	init_vector(&devito_F_vec,i,l);

	//Fills matrix with data (Should update it to be random)
	fill_matrix(A_vec.data,A_vec.size[0],A_vec.size[1]);
	fill_matrix(B_vec.data,B_vec.size[0],B_vec.size[1]);
	fill_matrix(C_vec.data,C_vec.size[0],C_vec.size[1]);
	fill_matrix(E_vec.data,E_vec.size[0],E_vec.size[1]);


	//Init timers
	struct profiler gemm_timers = {.section0 = 0};
	struct profiler devito_timers = {.section0 = 0};

	// gemm_mult_kernel(&A_vec,&B_vec,&gemm_C_vec,i-1,0,j-1,0,k-1,0,&gemm_timers);
	// devito_mult_kernel(&A_vec,&B_vec,&devito_C_vec,i-1,0,j-1,0,k-1,0,&devito_timers);
	int block_size = 32;
	int thread_number = 8;
	devito_chain_contraction_kernel(&A_vec, &B_vec, &C_vec, &devito_D_vec, &E_vec, &devito_F_vec, block_size, i-1,0,j-1,0,k-1,0,l-1,0,thread_number,&devito_timers);
	gemm_chain_contraction_kernel(&A_vec, &B_vec, &C_vec, &gemm_D_vec, &E_vec, &gemm_F_vec, block_size, i-1,0,j-1,0,k-1,0,l-1,0,thread_number,&gemm_timers);

	//PrintArrays
	if (((i < 10) && (j < 10)) && (k < 10)) {
		printf("A\n");
		print_matrix(A_vec.data,A_vec.size[0],A_vec.size[1]);
		printf("B\n");
		print_matrix(B_vec.data,B_vec.size[0],B_vec.size[1]);
		printf("C\n");
		print_matrix(C_vec.data,C_vec.size[0],C_vec.size[1]);
		printf("GEMM D\n");
		print_matrix(gemm_D_vec.data,gemm_D_vec.size[0],gemm_D_vec.size[1]);
		printf("devito D\n");
		print_matrix(devito_D_vec.data,devito_D_vec.size[0],devito_D_vec.size[1]);
		printf("E\n");
		print_matrix(E_vec.data,E_vec.size[0],E_vec.size[1]);
		printf("GEMM F\n");
		print_matrix(gemm_F_vec.data,gemm_F_vec.size[0],gemm_F_vec.size[1]);
		printf("devito F\n");
		print_matrix(devito_F_vec.data,devito_F_vec.size[0],devito_F_vec.size[1]);
	}
	
	// printf("The matrices are the same: %s\n",equal_matrix(gemm_F_vec.data,devito_F_vec.data,i,l) ? "true" : "false");
	printf("GEMM Multiplication took %f seconds\n",gemm_timers.section0);
	printf("Devito Multiplication took %f seconds\n",devito_timers.section0);
	
	//Free array storing matrices 
	destroy_vector(&A_vec);
	destroy_vector(&B_vec);
	destroy_vector(&C_vec);
	destroy_vector(&E_vec);
	destroy_vector(&gemm_D_vec);
	destroy_vector(&devito_D_vec);
	destroy_vector(&gemm_F_vec);
	destroy_vector(&devito_F_vec);

	return 0;
}


// Init and destroy vectors

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




