#define _POSIX_C_SOURCE 200809L
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define START_TIMER(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP_TIMER(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

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

	int size = 2;
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
	int block_size = 16;
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
	
	printf("The matrices are the same: %s\n",equal_matrix(gemm_F_vec.data,devito_F_vec.data,i,l) ? "true" : "false");
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

//GEMM Kernals

int gemm_mult_kernel(struct dataobj *restrict A_vec, struct dataobj *restrict B_vec, struct dataobj *restrict C_vec, const int i_M, const int i_m, const int j_M, const int j_m, const int k_M, const int k_m, struct profiler * timers) {
	START_TIMER(section0)
	cblas_sgemm(
		CblasRowMajor,					//Order - Specifies row-major (C) or column-major (Fortran) data ordering.
		CblasNoTrans,						//TransA - Specifies whether to transpose matrix A.
		CblasNoTrans,						//TransB - Specifies whether to transpose matrix B.
		i_M + 1,								//Number of rows in matrices A and C.
		k_M + 1,								//Number of columns in matrices B and C.
		j_M + 1,								//Number of columns in matrix A; number of rows in matrix B.
		1.0,										//Alpha - Scaling factor for the product of matrices A and B.
		(float *)A_vec->data, 	//Matrix A.
		i_M + 1,								//The size of the first dimension of matrix A; if you are passing a matrix A[m][n], the value should be m.
		(float *)B_vec->data,		//Matrix B 
		j_M + 1, 								//The size of the first dimension of matrix B; if you are passing a matrix B[m][n], the value should be m.
		0.0, 										//Beta - Scaling factor for matrix C.
		(float *)C_vec->data, 	//Matrix C
		i_M + 1);								//The size of the first dimension of matrix C; if you are passing a matrix C[m][n], the value should be m.
	STOP_TIMER(section0,timers)
	return 0;
}


int gemm_chain_contraction_kernel(struct dataobj *restrict A_vec, struct dataobj *restrict B_vec, struct dataobj *restrict C_vec, struct dataobj *restrict D_vec, struct dataobj *restrict E_vec, struct dataobj *restrict F_vec, const int i0_blk0_size, const int i_M, const int i_m, const int j_M, const int j_m, const int k_M, const int k_m, const int l_M, const int l_m, const int nthreads, struct profiler * timers) {
	START_TIMER(section0)

	//Bounds are very wrong need to fix this

	cblas_sgemm(
		CblasRowMajor,					//Order - Specifies row-major (C) or column-major (Fortran) data ordering.
		CblasNoTrans,						//TransA - Specifies whether to transpose matrix A.
		CblasNoTrans,						//TransB - Specifies whether to transpose matrix B.
		i_M + 1,								//Number of rows in matrices A and C.
		k_M + 1,								//Number of columns in matrices B and C.
		j_M + 1,								//Number of columns in matrix A; number of rows in matrix B.
		1.0,										//Alpha - Scaling factor for the product of matrices A and B.
		(float *)A_vec->data, 	//Matrix A.
		i_M + 1,								//The size of the first dimension of matrix A; if you are passing a matrix A[m][n], the value should be m.
		(float *)B_vec->data,		//Matrix B 
		j_M + 1, 								//The size of the first dimension of matrix B; if you are passing a matrix B[m][n], the value should be m.
		0.0, 										//Beta - Scaling factor for matrix C.
		(float *)D_vec->data, 	//Matrix C
		i_M + 1);								//The size of the first dimension of matrix C; if you are passing a matrix C[m][n], the value should be m.

	cblas_sgemm(
		CblasRowMajor,					//Order - Specifies row-major (C) or column-major (Fortran) data ordering.
		CblasNoTrans,						//TransA - Specifies whether to transpose matrix A.
		CblasNoTrans,						//TransB - Specifies whether to transpose matrix B.
		i_M + 1,								//Number of rows in matrices A and C.
		k_M + 1,								//Number of columns in matrices B and C.
		j_M + 1,								//Number of columns in matrix A; number of rows in matrix B.
		1.0,										//Alpha - Scaling factor for the product of matrices A and B.
		(float *)A_vec->data, 	//Matrix A.
		i_M + 1,								//The size of the first dimension of matrix A; if you are passing a matrix A[m][n], the value should be m.
		(float *)C_vec->data,		//Matrix B 
		j_M + 1, 								//The size of the first dimension of matrix B; if you are passing a matrix B[m][n], the value should be m.
		1.0, 										//Beta - Scaling factor for matrix C.
		(float *)D_vec->data, 	//Matrix C
		i_M + 1);								//The size of the first dimension of matrix C; if you are passing a matrix C[m][n], the value should be m.
	STOP_TIMER(section0,timers)

	
	cblas_sgemm(
		CblasRowMajor,					//Order - Specifies row-major (C) or column-major (Fortran) data ordering.
		CblasNoTrans,						//TransA - Specifies whether to transpose matrix A.
		CblasNoTrans,						//TransB - Specifies whether to transpose matrix B.
		i_M + 1,								//Number of rows in matrices A and C.
		k_M + 1,								//Number of columns in matrices B and C.
		j_M + 1,								//Number of columns in matrix A; number of rows in matrix B.
		1.0,										//Alpha - Scaling factor for the product of matrices A and B.
		(float *)D_vec->data, 	//Matrix A.
		i_M + 1,								//The size of the first dimension of matrix A; if you are passing a matrix A[m][n], the value should be m.
		(float *)E_vec->data,		//Matrix B 
		j_M + 1, 								//The size of the first dimension of matrix B; if you are passing a matrix B[m][n], the value should be m.
		0.0, 										//Beta - Scaling factor for matrix C.
		(float *)F_vec->data, 	//Matrix C
		i_M + 1);								//The size of the first dimension of matrix C; if you are passing a matrix C[m][n], the value should be m.
	STOP_TIMER(section0,timers)
	return 0;
}

//Devito Kernals

int devito_mult_kernel(struct dataobj *restrict A_vec, struct dataobj *restrict B_vec, struct dataobj *restrict D_vec, const int i_M, const int i_m, const int j_M, const int j_m, const int k_M, const int k_m, struct profiler * timers)
{
	float (*restrict A)[A_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[A_vec->size[1]]) A_vec->data;
	float (*restrict B)[B_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[B_vec->size[1]]) B_vec->data;
	float (*restrict D)[D_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[D_vec->size[1]]) D_vec->data;

	/* Begin section0 */
	START_TIMER(section0)
	for (int i = i_m; i <= i_M; i += 1)
	{
		for (int j = j_m; j <= j_M; j += 1)
		{
			for (int k = k_m; k <= k_M; k += 1)
			{
				float r0 = A[i][j]*B[j][k];
				D[i][k] += r0;
			}
		}
	}
	STOP_TIMER(section0,timers)
	/* End section0 */

	return 0;
}

int devito_chain_contraction_kernel(struct dataobj *restrict A_vec, struct dataobj *restrict B_vec, struct dataobj *restrict C_vec, struct dataobj *restrict D_vec, struct dataobj *restrict E_vec, struct dataobj *restrict F_vec, const int i0_blk0_size, const int i_M, const int i_m, const int j_M, const int j_m, const int k_M, const int k_m, const int l_M, const int l_m, const int nthreads, struct profiler * timers)
{
  float (*restrict A)[A_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[A_vec->size[1]]) A_vec->data;
  float (*restrict B)[B_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[B_vec->size[1]]) B_vec->data;
  float (*restrict C)[C_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[C_vec->size[1]]) C_vec->data;
  float (*restrict D)[D_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[D_vec->size[1]]) D_vec->data;
  float (*restrict E)[E_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[E_vec->size[1]]) E_vec->data;
  float (*restrict F)[F_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[F_vec->size[1]]) F_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  /* Begin section0 */
  START_TIMER(section0)
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(1) schedule(static,1)
    for (int i0_blk0 = i_m; i0_blk0 <= i_M; i0_blk0 += i0_blk0_size)
    {
      for (int i = i0_blk0; i <= MIN(i0_blk0 + i0_blk0_size - 1, i_M); i += 1)
      {
        for (int j = j_m; j <= j_M; j += 1)
        {
          #pragma omp simd aligned(A,B,C,D:32)
          for (int k = k_m; k <= k_M; k += 1)
          {
            float r0 = A[i][j]*B[j][k] + A[i][j]*C[j][k];
            D[i][k] += r0;
          }
        }
        for (int k = k_m; k <= k_M; k += 1)
        {
          #pragma omp simd aligned(D,E,F:32)
          for (int l = l_m; l <= l_M; l += 1)
          {
            float r1 = D[i][k]*E[k][l];
            F[i][l] += r1;
          }
        }
      }
    }
  }
  STOP_TIMER(section0,timers)
  /* End section0 */

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




