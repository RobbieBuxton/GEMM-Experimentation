#define START_TIMER(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP_TIMER(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include <stdio.h>
#include <limits.h> 
#include <stdlib.h>
#include <cblas.h>
#include "utils.h"
#include "mult.h"
#include "sys/time.h"

int main (int argc, char* argv[] ) {

    //Dimentions of the matrix's 
    // A = i*j, B = j*k, C = i*k-
    int i = 2;
    int j = 2;
    int k = 2;
    
		struct dataobj A_vec, B_vec, gemm_C_vec, devito_C_vec;
		init_vector(&A_vec,i,j);
		init_vector(&B_vec,j,k);

		fill_matrix(A_vec.data,i,j);
    fill_matrix(B_vec.data,j,k);

		init_vector(&gemm_C_vec,i,k);
		init_vector(&devito_C_vec,i,k);

		//Init timers
		struct profiler gemm_timers = {.section0 = 0};
		struct profiler devito_timers = {.section0 = 0};



    gemm_kernel(&A_vec,&B_vec,&gemm_C_vec,i-1,0,j-1,0,k-1,0,&gemm_timers);
		devito_kernel(&A_vec,&B_vec,&devito_C_vec,i-1,0,j-1,0,k-1,0,&devito_timers);

		//PrintArrays
		if (((i < 10) && (j < 10)) && (k < 10)) {
			printf("A\n");
    	print_matrix(A_vec.data,i,j);
    	printf("B\n");
    	print_matrix(B_vec.data,j,k);
    	printf("GEMM C\n");
    	print_matrix(gemm_C_vec.data,i,k);
			printf("devito C\n");
    	print_matrix(devito_C_vec.data,i,k);
		}
		
		printf("The matrices are the same: %s\n",equal_matrix(A_vec.data,B_vec.data,i,k) ? "true" : "false");
		printf("GEMM Multiplication took %f seconds\n",gemm_timers.section0);
		printf("Devito Multiplication took %f seconds\n",devito_timers.section0);
		
    //Free array storing matrices 
		destroy_vector(&A_vec);
		destroy_vector(&B_vec);
		destroy_vector(&gemm_C_vec);
		destroy_vector(&devito_C_vec);


    return 0;
}

int gemm_kernel(struct dataobj *restrict A_vec, struct dataobj *restrict B_vec, struct dataobj *restrict C_vec, const int i_M, const int i_m, const int j_M, const int j_m, const int k_M, const int k_m, struct profiler * timers) {
    START_TIMER(section0)
		cblas_sgemm(CblasRowMajor,					//Order - Specifies row-major (C) or column-major (Fortran) data ordering.
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

int devito_kernel(struct dataobj *restrict A_vec, struct dataobj *restrict B_vec, struct dataobj *restrict D_vec, const int i_M, const int i_m, const int j_M, const int j_m, const int k_M, const int k_m, struct profiler * timers)
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


