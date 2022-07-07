#include <stdio.h>
#include <limits.h> 
#include <stdlib.h>
#include <cblas.h>
#include "utils.h"
#include "mult.h"

int main (int argc, char* argv[] ) {


    //Dimentions of the matrix's 
    // A = i*j, B = j*k, C = i*k-
    int i = 2;
    int j = 2;
    int k = 2;
  
    // Allocate array storing matrices 
    float *A = malloc(sizeof(float)*i*j);
    float *B = malloc(sizeof(float)*j*k);
    float *C = malloc(sizeof(float)*i*k);
    
    fill_matrix(A,i,j);
    fill_matrix(B,j,k);

    struct dataobj A_vec = {.data = A};
    struct dataobj B_vec = {.data = B};
    struct dataobj C_vec = {.data = C};
    struct profiler timer = {.section0 = 0};

    kernel(&A_vec,&B_vec,&C_vec,i-1,0,j-1,0,k-1,0,&timer);

		printf("A\n");
    print_matrix(A_vec.data,i,j);
    printf("B\n");
    print_matrix(B_vec.data,j,k);
    printf("C\n");
    print_matrix(C_vec.data,i,k);

    //Free array storing matrices 
    free(A);
    free(B);
    free(C);

    return 0;
}

int kernel(struct dataobj *restrict A_vec, struct dataobj *restrict B_vec, struct dataobj *restrict C_vec, const int i_M, const int i_m, const int j_M, const int j_m, const int k_M, const int k_m, struct profiler * timers) {
    
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
  
    return 0;
}

