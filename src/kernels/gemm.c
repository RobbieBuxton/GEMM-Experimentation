
#define START_TIMER(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP_TIMER(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include "gemm.h"
#include <cblas.h>
#include "sys/time.h"

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