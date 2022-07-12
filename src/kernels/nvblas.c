#define START_TIMER(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP_TIMER(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include "nvblas.h"
#include <cblas.h>
#include "sys/time.h"

int nvblas_chain_contraction_kernel(struct dataobj *restrict A_vec, struct dataobj *restrict B_vec, struct dataobj *restrict C_vec, struct dataobj *restrict D_vec, struct dataobj *restrict E_vec, struct dataobj *restrict F_vec, const int i0_blk0_size, const int i_M, const int i_m, const int j_M, const int j_m, const int k_M, const int k_m, const int l_M, const int l_m, const int nthreads, struct profiler * timers, int iterations) {
//Bounds are very wrong need to fix this but fine while everything is square
	START_TIMER(section0)

	for (int iteration = 0; iteration < iterations; iteration++) {
		char transa = 'n';
		char transb = 'n';
		int m = i_M + 1;
		int n = k_M + 1;
		int k = j_M + 1;
		float alpha = 1.0;
		int lda = i_M + 1;
		int ldb = j_M + 1;
		float beta = 1.0;
		int ldc = i_M + 1;

		sgemm(			
			&transa,						
			&transb,						
			&m,								
			&n,								
			&k,							
			&alpha,										
			(float *)A_vec->data, 	
			&lda,								
			(float *)B_vec->data,		
			&ldb, 								
			&beta, 										
			(float *)D_vec->data, 	
			&ldc);

		sgemm(			
			&transa,						
			&transb,						
			&m,								
			&n,								
			&k,							
			&alpha,										
			(float *)A_vec->data, 	
			&lda,								
			(float *)C_vec->data,		
			&ldb, 								
			&beta, 										
			(float *)D_vec->data, 	
			&ldc);

		sgemm(			
			&transa,						
			&transb,						
			&m,								
			&n,								
			&k,							
			&alpha,										
			(float *)D_vec->data, 	
			&lda,								
			(float *)E_vec->data,		
			&ldb, 								
			&beta, 										
			(float *)F_vec->data, 	
			&ldc);


		// sgemm(			
		// 	'n',						
		// 	'n',						
		// 	i_M + 1,								
		// 	k_M + 1,								
		// 	j_M + 1,							
		// 	1.0,										
		// 	(float *)A_vec->data, 	
		// 	i_M + 1,								
		// 	(float *)B_vec->data,		
		// 	j_M + 1, 								
		// 	1.0, 										
		// 	(float *)D_vec->data, 	
		// 	i_M + 1);

		// sgemm(		
		// 	'n',						
		// 	'n',						
		// 	i_M + 1,								
		// 	k_M + 1,								
		// 	j_M + 1,						
		// 	1.0,										
		// 	(float *)A_vec->data, 	
		// 	i_M + 1,								
		// 	(float *)C_vec->data,		 
		// 	j_M + 1, 								
		// 	1.0, 										
		// 	(float *)D_vec->data, 	
		// 	i_M + 1);								

		// sgemm(
		// 	'n',					
		// 	'n',						
		// 	i_M + 1,							
		// 	k_M + 1,					
		// 	j_M + 1,								
		// 	1.0,										
		// 	(float *)D_vec->data, 	
		// 	i_M + 1,								
		// 	(float *)E_vec->data,		
		// 	j_M + 1, 								
		// 	1.0, 										
		// 	(float *)F_vec->data, 	
		// 	i_M + 1);					
	}


	
	STOP_TIMER(section0,timers)
	return 0;
}