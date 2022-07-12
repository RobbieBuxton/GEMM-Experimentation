#ifndef NVBLAS_H
#define NVBLAS_H
#include "../types.h"
extern void sgemm(const char* transa,
           const char* transb,
           const int* m,
           const int* n,
           const int* k,
           const float* alpha,
           const float* a,
           const int* lda,
           const float* b,
           const int* ldb,
           const float* beta,
           float* c,
           const int* ldc);

extern int nvblas_chain_contraction_kernel(struct dataobj *restrict A_vec, struct dataobj *restrict B_vec, struct dataobj *restrict C_vec, struct dataobj *restrict D_vec, struct dataobj *restrict E_vec, struct dataobj *restrict F_vec, const int i0_blk0_size, const int i_M, const int i_m, const int j_M, const int j_m, const int k_M, const int k_m, const int l_M, const int l_m, const int nthreads, struct profiler * timers, int iterations);

#endif