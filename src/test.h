#ifndef TEST_H
#define TEST_H
#include "types.h"
typedef int (*matrixKernel)(struct dataobj *restrict, struct dataobj *restrict, struct dataobj *restrict, struct dataobj *restrict, struct dataobj *restrict E_vec, struct dataobj *restrict, const int, const int, const int, const int, const int, const int, const int, const int, const int, const int, struct profiler*, int);
typedef int (*stencilKernel)(struct dataobj *restrict, const float, const float, const float, const int, const int, const int, const int, const int, const int, const int, const int, const int, const int, const int, const int, struct profiler *);

extern void init_vector(struct dataobj *restrict vect, int n, int m);
extern void destroy_vector(struct dataobj *restrict vect);
extern void test_chain_contraction(matrixKernel kernel,int size, int iterations, float sparcity, double *results);
extern void test_matrix_kernel(matrixKernel kernel, int steps, int step, int iterations, float sparcity);
extern void test_devito_stencil_kernel(int steps, int step, int iterations, int size);
extern void test_openblas_stencil_kernel(int steps, int step, int iterations,  int size);
#endif