#ifndef TEST_H
#define TEST_H
#include "types.h"
typedef void (*testKernel)(struct dataobj *restrict, struct dataobj *restrict, struct dataobj *restrict, struct dataobj *restrict, struct dataobj *restrict E_vec, struct dataobj *restrict, const int, const int, const int, const int, const int, const int, const int, const int, const int, const int, struct profiler*, int);

extern void init_vector(struct dataobj *restrict vect, int n, int m);
extern void destroy_vector(struct dataobj *restrict vect);
extern void test_chain_contraction(testKernel kernel,int size, int iterations, double *results);
extern void test_kernel(testKernel kernel, int steps, int step, int iterations);

#endif