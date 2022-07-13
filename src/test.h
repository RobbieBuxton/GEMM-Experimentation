#ifndef TEST_H
#define TEST_H
#include "types.h"
typedef int (*testedKernel)(struct dataobj *restrict, struct dataobj *restrict, struct dataobj *restrict, struct dataobj *restrict, struct dataobj *restrict E_vec, struct dataobj *restrict, const int, const int, const int, const int, const int, const int, const int, const int, const int, const int, struct profiler*, int);

extern void init_vector(struct dataobj *restrict vect, int n, int m);
extern void destroy_vector(struct dataobj *restrict vect);
extern void test_chain_contraction(testedKernel kernel,int size, int iterations, float sparcity, double *results);
extern void test_kernel(testedKernel kernel, int steps, int step, int iterations, float sparcity);

#endif