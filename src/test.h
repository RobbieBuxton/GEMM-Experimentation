#ifndef TEST_H
#define TEST_H
#include "types.h"

extern int devito_mult_kernel(struct dataobj *restrict A_vec, struct dataobj *restrict B_vec, struct dataobj *restrict C_vec, const int i_M, const int i_m, const int j_M, const int j_m, const int k_M, const int k_m, struct profiler * timers);
extern int devito_chain_contraction_kernel(struct dataobj *restrict A_vec, struct dataobj *restrict B_vec, struct dataobj *restrict C_vec, struct dataobj *restrict D_vec, struct dataobj *restrict E_vec, struct dataobj *restrict F_vec, const int i0_blk0_size, const int i_M, const int i_m, const int j_M, const int j_m, const int k_M, const int k_m, const int l_M, const int l_m, const int nthreads, struct profiler * timers);
extern void init_vector(struct dataobj *restrict vect, int n, int m);
extern void destroy_vector(struct dataobj *restrict vect);

#endif