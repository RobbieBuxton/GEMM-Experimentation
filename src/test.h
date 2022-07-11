#ifndef TEST_H
#define TEST_H
#include "types.h"

extern void test_chain_contraction(int size, int interations, double *results);
extern void init_vector(struct dataobj *restrict vect, int n, int m);
extern void destroy_vector(struct dataobj *restrict vect);

#endif