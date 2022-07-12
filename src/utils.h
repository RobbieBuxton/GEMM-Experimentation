#ifndef UTILS_H
#define UTILS_H
#include <stdbool.h>

extern void print_matrix(float *matrix,int n,int m);
extern void sparse_fill_matrix(float *matrix, int n, int m, float sparcity);
extern void index_fill_matrix(float *matrix, int n, int m);
extern bool equal_matrix(float *matrix_a,float *matrix_b, int n, int m);
#endif