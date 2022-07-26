#ifndef MATRIX_HELPERS_H
#define MATRIX_HELPERS_H
#include "../../types.h"
extern void test();
extern void diagonalize_matrix(float* A, int n, int m, float* PT, float* D, float* PINV);
extern void sgeev_( char* jobvl, char* jobvr, int* n, float* a,int* lda, float* wr, float* wi, float* vl, int* ldvl, float* vr, int* ldvr, float* work, int* lwork, int* info);
extern void sgetri_(int* n, float* a,int* lda, int* ipiv, float* work, int* lwork, int* info);
extern float sdot_(int*,float*,int*,float*,int*);
extern float* generate_binomial_table(int k); 
#endif