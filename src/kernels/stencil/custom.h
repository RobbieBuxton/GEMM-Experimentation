#ifndef CUSTOM_H
#define CUSTOM_H
#include "../../types.h"

extern int custom_linear_convection_kernel(float **stencil, struct dataobj *restrict u_vec, const float dt, const float h_x, const float h_y, const int x0_blk0_size, const int y0_blk0_size, const int time_M, const int time_m, const int x_M, const int x_m, const int y_M, const int y_m, struct profiler *timers, float *result);
extern void diagonalize_matrix(float *A, int n, int m, float *PT, float *D, float *PINV);
extern float sdot_(int *, float *, int *, float *, int *);
extern float vector_mag(float *vector, int size, int spacing);
extern int compare_stencil(int steps, int size_scale, int size_start, int iterations_scale, int iterations_start, float gamma, float a_n1, float a_1, float b_n1, float b_1);
#endif