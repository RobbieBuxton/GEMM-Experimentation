#ifndef DEVITO_S_H
#define DEVITO_S_H
#include "../../types.h"

extern int devito_linear_convection_kernel(float** stencil, struct dataobj *restrict u_vec, const float dt, const float h_x, const float h_y, const int x0_blk0_size, const int y0_blk0_size, const int time_M, const int time_m, const int x_M, const int x_m, const int y_M, const int y_m, struct profiler * timers, float* result);

#endif