#ifndef OPENBLAS_S_H
#define OPENBLAS_S_H
#include "../../types.h"

extern int openblas_linear_convection_kernel(struct dataobj *restrict u_vec, const float dt, const float h_x, const float h_y, const int i0x0_blk0_size, const int i0x_ltkn, const int i0x_rtkn, const int i0y0_blk0_size, const int i0y_ltkn, const int i0y_rtkn, const int time_M, const int time_m, const int x_M, const int x_m, const int y_M, const int y_m, struct profiler * timers);

#endif