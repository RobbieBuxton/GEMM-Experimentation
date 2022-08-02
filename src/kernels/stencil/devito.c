#define _POSIX_C_SOURCE 200809L
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define START_TIMER(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP_TIMER(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "xmmintrin.h"
#include "pmmintrin.h"
#include <stdio.h>
#include "../../types.h"
#include "../../utils.h"

int devito_linear_convection_kernel(float** stencil, struct dataobj *restrict u_vec, const float dt, const float h_x, const float h_y, const int x0_blk0_size, const int y0_blk0_size, const int time_M, const int time_m, const int x_M, const int x_m, const int y_M, const int y_m, struct profiler * timers,float * result)
{
  
	// printf("\ndt: %f h_x %f h_y %f i0x0_blk0_size %d i0x_ltkn %d i0x_rtkn %d i0y0_blk0_size %d i0y_ltkn %d i0y_rtkn %d time_M %d time_m %d x_M %d x_m %d y_M %d y_m %d\n",dt,h_x,h_y,i0x0_blk0_size,i0x_ltkn,i0x_rtkn,i0y0_blk0_size,i0y_ltkn,i0y_rtkn,time_M,time_m,x_M,x_m,y_M,y_m);

	float (*restrict u)[u_vec->size[1]][u_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]]) u_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

	int nthreads = 8;
	START_TIMER(section0)
	
  for (int time = time_m, t0 = (time)%(2), t1 = (time + 1)%(2); time <= time_M; time += 1, t0 = (time)%(2), t1 = (time + 1)%(2))
  {
		// printf("time: %d\n",time);
		// print_matrix((float*)u[t0],u_vec->size[1],u_vec->size[2]);

    /* Begin section0 */
		#pragma omp parallel num_threads(nthreads)
		{
			#pragma omp for collapse(1) schedule(static,1)
			for (int x0_blk0 = x_m; x0_blk0 <= x_M; x0_blk0 += x0_blk0_size)
			{
				for (int y0_blk0 = y_m; y0_blk0 <= y_M; y0_blk0 += y0_blk0_size)
				{
					for (int x = x0_blk0; x <= MIN(x0_blk0 + x0_blk0_size - 1, x_M); x += 1)
					{
						#pragma omp simd aligned(u:32)
						for (int y = y0_blk0; y <= MIN(y0_blk0 + y0_blk0_size - 1, y_M); y += 1)
						{
							// printf("u[t1][%d][%d] = %f(%+f*%+f %+f*%+f %+f*%+f %+f*%+f %+f*%+f)\n",
							// 			x + 1,
							// 			y + 1,
							// 			dt,
							// 			-r0,
							// 			(-u[t0][x][y + 1]),
							// 			-r0,
							// 			u[t0][x + 1][y + 1],
							// 			- r1,
							// 			(-u[t0][x + 1][y]),
							// 			-r1,
							// 			u[t0][x + 1][y + 1],
							// 			r2,
							// 			u[t0][x + 1][y + 1]);

							u[t1][x + 1][y + 1] = 
								stencil[0][0]*(u[t0][x + 1][y]) + stencil[0][2]*(u[t0][x + 1][y + 2]) +
								stencil[1][0]*(u[t0][x][y + 1]) + stencil[1][2]*(u[t0][x + 2][y + 1]) +  
								(stencil[0][1]+ stencil[1][1])*u[t0][x + 1][y + 1];
						}
					}
				}
			}
		}
 
    /* End section0 */
  }
	STOP_TIMER(section0,timers)
	// if (u_vec->size[1] < 30) {
	// 	printf("time: %d\n",(time_M + 1));
	// 	print_matrix((float*)u[(time_M + 1)%2],u_vec->size[1],u_vec->size[2]);
	// }

	for (int i = 1; i < u_vec->size[1]-1; i++) {
		for (int j = 1; j < u_vec->size[1]-1; j++) {
			result[(i-1) + (u_vec->size[1]-2) * (j-1)] = u[(time_M + 1)%2][i][j];
		}
	}
  return 0;
}