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

int devito_linear_convection_kernel(struct dataobj *restrict u_vec, const float dt, const float h_x, const float h_y, const int i0x0_blk0_size, const int i0x_ltkn, const int i0x_rtkn, const int i0y0_blk0_size, const int i0y_ltkn, const int i0y_rtkn, const int time_M, const int time_m, const int x_M, const int x_m, const int y_M, const int y_m, struct profiler * timers)
{
  
	printf("\ndt: %f h_x %f h_y %f i0x0_blk0_size %d i0x_ltkn %d i0x_rtkn %d i0y0_blk0_size %d i0y_ltkn %d i0y_rtkn %d time_M %d time_m %d x_M %d x_m %d y_M %d y_m %d\n",dt,h_x,h_y,i0x0_blk0_size,i0x_ltkn,i0x_rtkn,i0y0_blk0_size,i0y_ltkn,i0y_rtkn,time_M,time_m,x_M,x_m,y_M,y_m);

	float (*restrict u)[u_vec->size[1]][u_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]]) u_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  float r0 = 1.0F/h_x;
  float r1 = 1.0F/h_y;
  float r2 = 1.0F/dt;
	
	printf("\nr0: %f r1: %f r2: %f \n",r0,r1,r2);
	for (int time = time_m, t0 = (time)%(2), t1 = (time + 1)%(2); time <= time_M; time += 1, t0 = (time)%(2), t1 = (time + 1)%(2))
  {

		printf("time: %d\n",time);
		for (int x = 0; x < u_vec->size[1]; x++) {
			for (int y = 0; y < u_vec->size[2]; y++) {
				printf("%.2f ",u[t0][x][y]);
			}
			printf("\n");
		}
		printf("\n");
    /* Begin section0 */
    START_TIMER(section0)
    for (int i0x0_blk0 = i0x_ltkn + x_m; i0x0_blk0 <= -i0x_rtkn + x_M; i0x0_blk0 += i0x0_blk0_size)
    {
      for (int i0y0_blk0 = i0y_ltkn + y_m; i0y0_blk0 <= -i0y_rtkn + y_M; i0y0_blk0 += i0y0_blk0_size)
      {
        for (int i0x = i0x0_blk0; i0x <= MIN(i0x0_blk0 + i0x0_blk0_size - 1, -i0x_rtkn + x_M); i0x += 1)
        {
          #pragma omp simd aligned(u:32)
          for (int i0y = i0y0_blk0; i0y <= MIN(i0y0_blk0 + i0y0_blk0_size - 1, -i0y_rtkn + y_M); i0y += 1)
          {
						// printf("u[t1][%d][%d] = %f(%+f %+f %+f %+f %+f)\n",i0x + 1,i0y + 1,dt,-r0*(-u[t0][i0x][i0y + 1]),-r0*u[t0][i0x + 1][i0y + 1],- r1*(-u[t0][i0x + 1][i0y]),-r1*u[t0][i0x + 1][i0y + 1],r2*u[t0][i0x + 1][i0y + 1]);
            u[t1][i0x + 1][i0y + 1] = dt*(-r0*(-u[t0][i0x][i0y + 1]) - r0*u[t0][i0x + 1][i0y + 1] - r1*(-u[t0][i0x + 1][i0y]) - r1*u[t0][i0x + 1][i0y + 1] + r2*u[t0][i0x + 1][i0y + 1]);
          }
        }
      }
    }
    STOP_TIMER(section0,timers)
    /* End section0 */
  }

	printf("time: %d\n",(time_M + 1));
	for (int x = 0; x < u_vec->size[1]; x++) {
		for (int y = 0; y < u_vec->size[2]; y++) {
				printf("%.2f ",u[(time_M + 1)%(2)][x][y]);
			}
			printf("\n");
		}
	printf("\n");

  return 0;
}