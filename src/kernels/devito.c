#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define START_TIMER(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP_TIMER(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include <stdio.h>
#include <limits.h> 
#include <stdlib.h>
#include <cblas.h>
#include "sys/time.h"
#include "math.h"
#include "xmmintrin.h"
#include "pmmintrin.h"
#include "omp.h"
#include "devito.h"


//Devito Kernals

int devito_chain_contraction_kernel(struct dataobj *restrict A_vec, struct dataobj *restrict B_vec, struct dataobj *restrict C_vec, struct dataobj *restrict D_vec, struct dataobj *restrict E_vec, struct dataobj *restrict F_vec, const int i0_blk0_size, const int i_M, const int i_m, const int j_M, const int j_m, const int k_M, const int k_m, const int l_M, const int l_m, const int nthreads, struct profiler * timers)
{
  float (*restrict A)[A_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[A_vec->size[1]]) A_vec->data;
  float (*restrict B)[B_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[B_vec->size[1]]) B_vec->data;
  float (*restrict C)[C_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[C_vec->size[1]]) C_vec->data;
  float (*restrict D)[D_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[D_vec->size[1]]) D_vec->data;
  float (*restrict E)[E_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[E_vec->size[1]]) E_vec->data;
  float (*restrict F)[F_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[F_vec->size[1]]) F_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  /* Begin section0 */
  START_TIMER(section0)
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(1) schedule(static,1)
    for (int i0_blk0 = i_m; i0_blk0 <= i_M; i0_blk0 += i0_blk0_size)
    {
      for (int i = i0_blk0; i <= MIN(i0_blk0 + i0_blk0_size - 1, i_M); i += 1)
      {
        for (int j = j_m; j <= j_M; j += 1)
        {
          #pragma omp simd aligned(A,B,C,D:32)
          for (int k = k_m; k <= k_M; k += 1)
          {
            float r0 = A[i][j]*B[j][k] + A[i][j]*C[j][k];
            D[i][k] += r0;
          }
        }
        for (int k = k_m; k <= k_M; k += 1)
        {
          #pragma omp simd aligned(D,E,F:32)
          for (int l = l_m; l <= l_M; l += 1)
          {
            float r1 = D[i][k]*E[k][l];
            F[i][l] += r1;
          }
        }
      }
    }
  }
  STOP_TIMER(section0,timers)
  /* End section0 */

  return 0;
}
