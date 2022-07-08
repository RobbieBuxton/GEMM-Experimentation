struct dataobj
{
  void *restrict data;
  unsigned long * size;
  unsigned long * npsize;
  unsigned long * dsize;
  int * hsize;
  int * hofs;
  int * oofs;
};

struct profiler
{
  double section0;
};


extern int gemm_kernel(struct dataobj *restrict A_vec, struct dataobj *restrict B_vec, struct dataobj *restrict C_vec, const int i_M, const int i_m, const int j_M, const int j_m, const int k_M, const int k_m, struct profiler * timers);
extern int devito_kernel(struct dataobj *restrict A_vec, struct dataobj *restrict B_vec, struct dataobj *restrict C_vec, const int i_M, const int i_m, const int j_M, const int j_m, const int k_M, const int k_m, struct profiler * timers);
extern void init_vector(struct dataobj *restrict vect, int n, int m);