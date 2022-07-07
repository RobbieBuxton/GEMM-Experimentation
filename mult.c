#include <stdio.h>
#include <limits.h> 
#include <stdlib.h>
#include <cblas.h>

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



void printMatrix(double *matrix,int n,int m) {
    for(int i =0; i<n; i++){
        printf("|");
        for(int j=0; j<m; j++){
            printf("%.2f ",matrix[i*n+j]);
        }
        printf("|\n");
    }
    printf("\n");
}

int Kernel(struct dataobj *restrict A_vec, struct dataobj *restrict B_vec, struct dataobj *restrict C_vec, const int i_M, const int i_m, const int j_M, const int j_m, const int k_M, struct profiler * timers) {
    return 0; 
}


int main (int argc, char* argv[] ) {

    struct dataobj A_vec = {};
    struct dataobj B_vec = {};
    struct dataobj C_vec = {};


    //## OLD MULT 
    int n = atoi(argv[1]);

    // Create arrays that represent the matrices A,B,C
    double A[n*n];
    double B[n*n]; 
    double C[n*n];

    // Fill A and B
    for(int i =0; i <n; i++){
        for(int j=0; j<n; j++){
            A[i*n+j] = i*n+j;
            B[i*n+j] = i*n+j;
        }
    }

    // Calculate A*B=C
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n, B, n, 0.0, C, n);
    
    //Print Arrays 
    //A
    printf("A\n");
    printMatrix(A,n,n);
    //B
    printf("B\n");
    printMatrix(B,n,n);
    //C
    printf("C\n");
    printMatrix(C,n,n);
    return 0;
}

