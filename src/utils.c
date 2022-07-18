#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "utils.h"

void print_matrix(float *matrix, int m, int n) {
    for(int i =0; i<n; i++){
        printf("|");
        for(int j=0; j<m; j++){
						if (matrix[i*m+j] != 0) {
							printf("\x1B[32m%+.2f \x1B[0m",matrix[i*m+j]);
						} else {
							printf("%+.2f ",matrix[i*m+j]);
						}
        }
        printf("|\n");
    }
    printf("\n");
}

void fill_stencil(float *matrix, int n, int m, int halo_padding) {
		for(int i =0; i<n; i++){
			if ((i - halo_padding) >= 0 && (i + halo_padding) < n ) {
				for(int j=0; j<m; j++){
					if ((j - halo_padding) >= 0 && (j + halo_padding) < m ) {
						matrix[n*i + j] = 1;
					}
				}
			}
    }
} 

void sparse_fill_matrix(float *matrix, int n, int m,float sparcity) {
    for(int i =0; i<n; i++){
        for(int j=0; j<m; j++){
						if (((float)rand()/(float)(RAND_MAX)) <= sparcity) {
							matrix[n*i + j] = (float)rand()/(float)(RAND_MAX/100);
						} else {
							matrix[n*i + j] = 0;
						}
            
        }
    }
}

void index_fill_matrix(float *matrix, int n, int m) {
    for(int i =0; i<n; i++){
        for(int j=0; j<m; j++){
            matrix[n*i + j] = n*i + j;
        }
    }
}

bool equal_matrix(float *matrix_a,float *matrix_b, int n, int m) {
    for(int i =0; i<n; i++){
        for(int j=0; j<m; j++){
            if (matrix_a[n*i + j] != matrix_b[n*i + j]) {
								printf("Error matrices are different!!!!!\n");
								printf("At i: %d, j: %d A = %f, B = %f\n",i,j,matrix_a[n*i + j],matrix_b[n*i + j]);
                return false; 
            }
        }
    }
    return true;
}