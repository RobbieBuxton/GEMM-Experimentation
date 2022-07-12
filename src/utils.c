#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "utils.h"

void print_matrix(float *matrix, int n, int m) {
    for(int i =0; i<n; i++){
        printf("|");
        for(int j=0; j<m; j++){
            printf("%f ",matrix[i*n+j]);
        }
        printf("|\n");
    }
    printf("\n");
}

void random_fill_matrix(float *matrix, int n, int m) {
    for(int i =0; i<n; i++){
        for(int j=0; j<m; j++){
            matrix[n*i + j] = (float)rand()/(float)(RAND_MAX/100);
        }
    }
}

void sparse_fill_matrix(float *matrix, int n, int m,float sparcity) {
    for(int i =0; i<n; i++){
        for(int j=0; j<m; j++){
						printf("%f\n",(float)rand()/(float)(RAND_MAX));
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