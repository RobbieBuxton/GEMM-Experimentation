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

void fill_matrix(float *matrix, int n, int m) {
		int maxRandVal = 100;
    for(int i =0; i<n; i++){
        for(int j=0; j<m; j++){
            matrix[n*i + j] = (float)rand()/(float)(RAND_MAX/100);
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