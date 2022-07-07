#include <stdlib.h>
#include <stdio.h>

void print_matrix(float *matrix, int n, int m) {
    for(int i =0; i<n; i++){
        printf("|");
        for(int j=0; j<m; j++){
            printf("%.2f ",matrix[i*n+j]);
        }
        printf("|\n");
    }
    printf("\n");
}

void fill_matrix(float *matrix, int n, int m) {
    for(int i =0; i<n; i++){
        for(int j=0; j<m; j++){
            matrix[n*i + j] = n*i + j;
        }
    }
}