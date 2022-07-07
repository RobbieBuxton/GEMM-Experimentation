#include <stdlib.h>
#include <stdio.h>

void print_matrix(double *matrix, int n, int m) {
    for(int i =0; i<n; i++){
        printf("|");
        for(int j=0; j<m; j++){
            printf("%.2f ",matrix[i*n+j]);
        }
        printf("|\n");
    }
    printf("\n");
}