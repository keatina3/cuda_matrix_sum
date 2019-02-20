#include <stdio.h>
#include "gpu_test.h"
#include "matrix_gpu.h"

__global__ void sum_rows_gpu(float *A_vals, float *b, int n, int m){
	
}

__global__ void sum_cols_gpu(float *A_vals, float *b, int n, int m){

}

__global__ void sum_tot_gpu(float *A_vals, int n, int m){

}

extern int sum_rows_gpu(float *A_vals, int n, int m){
	printf("TESTING FUNCTION CALL\n");
	return 0;
}

extern int sum_cols_gpu(float *A_vals, int n, int m){
	
	return 0;
}

extern int vec_reduce_gpu(float *vec, int n){

	return 0;
}
