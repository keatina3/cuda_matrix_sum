#ifndef _MAT_GPU_H
#ifdef __cplusplus
extern "C" {
	void sum_rows_gpu(float *vals, float *row, int n, int m, float *tau);
	void sum_cols_gpu(float *A_vals, float *col, int n, int m, float *tau);
	void vec_reduce_gpu(float *vec, int n, float *sum, float *tau);
}
#endif

#define _MAT_GPU_H

__global__ void calc_sum_rows_gpu(float *A_vals, float *b, int n, int m);
__global__ void calc_sum_cols_gpu(float *A_vals, float *b, int n, int m);
__global__ void calc_vec_reduce_gpu(float *A_vals, float *b, int n, int m);

#endif
