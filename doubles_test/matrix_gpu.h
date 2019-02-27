#ifndef _MAT_GPU_H
#ifdef __cplusplus
extern "C" {
	void sum_rows_gpu(double *vals, double *row, int block, int n, int m, double *tau);
	void sum_cols_gpu(double *vals, double *col, int block, int n, int m, double *tau);
	void vec_reduce_gpu(double *vec, int block_size, int n, double *sum, double *tau);
}
#endif

#define _MAT_GPU_H

__global__ void calc_sum_rows_gpu(double *A_vals, double *b, int n, int m);
__global__ void calc_sum_cols_gpu(double *A_vals, double *b, int n, int m);
__global__ void calc_vec_reduce_gpu(double *A_vals, double *b, int n, int m);

#endif
