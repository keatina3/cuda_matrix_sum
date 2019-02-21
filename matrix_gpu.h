#ifndef _MAT_GPU_H
#define _MAT_GPU_H

__global__ void sum_rows_gpu(float *A_vals, float *b, int n, int m);
__global__ void sum_cols_gpu(float *A_vals, float *b, int n, int m);
__global__ void vec_reduce_gpu(float *A_vals, float *b, int n, int m);

#endif
