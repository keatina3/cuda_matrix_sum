#ifdef __cplusplus
extern "C" {
    int gpu_test(float **vals, int n, int m);
}
#endif

#ifndef _MAT_GPU_H
#define _MAT_GPU_H

__global__ void sum_rows_gpu(float *A_vals, float *b, int n, int m);
__global__ void sum_cols_gpu(float *A_vals, float *b, int n, int m);
__global__ float sum_tot_gpu(float *A_vals, int n, int m);

#endif
