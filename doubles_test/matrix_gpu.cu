#include <stdlib.h>
#include <sys/time.h>
#include "matrix_gpu.h"

extern int block_size;

__global__ void calc_sum_rows_gpu(double *A_vals, double *b, int n, int m){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int i;

	if(idx<n){
		b[idx]=0.0;
		for(i=0;i<m;i++){
			if(A_vals[i + idx*m] > 0.0)
				b[idx] += A_vals[i + idx*m];
			else
				b[idx] -= A_vals[i + idx*m];
		}
	}
}

__global__ void calc_sum_cols_gpu(double *A_vals, double *b, int n, int m){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int i;

	if(idx<m){
		b[idx] = 0.0;
		for(i=0;i<n;i++){
			if(A_vals[idx + i*m] > 0.0)
				b[idx] += A_vals[idx + i*m];
			else
				b[idx] -= A_vals[idx + i*m];
		}
	}
}
	
__global__ void calc_vec_reduce_gpu(double *vec, double *b, int n){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(idx<n/2)
		b[idx] = vec[2*idx] + vec[(2*idx)+1];
	if(n%2!=0 && idx == (n/2)-1)
		b[idx] += vec[(2*idx) + 2];
}

extern void sum_rows_gpu(double *A_vals, double *row, int n, int m, float *tau){
	double *A_vals_d, *row_d;
	struct timeval start, end;

	cudaMalloc( (void**)&A_vals_d, n*m*sizeof(double));
	cudaMalloc( (void**)&row_d, n*sizeof(double));

	cudaMemcpy(A_vals_d, A_vals, n*m*sizeof(double), cudaMemcpyHostToDevice);
	
	dim3 dimBlock(block_size);
	dim3 dimGrid((n/dimBlock.x) + (!(n%dimBlock.x)?0:1));

	gettimeofday(&start,NULL);
	calc_sum_rows_gpu <<<dimGrid,dimBlock>>> (A_vals_d, row_d, n, m);
	gettimeofday(&end,NULL);
	
	*tau = (double)(end.tv_sec-start.tv_sec) + (double)(end.tv_usec - start.tv_usec)/(1E06);

	cudaMemcpy(row, row_d, n*sizeof(double), cudaMemcpyDeviceToHost);
	
	cudaFree(A_vals_d); cudaFree(row_d);
}

extern void sum_cols_gpu(double *A_vals, double *col, int n, int m, float *tau){
	double *A_vals_d, *col_d;
	struct timeval start, end;

	cudaMalloc( (void**)&A_vals_d, n*m*sizeof(double));
	cudaMalloc( (void**)&col_d, m*sizeof(double));
	
	cudaMemcpy(A_vals_d, A_vals, n*m*sizeof(double), cudaMemcpyHostToDevice);

	dim3 dimBlock(block_size);
	dim3 dimGrid((m/dimBlock.x) + (!(m%dimBlock.x)?0:1));

	gettimeofday(&start,NULL);
	calc_sum_cols_gpu <<<dimGrid,dimBlock>>> (A_vals_d, col_d, n, m);
	gettimeofday(&end,NULL);
	
	*tau = (double)(end.tv_sec-start.tv_sec) + (double)(end.tv_usec - start.tv_usec)/(1E06);
	
	cudaMemcpy(col, col_d, m*sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(A_vals_d); cudaFree(col_d);
}

extern void vec_reduce_gpu(double *vec, int n, double* sum, float *tau){
	double *vec_d;
	struct timeval start, end;

	cudaMalloc( (void**)&vec_d, n*sizeof(double));

	cudaMemcpy(vec_d, vec, n*sizeof(double), cudaMemcpyHostToDevice);
	
	dim3 dimBlock(block_size);
	dim3 dimGrid((n/dimBlock.x) + (!(n%dimBlock.x)?0:1));

	gettimeofday(&start,NULL);
	while(n > 1){
		calc_vec_reduce_gpu <<<dimGrid,dimBlock>>> (vec_d, vec_d, n);
		n /= 2;
	}
	gettimeofday(&end,NULL);
	
	*tau = (double)(end.tv_sec-start.tv_sec) + (double)(end.tv_usec - start.tv_usec)/(1E06);
	
	cudaMemcpy(sum, &vec_d[0], sizeof(double), cudaMemcpyDeviceToHost);
	
	cudaFree(vec_d);
}
