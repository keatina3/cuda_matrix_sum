#include <stdio.h>
#include <stdlib.h>
#include "gpu_test.h"
#include "matrix_gpu.h"

extern int block_size;

__global__ void sum_rows_gpu(float *A_vals, float *b, int n, int m){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int i;

	if(idx<n){
		b[idx]=0.0;
		for(i=0;i<m;i++)
			b[idx] += A_vals[i + idx*m];
	}
}

__global__ void sum_cols_gpu(float *A_vals, float *b, int n, int m){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int i;

	if(idx<m){
		b[idx] = 0.0;
		for(i=0;i<n;i++)
			b[idx] += A_vals[idx + i*m];
	}
}
	
__global__ void vec_reduce_gpu(float *vec, float *b, int n){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(idx<n/2)
		b[idx] = vec[2*idx] + vec[(2*idx)+1];
	if(n%2!=0 && idx == (n/2)-1)
		b[idx] += vec[(2*idx) + 2];
	
}

extern int sum_rows_gpu(float *A_vals, int n, int m){
	float *A_vals_d, *row_d, *row;

	row = (float*)calloc(n,sizeof(float));

	cudaMalloc( (void**)&A_vals_d, n*m*sizeof(float));
	cudaMalloc( (void**)&row_d, n*sizeof(float));

	cudaMemcpy(A_vals_d, A_vals, n*m*sizeof(float), cudaMemcpyHostToDevice);
	
	dim3 dimBlock(block_size);
	dim3 dimGrid((n/dimBlock.x) + (!(n%dimBlock.x)?0:1));

	sum_rows_gpu <<<dimGrid,dimBlock>>> (A_vals_d, row_d, n, m);

	cudaMemcpy(row, row_d, n*sizeof(float), cudaMemcpyDeviceToHost);
	
	free(row);
	cudaFree(A_vals_d); cudaFree(row_d);
	
	printf("TESTING ROW FUNCTION CALL\n");
	return 0;
}

extern int sum_cols_gpu(float *A_vals, int n, int m){
	float *A_vals_d, *col_d, *col;

	col = (float*)calloc(m,sizeof(float));

	cudaMalloc( (void**)&A_vals_d, n*m*sizeof(float));
	cudaMalloc( (void**)&col_d, m*sizeof(float));

	cudaMemcpy(A_vals_d, A_vals, n*m*sizeof(float), cudaMemcpyHostToDevice);
	
	dim3 dimBlock(block_size);
	dim3 dimGrid((m/dimBlock.x) + (!(m%dimBlock.x)?0:1));

	sum_cols_gpu <<<dimGrid,dimBlock>>> (A_vals_d, col_d, n, m);

	cudaMemcpy(col, col_d, m*sizeof(float), cudaMemcpyDeviceToHost);
	
	free(col);
	cudaFree(A_vals_d); cudaFree(col_d);
	
	printf("TESTING COL FUNCTION CALL\n");
	return 0;
}

extern float vec_reduce_gpu(float *vec, int n){
	float sum, *vec_d;

	cudaMalloc( (void**)&vec_d, n*sizeof(float));

	cudaMemcpy(vec_d, vec, n*sizeof(float), cudaMemcpyHostToDevice);
	
	dim3 dimBlock(block_size);
	dim3 dimGrid((n/dimBlock.x) + (!(n%dimBlock.x)?0:1));

	while(n > 1){
		vec_reduce_gpu <<<dimGrid,dimBlock>>> (vec_d, vec_d, n);
		n /= 2;
	}
	cudaMemcpy(&sum, &vec_d[0], sizeof(float), cudaMemcpyDeviceToHost);
	
	printf("total sum = %f\n", sum);
	cudaFree(vec_d);
	
	printf("TESTING ROW FUNCTION CALL\n");
	return sum;
}
