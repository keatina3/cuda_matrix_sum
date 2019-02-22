#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <sys/time.h>
#include "matrix.h"

#define BLOCK_SIZE 256

extern int sum_rows_gpu(float *A_vals, float *row, int n, int m);
extern int sum_cols_gpu(float *A_vals, float *col, int n, int m);
extern int vec_reduce_gpu(float *vec, int n, float *sum);

int block_size = BLOCK_SIZE;

int main(int argc, char **argv){
	int m = 10, n = 10;
	int t = 0, r = 0, p = 0, option = 0;
	float **A, *A_vals, *row, *col, *rowGPU, *colGPU; 
	float tot_sum, tot_sumGPU;
	struct timeval start, end;
	float tauCPU, tauGPU;
	//int i;

	while((option=getopt(argc,argv,"n:m:rtp"))!=-1){
		switch(option){
			case 'n': n = atoi(optarg);
				break;
			case 'm': m = atoi(optarg);
				break;
			case 'r': r = 1;
				break;
			case 't': t = 1;
				break;
			case 'p': p = 1;
				break;
			default:
				printf("Incorrect options entered!\n");
				return 1;
		}
	}	
	if(argc != optind){
		printf("Too many arguments provided, exiting!\n");
		return 1;
	}

	gettimeofday(&start, NULL);
	if(r)
		srand48(start.tv_usec);
	else
		srand48(123456);

	A = (float**)malloc(n*sizeof(float*));
	A_vals = (float*)calloc(n*m,sizeof(float));
	row = (float*)calloc(n,sizeof(float));
	col = (float*)calloc(m,sizeof(float));
	rowGPU = (float*)calloc(n,sizeof(float));
	colGPU = (float*)calloc(m,sizeof(float));

	alloc_mat(A_vals, A, n, m);
	assign_vals(A_vals, n, m);
	printf("\n=================================================================\n");
	printf("\nBlock-size: %d; NxM: %dx%d;\n",block_size,n,m);
	if(n<=10 && m <= 10 && p){
		print_mat(A, n, m);
		printf("\n\n");
	}
	
	//////////////// SUM ROW ////////////////////////
	gettimeofday(&start,NULL);
	sum_rows(A, row, n, m);
	gettimeofday(&end,NULL);
	tauCPU = (float)(end.tv_sec-start.tv_sec) + (float)(end.tv_usec - start.tv_usec)/(1E06);
	
	gettimeofday(&start,NULL);
	sum_rows_gpu(A_vals, rowGPU, n, m);
	gettimeofday(&end,NULL);
	tauGPU = (float)(end.tv_sec-start.tv_sec) + (float)(end.tv_usec - start.tv_usec)/(1E06);
	
	printf("\n======================= Row Sum =====================\n");
	printf("SSE of CPU vals vs GPU: %0.7f\n", SSE(row, rowGPU, n));
	if(t){
		printf("CPU time-taken: %0.7f; GPU-time-taken: %0.7f;\n",tauCPU,tauGPU);
		printf("Speedup: %0.7f\n", tauCPU/tauGPU);
	}
	/////////////////////////////////////////////////

	//////////////// SUM COL ////////////////////////
	gettimeofday(&start,NULL);
	sum_cols(A, col, n, m);
	gettimeofday(&end,NULL);
	tauCPU = (float)(end.tv_sec-start.tv_sec) + (float)(end.tv_usec - start.tv_usec)/(1E06);
	
	gettimeofday(&start,NULL);
	sum_cols_gpu(A_vals, colGPU, n, m);
	gettimeofday(&end,NULL);
	tauGPU = (float)(end.tv_sec-start.tv_sec) + (float)(end.tv_usec - start.tv_usec)/(1E06);
	
	printf("\n======================= Col Sum =====================\n");
	printf("SSE of CPU vals vs GPU: %0.7f\n", SSE(col, colGPU, n));
	if(t){
		printf("CPU time-taken: %0.7f; GPU-time-taken: %0.7f;\n",tauCPU,tauGPU);
		printf("Speedup: %0.7f\n", tauCPU/tauGPU);
	}
	////////////////////////////////////////////////

	//////////////// VEC RED ///////////////////////
	gettimeofday(&start,NULL);
	tot_sum = vec_reduce(row, n);
	gettimeofday(&end,NULL);
	tauCPU = (float)(end.tv_sec-start.tv_sec) + (float)(end.tv_usec - start.tv_usec)/(1E06);
	
	gettimeofday(&start,NULL);
	//tot_sumGPU = vec_reduce_gpu(row, n);
	vec_reduce_gpu(row, n, &tot_sumGPU);
	gettimeofday(&end,NULL);
	tauGPU = (float)(end.tv_sec-start.tv_sec) + (float)(end.tv_usec - start.tv_usec)/(1E06);
	
	printf("\n======================= Reduce ======================\n");
	printf("CPU total sum: %f; GPU total sum: %f;\nerr: %0.7f%%;\n",
										tot_sum,tot_sumGPU, f_abs(tot_sumGPU - tot_sum)/tot_sum);
	if(t){
		printf("CPU time-taken: %0.7f; GPU-time-taken: %0.7f;\n",tauCPU,tauGPU);
		printf("Speedup: %0.7f\n", tauCPU/tauGPU);
	}
	////////////////////////////////////////////////
	printf("\n=================================================================\n\n");

	//printf("row[0] = %f, rowGPU[0] = %f\n", row[0], rowGPU[0]);
	//for(i=0;i<m;i++)
	//	printf("col[%d] = %f, colGPU[0] = %f\n", i, col[i], colGPU[i]);

	free(A); free(A_vals);
	free(row); free(rowGPU);
	free(col); free(colGPU);

	return 0;
}
