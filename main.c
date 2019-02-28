#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <sys/time.h>
#include "matrix.h"

#define BLOCK_SIZE 128

extern void sum_rows_gpu(float *A_vals, float *row, int block_size, int n, int m, float *tau);
extern void sum_cols_gpu(float *A_vals, float *col, int block_size, int n, int m, float *tau);
extern void vec_reduce_gpu(float *vec, int block_size, int n, float *sum, float *tau);

int is_empty(FILE* file);
void write_times(char* funcname, float tauCPU, float tauGPU, float tauGPUohead, int block, int n, int m, float err);

int main(int argc, char **argv){
	int m = 10, n = 10;
	int t = 0, r = 0, p = 0, w = 0, option = 0;
	int block_size=BLOCK_SIZE;
	float **A, *A_vals, *row, *col, *rowGPU, *colGPU; 
	float tot_sum, tot_sumGPU;
	struct timeval start, end;
	float tauCPU, tauGPU, tauGPUohead, sse;

	while((option=getopt(argc,argv,"n:m:b:rtwp"))!=-1){
		switch(option){
			case 'n': n = atoi(optarg);
				break;
			case 'm': m = atoi(optarg);
				break;
			case 'b': block_size = atoi(optarg);
				break;
			case 'r': r = 1;
				break;
			case 't': t = 1;
				break;
			case 'p': p = 1;
				break;
			case 'w': w = 1;
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
	sum_rows_gpu(A_vals, rowGPU, block_size, n, m, &tauGPU);
	gettimeofday(&end,NULL);
	tauGPUohead = (float)(end.tv_sec-start.tv_sec) + (float)(end.tv_usec - start.tv_usec)/(1E06);
	sse = SSE(row,rowGPU,n);
	
	printf("\n======================= Row Sum =====================\n");
	printf("SSE of CPU vals vs GPU: %0.7f\n", sse);
	if(t){
		printf("CPU time-taken: %0.7f;\nGPU-time-taken: %0.7f; GPU w o/head: %0.7f;\n",tauCPU,tauGPU,tauGPUohead);
		printf("Speedup: %0.7f\n", tauCPU/tauGPU);
	}
	if(w)
		write_times("row_sum.csv", tauCPU, tauGPU, tauGPUohead, block_size, m, n, sse);
	/////////////////////////////////////////////////

	//////////////// SUM COL ////////////////////////
	gettimeofday(&start,NULL);
	sum_cols(A, col, n, m);
	gettimeofday(&end,NULL);
	tauCPU = (float)(end.tv_sec-start.tv_sec) + (float)(end.tv_usec - start.tv_usec)/(1E06);
	
	gettimeofday(&start,NULL);
	sum_cols_gpu(A_vals, colGPU, block_size, n, m, &tauGPU);
	gettimeofday(&end,NULL);
	tauGPUohead = (float)(end.tv_sec-start.tv_sec) + (float)(end.tv_usec - start.tv_usec)/(1E06);
	sse = SSE(col,colGPU,m);

	printf("\n======================= Col Sum =====================\n");
	printf("SSE of CPU vals vs GPU: %0.7f\n", sse);
	if(t){
		printf("CPU time-taken: %0.7f;\nGPU-time-taken: %0.7f; GPU w o/head: %0.7f;\n",tauCPU,tauGPU,tauGPUohead);
		printf("Speedup: %0.7f\n", tauCPU/tauGPU);
	}
	if(w)
		write_times("col_sum.csv", tauCPU, tauGPU, tauGPUohead, block_size, m, n, sse);
	////////////////////////////////////////////////

	//////////////// VEC RED ///////////////////////
	gettimeofday(&start,NULL);
	tot_sum = vec_reduce(row, n);
	gettimeofday(&end,NULL);
	tauCPU = (float)(end.tv_sec-start.tv_sec) + (float)(end.tv_usec - start.tv_usec)/(1E06);
	
	gettimeofday(&start,NULL);
	vec_reduce_gpu(rowGPU, block_size, n, &tot_sumGPU, &tauGPU);
	gettimeofday(&end,NULL);
	tauGPUohead = (float)(end.tv_sec-start.tv_sec) + (float)(end.tv_usec - start.tv_usec)/(1E06);
	
	printf("\n======================= Reduce ======================\n");
	printf("CPU total sum: %f; GPU total sum: %f;\nerr: %0.7f%%;\n",
										tot_sum,tot_sumGPU, 100*f_abs(tot_sumGPU - tot_sum)/tot_sum);
	if(t){
		printf("CPU time-taken: %0.7f;\nGPU-time-taken: %0.7f; GPU w o/head: %0.7f;\n",tauCPU,tauGPU,tauGPUohead);
		printf("Speedup: %0.7f\n", tauCPU/tauGPU);
	}
	if(w)
		write_times("reduce.csv", tauCPU, tauGPU, tauGPUohead, block_size, n, m, 100*f_abs(tot_sumGPU-tot_sum)/tot_sum);
	////////////////////////////////////////////////
	printf("\n=================================================================\n\n");

	free(A); free(A_vals);
	free(row); free(rowGPU);
	free(col); free(colGPU);

	return 0;
}

int is_empty(FILE* file){
	size_t size;

	fseek(file, 0, SEEK_END);
	size = ftell(file);
	
	if(size)
		return 0;
	else
		return 1;
}

void write_times(char* fname, float tauCPU, float tauGPU, float tauGPUohead, int block, int n, int m, float err){
	FILE* fptr;
	
	fptr = fopen(fname, "a+");
	if(!fptr)
		printf("Couldn't open file %s\n",fname);

	if(is_empty(fptr))
		fprintf(fptr, "Block-size,NxM,CPU time,GPU time,GPU w/ o/head,Speedup,SSE/Err\n");
	fprintf(fptr, "%d,%dx%d,%f,%f,%f,%f,%f\n", block, n, m, tauCPU, tauGPU, tauGPUohead, tauCPU/tauGPU, err);

	fclose(fptr);
}
