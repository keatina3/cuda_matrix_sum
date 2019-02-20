#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <sys/time.h>
#include "matrix.h"
#include "gpu_test.h"

extern int sum_rows_gpu(float *A_vals, int n, int m);
extern int sum_cols_gpu(float *A_vals, int n, int m);
extern int vec_reduce_gpu(float *vec, int n);

int main(int argc, char **argv){
	int m = 10, n = 10;
	int t = 0, r = 0, option = 0;
	float **A, *A_vals, *row, *col, tot_sum;
	struct timeval seed;

	while((option=getopt(argc,argv,"n:m:rt"))!=-1){
		switch(option){
			case 'n': n = atoi(optarg);
				break;
			case 'm': m = atoi(optarg);
				break;
			case 'r': r = 1;
				break;
			case 't': t = 1;
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

	gettimeofday(&seed, NULL);
	if(r)
		srand48(seed.tv_usec);
	else
		srand48(123456);

	A = (float**)malloc(n*sizeof(float*));
	A_vals = (float*)calloc(n*m,sizeof(float));
	row = (float*)calloc(n,sizeof(float));
	col = (float*)calloc(m,sizeof(float));

	alloc_mat(A_vals, A, n, m);
	assign_vals(A_vals, n, m);
	print_mat(A, n, m);
	
	sum_rows(A, row, n, m);
	sum_cols(A, col, n, m);
	tot_sum = vec_reduce(row, n);
	printf("Total sum = %f\n",tot_sum);	
	
	sum_rows_gpu(A_vals, n, m);
	
	free(A);
	free(A_vals);
	free(row);
	free(col);

	return 0;
}