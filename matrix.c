#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

void alloc_mat(float *A_vals, float **A, int n, int m){
	int i;

	for(i=0;i<n;i++)
		A[i] = &A_vals[i*m];
}

void assign_vals(float *A_vals, int n, int m){
	int i;
	for(i=0;i<(n*m);i++){
		A_vals[i] = ((float)(drand48())*2.0)-1.0;
	}
}

void print_mat(float **A, int n, int m){
	int i,j;
	for(i=0;i<n;i++){
		for(j=0;j<m;j++)
			printf("%f ", A[i][j]);
		printf("\n");
	}
}

void sum_rows(float **A, float *b, int n, int m){
	int i,j;

	for(i=0;i<n;i++)
		for(j=0;j<m;j++)
			b[i] += f_abs(A[i][j]);
}

void sum_cols(float **A, float *b, int n, int m){
	int i,j;

	for(i=0;i<n;i++)
		for(j=0;j<m;j++)
			b[j] += f_abs(A[i][j]);
}

float vec_reduce(float *vec, int n){
	float sum = 0.0;
	int i;
	for(i=0;i<n;i++)
		sum += vec[i];

	return sum;
}

float f_abs(float a){
	if(a < 0.0)
		return -1.0*a;
	else
		return a;
}

float SSE(float *x, float *y, int n){
	float SSE = 0.0;
	int i;

	for(i=0;i<n;i++)
		SSE += (x[i]-y[i])*(x[i]-y[i]);
	return SSE;
}
