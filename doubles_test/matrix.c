#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

void alloc_mat(double *A_vals, double **A, int n, int m){
	int i;

	for(i=0;i<n;i++)
		A[i] = &A_vals[i*m];
}

void assign_vals(double *A_vals, int n, int m){
	int i;
	for(i=0;i<(n*m);i++){
		A_vals[i] = ((double)(drand48())*2.0)-1.0;
	}
}

void print_mat(double **A, int n, int m){
	int i,j;
	for(i=0;i<n;i++){
		for(j=0;j<m;j++)
			printf("%f ", A[i][j]);
		printf("\n");
	}
}

void sum_rows(double **A, double *b, int n, int m){
	int i,j;

	for(i=0;i<n;i++)
		for(j=0;j<m;j++)
			b[i] += f_abs(A[i][j]);
}

void sum_cols(double **A, double *b, int n, int m){
	int i,j;

	for(i=0;i<n;i++)
		for(j=0;j<m;j++)
			b[j] += f_abs(A[i][j]);
}

double vec_reduce(double *vec, int n){
	double sum = 0.0;
	int i;
	for(i=0;i<n;i++)
		sum += vec[i];

	return sum;
}

double f_abs(double a){
	if(a < 0.0)
		return -1.0*a;
	else
		return a;
}

double SSE(double *x, double *y, int n){
	double SSE = 0.0;
	int i;

	for(i=0;i<n;i++)
		SSE += (x[i]-y[i])*(x[i]-y[i]);
	return SSE;
}
