#ifndef _MAT_H_
#define _MAT_H_

void alloc_mat(float *A_vals, float **A, int n, int m);
void assign_vals(float *A_vals, int n, int m);
void print_mat(float **A, int n, int m);
void sum_rows(float **A, float *b, int n, int m);
void sum_cols(float **A, float *b, int n, int m);
float vec_reduce(float *vec, int n);
float f_abs(float a);
float SSE(float *x, float *y, int n);

#endif
