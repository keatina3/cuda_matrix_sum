#ifndef _MAT_H_
#define _MAT_H_

void alloc_mat(double *A_vals, double **A, int n, int m);
void assign_vals(double *A_vals, int n, int m);
void print_mat(double **A, int n, int m);
void sum_rows(double **A, double *b, int n, int m);
void sum_cols(double **A, double *b, int n, int m);
double vec_reduce(double *vec, int n);
double f_abs(double a);
double SSE(double *x, double *y, int n);

#endif
