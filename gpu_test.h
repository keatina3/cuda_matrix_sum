#ifdef __cplusplus
extern "C" {
    int sum_rows_gpu(float *vals, int n, int m);
	int sum_cols_gpu(float *A_vals, int n, int m);
	int vec_reduce_gpu(float *vec, int n);
}
#endif
