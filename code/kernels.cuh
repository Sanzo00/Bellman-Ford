#ifndef BELLMAN_FORD_KERNELS_CUH
#define BELLMAN_FORD_KERNELS_CUH

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void initializeArrayWithGridStride(const int N, int *p, const int val, bool sourceDifferent, const int source, const int sourceVal);
__global__ void initializeBooleanArrayWithGridStride(const int N, bool *p, const bool val, bool sourceDifferent, const int source, const bool sourceVal);
__global__ void relaxWithGridStride(int N, int *in_V, int *in_I, int *in_E, int *in_W, int *out_D, int *out_Di, bool *flag);
__global__ void updateIndexOfEdgesWithGridStide(int N, int *in_V, int *in_E, int l, int r);
__global__ void updateDistanceWithGridStride(int N, int *out_D, int *out_Di, bool *flag);
__global__ void updatePredWithGridStride(int N, int *in_V, int *in_I, int *in_E, int *in_W, int *out_D, int *out_P);

#endif // BELLMAN_FORD_KERNELS_CUH