#include "kernels.cuh"

__global__ void initializeArrayWithGridStride(const int N, int *p, const int val, bool sourceDifferent, const int source, const int sourceVal) {
    
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < N; i += stride) {
        p[i] = val;
        if (i == source && sourceDifferent) {
            p[i] = sourceVal;
        }
    }
}

__global__ void initializeBooleanArrayWithGridStride(const int N, bool *p, const bool val, bool sourceDifferent, const int source, const bool sourceVal) {
  
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < N; i += stride) {
        p[i] = val;
        if (i == source && sourceDifferent) {
            p[i] = sourceVal;
        }
    }
}

__global__ void relaxWithGridStride(int N, int *in_V, int *in_I, int *in_E, int *in_W, int *out_D, int *out_Di, bool *flag) {

    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < N; i += stride) {
        if (flag[i]) {
            flag[i] = false;
            for (int j = in_I[i]; j < in_I[i+1]; ++j) {
                int u = in_V[i];
                int w = in_W[j];
                int dis_u = out_D[i];
                int dis_v = out_D[in_E[j]];
                int dis = dis_u + w;
                if (dis < dis_v) {
                    atomicMin(&out_Di[in_E[j]], dis);
                }
            }
        }
    }
}

__global__ void updateIndexOfEdgesWithGridStide(int N, int *in_V, int *in_E, int l, int r) {
 
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    
    for (int i = tid; i < N; i += stride) {
        int left = l;
        int right = r;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (in_V[mid] == in_E[i]) {
                in_E[i] = mid;
                break;
            }

            if (in_V[mid] < in_E[i]) {
                left = mid + 1;
            }else {
                right = mid - 1;
            }
        }
    }
}

__global__ void updateDistanceWithGridStride(int N, int *out_D, int *out_Di, bool *flag) {
  
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    
    for (int i = tid; i < N; i += stride) {
        if (out_D[i] > out_Di[i]) {
            out_D[i] = out_Di[i];
            flag[i] = true;
        }
    }
}

__global__ void updatePredWithGridStride(int N, int *in_V, int *in_I, int *in_E, int *in_W, int *out_D, int *out_P) {
    
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < N; i += stride) {
        for (int j = in_I[i]; j < in_I[i+1]; ++j) {
            int u = in_V[i];
            int v = in_V[in_E[j]];
            int w = in_W[j];
            int dis_u = out_D[i];
            int dis_v = out_D[in_E[j]];
            if (dis_v == dis_u + w) {
                atomicMin(&out_P[in_E[j]], u);
            }
        }
    }
}