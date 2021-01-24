#ifndef BELLMAN_FORD_BELLMAN_CUH
#define BELLMAN_FORD_BELLMAN_CUH

#include "kernels.cuh"
#include "utilities.h"
#include <iostream>
#include <vector>
#include <limits>

using std::cout;
using std::endl;

void printCudaDevice();
int runBellmanFordOnGPU(const char *file, int blocks, int blockSize, int debug);

#endif // BELLMAN_FORD_BELLMAN_CUH