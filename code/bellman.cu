#include "bellman.cuh"

void printCudaDevice() {
    int device = 0;
    cudaSetDevice(device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    cout << "################# Using device " << device << " #################" << endl;
    cout << "Name: " << prop.name << endl;
    cout << "Total global Memory: " << prop.totalGlobalMem << endl;
    cout << "Compute capability: " << prop.major << "." << prop.minor << endl;
    cout << "Clock rate: " << prop.clockRate << endl;
    cout << "Multiprocessor count: " <<  prop.multiProcessorCount << endl;
    cout << "Shared memory per multiprocessor: " << prop.sharedMemPerBlock << endl;
    cout << "Registers per multiprocessor: " << prop.regsPerBlock << endl;
    cout << "Threads in warp: " << prop.warpSize << endl;
    cout << "Max threads per block: " << prop.maxThreadsPerBlock << endl;
    cout << "Max threads dimensions: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << endl;
    cout << "Max grid dimensions: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << endl;
    cout << "################# End of device stats #################" << endl;
    cout << endl;
}

int runBellmanFordOnGPU(const char *file, int blocks, int blockSize, int debug) {

    std::string inputFile(file);
    const int BLOCKS = blocks;
    const int BLOCK_SIZE = blockSize;
    const int DEBUG = debug;
    const int MAX_VAL = std::numeric_limits<int>::max();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cout << "Running Bellman Ford on GPU with Grid Stride Kernel & relax only when needed" << endl;
    cudaEventRecord(start, NULL);

    std::vector<int> V, I, E, W;
    loadVector((inputFile + "_V.csv").c_str(), V);
    loadVector((inputFile + "_I.csv").c_str(), I);
    loadVector((inputFile + "_E.csv").c_str(), E);
    loadVector((inputFile + "_W.csv").c_str(), W);

    if (DEBUG) {
        cout << "loadVector is done!" << endl;
    }
    printCudaDevice();
    cout << "Blocks: " << BLOCKS << " Block size: " << BLOCK_SIZE << endl;

    int *in_V, *in_I, *in_E, *in_W;
    int *out_D, *out_Di, *out_P;
    bool *flag;

    cudaMalloc((void**) &in_V, V.size() * sizeof(int));
    cudaMalloc((void**) &in_I, I.size() * sizeof(int));
    cudaMalloc((void**) &in_E, E.size() * sizeof(int));
    cudaMalloc((void**) &in_W, W.size() * sizeof(int));

    cudaMalloc((void**) &out_D, V.size() * sizeof(int));    
    cudaMalloc((void**) &out_Di, V.size() * sizeof(int));    
    cudaMalloc((void**) &out_P, V.size() * sizeof(int));    
    cudaMalloc((void**) &flag, V.size() * sizeof(bool));    

    cudaMemcpy(in_V, V.data(), V.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(in_I, I.data(), I.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(in_E, E.data(), E.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(in_W, W.data(), W.size() * sizeof(int), cudaMemcpyHostToDevice);

    initializeArrayWithGridStride<<<BLOCKS, BLOCK_SIZE>>> (V.size(), out_D, MAX_VAL, true, 0, 0);
    initializeArrayWithGridStride<<<BLOCKS, BLOCK_SIZE>>> (V.size(), out_Di, MAX_VAL, true, 0, 0);
    initializeArrayWithGridStride<<<BLOCKS, BLOCK_SIZE>>> (V.size(), out_P, MAX_VAL, true, 0, -1);
    initializeBooleanArrayWithGridStride<<<BLOCKS, BLOCK_SIZE>>> (V.size(), flag, false, true, 0, true);

    updateIndexOfEdgesWithGridStide<<<BLOCKS, BLOCK_SIZE>>> (E.size(), in_V, in_E, 0, V.size()-1);

    for (int round = 1; round < V.size(); ++round) {
        if (DEBUG && round % 10000 == 0) {
            cout << "######## round = " << round << " ########" << endl;
        }
        relaxWithGridStride<<<BLOCKS, BLOCK_SIZE>>> (V.size(), in_V, in_I, in_E, in_W, out_D, out_Di, flag);
        updateDistanceWithGridStride<<<BLOCKS, BLOCK_SIZE>>> (V.size(), out_D, out_Di, flag);
    }
    updatePredWithGridStride<<<BLOCKS, BLOCK_SIZE>>> (V.size(), in_V, in_I, in_E, in_W, out_D, out_P);

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    float elapsedTime = 0.0f;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    int *out_dis = new int[V.size()];
    int *out_pred = new int[V.size()];

    cudaMemcpy(out_dis, out_D, V.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_pred, out_P, V.size() * sizeof(int), cudaMemcpyDeviceToHost);

    makeOutputFileName(inputFile);
    storeResult(("output/" + inputFile + "_cuda.csv").c_str(), V, out_dis, out_pred);
    cout << "Results written to " << ("output/" + inputFile + "_cuda.csv").c_str() << endl;
    cout << "average time elapsed : " << elapsedTime << " milli seconds" << endl;

    free(out_dis);
    free(out_pred);
    cudaFree(in_V);
    cudaFree(in_I);
    cudaFree(in_E);
    cudaFree(in_W);
    cudaFree(out_D);
    cudaFree(out_Di);
    cudaFree(out_P);
    cudaFree(flag);

    return 0;
}
