#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <chrono>

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error in %s (line %d): %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)



__global__ void histogramBase(const int *input, int *globalHist, int N, int numBins) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int value = input[i];
        int bin = value * numBins / 1024;  
        globalHist[bin]++;
    }

}
__global__ void histogramAtomic(const int *input, int *globalHist, int N, int numBins) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int value = input[i];
        int bin = value * numBins / 1024;  
        atomicAdd(&globalHist[bin], 1);
    }

}

__global__ void histogramPrivatization(const int *input, int *globalHist, int N, int numBins) {

    __shared__ int sharedHist[256];  
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        sharedHist[i] = 0;
    }
    __syncthreads();

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalIdx < N) {

        int value = input[globalIdx];
        int bin = value * numBins / 1024; 
        atomicAdd(&sharedHist[bin], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        if (sharedHist[i] > 0) {  
            atomicAdd(&globalHist[i], sharedHist[i]);
        }
    }
}


#define CFACTOR 4

__global__ void histogramcoarsenedCon(const int *data, int *histo, int n, int numBins) {
    extern __shared__ int histo_shared[];
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        histo_shared[i] = 0;
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid * CFACTOR; i < n; i += gridDim.x * blockDim.x * CFACTOR) {
        for (int j = 0; j < CFACTOR && (i + j) < n; j++) {
            int bin = data[i+j] * numBins / 1024;
            atomicAdd(&histo_shared[bin], 1);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        atomicAdd(&histo[i], histo_shared[i]);
    }
}


__global__ void histogramcoarsenedInt(const int *data, int *histo, int n, int numBins) {
    extern __shared__ int histo_shared[];

    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        histo_shared[i] = 0;
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tid * CFACTOR; i < min((tid + 1) * CFACTOR, n); i++) {
        int bin = data[i] * numBins / 1024;
        atomicAdd(&histo_shared[bin], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        int bin_value = histo_shared[i];
        atomicAdd(&histo[i], bin_value);
    }
}

__global__ void histogramaggregate(const int *data, int *histo, int n, int numBins) {
    extern __shared__ int histo_shared[];
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        histo_shared[i] = 0;
    }
    __syncthreads();

 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

    int accumulator = 0;
    int prevBin = -1;


    for (int i = tid; i < n; i += totalThreads) {
        int bin = data[i] * numBins / 1024;
        if (bin == prevBin) {

            accumulator++;
        } else {

            if (prevBin != -1) {
                atomicAdd(&histo_shared[prevBin], accumulator);
            }

            accumulator = 1;
            prevBin = bin;
        }
    }

    if (prevBin != -1) {
        atomicAdd(&histo_shared[prevBin], accumulator);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        atomicAdd(&histo[i], histo_shared[i]);
    }
}



void histogramCPU(const int* input, int* hist, int N, int numBins) {
    for (int i = 0; i < N; i++) {
        int bin = input[i] * numBins / 1024;
        hist[bin]++;
    }
}

int main(int argc, char *argv[]) {

    int numBins = 256;        
    long long vecSize = 1 << 20;  
    if (argc == 4 && strcmp(argv[1], "-i") == 0) {
        numBins = atoi(argv[2]);
        vecSize = atoll(argv[3]);
    } else {
        printf("Usage: %s -i <BinNum> <VecDim>\n", argv[0]);
        printf("Using default numBins=%d, vecSize=%lld\n", numBins, vecSize);
    }
    if (numBins > 256) {
        fprintf(stderr, "Error: numBins exceeds 256 (max supported in this implementation).\n");
        return EXIT_FAILURE;
    }


    int *h_input = (int*)malloc(vecSize * sizeof(int));
    int *h_hist  = (int*)malloc(numBins * sizeof(int));
    if (!h_input || !h_hist) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

  
    srand((unsigned)time(NULL));
    for (long long i = 0; i < vecSize; ++i) {
        h_input[i] = rand() % 1024;
    }

    memset(h_hist, 0, numBins * sizeof(int));


    int *d_input = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input, vecSize * sizeof(int)));
    

    int *d_hist_base = nullptr;
    int *d_hist_atomic = nullptr;
    int *d_hist_private = nullptr;
    int *d_hist_con = nullptr;
    int *d_hist_int = nullptr;
    CHECK_CUDA(cudaMalloc(&d_hist_base, numBins * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_hist_atomic, numBins * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_hist_private, numBins * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_hist_con, numBins * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_hist_int, numBins * sizeof(int)));


    int *d_hist_agg = nullptr;
    CHECK_CUDA(cudaMalloc(&d_hist_agg, numBins * sizeof(int)));


    CHECK_CUDA(cudaMemcpy(d_input, h_input, vecSize * sizeof(int), cudaMemcpyHostToDevice));



    int* h_hist_cpu = (int*)malloc(numBins * sizeof(int));
    memset(h_hist_cpu, 0, numBins * sizeof(int));

    auto start_cpu = std::chrono::high_resolution_clock::now();
    histogramCPU(h_input, h_hist_cpu, vecSize, numBins);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = end_cpu - start_cpu;

   
    int threadsPerBlock = 256;  
    int blocksPerGrid   = (vecSize + threadsPerBlock - 1) / threadsPerBlock;

 
    cudaEvent_t start[6], stop[6];
    float milliseconds[6] = {0};
    for(int i = 0; i < 6; i++) {
        CHECK_CUDA(cudaEventCreate(&start[i]));
        CHECK_CUDA(cudaEventCreate(&stop[i]));
    }

 
    CHECK_CUDA(cudaEventRecord(start[0]));

 
    histogramBase<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_hist_base, (int)vecSize, numBins);
    CHECK_CUDA(cudaGetLastError());  
    CHECK_CUDA(cudaDeviceSynchronize());  


    CHECK_CUDA(cudaEventRecord(stop[0]));
    CHECK_CUDA(cudaEventSynchronize(stop[0]));


    CHECK_CUDA(cudaEventElapsedTime(&milliseconds[0], start[0], stop[0]));
    //printf("Kernel execution time: %f ms\n", milliseconds[0]);

    CHECK_CUDA(cudaMemset(d_hist_atomic, 0, numBins * sizeof(int)));
    CHECK_CUDA(cudaEventRecord(start[1]));
    histogramAtomic<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_hist_atomic, vecSize, numBins);
    CHECK_CUDA(cudaEventRecord(stop[1]));
    CHECK_CUDA(cudaEventSynchronize(stop[1]));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds[1], start[1], stop[1]));


    CHECK_CUDA(cudaMemset(d_hist_private, 0, numBins * sizeof(int)));
    CHECK_CUDA(cudaEventRecord(start[2]));
    histogramPrivatization<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_hist_private, vecSize, numBins);
    CHECK_CUDA(cudaEventRecord(stop[2]));
    CHECK_CUDA(cudaEventSynchronize(stop[2]));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds[2], start[2], stop[2]));

   
    CHECK_CUDA(cudaMemset(d_hist_con, 0, numBins * sizeof(int)));
    CHECK_CUDA(cudaEventRecord(start[3]));
    histogramcoarsenedCon<<<blocksPerGrid, threadsPerBlock, numBins * sizeof(int)>>>(
        d_input, d_hist_con, vecSize, numBins);
    CHECK_CUDA(cudaEventRecord(stop[3]));
    CHECK_CUDA(cudaEventSynchronize(stop[3]));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds[3], start[3], stop[3]));

 
    CHECK_CUDA(cudaMemset(d_hist_int, 0, numBins * sizeof(int)));
    CHECK_CUDA(cudaEventRecord(start[4]));
    histogramcoarsenedInt<<<blocksPerGrid, threadsPerBlock, numBins * sizeof(int)>>>(
        d_input, d_hist_int, vecSize, numBins);
    CHECK_CUDA(cudaEventRecord(stop[4]));
    CHECK_CUDA(cudaEventSynchronize(stop[4]));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds[4], start[4], stop[4]));


    CHECK_CUDA(cudaMemset(d_hist_agg, 0, numBins * sizeof(int)));
    CHECK_CUDA(cudaEventRecord(start[5]));
    histogramaggregate<<<blocksPerGrid, threadsPerBlock, numBins * sizeof(int)>>>(
        d_input, d_hist_agg, vecSize, numBins);
    CHECK_CUDA(cudaEventRecord(stop[5]));
    CHECK_CUDA(cudaEventSynchronize(stop[5]));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds[5], start[5], stop[5]));

 

    printf("CPU Time: %f ms\n", cpu_duration.count());
    printf("Base Time: %f ms\n", milliseconds[0]);
    printf("Atomic Time: %f ms\n", milliseconds[1]);
    printf("Privatization Time: %f ms\n", milliseconds[2]);
    printf("coarsened 1 Time: %f ms\n", milliseconds[3]);
    printf("coarsened 2 Time: %f ms\n", milliseconds[4]);
    printf("Aggregation Time: %f ms\n", milliseconds[5]);

    
    int *h_hist_base = (int*)malloc(numBins * sizeof(int));
    int *h_hist_atomic = (int*)malloc(numBins * sizeof(int));
    int *h_hist_private = (int*)malloc(numBins * sizeof(int));
    int *h_hist_con = (int*)malloc(numBins * sizeof(int));
    int *h_hist_int = (int*)malloc(numBins * sizeof(int));
    int *h_hist_agg = (int*)malloc(numBins * sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_hist_base, d_hist_base, numBins * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_hist_atomic, d_hist_atomic, numBins * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_hist_private, d_hist_private, numBins * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_hist_con, d_hist_con, numBins * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_hist_int, d_hist_int, numBins * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_hist_agg, d_hist_agg, numBins * sizeof(int), cudaMemcpyDeviceToHost));


    long long total_base = 0, total_atomic = 0, total_private = 0;
    for(int i = 0; i < numBins; i++) {
        total_base += h_hist_base[i];
        total_atomic += h_hist_atomic[i];
        total_private += h_hist_private[i];
    }
    
    long long total_con = 0, total_int = 0;
    for(int i = 0; i < numBins; i++) {
        total_con += h_hist_con[i];
        total_int += h_hist_int[i];
    }

    long long total_agg = 0;
    for(int i = 0; i < numBins; i++) {
        total_agg += h_hist_agg[i];
    }

    long long total_cpu = 0;
    for(int i = 0; i < numBins; i++) {
        total_cpu += h_hist_cpu[i];
    }

    // Calculate GFLOPS
    // Operations per element: 1 multiplication, 1 division, 1 atomic add, total 3 floating point operations
    float ops_per_element = 3.0f;
    float total_ops = vecSize * ops_per_element;
    float gflops = (total_ops * 1e-9f); 

    printf("\n=== Performance Statistics ===\n");
    printf("Total Operations: %.2f billion operations\n", gflops);
    printf("CPU Performance: %.2f GFLOPS\n", gflops / (cpu_duration.count() * 0.001f));
    printf("Base Version Performance: %.2f GFLOPS\n", gflops / (milliseconds[0] * 0.001f));
    printf("Atomic Version Performance: %.2f GFLOPS\n", gflops / (milliseconds[1] * 0.001f));
    printf("Privatization Version Performance: %.2f GFLOPS\n", gflops / (milliseconds[2] * 0.001f));
    printf("Coarsened 1 Performance: %.2f GFLOPS\n", gflops / (milliseconds[3] * 0.001f));
    printf("Coarsened 2 Performance: %.2f GFLOPS\n", gflops / (milliseconds[4] * 0.001f));
    printf("Aggregation Version Performance: %.2f GFLOPS\n", gflops / (milliseconds[5] * 0.001f));

    printf("\n=== Result Verification ===\n");
    printf("CPU Total: %lld\n", total_cpu);
    printf("Base Version Total: %lld\n", total_base);
    printf("Atomic Version Total: %lld\n", total_atomic);
    printf("Privatization Version Total: %lld\n", total_private);
    printf("Coarsened 1 Total: %lld\n", total_con);
    printf("Coarsened 2 Total: %lld\n", total_int);
    printf("Aggregation Version Total: %lld\n", total_agg);
    printf("Expected Total: %lld\n", vecSize);


    free(h_hist_base);
    free(h_hist_atomic);
    free(h_hist_private);
    CHECK_CUDA(cudaFree(d_hist_base));
    CHECK_CUDA(cudaFree(d_hist_atomic));
    CHECK_CUDA(cudaFree(d_hist_private));
    for(int i = 0; i < 6; i++) {
        CHECK_CUDA(cudaEventDestroy(start[i]));
        CHECK_CUDA(cudaEventDestroy(stop[i]));
    }


    free(h_hist_con);
    free(h_hist_int);
    free(h_hist_agg);
    CHECK_CUDA(cudaFree(d_hist_con));
    CHECK_CUDA(cudaFree(d_hist_int));
    CHECK_CUDA(cudaFree(d_hist_agg));

    free(h_hist_cpu);
    free(h_input);
    free(h_hist);
    CHECK_CUDA(cudaFree(d_input));
    return 0;
}
