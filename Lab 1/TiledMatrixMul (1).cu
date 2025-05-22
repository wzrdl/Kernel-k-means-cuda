#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cublas_v2.h>

void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Kernel 1: Naive matrix multiplication (non-coalesced memory access)
__global__ void matMulNaive(const float* A, const float* B, float* C, 
    int M, int N, int K) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;  
    int col = blockIdx.y * blockDim.y + threadIdx.y;  
    if (row >= M || col >= N) return;  

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a_val = A[row * K + k];   
        float b_val = B[k * N + col];   
        sum += a_val * b_val;
    }
    C[row * N + col] = sum;
}

// Kernel 2: Naive multiplication with coalesced global memory access
__global__ void matMulCoalesced(const float* A, const float* B, float* C, 
    int M, int N, int K) {
    // Compute global row and column index (swapped indexing for coalescing)
    int col = blockIdx.x * blockDim.x + threadIdx.x;  
    int row = blockIdx.y * blockDim.y + threadIdx.y;  
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a_val = A[row * K + k];   
        float b_val = B[k * N + col];   
        sum += a_val * b_val;
    }
    C[row * N + col] = sum;
}

// Kernel 3: Tiled matrix multiplication using shared memory (2D block tiling)
#define TILE_SIZE 32  
__global__ void matMulTiled(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    __shared__ float AsTile[TILE_SIZE][TILE_SIZE+1];  // +1 to prevent bank conflicts
    __shared__ float BsTile[TILE_SIZE][TILE_SIZE+1];
    
    int row0 = blockIdx.y * TILE_SIZE;
    int col0 = blockIdx.x * TILE_SIZE;
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    int global_row = row0 + ty;
    int global_col = col0 + tx;
    
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load A's tile [global_row, k] and B's tile [k, global_col] from this segment
        int tiled_k = t * TILE_SIZE;
        if (global_row < M && (tiled_k + tx) < K) { // bounds check
            AsTile[ty][tx] = A[global_row * K + (tiled_k + tx)];
        } else {
            AsTile[ty][tx] = 0.0f;
        }
        if ((tiled_k + ty) < K && global_col < N) { // bounds check
            BsTile[ty][tx] = B[(tiled_k + ty) * N + global_col];
        } else {
            BsTile[ty][tx] = 0.0f;
        }
        __syncthreads();  
        
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += AsTile[ty][k] * BsTile[k][tx];
        }
        __syncthreads();  
    }
    // Write the result to global memory
    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] = sum;
    }
}


// Kernel 4: Warp-level tiled optimization using shuffle operations
#define WARPS_PER_BLOCK 8    // number of warps per block
#define BLOCK_COLS 32        // block tile width must be 32 for warps
#define BLOCK_ROWS (WARPS_PER_BLOCK) 
__global__ void matMulWarpTiled(const float* A, const float* B, float* C,
                                int M, int N, int K) {
    
    int warpId = threadIdx.y;                
    int lane   = threadIdx.x;                
    
    int row = blockIdx.y * BLOCK_ROWS + warpId;
    int col = blockIdx.x * BLOCK_COLS + lane;
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    // Each warp collaboratively loads one 32-length segment of A and B
    for (int k0 = 0; k0 < K; k0 += 32) {
        float a_val = 0.0f;
        if (lane == 0 && (k0 + 0) < K) {
            // One thread (lane 0 of the warp) loads A for the current row
            a_val = A[row * K + (k0 + 0)];
        }
        // Broadcast A's value to all lanes in the warp
        a_val = __shfl_sync(0xFFFFFFFF, a_val, 0);
        float b_val = 0.0f;
        if ((k0 + lane) < K) {
            b_val = B[(k0 + lane) * N + col];
        }
        // Accumulate the products for this 32-length segment
        #pragma unroll
        for (int k_offset = 0; k_offset < 32; ++k_offset) {
            float a_shared = __shfl_sync(0xFFFFFFFF, a_val, 0);
            sum += a_shared * b_val;
        }
    }
    C[row * N + col] = sum;
}


// Kernel 5: 1D thread tiling (each thread computes a vector of outputs in one row)
#define TILE_M 32        
#define TILE_N 32        
#define THREAD_TILE_N 4  // each thread computes 4 output elements along the row
__global__ void matMul1DTiling(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    __shared__ float AsTile[TILE_M][TILE_M+1];
    __shared__ float BsTile[TILE_M][TILE_N+1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int global_row = blockIdx.y * TILE_M + ty;
    int global_col_start = blockIdx.x * TILE_N + tx * THREAD_TILE_N;

    float sum[THREAD_TILE_N] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int t = 0; t < (K + TILE_M - 1) / TILE_M; ++t) {
        int tiled_k = t * TILE_M;
        // Load sub-tiles of A and B into shared memory (coalesced)
        if (global_row < M && (tiled_k + tx) < K) {
            AsTile[ty][tx] = A[global_row * K + (tiled_k + tx)];
        } else {
            AsTile[ty][tx] = 0.0f;
        }
        // Each thread loads multiple B values contiguously for its 4 outputs
        for (int i = 0; i < THREAD_TILE_N; ++i) {
            int load_col = tx * THREAD_TILE_N + i;
            if ((tiled_k + ty) < K && (global_col_start + i) < N) {
                BsTile[ty][load_col] = B[(tiled_k + ty) * N + (global_col_start + i)];
            } else {
                BsTile[ty][load_col] = 0.0f;
            }
        }
        __syncthreads();
        // Compute partial results for this tile segment
        #pragma unroll
        for (int k = 0; k < TILE_M; ++k) {
            float a_val = AsTile[ty][k];
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_N; ++i) {
                float b_val = BsTile[k][tx * THREAD_TILE_N + i];
                sum[i] += a_val * b_val;
            }
        }
        __syncthreads();
    }
    // Write to global memory
    if (global_row < M) {
        for (int i = 0; i < THREAD_TILE_N; ++i) {
            int global_col = global_col_start + i;
            if (global_col < N) {
                C[global_row * N + global_col] = sum[i];
            }
        }
    }
}


// Kernel 6: 2D thread tiling (each thread computes a 2x2 block of C)
#undef THREAD_TILE_M
#define THREAD_TILE_M 2  // each thread computes 2 outputs in the row dimension
#undef THREAD_TILE_N
#define THREAD_TILE_N 2  // and 2 outputs in the col dimension (total 4 outputs per thread)
__global__ void matMul2DTiling(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    __shared__ float AsTile[TILE_M][TILE_M+1];
    __shared__ float BsTile[TILE_M][TILE_N+1];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int global_row = blockIdx.y * TILE_M + ty * THREAD_TILE_M;
    int global_col = blockIdx.x * TILE_N + tx * THREAD_TILE_N;
    float sum00 = 0.0f, sum01 = 0.0f, sum10 = 0.0f, sum11 = 0.0f;
    for (int t = 0; t < (K + TILE_M - 1) / TILE_M; ++t) {
        int tiled_k = t * TILE_M;
        // Load A tile (two rows per thread) and B tile (two cols per thread) into shared memory
        for (int i = 0; i < THREAD_TILE_M; ++i) {  // load 2 rows of A
            int r = ty * THREAD_TILE_M + i;
            if ((global_row + i) < M && (tiled_k + tx) < K) {
                AsTile[r][tx] = A[(global_row + i) * K + (tiled_k + tx)];
            } else {
                AsTile[r][tx] = 0.0f;
            }
        }
        for (int j = 0; j < THREAD_TILE_N; ++j) {  // load 2 cols of B
            int c = tx * THREAD_TILE_N + j;
            if ((tiled_k + ty) < K && (global_col + j) < N) {
                BsTile[ty][c] = B[(tiled_k + ty) * N + (global_col + j)];
            } else {
                BsTile[ty][c] = 0.0f;
            }
        }
        __syncthreads();
        // Compute using the shared tiles
        #pragma unroll
        for (int k = 0; k < TILE_M; ++k) {
            float a0 = AsTile[ty * THREAD_TILE_M + 0][k];
            float a1 = AsTile[ty * THREAD_TILE_M + 1][k];
            float b0 = BsTile[k][tx * THREAD_TILE_N + 0];
            float b1 = BsTile[k][tx * THREAD_TILE_N + 1];
            // Update the 2x2 block sums
            sum00 += a0 * b0;
            sum01 += a0 * b1;
            sum10 += a1 * b0;
            sum11 += a1 * b1;
        }
        __syncthreads();
    }
    // Write the 2x2 results to global memory
    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] = sum00;
    }
    if (global_row < M && (global_col + 1) < N) {
        C[global_row * N + (global_col + 1)] = sum01;
    }
    if ((global_row + 1) < M && global_col < N) {
        C[(global_row + 1) * N + global_col] = sum10;
    }
    if ((global_row + 1) < M && (global_col + 1) < N) {
        C[(global_row + 1) * N + (global_col + 1)] = sum11;
    }
}


int main(int argc, char** argv) {
    int M, K, N;
    if (argc == 5 && strcmp(argv[1], "-i") == 0) {
        M = atoi(argv[2]);
        K = atoi(argv[3]);
        N = atoi(argv[4]);
    } else {
        fprintf(stderr, "Usage: %s -i <rowDimA> <colDimA> <colDimB>\n", argv[0]);
        return 1;
    }
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);

    // Initialize matrices
    for (int i = 0; i < M*K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int j = 0; j < K*N; ++j) h_B[j] = static_cast<float>(rand()) / RAND_MAX;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // Copy inputs to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    
    // Define execution configurations for each kernel
    dim3 block_naive(16, 16);  
    dim3 grid_naive((N + block_naive.x - 1) / block_naive.x,
                    (M + block_naive.y - 1) / block_naive.y);
    dim3 block_coal(16, 16);   
    dim3 grid_coal((N + block_coal.x - 1) / block_coal.x,
                   (M + block_coal.y - 1) / block_coal.y);
    dim3 block_tiled(TILE_N, TILE_M); 
    dim3 grid_tiled((N + TILE_N - 1) / TILE_N,
                    (M + TILE_M - 1) / TILE_M);
   
    dim3 block_warp(32, WARPS_PER_BLOCK);
    dim3 grid_warp((N + BLOCK_COLS - 1) / BLOCK_COLS,
                   (M + BLOCK_ROWS - 1) / BLOCK_ROWS);

    dim3 block_1d(TILE_N / 4, TILE_M);
    dim3 grid_1d((N + TILE_N - 1) / TILE_N,
                 (M + TILE_M - 1) / TILE_M);

    dim3 block_2d(TILE_N / 2, TILE_M / 2);
    dim3 grid_2d((N + TILE_N - 1) / TILE_N,
                 (M + TILE_M - 1) / TILE_M);
    

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 1. Naive
    cudaEventRecord(start);
    matMulNaive<<<grid_naive, block_naive>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_ms = 0;
    cudaEventElapsedTime(&time_ms, start, stop);
    double time_s = time_ms / 1000.0;
    double tflops_naive = (2.0 * M * N * K) / (time_s * 1e12);
    printf("Naive: %.6f ms, %.3f TFLOPS\n", time_ms, tflops_naive);
    
    // 2. Coalesced
    cudaEventRecord(start);
    matMulCoalesced<<<grid_coal, block_coal>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    time_s = time_ms / 1000.0;
    double tflops_coal = (2.0 * M * N * K) / (time_s * 1e12);
    printf("Coalesced: %.6f ms, %.3f TFLOPS\n", time_ms, tflops_coal);
    
    // 3. Tiled (shared memory)
    cudaEventRecord(start);
    matMulTiled<<<grid_tiled, block_tiled>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    time_s = time_ms / 1000.0;
    double tflops_tiled = (2.0 * M * N * K) / (time_s * 1e12);
    printf("Tiled (Shared Memory): %.6f ms, %.3f TFLOPS\n", time_ms, tflops_tiled);
    
    // 4. Warp-tiled 
    cudaEventRecord(start);
    matMulWarpTiled<<<grid_warp, block_warp>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    time_s = time_ms / 1000.0;
    double tflops_warp = (2.0 * M * N * K) / (time_s * 1e12);
    //printf("%.3f\n", time_s);
    printf("Warp-Optimized: %.6f ms, %.3f TFLOPS\n", time_ms, tflops_warp);
    
    // 5. 1D thread tiling
    cudaEventRecord(start);
    matMul1DTiling<<<grid_1d, block_1d>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    time_s = time_ms / 1000.0;
    double tflops_1d = (2.0 * M * N * K) / (time_s * 1e12);
    //printf("%.3f\n", time_s);
    printf("1D Thread Tiling: %.6f ms, %.3f TFLOPS\n", time_ms, tflops_1d);
    
    // 6. 2D thread tiling
    cudaEventRecord(start);
    matMul2DTiling<<<grid_2d, block_2d>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    time_s = time_ms / 1000.0;
    double tflops_2d = (2.0 * M * N * K) / (time_s * 1e12);
    //printf("%.3f\n", time_s);
    printf("2D Thread Tiling: %.6f ms, %.3f TFLOPS\n", time_ms, tflops_2d);

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                &alpha,
                d_A, N,    
                d_B, K,
                &beta,
                d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    time_s = time_ms / 1000.0;
    double tflops_cuBLAS = (2.0 * M * N * K) / (time_s * 1e12);
    //printf("%.3f\n", time_s);
    printf("CUBLAS: %.6f ms, %.3f TFLOPS\n", time_ms, tflops_cuBLAS);
    
    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
