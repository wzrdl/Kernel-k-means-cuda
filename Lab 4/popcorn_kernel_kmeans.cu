
#include <iostream>                          
#include <vector>                            
#include <random>                            
#include <cuda_runtime.h>                   
#include <cublas_v2.h>                       
#include <cusparse_v2.h>                     
#include <cfloat>                            
#include <cusparse.h>                        
#include <chrono>                            
#include <cmath>                             
#include <fstream>                           
#include <thread>                            

// ---------- Error checking macros ----------
#define CUDA_CHECK(call) do {                                         \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__  \
                  << ": " << cudaGetErrorString(err) << std::endl;    \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while(0)


#define CUBLAS_CHECK(call) do {                                       \
    cublasStatus_t status = call;                                     \
    if (status != CUBLAS_STATUS_SUCCESS) {                            \
        std::cerr << "cuBLAS error in " << __FILE__ << ":" << __LINE__\
                  << std::endl;                                      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while(0)

#define CUSPARSE_CHECK(call) do {                                       \
    cusparseStatus_t s = call;                                          \
        if (s != CUSPARSE_STATUS_SUCCESS){                              \
            std::cerr<< "cuSPARSE error(" <<__LINE__<< ")" << std::endl;\
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while(0)

// Time measurement utility function
double getCurrentTime() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count() / 1000.0;
}

// ---------- CUDA kernel: Transform dot product matrix B to Gaussian kernel matrix K ----------
__global__ void gaussianKernelTransform(const float* B,  // Input: dot product matrix B
                                        float* K,        // Output: kernel matrix K
                                        const float* norms, // Each sample ||x||^2
                                        int N,            // Number of samples
                                        float sigma)      // σ parameter
{
    const int TILE_DIM = 16;
    int i = blockIdx.x * TILE_DIM + threadIdx.x;   
    int j = blockIdx.y * TILE_DIM + threadIdx.y;   
    if(i < N && j < N) {
        int idx = i + j * N; // Column-major storage
        float dist2 = norms[i] + norms[j] - 2.0f * B[idx];
        K[idx] = expf(-dist2 / (2.0f * sigma * sigma));
    }
}

// ---------- CUDA kernel: Transform dot product matrix B to polynomial kernel matrix K ----------
__global__ void polynomialKernelTransform(const float* B, // Input B
                                          float* K,       // Output K
                                          int N,          // Number of samples
                                          float c,        // Constant term c
                                          int d)          // Polynomial degree d
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx < N * N) {                                       
        K[idx] = powf(B[idx] + c, d);                        
    }
}

// ---------- CUDA kernel: Extract z vector based on assignments ----------
__global__ void gatherZ(const float* F,          // Input F = -2 * V * K (K×N, column-major)
                        const int* assign,       // Current sample cluster assignment array
                        float* z,                // Output vector z (length N)
                        int N, int K)            // N samples, K clusters
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; // Sample index i
    if (i < N) {
        int c = assign[i];                                  // Get the cluster c the sample belongs to
        float E_ic = F[i * K + c];                          // Get F(c, i) (column-major)
        z[i] = -0.5f * E_ic;                                // Multiply by -0.5 to get the term in the formula
    }
}

// ---------- CUDA kernel: Compute distances and reassign clusters ----------
__global__ void assignClusters(const float* Kdiag,   // K diagonal (N)
                               const float* F,       // F = E (K×N)
                               const float* cNorm,   // Vector clusterNorm (K)
                               int* newAssign,       // Output new cluster assignment
                               int N, int K)         // Dimensions
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < N) {
        float minVal = FLT_MAX;                            // Current minimum distance
        int best = -1;                                     // Best cluster
        for (int c = 0; c < K; ++c) {                      // Traverse all clusters
            float dist = Kdiag[i] + F[i * K + c] + cNorm[c]; // Formula D(i,c) = Kii + E + ||c||^2
            if (dist < minVal) { minVal = dist; best = c; } // Update optimal
        }
        newAssign[i] = best;                               
    }
}

// New: CUDA kernel to compare assignment arrays on GPU
__global__ void compareAssignKernel(const int* a, const int* b, int N, int* flag) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N) {
        if(a[tid] != b[tid])
            atomicExch(flag, 1);
    }
}

// CPU version of Gaussian kernel matrix computation
void cpuGaussianKernel(const float* X, float* K, int N, int D, float sigma) {
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            float dist = 0;
            for(int d = 0; d < D; d++) {
                float diff = X[i + d * N] - X[j + d * N];  // Column-major access
                dist += diff * diff;
            }
            K[i + j * N] = expf(-dist / (2.0f * sigma * sigma));
        }
    }
}

// CPU version of Kernel K-means iteration
void cpuKernelKMeans(const float* K, int* assign, int N, int numClusters, int maxIter) {
    std::vector<int> clusterSize(numClusters);
    
    for(int iter = 0; iter < maxIter; iter++) {
        // Compute the size of each cluster
        std::fill(clusterSize.begin(), clusterSize.end(), 0);
        for(int i = 0; i < N; i++) {
            clusterSize[assign[i]]++;
        }
        
        // For each point, compute the kernel distance to each cluster and reassign
        bool changed = false;
        for(int i = 0; i < N; i++) {
            float minDist = FLT_MAX;
            int bestC = assign[i];
            
            // Compute distance to each cluster
            for(int c = 0; c < numClusters; c++) {
                float dist = 0;
                
                // Compute ||φ(xi) - μc||^2
                // = K(xi,xi) - 2/|Cc|∑{xj∈Cc}K(xi,xj) + 1/|Cc|^2∑{xj,xk∈Cc}K(xj,xk)
                float term1 = K[i * N + i];  // K(xi,xi)
                
                float term2 = 0;  // 2/|Cc|∑K(xi,xj)
                float term3 = 0;  // 1/|Cc|^2∑K(xj,xk)
                
                if(clusterSize[c] > 0) {
                    // Compute term2
                    for(int j = 0; j < N; j++) {
                        if(assign[j] == c) {
                            term2 += K[i + j * N];
                        }
                    }
                    term2 = (2.0f * term2) / clusterSize[c];
                    
                    // Compute term3
                    for(int j = 0; j < N; j++) {
                        if(assign[j] == c) {
                            for(int k = 0; k < N; k++) {
                                if(assign[k] == c) {
                                    term3 += K[j + k * N];
                                }
                            }
                        }
                    }
                    term3 /= (float)(clusterSize[c] * clusterSize[c]);
                    
                    dist = term1 - term2 + term3;
                } else {
                    dist = FLT_MAX;  // Empty cluster
                }
                
                if(dist < minDist) {
                    minDist = dist;
                    bestC = c;
                }
            }
            
            if(bestC != assign[i]) {
                assign[i] = bestC;
                changed = true;
            }
        }
        
        if(!changed) {
            std::cout << "CPU version converged at iteration " << iter << std::endl;
            break;
        }
    }
}

// CPU version of sparse matrix-dense matrix multiplication (SpMM)
void cpuSpMM(const std::vector<int>& rowPtr, const std::vector<int>& colInd, 
             const std::vector<float>& values, const std::vector<float>& denseMatrix,
             std::vector<float>& result, int M, int N, int K, float alpha) {
    // M: Number of rows in sparse matrix, N: Number of columns in dense matrix, K: Number of rows in dense matrix
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            float sum = 0.0f;
            for(int p = rowPtr[i]; p < rowPtr[i+1]; p++) {
                int k = colInd[p];
                sum += values[p] * denseMatrix[k + j * K];  // Column-major order
            }
            result[j * M + i] = alpha * sum;
        }
    }
}

// CPU version of sparse matrix-vector multiplication (SpMV)
void cpuSpMV(const std::vector<int>& rowPtr, const std::vector<int>& colInd,
             const std::vector<float>& values, const std::vector<float>& vector,
             std::vector<float>& result, int M, float alpha) {
    for(int i = 0; i < M; i++) {
        float sum = 0.0f;
        for(int p = rowPtr[i]; p < rowPtr[i+1]; p++) {
            sum += values[p] * vector[colInd[p]];
        }
        result[i] = alpha * sum;
    }
}

// CPU version of Gaussian kernel matrix computation
void cpuGaussianTransform(const float* B, float* K, const float* norms, int N, float sigma) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            float dist2 = norms[i] + norms[j] - 2.0f * B[i + j * N];
            K[i + j * N] = expf(-dist2 / (2.0f * sigma * sigma));
        }
    }
}

// CPU version of z vector computation
void cpuGatherZ(const float* F, const int* assign, float* z, int N, int K) {
    for(int i = 0; i < N; i++) {
        int c = assign[i];
        float E_ic = F[i * K + c];  // Column-major order
        z[i] = -0.5f * E_ic;
    }
}

// CPU version of cluster assignment computation
void cpuAssignClusters(const float* Kdiag, const float* F, const float* cNorm, 
                      int* newAssign, int N, int K) {
    for(int i = 0; i < N; i++) {
        float minVal = FLT_MAX;
        int best = -1;
        for(int c = 0; c < K; c++) {
            float dist = Kdiag[i] + F[i + c * N] + cNorm[c];  // Column-major order
            if(dist < minVal) {
                minVal = dist;
                best = c;
            }
        }
        newAssign[i] = best;
    }
}

// Compute maximum relative error
float getMaxRelativeError(const std::vector<float>& cpu, const std::vector<float>& gpu, int size) {
    float maxError = 0.0f;
    for(int i = 0; i < size; i++) {
        float rel_error = fabs(cpu[i] - gpu[i]) / (fabs(cpu[i]) + 1e-6f);
        maxError = std::max(maxError, rel_error);
    }
    return maxError;
}

// Add helper function to save matrix to file
void saveMatrixToFile(const std::vector<float>& matrix, int rows, int cols, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }
    file << std::scientific;  // Use scientific notation
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            file << matrix[j * rows + i] << " ";  // Column-major order
        }
        file << "\n";
    }
    file.close();
}

// Add helper function to convert row-major to column-major order
void convertRowToColMajor(const std::vector<float>& rowMajor, std::vector<float>& colMajor, int rows, int cols) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            colMajor[j * rows + i] = rowMajor[i * cols + j];
        }
    }
}


__global__ void buildCSR(int* rowPtr,int* colInd,float* val,
    const int* assign,int N,int K)
{
    // Each thread processes one sample
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i>=N) return;
    int c = assign[i];

    // 1) Use atomic addition to count the size of each cluster
    __shared__ extern int shCount[]; // len=K
    if(threadIdx.x<K) shCount[threadIdx.x]=0;
    __syncthreads();
    atomicAdd(&shCount[c],1);
    __syncthreads();

    // 2) Thread 0 sorts and writes rowPtr
    if(threadIdx.x==0){
        int prefix = 0;
        for(int k=0;k<K;++k){
            rowPtr[k] = prefix;
            prefix += shCount[k];
        }
        rowPtr[K] = prefix;          // nnz == N
    }
    __syncthreads();

    // 3) Atomic addition again to locate colInd/val index
    int pos = atomicAdd(&rowPtr[c],1); // rowPtr has been reused
    colInd[pos] = i;
    val[pos]    = 1.0f / (float)shCount[c];
}

// Add test function
void runTest(int N, int K, int D, std::ofstream& log_file) {
    // Compute the number of floating-point operations per iteration
    double ops_per_iter = 
        2.0 * N * N * D +              // GEMM for B = X * X^T
        N * N +                        // Gaussian kernel transform
        2.0 * N * K +                 // SpMM operation
        N * K +                       // gatherZ
        N * K +                       // assignClusters
        N * K;                        // Other operations

    const int NUM_RUNS = (N == 2000) ? 1 : 5;  // Run only once for 2000 samples

    // Data generation part
    std::random_device rd;
    std::mt19937 gen_cpu(rd()), gen_gpu(rd()); // Use different random number generators

    std::vector<float> hX_row(N * D);
    int half = N / 2;
    float radius1 = 1.0f, radius2 = 3.0f;
    std::uniform_real_distribution<float> angleDist(0, 2 * M_PI);
    std::normal_distribution<float> noise(0.f, 0.1f);
    
    for (int i = 0; i < half; i++) {
        float angle = angleDist(gen_cpu);
        hX_row[i * D + 0] = radius1 * cos(angle) + noise(gen_cpu);
        hX_row[i * D + 1] = radius1 * sin(angle) + noise(gen_cpu);
    }
    for (int i = half; i < N; i++) {
        float angle = angleDist(gen_cpu);
        hX_row[i * D + 0] = radius2 * cos(angle) + noise(gen_cpu);
        hX_row[i * D + 1] = radius2 * sin(angle) + noise(gen_cpu);
    }

    std::vector<float> hX(N * D);
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < D; j++) {
            hX[j * N + i] = hX_row[i * D + j];
        }
    }

    std::vector<double> gpu_times(NUM_RUNS);
    std::vector<int> gpu_iters(NUM_RUNS);  // Record GPU iteration count

    for(int run = 0; run < NUM_RUNS; run++) {
        // GPU test
        cudaDeviceSynchronize();


        // GPU code part
        {
            float *dX = nullptr;                              // Device data pointer
            CUDA_CHECK(cudaMalloc(&dX, N * D * sizeof(float)));// Allocate memory for X
            CUDA_CHECK(cudaMemcpy(dX, hX.data(), N * D * sizeof(float), cudaMemcpyHostToDevice)); // Copy

            float *dB = nullptr, *dK = nullptr;               // Dot product matrix B and kernel matrix K
            CUDA_CHECK(cudaMalloc(&dB, N * N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dK, N * N * sizeof(float)));

            cublasHandle_t cublasH;                           // cuBLAS handle
            CUBLAS_CHECK(cublasCreate(&cublasH));            // Create

            const float alpha = 1.f, beta = 0.f;              // SGEMM coefficients
            CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
                                      N, N, D, &alpha,        // B = X * X^T
                                      dX, N, dX, N, &beta, dB, N));

            float *dNorms = nullptr; CUDA_CHECK(cudaMalloc(&dNorms, N*sizeof(float)));        // Allocate
            CUBLAS_CHECK(cublasScopy(cublasH, N, dB, N+1, dNorms, 1));                        // Copy diagonal

            dim3 blockDim(16, 16);
            dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
            gaussianKernelTransform<<<gridDim, blockDim>>>(dB, dK, dNorms, N, 1.0f);         // Transform to Gaussian kernel
            CUDA_CHECK(cudaDeviceSynchronize());                                               // Wait for GPU
            CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dNorms));                            // Free unused memory

            float *dKdiag = nullptr; CUDA_CHECK(cudaMalloc(&dKdiag, N*sizeof(float)));         // Diagonal array
            CUBLAS_CHECK(cublasScopy(cublasH, N, dK, N+1, dKdiag, 1));                         // Copy diagonal

            std::vector<int> hAssign(N); 
            std::uniform_int_distribution<int> ud_gpu(0,K-1);                                      // Random cluster
            for (auto &v:hAssign) v = ud_gpu(gen_gpu);                                                  // Fill
            int *dAssign=nullptr, *dNewAssign=nullptr;                                          // Device arrays
            CUDA_CHECK(cudaMalloc(&dAssign, N*sizeof(int)));
            CUDA_CHECK(cudaMalloc(&dNewAssign, N*sizeof(int)));
            CUDA_CHECK(cudaMemcpy(dAssign, hAssign.data(), N*sizeof(int), cudaMemcpyHostToDevice));

            std::vector<int>   hRowPtr(K+1);                      // CSR row pointer
            std::vector<int>   hColInd(N);                        // Column indices (up to N)
            std::vector<float> hVal(N);                           // Non-zero values

            auto buildCSR = [&](std::vector<int>& size){          // lambda: rebuild CSR
                int nz=0; hRowPtr[0]=0;                           // nnz start
                for(int c=0;c<K;++c){                             // Traverse clusters
                    for(int i=0;i<N;++i){                         // Traverse samples
                        if(hAssign[i]==c){                        // Belongs to the cluster
                            hColInd[nz]=i;                       // Write column index
                            hVal[nz] = size[c] ? 1.0f/size[c] : 0.0f;              // Value=1/Nc
                            ++nz;                                 // nnz++
                        }
                    }
                    hRowPtr[c+1]=nz;                              // Row pointer
                }
                return nz;                                        // Return nnz
            };

            std::vector<int> hSize(K,0); for(int i=0;i<N;++i) ++hSize[hAssign[i]];  // Compute initial cluster size
            int nnz = buildCSR(hSize);                                             // Build CSR

            int *dRowPtr=nullptr,*dColInd=nullptr; float *dVal=nullptr;            // GPU CSR
            CUDA_CHECK(cudaMalloc(&dRowPtr,(K+1)*sizeof(int)));
            CUDA_CHECK(cudaMalloc(&dColInd,nnz*sizeof(int)));
            CUDA_CHECK(cudaMalloc(&dVal,   nnz*sizeof(float)));
            CUDA_CHECK(cudaMemcpy(dRowPtr,hRowPtr.data(),(K+1)*sizeof(int),cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dColInd,hColInd.data(),nnz*sizeof(int),cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dVal,   hVal.data(),   nnz*sizeof(float),cudaMemcpyHostToDevice));

            cusparseHandle_t spH; CUSPARSE_CHECK(cusparseCreate(&spH));           // cuSPARSE handle

            cusparseSpMatDescr_t matV;                                            // Sparse matrix V
            CUSPARSE_CHECK(cusparseCreateCsr(&matV, K, N, nnz,                    // rows, cols, nnz
                                              dRowPtr,dColInd,dVal,               // CSR three arrays
                                              CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,
                                              CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

            cusparseDnMatDescr_t matK, matF;                                      // Dense matrices K / F
            CUSPARSE_CHECK(cusparseCreateDnMat(&matK, N, N, N, dK, CUDA_R_32F, CUSPARSE_ORDER_COL)); // N×N

            float *dF=nullptr; CUDA_CHECK(cudaMalloc(&dF, K*N*sizeof(float)));    // F (K×N)
            CUSPARSE_CHECK(cusparseCreateDnMat(&matF, K, N, K, dF, CUDA_R_32F, CUSPARSE_ORDER_COL));

            size_t bufSz=0; const float alphaSp=-2.f,betaSp=0.f;                   // α=-2, β=0
            CUSPARSE_CHECK(cusparseSpMM_bufferSize(spH,                              // Query size
                CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alphaSp, matV, matK, &betaSp, matF, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufSz));
            void* dBuf=nullptr; CUDA_CHECK(cudaMalloc(&dBuf, bufSz));              // Allocate buffer

            float *dZ=nullptr, *dCNorm=nullptr;                                     // z, ||c||^2 vectors
            CUDA_CHECK(cudaMalloc(&dZ, N*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dCNorm, K*sizeof(float)));

            cusparseDnVecDescr_t vecZ, vecCNorm;
            CUSPARSE_CHECK(cusparseCreateDnVec(&vecZ, N, dZ, CUDA_R_32F));
            CUSPARSE_CHECK(cusparseCreateDnVec(&vecCNorm, K, dCNorm, CUDA_R_32F));

            const float one = 1.f, zero = 0.f;
            size_t spmvBufSz = 0;
            CUSPARSE_CHECK(cusparseSpMV_bufferSize(spH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &one, matV, vecZ, &zero, vecCNorm, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, &spmvBufSz));
            void* dSpmvBuf = nullptr;
            CUDA_CHECK(cudaMalloc(&dSpmvBuf, spmvBufSz));

            int *dConvFlag = nullptr; // Device convergence flag
            CUDA_CHECK(cudaMalloc(&dConvFlag, sizeof(int)));

            // Record the start time of core computation
            cudaDeviceSynchronize();
            double gpuComputeStart = getCurrentTime();

            for(int it=0; it<1000; ++it){
                CUSPARSE_CHECK(cusparseSpMM(spH, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alphaSp, matV, matK, &betaSp, matF, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuf));

                int blks = (N+127)/128; 
                gatherZ<<<blks,128>>>(dF, dAssign, dZ, N, K);
                
                cusparseSpMV(spH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &one, matV, vecZ, &zero, vecCNorm, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, dSpmvBuf);
                
                
                blks = (N+255)/256; 
                assignClusters<<<blks,256>>>(dKdiag, dF, dCNorm, dNewAssign, N, K);
                
                       
                CUDA_CHECK(cudaMemset(dConvFlag, 0, sizeof(int)));
                blks = (N + 255) / 256;
                compareAssignKernel<<<blks,256>>>(dAssign, dNewAssign, N, dConvFlag);
                int hConvFlag = 0;
                CUDA_CHECK(cudaMemcpy(&hConvFlag, dConvFlag, sizeof(int), cudaMemcpyDeviceToHost)); // Only copy flag
                
                CUDA_CHECK(cudaMemcpy(dAssign, dNewAssign, N*sizeof(int), cudaMemcpyDeviceToDevice));
                
                CUDA_CHECK(cudaMemcpy(hAssign.data(), dAssign, N*sizeof(int), cudaMemcpyHostToHost));
                
                std::fill(hSize.begin(), hSize.end(), 0);
                for(int i = 0; i < N; ++i)
                    ++hSize[hAssign[i]]; // Use the latest hAssign to update size
                nnz = buildCSR(hSize); // Rebuild CSR host
                CUDA_CHECK(cudaMemcpy(dRowPtr, hRowPtr.data(), (K+1)*sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(dColInd, hColInd.data(), nnz*sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(dVal, hVal.data(), nnz*sizeof(float), cudaMemcpyHostToDevice));

                gpu_iters[run]++;
                if(hConvFlag == 0){ 
                    break; 
                }
            }

            // Ensure all GPU operations are complete
            cudaDeviceSynchronize();
            double gpuEndTime = getCurrentTime();
            gpu_times[run] = gpuEndTime - gpuComputeStart; // Only record computation time

            // Record results
            log_file << "\nPerformance analysis for run " << run + 1 << " (N=" << N << ", K=" << K << "):\n";
            log_file << "Number of iterations: " << gpu_iters[run] << std::endl;
            log_file << "Execution time: " << gpu_times[run] << " seconds" << std::endl;
            double total_ops = ops_per_iter * gpu_iters[run];
            double gpu_gflops = (total_ops / 1e9) / gpu_times[run];
            log_file << "GPU performance: " << gpu_gflops << " GFLOPS\n";

            CUDA_CHECK(cudaFree(dConvFlag));
            CUDA_CHECK(cudaFree(dSpmvBuf));
            CUDA_CHECK(cudaFree(dBuf)); CUDA_CHECK(cudaFree(dF)); CUDA_CHECK(cudaFree(dZ)); CUDA_CHECK(cudaFree(dCNorm));
            CUDA_CHECK(cudaFree(dRowPtr)); CUDA_CHECK(cudaFree(dColInd)); CUDA_CHECK(cudaFree(dVal));
            CUDA_CHECK(cudaFree(dAssign)); CUDA_CHECK(cudaFree(dNewAssign));
            CUDA_CHECK(cudaFree(dKdiag)); CUDA_CHECK(cudaFree(dX));
            CUSPARSE_CHECK(cusparseDestroySpMat(matV)); CUSPARSE_CHECK(cusparseDestroyDnMat(matK)); CUSPARSE_CHECK(cusparseDestroyDnMat(matF));
            CUSPARSE_CHECK(cusparseDestroyDnVec(vecZ)); CUSPARSE_CHECK(cusparseDestroyDnVec(vecCNorm));
            CUSPARSE_CHECK(cusparseDestroy(spH)); CUBLAS_CHECK(cublasDestroy(cublasH));      // Destroy handles
        }

        // Wait for GPU to cool down after each run
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // Compute average performance
    double gpu_avg = 0.0;
    double avg_gflops = 0.0;
    for(int i = 0; i < NUM_RUNS; i++) {
        gpu_avg += gpu_times[i];
        double total_ops = ops_per_iter * gpu_iters[i];
        avg_gflops += (total_ops / 1e9) / gpu_times[i];
    }
    gpu_avg /= NUM_RUNS;
    avg_gflops /= NUM_RUNS;

    // Update output format
    log_file << "\nOverall performance (N=" << N << ", K=" << K << "):\n";
    log_file << "Average GPU time: " << gpu_avg << " seconds\n";
    log_file << "Average GPU performance: " << avg_gflops << " GFLOPS\n\n";

    std::cout << "\nOverall performance (N=" << N << ", K=" << K << "):\n";
    std::cout << "Average GPU time: " << gpu_avg << " seconds\n";
    std::cout << "Average GPU performance: " << avg_gflops << " GFLOPS\n\n";
}

int main() {
    // Create log file
    std::ofstream log_file("clustering_performance.txt");
    log_file << "Clustering performance test results:\n" << std::endl;
    log_file << "Configuration\tGPU Time (seconds)\tSpeedup\n";
    
    // Test parameters
    const int D = 2;  // Feature dimension
    const std::vector<int> sample_sizes = {100, 500, 1000, 2000, 5000, 10000};
    const std::vector<int> cluster_nums = {2, 5, 10};
    
    // Run tests for all combinations
    for(int N : sample_sizes) {
        for(int K : cluster_nums) {
            if(N == 2000) {
                std::cout << "\n==== Large-scale test N=2000, K=" << K << " ====" << std::endl;
            }
            std::cout << "\n-------------------------------------------";
            std::cout << "\nTest configuration: N=" << N << ", K=" << K << std::endl;
            log_file << "\nTest configuration: N=" << N << ", K=" << K << std::endl;
            
            runTest(N, K, D, log_file);
            
            // Wait for GPU to cool down
            cudaDeviceSynchronize();
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    
    log_file.close();
    std::cout << "\nAll tests completed, detailed results saved to clustering_performance.txt" << std::endl;
    
    return 0;
}
