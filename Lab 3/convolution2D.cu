#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>

#include <cuda_runtime.h>

using namespace std;

#define MAX_KERNEL_SIZE 32
#define BLOCK_SIZE 32

__constant__ float d_mask_const[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

void convolution2D_CPU(const float* input, const float* mask, float* output, 
                       int dimX, int dimY, int dimK) {
    int radius = dimK / 2;
    for (int y = 0; y < dimY; ++y) {
        for (int x = 0; x < dimX; ++x) {
            float sum = 0.0f;
            for (int j = 0; j < dimK; ++j) {
                for (int i = 0; i < dimK; ++i) {
                    int imgY = y + j - radius;
                    int imgX = x + i - radius;
                    if (imgY >= 0 && imgY < dimY && imgX >= 0 && imgX < dimX) {
                        sum += input[imgY * dimX + imgX] * mask[j * dimK + i];
                    }
                }
            }
            output[y * dimX + x] = sum;
        }
    }
}

__global__ void conv2D_basic_kernel(const float* d_input, const float* d_mask, 
    float* d_output, int dimX, int dimY, int dimK) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dimX || y >= dimY) return;  
    int radius = dimK / 2;
    float sum = 0.0f;
    
    for (int j = 0; j < dimK; ++j) {
        for (int i = 0; i < dimK; ++i) {
            int imgY = y + j - radius;
            int imgX = x + i - radius;
            
            if (imgY >= 0 && imgY < dimY && imgX >= 0 && imgX < dimX) {
                float image_val = d_input[imgY * dimX + imgX];
                float mask_val  = d_mask[j * dimK + i];
                sum += image_val * mask_val;
            }
        }
    }
    d_output[y * dimX + x] = sum;
}


__global__ void conv2D_tiled_kernel(const float * d_input, const float * d_mask,
    float* d_output, int dimY , int dimX, int dimK) {
    int channels = 1;
    int w = BLOCK_SIZE + dimK - 1;	//width of shared memory
    __shared__ float N_ds[MAX_KERNEL_SIZE + BLOCK_SIZE - 1][MAX_KERNEL_SIZE + BLOCK_SIZE - 1];	//block of share memory


   
    int maskRadius = dimK/2;
    for (int k = 0; k < channels; k++) {
        int dest = threadIdx.y * BLOCK_SIZE + threadIdx.x;
        int destY = dest / w;     //col of shared memory
        int destX = dest % w;		//row of shared memory
        int srcY = blockIdx.y *BLOCK_SIZE + destY - maskRadius;  //row index to fetch data from input image
        int srcX = blockIdx.x *BLOCK_SIZE + destX - maskRadius;	//col index to fetch data from input image
        if(srcY>= 0 && srcY < dimX && srcX>=0 && srcX < dimY)
            N_ds[destY][destX] = d_input[(srcY *dimY +srcX) * channels + k];
        else
            N_ds[destY][destX] = 0;


        dest = threadIdx.y * BLOCK_SIZE+ threadIdx.x + BLOCK_SIZE * BLOCK_SIZE;
        destY = dest / w;
        destX = dest % w;
        srcY = blockIdx.y *BLOCK_SIZE + destY - maskRadius;
        srcX = blockIdx.x *BLOCK_SIZE + destX - maskRadius;
        if(destY < w){
            if(srcY>= 0 && srcY < dimX && srcX>=0 && srcX < dimY)
                N_ds[destY][destX] = d_input[(srcY *dimY +srcX) * channels + k];
            else
                N_ds[destY][destX] = 0;
        }

        __syncthreads();


        
        float accum = 0;
        int y, x;
        for (y= 0; y < dimK; y++)
            for(x = 0; x < dimK; x++)
                accum += N_ds[threadIdx.y + y][threadIdx.x + x] *d_mask[y * dimK + x];

        y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
        x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
        if(y < dimX && x < dimY)
            d_output[(y * dimY + x) * channels + k] = accum;
        __syncthreads();
    }

}





float rand_val() {
    return static_cast<float>(rand() % 16);
}

int main(int argc, char** argv) {
    if (argc != 7 || strcmp(argv[1], "-i") || strcmp(argv[3], "-j") || strcmp(argv[5], "-k")) {
        cerr << "Usage: ./convolution2D -i <dimX> -j <dimY> -k <dimK>\n";
        return 1;
    }
    int dimX = atoi(argv[2]);
    int dimY = atoi(argv[4]);
    int dimK = atoi(argv[6]);

    int image_size = dimX * dimY;
    int mask_size = dimK * dimK;

    vector<float> h_input(image_size), h_mask(mask_size), h_output_cpu(image_size);
    vector<float> h_output_basic(image_size), h_output_tiled(image_size);

    for (int i = 0; i < image_size; ++i) h_input[i] = rand_val();
    for (int i = 0; i < mask_size; ++i) h_mask[i] = rand_val();

    auto start = chrono::high_resolution_clock::now();
    convolution2D_CPU(h_input.data(), h_mask.data(), h_output_cpu.data(), dimX, dimY, dimK);
    auto end = chrono::high_resolution_clock::now();
    double cpu_time = chrono::duration<double, milli>(end - start).count();

    float *d_input, *d_mask, *d_output;
    cudaMalloc(&d_input, image_size * sizeof(float));
    cudaMalloc(&d_mask, mask_size * sizeof(float));
    cudaMalloc(&d_output, image_size * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), image_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask.data(), mask_size * sizeof(float), cudaMemcpyHostToDevice);
  

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((dimX + BLOCK_SIZE - 1) / BLOCK_SIZE, (dimY + BLOCK_SIZE - 1) / BLOCK_SIZE);

   
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    
    cudaMemset(d_output, 0, image_size * sizeof(float));
    cudaEventRecord(start_event);
    conv2D_basic_kernel<<<dimGrid, dimBlock>>>(d_input, d_mask, d_output, dimX, dimY, dimK);
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    cudaMemcpy(h_output_basic.data(), d_output, image_size * sizeof(float), cudaMemcpyDeviceToHost);

    float gpu_time_basic;
    cudaEventElapsedTime(&gpu_time_basic, start_event, stop_event);

  
    double error_basic = 0.0;
    for (int i = 0; i < image_size; ++i) {
        error_basic += fabs(h_output_cpu[i] - h_output_basic[i]);
    }

    
    cudaMemset(d_output, 0, image_size * sizeof(float));
    cudaEventRecord(start_event);
    conv2D_tiled_kernel<<<dimGrid, dimBlock>>>(d_input, d_mask, d_output, dimX, dimY, dimK);
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    cudaMemcpy(h_output_tiled.data(), d_output, image_size * sizeof(float), cudaMemcpyDeviceToHost);

    float gpu_time_tiled;
    cudaEventElapsedTime(&gpu_time_tiled, start_event, stop_event);

    // Calculate error for tiled kernel
    double error_tiled = 0.0;
    for (int i = 0; i < image_size; ++i) {
        error_tiled += fabs(h_output_cpu[i] - h_output_tiled[i]);
    }

    double total_ops = 2.0 * dimX * dimY * dimK * dimK;
    double cpu_gflops = total_ops / (cpu_time * 1e6);
    double gpu_gflops_basic = total_ops / (gpu_time_basic * 1e6);
    double gpu_gflops_tiled = total_ops / (gpu_time_tiled * 1e6);

    cout << "CPU Time: " << cpu_time << " ms, GFLOPS: " << cpu_gflops << "\n";
    cout << "GPU Basic Time: " << gpu_time_basic << " ms, GFLOPS: " << gpu_gflops_basic << "\n";
    cout << "GPU Basic Error: " << error_basic << "\n";
    cout << "GPU Tiled Time: " << gpu_time_tiled << " ms, GFLOPS: " << gpu_gflops_tiled << "\n";
    cout << "GPU Tiled Error: " << error_tiled << "\n";

    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_output);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    return 0;
}
