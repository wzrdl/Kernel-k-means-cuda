#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matMulCPU(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void randomInit(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)rand() / RAND_MAX;
    }
}

int main(int argc, char **argv) {
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

    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_C = (float *)malloc(sizeC);

    srand(time(NULL));
    randomInit(h_A, M * K);
    randomInit(h_B, K * N);

    double flops = 2.0 * M * N * K;

    clock_t cpu_start = clock();
    matMulCPU(h_A, h_B, h_C, M, N, K);
    clock_t cpu_end = clock();
    float cpu_time_ms = 1000.0f * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    //double cpu_tflops = flops / (cpu_time_ms / 1e3) / 1e12;
    printf("CPU\t\t%8.2f ms\t\n", cpu_time_ms);

    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
