#include <stdio.h>
#include <cuda.h>

#define N (1 << 24)   // 약 16M elements

// =====================
// Kernel A: Coalesced
// =====================
__global__ void coalesced(float* A, float* B, float* C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// =========================
// Kernel B: Non-Coalesced
// =========================
__global__ void non_coalesced(float* A, float* B, float* C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = 32;                 // warp size
    int bad_idx = idx * stride;
    if (bad_idx < N) {
        C[bad_idx] = A[bad_idx] + B[bad_idx];
    }
}

int main() {
    float *A, *B, *C;
    float *dA, *dB, *dC;

    size_t size = N * sizeof(float);

    // Host memory
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // Device memory
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    float ms;

    // =====================
    // Coalesced timing
    // =====================
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    coalesced<<<blocks, threads>>>(dA, dB, dC);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);
    printf("Coalesced time: %.3f ms\n", ms);

    // =========================
    // Non-coalesced timing
    // =========================
    cudaEventRecord(start);
    non_coalesced<<<blocks, threads>>>(dA, dB, dC);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);
    printf("Non-coalesced time: %.3f ms\n", ms);

    // Cleanup
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(A);
    free(B);
    free(C);

    return 0;
}