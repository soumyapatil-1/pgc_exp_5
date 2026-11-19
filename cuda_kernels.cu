#include "cuda_common.h"

// Each CUDA thread calculates one output matrix element and adds its bias.
__global__ void matrixMulBiasKernel(
    const float* A, const float* B, const float* bias, float* C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum + bias[col];
    }
}

// Each thread applies ReLU to one element.
__global__ void reluKernel(float* values, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        values[idx] = fmaxf(0.0f, values[idx]);
    }
}

// Each thread applies sigmoid to one element.
__global__ void sigmoidKernel(float* values, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        values[idx] = 1.0f / (1.0f + expf(-values[idx]));
    }
}


// Launch the four GPU kernels in forward-pass order.
void launchGpuForwardPass(const float* dX, const float* dW1, const float* db1,
                          float* dZ1, const float* dW2, const float* db2,
                          float* dZ2)
{
    dim3 block(16, 16);

    dim3 grid1((HIDDEN_SIZE + block.x - 1) / block.x,
               (BATCH + block.y - 1) / block.y);
    matrixMulBiasKernel<<<grid1, block>>>(
        dX, dW1, db1, dZ1, BATCH, HIDDEN_SIZE, INPUT_SIZE);

    int hiddenCount = BATCH * HIDDEN_SIZE;
    int threads = 256;
    int blocks = (hiddenCount + threads - 1) / threads;
    reluKernel<<<blocks, threads>>>(dZ1, hiddenCount);

    dim3 grid2((OUTPUT_SIZE + block.x - 1) / block.x,
               (BATCH + block.y - 1) / block.y);
    matrixMulBiasKernel<<<grid2, block>>>(
        dZ1, dW2, db2, dZ2, BATCH, OUTPUT_SIZE, HIDDEN_SIZE);

    int outputCount = BATCH * OUTPUT_SIZE;
    int outputBlocks = (outputCount + threads - 1) / threads;
    sigmoidKernel<<<outputBlocks, threads>>>(dZ2, outputCount);

    CUDA_CHECK(cudaGetLastError());
}

bool nearlyEqual(float a, float b, float tolerance = 1e-5f)
{
    return std::fabs(a - b) <= tolerance;
}

