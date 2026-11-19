#include "cuda_common.h"

// CPU implementation of matrix multiplication with bias.
void cpuMatMulBias(const std::vector<float>& A,
                   const std::vector<float>& B,
                   const std::vector<float>& bias,
                   std::vector<float>& C,
                   int M, int N, int K)
{
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum + bias[col];
        }
    }
}

void cpuReLU(std::vector<float>& values)
{
    for (float& x : values) {
        x = std::max(0.0f, x);
    }
}

void cpuSigmoid(std::vector<float>& values)
{
    for (float& x : values) {
        x = 1.0f / (1.0f + std::exp(-x));
    }
}

void cpuForwardPass(const std::vector<float>& X,
                    const std::vector<float>& W1,
                    const std::vector<float>& b1,
                    const std::vector<float>& W2,
                    const std::vector<float>& b2,
                    std::vector<float>& output)
{
    std::vector<float> z1(BATCH * HIDDEN_SIZE);
    cpuMatMulBias(X, W1, b1, z1, BATCH, HIDDEN_SIZE, INPUT_SIZE);
    cpuReLU(z1);

    std::vector<float> z2(BATCH * OUTPUT_SIZE);
    cpuMatMulBias(z1, W2, b2, z2, BATCH, OUTPUT_SIZE, HIDDEN_SIZE);
    cpuSigmoid(z2);
    output = z2;
}

