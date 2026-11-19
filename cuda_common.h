#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

constexpr int BATCH = 256;
constexpr int INPUT_SIZE = 128;
constexpr int HIDDEN_SIZE = 64;
constexpr int OUTPUT_SIZE = 10;
constexpr int REPEATS = 20;

__global__ void matrixMulBiasKernel(const float*, const float*, const float*, float*, int, int, int);
__global__ void reluKernel(float*, int);
__global__ void sigmoidKernel(float*, int);

void cpuForwardPass(const std::vector<float>&, const std::vector<float>&,
                    const std::vector<float>&, const std::vector<float>&,
                    const std::vector<float>&, std::vector<float>&);
void launchGpuForwardPass(const float*, const float*, const float*, float*,
                          const float*, const float*, float*);
bool testMatrixKernel();
bool testReLUKernel();
bool testSigmoidKernel();

#endif
