#include "cuda_common.h"

int main()
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable NVIDIA GPU was detected." << std::endl;
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "EXPERIMENT 5: NEURAL NETWORK LAYERS IN CUDA\n";
    std::cout << "GPU detected: " << prop.name << "\n\n";

    std::cout << "Testing individual CUDA kernels...\n";
    bool matrixOK = testMatrixKernel();
    bool reluOK = testReLUKernel();
    bool sigmoidOK = testSigmoidKernel();

    std::cout << "Matrix multiplication + bias: " << (matrixOK ? "PASSED" : "FAILED") << "\n";
    std::cout << "ReLU: " << (reluOK ? "PASSED" : "FAILED") << "\n";
    std::cout << "Sigmoid: " << (sigmoidOK ? "PASSED" : "FAILED") << "\n";

    if (!(matrixOK && reluOK && sigmoidOK)) {
        std::cerr << "A kernel test failed. Stop and debug before continuing.\n";
        return EXIT_FAILURE;
    }

    // Deterministic input and weights: CPU and GPU receive identical values.
    std::vector<float> X(BATCH * INPUT_SIZE);
    std::vector<float> W1(INPUT_SIZE * HIDDEN_SIZE);
    std::vector<float> b1(HIDDEN_SIZE);
    std::vector<float> W2(HIDDEN_SIZE * OUTPUT_SIZE);
    std::vector<float> b2(OUTPUT_SIZE);

    for (size_t i = 0; i < X.size(); ++i)
        X[i] = 0.5f * std::sin(static_cast<float>(i) * 0.013f);
    for (size_t i = 0; i < W1.size(); ++i)
        W1[i] = 0.08f * std::cos(static_cast<float>(i) * 0.017f);
    for (size_t i = 0; i < b1.size(); ++i)
        b1[i] = 0.01f * static_cast<float>(i % 7);
    for (size_t i = 0; i < W2.size(); ++i)
        W2[i] = 0.10f * std::sin(static_cast<float>(i) * 0.019f);
    for (size_t i = 0; i < b2.size(); ++i)
        b2[i] = 0.02f * static_cast<float>(i % 5);

    std::vector<float> cpuOutput(BATCH * OUTPUT_SIZE);
    auto cpuStart = std::chrono::steady_clock::now();
    for (int r = 0; r < REPEATS; ++r) {
        cpuForwardPass(X, W1, b1, W2, b2, cpuOutput);
    }
    auto cpuStop = std::chrono::steady_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(
        cpuStop - cpuStart).count() / REPEATS;

    // Allocate GPU memory.
    float *dX = nullptr, *dW1 = nullptr, *db1 = nullptr, *dZ1 = nullptr;
    float *dW2 = nullptr, *db2 = nullptr, *dZ2 = nullptr;

    CUDA_CHECK(cudaMalloc(&dX, X.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW1, W1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db1, b1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dZ1, BATCH * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW2, W2.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db2, b2.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dZ2, BATCH * OUTPUT_SIZE * sizeof(float)));

    // Copy inputs and weights to the GPU.
    CUDA_CHECK(cudaMemcpy(dX, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW1, W1.data(), W1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db1, b1.data(), b1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW2, W2.data(), W2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db2, b2.data(), b2.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Warm-up.
    launchGpuForwardPass(dX, dW1, db1, dZ1, dW2, db2, dZ2);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Kernel-only timing: inputs and weights already reside on the GPU.
    cudaEvent_t eventStart, eventStop;
    CUDA_CHECK(cudaEventCreate(&eventStart));
    CUDA_CHECK(cudaEventCreate(&eventStop));
    CUDA_CHECK(cudaEventRecord(eventStart));

    for (int r = 0; r < REPEATS; ++r) {
        launchGpuForwardPass(dX, dW1, db1, dZ1, dW2, db2, dZ2);
    }

    CUDA_CHECK(cudaEventRecord(eventStop));
    CUDA_CHECK(cudaEventSynchronize(eventStop));

    float gpuKernelMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuKernelMs, eventStart, eventStop));
    gpuKernelMs /= REPEATS;

    // End-to-end timing includes transfers, kernel execution and output transfer.
    std::vector<float> gpuOutput(BATCH * OUTPUT_SIZE);
    auto gpuTotalStart = std::chrono::steady_clock::now();

    for (int r = 0; r < REPEATS; ++r) {
        CUDA_CHECK(cudaMemcpy(dX, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dW1, W1.data(), W1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db1, b1.data(), b1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dW2, W2.data(), W2.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db2, b2.data(), b2.size() * sizeof(float), cudaMemcpyHostToDevice));

        launchGpuForwardPass(dX, dW1, db1, dZ1, dW2, db2, dZ2);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(gpuOutput.data(), dZ2,
                              gpuOutput.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }

    auto gpuTotalStop = std::chrono::steady_clock::now();
    double gpuEndToEndMs = std::chrono::duration<double, std::milli>(
        gpuTotalStop - gpuTotalStart).count() / REPEATS;

    // Compare CPU and GPU outputs.
    float maxDifference = 0.0f;
    for (size_t i = 0; i < cpuOutput.size(); ++i) {
        maxDifference = std::max(maxDifference,
                                 std::fabs(cpuOutput[i] - gpuOutput[i]));
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nForward-pass dimensions:\n";
    std::cout << "Input: " << BATCH << " x " << INPUT_SIZE << "\n";
    std::cout << "Hidden layer: " << BATCH << " x " << HIDDEN_SIZE << "\n";
    std::cout << "Output: " << BATCH << " x " << OUTPUT_SIZE << "\n";
    std::cout << "\nAverage CPU compute time: " << cpuMs << " ms\n";
    std::cout << "Average GPU kernel-only time: " << gpuKernelMs << " ms\n";
    std::cout << "Average GPU end-to-end time: " << gpuEndToEndMs << " ms\n";

    if (gpuKernelMs > 0.0f) {
        std::cout << "Kernel-only speedup: " << cpuMs / gpuKernelMs << "x\n";
    }
    if (gpuEndToEndMs > 0.0) {
        std::cout << "End-to-end speedup: " << cpuMs / gpuEndToEndMs << "x\n";
    }

    std::cout << "\nMaximum absolute output difference: " << maxDifference << "\n";
    std::cout << "CPU and GPU outputs: "
              << (maxDifference < 1e-4f ? "MATCH (within tolerance)" : "DO NOT MATCH")
              << "\n";

    std::cout << "\nFirst 10 output values (GPU):\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << gpuOutput[i] << (i == 9 ? '\n' : ' ');
    }

    CUDA_CHECK(cudaEventDestroy(eventStart));
    CUDA_CHECK(cudaEventDestroy(eventStop));

    cudaFree(dX);
    cudaFree(dW1);
    cudaFree(db1);
    cudaFree(dZ1);
    cudaFree(dW2);
    cudaFree(db2);
    cudaFree(dZ2);

    return (maxDifference < 1e-4f) ? EXIT_SUCCESS : EXIT_FAILURE;
}
