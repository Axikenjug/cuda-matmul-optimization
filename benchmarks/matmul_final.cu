#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <vector>
using namespace std;

__global__ void matmul_naive(const float* A, const float* B, float* C,
    int M, int N, int K
);

template <int TILE>
__global__ void matmul_tiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
);

template <int TILE>
// TILE = 16, R = 2
__global__ void matmul_reg_tiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
);

void genArray(float* arr, int length){
    for(int i = 0; i < length; i++){
        arr[i] = 10.0f * rand()/(float)RAND_MAX;
    }
}


int main(){
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cerr << "cuBLAS initialization failed\n";
        return 1;
    }
    
    status = cublasSetMathMode(
        handle,
        CUBLAS_DEFAULT_MATH
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        cerr << "cuBLAS initialization failed\n";
        return 1;
    }

    srand(42);
    vector<int> sizes = {128, 256, 512, 1024};
    float alpha = 1.0f;
    float beta = 0.0f;

    for(int i = 0; i < 4; i++){
        int M = sizes[i], N = sizes[i], K = sizes[i];
        cout << M << "x" << M << "running..." << endl;

        int len_A = M*K;
        int len_B = K*N;
        int len_C = M*N;

        int threads = 256;
        int blocks = (len_C + threads - 1)/threads;

        dim3 blockt (16, 16);
        dim3 gridt (
            (N+16-1)/16,
            (M+16-1)/16
        );

        dim3 blockrt (16, 16);
        dim3 gridrt (
            (N+2*16-1)/(2*16), 
            (M+16-1)/16
        );

        vector<float> A(len_A);
        vector<float> B(len_B);
        float* devA = nullptr;
        float* devB = nullptr;
        float* devC = nullptr;

        genArray(A.data(), len_A);
        genArray(B.data(), len_B);

        cudaError_t err = cudaMalloc((void**)&devA, sizeof(float)*len_A);
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            return 1;
        }
        err = cudaMalloc((void**)&devB, sizeof(float)*len_B);
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            return 1;
        }
        err = cudaMalloc((void**)&devC, sizeof(float)*len_C);
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            return 1;
        }

        err = cudaMemcpy(devA, A.data(), sizeof(float) * len_A, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            return 1;
        }
        err = cudaMemcpy(devB, B.data(), sizeof(float) * len_B, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            return 1;
        }

        for(int j = 0; j < 25; j++){ // naive warm up
            matmul_naive<<<blocks, threads>>>(devA, devB, devC, M, N, K);
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            return 1;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            cout << cudaGetErrorString(err) << endl;
            return 1;
        }

        for(int j = 0; j < 25; j++){ // tile warm up
            matmul_tiled<16><<<gridt, blockt>>>(devA, devB, devC, M, N, K);
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            return 1;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            cout << cudaGetErrorString(err) << endl;
            return 1;
        }

        for(int j = 0; j < 25; j++){ // reg_tile warm up
            matmul_reg_tiled<16><<<gridrt, blockrt>>>(devA, devB, devC, M, N, K);
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            return 1;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            cout << cudaGetErrorString(err) << endl;
            return 1;
        }

        for(int j = 0; j < 25; j++){ // cublas warm up
            status = cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N, M, K,
                &alpha,
                devB, N,
                devA, K,
                &beta,
                devC, N
            );
            if (status != CUBLAS_STATUS_SUCCESS) {
                cerr << "cuBLAS initialization failed\n";
                return 1;
            }
        }
        

        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        float mean_naive = 0.0f;
        float mean_tiled = 0.0f;
        float mean_rtiled = 0.0f;
        float mean_cublas = 0.0f;

        cudaEventRecord(start);
        for(int t = 0; t < 100; t++){
            matmul_naive<<<blocks, threads>>>(devA, devB, devC, M, N, K);
        }
        cudaEventRecord(stop);
        err = cudaEventSynchronize(stop);
        if (err != cudaSuccess) {
            cout << cudaGetErrorString(err) << endl;
            return 1;
        }

        err = cudaGetLastError();

        if (err != cudaSuccess) {
            cerr << "Kernel launch failed: "
            << cudaGetErrorString(err) << endl;
            return 1;
        }
            
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        mean_naive = elapsed_ms/100;

        cudaEventRecord(start);
        for(int t = 0; t < 100; t++){
            matmul_tiled<16><<<gridt, blockt>>>(devA, devB, devC, M, N, K);
        }
        cudaEventRecord(stop);
        err = cudaEventSynchronize(stop);
        if (err != cudaSuccess) {
            cout << cudaGetErrorString(err) << endl;
            return 1;
        }
        err = cudaGetLastError();

        if (err != cudaSuccess) {
            cerr << "Kernel launch failed: "
            << cudaGetErrorString(err) << endl;
            return 1;
        }
            
        elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        mean_tiled = elapsed_ms/100;
        
        cudaEventRecord(start);
        for(int t = 0; t < 100; t++){
            matmul_reg_tiled<16><<<gridrt, blockrt>>>(devA, devB, devC, M, N, K);
        }
        cudaEventRecord(stop);
        err = cudaEventSynchronize(stop);
        if (err != cudaSuccess) {
            cout << cudaGetErrorString(err) << endl;
            return 1;
        }
        err = cudaGetLastError();

        if (err != cudaSuccess) {
            cerr << "Kernel launch failed: "
            << cudaGetErrorString(err) << endl;
            return 1;
        }
            
        elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        mean_rtiled = elapsed_ms/100;
        
        cudaEventRecord(start);
        cublasStatus_t gemm_status = CUBLAS_STATUS_SUCCESS;

        for(int t = 0; t < 100; t++){
            cublasStatus_t s = cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N, M, K,
                &alpha,
                devB, N,
                devA, K,
                &beta,
                devC, N
            );
            if (s != CUBLAS_STATUS_SUCCESS) {
                gemm_status = s;
                break;
            }   
        }

        cudaEventRecord(stop);

        cudaError_t sync_err = cudaEventSynchronize(stop);
        if (gemm_status != CUBLAS_STATUS_SUCCESS ||
            sync_err != cudaSuccess) {
            cerr << "Benchmark execution failed\n";
            return 1;
        }

            
        elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        mean_cublas = elapsed_ms/100;

        double flops = 2.0 * M * N * K;

        double gflops_naive= flops / (mean_naive * 1e6);
        double gflops_tiled = flops / (mean_tiled * 1e6);
        double gflops_rtiled = flops / (mean_rtiled * 1e6);
        double gflops_cublas = flops / (mean_cublas * 1e6);


        cout << "Naive mean time: " << mean_naive << " ms, gflops: " << gflops_naive << "\n"
        << "Tiled mean time: " << mean_tiled << " ms, gflops: " << gflops_tiled << "\n"
        << "Reg Tiled mean time: " << mean_rtiled << " ms, gflops: " << gflops_rtiled << "\n" 
        << "cuBLAS mean time:  " << mean_cublas << "ms, gflops: " << gflops_cublas << "\n";

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(devA);
        cudaFree(devB);
        cudaFree(devC);
    }

    cublasDestroy(handle);
    return 0;
}