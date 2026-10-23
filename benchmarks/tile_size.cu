#include <cuda_runtime.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <cfloat>
#include <algorithm>
using namespace std;

template <int TILE>
__global__ void matmul_tiled(
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
    srand(42);
    vector<int> sizes = {128, 256, 512, 1024};

    for(int i = 0; i < 4; i++){
        const int M = sizes[i], N = sizes[i], K = sizes[i];

        cout << "size " << M << " x " << M << " running..." << endl;
        int len_A = M*K;
        int len_B = K*N;
        int len_C = M*N;

        dim3 block8 (8, 8);
        dim3 grid8 (
            (N+8-1)/8,
            (M+8-1)/8
        );
        dim3 block16 (16, 16);
        dim3 grid16 (
            (N+16-1)/16, 
            (M+16-1)/16
        );

        dim3 block32 (32, 32);
        dim3 grid32 (
            (N+32-1)/32, 
            (M+32-1)/32
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

        for(int j = 0; j < 25; j++){ // 8_tile warm up
            matmul_tiled<8><<<grid8, block8>>>(devA, devB, devC, M, N, K);
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

        for(int j = 0; j < 25; j++){ // 16_tile warm up
            matmul_tiled<16><<<grid16, block16>>>(devA, devB, devC, M, N, K);
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

        for(int j = 0; j < 25; j++){ // 32_tile warm up
            matmul_tiled<32><<<grid32, block32>>>(devA, devB, devC, M, N, K);
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


        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);


        float mean_8 = 0.0f;
        float mean_16 = 0.0f;
        float mean_32 = 0.0f;

        cudaEventRecord(start);
        for(int t = 0; t < 100; t++){
            matmul_tiled<8><<<grid8, block8>>>(devA, devB, devC, M, N, K);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
            
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        mean_8 = elapsed_ms/100;

        cudaEventRecord(start);
        for(int t = 0; t < 100; t++){
            matmul_tiled<16><<<grid16, block16>>>(devA, devB, devC, M, N, K);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
            
        elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        mean_16 = elapsed_ms/100;
        


        cudaEventRecord(start);
        for(int t = 0; t < 100; t++){
            matmul_tiled<32><<<grid32, block32>>>(devA, devB, devC, M, N, K);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
            
        elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        mean_32 = elapsed_ms/100;

        double flops = 2.0 * M * N * K;

        double gflops8 = flops / (mean_8 * 1e6);
        double gflops16 = flops / (mean_16 * 1e6);
        double gflops32 = flops / (mean_32 * 1e6);

        cout << "Tile 8 mean time: " << mean_8 << " ms, gflops: " << gflops8 << "\n" 
        << "Tile 16 mean time:  " << mean_16 << "ms, gflops: " << gflops16 << "\n"
        << "Tile 32 mean time: " << mean_32 << "ms, gflops: " << gflops32 << "\n";

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(devA);
        cudaFree(devB);
        cudaFree(devC);

    }

    return 0;
}