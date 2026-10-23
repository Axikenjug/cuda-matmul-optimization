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

__global__ void matmul_gpu(const float* A, const float* B, float* C,
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


void genArray(float* arr, int length){
    for(int i = 0; i < length; i++){
        arr[i] = 10.0f * rand()/(float)RAND_MAX;
    }
}


bool enough_close(const float* A, const float* B, int len, float atol, float rtol){
    for(int i = 0; i < len; i++){
        if (!isfinite(A[i]) || !isfinite(B[i])) {
            return false;
        }
        if(fabs(A[i]-B[i]) > atol+rtol*fabs(B[i])){
            return false;
        }
    }
    return true;
}


void matmul_cpu(const float* A,
            const float* B,
            float* C,
            int M, int N, int K
){
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            float sum = 0.0f;
            for(int k = 0; k < K; k++){
                sum += A[i*K+k]*B[k*N+j];
            }
            C[i*N+j] = sum;
        }
    }
}


int main(){
    srand(42);
    vector<int> sizes = {128, 256, 512, 1024};
    float atol = 1e-5f, rtol = 1e-4f;

    
    for(int i = 0; i < 4; i++){
        const int M = sizes[i], N = sizes[i], K = sizes[i];
        constexpr int TILE = 16;

        cout << "size " << M << " x " << M << "running..." << endl;

        int len_A = M*K;
        int len_B = K*N;
        int len_C = M*N;

        int threads = 256;
        int blocks = (len_C + threads - 1)/threads;

        dim3 block (TILE, TILE);
        dim3 grid (
            (N+TILE-1)/TILE,
            (M+TILE-1)/TILE
        );

        vector<float> A(len_A);
        vector<float> B(len_B);
        float* devA = nullptr;
        float* devB = nullptr;
        float* devC1 = nullptr;
        float* devC2 = nullptr;
        vector<float> cpu_result(len_C);

        genArray(A.data(), len_A);
        genArray(B.data(), len_B);

        matmul_cpu(A.data(), B.data(), cpu_result.data(), M, N, K);

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
        err = cudaMalloc((void**)&devC1, sizeof(float)*len_C);
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

        matmul_gpu<<<blocks, threads>>>(devA, devB, devC1, M, N, K);
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

        vector<float> gpu_result1(len_C);
        err = cudaMemcpy(gpu_result1.data(), devC1, sizeof(float) * len_C, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            return 1;
        }

        if(!enough_close(gpu_result1.data(), cpu_result.data(), len_C, atol, rtol)){
            cout << "Correctness FAIL" << endl;
            return 1;
        }

        err = cudaMalloc((void**)&devC2, sizeof(float)*len_C);
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            return 1;
        }
        


        matmul_tiled<TILE><<<grid, block>>>(devA, devB, devC2, M, N, K);
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

        vector<float> gpu_result2(len_C);
        err = cudaMemcpy(gpu_result2.data(), devC2, sizeof(float) * len_C, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            return 1;
        }

        if(!enough_close(gpu_result2.data(), cpu_result.data(), len_C, atol, rtol)){
            cout << "Correctness FAIL" << endl;
            return 1;
        }

        for(int j = 0; j < 25; j++){ // naive warm up
            matmul_gpu<<<blocks, threads>>>(devA, devB, devC1, M, N, K);
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

        
        for(int j = 0; j < 25; j++){ // tiled warm up
            matmul_tiled<<<grid, block>>>(devA, devB, devC2, M, N, K);
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

        
        float mean_naive = 0.0f;
        float mean_tile = 0.0f;

        for(int round = 0; round < 10; round++){
            for(int pos = 0; pos < 4; pos++){
                bool is_naive = (round%2 == 0) 
                    ? (pos == 0 || pos == 3) 
                    : (pos == 1 || pos == 2);
                if(is_naive){
                    cudaEventRecord(start);
                    for(int k = 0; k <100; k++){
                        matmul_gpu<<<blocks, threads>>>(devA, devB, devC1, M, N, K);
                    }
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
            
                    float elapsed_ms = 0.0f;
                    cudaEventElapsedTime(&elapsed_ms, start, stop);
                    float mean_time = elapsed_ms/100;
                    mean_naive += mean_time;

                    cout << "Round" << round << ": Naive\n"
                    << "mean: " << mean_time << "ms\n";
                }
                else{
                    cudaEventRecord(start);
                    for(int k = 0; k <100; k++){
                        matmul_tiled<<<grid, block>>>(devA, devB, devC2, M, N, K);
                    }
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
            
                    float elapsed_ms = 0.0f;
                    cudaEventElapsedTime(&elapsed_ms, start, stop);
                    float mean_time = elapsed_ms/100;
                    mean_tile += mean_time;

                    cout << "Round" << round << ": Tile\n"
                    << "mean: " << mean_time << "ms\n";
                }
            }
        }

        mean_naive /= 20;
        mean_tile /= 20;

        double speedup = mean_naive / mean_tile;

        double flops = 2.0 * M * N * K;
        double naive_gflops = flops / (mean_naive * 1e6);
        double tile_gflops = flops / (mean_tile * 1e6);


        
        cout << "In five rounds: \n"
        << "Naive mean: " << mean_naive
        << "; gflops: " << naive_gflops 
        << "\n Tiled mean: " << mean_tile
        << "; gflops: " << tile_gflops << endl;

        cout << "Tile speed up: " << speedup << endl;

    
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(devA);
        cudaFree(devB);
        cudaFree(devC1);
        cudaFree(devC2);
    }

    return 0;
}   
