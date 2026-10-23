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
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < M*N){
        int row = idx / N;
        int col = idx % N;

        float sum = 0.0f;

        for(int k = 0; k < K; k++){
            sum += A[row * K + k]*B[k * N + col]; 
        }
        C[row * N + col] = sum;
    } 
}




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
    vector<int> sizes = {128, 256, 512};
    float atol = 1e-5f, rtol = 1e-4f;

    for(int i = 0; i < 3; i++){
        const int M = sizes[i], N = sizes[i], K = sizes[i];

        cout << "size " << M << " x " << M << "running..." << endl;

        int len_A = M*K;
        int len_B = K*N;
        int len_C = M*N;
        int threads = 256;
        int blocks = (len_C + threads - 1)/threads;

        vector<float> A(len_A);
        vector<float> B(len_B);
        float* devA = nullptr;
        float* devB = nullptr;
        float* devC = nullptr;
        vector<float> cpu_result(len_C);

        genArray(A.data(), len_A);
        genArray(B.data(), len_B);

        matmul_cpu(A.data(), B.data(), cpu_result.data(), M, N, K);
        cudaMalloc((void**)&devA, sizeof(float)*len_A);
        cudaMalloc((void**)&devB, sizeof(float)*len_B);
        cudaMalloc((void**)&devC, sizeof(float)*len_C);

        cudaMemcpy(devA, A.data(), sizeof(float) * len_A, cudaMemcpyHostToDevice);
        cudaMemcpy(devB, B.data(), sizeof(float) * len_B, cudaMemcpyHostToDevice);

        matmul_gpu<<<blocks, threads>>>(devA, devB, devC, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            return 1;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            cout << cudaGetErrorString(err) << endl;
            return 1;
        }
        cudaDeviceSynchronize();

        vector<float> gpu_result(len_C);
        cudaMemcpy(gpu_result.data(), devC, sizeof(float) * len_C, cudaMemcpyDeviceToHost);
        if(!enough_close(gpu_result.data(), cpu_result.data(), len_C, atol, rtol)){
            cout << "Correctness FAIL" << endl;
            return 1;
        }

        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for(int j = 0; j < 2000; j++){ // warm up
            matmul_gpu<<<blocks, threads>>>(devA, devB, devC, M, N, K);
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

        float sum_mean_run_times = 0;
        vector<float> five_round_mean_run_times(5);

        float max_time = 0.0f;
        float min_time = FLT_MAX;

        for(int j = 0; j < 5; j++){
            cudaEventRecord(start);
            for(int k = 0; k <100; k++){
                matmul_gpu<<<blocks, threads>>>(devA, devB, devC, M, N, K);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float elapsed_ms = 0.0f;
            cudaEventElapsedTime(&elapsed_ms, start, stop);
            float mean_time = elapsed_ms/100;

            five_round_mean_run_times[j] = mean_time;
            max_time = max(max_time, mean_time);
            min_time = min(min_time, mean_time);
            sum_mean_run_times += mean_time;
        }
        float mean_run_time = sum_mean_run_times/5;

        for(int k = 0; k < 5; k++){
            cout << "Round " << k << "runs on average: " 
            << five_round_mean_run_times[k] << " ms" << endl;
        }

        cout << "Size " << M << "x" << N
        << " mean: " << mean_run_time << " ms"
        << ", max: " << max_time << " ms"
        << ", min: " << min_time << " ms\n";

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(devA);
        cudaFree(devB);
        cudaFree(devC);
    }
    
    return 0;
}   
