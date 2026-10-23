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
    vector<int> M = {16, 32, 19, 7, 64};
    vector<int> N = {16, 32, 29, 5, 128};
    vector<int> K = {16, 32, 23, 3, 32};
    float atol = 1e-5f, rtol = 1e-4f;
    
    for(int i = 0; i < 5; i++){
        int len_A = M[i]*K[i];
        int len_B = K[i]*N[i];
        int len_C = M[i]*N[i];
        
        dim3 block8 (8, 8);
        dim3 grid8 (
            (N[i]+8-1)/8, 
            (M[i]+8-1)/8
        );

        dim3 block16 (16, 16);
        dim3 grid16 (
            (N[i]+16-1)/16, 
            (M[i]+16-1)/16
        );

        dim3 block32 (32, 32);
        dim3 grid32 (
            (N[i]+32-1)/32, 
            (M[i]+32-1)/32
        );


        vector<float> A(len_A);
        vector<float> B(len_B);
        float* devA = nullptr;
        float* devB = nullptr;
        float* devC1 = nullptr;
        float* devC2 = nullptr;
        float* devC3 = nullptr;
        vector<float> cpu_result(len_C);

        genArray(A.data(), len_A);
        genArray(B.data(), len_B);

        matmul_cpu(A.data(), B.data(), cpu_result.data(), M[i], N[i], K[i]);

        cudaError_t err = cudaMalloc((void**)&devA, sizeof(float)*len_A);
        if(err != cudaSuccess){
            cout << "cuda FAIL" << endl;
            return 1;
        }
        err = cudaMalloc((void**)&devB, sizeof(float)*len_B);
        if(err != cudaSuccess){
            cout << "cuda FAIL" << endl;
            return 1;
        }
        err = cudaMalloc((void**)&devC1, sizeof(float)*len_C);
        if(err != cudaSuccess){
            cout << "cuda FAIL" << endl;
            return 1;
        }
        err = cudaMalloc((void**)&devC2, sizeof(float)*len_C);
        if(err != cudaSuccess){
            cout << "cuda FAIL" << endl;
            return 1;
        }
        err = cudaMalloc((void**)&devC3, sizeof(float)*len_C);
        if(err != cudaSuccess){
            cout << "cuda FAIL" << endl;
            return 1;
        }

        err = cudaMemcpy(devA, A.data(), sizeof(float) * len_A, cudaMemcpyHostToDevice);
        if(err != cudaSuccess){
            cout << "cuda FAIL" << endl;
            return 1;
        }
        err = cudaMemcpy(devB, B.data(), sizeof(float) * len_B, cudaMemcpyHostToDevice);
        if(err != cudaSuccess){
            cout << "cuda FAIL" << endl;
            return 1;
        }

        matmul_tiled<8><<<grid8, block8>>>(devA, devB, devC1, M[i], N[i], K[i]);
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
        matmul_tiled<16><<<grid16, block16>>>(devA, devB, devC2, M[i], N[i], K[i]);
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
        matmul_tiled<32><<<grid32, block32>>>(devA, devB, devC3, M[i], N[i], K[i]);
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

        vector<float> gpu_result(len_C);
        err = cudaMemcpy(gpu_result.data(), devC1, sizeof(float) * len_C, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            cout << cudaGetErrorString(err) << endl;
            return 1;
        }

        if(!enough_close(gpu_result.data(), cpu_result.data(), len_C, atol, rtol)){
            cout << "8 FAIL: M=" << M[i]
            << ", N=" << N[i]
            << ", K=" << K[i] << endl;
            return 1;
        }

        err = cudaMemcpy(gpu_result.data(), devC2, sizeof(float) * len_C, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            cout << cudaGetErrorString(err) << endl;
            return 1;
        }

        if(!enough_close(gpu_result.data(), cpu_result.data(), len_C, atol, rtol)){
            cout << "16 FAIL: M=" << M[i]
            << ", N=" << N[i]
            << ", K=" << K[i] << endl;
        }

        err = cudaMemcpy(gpu_result.data(), devC3, sizeof(float) * len_C, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            cout << cudaGetErrorString(err) << endl;
            return 1;
        }

        if(!enough_close(gpu_result.data(), cpu_result.data(), len_C, atol, rtol)){
            cout << "32 FAIL: M=" << M[i]
            << ", N=" << N[i]
            << ", K=" << K[i] << endl;
            return 1;
        }

        cudaFree(devA);
        cudaFree(devB);
        cudaFree(devC1);
        cudaFree(devC2);
        cudaFree(devC3);
    }

    cout << "PASS" << endl;
    return 0;
}