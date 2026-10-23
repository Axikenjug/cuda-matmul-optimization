#include <cuda_runtime.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cmath>
#include <iostream>
using namespace std;

__global__ void vecAdd(float* A, float* B, float* C, int vectorlength){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx < vectorlength){
        C[idx] = A[idx]+B[idx];
    }
}

void genArray(float* arr, int length){
    for(int i = 0; i < length; i++){
        arr[i] = rand()/(float)RAND_MAX;
    }
}

bool enough_close(const float* A, const float* B, int len, float atol, float rtol){
    for(int i = 0; i < len; i++){
        if(fabs(A[i]-B[i]) > atol+rtol*fabs(B[i])){
            return false;
        }
    }
    return true;
}

int main(){
    srand(time(nullptr));
    int length[3] = {1000, 1024, 1025};
    float atol = 1e-5f, rtol = 1e-4f;
    const int threads = 256;

    for(int i = 0; i < 3; i++){
        const int N = length[i];      
        const int blocks = (N + threads - 1)/threads;

        float A[N];
        float B[N];
        float* devA = nullptr;
        float* devB = nullptr;
        float* devC = nullptr;
        float* cpu_result = (float*)malloc(sizeof(float)*N);
        genArray(A, N);
        genArray(B, N);
    
        for(int j = 0; j < N; j++){
            cpu_result[j] = A[j]+B[j];
        }

        cudaMalloc((void**)&devA, sizeof(float)*N);
        cudaMalloc((void**)&devB, sizeof(float)*N);
        cudaMalloc((void**)&devC, sizeof(float)*N);

        cudaMemcpy(devA, A, sizeof(float) * N, cudaMemcpyHostToDevice);
        cudaMemcpy(devB, B, sizeof(float) * N, cudaMemcpyHostToDevice);

        vecAdd<<<blocks, threads>>>(devA, devB, devC, N);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            return 1;
        }
        cudaDeviceSynchronize();

        float* gpu_result = (float*)malloc(sizeof(float) * N);
        cudaMemcpy(gpu_result, devC, sizeof(float) * N, cudaMemcpyDeviceToHost);

        if(!enough_close(gpu_result, cpu_result, N, atol, rtol)){
            cout << "FAIL" << endl;
            return 1;
        }

        cudaFree(devA);
        cudaFree(devB);
        cudaFree(devC);
        free(cpu_result);
        free(gpu_result);
    }

    cout << "PASS" << endl;
    return 0;
}
