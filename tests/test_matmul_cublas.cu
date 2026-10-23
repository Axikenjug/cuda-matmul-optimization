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
    cublasHandle_t handle;

    cublasStatus_t status = cublasCreate(&handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
        cerr << "cuBLAS initialization failed\n";
        return 1;
    }

    srand(42);
    int M[3] = {2, 31, 64};
    int K[3] = {2, 9, 128};
    int N[3] = {3, 17, 65};
    float atol = 1e-5f, rtol = 1e-4f;
    float alpha = 1.0f;
    float beta = 0.0f;
    
    for(int i = 0; i < 3; i++){
        int len_A = M[i]*K[i];
        int len_B = K[i]*N[i];
        int len_C = M[i]*N[i];

        vector<float> A(len_A);
        vector<float> B(len_B);
        float* devA = nullptr;
        float* devB = nullptr;
        float* devC = nullptr;
        vector<float> cpu_result(len_C);

        genArray(A.data(), len_A);
        genArray(B.data(), len_B);

        matmul_cpu(A.data(), B.data(), cpu_result.data(), M[i], N[i], K[i]);

        cudaError_t err =cudaMalloc((void**)&devA, sizeof(float)*len_A);
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

        status = cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N[i], M[i], K[i],
            &alpha,
            devB, N[i],
            devA, K[i],
            &beta,
            devC, N[i]
        );

        if (status != CUBLAS_STATUS_SUCCESS) {
            cerr << "cuBLAS initialization failed\n";
            return 1;
        }

        vector<float> gpu_result(len_C);
        err = cudaMemcpy(gpu_result.data(), devC, sizeof(float) * len_C, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            return 1;
        }

        if(!enough_close(gpu_result.data(), cpu_result.data(), len_C, atol, rtol)){
            cout << "FAIL: M=" << M[i]
            << ", N=" << N[i]
            << ", K=" << K[i] << endl;
            return 1;
        }

        cudaFree(devA);
        cudaFree(devB);
        cudaFree(devC);
        
    }
    
    cublasDestroy(handle);
    cout << "PASS" << endl;
    return 0;
}