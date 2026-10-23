#include <cuda_runtime.h>
#include <memory.h>
using namespace std;

__global__ void matmul_naive(const float* A, const float* B, float* C,
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