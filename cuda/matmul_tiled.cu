#include <cuda_runtime.h>
#include <iostream>
#include <vector>
using namespace std;

template <int TILE>

__global__ void matmul_tiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
){ 
    float sum = 0;
    __shared__ float Atile[TILE][TILE];
    __shared__ float Btile[TILE][TILE];
    
    int row = blockIdx.y*TILE + threadIdx.y;
    int col = blockIdx.x*TILE + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    
    for(int p = 0; p < (K+TILE-1)/TILE; p++){
        if(row >= M || p * TILE + tx >= K){
            Atile[ty][tx] = 0.0f;
        }
        else{
            Atile[ty][tx] = A[row*K + p*TILE + tx];
        }

        if(p*TILE+ty >= K || col >= N){
            Btile[ty][tx] = 0.0f;
        }
        else{
            Btile[ty][tx] = B[(p*TILE + ty)*N + col];
        }

        __syncthreads();

        for(int q = 0; q < TILE; q++){
            sum += Atile[ty][q] * Btile[q][tx];
        }

        __syncthreads();
    }
    
    if(row < M && col < N){
        C[row * N + col] = sum;
    }
}

template __global__ void matmul_tiled<8>(
    const float*, const float*, float*,
    int, int, int
);

template __global__ void matmul_tiled<16>(
    const float*, const float*, float*,
    int, int, int
);

template __global__ void matmul_tiled<32>(
    const float*, const float*, float*,
    int, int, int
);