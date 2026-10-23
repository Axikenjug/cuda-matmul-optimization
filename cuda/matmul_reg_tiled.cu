#include <cuda_runtime.h>
#include <iostream>
#include <vector>
using namespace std;

template <int TILE>

__global__ void matmul_reg_tiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
){
    __shared__ float Atile[TILE][TILE],
                Btile[TILE][2*TILE];
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = blockIdx.y * TILE + ty;
    int col0 = 2 * blockIdx.x * TILE + tx;
    int col1 = col0 + TILE;

    float sum0 = 0, sum1 = 0;

    for(int p = 0; p < (K+TILE-1)/TILE; p++){
        if(row >= M || p*TILE+tx >= K){
            Atile[ty][tx] = 0.0f;
        }
        else{
            Atile[ty][tx] = A[row*K + p*TILE + tx];
        }
        
        if(col0 >= N || (p*TILE+ty) >= K){
            Btile[ty][tx] = 0.0f;
        }
        else{
            Btile[ty][tx] = B[(p*TILE + ty)*N + col0];
        }
        if(col1 >= N || (p*TILE+ty) >= K){
            Btile[ty][tx+TILE] = 0.0f;
        }
        else{
            Btile[ty][tx+TILE] = B[(p*TILE + ty)*N + col1];
        }
        __syncthreads();

        for(int q = 0; q < TILE; q++){
            float a = Atile[ty][q];

            sum0 += a * Btile[q][tx];
            sum1 += a * Btile[q][tx+TILE];
        }

        __syncthreads();

    }
    if(row < M && col0 < N){
        C[row * N + col0] = sum0;
    }
    if(row < M && col1 < N){
        C[row * N + col1] = sum1;
    }    

}


template __global__ void matmul_reg_tiled<8>(
    const float*, const float*, float*,
    int, int, int
);

template __global__ void matmul_reg_tiled<16>(
    const float*, const float*, float*,
    int, int, int
);

template __global__ void matmul_reg_tiled<32>(
    const float*, const float*, float*,
    int, int, int
);