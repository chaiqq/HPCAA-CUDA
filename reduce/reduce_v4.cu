#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <cublas_v2.h>

#include <iostream>
#include <sstream>
#include <cmath>

// N: sum up N elements
#define N (1024*1024)
#define BLOCK_SIZE 1024
// note: max number of thread per block = 1024

namespace err {
std::string PrevFile{};
int PrevLine{0};

/**
 * helper function to check for errors in CUDA calls
 * source: NVIDIA
 * */
//#define NDEBUG
void checkErr(const std::string &file, int line) {
#ifndef NDEBUG
    cudaError_t Error = cudaGetLastError();
    if (Error != cudaSuccess) {
        std::stringstream stream;
        stream << '\n'
               << file << ", line " << line << ": " << cudaGetErrorString(Error) << " (" << Error
               << ")\n";
        if (PrevLine > 0) {
            stream << "Previous CUDA call:" << '\n' << PrevFile << ", line " << PrevLine << '\n';
        }
        throw std::runtime_error(stream.str());
    }
    PrevFile = file;
    PrevLine = line;
#endif
}
} // namespace err

#define CHECK_ERR err::checkErr(__FILE__, __LINE__) //__FILE__, __LINE__ are from compiler

void __device__ warpReduce(volatile float* sdata, unsigned int tid){ //must volatile, why??
#if __CUDA_ARCH__ >= 800
    sdata[tid] += sdata[tid + 32]; __syncwarp();
    sdata[tid] += sdata[tid + 16]; __syncwarp();
    sdata[tid] += sdata[tid + 8]; __syncwarp();
    sdata[tid] += sdata[tid + 4]; __syncwarp();
    sdata[tid] += sdata[tid + 2]; __syncwarp();
    sdata[tid] += sdata[tid + 1]; __syncwarp();

#endif
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];

}

void __global__ reduce_4(float *d_in, float *d_out){


        extern __shared__ float sdata[];
        unsigned int tid = threadIdx.x;
        unsigned int i = (2*blockDim.x) * blockIdx.x + threadIdx.x;
        sdata[tid] = d_in[i] + d_in[i+blockDim.x];
        __syncthreads();

        for(unsigned int s = blockDim.x / 2; s > 32; s >>= 1){
            if(tid < s){
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if(tid < 32){
            warpReduce(sdata, tid);
        }
        if(tid == 0){
            d_out[blockIdx.x] = sdata[0];
        }
}

void  reduce(float *d_in, float *d_out, float *d_intermediate, dim3 block, dim3 grid){
    reduce_4<<<grid, block, block.x * sizeof(float)>>>( d_in, d_intermediate);
    reduce_4<<<1, grid.x / 2, grid.x* sizeof(float) >>>(d_intermediate, d_out);
}

size_t get1DGrid(size_t blockSize, size_t size) {
    return (size + blockSize - 1) / blockSize;
}

bool checkPow2(size_t x){
    return x > 0 && !(x & (x-1));
}


int main(){

    float *a = (float *) malloc (N * sizeof(float));
    float *dev_a{nullptr};
    cudaMalloc(&dev_a, N * sizeof(float));
    CHECK_ERR;

    float result_ref = 1.0f * N;
    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
    }
    std::cout << "ref: "<<result_ref << std::endl;

    cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice); CHECK_ERR;

    int grid_size = get1DGrid(BLOCK_SIZE, N) / 2;
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid(grid_size, 1, 1);
    bool isPow2 = checkPow2(N);

    float *result = (float *) malloc (sizeof(float));
    float *intermediate_result;
    float *dev_result;
    cudaMalloc(&intermediate_result, grid.x * sizeof(float)); CHECK_ERR;
    cudaMalloc(&dev_result, sizeof(float)); CHECK_ERR;

    // timer
    cudaEvent_t startTimer{}, stopTimer{};
    cudaEventCreate(&startTimer);
    cudaEventCreate(&stopTimer);
    cudaEventRecord(startTimer, 0);

    reduce(dev_a, dev_result, intermediate_result, block, grid);

    cudaEventRecord(stopTimer, 0);
    cudaEventSynchronize(stopTimer);
    CHECK_ERR;
    float gpuTime{};
    cudaEventElapsedTime(&gpuTime, startTimer, stopTimer);
    cudaEventDestroy(startTimer);
    cudaEventDestroy(stopTimer);

    cudaMemcpy(result, dev_result, sizeof(float), cudaMemcpyDeviceToHost); CHECK_ERR;

    std::cout <<"actual: "<< *result << std::endl;
    if(fabs(result_ref - *result) < 1e-10){
        std::cout << "Answer correct!" << std::endl;
    }
    else{
        std::cout << "Answer wrong!" << std::endl;
    }

    std::cout << "gpuTime = " << gpuTime <<" ms"<< std::endl;
    std::cout << "Bandwidth = " << N * 4 / gpuTime / 1e6 << " GB/s" << std::endl;

    cudaFree(dev_a); CHECK_ERR;
    cudaFree(dev_result); CHECK_ERR;
    cudaFree(intermediate_result); CHECK_ERR;

    return 0;
}
