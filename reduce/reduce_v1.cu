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



void __global__ reduce_1(float *d_in, float *d_out){


        extern __shared__ float sdata[];
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        sdata[tid] = d_in[i];
        __syncthreads();

        for(unsigned int s = 1; s < blockDim.x; s *= 2){
            unsigned int index = tid * 2 * s;
            if(index < blockDim.x){
                sdata[index] += sdata[index + s];
            }
            __syncthreads();
        }

        if(tid == 0){
            d_out[blockIdx.x] = sdata[0];
        }
}

void  reduce(float *d_in, float *d_out, float *d_intermediate, dim3 block, dim3 grid){
    reduce_1<<<grid, block, block.x * sizeof(float)>>>( d_in, d_intermediate);
    reduce_1<<<1, grid.x, grid.x* sizeof(float) >>>(d_intermediate, d_out);
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

    int grid_size = get1DGrid(BLOCK_SIZE, N);
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid(grid_size, 1, 1);
    bool isPow2 = checkPow2(N);
    std::cout << "grid size:" << grid_size << std::endl;

    float *result = (float *) malloc (sizeof(float));
    float *intermediate_result;
    float *dev_result;
    cudaMalloc(&intermediate_result, grid_size * sizeof(float)); CHECK_ERR;
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
