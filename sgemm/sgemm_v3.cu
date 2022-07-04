/* apply matrix multiplication A*B = C,
 * A: M rows, K cols
 * B: K rows, N cols
 * C: M rows, N cols
 */

// TODO: improve more
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>

#include <iostream>
#include <sstream>
#include <cmath>

#define M 1024
#define N 1024
#define K 1024

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

#define CHECK_ERR err::checkErr(__FILE__, __LINE__)

template <int TILE_SIZE>
__global__ void sgemm_2(float * __restrict__ A, float * __restrict__ B, float * __restrict__ C,
                    const int m, const int k, const int n){
    __shared__ float shr_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shr_B[TILE_SIZE][TILE_SIZE];
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    if(row < m && col < n){
        float Celem = 0.0f, Aelem = 0.0f, Belem = 0.0f;
        // load the first tile into registers
        Aelem = A[row * k + tx];
        Belem = B[ty * n + col];

        for(int phase = 1; phase < m / TILE_SIZE; phase++){
            // load  tiles of A and B to the shared mem.
            shr_A[ty][tx] = Aelem;
            shr_B[ty][tx] = Belem;
            __syncthreads();

            // load the next tile to the registers
            Aelem = A[row * k + tx + phase * TILE_SIZE];
            Belem = B[(phase * TILE_SIZE + ty) * n + col];

            for(int j = 0; j < TILE_SIZE; j++){
                Celem += shr_A[ty][j] * shr_B[j][tx];
            }
            __syncthreads();
        }

        // compute the last tile
        shr_A[ty][tx] = Aelem;
        shr_B[ty][tx] = Belem;
        __syncthreads();

        for (int j = 0; j < TILE_SIZE; ++j){
          Celem += shr_A[ty][j] * shr_B[j][tx];
        }

        C[row * n + col] = Celem;
    }
    
}

void initialize(float *X, int rows, int cols, bool flag = true);
void compute_ref_C(float *A, float *B, float *C, int m, int k, int n);
void print_matrix(float *A, int rows, int cols);
bool checkResult(float *C, float *ref, int rows, int cols);



int main(){
    cudaEvent_t startTimer, stopTimer;
    cudaEventCreate(&startTimer);
    cudaEventCreate(&stopTimer);

    float *A = (float*) malloc(M * K * sizeof(float));
    float *B = (float*) malloc(K * N * sizeof(float));
    float *C = (float*) malloc(M * N * sizeof(float));
    float *ref_C = (float*) malloc(M * N * sizeof(float));

    float *dev_A, *dev_B, *dev_C;
    cudaMalloc(&dev_A, M * K * sizeof(float)); CHECK_ERR;
    cudaMalloc(&dev_B, K * N * sizeof(float)); CHECK_ERR;
    cudaMalloc(&dev_C, M * N * sizeof(float)); CHECK_ERR;

    initialize(A, M, K);
    initialize(B, K, N);
    initialize(C, M, N, false);
    initialize(ref_C, M, N, false);
    compute_ref_C(A, B, ref_C, M, K, N);
    // print_matrix(ref_C, M, N);


    cudaMemcpy(dev_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice); CHECK_ERR;
    cudaMemcpy(dev_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice); CHECK_ERR;
    cudaMemcpy(dev_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice); CHECK_ERR;

    const int tile_size = 32;
    const int gridDim_x = std::ceil((float)N / (float)tile_size);
    const int gridDim_y = std::ceil((float)M / (float)tile_size);

    dim3 GRID(gridDim_x, gridDim_y,1);
    dim3 BLOCK(tile_size, tile_size, 1);

    cudaEventRecord(startTimer);
    sgemm_2<tile_size> <<<GRID, BLOCK>>> (dev_A, dev_B, dev_C, M, K, N);

    cudaEventRecord(stopTimer);
    cudaEventSynchronize(stopTimer);
    CHECK_ERR;
    float mySgemmTime{};
    cudaEventElapsedTime(&mySgemmTime, startTimer, stopTimer);
    CHECK_ERR;


    cudaMemcpy(C, dev_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    if(checkResult(C, ref_C, M, N) == true){
        std::cout << "Answer correct!" << std::endl;
    }else{
        std::cout << "Answer wrong!" << std::endl;
    }

    std::cout << "mySgemmTime = " << mySgemmTime <<" ms"<< std::endl;
    double FLOP = (double)M * (double)N * (double)K * 2.0f;
    double FLOPs = FLOP / (double) mySgemmTime / (double) 1000000;
    std::cout << "FLOPs = " << FLOPs << "GFLOPs" << std::endl;

/*
    // compare with Cublas
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f, beta = 0.0f;
    int numRepeats = 200;
    cudaEventRecord(startTimer);
    for (int i = 0; i < numRepeats; ++i){
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dev_B, N, dev_A, K, &beta, dev_C, N);
    }
    cudaEventRecord(stopTimer);
    cudaEventSynchronize(stopTimer);
    float cublasGemmTime{};
    cudaEventElapsedTime(&cublasGemmTime, startTimer, stopTimer);
    cublasGemmTime /= (float)numRepeats;
    CHECK_ERR;

    std::cout << "============================================"<<std::endl;
    cudaMemcpy(C, dev_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    if(checkResult(C, ref_C, M, N) == true){
        std::cout << "Cublas Answer correct!" << std::endl;
    }else{
        std::cout << "Cublas Answer wrong!" << std::endl;
    }
    std::cout << "CublasTime = " << cublasGemmTime <<" ms"<< std::endl;
    double cublasFLOPs = FLOP / (double) cublasGemmTime / (double) 1000000;
    std::cout << "FLOPs = " << cublasFLOPs << "GFLOPs" << std::endl;

    cublasDestroy(handle);
*/

    cudaEventDestroy(startTimer);
    cudaEventDestroy(stopTimer);
    
    cudaFree(dev_A); CHECK_ERR;
    cudaFree(dev_B); CHECK_ERR;
    cudaFree(dev_C); CHECK_ERR;

    return 0;

}


void initialize(float *X, int rows, int cols, bool flag){
    if(flag){
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                X[cols * i + j] = 0.5*i;
            }
        }
    }else{
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                X[cols * i + j] = 0.0f;
            }
        }
    }
}
void compute_ref_C(float *A, float *B, float *C, int m, int k, int n){
    for(int i = 0; i < m; i++){
        for(int l = 0; l < k; l++){
            for(int j = 0; j < n; j++){
                C[n * i + j] += A[k * i + l] * B[n * l + j];
            }
        }
    }
}

void print_matrix(float *A, int rows, int cols){
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            std::cout << A[cols * i + j] << ' ';
        }
        std::cout << std::endl;
    }
}

bool checkResult(float *C, float *ref, int rows, int cols){
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            if(fabs(C[i * cols + j] - ref[i * cols + j]) > 1e-10){
                return false;
            }
        }
    }
    return true;
}
