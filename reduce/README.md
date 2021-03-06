# Optimization Process
Based on https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf 
Kernels were tested on NVIDIA Quadro K3100M, Compute Capability = 3.0.
The reduction operation is memory bound. The arithmetic intensity = n FLOP / (n*4 Byte) = 0.25. So in the performance, we focus on the bandwidth it can achieve.
## Overview
The reduce kernel performs reduction  by 2 phases. 
1st phase: inside each block; each block then produces a partial sum;
2nd phase: collect the array of partial sums in the global memory, and launch the reduction kernel again

## reduce_v0: naive
Binary fan-in.
Each thread sums up 2 neighboring elements, then stores back to the location of the former element
eg: 
- 1st round:
    - thread 0: d[0] + d[1], stores back to d[0];
    - thread 2: d[2] + d[3], stores back to d[2];
- 2nd round: 
    - thread 0: d[0] + d[2], stores back to d[0];
    - thread 4: d[4] + d[6], stores back to d[4]


### performance
| Kernel   | Time(2^20 floats) | Bandwidth(GB/s) |
|----------|-------------------|-----------------|
| reduce_0 | 1.476ms           | 2.842           |


### problem
Highly divergent warps are very inefficient, and `%` operator is very slow.
Warp divergence: inside a warp, e.g. in 1st round, thread 0,2,4,... are working; while thread 1,3,5,... don't satisfy the condition and are not working. 

## reduce_v1: reduce warp divergence
Replace divergent branch in inner loop with strided index and non-divergent branch. Goal: inside a warp(32 threads), ensure that they go to the same branch.

eg:
- 1st round:
    - thread 0: d[0] + d[1], stores back to d[0];
    - thread 1: d[2] + d[3], stores back to d[2];
- 2nd round: 
    - thread 0: d[0] + d[2], stores back to d[0];
    - thread 1: d[4] + d[6], stores back to d[4];

### performance
| Kernel   | Time(2^20 floats) | Bandwidth(GB/s) | Speedup |
|----------|-------------------|-----------------|---------|
| reduce_0 | 1.476ms           | 2.842           | 1.0x    |
| reduce_1 | 1.382ms           | 3.035           | 1.07x   |

### problem
Shared Memory Bank Conflicts
- 1st round: 
    - thread 0 accesses d[0],d[1]; 
    - thread 16 accesses d[32],d[33];
Shared memory are organized as 32 banks, each bank with a width = a word = 4 byte.
Thread 0, 16,... 2-way bank conflicts

## reduce_v2: reduce bank conflicts
Sequential addressing:
- 1st round:
  - thread 0: d[0]+ d[0 + blocksize/2] -> d[0]
  - thread 1: d[1]+ d[1 + blocksize/2] ->d[1]
  - thread 31: d[31]+ d[31 + blocksize/2] -> d[31]
  - no bank conflict in a warp
- 2nd round:
  - thread 0: d[0]+ d[0 + blocksize / 4] -> d[0]
  - thread 1: d[1]+ d[1 + blocksize / 4] -> d[1]
### performance
| Kernel   | Time(2^20 floats) | Bandwidth(GB/s) | Speedup |
|----------|-------------------|-----------------|---------|
| reduce_0 | 1.476ms           | 2.842           | 1.0x    |
| reduce_1 | 1.382ms           | 3.035           | 1.07x   |
| reduce_2 | 0.763ms           | 5.496           | 1.93x   |

### problem
Idle threads. 1st round: half of the threads do nothing.

## reduce_v3: increase workload of a thread
Halve the number of blocks, and replace single load with two loads and first add of the reduction. Let idle threads also do additions when loading data from global memory to shared memory.

Be careful that in the second call of `reduce_3`, we still stick to the principle above, i.e. rescale the blocksize by 1/2.
### performance

| Kernel   | Time(2^20 floats) | Bandwidth(GB/s) | Speedup |
|----------|-------------------|-----------------|---------|
| reduce_0 | 1.476ms           | 2.842           | 1.0x    |
| reduce_1 | 1.382ms           | 3.035           | 1.07x   |
| reduce_2 | 0.763ms           | 5.496           | 1.93x   |
| reduce_3 | 0.439ms           | 9.560           | 3.36x   |

### problem
Instruction overhead: loop overhead
When the number of active threads < 32, then only a warp is working. No `__syncthreads()` needed anymore.

## reduce_v4: loop unroll
Unroll the last a few loops when only a warp (32 threads) active.

| Kernel   | Time(2^20 floats) | Bandwidth(GB/s) | Speedup |
|----------|-------------------|-----------------|---------|
| reduce_0 | 1.476ms           | 2.842           | 1.0x    |
| reduce_1 | 1.382ms           | 3.035           | 1.07x   |
| reduce_2 | 0.763ms           | 5.496           | 1.93x   |
| reduce_3 | 0.439ms           | 9.560           | 3.36x   |
| reduce_4 | 0.339ms           | 12.370           | 4.35x   |

## reduce_v5: complete loop unroll
Notice: in the slides, Mark Harris wrote "the block size is limited by the GPU to 512 threads". But for my GPU, the limitation is 1024 threads(compute capability >= 2.0).
| Kernel   | Time(2^20 floats) | Bandwidth(GB/s) | Speedup |
|----------|-------------------|-----------------|---------|
| reduce_0 | 1.476ms           | 2.842           | 1.0x    |
| reduce_1 | 1.382ms           | 3.035           | 1.07x   |
| reduce_2 | 0.763ms           | 5.496           | 1.93x   |
| reduce_3 | 0.439ms           | 9.560           | 3.36x   |
| reduce_4 | 0.339ms           | 12.370           | 4.35x   |
| reduce_5 | 0.311ms           | 13.504           | 4.75x   |

## reduce_v6: shuffle operation
The last reduction inside a warp can be done by shuffle operation.
However, the performance does not vary much.
| Kernel   | Time(2^20 floats) | Bandwidth(GB/s) | Speedup |
|----------|-------------------|-----------------|---------|
| reduce_0 | 1.476ms           | 2.842           | 1.0x    |
| reduce_1 | 1.382ms           | 3.035           | 1.07x   |
| reduce_2 | 0.763ms           | 5.496           | 1.93x   |
| reduce_3 | 0.439ms           | 9.560           | 3.36x   |
| reduce_4 | 0.339ms           | 12.370           | 4.35x   |
| reduce_5 | 0.311ms           | 13.504           | 4.75x   |
| reduce_6 | 0.302ms           | 13.885           | 4.89x   |


# TODO:
1. test on NV3070
2. number of elements != 2^x; modify the code
3. modify for flexible  blocksize and gridsize -- apply reduction multiple times?