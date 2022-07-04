

M 1024
N 1024
K 1024

CublasTime = 6.93575 ms
FLOPs = 496.073 GFLOPs

# sgemm_v0: naive
naive, global memory
Matrix multiplication with boundary checking. Can take not square matrices.
79.753 ms
30.815 GFLOPs

# sgemm_v1: tiling
Matrix multiplication with boundary checking. Can take not square matrices.
tiling, load global memory to shared memory, coalaced loading

TILE_SIZE = 32
mySegemmTime = 35.307 ms
FLOPs = 69.605 GFLOPs

I also tested TILE_SIZE=16, slower than TILE_SIZE=32. Intuitively, larger tilesize, fewer accesses to global memory.

# sgemm_v2: prefetch
The boundary check introduces too much warp divergence. To avoid this, a square matrix is a better choice. If A,B are not square, we can first do padding to make them square(maybe?). So for the purpose of practice, from this version, we assume the matrices are square. 

Prefetch: global->register->shared memory; while computing, threads retrieve data from shared memory; meanwhile, load data from global memory to register. 

TILE_SIZE = 32
mySgemmTime = 22.324 ms
FLOPs = 96.196 GFLOPs


# segmm_v3: bank conflict?
v2 has bank conflict while reading data from shr_A, shr_B. 16-way conflict? 