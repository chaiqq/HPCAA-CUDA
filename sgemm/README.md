Matrix multiplication with boundary checking. Can take not square matrices.

M 1000
N 1200
K 1024

CublasTime = 6.93575 ms
FLOPs = 496.073 GFLOPs

# sgemm_v0: 
naive, global memory
79.753 ms
30.815 GFLOPs

# sgemm_v1: 
tiling, load global memory to shared memory, coalaced loading

TILE_SIZE = 32
35.307 ms
69.605 GFLOPs

I also tested TILE_SIZE=16, slower than TILE_SIZE=32. Intuitively, larger tilesize, fewer accesses to global memory.

# sgemm_v2
register